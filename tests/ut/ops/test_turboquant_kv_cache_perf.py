#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarks and correctness checks for TurboQuant KV store/decode/quantize.

`turboquant_pack_kv_for_cache` + cache scatter + compact decode vs FP16 scatter/gather;
`turboquant_quantize_to_packed_bytes` + `turboquant_dequantize_from_packed_bytes` for
encode/decode operator coverage.

NPU performance tests are skipped without `torch.npu`. Toggle decode backend:

    VLLM_ASCEND_TURBOQUANT_DECODE_OP=1  # custom op when built (default)
    VLLM_ASCEND_TURBOQUANT_DECODE_OP=0  # PyTorch reference decode

Toggle encode (4-bit NPU pack kernel when built; 8-bit always uses PyTorch pack):

    VLLM_ASCEND_TURBOQUANT_ENCODE_OP=1  # default
    VLLM_ASCEND_TURBOQUANT_ENCODE_OP=0  # PyTorch reference pack

``VLLM_ASCEND_TURBOQUANT_MSE_IMPL=v2`` selects ``TurboQuantMSEV2`` (Lloyd–Max codebook;
same KV pack semantics as v1: unit direction for indices, fp16 ``‖x‖`` slot).

Layout: ``head_size=128``, ``num_heads=8``, ``block_size=128``; ``cache_token_slots`` in
``(2048, 4096)``; ``bits`` in ``(4, 8)``.

Quantize-only timing (NPU, ``run_turboquant_quantize_benchmark``): PyTorch pack vs
``VLLM_ASCEND_TURBOQUANT_ENCODE_OP=1`` (4-bit kernel when built).

Run with timing output (not captured): ``pytest -s tests/ut/ops/test_turboquant_kv_cache_perf.py``
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    _turboquant_pack_tables,
    _turboquant_slab_group4_to_row_format,
    ensure_turboquant_pack_tables_registered,
    refresh_turboquant_env_cache,
    turboquant_decode_kv_cache_compact,
    turboquant_dequantize_from_packed_bytes,
    turboquant_pack_kv_for_cache,
    turboquant_pack_kv_for_cache_to_cache,
    turboquant_packed_bytes_per_vector,
    turboquant_quantize_to_packed_bytes,
    turboquant_slab_block_size,
    turboquant_slab_row_bytes,
    unpack_uint4,
)

try:
    _NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception:
    _NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(
    not _NPU_AVAILABLE, reason="TurboQuant KV perf benchmark requires Ascend NPU"
)

# Unified layout for all cases: D=128, H=8, tokens per block=128.
KV_HEAD_DIM = 128
KV_NUM_HEADS = 8
KV_BLOCK_SIZE = 128

# TurboQuant MSE quantizer: v1 (legacy 4-bit table / env) or v2 (always Lloyd–Max tables).
TURBOQUANT_MSE_IMPLS = ("v1", "v2")

# TurboQuant index width (nibble-packed for 4-bit; one byte/dim for 8-bit).
TURBOQUANT_KV_BITS = (4, 8)

# Supported total KV cache capacities in token slots (num_blocks = slots // KV_BLOCK_SIZE).
CACHE_TOKEN_SLOTS = (2048, 4096)


@contextmanager
def _patched_turboquant_env(*args: Any, **kwargs: Any):
    with patch.dict(*args, **kwargs):
        refresh_turboquant_env_cache()
        try:
            yield
        finally:
            refresh_turboquant_env_cache()


@requires_npu
def test_turboquant_pack_kv_for_cache_encode_op0_vs_fused():
    """``turboquant_pack_kv_for_cache`` with ENCODE_OP=0 vs ENCODE_OP=1 must match.

    - ``VLLM_ASCEND_TURBOQUANT_ENCODE_OP=0``: PyTorch ``quantize`` + byte pack
      (``turboquant_quantize_to_packed_bytes`` path).
    - ``VLLM_ASCEND_TURBOQUANT_ENCODE_OP=1``: fused ``turboquant_pack_kv_for_cache`` NPU op.

    Optional profiling (both ENCODE_OP=0 and ENCODE_OP=1):
      ``VLLM_ASCEND_TURBOQUANT_PACK_PROFILE=1`` →
      ``/root/l00856060/perflog2/encode_op0`` and ``.../encode_op1``.
    """
    bits_key = 8
    bits_value = 8

    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache"):
        pytest.skip("turboquant_pack_kv_for_cache fused op not available")

    device = torch.device("npu")
    dtype = torch.float16
    T, H, D = 2, 2, KV_HEAD_DIM
    slot_w_k = turboquant_packed_bytes_per_vector(D, bits=bits_key)
    slot_w_v = turboquant_packed_bytes_per_vector(D, bits=bits_value)

    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()

    pack_kwargs = dict(
        key=key,
        value=value,
        bits_key=bits_key,
        bits_value=bits_value,
        slot_w_k=slot_w_k,
        slot_w_v=slot_w_v,
    )
    mse_env = {"VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1"}
    profile = os.environ.get("VLLM_ASCEND_TURBOQUANT_PACK_PROFILE", "0") == "1"
    profile_root = "/root/l00856060/perflog2"
    import torch_npu  # type: ignore

    def _pack_with_optional_profile(encode_op: str, trace_tag: str):
        env = {**mse_env, "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": encode_op}
        with _patched_turboquant_env(os.environ, env):
            if profile:
                trace_dir = f"{profile_root}/{trace_tag}"
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        trace_dir
                    ),
                ):
                    out = turboquant_pack_kv_for_cache(**pack_kwargs)
                    torch.npu.synchronize()
                    return out
            out = turboquant_pack_kv_for_cache(**pack_kwargs)
            torch.npu.synchronize()
            return out

    ref_k, ref_v = _pack_with_optional_profile("0", "encode_op0")
    fused_k, fused_v = _pack_with_optional_profile("1", "encode_op1")

    ref_k_u8 = ref_k.view(torch.uint8)
    ref_v_u8 = ref_v.view(torch.uint8)
    fused_k_u8 = fused_k.view(torch.uint8)
    fused_v_u8 = fused_v.view(torch.uint8)
    print(fused_k_u8.cpu().tolist())
    print(ref_k_u8.cpu().tolist())

    print(
        "ENCODE_OP=1 fused vs ENCODE_OP=0 ref:",
        f"key_shape={tuple(fused_k.shape)}",
        f"value_shape={tuple(fused_v.shape)}",
        f"key_bytes={int(fused_k_u8.numel())}",
        f"value_bytes={int(fused_v_u8.numel())}",
    )

    pk = turboquant_packed_bytes_per_vector(D, bits=8)

    def _assert_pack_close(name: str, fused: torch.Tensor, ref: torch.Tensor) -> None:
        fu, re = fused.view(torch.uint8), ref.view(torch.uint8)
        if torch.equal(fu, re):
            print(f"{name}: OK")
            return

        flat_f, flat_r = fu.reshape(-1), re.reshape(-1)
        i = int((flat_f != flat_r).nonzero(as_tuple=True)[0][0])
        row, col = divmod(i, fu.shape[-1])
        print(f"{name} first mismatch: flat_idx={i} row={row} col={col}")
        print(f"  ENCODE_OP=1 fused={int(flat_f[i])} ENCODE_OP=0 ref={int(flat_r[i])}")
        if col < D:
            print("  -> indices region")
        elif col < pk:
            print("  -> norm bytes region")
        else:
            print("  -> padding region")

        fused_idx = fu[..., :D].to(torch.int16)
        ref_idx = re[..., :D].to(torch.int16)
        idx_abs_diff = (fused_idx - ref_idx).abs()
        idx_mismatch = idx_abs_diff != 0
        idx_mismatch_count = int(idx_mismatch.sum().item())
        idx_tolerance_count = max(8, int(ref_idx.numel() * 0.01))
        non_adjacent_count = int((idx_abs_diff > 1).sum().item())
        print(
            f"{name} index mismatches: count={idx_mismatch_count} "
            f"allowed={idx_tolerance_count} non_adjacent={non_adjacent_count}"
        )
        assert non_adjacent_count == 0
        assert idx_mismatch_count <= idx_tolerance_count

        fused_norm = fu[..., D : D + 2].contiguous().view(torch.float16)
        ref_norm = re[..., D : D + 2].contiguous().view(torch.float16)
        torch.testing.assert_close(fused_norm, ref_norm, rtol=1e-3, atol=1e-3)

        if fu.shape[-1] > pk:
            assert torch.equal(fu[..., pk:], re[..., pk:])

    _assert_pack_close("key (ENCODE_OP=1 vs 0)", fused_k, ref_k)
    _assert_pack_close("value (ENCODE_OP=1 vs 0)", fused_v, ref_v)


@requires_npu
def test_turboquant_pack_kv_for_cache_v2_matches_registered_pack():
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_v2"):
        pytest.skip("turboquant_pack_kv_for_cache_v2 op not available")
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache"):
        pytest.skip("turboquant_pack_kv_for_cache op not available")

    device = torch.device("npu")
    dtype = torch.float16
    T, H, D = 5, 2, KV_HEAD_DIM
    bits_key = bits_value = 8
    P = turboquant_packed_bytes_per_vector(D, bits=bits_key)

    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()

    assert ensure_turboquant_pack_tables_registered(device, D, bits_key)
    ref_k, ref_v = torch.ops._C_ascend.turboquant_pack_kv_for_cache(
        key, value, P, P
    )
    v2_k, v2_v = torch.ops._C_ascend.turboquant_pack_kv_for_cache_v2(
        key, value, P, P
    )
    torch.npu.synchronize()

    torch.testing.assert_close(v2_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(v2_v, ref_v, rtol=0, atol=0)


@requires_npu
def test_turboquant_pack_kv_for_cache_to_cache_matches_old_path():
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_to_cache"):
        pytest.skip("turboquant_pack_kv_for_cache_to_cache op not available")

    device = torch.device("npu")
    dtype = torch.float16
    T, H, D = 7, 2, KV_HEAD_DIM
    bits_key = bits_value = 8
    P = turboquant_packed_bytes_per_vector(D, bits=bits_key)
    B, BS = 2, KV_BLOCK_SIZE
    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    slot_mapping = torch.tensor([0, 7, 130, -1, 5, 129, 3], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
    key_cache_old = torch.zeros(B, BS, H, P, dtype=torch.int8, device=device)
    value_cache_old = torch.zeros_like(key_cache_old)
    key_cache_new = torch.zeros_like(key_cache_old)
    value_cache_new = torch.zeros_like(value_cache_old)

    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key,
        value=value,
        bits_key=bits_key,
        bits_value=bits_value,
        slot_w_k=P,
        slot_w_v=P,
    )
    torch_npu = pytest.importorskip("torch_npu")
    torch_npu._npu_reshape_and_cache(
        key=packed_k,
        value=packed_v,
        key_cache=key_cache_old,
        value_cache=value_cache_old,
        slot_indices=slot_mapping,
    )
    turboquant_pack_kv_for_cache_to_cache(
        key=key,
        value=value,
        key_cache=key_cache_new,
        value_cache=value_cache_new,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        num_reqs=1,
        bits_key=bits_key,
        bits_value=bits_value,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(key_cache_new.view(torch.uint8), key_cache_old.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(value_cache_new.view(torch.uint8), value_cache_old.view(torch.uint8), rtol=0, atol=0)


def test_turboquant_4bit_slab_pack_decode_roundtrip_cpu() -> None:
    device = torch.device("cpu")
    dtype = torch.float16
    B, BS, H, D = 2, KV_BLOCK_SIZE, 2, KV_HEAD_DIM
    bits = 4
    row_w = turboquant_slab_row_bytes(D, bits=bits)
    T = 7
    slot_mapping = torch.tensor([0, 7, 130, -1, 5, 129, 3], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()

    key_cache_slab = torch.zeros(B, H, BS * row_w, dtype=torch.uint8, device=device)
    value_cache_slab = torch.zeros_like(key_cache_slab)
    key_cache_row = torch.zeros(B, BS, H, row_w, dtype=torch.uint8, device=device)
    value_cache_row = torch.zeros_like(key_cache_row)

    env = {
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
    }
    with _patched_turboquant_env(os.environ, env, clear=False):
        packed_k, packed_v = turboquant_pack_kv_for_cache(
            key=key,
            value=value,
            bits_key=bits,
            bits_value=bits,
            slot_w_k=row_w,
            slot_w_v=row_w,
        )
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache_slab,
            value_cache=value_cache_slab,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=1,
            bits_key=bits,
            bits_value=bits,
        )

        slot = slot_mapping.to(torch.int64)
        valid = slot >= 0
        tok_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        block_idx = torch.div(slot[valid], BS, rounding_mode="floor")
        block_off = slot[valid] - block_idx * BS
        head_idx = torch.arange(H, device=device, dtype=torch.int64)
        key_rows = _turboquant_slab_group4_to_row_format(
            key_cache_slab, head_size=D, bits=bits
        )
        value_rows = _turboquant_slab_group4_to_row_format(
            value_cache_slab, head_size=D, bits=bits
        )
        torch.testing.assert_close(
            key_rows[block_idx[:, None], block_off[:, None], head_idx[None, :], :],
            packed_k[tok_idx].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            value_rows[block_idx[:, None], block_off[:, None], head_idx[None, :], :],
            packed_v[tok_idx].view(torch.uint8),
            rtol=0,
            atol=0,
        )

        key_cache_row[block_idx[:, None], block_off[:, None], head_idx[None, :], :] = (
            packed_k[tok_idx].view(torch.uint8)
        )
        value_cache_row[block_idx[:, None], block_off[:, None], head_idx[None, :], :] = (
            packed_v[tok_idx].view(torch.uint8)
        )

        bt = torch.tensor([[0, 1, -1]], dtype=torch.int64, device=device)
        k_slab, v_slab, bt_slab = turboquant_decode_kv_cache_compact(
            key_cache=key_cache_slab,
            value_cache=value_cache_slab,
            block_tables=bt,
            head_size=D,
            dtype=dtype,
            bits=bits,
        )
        k_row, v_row, bt_row = turboquant_decode_kv_cache_compact(
            key_cache=key_cache_row,
            value_cache=value_cache_row,
            block_tables=bt,
            head_size=D,
            dtype=dtype,
            bits=bits,
        )

    assert turboquant_slab_block_size(key_cache_slab, head_size=D, bits=bits) == BS
    torch.testing.assert_close(k_slab, k_row, rtol=0, atol=0)
    torch.testing.assert_close(v_slab, v_row, rtol=0, atol=0)
    torch.testing.assert_close(bt_slab, bt_row, rtol=0, atol=0)


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_turboquant_4bit_pack_to_cache_op_matches_reference_cache(
    dtype: torch.dtype,
) -> None:
    if not _c_ascend_turboquant_op_available(
        "turboquant_pack_kv_for_cache_4bit"
    ):
        pytest.skip("turboquant_pack_kv_for_cache_4bit op not available")

    device = torch.device("npu:0")
    B, BS, H, D = 2, KV_BLOCK_SIZE, 2, KV_HEAD_DIM
    bits = 4
    row_w = turboquant_slab_row_bytes(D, bits=bits)
    T = 7
    slot_mapping = torch.arange(T, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
    torch.manual_seed(1234)
    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    key_cache_ref = torch.zeros(B, H, BS * row_w, dtype=torch.uint8, device=device)
    value_cache_ref = torch.zeros_like(key_cache_ref)
    key_cache_op = torch.zeros_like(key_cache_ref)
    value_cache_op = torch.zeros_like(value_cache_ref)
    env = {
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
    }
    with _patched_turboquant_env(os.environ, env, clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache_ref,
            value_cache=value_cache_ref,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=1,
            bits_key=bits,
            bits_value=bits,
        )
        torch.ops._C_ascend.turboquant_pack_kv_for_cache_4bit(
            key.contiguous(),
            value.contiguous(),
            slot_mapping,
            query_start_loc,
            *_turboquant_pack_tables(device, D, bits, dtype),
            key_cache_op,
            value_cache_op,
            1,
            BS,
        )
        torch.npu.synchronize()

    valid = slot_mapping.to(device="cpu", dtype=torch.int64) >= 0
    tok_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    slot_cpu = slot_mapping.to(device="cpu", dtype=torch.int64)[valid]
    block_idx = torch.div(slot_cpu, BS, rounding_mode="floor").to(device)
    block_off = (slot_cpu - torch.div(slot_cpu, BS, rounding_mode="floor") * BS).to(
        device
    )
    head_idx = torch.arange(H, device=device, dtype=torch.int64)

    def _assert_4bit_pack_close(
        name: str,
        op_cache: torch.Tensor,
        ref_cache: torch.Tensor,
    ) -> None:
        op_rows = _turboquant_slab_group4_to_row_format(
            op_cache, head_size=D, bits=bits
        )
        ref_rows = _turboquant_slab_group4_to_row_format(
            ref_cache, head_size=D, bits=bits
        )
        op_logical = op_rows[
            block_idx[:, None], block_off[:, None], head_idx[None, :], :
        ]
        ref_logical = ref_rows[
            block_idx[:, None], block_off[:, None], head_idx[None, :], :
        ]

        op_indices = unpack_uint4(op_logical[..., : row_w - 2], D).to(torch.int16)
        ref_indices = unpack_uint4(ref_logical[..., : row_w - 2], D).to(torch.int16)
        index_diff = (op_indices - ref_indices).abs()
        assert int((index_diff > 1).sum().item()) == 0
        assert int((index_diff != 0).sum().item()) <= max(
            32, int(ref_indices.numel() * 0.02)
        )
        torch.testing.assert_close(
            op_logical[..., row_w - 2 : row_w],
            ref_logical[..., row_w - 2 : row_w],
            rtol=0,
            atol=0,
        )

        with _patched_turboquant_env(
            os.environ,
            {
                "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
                "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
            },
            clear=False,
        ):
            op_decoded = turboquant_dequantize_from_packed_bytes(
                op_logical.reshape(-1, row_w), head_size=D, dtype=dtype, bits=bits
            )
            ref_decoded = turboquant_dequantize_from_packed_bytes(
                ref_logical.reshape(-1, row_w), head_size=D, dtype=dtype, bits=bits
            )
            torch.npu.synchronize()
        max_err = float((op_decoded.float() - ref_decoded.float()).abs().max().cpu())
        assert max_err <= 0.25, f"{name} decoded max error too large: {max_err}"

    _assert_4bit_pack_close("key_cache", key_cache_op, key_cache_ref)
    _assert_4bit_pack_close("value_cache", value_cache_op, value_cache_ref)


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_turboquant_4bit_pack_to_cache_seq_aware_decode_matches_reference(
    dtype: torch.dtype,
) -> None:
    if not _c_ascend_turboquant_op_available(
        "turboquant_pack_kv_for_cache_4bit"
    ):
        pytest.skip("turboquant_pack_kv_for_cache_4bit op not available")

    device = torch.device("npu:0")
    B, BS, H, D = 2, KV_BLOCK_SIZE, 2, KV_HEAD_DIM
    bits = 4
    row_w = turboquant_slab_row_bytes(D, bits=bits)
    slot_mapping = torch.tensor(
        [0, 4, 8, 128, 132, 136],
        dtype=torch.int32,
        device=device,
    )
    T = int(slot_mapping.numel())
    query_start_loc = torch.arange(T + 1, dtype=torch.int32, device=device)
    torch.manual_seed(4321)
    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    key_cache_general = torch.zeros(B, H, BS * row_w, dtype=torch.uint8, device=device)
    value_cache_general = torch.zeros_like(key_cache_general)
    key_cache_direct = torch.zeros_like(key_cache_general)
    value_cache_direct = torch.zeros_like(value_cache_general)

    ref_env = {
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
    }
    op_env = dict(ref_env)
    op_env["VLLM_ASCEND_TURBOQUANT_ENCODE_OP"] = "1"
    with _patched_turboquant_env(os.environ, ref_env, clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache_general,
            value_cache=value_cache_general,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=T,
            bits_key=bits,
            bits_value=bits,
        )
    with _patched_turboquant_env(os.environ, op_env, clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache_direct,
            value_cache=value_cache_direct,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=T,
            bits_key=bits,
            bits_value=bits,
        )
    torch.npu.synchronize()

    torch.testing.assert_close(key_cache_direct, key_cache_general, rtol=0, atol=0)
    torch.testing.assert_close(value_cache_direct, value_cache_general, rtol=0, atol=0)


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_turboquant_4bit_pack_to_cache_seq_aware_prefill_matches_reference(
    dtype: torch.dtype,
) -> None:
    if not _c_ascend_turboquant_op_available(
        "turboquant_pack_kv_for_cache_4bit"
    ):
        pytest.skip("turboquant_pack_kv_for_cache_4bit op not available")

    device = torch.device("npu:0")
    B, BS, H, D = 2, KV_BLOCK_SIZE, 2, KV_HEAD_DIM
    bits = 4
    row_w = turboquant_slab_row_bytes(D, bits=bits)
    slot_mapping = torch.tensor(
        [1, 2, 3, 128, 129, 130, 131, 132, 133],
        dtype=torch.int32,
        device=device,
    )
    T = int(slot_mapping.numel())
    query_start_loc = torch.tensor([0, 3, T], dtype=torch.int32, device=device)
    torch.manual_seed(5678)
    key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    key_cache_general = torch.zeros(B, H, BS * row_w, dtype=torch.uint8, device=device)
    value_cache_general = torch.zeros_like(key_cache_general)
    key_cache_fast = torch.zeros_like(key_cache_general)
    value_cache_fast = torch.zeros_like(value_cache_general)

    ref_env = {
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
    }
    op_env = dict(ref_env)
    op_env["VLLM_ASCEND_TURBOQUANT_ENCODE_OP"] = "1"
    with _patched_turboquant_env(os.environ, ref_env, clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache_general,
            value_cache=value_cache_general,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=2,
            bits_key=bits,
            bits_value=bits,
        )
    with _patched_turboquant_env(os.environ, op_env, clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache_fast,
            value_cache=value_cache_fast,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=2,
            bits_key=bits,
            bits_value=bits,
        )
    torch.npu.synchronize()

    torch.testing.assert_close(key_cache_fast, key_cache_general, rtol=0, atol=0)
    torch.testing.assert_close(value_cache_fast, value_cache_general, rtol=0, atol=0)


def _num_blocks_for_cache_tokens(total_token_slots: int, *, block_size: int = KV_BLOCK_SIZE) -> int:
    if total_token_slots % block_size != 0:
        raise ValueError(f"cache_token_slots {total_token_slots} must divide block_size {block_size}")
    return total_token_slots // block_size


def _sync_device(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def bench_device_ms(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    for _ in range(warmup):
        fn()
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync_device(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _fp16_kv_scatter(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    slot = slot_mapping.to(torch.int64)
    valid = slot >= 0
    if not torch.any(valid):
        return
    slot = slot[valid]
    tok_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    block_idx = torch.div(slot, block_size, rounding_mode="floor")
    block_off = slot - block_idx * block_size
    key_cache[block_idx, block_off] = key[tok_idx]
    value_cache[block_idx, block_off] = value[tok_idx]


def _turboquant_pack_and_scatter(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    bits: int = 4,
    bits_key: int | None = None,
    bits_value: int | None = None,
) -> None:
    if key.numel() == 0:
        return
    bk = bits if bits_key is None else bits_key
    bv = bits if bits_value is None else bits_value
    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key,
        value=value,
        bits_key=bk,
        bits_value=bv,
        slot_w_k=key_cache.shape[-1],
        slot_w_v=value_cache.shape[-1],
    )

    slot = slot_mapping.to(torch.int64)
    valid = slot >= 0
    if not torch.any(valid):
        return
    slot = slot[valid]
    tok_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    block_size = key_cache.shape[1]
    block_idx = torch.div(slot, block_size, rounding_mode="floor")
    block_off = slot - block_idx * block_size
    key_cache[block_idx, block_off] = packed_k[tok_idx].view(torch.uint8)
    value_cache[block_idx, block_off] = packed_v[tok_idx].view(torch.uint8)


@dataclass
class TurboQuantKvBenchConfig:
    """Inputs for `run_turboquant_kv_benchmark` (extend for new custom op shapes)."""

    # Total logical token slots in KV cache (2048 or 4096 with block_size=128).
    cache_token_slots: int = 2048
    block_size: int = KV_BLOCK_SIZE
    num_heads: int = KV_NUM_HEADS
    head_size: int = KV_HEAD_DIM
    # Per-iteration store: number of token rows written (≤ cache capacity).
    num_tokens: int = 2048
    dtype: torch.dtype = torch.float16
    warmup: int = 3
    repeat: int = 12
    # decode: number of sequences and max blocks per sequence (subset of num_blocks)
    num_seqs: int = 4
    max_blocks_per_seq: int = 12
    # ``v2`` = :class:`TurboQuantMSEV2` (Lloyd–Max); ``v1`` = legacy :class:`TurboQuantMSE`.
    mse_impl: str = "v2"
    bits: int = 4

    @property
    def num_blocks(self) -> int:
        return _num_blocks_for_cache_tokens(self.cache_token_slots, block_size=self.block_size)


@dataclass
class TurboQuantQuantizeBenchConfig:
    """Inputs for `run_turboquant_quantize_benchmark` (KV-shaped token rows × heads × dim)."""

    num_tokens: int = 2048
    num_heads: int = KV_NUM_HEADS
    head_size: int = KV_HEAD_DIM
    dtype: torch.dtype = torch.float16
    warmup: int = 3
    repeat: int = 12
    mse_impl: str = "v2"
    bits: int = 4


def run_turboquant_kv_benchmark(
    config: TurboQuantKvBenchConfig,
    *,
    device: torch.device,
) -> dict[str, float | str | int]:
    """Time store + compact decode for TurboQuant vs FP16 baseline. Pure timing helper."""

    with _patched_turboquant_env(
        os.environ, {"VLLM_ASCEND_TURBOQUANT_MSE_IMPL": config.mse_impl}
    ):
        B = config.num_blocks
        BS = config.block_size
        H = config.num_heads
        D = config.head_size
        T = min(config.num_tokens, B * BS)
        P = turboquant_packed_bytes_per_vector(D, bits=config.bits)
        dtype = config.dtype

        key_tq = torch.zeros(B, BS, H, P, dtype=torch.uint8, device=device)
        value_tq = torch.zeros_like(key_tq)
        key_fp = torch.zeros(B, BS, H, D, dtype=dtype, device=device)
        value_fp = torch.zeros_like(key_fp)

        key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
        value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
        slot_mapping = torch.randint(0, B * BS, (T,), device=device, dtype=torch.int64)

        decode_op_env = os.environ.get("VLLM_ASCEND_TURBOQUANT_DECODE_OP", "1")

        def do_store_tq() -> None:
            _turboquant_pack_and_scatter(
                key=key,
                value=value,
                key_cache=key_tq,
                value_cache=value_tq,
                slot_mapping=slot_mapping,
                bits=config.bits,
            )

        def do_store_fp() -> None:
            _fp16_kv_scatter(
                key=key,
                value=value,
                key_cache=key_fp,
                value_cache=value_fp,
                slot_mapping=slot_mapping,
                block_size=BS,
            )

        # Warm once so quantizer caches are built (same as steady-state serving).
        do_store_tq()
        do_store_fp()
        _sync_device(device)

        store_tq_ms = bench_device_ms(do_store_tq, device=device, warmup=config.warmup, repeat=config.repeat)
        store_fp_ms = bench_device_ms(do_store_fp, device=device, warmup=config.warmup, repeat=config.repeat)

        # block_tables: each seq references a contiguous stripe of logical blocks
        bt = torch.full(
            (config.num_seqs, config.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=device,
        )
        max_b = min(config.max_blocks_per_seq, B)
        for s in range(config.num_seqs):
            start = min(s * 3 % max(1, B - max_b), B - max_b)
            bt[s, :max_b] = torch.arange(start, start + max_b, device=device, dtype=torch.int32)

        bt64 = bt.to(torch.int64)

        def do_decode_tq() -> None:
            turboquant_decode_kv_cache_compact(
                key_cache=key_tq,
                value_cache=value_tq,
                block_tables=bt64,
                head_size=D,
                dtype=dtype,
                bits=config.bits,
            )

        def do_decode_fp() -> None:
            valid = bt64 >= 0
            if not torch.any(valid):
                return
            used_sorted, _ = torch.sort(bt64[valid].unique())
            _ = key_fp.index_select(0, used_sorted)
            _ = value_fp.index_select(0, used_sorted)

        decode_tq_ms = bench_device_ms(do_decode_tq, device=device, warmup=config.warmup, repeat=config.repeat)
        decode_fp_ms = bench_device_ms(do_decode_fp, device=device, warmup=config.warmup, repeat=config.repeat)

    return {
        "store_turboquant_ms": store_tq_ms,
        "store_fp16_ms": store_fp_ms,
        "decode_turboquant_compact_ms": decode_tq_ms,
        "decode_fp16_gather_ms": decode_fp_ms,
        "decode_op_env": decode_op_env,
        "device": str(device),
        "cache_token_slots": config.cache_token_slots,
        "mse_impl": config.mse_impl,
        "bits": config.bits,
    }


def run_turboquant_quantize_benchmark(
    config: TurboQuantQuantizeBenchConfig,
    *,
    device: torch.device,
) -> dict[str, float | str | int]:
    """Time ``turboquant_quantize_to_packed_bytes``: PyTorch pack (ENCODE_OP=0) vs kernel (ENCODE_OP=1).

    8-bit uses the PyTorch pack path for both toggles; expect similar timings.
    """

    H, D, T = config.num_heads, config.head_size, config.num_tokens
    dtype = config.dtype
    bits = config.bits
    x = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
    encode_op_env = os.environ.get("VLLM_ASCEND_TURBOQUANT_ENCODE_OP", "1")

    def make_do_quantize(encode_op: str) -> Callable[[], None]:
        def fn() -> None:
            with _patched_turboquant_env(
                os.environ,
                {
                    "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": config.mse_impl,
                    "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": encode_op,
                },
                clear=False,
            ):
                turboquant_quantize_to_packed_bytes(x, bits=bits)

        return fn

    do_py = make_do_quantize("0")
    do_custom = make_do_quantize("1")
    do_py()
    do_custom()
    _sync_device(device)

    ms_py = bench_device_ms(do_py, device=device, warmup=config.warmup, repeat=config.repeat)
    ms_custom = bench_device_ms(do_custom, device=device, warmup=config.warmup, repeat=config.repeat)

    return {
        "quantize_pytorch_encode_ms": ms_py,
        "quantize_custom_encode_op_ms": ms_custom,
        "quantize_ratio_py_to_custom": ms_py / max(ms_custom, 1e-9),
        "encode_op_env": encode_op_env,
        "device": str(device),
        "num_tokens": T,
        "mse_impl": config.mse_impl,
        "bits": bits,
    }


def _turboquant_qdq_fp16(
    x: torch.Tensor,
    *,
    bits: int,
    encode_op: int,
    decode_op: int,
    mse_impl: str,
) -> torch.Tensor:
    """Quantize then dequantize ``x`` under the given TurboQuant env toggles."""
    head_size = x.shape[-1]
    env = {
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": mse_impl,
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": str(encode_op),
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": str(decode_op),
    }
    with _patched_turboquant_env(os.environ, env, clear=False):
        packed = turboquant_quantize_to_packed_bytes(x, bits=bits)
        return turboquant_dequantize_from_packed_bytes(
            packed, head_size=head_size, dtype=x.dtype, bits=bits
        )


@pytest.mark.parametrize("tq_bits", TURBOQUANT_KV_BITS)
@pytest.mark.parametrize("mse_impl", TURBOQUANT_MSE_IMPLS)
def test_turboquant_quantize_roundtrip_cpu(mse_impl: str, tq_bits: int) -> None:
    """``turboquant_quantize_to_packed_bytes`` + reference decode reconstructs fp16 on CPU."""
    device = torch.device("cpu")
    H, D = KV_NUM_HEADS, KV_HEAD_DIM
    n = 256
    x = torch.randn(n, H, D, dtype=torch.float16, device=device)
    x_hat = _turboquant_qdq_fp16(
        x,
        bits=tq_bits,
        encode_op=0,
        decode_op=0,
        mse_impl=mse_impl,
    )
    torch.testing.assert_close(x_hat, x, rtol=0.06, atol=0.35)


@pytest.mark.parametrize("tq_bits", TURBOQUANT_KV_BITS)
@pytest.mark.parametrize("mse_impl", TURBOQUANT_MSE_IMPLS)
@requires_npu
def test_turboquant_quantize_npu_custom_encode_matches_reference(
    mse_impl: str, tq_bits: int
) -> None:
    """8-bit: ENCODE_OP toggles still use the PyTorch pack path (strict match).

    4-bit: NPU kernel quantizes in fp16; the reference uses fp32 ``y``/codebook distances for
    argmin, so indices can differ near decision boundaries. Allow a wider tolerance than 8-bit.
    """
    device = torch.device("npu:0")
    H, D = KV_NUM_HEADS, KV_HEAD_DIM
    n = 512
    x = torch.randn(n, H, D, dtype=torch.float16, device=device).contiguous()
    ref = _turboquant_qdq_fp16(x, bits=tq_bits, encode_op=0, decode_op=0, mse_impl=mse_impl)
    fast = _turboquant_qdq_fp16(x, bits=tq_bits, encode_op=1, decode_op=0, mse_impl=mse_impl)
    if tq_bits == 4:
        torch.testing.assert_close(fast, ref, rtol=0.15, atol=6.0)
    else:
        torch.testing.assert_close(fast, ref, rtol=0.01, atol=0.1)


@pytest.mark.parametrize("tq_bits", TURBOQUANT_KV_BITS)
@pytest.mark.parametrize("mse_impl", TURBOQUANT_MSE_IMPLS)
@pytest.mark.parametrize("num_tokens", (2048, 4096))
@requires_npu
def test_turboquant_quantize_performance_npu(
    mse_impl: str,
    tq_bits: int,
    num_tokens: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print ``turboquant_quantize_to_packed_bytes`` latency: ENCODE_OP=0 vs ENCODE_OP=1."""

    device = torch.device("npu:0")
    cfg = TurboQuantQuantizeBenchConfig(
        num_tokens=num_tokens, mse_impl=mse_impl, bits=tq_bits
    )
    stats = run_turboquant_quantize_benchmark(cfg, device=device)

    out = (
        f"[TurboQuant quantize bench] num_tokens={int(stats['num_tokens'])} "
        f"bits={int(stats['bits'])} MSE_IMPL={stats['mse_impl']} "
        f"(process VLLM_ASCEND_TURBOQUANT_ENCODE_OP={stats['encode_op_env']}) "
        f"device={stats['device']}\n"
        f"  quantize PyTorch pack (ENCODE_OP=0): {stats['quantize_pytorch_encode_ms']:.4f} ms\n"
        f"  quantize custom encode op (ENCODE_OP=1): {stats['quantize_custom_encode_op_ms']:.4f} ms "
        f"(ratio py/custom: {float(stats['quantize_ratio_py_to_custom']):.3f}x)\n"
    )
    with capsys.disabled():
        print(out, end="")

    assert float(stats["quantize_pytorch_encode_ms"]) > 0
    assert float(stats["quantize_custom_encode_op_ms"]) > 0


@pytest.mark.parametrize("tq_bits", TURBOQUANT_KV_BITS)
@pytest.mark.parametrize("cache_token_slots", CACHE_TOKEN_SLOTS)
@pytest.mark.parametrize("mse_impl", TURBOQUANT_MSE_IMPLS)
@requires_npu
def test_turboquant_kv_compact_matches_full_decode(
    cache_token_slots: int, mse_impl: str, tq_bits: int
) -> None:
    """Compact decode must match full-cache dequantize on the same blocks (NPU)."""
    device = torch.device("npu:0")
    BS, H, D = KV_BLOCK_SIZE, KV_NUM_HEADS, KV_HEAD_DIM
    B = _num_blocks_for_cache_tokens(cache_token_slots, block_size=BS)
    P = turboquant_packed_bytes_per_vector(D, bits=tq_bits)
    dtype = torch.float16

    key_cache = torch.zeros(B, BS, H, P, dtype=torch.uint8, device=device)
    value_cache = torch.zeros_like(key_cache)
    T = B * BS
    key = torch.randn(T, H, D, dtype=dtype, device=device)
    value = torch.randn(T, H, D, dtype=dtype, device=device)
    slot_mapping = torch.arange(T, device=device, dtype=torch.int64)
    env = {"VLLM_ASCEND_TURBOQUANT_MSE_IMPL": mse_impl}
    with _patched_turboquant_env(os.environ, env, clear=False):
        _turboquant_pack_and_scatter(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            bits=tq_bits,
        )

        bt = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32, device=device)
        k_c, v_c, _ = turboquant_decode_kv_cache_compact(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=bt.to(torch.int64),
            head_size=D,
            dtype=dtype,
            bits=tq_bits,
        )
        k_full = turboquant_dequantize_from_packed_bytes(
            key_cache, head_size=D, dtype=dtype, bits=tq_bits
        )
        v_full = turboquant_dequantize_from_packed_bytes(
            value_cache, head_size=D, dtype=dtype, bits=tq_bits
        )
    used = torch.tensor([0, 1, 2], device=device, dtype=torch.int64)
    torch.testing.assert_close(k_c, k_full.index_select(0, used), rtol=0.01, atol=0.1)
    torch.testing.assert_close(v_c, v_full.index_select(0, used), rtol=0.01, atol=0.1)


@pytest.mark.parametrize("tq_bits", TURBOQUANT_KV_BITS)
@pytest.mark.parametrize("mse_impl", TURBOQUANT_MSE_IMPLS)
@pytest.mark.parametrize("cache_token_slots", CACHE_TOKEN_SLOTS)
@pytest.mark.parametrize("decode_op", ("0", "1"))
@requires_npu
def test_turboquant_kv_performance_vs_fp16(
    cache_token_slots: int,
    decode_op: str,
    mse_impl: str,
    tq_bits: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report quant store/decode vs FP16 scatter/gather; env controls TurboQuant decode kernel."""

    device = torch.device("npu:0")
    cfg = TurboQuantKvBenchConfig(
        cache_token_slots=cache_token_slots, mse_impl=mse_impl, bits=tq_bits
    )
    with _patched_turboquant_env(
        os.environ, {"VLLM_ASCEND_TURBOQUANT_DECODE_OP": decode_op}, clear=False
    ):
        stats = run_turboquant_kv_benchmark(cfg, device=device)

    out = (
        f"[TurboQuant KV bench] cache_token_slots={int(stats['cache_token_slots'])} "
        f"bits={int(stats['bits'])} MSE_IMPL={stats['mse_impl']} "
        f"VLLM_ASCEND_TURBOQUANT_DECODE_OP={decode_op} "
        f"device={stats['device']}\n"
        f"  store turboquant: {stats['store_turboquant_ms']:.4f} ms\n"
        f"  store fp16 scatter: {stats['store_fp16_ms']:.4f} ms "
        f"(ratio fp16/tq: {float(stats['store_fp16_ms']) / max(float(stats['store_turboquant_ms']), 1e-9):.3f}x)\n"
        f"  decode compact (turboquant): {stats['decode_turboquant_compact_ms']:.4f} ms\n"
        f"  decode gather (fp16): {stats['decode_fp16_gather_ms']:.4f} ms "
        f"(ratio tq/fp16: {float(stats['decode_turboquant_compact_ms']) / max(float(stats['decode_fp16_gather_ms']), 1e-9):.3f}x)\n"
    )
    with capsys.disabled():
        print(out, end="")

    # Sanity: timings are positive finite (no graph / launch hard failure).
    assert float(stats["store_turboquant_ms"]) > 0
    assert float(stats["decode_turboquant_compact_ms"]) > 0


@pytest.mark.parametrize("tq_bits", TURBOQUANT_KV_BITS)
@pytest.mark.parametrize("cache_token_slots", CACHE_TOKEN_SLOTS)
@pytest.mark.parametrize("mse_impl", TURBOQUANT_MSE_IMPLS)
def test_turboquant_kv_roundtrip_cpu_reference(
    cache_token_slots: int, mse_impl: str, tq_bits: int
) -> None:
    """Reference decode path on CPU: store + compact decode stays self-consistent."""
    device = torch.device("cpu")
    BS, H, D = KV_BLOCK_SIZE, KV_NUM_HEADS, KV_HEAD_DIM
    B = _num_blocks_for_cache_tokens(cache_token_slots, block_size=BS)
    P = turboquant_packed_bytes_per_vector(D, bits=tq_bits)
    dtype = torch.float16

    key_cache = torch.zeros(B, BS, H, P, dtype=torch.uint8, device=device)
    value_cache = torch.zeros_like(key_cache)
    T = min(2048, B * BS)
    key = torch.randn(T, H, D, dtype=dtype, device=device)
    value = torch.randn(T, H, D, dtype=dtype, device=device)
    slot_mapping = torch.randint(0, B * BS, (T,), dtype=torch.int64)
    with _patched_turboquant_env(
        os.environ,
        {
            "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
            "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": mse_impl,
        },
        clear=False,
    ):
        _turboquant_pack_and_scatter(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            bits=tq_bits,
        )
        k_c, v_c, _ = turboquant_decode_kv_cache_compact(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=torch.tensor([[0, 1, -1]], dtype=torch.int64, device=device),
            head_size=D,
            dtype=dtype,
            bits=tq_bits,
        )

        k_full = turboquant_dequantize_from_packed_bytes(
            key_cache, head_size=D, dtype=dtype, bits=tq_bits
        )
        v_full = turboquant_dequantize_from_packed_bytes(
            value_cache, head_size=D, dtype=dtype, bits=tq_bits
        )
    used = torch.tensor([0, 1], dtype=torch.int64)
    torch.testing.assert_close(k_c, k_full.index_select(0, used), rtol=0.02, atol=0.2)
    torch.testing.assert_close(v_c, v_full.index_select(0, used), rtol=0.02, atol=0.2)


def test_turboquant_kv_mixed_key_value_bits_cpu() -> None:
    """K at 8-bit and V at 4-bit with narrow V cache rows roundtrips on CPU."""
    device = torch.device("cpu")
    BS, H, D = KV_BLOCK_SIZE, KV_NUM_HEADS, KV_HEAD_DIM
    B = 2
    bk, bv = 8, 4
    pk = turboquant_packed_bytes_per_vector(D, bits=bk)
    pv = turboquant_packed_bytes_per_vector(D, bits=bv)
    dtype = torch.float16
    key_cache = torch.zeros(B, BS, H, pk, dtype=torch.uint8, device=device)
    value_cache = torch.zeros(B, BS, H, pv, dtype=torch.uint8, device=device)
    T = 32
    key = torch.randn(T, H, D, dtype=dtype, device=device)
    value = torch.randn(T, H, D, dtype=dtype, device=device)
    slot_mapping = torch.randint(0, B * BS, (T,), dtype=torch.int64)
    with _patched_turboquant_env(
        os.environ,
        {"VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0", "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1"},
        clear=False,
    ):
        _turboquant_pack_and_scatter(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            bits_key=bk,
            bits_value=bv,
        )
        k_dec, v_dec, _ = turboquant_decode_kv_cache_compact(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=torch.tensor([[0, -1]], dtype=torch.int64, device=device),
            head_size=D,
            dtype=dtype,
            bits_key=bk,
            bits_value=bv,
        )
    # One block used; compare a few gathered slots
    sm = slot_mapping.to(torch.int64)
    for i in (0, min(5, T - 1)):
        bi = sm[i] // BS
        bo = sm[i] - bi * BS
        if bi != 0:
            continue
        torch.testing.assert_close(
            k_dec[0, bo, :, :],
            key[i],
            rtol=0.05,
            atol=0.3,
        )
        torch.testing.assert_close(
            v_dec[0, bo, :, :],
            value[i],
            rtol=0.05,
            atol=0.3,
        )
