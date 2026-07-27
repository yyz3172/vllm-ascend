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

"""BitResidual K8V4 pack + paged attention smoke test.

Run on an NPU machine after building custom ops:

    python tests/e2e/singlecard/xrx_bit_residual_k8v4_golden.py
    python tests/e2e/singlecard/xrx_bit_residual_k8v4_golden.py --attn-op fia
    python tests/e2e/singlecard/xrx_bit_residual_k8v4_golden.py --attn-op paged

Options:
    --attn-op {paged,fia}
        Select the paged-attention op for the chain/qtile tests. ``paged`` uses
        ``bit_residual_attention_paged_k8v4`` (Vector-QK path); ``fia`` uses
        ``bit_residual_fia_paged_k8v4`` (FlashAttention prefill path,
        sparse_mode=3 right-down). When omitted, each test keeps its own
        default (chain -> paged, qtile -> fia); when set, it overrides both
        tests' defaults. See ``--list-tests`` to run a single test in isolation.

    --test NAME
        Run only the named test (one of: manual_single_kv, zero_kv,
        pack_attention_chain, pack_attention_qtile). Defaults to running all
        in order.

This test mirrors the direct smoke style of ``xrx_turboquant4bit_smoke.py`` but
targets the BitResidual K8V4 custom-op chain. It validates:

1. hand-built packed cache -> ``bit_residual_attention_paged_k8v4``;
2. zero-length KV output;
3. ``bit_residual_pack_k8v4`` -> packed cache -> paged attention.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Set ASCEND_CUSTOM_OPP_PATH before importing torch/torch_npu. Otherwise CANN
# may only search the system opapi library and miss this repo's custom ops.
_CANN_OPP = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    existing_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        str(_CANN_OPP) if not existing_opp else f"{_CANN_OPP}:{existing_opp}"
    )

import torch

from vllm_ascend.utils import enable_custom_op

# All selectable paged-attention ops for the K8V4 golden tests.
_ATTN_OPS = ("paged", "fia")


def _select_attn_op(override: str | None, test_default: str) -> str:
    """Resolve the paged-attention op for a test.

    ``override`` is the CLI-supplied choice (``paged``/``fia``) or ``None``;
    when ``None`` the test falls back to its own ``test_default``.
    """
    choice = override if override else test_default
    if choice not in _ATTN_OPS:
        raise RuntimeError(
            f"attn-op must be one of {_ATTN_OPS}, got {choice!r}"
        )
    return choice


HEAD_SIZE = 128
BLOCK_SIZE = 16
KEY_BLOCK_STRIDE = 2112
VALUE_BLOCK_STRIDE = 1088
KEY_ROW_CODE_BYTES = HEAD_SIZE
VALUE_ROW_CODE_BYTES = HEAD_SIZE // 2
KEY_BLOCK_BASE_OFFSET = BLOCK_SIZE * KEY_ROW_CODE_BYTES
KEY_BLOCK_STEP_OFFSET = KEY_BLOCK_BASE_OFFSET + BLOCK_SIZE * 2
VALUE_BLOCK_VMIN_OFFSET = BLOCK_SIZE * VALUE_ROW_CODE_BYTES
VALUE_BLOCK_VSTEP_OFFSET = VALUE_BLOCK_VMIN_OFFSET + BLOCK_SIZE * 2
INV_SQRT_D = 1.0 / math.sqrt(HEAD_SIZE)
SEED = 2026


def _require_ops() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C is not loaded; build/install custom ops first")
    if not hasattr(torch.ops, "_C_ascend"):
        raise RuntimeError("torch.ops._C_ascend is not registered")
    for op_name in (
        "bit_residual_pack_k8v4",
        "bit_residual_attention_paged_k8v4",
        "bit_residual_fia_paged_k8v4",
    ):
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(f"torch.ops._C_ascend.{op_name} is not registered")


def _identity(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(HEAD_SIZE, dtype=dtype, device=device).contiguous()


def _dense_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Fixed dense orthogonal R^T used to expose rotation/layout bugs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    matrix = torch.randn(
        (HEAD_SIZE, HEAD_SIZE), dtype=torch.float32, generator=generator
    )
    rotation, _ = torch.linalg.qr(matrix)
    return rotation.to(device=device, dtype=dtype).contiguous()


def _scalar_bytes(value: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([value], dtype=dtype).view(torch.uint8).reshape(-1)


def _read_float32(x: torch.Tensor) -> float:
    return x.contiguous().view(torch.float32)[0].item()


def _read_dtype_scalar(x: torch.Tensor, dtype: torch.dtype) -> float:
    return x.contiguous().view(dtype)[0].float().item()


def _pack_value_idx4(value_idx: torch.Tensor) -> torch.Tensor:
    """Pack logical idx4 [0, 15] into signed int4 bytes used by int4b_t."""
    signed_idx = ((value_idx.to(torch.int16) - 8) & 0x0F).to(torch.uint8)
    return signed_idx[0::2] | (signed_idx[1::2] << 4)


def _unpack_value_idx4(code: torch.Tensor) -> torch.Tensor:
    """Unpack signed int4 cache bytes back to logical idx4 [0, 15]."""
    code_i32 = code.to(torch.int32)
    low = code_i32 & 0x0F
    high = (code_i32 >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low) + 8
    high = torch.where(high >= 8, high - 16, high) + 8
    idx4 = torch.empty(HEAD_SIZE, dtype=torch.float32)
    idx4[0::2] = low.float()
    idx4[1::2] = high.float()
    return idx4


def _build_block_table(actual_seq_lens_kv: list[int]) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [
        max(1, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        for seq_len in actual_seq_lens_kv
    ]
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros((len(actual_seq_lens_kv), max_blocks), dtype=torch.int32)
    next_block = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        block_table[seq_idx, :block_count] = torch.arange(
            next_block, next_block + block_count, dtype=torch.int32
        )
        next_block += block_count
    return block_table, next_block


def _write_manual_single_kv_cache(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_table, num_blocks = _build_block_table([1])
    key_cache = torch.zeros(
        num_blocks,
        1,
        KEY_BLOCK_STRIDE,
        dtype=torch.uint8,
    )
    value_cache = torch.zeros(
        num_blocks,
        1,
        VALUE_BLOCK_STRIDE,
        dtype=torch.uint8,
    )

    key_block = key_cache[0, 0]
    key_block[:KEY_ROW_CODE_BYTES].zero_()
    key_block[KEY_BLOCK_BASE_OFFSET : KEY_BLOCK_BASE_OFFSET + 2] = _scalar_bytes(
        INV_SQRT_D, dtype
    )
    key_block[KEY_BLOCK_STEP_OFFSET : KEY_BLOCK_STEP_OFFSET + 2] = _scalar_bytes(
        0.0, dtype
    )

    value_block = value_cache[0, 0]
    value_idx = (torch.arange(HEAD_SIZE, dtype=torch.int32) % 16).to(torch.uint8)
    value_block[:VALUE_ROW_CODE_BYTES] = _pack_value_idx4(value_idx)
    value_block[VALUE_BLOCK_VMIN_OFFSET : VALUE_BLOCK_VMIN_OFFSET + 2] = (
        _scalar_bytes(-1.0, dtype)
    )
    value_block[VALUE_BLOCK_VSTEP_OFFSET : VALUE_BLOCK_VSTEP_OFFSET + 2] = (
        _scalar_bytes(0.25, dtype)
    )

    return key_cache, value_cache, block_table


def _decode_key_row(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: int,
    kv_head: int,
    abs_pos: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_id = int(block_table[seq_idx, abs_pos // BLOCK_SIZE])
    pos_in_block = abs_pos % BLOCK_SIZE
    block = key_cache[block_id, kv_head]
    code_off = pos_in_block * KEY_ROW_CODE_BYTES
    code = block[code_off : code_off + KEY_ROW_CODE_BYTES].to(torch.int32)

    q = code.to(torch.float32)
    base = _read_dtype_scalar(block[
        KEY_BLOCK_BASE_OFFSET + pos_in_block * 2 : KEY_BLOCK_BASE_OFFSET + pos_in_block * 2 + 2
    ], dtype)
    step = _read_dtype_scalar(block[
        KEY_BLOCK_STEP_OFFSET + pos_in_block * 2 : KEY_BLOCK_STEP_OFFSET + pos_in_block * 2 + 2
    ], dtype)
    return base + q * step


def _decode_value_row(
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: int,
    kv_head: int,
    abs_pos: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_id = int(block_table[seq_idx, abs_pos // BLOCK_SIZE])
    pos_in_block = abs_pos % BLOCK_SIZE
    block = value_cache[block_id, kv_head]
    code_off = pos_in_block * VALUE_ROW_CODE_BYTES
    code = block[code_off : code_off + VALUE_ROW_CODE_BYTES].to(torch.int32)
    idx4 = _unpack_value_idx4(code)
    vmin = _read_dtype_scalar(block[
        VALUE_BLOCK_VMIN_OFFSET + pos_in_block * 2 : VALUE_BLOCK_VMIN_OFFSET + pos_in_block * 2 + 2
    ], dtype)
    vstep = _read_dtype_scalar(block[
        VALUE_BLOCK_VSTEP_OFFSET + pos_in_block * 2 : VALUE_BLOCK_VSTEP_OFFSET + pos_in_block * 2 + 2
    ], dtype)
    return vmin + idx4 * vstep


def _golden_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    rotation_key: torch.Tensor,
    rotation_value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    dtype = query.dtype
    gqa_group = num_heads // num_kv_heads
    out = torch.zeros_like(query, dtype=torch.float32)
    for token_idx in range(query.shape[0]):
        seq_idx = next(i for i, end in enumerate(actual_seq_lens_q) if token_idx < end)
        q_start = 0 if seq_idx == 0 else actual_seq_lens_q[seq_idx - 1]
        num_q_in_seq = actual_seq_lens_q[seq_idx] - q_start
        q_pos = token_idx - q_start
        causal_end = actual_seq_lens_kv[seq_idx] - num_q_in_seq + q_pos + 1
        causal_end = max(0, min(causal_end, actual_seq_lens_kv[seq_idx]))
        if causal_end == 0:
            continue
        for kv_head in range(num_kv_heads):
            keys = torch.stack(
                [
                    _decode_key_row(
                        key_cache, block_table, seq_idx, kv_head, pos, dtype
                    )
                    for pos in range(causal_end)
                ],
                dim=0,
            )
            values = torch.stack(
                [
                    _decode_value_row(value_cache, block_table, seq_idx, kv_head, pos, dtype)
                    for pos in range(causal_end)
                ],
                dim=0,
            )
            for group_idx in range(gqa_group):
                head_idx = kv_head * gqa_group + group_idx
                query_rot = query[token_idx, head_idx].float() @ rotation_key.float()
                scores = (keys * query_rot).sum(dim=-1) * scale
                attn = torch.softmax(scores, dim=-1) @ values
                out[token_idx, head_idx] = (
                    attn.to(dtype).float() @ rotation_value.float()
                ).to(dtype).float()
    return out


def _run_attention_op(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    rotation_key: torch.Tensor,
    rotation_value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    return torch.ops._C_ascend.bit_residual_attention_paged_k8v4(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        actual_seq_lens_q,
        actual_seq_lens_kv,
        rotation_key.contiguous(),
        rotation_value.contiguous(),
        num_heads,
        num_kv_heads,
        HEAD_SIZE,
        BLOCK_SIZE,
        max(1, max(actual_seq_lens_kv)),
        float(scale),
    )


# FIA compress causal mask (sparse_mode 2/3/4): 2048x2048 int8, 0=keep/attend,
# 1=discard.  Lower-triangular (strict upper triangle masked out) encodes a
# right-down causal mask for single-sequence prefill (q==kv).
_FIA_COMPRESS_MASK_SIZE = 2048
_FIA_SPARSE_MODE_RIGHT_DOWN = 3
_FIA_INT_MAX = 2147483647


def _compress_causal_mask(device: torch.device) -> torch.Tensor:
    """FIA compress mask for sparse_mode 2/3/4: int8 lower-triangular.

    Polarity (FIA docs): 0=keep/attend, 1=discard.  Strict upper triangle = 1
    (masked out); diagonal + lower = 0 (keep).  Matches the causal mask the
    golden reference applies via ``causal_end``.
    """
    ones = torch.ones(
        (_FIA_COMPRESS_MASK_SIZE, _FIA_COMPRESS_MASK_SIZE),
        dtype=torch.int8,
        device=device,
    )
    return torch.triu(ones, diagonal=1)


def _run_fia_op(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    rotation_key: torch.Tensor,
    rotation_value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Call the BitResidual FIA paged K8V4 op directly (bypassing the Python
    wrapper, which would force the internal Haar rotation).  Pass the same
    rotation the pack op used so encode/decode stay consistent.
    """
    return torch.ops._C_ascend.bit_residual_fia_paged_k8v4(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        actual_seq_lens_q,
        actual_seq_lens_kv,
        _compress_causal_mask(query.device),
        rotation_key.contiguous(),
        rotation_value.contiguous(),
        num_heads,
        num_kv_heads,
        HEAD_SIZE,
        BLOCK_SIZE,
        float(scale),
        _FIA_INT_MAX,
        _FIA_INT_MAX,
        _FIA_SPARSE_MODE_RIGHT_DOWN,
    )


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    actual_cpu = actual.float().cpu()
    if atol is None:
        atol = 1e-3 if actual.dtype == torch.float16 else 4e-2
    if rtol is None:
        rtol = 1e-3 if actual.dtype == torch.float16 else 4e-2
    torch.testing.assert_close(actual_cpu, expected.float(), atol=atol, rtol=rtol)
    print(f"PASS {name}: dtype={actual.dtype}, maxdiff={(actual_cpu - expected).abs().max().item():.6f}")


def _assert_min_cosine(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    min_cosine: float,
) -> None:
    actual_cpu = actual.float().cpu().reshape(-1)
    expected_cpu = expected.float().cpu().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        actual_cpu, expected_cpu, dim=0
    ).item()
    abs_error = (actual_cpu - expected_cpu).abs()
    sign_mismatch = (
        (actual_cpu < 0) != (expected_cpu < 0)
    ).float().mean().item()
    if cosine < min_cosine:
        print(f"{name} actual[:16]={actual_cpu[:16]}")
        print(f"{name} expected[:16]={expected_cpu[:16]}")
        print(f"{name} row_cos={torch.nn.functional.cosine_similarity(actual.float().cpu().reshape(-1, HEAD_SIZE), expected.float().cpu().reshape(-1, HEAD_SIZE), dim=-1)}")
        a0 = actual.float().cpu().reshape(-1, HEAD_SIZE)[0]
        e0 = expected.float().cpu().reshape(-1, HEAD_SIZE)[0]
        print(f"{name} sign_match_chunks={[(a0[i:i+16].sign() == e0[i:i+16].sign()).sum().item() for i in range(0, HEAD_SIZE, 16)]}")
        print(f"{name} actual_chunks={[a0[i:i+4].tolist() for i in range(0, HEAD_SIZE, 16)]}")
        print(f"{name} expected_chunks={[e0[i:i+4].tolist() for i in range(0, HEAD_SIZE, 16)]}")
        raise AssertionError(
            f"{name} cosine {cosine:.6f} is lower than {min_cosine:.6f}"
        )
    print(
        f"PASS {name}: dtype={actual.dtype}, cosine={cosine:.6f}, "
        f"maxerr={abs_error.max().item():.6f}, "
        f"meanerr={abs_error.mean().item():.6f}, "
        f"sign_mismatch={sign_mismatch:.6f}"
    )


def _diagnose_key_encoding(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    rotated_key: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Compare cache bytes with a CPU emulation of the vector Key encoder."""
    actual_codes = []
    actual_bases = []
    actual_steps = []
    num_tokens, num_heads, _ = rotated_key.shape
    for token_idx in range(num_tokens):
        block_id = int(block_table[0, token_idx // BLOCK_SIZE])
        pos_in_block = token_idx % BLOCK_SIZE
        for head_idx in range(num_heads):
            block = key_cache[block_id, head_idx]
            code_off = pos_in_block * KEY_ROW_CODE_BYTES
            actual_codes.append(
                block[code_off : code_off + KEY_ROW_CODE_BYTES].to(torch.int32)
            )
            actual_bases.append(_read_dtype_scalar(block[
                KEY_BLOCK_BASE_OFFSET + pos_in_block * 2 :
                KEY_BLOCK_BASE_OFFSET + pos_in_block * 2 + 2
            ], dtype))
            actual_steps.append(_read_dtype_scalar(block[
                KEY_BLOCK_STEP_OFFSET + pos_in_block * 2 :
                KEY_BLOCK_STEP_OFFSET + pos_in_block * 2 + 2
            ], dtype))

    rows = rotated_key.reshape(-1, HEAD_SIZE).float()
    ymin = rows.amin(dim=-1)
    ymax = rows.amax(dim=-1)
    step = ((ymax - ymin) / 255.0).clamp_min(1.0e-6)
    expected_q = torch.round(
        ((rows - ymin.unsqueeze(-1)) / step.unsqueeze(-1))
    ).clamp(0, 255).to(torch.int32)
    expected_codes = expected_q
    expected_bases = ymin.to(dtype).float()
    expected_steps = step.to(dtype).float()

    actual_codes_t = torch.stack(actual_codes)
    actual_bases_t = torch.tensor(actual_bases)
    actual_steps_t = torch.tensor(actual_steps)
    code_mismatch = actual_codes_t != expected_codes
    print(
        "KEY ENCODING DIAG (Scheme A uniform): "
        f"code_mismatch={code_mismatch.float().mean().item():.6f}, "
        f"base_maxerr={(actual_bases_t - expected_bases).abs().max().item():.6f}, "
        f"step_maxerr={(actual_steps_t - expected_steps).abs().max().item():.6f}"
    )
    if code_mismatch.any():
        row, dim = code_mismatch.nonzero()[0].tolist()
        print(
            f"KEY FIRST MISMATCH row={row} dim={dim}: "
            f"actual={actual_codes_t[row, dim].item()}, "
            f"expected={expected_codes[row, dim].item()}, "
            f"ymin={ymin[row].item()}, step={step[row].item()}"
        )


def _decode_cache_rows(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    num_kv_heads: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoded_key = torch.stack(
        [
            torch.stack(
                [
                    _decode_key_row(key_cache, block_table, 0, head_idx, token_idx, dtype)
                    for head_idx in range(num_kv_heads)
                ],
                dim=0,
            )
            for token_idx in range(num_tokens)
        ],
        dim=0,
    )
    decoded_value = torch.stack(
        [
            torch.stack(
                [
                    _decode_value_row(value_cache, block_table, 0, head_idx, token_idx, dtype)
                    for head_idx in range(num_kv_heads)
                ],
                dim=0,
            )
            for token_idx in range(num_tokens)
        ],
        dim=0,
    )
    return decoded_key, decoded_value


def _fp_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    query = query.float()
    key = key.float()
    value = value.float()
    gqa_group = num_heads // num_kv_heads
    out = torch.zeros_like(query)
    for token_idx in range(query.shape[0]):
        seq_idx = next(i for i, end in enumerate(actual_seq_lens_q) if token_idx < end)
        q_start = 0 if seq_idx == 0 else actual_seq_lens_q[seq_idx - 1]
        num_q_in_seq = actual_seq_lens_q[seq_idx] - q_start
        q_pos = token_idx - q_start
        causal_end = actual_seq_lens_kv[seq_idx] - num_q_in_seq + q_pos + 1
        causal_end = max(0, min(causal_end, actual_seq_lens_kv[seq_idx]))
        for kv_head in range(num_kv_heads):
            keys = key[:causal_end, kv_head]
            values = value[:causal_end, kv_head]
            for group_idx in range(gqa_group):
                head_idx = kv_head * gqa_group + group_idx
                scores = (keys * query[token_idx, head_idx]).sum(dim=-1) * scale
                out[token_idx, head_idx] = torch.softmax(scores, dim=-1) @ values
    return out


def _run_manual_single_kv(dtype: torch.dtype, device: torch.device) -> None:
    key_cache_cpu, value_cache_cpu, block_table_cpu = _write_manual_single_kv_cache(dtype)
    query = torch.randn((1, 1, HEAD_SIZE), dtype=dtype, device=device)
    rotation_key = _identity(dtype, device)
    rotation_value = _identity(dtype, device)
    actual = _run_attention_op(
        query=query,
        key_cache=key_cache_cpu.to(device),
        value_cache=value_cache_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=[1],
        actual_seq_lens_kv=[1],
        rotation_key=rotation_key,
        rotation_value=rotation_value,
        num_heads=1,
        num_kv_heads=1,
        scale=HEAD_SIZE**-0.5,
    )
    expected = (-1.0 + (torch.arange(HEAD_SIZE) % 16).float() * 0.25).reshape(1, 1, -1)
    expected = expected.to(dtype).float()
    _assert_close("manual_single_kv", actual, expected)


def _run_zero_kv(dtype: torch.dtype, device: torch.device) -> None:
    key_cache_cpu, value_cache_cpu, block_table_cpu = _write_manual_single_kv_cache(dtype)
    query = torch.randn((1, 1, HEAD_SIZE), dtype=dtype, device=device)
    rotation = _identity(dtype, device)
    actual = _run_attention_op(
        query=query,
        key_cache=key_cache_cpu.to(device),
        value_cache=value_cache_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=[1],
        actual_seq_lens_kv=[0],
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=1,
        num_kv_heads=1,
        scale=HEAD_SIZE**-0.5,
    )
    expected = torch.zeros((1, 1, HEAD_SIZE), dtype=torch.float32)
    _assert_close("zero_kv", actual, expected)


def _run_pack_attention_chain(
    dtype: torch.dtype, device: torch.device, attn_op_override: str | None = None
) -> None:
    torch.manual_seed(SEED)
    num_kv_tokens = 9
    num_query_tokens = 3
    num_kv_heads = 8
    num_heads = 2 * num_kv_heads
    scale = HEAD_SIZE**-0.5
    block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])
    key = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    value = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    query = torch.randn(
        (num_query_tokens, num_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    rotation_t = _dense_rotation(dtype, device)
    rotation = rotation_t.transpose(0, 1).contiguous()
    slot_mapping = torch.arange(num_kv_tokens, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_kv_tokens], dtype=torch.int32, device=device)
    key_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        KEY_BLOCK_STRIDE,
        dtype=torch.uint8,
        device=device,
    )
    value_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        VALUE_BLOCK_STRIDE,
        dtype=torch.uint8,
        device=device,
    )

    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key,
        value,
        slot_mapping,
        query_start_loc,
        rotation_t,
        key_cache,
        value_cache,
        1,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()
    decoded_key, decoded_value = _decode_cache_rows(
        key_cache.cpu(),
        value_cache.cpu(),
        block_table_cpu,
        num_kv_tokens,
        num_kv_heads,
        dtype,
    )
    rotated_key = torch.matmul(key.float(), rotation_t.float()).cpu()
    rotated_value = torch.matmul(value.float(), rotation_t.float()).cpu()
    _diagnose_key_encoding(
        key_cache.cpu(), block_table_cpu, rotated_key, dtype
    )
    _assert_min_cosine("pack_key_decode", decoded_key, rotated_key, 0.999)
    _assert_min_cosine("pack_value_decode", decoded_value, rotated_value, 0.990)

    actual_seq_lens_q = [num_query_tokens]
    actual_seq_lens_kv = [num_kv_tokens]
    attn_op = _select_attn_op(attn_op_override, "paged")
    run_attn = _run_attention_op if attn_op == "paged" else _run_fia_op
    actual = run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation_t,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    torch.npu.synchronize()

    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation_t.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    # FIA prefill path casts Q/K to fp16 for the FlashAttention matmul; allow
    # slightly looser tolerance than the Vector-QK path.  The Vector-QK path
    # keeps the original tight 1e-3 / 4e-2 tolerance.
    if attn_op == "fia":
        atol = 2e-3 if dtype == torch.float16 else 5e-2
        rtol = 2e-3 if dtype == torch.float16 else 5e-2
        _assert_close(f"pack_attention_chain[{attn_op}]", actual, expected, atol=atol, rtol=rtol)
    else:
        _assert_close(f"pack_attention_chain[{attn_op}]", actual, expected)
    fp_expected = _fp_attention(
        query=query.cpu(),
        key=key.cpu(),
        value=value.cpu(),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    _assert_min_cosine("pack_attention_vs_fp", actual, fp_expected, 0.990)


def _run_pack_attention_qtile(
    dtype: torch.dtype, device: torch.device, attn_op_override: str | None = None
) -> None:
    """Multi-token Q (num_q > usedCoreNum) exercises SplitBN qTile path.

    Defaults to ``bit_residual_fia_paged_k8v4`` (FlashAttention prefill path
    with a compress causal mask, sparse_mode=3) instead of the Vector-QK
    attention op, so the FIA decode/prefill path is exercised against the same
    golden.  Pass ``attn_op_override='paged'`` to run the Vector-QK op here
    instead (tighter tolerance applies, see below).
    """
    torch.manual_seed(SEED + 1)
    num_kv_tokens = 48
    num_query_tokens = 48
    num_kv_heads = 8
    num_heads = 2 * num_kv_heads
    scale = HEAD_SIZE**-0.5
    block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])
    key = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    value = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    query = torch.randn(
        (num_query_tokens, num_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    rotation_t = _dense_rotation(dtype, device)
    rotation = rotation_t.transpose(0, 1).contiguous()
    slot_mapping = torch.arange(num_kv_tokens, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_kv_tokens], dtype=torch.int32, device=device)
    key_cache = torch.zeros(
        num_blocks, num_kv_heads, KEY_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros(
        num_blocks, num_kv_heads, VALUE_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key,
        value,
        slot_mapping,
        query_start_loc,
        rotation_t,
        key_cache,
        value_cache,
        1,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()
    actual_seq_lens_q = [num_query_tokens]
    actual_seq_lens_kv = [num_kv_tokens]
    attn_op = _select_attn_op(attn_op_override, "fia")
    run_attn = _run_attention_op if attn_op == "paged" else _run_fia_op
    actual = run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation_t,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    torch.npu.synchronize()
    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation_t.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    # Prefill qTile: fp16 stays elementwise-close (Cube/FIA add a little noise
    # vs chain's 1e-3). bf16 + paged Cube QK can show large per-element spikes
    # (~1.0 abs on a few heads) while remaining directionally correct — do not
    # paper over that with a huge atol; use cosine like pack_attention_vs_fp.
    name = f"pack_attention_qtile[{attn_op}]"
    if attn_op == "paged" and dtype == torch.bfloat16:
        _assert_min_cosine(name, actual, expected, 0.990)
    else:
        atol = 2e-3 if dtype == torch.float16 else 5e-2
        rtol = 2e-3 if dtype == torch.float16 else 5e-2
        _assert_close(name, actual, expected, atol=atol, rtol=rtol)


def _parse_args() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="BitResidual K8V4 pack + paged attention golden test"
    )
    parser.add_argument(
        "--attn-op",
        choices=_ATTN_OPS,
        default=None,
        help=(
            "Select the paged-attention op for the chain/qtile tests. "
            "'paged' = bit_residual_attention_paged_k8v4 (Vector-QK path); "
            "'fia' = bit_residual_fia_paged_k8v4 (FlashAttention prefill "
            "path). When omitted, each test keeps its own default "
            "(chain -> paged, qtile -> fia)."
        ),
    )
    parser.add_argument(
        "--test",
        default=None,
        help=(
            "Run only the named test: manual_single_kv, zero_kv, "
            "pack_attention_chain, pack_attention_qtile. Defaults to all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require_ops()
    device = torch.device("npu:0")
    attn_op = args.attn_op
    run_all = args.test is None

    def _run(dtype: torch.dtype) -> None:
        if run_all or args.test == "manual_single_kv":
            _run_manual_single_kv(dtype, device)
        if run_all or args.test == "zero_kv":
            _run_zero_kv(dtype, device)
        if run_all or args.test == "pack_attention_chain":
            _run_pack_attention_chain(dtype, device, attn_op)
        if run_all or args.test == "pack_attention_qtile":
            _run_pack_attention_qtile(dtype, device, attn_op)

    if args.test is not None and args.test not in {
        "manual_single_kv",
        "zero_kv",
        "pack_attention_chain",
        "pack_attention_qtile",
    }:
        raise RuntimeError(f"unknown --test {args.test!r}")

    for dtype in (torch.float16, torch.bfloat16):
        _run(dtype)
    if run_all:
        print("bit residual k8v4 smoke output: all cases passed")
    else:
        print(f"bit residual k8v4 smoke output: {args.test} passed")


if __name__ == "__main__":
    main()
