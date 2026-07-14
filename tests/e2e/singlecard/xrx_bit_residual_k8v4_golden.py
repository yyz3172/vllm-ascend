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

    python tests/e2e/singlecard/xrx_bit_residual_k8v4_smoke.py

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
    for op_name in ("bit_residual_pack_k8v4", "bit_residual_attention_paged_k8v4"):
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
    value_block[:VALUE_ROW_CODE_BYTES] = (
        value_idx[0::2] | (value_idx[1::2] << 4)
    )
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

    q7 = (code & 0x7F).float()
    sign = (code >> 7).float()
    sign_val = torch.where(sign == 0, 1.0, -1.0)
    base = _read_dtype_scalar(block[
        KEY_BLOCK_BASE_OFFSET + pos_in_block * 2 : KEY_BLOCK_BASE_OFFSET + pos_in_block * 2 + 2
    ], dtype)
    step = _read_dtype_scalar(block[
        KEY_BLOCK_STEP_OFFSET + pos_in_block * 2 : KEY_BLOCK_STEP_OFFSET + pos_in_block * 2 + 2
    ], dtype)
    return (base + q7 * step) * sign_val


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
    idx4 = torch.empty(HEAD_SIZE, dtype=torch.float32)
    idx4[0::2] = (code & 0x0F).float()
    idx4[1::2] = (code >> 4).float()
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


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.float().cpu()
    atol = 1e-3 if actual.dtype == torch.float16 else 4e-2
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
    abs_rows = rows.abs()
    abs_max = abs_rows.amax(dim=-1)
    normalized = (abs_rows / abs_max.clamp_min(1.0e-12).unsqueeze(-1)).half()
    base_norm = normalized.amin(dim=-1)
    step_norm = ((torch.ones_like(base_norm) - base_norm) / 127.0).half()
    safe_step = step_norm.clamp_min(torch.tensor(1.0e-6, dtype=torch.float16))
    expected_q = torch.round(
        ((normalized - base_norm.unsqueeze(-1)) / safe_step.unsqueeze(-1)).float()
    ).clamp(0, 127).to(torch.int32)
    expected_sign = (rows < 0).to(torch.int32)
    expected_codes = expected_q | (expected_sign << 7)
    expected_bases = (base_norm * abs_max.half()).to(dtype).float()
    expected_steps = (step_norm * abs_max.half()).to(dtype).float()

    actual_codes_t = torch.stack(actual_codes)
    actual_bases_t = torch.tensor(actual_bases)
    actual_steps_t = torch.tensor(actual_steps)
    code_mismatch = actual_codes_t != expected_codes
    q_mismatch = (actual_codes_t & 0x7F) != expected_q
    sign_mismatch = (actual_codes_t >> 7) != expected_sign
    print(
        "KEY ENCODING DIAG: "
        f"code_mismatch={code_mismatch.float().mean().item():.6f}, "
        f"q_mismatch={q_mismatch.float().mean().item():.6f}, "
        f"sign_mismatch={sign_mismatch.float().mean().item():.6f}, "
        f"base_maxerr={(actual_bases_t - expected_bases).abs().max().item():.6f}, "
        f"step_maxerr={(actual_steps_t - expected_steps).abs().max().item():.6f}"
    )
    if code_mismatch.any():
        row, dim = code_mismatch.nonzero()[0].tolist()
        print(
            f"KEY FIRST MISMATCH row={row} dim={dim}: "
            f"actual={actual_codes_t[row, dim].item()}, "
            f"expected={expected_codes[row, dim].item()}, "
            f"normalized={normalized[row, dim].item()}, "
            f"base={base_norm[row].item()}, step={step_norm[row].item()}"
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


def _run_pack_attention_chain(dtype: torch.dtype, device: torch.device) -> None:
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
    actual = _run_attention_op(
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
    _assert_close("pack_attention_chain", actual, expected)
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


def main() -> None:
    _require_ops()
    device = torch.device("npu:0")
    for dtype in (torch.float16, torch.bfloat16):
        _run_manual_single_kv(dtype, device)
        _run_zero_kv(dtype, device)
        _run_pack_attention_chain(dtype, device)
    print("bit residual k8v4 smoke output: all cases passed")


if __name__ == "__main__":
    main()
