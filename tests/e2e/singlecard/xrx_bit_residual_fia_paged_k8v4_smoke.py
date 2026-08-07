#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# BitResidual FIA Paged K8V4 correctness smoke.
#
# Validates:
# 1. FIA vs CPU golden (_golden_attention)
# 2. FIA vs bit_residual_attention_paged_k8v4
# 3. decode / GQA / non-identity rotation / pack chain
# 4. Prefill: multi-Q + sparse_mode=3 compress causal mask
# 5. FlashDecode: long KV (kv>~s2Base=512) tiling key 1 path
# 6. bf16: pack meta + query/rotation (serving dtype for Qwen bf16)
# 7. mm1 S2 Align32 workspace: KV lens with S2%32!=0 (host must size WS to 32)
#
# Run on NPU after building custom ops:
#   bash script/lcy/bit_residual_fia_paged_k8v4/rebuild_op.sh
#   python tests/e2e/singlecard/xrx_bit_residual_fia_paged_k8v4_smoke.py

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

_CANN_OPP = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    existing_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        str(_CANN_OPP) if not existing_opp else f"{_CANN_OPP}:{existing_opp}"
    )

import torch

from vllm_ascend.utils import enable_custom_op

from tests.e2e.singlecard.xrx_bit_residual_k8v4_golden import (
    BLOCK_SIZE,
    HEAD_SIZE,
    KEY_BLOCK_STRIDE,
    SEED,
    VALUE_BLOCK_STRIDE,
    _assert_close,
    _assert_min_cosine,
    _build_block_table,
    _decode_cache_rows,
    _dense_rotation,
    _golden_attention,
    _identity,
    _write_manual_single_kv_cache,
)

SCALE = HEAD_SIZE**-0.5
FIA_VS_ATTN_ATOL = 2e-3
FIA_VS_GOLDEN_ATOL = 2e-3
# bf16: pack/dequant + FIA accumulate; measured max ~7.8e-3 on smoke shapes.
FIA_VS_ATTN_ATOL_BF16 = 1.5e-2
FIA_VS_GOLDEN_ATOL_BF16 = 1.5e-2
# FIA compress causal/band mask (sparse_mode 2/3/4): 2048x2048, 0=keep.
COMPRESS_MASK_SIZE = 2048
SPARSE_MODE_RIGHT_DOWN = 3
INT_MAX = 2147483647
# s2Base=512; kv beyond this can enable FlashDecode when SplitCore splits S2.
FD_KV_TOKENS = 1000
# Host mm1/vec1 WS must AlignUp(S2, BYTE_BLOCK=32). Align16 under-sizes when
# S2%32!=0 (e.g. 144→host 144 vs device stride 160) → Fixpipe OOB → NaN.
MM1_ALIGN32_DIRTY_KV = (144, 168, 176, 200)
MM1_ALIGN32_CLEAN_KV = (160, 192)
MM1_ALIGN32_REPEATS = 8


def _require_npu() -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C is not loaded; build/install custom ops first")
    for op_name in (
        "bit_residual_pack_k8v4",
        "bit_residual_attention_paged_k8v4",
        "bit_residual_fia_paged_k8v4",
    ):
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(f"missing custom op: {op_name}")
    return torch.device("npu")


def _run_attn(
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
        float(SCALE),
    )


def _compress_causal_mask(device: torch.device) -> torch.Tensor:
    """FIA compress mask for sparse_mode 2/3/4: 2048x2048 int8 lower-triangular.

    Polarity (FIA docs): 0=keep/attend, 1=discard. Must be lower-triangular;
    an all-zero mask is a no-op for multi-Q (matches full attention).
    """
    ones = torch.ones(
        (COMPRESS_MASK_SIZE, COMPRESS_MASK_SIZE), dtype=torch.int8, device=device
    )
    # Upper triangle (strict) = 1 (masked out); diagonal+lower = 0 (keep).
    return torch.triu(ones, diagonal=1)


def _run_fia(
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
    atten_mask: torch.Tensor | None = None,
    sparse_mode: int = 0,
    pre_tokens: int = INT_MAX,
    next_tokens: int = INT_MAX,
) -> torch.Tensor:
    return torch.ops._C_ascend.bit_residual_fia_paged_k8v4(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        actual_seq_lens_q,
        actual_seq_lens_kv,
        atten_mask,
        rotation_key.contiguous(),
        rotation_value.contiguous(),
        num_heads,
        num_kv_heads,
        HEAD_SIZE,
        BLOCK_SIZE,
        float(SCALE),
        pre_tokens,
        next_tokens,
        sparse_mode,
    )


def _pack_kv_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    rotation_t: torch.Tensor,
    num_blocks: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = key.device
    num_kv_tokens = key.shape[0]
    key_cache = torch.zeros(
        num_blocks, num_kv_heads, KEY_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros(
        num_blocks, num_kv_heads, VALUE_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    torch.ops._C_ascend.bit_residual_pack_k8v4(
        key,
        value,
        torch.arange(num_kv_tokens, dtype=torch.int32, device=device),
        torch.tensor([0, num_kv_tokens], dtype=torch.int32, device=device),
        rotation_t.contiguous(),
        key_cache,
        value_cache,
        1,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()
    return key_cache, value_cache


def _compare_tria(
    name: str,
    out_fia: torch.Tensor,
    out_attn: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol_golden: float | None = None,
    atol_attn: float | None = None,
) -> None:
    """Compare FIA vs golden and FIA vs attn (CPU tensors for golden)."""
    if atol_golden is None or atol_attn is None:
        if out_fia.dtype == torch.bfloat16:
            atol_golden = FIA_VS_GOLDEN_ATOL_BF16 if atol_golden is None else atol_golden
            atol_attn = FIA_VS_ATTN_ATOL_BF16 if atol_attn is None else atol_attn
        else:
            atol_golden = FIA_VS_GOLDEN_ATOL if atol_golden is None else atol_golden
            atol_attn = FIA_VS_ATTN_ATOL if atol_attn is None else atol_attn

    fia_cpu = out_fia.float().cpu()
    attn_cpu = out_attn.float().cpu()
    exp_cpu = expected.float().cpu()

    diff_golden = (fia_cpu - exp_cpu).abs().max().item()
    diff_attn = (fia_cpu - attn_cpu).abs().max().item()
    print(
        f"{name}: fia_vs_golden={diff_golden:.6f}, fia_vs_attn={diff_attn:.6f}"
    )
    if diff_golden > atol_golden:
        raise AssertionError(
            f"{name}: FIA vs golden max_diff {diff_golden} exceeds {atol_golden}"
        )
    if diff_attn > atol_attn:
        raise AssertionError(
            f"{name}: FIA vs attn max_diff {diff_attn} exceeds {atol_attn}"
        )
    print(f"PASS {name}")


def test_manual_single_kv(device: torch.device) -> None:
    """Hand-built 1-token cache; golden is closed-form value decode."""
    dtype = torch.float16
    key_cache_cpu, value_cache_cpu, block_table_cpu = _write_manual_single_kv_cache(dtype)
    query = torch.randn((1, 1, HEAD_SIZE), dtype=dtype, device=device)
    rotation_key = _identity(dtype, device)
    rotation_value = _identity(dtype, device)
    actual_seq_lens_q = [1]
    actual_seq_lens_kv = [1]

    out_attn = _run_attn(
        query=query,
        key_cache=key_cache_cpu.to(device),
        value_cache=value_cache_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation_key,
        rotation_value=rotation_value,
        num_heads=1,
        num_kv_heads=1,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache_cpu.to(device),
        value_cache=value_cache_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation_key,
        rotation_value=rotation_value,
        num_heads=1,
        num_kv_heads=1,
    )
    # Manual cache: K base=1/sqrt(D), step=0 → key=const; V=vmin+idx*vstep.
    # With identity R and single KV, attn out ≈ decoded V (softmax trivial).
    expected = (-1.0 + (torch.arange(HEAD_SIZE) % 16).float() * 0.25).reshape(1, 1, -1)
    _compare_tria("manual_single_kv", out_fia, out_attn, expected)


def test_decode_multi_kv_gqa(device: torch.device) -> None:
    """Decode-style q=1, kv crosses blocks, GQA 8/2, identity rotation."""
    dtype = torch.float16
    torch.manual_seed(SEED)
    num_kv_tokens = 32  # 2 blocks
    num_kv_heads = 2
    num_heads = 8
    block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])

    key = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    value = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    query = torch.randn((1, num_heads, HEAD_SIZE), dtype=dtype, device=device).contiguous()
    rotation = _identity(dtype, device)
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
        rotation,  # R^T = I
        key_cache,
        value_cache,
        1,
        BLOCK_SIZE,
    )
    torch.npu.synchronize()

    actual_seq_lens_q = [1]
    actual_seq_lens_kv = [num_kv_tokens]
    out_attn = _run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=SCALE,
    )
    _compare_tria("decode_multi_kv_gqa", out_fia, out_attn, expected)


def test_pack_fia_chain_dense_rotation(device: torch.device) -> None:
    """pack -> FIA/attn with dense orthogonal rotations (GQA multi-KV)."""
    dtype = torch.float16
    torch.manual_seed(SEED)
    num_kv_tokens = 9
    num_query_tokens = 1  # decode-style: avoid causal/mask mismatch with sparse_mode=0
    num_kv_heads = 4
    num_heads = 8
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
    rotation_t = _dense_rotation(dtype, device)  # R^T for pack / Q side
    rotation = rotation_t.transpose(0, 1).contiguous()  # R for O side
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
    _assert_min_cosine("pack_key_decode", decoded_key, rotated_key, 0.999)
    _assert_min_cosine("pack_value_decode", decoded_value, rotated_value, 0.990)

    actual_seq_lens_q = [num_query_tokens]
    actual_seq_lens_kv = [num_kv_tokens]
    out_attn = _run_attn(
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
    )
    out_fia = _run_fia(
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
        scale=SCALE,
    )
    _compare_tria("pack_fia_dense_rotation", out_fia, out_attn, expected)
    _assert_close("attn_vs_golden_dense_rotation", out_attn, expected)


def test_prefill_causal_gqa(device: torch.device) -> None:
    """Prefill: multi-Q + right-down causal (sparse_mode=3) vs attn/golden."""
    dtype = torch.float16
    torch.manual_seed(SEED)
    num_kv_tokens = 32
    num_query_tokens = 8  # chunked prefill: attend with right-down causal
    num_kv_heads = 2
    num_heads = 8
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
    rotation = _identity(dtype, device)
    key_cache, value_cache = _pack_kv_cache(
        key=key,
        value=value,
        rotation_t=rotation,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
    )

    actual_seq_lens_q = [num_query_tokens]
    actual_seq_lens_kv = [num_kv_tokens]
    atten_mask = _compress_causal_mask(device)

    out_attn = _run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        atten_mask=atten_mask,
        sparse_mode=SPARSE_MODE_RIGHT_DOWN,
    )
    torch.npu.synchronize()

    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=SCALE,
    )
    _compare_tria("prefill_causal_gqa", out_fia, out_attn, expected)


def test_prefill_full_seq_dense_rotation(device: torch.device) -> None:
    """Full-sequence prefill (q==kv) + dense R + causal mask."""
    dtype = torch.float16
    torch.manual_seed(SEED + 1)
    num_tokens = 16
    num_kv_heads = 4
    num_heads = 8
    block_table_cpu, num_blocks = _build_block_table([num_tokens])

    key = torch.randn(
        (num_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    value = torch.randn(
        (num_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    query = torch.randn(
        (num_tokens, num_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    rotation_t = _dense_rotation(dtype, device)
    rotation = rotation_t.transpose(0, 1).contiguous()
    key_cache, value_cache = _pack_kv_cache(
        key=key,
        value=value,
        rotation_t=rotation_t,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
    )

    actual_seq_lens_q = [num_tokens]
    actual_seq_lens_kv = [num_tokens]
    atten_mask = _compress_causal_mask(device)

    out_attn = _run_attn(
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
    )
    out_fia = _run_fia(
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
        atten_mask=atten_mask,
        sparse_mode=SPARSE_MODE_RIGHT_DOWN,
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
        scale=SCALE,
    )
    _compare_tria("prefill_full_seq_dense_rotation", out_fia, out_attn, expected)


def test_decode_kv_len_regression(device: torch.device) -> None:
    """Decode FIA vs attn across KV lengths that NaN'd under dual-AIV dequant.

    Mid-block element half-split (e.g. kv=76 → half=38) stresses ping-pong
    meta/codes MTE2 reuse. Regression gate: no NaN and FIA close to attn.
    """
    dtype = torch.float16
    num_kv_heads = 2
    num_heads = 8
    kv_lens_atol = [32, 48, 64, 76, 78, 80, 96]
    kv_lens_finite = [112, 128]
    repeats = 3

    for num_kv_tokens in kv_lens_atol + kv_lens_finite:
        check_atol = num_kv_tokens in kv_lens_atol
        for rep in range(repeats):
            torch.manual_seed(SEED + num_kv_tokens * 17 + rep)
            block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])
            key = torch.randn(
                (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
            ).contiguous()
            value = torch.randn(
                (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
            ).contiguous()
            query = torch.randn(
                (1, num_heads, HEAD_SIZE), dtype=dtype, device=device
            ).contiguous()
            rotation = _identity(dtype, device)
            key_cache, value_cache = _pack_kv_cache(
                key=key,
                value=value,
                rotation_t=rotation,
                num_blocks=num_blocks,
                num_kv_heads=num_kv_heads,
            )
            actual_seq_lens_q = [1]
            actual_seq_lens_kv = [num_kv_tokens]
            out_attn = _run_attn(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table_cpu.to(device),
                actual_seq_lens_q=actual_seq_lens_q,
                actual_seq_lens_kv=actual_seq_lens_kv,
                rotation_key=rotation,
                rotation_value=rotation,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            out_fia = _run_fia(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table_cpu.to(device),
                actual_seq_lens_q=actual_seq_lens_q,
                actual_seq_lens_kv=actual_seq_lens_kv,
                rotation_key=rotation,
                rotation_value=rotation,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            torch.npu.synchronize()
            if torch.isnan(out_fia).any() or torch.isnan(out_attn).any():
                raise AssertionError(
                    f"decode_kv_len_regression: NaN at kv={num_kv_tokens} rep={rep}"
                )
            diff = (out_fia.float() - out_attn.float()).abs().max().item()
            if check_atol and diff > FIA_VS_ATTN_ATOL:
                raise AssertionError(
                    f"decode_kv_len_regression: kv={num_kv_tokens} rep={rep} "
                    f"max_diff {diff} exceeds {FIA_VS_ATTN_ATOL}"
                )
    print("PASS decode_kv_len_regression")


def test_mm1_s2_align32_workspace(device: torch.device) -> None:
    """Regression: host sInnerSizeAlign must match device BYTE_BLOCK=32.

    When S2%32!=0, AlignUp(S2, 16) under-sizes mm1/vec1 workspace vs Fixpipe
    dstStride Align(S2, 32) → OOB writes → nondeterministic NaN. Dirty-band
    KV lengths must stay finite across repeats; clean (%32==0) are controls.
    """
    dtype = torch.float16
    num_kv_heads = 2
    num_heads = 8
    kv_lens = list(MM1_ALIGN32_DIRTY_KV) + list(MM1_ALIGN32_CLEAN_KV)

    for num_kv_tokens in kv_lens:
        for rep in range(MM1_ALIGN32_REPEATS):
            torch.manual_seed(SEED + 101 * num_kv_tokens + rep)
            block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])
            key = torch.randn(
                (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
            ).contiguous()
            value = torch.randn(
                (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
            ).contiguous()
            query = torch.randn(
                (1, num_heads, HEAD_SIZE), dtype=dtype, device=device
            ).contiguous()
            rotation = _identity(dtype, device)
            key_cache, value_cache = _pack_kv_cache(
                key=key,
                value=value,
                rotation_t=rotation,
                num_blocks=num_blocks,
                num_kv_heads=num_kv_heads,
            )
            actual_seq_lens_q = [1]
            actual_seq_lens_kv = [num_kv_tokens]
            out_attn = _run_attn(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table_cpu.to(device),
                actual_seq_lens_q=actual_seq_lens_q,
                actual_seq_lens_kv=actual_seq_lens_kv,
                rotation_key=rotation,
                rotation_value=rotation,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            out_fia = _run_fia(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table_cpu.to(device),
                actual_seq_lens_q=actual_seq_lens_q,
                actual_seq_lens_kv=actual_seq_lens_kv,
                rotation_key=rotation,
                rotation_value=rotation,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            torch.npu.synchronize()
            if not torch.isfinite(out_fia).all() or not torch.isfinite(out_attn).all():
                raise AssertionError(
                    f"mm1_s2_align32_workspace: non-finite at kv={num_kv_tokens} "
                    f"rep={rep} (S2%32={num_kv_tokens % 32})"
                )
            diff = (out_fia.float() - out_attn.float()).abs().max().item()
            if diff > FIA_VS_ATTN_ATOL:
                raise AssertionError(
                    f"mm1_s2_align32_workspace: kv={num_kv_tokens} rep={rep} "
                    f"max_diff {diff} exceeds {FIA_VS_ATTN_ATOL}"
                )
    print("PASS mm1_s2_align32_workspace")


def test_flash_decode_long_kv(device: torch.device) -> None:
    """Long-KV decode (kv=1000 > s2Base=512): multi-s2Base / FD-capable path.

    Exercises SplitCore S2 tiling beyond one s2Base chunk. With enough parallel
    (b,n2) work, tiling may set key=1 (FlashDecode reduce); either way output
    must match attn/golden.
    """
    dtype = torch.float16
    torch.manual_seed(SEED + 2)
    num_kv_tokens = FD_KV_TOKENS
    num_kv_heads = 2
    num_heads = 8
    block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])

    key = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    value = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    query = torch.randn((1, num_heads, HEAD_SIZE), dtype=dtype, device=device).contiguous()
    rotation = _identity(dtype, device)
    key_cache, value_cache = _pack_kv_cache(
        key=key,
        value=value,
        rotation_t=rotation,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
    )

    actual_seq_lens_q = [1]
    actual_seq_lens_kv = [num_kv_tokens]
    out_attn = _run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    torch.npu.synchronize()

    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=SCALE,
    )
    _compare_tria("flash_decode_long_kv", out_fia, out_attn, expected)


def test_flash_decode_multibatch_vs_attn(device: torch.device) -> None:
    """Multi-batch long KV to stress FD reduce (FIA vs attn only; golden too heavy)."""
    dtype = torch.float16
    torch.manual_seed(SEED + 3)
    num_kv_tokens = FD_KV_TOKENS
    num_kv_heads = 4
    num_heads = 8
    batch = 8
    actual_seq_lens_kv = [num_kv_tokens] * batch
    actual_seq_lens_q = list(range(1, batch + 1))
    total_q = batch
    block_table_cpu, num_blocks = _build_block_table(actual_seq_lens_kv)

    key_cache = torch.zeros(
        num_blocks, num_kv_heads, KEY_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros(
        num_blocks, num_kv_heads, VALUE_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    rotation = _identity(dtype, device)
    query = torch.randn(
        (total_q, num_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()

    for seq_idx in range(batch):
        key = torch.randn(
            (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
        ).contiguous()
        value = torch.randn(
            (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
        ).contiguous()
        slots = []
        for pos in range(num_kv_tokens):
            blk = int(block_table_cpu[seq_idx, pos // BLOCK_SIZE].item())
            slots.append(blk * BLOCK_SIZE + (pos % BLOCK_SIZE))
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
        torch.ops._C_ascend.bit_residual_pack_k8v4(
            key,
            value,
            slot_mapping,
            torch.tensor([0, num_kv_tokens], dtype=torch.int32, device=device),
            rotation,
            key_cache,
            value_cache,
            1,
            BLOCK_SIZE,
        )
    torch.npu.synchronize()

    block_table = block_table_cpu.to(device)
    out_attn = _run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    torch.npu.synchronize()

    diff = (out_fia.float() - out_attn.float()).abs().max().item()
    print(f"flash_decode_multibatch_vs_attn: fia_vs_attn={diff:.6f}")
    if diff > FIA_VS_ATTN_ATOL:
        raise AssertionError(
            f"flash_decode_multibatch_vs_attn: FIA vs attn max_diff {diff} "
            f"exceeds {FIA_VS_ATTN_ATOL}"
        )
    print("PASS flash_decode_multibatch_vs_attn")


def test_bf16_decode_multi_kv_gqa(device: torch.device) -> None:
    """bf16 decode: pack meta is bf16; FIA must read bf16 (not half) metadata."""
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    num_kv_tokens = 32
    num_kv_heads = 2
    num_heads = 8
    block_table_cpu, num_blocks = _build_block_table([num_kv_tokens])

    key = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    value = torch.randn(
        (num_kv_tokens, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device
    ).contiguous()
    query = torch.randn((1, num_heads, HEAD_SIZE), dtype=dtype, device=device).contiguous()
    rotation = _identity(dtype, device)
    key_cache, value_cache = _pack_kv_cache(
        key=key,
        value=value,
        rotation_t=rotation,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
    )

    actual_seq_lens_q = [1]
    actual_seq_lens_kv = [num_kv_tokens]
    out_attn = _run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=SCALE,
    )
    _compare_tria("bf16_decode_multi_kv_gqa", out_fia, out_attn, expected)


def test_bf16_pack_fia_dense_rotation(device: torch.device) -> None:
    """bf16 pack → FIA/attn with dense orthogonal R (serving-like Haar path)."""
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    num_kv_tokens = 9
    num_query_tokens = 1
    num_kv_heads = 4
    num_heads = 8
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
    key_cache, value_cache = _pack_kv_cache(
        key=key,
        value=value,
        rotation_t=rotation_t,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
    )

    actual_seq_lens_q = [num_query_tokens]
    actual_seq_lens_kv = [num_kv_tokens]
    out_attn = _run_attn(
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
    )
    out_fia = _run_fia(
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
    )
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
        scale=SCALE,
    )
    _compare_tria("bf16_pack_fia_dense_rotation", out_fia, out_attn, expected)


def test_bf16_prefill_causal_gqa(device: torch.device) -> None:
    """bf16 prefill: multi-Q + sparse_mode=3 vs attn/golden."""
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    num_kv_tokens = 32
    num_query_tokens = 8
    num_kv_heads = 2
    num_heads = 8
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
    rotation = _identity(dtype, device)
    key_cache, value_cache = _pack_kv_cache(
        key=key,
        value=value,
        rotation_t=rotation,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
    )
    atten_mask = _compress_causal_mask(device)

    actual_seq_lens_q = [num_query_tokens]
    actual_seq_lens_kv = [num_kv_tokens]
    out_attn = _run_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    out_fia = _run_fia(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        atten_mask=atten_mask,
        sparse_mode=SPARSE_MODE_RIGHT_DOWN,
    )
    expected = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache.cpu(),
        value_cache=value_cache.cpu(),
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        rotation_key=rotation.cpu(),
        rotation_value=rotation.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=SCALE,
    )
    _compare_tria("bf16_prefill_causal_gqa", out_fia, out_attn, expected)


def main() -> None:
    device = _require_npu()
    test_manual_single_kv(device)
    test_decode_multi_kv_gqa(device)
    test_pack_fia_chain_dense_rotation(device)
    test_prefill_causal_gqa(device)
    test_prefill_full_seq_dense_rotation(device)
    test_decode_kv_len_regression(device)
    test_mm1_s2_align32_workspace(device)
    test_flash_decode_long_kv(device)
    test_flash_decode_multibatch_vs_attn(device)
    test_bf16_decode_multi_kv_gqa(device)
    test_bf16_pack_fia_dense_rotation(device)
    test_bf16_prefill_causal_gqa(device)
    print("All BitResidual FIA Prefill/FD correctness smoke tests passed.")


if __name__ == "__main__":
    main()
