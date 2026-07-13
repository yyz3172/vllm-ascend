#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fused pack -> fused FIA vs decode+FIA fallback on the same cache."""

from __future__ import annotations

import os

import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    _turboquant_fused_infer_attention_score_k8v4_impl,
    refresh_turboquant_env_cache,
    turboquant_fused_infer_attention_score_k8v4,
    turboquant_pack_kv_for_cache_to_cache,
    turboquant_packed_bytes_per_vector,
)

D = 128
BITS_KEY, BITS_VALUE = 8, 4
SEED = 42


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def main() -> None:
    device = torch.device("npu")
    torch.manual_seed(SEED)
    os.environ["VLLM_ASCEND_TURBOQUANT_ENCODE_OP"] = "1"
    os.environ["VLLM_ASCEND_TURBOQUANT_FUSED_FIA_K8V4"] = "1"
    refresh_turboquant_env_cache()

    num_blocks, block_size, num_heads, num_kv_heads = 4, 16, 16, 8
    slot_w = max(
        turboquant_packed_bytes_per_vector(D, bits=BITS_KEY),
        turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE),
    )
    T = 5
    key = torch.randn(T, num_kv_heads, D, dtype=torch.bfloat16, device=device)
    value = torch.randn(T, num_kv_heads, D, dtype=torch.bfloat16, device=device)
    key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, slot_w, dtype=torch.int8, device=device
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(T, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)

    turboquant_pack_kv_for_cache_to_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        num_reqs=1,
        bits_key=BITS_KEY,
        bits_value=BITS_VALUE,
    )
    torch.npu.synchronize()

    query = torch.randn(1, num_heads, D, dtype=torch.bfloat16, device=device)
    block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    seq_q = [1]
    seq_kv = [T]
    scale = 1.0 / (D**0.5)
    atten_mask_fp16 = torch.zeros(1, 1, 2048, 2048, dtype=torch.float16, device=device)
    atten_mask_bool = torch.zeros(1, 1, 2048, 2048, dtype=torch.bool, device=device)

    fused = turboquant_fused_infer_attention_score_k8v4(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        atten_mask=atten_mask_fp16,
        actual_seq_lengths_q=seq_q,
        actual_seq_lengths_kv=seq_kv,
        head_size=D,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        scale=scale,
    )
    ref = _turboquant_fused_infer_attention_score_k8v4_impl(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        atten_mask=atten_mask_bool,
        actual_seq_lengths_q=seq_q,
        actual_seq_lengths_kv=seq_kv,
        head_size=D,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        scale=scale,
    )
    torch.npu.synchronize()

    diff = (fused.float() - ref.float()).abs()
    cos = _cos(fused, ref)
    print("fused finite:", bool(torch.isfinite(fused).all()))
    print("ref finite:", bool(torch.isfinite(ref).all()))
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("cos:", cos)
    print("fused sample:", fused.reshape(-1)[:8].float().tolist())
    print("ref sample:", ref.reshape(-1)[:8].float().tolist())
    ok = torch.isfinite(fused).all() and cos > 0.95
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
