"""Compare fused K8V4 read vs decode+FIA fallback on bf16 norms."""

from __future__ import annotations

import os

import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    _turboquant_fused_infer_attention_score_k8v4_impl,
    turboquant_fused_infer_attention_score_k8v4,
    turboquant_pack_kv_for_cache,
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
    if not torch.npu.is_available():
        print("skip: no NPU")
        return
    device = torch.device("npu")
    torch.manual_seed(SEED)

    num_blocks, block_size, num_heads, num_kv_heads = 4, 16, 16, 8
    slot_w = max(
        turboquant_packed_bytes_per_vector(D, bits=BITS_KEY),
        turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE),
    )

    # Use CPU reference pack so fallback decode does not require TBE JIT on NPU.
    key = torch.randn(num_blocks, block_size, num_kv_heads, D, dtype=torch.bfloat16)
    value = torch.randn(num_blocks, block_size, num_kv_heads, D, dtype=torch.bfloat16)

    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key.reshape(-1, num_kv_heads, D),
        value=value.reshape(-1, num_kv_heads, D),
        bits_key=BITS_KEY,
        bits_value=BITS_VALUE,
        slot_w_k=slot_w,
        slot_w_v=slot_w,
    )
    key_cache = packed_k.reshape(num_blocks, block_size, num_kv_heads, slot_w).to(device)
    value_cache = packed_v.reshape(num_blocks, block_size, num_kv_heads, slot_w).to(device)

    query = torch.randn(2, num_heads, D, dtype=torch.bfloat16, device=device)

    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    seq_q = [1, 2]
    seq_kv = [5, 7]
    scale = 1.0 / (D**0.5)
    atten_mask = torch.zeros(2, 1, 1, 256, dtype=torch.float16, device=device)

    fused = turboquant_fused_infer_attention_score_k8v4(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        atten_mask=atten_mask,
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
        atten_mask=atten_mask,
        actual_seq_lengths_q=seq_q,
        actual_seq_lengths_kv=seq_kv,
        head_size=D,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        scale=scale,
    )

    diff = (fused.float() - ref.float()).abs()
    print("fused finite:", torch.isfinite(fused).all().item())
    print("ref finite:", torch.isfinite(ref).all().item())
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("cos fused vs ref:", _cos(fused, ref))
    ok = torch.isfinite(fused).all() and _cos(fused, ref) > 0.99
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ASCEND_TURBOQUANT_ENCODE_OP", "0")
    main()
