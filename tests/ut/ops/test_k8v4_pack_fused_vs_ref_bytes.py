#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare fused vs reference K8V4 pack bytes on NPU."""

from __future__ import annotations

import os

import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    refresh_turboquant_env_cache,
    turboquant_pack_kv_for_cache,
    turboquant_pack_kv_for_cache_to_cache,
    turboquant_packed_bytes_per_vector,
    turboquant_dequantize_from_packed_bytes,
)

D = 128
BITS_KEY, BITS_VALUE = 8, 4
SEED = 7


def main() -> None:
    device = torch.device("npu")
    torch.manual_seed(SEED)
    T, H = 4, 8
    slot_w = max(
        turboquant_packed_bytes_per_vector(D, bits=BITS_KEY),
        turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE),
    )
    key = torch.randn(T, H, D, dtype=torch.bfloat16, device=device)
    value = torch.randn(T, H, D, dtype=torch.bfloat16, device=device)

    # Reference pack
    os.environ["VLLM_ASCEND_TURBOQUANT_ENCODE_OP"] = "0"
    refresh_turboquant_env_cache()
    ref_k, ref_v = turboquant_pack_kv_for_cache(
        key=key, value=value, bits_key=BITS_KEY, bits_value=BITS_VALUE,
        slot_w_k=slot_w, slot_w_v=slot_w,
    )

    # Fused pack via to_cache then gather
    os.environ["VLLM_ASCEND_TURBOQUANT_ENCODE_OP"] = "1"
    refresh_turboquant_env_cache()
    num_blocks, block_size = 2, 16
    key_cache = torch.zeros(
        num_blocks, block_size, H, slot_w, dtype=torch.int8, device=device
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
    # slots 0..T-1 are in block0 positions 0..T-1
    fused_k = key_cache[0, :T].reshape(T, H, slot_w).contiguous()
    fused_v = value_cache[0, :T].reshape(T, H, slot_w).contiguous()

    rk = ref_k.view(torch.uint8)
    fk = fused_k.view(torch.uint8)
    rv = ref_v.view(torch.uint8)
    fv = fused_v.view(torch.uint8)

    def report(name, a, b):
        diff = (a.int() - b.int()).abs()
        print(f"{name}: shape={tuple(a.shape)} max_diff={int(diff.max())} "
              f"mean_diff={float(diff.float().mean()):.4f} "
              f"eq={(a == b).all().item()}")
        # norm bytes at offset 128 for 8-bit (128 idx + 2 norm)
        if a.shape[-1] >= 130:
            na = a[..., 128:130]
            nb = b[..., 128:130]
            nd = (na.int() - nb.int()).abs()
            print(f"  norm@128 max_diff={int(nd.max())} eq={(na==nb).all().item()} "
                  f"sample_ref={na.reshape(-1,2)[0].tolist()} sample_fused={nb.reshape(-1,2)[0].tolist()}")

    report("key", rk, fk)
    report("val", rv, fv)

    # Dequant both and compare
    dk_ref = turboquant_dequantize_from_packed_bytes(ref_k, head_size=D, bits=BITS_KEY, dtype=torch.bfloat16)
    dk_fus = turboquant_dequantize_from_packed_bytes(fused_k, head_size=D, bits=BITS_KEY, dtype=torch.bfloat16)
    dv_ref = turboquant_dequantize_from_packed_bytes(ref_v, head_size=D, bits=BITS_VALUE, dtype=torch.bfloat16)
    dv_fus = turboquant_dequantize_from_packed_bytes(fused_v, head_size=D, bits=BITS_VALUE, dtype=torch.bfloat16)
    for name, a, b in [("key_deq", dk_ref, dk_fus), ("val_deq", dv_ref, dv_fus)]:
        diff = (a.float() - b.float()).abs()
        cos = float(torch.dot(a.float().reshape(-1), b.float().reshape(-1)) /
                    (a.float().norm() * b.float().norm() + 1e-8))
        print(f"{name}: max={float(diff.max()):.6f} mean={float(diff.mean()):.6f} cos={cos:.6f}")


if __name__ == "__main__":
    main()
