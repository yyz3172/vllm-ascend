#!/usr/bin/env python3
"""Microbench for K8V4 CPU hot-path helpers (C1/C2/C3/C4)."""

from __future__ import annotations

import time

import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    _fp16_mask_on_device,
    _get_fused_fia_k8v4_op,
    _get_pack_k8v4_op,
    _int32_seq_lens_on_device,
    _prepare_k8v4_fused_read_caches,
    refresh_turboquant_env_cache,
)


def _us(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    if torch.npu.is_available():
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.npu.is_available():
        torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main() -> None:
    refresh_turboquant_env_cache()
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    seq = [1, 2]
    seq2 = [1, 2]  # same values, new list object

    # Cold then warm seq cache
    cold = _us(lambda: _int32_seq_lens_on_device([3, 5, 9], device), iters=50)
    warm = _us(lambda: _int32_seq_lens_on_device(seq2, device), iters=500)
    # Prime cache for seq
    t = _int32_seq_lens_on_device(seq, device)
    warm2 = _us(lambda: _int32_seq_lens_on_device(seq, device), iters=500)
    assert t.data_ptr() == _int32_seq_lens_on_device(seq, device).data_ptr()

    mask = torch.zeros(2, 1, 1, 256, dtype=torch.bfloat16, device=device)
    m0 = _us(lambda: _fp16_mask_on_device(mask), iters=200)
    m1 = _us(lambda: _fp16_mask_on_device(mask), iters=500)

    cache = torch.zeros(4, 16, 8, 130, dtype=torch.uint8, device=device)
    prep = _us(
        lambda: _prepare_k8v4_fused_read_caches(
            cache, cache, head_size=128, bits_key=8, bits_value=4,
            activation_dtype=torch.bfloat16,
        ),
        iters=500,
    )
    k, v = _prepare_k8v4_fused_read_caches(
        cache, cache, head_size=128, bits_key=8, bits_value=4,
        activation_dtype=torch.bfloat16,
    )
    assert k.data_ptr() == cache.data_ptr() and v.data_ptr() == cache.data_ptr()

    gate = _us(lambda: (_get_fused_fia_k8v4_op(), _get_pack_k8v4_op()), iters=2000)

    print(f"seq cold(new lens)     {cold:.1f} us")
    print(f"seq warm(same values)  {warm:.1f} us")
    print(f"seq warm(same list)    {warm2:.1f} us")
    print(f"mask first/cached      {m0:.1f} / {m1:.1f} us")
    print(f"prep_cache (no-op)   {prep:.1f} us")
    print(f"gate handle cache      {gate:.1f} us")
    print("PASS")


if __name__ == "__main__":
    main()
