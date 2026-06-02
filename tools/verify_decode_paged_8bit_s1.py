#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""S1 verification for turboquant_decode_paged_8bit.

Validates the addressing path only: S1 kernel just zero-extends the first 128 bytes of
each packed row to fp16. We assemble a reference by direct PyTorch indexing and compare.

Run on NPU after building the custom op:
    python tools/verify_decode_paged_8bit_s1.py

Exits 0 on success, non-zero on mismatch.
"""

from __future__ import annotations

import sys

import torch

import vllm_ascend  # noqa: F401  - registers _C_ascend ops
from vllm_ascend.utils import enable_custom_op


def make_inputs(
    *,
    num_blocks_total: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    batch: int,
    max_bps: int,
    device: torch.device,
    seed: int = 0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = head_size + 2

    # Random packed K/V cache. We control byte values so a typo in addressing surfaces
    # as a numerical mismatch (rather than accidental zero match).
    key_cache_cpu = torch.randint(0, 256, (num_blocks_total, block_size, num_kv_heads, P),
                                  generator=g, dtype=torch.uint8)
    value_cache_cpu = torch.randint(0, 256, (num_blocks_total, block_size, num_kv_heads, P),
                                    generator=g, dtype=torch.uint8)

    # Build seq lengths bounded by max_bps. Each entry must be in (0, max_bps * block_size].
    # We mix shapes within that envelope.
    candidates = [
        block_size * max_bps,            # exactly max_bps blocks
        block_size * max(1, max_bps - 1) + 17,  # max_bps blocks (or 1 when max_bps=1: → 17 → 1 block)
        block_size * max(1, max_bps // 2 + 1),  # roughly half the cap
        block_size,                      # exactly 1 block
    ]
    raw = torch.tensor(candidates[:batch], dtype=torch.int32)
    while raw.numel() < batch:
        raw = torch.cat([raw, torch.tensor([block_size], dtype=torch.int32)])
    # Clamp to legal range: [1, max_bps * block_size]
    cap = int(max_bps * block_size)
    actual_seq_lengths_kv = raw.clamp(min=1, max=cap).to(torch.int32)

    bs = block_size
    num_blocks_per_seq = (actual_seq_lengths_kv + bs - 1) // bs
    block_offsets = num_blocks_per_seq.cumsum(0) - num_blocks_per_seq
    total_blocks = int(num_blocks_per_seq.sum().item())
    assert total_blocks <= num_blocks_total, (
        f"total_blocks={total_blocks} exceeds num_blocks_total={num_blocks_total}; "
        "increase num_blocks_total in test config")

    # Random unique physical block ids per seq (no overlap, just to exercise scatter).
    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(num_blocks_total, generator=rng)[:total_blocks].to(torch.int32)

    block_table = torch.zeros((batch, max_bps), dtype=torch.int32)
    cursor = 0
    for s in range(batch):
        nb = int(num_blocks_per_seq[s].item())
        block_table[s, :nb] = perm[cursor:cursor + nb]
        cursor += nb

    # gather_block_ids[i] = block_table[seq_of(i), j_of(i)]
    arange_j = torch.arange(max_bps, dtype=torch.int32)
    j_mask = arange_j.unsqueeze(0) < num_blocks_per_seq.unsqueeze(1)  # [batch, max_bps]
    gather_block_ids = block_table[j_mask].to(torch.int32)            # [total_blocks]
    assert gather_block_ids.numel() == total_blocks

    # Move to NPU.
    key_cache = key_cache_cpu.to(device)
    value_cache = value_cache_cpu.to(device)
    gather_block_ids_dev = gather_block_ids.to(device)

    # Codebook / rotation are S1-irrelevant; supply legitimate fp16 placeholders.
    codebook = torch.linspace(-0.5, 0.5, 256, dtype=torch.float16, device=device)
    rotation = torch.eye(head_size, dtype=torch.float16, device=device)

    return {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "gather_block_ids": gather_block_ids_dev,
        "codebook": codebook,
        "rotation": rotation,
        # Reference inputs (CPU)
        "key_cache_cpu": key_cache_cpu,
        "value_cache_cpu": value_cache_cpu,
        "gather_block_ids_cpu": gather_block_ids,
        "block_table_cpu": block_table,
        "num_blocks_per_seq_cpu": num_blocks_per_seq,
        "block_offsets_cpu": block_offsets,
        "total_blocks": total_blocks,
        "head_size": head_size,
        "block_size": block_size,
    }


def reference_s1_output(
    cache_cpu: torch.Tensor,           # [num_blocks_total, BS, H, P=130] uint8
    gather_block_ids_cpu: torch.Tensor,  # [total_blocks] int32
    head_size: int,
) -> torch.Tensor:
    """S1 reference: take the first 128 bytes of each packed row, zero-extend to fp16."""
    selected = cache_cpu.index_select(0, gather_block_ids_cpu.to(torch.int64))
    # selected: [total_blocks, BS, H, 130]
    idx_bytes = selected[..., :head_size]                    # [total_blocks, BS, H, 128] uint8
    return idx_bytes.to(torch.float16)


def run_one_case(name: str, **cfg) -> bool:
    device = torch.device("npu", 0)
    print(f"\n[case {name}] cfg={cfg}")
    bundle = make_inputs(device=device, **cfg)

    enable_custom_op()
    op = getattr(torch.ops._C_ascend, "turboquant_decode_paged_8bit", None)
    if op is None:
        print("  FAIL: torch.ops._C_ascend.turboquant_decode_paged_8bit not registered")
        return False

    key_out, value_out = op(
        bundle["key_cache"],
        bundle["value_cache"],
        bundle["gather_block_ids"],
        bundle["codebook"],
        bundle["rotation"],
        bundle["head_size"],
        bundle["block_size"],
        0,    # out_dtype = fp16
    )
    torch.npu.synchronize()

    ref_key = reference_s1_output(bundle["key_cache_cpu"],
                                   bundle["gather_block_ids_cpu"],
                                   bundle["head_size"])
    ref_value = reference_s1_output(bundle["value_cache_cpu"],
                                     bundle["gather_block_ids_cpu"],
                                     bundle["head_size"])
    got_key = key_out.cpu()
    got_value = value_out.cpu()

    ok_k = torch.equal(got_key, ref_key)
    ok_v = torch.equal(got_value, ref_value)

    if not ok_k:
        diff = (got_key.to(torch.float32) - ref_key.to(torch.float32)).abs()
        idx = diff.flatten().argmax()
        ts = list(diff.shape)
        # Decompose flat idx
        coord = []
        rem = int(idx.item())
        for d in reversed(ts):
            coord.append(rem % d)
            rem //= d
        coord.reverse()
        print(f"  K mismatch: max abs diff={float(diff.max().item())}, "
              f"first divergent coord={coord}, "
              f"got={float(got_key.flatten()[idx])} ref={float(ref_key.flatten()[idx])}")
    else:
        print("  K OK")

    if not ok_v:
        print(f"  V mismatch: max abs diff="
              f"{float((got_value.to(torch.float32) - ref_value.to(torch.float32)).abs().max())}")
    else:
        print("  V OK")

    return ok_k and ok_v


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available, skipping S1 verification")
        return 0

    cases = [
        # name, cfg
        ("small",      dict(num_blocks_total=16,  block_size=128, num_kv_heads=8,
                            head_size=128, batch=4, max_bps=4)),
        ("uneven",     dict(num_blocks_total=64,  block_size=128, num_kv_heads=8,
                            head_size=128, batch=4, max_bps=8)),
        ("single_seq", dict(num_blocks_total=16,  block_size=128, num_kv_heads=8,
                            head_size=128, batch=1, max_bps=4)),
        ("max_bps_1",  dict(num_blocks_total=8,   block_size=128, num_kv_heads=8,
                            head_size=128, batch=4, max_bps=1)),
    ]
    all_ok = True
    for name, cfg in cases:
        try:
            ok = run_one_case(name, **cfg)
            all_ok = all_ok and ok
        except Exception as exc:
            print(f"  EXCEPTION: {exc!r}")
            all_ok = False

    print("\n" + ("ALL S1 CASES PASSED" if all_ok else "S1 VERIFICATION FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
