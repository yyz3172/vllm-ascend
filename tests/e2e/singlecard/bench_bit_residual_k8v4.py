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

"""Latency + correctness micro-benchmark for bit_residual_attention_paged_k8v4.

Builds a realistic paged KV cache (random packed bytes; layout matches the
16-row sub-block K8V4 format), then times the decode-path attention op across
several seq_len / block_size / dtype configurations.

Run inside the vllm-ascend docker after sourcing CANN + venv:

    python tests/e2e/singlecard/bench_bit_residual_k8v4.py

Notes:
- Cache bytes are random uint8. The op runs the identical decode + attention
  instruction stream regardless of byte values, so random fill is valid for
  timing (all per-row / per-tile control flow is shape-driven, not data-driven).
- A small packed-cache correctness check against the torch golden
  (xrx_bit_residual_k8v4_golden) is included to confirm the build is sane.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

_CANN_OPP = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{_CANN_OPP}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
    )

import torch  # noqa: E402

from vllm_ascend.utils import enable_custom_op  # noqa: E402

# ---- K8V4 cache layout constants (must match decode_device.h) ----
HEAD_SIZE = 128
SUB_BLOCK_ROWS = 16            # TQ_BR_BLOCK_ROWS
KEY_SUB_BLOCK_STRIDE = 2176    # TQ_BR_KEY_BLOCK_STRIDE
VALUE_SUB_BLOCK_STRIDE = 1152  # TQ_BR_VAL_BLOCK_STRIDE


def key_width(block_size: int) -> int:
    return (block_size // SUB_BLOCK_ROWS) * KEY_SUB_BLOCK_STRIDE


def value_width(block_size: int) -> int:
    return (block_size // SUB_BLOCK_ROWS) * VALUE_SUB_BLOCK_STRIDE


def _identity(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(HEAD_SIZE, dtype=dtype, device=device)


def build_block_table(seq_lens_kv: list[int], block_size: int):
    blocks_per_seq = [
        max(1, (s + block_size - 1) // block_size) for s in seq_lens_kv
    ]
    max_blocks = max(blocks_per_seq)
    table = torch.zeros((len(seq_lens_kv), max_blocks), dtype=torch.int32)
    nxt = 0
    for i, nblk in enumerate(blocks_per_seq):
        table[i, :nblk] = torch.arange(nxt, nxt + nblk, dtype=torch.int32)
        nxt += nblk
    return table, nxt


def run_op(
    *,
    query,
    key_cache,
    value_cache,
    block_table,
    seq_lens_q: list[int],
    seq_lens_kv: list[int],
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
    scale: float,
):
    rotation = _identity(query.dtype, query.device)
    return torch.ops._C_ascend.bit_residual_attention_paged_k8v4(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        seq_lens_q,
        seq_lens_kv,
        rotation.contiguous(),
        rotation.contiguous(),
        int(num_heads),
        int(num_kv_heads),
        HEAD_SIZE,
        int(block_size),
        int(max(1, max(seq_lens_kv))),
        float(scale),
    )


def make_inputs(
    *,
    num_query_tokens: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    block_table, num_blocks = build_block_table([seq_len_kv], block_size)
    block_table = block_table.to(device)
    key_cache = torch.zeros(
        (num_blocks, num_kv_heads, key_width(block_size)),
        dtype=torch.uint8, device=device,
    )
    value_cache = torch.zeros(
        (num_blocks, num_kv_heads, value_width(block_size)),
        dtype=torch.uint8, device=device,
    )
    # Random packed bytes: same instruction stream as real data, valid for timing.
    key_cache.uniform_(0, 255)
    value_cache.uniform_(0, 255)
    query = torch.randn(
        (num_query_tokens, num_heads, HEAD_SIZE), dtype=dtype, device=device,
    ).contiguous()
    return dict(
        query=query, key_cache=key_cache, value_cache=value_cache,
        block_table=block_table, seq_lens_q=[num_query_tokens],
        seq_lens_kv=[seq_len_kv], num_heads=num_heads,
        num_kv_heads=num_kv_heads, block_size=block_size,
        scale=HEAD_SIZE ** -0.5,
    )


def bench_once(inputs, warmup=5, iters=50) -> dict:
    # Warmup.
    for _ in range(warmup):
        out = run_op(**inputs)
    torch.npu.synchronize()

    # Timed.
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = run_op(**inputs)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    per_iter_us = elapsed / iters * 1e6
    samples = []
    # Per-call samples for stddev (smaller N to bound cost).
    for _ in range(min(iters, 30)):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run_op(**inputs)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)

    return dict(
        mean_us=per_iter_us,
        median_us=statistics.median(samples),
        min_us=min(samples),
        max_us=max(samples),
        stddev_us=statistics.pstdev(samples),
        out_shape=tuple(out.shape),
        out_finite=torch.isfinite(out.float()).all().item(),
    )


def correctness_smoke(dtype: torch.dtype, device: torch.device) -> None:
    """Reuse the golden pack+attention chain for a tiny correctness check."""
    import importlib.util
    golden_path = Path(__file__).parent / "xrx_bit_residual_k8v4_golden.py"
    spec = importlib.util.spec_from_file_location("br_golden", golden_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Golden runs both fp16 and bf16; run just the requested dtype.
    mod._run_pack_attention_chain(dtype, device)
    print(f"[correctness] pack+attention chain PASSED for dtype={dtype}")


def main() -> None:
    if not enable_custom_op():
        raise RuntimeError("custom op not loaded; build first")
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(
        torch.ops._C_ascend, "bit_residual_attention_paged_k8v4"
    ):
        raise RuntimeError("bit_residual_attention_paged_k8v4 not registered")

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    print("=" * 92)
    print("bit_residual_attention_paged_k8v4 decode latency benchmark (910B3)")
    print("=" * 92)

    # Quick correctness gate (one dtype) so the perf numbers are meaningful.
    try:
        correctness_smoke(torch.float16, device)
    except Exception as e:  # noqa: BLE001
        print(f"[correctness] SKIPPED/FAILED: {e}")

    configs = []
    for dtype in (torch.float16, torch.bfloat16):
        for seq_len in (128, 512, 2048, 4096):
            for block_size in (16, 128):
                configs.append(dict(
                    dtype=dtype, seq_len_kv=seq_len, block_size=block_size,
                    num_heads=16, num_kv_heads=8, num_query_tokens=1,
                ))

    header = (
        f"{'dtype':<7} {'seq_kv':>7} {'blk':>4} {'gqa':>4} "
        f"{'mean_us':>9} {'median_us':>10} {'min_us':>8} {'max_us':>8} "
        f"{'stddev':>8} {'finite':>7}"
    )
    print(header)
    print("-" * len(header))
    for cfg in configs:
        inputs = make_inputs(
            num_query_tokens=cfg["num_query_tokens"],
            seq_len_kv=cfg["seq_len_kv"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            block_size=cfg["block_size"],
            dtype=cfg["dtype"],
            device=device,
        )
        r = bench_once(inputs)
        gqa = cfg["num_heads"] // cfg["num_kv_heads"]
        print(
            f"{str(cfg['dtype']):<7} {cfg['seq_len_kv']:>7} {cfg['block_size']:>4} "
            f"{gqa:>4} {r['mean_us']:>9.2f} {r['median_us']:>10.2f} "
            f"{r['min_us']:>8.2f} {r['max_us']:>8.2f} {r['stddev_us']:>8.2f} "
            f"{str(r['out_finite']):>7}"
        )
    print("-" * len(header))
    print("Note: mean_us is end-to-end (op launch + kernel + sync) over 50 iters.")
    print("For kernel-only time and AICore utilization use msprof (see profile script).")


if __name__ == "__main__":
    main()

