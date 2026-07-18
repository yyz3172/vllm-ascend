#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Host-side FlashDecode cost-model probe for BitResidual K8V4.
# Mirrors PickFlashDecodeKvSplitPart in
# csrc/bit_residual_attention_paged_k8v4/op_host/bit_residual_attention_paged_k8v4_tiling.cpp
#
# B1: measure before raising TQ_BR_FLASH_DECODE_MAX_Q_TOKENS — do NOT blindly lift the cap.

from __future__ import annotations

import argparse
import math


MIN_KV_LEN = 1024
MIN_SEGMENT = 512
CURRENT_MAX_Q = 32


def ceil_div(a: int, b: int) -> int:
    return a if b == 0 else (a + b - 1) // b


def pick_kv_split_part(task_count: int, max_kv_len: int, core_num: int) -> tuple[int, int]:
    if task_count == 0 or core_num == 0 or max_kv_len < MIN_KV_LEN:
        return 1, ceil_div(task_count, core_num) * max_kv_len
    max_p = max(1, min(core_num, max_kv_len // MIN_SEGMENT))
    best_p = 1
    best_cost = ceil_div(task_count, core_num) * max_kv_len
    for p in range(2, max_p + 1):
        concurrent_bn = core_num // p
        if concurrent_bn == 0:
            break
        waves = ceil_div(task_count, concurrent_bn)
        seg_len = ceil_div(max_kv_len, p)
        cost = waves * seg_len
        if cost < best_cost:
            best_cost = cost
            best_p = p
    return best_p, best_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-len", type=int, default=2000)
    parser.add_argument("--cores", type=int, default=20)
    parser.add_argument("--kv-heads", type=int, default=8, help="taskCount = Q * kv_heads")
    parser.add_argument(
        "--q-tokens",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 241],
        help="Q token counts to evaluate (241 ≈ long_query chunk)",
    )
    args = parser.parse_args()

    print(
        f"FD cost model · kv_len={args.kv_len} cores={args.cores} "
        f"kv_heads={args.kv_heads} · current MAX_Q={CURRENT_MAX_Q}"
    )
    print(
        f"{'Q':>5} {'tasks':>7} {'P1_cost':>10} {'bestP':>6} {'best_cost':>10} "
        f"{'Δvs_P1':>8} {'eligible_today':>15} {'recommend'}"
    )
    for q in args.q_tokens:
        tasks = q * args.kv_heads
        p1_cost = ceil_div(tasks, args.cores) * args.kv_len
        best_p, best_cost = pick_kv_split_part(tasks, args.kv_len, args.cores)
        delta = best_cost - p1_cost
        delta_pct = (100.0 * delta / p1_cost) if p1_cost else 0.0
        eligible = q <= CURRENT_MAX_Q and best_p > 1
        # Only recommend raising MAX_Q when model shows clear win (>5%) and P>1.
        if best_p > 1 and delta_pct <= -5.0:
            recommend = "consider_raise_MAX_Q"
        elif best_p > 1 and delta_pct < 0:
            recommend = "marginal_keep_cap"
        else:
            recommend = "keep_SplitBN"
        print(
            f"{q:5d} {tasks:7d} {p1_cost:10d} {best_p:6d} {best_cost:10d} "
            f"{delta_pct:7.1f}% {str(eligible):>15} {recommend}"
        )
    print()
    print(
        "Note: model ignores Combine + causal imbalance on multi-Q prefill. "
        "B2 must not raise MAX_Q unless Δ is clearly negative AND e2e confirms."
    )


if __name__ == "__main__":
    main()
