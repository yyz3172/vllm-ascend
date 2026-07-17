#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Summarize BitResidual FIA ``msprof op`` dumps (L6 / Step1 performance path).

Usage::

    python3 tools/bit_residual_fia_paged_k8v4/analyze_opprof.py \\
        tools/bit_residual_fia_paged_k8v4/prof_output/tnd_pa_bit_residual/kv1000/op/OPPROF_xxx

    # Or auto-pick latest under a parent dir:
    python3 tools/bit_residual_fia_paged_k8v4/analyze_opprof.py --latest \\
        tools/bit_residual_fia_paged_k8v4/prof_output/tnd_pa_bit_residual/kv1000/op

What to watch (and what it means)
---------------------------------

OpBasicInfo.csv
  Task Duration(us)
      End-to-end kernel wall time on device for one launch. Primary KPI for
      P2/P3 deltas and for ``BR <= TQ * 1.15`` (P3).
  Block Dim / Mix Block Dim
      Cube / Vector core occupancy of this launch. Sanity-check tiling split.
  Current Freq / Rated Freq
      Chip frequency during the sample; large downclock skews comparisons.

PipeUtilization.csv  (per sub_block: cube* / vector*)
  aic_time(us) / aiv_time(us)
      Average busy time on AIC (Cube) vs AIV (Vector). Compare means across
      cores → **Pipe bound**:
        * AIV >> AIC  → vector/dequant/softmax/MTE likely bottleneck → P2
          (batch dequant via tmpBuff1, s2_sub=8~16) is plausible.
        * AIC >> AIV  → matmul-dominated → P2 dequant batching has limited
          upside; look at tiling / FD / Cube efficiency instead.
        * AIC ≈ AIV   → balanced; only ship P2 if A/B shows clear Task
          Duration drop.
  aic_cube_ratio / aic_mte2_ratio
      Fraction of AIC time in Cube MAC vs MTE2 (GM→L1). Low cube + high mte2
      ⇒ Cube starved on loads.
  aiv_vec_ratio / aiv_mte2_ratio / aiv_mte3_ratio
      Vector compute vs MTE2 (GM→UB) vs MTE3 (UB→GM). High mte2 with low vec
      on AIV often means dequant/load trip count (s2_sub=1) dominates → P2.

ArithmeticUtilization.csv
  aic_cube_fops / aiv_vec_fops
      Rough arithmetic volume. Use with Pipe ratios, not alone.
  aic_cube_total_instr_number
      Cube instruction pressure; secondary.

Memory / MemoryUB (if present)
  UB / L1 pressure hints. P2 borrows tmpBuff1 (32KB); watch for UB conflict
  regressions after raising s2_sub.

How to decide P2 / P3
---------------------
  1. Record Task Duration + Pipe bound on this BR FIA dump (kv=1000 L6).
  2. If AIV-bound and aiv_mte2/vec look hot → try P2; re-run this script.
  3. Compare Task Duration to TQ FIA same shape (P3 bar: BR <= TQ * 1.15).
  4. Serving torch profiler is for E2E A/B only — do not use it to attribute
     dequant vs Cube.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


METRIC_HELP = """
Key metrics (short):
  Task Duration     — kernel wall time (us); main KPI
  Pipe bound        — max(avg AIC time, avg AIV time); which pipe is slower
  aic_cube_ratio    — Cube MAC share of AIC time
  aic_mte2_ratio    — AIC GM→L1 load share
  aiv_vec_ratio     — Vector compute share of AIV time
  aiv_mte2_ratio    — AIV GM→UB load share (dequant-sensitive)
  aiv_mte3_ratio    — AIV UB→GM store share
See module docstring for full interpretation / P2-P3 decision tree.
""".strip()


def _fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _avg(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _fmt(v: float | None, suffix: str = "") -> str:
    if v is None:
        return "NA"
    return f"{v:.3f}{suffix}"


def _resolve_opprof_dir(path: Path) -> Path:
    """Accept OPPROF root, .../dump, or a parent containing OPPROF_*."""
    if path.is_file():
        path = path.parent
    if (path / "OpBasicInfo.csv").is_file():
        return path
    if (path / "dump" / "OpBasicInfo.csv").is_file():
        return path
    # dump/ may only have fdata; CSVs live on OPPROF root
    if path.name == "dump" and (path.parent / "OpBasicInfo.csv").is_file():
        return path.parent
    raise FileNotFoundError(
        f"OpBasicInfo.csv not found under {path} "
        "(pass OPPROF_* directory from msprof op --output)"
    )


def _latest_opprof(parent: Path) -> Path:
    cands = sorted(
        parent.glob("OPPROF_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"no OPPROF_* under {parent}")
    return cands[0]


def summarize(opprof_dir: Path) -> dict[str, Any]:
    opprof_dir = _resolve_opprof_dir(opprof_dir)
    result: dict[str, Any] = {"opprof_dir": str(opprof_dir)}

    obi = opprof_dir / "OpBasicInfo.csv"
    if obi.is_file():
        with obi.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            r = rows[0]
            result["op_name"] = r.get("Op Name", "")
            result["task_duration_us"] = _fnum(r.get("Task Duration(us)"))
            result["block_dim"] = r.get("Block Dim", "")
            result["mix_block_dim"] = r.get("Mix Block Dim", "")
            result["current_freq"] = r.get("Current Freq", "")
            result["rated_freq"] = r.get("Rated Freq", "")

    pu = opprof_dir / "PipeUtilization.csv"
    if pu.is_file():
        with pu.open(newline="") as f:
            rows = list(csv.DictReader(f))
        aic = [r for r in rows if str(r.get("sub_block_id", "")).startswith("cube")]
        aiv = [r for r in rows if str(r.get("sub_block_id", "")).startswith("vector")]
        aic_t = _avg([_fnum(r.get("aic_time(us)")) for r in aic])
        aiv_t = _avg([_fnum(r.get("aiv_time(us)")) for r in aiv])
        result["aic_cores"] = len(aic)
        result["aiv_cores"] = len(aiv)
        result["aic_time_us"] = aic_t
        result["aiv_time_us"] = aiv_t
        result["aic_cube_ratio"] = _avg([_fnum(r.get("aic_cube_ratio")) for r in aic])
        result["aic_mte2_ratio"] = _avg([_fnum(r.get("aic_mte2_ratio")) for r in aic])
        result["aiv_vec_ratio"] = _avg([_fnum(r.get("aiv_vec_ratio")) for r in aiv])
        result["aiv_mte2_ratio"] = _avg([_fnum(r.get("aiv_mte2_ratio")) for r in aiv])
        result["aiv_mte3_ratio"] = _avg([_fnum(r.get("aiv_mte3_ratio")) for r in aiv])
        if aic_t is not None and aiv_t is not None:
            result["pipe_bound"] = "AIV" if aiv_t > aic_t else "AIC"
            result["pipe_bound_us"] = max(aic_t, aiv_t)
            if aic_t > 0:
                result["aiv_over_aic"] = aiv_t / aic_t
        else:
            result["pipe_bound"] = "unknown"

    au = opprof_dir / "ArithmeticUtilization.csv"
    if au.is_file():
        with au.open(newline="") as f:
            rows = list(csv.DictReader(f))
        aic = [r for r in rows if str(r.get("sub_block_id", "")).startswith("cube")]
        aiv = [r for r in rows if str(r.get("sub_block_id", "")).startswith("vector")]
        result["cube_instr"] = _avg(
            [_fnum(r.get("aic_cube_total_instr_number")) for r in aic]
        )
        result["cube_fops"] = _avg([_fnum(r.get("aic_cube_fops")) for r in aic])
        result["vec_fops"] = _avg([_fnum(r.get("aiv_vec_fops")) for r in aiv])

    return result


def _p2_p3_hint(s: dict[str, Any]) -> str:
    bound = s.get("pipe_bound", "unknown")
    aiv_mte2 = s.get("aiv_mte2_ratio")
    aiv_vec = s.get("aiv_vec_ratio")
    lines = []
    if bound == "AIV":
        lines.append(
            "Pipe bound=AIV → P2 (batch dequant / s2_sub) is a plausible next step."
        )
        if aiv_mte2 is not None and aiv_vec is not None and aiv_mte2 > aiv_vec:
            lines.append(
                "  aiv_mte2_ratio > aiv_vec_ratio → load/dequant trips likely dominate "
                "vector compute; prioritize s2_sub batching."
            )
    elif bound == "AIC":
        lines.append(
            "Pipe bound=AIC → Cube/MM path dominates; P2 dequant batching likely "
            "limited ROI. Prefer tiling/FD/Cube efficiency before Queue (P3)."
        )
    else:
        lines.append("Pipe bound unknown (missing PipeUtilization.csv).")

    lines.append(
        "P3: compare Task Duration to TQ FIA same shape; target BR <= TQ * 1.15."
    )
    return "\n".join(lines)


def print_summary(s: dict[str, Any]) -> None:
    print()
    print(f"=== Op metrics summary: {s['opprof_dir']} ===")
    print()
    print("--- OpBasicInfo (wall time) ---")
    print(f"  Op Name         : {s.get('op_name', '')}")
    print(f"  Task Duration   : {_fmt(s.get('task_duration_us'), ' us')}  "
          f"← primary KPI")
    print(f"  Block Dim       : {s.get('block_dim', '')}  "
          f"Mix Block Dim: {s.get('mix_block_dim', '')}")
    print(f"  Freq            : {s.get('current_freq', '')} / "
          f"rated {s.get('rated_freq', '')} MHz")

    print()
    print("--- PipeUtilization (AIC Cube vs AIV Vector) ---")
    print(f"  AIC cores       : {s.get('aic_cores', 0)}  "
          f"avg time={_fmt(s.get('aic_time_us'), ' us')}")
    print(f"    cube_ratio    : {_fmt(s.get('aic_cube_ratio'))}  "
          f"mte2_ratio={_fmt(s.get('aic_mte2_ratio'))}")
    print(f"  AIV cores       : {s.get('aiv_cores', 0)}  "
          f"avg time={_fmt(s.get('aiv_time_us'), ' us')}")
    print(f"    vec_ratio     : {_fmt(s.get('aiv_vec_ratio'))}  "
          f"mte2_ratio={_fmt(s.get('aiv_mte2_ratio'))}  "
          f"mte3_ratio={_fmt(s.get('aiv_mte3_ratio'))}")
    bound = s.get("pipe_bound", "unknown")
    if bound in ("AIC", "AIV"):
        ratio = s.get("aiv_over_aic")
        extra = f"  (AIV/AIC={ratio:.2f})" if isinstance(ratio, float) else ""
        print(f"  Pipe bound      : {bound} "
              f"(max avg {_fmt(s.get('pipe_bound_us'), ' us')}){extra}")
    else:
        print("  Pipe bound      : unknown")

    if any(k in s for k in ("cube_instr", "cube_fops", "vec_fops")):
        print()
        print("--- ArithmeticUtilization ---")
        print(f"  Cube instr#     : avg={s.get('cube_instr')}")
        print(f"  Cube FOPS       : avg={s.get('cube_fops')}")
        print(f"  Vec FOPS        : avg={s.get('vec_fops')}")

    print()
    print("--- P2 / P3 reading ---")
    print(_p2_p3_hint(s))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("What to watch")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=METRIC_HELP,
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="OPPROF_* dir, or parent dir when used with --latest",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Pick newest OPPROF_* under path (default: path is OPPROF itself)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print full metric glossary and exit",
    )
    args = parser.parse_args()

    if args.explain:
        print(__doc__)
        return 0

    if args.path is None:
        parser.error("path is required (or pass --explain)")

    target = args.path
    if args.latest:
        target = _latest_opprof(args.path)

    summary = summarize(target)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
