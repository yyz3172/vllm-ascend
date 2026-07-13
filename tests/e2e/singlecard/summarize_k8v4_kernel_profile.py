#!/usr/bin/env python3
"""Summarize K8V4 Pack/FIA device metrics from Ascend kernel_details.csv."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


TARGET_OPS = (
    "TurboquantPackKvForCacheK8v4",
    "TurboquantFusedInferAttentionScoreK8v4",
)


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v in ("", "N/A", None):
        return 0.0
    return float(v)


def summarize(profile_dir: Path) -> dict[str, dict[str, float]]:
    csv_path = profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    buckets: dict[str, list[dict[str, float]]] = {op: [] for op in TARGET_OPS}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            if name not in buckets:
                continue
            buckets[name].append({
                "duration_us": _f(row, "Duration(us)"),
                "aicore_us": _f(row, "aicore_time(us)"),
                "aic_mac_us": _f(row, "aic_mac_time(us)"),
                "aic_scalar_us": _f(row, "aic_scalar_time(us)"),
                "aiv_us": _f(row, "aiv_time(us)"),
                "aiv_vec_us": _f(row, "aiv_vec_time(us)"),
                "aiv_scalar_us": _f(row, "aiv_scalar_time(us)"),
                "cube_util": _f(row, "cube_utilization(%)"),
            })

    out: dict[str, dict[str, float]] = {}
    for op, rows in buckets.items():
        if not rows:
            continue
        def stat(key: str, fn=statistics.mean):
            return fn(r[key] for r in rows)
        out[op] = {
            "count": float(len(rows)),
            "duration_us_mean": stat("duration_us"),
            "duration_us_median": stat("duration_us", statistics.median),
            "duration_us_p90": sorted(r["duration_us"] for r in rows)[
                max(0, int(len(rows) * 0.9) - 1)
            ],
            "aic_mac_us_mean": stat("aic_mac_us"),
            "aic_scalar_us_mean": stat("aic_scalar_us"),
            "aiv_scalar_us_mean": stat("aiv_scalar_us"),
            "aiv_vec_us_mean": stat("aiv_vec_us"),
            "cube_util_mean": stat("cube_util"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dirs", nargs="+", help="rank0_* ascend_pt dirs")
    ap.add_argument("--labels", nargs="*", default=[], help="labels for each dir")
    args = ap.parse_args()

    labels = args.labels or [Path(p).name for p in args.profile_dirs]
    if len(labels) != len(args.profile_dirs):
        raise SystemExit("labels count must match profile_dirs")

    for label, p in zip(labels, args.profile_dirs):
        stats = summarize(Path(p))
        print(f"\n=== {label} ===")
        for op in TARGET_OPS:
            s = stats.get(op)
            if not s:
                print(f"{op}: (no rows)")
                continue
            print(
                f"{op}: n={int(s['count'])} "
                f"duration mean/med/p90="
                f"{s['duration_us_mean']:.1f}/"
                f"{s['duration_us_median']:.1f}/"
                f"{s['duration_us_p90']:.1f} us | "
                f"aic_mac={s['aic_mac_us_mean']:.3f} "
                f"aic_scalar={s['aic_scalar_us_mean']:.3f} "
                f"aiv_scalar={s['aiv_scalar_us_mean']:.1f} "
                f"aiv_vec={s['aiv_vec_us_mean']:.1f} "
                f"cube_util={s['cube_util_mean']:.1f}%"
            )


if __name__ == "__main__":
    main()
