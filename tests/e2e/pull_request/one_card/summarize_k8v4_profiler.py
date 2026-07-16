#!/usr/bin/env python3
"""Summarize K8V4 BitResidual / TurboQuant metrics from Ascend profiler output.

Reads ``ASCEND_PROFILER_OUTPUT/operator_details.csv`` and
``ASCEND_PROFILER_OUTPUT/kernel_details.csv`` under one or more
``rank0_*_ascend_pt`` directories produced by ``xrx_bit_residual_k8v4_smoke.py``
(with ``XRX_K8V4_PROFILE=1``) or similar torch-profiler runs.

Example::

    python tests/e2e/singlecard/summarize_k8v4_profiler.py \\
        /root/yyz/pytorch_profiler/BitResidual/260713/k8v4_0.6B/rank0_565817_20260713062511278_ascend_pt

    python tests/e2e/singlecard/summarize_k8v4_profiler.py \\
        --labels baseline optimized \\
        profile_a/rank0_xxx_ascend_pt profile_b/rank0_yyy_ascend_pt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


UNIFIED_ATTENTION = "vllm::unified_attention_with_output"

# operator_details.csv names (PyTorch / custom-op wrapper level)
OPERATOR_TARGETS = {
    "bit_residual": {
        "pack_wrapper": "_C_ascend::bit_residual_pack_k8v4",
        "attn_wrapper": "_C_ascend::bit_residual_attention_paged_k8v4",
        "pack_aclnn": "aclnnBitResidualPackK8v4",
        "attn_aclnn": "aclnnBitResidualAttentionPagedK8v4",
    },
    "turboquant": {
        "pack_wrapper": "_C_ascend::turboquant_pack_kv_for_cache_k8v4",
        "attn_wrapper": "_C_ascend::turboquant_fused_infer_attention_score_k8v4",
        "pack_aclnn": "aclnnTurboquantPackKvForCacheK8v4",
        "attn_aclnn": "aclnnTurboquantFusedInferAttentionScoreK8v4",
    },
}

HOST_OVERHEAD_OPS = (
    "aten::copy_",
    "aten::contiguous",
    "npu::_npu_dtype_cast",
    "aten::to",
)

# kernel_details.csv kernel names (device AICore level)
KERNEL_TARGETS = {
    "bit_residual": (
        "BitResidualPackK8v4",
        "BitResidualAttentionPagedK8v4",
    ),
    "turboquant": (
        "TurboquantPackKvForCacheK8v4",
        "TurboquantFusedInferAttentionScoreK8v4",
    ),
}


@dataclass
class DurationStats:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0

    @classmethod
    def from_values(cls, values: Iterable[float]) -> DurationStats:
        vals = [float(v) for v in values]
        if not vals:
            return cls()
        ordered = sorted(vals)
        p90_idx = max(0, int(len(ordered) * 0.9) - 1)
        return cls(
            count=len(vals),
            mean=statistics.mean(vals),
            median=statistics.median(vals),
            p90=ordered[p90_idx],
            min=ordered[0],
            max=ordered[-1],
            sum=sum(vals),
        )

    def fmt(self, unit: str = "us") -> str:
        if self.count == 0:
            return "n/a"
        return (
            f"n={self.count} mean/med/p90="
            f"{self.mean:.1f}/{self.median:.1f}/{self.p90:.1f} {unit}"
        )


@dataclass
class KernelStats:
    duration: DurationStats = field(default_factory=DurationStats)
    aic_mac_mean: float = 0.0
    aic_scalar_mean: float = 0.0
    aiv_scalar_mean: float = 0.0
    aiv_vec_mean: float = 0.0
    cube_util_mean: float = 0.0


@dataclass
class LayerBlock:
    ops: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ProfileSummary:
    profile_dir: Path
    flavor: str
    layer_blocks: list[LayerBlock] = field(default_factory=list)
    steady_blocks: list[LayerBlock] = field(default_factory=list)
    operator_stats: dict[str, DurationStats] = field(default_factory=dict)
    kernel_stats: dict[str, KernelStats] = field(default_factory=dict)
    host_overhead: dict[str, DurationStats] = field(default_factory=dict)


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "N/A", None):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _resolve_profile_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "ASCEND_PROFILER_OUTPUT" / "operator_details.csv").exists():
        return path
    candidates = sorted(path.glob("rank0_*_ascend_pt"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{path} contains multiple rank0_* dirs; pass one explicitly: "
            + ", ".join(p.name for p in candidates[:5])
        )
    raise FileNotFoundError(f"profiler output not found under {path}")


def _detect_flavor(operator_rows: list[dict[str, str]]) -> str:
    names = {row.get("Name", "") for row in operator_rows}
    br_hits = sum(
        1
        for key in OPERATOR_TARGETS["bit_residual"].values()
        if key in names
    )
    tq_hits = sum(
        1
        for key in OPERATOR_TARGETS["turboquant"].values()
        if key in names
    )
    if br_hits >= tq_hits and br_hits > 0:
        return "bit_residual"
    if tq_hits > 0:
        return "turboquant"
    return "unknown"


def _split_attention_layers(rows: list[dict[str, str]]) -> list[LayerBlock]:
    blocks: list[LayerBlock] = []
    current: LayerBlock | None = None
    for row in rows:
        name = row.get("Name", "")
        if name == UNIFIED_ATTENTION:
            if current is not None:
                blocks.append(current)
            current = LayerBlock()
            current.ops.append(row)
            continue
        if current is not None:
            current.ops.append(row)
    if current is not None:
        blocks.append(current)
    return blocks


def _pick_steady_blocks(
    blocks: list[LayerBlock],
    skip_warmup: int,
    skip_tail: int,
) -> list[LayerBlock]:
    if not blocks:
        return []
    start = min(skip_warmup, len(blocks))
    end = len(blocks) - skip_tail if skip_tail > 0 else len(blocks)
    end = max(start, end)
    return blocks[start:end]


def _collect_operator_values(
    blocks: list[LayerBlock],
    op_name: str,
    column: str,
) -> list[float]:
    values: list[float] = []
    for block in blocks:
        for row in block.ops:
            if row.get("Name", "") == op_name:
                values.append(_f(row, column))
                break
    return values


def _collect_host_overhead(
    blocks: list[LayerBlock],
    op_name: str,
    column: str,
) -> list[float]:
    values: list[float] = []
    for block in blocks:
        per_layer = 0.0
        for row in block.ops:
            if row.get("Name", "") == op_name:
                per_layer += _f(row, column)
        if per_layer > 0:
            values.append(per_layer)
    return values


def _read_operator_rows(profile_dir: Path) -> list[dict[str, str]]:
    csv_path = profile_dir / "ASCEND_PROFILER_OUTPUT" / "operator_details.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize_kernels(profile_dir: Path, flavor: str) -> dict[str, KernelStats]:
    csv_path = profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if not csv_path.exists():
        return {}

    targets = KERNEL_TARGETS.get(flavor, ())
    buckets: dict[str, list[dict[str, float]]] = {name: [] for name in targets}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Name", "")
            if name not in buckets:
                continue
            buckets[name].append(
                {
                    "duration_us": _f(row, "Duration(us)"),
                    "aic_mac_us": _f(row, "aic_mac_time(us)"),
                    "aic_scalar_us": _f(row, "aic_scalar_time(us)"),
                    "aiv_scalar_us": _f(row, "aiv_scalar_time(us)"),
                    "aiv_vec_us": _f(row, "aiv_vec_time(us)"),
                    "cube_util": _f(row, "cube_utilization(%)"),
                }
            )

    out: dict[str, KernelStats] = {}
    for name, rows in buckets.items():
        if not rows:
            continue
        out[name] = KernelStats(
            duration=DurationStats.from_values(r["duration_us"] for r in rows),
            aic_mac_mean=statistics.mean(r["aic_mac_us"] for r in rows),
            aic_scalar_mean=statistics.mean(r["aic_scalar_us"] for r in rows),
            aiv_scalar_mean=statistics.mean(r["aiv_scalar_us"] for r in rows),
            aiv_vec_mean=statistics.mean(r["aiv_vec_us"] for r in rows),
            cube_util_mean=statistics.mean(r["cube_util"] for r in rows),
        )
    return out


def summarize_profile(
    profile_dir: Path,
    skip_warmup: int = 1,
    skip_tail: int = 1,
) -> ProfileSummary:
    profile_dir = _resolve_profile_dir(profile_dir)
    rows = _read_operator_rows(profile_dir)
    flavor = _detect_flavor(rows)
    layer_blocks = _split_attention_layers(rows)
    steady_blocks = _pick_steady_blocks(layer_blocks, skip_warmup, skip_tail)

    operator_stats: dict[str, DurationStats] = {}
    target_ops = OPERATOR_TARGETS.get(flavor, {})
    for label, op_name in target_ops.items():
        operator_stats[label] = DurationStats.from_values(
            _collect_operator_values(steady_blocks, op_name, "Device Self Duration(us)")
        )
        operator_stats[f"{label}_host"] = DurationStats.from_values(
            _collect_operator_values(steady_blocks, op_name, "Host Self Duration(us)")
        )

    operator_stats["unified_attention_host"] = DurationStats.from_values(
        _collect_operator_values(steady_blocks, UNIFIED_ATTENTION, "Host Self Duration(us)")
    )
    operator_stats["unified_attention_device"] = DurationStats.from_values(
        _collect_operator_values(steady_blocks, UNIFIED_ATTENTION, "Device Total Duration(us)")
    )

    host_overhead: dict[str, DurationStats] = {}
    for op_name in HOST_OVERHEAD_OPS:
        host_overhead[op_name] = DurationStats.from_values(
            _collect_host_overhead(steady_blocks, op_name, "Host Self Duration(us)")
        )
        host_overhead[f"{op_name}_device"] = DurationStats.from_values(
            _collect_host_overhead(steady_blocks, op_name, "Device Total Duration(us)")
        )

    return ProfileSummary(
        profile_dir=profile_dir,
        flavor=flavor,
        layer_blocks=layer_blocks,
        steady_blocks=steady_blocks,
        operator_stats=operator_stats,
        kernel_stats=_summarize_kernels(profile_dir, flavor),
        host_overhead=host_overhead,
    )


def _print_summary(label: str, summary: ProfileSummary) -> None:
    print(f"\n=== {label} ===")
    print(f"path   : {summary.profile_dir}")
    print(f"flavor : {summary.flavor}")
    print(
        f"layers : total={len(summary.layer_blocks)} "
        f"steady={len(summary.steady_blocks)}"
    )

    print("\n[operator_details / per-layer steady-state]")
    rows_to_print = [
        ("pack_wrapper (host)", "pack_wrapper_host"),
        ("attn_wrapper (host)", "attn_wrapper_host"),
        ("pack_aclnn (device)", "pack_aclnn"),
        ("attn_aclnn (device)", "attn_aclnn"),
        ("unified_attention (host)", "unified_attention_host"),
        ("unified_attention (device total)", "unified_attention_device"),
    ]
    for label, key in rows_to_print:
        stats = summary.operator_stats.get(key)
        if stats is None or stats.count == 0:
            continue
        print(f"  {label:32s} {stats.fmt()}")

    pack_dev = summary.operator_stats.get("pack_aclnn")
    attn_dev = summary.operator_stats.get("attn_aclnn")
    if pack_dev and attn_dev and pack_dev.count and attn_dev.count:
        total = pack_dev.mean + attn_dev.mean
        print(f"  {'pack+attn device total':28s} ~{total:.1f} us/layer")

    print("\n[host overhead inside attention blocks]")
    for op_name in HOST_OVERHEAD_OPS:
        stats = summary.host_overhead.get(op_name)
        if stats is None or stats.count == 0:
            continue
        dev_stats = summary.host_overhead.get(f"{op_name}_device")
        dev_part = f" | device sum/layer {dev_stats.mean:.1f} us" if dev_stats and dev_stats.count else ""
        print(f"  {op_name:28s} host sum/layer {stats.mean:.1f} us{dev_part}")

    if summary.kernel_stats:
        print("\n[kernel_details / device kernel]")
        for name, stats in summary.kernel_stats.items():
            print(
                f"  {name}: {stats.duration.fmt()} | "
                f"aic_mac={stats.aic_mac_mean:.2f} "
                f"aic_scalar={stats.aic_scalar_mean:.2f} "
                f"aiv_scalar={stats.aiv_scalar_mean:.1f} "
                f"aiv_vec={stats.aiv_vec_mean:.1f} "
                f"cube_util={stats.cube_util_mean:.1f}%"
            )


def _print_compare(base_label: str, base: ProfileSummary, other_label: str, other: ProfileSummary) -> None:
    print(f"\n=== compare: {other_label} vs {base_label} ===")
    keys = sorted(set(base.operator_stats) | set(other.operator_stats))
    for key in keys:
        b = base.operator_stats.get(key)
        o = other.operator_stats.get(key)
        if not b or not o or b.count == 0 or o.count == 0:
            continue
        delta = o.mean - b.mean
        print(f"  {key:28s} {b.mean:7.2f} -> {o.mean:7.2f} us ({delta:+.2f})")

    b_k = base.kernel_stats
    o_k = other.kernel_stats
    for name in sorted(set(b_k) | set(o_k)):
        bs = b_k.get(name)
        os_ = o_k.get(name)
        if not bs or not os_ or bs.duration.count == 0 or os_.duration.count == 0:
            continue
        delta = os_.duration.mean - bs.duration.mean
        print(
            f"  kernel {name:24s} {bs.duration.mean:7.2f} -> "
            f"{os_.duration.mean:7.2f} us ({delta:+.2f})"
        )


def _to_json(summary: ProfileSummary) -> dict:
    def stats_dict(stats: DurationStats) -> dict:
        return {
            "count": stats.count,
            "mean": stats.mean,
            "median": stats.median,
            "p90": stats.p90,
            "min": stats.min,
            "max": stats.max,
            "sum": stats.sum,
        }

    return {
        "profile_dir": str(summary.profile_dir),
        "flavor": summary.flavor,
        "layers_total": len(summary.layer_blocks),
        "layers_steady": len(summary.steady_blocks),
        "operator_stats": {k: stats_dict(v) for k, v in summary.operator_stats.items()},
        "host_overhead": {k: stats_dict(v) for k, v in summary.host_overhead.items()},
        "kernel_stats": {
            name: {
                "duration": stats_dict(stats.duration),
                "aic_mac_mean": stats.aic_mac_mean,
                "aic_scalar_mean": stats.aic_scalar_mean,
                "aiv_scalar_mean": stats.aiv_scalar_mean,
                "aiv_vec_mean": stats.aiv_vec_mean,
                "cube_util_mean": stats.cube_util_mean,
            }
            for name, stats in summary.kernel_stats.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize K8V4 profiler metrics from Ascend torch profiler output.",
    )
    parser.add_argument(
        "profile_dirs",
        nargs="+",
        help="rank0_*_ascend_pt dir, or parent dir containing exactly one such subdir",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="display labels for each profile dir (default: directory basename)",
    )
    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=1,
        help="skip first N unified_attention layers when computing steady-state (default: 1)",
    )
    parser.add_argument(
        "--skip-tail",
        type=int,
        default=1,
        help="skip last N unified_attention layers when computing steady-state (default: 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of text report",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="when 2+ dirs are given, print delta against the first dir",
    )
    args = parser.parse_args()

    labels = args.labels or [Path(p).name for p in args.profile_dirs]
    if len(labels) != len(args.profile_dirs):
        raise SystemExit("labels count must match profile_dirs")

    summaries = [
        summarize_profile(
            Path(path),
            skip_warmup=args.skip_warmup,
            skip_tail=args.skip_tail,
        )
        for path in args.profile_dirs
    ]

    if args.json:
        payload = {label: _to_json(summary) for label, summary in zip(labels, summaries)}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for label, summary in zip(labels, summaries):
        _print_summary(label, summary)

    if args.compare and len(summaries) >= 2:
        base = summaries[0]
        for label, summary in zip(labels[1:], summaries[1:]):
            _print_compare(labels[0], base, label, summary)


if __name__ == "__main__":
    main()
