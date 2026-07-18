#!/usr/bin/env python3
"""Summarize K8V4 BitResidual / TurboQuant metrics from Ascend profiler output.

Reads ``ASCEND_PROFILER_OUTPUT/operator_details.csv`` and
``ASCEND_PROFILER_OUTPUT/kernel_details.csv`` under one or more
``rank0_*_ascend_pt`` directories produced by ``xrx_bit_residual_k8v4_smoke.py``
(with ``XRX_K8V4_PROFILE=1``) or similar torch-profiler runs.

Attention kernels are also split into decode vs prefill using the query token
count from ``Input Shapes`` (first dim of the first tensor). Default rule for
long-query profiles: ``Q <= --decode-q-max`` → decode, else prefill
(typical: decode Q=num_prompts, chunked-prefill Q≈241).

Example::

    python tests/e2e/singlecard/summarize_k8v4_profiler.py \\
        /root/yyz/pytorch_profiler/BitResidual/260713/k8v4_0.6B/rank0_565817_20260713062511278_ascend_pt

    python tests/e2e/singlecard/summarize_k8v4_profiler.py \\
        --compare --labels baseline optimized \\
        profile_a/ profile_b/

    python tests/e2e/singlecard/summarize_k8v4_profiler.py \\
        --compare -v baseline/ optimized/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
        "fia_wrapper": "_C_ascend::bit_residual_fia_paged_k8v4",
        "pack_aclnn": "aclnnBitResidualPackK8v4",
        "attn_aclnn": "aclnnBitResidualAttentionPagedK8v4",
        "fia_aclnn": "aclnnBitResidualFiaPagedK8v4",
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
        "BitResidualFiaPagedK8v4",
    ),
    "turboquant": (
        "TurboquantPackKvForCacheK8v4",
        "TurboquantFusedInferAttentionScoreK8v4",
    ),
}

# Kernels whose first Input Shapes dim is query tokens (TND / batched Q).
ATTN_PHASE_KERNELS = {
    "BitResidualAttentionPagedK8v4",
    "BitResidualFiaPagedK8v4",
    "TurboquantFusedInferAttentionScoreK8v4",
}

_SHAPE_FIRST_DIM_RE = re.compile(r'["\s]*(\d+)\s*,')


def _percentile(ordered: list[float], p: float) -> float:
    """Nearest-rank percentile on a sorted ascending list (p in (0, 1])."""
    if not ordered:
        return 0.0
    n = len(ordered)
    idx = min(n - 1, max(0, math.ceil(n * p) - 1))
    return ordered[idx]


@dataclass
class DurationStats:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0

    @classmethod
    def from_values(cls, values: Iterable[float]) -> DurationStats:
        vals = [float(v) for v in values]
        if not vals:
            return cls()
        ordered = sorted(vals)
        return cls(
            count=len(vals),
            mean=statistics.mean(vals),
            median=statistics.median(vals),
            p90=_percentile(ordered, 0.90),
            p99=_percentile(ordered, 0.99),
            min=ordered[0],
            max=ordered[-1],
            sum=sum(vals),
        )

    def value(self, stat: str = "mean") -> float:
        if stat == "mean":
            return self.mean
        if stat == "p90":
            return self.p90
        if stat == "p99":
            return self.p99
        return self.median

    def fmt(self, stat: str = "mean", *, detail: bool = False) -> str:
        if self.count == 0:
            return "n/a"
        abbr = _STAT_ABBREV.get(stat, stat)
        primary = _fmt_dur(self.value(stat))
        if not detail:
            return f"n={self.count} {abbr}={primary}"
        extras = []
        if stat != "median":
            extras.append(f"med={_fmt_dur(self.median)}")
        if stat != "mean":
            extras.append(f"mean={_fmt_dur(self.mean)}")
        if stat != "p90":
            extras.append(f"p90={_fmt_dur(self.p90)}")
        if stat != "p99":
            extras.append(f"p99={_fmt_dur(self.p99)}")
        return f"n={self.count} {abbr}={primary} ({' '.join(extras)})"

    def fmt_triple(self) -> str:
        """Compact n=… med/mean/p90=… line."""
        if self.count == 0:
            return "n/a"
        return f"n={self.count} {_fmt_med_mean_p90(self.median, self.mean, self.p90)}"


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
    # Keys: "{kernel}[{phase} Q={n}]" e.g. BitResidualAttentionPagedK8v4[decode Q=16]
    kernel_phase_stats: dict[str, KernelStats] = field(default_factory=dict)
    decode_q_max: int = 32
    host_overhead: dict[str, DurationStats] = field(default_factory=dict)


# Absolute delta below this (us) is treated as noise in compare tables.
_COMPARE_NOISE_US = 1.0
_STAT_ABBREV = {"median": "med", "mean": "mean", "p90": "p90", "p99": "p99"}


def _fmt_med_mean_p90(median_us: float, mean_us: float, p90_us: float) -> str:
    """Format med/mean/p90 with a shared unit when possible."""
    vals = (median_us, mean_us, p90_us)
    use_ms = all(abs(v) >= 1000.0 for v in vals)
    if use_ms:
        body = "/".join(f"{v / 1000.0:.2f}" for v in vals)
        unit = "ms"
    else:
        body = "/".join(f"{v:.1f}" for v in vals)
        unit = "us"
    return f"med/mean/p90={body} {unit}"


def _fmt_dur(us: float, *, width: int | None = None) -> str:
    """Format microseconds with automatic us/ms scaling."""
    if abs(us) >= 1000.0:
        text = f"{us / 1000.0:.2f} ms"
    else:
        text = f"{us:.1f} us"
    if width is None:
        return text
    return f"{text:>{width}}"


def _fmt_delta_us(delta_us: float) -> str:
    sign = "+" if delta_us >= 0 else "-"
    mag = abs(delta_us)
    if mag >= 1000.0:
        return f"{sign}{mag / 1000.0:.2f} ms"
    return f"{sign}{mag:.1f} us"


def _speedup(base_us: float, other_us: float) -> float | None:
    if other_us <= 0 or base_us <= 0:
        return None
    return base_us / other_us


def _pct_change(base_us: float, other_us: float) -> float | None:
    if base_us == 0:
        return None
    return (other_us - base_us) / base_us * 100.0


def _fmt_speedup(base_us: float, other_us: float) -> str:
    ratio = _speedup(base_us, other_us)
    if ratio is None:
        return "   n/a"
    return f"{ratio:5.1f}×"


def _fmt_pct(base_us: float, other_us: float) -> str:
    pct = _pct_change(base_us, other_us)
    if pct is None:
        return "   n/a"
    return f"{pct:+.1f}%"


def _parse_phase_key(key: str) -> tuple[str, str] | None:
    """Split 'Kernel[decode Q=16]' -> ('Kernel', 'decode Q=16')."""
    if "[" not in key or not key.endswith("]"):
        return None
    base, rest = key.split("[", 1)
    return base, rest[:-1]


def _attn_kernel_names(summary: ProfileSummary) -> list[str]:
    names = [
        n
        for n in summary.kernel_stats
        if n in ATTN_PHASE_KERNELS or "Attention" in n or "Fia" in n or "InferAttention" in n
    ]
    return sorted(names)


def _pack_kernel_names(summary: ProfileSummary) -> list[str]:
    names = [n for n in summary.kernel_stats if "Pack" in n]
    return sorted(names)


def _fmt_hw(stats: KernelStats) -> str:
    return (
        f"aic_mac={stats.aic_mac_mean:.2f} "
        f"aic_scalar={stats.aic_scalar_mean:.2f} "
        f"aiv_scalar={stats.aiv_scalar_mean:.1f} "
        f"aiv_vec={stats.aiv_vec_mean:.1f}"
    )


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


def _parse_query_tokens(input_shapes: str) -> int | None:
    """Return query token count (first dim of first tensor) from Input Shapes."""
    if not input_shapes:
        return None
    match = _SHAPE_FIRST_DIM_RE.match(input_shapes.strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _phase_for_q(q_tokens: int | None, decode_q_max: int) -> str:
    if q_tokens is None:
        return "unknown"
    if q_tokens <= decode_q_max:
        return "decode"
    return "prefill"


def _kernel_stats_from_rows(rows: list[dict[str, float]]) -> KernelStats:
    return KernelStats(
        duration=DurationStats.from_values(r["duration_us"] for r in rows),
        aic_mac_mean=statistics.mean(r["aic_mac_us"] for r in rows),
        aic_scalar_mean=statistics.mean(r["aic_scalar_us"] for r in rows),
        aiv_scalar_mean=statistics.mean(r["aiv_scalar_us"] for r in rows),
        aiv_vec_mean=statistics.mean(r["aiv_vec_us"] for r in rows),
        cube_util_mean=statistics.mean(r["cube_util"] for r in rows),
    )


def _summarize_kernels(
    profile_dir: Path,
    flavor: str,
    decode_q_max: int = 32,
) -> tuple[dict[str, KernelStats], dict[str, KernelStats]]:
    csv_path = profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if not csv_path.exists():
        return {}, {}

    targets = set(KERNEL_TARGETS.get(flavor, ()))
    buckets: dict[str, list[dict[str, float]]] = {name: [] for name in targets}
    phase_buckets: dict[str, list[dict[str, float]]] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Name", "")
            if name not in buckets:
                continue
            metrics = {
                "duration_us": _f(row, "Duration(us)"),
                "aic_mac_us": _f(row, "aic_mac_time(us)"),
                "aic_scalar_us": _f(row, "aic_scalar_time(us)"),
                "aiv_scalar_us": _f(row, "aiv_scalar_time(us)"),
                "aiv_vec_us": _f(row, "aiv_vec_time(us)"),
                "cube_util": _f(row, "cube_utilization(%)"),
            }
            buckets[name].append(metrics)
            if name in ATTN_PHASE_KERNELS:
                q_tokens = _parse_query_tokens(row.get("Input Shapes", "") or "")
                phase = _phase_for_q(q_tokens, decode_q_max)
                q_label = "Q=?" if q_tokens is None else f"Q={q_tokens}"
                key = f"{name}[{phase} {q_label}]"
                phase_buckets.setdefault(key, []).append(metrics)

    out = {
        name: _kernel_stats_from_rows(rows)
        for name, rows in buckets.items()
        if rows
    }
    phase_out = {
        key: _kernel_stats_from_rows(rows)
        for key, rows in sorted(phase_buckets.items())
        if rows
    }
    return out, phase_out


def summarize_profile(
    profile_dir: Path,
    skip_warmup: int = 1,
    skip_tail: int = 1,
    decode_q_max: int = 32,
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

    kernel_stats, kernel_phase_stats = _summarize_kernels(
        profile_dir, flavor, decode_q_max=decode_q_max
    )
    return ProfileSummary(
        profile_dir=profile_dir,
        flavor=flavor,
        layer_blocks=layer_blocks,
        steady_blocks=steady_blocks,
        operator_stats=operator_stats,
        kernel_stats=kernel_stats,
        kernel_phase_stats=kernel_phase_stats,
        decode_q_max=decode_q_max,
        host_overhead=host_overhead,
    )


def _print_op_row(
    label: str,
    stats: DurationStats | None,
    *,
    indent: int = 2,
) -> None:
    if stats is None or stats.count == 0:
        return
    pad = " " * indent
    print(f"{pad}{label:30s} {stats.fmt_triple()}")


def _print_kernel_row(
    label: str,
    stats: KernelStats,
    *,
    verbose: bool,
    indent: int = 2,
) -> None:
    pad = " " * indent
    dur = stats.duration.fmt_triple()
    cube = f"cube={stats.cube_util_mean:.1f}%"
    line = f"{pad}{label:30s} {dur}  {cube}"
    if verbose:
        line += f"  {_fmt_hw(stats)}"
    print(line)


def _host_device_ratio_line(summary: ProfileSummary) -> str | None:
    """One-liner: mean host time as % of mean device time for unified_attention."""
    host = summary.operator_stats.get("unified_attention_host")
    device = summary.operator_stats.get("unified_attention_device")
    if not host or not device or host.count == 0 or device.count == 0:
        return None
    if device.mean <= 0:
        return None
    pct = host.mean / device.mean * 100.0
    return (
        f"mean host {_fmt_dur(host.mean)} = {pct:.2f}% of "
        f"device {_fmt_dur(device.mean)}"
    )


def _print_summary(
    label: str,
    summary: ProfileSummary,
    *,
    verbose: bool = False,
) -> None:
    print(f"\n=== {label} ===")
    print(f"path   : {summary.profile_dir}")
    print(f"flavor : {summary.flavor}")
    print(
        f"layers : total={len(summary.layer_blocks)} "
        f"steady={len(summary.steady_blocks)}"
    )
    print(
        "stats  : med/mean/p90"
        + ("  (+ pack/attn device, wrappers, aten overhead, HW)" if verbose else "")
    )

    print("\n[latency / per-layer steady-state]")
    _print_op_row(
        "unified_attention device",
        summary.operator_stats.get("unified_attention_device"),
    )
    ratio = _host_device_ratio_line(summary)
    if ratio:
        print(f"  {'host / device':30s} {ratio}")

    if verbose:
        for row_label, key in (
            ("pack device", "pack_aclnn"),
            ("attn device", "attn_aclnn"),
            ("fia device", "fia_aclnn"),
            ("unified_attention host", "unified_attention_host"),
        ):
            _print_op_row(row_label, summary.operator_stats.get(key))
        print("\n[host wrappers]")
        for row_label, key in (
            ("pack_wrapper host", "pack_wrapper_host"),
            ("attn_wrapper host", "attn_wrapper_host"),
            ("fia_wrapper host", "fia_wrapper_host"),
        ):
            _print_op_row(row_label, summary.operator_stats.get(key))

    if summary.kernel_stats or summary.kernel_phase_stats:
        print(
            f"\n[kernel / decode vs prefill "
            f"(Q<={summary.decode_q_max} → decode)]"
        )
        for name in _attn_kernel_names(summary):
            print(f"  {name}")
            all_stats = summary.kernel_stats.get(name)
            if all_stats and all_stats.duration.count:
                _print_kernel_row(
                    "all",
                    all_stats,
                    verbose=verbose,
                    indent=4,
                )
            phase_keys = [
                k
                for k in summary.kernel_phase_stats
                if k.startswith(f"{name}[")
            ]
            # decode before prefill, then unknown
            def _phase_sort(key: str) -> tuple[int, str]:
                parsed = _parse_phase_key(key)
                phase = parsed[1] if parsed else key
                if phase.startswith("decode"):
                    return (0, phase)
                if phase.startswith("prefill"):
                    return (1, phase)
                return (2, phase)

            for key in sorted(phase_keys, key=_phase_sort):
                parsed = _parse_phase_key(key)
                phase_label = parsed[1] if parsed else key
                _print_kernel_row(
                    phase_label,
                    summary.kernel_phase_stats[key],
                    verbose=verbose,
                    indent=4,
                )

        for name in _pack_kernel_names(summary):
            _print_kernel_row(
                name,
                summary.kernel_stats[name],
                verbose=verbose,
                indent=2,
            )

    if verbose:
        overhead_items = [
            (op, summary.host_overhead.get(op))
            for op in HOST_OVERHEAD_OPS
            if summary.host_overhead.get(op) and summary.host_overhead[op].count
        ]
        if overhead_items:
            print("\n[host overhead inside attention blocks]")
            for op_name, stats in overhead_items:
                assert stats is not None
                dev_stats = summary.host_overhead.get(f"{op_name}_device")
                host_part = f"host {stats.fmt_triple()}"
                if dev_stats and dev_stats.count:
                    host_part += f"  device {dev_stats.fmt_triple()}"
                print(f"  {op_name:28s} {host_part}")


def _col_width(*labels: str, minimum: int = 10) -> int:
    return max(minimum, max(len(x) for x in labels))


def _print_compare_table_header(
    base_label: str,
    other_label: str,
    *,
    col_w: int,
) -> None:
    print(
        f"  {'metric':32s} {base_label:>{col_w}s} {other_label:>{col_w}s} "
        f"{'delta':>10s} {'speedup':>7s} {'pct':>8s}"
    )


def _print_compare_dur_row(
    metric: str,
    base_us: float,
    other_us: float,
    *,
    col_w: int,
    indent: int = 2,
    force: bool = False,
) -> bool:
    delta = other_us - base_us
    if not force and abs(delta) < _COMPARE_NOISE_US:
        return False
    pad = " " * indent
    b = _fmt_dur(base_us, width=col_w)
    o = _fmt_dur(other_us, width=col_w)
    d = f"{_fmt_delta_us(delta):>10}"
    sp = _fmt_speedup(base_us, other_us)
    pct = _fmt_pct(base_us, other_us)
    print(f"{pad}{metric:32s} {b} {o} {d} {sp:>7s} {pct:>8s}")
    return True


def _print_compare(
    base_label: str,
    base: ProfileSummary,
    other_label: str,
    other: ProfileSummary,
    *,
    verbose: bool = False,
) -> None:
    stat = "mean"
    print(f"\n=== compare: {base_label} vs {other_label} ({stat}) ===")
    col_w = _col_width(base_label, other_label, minimum=10)
    _print_compare_table_header(base_label, other_label, col_w=col_w)

    attn_names = sorted(
        set(_attn_kernel_names(base)) | set(_attn_kernel_names(other))
    )

    # Operator latency (ordered, skip noise)
    op_rows = [
        ("unified_attention device", "unified_attention_device"),
    ]
    if verbose:
        op_rows.extend(
            [
                ("attn device", "attn_aclnn"),
                ("fia device", "fia_aclnn"),
                ("pack device", "pack_aclnn"),
                ("unified_attention host", "unified_attention_host"),
                ("pack_wrapper host", "pack_wrapper_host"),
                ("attn_wrapper host", "attn_wrapper_host"),
                ("fia_wrapper host", "fia_wrapper_host"),
                ("pack_aclnn host", "pack_aclnn_host"),
                ("attn_aclnn host", "attn_aclnn_host"),
                ("fia_aclnn host", "fia_aclnn_host"),
            ]
        )
    for metric, key in op_rows:
        b = base.operator_stats.get(key)
        o = other.operator_stats.get(key)
        if not b or not o or b.count == 0 or o.count == 0:
            continue
        _print_compare_dur_row(
            metric,
            b.value(stat),
            o.value(stat),
            col_w=col_w,
            force=key == "unified_attention_device",
        )

    # Kernels: attn all + phases, then pack
    for name in attn_names:
        bs = base.kernel_stats.get(name)
        os_ = other.kernel_stats.get(name)
        if bs and os_ and bs.duration.count and os_.duration.count:
            _print_compare_dur_row(
                name,
                bs.duration.value(stat),
                os_.duration.value(stat),
                col_w=col_w,
                force=True,
            )

        phase_keys = [
            k
            for k in sorted(set(base.kernel_phase_stats) | set(other.kernel_phase_stats))
            if k.startswith(f"{name}[")
        ]

        def _phase_sort(key: str) -> tuple[int, str]:
            parsed = _parse_phase_key(key)
            phase = parsed[1] if parsed else key
            if phase.startswith("decode"):
                return (0, phase)
            if phase.startswith("prefill"):
                return (1, phase)
            return (2, phase)

        for key in sorted(phase_keys, key=_phase_sort):
            bks = base.kernel_phase_stats.get(key)
            oks = other.kernel_phase_stats.get(key)
            if not bks or not oks or not bks.duration.count or not oks.duration.count:
                continue
            parsed = _parse_phase_key(key)
            phase = parsed[1] if parsed else key
            _print_compare_dur_row(
                f"  {phase}",
                bks.duration.value(stat),
                oks.duration.value(stat),
                col_w=col_w,
                force=True,
            )

    for name in sorted(set(_pack_kernel_names(base)) | set(_pack_kernel_names(other))):
        bs = base.kernel_stats.get(name)
        os_ = other.kernel_stats.get(name)
        if not bs or not os_ or not bs.duration.count or not os_.duration.count:
            continue
        _print_compare_dur_row(
            name,
            bs.duration.value(stat),
            os_.duration.value(stat),
            col_w=col_w,
            force=True,
        )

    if verbose:
        print("\n[host overhead deltas]")
        _print_compare_table_header(base_label, other_label, col_w=col_w)
        for op_name in HOST_OVERHEAD_OPS:
            b = base.host_overhead.get(op_name)
            o = other.host_overhead.get(op_name)
            if not b or not o or b.count == 0 or o.count == 0:
                continue
            _print_compare_dur_row(
                op_name, b.value(stat), o.value(stat), col_w=col_w
            )


def _to_json(summary: ProfileSummary) -> dict:
    def stats_dict(stats: DurationStats) -> dict:
        return {
            "count": stats.count,
            "mean": stats.mean,
            "median": stats.median,
            "p90": stats.p90,
            "p99": stats.p99,
            "min": stats.min,
            "max": stats.max,
            "sum": stats.sum,
        }

    def kernel_dict(stats: KernelStats) -> dict:
        return {
            "duration": stats_dict(stats.duration),
            "aic_mac_mean": stats.aic_mac_mean,
            "aic_scalar_mean": stats.aic_scalar_mean,
            "aiv_scalar_mean": stats.aiv_scalar_mean,
            "aiv_vec_mean": stats.aiv_vec_mean,
            "cube_util_mean": stats.cube_util_mean,
        }

    return {
        "profile_dir": str(summary.profile_dir),
        "flavor": summary.flavor,
        "decode_q_max": summary.decode_q_max,
        "layers_total": len(summary.layer_blocks),
        "layers_steady": len(summary.steady_blocks),
        "operator_stats": {k: stats_dict(v) for k, v in summary.operator_stats.items()},
        "host_overhead": {k: stats_dict(v) for k, v in summary.host_overhead.items()},
        "kernel_stats": {
            name: kernel_dict(stats) for name, stats in summary.kernel_stats.items()
        },
        "kernel_phase_stats": {
            name: kernel_dict(stats)
            for name, stats in summary.kernel_phase_stats.items()
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "show host wrappers, host overhead inside attention blocks, "
            "and HW counters (aic/aiv)"
        ),
    )
    parser.add_argument(
        "--decode-q-max",
        type=int,
        default=32,
        help=(
            "classify attention kernels by Input Shapes query tokens: "
            "Q<=this → decode, else prefill (default: 32; long-query decode "
            "is usually num_prompts=16, chunked prefill Q≈241)"
        ),
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
            decode_q_max=args.decode_q_max,
        )
        for path in args.profile_dirs
    ]

    if args.json:
        payload = {label: _to_json(summary) for label, summary in zip(labels, summaries)}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.compare and len(summaries) >= 2:
        base = summaries[0]
        for label, summary in zip(labels[1:], summaries[1:]):
            _print_compare(
                labels[0],
                base,
                label,
                summary,
                verbose=args.verbose,
            )

    for label, summary in zip(labels, summaries):
        _print_summary(label, summary, verbose=args.verbose)


if __name__ == "__main__":
    main()
