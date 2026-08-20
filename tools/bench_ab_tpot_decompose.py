#!/usr/bin/env python3
"""Build NOTQ vs TQ TPOT decomposition from bench logs + Ascend profiler dumps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.e2e.singlecard.summarize_k8v4_profiler import (  # noqa: E402
    ProfileSummary,
    _pack_kernel_names,
    _resolve_profile_dir,
    summarize_profile,
)

_NUM_LAYERS = 36


@dataclass
class BenchMetrics:
    median_tpot_ms: float | None = None
    mean_tpot_ms: float | None = None
    median_ttft_ms: float | None = None
    mean_ttft_ms: float | None = None
    out_tok_s: float | None = None
    duration_s: float | None = None
    req_throughput: float | None = None


def _f(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    return float(m.group(1))


def parse_bench_log(path: Path) -> BenchMetrics:
    text = path.read_text(encoding="utf-8", errors="replace")
    return BenchMetrics(
        median_tpot_ms=_f(text, r"Median TPOT \(ms\):\s+([\d.]+)"),
        mean_tpot_ms=_f(text, r"Mean TPOT \(ms\):\s+([\d.]+)"),
        median_ttft_ms=_f(text, r"Median TTFT \(ms\):\s+([\d.]+)"),
        mean_ttft_ms=_f(text, r"Mean TTFT \(ms\):\s+([\d.]+)"),
        out_tok_s=_f(text, r"Output token throughput \(tok/s\):\s+([\d.]+)"),
        duration_s=_f(text, r"Benchmark duration \(s\):\s+([\d.]+)"),
        req_throughput=_f(text, r"Request throughput \(req/s\):\s+([\d.]+)"),
    )


def _pick_attn_us(summary: ProfileSummary) -> float:
    kernel_candidates = [
        n
        for n in summary.kernel_stats
        if any(
            k in n
            for k in (
                "FusedInferAttention",
                "BitResidualFia",
                "BitResidualAttention",
                "TurboquantFusedInfer",
            )
        )
    ]
    if kernel_candidates:
        return max(summary.kernel_stats[n].duration.median for n in kernel_candidates)
    ua = summary.operator_stats.get("unified_attention_device")
    if ua and ua.median:
        return ua.median
    return 0.0


def _pick_pack_us(summary: ProfileSummary) -> float:
    names = _pack_kernel_names(summary)
    if not names:
        return 0.0
    return max(summary.kernel_stats[n].duration.median for n in names)


def _step_trace_means(summary: ProfileSummary) -> dict[str, float]:
    if not summary.step_trace:
        return {}
    cols = ("Computing", "Stage", "Free", "Preparing")
    out: dict[str, float] = {}
    for col in cols:
        vals = [row[col] for row in summary.step_trace if col in row]
        if vals:
            out[col] = sum(vals) / len(vals)
    return out


def _load_summary(prof_parent: Path) -> ProfileSummary:
    prof_dir = _resolve_profile_dir(prof_parent, latest=True)
    return summarize_profile(prof_dir, skip_warmup=1, skip_tail=1, decode_q_max=32)


def _infer_query_tokens(summary: ProfileSummary) -> int:
    for key in summary.kernel_phase_stats:
        if "Q=" in key:
            try:
                return int(key.split("Q=")[-1].rstrip("]"))
            except ValueError:
                pass
    import csv

    csv_path = summary.profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = row.get("Name", "")
                if "Attention" not in name and "Fia" not in name and "FusedInfer" not in name:
                    continue
                shapes = row.get("Input Shapes", "").replace('"', "").split(";")
                if shapes and shapes[0]:
                    parts = shapes[0].split(",")
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])
                        except ValueError:
                            pass
                break
    return 32


def build_decomposition(
    notq_bench: BenchMetrics,
    tq_bench: BenchMetrics,
    notq_prof: ProfileSummary,
    tq_prof: ProfileSummary,
) -> dict:
    notq_attn = _pick_attn_us(notq_prof)
    tq_attn = _pick_attn_us(tq_prof)
    notq_pack = _pick_pack_us(notq_prof)
    tq_pack = _pick_pack_us(tq_prof)

    notq_fwd_ms = _NUM_LAYERS * (notq_attn + notq_pack) / 1000.0
    tq_fwd_ms = _NUM_LAYERS * (tq_attn + tq_pack) / 1000.0
    kernel_delta_ms = tq_fwd_ms - notq_fwd_ms

    q_tokens = max(_infer_query_tokens(notq_prof), _infer_query_tokens(tq_prof), 1)
    # One engine step may emit ~q_tokens × batch; TPOT is per client token — amortize.
    kernel_delta_per_tpot_ms = kernel_delta_ms / q_tokens

    notq_trace = _step_trace_means(notq_prof)
    tq_trace = _step_trace_means(tq_prof)

    median_tpot_delta = None
    if notq_bench.median_tpot_ms is not None and tq_bench.median_tpot_ms is not None:
        median_tpot_delta = tq_bench.median_tpot_ms - notq_bench.median_tpot_ms

    unattributed_ms = None
    if median_tpot_delta is not None:
        unattributed_ms = median_tpot_delta - kernel_delta_per_tpot_ms

    rows = [
        {
            "component": "bench_median_tpot_ms",
            "notq": notq_bench.median_tpot_ms,
            "tq": tq_bench.median_tpot_ms,
            "delta_tq_minus_notq": median_tpot_delta,
            "note": "端到端 decode token 间隔（含非算子）",
        },
        {
            "component": "bench_mean_ttft_ms",
            "notq": notq_bench.mean_ttft_ms,
            "tq": tq_bench.mean_ttft_ms,
            "delta_tq_minus_notq": (
                (tq_bench.mean_ttft_ms - notq_bench.mean_ttft_ms)
                if notq_bench.mean_ttft_ms and tq_bench.mean_ttft_ms
                else None
            ),
            "note": "prefill 主导，与 TPOT 分解分开看",
        },
        {
            "component": "out_tok_s",
            "notq": notq_bench.out_tok_s,
            "tq": tq_bench.out_tok_s,
            "delta_tq_minus_notq": (
                (tq_bench.out_tok_s - notq_bench.out_tok_s)
                if notq_bench.out_tok_s and tq_bench.out_tok_s
                else None
            ),
            "note": "吞吐",
        },
        {
            "component": f"kernel_attn_per_layer_us (×{_NUM_LAYERS})",
            "notq": notq_attn,
            "tq": tq_attn,
            "delta_tq_minus_notq": tq_attn - notq_attn,
            "note": "device attention kernel",
        },
        {
            "component": f"kernel_pack_per_layer_us (×{_NUM_LAYERS})",
            "notq": notq_pack,
            "tq": tq_pack,
            "delta_tq_minus_notq": tq_pack - notq_pack,
            "note": "TQ Pack；NOTQ≈0",
        },
        {
            "component": f"kernel_fwd_budget_ms (36L×(attn+pack), /Q={q_tokens})",
            "notq": round(notq_fwd_ms / q_tokens, 3),
            "tq": round(tq_fwd_ms / q_tokens, 3),
            "delta_tq_minus_notq": round(kernel_delta_per_tpot_ms, 3),
            "note": "按 profile Q 归一化到 per-token 粗算",
        },
        {
            "component": "kernel_fwd_budget_ms (36L, per engine step)",
            "notq": round(notq_fwd_ms, 3),
            "tq": round(tq_fwd_ms, 3),
            "delta_tq_minus_notq": round(kernel_delta_ms, 3),
            "note": "整步 forward，未除 Q",
        },
        {
            "component": "unattributed_to_kernels_ms",
            "notq": None,
            "tq": None,
            "delta_tq_minus_notq": (
                round(unattributed_ms, 3) if unattributed_ms is not None else None
            ),
            "note": "median_tpot_delta − kernel_fwd_budget_delta",
        },
        {
            "component": "profile_step_stage_mean_us",
            "notq": notq_trace.get("Stage"),
            "tq": tq_trace.get("Stage"),
            "delta_tq_minus_notq": (
                tq_trace.get("Stage", 0) - notq_trace.get("Stage", 0)
                if notq_trace.get("Stage") and tq_trace.get("Stage")
                else None
            ),
            "note": "每 engine step wall（含 Free）",
        },
        {
            "component": "profile_step_computing_mean_us",
            "notq": notq_trace.get("Computing"),
            "tq": tq_trace.get("Computing"),
            "delta_tq_minus_notq": (
                tq_trace.get("Computing", 0) - notq_trace.get("Computing", 0)
                if notq_trace.get("Computing") and tq_trace.get("Computing")
                else None
            ),
            "note": "NPU 算力时间 / step",
        },
        {
            "component": "profile_step_free_mean_us",
            "notq": notq_trace.get("Free"),
            "tq": tq_trace.get("Free"),
            "delta_tq_minus_notq": (
                tq_trace.get("Free", 0) - notq_trace.get("Free", 0)
                if notq_trace.get("Free") and tq_trace.get("Free")
                else None
            ),
            "note": "pipeline 空转 / 调度 bubble",
        },
    ]

    non_kernel_factors = [
        "MLP/RMSNorm/RoPE 等非 attention 层（profile 未逐项展开，通常 NOTQ≈TQ）",
        "Sampler / top-p / CPU-NPU sync（每 token 一次，通常 <1ms）",
        "Scheduler 凑 batch、async scheduling 导致的 step Free 增大",
        "Prefill/decode 混合 batch 形状变化 → ACL graph lazy capture（应被 num-warmups 避开）",
        "Python/engine 调度、请求排队（高并发时放大 TTFT）",
        "Chunked prefill（Q=32）下 KV 长度不同 → attention 绝对耗时不同",
    ]

    return {
        "num_layers": _NUM_LAYERS,
        "notq_profile_dir": str(notq_prof.profile_dir),
        "tq_profile_dir": str(tq_prof.profile_dir),
        "rows": rows,
        "non_kernel_factors": non_kernel_factors,
        "query_tokens_per_profile_step": q_tokens,
        "interpretation": {
            "kernel_explains_tpot_pct": (
                round(min(100.0, 100.0 * kernel_delta_per_tpot_ms / median_tpot_delta), 1)
                if median_tpot_delta and median_tpot_delta > 0 and kernel_delta_per_tpot_ms > 0
                else None
            ),
            "residual_tpot_ms": (
                round(unattributed_ms, 3) if unattributed_ms is not None else None
            ),
        },
    }


def _fmt_val(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1000 and abs(v) < 1_000_000:
        return f"{v / 1000.0:.3f} ms"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000.0:.3f} s"
    return f"{v:.3f}"


def format_table(data: dict) -> str:
    lines = [
        "=== TPOT / 端到端时延分解 (NOTQ vs TQ) ===",
        f"NOTQ profile: {data['notq_profile_dir']}",
        f"TQ profile:   {data['tq_profile_dir']}",
        "",
        f"{'component':<42} {'NOTQ':>14} {'TQ':>14} {'Δ(TQ-NOTQ)':>14}  note",
        "-" * 110,
    ]
    for row in data["rows"]:
        lines.append(
            f"{row['component']:<42} {_fmt_val(row['notq']):>14} "
            f"{_fmt_val(row['tq']):>14} {_fmt_val(row['delta_tq_minus_notq']):>14}  "
            f"{row.get('note', '')}"
        )
    pct = data["interpretation"].get("kernel_explains_tpot_pct")
    residual = data["interpretation"].get("residual_tpot_ms")
    lines.extend(
        [
            "",
            f"kernel_fwd_budget 可解释 median TPOT 增量的约: {pct if pct is not None else 'n/a'}%",
            f"仍未归因到 attn+pack 的 TPOT 增量: {residual if residual is not None else 'n/a'} ms",
            "",
            "除算子 device 时延外，影响端到端 TPOT/吞吐的因素:",
        ]
    )
    for i, factor in enumerate(data["non_kernel_factors"], 1):
        lines.append(f"  {i}. {factor}")
    return "\n".join(lines) + "\n"


def _find_bench_log(variant_dir: Path) -> Path:
    candidates = sorted(variant_dir.glob("mc*/mc*_r*_i*_o*/bench.log"))
    if not candidates:
        candidates = sorted(variant_dir.glob("logs/**/bench_*.log"))
    if not candidates:
        raise FileNotFoundError(f"bench.log not found under {variant_dir}")
    return candidates[-1]


def _find_prof_parent(variant_dir: Path, ab_root: Path, variant: str) -> Path:
    for p in (
        ab_root / f"{variant}_prof",
        variant_dir / "logs" / "mc16_dev7_port9555" / "profiler",
        variant_dir / "logs" / "**" / "profiler",
    ):
        if p.is_dir():
            try:
                _resolve_profile_dir(p, latest=True)
                return p
            except FileNotFoundError:
                continue
    # glob fallback
    hits = sorted(ab_root.glob(f"{variant}_prof/**/rank0_*_ascend_pt"), key=lambda x: x.stat().st_mtime)
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"profiler dir not found for {variant} under {ab_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NOTQ vs TQ TPOT decomposition table")
    parser.add_argument(
        "--ab-root",
        type=Path,
        help="A/B run root with notq/ tq/ and optional *_prof/",
    )
    parser.add_argument("--notq-bench-log", type=Path)
    parser.add_argument("--tq-bench-log", type=Path)
    parser.add_argument("--notq-profile-dir", type=Path)
    parser.add_argument("--tq-profile-dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="write table + JSON here")
    args = parser.parse_args()

    if args.ab_root:
        ab_root = args.ab_root.resolve()
        notq_bench = _find_bench_log(ab_root / "notq")
        tq_bench = _find_bench_log(ab_root / "tq")
        notq_prof_parent = _find_prof_parent(ab_root / "notq", ab_root, "notq")
        tq_prof_parent = _find_prof_parent(ab_root / "tq", ab_root, "tq")
    else:
        if not all(
            [
                args.notq_bench_log,
                args.tq_bench_log,
                args.notq_profile_dir,
                args.tq_profile_dir,
            ]
        ):
            parser.error("provide --ab-root or all four --notq-* / --tq-* paths")
        notq_bench = args.notq_bench_log
        tq_bench = args.tq_bench_log
        notq_prof_parent = args.notq_profile_dir
        tq_prof_parent = args.tq_profile_dir
        ab_root = args.output.parent if args.output else Path(".")

    data = build_decomposition(
        parse_bench_log(notq_bench),
        parse_bench_log(tq_bench),
        _load_summary(notq_prof_parent),
        _load_summary(tq_prof_parent),
    )
    text = format_table(data)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {args.output} and {json_path}")


if __name__ == "__main__":
    main()
