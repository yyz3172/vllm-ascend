#!/usr/bin/env python3
"""Summarize TurboQuant pack durations from an Ascend trace_view.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


PY_WRAPPER_NAME = (
    "vllm_ascend/ops/turboquant_kv_cache.py(647): "
    "turboquant_pack_kv_for_cache"
)
KERNEL_NAME = "TurboquantPackKvForCacheFused"


def _trace_path(path: str) -> Path:
    trace_path = Path(path)
    if trace_path.is_dir():
        trace_path = trace_path / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace_view.json not found: {trace_path}")
    return trace_path


def _durations(trace_path: Path, event_name: str) -> list[float]:
    with trace_path.open("r", encoding="utf-8") as f:
        events = json.load(f)
    return [
        float(event["dur"])
        for event in events
        if event.get("ph") == "X"
        and event.get("name") == event_name
        and "dur" in event
    ]


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"avg_us": 0.0, "count": 0, "max_us": 0.0, "min_us": 0.0}
    return {
        "avg_us": mean(values),
        "count": len(values),
        "max_us": max(values),
        "min_us": min(values),
    }


def _collect(path: str) -> dict[str, object]:
    trace_path = _trace_path(path)
    return {
        "trace": str(trace_path),
        "python_wrapper": _stats(_durations(trace_path, PY_WRAPPER_NAME)),
        "kernel": _stats(_durations(trace_path, KERNEL_NAME)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute average TurboQuant pack durations from trace_view.json"
    )
    parser.add_argument(
        "trace",
        help=(
            "Profiler directory containing ASCEND_PROFILER_OUTPUT/trace_view.json "
            "or the trace_view.json path"
        ),
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline profiler directory or trace_view.json for delta.",
    )
    args = parser.parse_args()

    metrics = _collect(args.trace)
    if args.baseline:
        baseline = _collect(args.baseline)
        metrics["baseline"] = baseline
        metrics["delta_pct"] = {}
        for key in ("python_wrapper", "kernel"):
            base = float(baseline[key]["avg_us"])
            cur = float(metrics[key]["avg_us"])
            metrics["delta_pct"][key] = 0.0 if base == 0.0 else (cur - base) / base * 100.0

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
