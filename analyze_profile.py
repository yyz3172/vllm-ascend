#!/usr/bin/env python3
"""Analyze ASCEND profiler trace_view.json for specific function durations.

Extracts and calculates average execution time for:
- vllm_ascend/attention/attention_v1.py: reshape_and_cache
- vllm_ascend/ops/turboquant_kv_cache.py: turboquant_pack_kv_for_cache_to_cache
- aten::contiguous
- TurboquantPackKvForCacheFused (or TurboquantPackKvForCacheToCache)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Pattern


def find_trace_view_json(base_dir: str) -> Path:
    """Find trace_view.json in ASCEND_PROFILER_OUTPUT directory.

    Args:
        base_dir: Base directory to search in.

    Returns:
        Path to trace_view.json file.

    Raises:
        FileNotFoundError: If trace_view.json is not found.
    """
    trace_path = Path(base_dir) / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"trace_view.json not found at {trace_path}"
        )
    return trace_path


def load_trace_data(trace_path: Path) -> list:
    """Load trace data from JSON file.

    Args:
        trace_path: Path to trace_view.json file.

    Returns:
        List of trace events.
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_events_exact(trace_data: list, name: str) -> list[dict]:
    """Extract durations for events with exact name match.

    Args:
        trace_data: List of trace events.
        name: Exact event name to match.

    Returns:
        List of matching trace events.
    """
    return [
        event
        for event in trace_data
        if event.get("name") == name and "dur" in event
    ]


def extract_events_regex(trace_data: list, pattern: Pattern[str]) -> list[dict]:
    """Extract events whose name matches a regex pattern."""
    return [
        event
        for event in trace_data
        if pattern.search(event.get("name", "")) and "dur" in event
    ]


def calculate_average(durations: list) -> float:
    """Calculate average of a list of durations.

    Args:
        durations: List of duration values.

    Returns:
        Average duration, or 0.0 if list is empty.
    """
    if not durations:
        return 0.0
    return sum(durations) / len(durations)


def percentile(sorted_durations: list[float], ratio: float) -> float:
    if not sorted_durations:
        return 0.0
    index = int((len(sorted_durations) - 1) * ratio)
    return sorted_durations[index]


def print_duration_summary(events: list[dict], indent: str = "  ") -> None:
    durations = sorted(float(event["dur"]) for event in events)
    count = len(durations)
    print(f"{indent}Count: {count}")
    if count == 0:
        return

    avg_without_max = (
        calculate_average(durations[:-1])
        if count > 1
        else durations[0]
    )
    names = sorted({event.get("name", "") for event in events})
    print(f"{indent}Average duration: {calculate_average(durations):.2f} us")
    print(f"{indent}Min duration: {durations[0]:.2f} us")
    print(f"{indent}P50 duration: {percentile(durations, 0.50):.2f} us")
    print(f"{indent}P90 duration: {percentile(durations, 0.90):.2f} us")
    print(f"{indent}P99 duration: {percentile(durations, 0.99):.2f} us")
    print(f"{indent}Max duration: {durations[-1]:.2f} us")
    print(f"{indent}Average without max: {avg_without_max:.2f} us")
    if len(names) == 1:
        print(f"{indent}Matched event: {names[0]}")
    elif len(names) <= 5:
        print(f"{indent}Matched events: {', '.join(names)}")
    else:
        print(f"{indent}Matched events: {len(names)} unique names")


def analyze_profile(base_dir: str) -> None:
    """Analyze profile data and print average durations.

    Args:
        base_dir: Base directory containing profiler output.
    """
    trace_path = find_trace_view_json(base_dir)
    trace_data = load_trace_data(trace_path)

    # Keep the historical pattern labels stable, but make Python function
    # matching line-number independent so source edits do not break analysis.
    patterns = [
        (
            "vllm_ascend/attention/attention_v1.py(...): forward",
            "vllm_ascend/attention/attention_v1.py(1212): forward",
            re.compile(
                r"^vllm_ascend/attention/attention_v1\.py\(\d+\): forward$"
            ),
        ),
        (
            "vllm_ascend/attention/attention_v1.py(...): reshape_and_cache",
            "vllm_ascend/attention/attention_v1.py(1112): reshape_and_cache",
            re.compile(
                r"^vllm_ascend/attention/attention_v1\.py\(\d+\): "
                r"reshape_and_cache$"
            ),
        ),
        (
            "vllm_ascend/ops/turboquant_kv_cache.py(...): turboquant_pack_kv_for_cache_to_cache",
            "vllm_ascend/ops/turboquant_kv_cache.py(1160): turboquant_pack_kv_for_cache_to_cache",
            re.compile(
                r"^vllm_ascend/ops/turboquant_kv_cache\.py\(\d+\): "
                r"turboquant_pack_kv_for_cache_to_cache$"
            ),
        ),
        ("aten::contiguous", "aten::contiguous", None),
        ("BitResidualPackK8v4", "BitResidualPackK8v4", None),
        ("BitResidualAttentionPagedK8v4", "BitResidualAttentionPagedK8v4", None),
        ("TurboquantPackKvForCache4bit", "TurboquantPackKvForCache4bit", None),
        ("TurboquantAttentionPaged4bit", "TurboquantAttentionPaged4bit", None),
        ("TurboquantAttentionPaged8bit", "TurboquantAttentionPaged8bit", None),
        ("TurboquantPackKvForCacheFused", "TurboquantPackKvForCacheFused", None),
        (
            "TurboquantPackKvForCacheToCache",
            "TurboquantPackKvForCacheToCache",
            None,
        ),
        ("TurboquantPackKvForCacheV2", "TurboquantPackKvForCacheV2", None),
        ("TurboquantPackKvForCacheV3", "TurboquantPackKvForCacheV3", None),
    ]

    print(f"Analyzing profile data from: {trace_path}")
    print("=" * 60)

    for label, exact, regex_pattern in patterns:
        events = extract_events_exact(trace_data, exact)

        print(f"\nPattern: {label}")
        if events:
            print_duration_summary(events)
            continue

        print("  No exact match found")
        if regex_pattern is not None:
            events = extract_events_regex(trace_data, regex_pattern)
            if events:
                print("  Regex match found:")
                print_duration_summary(events, indent="    ")
                continue


def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <profile_directory>")
        print("\nExample:")
        print(f"  {sys.argv[0]} /root/x00827378/perflog2/rank0_365135_20260615111639773_ascend_pt")
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    analyze_profile(base_dir)


if __name__ == "__main__":
    main()
