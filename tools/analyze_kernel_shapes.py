#!/usr/bin/env python3
"""Analyze ASCEND profiler kernel_details.csv by Input Shapes for a given kernel.

Groups kernel events by their Input Shapes, computes per-shape duration statistics
and AIV pipeline breakdown (scalar/vec/mte2/mte3), identifies bottleneck type,
and prints a Shape Summary table.

Usage:
  python tools/analyze_kernel_shapes.py <profile_dir> <kernel_name>

Example:
  python tools/analyze_kernel_shapes.py \
      mytmp/perflog_k8v4/rank0_1126175_20260714110025489_ascend_pt \
      BitResidualPackK8v4

The <profile_dir> should contain ASCEND_PROFILER_OUTPUT/kernel_details.csv.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def find_kernel_details_csv(base_dir: str) -> Path:
    csv_path = Path(base_dir) / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"kernel_details.csv not found at {csv_path}")
    return csv_path


def parse_shape_field(shape_str: str) -> list[list[int]]:
    """Parse a kernel_details Input Shapes field like "1936,8,128;1936,8,128;128,128;1936;2"
    into [[1936,8,128], [1936,8,128], [128,128], [1936], [2]].

    Empty `;`-separated slots (optional / absent tensors, e.g. ``16;16;;128,128``)
    are kept as ``[]`` so positional indices stay aligned.
    """
    shape_str = shape_str.strip().strip('"')
    if not shape_str:
        return []
    shapes: list[list[int]] = []
    for part in shape_str.split(";"):
        part = part.strip()
        if not part:
            shapes.append([])
            continue
        shapes.append([int(x) for x in part.split(",") if x.strip()])
    return shapes


def compute_pct(value: float, total: float) -> float:
    return value / total * 100.0 if total > 0 else 0.0


def analyze_kernel_shapes(base_dir: str, kernel_name: str) -> None:
    csv_path = find_kernel_details_csv(base_dir)
    print("Profile dir:  %s" % base_dir)
    print("Kernel:       %s" % kernel_name)
    print("CSV source:   %s" % csv_path)
    print()

    # Read CSV
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)

        col = {name: header.index(name) for name in [
            "Name", "Input Shapes", "Output Shapes", "Input Data Types",
            "Duration(us)", "aiv_vec_time(us)", "aiv_scalar_time(us)",
            "aiv_mte2_time(us)", "aiv_mte3_time(us)",
            "aiv_scalar_ratio", "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
            "aic_scalar_time(us)", "aic_mac_time(us)",
        ]}

        groups = defaultdict(list)
        for row in reader:
            if kernel_name not in row[col["Name"]]:
                continue
            groups[row[col["Input Shapes"]]].append({
                "output": row[col["Output Shapes"]],
                "input_dtype": row[col["Input Data Types"]],
                "dur": float(row[col["Duration(us)"]]),
                "aiv_vec": float(row[col["aiv_vec_time(us)"]]),
                "aiv_scalar": float(row[col["aiv_scalar_time(us)"]]),
                "aiv_mte2": float(row[col["aiv_mte2_time(us)"]]),
                "aiv_mte3": float(row[col["aiv_mte3_time(us)"]]),
                "aic_scalar": float(row[col["aic_scalar_time(us)"]]),
                "aic_mac": float(row[col["aic_mac_time(us)"]]),
            })

    total_events = sum(len(v) for v in groups.values())
    if total_events == 0:
        print("No events found for kernel '%s'" % kernel_name)
        return

    print("Total events: %d  Unique shapes: %d" % (total_events, len(groups)))
    print("=" * 90)

    # Per-shape detail
    shape_info = []
    for shape_str, events in sorted(groups.items(),
                                    key=lambda x: -sum(e["dur"] for e in x[1]) / len(x[1])):
        durs = sorted(e["dur"] for e in events)
        avg_dur = sum(durs) / len(durs)
        avg_vec = sum(e["aiv_vec"] for e in events) / len(events)
        avg_scalar = sum(e["aiv_scalar"] for e in events) / len(events)
        avg_mte2 = sum(e["aiv_mte2"] for e in events) / len(events)
        avg_mte3 = sum(e["aiv_mte3"] for e in events) / len(events)

        shapes = parse_shape_field(shape_str)
        rows = shapes[0][0] if len(shapes) > 0 and len(shapes[0]) > 0 else 0
        # For BitResidualPackK8v4: 5th input is query_start_loc.
        # query_start_loc values are prefix token offsets; profiler Input Shapes
        # only gives its dim0, which is num_reqs + 1.
        q_start_loc_dim0 = shapes[4][0] if len(shapes) > 4 and len(shapes[4]) > 0 else 0
        num_reqs = q_start_loc_dim0 - 1 if q_start_loc_dim0 > 0 else 0

        bottleneck = "SCALAR" if avg_scalar > avg_vec else "VEC"

        p50_idx = len(durs) // 2
        p90_idx = min(int(len(durs) * 0.9), len(durs) - 1)

        shape_info.append({
            "shape_str": shape_str,
            "rows": rows,
            "q_start_loc_dim0": q_start_loc_dim0,
            "num_reqs": num_reqs,
            "count": len(events),
            "avg_dur": avg_dur,
            "p50": durs[p50_idx],
            "p90": durs[p90_idx],
            "max": durs[-1],
            "avg_vec": avg_vec,
            "avg_scalar": avg_scalar,
            "avg_mte2": avg_mte2,
            "avg_mte3": avg_mte3,
            "bottleneck": bottleneck,
            "dur_per_row": avg_dur / rows if rows > 0 else 0,
            "scalar_per_row": avg_scalar / rows if rows > 0 else 0,
            "vec_per_row": avg_vec / rows if rows > 0 else 0,
            "rows_per_req": rows / num_reqs if num_reqs > 0 else 0,
            "output": events[0]["output"],
            "input_dtype": events[0]["input_dtype"],
        })

    # Detail per shape
    for info in shape_info:
        print()
        print("Input Shape:  %s" % info["shape_str"])
        print("Output Shape: %s" % info["output"])
        print("Input Dtype:  %s" % info["input_dtype"])
        print("Rows=%d, QStartLocDim0=%d, NumReqs=%d, Count=%d" %
              (info["rows"], info["q_start_loc_dim0"], info["num_reqs"], info["count"]))
        print("Avg Duration: %.2fus  P50: %.2fus  P90: %.2fus  Max: %.2fus" %
              (info["avg_dur"], info["p50"], info["p90"], info["max"]))
        print()
        print("  AIV Pipeline Time Breakdown:")
        print("    aiv_scalar: %.2fus (%.1f%%)" %
              (info["avg_scalar"], compute_pct(info["avg_scalar"], info["avg_dur"])))
        print("    aiv_vec:    %.2fus (%.1f%%)" %
              (info["avg_vec"], compute_pct(info["avg_vec"], info["avg_dur"])))
        print("    aiv_mte2:   %.2fus (%.1f%%)" %
              (info["avg_mte2"], compute_pct(info["avg_mte2"], info["avg_dur"])))
        print("    aiv_mte3:   %.2fus (%.1f%%)" %
              (info["avg_mte3"], compute_pct(info["avg_mte3"], info["avg_dur"])))
        print()
        print("  Per-row: %.1fus/row (scalar %.1fus/row, vec %.1fus/row)" %
              (info["dur_per_row"], info["scalar_per_row"], info["vec_per_row"]))
        print("  Avg rows/request: %.1f" % info["rows_per_req"])
        print("  Bottleneck: %s" % info["bottleneck"])

    # Summary table
    print()
    print("=" * 90)
    print("SHAPE SUMMARY TABLE")
    print("=" * 90)
    header_fmt = "%-6s %-10s %-8s %-10s %-7s %-8s %-7s %-7s %-7s %-9s"
    print(header_fmt % (
        "Rows", "QStartDim", "NumReqs", "Count", "Avg(us)", "Scalar%", "Vec%",
        "MTE2%", "MTE3%", "Bottleneck"))
    print("-" * 90)
    for info in sorted(shape_info, key=lambda x: -x["avg_dur"]):
        print(header_fmt % (
            info["rows"], info["q_start_loc_dim0"], info["num_reqs"], info["count"],
            "%.0f" % info["avg_dur"],
            "%.1f" % compute_pct(info["avg_scalar"], info["avg_dur"]),
            "%.1f" % compute_pct(info["avg_vec"], info["avg_dur"]),
            "%.1f" % compute_pct(info["avg_mte2"], info["avg_dur"]),
            "%.1f" % compute_pct(info["avg_mte3"], info["avg_dur"]),
            info["bottleneck"]))

    # Per-row comparison
    print()
    print("=" * 90)
    print("PER-ROW COST COMPARISON")
    print("=" * 90)
    row_fmt = "Rows=%-6d Reqs=%-3d | dur=%-7.0fus (%.1fus/r) | scalar=%-7.0fus (%.1fus/r) | vec=%-6.0fus (%.1fus/r)"
    for info in sorted(shape_info, key=lambda x: -x["avg_dur"]):
        print(row_fmt % (
            info["rows"], info["num_reqs"],
            info["avg_dur"], info["dur_per_row"],
            info["avg_scalar"], info["scalar_per_row"],
            info["avg_vec"], info["vec_per_row"]))


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: %s <profile_dir> <kernel_name>" % sys.argv[0])
        print()
        print("Example:")
        print("  %s mytmp/perflog_k8v4/rank0_xxx_ascend_pt BitResidualPackK8v4" % sys.argv[0])
        sys.exit(1)

    base_dir = sys.argv[1]
    kernel_name = sys.argv[2]

    if not Path(base_dir).is_dir():
        print("Error: Directory not found: %s" % base_dir)
        sys.exit(1)

    analyze_kernel_shapes(base_dir, kernel_name)


if __name__ == "__main__":
    main()
