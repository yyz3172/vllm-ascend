#!/usr/bin/env python3
#
# Summarize outputs produced by profile_turboquant4bit_pack_branches.sh.

from __future__ import annotations

import argparse
import re
from pathlib import Path


CASE_ORDER = [
    "general_contiguous",
    "direct_contiguous",
    "contiguous_fast_contiguous",
    "contiguous_fast_scatter_groups",
    "contiguous_fast_swap_pairs",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        return None
    return float(match.group(1))


def first_str(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        return ""
    return match.group(1).strip()


def parse_pipe_metrics(text: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for prefix, line_re in (
        ("cube0", r"^\s*cube0 rows=.*$"),
        ("vector0", r"^\s*vector0 rows=.*$"),
        ("vector1", r"^\s*vector1 rows=.*$"),
    ):
        line = first_str(f"({line_re})", text)
        for name in (
            "aic_time",
            "aic_scalar_time",
            "aic_mte2_time",
            "aiv_time",
            "aiv_scalar_time",
            "aiv_mte2_time",
        ):
            value = first_float(rf"{name}\(us\)=([0-9.]+)/", line)
            if value is not None:
                metrics[f"{prefix}_{name}"] = value
    return metrics


def parse_top_repo_lines(text: str, limit: int) -> list[tuple[int, str]]:
    marker = "  Top repo source lines:\n"
    start = text.find(marker)
    if start < 0:
        return []
    tail = text[start + len(marker):]
    end = tail.find("  Top basic blocks:")
    if end >= 0:
        tail = tail[:end]

    rows: list[tuple[int, str]] = []
    for line in tail.splitlines():
        match = re.match(r"\s*count=(\d+) bb=\d+ (.*)$", line)
        if match is None:
            continue
        rows.append((int(match.group(1)), match.group(2)))
        if len(rows) >= limit:
            break
    return rows


def summarize_case(case_dir: Path, top_lines: int) -> dict[str, object]:
    hot = read_text(case_dir / "hotspot_lines.log")
    op_log = read_text(case_dir / "op.profile.log")
    run_script = read_text(case_dir / "run.sh")

    data: dict[str, object] = {
        "case": case_dir.name,
        "pack_mode": first_str(r"--pack-mode \"?([^\"\\\n ]+)", run_script),
        "slot_pattern": first_str(r"--slot-pattern \"?([^\"\\\n ]+)", run_script),
        "app_avg_us": first_float(r"pack4bit_to_cache_avg_us=([0-9.]+)", op_log),
        "duration_us": first_float(r"duration_us=([0-9.]+)", hot),
        "task_duration_us": first_float(r"Task Duration\(us\):\s*([0-9.]+)", op_log),
        "op_name": first_str(r"op=([^ ]+)", hot),
        "opprof": read_text(case_dir / "opprof_path.txt").strip(),
        "top_repo_lines": parse_top_repo_lines(hot, top_lines),
    }
    data.update(parse_pipe_metrics(hot))
    return data


def fmt_us(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "-"


def markdown_report(rows: list[dict[str, object]], top_lines: int) -> str:
    out = [
        "# TurboquantPackKvForCache4bit Branch Profile Summary",
        "",
        "| case | pack_mode | slot_pattern | app_avg_us | task_duration_us | cube0_aic_time | cube0_scalar | cube0_mte2 | vector0_aiv_time | vector0_scalar | vector0_mte2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            "| {case} | {pack_mode} | {slot_pattern} | {app_avg} | {task} | {cube_time} | {cube_scalar} | {cube_mte2} | {vec_time} | {vec_scalar} | {vec_mte2} |".format(
                case=row["case"],
                pack_mode=row["pack_mode"],
                slot_pattern=row["slot_pattern"],
                app_avg=fmt_us(row.get("app_avg_us")),
                task=fmt_us(row.get("task_duration_us") or row.get("duration_us")),
                cube_time=fmt_us(row.get("cube0_aic_time")),
                cube_scalar=fmt_us(row.get("cube0_aic_scalar_time")),
                cube_mte2=fmt_us(row.get("cube0_aic_mte2_time")),
                vec_time=fmt_us(row.get("vector0_aiv_time")),
                vec_scalar=fmt_us(row.get("vector0_aiv_scalar_time")),
                vec_mte2=fmt_us(row.get("vector0_aiv_mte2_time")),
            ))

    out.extend(["", f"## Top {top_lines} Repo Lines", ""])
    for row in rows:
        out.append(f"### {row['case']}")
        top_repo_lines = row["top_repo_lines"]
        if not top_repo_lines:
            out.append("- no repo line data")
            continue
        for count, location in top_repo_lines:
            out.append(f"- count={count}: {location}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile_dir",
        nargs="?",
        default="mytmp/pack4bit_branch_profiles",
        help="Directory produced by profile_turboquant4bit_pack_branches.sh",
    )
    parser.add_argument("--top-lines", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    case_dirs = [profile_dir / name for name in CASE_ORDER if (profile_dir / name).exists()]
    if not case_dirs:
        case_dirs = sorted(path for path in profile_dir.iterdir() if path.is_dir())
    rows = [summarize_case(case_dir, args.top_lines) for case_dir in case_dirs]
    report = markdown_report(rows, args.top_lines)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
