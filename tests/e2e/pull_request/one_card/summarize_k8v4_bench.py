#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""汇总统计 TurboQuant K8V4 serve+bench 多 run 结果（中文使用说明）。

本脚本输入一个或多个 run 目录（即 turboquant_k8v4_serve_bench.py 产出的
<output-dir>/<时间戳>/ 目录），递归遍历其下所有子目录里的
vllm-infqps-*.json 结果文件，从每个 JSON 提取关键字段，并从路径还原
fia / 输入输出长度 / 并发，拼成一张总表输出到终端并落盘 summary.log。

【目录层级】期望形如：
  <run 目录>/<fia>/in<i>_out<o>_conc<c>/vllm-infqps-*.json
  如 .../20260722-082749/off/in200_out200_conc16/vllm-infqps-...json

【关键字段】与 turboquant_k8v4_serve_bench.py 的 summary.log 一致：
  Successful requests / Benchmark duration / Total input tokens /
  Total generated tokens / Total token throughput / Mean TTFT / Mean TPOT

【典型用法】
  # 单个 run
  python tests/e2e/singlecard/summarize_k8v4_bench.py \\
      /root/yyz/perflog/k8v4_serve_bench/20260722-082749

  # 多个 run 拼成一张总表（多一列 run 区分）
  python tests/e2e/singlecard/summarize_k8v4_bench.py \\
      /root/yyz/perflog/k8v4_serve_bench/20260722-082749 \\
      /root/yyz/perflog/k8v4_serve_bench/20260722-091225

  # 指定落盘路径（默认写到第一个输入目录下 summary.log）
  python tests/e2e/singlecard/summarize_k8v4_bench.py <dir...> -O /tmp/sum.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 关键字段：JSON 键 -> 展示列名
FIELDS = [
    ("completed", "COMPLETED"),
    ("duration", "DURATION"),
    ("total_input_tokens", "IN_TOK"),
    ("total_output_tokens", "OUT_TOK"),
    ("total_token_throughput", "TOK/S"),
    ("mean_ttft_ms", "TTFT"),
    ("mean_tpot_ms", "TPOT"),
]

# 子目录名形如 in200_out200_conc16
_GROUP_RE = re.compile(r"^in(\d+)_out(\d+)_conc(\d+)$")
# 结果文件名形如 vllm-infqps-concurrency16-Qwen3-0.6B-<ts>.json
_RESULT_RE = re.compile(r"^vllm-infqps.*\.json$")


def _parse_group_dir(name: str) -> tuple[int, int, int] | None:
    """in200_out200_conc16 -> (200, 200, 16)."""
    m = _GROUP_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _load_row(json_path: Path, run: str) -> dict | None:
    """从结果 JSON 提取关键字段，并从路径还原 fia/in/out/conc/run."""
    # fia = 父目录的父目录名（<run>/<fia>/<group>/...json）
    group_dir = json_path.parent
    fia_dir = group_dir.parent
    parsed = _parse_group_dir(group_dir.name)
    if parsed is None:
        return None  # 不符合命名约定，跳过
    input_len, output_len, concurrency = parsed
    try:
        with open(json_path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] 读取失败 {json_path}: {e}", file=sys.stderr)
        return None
    row: dict = {"run": run, "fia": fia_dir.name, "in": input_len,
                 "out": output_len, "conc": concurrency}
    for key, _col in FIELDS:
        row[key] = d.get(key)
    return row


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return str(v)


def _print_table(rows: list[dict], *, out_file=None) -> None:
    """打印对齐表格到 stdout，同时写入 out_file（若给定）。"""
    cols = ["run", "fia", "in", "out", "conc"] + [k for k, _ in FIELDS]
    # 列展示名
    header_names = {"run": "RUN", "fia": "FIA", "in": "IN", "out": "OUT",
                    "conc": "CONC"}
    # 列宽
    widths = {}
    for c in cols:
        head = header_names.get(c, dict(FIELDS).get(c, c))
        widths[c] = max(len(head), max((len(_fmt(r.get(c))) for r in rows), default=0))

    def line(parts: list[str]) -> str:
        return "  ".join(parts)

    header_parts = []
    for c in cols:
        head = header_names.get(c, dict(FIELDS).get(c, c))
        header_parts.append(head.rjust(widths[c]))
    header = line(header_parts)
    bar = "=" * len(header)
    sep = "-" * len(header)

    output_lines = ["", bar, header, sep]
    # 排序：run, fia, in, out, conc，便于同 run 内聚、参数递增
    for r in sorted(rows, key=lambda x: (str(x.get("run")), str(x.get("fia")),
                                         x.get("in", 0), x.get("out", 0),
                                         x.get("conc", 0))):
        parts = [_fmt(r.get(c)).rjust(widths[c]) for c in cols]
        output_lines.append(line(parts))
    output_lines.append(bar)

    text = "\n".join(output_lines)
    print(text)
    if out_file is not None:
        out_file.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总统计 TurboQuant K8V4 serve+bench 多 run 结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dirs", nargs="+", metavar="DIR",
        help="一个或多个 run 目录（turboquant_k8v4_serve_bench 的 <时间戳> 输出目录），"
             "递归遍历其下 vllm-infqps-*.json。",
    )
    parser.add_argument(
        "-O", "--output", default=None,
        help="汇总表落盘路径；默认写到第一个输入目录下的 summary.log。",
    )
    args = parser.parse_args()

    # 校验输入目录
    run_dirs: list[Path] = []
    for d in args.dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[!] 不是目录: {d}", file=sys.stderr)
            return 2
        run_dirs.append(p)

    all_rows: list[dict] = []
    total_json = 0
    for rd in run_dirs:
        run_name = rd.name  # 如 20260722-082749
        # 递归找所有结果 json
        jsons = [p for p in rd.rglob("*.json") if _RESULT_RE.match(p.name)]
        if not jsons:
            print(f"[~] {rd} 下无 vllm-infqps-*.json", file=sys.stderr)
            continue
        for jp in sorted(jsons):
            row = _load_row(jp, run_name)
            if row is None:
                print(f"[~] 跳过不符合命名的: {jp}", file=sys.stderr)
                continue
            all_rows.append(row)
            total_json += 1

    if not all_rows:
        print("[!] 未找到任何可统计的结果 JSON", file=sys.stderr)
        return 1

    # 落盘
    out_path = Path(args.output) if args.output else run_dirs[0] / "summary.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n汇总: {total_json} 个结果 JSON，来自 {len(run_dirs)} 个 run 目录")
    print(f"run 目录: {[str(r) for r in run_dirs]}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 汇总: {total_json} 个 JSON, {len(run_dirs)} 个 run 目录\n")
        f.write(f"# run: {[str(r) for r in run_dirs]}\n")
        _print_table(all_rows, out_file=f)

    print(f"\n[+] 汇总已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
