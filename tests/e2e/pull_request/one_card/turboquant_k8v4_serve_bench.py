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

"""TurboQuant K8V4 serve + bench 多组全排列驱动脚本（中文使用说明）。

本脚本一条命令自动完成「起 vllm serve → 等就绪 → 跑 vllm bench serve → 关
serve」全流程，并支持对 FIA / 输入输出长度 / 并发数做多组全排列组合，便于
对比改算子前后的端到端 serving 吞吐与延迟。

【参数规则】多组参数均用逗号分隔：
  -L/--io        输入:输出 对，多组逗号分隔，如 200:200,1024:200,4096:512
  -c/--concurrency 并发数，多组逗号分隔，如 16,32,64
  -f/--fia       FIA 开关，取值 off/prefill/decode/all，多组逗号分隔，如 off,all
  全部组合做笛卡尔积：fia × io × concurrency，每组独立跑一次 bench。

【serve 处理】FIA 是 serve 侧（引擎进程）的环境变量，每个 -f 值需重启 serve：
  外层按 -f 循环，每个 -f 值起一次 serve；内层跑完所有 -L × -c 排列，serve 复用。

【日志结构】多级目录（时间戳只在最外一层，整次运行归到一个 run 目录）：
  <output-dir>/<时间戳>/                        # 一次运行一个 run 目录
    <fia>/serve.log                             #   该 fia 的 serve 完整日志（每个 fia 重启 serve）
    <fia>/in<i>_out<o>_conc<c>/{bench.log,result.json}
    summary.log                                 #   跨所有组的总汇总表

【bench 输出】只回显关键字段（从 vllm 写出的 result JSON 提取，不依赖文本对齐）：
  Successful requests / Benchmark duration / Total input tokens /
  Total generated tokens / Total token throughput / Mean TTFT / Mean TPOT。
  全部组跑完后打印一张总汇总表，并额外写一份到 <output-dir>/<时间戳>/summary.log。

典型用法：
    # 默认单组：fia=off, io=200:200, conc=16
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5

    # 多组全排列：2 个 fia × 2 组 io × 2 个并发 = 8 组
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5 \\
        -f off,all -L 200:200,1024:200 -c 16,32

    # 保留 serve 不关（便于挂 profiler），仅单组时常用
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -k
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---- Defaults mirroring the reference command pair --------------------------
DEFAULT_MODEL = "/root/yyz/models/Qwen3-0.6B"
DEFAULT_PORT = 9875
DEFAULT_KV_BITS = "8,4"  # -> additional_config={"turboquant_kv_bits":[8,4]}
DEFAULT_KV_CACHE_DTYPE = "turboquant"
DEFAULT_OUTPUT_ROOT = "/root/yyz/perflog/k8v4_serve_bench"

# Bench defaults (match the reference `vllm bench serve` command)
DEFAULT_IO = "200:200"
DEFAULT_NUM_PROMPTS = 256
DEFAULT_CONCURRENCY = "16"
DEFAULT_FIA = "off"

# Ready-check defaults
DEFAULT_READY_TIMEOUT_S = 120
DEFAULT_READY_INTERVAL_S = 2.0

# FIA toggles (see vllm_ascend/envs.py). The bit_residual k8v4 path routes
# prefill/decode attention to ``bit_residual_fia_paged_k8v4`` when these are ON;
# OFF falls back to the vector paged attn op ``bit_residual_attention_paged_k8v4``.
FIA_ENV_PREFILL = "VLLM_ASCEND_BIT_RESIDUAL_FIA"
FIA_ENV_DECODE = "VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA"
FIA_CHOICES = ("off", "prefill", "decode", "all")
# NPU visible-devices env (Ascend's CUDA_VISIBLE_DEVICES equivalent; see
# vllm_ascend/platform.py device_control_env_var).
NPU_VISIBLE_ENV = "ASCEND_RT_VISIBLE_DEVICES"
# Env defaults applied to the serve subprocess so the bit_residual k8v4 path is
# fully enabled (matches the offline smoke/long-query scripts).
SERVE_ENV_DEFAULTS = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
    "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
}

# tqdm progress bars render as lines like "  38%|██▊  | 97/256 [00:46<01:04, 2.45it/s]".
_TQDM_RE = re.compile(r"^\s*\d+%\|")


# ---------------------------------------------------------------------------
# Multi-value parsers (comma-separated)
# ---------------------------------------------------------------------------
def _parse_io_pairs(s: str) -> list[tuple[int, int]]:
    """Parse "200:200,1024:200" into [(200,200),(1024,200)]."""
    pairs: list[tuple[int, int]] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--io expects input:output pairs, got {chunk!r}"
            )
        i_str, o_str = chunk.split(":", 1)
        try:
            i_len, o_len = int(i_str), int(o_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--io pair must be ints, got {chunk!r}"
            )
        if i_len <= 0 or o_len <= 0:
            raise argparse.ArgumentTypeError(f"--io lengths must be >0, got {chunk!r}")
        pairs.append((i_len, o_len))
    if not pairs:
        raise argparse.ArgumentTypeError("--io needs at least one input:output pair")
    return pairs


def _parse_int_list(s: str, name: str) -> list[int]:
    """Parse "16,32,64" into [16,32,64]."""
    vals: list[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            v = int(chunk)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be ints, got {chunk!r}")
        if v <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be >0, got {chunk!r}")
        vals.append(v)
    if not vals:
        raise argparse.ArgumentTypeError(f"{name} needs at least one value")
    return vals


def _parse_fia_list(s: str) -> list[str]:
    """Parse "off,all,prefill" into ["off","all","prefill"] with validation."""
    vals: list[str] = []
    for chunk in s.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        if chunk not in FIA_CHOICES:
            raise argparse.ArgumentTypeError(
                f"--fia values must be one of {FIA_CHOICES}, got {chunk!r}"
            )
        vals.append(chunk)
    if not vals:
        raise argparse.ArgumentTypeError("--fia needs at least one value")
    return vals


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _kv_bits_to_config(kv_bits: str) -> str:
    """Turn "8,4" into the additional_config JSON {"turboquant_kv_bits":[8,4]}."""
    parts = [int(x.strip()) for x in kv_bits.split(",") if x.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"--kv-bits must be two comma-separated ints, got {kv_bits!r}")
    return json.dumps({"turboquant_kv_bits": parts})


def _fia_env(fia: str) -> dict[str, str]:
    """Map a single fia choice to the VLLM_ASCEND_BIT_RESIDUAL_*FIA env vars."""
    prefill = "1" if fia in ("prefill", "all") else "0"
    decode = "1" if fia in ("decode", "all") else "0"
    return {FIA_ENV_PREFILL: prefill, FIA_ENV_DECODE: decode}


def _build_serve_cmd(
    *, model: str, kv_cache_dtype: str, kv_bits: str, port: int, host: str,
    trust_remote_code: bool, serve_extra: str,
) -> list[str]:
    cmd: list[str] = [
        "vllm", "serve", model,
        f"--kv_cache_dtype={kv_cache_dtype}",
        f"--additional-config={_kv_bits_to_config(kv_bits)}",
        f"--port={port}",
        f"--host={host}",
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if serve_extra:
        cmd += shlex.split(serve_extra)
    return cmd


def _build_bench_cmd(
    *, model: str, port: int, host: str, input_len: int, output_len: int,
    num_prompts: int, concurrency: int, request_rate: float | None,
    bench_extra: str, result_dir: Path,
) -> list[str]:
    cmd: list[str] = [
        "vllm", "bench", "serve",
        "--backend=vllm",
        f"--port={port}",
        f"--host={host}",
        f"--model={model}",
        "--dataset-name=random",
        f"--input-len={input_len}",
        f"--output-len={output_len}",
        f"--num-prompts={num_prompts}",
        f"--max-concurrency={concurrency}",
        "--save-result",
        f"--result-dir={result_dir}",
    ]
    if request_rate is not None:
        cmd.append(f"--request-rate={request_rate}")
    if bench_extra:
        cmd += shlex.split(bench_extra)
    return cmd


# ---------------------------------------------------------------------------
# serve log streaming (background thread, no terminal echo)
# ---------------------------------------------------------------------------
class _ServeLogStreamer:
    """Background-thread reader for the serve process stdout/stderr.

    A long-lived `vllm serve` emits a lot of output during startup. If nobody
    drains the pipe, its kernel buffer (typically 64 KiB) fills and the serve
    process blocks on write — which then looks exactly like "server never
    becomes ready". This reader keeps the pipe drained into ``log_path``
    (silently — output is NOT echoed to the terminal; on failure the caller
    prints the tail via ``.tail()``), while the main thread polls /health.
    """

    def __init__(self, proc: subprocess.Popen, log_path: Path):
        self._proc = proc
        self._log_path = log_path
        self._lines: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w", buffering=1) as f:
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                self._lines.append(line)
                f.write(line)

    def tail(self, n: int = 40) -> str:
        return "".join(self._lines[-n:])

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# bench streaming (tqdm progress only on terminal; key fields from JSON)
# ---------------------------------------------------------------------------
def _stream_bench(proc: subprocess.Popen, log_path: Path) -> int:
    """Stream bench output to ``log_path`` fully; echo only the tqdm progress line.

    Everything (INFO/namespace/warnings/result table) goes to the file for
    later debugging. To the terminal we echo only the tqdm progress bar, which
    refreshes in place via ``\\r`` (one line, not a flood). The actual result
    metrics are extracted from the result JSON and printed separately by the
    caller — so we deliberately do NOT echo the result table here.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", buffering=1) as f:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
            f.write(line)
            stripped = line.strip()
            if _TQDM_RE.match(line):
                sys.stdout.write(f"\r{stripped}")
                sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return proc.wait()


def _extract_bench_metrics(group_dir: Path) -> dict | None:
    """Pull the key fields from the vllm-written result JSON in ``group_dir``.

    Returns None if no json landed (bench failed before writing). vllm names the
    file itself (see compute_result_filename), so we glob for *.json.
    """
    jsons = sorted(group_dir.glob("*.json"))
    if not jsons:
        return None
    try:
        with open(jsons[0]) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "json_path": str(jsons[0]),
        "completed": d.get("completed"),
        "duration": d.get("duration"),
        "total_input_tokens": d.get("total_input_tokens"),
        "total_output_tokens": d.get("total_output_tokens"),
        "total_token_throughput": d.get("total_token_throughput"),
        "mean_ttft_ms": d.get("mean_ttft_ms"),
        "mean_tpot_ms": d.get("mean_tpot_ms"),
    }


# ---------------------------------------------------------------------------
# serve lifecycle helpers
# ---------------------------------------------------------------------------
def _wait_for_ready(
    base_url: str, timeout_s: float, interval_s: float, *, serve_proc: subprocess.Popen,
) -> None:
    """Poll /health until 200 or timeout. Fail fast if serve exits early."""
    health_url = f"{base_url}/health"
    deadline = time.perf_counter() + timeout_s
    last_err: str | None = None
    while time.perf_counter() < deadline:
        if serve_proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve process exited early (code {serve_proc.returncode}) "
                f"before /health came up; see serve.log"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=interval_s) as resp:
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = repr(e)
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(interval_s)
    raise RuntimeError(
        f"server not ready at {health_url} within {timeout_s:.0f}s (last: {last_err})"
    )


def _terminate(proc: subprocess.Popen, grace_s: float = 10.0) -> int:
    """SIGTERM the process, then SIGKILL if it doesn't die within grace_s."""
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.terminate()
    except ProcessLookupError:
        return proc.returncode
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)
    return proc.returncode


# ---------------------------------------------------------------------------
# one (fia, io, conc) combination runner
# ---------------------------------------------------------------------------
def _run_one_combo(
    *, serve_proc: subprocess.Popen, serve_streamer: _ServeLogStreamer,
    bench_cmd: list[str], group_dir: Path, label: str,
) -> tuple[int, dict | None]:
    """Run one bench combination into ``group_dir``. Returns (bench_rc, metrics)."""
    print(f"\n--- {label} ---")
    bench_log = group_dir / "bench.log"
    print(f"    bench cmd: {' '.join(shlex.quote(c) for c in bench_cmd)}")
    bench_proc = subprocess.Popen(
        bench_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    bench_rc = _stream_bench(bench_proc, bench_log)
    if bench_rc != 0:
        print(f"[!] bench failed (exit {bench_rc}); see {bench_log}", file=sys.stderr)
    metrics = _extract_bench_metrics(group_dir)
    if metrics:
        print(
            f"    => Successful requests: {metrics['completed']} | "
            f"Duration: {metrics['duration']:.2f}s | "
            f"Input tokens: {metrics['total_input_tokens']} | "
            f"Output tokens: {metrics['total_output_tokens']} | "
            f"Total tok/s: {metrics['total_token_throughput']:.2f} | "
            f"Mean TTFT: {metrics['mean_ttft_ms']:.2f}ms | "
            f"Mean TPOT: {metrics['mean_tpot_ms']:.2f}ms"
        )
    else:
        print(f"[~] no result json in {group_dir}")
    return bench_rc, metrics


# ---------------------------------------------------------------------------
# summary table
# ---------------------------------------------------------------------------
def _print_summary(rows: list[dict], log_path: Path | None = None) -> None:
    """Print one row per combo with the key fields; also write to ``log_path``.

    The same table is written to stdout (for the live terminal) and, if
    ``log_path`` is given, appended to that file so the summary survives in a
    durable log alongside the per-combo serve/bench logs.
    """
    lines: list[str] = []
    if not rows:
        msg = "[no results to summarize]"
        print(f"\n{msg}")
        lines.append(msg)
    else:
        cols = ["fia", "in", "out", "conc", "completed", "duration",
                "in_tok", "out_tok", "tok/s", "ttft", "tpot"]
        widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in cols}
        header = "  ".join(c.upper().rjust(widths[c]) for c in cols)
        bar = "=" * len(header)
        lines.append("")
        lines.append(bar)
        lines.append(header)
        lines.append("-" * len(header))
        for r in rows:
            lines.append("  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in cols))
        lines.append(bar)
        print("\n".join(lines))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive vllm serve + vllm bench serve for TurboQuant K8V4 "
                    "with multi-config permutations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- shared ----
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="model path/name.")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="serve & bench port.")
    parser.add_argument("--host", default="127.0.0.1", help="serve & bench host.")
    parser.add_argument(
        "--kv-cache-dtype", default=DEFAULT_KV_CACHE_DTYPE,
        help="kv_cache_dtype passed to `vllm serve`.",
    )
    parser.add_argument(
        "--kv-bits", default=DEFAULT_KV_BITS,
        help='comma-separated pair for additional_config turboquant_kv_bits, e.g. "8,4".',
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=True,
        help="pass --trust-remote-code to vllm serve (on by default for local HF paths).",
    )

    # ---- serve group ----
    serve = parser.add_argument_group("serve")
    serve.add_argument(
        "-d", "--device", default=None,
        help="NPU card(s) for the serve process via "
             f"{NPU_VISIBLE_ENV}; e.g. '0' or '0,1'. "
             "Default (None) leaves the inherited env untouched.",
    )
    serve.add_argument(
        "--serve-extra", default="",
        help="extra args appended verbatim to `vllm serve` (shell-split), e.g. "
             "'--gpu-memory-utilization 0.1'.",
    )
    serve.add_argument(
        "--serve-ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S,
        help="seconds to wait for /health before giving up.",
    )
    serve.add_argument(
        "--serve-ready-interval", type=float, default=DEFAULT_READY_INTERVAL_S,
        help="seconds between /health polls.",
    )
    serve.add_argument(
        "-k", "--keep-server", action="store_true",
        help="leave the serve process alive after bench (single-fia mode only; "
             "ignored when -f has multiple values).",
    )

    # ---- bench group (multi-value, comma-separated) ----
    bench = parser.add_argument_group("bench")
    bench.add_argument(
        "-L", "--io", default=DEFAULT_IO,
        help="input:output pairs, comma-separated, e.g. '200:200,1024:200'.",
    )
    bench.add_argument("-n", "--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    bench.add_argument(
        "-c", "--concurrency", default=DEFAULT_CONCURRENCY,
        help="max-concurrency values, comma-separated, e.g. '16,32,64'.",
    )
    bench.add_argument(
        "-f", "--fia", default=DEFAULT_FIA,
        help=f"FIA setting(s), comma-separated; each one of {FIA_CHOICES}, e.g. 'off,all'.",
    )
    bench.add_argument(
        "--request-rate", type=float, default=None,
        help="request rate (RPS); default omits the flag (open-loop, inf).",
    )
    bench.add_argument(
        "--bench-extra", default="",
        help="extra args appended verbatim to `vllm bench serve` (shell-split).",
    )

    # ---- output ----
    out = parser.add_argument_group("output")
    out.add_argument(
        "-O", "--output-dir", default=DEFAULT_OUTPUT_ROOT,
        help="root dir for logs/json; per-fia subdir <fia>_<ts>/ holds per-combo "
             "subdirs in<i>_out<o>_conc<c>/.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] model path not found: {args.model}", file=sys.stderr)
        return 2

    io_pairs = _parse_io_pairs(args.io)
    conc_list = _parse_int_list(args.concurrency, "--concurrency")
    fia_list = _parse_fia_list(args.fia)
    # num_combos = len(fia_list) * len(io_pairs) * len(conc_list)

    base_url = f"http://{args.host}:{args.port}"
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    overall_rc = 0
    summary_rows: list[dict] = []

    print("=" * 80)
    print("TurboQuant K8V4 serve+bench driver (multi-config permutations)")
    print("=" * 80)
    print(f"  fia   : {fia_list}")
    print(f"  io    : {io_pairs}")
    print(f"  conc  : {conc_list}")
    print(f"  total combos: {len(fia_list) * len(io_pairs) * len(conc_list)} "
          f"({len(fia_list)} fia x {len(io_pairs)} io x {len(conc_list)} conc)")
    print(f"  output root : {run_dir}")
    print("=" * 80)

    # ---- outer loop: per fia (each fia restarts serve) ----
    for fia_idx, fia in enumerate(fia_list):
        fia_root = run_dir / fia
        fia_root.mkdir(parents=True, exist_ok=True)
        serve_log = fia_root / "serve.log"

        serve_env = os.environ.copy()
        serve_env.update(SERVE_ENV_DEFAULTS)
        if args.device is not None:
            serve_env[NPU_VISIBLE_ENV] = str(args.device)
        serve_env.update(_fia_env(fia))

        serve_cmd = _build_serve_cmd(
            model=args.model, kv_cache_dtype=args.kv_cache_dtype, kv_bits=args.kv_bits,
            port=args.port, host=args.host, trust_remote_code=args.trust_remote_code,
            serve_extra=args.serve_extra,
        )

        print(f"\n[fia {fia_idx + 1}/{len(fia_list)}] {fia}  ->  {fia_root}")
        print(f"  serve env : {NPU_VISIBLE_ENV}={serve_env.get(NPU_VISIBLE_ENV, '(inherited)')} "
              f"{FIA_ENV_PREFILL}={serve_env[FIA_ENV_PREFILL]} "
              f"{FIA_ENV_DECODE}={serve_env[FIA_ENV_DECODE]}")
        print(f"  serve cmd : {' '.join(shlex.quote(c) for c in serve_cmd)}")
        print(f"  serve log : {serve_log}")

        # ---- launch serve ----
        print(f"  [launching vllm serve ...]")
        serve_proc = subprocess.Popen(
            serve_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=serve_env,
        )
        serve_streamer = _ServeLogStreamer(serve_proc, serve_log)

        is_last_fia = (fia_idx == len(fia_list) - 1)
        keep = args.keep_server and is_last_fia and len(fia_list) == 1

        try:
            print(f"  [waiting for {base_url}/health (timeout {args.serve_ready_timeout:.0f}s) ...]")
            try:
                _wait_for_ready(
                    base_url, args.serve_ready_timeout, args.serve_ready_interval,
                    serve_proc=serve_proc,
                )
            except RuntimeError as e:
                print(f"  [!] {e}", file=sys.stderr)
                print(f"  [!] serve.log tail:\n{serve_streamer.tail(40)}", file=sys.stderr)
                overall_rc = 3
                continue  # try next fia
            print(f"  [server is ready.]")

            # ---- inner loop: all io x conc combos on this serve ----
            for (input_len, output_len), concurrency in itertools.product(
                io_pairs, conc_list
            ):
                group_dir = fia_root / f"in{input_len}_out{output_len}_conc{concurrency}"
                group_dir.mkdir(parents=True, exist_ok=True)
                bench_cmd = _build_bench_cmd(
                    model=args.model, port=args.port, host=args.host,
                    input_len=input_len, output_len=output_len,
                    num_prompts=args.num_prompts, concurrency=concurrency,
                    request_rate=args.request_rate, bench_extra=args.bench_extra,
                    result_dir=group_dir,
                )
                label = f"fia={fia} in={input_len} out={output_len} conc={concurrency}"
                rc, metrics = _run_one_combo(
                    serve_proc=serve_proc, serve_streamer=serve_streamer,
                    bench_cmd=bench_cmd, group_dir=group_dir, label=label,
                )
                if rc != 0:
                    overall_rc = max(overall_rc, rc)
                if metrics:
                    summary_rows.append({
                        "fia": fia, "in": input_len, "out": output_len,
                        "conc": concurrency,
                        "completed": metrics["completed"],
                        "duration": metrics["duration"],
                        "in_tok": metrics["total_input_tokens"],
                        "out_tok": metrics["total_output_tokens"],
                        "tok/s": metrics["total_token_throughput"],
                        "ttft": metrics["mean_ttft_ms"],
                        "tpot": metrics["mean_tpot_ms"],
                    })
        finally:
            if keep:
                print(f"  [--keep-server] leaving serve alive (pid {serve_proc.pid})")
            else:
                print(f"  [stopping serve (pid {serve_proc.pid}) ...]")
                _terminate(serve_proc)
                serve_streamer.join(timeout=5.0)
                print(f"  [serve stopped.]")

    # ---- final summary (stdout + a durable summary.log) ----
    summary_log = run_dir / "summary.log"
    _print_summary(summary_rows, log_path=summary_log)
    print(f"\n[+] summary written to: {summary_log}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
