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

【日志结构】多级目录（日期/时间两层，时间后可拼 --ts-suffix 后缀，整次运行归到一个 run 目录）：
  <output-dir>/<日期>/<时间>[<后缀>]/               # 一次运行一个 run 目录
    <fia>/serve.log                             #   该 fia 的 serve 完整日志（每个 fia 重启 serve）
    <fia>/in<i>_out<o>_conc<c>/{bench.log,result.json}
    summary.log                                 #   跨所有组的总汇总表
  （日期=YYYYMMDD，时间=HHMMSS；后缀默认空串。例 --ts-suffix _eager → <日期>/HHMMSS_eager/）

【bench 输出】只回显关键字段（从 vllm 写出的 result JSON 提取，不依赖文本对齐）：
  Successful requests / Benchmark duration / Total input tokens /
  Total generated tokens / Total token throughput / Mean TTFT / Mean TPOT。
  全部组跑完后打印一张总汇总表，并额外写一份到 <output-dir>/<日期>/<时间>[<后缀>]/summary.log。

典型用法：
    # 默认单组：fia=off, io=200:200, conc=16
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5

    # 多组全排列：2 个 fia × 2 组 io × 2 个并发 = 8 组
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5 \\
        -f off,all -L 200:200,1024:200 -c 16,32

    # 保留 serve 不关（便于挂 profiler），仅单组时常用
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -k

【profiler】--profile 开启 torch profiler（NPU 上落盘 Ascend msprof trace）：
  serve 带 --profiler-config.*（profiler=torch + torch_profiler_dir + with_stack），
  bench 带 --profile（warmup 后 POST /start_profile，跑完 POST /stop_profile，
  故 trace 只含压测段、不含 warmup）。profiler 目录是 serve 级（启动时固定）配置，
  所以开 --profile 时每个 (fia,io,conc) combo 各重启一次 serve、各落独立 trace 目录
  （<output-dir>/<日期>/<时间>[<后缀>]/<fia>/in<i>_out<o>_conc<c>/profiler/，复用 --output-dir，
  无需额外参数；要落到别处磁盘就把 --output-dir 指过去）。不开 --profile 时仍走原
  per-fia 复用 serve 模式，默认行为零回归。可加 --profiler-ignore-frontend 只采 NPU
  worker、跳过前端 CPU trace 以降开销。

    # 单组 + profiler：trace 落 <output-dir>/<日期>/<时间>/off/in200_out200_conc16/profiler/
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5 \\
        --profile

    # 多组 + profiler：每个 combo 各重启 serve、各一份 trace
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5 \\
        --profile -f off,all -L 200:200,1024:200 -c 16,32

    # 落到大磁盘：把 --output-dir 指过去，profiler trace 作为其子目录跟过去
    python tests/e2e/singlecard/turboquant_k8v4_serve_bench.py -p 31720 -d 5 \\
        --profile -O /root/yyz/perfprof/runA
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---- 默认值（与参考命令对保持一致）----------------------------------------
DEFAULT_MODEL = "/root/yyz/models/Qwen3-0.6B"
DEFAULT_PORT = 9875
DEFAULT_KV_BITS = "8,4"  # -> additional_config={"turboquant_kv_bits":[8,4]}
DEFAULT_KV_CACHE_DTYPE = "turboquant"
DEFAULT_OUTPUT_ROOT = "/root/yyz/perflog/k8v4_serve_bench"

# bench 默认值（与参考 `vllm bench serve` 命令对齐）
DEFAULT_IO = "200:200"
DEFAULT_NUM_PROMPTS = 256
DEFAULT_CONCURRENCY = "16"
DEFAULT_FIA = "off"

# 就绪检测默认值
DEFAULT_READY_TIMEOUT_S = 120
DEFAULT_READY_INTERVAL_S = 2.0

# FIA 开关（见 vllm_ascend/envs.py）。bit_residual k8v4 路径在这些开关打开时，
# 把 prefill/decode 的 attention 走到 ``bit_residual_fia_paged_k8v4``；
# 关闭时回退到向量 paged attn 算子 ``bit_residual_attention_paged_k8v4``。
FIA_ENV_PREFILL = "VLLM_ASCEND_BIT_RESIDUAL_FIA"
FIA_ENV_DECODE = "VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA"
FIA_CHOICES = ("off", "prefill", "decode", "all")
# NPU 可见设备环境变量（Ascend 版的 CUDA_VISIBLE_DEVICES；见
# vllm_ascend/platform.py 的 device_control_env_var）。
NPU_VISIBLE_ENV = "ASCEND_RT_VISIBLE_DEVICES"
# 作用到 serve 子进程的默认环境变量，使 bit_residual k8v4 路径完整启用
# （与离线 smoke / 长查询脚本保持一致）。
SERVE_ENV_DEFAULTS = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
    "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
}

# tqdm 进度条会渲染成形如 "  38%|██▊  | 97/256 [00:46<01:04, 2.45it/s]" 的行。
_TQDM_RE = re.compile(r"^\s*\d+%\|")


# ---------------------------------------------------------------------------
# 多值解析（逗号分隔）
# ---------------------------------------------------------------------------
def _parse_io_pairs(s: str) -> list[tuple[int, int]]:
    """把 "200:200,1024:200" 解析成 [(200,200),(1024,200)]。"""
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
    """把 "16,32,64" 解析成 [16,32,64]。"""
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
    """把 "off,all,prefill" 解析成 ["off","all","prefill"]，并校验取值合法。"""
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
# 配置辅助
# ---------------------------------------------------------------------------
def _kv_bits_to_config(kv_bits: str) -> str:
    """把 "8,4" 转成 additional_config 的 JSON {"turboquant_kv_bits":[8,4]}。"""
    parts = [int(x.strip()) for x in kv_bits.split(",") if x.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"--kv-bits must be two comma-separated ints, got {kv_bits!r}")
    return json.dumps({"turboquant_kv_bits": parts})


def _fia_env(fia: str) -> dict[str, str]:
    """把单个 fia 取值映射到 VLLM_ASCEND_BIT_RESIDUAL_*FIA 环境变量。"""
    prefill = "1" if fia in ("prefill", "all") else "0"
    decode = "1" if fia in ("decode", "all") else "0"
    return {FIA_ENV_PREFILL: prefill, FIA_ENV_DECODE: decode}


def _build_serve_cmd(
    *, model: str, kv_cache_dtype: str, kv_bits: str, port: int, host: str,
    trust_remote_code: bool, serve_extra: str,
    profiler_dir: str | None = None, profiler_ignore_frontend: bool = False,
    profiler_active_iterations: int | None = None,
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
    # 给了 profiler_dir 时，通过嵌套的 --profiler-config.* CLI 让 serve 进程开启
    # torch profiler（vLLM 只有在配置了 profiler 时才会挂载 /start_profile、
    # /stop_profile 端点）。目录必须是绝对路径——ProfilerConfig 会拒绝相对路径；
    # NPU 上该目录落盘的是 Ascend（torch_npu.profiler）trace。
    if profiler_dir is not None:
        cmd += [
            "--profiler-config.profiler=torch",
            f"--profiler-config.torch_profiler_dir={profiler_dir}",
            "--profiler-config.torch_profiler_with_stack=true",
        ]
        if profiler_ignore_frontend:
            cmd.append("--profiler-config.ignore_frontend=true")
        # active_iterations 控制 schedule 里 ACTIVE 段长度；默认 vLLM=5，
        # 实测 TopK 往往只录到 ~4 次。需要更多采样（如 20）时显式拉高。
        if profiler_active_iterations is not None:
            cmd.append(
                f"--profiler-config.active_iterations={profiler_active_iterations}"
            )
    if serve_extra:
        cmd += shlex.split(serve_extra)
    return cmd


def _build_bench_cmd(
    *, model: str, port: int, host: str, input_len: int, output_len: int,
    num_prompts: int, concurrency: int, request_rate: float | None,
    bench_extra: str, result_dir: Path, profile: bool = False,
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
    # --profile 让 `vllm bench serve` 在 warmup 之后 POST /start_profile、主流程
    # 跑完 POST /stop_profile，因此 trace 精确只覆盖压测段（不含 warmup）。
    # 要求 serve 已经带 --profiler-config.* 启动（见 _build_serve_cmd）。
    if profile:
        cmd.append("--profile")
    if bench_extra:
        cmd += shlex.split(bench_extra)
    return cmd


# ---------------------------------------------------------------------------
# serve 日志流式读取（后台线程，不回显到终端）
# ---------------------------------------------------------------------------
class _ServeLogStreamer:
    """serve 进程 stdout/stderr 的后台线程读取器。

    常驻的 `vllm serve` 启动期间会产出大量输出。若没人消费管道，其内核缓冲区
    （通常 64 KiB）会被写满，serve 进程阻塞在 write 上——表现就和"服务一直
    起不来"一样。本读取器把管道内容排空写入 ``log_path``（静默——输出不会
    回显到终端；失败时由调用方通过 ``.tail()`` 打印尾部），同时主线程轮询
    /health。
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
# bench 输出流式处理（终端只回显 tqdm 进度条；关键字段从 JSON 取）
# ---------------------------------------------------------------------------
def _stream_bench(proc: subprocess.Popen, log_path: Path) -> int:
    """把 bench 输出完整写入 ``log_path``；终端只回显 tqdm 进度行。

    所有内容（INFO/namespace/警告/result 表）都进文件以便事后排查。终端只回显
    tqdm 进度条，它通过 ``\\r`` 原地刷新（只占一行，不会刷屏）。真正的结果
    指标由调用方从 result JSON 提取后单独打印——所以这里刻意不回显 result 表。
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
    """从 ``group_dir`` 里 vllm 写出的 result JSON 中提取关键字段。

    若没有 json 落盘（bench 在写文件前就失败了）返回 None。vllm 自己命名该文件
    （见 compute_result_filename），所以这里 glob *.json。
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
# serve 生命周期辅助
# ---------------------------------------------------------------------------
def _port_in_use(host: str, port: int) -> bool:
    """若已有进程在 (host, port) 上监听则返回 True。

    vllm serve 用 SO_REUSEADDR 绑定，所以两个 serve 能同时 LISTEN 同一端口而不
    报错——连接会被轮询分发到两者之间，bench 请求会静默打到错误（通常已过载）
    的 server。本守卫拒绝在已被占用的端口上再起第二个 serve。
    """
    # 连接用的 host 归一化：'' / '0.0.0.0' -> 127.0.0.1；vllm 默认 host 是
    # 127.0.0.1。探测的是 bench 实际会用的那个回环地址。
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", None) else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((probe_host, port)) == 0


def _wait_for_ready(
    base_url: str, timeout_s: float, interval_s: float, *, serve_proc: subprocess.Popen,
) -> None:
    """轮询 /health 直到 200 或超时。serve 提前退出则快速失败。"""
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
    """先 SIGTERM 进程，若 grace_s 内未退出再 SIGKILL。"""
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
# 单个 (fia, io, conc) 组合的执行器
# ---------------------------------------------------------------------------
def _run_one_combo(
    *, serve_proc: subprocess.Popen, serve_streamer: _ServeLogStreamer,
    bench_cmd: list[str], group_dir: Path, label: str,
) -> tuple[int, dict | None]:
    """把一个 bench 组合跑进 ``group_dir``。返回 (bench_rc, metrics)。"""
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
# 汇总表
# ---------------------------------------------------------------------------
def _print_summary(rows: list[dict], log_path: Path | None = None) -> None:
    """每组 combo 打印一行关键字段；同时写入 ``log_path``。

    同一张表写到 stdout（给现场终端看），并在给出 ``log_path`` 时追加到该文件，
    使汇总与各 combo 的 serve/bench 日志一同留在持久日志里。
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
        # profiler 目录是长路径；为保持对齐表的紧凑，把各组 profiler 目录单
        # 列在表下方的独立块里（仅当某组确有值时才出现——即用 --profile 时）。
        prof_rows = [r for r in rows if r.get("profiler_dir")]
        if prof_rows:
            lines.append("")
            lines.append("profiler dir per combo:")
            for r in prof_rows:
                lines.append(
                    f"  fia={r['fia']} in={r['in']} out={r['out']} conc={r['conc']} "
                    f"-> {r['profiler_dir']}"
                )
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
# 主入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive vllm serve + vllm bench serve for TurboQuant K8V4 "
                    "with multi-config permutations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- 共用 ----
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

    # ---- serve 参数组 ----
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

    # ---- bench 参数组（多值，逗号分隔）----
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

    # ---- 输出 ----
    out = parser.add_argument_group("output")
    out.add_argument(
        "-O", "--output-dir", default=DEFAULT_OUTPUT_ROOT,
        help="root dir for logs/json; run 目录按 日期/时间 两层下挂: "
             "<output-dir>/<日期>/<时间>[<后缀>]/<fia>/in<i>_out<o>_conc<c>/ 。"
             "(日期=YYYYMMDD，时间=HHMMSS，后缀由 --ts-suffix 给)。",
    )
    out.add_argument(
        "--ts-suffix", default="",
        help="时间目录的后缀，拼在时间之后(默认空串)。用于给同一次运行打标签，"
             "如 --ts-suffix _eager 把目录变成 <日期>/HHMMSS_eager/。注意需自带分隔符"
             "(_eager 而非 eager)。",
    )

    # ---- profiler 参数 ----
    prof = parser.add_argument_group("profiler")
    prof.add_argument(
        "--profile", action="store_true",
        help="enable torch profiler: inject --profiler-config.* into `vllm serve` "
             "and --profile into `vllm bench serve`. bench POSTs /start_profile after "
             "warmup and /stop_profile after the run, so the trace covers only the "
             "bench segment. On NPU the trace (msprof-style) lands under --output-dir "
             "(<output-dir>/<日期>/<时间>[<后缀>]/<fia>/in<i>_out<o>_conc<c>/profiler/). When set, serve "
             "is restarted per combo so each combo gets an isolated profiler dir "
             "(slower: one serve boot per combo).",
    )
    prof.add_argument(
        "--profiler-ignore-frontend", action="store_true",
        help="only profile the NPU worker (EngineCore subprocess), skip the AsyncLLM "
             "frontend CPU trace, to reduce overhead. Adds "
             "--profiler-config.ignore_frontend=true to `vllm serve`.",
    )
    prof.add_argument(
        "--profiler-active-iterations", type=int, default=None,
        help="set --profiler-config.active_iterations=N on `vllm serve` when "
             "--profile is on. Default leaves vLLM's own default (5). Use 20 to "
             "capture ~20 decode steps / TopK calls in the Ascend trace.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] model path not found: {args.model}", file=sys.stderr)
        return 2

    io_pairs = _parse_io_pairs(args.io)
    conc_list = _parse_int_list(args.concurrency, "--concurrency")
    fia_list = _parse_fia_list(args.fia)

    base_url = f"http://{args.host}:{args.port}"
    # run 目录拆成 日期/时间 两层，时间后可拼 --ts-suffix 后缀，便于按天归档与打标签：
    #   <output-dir>/<日期>/<时间>[<后缀>]/...
    now = datetime.datetime.now()
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S") + args.ts_suffix
    run_dir = Path(args.output_dir) / date_part / time_part
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

    # ---- serve 起停辅助（下方两种模式共用）----
    # _launch 返回 (proc, streamer, busy)：busy=True 表示端口已被占用
    # （调用方中止整次运行）；proc 为 None 表示 serve 没起来（调用方跳过
    # 这个 combo/fia）。借此让 per-fia 与 per-combo 两条分支保持精简。
    def _launch(fia: str, serve_log: Path, profiler_dir: str | None):
        serve_env = os.environ.copy()
        serve_env.update(SERVE_ENV_DEFAULTS)
        if args.device is not None:
            serve_env[NPU_VISIBLE_ENV] = str(args.device)
        serve_env.update(_fia_env(fia))

        serve_cmd = _build_serve_cmd(
            model=args.model, kv_cache_dtype=args.kv_cache_dtype, kv_bits=args.kv_bits,
            port=args.port, host=args.host, trust_remote_code=args.trust_remote_code,
            serve_extra=args.serve_extra,
            profiler_dir=profiler_dir,
            profiler_ignore_frontend=args.profiler_ignore_frontend,
            profiler_active_iterations=(
                args.profiler_active_iterations if profiler_dir is not None else None
            ),
        )
        print(f"  serve env : {NPU_VISIBLE_ENV}={serve_env.get(NPU_VISIBLE_ENV, '(inherited)')} "
              f"{FIA_ENV_PREFILL}={serve_env[FIA_ENV_PREFILL]} "
              f"{FIA_ENV_DECODE}={serve_env[FIA_ENV_DECODE]}")
        print(f"  serve cmd : {' '.join(shlex.quote(c) for c in serve_cmd)}")
        print(f"  serve log : {serve_log}")
        if profiler_dir:
            print(f"  profiler dir : {profiler_dir}")
            if args.profiler_active_iterations is not None:
                print(f"  profiler active_iterations : {args.profiler_active_iterations}")

        # 防止有旧 serve（或本脚本的另一个实例）已在端口上 LISTEN。vllm 用
        # SO_REUSEADDR 绑定，第二次 bind 也会“成功”并静默分流 bench 流量。
        if _port_in_use(args.host, args.port):
            print(
                f"  [!] 端口 {args.port} 已被占用（可能有一个旧的 vllm serve 没关，"
                f"或另一个本脚本实例在跑）。请换端口（-p）或先 kill 旧 serve 后重试。",
                file=sys.stderr,
            )
            return None, None, True  # 端口忙 -> 中止整次运行
        print(f"  [launching vllm serve ...]")
        proc = subprocess.Popen(
            serve_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=serve_env,
        )
        streamer = _ServeLogStreamer(proc, serve_log)
        print(f"  [waiting for {base_url}/health (timeout {args.serve_ready_timeout:.0f}s) ...]")
        try:
            _wait_for_ready(
                base_url, args.serve_ready_timeout, args.serve_ready_interval,
                serve_proc=proc,
            )
        except RuntimeError as e:
            print(f"  [!] {e}", file=sys.stderr)
            print(f"  [!] serve.log tail:\n{streamer.tail(40)}", file=sys.stderr)
            _terminate(proc)
            streamer.join(timeout=5.0)
            return None, None, False  # 端口不忙但 serve 没起来 -> 跳过本 combo
        print(f"  [server is ready.]")
        return proc, streamer, False

    def _bench_combo(fia: str, il: int, ol: int, c: int, proc, streamer,
                     group_dir: Path, profiler_dir: str | None) -> int:
        """把一个 (fia, io, conc) bench 跑进 group_dir。返回 bench 退出码。"""
        group_dir.mkdir(parents=True, exist_ok=True)
        bench_cmd = _build_bench_cmd(
            model=args.model, port=args.port, host=args.host,
            input_len=il, output_len=ol,
            num_prompts=args.num_prompts, concurrency=c,
            request_rate=args.request_rate, bench_extra=args.bench_extra,
            result_dir=group_dir, profile=args.profile,
        )
        label = f"fia={fia} in={il} out={ol} conc={c}"
        rc, metrics = _run_one_combo(
            serve_proc=proc, serve_streamer=streamer,
            bench_cmd=bench_cmd, group_dir=group_dir, label=label,
        )
        if metrics:
            summary_rows.append({
                "fia": fia, "in": il, "out": ol, "conc": c,
                "completed": metrics["completed"],
                "duration": metrics["duration"],
                "in_tok": metrics["total_input_tokens"],
                "out_tok": metrics["total_output_tokens"],
                "tok/s": metrics["total_token_throughput"],
                "ttft": metrics["mean_ttft_ms"],
                "tpot": metrics["mean_tpot_ms"],
                "profiler_dir": profiler_dir,
            })
        return rc

    def _stop(proc, streamer, keep: bool) -> None:
        if keep:
            print(f"  [--keep-server] leaving serve alive (pid {proc.pid})")
        else:
            print(f"  [stopping serve (pid {proc.pid}) ...]")
            _terminate(proc)
            streamer.join(timeout=5.0)
            print(f"  [serve stopped.]")

    # ---- 分支：per-combo 重启 serve（仅当 --profile）----
    # profiler_dir 是 server 级（serve 启动时固定）配置，因此必须每个 combo
    # 重启一次 serve，才能让每个 combo 各得一个隔离的 profiler 目录。
    if args.profile:
        print(f"  profiler  : enabled (per-combo serve restart; isolated trace dir per combo)")
        if args.profiler_ignore_frontend:
            print(f"             ignore_frontend=True (NPU worker only)")
        if args.profiler_active_iterations is not None:
            print(f"             active_iterations={args.profiler_active_iterations}")
        print("=" * 80)
        combos = [(fia, il, ol, c)
                  for fia in fia_list for (il, ol) in io_pairs for c in conc_list]
        for idx, (fia, il, ol, c) in enumerate(combos):
            fia_root = run_dir / fia
            fia_root.mkdir(parents=True, exist_ok=True)
            group_dir = fia_root / f"in{il}_out{ol}_conc{c}"
            group_dir.mkdir(parents=True, exist_ok=True)
            # 每 combo 隔离的 profiler 目录，与该组的日志放在一起。
            profiler_dir = str((group_dir / "profiler").resolve())
            (group_dir / "profiler").mkdir(parents=True, exist_ok=True)
            serve_log = group_dir / "serve.log"

            print(f"\n[combo {idx + 1}/{len(combos)}] "
                  f"fia={fia} in={il} out={ol} conc={c}  ->  {group_dir}")
            proc, streamer, busy = _launch(fia, serve_log, profiler_dir)
            if busy:
                return 4
            if proc is None:
                overall_rc = 3
                continue  # serve 没起来；尝试下一个 combo
            is_last = (idx == len(combos) - 1)
            keep = args.keep_server and is_last and len(combos) == 1
            try:
                rc = _bench_combo(fia, il, ol, c, proc, streamer, group_dir, profiler_dir)
                if rc != 0:
                    overall_rc = max(overall_rc, rc)
            finally:
                _stop(proc, streamer, keep)

    # ---- 分支：per-fia 复用 serve（默认；不开 profiler）----
    # 原始行为：每个 fia 起一次 serve，所有 io×conc 组合共用它。
    else:
        for fia_idx, fia in enumerate(fia_list):
            fia_root = run_dir / fia
            fia_root.mkdir(parents=True, exist_ok=True)
            serve_log = fia_root / "serve.log"

            print(f"\n[fia {fia_idx + 1}/{len(fia_list)}] {fia}  ->  {fia_root}")
            proc, streamer, busy = _launch(fia, serve_log, None)
            if busy:
                return 4
            if proc is None:
                overall_rc = 3
                continue  # 尝试下一个 fia
            is_last_fia = (fia_idx == len(fia_list) - 1)
            keep = args.keep_server and is_last_fia and len(fia_list) == 1
            try:
                for (il, ol), c in itertools.product(io_pairs, conc_list):
                    group_dir = fia_root / f"in{il}_out{ol}_conc{c}"
                    rc = _bench_combo(fia, il, ol, c, proc, streamer, group_dir, None)
                    if rc != 0:
                        overall_rc = max(overall_rc, rc)
            finally:
                _stop(proc, streamer, keep)

    # ---- 最终汇总（stdout + 持久 summary.log）----
    summary_log = run_dir / "summary.log"
    _print_summary(summary_rows, log_path=summary_log)
    print(f"\n[+] summary written to: {summary_log}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
