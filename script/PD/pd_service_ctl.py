#!/usr/bin/env python3
"""
PD 分离：Prefill / Decode / 负载均衡代理 的启动与停止。

两种用法：
  1) 命令行：``python pd_service_ctl.py start ...`` 一键起 prefill/decode/代理；``python pd_service_ctl.py stop`` 停止。
     具体参数见 ``--help``。
  2) 代码：构造 ``PdServiceCtl`` / ``PdRuntimeConfig`` 调用 ``start_*``、``stop``（如 ``run_sharegpt_sweep.py``）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

PKG_DIR = Path(__file__).resolve().parent

PID_PREFILL = Path("/tmp/vllm_prefill.pid")
PID_DECODE = Path("/tmp/vllm_decode.pid")
PID_PROXY = Path("/tmp/vllm_proxy.pid")

VLLM_VENV = Path("/root/autodl-tmp/py_venv/vllm")

_FALLBACK_NIC_NAME = "eth0"
_FALLBACK_LOCAL_IP = "172.17.0.4"
PD_MODE = "1P1_1D1"


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ipv4_on_dev(dev: str) -> Optional[str]:
    """用 ``ip -json addr`` 取某网卡首个全局 IPv4。"""
    try:
        p = subprocess.run(
            ["ip", "-json", "addr", "show", "dev", dev],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        data = json.loads(p.stdout)
        if not data:
            return None
        for info in data[0].get("addr_info", []):
            if info.get("family") != "inet":
                continue
            loc = info.get("local")
            if loc and not loc.startswith("127."):
                return loc
    except (json.JSONDecodeError, OSError, KeyError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ip_json() -> Optional[Tuple[str, str]]:
    try:
        p = subprocess.run(
            ["ip", "-json", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        routes = json.loads(p.stdout)
        if not routes:
            return None
        r0 = routes[0]
        dev = r0.get("dev")
        if not dev:
            return None
        prefsrc = r0.get("prefsrc")
        if prefsrc:
            return dev, prefsrc
        ip = _ipv4_on_dev(dev)
        if ip:
            return dev, ip
    except (json.JSONDecodeError, OSError, KeyError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ip_text() -> Optional[Tuple[str, str]]:
    try:
        p = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        line = (p.stdout or "").strip().splitlines()[0]
        m = re.search(r"\bdefault\s+(?:via\s+\S+\s+)?dev\s+(\S+)", line)
        if not m:
            return None
        dev = m.group(1)
        m2 = re.search(r"\bsrc\s+(\d+\.\d+\.\d+\.\d+)", line)
        if m2:
            return dev, m2.group(1)
        ip = _ipv4_on_dev(dev)
        if ip:
            return dev, ip
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ifconfig() -> Optional[Tuple[str, str]]:
    """解析 ``ifconfig`` / ``ifconfig -a``（无 ``ip`` 的旧环境兜底）。"""
    for cmd in (["ifconfig"], ["ifconfig", "-a"]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
            if p.returncode != 0:
                continue
            text = p.stdout or ""
        except OSError:
            continue
        for block in text.split("\n\n"):
            lines = [ln for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            m0 = re.match(r"^(\S+):", lines[0].strip())
            if not m0:
                continue
            iface = m0.group(1)
            if iface == "lo":
                continue
            rest = "\n".join(lines[1:])
            m = re.search(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", rest)
            if not m:
                continue
            ip = m.group(1)
            if ip.startswith("127."):
                continue
            return iface, ip
    return None


def _detect_default_nic_ip() -> Tuple[str, str]:
    """
    默认网卡与 IPv4：优先 ``ip`` JSON/文本，其次 ``ifconfig``；均失败则用模块内常量。
    """
    for fn in (_detect_nic_ip_via_ip_json, _detect_nic_ip_via_ip_text, _detect_nic_ip_via_ifconfig):
        t = fn()
        if t:
            return t
    print(
        f"{ts()} [pd] NIC_NAME/LOCAL_IP 探测失败（ip / ifconfig 均未能解析），"
        f"使用回退值: NIC_NAME={_FALLBACK_NIC_NAME!r} LOCAL_IP={_FALLBACK_LOCAL_IP!r}",
        file=sys.stderr,
    )
    return _FALLBACK_NIC_NAME, _FALLBACK_LOCAL_IP


NIC_NAME, LOCAL_IP = _detect_default_nic_ip()

LogFn = Callable[[str], None]


@dataclass
class PdRuntimeConfig:
    """启动 PD 服务所需的最小运行时配置（与压测 batch 无关）。"""

    topo_dir: Path
    vllm_venv: Path
    pd_mode: str = ""
    prefill_port: int = 9000
    vllm_port: int = 9010
    proxy_sleep_s: int = 10
    ready_timeout_s: int = 150
    nic_name: str = NIC_NAME
    local_ip: str = LOCAL_IP


def log_default(msg: str) -> None:
    print(f"{ts()} [pd] {msg}")


def pd_proxy_path(topo_dir: Path) -> Optional[Path]:
    p = topo_dir / "pd_proxy.py"
    return p if p.is_file() else None


def _bash_start_background(
    *,
    run_script: Path,
    log_file: Optional[Path],
    pid_file: Path,
    venv_activate: Path,
    cwd: Path,
    extra_env: Optional[dict[str, str]] = None,
) -> None:
    if not venv_activate.is_file():
        raise FileNotFoundError(f"venv activate 不存在: {venv_activate}")
    if not run_script.is_file():
        raise FileNotFoundError(f"入口不存在: {run_script}")
    if run_script.suffix == ".py":
        launcher = f'python3 "{run_script}"'
    elif run_script.suffix == ".sh":
        launcher = f'bash "{run_script}"'
    else:
        raise ValueError(f"不支持的入口后缀: {run_script}")
    redir = f'>> "{log_file}" 2>&1' if log_file is not None else ""
    inner = f"""
set -euo pipefail
source "{venv_activate}"
# Some activate scripts (or cluster env) export invalid thread vars like "false",
# which makes torch/torch_npu abort at import time. Force safe integers after
# activating the venv.
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
for k in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS; do
  v="${{!k}}"
  if [[ ! "$v" =~ ^[0-9]+$ ]]; then export "$k"=1; fi
done
set -m
nohup {launcher} {redir} &
echo $! > "{pid_file}"
"""
    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    subprocess.run(["/bin/bash", "-c", inner], check=True, cwd=str(cwd), env=run_env)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _read_pid_file(pid_file: Path) -> Optional[int]:
    if not pid_file.is_file():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def _direct_child_pids(pid: int) -> list[int]:
    """列出 ``pid`` 的直接子进程（Linux ``pgrep -P``）。"""
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return []
    kids: list[int] = []
    for line in (out.stdout or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            kids.append(int(line))
        except ValueError:
            continue
    return kids


def _collect_pid_tree(root: int) -> set[int]:
    """自 ``root`` 起 BFS 收集整棵子进程树（含 ``root``）。"""
    seen: set[int] = set()
    stack = [root]
    while stack:
        p = stack.pop()
        if p <= 0 or p in seen:
            continue
        seen.add(p)
        for c in _direct_child_pids(p):
            if c not in seen:
                stack.append(c)
    return seen


def _kill_pid_tree(pid: int, log: LogFn, label: str, wait_cap: int = 30) -> None:
    """
    结束以 ``pid`` 为根的进程树。

    ``run_decode.sh`` 无参时 pid 文件指向外层 bash，真实 vllm 在子 shell/子进程组里，
    仅用 ``killpg`` 杀不到；这里按父子关系整树 ``SIGKILL``。代理、prefill 同理兜底。
    """
    if not _pid_alive(pid):
        return
    tree = _collect_pid_tree(pid)
    others = sorted((tree - {pid}), reverse=True)
    for p in others:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    t0 = time.time()
    while time.time() - t0 < wait_cap:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)

    if _pid_alive(pid):
        log(f"[stop {label}] pid {pid} 在 {wait_cap}s 内仍存在（可能已僵尸），放弃。")


class PdServiceCtl:
    """
    PD 服务编排：prefill / decode / 代理 的启动与 ``stop`` 回收。

    ``log`` 为行级回调；``run_sharegpt_sweep`` 可传入写入 sweep.log 的函数。
    """

    def __init__(self, rt: PdRuntimeConfig, log: LogFn = log_default) -> None:
        self._rt = rt
        self._log = log

    @property
    def rt(self) -> PdRuntimeConfig:
        return self._rt

    @property
    def has_proxy_script(self) -> bool:
        return pd_proxy_path(self._rt.topo_dir) is not None

    def _runner_path(self, stem: str) -> Path:
        p = self._rt.topo_dir / f"{stem}.sh"
        if not p.is_file():
            mode = self._rt.pd_mode or str(self._rt.topo_dir)
            raise FileNotFoundError(f"缺少拓扑入口 {p}（pd_mode={mode!r}）")
        return p

    def _npu_iface_env(self) -> dict[str, str]:
        return {
            "NIC_NAME": self._rt.nic_name,
            "LOCAL_IP": self._rt.local_ip,
        }

    def wait_for_port(
        self,
        port: int,
        name: str,
        timeout_s: Optional[int] = None,
    ) -> bool:
        t = self._rt.ready_timeout_s if timeout_s is None else timeout_s
        self._log(f"等待 {name} 端口 {port} 就绪（超时 {t}s）...")
        elapsed = 0
        url = f"http://localhost:{port}/health"
        step = 5
        while elapsed < t:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        self._log(f"{name} 已就绪（已等待 {elapsed}s）。")
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(step)
            elapsed += step
        self._log(f"ERROR: {name} 在 {t}s 内未就绪。")
        return False

    def _is_port_ready(self, port: int) -> bool:
        url = f"http://localhost:{port}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def wait_for_ports(self, ports: list[tuple[int, str]], *, timeout_s: Optional[int] = None) -> bool:
        """
        等待多个端口就绪（共享超时窗口）。

        用于并行拉起场景：prefill/decode 都已启动，但 readiness 需要统一等待。
        """
        t = self._rt.ready_timeout_s if timeout_s is None else timeout_s
        names = ", ".join([f"{n}:{p}" for p, n in ports])
        self._log(f"等待端口就绪（{names}，共享超时 {t}s）...")
        deadline = time.time() + t
        step = 2.0
        ready: dict[int, bool] = {p: False for p, _ in ports}
        while time.time() < deadline:
            changed = False
            for p, _n in ports:
                if ready[p]:
                    continue
                if self._is_port_ready(p):
                    ready[p] = True
                    changed = True
            if all(ready.values()):
                self._log("端口均已就绪。")
                return True
            if changed:
                not_ready = ", ".join([f"{n}:{p}" for p, n in ports if not ready[p]])
                self._log(f"仍未就绪: {not_ready}")
            time.sleep(step)
        not_ready = ", ".join([f"{n}:{p}" for p, n in ports if not ready[p]])
        self._log(f"ERROR: 在 {t}s 内未全部就绪，仍未就绪: {not_ready}")
        return False

    def _ensure_pid_alive(self, pid_file: Path, name: str, *, grace_s: float = 2.0) -> bool:
        """
        启动后快速校验：pid 文件存在且进程未立刻退出。

        端口 readiness（/health）由 ``wait_for_port`` 负责。
        """
        pid = _read_pid_file(pid_file)
        if pid is None:
            self._log(f"ERROR: {name} 启动后未生成 pid 文件: {pid_file}")
            return False
        t0 = time.time()
        while time.time() - t0 < grace_s:
            if _pid_alive(pid):
                return True
            time.sleep(0.1)
        self._log(f"ERROR: {name} 进程疑似已退出（pid={pid}）")
        return False

    def _cleanup_stale_pid_files(self) -> None:
        """清理 pid 文件存在但进程已退出的残留，避免误判运行中。"""
        for pid_file, label in ((PID_PROXY, "proxy"), (PID_DECODE, "decode"), (PID_PREFILL, "prefill")):
            pid = _read_pid_file(pid_file)
            if pid is None:
                continue
            if not _pid_alive(pid):
                pid_file.unlink(missing_ok=True)
                self._log(f"[start] 清理残留 pid 文件: {pid_file}（{label} pid={pid} 已不存在）")

    def _running_components(self) -> list[tuple[str, int, Path]]:
        """返回仍在运行的组件列表：[(label, pid, pid_file), ...]"""
        running: list[tuple[str, int, Path]] = []
        for pid_file, label in ((PID_PROXY, "proxy"), (PID_DECODE, "decode"), (PID_PREFILL, "prefill")):
            pid = _read_pid_file(pid_file)
            if pid is None:
                continue
            if _pid_alive(pid):
                running.append((label, pid, pid_file))
        return running

    def start_prefill(self, log_dir: Path) -> None:
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        run_script = self._runner_path("run_prefill")
        _bash_start_background(
            run_script=run_script,
            log_file=None,
            pid_file=PID_PREFILL,
            venv_activate=self._rt.vllm_venv / "bin" / "activate",
            cwd=self._rt.topo_dir,
            extra_env={
                "LOG_DIR": str(log_dir),
                **self._npu_iface_env(),
            },
        )

    def start_decode(self, log_dir: Path) -> None:
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        run_script = self._runner_path("run_decode")
        env = os.environ.copy()
        env["LOG_DIR"] = str(log_dir)
        env.update(self._npu_iface_env())
        activate = self._rt.vllm_venv / "bin" / "activate"
        if not activate.is_file():
            raise FileNotFoundError(f"venv activate 不存在: {self._rt.vllm_venv}")
        if run_script.suffix == ".py":
            launcher = f'python3 "{run_script}"'
        elif run_script.suffix == ".sh":
            launcher = f'bash "{run_script}"'
        else:
            raise ValueError(f"不支持的入口后缀: {run_script}")
        inner = f"""
set -euo pipefail
source "{activate}"
set -m
nohup {launcher} &
echo $! > "{PID_DECODE}"
"""
        subprocess.run(
            ["/bin/bash", "-c", inner],
            check=True,
            cwd=str(self._rt.topo_dir),
            env=env,
        )

    def start_proxy(self, log_dir: Path) -> None:
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        proxy = pd_proxy_path(self._rt.topo_dir)
        if proxy is None:
            mode = self._rt.pd_mode or str(self._rt.topo_dir)
            raise FileNotFoundError(f"未找到 pd_proxy.py（{mode}）")
        _bash_start_background(
            run_script=proxy,
            log_file=log_dir / "proxy.log",
            pid_file=PID_PROXY,
            venv_activate=self._rt.vllm_venv / "bin" / "activate",
            cwd=self._rt.topo_dir,
            extra_env={"LOCAL_IP": self._rt.local_ip},
        )

    def stop_proxy_if_running(self, apply: bool) -> None:
        if not apply:
            return
        if not PID_PROXY.is_file():
            return
        try:
            pid = int(PID_PROXY.read_text().strip())
        except ValueError:
            PID_PROXY.unlink(missing_ok=True)
            return
        if not _pid_alive(pid):
            PID_PROXY.unlink(missing_ok=True)
            return
        self._log(f"[stop proxy] 结束代理 PID {pid}（含子进程）...")
        _kill_pid_tree(pid, self._log, "proxy")
        PID_PROXY.unlink(missing_ok=True)
        self._log("[stop proxy] 代理已停止。")

    def stop_vllm(self) -> None:
        self._log("停止 vLLM（先 decode 再 prefill）...")
        for pid_file, label in ((PID_DECODE, "decode"), (PID_PREFILL, "prefill")):
            if not pid_file.is_file():
                continue
            try:
                pid = int(pid_file.read_text().strip())
            except ValueError:
                pid_file.unlink(missing_ok=True)
                continue
            if not _pid_alive(pid):
                pid_file.unlink(missing_ok=True)
                continue
            self._log(f"[stop {label}] 结束 {label} PID {pid}（含子进程）...")
            _kill_pid_tree(pid, self._log, label)
            pid_file.unlink(missing_ok=True)
            self._log(f"[stop {label}] 已停止。")

    def stop(self, *, use_proxy: bool = True) -> None:
        """先停代理（若 ``use_proxy``），再停 decode、prefill。"""
        self.stop_proxy_if_running(use_proxy)
        self.stop_vllm()

    def start_stack(
        self,
        log_dir: Path,
        *,
        with_proxy: Optional[bool] = None,
        ready_timeout_s: Optional[int] = None,
        start_mode: str = "parallel",
    ) -> int:
        """
        顺序启动 prefill → decode →（可选）代理。
        ``with_proxy`` 为 ``None`` 时：存在 ``pd_proxy.py`` 则起代理。
        返回 0 成功；失败时会尽量回收已起进程。
        """
        # 防重入：未 stop 前再次 start 直接报错
        self._cleanup_stale_pid_files()
        running = self._running_components()
        if running:
            detail = ", ".join([f"{lbl}(pid={pid}, pid_file={pf})" for lbl, pid, pf in running])
            self._log(f"ERROR: PD 服务已在运行，禁止重复 start；请先 stop。运行中: {detail}")
            return 1

        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        has_proxy = self.has_proxy_script
        if with_proxy is None:
            wp = has_proxy
        else:
            wp = with_proxy
        if wp and not has_proxy:
            self._log("ERROR: 指定了代理但当前拓扑无 pd_proxy.py")
            return 1

        if start_mode not in ("serial", "parallel"):
            self._log(f"ERROR: 不支持的 start_mode={start_mode!r}（仅支持 serial/parallel）")
            return 1

        t = self._rt.ready_timeout_s if ready_timeout_s is None else ready_timeout_s

        if start_mode == "serial":
            try:
                self._log("启动 prefill...")
                self.start_prefill(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: prefill 启动失败: {e}")
                self.stop_vllm()
                return 1

            if not self._ensure_pid_alive(PID_PREFILL, "prefill"):
                self.stop_vllm()
                return 1

            if ready_timeout_s != 0:
                if not self.wait_for_port(self._rt.prefill_port, "prefill", timeout_s=t):
                    self.stop_vllm()
                    return 1

            try:
                self._log("启动 decode...")
                self.start_decode(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: decode 启动失败: {e}")
                self.stop_vllm()
                return 1

            if not self._ensure_pid_alive(PID_DECODE, "decode"):
                self.stop_vllm()
                return 1

            if ready_timeout_s != 0:
                if not self.wait_for_port(self._rt.vllm_port, "decode", timeout_s=t):
                    self.stop_vllm()
                    return 1
        else:
            # parallel: 不等待 prefill readiness，就直接拉起 decode；然后统一等待两个端口就绪
            try:
                self._log("并行拉起：启动 prefill...")
                self.start_prefill(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: prefill 启动失败: {e}")
                self.stop_vllm()
                return 1

            try:
                self._log("并行拉起：启动 decode...")
                self.start_decode(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: decode 启动失败: {e}")
                self.stop_vllm()
                return 1

            if not self._ensure_pid_alive(PID_PREFILL, "prefill"):
                self.stop_vllm()
                return 1
            if not self._ensure_pid_alive(PID_DECODE, "decode"):
                self.stop_vllm()
                return 1

            if ready_timeout_s != 0:
                if not self.wait_for_ports(
                    [(self._rt.prefill_port, "prefill"), (self._rt.vllm_port, "decode")],
                    timeout_s=t,
                ):
                    self.stop_vllm()
                    return 1

        if wp:
            try:
                self._log("启动代理...")
                self.start_proxy(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: 代理启动失败: {e}")
                self.stop(use_proxy=True)
                return 1
            time.sleep(self._rt.proxy_sleep_s)
            if not self._ensure_pid_alive(PID_PROXY, "proxy", grace_s=0.1):
                self.stop(use_proxy=True)
                return 1

        self._log(f"PD 服务已拉起，日志目录: {log_dir}")
        return 0


# ---------------------------------------------------------------------------
# 命令行：解析与 Namespace → PdRuntimeConfig（仅 start 子命令会用到后者）
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PD Prefill / Decode / 代理 启停")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="启动 prefill → decode →（可选）代理")
    p_start.add_argument(
        "--pd_mode",
        default=PD_MODE,
        help="相对当前脚本目录的拓扑子路径（默认 PD_MODE 常量）",
    )
    p_start.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs"),
        help="prefill/decode/proxy 日志目录（默认当前目录下 logs/）",
    )
    p_start.add_argument(
        "--vllm_venv",
        type=Path,
        default=VLLM_VENV,
        help="vLLM 所在虚拟环境目录（默认 /root/autodl-tmp/py_venv/vllm）",
    )
    p_start.add_argument(
        "--nic_name",
        default=NIC_NAME,
        help="注入脚本 HCCL/GLOO 等所用网卡名（默认与模块常量 NIC_NAME 一致）",
    )
    p_start.add_argument(
        "--local_ip",
        default=LOCAL_IP,
        help="本机可达 IP：NPU 面 HCCL_IF_IP 与代理连 prefill/decode 的 host（默认 LOCAL_IP 常量）",
    )
    p_start.add_argument(
        "--start_mode",
        choices=("serial", "parallel"),
        default="parallel",
        help="组件拉起模式：serial=串行（prefill ready 后再起 decode）；parallel=并行（prefill/decode 都拉起后再统一等 ready）",
    )

    sub.add_parser("stop", help="停止代理（若曾启动）与 decode、prefill")

    return parser


def _runtime_from_args(args: argparse.Namespace) -> PdRuntimeConfig:
    rel = args.pd_mode.strip("/").replace("\\", "/")
    topo = (PKG_DIR / rel).resolve()
    return PdRuntimeConfig(
        topo_dir=topo,
        vllm_venv=Path(args.vllm_venv).resolve(),
        pd_mode=args.pd_mode,
        nic_name=args.nic_name,
        local_ip=args.local_ip,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_cli_parser().parse_args(argv)

    if args.cmd == "stop":
        # 仅读写 /tmp/vllm_*.pid，不依赖拓扑目录；rt 占位满足构造即可。
        PdServiceCtl(
            PdRuntimeConfig(topo_dir=PKG_DIR, vllm_venv=VLLM_VENV.resolve()),
            log=log_default,
        ).stop(use_proxy=True)
        return 0

    rt = _runtime_from_args(args)
    ctl = PdServiceCtl(rt, log=log_default)
    log_dir = Path(args.log_dir).resolve()
    return ctl.start_stack(log_dir, with_proxy=None, ready_timeout_s=None, start_mode=args.start_mode)


if __name__ == "__main__":
    raise SystemExit(main())
