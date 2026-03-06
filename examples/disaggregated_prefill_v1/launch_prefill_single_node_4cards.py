#!/usr/bin/env python3
# 单机 4 卡 Prefill：1 个 DP 引擎，TP=4，使用 NPU 0,1,2,3
# python launch_prefill_single_node_4cards.py
# 脚本会调用同目录下 Qwen3-32B/run_prefill_4cards.sh。
import os
import sys
import multiprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(SCRIPT_DIR, "Qwen3-32B", "run_prefill_4cards.sh")

# DP 与端口配置（单机单进程时仅一组即可，扩展时可循环多组）
dp_size = 1              # DP 总副本数（当前单机单 Prefill 引擎为 1）
dp_rank_local = 0        # 本进程在本机内的 DP 秩
dp_rank = 0              # 本进程的全局 DP 秩
dp_ip = "127.0.0.1"      # DP 通信用的本机 IP，多机时改为实际 IP
dp_port = 13395          # DP 通信用的 master 端口（需与 Decode 侧不同，避免 EADDRINUSE）
engine_port = 9000       # 本进程 vLLM 服务监听端口
dp_size_local = 1        # 本机上的 DP 进程数（当前为 1）
visible_devices = "0,1,2,3"  # 本进程可见的 NPU 设备，Prefill 使用卡 0–3


def run_command(dp_rank_local_, dp_rank_, engine_port_, dp_size_local_, visible_devices_):
    """启动一个 Prefill 进程：设置可见设备并调用 shell 模板。多进程时各进程传入不同的 rank/port/devices。"""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices_
    command = (
        f"bash {template_path} {dp_size} {dp_ip} {dp_port} "
        f"{dp_rank_local_} {dp_rank_} {engine_port_} {dp_size_local_}"
    )
    os.system(command)


if __name__ == "__main__":
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        print("Create Qwen3-32B/run_prefill_4cards.sh.")
        sys.exit(1)

    process = multiprocessing.Process(
        target=run_command,
        args=(dp_rank_local, dp_rank, engine_port, dp_size_local, visible_devices),
    )
    process.start()
    process.join()
