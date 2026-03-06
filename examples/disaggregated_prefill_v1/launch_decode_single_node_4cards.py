#!/usr/bin/env python3
# 单机 4 卡 Decode：1 个 DP 引擎，TP=4，使用 NPU 4,5,6,7
# python launch_decode_single_node_4cards.py
# 脚本会调用同目录下 Qwen3-32B/run_decode_4cards.sh。

import os
import sys
import multiprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(SCRIPT_DIR, "Qwen3-32B", "run_decode_4cards.sh")

# DP 与端口配置（单机单进程时仅一组即可，扩展时可循环多组）
dp_size = 1              # DP 总副本数（当前单机单 Decode 引擎为 1）
dp_rank_local = 0         # 本进程在本机内的 DP 秩
dp_rank = 0               # 本进程的全局 DP 秩
dp_ip = "127.0.0.1"       # DP 通信用的本机 IP，多机时改为实际 IP
dp_port = 13495           # DP 通信用的 master 端口（需与 Prefill 的 13395 不同，避免 EADDRINUSE）
engine_port = 9010        # 本进程 vLLM 服务监听端口
dp_size_local_for_process = 1  # 本进程内 DP 大小（单引擎为 1）
visible_devices = "4,5,6,7"     # 本进程可见的 NPU 设备，Decode 使用卡 4–7


def run_command(dp_rank_local_, dp_rank_, engine_port_, dp_size_local_, visible_devices_):
    """启动一个 Decode 进程：设置可见设备并调用 shell 模板。$8 传入模板作为 ASCEND_RT_VISIBLE_DEVICES。多进程时各进程传入不同的 rank/port/devices。"""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices_
    command = (
        f"bash {template_path} {dp_size} {dp_ip} {dp_port} "
        f"{dp_rank_local_} {dp_rank_} {engine_port_} {dp_size_local_} {visible_devices_}"
    )
    os.system(command)


if __name__ == "__main__":
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        print("Create Qwen3-32B/run_decode_4cards.sh.")
        sys.exit(1)

    process = multiprocessing.Process(
        target=run_command,
        args=(dp_rank_local, dp_rank, engine_port, dp_size_local_for_process, visible_devices),
    )
    process.start()
    process.join()
