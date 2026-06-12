#!/usr/bin/env python3
"""1P1+1D1 负载均衡代理：1 个 P（9000）+ 1 个 D（9010），对外 8000。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[3]
print(_CODE_ROOT)

PROXY_SCRIPT = (
    _CODE_ROOT
    / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
)


def main() -> None:
    if not PROXY_SCRIPT.is_file():
        print(f"pd_proxy: 未找到 {PROXY_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    host = os.environ.get("LOCAL_IP") or os.environ.get("PREFILL_HOST", "172.17.0.4")
    port = os.environ.get("PROXY_PORT", "9876")
    argv = [
        sys.executable,
        str(PROXY_SCRIPT),
        "--port",
        port,
        "--host",
        "0.0.0.0",
        "--prefiller-hosts",
        host,
        "--prefiller-ports",
        "9000",
        "--decoder-hosts",
        host,
        "--decoder-ports",
        "9010",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
