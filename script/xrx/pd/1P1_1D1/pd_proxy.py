#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[4]
PROXY_SCRIPT = CODE_ROOT / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"


def main() -> None:
    if not PROXY_SCRIPT.is_file():
        print(f"pd_proxy: missing proxy script: {PROXY_SCRIPT}", file=sys.stderr)
        raise SystemExit(1)

    host = os.environ.get("LOCAL_IP", "127.0.0.1")
    proxy_port = os.environ.get("PROXY_PORT", "9878")
    prefill_port = os.environ.get("PREFILL_PORT", "9000")
    decode_port = os.environ.get("DECODE_PORT", "9010")

    argv = [
        sys.executable,
        str(PROXY_SCRIPT),
        "--host",
        "0.0.0.0",
        "--port",
        proxy_port,
        "--prefiller-hosts",
        host,
        "--prefiller-ports",
        prefill_port,
        "--decoder-hosts",
        host,
        "--decoder-ports",
        decode_port,
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
