#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""累计 / 长 KV 抽测 + FlashDecode smoke（原 verify_cum_matrix.py）。

用法:
  python tests/ut/ops/fia_nan_unit/verify_cum_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    _REPO,
    require_npu,
    resolve_out_dir,
    run_case,
    run_smoke_suite,
)

# Historical cum matrix: B=16, selected rem / long-KV points.
KV_LENS = (23, 96, 160, 208, 224, 240, 520, 1000)
BATCH = 16
TRIALS = 4


def main() -> int:
    device = require_npu()
    out_dir = resolve_out_dir("verify_cum_matrix")
    rows: list[dict] = []

    print(f"[verify_cum_matrix] repo={_REPO} out={out_dir}", flush=True)
    for kv in KV_LENS:
        r = run_case(BATCH, kv, device=device, trials=TRIALS)
        rows.append(r)
        print(json.dumps(r), flush=True)

    smoke = run_smoke_suite(device, names=("FD_multi", "FD_single"))
    smoke_ok = all(v == "PASS" for v in smoke.values())
    failed = [r for r in rows if not r["ok"]]
    summary = {
        "suite": "verify_cum_matrix",
        "failed_cases": len(failed),
        "smoke": smoke,
        "smoke_ok": smoke_ok,
        "ALL_PASS": len(failed) == 0 and smoke_ok,
        "out_dir": str(out_dir),
    }
    (out_dir / "verify_cum_matrix_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (out_dir / "verify_cum_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    print("DONE", flush=True)
    return 0 if summary["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
