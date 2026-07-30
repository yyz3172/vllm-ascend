#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""完整 FIA 内核回归（原 verify_full_fia.py）：长 KV 矩阵 + 多 batch/fp16 + smoke。

用法:
  python tests/ut/ops/fia_nan_unit/verify_full_fia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from common import (  # noqa: E402
    _REPO,
    require_npu,
    resolve_out_dir,
    run_case,
    run_smoke_suite,
)

# Historical full matrix (B=16 bf16) + smaller-B fp16 long-KV.
KV_LENS_BF16_B16 = (
    23,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    520,
    1000,
)
BATCHES_FP16 = (1, 2, 4, 8)
KV_FP16 = 1000
TRIALS = 4


def main() -> int:
    device = require_npu()
    out_dir = resolve_out_dir("verify_full_fia")
    rows: list[dict] = []

    print(f"[verify_full_fia] repo={_REPO} out={out_dir}", flush=True)
    for kv in KV_LENS_BF16_B16:
        r = run_case(16, kv, device=device, trials=TRIALS)
        rows.append(r)
        print(json.dumps(r), flush=True)

    for batch in BATCHES_FP16:
        r = run_case(
            batch,
            KV_FP16,
            device=device,
            trials=TRIALS,
            num_heads=8,
            num_kv_heads=4,
            dtype=torch.float16,
        )
        rows.append(r)
        print(json.dumps(r), flush=True)

    smoke = run_smoke_suite(
        device, names=("FD_multi", "FD_single", "bf16_decode")
    )
    smoke_ok = all(v == "PASS" for v in smoke.values())
    failed = [r for r in rows if not r["ok"]]
    summary = {
        "suite": "verify_full_fia",
        "failed_cases": len(failed),
        "smoke": smoke,
        "smoke_ok": smoke_ok,
        "ALL_PASS": len(failed) == 0 and smoke_ok,
        "out_dir": str(out_dir),
    }
    (out_dir / "verify_full_fia_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (out_dir / "verify_full_fia_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return 0 if summary["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
