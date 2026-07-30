#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Rem 矩阵内核冒烟（原 20260728_dual_aiv_smoke.sh → 纯 Python）。

覆盖 B∈{1,16} × kv∈{16…128}（含 rem0/8/16），检查 FIA NaN 与 vs Paged maxdiff。

用法:
  python tests/ut/ops/fia_nan_unit/verify_rem_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DTYPE,
    SEED,
    _REPO,
    _identity,
    require_npu,
    resolve_out_dir,
    run_once,
)

# Same shape set as historical dual_aiv smoke / fia_nan_unit short matrix.
KV_LENS = (16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 96, 104, 112, 128)
BATCHES = (1, 16)
TRIALS = 3
MAXDIFF_FAIL = 1.0


def main() -> int:
    device = require_npu()
    rot = _identity(DTYPE, device)
    out_dir = resolve_out_dir("verify_rem_matrix")
    records: list[dict] = []
    fail = 0

    print(f"[verify_rem_matrix] repo={_REPO} out={out_dir}", flush=True)
    for batch in BATCHES:
        for kv in KV_LENS:
            nans = 0
            mds: list[float] = []
            for t in range(TRIALS):
                rec = run_once(
                    batch=batch,
                    kv=kv,
                    seed=SEED + t,
                    device=device,
                    rotation=rot,
                )
                records.append({**rec, "suite": "rem_matrix"})
                if not rec["fia_finite"]:
                    nans += 1
                if rec["maxdiff"] is not None:
                    mds.append(rec["maxdiff"])
            rem = kv % 32
            bad = nans > 0 or (mds and max(mds) > MAXDIFF_FAIL)
            md_s = f"maxdiff={max(mds):.4g}" if mds else "maxdiff=n/a"
            status = "FAIL" if bad else "OK"
            if bad:
                fail += 1
            print(
                f"B={batch:2d} kv={kv:3d} rem32={rem:2d} "
                f"fia_nan={nans}/{TRIALS} {md_s} {status}",
                flush=True,
            )

    summary = {
        "suite": "verify_rem_matrix",
        "smoke_fails": fail,
        "n_records": len(records),
        "fia_nan_cases": sum(1 for r in records if not r["fia_finite"]),
        "ALL_PASS": fail == 0,
        "out_dir": str(out_dir),
    }
    (out_dir / "verify_rem_matrix_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    (out_dir / "verify_rem_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    print("SMOKE_FAILS", fail, flush=True)
    print("SMOKE_DONE", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
