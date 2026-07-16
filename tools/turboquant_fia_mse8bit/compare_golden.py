#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the FIA TurboQuant P0 kernel output against the golden reference.

The C++ test (test_aclnn_fused_infer_attention_score_tnd_pa_turboquant.cpp)
must dump its attentionOut (fp16, TND [S,H,D]) to a file. The simplest way is
to set the env GOLDEN_OUT_PATH; if the test is patched to write outHostData
there when that env is set, this script reads it and compares.

Usage
-----
  # 1. generate golden (pick the Π mode matching what the C++ test uses)
  python3 golden_tnd_pa_turboquant.py --pi identity   --out-dir golden_identity
  python3 golden_tnd_pa_turboquant.py --pi rotation   --out-dir golden_rotation

  # 2. run the C++ kernel with GOLDEN_OUT_PATH set (after patching the test
  #    to dump outHostData, see fi_turboquant_pi_impl_review.md §3 step 1)
  GOLDEN_OUT_PATH=./kernel_out.bin ./output/test_aclnn_fused_infer_attention_score_tnd_pa_turboquant

  # 3. compare
  python3 compare_golden.py --kernel kernel_out.bin --golden golden_identity/golden_out.bin
  python3 compare_golden.py --kernel kernel_out.bin --golden golden_rotation/golden_out.bin --atol 2e-2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Compare kernel output vs golden")
    ap.add_argument("--kernel", required=True, help="kernel output fp16 bin [S,H,D] TND")
    ap.add_argument("--golden", required=True, help="golden_out.bin from golden_tnd_pa_turboquant.py")
    ap.add_argument("--atol", type=float, default=None, help="override atol (else from golden_meta.json)")
    ap.add_argument("--rtol", type=float, default=None, help="override rtol (else from golden_meta.json)")
    ap.add_argument("--dump-diff", default=None, help="write per-element abs-diff fp16 to this path")
    args = ap.parse_args()

    ker = np.fromfile(args.kernel, dtype=np.float16)
    gol = np.fromfile(args.golden, dtype=np.float16)
    if ker.size != gol.size:
        print(f"[FAIL] size mismatch: kernel={ker.size} golden={gol.size}")
        sys.exit(1)

    # load tolerance from golden_meta.json sibling unless overridden
    meta_path = Path(args.golden).parent / "golden_meta.json"
    atol = args.atol
    rtol = args.rtol
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if atol is None:
            atol = meta.get("tolerance", {}).get("atol", 1e-3)
        if rtol is None:
            rtol = meta.get("tolerance", {}).get("rtol", 1e-3)
        pi_mode = meta.get("pi_mode", "?")
        print(f"[cmp] golden Π mode: {pi_mode}")
    if atol is None:
        atol = 1e-3
    if rtol is None:
        rtol = 1e-3
    print(f"[cmp] tolerance: atol={atol} rtol={rtol}")

    kf = ker.astype(np.float64)
    gf = gol.astype(np.float64)
    abs_diff = np.abs(kf - gf)
    rel_diff = abs_diff / (np.abs(gf) + 1e-9)
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    mean_abs = float(abs_diff.mean())
    n_fail_abs = int(np.sum(abs_diff > atol))
    n_fail_rel = int(np.sum(rel_diff > rtol))
    n_total = ker.size

    print(f"[cmp] elements      : {n_total}")
    print(f"[cmp] max abs diff  : {max_abs:.6f}")
    print(f"[cmp] mean abs diff : {mean_abs:.6f}")
    print(f"[cmp] max rel diff  : {max_rel:.6f}")
    print(f"[cmp] fail(abs>tol) : {n_fail_abs}/{n_total} ({100.0*n_fail_abs/n_total:.2f}%)")
    print(f"[cmp] fail(rel>tol) : {n_fail_rel}/{n_total} ({100.0*n_fail_rel/n_total:.2f}%)")

    # show a few worst elements
    flat_idx = np.argsort(abs_diff)[::-1][:8]
    # shape is [S,H,D] = [16,16,128]
    print("[cmp] worst elements (flat_idx, kernel, golden, abs_diff):")
    for fi in flat_idx:
        s = fi // (16 * 128)
        h = (fi % (16 * 128)) // 128
        d = fi % 128
        print(f"        [{s},{h},{d}] ker={kf[fi]:.6f} gol={gf[fi]:.6f} diff={abs_diff[fi]:.6f}")

    if args.dump_diff:
        abs_diff.astype(np.float16).tofile(args.dump_diff)
        print(f"[cmp] wrote per-element abs-diff to {args.dump_diff}")

    if max_abs <= atol and max_rel <= rtol:
        print("[cmp] PASS ✅ (max_abs<atol AND max_rel<rtol)")
        sys.exit(0)
    elif max_abs <= atol:
        print("[cmp] PASS (max_abs<atol; rel on near-zero elements is noisy, ignored)")
        sys.exit(0)
    else:
        print(f"[cmp] FAIL ❌ (max_abs={max_abs:.6f} > atol={atol})")
        sys.exit(2)


if __name__ == "__main__":
    main()
