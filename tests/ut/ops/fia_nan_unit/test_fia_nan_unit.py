#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
FIA K8V4 NaN 回归单元测试（serve c16 慢路径相关形状）。

关注：bit_residual_fia_paged_k8v4 输出是否出现 NaN/Inf。
不测吞吐；Paged attn 仅作对照（应始终有限）。

需 NPU + 已 rebuild 的自定义算子（非纯 CPU CI）。

Run（容器内，卡 5）:
  cd /vllm-workspace/vllm-ascend
  ASCEND_RT_VISIBLE_DEVICES=5 ASCEND_CUSTOM_OPP_PATH=\\
    /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend \\
    python tests/ut/ops/fia_nan_unit/test_fia_nan_unit.py -v

或:
  bash tests/ut/ops/fia_nan_unit/run.sh

产物默认写到 /root/yyz/fia_nan_unit/YYYYMMDD_HHMMSS/（避免污染仓库）；
也可设环境变量 FIA_NAN_UNIT_OUT 指定输出目录。
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# tests/ut/ops/fia_nan_unit -> repo root
REPO_ROOT = SCRIPT_DIR.parents[3]
REPO_CANDIDATES = [
    REPO_ROOT,
    Path("/vllm-workspace/vllm-ascend"),
    Path("/root/yyz/code/vllm-project/vllm-ascend"),
]


def _resolve_out_dir() -> Path:
    """Prefer FIA_NAN_UNIT_OUT; else /root/yyz/fia_nan_unit/<YYYYMMDD_HHMMSS>/."""
    env = os.environ.get("FIA_NAN_UNIT_OUT", "").strip()
    if env:
        out = Path(env)
    else:
        out = Path("/root/yyz/fia_nan_unit") / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


OUT_DIR = _resolve_out_dir()


def _setup_repo() -> Path:
    for root in REPO_CANDIDATES:
        if (root / "vllm_ascend").is_dir():
            sys.path.insert(0, str(root))
            os.chdir(root)
            cann = root / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
            if cann.is_dir():
                prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
                    str(cann) if not prev else f"{cann}:{prev}"
                )
            return root
    raise RuntimeError("vllm-ascend repo not found under known paths")


_REPO = _setup_repo()

import torch  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

from tests.e2e.singlecard.xrx_bit_residual_fia_paged_k8v4_smoke import (  # noqa: E402
    HEAD_SIZE,
    _run_attn,
    _run_fia,
)
from tests.e2e.singlecard.xrx_bit_residual_k8v4_golden import (  # noqa: E402
    BLOCK_SIZE,
    KEY_BLOCK_STRIDE,
    SEED,
    VALUE_BLOCK_STRIDE,
    _build_block_table,
    _identity,
)

# Qwen3-0.6B-ish decode shapes used in serve c16 investigation.
NUM_HEADS = 16
NUM_KV_HEADS = 8
DTYPE = torch.bfloat16

# Cover rem0 / rem8 / rem16 and short serve-like lengths (20:20 grows ~20→40).
KV_LENS = (16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 96, 104, 112, 128)
BATCHES = (1, 16)
TRIALS = 3
RECORDS: list[dict] = []


def _require_npu() -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise unittest.SkipTest("torch.npu is not available")
    if not enable_custom_op():
        raise unittest.SkipTest("vllm_ascend_C not loaded; rebuild custom ops first")
    for name in (
        "bit_residual_pack_k8v4",
        "bit_residual_attention_paged_k8v4",
        "bit_residual_fia_paged_k8v4",
    ):
        if not hasattr(torch.ops._C_ascend, name):
            raise unittest.SkipTest(f"missing op: {name}")
    return torch.device("npu")


def _run_once(
    *,
    batch: int,
    kv: int,
    seed: int,
    device: torch.device,
    rotation: torch.Tensor,
) -> dict:
    torch.manual_seed(seed)
    asq = list(range(1, batch + 1))
    ask = [kv] * batch
    q = torch.randn((batch, NUM_HEADS, HEAD_SIZE), dtype=DTYPE, device=device) * 0.5
    bt_cpu, nb = _build_block_table(ask)
    kc = torch.zeros(nb, NUM_KV_HEADS, KEY_BLOCK_STRIDE, dtype=torch.uint8, device=device)
    vc = torch.zeros(
        nb, NUM_KV_HEADS, VALUE_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    for seq in range(batch):
        key = torch.randn((kv, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE, device=device) * 0.5
        value = (
            torch.randn((kv, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE, device=device) * 0.5
        )
        slots = [
            int(bt_cpu[seq, pos // BLOCK_SIZE]) * BLOCK_SIZE + (pos % BLOCK_SIZE)
            for pos in range(kv)
        ]
        torch.ops._C_ascend.bit_residual_pack_k8v4(
            key,
            value,
            torch.tensor(slots, dtype=torch.int32, device=device),
            torch.tensor([0, kv], dtype=torch.int32, device=device),
            rotation,
            kc,
            vc,
            1,
            BLOCK_SIZE,
        )
    bt = bt_cpu.to(device)
    fia = _run_fia(
        query=q,
        key_cache=kc,
        value_cache=vc,
        block_table=bt,
        actual_seq_lens_q=asq,
        actual_seq_lens_kv=ask,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
    )
    attn = _run_attn(
        query=q,
        key_cache=kc,
        value_cache=vc,
        block_table=bt,
        actual_seq_lens_q=asq,
        actual_seq_lens_kv=ask,
        rotation_key=rotation,
        rotation_value=rotation,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
    )
    fia_f = fia.float()
    attn_f = attn.float()
    fia_nan = int(torch.isnan(fia_f).sum().item())
    fia_inf = int(torch.isinf(fia_f).sum().item())
    attn_nan = int(torch.isnan(attn_f).sum().item())
    fia_ok = bool(torch.isfinite(fia_f).all().item())
    attn_ok = bool(torch.isfinite(attn_f).all().item())
    nan_rows = 0
    if fia.ndim >= 2 and not fia_ok:
        # (B, H, D) — count rows with any non-finite in last dim over H*D flatten per batch
        flat = fia_f.view(batch, -1)
        nan_rows = int((~torch.isfinite(flat)).any(dim=-1).sum().item())
    maxdiff = None
    if fia_ok and attn_ok:
        maxdiff = float((fia_f - attn_f).abs().max().item())
    return {
        "batch": batch,
        "kv": kv,
        "rem32": kv % 32,
        "seed": seed,
        "fia_finite": fia_ok,
        "attn_finite": attn_ok,
        "fia_nan": fia_nan,
        "fia_inf": fia_inf,
        "attn_nan": attn_nan,
        "nan_rows": nan_rows,
        "maxdiff": maxdiff,
    }


class FiaNanUnitTest(unittest.TestCase):
    """Matrix: batch × kv × trials — fail if FIA emits NaN/Inf."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = _require_npu()
        cls.rotation = _identity(DTYPE, cls.device)

    def _check_case(self, batch: int, kv: int) -> None:
        fails: list[str] = []
        for t in range(TRIALS):
            rec = _run_once(
                batch=batch,
                kv=kv,
                seed=SEED + t,
                device=self.device,
                rotation=self.rotation,
            )
            RECORDS.append(rec)
            if not rec["attn_finite"]:
                fails.append(
                    f"trial={t} Paged also non-finite (nan={rec['attn_nan']}) — env/op broken?"
                )
            elif not rec["fia_finite"]:
                fails.append(
                    f"trial={t} FIA NaN/Inf nan={rec['fia_nan']} inf={rec['fia_inf']} "
                    f"nan_rows={rec['nan_rows']} rem32={rec['rem32']}"
                )
        if fails:
            self.fail(
                f"B={batch} kv={kv} rem32={kv % 32} NaN regressions:\n  "
                + "\n  ".join(fails)
            )


def _add_matrix_tests() -> None:
    for batch in BATCHES:
        for kv in KV_LENS:

            def _make(b: int = batch, k: int = kv):
                def _test(self: FiaNanUnitTest) -> None:
                    self._check_case(b, k)

                return _test

            name = f"test_fia_finite_B{batch}_kv{kv}_rem{kv % 32}"
            setattr(FiaNanUnitTest, name, _make())


_add_matrix_tests()


class FiaNanServeLikeTest(unittest.TestCase):
    """Serve 20:20×c16 相关短 KV（20/24/40）加重试。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = _require_npu()
        cls.rotation = _identity(DTYPE, cls.device)

    def test_c16_short_kv_no_nan(self) -> None:
        device = self.device
        rot = self.rotation
        bad: list[str] = []
        for kv in (20, 24, 40):
            for t in range(5):
                rec = _run_once(
                    batch=16, kv=kv, seed=SEED + 100 + t, device=device, rotation=rot
                )
                RECORDS.append({**rec, "suite": "serve_like"})
                if not rec["fia_finite"]:
                    bad.append(
                        f"kv={kv} t={t} nan={rec['fia_nan']} rows={rec['nan_rows']}"
                    )
        if bad:
            self.fail("serve-like c16 short-KV NaN:\n  " + "\n  ".join(bad))


def _flush_records() -> None:
    out = OUT_DIR / "fia_nan_unit_results.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for rec in RECORDS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "n_records": len(RECORDS),
        "fia_nan_cases": sum(1 for r in RECORDS if not r["fia_finite"]),
        "attn_nan_cases": sum(1 for r in RECORDS if not r["attn_finite"]),
        "repo": str(_REPO),
        "out_dir": str(OUT_DIR),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT_DIR / "fia_nan_unit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[fia_nan_unit] out_dir={OUT_DIR}", flush=True)
    print(f"[fia_nan_unit] wrote {out} summary={summary}", flush=True)


if __name__ == "__main__":
    print(f"[fia_nan_unit] OUT_DIR={OUT_DIR}", flush=True)
    prog = unittest.main(verbosity=2, exit=False)
    _flush_records()
    ok = prog.result.wasSuccessful() and all(r["fia_finite"] for r in RECORDS)
    sys.exit(0 if ok else 1)
