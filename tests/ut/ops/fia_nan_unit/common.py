# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared helpers for FIA K8V4 NaN / rem-matrix kernel verifies (NPU)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REPO_CANDIDATES = [
    REPO_ROOT,
    Path("/vllm-workspace/vllm-ascend"),
    Path("/root/yyz/code/vllm-project/vllm-ascend"),
]

DEFAULT_OUT_BASE = Path("/root/yyz/fia_nan_unit")


def setup_repo() -> Path:
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
            os.environ.setdefault("VLLM_ASCEND_BIT_RESIDUAL_FIA_FORCE", "1")
            return root
    raise RuntimeError("vllm-ascend repo not found under known paths")


_REPO = setup_repo()

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

NUM_HEADS = 16
NUM_KV_HEADS = 8
DTYPE = torch.bfloat16


def resolve_out_dir(subdir: str | None = None) -> Path:
    """Prefer FIA_NAN_UNIT_OUT; else /root/yyz/fia_nan_unit/<ts>[_subdir]/."""
    env = os.environ.get("FIA_NAN_UNIT_OUT", "").strip()
    if env:
        out = Path(env)
    else:
        base = Path(os.environ.get("FIA_NAN_UNIT_BASE", str(DEFAULT_OUT_BASE)))
        name = time.strftime("%Y%m%d_%H%M%S")
        if subdir:
            name = f"{name}_{subdir}"
        out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def require_npu() -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C not loaded; rebuild custom ops first")
    for name in (
        "bit_residual_pack_k8v4",
        "bit_residual_attention_paged_k8v4",
        "bit_residual_fia_paged_k8v4",
    ):
        if not hasattr(torch.ops._C_ascend, name):
            raise RuntimeError(f"missing op: {name}")
    return torch.device("npu")


def run_once(
    *,
    batch: int,
    kv: int,
    seed: int,
    device: torch.device,
    rotation: torch.Tensor,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    dtype: torch.dtype = DTYPE,
) -> dict:
    """Pack random KV, run FIA + Paged, return finiteness / maxdiff stats."""
    torch.manual_seed(seed)
    asq = list(range(1, batch + 1))
    ask = [kv] * batch
    q = torch.randn((batch, num_heads, HEAD_SIZE), dtype=dtype, device=device) * 0.5
    bt_cpu, nb = _build_block_table(ask)
    kc = torch.zeros(
        nb, num_kv_heads, KEY_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    vc = torch.zeros(
        nb, num_kv_heads, VALUE_BLOCK_STRIDE, dtype=torch.uint8, device=device
    )
    for seq in range(batch):
        key = (
            torch.randn((kv, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device) * 0.5
        )
        value = (
            torch.randn((kv, num_kv_heads, HEAD_SIZE), dtype=dtype, device=device) * 0.5
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
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
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
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    fia_f = fia.float()
    attn_f = attn.float()
    fia_ok = bool(torch.isfinite(fia_f).all().item())
    attn_ok = bool(torch.isfinite(attn_f).all().item())
    fia_nan = int(torch.isnan(fia_f).sum().item())
    fia_inf = int(torch.isinf(fia_f).sum().item())
    attn_nan = int(torch.isnan(attn_f).sum().item())
    nan_rows = 0
    if fia.ndim >= 2 and not fia_ok:
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


def run_case(
    batch: int,
    kv: int,
    *,
    device: torch.device,
    trials: int = 4,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    dtype: torch.dtype = DTYPE,
    seed0: int = SEED,
    maxdiff_tol: float | None = None,
) -> dict:
    """Multi-trial aggregate used by cum/full verifies."""
    if maxdiff_tol is None:
        maxdiff_tol = 0.02 if dtype == torch.bfloat16 else 0.005
    rot = _identity(dtype, device)
    bad = 0
    diffs: list[float] = []
    for t in range(trials):
        rec = run_once(
            batch=batch,
            kv=kv,
            seed=seed0 + t,
            device=device,
            rotation=rot,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )
        if not rec["fia_finite"]:
            bad += 1
        elif rec["maxdiff"] is not None:
            diffs.append(rec["maxdiff"])
    max_diff = max(diffs) if diffs else None
    return {
        "B": batch,
        "kv": kv,
        "rem32": kv % 32,
        "nan": bad,
        "max_diff": max_diff,
        "ok": bad == 0 and (max_diff if max_diff is not None else 0.0) < maxdiff_tol,
        "trials": trials,
        "dtype": str(dtype).replace("torch.", ""),
    }


def run_smoke_suite(device: torch.device, names: tuple[str, ...] | None = None) -> dict:
    """Run selected xrx FIA smoke tests; return {name: ok/error}."""
    import tests.e2e.singlecard.xrx_bit_residual_fia_paged_k8v4_smoke as smoke

    suite = {
        "FD_multi": smoke.test_flash_decode_multibatch_vs_attn,
        "FD_single": smoke.test_flash_decode_long_kv,
        "bf16_decode": smoke.test_bf16_decode_multi_kv_gqa,
    }
    if names is not None:
        suite = {k: suite[k] for k in names if k in suite}
    results: dict[str, str] = {}
    for name, fn in suite.items():
        try:
            fn(device)
            results[name] = "PASS"
            print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001 — report and continue
            results[name] = f"FAIL {str(e)[:160]}"
            print(results[name], flush=True)
    return results


__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_OUT_BASE",
    "DTYPE",
    "HEAD_SIZE",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "REPO_ROOT",
    "SEED",
    "_REPO",
    "_identity",
    "require_npu",
    "resolve_out_dir",
    "run_case",
    "run_once",
    "run_smoke_suite",
    "setup_repo",
]
