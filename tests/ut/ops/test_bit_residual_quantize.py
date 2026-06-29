"""
Comparison tool: BitResidual 4-bit vs TurboQuant 4-bit MSE quantization.

Both schemes use 4 bits per dimension (plus norm overhead) and share a
random orthogonal rotation matrix R. BitResidual decomposes the 4 bits
into 1-bit sign + 3-bit uniform residual; TurboQuant uses Lloyd-Max
codebook lookup (argmin |y - codebook[k]|).

This tool compares reconstruction error for the same input data under
both methods. Random rotation is mandatory for BitResidual (enforced
by passing _make_rotation() to every quantize/dequantize call).

Usage:
    python3 tests/ut/ops/test_bit_residual_quantize.py              # run all
    python3 tests/ut/ops/test_bit_residual_quantize.py --fused      # include NPU fused op tests
    python3 tests/ut/ops/test_bit_residual_quantize.py --basic      # only basic correctness
    python3 tests/ut/ops/test_bit_residual_quantize.py --comparison # only TQ comparison
    python3 tests/ut/ops/test_bit_residual_quantize.py --real-data  # only real data
"""

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from vllm_ascend.ops.bit_residual_quantize import (
    bit_residual_quantize,
    bit_residual_dequantize,
    compute_metrics,
    BITRESIDUAL_ROW_BYTES,
)
from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    _get_quantizer,
    _turboquant_pack_tables,
    _turboquant_slab_group4_to_row_format,
    refresh_turboquant_env_cache,
    turboquant_quantize_to_packed_bytes,
    turboquant_dequantize_from_packed_bytes,
    turboquant_packed_bytes_per_vector,
    turboquant_slab_row_bytes,
    unpack_uint4,
)

try:
    _NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception:
    _NPU_AVAILABLE = False

D = 128  # head_size
SEED = 42
REAL_SMOKE_DUMP_PATH = Path(__file__).with_name("data") / "tq4bit_real_smoke_dump.pt"

_passed = 0
_failed = 0
_skipped = 0


def _pass(name: str) -> None:
    global _passed
    _passed += 1
    print(f"  [PASS] {name}")


def _fail(name: str, msg: str) -> None:
    global _failed
    _failed += 1
    print(f"  [FAIL] {name}: {msg}")


def _skip(name: str, reason: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  [SKIP] {name}: {reason}")


def _make_rotation(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate the same rotation matrix as TurboQuant (Haar orthogonal, seed=42).

    BitResidual must use the identical rotation matrix as TurboQuant for
    fair comparison — both share the same random orthogonal transform.
    """
    return _get_quantizer(D, bits=4, device=device).rotation.to(device=device, dtype=torch.float32)


def _turboquant_4bit_reference(x: torch.Tensor) -> dict:
    """Quantize + dequantize with TurboQuant 4-bit MSE (v1-like, CPU)."""
    bits = 4
    packed = turboquant_quantize_to_packed_bytes(x, bits=bits)
    reconstructed = turboquant_dequantize_from_packed_bytes(
        packed, head_size=x.shape[-1], dtype=x.dtype, bits=bits
    )
    return {"packed": packed, "reconstructed": reconstructed}


def _print_metrics_table(title: str, *rows: tuple[str, dict]) -> None:
    """Print a comparison table of metrics from compute_metrics() dicts."""
    metric_keys = ["mse", "max_error", "relative_error", "cosine_similarity"]
    col_width = 15
    print(f"\n{'='*60}")
    print(f"{title}")
    header = f"{'Metric':<20}" + "".join(f"{label:<{col_width}}" for label, _ in rows)
    print(header)
    print(f"{'-'*60}")
    for mk in metric_keys:
        line = f"{mk:<20}" + "".join(f"{m[mk]:<{col_width}.6f}" for _, m in rows)
        print(line)
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------

def test_roundtrip_shape() -> None:
    """Quantize then dequantize should return same shape."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        name = f"roundtrip_shape/{dtype}"
        device = torch.device("cpu")
        R = _make_rotation(device)
        x = torch.randn(10, D, dtype=dtype, device=device)
        packed = bit_residual_quantize(x, R, head_size=D)
        x_hat = bit_residual_dequantize(packed, R, head_size=D, dtype=dtype)
        metrics = compute_metrics(x, x_hat)
        _print_metrics_table(f"Roundtrip shape, dtype={dtype}", ("BitResidual", metrics))
        if x_hat.shape == x.shape:
            _pass(name)
        else:
            _fail(name, f"shape mismatch: {x_hat.shape} != {x.shape}")


def test_zero_vector() -> None:
    """Zero vector should reconstruct with near-zero error."""
    name = "zero_vector"
    R = _make_rotation(torch.device("cpu"))
    x = torch.zeros(5, D, dtype=torch.float32)
    packed = bit_residual_quantize(x, R, head_size=D)
    x_hat = bit_residual_dequantize(packed, R, head_size=D, dtype=torch.float32)
    metrics = compute_metrics(x, x_hat)
    _print_metrics_table("Zero vector", ("BitResidual", metrics))
    max_err = metrics["max_error"]
    if max_err < 0.01:
        _pass(name)
    else:
        _fail(name, f"max_error={max_err}")


def test_packed_keys() -> None:
    """Quantize should return expected dict keys."""
    name = "packed_keys"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float16)
    packed = bit_residual_quantize(x, R, head_size=D)
    expected = ["norms", "signs", "bases", "steps", "q3_indices"]
    missing = [k for k in expected if k not in packed]
    if not missing:
        _pass(name)
    else:
        _fail(name, f"missing keys: {missing}")


def test_sign_bits_range() -> None:
    """Sign bits should be boolean."""
    name = "sign_bits_range"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float16)
    packed = bit_residual_quantize(x, R, head_size=D)
    ok = packed["signs"].dtype == torch.bool and packed["signs"].shape[-1] == D
    if ok:
        _pass(name)
    else:
        _fail(name, f"dtype={packed['signs'].dtype}, shape[-1]={packed['signs'].shape[-1]}")


def test_q3_indices_range() -> None:
    """3-bit indices should be in range 0..7."""
    name = "q3_indices_range"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float16)
    packed = bit_residual_quantize(x, R, head_size=D)
    q3 = packed["q3_indices"]
    ok = q3.dtype == torch.uint8 and q3.min().item() >= 0 and q3.max().item() <= 7
    if ok:
        _pass(name)
    else:
        _fail(name, f"dtype={q3.dtype}, min={q3.min()}, max={q3.max()}")


def test_per_vector_norms() -> None:
    """Norms should match actual L2 norm of each vector."""
    name = "per_vector_norms"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float32)
    packed = bit_residual_quantize(x, R, head_size=D)
    norms_hat = packed["norms"].float().squeeze(-1)
    norms_true = x.norm(p=2, dim=-1)
    diff = (norms_hat - norms_true).abs().max().item()
    if diff < 0.01:
        _pass(name)
    else:
        _fail(name, f"norm diff={diff}")


def test_rotation_balances_signs() -> None:
    """After random rotation, sign bits should be roughly 50/50."""
    name = "rotation_balances_signs"
    R = _make_rotation(torch.device("cpu"))
    torch.manual_seed(SEED)
    x = torch.randn(1000, D, dtype=torch.float32)
    packed = bit_residual_quantize(x, R, head_size=D)
    sign_ratio = packed["signs"].float().mean().item()
    print(f"\n  Sign ratio after random rotation: {sign_ratio:.4f} (expected ~0.5)")
    if 0.45 < sign_ratio < 0.55:
        _pass(name)
    else:
        _fail(name, f"sign_ratio={sign_ratio}")


def test_multi_head_shapes() -> None:
    """Should work with [N, D] and [N, H, D] shapes."""
    for shape in [(32, D), (8, 4, D)]:
        name = f"multi_head_shapes/{shape}"
        R = _make_rotation(torch.device("cpu"))
        x = torch.randn(*shape, dtype=torch.float16)
        packed = bit_residual_quantize(x, R, head_size=D)
        x_hat = bit_residual_dequantize(packed, R, head_size=D, dtype=torch.float16)
        metrics = compute_metrics(x, x_hat)
        _print_metrics_table(f"Multi-head shape={shape}", ("BitResidual", metrics))
        ok = x_hat.shape == x.shape and metrics["cosine_similarity"] > 0.95
        if ok:
            _pass(name)
        else:
            _fail(name, f"shape={x_hat.shape}, cos_sim={metrics['cosine_similarity']:.4f}")


def test_rotation_balances_skewed_data() -> None:
    """Random rotation balances sign bits on skewed data — mandatory for robustness."""
    name = "rotation_balances_skewed_data"
    N = 256
    torch.manual_seed(SEED)

    # Deliberately skewed data (all positive → sign imbalance without rotation)
    x_skewed = torch.randn(N, D, dtype=torch.float32).abs()

    # With random rotation (mandatory)
    R_random = _make_rotation(torch.device("cpu"))
    packed_rot = bit_residual_quantize(x_skewed, R_random, head_size=D)
    x_hat_rot = bit_residual_dequantize(packed_rot, R_random, head_size=D, dtype=torch.float32)
    metrics_rot = compute_metrics(x_skewed, x_hat_rot)

    # Without rotation (R = Identity)
    R_identity = torch.eye(D, dtype=torch.float32)
    packed_no_rot = bit_residual_quantize(x_skewed, R_identity, head_size=D)
    x_hat_no_rot = bit_residual_dequantize(packed_no_rot, R_identity, head_size=D, dtype=torch.float32)
    metrics_no_rot = compute_metrics(x_skewed, x_hat_no_rot)

    sign_ratio_rot = packed_rot["signs"].float().mean().item()
    sign_ratio_no_rot = packed_no_rot["signs"].float().mean().item()

    _print_metrics_table(
        f"Rotation on skewed data (N={N})",
        ("WithRotation", metrics_rot),
        ("NoRotation", metrics_no_rot),
    )
    print(f"  sign_ratio: with_rotation={sign_ratio_rot:.4f}, "
          f"no_rotation={sign_ratio_no_rot:.4f}")

    ok1 = abs(sign_ratio_rot - 0.5) < 0.10
    ok2 = sign_ratio_no_rot > 0.90
    ok3 = metrics_rot["cosine_similarity"] > 0.90
    ok4 = metrics_no_rot["cosine_similarity"] > 0.90
    if ok1 and ok2 and ok3 and ok4:
        _pass(name)
    else:
        msgs = []
        if not ok1: msgs.append(f"rot sign_ratio={sign_ratio_rot:.4f} not near 0.5")
        if not ok2: msgs.append(f"no_rot sign_ratio={sign_ratio_no_rot:.4f} not >0.90")
        if not ok3: msgs.append(f"rot cos_sim={metrics_rot['cosine_similarity']:.4f}")
        if not ok4: msgs.append(f"no_rot cos_sim={metrics_no_rot['cosine_similarity']:.4f}")
        _fail(name, "; ".join(msgs))


# ---------------------------------------------------------------------------
# Head-to-head comparison: BitResidual vs TurboQuant
# ---------------------------------------------------------------------------

def test_single_head_comparison() -> None:
    """Compare BitResidual vs TurboQuant for [N, D] tensors."""
    for dtype in [torch.float16, torch.bfloat16]:
        for N in [16, 128, 512]:
            name = f"single_head/N={N}/{dtype}"
            device = torch.device("cpu")
            torch.manual_seed(SEED)
            R = _make_rotation(device)
            x = torch.randn(N, D, dtype=dtype, device=device)

            packed_br = bit_residual_quantize(x, R, head_size=D)
            x_hat_br = bit_residual_dequantize(packed_br, R, head_size=D, dtype=dtype)
            metrics_br = compute_metrics(x, x_hat_br)

            x_hat_tq = _turboquant_4bit_reference(x)["reconstructed"]
            metrics_tq = compute_metrics(x, x_hat_tq)

            tq_row = turboquant_slab_row_bytes(D, bits=4)
            _print_metrics_table(
                f"Single head: N={N}, dtype={dtype}",
                ("BitResidual", metrics_br),
                ("TurboQuant", metrics_tq),
            )
            print(f"  Storage/row: BitResidual={BITRESIDUAL_ROW_BYTES}, TurboQuant={tq_row}")

            ok = metrics_br["cosine_similarity"] > 0.90 and metrics_tq["cosine_similarity"] > 0.90
            if ok:
                _pass(name)
            else:
                _fail(name, f"BR cos={metrics_br['cosine_similarity']:.4f}, TQ cos={metrics_tq['cosine_similarity']:.4f}")


def test_multi_head_comparison() -> None:
    """Compare for [T, H, D] (typical KV cache layout)."""
    for dtype in [torch.float16, torch.bfloat16]:
        name = f"multi_head/{dtype}"
        T, H = 32, 8
        device = torch.device("cpu")
        torch.manual_seed(SEED)
        R = _make_rotation(device)
        x = torch.randn(T, H, D, dtype=dtype, device=device)

        packed_br = bit_residual_quantize(x, R, head_size=D)
        x_hat_br = bit_residual_dequantize(packed_br, R, head_size=D, dtype=dtype)
        metrics_br = compute_metrics(x, x_hat_br)

        x_hat_tq = _turboquant_4bit_reference(x)["reconstructed"]
        metrics_tq = compute_metrics(x, x_hat_tq)

        tq_row = turboquant_slab_row_bytes(D, bits=4)
        _print_metrics_table(
            f"Multi-head: T={T}, H={H}, dtype={dtype}",
            ("BitResidual", metrics_br),
            ("TurboQuant", metrics_tq),
        )
        print(f"  Storage/row: BitResidual={BITRESIDUAL_ROW_BYTES}, TurboQuant={tq_row}")

        ok = metrics_br["cosine_similarity"] > 0.90 and metrics_tq["cosine_similarity"] > 0.90
        if ok:
            _pass(name)
        else:
            _fail(name, f"BR cos={metrics_br['cosine_similarity']:.4f}, TQ cos={metrics_tq['cosine_similarity']:.4f}")


def test_per_vector_error_distribution() -> None:
    """Show per-vector error distribution for both methods."""
    name = "per_vector_error_distribution"
    N = 256
    torch.manual_seed(SEED)
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(N, D, dtype=torch.float32)

    packed_br = bit_residual_quantize(x, R, head_size=D)
    x_hat_br = bit_residual_dequantize(packed_br, R, head_size=D, dtype=torch.float32)

    x_f16 = x.to(torch.float16)
    x_hat_tq = _turboquant_4bit_reference(x_f16)["reconstructed"].float()

    orig_norm = x.norm(p=2, dim=-1)
    br_rel = (x - x_hat_br).norm(p=2, dim=-1) / (orig_norm + 1e-10)
    tq_rel = (x - x_hat_tq).norm(p=2, dim=-1) / (orig_norm + 1e-10)

    print(f"\nPer-vector relative error distribution (N={N}):")
    print(f"{'Method':<15} {'mean':<10} {'p50':<10} {'p90':<10} {'p99':<10} {'max':<10}")
    print(f"{'-'*55}")
    print(f"{'BitResidual':<15} {br_rel.mean():<10.4f} {br_rel.median():<10.4f} "
          f"{br_rel.quantile(0.9):<10.4f} {br_rel.quantile(0.99):<10.4f} {br_rel.max():<10.4f}")
    print(f"{'TurboQuant':<15} {tq_rel.mean():<10.4f} {tq_rel.median():<10.4f} "
          f"{tq_rel.quantile(0.9):<10.4f} {tq_rel.quantile(0.99):<10.4f} {tq_rel.max():<10.4f}")

    br_cos = torch.nn.functional.cosine_similarity(x, x_hat_br, dim=-1)
    tq_cos = torch.nn.functional.cosine_similarity(x, x_hat_tq, dim=-1)

    print(f"\nPer-vector cosine similarity distribution:")
    print(f"{'Method':<15} {'mean':<10} {'p50':<10} {'p90':<10} {'p99':<10} {'min':<10}")
    print(f"{'-'*55}")
    print(f"{'BitResidual':<15} {br_cos.mean():<10.4f} {br_cos.median():<10.4f} "
          f"{br_cos.quantile(0.9):<10.4f} {br_cos.quantile(0.99):<10.4f} {br_cos.min():<10.4f}")
    print(f"{'TurboQuant':<15} {tq_cos.mean():<10.4f} {tq_cos.median():<10.4f} "
          f"{tq_cos.quantile(0.9):<10.4f} {tq_cos.quantile(0.99):<10.4f} {tq_cos.min():<10.4f}")

    _pass(name)  # informational, no strict assertion


def test_different_input_scales() -> None:
    """Test how both methods handle different input magnitude scales."""
    for scale in [0.01, 0.05, 0.2, 1.0]:
        name = f"input_scale/{scale}"
        N = 128
        torch.manual_seed(SEED)
        R = _make_rotation(torch.device("cpu"))
        x = torch.randn(N, D, dtype=torch.float32) * scale

        packed_br = bit_residual_quantize(x, R, head_size=D)
        x_hat_br = bit_residual_dequantize(packed_br, R, head_size=D, dtype=torch.float32)
        metrics_br = compute_metrics(x, x_hat_br)

        x_f16 = x.to(torch.float16)
        x_hat_tq = _turboquant_4bit_reference(x_f16)["reconstructed"].float()
        metrics_tq = compute_metrics(x, x_hat_tq)

        _print_metrics_table(
            f"Scale={scale:.3f}, N={N}",
            ("BitResidual", metrics_br),
            ("TurboQuant", metrics_tq),
        )

        if metrics_br["relative_error"] < 0.35:
            _pass(name)
        else:
            _fail(name, f"rel_err={metrics_br['relative_error']:.4f}")


# ---------------------------------------------------------------------------
# Real data tests
# ---------------------------------------------------------------------------

def _load_real_data() -> dict | None:
    """Load real KV cache data from smoke dump."""
    if not REAL_SMOKE_DUMP_PATH.exists():
        return None
    try:
        dump = torch.load(REAL_SMOKE_DUMP_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        dump = torch.load(REAL_SMOKE_DUMP_PATH, map_location="cpu")
    return dump


def test_real_data_roundtrip_quality() -> None:
    """BitResidual roundtrip on real data should achieve cos_sim > 0.90."""
    name = "real_data_roundtrip_quality"
    dump = _load_real_data()
    if dump is None:
        _skip(name, "Real smoke dump not found")
        return

    R = _make_rotation(torch.device("cpu"))
    all_ok = True
    all_metrics = []
    for s in dump["samples"]:
        for kv in ["key_original", "value_original"]:
            x = s[kv].to(torch.float32)
            packed = bit_residual_quantize(x, R, head_size=D)
            x_hat = bit_residual_dequantize(packed, R, head_size=D, dtype=torch.float32)
            metrics = compute_metrics(x, x_hat)
            all_metrics.append(metrics)
            if metrics["cosine_similarity"] <= 0.90:
                all_ok = False
                print(f"    FAIL: {kv} cos_sim={metrics['cosine_similarity']:.4f}, shape={x.shape}")

    avg_cos = sum(m["cosine_similarity"] for m in all_metrics) / len(all_metrics)
    avg_mse = sum(m["mse"] for m in all_metrics) / len(all_metrics)
    print(f"\n  Real data roundtrip ({len(all_metrics)} tensors): "
          f"avg_cos_sim={avg_cos:.6f}, avg_mse={avg_mse:.6f}")

    if all_ok:
        _pass(name)
    else:
        _fail(name, "some vectors had cos_sim <= 0.90")


def test_real_data_vs_turboquant() -> None:
    """Compare BitResidual vs TurboQuant on real KV data."""
    dump = _load_real_data()
    if dump is None:
        _skip("real_data_vs_turboquant", "Real smoke dump not found")
        return

    for kv_type in ["key_original", "value_original"]:
        name = f"real_data_vs_tq/{kv_type}"
        R = _make_rotation(torch.device("cpu"))

        all_br_metrics = []
        all_tq_metrics = []

        for s in dump["samples"]:
            x_orig = s[kv_type].to(torch.float32)

            packed_br = bit_residual_quantize(x_orig, R, head_size=D)
            x_hat_br = bit_residual_dequantize(packed_br, R, head_size=D, dtype=torch.float32)
            metrics_br = compute_metrics(x_orig, x_hat_br)

            x_bf16 = s[kv_type]
            x_hat_tq = _turboquant_4bit_reference(x_bf16)["reconstructed"].float()
            metrics_tq = compute_metrics(x_orig, x_hat_tq)

            all_br_metrics.append(metrics_br)
            all_tq_metrics.append(metrics_tq)

        br_avg = {k: sum(m[k] for m in all_br_metrics) / len(all_br_metrics)
                  for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}
        tq_avg = {k: sum(m[k] for m in all_tq_metrics) / len(all_tq_metrics)
                  for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}

        tq_row = turboquant_slab_row_bytes(D, bits=4)
        _print_metrics_table(
            f"Real data: {kv_type}, {len(dump['samples'])} samples",
            ("BitResidual", br_avg),
            ("TurboQuant", tq_avg),
        )
        print(f"  Storage/row: BitResidual={BITRESIDUAL_ROW_BYTES}, TurboQuant={tq_row}")

        ok = br_avg["cosine_similarity"] > 0.90 and tq_avg["cosine_similarity"] > 0.90
        if ok:
            _pass(name)
        else:
            _fail(name, f"BR cos={br_avg['cosine_similarity']:.4f}, TQ cos={tq_avg['cosine_similarity']:.4f}")


def test_real_data_per_vector_error_distribution() -> None:
    """Per-vector error distribution on real KV data for both methods."""
    name = "real_data_per_vector_error_dist"
    dump = _load_real_data()
    if dump is None:
        _skip(name, "Real smoke dump not found")
        return

    R = _make_rotation(torch.device("cpu"))
    all_br_rel, all_tq_rel, all_br_cos, all_tq_cos = [], [], [], []

    for s in dump["samples"]:
        for kv_type in ["key_original", "value_original"]:
            x = s[kv_type].to(torch.float32)
            orig_norm = x.norm(p=2, dim=-1)

            packed_br = bit_residual_quantize(x, R, head_size=D)
            x_hat_br = bit_residual_dequantize(packed_br, R, head_size=D, dtype=torch.float32)
            br_rel = (x - x_hat_br).norm(p=2, dim=-1) / (orig_norm + 1e-10)
            br_cos = torch.nn.functional.cosine_similarity(x, x_hat_br, dim=-1)

            x_bf16 = s[kv_type]
            x_hat_tq = _turboquant_4bit_reference(x_bf16)["reconstructed"].float()
            tq_rel = (x - x_hat_tq).norm(p=2, dim=-1) / (orig_norm + 1e-10)
            tq_cos = torch.nn.functional.cosine_similarity(x, x_hat_tq, dim=-1)

            all_br_rel.append(br_rel)
            all_tq_rel.append(tq_rel)
            all_br_cos.append(br_cos)
            all_tq_cos.append(tq_cos)

    br_rel_all = torch.cat(all_br_rel)
    tq_rel_all = torch.cat(all_tq_rel)
    br_cos_all = torch.cat(all_br_cos)
    tq_cos_all = torch.cat(all_tq_cos)

    print(f"\nReal data per-vector error distribution ({br_rel_all.numel()} vectors):")
    print(f"{'Method':<15} {'mean':<10} {'p50':<10} {'p90':<10} {'p99':<10} {'max':<10}")
    print(f"{'-'*55}")
    print(f"{'BitResidual':<15} {br_rel_all.mean():<10.4f} {br_rel_all.median():<10.4f} "
          f"{br_rel_all.quantile(0.9):<10.4f} {br_rel_all.quantile(0.99):<10.4f} {br_rel_all.max():<10.4f}")
    print(f"{'TurboQuant':<15} {tq_rel_all.mean():<10.4f} {tq_rel_all.median():<10.4f} "
          f"{tq_rel_all.quantile(0.9):<10.4f} {tq_rel_all.quantile(0.99):<10.4f} {tq_rel_all.max():<10.4f}")

    print(f"\nReal data per-vector cosine similarity distribution:")
    print(f"{'Method':<15} {'mean':<10} {'p50':<10} {'p90':<10} {'p99':<10} {'min':<10}")
    print(f"{'-'*55}")
    print(f"{'BitResidual':<15} {br_cos_all.mean():<10.4f} {br_cos_all.median():<10.4f} "
          f"{br_cos_all.quantile(0.9):<10.4f} {br_cos_all.quantile(0.99):<10.4f} {br_cos_all.min():<10.4f}")
    print(f"{'TurboQuant':<15} {tq_cos_all.mean():<10.4f} {tq_cos_all.median():<10.4f} "
          f"{tq_cos_all.quantile(0.9):<10.4f} {tq_cos_all.quantile(0.99):<10.4f} {tq_cos_all.min():<10.4f}")

    ok = br_cos_all.mean() > 0.90 and tq_cos_all.mean() > 0.90
    if ok:
        _pass(name)
    else:
        _fail(name, f"BR mean_cos={br_cos_all.mean():.4f}, TQ mean_cos={tq_cos_all.mean():.4f}")


# ---------------------------------------------------------------------------
# Helpers for fused 4-bit NPU op tests
# ---------------------------------------------------------------------------

KV_BLOCK_SIZE = 128


@contextmanager
def _patched_turboquant_env(*args: Any, **kwargs: Any):
    with patch.dict(*args, **kwargs):
        refresh_turboquant_env_cache()
        try:
            yield
        finally:
            refresh_turboquant_env_cache()


def _gather_4bit_slab_rows_for_test(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    head_size: int,
    bits: int,
) -> torch.Tensor:
    row_w = turboquant_slab_row_bytes(head_size, bits=bits)
    block_size = cache.shape[-1] // row_w
    valid = slot_mapping.to(device="cpu", dtype=torch.int64) >= 0
    slot_cpu = slot_mapping.to(device="cpu", dtype=torch.int64)[valid]
    block_idx_cpu = torch.div(slot_cpu, block_size, rounding_mode="floor")
    block_idx = block_idx_cpu.to(cache.device)
    block_off = (slot_cpu - block_idx_cpu * block_size).to(cache.device)
    head_idx = torch.arange(cache.shape[1], device=cache.device, dtype=torch.int64)
    rows = _turboquant_slab_group4_to_row_format(
        cache, head_size=head_size, bits=bits
    )
    return rows[block_idx[:, None], block_off[:, None], head_idx[None, :], :]


def _formula_dequantize_4bit_for_test(
    packed_rows: torch.Tensor,
    rotation: torch.Tensor,
    *,
    head_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize 4-bit packed rows using the cubic formula (no codebook lookup)."""
    row_w = turboquant_slab_row_bytes(head_size, bits=4)
    packed_u8 = packed_rows.view(torch.uint8)[..., :row_w]
    indices = unpack_uint4(packed_u8[..., : row_w - 2], head_size).to(torch.float32)
    norm_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
    norms = (
        packed_u8[..., row_w - 2 : row_w]
        .contiguous()
        .view(norm_dtype)
        .view(*packed_u8.shape[:-1])
    )
    delta = indices - 7.5
    y_hat = 0.0001926 * (delta**3) + 0.020799 * delta
    out = y_hat @ rotation.to(device=packed_rows.device, dtype=torch.float32)
    return (out * norms.to(torch.float32).unsqueeze(-1)).to(dtype=dtype)


# ---------------------------------------------------------------------------
# Fused 4-bit NPU op tests
# ---------------------------------------------------------------------------

def test_fused_op_vs_bitresidual_random() -> None:
    """Compare BitResidual vs fused 4-bit NPU op on random data."""
    if not _NPU_AVAILABLE:
        for dtype in [torch.float16, torch.bfloat16]:
            _skip(f"fused_random/{dtype}", "NPU not available")
        return
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_4bit"):
        for dtype in [torch.float16, torch.bfloat16]:
            _skip(f"fused_random/{dtype}", "turboquant_pack_kv_for_cache_4bit op not available")
        return

    for dtype in [torch.float16, torch.bfloat16]:
        name = f"fused_random/{dtype}"
        device = torch.device("npu:0")
        T, H = 7, 8
        bits = 4
        row_w = turboquant_slab_row_bytes(D, bits=bits)
        BS = KV_BLOCK_SIZE
        B = 2

        torch.manual_seed(SEED)
        key = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
        value = torch.randn(T, H, D, dtype=dtype, device=device).contiguous()
        slot_mapping = torch.arange(T, dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
        cb_k, rot_t_k = _turboquant_pack_tables(device, D, bits, dtype)
        rotation = _get_quantizer(D, bits, device).rotation.to(torch.float32)

        # Fused 4-bit NPU op
        key_cache = torch.zeros(B, H, BS * row_w, dtype=torch.uint8, device=device)
        value_cache = torch.zeros_like(key_cache)
        env = {
            "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
            "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
            "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
        }
        with _patched_turboquant_env(os.environ, env, clear=False):
            torch.ops._C_ascend.turboquant_pack_kv_for_cache_4bit(
                key.contiguous(), value.contiguous(), slot_mapping, query_start_loc,
                cb_k, rot_t_k, key_cache, value_cache, 1, BS,
            )
            torch.npu.synchronize()

        key_fused_packed = _gather_4bit_slab_rows_for_test(key_cache, slot_mapping, head_size=D, bits=bits)
        value_fused_packed = _gather_4bit_slab_rows_for_test(value_cache, slot_mapping, head_size=D, bits=bits)
        key_fused_deq = _formula_dequantize_4bit_for_test(key_fused_packed, rotation, head_size=D, dtype=dtype)
        value_fused_deq = _formula_dequantize_4bit_for_test(value_fused_packed, rotation, head_size=D, dtype=dtype)

        # BitResidual (CPU, with rotation — mandatory)
        R_br = _make_rotation(torch.device("cpu"))
        key_br_deq = bit_residual_dequantize(bit_residual_quantize(key.cpu(), R_br, head_size=D), R_br, head_size=D, dtype=dtype)
        value_br_deq = bit_residual_dequantize(bit_residual_quantize(value.cpu(), R_br, head_size=D), R_br, head_size=D, dtype=dtype)

        # TurboQuant Python ref
        key_tq_deq = turboquant_dequantize_from_packed_bytes(
            turboquant_quantize_to_packed_bytes(key, bits=bits), head_size=D, dtype=dtype, bits=bits)
        value_tq_deq = turboquant_dequantize_from_packed_bytes(
            turboquant_quantize_to_packed_bytes(value, bits=bits), head_size=D, dtype=dtype, bits=bits)

        key_f32 = key.cpu().float()
        value_f32 = value.cpu().float()

        m_br_key = compute_metrics(key_f32, key_br_deq.float())
        m_br_val = compute_metrics(value_f32, value_br_deq.float())
        m_tq_key = compute_metrics(key_f32, key_tq_deq.cpu().float())
        m_tq_val = compute_metrics(value_f32, value_tq_deq.cpu().float())
        m_fused_key = compute_metrics(key_f32, key_fused_deq.cpu().float())
        m_fused_val = compute_metrics(value_f32, value_fused_deq.cpu().float())

        tq_row = turboquant_slab_row_bytes(D, bits=4)
        _print_metrics_table(
            f"Random data: key, dtype={dtype}, T={T}, H={H}",
            ("BitResidual", m_br_key), ("TQ-Python", m_tq_key), ("TQ-Fused", m_fused_key),
        )
        _print_metrics_table(
            f"Random data: value, dtype={dtype}, T={T}, H={H}",
            ("BitResidual", m_br_val), ("TQ-Python", m_tq_val), ("TQ-Fused", m_fused_val),
        )
        print(f"  Storage/row: BitResidual={BITRESIDUAL_ROW_BYTES}, TurboQuant={tq_row}")

        all_ok = True
        for m, label in [(m_br_key, "BR key"), (m_br_val, "BR val"),
                         (m_tq_key, "TQ key"), (m_tq_val, "TQ val"),
                         (m_fused_key, "Fused key"), (m_fused_val, "Fused val")]:
            if m["cosine_similarity"] <= 0.90:
                all_ok = False
                print(f"    FAIL: {label} cos_sim={m['cosine_similarity']:.4f}")

        if all_ok:
            _pass(name)
        else:
            _fail(name, "some methods had cos_sim <= 0.90")


def test_fused_op_vs_bitresidual_real_data() -> None:
    """Compare BitResidual vs fused 4-bit NPU op on real KV data."""
    dump = _load_real_data()
    if dump is None:
        _skip("fused_real_data/key", "Real smoke dump not found")
        _skip("fused_real_data/value", "Real smoke dump not found")
        return
    if not _NPU_AVAILABLE:
        for dtype in [torch.float16, torch.bfloat16]:
            _skip(f"fused_real_data/{dtype}", "NPU not available")
        return
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_4bit"):
        for dtype in [torch.float16, torch.bfloat16]:
            _skip(f"fused_real_data/{dtype}", "fused op not available")
        return

    for dtype in [torch.float16, torch.bfloat16]:
        name = f"fused_real_data/{dtype}"
        device = torch.device("npu:0")
        bits = int(dump["bits"])
        BS = int(dump["block_size"])
        row_w = turboquant_slab_row_bytes(D, bits=bits)
        cb_k, rot_t_k = _turboquant_pack_tables(device, D, bits, dtype)
        rotation = _get_quantizer(D, bits, device).rotation.to(torch.float32)
        R_br = _make_rotation(torch.device("cpu"))

        env = {
            "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
            "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
            "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
        }

        all_br_metrics = {"key": [], "value": []}
        all_tq_metrics = {"key": [], "value": []}
        all_fused_metrics = {"key": [], "value": []}

        with _patched_turboquant_env(os.environ, env, clear=False):
            for sample in dump["samples"]:
                key_orig = sample["key_original"].to(device=device, dtype=dtype).contiguous()
                value_orig = sample["value_original"].to(device=device, dtype=dtype).contiguous()
                slot_mapping = sample["slot_mapping"].to(device=device, dtype=torch.int32)
                T, H, _ = key_orig.shape
                query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
                B = int(slot_mapping.to(torch.int64).max().item()) // BS + 1

                key_cache = torch.zeros(B, H, BS * row_w, dtype=torch.uint8, device=device)
                value_cache = torch.zeros_like(key_cache)

                torch.ops._C_ascend.turboquant_pack_kv_for_cache_4bit(
                    key_orig, value_orig, slot_mapping, query_start_loc,
                    cb_k, rot_t_k, key_cache, value_cache, 1, BS,
                )
                torch.npu.synchronize()

                key_fused_packed = _gather_4bit_slab_rows_for_test(key_cache, slot_mapping, head_size=D, bits=bits)
                value_fused_packed = _gather_4bit_slab_rows_for_test(value_cache, slot_mapping, head_size=D, bits=bits)
                key_fused_deq = _formula_dequantize_4bit_for_test(key_fused_packed, rotation, head_size=D, dtype=dtype)
                value_fused_deq = _formula_dequantize_4bit_for_test(value_fused_packed, rotation, head_size=D, dtype=dtype)

                key_br_packed = bit_residual_quantize(key_orig.cpu(), R_br, head_size=D)
                value_br_packed = bit_residual_quantize(value_orig.cpu(), R_br, head_size=D)
                key_br_deq = bit_residual_dequantize(key_br_packed, R_br, head_size=D, dtype=dtype)
                value_br_deq = bit_residual_dequantize(value_br_packed, R_br, head_size=D, dtype=dtype)

                key_tq_deq = turboquant_dequantize_from_packed_bytes(
                    turboquant_quantize_to_packed_bytes(key_orig, bits=bits), head_size=D, dtype=dtype, bits=bits)
                value_tq_deq = turboquant_dequantize_from_packed_bytes(
                    turboquant_quantize_to_packed_bytes(value_orig, bits=bits), head_size=D, dtype=dtype, bits=bits)

                for kv_label, orig_f32, br_deq, tq_deq, fused_deq in [
                    ("key", key_orig.cpu().float(), key_br_deq.float(), key_tq_deq.cpu().float(), key_fused_deq.cpu().float()),
                    ("value", value_orig.cpu().float(), value_br_deq.float(), value_tq_deq.cpu().float(), value_fused_deq.cpu().float()),
                ]:
                    all_br_metrics[kv_label].append(compute_metrics(orig_f32, br_deq))
                    all_tq_metrics[kv_label].append(compute_metrics(orig_f32, tq_deq))
                    all_fused_metrics[kv_label].append(compute_metrics(orig_f32, fused_deq))

        tq_row = turboquant_slab_row_bytes(D, bits=4)
        all_ok = True
        for kv_label in ["key", "value"]:
            br_avg = {k: sum(m[k] for m in all_br_metrics[kv_label]) / len(all_br_metrics[kv_label])
                      for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}
            tq_avg = {k: sum(m[k] for m in all_tq_metrics[kv_label]) / len(all_tq_metrics[kv_label])
                      for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}
            fused_avg = {k: sum(m[k] for m in all_fused_metrics[kv_label]) / len(all_fused_metrics[kv_label])
                         for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}

            _print_metrics_table(
                f"Real data {kv_label}: {len(dump['samples'])} samples, dtype={dtype}",
                ("BitResidual", br_avg), ("TQ-Python", tq_avg), ("TQ-Fused", fused_avg),
            )
            print(f"  Storage/row: BitResidual={BITRESIDUAL_ROW_BYTES}, TurboQuant={tq_row}")

            for label, avg in [("BR", br_avg), ("TQ", tq_avg), ("Fused", fused_avg)]:
                if avg["cosine_similarity"] <= 0.90:
                    all_ok = False
                    print(f"    FAIL: {label} {kv_label} cos_sim={avg['cosine_similarity']:.4f}")

        if all_ok:
            _pass(name)
        else:
            _fail(name, "some methods had cos_sim <= 0.90")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "basic": [
        test_roundtrip_shape, test_zero_vector, test_packed_keys,
        test_sign_bits_range, test_q3_indices_range, test_per_vector_norms,
        test_rotation_balances_signs, test_multi_head_shapes,
        test_rotation_balances_skewed_data,
    ],
    "comparison": [
        test_single_head_comparison, test_multi_head_comparison,
        test_per_vector_error_distribution, test_different_input_scales,
    ],
    "real-data": [
        test_real_data_roundtrip_quality, test_real_data_vs_turboquant,
        test_real_data_per_vector_error_distribution,
    ],
    "fused": [
        test_fused_op_vs_bitresidual_random, test_fused_op_vs_bitresidual_real_data,
    ],
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BitResidual vs TurboQuant 4-bit comparison tool")
    parser.add_argument("--basic", action="store_true", help="Run basic correctness tests")
    parser.add_argument("--comparison", action="store_true", help="Run TQ comparison tests")
    parser.add_argument("--real-data", action="store_true", help="Run real data tests")
    parser.add_argument("--fused", action="store_true", help="Run fused NPU op tests")
    args = parser.parse_args()

    # If no specific category is selected, run all (except fused unless --fused)
    if not (args.basic or args.comparison or args.real_data or args.fused):
        categories = ["basic", "comparison", "real-data"]
    else:
        categories = [c for c in ["basic", "comparison", "real-data", "fused"]
                      if getattr(args, c.replace("-", "_"))]

    print(f"\n{'#'*60}")
    print(f"# BitResidual 4-bit vs TurboQuant 4-bit comparison tool")
    print(f"# D={D}, SEED={SEED}, categories={categories}")
    print(f"{'#'*60}\n")

    for cat in categories:
        print(f"\n--- [{cat}] ---")
        for test_fn in ALL_TESTS[cat]:
            test_fn()

    print(f"\n{'#'*60}")
    print(f"# Results: {_passed} passed, {_failed} failed, {_skipped} skipped")
    print(f"{'#'*60}\n")

    if _failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
