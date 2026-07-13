"""
Unit tests for Cubic-polynomial 4-bit quantization.

Tests the cubic_quantize/cubic_dequantize roundtrip, index range,
multi-head shapes, and comparison against TurboQuant 4-bit.

Usage:
    python3 tests/ut/ops/test_cubic_quant.py
    python3 tests/ut/ops/test_cubic_quant.py --comparison  # include TQ comparison
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.ops.cubic_quant import (
    CUBIC_QUANT_ROW_BYTES,
    cubic_dequantize,
    cubic_quantize,
    compute_metrics,
    generate_rotation_matrix,
)
from vllm_ascend.ops.turboquant_kv_cache import (
    _get_quantizer,
    turboquant_dequantize_from_packed_bytes,
    turboquant_quantize_to_packed_bytes,
    turboquant_slab_row_bytes,
)

D = 128
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
    """Generate the same rotation matrix as TurboQuant (Haar orthogonal, seed=42)."""
    return _get_quantizer(D, bits=4, device=device).rotation.to(device=device, dtype=torch.float32)


def _turboquant_4bit_reference(x: torch.Tensor) -> dict:
    """Quantize + dequantize with TurboQuant 4-bit MSE (CPU)."""
    packed = turboquant_quantize_to_packed_bytes(x, bits=4)
    reconstructed = turboquant_dequantize_from_packed_bytes(
        packed, head_size=x.shape[-1], dtype=x.dtype, bits=4
    )
    return {"packed": packed, "reconstructed": reconstructed}


def _print_metrics_table(title: str, *rows: tuple[str, dict]) -> None:
    """Print a comparison table of metrics from compute_metrics() dicts."""
    metric_keys = ["mse", "max_error", "relative_error", "cosine_similarity"]
    col_width = 18
    print(f"\n{'=' * 60}")
    print(f"{title}")
    header = f"{'Metric':<20}" + "".join(f"{label:<{col_width}}" for label, _ in rows)
    print(header)
    print(f"{'-' * 60}")
    for mk in metric_keys:
        line = f"{mk:<20}" + "".join(f"{m[mk]:<{col_width}.6f}" for _, m in rows)
        print(line)
    print(f"{'=' * 60}")


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
        packed = cubic_quantize(x, R, head_size=D)
        x_hat = cubic_dequantize(packed, R, head_size=D, dtype=dtype)
        metrics = compute_metrics(x, x_hat)
        _print_metrics_table(f"Roundtrip shape, dtype={dtype}", ("CubicQuant", metrics))
        if x_hat.shape == x.shape:
            _pass(name)
        else:
            _fail(name, f"shape mismatch: {x_hat.shape} != {x.shape}")


def test_zero_vector() -> None:
    """Zero vector should reconstruct with near-zero error."""
    name = "zero_vector"
    R = _make_rotation(torch.device("cpu"))
    x = torch.zeros(5, D, dtype=torch.float32)
    packed = cubic_quantize(x, R, head_size=D)
    x_hat = cubic_dequantize(packed, R, head_size=D, dtype=torch.float32)
    metrics = compute_metrics(x, x_hat)
    _print_metrics_table("Zero vector", ("CubicQuant", metrics))
    max_err = metrics["max_error"]
    if max_err < 0.1:
        _pass(name)
    else:
        _fail(name, f"max_error={max_err}")


def test_packed_keys() -> None:
    """Quantize should return expected dict keys."""
    name = "packed_keys"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float16)
    packed = cubic_quantize(x, R, head_size=D)
    expected = ["norms", "indices"]
    missing = [k for k in expected if k not in packed]
    if not missing:
        _pass(name)
    else:
        _fail(name, f"missing keys: {missing}")


def test_indices_shape() -> None:
    """Packed indices should be D//2 bytes per vector."""
    name = "indices_shape"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float16)
    packed = cubic_quantize(x, R, head_size=D)
    ok = packed["indices"].dtype == torch.uint8 and packed["indices"].shape[-1] == D // 2
    if ok:
        _pass(name)
    else:
        _fail(name, f"dtype={packed['indices'].dtype}, shape[-1]={packed['indices'].shape[-1]}")


def test_per_vector_norms() -> None:
    """Norms should match actual L2 norm of each vector."""
    name = "per_vector_norms"
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(5, D, dtype=torch.float32)
    packed = cubic_quantize(x, R, head_size=D)
    norms_hat = packed["norms"].float().squeeze(-1)
    norms_true = x.norm(p=2, dim=-1)
    diff = (norms_hat - norms_true).abs().max().item()
    if diff < 0.01:
        _pass(name)
    else:
        _fail(name, f"norm diff={diff}")


def test_rotation_balances_indices() -> None:
    """After random rotation, indices should span a reasonable range."""
    name = "rotation_balances_indices"
    R = _make_rotation(torch.device("cpu"))
    torch.manual_seed(SEED)
    x = torch.randn(1000, D, dtype=torch.float32)
    packed = cubic_quantize(x, R, head_size=D)
    # Unpack indices to check distribution
    from vllm_ascend.ops.cubic_quant import _unpack_uint4
    indices = _unpack_uint4(packed["indices"], D)
    unique_vals = indices.unique()
    print(f"\n  Unique index values: {len(unique_vals)} (range 0..15)")
    if len(unique_vals) >= 8:
        _pass(name)
    else:
        _fail(name, f"only {len(unique_vals)} unique values")


def test_multi_head_shapes() -> None:
    """Should work with [N, D] and [N, H, D] shapes."""
    for shape in [(32, D), (8, 4, D)]:
        name = f"multi_head_shapes/{shape}"
        R = _make_rotation(torch.device("cpu"))
        x = torch.randn(*shape, dtype=torch.float16)
        packed = cubic_quantize(x, R, head_size=D)
        x_hat = cubic_dequantize(packed, R, head_size=D, dtype=torch.float16)
        metrics = compute_metrics(x, x_hat)
        _print_metrics_table(f"Multi-head shape={shape}", ("CubicQuant", metrics))
        ok = x_hat.shape == x.shape and metrics["cosine_similarity"] > 0.90
        if ok:
            _pass(name)
        else:
            _fail(name, f"shape={x_hat.shape}, cos_sim={metrics['cosine_similarity']:.4f}")


# ---------------------------------------------------------------------------
# Comparison: CubicQuant vs TurboQuant 4-bit
# ---------------------------------------------------------------------------

def test_single_head_comparison() -> None:
    """Compare CubicQuant vs TurboQuant for [N, D] tensors."""
    for dtype in [torch.float16, torch.bfloat16]:
        for N in [16, 128, 512]:
            name = f"single_head/N={N}/{dtype}"
            device = torch.device("cpu")
            torch.manual_seed(SEED)
            R = _make_rotation(device)
            x = torch.randn(N, D, dtype=dtype, device=device)

            packed_cq = cubic_quantize(x, R, head_size=D)
            x_hat_cq = cubic_dequantize(packed_cq, R, head_size=D, dtype=dtype)
            metrics_cq = compute_metrics(x, x_hat_cq)

            x_hat_tq = _turboquant_4bit_reference(x)["reconstructed"]
            metrics_tq = compute_metrics(x, x_hat_tq)

            tq_row = turboquant_slab_row_bytes(D, bits=4)
            _print_metrics_table(
                f"Single head: N={N}, dtype={dtype}",
                ("CubicQuant", metrics_cq),
                ("TurboQuant", metrics_tq),
            )
            print(f"  Storage/row: CubicQuant={CUBIC_QUANT_ROW_BYTES}, TurboQuant={tq_row}")

            ok = metrics_cq["cosine_similarity"] > 0.90 and metrics_tq["cosine_similarity"] > 0.90
            if ok:
                _pass(name)
            else:
                _fail(name, f"CQ cos={metrics_cq['cosine_similarity']:.4f}, TQ cos={metrics_tq['cosine_similarity']:.4f}")


def test_per_vector_error_distribution() -> None:
    """Show per-vector error distribution for both methods."""
    name = "per_vector_error_distribution"
    N = 256
    torch.manual_seed(SEED)
    R = _make_rotation(torch.device("cpu"))
    x = torch.randn(N, D, dtype=torch.float32)

    packed_cq = cubic_quantize(x, R, head_size=D)
    x_hat_cq = cubic_dequantize(packed_cq, R, head_size=D, dtype=torch.float32)

    x_f16 = x.to(torch.float16)
    x_hat_tq = _turboquant_4bit_reference(x_f16)["reconstructed"].float()

    orig_norm = x.norm(p=2, dim=-1)
    cq_rel = (x - x_hat_cq).norm(p=2, dim=-1) / (orig_norm + 1e-10)
    tq_rel = (x - x_hat_tq).norm(p=2, dim=-1) / (orig_norm + 1e-10)

    print(f"\nPer-vector relative error distribution (N={N}):")
    print(f"{'Method':<15} {'mean':<10} {'p50':<10} {'p90':<10} {'p99':<10} {'max':<10}")
    print(f"{'-' * 55}")
    print(f"{'CubicQuant':<15} {cq_rel.mean():<10.4f} {cq_rel.median():<10.4f} "
          f"{cq_rel.quantile(0.9):<10.4f} {cq_rel.quantile(0.99):<10.4f} {cq_rel.max():<10.4f}")
    print(f"{'TurboQuant':<15} {tq_rel.mean():<10.4f} {tq_rel.median():<10.4f} "
          f"{tq_rel.quantile(0.9):<10.4f} {tq_rel.quantile(0.99):<10.4f} {tq_rel.max():<10.4f}")

    _pass(name)  # informational


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
    """CubicQuant roundtrip on real data should achieve cos_sim > 0.90."""
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
            packed = cubic_quantize(x, R, head_size=D)
            x_hat = cubic_dequantize(packed, R, head_size=D, dtype=torch.float32)
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
    """Compare CubicQuant vs TurboQuant on real KV data."""
    dump = _load_real_data()
    if dump is None:
        _skip("real_data_vs_tq", "Real smoke dump not found")
        return

    for kv_type in ["key_original", "value_original"]:
        name = f"real_data_vs_tq/{kv_type}"
        R = _make_rotation(torch.device("cpu"))

        all_cq_metrics = []
        all_tq_metrics = []

        for s in dump["samples"]:
            x_orig = s[kv_type].to(torch.float32)

            packed_cq = cubic_quantize(x_orig, R, head_size=D)
            x_hat_cq = cubic_dequantize(packed_cq, R, head_size=D, dtype=torch.float32)
            metrics_cq = compute_metrics(x_orig, x_hat_cq)

            x_bf16 = s[kv_type]
            x_hat_tq = _turboquant_4bit_reference(x_bf16)["reconstructed"].float()
            metrics_tq = compute_metrics(x_orig, x_hat_tq)

            all_cq_metrics.append(metrics_cq)
            all_tq_metrics.append(metrics_tq)

        cq_avg = {k: sum(m[k] for m in all_cq_metrics) / len(all_cq_metrics)
                  for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}
        tq_avg = {k: sum(m[k] for m in all_tq_metrics) / len(all_tq_metrics)
                  for k in ["mse", "max_error", "relative_error", "cosine_similarity"]}

        tq_row = turboquant_slab_row_bytes(D, bits=4)
        _print_metrics_table(
            f"Real data: {kv_type}, {len(dump['samples'])} samples",
            ("CubicQuant", cq_avg),
            ("TurboQuant", tq_avg),
        )
        print(f"  Storage/row: CubicQuant={CUBIC_QUANT_ROW_BYTES}, TurboQuant={tq_row}")

        ok = cq_avg["cosine_similarity"] > 0.90 and tq_avg["cosine_similarity"] > 0.90
        if ok:
            _pass(name)
        else:
            _fail(name, f"CQ cos={cq_avg['cosine_similarity']:.4f}, TQ cos={tq_avg['cosine_similarity']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "basic": [
        test_roundtrip_shape, test_zero_vector, test_packed_keys,
        test_indices_shape, test_per_vector_norms,
        test_rotation_balances_indices, test_multi_head_shapes,
    ],
    "comparison": [
        test_single_head_comparison, test_per_vector_error_distribution,
    ],
    "real-data": [
        test_real_data_roundtrip_quality, test_real_data_vs_turboquant,
    ],
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CubicQuant 4-bit unit tests")
    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    args = parser.parse_args()

    if not (args.basic or args.comparison or args.real_data):
        categories = ["basic", "comparison", "real-data"]
    else:
        categories = [c for c in ["basic", "comparison", "real-data"]
                      if getattr(args, c.replace("-", "_"))]

    print(f"\n{'#' * 60}")
    print(f"# CubicQuant 4-bit unit tests")
    print(f"# D={D}, SEED={SEED}, categories={categories}")
    print(f"{'#'*60}\n")

    for cat in categories:
        print(f"\n--- [{cat}] ---")
        for test_fn in ALL_TESTS[cat]:
            test_fn()

    print(f"\n{'#' * 60}")
    print(f"# Results: {_passed} passed, {_failed} failed, {_skipped} skipped")
    print(f"{'#' * 60}\n")

    if _failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
