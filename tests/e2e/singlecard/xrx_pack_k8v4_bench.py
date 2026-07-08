"""Performance benchmark comparing three KV cache pack ops:

  1. bit_residual_pack_k8v4  — 1-bit sign + 7-bit residual (manual Mmad)
  2. turboquant_pack_kv_for_cache_4bit — 4-bit codebook (manual Mmad)
  3. turboquant_pack_kv_for_cache_v2_to_cache — 8-bit codebook (fused)

All three receive the same key/value/slot_mapping inputs so the comparison is
fair.  The benchmark measures wall-clock time per call (NPU synchronize inside
the loop) and reports throughput (tokens/s).

Usage:
    python tests/e2e/singlecard/xrx_pack_k8v4_bench.py
    # Adjust token count / heads / block_size via env vars:
    BR_BENCH_TOKENS=3200 BR_BENCH_HEADS=8 BR_BENCH_BS=128 python ...
    # Skip individual ops:
    BR_BENCH_SKIP=4bit,v2 python ...
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Set ASCEND_CUSTOM_OPP_PATH before CANN runtime init.
_CANN_OPP = os.path.join(
    str(REPO_ROOT), "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend"
)
if os.path.isdir(_CANN_OPP):
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        _CANN_OPP if not existing else f"{_CANN_OPP}:{existing}"
)

import torch

from vllm_ascend.utils import enable_custom_op

# ---------------------------------------------------------------------------
# Configurable parameters (env overrides)
# ---------------------------------------------------------------------------
D = 128
H = int(os.getenv("BR_BENCH_HEADS", "8"))
BS = int(os.getenv("BR_BENCH_BS", "128"))
TOKENS = int(os.getenv("BR_BENCH_TOKENS", "3200"))
NUM_REQS = int(os.getenv("BR_BENCH_NUM_REQS", "1"))
WARMUP_ITERS = int(os.getenv("BR_BENCH_WARMUP", "3"))
BENCH_ITERS = int(os.getenv("BR_BENCH_ITERS", "10"))
SEED = 42
SKIP = set(os.getenv("BR_BENCH_SKIP", "").split(",")) if os.getenv("BR_BENCH_SKIP") else set()

# Bit-residual layout constants
BR_GROUP_STRIDE = 288
BR_KEY_GROUP_ROWS = 2
BR_VALUE_GROUP_ROWS = 4

# TurboQuant 4-bit slab layout constants
TQ4_GROUP_ROWS = 4


@contextlib.contextmanager
def _patched_env(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _make_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 17)
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=gen, dtype=torch.float32))
    return q.to(device=device, dtype=dtype).contiguous()


def _bench_op(
    label: str,
    op_fn,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Run warmup + timed iterations, return {time_ms, tokens_per_sec}."""
    # Warmup
    for _ in range(warmup):
        op_fn()
        torch.npu.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        op_fn()
        torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_call = elapsed / iters * 1000
    tok_per_sec = TOKENS * iters / elapsed
    return {
        "label": label,
        "ms_per_call": ms_per_call,
        "tokens_per_sec": tok_per_sec,
        "iters": iters,
        "warmup": warmup,
    }


def _run_br_k8v4_bench(
    key: torch.Tensor,
    value: torch.Tensor,
    rotation_t: torch.Tensor,
    slots: torch.Tensor,
    qsl: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    num_blocks = int((TOKENS + BS - 1) // BS + 2)
    kc = torch.zeros(
        num_blocks, H, (BS // BR_KEY_GROUP_ROWS) * BR_GROUP_STRIDE,
        dtype=torch.uint8, device=device,
    )
    vc = torch.zeros(
        num_blocks, H, (BS // BR_VALUE_GROUP_ROWS) * BR_GROUP_STRIDE,
        dtype=torch.uint8, device=device,
    )

    def call():
        torch.ops._C_ascend.bit_residual_pack_k8v4(
            key, value, slots, qsl, rotation_t, kc, vc, NUM_REQS, BS
        )

    return _bench_op("bit_residual_pack_k8v4", call, warmup, iters)


def _run_tq4bit_bench(
    key: torch.Tensor,
    value: torch.Tensor,
    codebook: torch.Tensor,
    rotation_t: torch.Tensor,
    slots: torch.Tensor,
    qsl: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    from vllm_ascend.ops.turboquant_kv_cache import (
        turboquant_slab_row_bytes,
    )
    row_w = turboquant_slab_row_bytes(D, bits=4)
    num_blocks = int((TOKENS + BS - 1) // BS + 2)
    kc = torch.zeros(num_blocks, H, BS * row_w, dtype=torch.uint8, device=device)
    vc = torch.zeros_like(kc)

    def call():
        torch.ops._C_ascend.turboquant_pack_kv_for_cache_4bit(
            key, value, slots, qsl, codebook, rotation_t, kc, vc, NUM_REQS, BS
        )

    return _bench_op("turboquant_pack_kv_for_cache_4bit", call, warmup, iters)


def _run_tq_v2_to_cache_bench(
    key: torch.Tensor,
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    # v2_to_cache writes into a standard 8-bit cache layout:
    # [num_blocks, block_size, num_heads, slot_w]
    # slot_w = head_size*2 + 2  (8-bit packed indices + 2-byte norm)
    num_blocks = int((TOKENS + BS - 1) // BS + 2)
    slot_w_k = D * 2 + 2  # 8-bit: packed uint8 indices + 2-byte norm
    slot_w_v = D * 2 + 2
    kc = torch.zeros(
        num_blocks, BS, H, slot_w_k, dtype=torch.uint8, device=device,
    )
    vc = torch.zeros(
        num_blocks, BS, H, slot_w_v, dtype=torch.uint8, device=device,
    )
    slots = torch.arange(0, TOKENS, dtype=torch.int32, device=device)

    def call():
        torch.ops._C_ascend.turboquant_pack_kv_for_cache_v2_to_cache(
            key, value, slots, kc, vc, slot_w_k, slot_w_v
        )

    return _bench_op("turboquant_pack_kv_for_cache_v2_to_cache", call, warmup, iters)


def main() -> None:
    enable_custom_op()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    device = torch.device("npu:0")
    dtype_str = os.getenv("BR_BENCH_DTYPE", "bf16")
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print("=" * 70)
    print("KV Cache Pack Op Performance Benchmark")
    print("=" * 70)
    print(f"  dtype       : {dtype}")
    print(f"  head_dim (D): {D}")
    print(f"  num_heads(H): {H}")
    print(f"  block_size  : {BS}")
    print(f"  tokens (T)  : {TOKENS}")
    print(f"  num_reqs    : {NUM_REQS}")
    print(f"  warmup      : {WARMUP_ITERS}")
    print(f"  bench_iters : {BENCH_ITERS}")
    print(f"  skip        : {SKIP or '(none)'}")
    print()

    # Generate input data
    torch.manual_seed(SEED)
    key = torch.randn(TOKENS, H, D, dtype=dtype, device=device).contiguous()
    value = torch.randn(TOKENS, H, D, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(0, TOKENS, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, TOKENS], dtype=torch.int32, device=device)

    results: list[dict[str, float]] = []

    # --- bit_residual_pack_k8v4 ---
    if "k8v4" not in SKIP:
        print(f"[1/3] Benchmarking bit_residual_pack_k8v4 ...")
        r = _run_br_k8v4_bench(
            key, value, rotation_t, slots, qsl, device, dtype,
            WARMUP_ITERS, BENCH_ITERS,
        )
        results.append(r)

    # --- turboquant_pack_kv_for_cache_4bit ---
    if "4bit" not in SKIP:
        # Need to set up TQ env + register pack tables + get codebook
        import io
        from vllm_ascend.ops.turboquant_kv_cache import (
            _turboquant_pack_tables,
            ensure_turboquant_pack_tables_registered,
            refresh_turboquant_env_cache,
            turboquant_slab_row_bytes,
        )

        env = {
            "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
            "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
            "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "1",
            "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "1",
        }
        # Suppress verbose TurboQuantMSE codebook print output
        with _patched_env(env), contextlib.redirect_stdout(io.StringIO()):
            refresh_turboquant_env_cache()
            cb, rot_tq = _turboquant_pack_tables(device, D, 4, dtype)
            ensure_turboquant_pack_tables_registered(device, D, 4)

        print(f"[2/3] Benchmarking turboquant_pack_kv_for_cache_4bit ...")
        r = _run_tq4bit_bench(
            key, value, cb, rot_tq, slots, qsl, device, dtype,
            WARMUP_ITERS, BENCH_ITERS,
        )
        results.append(r)

    # --- turboquant_pack_kv_for_cache_v2_to_cache ---
    if "v2" not in SKIP:
        import io
        from vllm_ascend.ops.turboquant_kv_cache import (
            ensure_turboquant_pack_tables_registered,
            refresh_turboquant_env_cache,
        )

        env = {
            "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v2",
            "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "1",
        }
        # Suppress verbose TurboQuantMSE codebook print output
        with _patched_env(env), contextlib.redirect_stdout(io.StringIO()):
            refresh_turboquant_env_cache()
            ensure_turboquant_pack_tables_registered(device, D, 8)

            print(f"[3/3] Benchmarking turboquant_pack_kv_for_cache_v2_to_cache ...")
            r = _run_tq_v2_to_cache_bench(
                key, value, device, dtype,
                WARMUP_ITERS, BENCH_ITERS,
            )
            results.append(r)

    # --- Report ---
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  {'Op':<40s} {'ms/call':>10s} {'tok/s':>12s}")
    print(f"  {'-'*40} {'-'*10} {'-'*12}")
    for r in results:
        print(
            f"  {r['label']:<40s} "
            f"{r['ms_per_call']:10.2f} "
            f"{r['tokens_per_sec']:12.1f}"
        )
    print()

    # Relative comparison (if ≥2 results)
    if len(results) >= 2:
        baseline = results[0]
        print("=" * 70)
        print(f"Relative to {baseline['label']}")
        print("=" * 70)
        for r in results[1:]:
            speedup = baseline["ms_per_call"] / r["ms_per_call"]
            print(
                f"  {r['label']:<40s} "
                f"speedup={speedup:.2f}x "
                f"(ms: {r['ms_per_call']:.2f} vs {baseline['ms_per_call']:.2f})"
            )
        print()


if __name__ == "__main__":
    main()
