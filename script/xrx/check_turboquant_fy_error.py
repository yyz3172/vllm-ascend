#!/usr/bin/env python3
"""Check TurboQuant 4-bit fy/fyn approximation errors.

This script mirrors the formula experiment in:
  csrc/turboquant_pack_kv_for_cache4bit/op_kernel/turboquant_pack_kv_for_cache4bit.cpp
  csrc/turboquant_attention_paged4bit/op_kernel/decode_device.h

It reports:
  1. inverse consistency for candidate_fyn(fy(x)) and exact_fyn(fy(x))
  2. fy(code - 7.5) drift from the current fixed 4-bit MSE codebook
  3. old codebook-nearest quantization vs formula fyn+round quantization
"""

from __future__ import annotations

import argparse
import math
import struct
from collections.abc import Callable, Iterable


FY_LINEAR = 0.020799
FY_CUBIC = 0.0001926
FY_B = FY_CUBIC / FY_LINEAR

# Exact fp32 bit pattern copied from vllm_ascend/ops/turboquant_kv_cache.py.
TURBOQUANT_4BIT_CODEBOOK_FP32_LE_HEX = (
    "8ff666becf1129be561000bed7bdc0bdd5c48cbd49a43fbde5d9ddbc0b110cbc"
    "07c5183ca23ae13cb1f13f3d4fe48c3d92b5c03dbd46fe3dcda8263e59ee613e"
)


def fixed_4bit_codebook() -> list[float]:
    raw = bytes.fromhex(TURBOQUANT_4BIT_CODEBOOK_FP32_LE_HEX)
    return list(struct.unpack("<16f", raw))


def fy_from_x(x: float) -> float:
    return FY_CUBIC * x * x * x + FY_LINEAR * x


def fy_from_code(code: int) -> float:
    return fy_from_x(float(code) - 7.5)


def fyn_exact_cardano(y: float) -> float:
    """Exact real inverse of y = FY_LINEAR * x + FY_CUBIC * x^3."""
    if y == 0.0:
        return 0.0
    scale = 2.0 * math.sqrt(FY_LINEAR / (3.0 * FY_CUBIC))
    arg = (3.0 * y / (2.0 * FY_LINEAR)) * math.sqrt(3.0 * FY_CUBIC / FY_LINEAR)
    return scale * math.sinh(math.asinh(arg) / 3.0)


def fyn_quadratic_direct(y: float) -> float:
    """Candidate inverse from the quadratic-u derivation: sign(y) * sqrt(u)."""
    if y == 0.0:
        return 0.0
    t = y / FY_LINEAR
    u = (-1.0 + math.sqrt(1.0 + 4.0 * FY_B * t * t)) / (2.0 * FY_B)
    return math.copysign(math.sqrt(max(u, 0.0)), y)


def fyn_quadratic_kernel_form(y: float) -> float:
    """Candidate inverse form currently used by the AscendC pack kernel."""
    if y == 0.0:
        return 0.0
    t = y / FY_LINEAR
    u = (-1.0 + math.sqrt(1.0 + 4.0 * FY_B * t * t)) / (2.0 * FY_B)
    return y / (FY_LINEAR * (1.0 + FY_B * u))


def clamp_code(v: float) -> int:
    return max(0, min(15, int(round(v))))


def formula_quantize_kernel(y: float) -> int:
    return clamp_code(fyn_quadratic_kernel_form(y) + 7.5)


def formula_quantize_exact(y: float) -> int:
    return clamp_code(fyn_exact_cardano(y) + 7.5)


def nearest_codebook_quantize(y: float, codebook: list[float]) -> int:
    return min(range(len(codebook)), key=lambda i: abs(y - codebook[i]))


def linspace(start: float, end: float, count: int) -> Iterable[float]:
    if count <= 1:
        yield start
        return
    step = (end - start) / float(count - 1)
    for i in range(count):
        yield start + step * i


def stats(values: Iterable[float]) -> tuple[float, float, float]:
    vals = [abs(v) for v in values]
    if not vals:
        return 0.0, 0.0, 0.0
    max_v = max(vals)
    mean_v = math.fsum(vals) / len(vals)
    rms_v = math.sqrt(math.fsum(v * v for v in vals) / len(vals))
    return max_v, mean_v, rms_v


def print_stats(name: str, values: Iterable[float]) -> None:
    max_v, mean_v, rms_v = stats(values)
    print(f"{name:<38} max={max_v:.12e} mean={mean_v:.12e} rms={rms_v:.12e}")


def check_inverse(samples: int) -> None:
    xs = list(linspace(-7.5, 7.5, samples))
    x_from_exact_inv = [fyn_exact_cardano(fy_from_x(x)) - x for x in xs]
    y_from_exact_inv = [fy_from_x(fyn_exact_cardano(fy_from_x(x))) - fy_from_x(x) for x in xs]
    x_from_kernel_inv = [fyn_quadratic_kernel_form(fy_from_x(x)) - x for x in xs]
    y_from_kernel_inv = [
        fy_from_x(fyn_quadratic_kernel_form(fy_from_x(x))) - fy_from_x(x)
        for x in xs
    ]
    x_from_direct_inv = [fyn_quadratic_direct(fy_from_x(x)) - x for x in xs]
    kernel_vs_exact = [
        fyn_quadratic_kernel_form(fy_from_x(x)) - fyn_exact_cardano(fy_from_x(x))
        for x in xs
    ]

    print("== Inverse consistency over x in [-7.5, 7.5] ==")
    print_stats("abs(fyn_exact(fy(x)) - x)", x_from_exact_inv)
    print_stats("abs(fy(fyn_exact(fy(x))) - fy(x))", y_from_exact_inv)
    print_stats("abs(fyn_kernel_quad(fy(x)) - x)", x_from_kernel_inv)
    print_stats("abs(fy(fyn_kernel_quad(fy(x))) - fy(x))", y_from_kernel_inv)
    print_stats("abs(fyn_direct_quad(fy(x)) - x)", x_from_direct_inv)
    print_stats("abs(fyn_kernel_quad - fyn_exact)", kernel_vs_exact)
    print()


def print_code_table(codebook: list[float]) -> None:
    print("== 4-bit code points: current codebook vs fy(code - 7.5) ==")
    print(
        "code        x              codebook                  fy"
        "              fy-codebook     kernel_qf kq   exact_qf eq"
    )
    for code, cb in enumerate(codebook):
        x = float(code) - 7.5
        f = fy_from_x(x)
        kernel_q_float = fyn_quadratic_kernel_form(f) + 7.5
        kernel_q = formula_quantize_kernel(f)
        exact_q_float = fyn_exact_cardano(f) + 7.5
        exact_q = formula_quantize_exact(f)
        print(
            f"{code:4d} {x:12.6f} {cb:22.12e} {f:22.12e}"
            f" {f - cb:18.12e} {kernel_q_float:13.9f} {kernel_q:2d}"
            f" {exact_q_float:10.6f} {exact_q:2d}"
        )
    print_stats("abs(fy(code - 7.5) - codebook)", (fy_from_code(i) - codebook[i] for i in range(16)))
    print()


def check_quantizer_decision(
    name: str,
    y_min: float,
    y_max: float,
    samples: int,
    codebook: list[float],
    formula_quantize: Callable[[float], int],
) -> None:
    mismatches = 0
    max_code_delta = 0
    old_errors: list[float] = []
    new_errors: list[float] = []
    new_minus_old_errors: list[float] = []
    first_examples: list[tuple[float, int, int, float, float]] = []

    for y in linspace(y_min, y_max, samples):
        old_q = nearest_codebook_quantize(y, codebook)
        new_q = formula_quantize(y)
        old_recon = codebook[old_q]
        new_recon = fy_from_code(new_q)
        old_err = abs(y - old_recon)
        new_err = abs(y - new_recon)

        old_errors.append(old_err)
        new_errors.append(new_err)
        new_minus_old_errors.append(new_err - old_err)

        if old_q != new_q:
            mismatches += 1
            max_code_delta = max(max_code_delta, abs(new_q - old_q))
            if len(first_examples) < 8:
                first_examples.append((y, old_q, new_q, old_err, new_err))

    print(f"== Quantization decision comparison: {name} ==")
    print(f"range=[{y_min:.12e}, {y_max:.12e}], samples={samples}")
    print(f"mismatch_count={mismatches} ({mismatches / samples:.6%}), max_code_delta={max_code_delta}")
    print_stats("old abs(y - codebook[old_q])", old_errors)
    print_stats("new abs(y - fy(new_q - 7.5))", new_errors)
    print_stats("abs(new_err - old_err)", new_minus_old_errors)
    if first_examples:
        print("first mismatches: y old_q new_q old_err new_err")
        for y, old_q, new_q, old_err, new_err in first_examples:
            print(f"  {y: .12e} {old_q:5d} {new_q:5d} {old_err:.12e} {new_err:.12e}")
    print()


def check_prompt_simplification(samples: int) -> None:
    """Show the error if the final substituted formula omits division by a^2."""

    def prompt_simplified_fyn(y: float) -> float:
        if y == 0.0:
            return 0.0
        u = (-1.0 + math.sqrt(1.0 + 4.0 * FY_B * y * y)) / (2.0 * FY_B)
        return math.copysign(math.sqrt(max(u, 0.0)), y)

    xs = list(linspace(-7.5, 7.5, samples))
    errors = [prompt_simplified_fyn(fy_from_x(x)) - x for x in xs]
    print("== Optional check: substituted formula without (y / a)^2 ==")
    print_stats("abs(fyn_simplified(fy(x)) - x)", errors)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=100_001,
        help="Number of dense samples for inverse and decision checks.",
    )
    parser.add_argument(
        "--check-prompt-simplification",
        action="store_true",
        help="Also report the error of the substituted form that uses 4*b*y*y.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise SystemExit("--samples must be >= 2")

    codebook = fixed_4bit_codebook()
    formula_min = fy_from_code(0)
    formula_max = fy_from_code(15)
    codebook_min = min(codebook)
    codebook_max = max(codebook)

    print(f"FY_LINEAR={FY_LINEAR:.12e}, FY_CUBIC={FY_CUBIC:.12e}, FY_B={FY_B:.12e}")
    print()

    check_inverse(args.samples)
    print_code_table(codebook)
    check_quantizer_decision(
        "kernel quadratic fyn, current codebook support",
        codebook_min,
        codebook_max,
        args.samples,
        codebook,
        formula_quantize_kernel,
    )
    check_quantizer_decision(
        "kernel quadratic fyn, formula support",
        formula_min,
        formula_max,
        args.samples,
        codebook,
        formula_quantize_kernel,
    )
    check_quantizer_decision(
        "exact fyn, formula support",
        formula_min,
        formula_max,
        args.samples,
        codebook,
        formula_quantize_exact,
    )
    check_quantizer_decision(
        "exact fyn, union support",
        min(codebook_min, formula_min),
        max(codebook_max, formula_max),
        args.samples,
        codebook,
        formula_quantize_exact,
    )
    if args.check_prompt_simplification:
        check_prompt_simplification(args.samples)


if __name__ == "__main__":
    main()
