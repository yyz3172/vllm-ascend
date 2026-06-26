"""
BitResidual 4-bit Quantization for KV Cache vectors (head_size=128).

Algorithm per vector x of dimension D=128:
  1. Compute L2 norm: norm = ||x||
  2. Normalize: n = x / norm
  3. Rotate: y = n @ R^T   (R is a pre-generated random orthogonal matrix)
  4. Sign bit: sign = (y >= 0). Each dim gets 1 bit (1=positive, 0=negative).
     bit_vec = sign * (1/sqrt(D)) - (1-sign) * (1/sqrt(D)) = ±1/sqrt(D)
  5. Residual: err = y - bit_vec
  6. 3-bit uniform quantization of err:
     base = min(err), step = (max(err) - min(err)) / 7
     q3 = round((err - base) / step)   # 0..7, stored as 3 bits
  7. Reconstructed rotated vector: y_hat = bit_vec + base + q3 * step
  8. Reconstructed normalized vector: n_hat = y_hat @ R
  9. Reconstructed original vector: x_hat = n_hat * norm

The rotation matrix R ensures roughly 50/50 positive/negative distribution
in the rotated space, making the 1-bit sign approximation more effective.

Storage per vector (D=128):
  - norm:   2 bytes (fp16)
  - sign:   128 bits = 16 bytes
  - base:   2 bytes (fp16)
  - step:   2 bytes (fp16)
  - q3:     128 * 3 bits = 48 bytes (tight packing)
  Total:    16 + 48 + 2 + 2 + 2 = 70 bytes (vs TurboQuant 4-bit = 66 bytes)

R is shared across all vectors (like TurboQuant's rotation), stored once.
"""

from __future__ import annotations

import math
import torch


# ---------------------------------------------------------------------------
# Rotation matrix generation (shared, like TurboQuant)
# ---------------------------------------------------------------------------

def generate_rotation_matrix(
    head_size: int = 128,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a random orthogonal matrix R of shape [D, D].

    Uses QR decomposition of a random matrix to produce a uniformly
    random orthogonal matrix (same approach as TurboQuant).
    """
    torch.manual_seed(seed)
    A = torch.randn(head_size, head_size, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)
    # Ensure Q is truly orthogonal (sign convention)
    Q = Q * torch.sign(torch.diag(Q))  # make diagonal positive
    return Q


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------

def bit_residual_quantize(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    head_size: int = 128,
) -> dict:
    """Quantize a batch of vectors using BitResidual 4-bit scheme.

    Args:
        x: shape [N, head_size] or [N, H, head_size], float16/bf16/float32
        rotation: [head_size, head_size] orthogonal rotation matrix R
        head_size: vector dimension (must be 128)

    Returns:
        dict with quantized components:
        - norms:      [N, 1] or [N, H, 1], same dtype as x — L2 norms
        - signs:      [N, D] or [N, H, D], bool — sign bits in rotated space
        - bases:      [N, 1] or [N, H, 1], same dtype as x — residual min
        - steps:      [N, 1] or [N, H, 1], same dtype as x — residual step
        - q3_indices: [N, D] or [N, H, D], uint8 — 3-bit quantized indices (0..7)
    """
    original_shape = x.shape
    if x.shape[-1] != head_size:
        raise ValueError(f"Last dim must be {head_size}, got {x.shape[-1]}")

    D = head_size
    sqrt_D = math.sqrt(D)

    # Work in float32 for precision
    x_f32 = x.float()
    N_total = x_f32.numel() // D
    x_flat = x_f32.reshape(N_total, D)
    R = rotation.float().to(x.device)  # [D, D]

    # 1. L2 norm
    norms = x_flat.norm(p=2, dim=-1, keepdim=True)  # [N, 1]

    # 2. Normalize
    eps = 1e-10
    n = x_flat / (norms + eps)  # [N, D]

    # 3. Rotate: y = n @ R^T
    y = n @ R.T  # [N, D]

    # 4. Sign bits in rotated space
    signs = (y >= 0)  # [N, D], bool

    # bit_vec = ±1/sqrt(D)
    bit_vec = signs.float() * (1.0 / sqrt_D) + (~signs).float() * (-1.0 / sqrt_D)

    # 5. Residual
    err = y - bit_vec  # [N, D]

    # 6. 3-bit uniform quantization of the residual
    err_min = err.min(dim=-1, keepdim=True).values  # [N, 1]
    err_max = err.max(dim=-1, keepdim=True).values  # [N, 1]
    range_val = err_max - err_min
    step = range_val / 7.0
    step = torch.where(range_val < eps, torch.full_like(step, eps), step)

    q3_continuous = (err - err_min) / step
    q3_indices = torch.clamp(torch.round(q3_continuous), 0, 7).to(torch.uint8)

    # Reshape to original shape
    if len(original_shape) == 3:
        _, H, D_ = original_shape
        norms = norms.reshape(_, H, 1)
        signs = signs.reshape(_, H, D_)
        bases = err_min.reshape(_, H, 1)
        steps = step.reshape(_, H, 1)
        q3_indices = q3_indices.reshape(_, H, D_)
    else:
        bases = err_min
        steps = step

    return {
        "norms": norms.to(x.dtype),
        "signs": signs,
        "bases": bases.to(x.dtype),
        "steps": steps.to(x.dtype),
        "q3_indices": q3_indices,
    }


# ---------------------------------------------------------------------------
# Dequantize
# ---------------------------------------------------------------------------

def bit_residual_dequantize(
    packed: dict,
    rotation: torch.Tensor,
    *,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct vectors from BitResidual 4-bit packed data.

    Args:
        packed: dict from bit_residual_quantize
        rotation: [head_size, head_size] orthogonal rotation matrix R
        head_size: vector dimension
        dtype: output dtype

    Returns:
        Reconstructed tensor with same shape as original input.
    """
    norms = packed["norms"].float()
    signs = packed["signs"]
    bases = packed["bases"].float()
    steps = packed["steps"].float()
    q3_indices = packed["q3_indices"].float()
    R = rotation.float().to(norms.device)

    D = head_size
    sqrt_D = math.sqrt(D)

    # Reconstruct bit_vec in rotated space
    bit_vec = signs.float() * (1.0 / sqrt_D) + (~signs).float() * (-1.0 / sqrt_D)

    # Reconstruct residual
    err_hat = bases + q3_indices * steps

    # Reconstruct rotated normalized vector
    y_hat = bit_vec + err_hat  # [N, D]

    # Rotate back: n_hat = y_hat @ R
    n_hat = y_hat @ R  # [N, D]

    # Reconstruct original vector
    x_hat = n_hat * norms

    return x_hat.to(dtype)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict:
    """Compute MSE, max error, cosine similarity, and relative error."""
    orig_f = original.float()
    recon_f = reconstructed.float()

    # Per-vector MSE
    mse_per_vec = ((orig_f - recon_f) ** 2).mean(dim=-1)
    mse = mse_per_vec.mean().item()

    # Max absolute error
    max_err = (orig_f - recon_f).abs().max().item()

    # Relative error per vector: ||x - x_hat|| / ||x||
    orig_norm = orig_f.norm(p=2, dim=-1)
    err_norm = (orig_f - recon_f).norm(p=2, dim=-1)
    rel_err = (err_norm / (orig_norm + 1e-10)).mean().item()

    # Cosine similarity per vector
    cos_sim = torch.nn.functional.cosine_similarity(orig_f, recon_f, dim=-1).mean().item()

    return {
        "mse": mse,
        "max_error": max_err,
        "relative_error": rel_err,
        "cosine_similarity": cos_sim,
    }


# ---------------------------------------------------------------------------
# Byte storage size (for comparison)
# ---------------------------------------------------------------------------

BITRESIDUAL_ROW_BYTES = 70  # signs(16) + q3_tight(48) + norm(2) + base(2) + step(2)
