"""
Cubic-Polynomial 4-bit Quantization for KV Cache vectors (head_size=128).

Algorithm per vector x of dimension D=128:
  1. Compute L2 norm: norm = ||x||
  2. Normalize: n = x / norm
  3. Rotate: y = n @ R^T   (R is a pre-generated random orthogonal matrix)
  4. Cubic polynomial mapping per coordinate:
       p(x) = -2.082971e+02 * x^3 - 1.944945e-01 * x^2 + 4.317486e+01 * x + 7.511501e+00
     Clamp the result to [0, 15] and round to integer → 4-bit index.
  5. Dequantize: look up each 4-bit index in a 16-entry codebook table `tab`
     to get the reconstructed rotated-space value t.
  6. Inverse rotate: n_hat = t @ R
  7. Restore scale: x_hat = n_hat * norm

Storage per vector (D=128):
  - norm:    2 bytes (fp16)
  - indices: 128 * 4 bits = 64 bytes (packed, 2 per byte)
  Total:     66 bytes (same as TurboQuant 4-bit)

R is shared across all vectors (like TurboQuant's rotation), stored once.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Codebook lookup table (16 entries, index 0..15)
# ---------------------------------------------------------------------------

_CUBIC_QUANT_TABLE: list[float] = [
    -2.378270e-01,
    -1.777851e-01,
    -1.411261e-01,
    -1.110442e-01,
    -8.417787e-02,
    -5.915329e-02,
    -3.521391e-02,
    -1.185461e-02,
    +1.132200e-02,
    +3.468274e-02,
    +5.862524e-02,
    +8.365526e-02,
    +1.105308e-01,
    +1.406294e-01,
    +1.773277e-01,
    +2.375010e-01,
]

# Coefficients of the cubic polynomial p(x) = a*x^3 + b*x^2 + c*x + d
_CUBIC_A = -2.082971e+02
_CUBIC_B = -1.944945e-01
_CUBIC_C = 4.317486e+01
_CUBIC_D = 7.511501e+00


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

def cubic_quantize(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    head_size: int = 128,
) -> dict:
    """Quantize a batch of vectors using cubic-polynomial 4-bit scheme.

    Args:
        x: shape [N, head_size] or [N, H, head_size], float16/bf16/float32
        rotation: [head_size, head_size] orthogonal rotation matrix R
        head_size: vector dimension (must be 128)

    Returns:
        dict with quantized components:
        - norms:   [N, 1] or [N, H, 1], same dtype as x — L2 norms
        - indices: [N, D//2] or [N, H, D//2], uint8 — packed 4-bit indices
    """
    original_shape = x.shape
    if x.shape[-1] != head_size:
        raise ValueError(f"Last dim must be {head_size}, got {x.shape[-1]}")

    D = head_size

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

    # 4. Cubic polynomial mapping + clamp to [0, 15]
    #    p(x) = a*x^3 + b*x^2 + c*x + d
    y_idx = _CUBIC_A * (y ** 3) + _CUBIC_B * (y ** 2) + _CUBIC_C * y + _CUBIC_D
    indices = torch.clamp(torch.round(y_idx), 0, 15).to(torch.uint8)  # [N, D]

    # Pack 4-bit indices: 2 per byte
    packed = _pack_uint4(indices)  # [N, D//2]

    # Reshape to original shape
    if len(original_shape) == 3:
        N, H, _ = original_shape
        norms = norms.reshape(N, H, 1)
        packed = packed.reshape(N, H, D // 2)

    return {
        "norms": norms.to(x.dtype),
        "indices": packed,
    }


def _pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint4 indices: dim 0..D/2-1 in low nibble, dim D/2..D-1 in high."""
    if indices.shape[-1] % 2 != 0:
        raise ValueError(f"Last dim must be even, got {indices.shape[-1]}")
    half_dim = indices.shape[-1] // 2
    low = indices[..., :half_dim] & 0x0F
    high = (indices[..., half_dim:] & 0x0F) << 4
    return (high | low).to(torch.uint8)


# ---------------------------------------------------------------------------
# Dequantize
# ---------------------------------------------------------------------------

def cubic_dequantize(
    packed: dict,
    rotation: torch.Tensor,
    *,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct vectors from cubic-polynomial 4-bit packed data.

    Args:
        packed: dict from cubic_quantize
        rotation: [head_size, head_size] orthogonal rotation matrix R
        head_size: vector dimension
        dtype: output dtype

    Returns:
        Reconstructed tensor with same shape as original input.
    """
    norms = packed["norms"].float()
    indices_packed = packed["indices"]
    R = rotation.float().to(norms.device)

    D = head_size

    # Unpack 4-bit indices
    indices = _unpack_uint4(indices_packed, D)  # [N, D], uint8

    # Lookup in codebook table
    table = torch.tensor(_CUBIC_QUANT_TABLE, dtype=torch.float32, device=norms.device)
    t = table[indices.long()]  # [N, D], float32

    # Inverse rotate: n_hat = t @ R
    n_hat = t @ R  # [N, D]

    # Restore scale
    x_hat = n_hat * norms

    return x_hat.to(dtype)


def _unpack_uint4(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes into uint4 indices using low-half/high-half layout."""
    if packed.dtype == torch.int8:
        packed = packed.view(dtype=torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    half_dim = orig_dim // 2
    out = torch.empty(
        packed.shape[:-1] + (orig_dim,), dtype=torch.uint8, device=packed.device
    )
    out[..., :half_dim] = low
    out[..., half_dim:] = high
    return out


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

CUBIC_QUANT_ROW_BYTES = 66  # indices(64) + norm(2)
