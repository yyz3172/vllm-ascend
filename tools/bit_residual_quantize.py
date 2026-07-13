"""
BitResidual 8-bit Quantization for KV Cache vectors (head_size=128).

Algorithm per vector x of dimension D=128:
  1. Compute L2 norm: norm = ||x||
  2. Normalize: n = x / norm
  3. Rotate: y = n @ R^T   (R is a pre-generated random orthogonal matrix)
  4. Sign bit: sign = (y >= 0). Each dim gets 1 bit (1=positive, 0=negative).
     bit_vec = sign * (1/sqrt(D)) - (1-sign) * (1/sqrt(D)) = ±1/sqrt(D)
  5. Residual: err = y - bit_vec
  6. 7-bit uniform quantization of err:
     base = min(err), step = (max(err) - min(err)) / 127
     q7 = round((err - base) / step)   # 0..127, stored as 7 bits
  7. Reconstructed rotated vector: y_hat = bit_vec + base + q7 * step
  8. Reconstructed normalized vector: n_hat = y_hat @ R
  9. Reconstructed original vector: x_hat = n_hat * norm

The rotation matrix R ensures roughly 50/50 positive/negative distribution
in the rotated space, making the 1-bit sign approximation more effective.

Storage per vector (D=128):
  - norm:   2 bytes (fp16)
  - sign:   128 bits = 16 bytes
  - base:   2 bytes (fp16)
  - step:   2 bytes (fp16)
  - sign/q7: 128 bytes, one byte per dim: bit0 is sign, bits1..7 are q7
  Total per vector: 128 + 2 + 2 + 2 = 134 bytes

Paged cache storage is field-major within each (page block, head):
  [all sign/q7 bytes][all norm scalars][all base scalars][all step scalars].

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
    """Quantize a batch of vectors using BitResidual 8-bit scheme.

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
        - q7_indices: [N, D] or [N, H, D], uint8 — 7-bit quantized indices (0..127)
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

    # 6. 7-bit uniform quantization of the residual
    err_min = err.min(dim=-1, keepdim=True).values  # [N, 1]
    err_max = err.max(dim=-1, keepdim=True).values  # [N, 1]
    range_val = err_max - err_min
    step = range_val / 127.0
    step = torch.where(range_val < eps, torch.full_like(step, eps), step)

    q7_continuous = (err - err_min) / step
    q7_indices = torch.clamp(torch.round(q7_continuous), 0, 127).to(torch.uint8)

    # Reshape to original shape
    if len(original_shape) == 3:
        _, H, D_ = original_shape
        norms = norms.reshape(_, H, 1)
        signs = signs.reshape(_, H, D_)
        bases = err_min.reshape(_, H, 1)
        steps = step.reshape(_, H, 1)
        q7_indices = q7_indices.reshape(_, H, D_)
    else:
        bases = err_min
        steps = step

    return {
        "norms": norms.to(x.dtype),
        "signs": signs,
        "bases": bases.to(x.dtype),
        "steps": steps.to(x.dtype),
        "q7_indices": q7_indices,
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
    """Reconstruct vectors from BitResidual 8-bit packed data.

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
    q7_indices = packed["q7_indices"].float()
    R = rotation.float().to(norms.device)

    D = head_size
    sqrt_D = math.sqrt(D)

    # Reconstruct bit_vec in rotated space
    bit_vec = signs.float() * (1.0 / sqrt_D) + (~signs).float() * (-1.0 / sqrt_D)

    # Reconstruct residual
    err_hat = bases + q7_indices * steps

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
# Byte row storage helpers
# ---------------------------------------------------------------------------

BITRESIDUAL_CODE_BYTES = 128
BITRESIDUAL_SCALAR_BYTES = 2
BITRESIDUAL_ROW_BYTES = (
    BITRESIDUAL_CODE_BYTES + 3 * BITRESIDUAL_SCALAR_BYTES
)


def bit_residual_block_bytes(block_size: int) -> int:
    """Packed bytes for one ``(page block, head)`` BitResidual slab."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return block_size * BITRESIDUAL_ROW_BYTES


def _bit_residual_code_bytes(packed: dict) -> torch.Tensor:
    signs = packed["signs"].reshape(-1, packed["signs"].shape[-1]).to(torch.uint8)
    q7 = packed["q7_indices"].reshape(-1, packed["q7_indices"].shape[-1]).to(torch.uint8)
    if signs.shape[-1] != 128 or q7.shape[-1] != 128:
        raise ValueError("BitResidual byte rows require head_size=128.")
    return ((q7 & 0x7F) << 1) | (signs & 0x01)


def bit_residual_pack_to_block_bytes(
    packed: dict,
    *,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    """Pack quantized components into field-major page blocks.

    Input components must have shape ``[num_blocks, num_heads, block_size, 128]``
    for code-bearing tensors and ``[..., 1]`` for scalars. The returned cache has
    shape ``[num_blocks, num_heads, block_size * 134]`` with layout:
    ``codes | norms | bases | steps``.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("BitResidual block cache stores scalars as fp16/bf16 only.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    signs_shape = packed["signs"].shape
    if len(signs_shape) != 4 or signs_shape[-2:] != (block_size, 128):
        raise ValueError(
            "BitResidual block pack expects signs shape "
            "[num_blocks, num_heads, block_size, 128]."
        )
    num_blocks, num_heads = signs_shape[:2]
    code = _bit_residual_code_bytes(packed).reshape(
        num_blocks, num_heads, block_size, BITRESIDUAL_CODE_BYTES
    )
    norms = packed["norms"].reshape(num_blocks, num_heads, block_size, 1).to(dtype)
    bases = packed["bases"].reshape(num_blocks, num_heads, block_size, 1).to(dtype)
    steps = packed["steps"].reshape(num_blocks, num_heads, block_size, 1).to(dtype)
    block = torch.empty(
        (num_blocks, num_heads, bit_residual_block_bytes(block_size)),
        device=code.device,
        dtype=torch.uint8,
    )
    code_bytes = block_size * BITRESIDUAL_CODE_BYTES
    scalar_bytes = block_size * BITRESIDUAL_SCALAR_BYTES
    norm_off = code_bytes
    base_off = norm_off + scalar_bytes
    step_off = base_off + scalar_bytes
    block[..., :code_bytes] = code.reshape(num_blocks, num_heads, code_bytes)
    block[..., norm_off:base_off] = norms.contiguous().view(torch.uint8).reshape(
        num_blocks, num_heads, scalar_bytes
    )
    block[..., base_off:step_off] = bases.contiguous().view(torch.uint8).reshape(
        num_blocks, num_heads, scalar_bytes
    )
    block[..., step_off:] = steps.contiguous().view(torch.uint8).reshape(
        num_blocks, num_heads, scalar_bytes
    )
    return block


def bit_residual_unpack_from_block_bytes(
    blocks: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_size: int,
) -> dict:
    """Unpack field-major page blocks into quantized components."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("BitResidual block cache stores scalars as fp16/bf16 only.")
    if blocks.shape[-1] != bit_residual_block_bytes(block_size):
        raise ValueError(
            f"Expected last dim {bit_residual_block_bytes(block_size)}, "
            f"got {blocks.shape[-1]}."
        )
    if blocks.ndim != 3:
        raise ValueError("BitResidual block cache must be [num_blocks, num_heads, bytes].")
    num_blocks, num_heads = blocks.shape[:2]
    code_bytes = block_size * BITRESIDUAL_CODE_BYTES
    scalar_bytes = block_size * BITRESIDUAL_SCALAR_BYTES
    norm_off = code_bytes
    base_off = norm_off + scalar_bytes
    step_off = base_off + scalar_bytes
    code = blocks[..., :code_bytes].reshape(
        num_blocks, num_heads, block_size, BITRESIDUAL_CODE_BYTES
    )
    signs = (code & 0x01) != 0
    q7_indices = (code >> 1) & 0x7F
    norms = blocks[..., norm_off:base_off].contiguous().view(dtype).reshape(
        num_blocks, num_heads, block_size, 1
    )
    bases = blocks[..., base_off:step_off].contiguous().view(dtype).reshape(
        num_blocks, num_heads, block_size, 1
    )
    steps = blocks[..., step_off:].contiguous().view(dtype).reshape(
        num_blocks, num_heads, block_size, 1
    )
    return {
        "norms": norms,
        "signs": signs,
        "bases": bases,
        "steps": steps,
        "q7_indices": q7_indices,
    }


def bit_residual_pack_to_bytes(packed: dict, *, dtype: torch.dtype) -> torch.Tensor:
    """Pack independent rows as ``codes | norm | base | step`` for tests/tools."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("BitResidual byte rows store scalars as fp16/bf16 only.")
    code = _bit_residual_code_bytes(packed)
    scalars = torch.cat(
        [
            packed["norms"].reshape(-1, 1).to(dtype),
            packed["bases"].reshape(-1, 1).to(dtype),
            packed["steps"].reshape(-1, 1).to(dtype),
        ],
        dim=-1,
    ).contiguous().view(torch.uint8)
    rows = torch.empty(
        (code.shape[0], BITRESIDUAL_ROW_BYTES),
        device=code.device,
        dtype=torch.uint8,
    )
    rows[:, :BITRESIDUAL_CODE_BYTES] = code
    rows[:, BITRESIDUAL_CODE_BYTES:] = scalars.reshape(code.shape[0], 6)
    return rows


def bit_residual_unpack_from_bytes(
    rows: torch.Tensor,
    *,
    dtype: torch.dtype,
    original_shape: tuple[int, ...] | None = None,
) -> dict:
    """Unpack independent test/tool rows into quantized components."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("BitResidual byte rows store scalars as fp16/bf16 only.")
    if rows.shape[-1] != BITRESIDUAL_ROW_BYTES:
        raise ValueError(
            f"Expected last dim {BITRESIDUAL_ROW_BYTES}, got {rows.shape[-1]}."
        )
    flat = rows.reshape(-1, BITRESIDUAL_ROW_BYTES).contiguous()
    code = flat[:, :BITRESIDUAL_CODE_BYTES]
    signs = (code & 0x01) != 0
    q7_indices = (code >> 1) & 0x7F
    scalars = flat[:, BITRESIDUAL_CODE_BYTES:].contiguous().view(dtype).reshape(-1, 3)
    norms = scalars[:, 0:1]
    bases = scalars[:, 1:2]
    steps = scalars[:, 2:3]
    if original_shape is not None:
        if original_shape[-1] != 128:
            raise ValueError("original_shape must end with head_size=128.")
        prefix = original_shape[:-1]
        signs = signs.reshape(*prefix, 128)
        q7_indices = q7_indices.reshape(*prefix, 128)
        norms = norms.reshape(*prefix, 1)
        bases = bases.reshape(*prefix, 1)
        steps = steps.reshape(*prefix, 1)
    return {
        "norms": norms,
        "signs": signs,
        "bases": bases,
        "steps": steps,
        "q7_indices": q7_indices,
    }
