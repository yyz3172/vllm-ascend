# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


BITRESIDUAL_HEAD_SIZE = 128
BITRESIDUAL_SCALAR_BYTES = 2
BITRESIDUAL_ROW_BYTES = BITRESIDUAL_HEAD_SIZE + 3 * BITRESIDUAL_SCALAR_BYTES


def _validate_head_size(head_size: int) -> None:
    if head_size != BITRESIDUAL_HEAD_SIZE:
        raise ValueError(
            f"BitResidual reference supports head_size={BITRESIDUAL_HEAD_SIZE}, "
            f"got {head_size}."
        )


def _validate_rotation(rotation: torch.Tensor, head_size: int) -> torch.Tensor:
    if rotation.shape != (head_size, head_size):
        raise ValueError(
            f"rotation must be [{head_size}, {head_size}], got {tuple(rotation.shape)}."
        )
    return rotation.to(dtype=torch.float32)


def _scalar_nbytes(dtype: torch.dtype) -> int:
    nbytes = torch.empty((), dtype=dtype).element_size()
    if nbytes != BITRESIDUAL_SCALAR_BYTES:
        raise ValueError("BitResidual byte packing expects fp16/bf16 scalar storage.")
    return nbytes


def _scalar_to_bytes(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    nbytes = _scalar_nbytes(dtype)
    return values.to(dtype=dtype).contiguous().view(torch.uint8).reshape(
        *values.shape[:-1], nbytes
    )


def _bytes_to_scalar(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    _scalar_nbytes(dtype)
    return values.contiguous().view(dtype).reshape(*values.shape[:-1], 1)


def bit_residual_quantize(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    head_size: int = BITRESIDUAL_HEAD_SIZE,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    _validate_head_size(head_size)
    if x.shape[-1] != head_size:
        raise ValueError(f"x last dim must be {head_size}, got {x.shape[-1]}.")

    rotation_f32 = _validate_rotation(rotation, head_size).to(device=x.device)
    original_shape = tuple(x.shape)
    x_flat = x.to(dtype=torch.float32).reshape(-1, head_size)
    norms = x_flat.norm(p=2, dim=-1, keepdim=True)
    y = (x_flat / (norms + 1e-10)) @ rotation_f32.T

    signs = y >= 0
    sign_unit = torch.where(
        signs,
        torch.full_like(y, 1.0 / math.sqrt(head_size)),
        torch.full_like(y, -1.0 / math.sqrt(head_size)),
    )
    residual = y - sign_unit
    bases = residual.min(dim=-1, keepdim=True).values
    max_values = residual.max(dim=-1, keepdim=True).values
    ranges = max_values - bases
    steps = torch.where(
        ranges > 0,
        ranges / 127.0,
        torch.ones_like(ranges),
    )
    q7_indices = torch.round((residual - bases) / steps).clamp_(0, 127).to(
        torch.uint8
    )

    packed_shape = original_shape[:-1]
    return {
        "norms": norms.reshape(*packed_shape, 1),
        "signs": signs.reshape(*original_shape),
        "bases": bases.reshape(*packed_shape, 1),
        "steps": steps.reshape(*packed_shape, 1),
        "q7_indices": q7_indices.reshape(*original_shape),
        "original_shape": original_shape,
    }


def bit_residual_dequantize(
    packed: dict[str, torch.Tensor | tuple[int, ...]],
    rotation: torch.Tensor,
    *,
    head_size: int = BITRESIDUAL_HEAD_SIZE,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    _validate_head_size(head_size)
    q7_indices = packed["q7_indices"]
    signs = packed["signs"]
    norms = packed["norms"]
    bases = packed["bases"]
    steps = packed["steps"]
    if not (
        isinstance(q7_indices, torch.Tensor)
        and isinstance(signs, torch.Tensor)
        and isinstance(norms, torch.Tensor)
        and isinstance(bases, torch.Tensor)
        and isinstance(steps, torch.Tensor)
    ):
        raise TypeError("packed BitResidual fields must be tensors.")

    rotation_f32 = _validate_rotation(rotation, head_size).to(device=q7_indices.device)
    q7_flat = q7_indices.to(dtype=torch.float32).reshape(-1, head_size)
    sign_flat = signs.reshape(-1, head_size)
    norm_flat = norms.to(dtype=torch.float32).reshape(-1, 1)
    base_flat = bases.to(dtype=torch.float32).reshape(-1, 1)
    step_flat = steps.to(dtype=torch.float32).reshape(-1, 1)

    sign_unit = torch.where(
        sign_flat,
        torch.full_like(q7_flat, 1.0 / math.sqrt(head_size)),
        torch.full_like(q7_flat, -1.0 / math.sqrt(head_size)),
    )
    y_hat = sign_unit + base_flat + q7_flat * step_flat
    x_hat = (y_hat @ rotation_f32) * norm_flat
    return x_hat.reshape(q7_indices.shape).to(dtype=dtype)


def bit_residual_pack_to_bytes(
    packed: dict[str, torch.Tensor | tuple[int, ...]],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    q7_indices = packed["q7_indices"]
    signs = packed["signs"]
    norms = packed["norms"]
    bases = packed["bases"]
    steps = packed["steps"]
    if not (
        isinstance(q7_indices, torch.Tensor)
        and isinstance(signs, torch.Tensor)
        and isinstance(norms, torch.Tensor)
        and isinstance(bases, torch.Tensor)
        and isinstance(steps, torch.Tensor)
    ):
        raise TypeError("packed BitResidual fields must be tensors.")

    code = ((q7_indices & 0x7F) << 1) | (signs.to(torch.uint8) & 0x01)
    code_rows = code.reshape(-1, BITRESIDUAL_HEAD_SIZE)
    return torch.cat(
        [
            code_rows,
            _scalar_to_bytes(norms.reshape(-1, 1), dtype),
            _scalar_to_bytes(bases.reshape(-1, 1), dtype),
            _scalar_to_bytes(steps.reshape(-1, 1), dtype),
        ],
        dim=-1,
    )


def bit_residual_unpack_from_bytes(
    rows: torch.Tensor,
    *,
    dtype: torch.dtype,
    original_shape: tuple[int, ...],
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    if rows.dtype != torch.uint8:
        raise ValueError(f"rows must be uint8, got {rows.dtype}.")
    if rows.shape[-1] != BITRESIDUAL_ROW_BYTES:
        raise ValueError(
            f"rows last dim must be {BITRESIDUAL_ROW_BYTES}, got {rows.shape[-1]}."
        )
    if original_shape[-1] != BITRESIDUAL_HEAD_SIZE:
        raise ValueError(
            f"original_shape last dim must be {BITRESIDUAL_HEAD_SIZE}, "
            f"got {original_shape[-1]}."
        )

    nbytes = _scalar_nbytes(dtype)
    code = rows[:, :BITRESIDUAL_HEAD_SIZE].reshape(*original_shape)
    norm_off = BITRESIDUAL_HEAD_SIZE
    base_off = norm_off + nbytes
    step_off = base_off + nbytes
    packed_shape = original_shape[:-1] + (1,)
    return {
        "norms": _bytes_to_scalar(rows[:, norm_off:base_off], dtype).reshape(
            packed_shape
        ),
        "signs": (code & 0x01).to(torch.bool),
        "bases": _bytes_to_scalar(rows[:, base_off:step_off], dtype).reshape(
            packed_shape
        ),
        "steps": _bytes_to_scalar(rows[:, step_off:step_off + nbytes], dtype).reshape(
            packed_shape
        ),
        "q7_indices": ((code >> 1) & 0x7F).to(torch.uint8),
        "original_shape": original_shape,
    }


def bit_residual_block_bytes(block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")
    return block_size * BITRESIDUAL_ROW_BYTES


def bit_residual_pack_to_block_bytes(
    packed: dict[str, torch.Tensor | tuple[int, ...]],
    *,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")

    q7_indices = packed["q7_indices"]
    signs = packed["signs"]
    norms = packed["norms"]
    bases = packed["bases"]
    steps = packed["steps"]
    if not (
        isinstance(q7_indices, torch.Tensor)
        and isinstance(signs, torch.Tensor)
        and isinstance(norms, torch.Tensor)
        and isinstance(bases, torch.Tensor)
        and isinstance(steps, torch.Tensor)
    ):
        raise TypeError("packed BitResidual fields must be tensors.")
    if q7_indices.shape[-2:] != (block_size, BITRESIDUAL_HEAD_SIZE):
        raise ValueError(
            "q7_indices must end with "
            f"[{block_size}, {BITRESIDUAL_HEAD_SIZE}], got {tuple(q7_indices.shape)}."
        )

    code = ((q7_indices & 0x7F) << 1) | (signs.to(torch.uint8) & 0x01)
    prefix_shape = q7_indices.shape[:-2]
    code_bytes = code.reshape(*prefix_shape, block_size * BITRESIDUAL_HEAD_SIZE)
    scalar_shape = (*prefix_shape, block_size * BITRESIDUAL_SCALAR_BYTES)
    return torch.cat(
        [
            code_bytes,
            _scalar_to_bytes(norms, dtype).reshape(scalar_shape),
            _scalar_to_bytes(bases, dtype).reshape(scalar_shape),
            _scalar_to_bytes(steps, dtype).reshape(scalar_shape),
        ],
        dim=-1,
    )


def bit_residual_unpack_from_block_bytes(
    blocks: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_size: int,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    if blocks.dtype != torch.uint8:
        raise ValueError(f"blocks must be uint8, got {blocks.dtype}.")
    expected_bytes = bit_residual_block_bytes(block_size)
    if blocks.shape[-1] != expected_bytes:
        raise ValueError(
            f"blocks last dim must be {expected_bytes}, got {blocks.shape[-1]}."
        )

    scalar_bytes = block_size * _scalar_nbytes(dtype)
    code_bytes = block_size * BITRESIDUAL_HEAD_SIZE
    norm_off = code_bytes
    base_off = norm_off + scalar_bytes
    step_off = base_off + scalar_bytes
    prefix_shape = blocks.shape[:-1]
    vector_shape = (*prefix_shape, block_size, BITRESIDUAL_HEAD_SIZE)
    scalar_shape = (*prefix_shape, block_size, 1)
    code = blocks[..., :code_bytes].reshape(vector_shape)
    return {
        "norms": blocks[..., norm_off:base_off]
        .contiguous()
        .view(dtype)
        .reshape(scalar_shape),
        "signs": (code & 0x01).to(torch.bool),
        "bases": blocks[..., base_off:step_off]
        .contiguous()
        .view(dtype)
        .reshape(scalar_shape),
        "steps": blocks[..., step_off:step_off + scalar_bytes]
        .contiguous()
        .view(dtype)
        .reshape(scalar_shape),
        "q7_indices": ((code >> 1) & 0x7F).to(torch.uint8),
        "original_shape": vector_shape,
    }


def compute_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    x_f32 = x.to(dtype=torch.float32)
    x_hat_f32 = x_hat.to(dtype=torch.float32)
    diff = x_f32 - x_hat_f32
    mse = diff.square().mean().item()
    max_error = diff.abs().max().item()
    x_norm = x_f32.norm().item()
    diff_norm = diff.norm().item()
    relative_error = diff_norm / (x_norm + 1e-10)
    if x_f32.shape[-1] == x_hat_f32.shape[-1]:
        cosine_similarity = F.cosine_similarity(
            x_f32.reshape(-1, x_f32.shape[-1]),
            x_hat_f32.reshape(-1, x_hat_f32.shape[-1]),
            dim=-1,
            eps=1e-10,
        ).mean().item()
    else:
        cosine_similarity = float("nan")
    return {
        "mse": mse,
        "max_error": max_error,
        "relative_error": relative_error,
        "cosine_similarity": cosine_similarity,
    }

