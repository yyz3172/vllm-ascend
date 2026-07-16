# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# BitResidual K8/V4 dequant + PA layout golden (CPU reference).
# Matches csrc/bit_residual_pack_k8v4/IMPL_DESIGN.md and
# tests/e2e/singlecard/xrx_bit_residual_k8v4_golden.py decode helpers.

from __future__ import annotations

from typing import Literal

import torch

HEAD_SIZE = 128
BLOCK_ROWS = 16
KEY_BLOCK_STRIDE = BLOCK_ROWS * (HEAD_SIZE + 4)  # 2112
VALUE_BLOCK_STRIDE = BLOCK_ROWS * (HEAD_SIZE // 2 + 4)  # 1088
KEY_ROW_CODE_BYTES = HEAD_SIZE
VALUE_ROW_CODE_BYTES = HEAD_SIZE // 2
KEY_QUANT_LEVELS = 127
VAL_QUANT_LEVELS = 15


def key_code_offset(pos_in_block: int) -> int:
    return pos_in_block * KEY_ROW_CODE_BYTES


def key_base_offset(block_size: int, pos_in_block: int) -> int:
    return block_size * KEY_ROW_CODE_BYTES + pos_in_block * 2


def key_step_offset(block_size: int, pos_in_block: int) -> int:
    return block_size * (KEY_ROW_CODE_BYTES + 2) + pos_in_block * 2


def value_code_offset(pos_in_block: int) -> int:
    return pos_in_block * VALUE_ROW_CODE_BYTES


def value_vmin_offset(block_size: int, pos_in_block: int) -> int:
    return block_size * VALUE_ROW_CODE_BYTES + pos_in_block * 2


def value_vstep_offset(block_size: int, pos_in_block: int) -> int:
    return block_size * (VALUE_ROW_CODE_BYTES + 2) + pos_in_block * 2


def key_head_stride(block_size: int) -> int:
    return block_size * (HEAD_SIZE + 4)


def value_head_stride(block_size: int) -> int:
    return block_size * (VALUE_ROW_CODE_BYTES + 4)


def pack_value_idx4(value_idx: torch.Tensor) -> torch.Tensor:
    signed_idx = ((value_idx.to(torch.int16) - 8) & 0x0F).to(torch.uint8)
    return signed_idx[0::2] | (signed_idx[1::2] << 4)


def unpack_value_idx4(code: torch.Tensor) -> torch.Tensor:
    code_i32 = code.to(torch.int32)
    low = code_i32 & 0x0F
    high = (code_i32 >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low) + 8
    high = torch.where(high >= 8, high - 16, high) + 8
    idx4 = torch.empty(HEAD_SIZE, dtype=torch.float32)
    idx4[0::2] = low.float()
    idx4[1::2] = high.float()
    return idx4


def decode_key_row(
    codes: torch.Tensor,
    base: float,
    step: float,
) -> torch.Tensor:
    code = codes.to(torch.int32)
    q7 = (code & 0x7F).float()
    sign = (code >> 7).float()
    sign_val = torch.where(sign == 0, 1.0, -1.0)
    err = base + q7 * step
    return err * sign_val


def decode_value_row(
    packed_codes: torch.Tensor,
    vmin: float,
    vstep: float,
) -> torch.Tensor:
    idx4 = unpack_value_idx4(packed_codes)
    return vmin + idx4 * vstep


def encode_key_row(y: torch.Tensor, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, float, float]:
    """Pack-side K8 encode (matches bit_residual_pack_k8v4_aiv.h semantics)."""
    y_f = y.to(torch.float32)
    m = y_f.abs().max().item()
    m = max(m, 1.0e-12)
    abs_norm = y_f.abs() / m
    base_norm = abs_norm.min().item()
    step_norm = (1.0 - base_norm) / KEY_QUANT_LEVELS if base_norm < 1.0 else 1.0
    step_norm = max(step_norm, 1.0e-6)
    q7 = torch.round((abs_norm - base_norm) / step_norm).clamp_(0, KEY_QUANT_LEVELS).to(torch.int32)
    sign = (y_f < 0).to(torch.int32)
    code = (q7 | (sign << 7)).to(torch.uint8)
    base = base_norm * m
    step = step_norm * m
    return code, float(torch.tensor(base, dtype=dtype)), float(torch.tensor(step, dtype=dtype))


def encode_value_row(y: torch.Tensor, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, float, float]:
    y_f = y.to(torch.float32)
    m = y_f.abs().max().item()
    m = max(m, 1.0e-12)
    y_norm = y_f / m
    vmin_norm = y_norm.min().item()
    vmax_norm = y_norm.max().item()
    vstep_norm = (vmax_norm - vmin_norm) / VAL_QUANT_LEVELS
    vstep_norm = max(vstep_norm, 1.0e-6)
    idx4 = torch.round((y_norm - vmin_norm) / vstep_norm).clamp_(0, VAL_QUANT_LEVELS).to(torch.int32)
    packed = pack_value_idx4(idx4.to(torch.uint8))
    vmin = vmin_norm * m
    vstep = vstep_norm * m
    return packed, float(torch.tensor(vmin, dtype=dtype)), float(torch.tensor(vstep, dtype=dtype))


def read_fp16_scalar(raw: torch.Tensor, offset: int, dtype: torch.dtype) -> float:
    return raw[offset : offset + 2].contiguous().view(dtype)[0].float().item()


def decode_key_from_head_block(
    block: torch.Tensor,
    pos_in_block: int,
    block_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    off = key_code_offset(pos_in_block)
    codes = block[off : off + KEY_ROW_CODE_BYTES]
    base = read_fp16_scalar(block, key_base_offset(block_size, pos_in_block), dtype)
    step = read_fp16_scalar(block, key_step_offset(block_size, pos_in_block), dtype)
    return decode_key_row(codes, base, step)


def decode_value_from_head_block(
    block: torch.Tensor,
    pos_in_block: int,
    block_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    off = value_code_offset(pos_in_block)
    codes = block[off : off + VALUE_ROW_CODE_BYTES]
    vmin = read_fp16_scalar(block, value_vmin_offset(block_size, pos_in_block), dtype)
    vstep = read_fp16_scalar(block, value_vstep_offset(block_size, pos_in_block), dtype)
    return decode_value_row(codes, vmin, vstep)


def head_block_base(
    block_id: int,
    kv_head: int,
    num_kv_heads: int,
    block_size: int,
    kind: Literal["key", "value"],
) -> int:
    stride = key_head_stride(block_size) if kind == "key" else value_head_stride(block_size)
    return (block_id * num_kv_heads + kv_head) * stride
