# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""L1/L2 CPU tests for BitResidual FIA dequant golden."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "bit_residual_fia_paged_k8v4"))

from golden_dequant import (  # noqa: E402
    HEAD_SIZE,
    decode_key_from_head_block,
    decode_key_row,
    decode_value_from_head_block,
    decode_value_row,
    encode_key_row,
    encode_value_row,
    key_base_offset,
    key_code_offset,
    key_head_stride,
    key_step_offset,
    pack_value_idx4,
    unpack_value_idx4,
    value_code_offset,
    value_head_stride,
    value_vmin_offset,
    value_vstep_offset,
)

BLOCK_SIZE = 16
INV_SQRT_D = 1.0 / math.sqrt(HEAD_SIZE)


def _scalar_bytes(value: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([value], dtype=dtype).view(torch.uint8).reshape(-1)


class TestKeyDecode:
    def test_k1_constant_row(self) -> None:
        base = INV_SQRT_D
        step = 0.0
        codes = torch.zeros(HEAD_SIZE, dtype=torch.uint8)
        out = decode_key_row(codes, base, step)
        expected = torch.full((HEAD_SIZE,), INV_SQRT_D)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)

    def test_k2_hand_example_first4(self) -> None:
        dtype = torch.float16
        y = torch.tensor([0.80, -0.20, 0.50, -0.10], dtype=torch.float32)
        code, base, step = encode_key_row(y, dtype=dtype)
        out = decode_key_row(
            torch.cat([code, torch.zeros(HEAD_SIZE - 4, dtype=torch.uint8)]),
            base,
            step,
        )
        torch.testing.assert_close(out[:4], y, atol=1e-2, rtol=1e-2)

    def test_key_roundtrip_random(self) -> None:
        gen = torch.Generator()
        gen.manual_seed(42)
        y = torch.randn(HEAD_SIZE, generator=gen)
        code, base, step = encode_key_row(y)
        out = decode_key_row(code, base, step)
        cos = torch.nn.functional.cosine_similarity(out, y, dim=0)
        assert cos.item() > 0.99


class TestValueDecode:
    def test_v1_linear_idx4(self) -> None:
        vmin = -1.0
        vstep = 0.25
        idx4 = (torch.arange(HEAD_SIZE, dtype=torch.int32) % 16).to(torch.uint8)
        packed = pack_value_idx4(idx4)
        out = decode_value_row(packed, vmin, vstep)
        expected = -1.0 + 0.25 * (torch.arange(HEAD_SIZE) % 16).float()
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_v2_hand_example_first4(self) -> None:
        dtype = torch.float16
        y = torch.tensor([0.80, -0.20, 0.50, -0.10], dtype=torch.float32)
        packed, vmin, vstep = encode_value_row(y, dtype=dtype)
        out = decode_value_row(
            torch.cat([packed, torch.zeros(HEAD_SIZE // 2 - 2, dtype=torch.uint8)]),
            vmin,
            vstep,
        )
        torch.testing.assert_close(out[:4], y, atol=0.06, rtol=0.06)

    def test_unpack_value_idx4(self) -> None:
        idx4 = torch.arange(HEAD_SIZE, dtype=torch.int32) % 16
        packed = pack_value_idx4(idx4.to(torch.uint8))
        recovered = unpack_value_idx4(packed)
        torch.testing.assert_close(recovered, idx4.float())


class TestLayout:
    def test_l_a_single_block_offsets(self) -> None:
        pos = 3
        assert key_code_offset(pos) == 3 * 128
        assert key_base_offset(BLOCK_SIZE, pos) == BLOCK_SIZE * 128 + 3 * 2
        assert key_step_offset(BLOCK_SIZE, pos) == BLOCK_SIZE * (128 + 2) + 3 * 2
        assert value_code_offset(pos) == 3 * 64
        assert value_vmin_offset(BLOCK_SIZE, pos) == BLOCK_SIZE * 64 + 3 * 2
        assert value_vstep_offset(BLOCK_SIZE, pos) == BLOCK_SIZE * (64 + 2) + 3 * 2

    def test_l_c_head_stride(self) -> None:
        assert key_head_stride(BLOCK_SIZE) == BLOCK_SIZE * (128 + 4)
        assert value_head_stride(BLOCK_SIZE) == BLOCK_SIZE * (64 + 4)

    def test_decode_from_head_block(self) -> None:
        dtype = torch.float16
        block = torch.zeros(key_head_stride(BLOCK_SIZE), dtype=torch.uint8)
        pos = 0
        block[key_code_offset(pos) : key_code_offset(pos) + HEAD_SIZE].zero_()
        block[key_base_offset(BLOCK_SIZE, pos) : key_base_offset(BLOCK_SIZE, pos) + 2] = (
            _scalar_bytes(INV_SQRT_D, dtype)
        )
        block[key_step_offset(BLOCK_SIZE, pos) : key_step_offset(BLOCK_SIZE, pos) + 2] = (
            _scalar_bytes(0.0, dtype)
        )
        out = decode_key_from_head_block(block, pos, BLOCK_SIZE, dtype=dtype)
        torch.testing.assert_close(
            out, torch.full((HEAD_SIZE,), INV_SQRT_D), atol=1e-3, rtol=1e-3
        )

        vblock = torch.zeros(value_head_stride(BLOCK_SIZE), dtype=torch.uint8)
        idx4 = (torch.arange(HEAD_SIZE, dtype=torch.int32) % 16).to(torch.uint8)
        vblock[value_code_offset(pos) : value_code_offset(pos) + HEAD_SIZE // 2] = pack_value_idx4(
            idx4
        )
        vblock[value_vmin_offset(BLOCK_SIZE, pos) : value_vmin_offset(BLOCK_SIZE, pos) + 2] = (
            _scalar_bytes(-1.0, dtype)
        )
        vblock[value_vstep_offset(BLOCK_SIZE, pos) : value_vstep_offset(BLOCK_SIZE, pos) + 2] = (
            _scalar_bytes(0.25, dtype)
        )
        vout = decode_value_from_head_block(vblock, pos, BLOCK_SIZE, dtype=dtype)
        expected = -1.0 + 0.25 * (torch.arange(HEAD_SIZE) % 16).float()
        torch.testing.assert_close(vout, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
