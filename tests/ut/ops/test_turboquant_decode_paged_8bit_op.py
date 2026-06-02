#
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

"""Tests for the 8-bit paged decode op (design doc §2.6.8).

The host index computation (``turboquant_paged_decode_host_indices``) is pure
tensor math and runs on CPU. The kernel numeric golden (D1/D8) requires NPU and
the compiled custom op; those are skipped without ``torch.npu`` / the op.
"""

import os

import pytest
import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    turboquant_dequantize_from_packed_bytes,
    turboquant_paged_decode_host_indices,
    turboquant_quantize_to_packed_bytes,
)

HEAD_SIZE = 128
BLOCK_SIZE = 128

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
OP_AVAILABLE = NPU_AVAILABLE and _c_ascend_turboquant_op_available(
    "turboquant_decode_paged_8bit"
)


# --------------------------------------------------------------------------- #
# Host index computation (CPU) — design doc §2.6.2 / §2.6.8 D2, D4, D5         #
# --------------------------------------------------------------------------- #
def test_host_indices_d2_compact_layout():
    """D2: batch=3, num_blocks=[2,1,3] -> exact gather order + bt_compact."""
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],  # seq0: only first 2 live
            [20, 21, 22, 23],  # seq1: only first 1 live
            [30, 31, 32, 33],  # seq2: first 3 live
        ],
        dtype=torch.int32,
    )
    actual_kv = [200, 50, 350]  # ceil/128 = [2, 1, 3]

    gather, bt_compact, total = turboquant_paged_decode_host_indices(
        block_table, actual_kv, BLOCK_SIZE
    )

    assert total == 6
    # Live blocks of each seq, concatenated in seq order.
    assert gather.tolist() == [10, 11, 20, 30, 31, 32]
    assert gather.dtype == torch.int32
    # Valid -> block_offsets[s] + j; padding -> block_offsets[s].
    assert bt_compact.tolist() == [
        [0, 1, 0, 0],
        [2, 2, 2, 2],
        [3, 4, 5, 3],
    ]
    assert bt_compact.dtype == block_table.dtype


def test_host_indices_single_seq_single_block():
    """D1: batch=1, one block."""
    block_table = torch.tensor([[7]], dtype=torch.int32)
    gather, bt_compact, total = turboquant_paged_decode_host_indices(
        block_table, [100], BLOCK_SIZE
    )
    assert total == 1
    assert gather.tolist() == [7]
    assert bt_compact.tolist() == [[0]]


def test_host_indices_kvlen_non_multiple_of_block():
    """D3: kvLen not a multiple of block_size -> ceil block count."""
    block_table = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    # 300 -> ceil(300/128) = 3 blocks (last block partial, still counted).
    gather, bt_compact, total = turboquant_paged_decode_host_indices(
        block_table, [300], BLOCK_SIZE
    )
    assert total == 3
    assert gather.tolist() == [5, 6, 7]
    assert bt_compact.tolist() == [[0, 1, 2, 0]]


def test_host_indices_shared_physical_block():
    """D4: a physical block referenced by multiple seqs appears once per ref."""
    block_table = torch.tensor(
        [
            [9, 0, 0, 0],
            [9, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    gather, bt_compact, total = turboquant_paged_decode_host_indices(
        block_table, [10, 10], BLOCK_SIZE
    )
    assert total == 2
    # Both seqs reference physical block 9 -> decoded twice (arange-style), each
    # seq pointing at its own compact slot.
    assert gather.tolist() == [9, 9]
    assert bt_compact.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1]]


def test_host_indices_padding_indices_in_range():
    """D5: every bt_compact entry indexes a valid workspace slot [0, total)."""
    block_table = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    actual_kv = [200, 128, 600]  # [2, 1, 5]
    gather, bt_compact, total = turboquant_paged_decode_host_indices(
        block_table, actual_kv, BLOCK_SIZE
    )
    assert total == 8
    assert gather.numel() == total
    assert int(bt_compact.min()) >= 0
    assert int(bt_compact.max()) < total


def test_host_indices_tensor_seq_lengths():
    """actual_seq_lengths_kv accepts a tensor as well as a list."""
    block_table = torch.tensor([[3, 4]], dtype=torch.int32)
    g_list, bt_list, t_list = turboquant_paged_decode_host_indices(
        block_table, [256], BLOCK_SIZE
    )
    g_t, bt_t, t_t = turboquant_paged_decode_host_indices(
        block_table, torch.tensor([256], dtype=torch.int32), BLOCK_SIZE
    )
    assert t_list == t_t == 2
    assert g_list.tolist() == g_t.tolist()
    assert bt_list.tolist() == bt_t.tolist()


# --------------------------------------------------------------------------- #
# Kernel numeric golden (NPU) — design doc §2.6.8 D1 / D8                      #
# --------------------------------------------------------------------------- #
def _build_packed_cache(num_blocks, num_kv_heads, device, seed=0):
    """Random fp16 K/V -> 8-bit packed uint8 cache [num_blocks, BS, H, 130]."""
    torch.manual_seed(seed)
    shape = (num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
    key = torch.randn(shape, dtype=torch.float16, device=device)
    value = torch.randn(shape, dtype=torch.float16, device=device)
    key_packed = turboquant_quantize_to_packed_bytes(key, bits=8)
    value_packed = turboquant_quantize_to_packed_bytes(value, bits=8)
    return key_packed, value_packed


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + compiled decode op")
@pytest.mark.parametrize("mode", [0, 1])
def test_decode_paged_op_matches_pytorch_golden(mode):
    """D1/D8: op output (per compact block) vs PyTorch dequantize, both layouts."""
    os.environ["VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT"] = "1"
    os.environ["VLLM_ASCEND_TURBOQUANT_DECODE_OP_8BIT_MODE"] = str(mode)
    device = torch.device("npu")
    num_kv_heads = 8
    num_blocks = 6

    key_packed, value_packed = _build_packed_cache(num_blocks, num_kv_heads, device)
    # block_table referencing a subset (with one shared block) in seq order.
    block_table = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.int32, device=device)
    actual_kv = [3 * BLOCK_SIZE, 2 * BLOCK_SIZE]

    gather, bt_compact, total = turboquant_paged_decode_host_indices(
        block_table, actual_kv, BLOCK_SIZE
    )

    from vllm_ascend.ops.turboquant_kv_cache import _get_quantizer

    quantizer = _get_quantizer(HEAD_SIZE, 8, device)
    codebook = quantizer.codebook.to(device=device, dtype=torch.float16)
    rotation = quantizer.rotation.to(device=device, dtype=torch.float16)

    key_out, value_out = torch.ops._C_ascend.turboquant_decode_paged_8bit(
        key_packed.contiguous(),
        value_packed.contiguous(),
        gather.contiguous(),
        codebook,
        rotation,
        HEAD_SIZE,
        BLOCK_SIZE,
        0,
        mode,
    )

    # Golden: PyTorch dequantize of each gathered physical block, in compact order.
    gathered_key = key_packed.index_select(0, gather.to(torch.int64))
    gathered_value = value_packed.index_select(0, gather.to(torch.int64))
    key_golden = turboquant_dequantize_from_packed_bytes(
        gathered_key, head_size=HEAD_SIZE, dtype=torch.float16, bits=8
    )
    value_golden = turboquant_dequantize_from_packed_bytes(
        gathered_value, head_size=HEAD_SIZE, dtype=torch.float16, bits=8
    )

    assert key_out.shape == key_golden.shape
    torch.testing.assert_close(
        key_out.float(), key_golden.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        value_out.float(), value_golden.float(), atol=2e-2, rtol=2e-2
    )
