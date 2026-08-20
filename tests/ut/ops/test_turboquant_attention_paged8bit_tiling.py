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

"""Unit tests for TurboQuant Scheme B matmul tiling strategy."""

import pytest

from vllm_ascend.ops.turboquant_attention_tiling import (
    TQ_ATTN_QKPV_CUBE,
    TQ_ATTN_QKPV_VECTOR,
    TQ_ATTN_SPLIT_BN,
    TQ_ATTN_SPLIT_BNS,
    compute_turboquant_attention_tiling,
    is_flash_decode,
    pick_kv_tile_rows,
    pick_qk_pv_mode,
    split_bn,
    split_bns,
)


def test_pick_kv_tile_rows_aligns_block_size():
    assert pick_kv_tile_rows(16) == 32
    assert pick_kv_tile_rows(32) == 32
    # PA block_size=128 exceeds UB cap; kernel tiles KV in 32-row chunks.
    assert pick_kv_tile_rows(128) == 32


def test_is_flash_decode_small_batch_long_kv():
    # batch=1, kv_heads=4 -> bn=4, aic=24 -> flash decode eligible when kv>=2048
    assert is_flash_decode(1, 4, 4096, 24) is True
    assert is_flash_decode(16, 4, 4096, 24) is False


def test_pick_qk_pv_mode_gating():
    assert pick_qk_pv_mode(1, 32) == TQ_ATTN_QKPV_VECTOR
    assert pick_qk_pv_mode(8, 32) == TQ_ATTN_QKPV_CUBE
    assert pick_qk_pv_mode(16, 16) == TQ_ATTN_QKPV_VECTOR


def test_split_bn_large_batch():
    used, starts = split_bn(bn=64, core_num=24)
    assert used == 24
    assert starts[0] == 0
    assert len(starts) == 24


def test_split_bns_small_batch():
    used, kv_split, seg_len, starts = split_bns(bn=4, max_kv_len=4096, block_size=16, core_num=24)
    assert used == 4 * kv_split
    assert kv_split == 6
    assert seg_len % 16 == 0
    assert len(starts) == used


def test_compute_tiling_split_bn_decode_batch():
    t = compute_turboquant_attention_tiling(
        num_tokens=16,
        num_heads=32,
        num_kv_heads=4,
        block_size=16,
        max_kv_len=1024,
        aic_num=24,
    )
    assert t["split_mode"] == TQ_ATTN_SPLIT_BN
    assert t["gqa_group"] == 8
    assert t["qk_pv_mode"] == TQ_ATTN_QKPV_CUBE
    assert t["tiling_key"] == 1


def test_compute_tiling_split_bns_flash_decode():
    t = compute_turboquant_attention_tiling(
        num_tokens=1,
        num_heads=32,
        num_kv_heads=4,
        block_size=16,
        max_kv_len=4096,
        aic_num=24,
    )
    assert t["split_mode"] == TQ_ATTN_SPLIT_BNS
    assert t["kv_split_part"] == 6
    assert t["tiling_key"] in (2, 3)


def test_compute_tiling_uses_actual_seq_len_for_flash_decode():
    t = compute_turboquant_attention_tiling(
        num_tokens=1,
        num_heads=32,
        num_kv_heads=4,
        block_size=16,
        max_kv_len=4096,
        max_actual_seq_len=512,
        aic_num=24,
    )
    assert t["max_kv_len"] == 4096
    assert t["max_actual_seq_len"] == 512
    assert t["split_mode"] == TQ_ATTN_SPLIT_BN


def test_compute_tiling_mqa_vector_qkpv():
    t = compute_turboquant_attention_tiling(
        num_tokens=8,
        num_heads=8,
        num_kv_heads=8,
        block_size=16,
        max_kv_len=512,
        aic_num=24,
    )
    assert t["gqa_group"] == 1
    assert t["qk_pv_mode"] == TQ_ATTN_QKPV_VECTOR
    assert t["tiling_key"] == 0
