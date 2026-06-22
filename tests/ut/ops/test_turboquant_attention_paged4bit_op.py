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

"""Correctness tests for TurboQuant 4-bit slab paged attention."""

from __future__ import annotations

import math

import pytest
import torch

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    _turboquant_fused_8bit_decode_tables,
    _turboquant_pack_tables,
    refresh_turboquant_env_cache,
    turboquant_attention_paged4bit,
)

HEAD_SIZE = 128
BLOCK_SIZE = 16
ROW_BYTES = HEAD_SIZE // 2 + 2
FY_LINEAR = 0.020799
FY_CUBIC = 0.0001926

try:
    NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception:
    NPU_AVAILABLE = False

OP_AVAILABLE = (
    NPU_AVAILABLE
    and torch_npu is not None
    and _c_ascend_turboquant_op_available("turboquant_attention_paged4bit")
)


def _build_block_table(
    actual_seq_lens_kv: list[int],
    block_size: int,
) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [
        (length + block_size - 1) // block_size
        for length in actual_seq_lens_kv
    ]
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq)
    block_table = torch.zeros((len(actual_seq_lens_kv), max_blocks), dtype=torch.int32)
    next_block = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        block_table[seq_idx, :block_count] = torch.arange(
            next_block, next_block + block_count, dtype=torch.int32
        )
        next_block += block_count
    return block_table, total_blocks


def _build_slab_cache(
    *,
    num_blocks: int,
    num_kv_heads: int,
    seed: int,
    norm_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        16,
        (num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE),
        generator=generator,
        dtype=torch.uint8,
    )
    norms = (
        torch.rand(
            (num_blocks, num_kv_heads, BLOCK_SIZE),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
        + 0.25
    ).to(norm_dtype)
    groups_per_block = BLOCK_SIZE // 4
    group_bytes = ROW_BYTES * 4
    cache_group = torch.zeros(
        (num_blocks, num_kv_heads, groups_per_block, group_bytes),
        dtype=torch.uint8,
    )
    words = torch.zeros(
        (num_blocks, num_kv_heads, groups_per_block, HEAD_SIZE),
        dtype=torch.int32,
    )
    for group_row in range(4):
        idx = indices[:, group_row::4, :, :].permute(0, 2, 1, 3).to(torch.int32)
        words |= (idx & 0x0F) << (group_row * 4)
    cache_group[..., : HEAD_SIZE * 2 : 2] = (words & 0xFF).to(torch.uint8)
    cache_group[..., 1 : HEAD_SIZE * 2 : 2] = ((words >> 8) & 0xFF).to(torch.uint8)

    norm_bytes = norms.view(torch.uint8).reshape(num_blocks, num_kv_heads, BLOCK_SIZE, 2)
    for group_row in range(4):
        cache_group[
            ...,
            HEAD_SIZE * 2 + group_row * 2 : HEAD_SIZE * 2 + (group_row + 1) * 2,
        ] = norm_bytes[:, :, group_row::4, :]
    return cache_group.reshape(num_blocks, num_kv_heads, BLOCK_SIZE * ROW_BYTES)


def _decode_slab_row(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: int,
    kv_head: int,
    abs_pos: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_id = int(block_table[seq_idx, abs_pos // BLOCK_SIZE])
    pos_in_block = abs_pos % BLOCK_SIZE
    group_idx = pos_in_block // 4
    group_row = pos_in_block % 4
    group_base = group_idx * ROW_BYTES * 4
    group = cache[block_id, kv_head, group_base : group_base + ROW_BYTES * 4]
    raw = group[: HEAD_SIZE * 2]
    words = raw[0::2].long() | (raw[1::2].long() << 8)
    indices = ((words >> (group_row * 4)) & 0x0F).long()
    norm_base = HEAD_SIZE * 2 + group_row * 2
    norm = group[norm_base : norm_base + 2].contiguous().view(dtype).float()[0]
    x = indices.float() - 7.5
    y = (FY_CUBIC * x * x * x + FY_LINEAR * x).to(dtype).float()
    return (y * norm).to(dtype)


def _golden_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    q_rotation: torch.Tensor,
    out_rotation: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    out = torch.empty_like(query, dtype=torch.float32)
    dtype = query.dtype
    gqa_group = num_heads // num_kv_heads
    for token_idx in range(query.shape[0]):
        seq_idx = next(i for i, end in enumerate(actual_seq_lens_q) if token_idx < end)
        q_start = 0 if seq_idx == 0 else actual_seq_lens_q[seq_idx - 1]
        num_q_in_seq = actual_seq_lens_q[seq_idx] - q_start
        q_pos = token_idx - q_start
        causal_kv_end = actual_seq_lens_kv[seq_idx] - num_q_in_seq + q_pos + 1
        for kv_head in range(num_kv_heads):
            keys = torch.stack(
                [
                    _decode_slab_row(
                        key_cache,
                        block_table,
                        seq_idx,
                        kv_head,
                        pos,
                        dtype,
                    )
                    for pos in range(causal_kv_end)
                ],
                dim=0,
            )
            values = torch.stack(
                [
                    _decode_slab_row(
                        value_cache,
                        block_table,
                        seq_idx,
                        kv_head,
                        pos,
                        dtype,
                    )
                    for pos in range(causal_kv_end)
                ],
                dim=0,
            )
            for group_idx in range(gqa_group):
                head_idx = kv_head * gqa_group + group_idx
                query_rot = query[token_idx, head_idx].float() @ q_rotation.float()
                scores = (keys.float() * query_rot).sum(dim=-1) * scale
                out_y = torch.softmax(scores, dim=-1) @ values.float()
                out[token_idx, head_idx] = (
                    out_y.to(dtype).float() @ out_rotation.float()
                ).to(dtype).float()
    return out


def _run_custom_op(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    cb_k, rot_t_k = _turboquant_pack_tables(
        device=query.device,
        head_size=HEAD_SIZE,
        bits=4,
        dtype=query.dtype,
    )
    _, _, _, rot_v = _turboquant_fused_8bit_decode_tables(
        device=query.device,
        head_size=HEAD_SIZE,
        bits_key=4,
        bits_value=4,
        dtype=query.dtype,
    )
    return torch.ops._C_ascend.turboquant_attention_paged4bit(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        actual_seq_lens_q,
        actual_seq_lens_kv,
        cb_k.contiguous(),
        rot_t_k.contiguous(),
        cb_k.contiguous(),
        rot_v.contiguous(),
        num_heads,
        num_kv_heads,
        HEAD_SIZE,
        BLOCK_SIZE,
        max(actual_seq_lens_kv),
        float(scale),
    )


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + turboquant_attention_paged4bit")
def test_attention_paged4bit_wrapper_accepts_large_gqa_group(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_TURBOQUANT_DECODE_OP", "1")
    monkeypatch.setenv("VLLM_ASCEND_TURBOQUANT_MSE_IMPL", "v1")
    refresh_turboquant_env_cache()
    try:
        device = torch.device("npu")
        num_tokens = 4
        num_heads = 16
        num_kv_heads = 1
        actual_seq_lens_q = [num_tokens]
        actual_seq_lens_kv = [16]
        scale = 1.0 / math.sqrt(HEAD_SIZE)
        block_table_cpu, total_blocks = _build_block_table(
            actual_seq_lens_kv,
            BLOCK_SIZE,
        )
        key_cache_cpu = _build_slab_cache(
            num_blocks=total_blocks,
            num_kv_heads=num_kv_heads,
            seed=29,
        )
        value_cache_cpu = _build_slab_cache(
            num_blocks=total_blocks,
            num_kv_heads=num_kv_heads,
            seed=129,
        )
        query = torch.randn(
            (num_tokens, num_heads, HEAD_SIZE),
            dtype=torch.float16,
            device=device,
        )

        out = turboquant_attention_paged4bit(
            query=query,
            key_cache=key_cache_cpu.to(device),
            value_cache=value_cache_cpu.to(device),
            block_tables=block_table_cpu.to(device),
            actual_seq_lengths_q=actual_seq_lens_q,
            actual_seq_lengths_kv=actual_seq_lens_kv,
            head_size=HEAD_SIZE,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            block_size=BLOCK_SIZE,
            scale=scale,
        )
        assert out is not None
        assert out.shape == query.shape
    finally:
        monkeypatch.undo()
        refresh_turboquant_env_cache()


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + turboquant_attention_paged4bit")
def test_attention_paged4bit_long_decode_matches_cpu_reference():
    device = torch.device("npu")
    num_tokens = 1
    num_heads = 8
    num_kv_heads = 1
    actual_seq_lens_q = [num_tokens]
    actual_seq_lens_kv = [1024]
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    block_table_cpu, total_blocks = _build_block_table(
        actual_seq_lens_kv,
        BLOCK_SIZE,
    )
    key_cache_cpu = _build_slab_cache(
        num_blocks=total_blocks,
        num_kv_heads=num_kv_heads,
        seed=31,
    )
    value_cache_cpu = _build_slab_cache(
        num_blocks=total_blocks,
        num_kv_heads=num_kv_heads,
        seed=131,
    )
    query = torch.randn(
        (num_tokens, num_heads, HEAD_SIZE),
        dtype=torch.float16,
        device=device,
    )
    _, rot_t_k = _turboquant_pack_tables(
        device=device,
        head_size=HEAD_SIZE,
        bits=4,
        dtype=torch.float16,
    )
    _, _, _, rot_v = _turboquant_fused_8bit_decode_tables(
        device=device,
        head_size=HEAD_SIZE,
        bits_key=4,
        bits_value=4,
        dtype=torch.float16,
    )

    actual = _run_custom_op(
        query=query,
        key_cache=key_cache_cpu.to(device),
        value_cache=value_cache_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    golden = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache_cpu,
        value_cache=value_cache_cpu,
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        q_rotation=rot_t_k.cpu(),
        out_rotation=rot_v.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )

    torch.testing.assert_close(actual.float().cpu(), golden, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + turboquant_attention_paged4bit")
@pytest.mark.parametrize(
    "num_tokens,num_heads,num_kv_heads,actual_seq_lens_q,actual_seq_lens_kv,dtype",
    [
        (4, 4, 1, [4], [16], torch.float16),
        (5, 4, 2, [3, 5], [20, 9], torch.float16),
        (20, 4, 1, [20], [32], torch.float16),
        (4, 16, 1, [4], [16], torch.float16),
        (20, 16, 1, [20], [32], torch.float16),
        (4, 4, 1, [4], [16], torch.bfloat16),
        (5, 4, 2, [3, 5], [20, 9], torch.bfloat16),
        (4, 16, 1, [4], [16], torch.bfloat16),
    ],
)
def test_attention_paged4bit_matches_cpu_reference(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    dtype: torch.dtype,
):
    device = torch.device("npu")
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    block_table_cpu, total_blocks = _build_block_table(
        actual_seq_lens_kv,
        BLOCK_SIZE,
    )
    key_cache_cpu = _build_slab_cache(
        num_blocks=total_blocks,
        num_kv_heads=num_kv_heads,
        seed=17,
        norm_dtype=dtype,
    )
    value_cache_cpu = _build_slab_cache(
        num_blocks=total_blocks,
        num_kv_heads=num_kv_heads,
        seed=117,
        norm_dtype=dtype,
    )
    torch.manual_seed(1)
    query = torch.randn(
        (num_tokens, num_heads, HEAD_SIZE), dtype=dtype, device=device
    )
    _, rot_t_k = _turboquant_pack_tables(
        device=device,
        head_size=HEAD_SIZE,
        bits=4,
        dtype=dtype,
    )
    _, _, _, rot_v = _turboquant_fused_8bit_decode_tables(
        device=device,
        head_size=HEAD_SIZE,
        bits_key=4,
        bits_value=4,
        dtype=dtype,
    )

    actual = _run_custom_op(
        query=query,
        key_cache=key_cache_cpu.to(device),
        value_cache=value_cache_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    golden = _golden_attention(
        query=query.cpu(),
        key_cache=key_cache_cpu,
        value_cache=value_cache_cpu,
        block_table=block_table_cpu,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        q_rotation=rot_t_k.cpu(),
        out_rotation=rot_v.cpu(),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )

    torch.testing.assert_close(actual.float().cpu(), golden, atol=1e-3, rtol=1e-3)
