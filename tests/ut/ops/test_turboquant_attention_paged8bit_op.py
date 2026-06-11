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

"""Correctness tests for TurboQuant Scheme B fused attention.

The custom op consumes packed uint8 PA K/V cache directly. The golden path uses
the PyTorch TurboQuant dequantizer to reconstruct fp16 K/V rows and then computes
decode attention with ``torch_npu.npu_fused_infer_attention_score``. Tests are
skipped unless an Ascend NPU and the ``turboquant_attention_paged8bit`` custom op
are available.
"""

from __future__ import annotations

import pytest
import torch

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover
    torch_npu = None

from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    _get_quantizer,
    turboquant_dequantize_from_packed_bytes,
    turboquant_quantize_to_packed_bytes,
)

HEAD_SIZE = 128
BLOCK_SIZE = 16
BITS = 8
FIA_ATTEN_MASK_SIZE = 2048

try:
    NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception:
    NPU_AVAILABLE = False

OP_AVAILABLE = (
    NPU_AVAILABLE
    and torch_npu is not None
    and _c_ascend_turboquant_op_available("turboquant_attention_paged8bit")
)


def _build_block_table(
    actual_seq_lens_kv: list[int],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [(length + block_size - 1) // block_size for length in actual_seq_lens_kv]
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq)

    block_table = torch.zeros(
        (len(actual_seq_lens_kv), max_blocks), device=device, dtype=torch.int32
    )
    next_block = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        ids = torch.arange(
            next_block, next_block + block_count, device=device, dtype=torch.int32
        )
        block_table[seq_idx, :block_count] = ids
        next_block += block_count
    return block_table, total_blocks


def _build_packed_cache(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    shape = (num_blocks, block_size, num_kv_heads, HEAD_SIZE)
    key = torch.randn(shape, dtype=torch.float16, device=device)
    value = torch.randn(shape, dtype=torch.float16, device=device)
    return (
        turboquant_quantize_to_packed_bytes(key, bits=BITS).contiguous(),
        turboquant_quantize_to_packed_bytes(value, bits=BITS).contiguous(),
    )


def _npu_fia_turboquant_attention_golden(
    *,
    query: torch.Tensor,
    key_cache_packed: torch.Tensor,
    value_cache_packed: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    block_size: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    key_dec = turboquant_dequantize_from_packed_bytes(
        key_cache_packed, head_size=HEAD_SIZE, dtype=query.dtype, bits=BITS
    )
    value_dec = turboquant_dequantize_from_packed_bytes(
        value_cache_packed, head_size=HEAD_SIZE, dtype=query.dtype, bits=BITS
    )

    num_tokens, num_heads, _ = query.shape
    atten_mask = torch.triu(
        torch.ones(
            FIA_ATTEN_MASK_SIZE,
            FIA_ATTEN_MASK_SIZE,
            dtype=torch.int8,
            device=query.device,
        ),
        diagonal=1,
    )
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key_dec.flatten(2, 3).contiguous(),
        value=value_dec.flatten(2, 3).contiguous(),
        atten_mask=atten_mask,
        block_table=block_table.contiguous(),
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=actual_seq_lens_q,
        actual_seq_lengths_kv=actual_seq_lens_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=3,
    )
    return attn_output.view_as(query)


def _run_custom_op(
    *,
    query: torch.Tensor,
    key_cache_packed: torch.Tensor,
    value_cache_packed: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lens_q: list[int],
    actual_seq_lens_kv: list[int],
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    quantizer = _get_quantizer(HEAD_SIZE, BITS, query.device)
    codebook = quantizer.codebook.to(device=query.device, dtype=torch.float16)
    rotation = quantizer.rotation.to(device=query.device, dtype=torch.float16)
    actual_seq_len_q = torch.tensor(
        actual_seq_lens_q, device=query.device, dtype=torch.int64
    )
    actual_seq_len_kv = torch.tensor(
        actual_seq_lens_kv, device=query.device, dtype=torch.int64
    )
    return torch.ops._C_ascend.turboquant_attention_paged8bit(
        query.contiguous(),
        key_cache_packed.contiguous(),
        value_cache_packed.contiguous(),
        block_table.contiguous(),
        actual_seq_len_q,  # Python list
        actual_seq_lens_kv,  # Python list
        codebook,
        rotation,
        codebook,
        rotation,
        num_heads,
        num_kv_heads,
        HEAD_SIZE,
        block_size,
        max(actual_seq_lens_kv),
        float(scale),
    )


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + turboquant_attention_paged8bit")
@pytest.mark.parametrize(
    "num_tokens,num_heads,num_kv_heads,actual_seq_lens_kv",
    [
        # SplitBN: enough (token, kv_head) work; G=8 also exercises Cube tiling selection.
        (4, 8, 1, [17, 23, 31, 11]),
        # Vector QK/PV fallback: G=1, validates MHA/MQA-like skinny QK.
        (2, 4, 4, [19, 7]),
    ],
)
def test_attention_paged8bit_matches_fia_splitbn(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    actual_seq_lens_kv: list[int],
):
    device = torch.device("npu")
    scale = HEAD_SIZE**-0.5
    block_table, total_blocks = _build_block_table(
        actual_seq_lens_kv, BLOCK_SIZE, device
    )
    key_packed, value_packed = _build_packed_cache(
        num_blocks=total_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        device=device,
        seed=0,
    )
    torch.manual_seed(1)
    query = torch.randn(
        (num_tokens, num_heads, HEAD_SIZE), dtype=torch.float16, device=device
    )

    actual_seq_lens_q = list(range(1, num_tokens + 1))
    golden = _npu_fia_turboquant_attention_golden(
        query=query,
        key_cache_packed=key_packed,
        value_cache_packed=value_packed,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    actual = _run_custom_op(
        query=query,
        key_cache_packed=key_packed,
        value_cache_packed=value_packed,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_size=BLOCK_SIZE,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )

    torch.testing.assert_close(actual.float(), golden.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + turboquant_attention_paged8bit")
def test_attention_paged8bit_matches_fia_splitbns_flashdecode():
    """Small BN + long KV triggers SplitBNS; compare final combined output."""
    device = torch.device("npu")
    num_tokens = 1
    num_heads = 8
    num_kv_heads = 1
    actual_seq_lens_kv = [2051]
    scale = HEAD_SIZE**-0.5

    block_table, total_blocks = _build_block_table(
        actual_seq_lens_kv, BLOCK_SIZE, device
    )
    key_packed, value_packed = _build_packed_cache(
        num_blocks=total_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        device=device,
        seed=2,
    )
    torch.manual_seed(3)
    query = torch.randn(
        (num_tokens, num_heads, HEAD_SIZE), dtype=torch.float16, device=device
    )

    actual_seq_lens_q = list(range(1, num_tokens + 1))
    golden = _npu_fia_turboquant_attention_golden(
        query=query,
        key_cache_packed=key_packed,
        value_cache_packed=value_packed,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    actual = _run_custom_op(
        query=query,
        key_cache_packed=key_packed,
        value_cache_packed=value_packed,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_size=BLOCK_SIZE,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )

    torch.testing.assert_close(actual.float(), golden.float(), atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not OP_AVAILABLE, reason="requires NPU + turboquant_attention_paged8bit")
def test_attention_paged8bit_mixed_decode_prefill_batch():
    """ChunkedPrefill-like mixed batch from test_sampler2 (Qwen3-0.6B layout).

    seq0: 1 decode token with kv_len=6
    seq1: 4 prefill tokens with kv_len=4
    total tokens=5, bn=40, block_size=128 (chunked prefill)
    """
    device = torch.device("npu")
    num_heads = 16
    num_kv_heads = 8
    block_size = 128
    actual_seq_lens_kv = [6, 4]
    actual_seq_lens_q = [1, 5]
    num_tokens = actual_seq_lens_q[-1]
    scale = HEAD_SIZE**-0.5

    block_table, total_blocks = _build_block_table(
        actual_seq_lens_kv, block_size, device
    )
    key_packed, value_packed = _build_packed_cache(
        num_blocks=total_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        device=device,
        seed=42,
    )
    torch.manual_seed(43)
    query = torch.randn(
        (num_tokens, num_heads, HEAD_SIZE), dtype=torch.float16, device=device
    )

    golden = _npu_fia_turboquant_attention_golden(
        query=query,
        key_cache_packed=key_packed,
        value_cache_packed=value_packed,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )
    actual = _run_custom_op(
        query=query,
        key_cache_packed=key_packed,
        value_cache_packed=value_packed,
        block_table=block_table,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
    )

    torch.testing.assert_close(actual.float(), golden.float(), atol=4e-2, rtol=4e-2)
