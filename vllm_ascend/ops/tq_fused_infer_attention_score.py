"""
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
the CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import torch


def tq_fused_infer_attention_score(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    atten_mask: torch.Tensor | None,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    head_size: int,
    num_heads: int,
    num_key_value_heads: int,
    block_size: int,
    scale: float,
) -> torch.Tensor | None:
    """Call the fused TqFusedInferAttentionScore op when supported.

    This is a wrapper for the TqFusedInferAttentionScore operator which supports
    both decode and prefill scenarios with paged attention.

    Returns ``None`` when the custom op is unavailable, so callers can keep
    existing fallbacks.
    """
    # Check basic constraints
    if (
        head_size <= 0
        or num_heads <= 0
        or num_key_value_heads <= 0
        or num_heads % num_key_value_heads != 0
        or block_size <= 0
        or block_tables.numel() == 0
    ):
        return None

    # Check if the op is available
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(
        torch.ops._C_ascend, "tq_fused_infer_attention_score"
    ):
        return None

    # Validate sequence lengths
    actual_seq_lengths_q = [int(length) for length in actual_seq_lengths_q]
    actual_seq_lengths_kv = [int(length) for length in actual_seq_lengths_kv]

    if (
        not actual_seq_lengths_q
        or len(actual_seq_lengths_q) != len(actual_seq_lengths_kv)
        or actual_seq_lengths_q[-1] != query.shape[0]
    ):
        return None

    if any(length < 0 for length in actual_seq_lengths_kv):
        return None

    # Check monotonicity for actual_seq_lengths_q
    if any(
        current < previous
        for previous, current in zip(actual_seq_lengths_q, actual_seq_lengths_q[1:])
    ):
        return None

    try:
        fused = torch.ops._C_ascend.tq_fused_infer_attention_score
        out = fused(
            query=query.contiguous(),
            key_cache=key_cache.contiguous(),
            value_cache=value_cache.contiguous(),
            block_tables=block_tables.contiguous(),
            atten_mask=atten_mask,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            block_size=block_size,
            scale_value=scale,
        )
        return out
    except Exception as e:
        # Log the exception if needed
        import logging
        logging.warning(f"TqFusedInferAttentionScore op failed: {e}")
        return None
