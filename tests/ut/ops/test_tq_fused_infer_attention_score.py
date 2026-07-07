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

"""Unit tests for tq_fused_infer_attention_score operator.

This test verifies the Python wrapper and C++ binding for the
tq_fused_infer_attention_score operator.
"""

import pytest
import torch


def test_tq_fused_infer_attention_score_op_availability():
    """Test if the tq_fused_infer_attention_score op is available."""
    # Check if torch.ops._C_ascend exists
    assert hasattr(torch.ops, "_C_ascend"), "torch.ops._C_ascend not found"
    
    # Check if tq_fused_infer_attention_score is registered
    assert hasattr(torch.ops._C_ascend, "tq_fused_infer_attention_score"), \
        "tq_fused_infer_attention_score op not registered"


@pytest.mark.skipif(
    not hasattr(torch.ops, "_C_ascend") or 
    not hasattr(torch.ops._C_ascend, "tq_fused_infer_attention_score"),
    reason="tq_fused_infer_attention_score op not available"
)
def test_tq_fused_infer_attention_score_basic():
    """Test basic functionality of tq_fused_infer_attention_score."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # Create test tensors
    batch_size = 2
    num_heads = 8
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 4
    seq_len = 32
    
    # Query tensor: [num_tokens, num_heads, head_size]
    query = torch.randn(seq_len, num_heads, head_size, dtype=torch.float16, device="npu")
    
    # Key/Value cache: [2, num_blocks, block_size, num_kv_heads, head_size]
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                            dtype=torch.float16, device="npu")
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                              dtype=torch.float16, device="npu")
    
    # Block table: [batch_size, max_num_blocks]
    block_table = torch.randint(0, num_blocks, (batch_size, num_blocks), 
                               dtype=torch.int32, device="npu")
    
    # Attention mask: [batch_size, seq_len, seq_len]
    atten_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.float16, device="npu")
    
    # Actual sequence lengths
    actual_seq_lengths_q = [seq_len] * batch_size
    actual_seq_lengths_kv = [seq_len] * batch_size
    
    scale = 1.0 / (head_size ** 0.5)
    
    # Call the op
    output = torch.ops._C_ascend.tq_fused_infer_attention_score(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        atten_mask=atten_mask,
        actual_seq_len_q=actual_seq_lengths_q,
        actual_seq_len_kv=actual_seq_lengths_kv,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        scale_value=scale,
    )
    
    # Verify output shape
    assert output.shape == query.shape, f"Expected shape {query.shape}, got {output.shape}"
    assert output.dtype == query.dtype, f"Expected dtype {query.dtype}, got {output.dtype}"
    assert output.device == query.device, f"Expected device {query.device}, got {output.device}"


@pytest.mark.skipif(
    not hasattr(torch.ops, "_C_ascend") or 
    not hasattr(torch.ops._C_ascend, "tq_fused_infer_attention_score"),
    reason="tq_fused_infer_attention_score op not available"
)
def test_tq_fused_infer_attention_score_decode_only():
    """Test tq_fused_infer_attention_score in decode-only scenario."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # Decode scenario: single token per request
    batch_size = 4
    num_heads = 8
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 8
    
    # Single token query for decode
    query = torch.randn(batch_size, num_heads, head_size, dtype=torch.float16, device="npu")
    
    # Key/Value cache with history
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                            dtype=torch.float16, device="npu")
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                              dtype=torch.float16, device="npu")
    
    # Block table
    block_table = torch.randint(0, num_blocks, (batch_size, num_blocks), 
                               dtype=torch.int32, device="npu")
    
    # Attention mask for decode (causal)
    atten_mask = torch.ones(batch_size, 1, 1, dtype=torch.float16, device="npu")
    
    # Sequence lengths: q_len=1 for decode, kv_len varies
    actual_seq_lengths_q = [1] * batch_size
    actual_seq_lengths_kv = [10, 20, 15, 25]  # Different history lengths
    
    scale = 1.0 / (head_size ** 0.5)
    
    # Call the op
    output = torch.ops._C_ascend.tq_fused_infer_attention_score(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        atten_mask=atten_mask,
        actual_seq_len_q=actual_seq_lengths_q,
        actual_seq_len_kv=actual_seq_lengths_kv,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        scale_value=scale,
    )
    
    # Verify output shape: [batch_size, num_heads, head_size]
    expected_shape = (batch_size, num_heads, head_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


@pytest.mark.skipif(
    not hasattr(torch.ops, "_C_ascend") or 
    not hasattr(torch.ops._C_ascend, "tq_fused_infer_attention_score"),
    reason="tq_fused_infer_attention_score op not available"
)
def test_tq_fused_infer_attention_score_gqa():
    """Test tq_fused_infer_attention_score with Grouped Query Attention (GQA)."""
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    
    # GQA scenario: num_heads > num_kv_heads
    batch_size = 2
    num_heads = 32  # Query heads
    num_kv_heads = 8  # KV heads (GQA group = 4)
    head_size = 128
    block_size = 16
    num_blocks = 4
    seq_len = 16
    
    # Query tensor
    query = torch.randn(seq_len, num_heads, head_size, dtype=torch.float16, device="npu")
    
    # Key/Value cache with fewer heads
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                            dtype=torch.float16, device="npu")
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                              dtype=torch.float16, device="npu")
    
    # Block table
    block_table = torch.randint(0, num_blocks, (batch_size, num_blocks), 
                               dtype=torch.int32, device="npu")
    
    # Attention mask
    atten_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.float16, device="npu")
    
    # Sequence lengths
    actual_seq_lengths_q = [seq_len] * batch_size
    actual_seq_lengths_kv = [seq_len] * batch_size
    
    scale = 1.0 / (head_size ** 0.5)
    
    # Call the op
    output = torch.ops._C_ascend.tq_fused_infer_attention_score(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        atten_mask=atten_mask,
        actual_seq_len_q=actual_seq_lengths_q,
        actual_seq_len_kv=actual_seq_lengths_kv,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        scale_value=scale,
    )
    
    # Verify output shape
    assert output.shape == query.shape, f"Expected shape {query.shape}, got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
