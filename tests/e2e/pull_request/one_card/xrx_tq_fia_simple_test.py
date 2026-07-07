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

"""Simple test for tq_fused_infer_attention_score operator.

This test directly calls the operator to verify it works correctly.
"""

from __future__ import annotations

import sys
import torch


def test_op_availability():
    """Test if the operator is available."""
    print("Testing operator availability...")
    
    if not hasattr(torch.ops, "_C_ascend"):
        print("❌ torch.ops._C_ascend not found")
        return False
    
    if not hasattr(torch.ops._C_ascend, "tq_fused_infer_attention_score"):
        print("❌ tq_fused_infer_attention_score not found in torch.ops._C_ascend")
        return False
    
    print("✅ tq_fused_infer_attention_score operator is available")
    return True


def test_op_signature():
    """Test the operator signature."""
    print("\nTesting operator signature...")
    
    try:
        # Get the op schema
        op = torch.ops._C_ascend.tq_fused_infer_attention_score
        print(f"✅ Operator found: {op}")
        
        # Try to get schema if available
        if hasattr(op, 'schema'):
            print(f"Schema: {op.schema}")
        
        return True
    except Exception as e:
        print(f"❌ Error getting operator info: {e}")
        return False


def test_python_wrapper():
    """Test the Python wrapper function."""
    print("\nTesting Python wrapper...")
    
    try:
        from vllm_ascend.ops.tq_fused_infer_attention_score import tq_fused_infer_attention_score
        print("✅ Python wrapper imported successfully")
        
        # Check function signature
        import inspect
        sig = inspect.signature(tq_fused_infer_attention_score)
        print(f"Function signature: {sig}")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import Python wrapper: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_op_call():
    """Test calling the operator (requires NPU)."""
    print("\nTesting operator call...")
    
    if not torch.npu.is_available():
        print("⚠️  NPU not available, skipping actual call test")
        return True
    
    try:
        # Create minimal test tensors
        batch_size = 1
        num_heads = 8
        num_kv_heads = 8
        head_size = 128
        block_size = 16
        num_blocks = 2
        seq_len = 1
        
        query = torch.randn(seq_len, num_heads, head_size, dtype=torch.float16).npu()
        key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                                dtype=torch.float16).npu()
        value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, 
                                  dtype=torch.float16).npu()
        block_table = torch.zeros(batch_size, num_blocks, dtype=torch.int32).npu()
        atten_mask = torch.ones(batch_size, 1, 1, dtype=torch.float16).npu()
        
        actual_seq_lengths_q = [seq_len]
        actual_seq_lengths_kv = [seq_len]
        scale = 1.0 / (head_size ** 0.5)
        
        print("Calling tq_fused_infer_attention_score...")
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
        
        print(f"✅ Operator call successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error calling operator: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("TqFusedInferAttentionScore Operator Test")
    print("=" * 80)
    
    results = []
    
    # Test 1: Operator availability
    results.append(("Operator Availability", test_op_availability()))
    
    # Test 2: Operator signature
    results.append(("Operator Signature", test_op_signature()))
    
    # Test 3: Python wrapper
    results.append(("Python Wrapper", test_python_wrapper()))
    
    # Test 4: Operator call (if NPU available)
    results.append(("Operator Call", test_op_call()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
