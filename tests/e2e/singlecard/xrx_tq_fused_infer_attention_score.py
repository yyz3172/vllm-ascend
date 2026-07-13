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

"""Smoke test for tq_fused_infer_attention_score operator integration.

This test verifies that the tq_fused_infer_attention_score operator can be
successfully called through the forward_fused_infer_attention framework.
"""

from __future__ import annotations

import contextlib
import os

import torch
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODEL_PATH = "/root/x00827378/model/Qwen3-0.6B"
PROFILE_DIR = "/root/x00827378/perflog2"


@contextlib.contextmanager
def _patched_env(env: dict[str, str]):
    """Context manager to temporarily patch environment variables."""
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def main() -> None:
    """Run tq_fused_infer_attention_score smoke test."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    # Environment configuration for TqFusedInferAttentionScore
    env = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        # Enable TqFusedInferAttentionScore operator
        "VLLM_ASCEND_TQ_FUSED_INFER_ATTENTION_SCORE": "1",
        # Use standard fp16 KV cache (not turboquant) to test the new operator
        "VLLM_ASCEND_FORCE_TQ_FIA": "1",
    }
    
    with _patched_env(env):
        clear_ascend_config()
        
        # Configure LLM with settings that will trigger forward_fused_infer_attention
        # and use tq_fused_infer_attention_score when available
        enable_profile = os.getenv("XRX_TQ_FIA_PROFILE", "0") == "1"
        profiler_config = None
        if enable_profile:
            profiler_config = {
                "profiler": "torch",
                "torch_profiler_dir": PROFILE_DIR,
                "torch_profiler_with_stack": True,
            }
        
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            max_model_len=2560,
            block_size=16,
            # Use auto kv_cache_dtype to let the system decide
            kv_cache_dtype="auto",
            gpu_memory_utilization=0.05,
            enforce_eager=True,
            enable_chunked_prefill=True,
            disable_log_stats=True,
            profiler_config=profiler_config,
        )
        
        if enable_profile:
            llm.start_profile()
        
        try:
            # Test with various prompts to exercise different code paths
            test_prompts = [
                "Hello, my name is",
                "介绍大模型推理优化技术",
                "The future of artificial intelligence",
            ]
            
            print(f"Running tq_fused_infer_attention_score smoke test with {len(test_prompts)} prompts...")
            
            outs = llm.generate(
                test_prompts,
                sampling_params=SamplingParams(temperature=0.7, max_tokens=40),
            )
            
            print("\n" + "=" * 80)
            print("TqFusedInferAttentionScore Smoke Test Results:")
            print("=" * 80)
            for i, out in enumerate(outs):
                print(f"\nPrompt {i+1}: {test_prompts[i]}")
                print(f"Output: {out.outputs[0].text}")
            print("\n" + "=" * 80)
            
        finally:
            if enable_profile:
                llm.stop_profile()
    
    print("\n✅ tq_fused_infer_attention_score smoke test completed successfully!")


if __name__ == "__main__":
    main()
