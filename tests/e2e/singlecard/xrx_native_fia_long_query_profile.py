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

"""Native FIA (fp16 KV cache) long-query profile — baseline counterpart to
``xrx_turboquant4bit_long_query_profile.py``.

Runs the standard ``npu_fused_infer_attention_score`` path without TurboQuant
quantization.  Unsets all ``VLLM_ASCEND_TURBOQUANT_*`` environment variables to
guarantee the native code path is exercised even if the process previously
inherited TQ settings.
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import torch

MODEL_PATH = os.getenv("XRX_FIA_MODEL_PATH", "/root/x00827378/model/Qwen3-0.6B")
PROFILE_DIR = os.getenv("XRX_FIA_PROFILE_DIR", "/root/x00827378/perflog_fia")


@contextlib.contextmanager
def _patched_env(env: dict[str, str], unset: tuple[str, ...] = ()):
    old = os.environ.copy()
    for name in unset:
        os.environ.pop(name, None)
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _build_prompt(model_path: str, target_tokens: int) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    seed = (
        "TurboQuant compresses key and value cache tensors for long context "
        "inference while preserving attention quality. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("tokenizer produced an empty seed prompt")
    prompt_ids = (seed_ids * math.ceil(target_tokens / len(seed_ids)))[:target_tokens]
    return tokenizer.decode(prompt_ids, skip_special_tokens=True)


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    prompt_tokens = _int_env("XRX_FIA_LONG_PROMPT_TOKENS", 2000)
    num_prompts = _int_env("XRX_FIA_LONG_NUM_PROMPTS", 16)
    max_tokens = _int_env("XRX_FIA_LONG_OUTPUT_TOKENS", 8)
    max_model_len = _int_env(
        "XRX_FIA_LONG_MAX_MODEL_LEN",
        max(256, prompt_tokens + max_tokens + 32),
    )
    gpu_memory_utilization = _float_env("XRX_FIA_GPU_MEMORY_UTILIZATION", 0.05)
    enable_profile = os.getenv("XRX_FIA_ENABLE_PROFILE", "1") == "1"
    profile_warmup_generates = _int_env("XRX_FIA_PROFILE_WARMUP_GENERATES", 0)

    os.makedirs(PROFILE_DIR, exist_ok=True)
    unset = (
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP",
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP",
        "VLLM_ASCEND_TURBOQUANT_PACK_OP",
    )
    env = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
    }
    with _patched_env(env, unset):
        from vllm import LLM, SamplingParams

        from vllm_ascend.ascend_config import clear_ascend_config

        clear_ascend_config()
        prompt = _build_prompt(MODEL_PATH, prompt_tokens)
        prompts = [prompt for _ in range(num_prompts)]

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
            max_model_len=max_model_len,
            block_size=16,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            enable_chunked_prefill=True,
            disable_log_stats=True,
            profiler_config=profiler_config,
        )
        if enable_profile:
            llm.start_profile()
        try:
            for _ in range(profile_warmup_generates):
                llm.generate(
                    ["profile warmup"],
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                )
            start = time.perf_counter()
            outs = llm.generate(
                prompts,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
            )
            elapsed_s = time.perf_counter() - start
        finally:
            if enable_profile:
                llm.stop_profile()
    print(
        "fia long query profile:",
        {
            "num_prompts": num_prompts,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "enable_profile": enable_profile,
            "profile_warmup_generates": profile_warmup_generates,
            "elapsed_s": elapsed_s,
            "output_chars": [len(out.outputs[0].text) for out in outs],
            "profile_dir": PROFILE_DIR,
        },
    )


if __name__ == "__main__":
    main()
