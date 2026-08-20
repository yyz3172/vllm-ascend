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

from __future__ import annotations

import contextlib
import gc
import os
import sys

import torch
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODEL_PATH = "/root/x00827378/model/Qwen3-0.6B"
PROMPT = "Hello, my name is"
MAX_TOKENS = 20


@contextlib.contextmanager
def _patched_env(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _cleanup_npu_memory() -> None:
    gc.collect()
    if hasattr(torch, "npu"):
        with contextlib.suppress(Exception):
            torch.npu.synchronize()
        with contextlib.suppress(Exception):
            torch.npu.empty_cache()


def _run_case(
    name: str,
    *,
    kv_cache_dtype: str | None = None,
    additional_config: dict | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
    }
    if extra_env:
        env.update(extra_env)
    print(f"\n=== {name} ===", flush=True)
    print("env:", {k: env[k] for k in sorted(env)}, flush=True)
    with _patched_env(env):
        clear_ascend_config()
        kwargs = {
            "model": MODEL_PATH,
            "trust_remote_code": True,
            "max_model_len": 256,
            "block_size": 16,
            "gpu_memory_utilization": 0.10,
            "enforce_eager": True,
            "enable_chunked_prefill": True,
            "disable_log_stats": True,
        }
        if kv_cache_dtype is not None:
            kwargs["kv_cache_dtype"] = kv_cache_dtype
        if additional_config is not None:
            kwargs["additional_config"] = additional_config
        llm = LLM(**kwargs)
        try:
            outs = llm.generate(
                [PROMPT],
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                ),
            )
            print(f"{name} output:", [out.outputs[0].text for out in outs], flush=True)
        finally:
            del llm
    _cleanup_npu_memory()


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    selected = set(sys.argv[1:])
    run_all = not selected

    def should_run(name: str) -> bool:
        return run_all or name in selected

    if should_run("fp16_auto"):
        _run_case("fp16_auto")
    common_4bit = {
        "kv_cache_dtype": "turboquant",
        "additional_config": {"turboquant_kv_bits": [4, 4]},
    }
    if should_run("4bit_ref_ref"):
        _run_case(
            "4bit_ref_ref",
            **common_4bit,
            extra_env={
                "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
                "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
                "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
                "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
            },
        )
    if should_run("4bit_row_ref_ref"):
        _run_case(
            "4bit_row_ref_ref",
            **common_4bit,
            extra_env={
                "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "0",
                "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
                "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
                "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
            },
        )
    if should_run("4bit_op_ref"):
        _run_case(
            "4bit_op_ref",
            **common_4bit,
            extra_env={
                "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
                "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
                "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "1",
                "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
            },
        )
    if should_run("4bit_op_fused"):
        _run_case(
            "4bit_op_fused",
            **common_4bit,
            extra_env={
                "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
                "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
                "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "1",
                "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "1",
            },
        )
    if should_run("8bit_ref_ref"):
        _run_case(
            "8bit_ref_ref",
            kv_cache_dtype="turboquant",
            additional_config={"turboquant_kv_bits": [8, 8]},
            extra_env={
                "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "0",
                "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
                "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "0",
                "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
            },
        )


if __name__ == "__main__":
    main()
