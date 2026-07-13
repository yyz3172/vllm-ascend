#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Greedy A/B for K8V4 fused vs reference decode quality."""

from __future__ import annotations

import contextlib
import os
import sys

import torch
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.ops.turboquant_kv_cache import refresh_turboquant_env_cache

MODEL_PATH = "/root/yyz/models/Qwen3-0.6B"
PROMPTS = ["Hello, my name is", "介绍turboquant技术"]


def _spawn_child_env(extra: dict[str, str]) -> dict[str, str]:
    inherited: dict[str, str] = {}
    for key, value in os.environ.items():
        if (
            key.startswith(("ASCEND_", "LD_", "PYTHON", "HCCL_", "GE_", "TOOLCHAIN_"))
            or key in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH")
        ):
            inherited[key] = value
    inherited.update(extra)
    return inherited


@contextlib.contextmanager
def _patched_env(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def run_one(tag: str, encode: str, fused_fia: str) -> list[str]:
    env = _spawn_child_env({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": encode,
        "VLLM_ASCEND_TURBOQUANT_FUSED_FIA_K8V4": fused_fia,
    })
    with _patched_env(env):
        clear_ascend_config()
        refresh_turboquant_env_cache()
        llm = LLM(
            model=MODEL_PATH,
            seed=0,
            trust_remote_code=True,
            max_model_len=256,
            block_size=16,
            kv_cache_dtype="turboquant",
            gpu_memory_utilization=0.05,
            additional_config={"turboquant_kv_bits": [8, 4]},
            enforce_eager=True,
            enable_chunked_prefill=True,
            disable_log_stats=True,
        )
        outs = llm.generate(
            PROMPTS,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=40),
        )
    texts = [out.outputs[0].text for out in outs]
    print(f"=== {tag} encode={encode} fia={fused_fia} ===")
    for t in texts:
        print(repr(t))
    return texts


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    modes = [
        ("fused", "1", "1"),
        ("ref", "0", "0"),
        ("pack_fused_fia_ref", "1", "0"),
        ("pack_ref_fia_fused", "0", "1"),
    ]
    want = sys.argv[1:] if len(sys.argv) > 1 else ["fused", "ref"]
    for tag, enc, fia in modes:
        if tag in want:
            run_one(tag, enc, fia)


if __name__ == "__main__":
    main()
