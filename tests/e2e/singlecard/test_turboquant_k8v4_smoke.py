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

"""Minimal end-to-end smoke test for TurboQuant K8V4 (key 8-bit, value 4-bit).

Mirrors ``xrx_turboquant4bit_smoke.py`` but runs the asymmetric K8V4 config so
you can quickly sanity-check that the K8V4 read/write path produces sane text.

Toggle the fused K8V4 kernels on/off (no rebuild) via env vars:
  * VLLM_ASCEND_TURBOQUANT_FUSED_FIA_K8V4  -> fused read op   (1=on, 0=ref)
  * VLLM_ASCEND_TURBOQUANT_ENCODE_OP       -> fused write/pack (1=on, 0=ref)

Usage (inside the container):
  cd /root/yyz/code/vllm-project/vllm-ascend
  PYTHONPATH=.:/root/yyz/code/vllm-project/vllm \
    python tests/e2e/singlecard/test_turboquant_k8v4_smoke.py
"""

from __future__ import annotations

import contextlib
import os

import torch
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.ops.turboquant_kv_cache import refresh_turboquant_env_cache

MODEL_PATH = "/root/yyz/models/Qwen3-0.6B"
PROFILE_DIR = "/root/yyz/pytorch_profiler/TurboQuant/260711/k8v4_0.6B"
PROFILE_WARMUP_ITERATIONS = int(os.getenv("XRX_K8V4_PROFILE_WARMUP_ITERATIONS", "2"))
PROFILE_ACTIVE_ITERATIONS = int(os.getenv("XRX_K8V4_PROFILE_ACTIVE_ITERATIONS", "2"))


def _spawn_child_env(extra: dict[str, str]) -> dict[str, str]:
    """Preserve Ascend/CANN paths for vLLM ``spawn`` worker processes."""
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


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    env = _spawn_child_env({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        # K8V4 uses the standard paged TurboQuant cache (not the 4-bit slab cache).
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
        # Fused write (pack) op; set to 0 to compare against the reference path.
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": os.getenv(
            "VLLM_ASCEND_TURBOQUANT_ENCODE_OP", "1"),
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "1",
        # Fused K8V4 read op (default on; set to 0 for decode+FIA fallback).
        "VLLM_ASCEND_TURBOQUANT_FUSED_FIA_K8V4": os.getenv(
            "VLLM_ASCEND_TURBOQUANT_FUSED_FIA_K8V4", "1"),
    })
    with _patched_env(env):
        clear_ascend_config()
        refresh_turboquant_env_cache()
        enable_profile = os.getenv("XRX_K8V4_PROFILE", "0") == "1"
        profiler_config = None
        if enable_profile:
            profiler_config = {
                "profiler": "torch",
                "torch_profiler_dir": PROFILE_DIR,
                "torch_profiler_with_stack": True,
                "warmup_iterations": PROFILE_WARMUP_ITERATIONS,
                "active_iterations": PROFILE_ACTIVE_ITERATIONS,
            }
        seed_cur = 0  # time.time_ns() % (2**31)
        print("llm seed is ", seed_cur)
        llm = LLM(
            model=MODEL_PATH,
            seed=seed_cur,
            trust_remote_code=True,
            max_model_len=256,
            block_size=16,
            kv_cache_dtype="turboquant",
            gpu_memory_utilization=0.05,
            additional_config={"turboquant_kv_bits": [8, 4]},
            enforce_eager=True,
            enable_chunked_prefill=True,
            disable_log_stats=True,
            profiler_config=profiler_config,
        )
        if enable_profile:
            llm.start_profile()
        try:
            outs = llm.generate(
                ["Hello, my name is", "介绍turboquant技术"],
                sampling_params=SamplingParams(temperature=0.7, max_tokens=40),
            )
        finally:
            if enable_profile:
                llm.stop_profile()
    print("k8v4 smoke output:", [out.outputs[0].text for out in outs])


if __name__ == "__main__":
    main()
