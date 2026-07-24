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

"""BitResidual K8V4 end-to-end smoke test (mirrors xrx_turboquant4bit_smoke.py).

Uses ``turboquant_kv_bits=[8, 4]`` to activate the BitResidual k8v4 pack +
paged attention path inside the vLLM inference pipeline.

Run on an NPU machine after building custom ops:

    python tests/e2e/singlecard/xrx_bit_residual_k8v4_smoke.py

Env:
    XRX_K8V4_MODEL_PATH   model path (default: /root/l00856060/model/Qwen3-0.6B)
    XRX_K8V4_PROFILE_DIR  torch profiler dump dir (default: /root/l00856060/perflog2)
    XRX_K8V4_PROFILE      set to 1 to enable torch profiler
"""

from __future__ import annotations

import contextlib
import os

import torch
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODEL_PATH = os.getenv("MODEL_PATH", "/root/yyz/models/Qwen3-0.6B")
PROFILE_DIR = os.getenv(
    "PROFILE_DIR",
    "/root/yyz/pytorch_profiler/BitResidualSmoke/260723/paged",
)


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

    env = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        # BitResidual k8v4 reuses the turboquant MSE v1 rotation matrix.
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
    }
    enable_profile = os.getenv("XRX_K8V4_PROFILE", "0") == "1"
    with _patched_env(env):
        clear_ascend_config()
        profiler_config = None
        if enable_profile:
            os.makedirs(PROFILE_DIR, exist_ok=True)
            profiler_config = {
                "profiler": "torch",
                "torch_profiler_dir": PROFILE_DIR,
                "torch_profiler_with_stack": True,
            }
        seed_cur = 0
        print("llm seed is ", seed_cur)
        llm = LLM(
            model=MODEL_PATH,
            seed=seed_cur,
            trust_remote_code=True,
            max_model_len=256,
            block_size=16,
            kv_cache_dtype="turboquant",
            gpu_memory_utilization=0.05,
            # bits_key=8, bits_value=4 activates the BitResidual k8v4 path
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
                use_tqdm=False,
            )
        finally:
            if enable_profile:
                llm.stop_profile()
    print("k8v4 smoke output:", [out.outputs[0].text for out in outs])
    if enable_profile:
        print("k8v4 smoke profile_dir:", PROFILE_DIR)


if __name__ == "__main__":
    main()
