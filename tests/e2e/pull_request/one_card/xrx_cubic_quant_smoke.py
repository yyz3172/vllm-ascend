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
"""Cubic-polynomial 4-bit KV cache quick verification smoke test.

Starts an LLM with kv_cache_dtype="cubic_quant", sends a few prompts via
``llm.generate``, and prints the decoded output. This validates that the
cubic-quant quantize→dequantize path is wired end-to-end in
``reshape_and_cache`` (pack via cubic polynomial mapping, then immediately
unpack via codebook lookup).

Usage (on an NPU machine):
    python tests/e2e/singlecard/xrx_cubic_quant_smoke.py

With custom model path:
    BR_SMOKE_MODEL_PATH=../model/Qwen3-0.6B python tests/e2e/singlecard/xrx_cubic_quant_smoke.py
"""

from __future__ import annotations

import os

from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODEL_PATH = os.getenv("BR_SMOKE_MODEL_PATH", "../model/Qwen3-0.6B")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")

    clear_ascend_config()
    seed_cur = 0
    print("CubicQuant smoke: seed =", seed_cur)
    llm = LLM(
        model=MODEL_PATH,
        seed=seed_cur,
        trust_remote_code=True,
        max_model_len=256,
        block_size=16,
        kv_cache_dtype="cubic_quant",
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        enable_chunked_prefill=True,
        disable_log_stats=True,
        dtype="float16",
    )
    outs = llm.generate(
        ["Hello, my name is", "介绍turboquant技术"],
        sampling_params=SamplingParams(temperature=0.7, max_tokens=40),
    )

    print("\n===== CubicQuant 4-bit KV Cache Smoke Output =====")
    for i, out in enumerate(outs):
        print(f"[Prompt {i}] {out.prompt!r}")
        print(f"  -> {out.outputs[0].text}")
    print("===================================================")
    print("Done.")


if __name__ == "__main__":
    main()
