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

import argparse
import contextlib
import os

MODEL_PATH = "/root/x00827378/model/Qwen3-0.6B"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-op", choices=("0", "1"), default="1")
    parser.add_argument("--decode-op", choices=("0", "1"), default="1")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


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
    args = _parse_args()

    env = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "1",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": args.encode_op,
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": args.decode_op,
    }
    os.environ.update(env)

    import torch
    from vllm import LLM, SamplingParams

    from vllm_ascend.ascend_config import clear_ascend_config

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    with _patched_env(env):
        clear_ascend_config()
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            max_model_len=256,
            block_size=16,
            kv_cache_dtype="turboquant",
            gpu_memory_utilization=0.10,
            additional_config={"turboquant_kv_bits": [4, 4]},
            enforce_eager=True,
            enable_chunked_prefill=True,
            disable_log_stats=True,
        )
        outs = llm.generate(
            ["Hello, my name is", "介绍turboquant技术"],
            sampling_params=SamplingParams(
                temperature=args.temperature,
                max_tokens=20,
            ),
        )
    print(
        f"debug output encode={args.encode_op} decode={args.decode_op}:",
        [out.outputs[0].text for out in outs],
    )


if __name__ == "__main__":
    main()
