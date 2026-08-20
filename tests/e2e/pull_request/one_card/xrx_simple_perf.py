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
import time

import torch
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODEL_PATH = "/root/x00827378/model/Qwen3-0.6B"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.3
MAX_TOKENS = 50
NUM_CLIENTS = 5
REQUESTS_PER_CLIENT = 5

BASE_PROMPTS = [
    "Hello, my name is",
    "介绍TurboQuant",
]


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
            torch.npu.empty_cache()
        with contextlib.suppress(Exception):
            torch.npu.synchronize()


def _output_token_count(output) -> int:
    return sum(len(sample.token_ids) for sample in output.outputs)


def _prompt_token_count(output) -> int:
    return len(output.prompt_token_ids or [])


def _build_client_requests() -> tuple[list[str], list[tuple[int, int]]]:
    prompts: list[str] = []
    request_ids: list[tuple[int, int]] = []
    for client_id in range(NUM_CLIENTS):
        for request_id in range(REQUESTS_PER_CLIENT):
            base_prompt = BASE_PROMPTS[request_id % len(BASE_PROMPTS)]
            prompts.append(f"[client={client_id} request={request_id}] {base_prompt}")
            request_ids.append((client_id, request_id))
    return prompts, request_ids


def run_simple_perf() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is not available")

    env = {
        # torch_npu cannot be re-initialized in forked subprocesses.
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        # Match xrx_test_sampler.py: force non-custom TurboQuant decode reference path.
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "0",
        # Keep experimental 8-bit fused attention paths out of this perf run.
        "VLLM_ASCEND_TURBOQUANT_ATTENTION_OP_8BIT": "0",
        "VLLM_ASCEND_TURBOQUANT_FUSED_FIA_8BIT": "0",
    }

    with _patched_env(env):
        clear_ascend_config()
        llm: LLM | None = None
        try:
            llm = LLM(
                model=MODEL_PATH,
                trust_remote_code=True,
                kv_cache_dtype="turboquant",
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                additional_config={"turboquant_kv_bits": [8, 8]},
                enforce_eager=True,
            )
            sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
            prompts, request_ids = _build_client_requests()

            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            elapsed = time.perf_counter() - start

            prompt_tokens = sum(_prompt_token_count(output) for output in outputs)
            output_tokens = sum(_output_token_count(output) for output in outputs)
            total_tokens = prompt_tokens + output_tokens
            client_prompt_tokens = [0] * NUM_CLIENTS
            client_output_tokens = [0] * NUM_CLIENTS
            for output, (client_id, _) in zip(outputs, request_ids):
                client_prompt_tokens[client_id] += _prompt_token_count(output)
                client_output_tokens[client_id] += _output_token_count(output)

            print("\nSimple perf result")
            print(f"clients: {NUM_CLIENTS}")
            print(f"requests_per_client: {REQUESTS_PER_CLIENT}")
            print(f"total_requests: {len(prompts)}")
            print(f"prompt_tokens: {prompt_tokens}")
            print(f"output_tokens: {output_tokens}")
            print(f"elapsed_s: {elapsed:.3f}")
            print(f"output_tok_per_s: {output_tokens / elapsed:.2f}")
            print(f"total_tok_per_s: {total_tokens / elapsed:.2f}")
            for client_id in range(NUM_CLIENTS):
                client_total_tokens = client_prompt_tokens[client_id] + client_output_tokens[client_id]
                print(
                    f"client[{client_id}]: requests={REQUESTS_PER_CLIENT}, "
                    f"prompt_tokens={client_prompt_tokens[client_id]}, "
                    f"output_tokens={client_output_tokens[client_id]}, "
                    f"total_tokens={client_total_tokens}"
                )
            for index, (output, (client_id, request_id)) in enumerate(zip(outputs, request_ids)):
                text = output.outputs[0].text if output.outputs else ""
                print(f"output[{index}] client={client_id} request={request_id}: {text!r}")
        finally:
            if llm is not None:
                del llm
            clear_ascend_config()
            _cleanup_npu_memory()


if __name__ == "__main__":
    run_simple_perf()
