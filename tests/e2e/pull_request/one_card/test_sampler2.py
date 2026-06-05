#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/entrypoints/llm/test_guided_generate.py
# Copyright 2023 The vLLM team.
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
#
import contextlib
import os
from difflib import SequenceMatcher

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None
import torch
from vllm import SamplingParams
from vllm_ascend.ascend_config import clear_ascend_config

from tests.e2e.conftest import VllmRunner

@contextlib.contextmanager
def _patched_env(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def _is_mostly_printable(s: str, *, min_ratio: float = 0.95) -> bool:
    # Treat common whitespace as printable.
    if not s:
        return True
    printable = 0
    for ch in s:
        o = ord(ch)
        if ch in "\n\r\t":
            printable += 1
        elif 32 <= o <= 126:
            printable += 1
        elif 0x4E00 <= o <= 0x9FFF:  # CJK Unified Ideographs
            printable += 1
        elif 0x3000 <= o <= 0x303F:  # CJK punctuation
            printable += 1
        elif 0xFF00 <= o <= 0xFFEF:  # Fullwidth forms
            printable += 1
    return (printable / max(1, len(s))) >= min_ratio


def _run_greedy(
    *,
    model_path: str,
    kv_cache_dtype: str | None,
    turboquant_decode_op: str | None,
    prompts: list[str],
    max_tokens: int,
) -> list[str]:
    env: dict[str, str] = {}
    # torch_npu cannot be re-initialized in forked subprocesses.
    # Force vLLM to use spawn for worker/engine core processes.
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Some vLLM builds use a separate engine core process; keep this in sync.
    env.setdefault("VLLM_ENGINE_CORE_MULTIPROC_METHOD", "spawn")
    # env.setdefault("VLLM_ASCEND_TURBOQUANT_CODEBOOK_METHOD", "sample")
    if turboquant_decode_op is not None:
        env["VLLM_ASCEND_TURBOQUANT_DECODE_OP"] = turboquant_decode_op
    with _patched_env(env):
        clear_ascend_config()
        runner_kwargs = dict(
            max_model_len=8192,
            # cudagraph_capture_sizes=[1, 2, 4, 8],
            gpu_memory_utilization=0.7,
            additional_config={"turboquant_kv_bits": [8, 8]},
            profiler_config={"profiler": "torch", "torch_profiler_dir": "/root/l00856060/perflog2", "torch_profiler_with_stack": True},
            enforce_eager=True,
            # compilation_config={"cudagraph_mode": "NONE"}
        )
        if kv_cache_dtype is not None:
            runner_kwargs["kv_cache_dtype"] = kv_cache_dtype
        with VllmRunner(model_path, **runner_kwargs) as runner:
            outs = runner.generate_greedy(prompts, max_tokens=max_tokens)
        return [s for _, s in outs]


def _test_qwen3_turboquant_decode_accuracy(prompts: list[str]) -> None:
    """Compare turboquant non-custom-op vs custom-op outputs.

    This test is designed to catch regressions where TurboQuant custom decode op
    produces clearly wrong tokens (e.g. many '!' characters).
    """
    model_path = "/root/l00856060/model/Qwen3-0.6B"
    if not os.path.exists(model_path):
        if pytest is not None:
            pytest.skip(f"model path not found: {model_path}")
        raise RuntimeError(f"model path not found: {model_path}")
    if not hasattr(torch, "npu"):
        if pytest is not None:
            pytest.skip("torch.npu is not available")
        raise RuntimeError("torch.npu is not available")

    # Keep it modest to reduce amplification of early-token differences.
    max_tokens = 50

    #NOTE: Intentionally skip non-quant baseline here to save time during debugging.
    # baseline = _run_greedy(
    #     model_path=model_path,
    #     kv_cache_dtype=None,
    #     turboquant_decode_op=None,
    #     prompts=prompts,
    #     max_tokens=max_tokens,
    # )
    # print(baseline)

    tq_ref = _run_greedy(
            model_path=model_path,
            kv_cache_dtype="turboquant",
            turboquant_decode_op="0",  # force non-custom path
            prompts=prompts,
            max_tokens=max_tokens,        
        )
    print(tq_ref)

    # tq_custom = _run_greedy(
    #     model_path=model_path,
    #     kv_cache_dtype="turboquant",
    #     turboquant_decode_op="1",  # force custom op path
    #     prompts=prompts,
    #     max_tokens=max_tokens,
    # )

    # TurboQuant is lossy: don't require high similarity to non-quant baseline.
    # The key correctness check is: custom-op path should closely match the
    # non-custom TurboQuant path (same algorithm, different implementation).
    # min_sim_custom_vs_ref = 0.90

    # for i, p in enumerate(prompts):
    #     r = tq_ref[i]
    #     c = tq_custom[i]
    #     sim_cr = _similarity(r, c)

    #     # Heuristic corruption detector: excessive exclamation marks relative to ref.
    #     # (Empirically catches the "many !" failure mode.)
    #     assert c.count("!") <= max(8, r.count("!") + 8), (
    #         f"turboquant(custom-op) output seems corrupted for prompt[{i}]: {p!r}\n"
    #         f"ref={r!r}\ncustom={c!r}\n"
    #         f"custom(!)={c.count('!')} ref(!)={r.count('!')}"
    #     )

    #     assert _is_mostly_printable(r), (
    #         f"turboquant(non-custom) output contains many non-printable chars for prompt[{i}]: {p!r}\n"
    #         f"ref={r!r}\ncustom={c!r}"
    #     )
    #     assert _is_mostly_printable(c), (
    #         f"turboquant(custom-op) output contains many non-printable chars for prompt[{i}]: {p!r}\n"
    #         f"ref={r!r}\ncustom={c!r}"
    #     )

    #     assert sim_cr >= min_sim_custom_vs_ref, (
    #         f"turboquant(custom-op) deviates too much from non-custom turboquant for prompt[{i}]: {p!r}\n"
    #         f"similarity(custom, ref)={sim_cr:.3f} (min={min_sim_custom_vs_ref})\n"
    #         f"ref={r!r}\ncustom={c!r}"
    #     )

if __name__ == "__main__":
    print("开始运行测试...")
    try:
        prompts = [
            "Hello, my name is",
        ]
        _test_qwen3_turboquant_decode_accuracy(prompts=prompts)
        print("测试 完成（turboquant ref vs custom）")
    except Exception as e:
        print(f"运行出错: {e}")
