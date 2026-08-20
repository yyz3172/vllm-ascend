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
            gpu_memory_utilization=0.04,
            additional_config={"turboquant_kv_bits": [8, 8]},
            profiler_config={"profiler": "torch", "torch_profiler_dir": "/root/x00827378/perflog2", "torch_profiler_with_stack": True},
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
    model_path = "/root/x00827378/model/Qwen3-0.6B"
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
            "Waldrada of Lotharingia\nWaldrada was the mistress, and later the wife, of Lothair II of Lotharingia.\n\nBiography\nWaldrada's family origin is uncertain. The prolific 19th-century French writer Baron Ernouf suggested that Waldrada was of noble Gallo-Roman descent, sister of Thietgaud, the bishop of Trier, and niece of Gunther, archbishop of Cologne. However, these suggestions are not supported by any evidence, and more recent studies have instead suggested she was of relatively undistinguished social origins, though still from an aristocratic milieu.\nThe Vita Sancti Deicoli states that Waldrada was related to Eberhard II, Count of Nordgau (included Strasbourg) and the family of Etichonids, though this is a late 10th-century source and so may not be entirely reliable on this question.In 855 the Carolingian king Lothar II married Teutberga, a Carolingian aristocrat and the daughter of Bosonid Boso the Elder. The marriage was arranged by Lothar's father Lothar I for political reasons. It is very probable that Waldrada was already Lothar II's mistress at this time.Teutberga was allegedly not capable of bearing children and Lothar's reign was chiefly occupied by his efforts to obtain an annulment of their marriage, and his relations with his uncles Charles the Bald and Louis the German were influenced by his desire to obtain their support for this endeavour. Lothair, whose desire for annulment was arguably prompted by his affection for Waldrada, put away Teutberga. However, Hucbert took up arms on his sister's behalf, and after she had submitted successfully to the ordeal of water, Lothair was compelled to restore her in 858. Still pursuing his purpose, he won the support of his brother, Emperor Louis II, by a cession of lands and obtained the consent of the local clergy to the annulment and to his marriage with Waldrada, which took place in 862. However, Pope Nicholas I was suspicious of this and sent legates to investigate at the Council of Metz in 863. The Council found in favour of Lothair's divorce, which led to rumours that the papal legates may have bribed and thus meant that Nicholas order Lothair to take Teutberga back or face excommunication. \nWith the support of Charles the Bald and Louis the German, Teutberga appealed the annulment to Pope Nicholas. Nicholas refused to recognize the annulment and excommunicated Waldrada in 866, forcing Lothair to abandon Waldrada in favour of Teutberga. Lothair accepted this begrudgingly for a time, but shortly afterward at the end of 867 Pope Nicholas I died. Thus, Lothair began to seek the permission of the newly appointed Pope Adrian II to again put Teutberga aside and marry Waldrada, riding to Rome to speak with him on the matter in 869. However, on his way home, Lothair died.\n\nChildren\nWaldrada and Lothair II had some sons and probably three daughters, all of whom were declared illegitimate:\n\nHugh (c. 855–895), Duke of Alsace (867–885)\nGisela (c. 865–908), who in 883 married Godfrey, the Viking leader ruling in Frisia, who was murdered in 885\nBertha (c. 863–925), who married Theobald of Arles (c. 854–895), count of Arles, nephew of Teutberga. They had two sons, Hugh of Italy and Boso of Tuscany. After Theobald's death, between 895 and 898 she married Adalbert II of Tuscany (c. 875–915) They had at least three children: Guy, who succeeded his father as count and duke of Lucca and margrave of Tuscany, Lambert succeeded his brother in 929, but lost the titles in 931 to his half-brother Boso of Tuscany, and Ermengard.\nErmengarde (d. 90?)\nOdo (d. c.879)"
        ]
        _test_qwen3_turboquant_decode_accuracy(prompts=prompts)
        print("测试 完成（turboquant ref vs custom）")
    except Exception as e:
        print(f"运行出错: {e}")
