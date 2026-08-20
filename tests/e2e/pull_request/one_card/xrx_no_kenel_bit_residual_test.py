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
"""BitResidual KV cache quick verification smoke test.

Starts an LLM with kv_cache_dtype="bit_residual", sends a few prompts via
``llm.generate``, and prints the decoded output. This validates that the
BitResidual quantize→dequantize path is wired end-to-end in
``reshape_and_cache``.

By default runs both 4-bit and 8-bit (configurable via ``--bits``).

Usage (on an NPU machine):
    python tests/e2e/singlecard/xrx_bit_residual_smoke.py
    python tests/e2e/singlecard/xrx_bit_residual_smoke.py --bits 4
    python tests/e2e/singlecard/xrx_bit_residual_smoke.py --bits 8

With custom model path:
    BR_SMOKE_MODEL_PATH=../model/Qwen3-0.6B python tests/e2e/singlecard/xrx_bit_residual_smoke.py
"""

from __future__ import annotations

import argparse
import os

from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODEL_PATH = os.getenv("BR_SMOKE_MODEL_PATH", "../model/Qwen3-0.6B")


def run_smoke(bits: int) -> None:
    """Run BitResidual smoke test at the given bit width."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model path not found: {MODEL_PATH}")

    clear_ascend_config()
    seed_cur = 0
    print(f"BitResidual {bits}-bit smoke: seed = {seed_cur}")
    llm = LLM(
        model=MODEL_PATH,
        seed=seed_cur,
        trust_remote_code=True,
        max_model_len=256,
        block_size=16,
        kv_cache_dtype="bit_residual",
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        enable_chunked_prefill=True,
        disable_log_stats=True,
        dtype="float16",
        additional_config={"bit_residual_kv_bits": bits},
    )
    outs = llm.generate(
        ["Hello, my name is", "介绍turboquant技术"],
        sampling_params=SamplingParams(temperature=0.7, max_tokens=40),
    )

    print(f"\n===== BitResidual {bits}-bit KV Cache Smoke Output =====")
    for i, out in enumerate(outs):
        print(f"[Prompt {i}] {out.prompt!r}")
        print(f"  -> {out.outputs[0].text}")
    print("=" * (51 + len(str(bits))))
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BitResidual KV cache smoke test (4-bit / 8-bit)"
    )
    parser.add_argument(
        "--bits", type=int, nargs="*", default=[4, 8],
        help="Bit width(s) to test (default: both 4 and 8)"
    )
    args = parser.parse_args()

    bits_list = [b for b in args.bits if b in (4, 8)]
    if not bits_list:
        raise ValueError(f"Invalid --bits: {args.bits}. Allowed: 4, 8.")

    for bits in bits_list:
        print(f"\n{'#' * 60}")
        print(f"# BitResidual {bits}-bit smoke")
        print(f"{'#' * 60}")
        run_smoke(bits)


if __name__ == "__main__":
    main()
