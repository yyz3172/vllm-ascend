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

"""BitResidual K8V4 bs=1 long-KV decode microbench (FlashDecode eligibility).

Qwen3-0.6B: taskCount = numTokens * 8. With one prompt / one decode token,
taskCount=8 <= ~10, so IsFlashDecodeK8v4 can enable SplitBNS when KV>=1024.

Defaults (override via env):
  XRX_K8V4_LONG_NUM_PROMPTS=1
  XRX_K8V4_LONG_PROMPT_TOKENS=2000
  XRX_K8V4_LONG_OUTPUT_TOKENS=16
  XRX_K8V4_PROFILE_DIR=.../BitResidualBs1Long/...

Example::

    ASCEND_RT_VISIBLE_DEVICES=5 python \\
        tests/e2e/singlecard/xrx_bit_residual_k8v4_bs1_long_profile.py
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    os.environ.setdefault("XRX_K8V4_LONG_NUM_PROMPTS", "1")
    os.environ.setdefault("XRX_K8V4_LONG_PROMPT_TOKENS", "2000")
    os.environ.setdefault("XRX_K8V4_LONG_OUTPUT_TOKENS", "16")
    os.environ.setdefault("XRX_K8V4_ENABLE_PROFILE", "1")
    os.environ.setdefault(
        "XRX_K8V4_PROFILE_DIR",
        "/root/yyz/pytorch_profiler/BitResidualBs1Long/260715/k8v4_0.6B",
    )
    target = Path(__file__).with_name("xrx_bit_residual_k8v4_long_query_profile.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
