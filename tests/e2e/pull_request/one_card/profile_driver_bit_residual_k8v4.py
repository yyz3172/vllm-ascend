#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Single-config driver for msprof profiling of bit_residual_attention_paged_k8v4.

Runs ONE shape in a tight loop so msprof captures a clean kernel timeline without
the benchmark's multi-config noise. Config via env vars:

    PROF_SEQ (default 2048)   KV sequence length
    PROF_BLK (default 128)    paged block size
    PROF_DTYPE (default fp16) fp16 | bf16
    PROF_ITERS (default 20)   timed iterations
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
_CANN_OPP = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{_CANN_OPP}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
    )

import torch  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from bench_bit_residual_k8v4 import make_inputs, run_op  # noqa: E402


def main() -> None:
    enable_custom_op()
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    seq = int(os.environ.get("PROF_SEQ", "2048"))
    blk = int(os.environ.get("PROF_BLK", "128"))
    dtype = torch.bfloat16 if os.environ.get("PROF_DTYPE", "fp16") == "bf16" else torch.float16
    iters = int(os.environ.get("PROF_ITERS", "20"))

    inputs = make_inputs(
        num_query_tokens=1, seq_len_kv=seq, num_heads=16, num_kv_heads=8,
        block_size=blk, dtype=dtype, device=device,
    )
    # Warmup.
    for _ in range(5):
        run_op(**inputs)
    torch.npu.synchronize()
    # Tight loop — msprof captures these kernels.
    for _ in range(iters):
        run_op(**inputs)
    torch.npu.synchronize()
    print(f"profile_driver done: seq={seq} blk={blk} dtype={dtype} iters={iters}")


if __name__ == "__main__":
    main()

