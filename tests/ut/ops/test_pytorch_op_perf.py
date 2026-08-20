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

"""Micro-benchmarks for selected PyTorch elementwise ops (CPU / NPU).

Run with printed timings: ``pytest -s tests/ut/ops/test_pytorch_op_perf.py``
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest
import torch


def _sync_device(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def bench_device_ms(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    for _ in range(warmup):
        fn()
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync_device(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _perf_device() -> torch.device:
    try:
        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    return torch.device("cpu")


class TestPyTorchOpPerf:
    """Timing-only checks for PyTorch ops; asserts positive finite latency."""

    @pytest.mark.parametrize("rows", [1, 1000])
    def test_bitwise_and_or_int32(self, rows: int, capsys: pytest.CaptureFixture[str]) -> None:
        """Benchmark ``&`` and ``|`` on ``(rows, 128)`` int32 tensors."""
        device = _perf_device()
        shape = (rows, 128)
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        a = torch.randint(0, 2**30, shape, dtype=torch.int32, device=device, generator=gen)
        b = torch.randint(0, 2**30, shape, dtype=torch.int32, device=device, generator=gen)

        def run_and() -> None:
            c = a & b
            assert c.shape == shape

        def run_or() -> None:
            c = a | b
            assert c.shape == shape

        ms_and = bench_device_ms(run_and, device=device)
        ms_or = bench_device_ms(run_or, device=device)

        out = (
            f"bitwise ops shape={shape} device={device}:\n"
            f"  &: {ms_and:.4f} ms/iter\n"
            f"  |: {ms_or:.4f} ms/iter\n"
        )
        with capsys.disabled():
            print(out, end="")

        assert ms_and > 0
        assert ms_or > 0
