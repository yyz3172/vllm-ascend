#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Isolate a pure AIC/Cube matmul path for TurboQuant rotate.
#
# probe_mode:
#   0 = MIX 1C2V KFC REGIST_MATMUL_OBJ only (matches fused pack handshake)
#   1 = pure AIC GM [M_pad,128]@[128,128] matmul via mm.Init (no REGIST)
#
# Build (torch binding + aclnn in libopapi.so — ``build_ext`` alone is not enough):
#   cd vllm-ascend && COMPILE_CUSTOM_KERNELS=1 python setup.py build_ext --inplace
#   # or: pip install -e .
# Verify aclnn symbol:
#   nm -D vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libopapi.so \
#     | grep -i TurboquantRotateMatmulProbe
#
# Run (blocking + debug prints):
#   ASCEND_LAUNCH_BLOCKING=1 VLLM_ASCEND_TURBOQUANT_DEBUG_BLOCKS=1 \
#     pytest -s tests/ut/ops/test_turboquant_rotate_matmul_probe.py
#

from __future__ import annotations

import os
import signal
from contextlib import contextmanager

import pytest
import torch

import vllm_ascend
from vllm_ascend.ops.turboquant_kv_cache import _c_ascend_turboquant_op_available
from vllm_ascend.utils import enable_custom_op

# Load vllm_ascend_C and point CANN at vendor libopapi (aclnnTurboquantRotateMatmulProbe).
_custom_opp = os.path.join(
    os.path.dirname(vllm_ascend.__file__),
    "_cann_ops_custom",
    "vendors",
    "vllm-ascend",
)
if os.path.isdir(_custom_opp):
    _prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{_custom_opp}:{_prev}" if _prev else _custom_opp
    )
enable_custom_op()

try:
    _NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception:
    _NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(
    not _NPU_AVAILABLE, reason="TurboQuant matmul probe requires Ascend NPU"
)

PROBE_TIMEOUT_SEC = int(os.environ.get("VLLM_ASCEND_TQ_MM_PROBE_TIMEOUT", "120"))
D = 128


@contextmanager
def _probe_timeout(seconds: int):
    """Fail the test if NPU op blocks longer than ``seconds`` (Unix only)."""

    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"probe exceeded {seconds}s (likely REGIST/matmul hang)")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _run_probe(a: torch.Tensor, b: torch.Tensor, probe_mode: int) -> torch.Tensor:
    with _probe_timeout(PROBE_TIMEOUT_SEC):
        out = torch.ops._C_ascend.turboquant_rotate_matmul_probe(a, b, probe_mode)
        torch.npu.synchronize()
    return out


# Each parametrized m runs one separate kernel launch (not double-enter per case).
@requires_npu
@pytest.mark.parametrize("m", [1, 16])
def test_turboquant_rotate_matmul_probe_regist_only(m: int):
    """Mode 0: MIX KFC REGIST only — should finish quickly if 1C2V handshake is OK."""

    if not _c_ascend_turboquant_op_available("turboquant_rotate_matmul_probe"):
        pytest.skip("turboquant_rotate_matmul_probe not built")

    device = torch.device("npu")
    a = torch.randn(m, D, dtype=torch.float16, device=device).contiguous()
    b = torch.randn(D, D, dtype=torch.float16, device=device).contiguous()

    c = _run_probe(a, b, probe_mode=0)
    assert c.shape == (m, D)


@requires_npu
@pytest.mark.parametrize("m", [1, 16])
def test_turboquant_rotate_matmul_probe_matmul_correctness(m: int):
    """Mode 1: pure GM matmul (no REGIST); compare with torch.matmul (fp32 ref)."""

    if not _c_ascend_turboquant_op_available("turboquant_rotate_matmul_probe"):
        pytest.skip("turboquant_rotate_matmul_probe not built")

    device = torch.device("npu")
    torch.manual_seed(42 + m)
    a = torch.randn(m, D, dtype=torch.float16, device=device).contiguous()
    b = torch.randn(D, D, dtype=torch.float16, device=device).contiguous()

    c = _run_probe(a, b, probe_mode=1)

    ref = (a.float() @ b.float()).to(torch.float16)
    if torch.isnan(c).any().item():
        pytest.fail(
            f"m={m}: probe output contains NaN (kernel likely did not write c; "
            "check plog for [TQ_MM_PROBE_OP] mode=1 enter/after gm matmul)"
        )
    max_err = (c.float() - ref).abs().max().item()
    assert max_err < 0.25, f"m={m} max_err={max_err}"
