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

"""Accuracy and profiling tests for ``turboquant_pack_kv_for_cache_to_cache``.

Matches the production path in ``attention_v1.py``.  Accuracy compares the fused
pack-to-cache kernel against the decomposed fallback:

    ``turboquant_pack_kv_for_cache`` + ``_npu_reshape_and_cache``

(same reference as ``test_turboquant_pack_kv_for_cache_to_cache_matches_old_path``).

Batch sizes (``num_tokens``):

    - Small  (T=1):     decode scenario
    - Medium (T=512):   moderate prefill
    - Large  (T=2048):  large-scale prefill

Accuracy: byte-exact match on paged KV cache contents.
Profiling: uses ``torch_npu.profiler`` with Python call stacks (``with_stack`` /
``with_modules``); traces saved under TRACE_DIR for TensorBoard / msprof.

Run:

    export ASCEND_RT_VISIBLE_DEVICES=4
    pytest -s tests/ut/ops/test_turboquant_pack_kv_fused_batch.py -v

Profiler trace saved to TRACE_DIR for each batch size.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    _c_ascend_turboquant_op_available,
    ensure_turboquant_pack_tables_registered,
    turboquant_pack_kv_for_cache,
    turboquant_pack_kv_for_cache_to_cache,
    turboquant_packed_bytes_per_vector,
)

try:
    _NPU_AVAILABLE = bool(torch.npu.is_available())
except Exception:
    _NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(
    not _NPU_AVAILABLE, reason="Requires Ascend NPU"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_SIZE = 128
NUM_KV_HEADS = 8
BLOCK_SIZE = 128
BITS = 8

BATCH_SIZES = (1, 512, 2048)

# Target physical device via ASCEND_RT_VISIBLE_DEVICES.  When the env var
# is set, physical device 4 appears as npu:0 inside the process.
DEVICE = torch.device("npu:0")

TRACE_DIR = "/root/yyz/pytorch_profiler/TurboQuant/pack_kv_to_cache_batch"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync_device() -> None:
    if DEVICE.type == "npu":
        torch.npu.synchronize()


def _make_paged_cache(
    num_tokens: int,
    num_kv_heads: int,
    slot_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged KV cache + sequential int32 slot_mapping (attention_v1 layout)."""
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = torch.zeros(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        slot_w,
        dtype=torch.int8,
        device=DEVICE,
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    return key_cache, value_cache, slot_mapping


def _pack_to_cache_reference(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    bits_key: int,
    bits_value: int,
    slot_w_k: int,
    slot_w_v: int,
) -> None:
    """``turboquant_pack_kv_for_cache`` + ``_npu_reshape_and_cache`` fallback path."""
    import torch_npu  # type: ignore

    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key,
        value=value,
        bits_key=bits_key,
        bits_value=bits_value,
        slot_w_k=slot_w_k,
        slot_w_v=slot_w_v,
    )
    key_cache.zero_()
    value_cache.zero_()
    torch_npu._npu_reshape_and_cache(
        key=packed_k,
        value=packed_v,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=slot_mapping,
    )
    _sync_device()


def _pack_to_cache_fused(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    bits_key: int,
    bits_value: int,
) -> None:
    """Run ``turboquant_pack_kv_for_cache_to_cache`` (production fused kernel)."""
    env = {
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "1",
    }
    key_cache.zero_()
    value_cache.zero_()
    with patch.dict(os.environ, env, clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            bits_key=bits_key,
            bits_value=bits_value,
        )
    _sync_device()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@requires_npu
def test_pack_kv_fused_accuracy(num_tokens: int) -> None:
    """Fused pack-to-cache must match ``pack`` + ``_npu_reshape_and_cache``."""
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_to_cache"):
        pytest.skip("turboquant_pack_kv_for_cache_to_cache op not available")
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache"):
        pytest.skip("turboquant_pack_kv_for_cache op not available")

    dtype = torch.float16
    T = num_tokens
    H = NUM_KV_HEADS
    D = HEAD_SIZE

    slot_w_k = turboquant_packed_bytes_per_vector(D, bits=BITS)
    slot_w_v = turboquant_packed_bytes_per_vector(D, bits=BITS)

    torch.manual_seed(42)
    key = torch.randn(T, H, D, dtype=dtype, device=DEVICE).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=DEVICE).contiguous()

    ensure_turboquant_pack_tables_registered(DEVICE, D, BITS)

    key_cache_ref, value_cache_ref, slot_mapping = _make_paged_cache(T, H, slot_w_k)
    key_cache_fused = torch.zeros_like(key_cache_ref)
    value_cache_fused = torch.zeros_like(value_cache_ref)

    # Reference: pack + scatter (attention_v1 fallback when to_cache unavailable)
    _pack_to_cache_reference(
        key=key,
        value=value,
        key_cache=key_cache_ref,
        value_cache=value_cache_ref,
        slot_mapping=slot_mapping,
        bits_key=BITS,
        bits_value=BITS,
        slot_w_k=slot_w_k,
        slot_w_v=slot_w_v,
    )

    # Fused: turboquant_pack_kv_for_cache_to_cache (production path)
    _pack_to_cache_fused(
        key=key,
        value=value,
        key_cache=key_cache_fused,
        value_cache=value_cache_fused,
        slot_mapping=slot_mapping,
        bits_key=BITS,
        bits_value=BITS,
    )

    ref_k_u8 = key_cache_ref.view(torch.uint8)
    ref_v_u8 = value_cache_ref.view(torch.uint8)
    fused_k_u8 = key_cache_fused.view(torch.uint8)
    fused_v_u8 = value_cache_fused.view(torch.uint8)

    max_diff_k = (fused_k_u8.to(torch.int16) - ref_k_u8.to(torch.int16)).abs().max().item()
    max_diff_v = (fused_v_u8.to(torch.int16) - ref_v_u8.to(torch.int16)).abs().max().item()

    print(
        f"\n[Accuracy] num_tokens={T} num_kv_heads={H} bits={BITS}\n"
        f"  key_cache   max_abs_diff = {max_diff_k}\n"
        f"  value_cache max_abs_diff = {max_diff_v}"
    )

    torch.testing.assert_close(fused_k_u8, ref_k_u8, rtol=0, atol=0)
    torch.testing.assert_close(fused_v_u8, ref_v_u8, rtol=0, atol=0)


def _parse_chrome_trace_op_stats(
    trace_path: str,
) -> list[dict]:
    """Parse a Chrome-trace JSON and return Turboquant-related NPU op events.

    Each returned dict contains: ``name``, ``dur_us``, ``cat``.
    """
    # export_chrome_trace produces <dir>/<dir>.json
    json_path = trace_path
    if not json_path.endswith(".json"):
        json_path = os.path.join(trace_path, os.path.basename(trace_path) + ".json")
    if not os.path.isfile(json_path):
        return []

    with open(json_path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    results: list[dict] = []
    for evt in events:
        name = evt.get("name", "")
        if "Turboquant" not in name and "turboquant" not in name:
            continue
        # Chrome-trace 'dur' is in microseconds
        dur = evt.get("dur", 0.0)
        cat = evt.get("cat", "")
        results.append({"name": name, "dur_us": dur, "cat": cat})

    return results


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@requires_npu
def test_pack_kv_fused_profiling(
    num_tokens: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Profile ``turboquant_pack_kv_for_cache_to_cache`` with ``torch_npu.profiler``.

    Collects operator-level timing plus Python call stacks (``with_stack``,
    ``with_modules``).  The full trace is saved for offline analysis.
    """
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_to_cache"):
        pytest.skip("turboquant_pack_kv_for_cache_to_cache op not available")

    import torch_npu  # type: ignore
    import torch_npu.profiler  # type: ignore

    dtype = torch.float16
    T = num_tokens
    H = NUM_KV_HEADS
    D = HEAD_SIZE

    slot_w_k = turboquant_packed_bytes_per_vector(D, bits=BITS)

    torch.manual_seed(42)
    key = torch.randn(T, H, D, dtype=dtype, device=DEVICE).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=DEVICE).contiguous()
    ensure_turboquant_pack_tables_registered(DEVICE, D, BITS)
    key_cache, value_cache, slot_mapping = _make_paged_cache(T, H, slot_w_k)

    # Warmup
    _pack_to_cache_fused(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        bits_key=BITS,
        bits_value=BITS,
    )
    _sync_device()

    # Profile
    tag = f"pack_kv_to_cache_tokens{T}"
    trace_dir = f"{TRACE_DIR}/{tag}"

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        # Keep FRAMEWORK/ data so Python call stacks from with_stack are retained.
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            trace_dir
        ),
        with_stack=True,
        with_modules=True,
        experimental_config=experimental_config,
    ) as prof:
        _pack_to_cache_fused(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            bits_key=BITS,
            bits_value=BITS,
        )

    # Also export Chrome-trace JSON for programmatic parsing
    chrome_path = f"{trace_dir}/chrome_trace.json"
    os.makedirs(trace_dir, exist_ok=True)
    prof.export_chrome_trace(chrome_path)

    # Parse Chrome trace to extract Turboquant op durations
    op_stats = _parse_chrome_trace_op_stats(chrome_path)

    out = f"\n[PackKV-to-Cache Profiling] num_tokens={T} num_kv_heads={H} bits={BITS}\n"
    out += f"  trace_dir: {trace_dir}\n"
    out += "  python stacks: enabled (with_stack=True, with_modules=True)\n"

    if op_stats:
        out += f"  {'Name':<45} {'Duration (us)':>14} {'Category':<15}\n"
        out += f"  {'-'*45} {'-'*14} {'-'*15}\n"
        for s in op_stats:
            out += (
                f"  {s['name']:<45} {s['dur_us']:>14.3f} {s['cat']:<15}\n"
            )
    else:
        out += "  (No Turboquant events found in Chrome trace.\n"
        out += "   Check the full trace under trace_dir with msprof or TensorBoard.)\n"

    with capsys.disabled():
        print(out, end="")

    # Sanity: at least one Turboquant event should appear
    assert len(op_stats) > 0, (
        f"No Turboquant events found in {chrome_path}. "
        "Check that the fused kernel was actually invoked."
    )
