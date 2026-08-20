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

Matches the production path in ``attention_v1.py``.  With ``PACK_OP=v2`` (default),
Python routes to the monolithic C++ ``aclnnTurboquantPackKvForCacheV2ToCache`` kernel.

Accuracy compares the v2-to-cache binding against the legacy decomposed fallback:

    ``turboquant_pack_kv_for_cache``  +  ``_npu_reshape_and_cache``

Tests pin ``VLLM_ASCEND_TURBOQUANT_PACK_OP=v2`` so the binding uses the v2
monolithic to_cache kernel.

Batch sizes (``num_tokens``):

    - Small  (T=1):     decode scenario
    - Medium (T=512):   moderate prefill
    - Large  (T=2048):  large-scale prefill

Accuracy: byte-exact match on paged KV cache contents.
Profiling: uses ``torch_npu.profiler`` with Python call stacks (``with_stack`` /
``with_modules``); traces saved under TRACE_DIR for TensorBoard / msprof.
Expect ``TurboquantPackKvForCacheV2ToCache`` (v2 monolithic); no ReshapeAndCache.

Run:

    export ASCEND_RT_VISIBLE_DEVICES=4
    pytest -s tests/ut/ops/test_turboquant_pack_kv_fused_batch.py -v

Profiler trace saved to TRACE_DIR for each batch size.
"""

from __future__ import annotations

import json
import os
import re
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

# Pin pack kernel so Python routing uses the v2 monolithic to_cache op.
DEFAULT_PACK_OP = "v2"

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


def _turboquant_test_env(*, pack_op: str = DEFAULT_PACK_OP) -> dict[str, str]:
    return {
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": "1",
        "VLLM_ASCEND_TURBOQUANT_PACK_OP": pack_op,
    }


def _require_pack_to_cache_ops(pack_op: str = DEFAULT_PACK_OP) -> None:
    if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache_to_cache"):
        pytest.skip("turboquant_pack_kv_for_cache_to_cache routing entry not available")
    if pack_op == "v2":
        if not _c_ascend_turboquant_op_available(
            "turboquant_pack_kv_for_cache_v2_to_cache"
        ):
            pytest.skip("turboquant_pack_kv_for_cache_v2_to_cache op not available")
        if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache"):
            pytest.skip("turboquant_pack_kv_for_cache op not available")
        return
    if pack_op == "fused":
        if not _c_ascend_turboquant_op_available("turboquant_pack_kv_for_cache"):
            pytest.skip("turboquant_pack_kv_for_cache op not available")
        return
    pack_op_name = f"turboquant_pack_kv_for_cache_{pack_op}"
    if not _c_ascend_turboquant_op_available(pack_op_name):
        pytest.skip(f"{pack_op_name} op not available")


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
    """Legacy pack + ``_npu_reshape_and_cache`` reference path."""
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


def _pack_to_cache_binding(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    bits_key: int,
    bits_value: int,
    pack_op: str = DEFAULT_PACK_OP,
) -> None:
    """Run ``turboquant_pack_kv_for_cache_to_cache`` (Python routing to v2 C++ op)."""
    key_cache.zero_()
    value_cache.zero_()
    query_start_loc = torch.tensor(
        [0, key.shape[0]],
        device=slot_mapping.device,
        dtype=torch.int32,
    )
    with patch.dict(os.environ, _turboquant_test_env(pack_op=pack_op), clear=False):
        turboquant_pack_kv_for_cache_to_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            num_reqs=1,
            bits_key=bits_key,
            bits_value=bits_value,
        )
    _sync_device()


def _normalize_trace_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _classify_trace_op(name: str) -> str | None:
    """Return ``fused_pack`` / ``pack`` / ``scatter`` / ``other_turboquant`` or None."""
    norm = _normalize_trace_name(name)
    # Monolithic pack+scatter kernels (check *tocache before bare v2/v3).
    if (
        "turboquantpackkvforcachev2tocache" in norm
        or "turboquantpackkvforcachev3tocache" in norm
        or "turboquantpackkvforcachetocache" in norm
    ):
        return "fused_pack"
    if "turboquantpackkvforcachev2" in norm or "turboquantpackkvforcachev3" in norm:
        return "pack"
    if "reshapeandcache" in norm or "reshapecache" in norm:
        return "scatter"
    if "turboquant" in norm:
        return "other_turboquant"
    return None


def _parse_chrome_trace_op_stats(
    trace_path: str,
) -> list[dict]:
    """Parse Chrome trace JSON and return pack/scatter/TurboQuant op events.

    Each dict contains: ``name``, ``dur_us``, ``cat``, ``kind``.
    """
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
        kind = _classify_trace_op(name)
        if kind is None:
            continue
        dur = evt.get("dur", 0.0)
        cat = evt.get("cat", "")
        results.append({"name": name, "dur_us": dur, "cat": cat, "kind": kind})

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@requires_npu
def test_pack_kv_fused_accuracy(num_tokens: int) -> None:
    """v2-to-cache binding must match legacy pack + scatter."""
    _require_pack_to_cache_ops(DEFAULT_PACK_OP)

    dtype = torch.float16
    T = num_tokens
    H = NUM_KV_HEADS
    D = HEAD_SIZE

    slot_w_k = turboquant_packed_bytes_per_vector(D, bits=BITS)
    slot_w_v = turboquant_packed_bytes_per_vector(D, bits=BITS)

    torch.manual_seed(42)
    key = torch.randn(T, H, D, dtype=dtype, device=DEVICE).contiguous()
    value = torch.randn(T, H, D, dtype=dtype, device=DEVICE).contiguous()

    with patch.dict(os.environ, _turboquant_test_env(), clear=False):
        ensure_turboquant_pack_tables_registered(DEVICE, D, BITS)

    key_cache_ref, value_cache_ref, slot_mapping = _make_paged_cache(T, H, slot_w_k)
    key_cache_binding = torch.zeros_like(key_cache_ref)
    value_cache_binding = torch.zeros_like(value_cache_ref)

    # Reference: legacy pack + _npu_reshape_and_cache
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

    # Binding: v2 monolithic to_cache (production default path)
    _pack_to_cache_binding(
        key=key,
        value=value,
        key_cache=key_cache_binding,
        value_cache=value_cache_binding,
        slot_mapping=slot_mapping,
        bits_key=BITS,
        bits_value=BITS,
    )

    ref_k_u8 = key_cache_ref.view(torch.uint8)
    ref_v_u8 = value_cache_ref.view(torch.uint8)
    binding_k_u8 = key_cache_binding.view(torch.uint8)
    binding_v_u8 = value_cache_binding.view(torch.uint8)

    max_diff_k = (
        (binding_k_u8.to(torch.int16) - ref_k_u8.to(torch.int16)).abs().max().item()
    )
    max_diff_v = (
        (binding_v_u8.to(torch.int16) - ref_v_u8.to(torch.int16)).abs().max().item()
    )

    print(
        f"\n[Accuracy] num_tokens={T} num_kv_heads={H} bits={BITS} "
        f"pack_op={DEFAULT_PACK_OP}\n"
        f"  key_cache   max_abs_diff = {max_diff_k}\n"
        f"  value_cache max_abs_diff = {max_diff_v}"
    )

    torch.testing.assert_close(binding_k_u8, ref_k_u8, rtol=0, atol=0)
    torch.testing.assert_close(binding_v_u8, ref_v_u8, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@requires_npu
def test_pack_kv_fused_profiling(
    num_tokens: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Profile v2-to-cache binding (monolithic TurboquantPackKvForCacheV2ToCache)."""
    _require_pack_to_cache_ops(DEFAULT_PACK_OP)

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
    with patch.dict(os.environ, _turboquant_test_env(), clear=False):
        ensure_turboquant_pack_tables_registered(DEVICE, D, BITS)
    key_cache, value_cache, slot_mapping = _make_paged_cache(T, H, slot_w_k)

    # Warmup
    _pack_to_cache_binding(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        bits_key=BITS,
        bits_value=BITS,
    )
    _sync_device()

    tag = f"pack_kv_to_cache_{DEFAULT_PACK_OP}_tokens{T}"
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
        _pack_to_cache_binding(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            bits_key=BITS,
            bits_value=BITS,
        )

    chrome_path = f"{trace_dir}/chrome_trace.json"
    os.makedirs(trace_dir, exist_ok=True)
    prof.export_chrome_trace(chrome_path)

    op_stats = _parse_chrome_trace_op_stats(chrome_path)
    fused_ops = [s for s in op_stats if s["kind"] == "fused_pack"]
    scatter_ops = [s for s in op_stats if s["kind"] == "scatter"]

    out = (
        f"\n[PackKV-to-Cache Profiling] num_tokens={T} num_kv_heads={H} "
        f"bits={BITS} pack_op={DEFAULT_PACK_OP}\n"
    )
    out += f"  trace_dir: {trace_dir}\n"
    out += "  python stacks: enabled (with_stack=True, with_modules=True)\n"
    out += "  expected ops: TurboquantPackKvForCacheV2ToCache (no ReshapeAndCache)\n"

    if op_stats:
        out += f"  {'Name':<45} {'Duration (us)':>14} {'Kind':<12} {'Category':<15}\n"
        out += f"  {'-'*45} {'-'*14} {'-'*12} {'-'*15}\n"
        for s in op_stats:
            out += (
                f"  {s['name']:<45} {s['dur_us']:>14.3f} "
                f"{s['kind']:<12} {s['cat']:<15}\n"
            )
        fused_us = sum(s["dur_us"] for s in fused_ops)
        scatter_us = sum(s["dur_us"] for s in scatter_ops)
        out += f"  monolithic pack subtotal (us): {fused_us:.3f}\n"
        out += f"  scatter subtotal (us):       {scatter_us:.3f}\n"
    else:
        out += (
            "  (No pack/scatter events found in Chrome trace.\n"
            "   Check kernel_detail.csv under trace_dir with msprof.)\n"
        )

    with capsys.disabled():
        print(out, end="")

    assert len(fused_ops) > 0, (
        f"No TurboquantPackKvForCacheV2ToCache events in {chrome_path}. "
        f"pack_op={DEFAULT_PACK_OP}"
    )
    assert len(scatter_ops) == 0, (
        f"Unexpected ReshapeAndCache events in {chrome_path} for v2 pack_op. "
        "v2_to_cache should not call scatter separately."
    )
