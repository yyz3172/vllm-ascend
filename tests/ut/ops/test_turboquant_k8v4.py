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

"""Reference-path unit tests for TurboQuant **K8V4** (key 8-bit, value 4-bit).

These tests exercise the pure-PyTorch reference encode/decode paths (no custom
NPU op required), so they run on CPU. They verify that:

* K and V are quantized with their own bit widths (8 vs 4);
* asymmetric packed widths (P_k=130, P_v=66) round-trip correctly;
* the shared-``max`` cache layout (both rows padded to 130 bytes) round-trips;
* the paged compact decode returns K/V decoded with the right per-side codebook.

Run standalone:  python3 tests/ut/ops/test_turboquant_k8v4.py
Run via pytest:  pytest -s tests/ut/ops/test_turboquant_k8v4.py
"""

from __future__ import annotations

import torch

from vllm_ascend.ops.turboquant_kv_cache import (
    turboquant_decode_kv_cache_compact,
    turboquant_dequantize_from_packed_bytes,
    turboquant_pack_kv_for_cache,
    turboquant_packed_bytes_per_vector,
    turboquant_quantize_to_packed_bytes,
)

D = 128
SEED = 42
BITS_KEY = 8
BITS_VALUE = 4


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float(), b.float(), dim=-1
    ).mean().item()


def test_packed_widths_are_asymmetric() -> None:
    """K (8-bit) uses 130 bytes/vector, V (4-bit) uses 66 bytes/vector."""
    pk = turboquant_packed_bytes_per_vector(D, bits=BITS_KEY)
    pv = turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE)
    assert pk == D + 2 == 130
    assert pv == D // 2 + 2 == 66
    assert pk != pv


def test_roundtrip_per_side() -> None:
    """Quantize/dequantize each side with its own bit width; K should beat V."""
    torch.manual_seed(SEED)
    x = torch.randn(256, D, dtype=torch.float16)

    packed_k = turboquant_quantize_to_packed_bytes(x, bits=BITS_KEY)
    packed_v = turboquant_quantize_to_packed_bytes(x, bits=BITS_VALUE)
    assert packed_k.shape[-1] == 130
    assert packed_v.shape[-1] == 66

    k_hat = turboquant_dequantize_from_packed_bytes(
        packed_k, head_size=D, dtype=torch.float16, bits=BITS_KEY
    )
    v_hat = turboquant_dequantize_from_packed_bytes(
        packed_v, head_size=D, dtype=torch.float16, bits=BITS_VALUE
    )
    cos_k = _cos(x, k_hat)
    cos_v = _cos(x, v_hat)
    print(f"[K8V4] roundtrip cos: key(8bit)={cos_k:.5f} value(4bit)={cos_v:.5f}")
    assert cos_k > 0.90, f"key cos_sim too low: {cos_k}"
    assert cos_v > 0.90, f"value cos_sim too low: {cos_v}"
    # 8-bit key should reconstruct at least as well as 4-bit value.
    assert cos_k >= cos_v - 1e-3


def test_pack_kv_for_cache_asymmetric_slots() -> None:
    """Reference pack keeps K/V on separate widths and decodes correctly."""
    torch.manual_seed(SEED)
    key = torch.randn(64, 4, D, dtype=torch.float16)
    value = torch.randn(64, 4, D, dtype=torch.float16)
    pk = turboquant_packed_bytes_per_vector(D, bits=BITS_KEY)
    pv = turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE)

    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key,
        value=value,
        bits_key=BITS_KEY,
        bits_value=BITS_VALUE,
        slot_w_k=pk,
        slot_w_v=pv,
    )
    assert packed_k.shape[-1] == pk
    assert packed_v.shape[-1] == pv

    k_hat = turboquant_dequantize_from_packed_bytes(
        packed_k, head_size=D, dtype=torch.float16, bits=BITS_KEY
    )
    v_hat = turboquant_dequantize_from_packed_bytes(
        packed_v, head_size=D, dtype=torch.float16, bits=BITS_VALUE
    )
    assert k_hat.shape == key.shape
    assert v_hat.shape == value.shape
    assert _cos(key, k_hat) > 0.90
    assert _cos(value, v_hat) > 0.90


def test_pack_kv_for_cache_shared_max_slots() -> None:
    """Shared-``max`` layout (both rows padded to 130) also round-trips.

    This mirrors the actual on-device cache created by ``get_kv_cache_shape``,
    which uses ``max(P_k, P_v)`` for both K and V.
    """
    torch.manual_seed(SEED)
    key = torch.randn(64, 4, D, dtype=torch.float16)
    value = torch.randn(64, 4, D, dtype=torch.float16)
    slot_w = max(
        turboquant_packed_bytes_per_vector(D, bits=BITS_KEY),
        turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE),
    )

    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key,
        value=value,
        bits_key=BITS_KEY,
        bits_value=BITS_VALUE,
        slot_w_k=slot_w,
        slot_w_v=slot_w,
    )
    assert packed_k.shape[-1] == slot_w
    assert packed_v.shape[-1] == slot_w  # V padded from 66 -> 130

    # Dequantize reads only the logical prefix (66 bytes) for the 4-bit value.
    v_hat = turboquant_dequantize_from_packed_bytes(
        packed_v, head_size=D, dtype=torch.float16, bits=BITS_VALUE
    )
    k_hat = turboquant_dequantize_from_packed_bytes(
        packed_k, head_size=D, dtype=torch.float16, bits=BITS_KEY
    )
    assert _cos(key, k_hat) > 0.90
    assert _cos(value, v_hat) > 0.90


def test_decode_kv_cache_compact_k8v4() -> None:
    """Paged compact decode uses bits_key for K and bits_value for V."""
    torch.manual_seed(SEED)
    num_blocks, block_size, num_heads = 6, 16, 4
    slot_w = max(
        turboquant_packed_bytes_per_vector(D, bits=BITS_KEY),
        turboquant_packed_bytes_per_vector(D, bits=BITS_VALUE),
    )

    key = torch.randn(num_blocks, block_size, num_heads, D, dtype=torch.float16)
    value = torch.randn(num_blocks, block_size, num_heads, D, dtype=torch.float16)

    packed_k, packed_v = turboquant_pack_kv_for_cache(
        key=key.reshape(-1, num_heads, D),
        value=value.reshape(-1, num_heads, D),
        bits_key=BITS_KEY,
        bits_value=BITS_VALUE,
        slot_w_k=slot_w,
        slot_w_v=slot_w,
    )
    key_cache = packed_k.reshape(num_blocks, block_size, num_heads, slot_w)
    value_cache = packed_v.reshape(num_blocks, block_size, num_heads, slot_w)

    block_tables = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.int32)
    k_dec, v_dec, bt_compact = turboquant_decode_kv_cache_compact(
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        head_size=D,
        dtype=torch.float16,
        bits_key=BITS_KEY,
        bits_value=BITS_VALUE,
    )
    # Compare decoded workspace against the original blocks it maps to.
    used = torch.unique(block_tables[block_tables >= 0])
    for compact_i, phys in enumerate(used.tolist()):
        cos_k = _cos(key[phys], k_dec[compact_i])
        cos_v = _cos(value[phys], v_dec[compact_i])
        assert cos_k > 0.90, f"block {phys} key cos={cos_k}"
        assert cos_v > 0.90, f"block {phys} value cos={cos_v}"
    assert bt_compact.shape == block_tables.shape


def _run_all() -> None:
    test_packed_widths_are_asymmetric()
    test_roundtrip_per_side()
    test_pack_kv_for_cache_asymmetric_slots()
    test_pack_kv_for_cache_shared_max_slots()
    test_decode_kv_cache_compact_k8v4()
    print("All K8V4 reference-path tests passed.")


if __name__ == "__main__":
    _run_all()
