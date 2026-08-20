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

from __future__ import annotations

from types import SimpleNamespace

import torch

import vllm_ascend.ops.turboquant_kv_cache as tq_ops


def _paged_attention_inputs(num_tokens: int = 1) -> dict:
    return {
        "query": torch.randn(num_tokens, 8, 128, dtype=torch.float16),
        "key_cache": torch.empty(1, 16, 1, 128, dtype=torch.uint8),
        "value_cache": torch.empty(1, 16, 1, 128, dtype=torch.uint8),
        "block_tables": torch.ones(1, 1, dtype=torch.int32),
        "actual_seq_lengths_q": list(range(1, num_tokens + 1)),
        "actual_seq_lengths_kv": [16] * num_tokens,
        "head_size": 128,
        "num_heads": 8,
        "num_key_value_heads": 1,
        "block_size": 16,
        "scale": 128**-0.5,
    }


def test_attention_paged8bit_unavailable_op_returns_none(monkeypatch):
    monkeypatch.setattr(tq_ops, "_c_ascend_turboquant_op_available", lambda _: False)

    assert tq_ops.turboquant_attention_paged8bit(**_paged_attention_inputs()) is None


def test_attention_paged8bit_rejects_invalid_seq_lens_before_codebook_setup(monkeypatch):
    monkeypatch.setattr(tq_ops, "_c_ascend_turboquant_op_available", lambda _: True)

    def fail_decode_tables(*args, **kwargs):
        raise AssertionError("decode tables should not be prepared for invalid metadata")

    monkeypatch.setattr(tq_ops, "_turboquant_fused_8bit_decode_tables", fail_decode_tables)

    inputs = _paged_attention_inputs(num_tokens=2)
    inputs["actual_seq_lengths_q"] = [2, 4]
    assert tq_ops.turboquant_attention_paged8bit(**inputs) is None


def test_attention_paged8bit_calls_fused_op_when_enabled(monkeypatch):
    monkeypatch.setattr(tq_ops, "_c_ascend_turboquant_op_available", lambda _: True)

    codebook = torch.empty(256, 128, dtype=torch.float16)
    rotation = torch.empty(128, 128, dtype=torch.float16)
    monkeypatch.setattr(
        tq_ops,
        "_turboquant_fused_8bit_decode_tables",
        lambda *args, **kwargs: (codebook, rotation, codebook, rotation),
    )

    calls = []

    def fake_attention(*args):
        calls.append(args)
        return torch.zeros(1, 8, 128, dtype=torch.float16)

    monkeypatch.setattr(
        tq_ops.torch.ops,
        "_C_ascend",
        SimpleNamespace(turboquant_attention_paged8bit=fake_attention),
        raising=False,
    )

    out = tq_ops.turboquant_attention_paged8bit(**_paged_attention_inputs())

    assert out.shape == (1, 8, 128)
    assert len(calls) == 1
    assert calls[0][4] == [1]
    assert calls[0][5] == [16]


def test_fused_fia_missing_op_uses_decode_fallback(monkeypatch):
    expected = torch.ones(1, 8, 128, dtype=torch.float16)
    fallback_calls = []

    def fake_fallback(*args):
        fallback_calls.append(args)
        return expected

    monkeypatch.setattr(tq_ops, "_turboquant_fused_infer_attention_score_8bit_impl", fake_fallback)
    monkeypatch.setattr(
        tq_ops.torch.ops,
        "_C_ascend",
        SimpleNamespace(),
        raising=False,
    )

    out = tq_ops.turboquant_fused_infer_attention_score_8bit(
        query=torch.randn(1, 8, 128, dtype=torch.float16),
        key_cache=torch.empty(1, 16, 1, 128, dtype=torch.uint8),
        value_cache=torch.empty(1, 16, 1, 128, dtype=torch.uint8),
        block_tables=torch.ones(1, 1, dtype=torch.int32),
        atten_mask=torch.zeros(1, 1, dtype=torch.int8),
        actual_seq_lengths_q=[1],
        actual_seq_lengths_kv=[16],
        head_size=128,
        num_heads=8,
        num_key_value_heads=1,
        block_size=16,
        scale=128**-0.5,
    )

    assert out is expected
    assert len(fallback_calls) == 1
