# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for BitResidual k8v4 Prefill→FIA / Decode→attn routing helpers."""

from __future__ import annotations

import importlib

import pytest


def _should_use_br_fia(
    *,
    attn_state_name: str,
    fia_env: bool,
    decode_fia_env: bool = False,
    has_block_table: bool = True,
) -> bool:
    """Mirror attention_v1 BR FIA gate (no NPU).

    PrefillNoCache keeps block_table=None and uses float key/value — never BR FIA.
    """
    if not has_block_table:
        return False
    if attn_state_name in ("PrefillCacheHit", "ChunkedPrefill"):
        return fia_env
    if attn_state_name == "DecodeOnly":
        return decode_fia_env
    return False


@pytest.mark.parametrize(
    "state,fia_env,decode_fia,has_bt,expect",
    [
        ("DecodeOnly", True, False, True, False),
        ("DecodeOnly", False, True, True, True),
        ("DecodeOnly", True, True, True, True),
        ("ChunkedPrefill", True, False, True, True),
        ("PrefillCacheHit", True, False, True, True),
        ("PrefillNoCache", True, True, False, False),
        ("PrefillNoCache", True, True, True, False),  # state alone never selects FIA
        ("ChunkedPrefill", False, False, True, False),
        ("PrefillCacheHit", False, True, True, False),
    ],
)
def test_br_fia_route_gate(
    state: str,
    fia_env: bool,
    decode_fia: bool,
    has_bt: bool,
    expect: bool,
) -> None:
    assert (
        _should_use_br_fia(
            attn_state_name=state,
            fia_env=fia_env,
            decode_fia_env=decode_fia,
            has_block_table=has_bt,
        )
        is expect
    )


def test_bit_residual_fia_env_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_BIT_RESIDUAL_FIA", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA", raising=False)
    envs = importlib.reload(importlib.import_module("vllm_ascend.envs"))
    assert envs.VLLM_ASCEND_BIT_RESIDUAL_FIA is False
    assert envs.VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA is False


def test_bit_residual_fia_env_can_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_BIT_RESIDUAL_FIA", "1")
    monkeypatch.setenv("VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA", "1")
    envs = importlib.reload(importlib.import_module("vllm_ascend.envs"))
    assert envs.VLLM_ASCEND_BIT_RESIDUAL_FIA is True
    assert envs.VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA is True


def test_wrapper_symbol_exported() -> None:
    """Assert wrappers exist in source without importing torch_npu (host CI)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "vllm_ascend" / "ops" / "turboquant_kv_cache.py"
    text = src.read_text(encoding="utf-8")
    assert "def bit_residual_fia_paged_k8v4(" in text
    assert "def bit_residual_attention_paged_k8v4(" in text
    # Both wrappers accept optional in-place out= (caller buffer).
    assert "out: torch.Tensor | None = None," in text
    fia_idx = text.index("def bit_residual_fia_paged_k8v4(")
    fia_chunk = text[fia_idx : fia_idx + 1200]
    assert "out: torch.Tensor | None = None," in fia_chunk
