#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Fallback when upstream vLLM lacks :class:`~vllm.v1.kv_cache_interface.TQFullAttentionSpec`.
# Mixed K/V TurboQuant uses ``tq_slot_size = P_k + P_v`` bytes per logical (block × head)
# row across K+V (split into separate K/V tensors in the ascend runner).

from __future__ import annotations

from dataclasses import dataclass, replace
from typing_extensions import Self

from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True)
class AscendTQFullAttentionSpecFallback(FullAttentionSpec):
    """Mirrors upstream TQ-full page size via ``tq_slot_size`` (K+V packed widths)."""

    tq_slot_size: int = 0

    @property
    def page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        assert all(getattr(s, "tq_slot_size", 0) == getattr(specs[0],
                                                             "tq_slot_size", 0)
                   for s in specs), ("All TurboQuant-full layers must use the "
                                     "same tq_slot_size.")
        return replace(merged, tq_slot_size=getattr(specs[0], "tq_slot_size", 0))


def turboquant_attention_spec_cls() -> type:
    """Prefer upstream ``TQFullAttentionSpec`` when present."""
    try:
        from vllm.v1.kv_cache_interface import (  # noqa: PLC0415
            TQFullAttentionSpec as _TQ,
        )
        return _TQ  # type: ignore[no-any-return]
    except ImportError:
        return AscendTQFullAttentionSpecFallback


def _patch_turboquant_page_size_bytes() -> None:
    """Monkey-patch ``TurboQuantAttentionSpec.page_size_bytes`` for k8v4.

    Upstream uses ``2 * block_size * num_kv_heads * max(pk, pv)`` which
    zero-pads the narrower (4-bit value) side to match the wider (8-bit key)
    side.  This is correct for the symmetric slab layout but **overestimates**
    by ~30% for the BitResidual k8v4 asymmetric layout where K and V have
    separate packed widths.

    The inflated page_size_bytes causes the scheduler to allocate fewer
    ``num_gpu_blocks`` than the physical cache can hold, leaving ~23% of KV
    cache capacity invisible to the scheduler.  Under prefix-cache pressure
    this triggers preemptions that degrade TPOT.

    For k8v4 (bits_key=8, bits_value=4, head_size=128) we replace the
    ``2 * max(pk, pv)`` formula with the actual asymmetric width
    ``pk_k8v4 + pv_k8v4`` computed from the BitResidual sub-block strides.
    """
    try:
        from vllm.v1.kv_cache_interface import (  # noqa: PLC0415
            TurboQuantAttentionSpec,
        )
    except ImportError:
        return

    from vllm_ascend.ops.turboquant_kv_cache import (
        bit_residual_k8v4_key_packed_width,
        bit_residual_k8v4_value_packed_width,
    )

    _original_page_size_bytes = TurboQuantAttentionSpec.page_size_bytes

    @property  # type: ignore[misc]
    def _patched_page_size_bytes(self: "TurboQuantAttentionSpec") -> int:
        # BitResidual k8v4 asymmetric layout: use actual per-head packed
        # widths instead of the symmetric max(pk, pv) padding.
        if (self.bits_key == 8 and self.bits_value == 4
                and self.head_size == 128):
            pk_k8v4 = bit_residual_k8v4_key_packed_width(self.block_size)
            pv_k8v4 = bit_residual_k8v4_value_packed_width(self.block_size)
            # pk_k8v4/pv_k8v4 are per-(block, head) totals (already include
            # the block_size/16 sub-block count), so we only multiply by
            # num_kv_heads -- NOT by block_size.
            return self.num_kv_heads * (pk_k8v4 + pv_k8v4)
        # All other TurboQuant configs (4/4, 8/8, etc.) use the original
        # symmetric formula.
        return _original_page_size_bytes.fget(self)  # type: ignore[union-attr]

    TurboQuantAttentionSpec.page_size_bytes = _patched_page_size_bytes


def register_turboquant_kv_spec_managers() -> None:
    """Wire TQ-full spec classes into vLLM's ``spec_manager_map`` (same as TurboQuant)."""
    from vllm.v1.core.single_type_kv_cache_manager import (  # noqa: PLC0415
        FullAttentionManager,
        spec_manager_map,
    )

    spec_manager_map.setdefault(AscendTQFullAttentionSpecFallback,
                                FullAttentionManager)
    try:
        from vllm.v1.kv_cache_interface import (  # noqa: PLC0415
            TQFullAttentionSpec as _TQF,
        )
    except ImportError:
        pass
    else:
        spec_manager_map.setdefault(_TQF, FullAttentionManager)

    # Patch page_size_bytes for k8v4 so the scheduler sees the correct
    # (smaller) per-block size and allocates more blocks from the same
    # available memory.
    _patch_turboquant_page_size_bytes()


register_turboquant_kv_spec_managers()
