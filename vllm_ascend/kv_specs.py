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


register_turboquant_kv_spec_managers()
