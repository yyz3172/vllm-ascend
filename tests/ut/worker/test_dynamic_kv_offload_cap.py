# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DynamicKV offload chronological cap helper."""

import torch

from vllm_ascend.attention.dynamic_kv import cap_keep_indices_chronological


def _cap(
    *,
    idx_old: list[int],
    old_budget: int,
    tail_idx: list[int],
    cap: int,
) -> tuple[list[int], bool]:
    device = torch.device("cpu")
    idx_t = torch.tensor(idx_old, dtype=torch.long)
    tail_t = torch.tensor(tail_idx, dtype=torch.long)
    keep, trunc = cap_keep_indices_chronological(
        idx_old=idx_t,
        old_budget=old_budget,
        tail_idx=tail_t,
        cap=cap,
        device=device,
    )
    return [int(x) for x in keep.tolist()], trunc


def test_cap_no_truncation_when_merge_fits():
    keep, trunc = _cap(
        idx_old=[0, 1, 2, 3, 4],
        old_budget=5,
        tail_idx=[10, 11],
        cap=20,
    )
    assert not trunc
    assert keep == [0, 1, 2, 3, 4, 10, 11]


def test_cap_tail_always_kept_when_old_must_drop():
    # Merge unique: 0..9 + tail 18..19 -> 12 tokens; cap=8 -> drop oldest old, keep tail.
    keep, trunc = _cap(
        idx_old=list(range(10)),
        old_budget=10,
        tail_idx=[18, 19],
        cap=8,
    )
    assert trunc
    assert 18 in keep and 19 in keep
    assert len(keep) == 8
    assert keep == sorted(keep)


def test_cap_only_old_region_exceeds_cap_keeps_largest_old_indices():
    # No tail in merge; cap below unique count -> last C chronological.
    keep, trunc = _cap(
        idx_old=list(range(20)),
        old_budget=20,
        tail_idx=[],
        cap=5,
    )
    assert trunc
    assert keep == [15, 16, 17, 18, 19]


def test_cap_tail_window_larger_than_cap_keeps_last_c_chronological():
    keep, trunc = _cap(
        idx_old=[],
        old_budget=0,
        tail_idx=[0, 1, 2, 3, 4],
        cap=3,
    )
    assert trunc
    assert keep == [2, 3, 4]
