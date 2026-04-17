"""
DynamicKV helpers for vLLM-Ascend:

- Paged-cache gather for full-sequence K/V (`gather_kv_from_paged_cache*`).
- DynamicKV: per-layer scores/indices and cross-layer budget (`DynamicKVConfig`,
  `scores_and_indices_old`, `update_and_reset_budget`).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch


def _pool_1d(scores: torch.Tensor, pooling: str, kernel_size: int) -> torch.Tensor:
    """scores: [T]"""
    if pooling == "none" or kernel_size <= 1 or scores.numel() == 0:
        return scores
    x = scores[None, None, :]  # [1,1,T]
    pad = kernel_size // 2
    if pooling == "avgpool":
        y = torch.nn.functional.avg_pool1d(
            x, kernel_size=kernel_size, stride=1, padding=pad
        )
    elif pooling == "maxpool":
        y = torch.nn.functional.max_pool1d(
            x, kernel_size=kernel_size, stride=1, padding=pad
        )
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return y[0, 0, :]


def gather_kv_from_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather contiguous K/V for one request from paged cache.

    key_cache/value_cache: [num_blocks, block_size, num_kv_heads, head_size]
    block_table_row: [max_blocks_per_seq] physical block ids
    returns:
      key/value: [seq_len, num_kv_heads, head_size]
    """
    if seq_len <= 0:
        empty = key_cache[:0, :0]
        return empty, empty
    block_size = int(key_cache.shape[1])
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    phys = block_table_row[:num_blocks_needed].to(torch.long)
    k = key_cache.index_select(0, phys).reshape(-1, key_cache.shape[2], key_cache.shape[3])
    v = value_cache.index_select(0, phys).reshape(-1, value_cache.shape[2], value_cache.shape[3])
    return k[:seq_len], v[:seq_len]


def gather_kv_from_paged_cache_batched(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Batched gather of contiguous K/V for multiple requests.

    Inputs:
      - key_cache/value_cache: [num_blocks, block_size, num_kv_heads, head_size]
      - block_tables: [B, max_blocks_per_seq] (physical block ids)
      - seq_lens: [B]

    Returns:
      - k_packed/v_packed: [sum(seq_lens), Hkv, D]
      - slots_packed: [sum(seq_lens)] int32 (slot indices for each packed token)
      - cu_seqlens: cumulative token ends, length B (same shape as actual_seq_lengths_kv style)
    """
    B = int(len(seq_lens))
    if B == 0:
        empty = key_cache[:0, :0].reshape(0, key_cache.shape[2], key_cache.shape[3])
        return empty, empty, torch.empty((0,), device=key_cache.device, dtype=torch.int32), []

    block_size = int(key_cache.shape[1])
    # Build packed physical block ids.
    phys_list: list[torch.Tensor] = []
    token_offsets_in_packed: list[int] = []
    cu: list[int] = []
    total = 0
    for i, L in enumerate(seq_lens):
        L = int(L)
        token_offsets_in_packed.append(total)
        if L <= 0:
            cu.append(total)
            continue
        num_blocks_needed = (L + block_size - 1) // block_size
        phys = block_tables[i, :num_blocks_needed].to(torch.long)
        phys_list.append(phys)
        total += L
        cu.append(total)

    if total <= 0:
        empty = key_cache[:0, :0].reshape(0, key_cache.shape[2], key_cache.shape[3])
        return empty, empty, torch.empty((0,), device=key_cache.device, dtype=torch.int32), cu

    phys_all = torch.cat(phys_list, dim=0) if phys_list else torch.empty((0,), device=key_cache.device, dtype=torch.long)
    kv_blocks = key_cache.index_select(0, phys_all)  # [sum(blocks), block_size, H, D]
    k_all = kv_blocks.reshape(-1, key_cache.shape[2], key_cache.shape[3])[:total]
    v_all = value_cache.index_select(0, phys_all).reshape(-1, value_cache.shape[2], value_cache.shape[3])[:total]

    # slots: phys_id * block_size + offset_in_block
    # We construct it by repeating each phys_id for block_size positions.
    if phys_all.numel() == 0:
        slots = torch.empty((0,), device=key_cache.device, dtype=torch.int32)
    else:
        phys_rep = phys_all.to(torch.int32).repeat_interleave(block_size)
        off = torch.arange(block_size, device=key_cache.device, dtype=torch.int32).repeat(phys_all.numel())
        slots = (phys_rep * block_size + off)[:total]

    return k_all, v_all, slots, cu


# ---------------------------------------------------------------------------
# DynamicKV: per-layer old-token scores / indices + cross-layer budget.
# (Aligned with open-source DynamicKV reference under code/DynamicKV/.)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicKVConfig:
    num_hidden_layers: int
    window_size: int
    max_capacity_prompt: int
    pooling: str = "avgpool"  # avgpool|maxpool|none
    kernel_size: int = 7
    radio_max: float = 10.0
    radio_min: float = 0.1

    @property
    def base(self) -> int:
        return max(int(self.max_capacity_prompt) - int(self.window_size), 0)


def _compute_token_scores_old(
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    window_size: int,
) -> torch.Tensor:
    """Return token importance scores for old tokens: [old_len]."""
    assert query_last.dim() == 3 and key_full.dim() == 3
    W = int(window_size)
    L = int(key_full.shape[0])
    if W <= 0 or L <= W:
        return key_full.new_zeros((0,))

    Hq = int(query_last.shape[1])
    Hkv = int(key_full.shape[1])
    D = int(query_last.shape[2])
    assert key_full.shape[2] == D
    assert Hq % Hkv == 0

    scale = 1.0 / math.sqrt(D)
    rep = Hq // Hkv
    qg = query_last.view(W, Hkv, rep, D).permute(1, 2, 0, 3)  # [Hkv, rep, W, D]
    kg = key_full.permute(1, 2, 0)  # [Hkv, D, L]
    attn_logits = torch.matmul(qg, kg).reshape(Hq, W, L) * scale
    attn = torch.nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(
        query_last.dtype
    )
    token_scores = attn.sum(dim=1).sum(dim=0)  # [L]
    return token_scores[: L - W]


def scores_and_indices_old(
    *,
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    cfg: DynamicKVConfig,
    budget_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lighter per-layer step: scores over old tokens + top-k indices (sorted).
    """
    W = int(cfg.window_size)
    L = int(key_full.shape[0])
    if W <= 0 or L <= W or query_last.numel() == 0:
        return key_full.new_zeros((0,)), key_full.new_zeros((0,), dtype=torch.long)

    old_len = L - W
    budget_size = int(min(int(budget_size), old_len))
    scores_old = _compute_token_scores_old(query_last, key_full, W)  # [old_len]
    scores_old = _pool_1d(scores_old, cfg.pooling, int(cfg.kernel_size))
    if budget_size <= 0:
        indices_old = key_full.new_zeros((0,), dtype=torch.long)
    else:
        indices_old = torch.topk(scores_old, k=budget_size, dim=0).indices
        indices_old, _ = torch.sort(indices_old)
    return scores_old, indices_old


def update_and_reset_budget(
    *,
    per_layer_scores_old: List[torch.Tensor],  # each [old_len]
    per_layer_k_budget: List[torch.Tensor],  # each [budget_size+W, Hkv, D]
    per_layer_v_budget: List[torch.Tensor],
    cfg: DynamicKVConfig,
    budget_size: int,
    head_factor: int = 1,
) -> List[int]:
    """
    Reallocate budgets across layers.

    Same role as upstream DynamicKV `update_and_reset_budget`.
    Returns per-layer old-token counts (excluding window tokens).

    ``head_factor`` is ignored: scores are head-aggregated, so cross-layer
    top-k uses ``base * num_layers`` tokens (see comments below), not
    ``base * head_factor * num_layers``.
    """
    num_layers = int(cfg.num_hidden_layers)
    base = int(cfg.base)
    if num_layers <= 0 or base <= 0:
        return [0] * max(num_layers, 0)

    need_fill_kv = base * num_layers
    min_budget = int(cfg.radio_min * base)

    valid_scores = [s for s in per_layer_scores_old if isinstance(s, torch.Tensor)]
    if len(valid_scores) != num_layers:
        return [base] * num_layers
    old_lens = [int(s.numel()) for s in per_layer_scores_old]
    min_old_len = min(old_lens) if old_lens else 0
    if min_old_len <= 0:
        return [0] * num_layers

    gather_attn = torch.stack(
        [s[:min_old_len].to(torch.float32) for s in per_layer_scores_old], dim=0
    )  # [L, old_len]

    flat = gather_attn.reshape(-1)
    flat_n = int(flat.numel())
    # Upstream DynamicKV (dynamic_v11.py) uses ``tk = base * head_num *
    # num_layers`` over a flat tensor whose volume includes a head axis.
    # Our ``per_layer_scores_old`` is head-aggregated in
    # ``_compute_token_scores_old``, so ``flat_n == num_layers * min_old_len``.
    # Using ``base * head_factor * num_layers`` here double-counts heads and
    # often makes ``tk == flat_n``, so ``topk`` covers every cell and layer
    # counts become uniform. Match upstream *density* with ``base * num_layers``.
    # ``head_factor`` is kept in the signature for call-site compatibility only.
    tk = max(1, min(int(base) * int(num_layers), flat_n))
    if tk <= 0:
        return [base] * num_layers

    topk_indices = torch.topk(flat, k=tk, dim=0).indices
    dim1 = min_old_len
    indices_0 = torch.div(topk_indices, dim1, rounding_mode="floor")  # layer ids
    counts = torch.bincount(indices_0, minlength=num_layers).to(torch.float32)
    if float(counts.sum().item()) <= 0:
        counts = torch.ones((num_layers,), device=counts.device, dtype=counts.dtype)

    norm = counts / counts.sum()
    budget_length_fix = [int((budget_size * t).item()) for t in norm]

    ss_radio = (sum(budget_length_fix) / need_fill_kv) if need_fill_kv > 0 else 1.0
    if ss_radio <= 0:
        ss_radio = 1.0
    budget_length_fix = [int(k // ss_radio) for k in budget_length_fix]

    diff = need_fill_kv - sum(budget_length_fix)
    budget_length_fix[-1] += diff

    budget_length_fix = [max(min_budget, int(x)) for x in budget_length_fix]
    diff = need_fill_kv - sum(budget_length_fix)
    budget_length_fix[-1] += diff

    out: List[int] = []
    for i in range(num_layers):
        max_old = max(int(per_layer_k_budget[i].shape[0]) - int(cfg.window_size), 0)
        out.append(int(min(budget_length_fix[i], max_old)))
    diff = need_fill_kv - sum(out)
    out[-1] = int(max(0, out[-1] + diff))
    return out

