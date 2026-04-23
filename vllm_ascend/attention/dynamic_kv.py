"""
DynamicKV helpers for vLLM-Ascend:

- Paged-cache gather for full-sequence K/V (`gather_kv_from_paged_cache*`).
- DynamicKV: per-layer scores/indices and cross-layer budget (`DynamicKVConfig`,
  `scores_and_indices_old`, `update_and_reset_budget`, `cap_keep_indices_chronological`).
- Validation mode: save/retrieve important_mask for verifying token selection.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Validation mode: per-request per-layer important_mask storage.
# Used for verifying token selection effectiveness without compression.
# ---------------------------------------------------------------------------

# Structure: {request_id: {layer_idx: important_mask}}
_VALIDATION_MASKS: Dict[str, Dict[int, torch.Tensor]] = {}


def save_validation_mask(
    request_id: str,
    layer_idx: int,
    important_mask: torch.Tensor,
) -> None:
    """Save important_mask for validation mode.
    
    Args:
        request_id: Unique request identifier.
        layer_idx: Layer index.
        important_mask: [seq_len] bool tensor, True = important token.
    """
    if request_id not in _VALIDATION_MASKS:
        _VALIDATION_MASKS[request_id] = {}
    _VALIDATION_MASKS[request_id][layer_idx] = important_mask


def get_validation_mask(
    request_id: str,
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """Retrieve important_mask for validation mode.
    
    Returns:
        important_mask tensor or None if not found.
    """
    req_masks = _VALIDATION_MASKS.get(request_id)
    if req_masks is None:
        return None
    return req_masks.get(layer_idx)


def clear_validation_masks(request_id: Optional[str] = None) -> None:
    """Clear validation masks.
    
    Args:
        request_id: If provided, only clear masks for this request.
                   If None, clear all masks.
    """
    global _VALIDATION_MASKS
    if request_id is None:
        _VALIDATION_MASKS = {}
    elif request_id in _VALIDATION_MASKS:
        del _VALIDATION_MASKS[request_id]


def build_attention_mask_from_important_mask(
    important_mask: torch.Tensor,
    query_len: int = 1,
) -> torch.Tensor:
    """Convert 1D important_mask to 2D attention mask (uint8/bool).
    
    Args:
        important_mask: [seq_len] bool tensor, True = important token.
        query_len: Number of query tokens (1 for decode).
    
    Returns:
        attn_mask: [query_len, seq_len] uint8 tensor.
            0 for important tokens (keep), 1 for unimportant tokens (mask out).
    """
    seq_len = important_mask.shape[0]
    if important_mask.dtype != torch.bool:
        important_mask = important_mask.to(torch.bool)
    # FlashAttention kernels on Ascend expect atten_mask to be bool/uint8.
    # Use uint8: 1 = masked (unimportant), 0 = keep (important).
    key_mask = (~important_mask).to(torch.uint8)
    # Broadcast to [query_len, seq_len]
    attn_mask = key_mask.unsqueeze(0).expand(query_len, seq_len)
    return attn_mask.contiguous()


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
    # NOTE: `torch.matmul` broadcasts batch dims by right-alignment.
    # qg batch dims are [Hkv, rep], kg batch dims are [Hkv]. Without an extra
    # singleton dim, kg's [Hkv] would align to rep and can fail when rep != Hkv
    # (e.g. TP=1 with GQA: Hkv=8, rep=4). Unsqueeze so kg batch dims become
    # [Hkv, 1] and broadcast correctly to [Hkv, rep].
    kg = key_full.permute(1, 2, 0).unsqueeze(1)  # [Hkv, 1, D, L]
    attn_logits = torch.matmul(qg, kg).reshape(Hq, W, L) * scale
    # Apply causal mask within the tail window (align with open-source DynamicKV).
    # Only affects attention from the last W queries to the last W keys.
    if W > 0 and L >= W:
        # mask[i, j] = -inf when j > i (future), else 0
        mask = torch.full((W, W),
                          torch.finfo(attn_logits.dtype).min,
                          device=attn_logits.device,
                          dtype=attn_logits.dtype)
        mask = torch.triu(mask, diagonal=1)
        attn_logits[:, :, L - W:L] = attn_logits[:, :, L - W:L] + mask[None, :, :]
    attn = torch.nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(
        query_last.dtype
    )
    token_scores = attn.sum(dim=1).sum(dim=0)  # [L]
    return token_scores[: L - W]


def _compute_token_scores_old_per_kv_head(
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    window_size: int,
) -> torch.Tensor:
    """Return token importance scores for old tokens per KV head: [Hkv, old_len]."""
    assert query_last.dim() == 3 and key_full.dim() == 3
    W = int(window_size)
    L = int(key_full.shape[0])
    if W <= 0 or L <= W:
        return key_full.new_zeros((int(key_full.shape[1]), 0))

    Hq = int(query_last.shape[1])
    Hkv = int(key_full.shape[1])
    D = int(query_last.shape[2])
    assert key_full.shape[2] == D
    assert Hq % Hkv == 0

    scale = 1.0 / math.sqrt(D)
    rep = Hq // Hkv
    # q: [Hkv, rep, W, D]
    qg = query_last.view(W, Hkv, rep, D).permute(1, 2, 0, 3)
    # k: [Hkv, 1, D, L] (broadcast across rep)
    kg = key_full.permute(1, 2, 0).unsqueeze(1)
    attn_logits = torch.matmul(qg, kg) * scale  # [Hkv, rep, W, L]
    # Apply causal mask within the tail window (align with open-source DynamicKV).
    if W > 0 and L >= W:
        mask = torch.full((W, W),
                          torch.finfo(attn_logits.dtype).min,
                          device=attn_logits.device,
                          dtype=attn_logits.dtype)
        mask = torch.triu(mask, diagonal=1)
        attn_logits[:, :, :, L - W:L] = attn_logits[:, :, :, L - W:L] + mask[None, None, :, :]
    attn = torch.nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(
        query_last.dtype
    )
    # Sum over rep and W -> [Hkv, L]
    token_scores = attn.sum(dim=1).sum(dim=1)
    return token_scores[:, : L - W]


def scores_and_indices_old_per_kv_head(
    *,
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    cfg: DynamicKVConfig,
    budget_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-layer step (open-source style): per-head scores + per-head top-k indices.

    Returns:
      - scores_old: [Hkv, old_len]
      - indices_old: [Hkv, k] (NOT sorted chronologically; matches upstream behavior)
    """
    W = int(cfg.window_size)
    L = int(key_full.shape[0])
    Hkv = int(key_full.shape[1])
    if W <= 0 or L <= W or query_last.numel() == 0:
        return (
            key_full.new_zeros((Hkv, 0)),
            key_full.new_zeros((Hkv, 0), dtype=torch.long),
        )

    old_len = L - W
    k = int(min(int(budget_size), old_len))
    scores_old = _compute_token_scores_old_per_kv_head(query_last, key_full, W)  # [Hkv, old_len]
    if cfg.pooling != "none" and int(cfg.kernel_size) > 1 and old_len > 0:
        pooled: list[torch.Tensor] = []
        for h in range(Hkv):
            pooled.append(_pool_1d(scores_old[h], cfg.pooling, int(cfg.kernel_size)))
        scores_old = torch.stack(pooled, dim=0)
    if k <= 0:
        indices_old = key_full.new_zeros((Hkv, 0), dtype=torch.long)
    else:
        indices_old = torch.topk(scores_old, k=k, dim=-1).indices  # [Hkv, k]
    return scores_old, indices_old


def update_and_reset_budget_per_kv_head(
    *,
    per_layer_scores_old: List[torch.Tensor],  # each [Hkv, old_len]
    per_layer_k_budget: List[torch.Tensor],
    per_layer_v_budget: List[torch.Tensor],
    cfg: DynamicKVConfig,
    budget_size: int,
) -> List[int]:
    """Cross-layer budget reallocation (open-source density) for per-head scores.

    Matches upstream `tk = base * head_num * num_layers` behavior by flattening
    `[layer, head, token]` volume.
    Returns per-layer old-token counts (excluding window tokens).
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

    # Align old_len across layers by min.
    old_lens = [int(s.shape[-1]) for s in per_layer_scores_old]
    min_old_len = min(old_lens) if old_lens else 0
    if min_old_len <= 0:
        return [0] * num_layers

    # Align head count by min (defensive; should be constant).
    head_lens = [int(s.shape[0]) for s in per_layer_scores_old]
    H = min(head_lens) if head_lens else 0
    if H <= 0:
        return [base] * num_layers

    gather_attn = torch.stack(
        [s[:H, :min_old_len].to(torch.float32) for s in per_layer_scores_old], dim=0
    )  # [L, H, old_len]

    flat = gather_attn.reshape(-1)
    flat_n = int(flat.numel())
    tk = max(1, min(int(base) * int(H) * int(num_layers), flat_n))
    if tk <= 0:
        return [base] * num_layers

    topk_indices = torch.topk(flat, k=tk, dim=0).indices
    dim1 = H * min_old_len
    indices_0 = torch.div(topk_indices, dim1, rounding_mode="floor")  # layer ids
    counts = torch.bincount(indices_0, minlength=num_layers).to(torch.float32)
    if float(counts.sum().item()) <= 0:
        counts = torch.ones((num_layers,), device=counts.device, dtype=counts.dtype)

    norm = counts / counts.sum()
    budget_length_fix = [int((budget_size * t).item()) for t in norm]

    ss_radio = (sum(budget_length_fix) / need_fill_kv) if need_fill_kv > 0 else 1.0
    if ss_radio <= 0:
        ss_radio = 1.0
    budget_length_fix = [int(kv // ss_radio) for kv in budget_length_fix]

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


def scores_and_indices_old(
    *,
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    cfg: DynamicKVConfig,
    budget_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lighter per-layer step: scores over old tokens + top-k indices (sorted).

    NOTE: This version uses head-aggregated scores. For per-head scores with
    aggregated indices (matching open-source DynamicKV density), use
    ``scores_and_indices_old_perhead_aggregated`` instead.
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
        # Keep importance order (no sort): truncation idx[:budget] keeps most important.
        # Final chronological order is restored by cap_keep_indices_chronological.
    return scores_old, indices_old


def scores_and_indices_old_perhead_aggregated(
    *,
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    cfg: DynamicKVConfig,
    budget_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-layer step with per-head scores + aggregated indices (union strategy).

    To preserve all tokens that ANY head considers important (matching the semantic
    of open-source per-head compression under vLLM's token-level Paged KV Cache):
    - Compute per-head scores [Hkv, old_len] for cross-layer budget allocation
    - Each head selects its top-k tokens
    - Return the UNION of all heads' selections (sorted by importance for truncation)

    Returns:
      - scores_old: [Hkv, old_len] (per-head, for cross-layer topk)
      - indices_old: [union_size] (union of all heads' top-k, importance order)
    """
    W = int(cfg.window_size)
    L = int(key_full.shape[0])
    Hkv = int(key_full.shape[1])
    if W <= 0 or L <= W or query_last.numel() == 0:
        return (
            key_full.new_zeros((Hkv, 0)),
            key_full.new_zeros((0,), dtype=torch.long),
        )

    old_len = L - W
    budget_size = int(min(int(budget_size), old_len))

    # Step 1: Compute per-head scores [Hkv, old_len]
    scores_old = _compute_token_scores_old_per_kv_head(query_last, key_full, W)

    # Step 2: Apply pooling per head
    if cfg.pooling != "none" and int(cfg.kernel_size) > 1 and old_len > 0:
        pooled: list[torch.Tensor] = []
        for h in range(Hkv):
            pooled.append(_pool_1d(scores_old[h], cfg.pooling, int(cfg.kernel_size)))
        scores_old = torch.stack(pooled, dim=0)

    if budget_size <= 0:
        return scores_old, key_full.new_zeros((0,), dtype=torch.long)

    # Step 3: Per-head top-k indices [Hkv, k]
    indices_per_head = torch.topk(scores_old, k=budget_size, dim=-1).indices

    # Step 4: Union of all heads' selections
    # This ensures no head loses tokens it considers important.
    flat_indices = indices_per_head.flatten()  # [Hkv * k]
    union_indices = torch.unique(flat_indices)  # unique tokens selected by any head

    # Sort union by combined importance score (for truncation: idx[:budget] keeps best)
    # Combined score = selection frequency × aggregated importance
    token_counts = torch.zeros(old_len, device=key_full.device, dtype=torch.float32)
    token_counts.scatter_add_(
        0,
        flat_indices.to(torch.long),
        torch.ones_like(flat_indices, dtype=torch.float32),
    )
    token_importance = scores_old.sum(dim=0)  # [old_len]
    combined_score = token_counts * token_importance.to(torch.float32)

    # Get scores for union tokens and sort by importance (descending)
    union_scores = combined_score[union_indices.to(torch.long)]
    sorted_order = torch.argsort(union_scores, descending=True)
    indices_sorted_by_importance = union_indices[sorted_order]

    return scores_old, indices_sorted_by_importance.to(torch.long)


def cap_keep_indices_chronological(
    *,
    idx_old: torch.Tensor,
    old_budget: int,
    tail_idx: torch.Tensor,
    cap: int,
    device: torch.device,
) -> Tuple[torch.Tensor, bool]:
    """Merge old-segment top-k indices with tail window, unique-sort, then cap.

    If more than ``cap`` distinct positions: **always keep the full tail window**
    (all indices in ``tail_idx`` that appear in the merge), then fill remaining
    slots with **old-region** indices **closest to the tail** (largest original
    indices first). Shared by ``impl=attn`` rewrite and ``impl=offload`` path.

    Returns ``(keep_sorted, truncated)`` where ``truncated`` is True if we had
    to drop any index that would have been kept in the uncapped merge.
    """
    C = int(cap)
    if C <= 0:
        return torch.zeros(0, device=device, dtype=torch.long), False

    old_part = idx_old[: int(old_budget)].to(torch.long)
    if tail_idx.numel() == 0:
        merged = old_part
    else:
        merged = torch.cat([old_part, tail_idx.to(torch.long)], dim=0)
    keep_sorted = torch.unique(merged, sorted=True)
    n0 = int(keep_sorted.numel())
    if n0 <= C:
        return keep_sorted, False

    tail_set = {int(x) for x in tail_idx.tolist()}
    ks_list = [int(x) for x in keep_sorted.tolist()]
    tail_in = [x for x in ks_list if x in tail_set]
    old_in = [x for x in ks_list if x not in tail_set]

    if len(tail_in) + len(old_in) <= C:
        return keep_sorted, False

    if len(tail_in) >= C:
        # Extremely rare (e.g. W > C): keep the last C chronological positions.
        kept = ks_list[-C:]
        return torch.tensor(kept, device=device, dtype=torch.long), True

    budget_old = C - len(tail_in)
    old_kept = old_in[-budget_old:] if budget_old > 0 else []
    result = sorted(set(tail_in) | set(old_kept))
    return torch.tensor(result, device=device, dtype=torch.long), True


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

