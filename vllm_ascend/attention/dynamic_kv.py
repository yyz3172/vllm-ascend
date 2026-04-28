"""
DynamicKV helpers for vLLM-Ascend:

- Paged-cache gather for full-sequence K/V (`gather_kv_from_paged_cache*`).
- DynamicKV: per-layer scores/indices and cross-layer budget (`DynamicKVConfig`,
  `scores_and_indices_old`, `update_and_reset_budget`, `cap_keep_indices_chronological`).
- Validation mode: save/retrieve important_mask for verifying token selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
# Scratch buffers for chunked softmax to reduce allocator fragmentation.
# Keyed by (device, Hkv, rep, W, chunk_size). Values are float32 tensors.
_DYNKV_SCRATCH: dict[tuple[str, int, int, int, int], torch.Tensor] = {}


def _get_dynkv_softmax_scratch(
    *,
    device: torch.device,
    Hkv: int,
    rep: int,
    W: int,
    chunk_size: int,
) -> torch.Tensor:
    key = (str(device), int(Hkv), int(rep), int(W), int(chunk_size))
    buf = _DYNKV_SCRATCH.get(key)
    if isinstance(buf, torch.Tensor) and buf.device == device and buf.dtype == torch.float32:
        if buf.shape == (Hkv, rep, W, chunk_size):
            return buf
    buf = torch.empty((Hkv, rep, W, chunk_size), device=device, dtype=torch.float32)
    _DYNKV_SCRATCH[key] = buf
    return buf


def clear_dynkv_softmax_scratch(device: torch.device | None = None) -> None:
    """Clear cached scratch buffers to reduce reserved memory.

    When running under tight memory / fragmentation, keeping large persistent
    scratch buffers can increase `reserved` and make small allocations fail.
    """
    if device is None:
        _DYNKV_SCRATCH.clear()
        return
    d = str(device)
    for k in list(_DYNKV_SCRATCH.keys()):
        if k and k[0] == d:
            _DYNKV_SCRATCH.pop(k, None)



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


def _token_indices_to_slots(
    *,
    block_table_row: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map logical token indices to flattened paged-cache slots.

    block_table_row: [max_blocks_per_seq] physical block ids
    token_indices: [T] logical positions in [0, seq_len)
    returns: [T] slots in [0, num_blocks*block_size)
    """
    if token_indices.numel() == 0:
        return token_indices.new_empty((0,), dtype=torch.long)
    bs = int(block_size)
    ti = token_indices.to(torch.long)
    block_idx = torch.div(ti, bs, rounding_mode="floor")
    off = ti - block_idx * bs
    phys = block_table_row.index_select(0, block_idx).to(torch.long)
    return phys * bs + off


def _compute_token_scores_old_per_kv_head_paged_cache(
    *,
    query_last: torch.Tensor,  # [W, Hq, D]
    key_cache: torch.Tensor,  # [num_blocks, block_size, Hkv, D]
    block_table_row: torch.Tensor,  # [max_blocks_per_seq]
    seq_len: int,
    window_size: int,
    softmax_chunk_size: int,
) -> torch.Tensor:
    """Compute per-head old-token scores without gathering full key_full.

    Returns scores_old: [Hkv, old_len] where old_len = seq_len - W_eff.
    """
    L = int(seq_len)
    if L <= 1:
        return key_cache.new_zeros((int(key_cache.shape[2]), 0))
    W_cfg = int(window_size)
    if W_cfg <= 0:
        return key_cache.new_zeros((int(key_cache.shape[2]), 0))

    # q_last effective tail (must match captured length and <= L).
    W_eff = int(min(W_cfg, int(query_last.shape[0]), L))
    if W_eff <= 0 or L <= W_eff:
        return key_cache.new_zeros((int(key_cache.shape[2]), 0))
    if int(query_last.shape[0]) != W_eff:
        query_last = query_last[-W_eff:].contiguous()

    Hq = int(query_last.shape[1])
    Hkv = int(key_cache.shape[2])
    D = int(query_last.shape[2])
    assert int(key_cache.shape[3]) == D
    assert Hq % Hkv == 0
    rep = Hq // Hkv

    block_size = int(key_cache.shape[1])
    k_flat = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])

    # q: [Hkv, rep, W, D]
    qg = query_last.view(W_eff, Hkv, rep, D).permute(1, 2, 0, 3)
    scale = 1.0 / math.sqrt(D)

    chunk_size = max(128, int(softmax_chunk_size))
    scratch = _get_dynkv_softmax_scratch(
        device=qg.device, Hkv=Hkv, rep=rep, W=W_eff, chunk_size=chunk_size
    )

    tail_base = int(L - W_eff)
    q_idx = torch.arange(W_eff, device=qg.device)

    # Pass 1: max logits
    max_logits = torch.full(
        (Hkv, rep, W_eff),
        torch.finfo(torch.float32).min,
        device=qg.device,
        dtype=torch.float32,
    )
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        c = int(end - start)
        ti = torch.arange(start, end, device=qg.device, dtype=torch.long)
        slots = _token_indices_to_slots(
            block_table_row=block_table_row, token_indices=ti, block_size=block_size
        )
        k_chunk = k_flat.index_select(0, slots)  # [c, Hkv, D]
        kg = k_chunk.permute(1, 2, 0).unsqueeze(1)  # [Hkv, 1, D, c]
        logits = torch.matmul(qg, kg) * scale  # [Hkv, rep, W, c]

        # Causal mask for keys in tail window.
        if end > tail_base and start < L:
            ov_s = max(start, tail_base)
            ov_e = end
            ov_len = ov_e - ov_s
            if ov_len > 0:
                key_j = (torch.arange(ov_s, ov_e, device=qg.device) - tail_base)
                m = (key_j[None, :] > q_idx[:, None]).to(logits.dtype)
                m = m * torch.finfo(logits.dtype).min
                logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] = (
                    logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] + m[None, None, :, :]
                )

        s = scratch[..., :c]
        s.copy_(logits)  # cast to fp32
        max_logits = torch.maximum(max_logits, s.amax(dim=-1))

    # Pass 2: sumexp
    sumexp = torch.zeros((Hkv, rep, W_eff), device=qg.device, dtype=torch.float32)
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        c = int(end - start)
        ti = torch.arange(start, end, device=qg.device, dtype=torch.long)
        slots = _token_indices_to_slots(
            block_table_row=block_table_row, token_indices=ti, block_size=block_size
        )
        k_chunk = k_flat.index_select(0, slots)
        kg = k_chunk.permute(1, 2, 0).unsqueeze(1)
        logits = torch.matmul(qg, kg) * scale

        if end > tail_base and start < L:
            ov_s = max(start, tail_base)
            ov_e = end
            ov_len = ov_e - ov_s
            if ov_len > 0:
                key_j = (torch.arange(ov_s, ov_e, device=qg.device) - tail_base)
                m = (key_j[None, :] > q_idx[:, None]).to(logits.dtype)
                m = m * torch.finfo(logits.dtype).min
                logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] = (
                    logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] + m[None, None, :, :]
                )

        s = scratch[..., :c]
        s.copy_(logits)
        s.sub_(max_logits[..., None])
        torch.exp(s, out=s)
        sumexp = sumexp + s.sum(dim=-1)

    sumexp = torch.clamp(sumexp, min=1e-20)

    # Pass 3: accumulate token scores for old tokens only.
    old_len = int(L - W_eff)
    token_scores_old = torch.zeros((Hkv, old_len), device=qg.device, dtype=torch.float32)
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        c = int(end - start)
        ti = torch.arange(start, end, device=qg.device, dtype=torch.long)
        slots = _token_indices_to_slots(
            block_table_row=block_table_row, token_indices=ti, block_size=block_size
        )
        k_chunk = k_flat.index_select(0, slots)
        kg = k_chunk.permute(1, 2, 0).unsqueeze(1)
        logits = torch.matmul(qg, kg) * scale

        if end > tail_base and start < L:
            ov_s = max(start, tail_base)
            ov_e = end
            ov_len = ov_e - ov_s
            if ov_len > 0:
                key_j = (torch.arange(ov_s, ov_e, device=qg.device) - tail_base)
                m = (key_j[None, :] > q_idx[:, None]).to(logits.dtype)
                m = m * torch.finfo(logits.dtype).min
                logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] = (
                    logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] + m[None, None, :, :]
                )

        s = scratch[..., :c]
        s.copy_(logits)
        s.sub_(max_logits[..., None])
        torch.exp(s, out=s)
        s.div_(sumexp[..., None])
        # Sum over rep and W => [Hkv, c]
        contrib = s.sum(dim=1).sum(dim=1)
        # Only keep old tokens (positions < old_len).
        if start < old_len:
            e2 = min(end, old_len)
            token_scores_old[:, start:e2] = token_scores_old[:, start:e2] + contrib[:, : (e2 - start)]

    return token_scores_old


def scores_and_indices_old_perhead_aggregated_paged_cache(
    *,
    query_last: torch.Tensor,  # [W, Hq, D]
    key_cache: torch.Tensor,  # [num_blocks, block_size, Hkv, D]
    block_table_row: torch.Tensor,  # [max_blocks_per_seq]
    seq_len: int,
    cfg: DynamicKVConfig,
    budget_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming variant of `scores_and_indices_old_perhead_aggregated` for paged cache."""
    W = int(cfg.window_size)
    L = int(seq_len)
    Hkv = int(key_cache.shape[2])
    if W <= 0 or L <= W or query_last.numel() == 0:
        return key_cache.new_zeros((Hkv, 0)), key_cache.new_zeros((0,), dtype=torch.long)

    old_len = L - W
    budget_size = int(min(int(budget_size), old_len))
    scores_old = _compute_token_scores_old_per_kv_head_paged_cache(
        query_last=query_last,
        key_cache=key_cache,
        block_table_row=block_table_row,
        seq_len=L,
        window_size=W,
        softmax_chunk_size=int(getattr(cfg, "softmax_chunk_size", 1024) or 1024),
    )

    # Pooling per head
    if cfg.pooling != "none" and int(cfg.kernel_size) > 1 and old_len > 0:
        pooled: list[torch.Tensor] = []
        for h in range(Hkv):
            pooled.append(_pool_1d(scores_old[h], cfg.pooling, int(cfg.kernel_size)))
        scores_old = torch.stack(pooled, dim=0)

    if budget_size <= 0:
        return scores_old, key_cache.new_zeros((0,), dtype=torch.long)

    indices_per_head = torch.topk(scores_old, k=budget_size, dim=-1).indices
    flat_indices = indices_per_head.flatten()
    union_indices = torch.unique(flat_indices)
    # Importance for union: max over heads
    union_scores = scores_old.index_select(1, union_indices).amax(dim=0)
    union_order = torch.topk(union_scores, k=int(union_scores.numel()), dim=0).indices
    indices_old = union_indices.index_select(0, union_order)
    return scores_old, indices_old


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
    # Chunk size (in tokens) for chunked softmax in token score computation.
    # This controls the key-length chunking: higher -> faster but higher peak memory.
    softmax_chunk_size: int = 1024
    radio_max: float = 10.0
    radio_min: float = 0.1

    @property
    def base(self) -> int:
        return max(int(self.max_capacity_prompt) - int(self.window_size), 0)


def _compute_token_scores_old(
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    window_size: int,
    *,
    softmax_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return token importance scores for old tokens: [old_len]."""
    assert query_last.dim() == 3 and key_full.dim() == 3
    # Reuse the chunked per-kv-head implementation and sum across KV heads.
    # This avoids materializing [Hq, W, L] and matches the semantics:
    # token_scores_old = sum_{head} sum_{rep,w} softmax(logits)[token]
    scores_h = _compute_token_scores_old_per_kv_head(
        query_last,
        key_full,
        int(window_size),
        softmax_chunk_size=softmax_chunk_size,
    )
    if scores_h.numel() == 0:
        return key_full.new_zeros((0,))
    return scores_h.sum(dim=0)


def _compute_token_scores_old_per_kv_head(
    query_last: torch.Tensor,  # [W, Hq, D]
    key_full: torch.Tensor,  # [L, Hkv, D]
    window_size: int,
    *,
    softmax_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return token importance scores for old tokens per KV head: [Hkv, old_len]."""
    assert query_last.dim() == 3 and key_full.dim() == 3
    # Window size is a *target* tail size; for correctness we must match the
    # actual captured tail length.
    W_cfg = int(window_size)
    L = int(key_full.shape[0])
    if W_cfg <= 0 or L <= 1:
        return key_full.new_zeros((int(key_full.shape[1]), 0))

    Hq = int(query_last.shape[1])
    Hkv = int(key_full.shape[1])
    D = int(query_last.shape[2])
    assert key_full.shape[2] == D
    assert Hq % Hkv == 0

    W_eff = int(min(W_cfg, int(query_last.shape[0]), int(L)))
    if W_eff <= 0 or L <= W_eff:
        return key_full.new_zeros((Hkv, 0))
    if int(query_last.shape[0]) != W_eff:
        query_last = query_last[-W_eff:].contiguous()

    scale = 1.0 / math.sqrt(D)
    rep = Hq // Hkv
    # q: [Hkv, rep, W, D]
    qg = query_last.view(W_eff, Hkv, rep, D).permute(1, 2, 0, 3)

    # Chunked softmax to avoid materializing [Hkv, rep, W, L].
    chunk_size = int(softmax_chunk_size) if softmax_chunk_size is not None else 1024
    chunk_size = max(128, chunk_size)

    device = qg.device
    logit_dtype = qg.dtype
    scratch = _get_dynkv_softmax_scratch(
        device=device, Hkv=Hkv, rep=rep, W=W_eff, chunk_size=chunk_size
    )

    # 1) pass: compute per-(Hkv,rep,W) max over L for numerically-stable softmax.
    max_logits = torch.full((Hkv, rep, W_eff),
                            torch.finfo(torch.float32).min,
                            device=device,
                            dtype=torch.float32)

    # Optional causal mask within tail window: only affects keys in [L-W_eff, L).
    tail_base = int(L - W_eff)
    q_idx = torch.arange(W_eff, device=device)
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        c = int(end - start)
        k_chunk = key_full[start:end]  # [c, Hkv, D]
        # k: [Hkv, 1, D, c]
        kg = k_chunk.permute(1, 2, 0).unsqueeze(1)
        logits = torch.matmul(qg, kg) * scale  # [Hkv, rep, W, c]
        # Apply causal mask for overlap with tail window.
        if end > tail_base and start < L:
            ov_s = max(start, tail_base)
            ov_e = end
            ov_len = ov_e - ov_s
            if ov_len > 0:
                key_j = (torch.arange(ov_s, ov_e, device=device) - tail_base)  # [ov_len] in [0,W_eff)
                # mask[w, j] = -inf if j > w else 0
                m = (key_j[None, :] > q_idx[:, None]).to(logit_dtype)
                m = m * torch.finfo(logit_dtype).min
                logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] = (
                    logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] + m[None, None, :, :]
                )
        # Reuse scratch: cast logits -> fp32 into scratch slice, then reduce.
        s = scratch[..., :c]
        s.copy_(logits)  # casts to float32
        max_logits = torch.maximum(max_logits, s.amax(dim=-1))

    # 2) pass: compute sumexp over L for each (Hkv,rep,W)
    sumexp = torch.zeros((Hkv, rep, W_eff), device=device, dtype=torch.float32)
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        c = int(end - start)
        k_chunk = key_full[start:end]
        kg = k_chunk.permute(1, 2, 0).unsqueeze(1)
        logits = torch.matmul(qg, kg) * scale
        if end > tail_base and start < L:
            ov_s = max(start, tail_base)
            ov_e = end
            ov_len = ov_e - ov_s
            if ov_len > 0:
                key_j = (torch.arange(ov_s, ov_e, device=device) - tail_base)
                m = (key_j[None, :] > q_idx[:, None]).to(logit_dtype)
                m = m * torch.finfo(logit_dtype).min
                logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] = (
                    logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] + m[None, None, :, :]
                )
        # exp(logits - max) with scratch reuse to avoid large allocations.
        s = scratch[..., :c]
        s.copy_(logits)  # fp32
        s.sub_(max_logits[..., None])
        torch.exp(s, out=s)
        sumexp = sumexp + s.sum(dim=-1)

    # Avoid division by zero.
    sumexp = torch.clamp(sumexp, min=1e-20)

    # 3) pass: accumulate token scores without storing full attention matrix.
    token_scores = torch.zeros((Hkv, L), device=device, dtype=torch.float32)
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        c = int(end - start)
        k_chunk = key_full[start:end]
        kg = k_chunk.permute(1, 2, 0).unsqueeze(1)
        logits = torch.matmul(qg, kg) * scale
        if end > tail_base and start < L:
            ov_s = max(start, tail_base)
            ov_e = end
            ov_len = ov_e - ov_s
            if ov_len > 0:
                key_j = (torch.arange(ov_s, ov_e, device=device) - tail_base)
                m = (key_j[None, :] > q_idx[:, None]).to(logit_dtype)
                m = m * torch.finfo(logit_dtype).min
                logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] = (
                    logits[:, :, :, (ov_s - start):(ov_s - start + ov_len)] + m[None, None, :, :]
                )
        # Reuse scratch: compute exp and normalize in-place.
        s = scratch[..., :c]
        s.copy_(logits)  # fp32
        s.sub_(max_logits[..., None])
        torch.exp(s, out=s)
        s.div_(sumexp[..., None])
        # Sum over rep and W => [Hkv, c]
        token_scores[:, start:end] = token_scores[:, start:end] + s.sum(dim=1).sum(dim=1)

    # Return only old tokens: [Hkv, L - W_eff]
    old_len = int(L - W_eff)
    if old_len <= 0:
        return key_full.new_zeros((Hkv, 0))
    return token_scores[:, :old_len].to(query_last.dtype)


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
    scores_old = _compute_token_scores_old_per_kv_head(
        query_last,
        key_full,
        W,
        softmax_chunk_size=int(getattr(cfg, "softmax_chunk_size", 1024) or 1024),
    )  # [Hkv, old_len]
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
    scores_old = _compute_token_scores_old(
        query_last,
        key_full,
        W,
        softmax_chunk_size=int(getattr(cfg, "softmax_chunk_size", 1024) or 1024),
    )  # [old_len]
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
    scores_old = _compute_token_scores_old_per_kv_head(
        query_last,
        key_full,
        W,
        softmax_chunk_size=int(getattr(cfg, "softmax_chunk_size", 1024) or 1024),
    )

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

