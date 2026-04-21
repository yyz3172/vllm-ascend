"""
DynamicKV offload implementation for vLLM-Ascend.

Post-prefill pass (reference ``code/DynamicKV/`` does in-model gather instead).

When ``prompt_len <= prompt_kv_len_budget``, metadata is a full prefix ``0..L-1`` (no
compression). When ``L > prompt_kv_len_budget``, scores + cross-layer budget select
old tokens; retained K/V are **packed** into the leading prompt slots via
``index_copy_`` (see ``run_offload_rewrite_and_build_updates``).

Env: ``VLLM_ASCEND_DYNKV_STRICT=1`` aborts if ``L > prompt_kv_len_budget`` but Q
capture is missing (no compression possible).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.attention.dynamic_kv import (DynamicKVConfig,
                                              cap_keep_indices_chronological,
                                              gather_kv_from_paged_cache_batched,
                                              scores_and_indices_old,
                                              scores_and_indices_old_per_kv_head,
                                              update_and_reset_budget)
from vllm_ascend.attention.dynamic_kv import update_and_reset_budget_per_kv_head


@dataclass
class OffloadCaptureContext:
    # Request IDs in current batch order.
    req_ids: list[str]
    # Per-request query slices into q_proj output [T,...].
    # start[i], end[i] define tokens belonging to req i in this step.
    q_start: list[int]
    q_end: list[int]
    # Total prompt length (seq len) per request.
    seq_lens: list[int]
    # True when this step completes prompt (last chunk).
    is_last_chunk: bool


def try_extract_layer_idx_from_name(name: str) -> int | None:
    try:
        return int(extract_layer_index(name, num_attn_module=1))
    except Exception:
        return None


def _dynamic_kv_payload_full_prefix(
    *,
    num_layers: int,
    layer_indices: list[int],
    L: int,
) -> dict[str, Any]:
    """Full prompt prefix ``0..L-1`` for each layer present (no compression)."""
    kl = list(range(int(L)))
    full_lens: list[int] = [-1] * num_layers
    full_idx: list[list[int]] = [[] for _ in range(num_layers)]
    for li in layer_indices:
        full_lens[li] = int(L)
        full_idx[li] = list(kl)
    return {
        "per_layer_kv_lens": full_lens,
        "per_layer_keep_indices": full_idx,
    }


def install_qproj_hooks(
    *,
    model: torch.nn.Module,
    should_enable: Callable[[], bool],
    get_capture_ctx: Callable[[], OffloadCaptureContext | None],
    q_last_store: dict[str, dict[int, torch.Tensor]],
    window_size: int,
) -> list[Any]:
    """
    Install forward hooks on q_proj-like modules.

    Stores per request per layer q_last tensor: [W, Hq, D] on NPU.
    """
    handles: list[Any] = []

    def _store_q_slices(
        q: torch.Tensor,
        *,
        module_name: str,
        ctx: OffloadCaptureContext,
    ) -> None:
        """q: [T, Hq, D] on device."""
        layer_idx = try_extract_layer_idx_from_name(module_name)
        if layer_idx is None:
            return
        W = int(window_size)
        for i, rid in enumerate(ctx.req_ids):
            s = int(ctx.q_start[i])
            e = int(ctx.q_end[i])
            if e <= s:
                continue
            q_req = q[s:e]
            if q_req.numel() == 0:
                continue
            q_last = q_req[-min(W, int(q_req.shape[0])):].contiguous()
            per_req = q_last_store.setdefault(rid, {})
            per_req[layer_idx] = q_last

    def _hook_q_proj(module, inputs, output, *, module_name: str):
        if not should_enable():
            return
        ctx = get_capture_ctx()
        if ctx is None:
            return
        if output is None or not isinstance(output, torch.Tensor):
            return
        q = output
        if q.dim() == 2:
            q = q[:, None, :]
        if q.dim() != 3:
            return
        _store_q_slices(q, module_name=module_name, ctx=ctx)

    def _hook_qkv_proj(module, inputs, output, *, module_name: str):
        """Llama/Mistral: fused QKV linear; slice Q from concatenated output."""
        if not should_enable():
            return
        ctx = get_capture_ctx()
        if ctx is None:
            return
        out = output[0] if isinstance(output, tuple) else output
        if out is None or not isinstance(out, torch.Tensor):
            return
        osizes = getattr(module, "output_sizes", None)
        nh = getattr(module, "num_heads", None)
        hd = getattr(module, "head_size", None)
        if (
            not isinstance(osizes, (list, tuple))
            or len(osizes) < 1
            or nh is None
            or hd is None
        ):
            return
        # Column-parallel QKV: each rank's last dim matches
        # ``output_partition_sizes[0]`` (gather_output=False). Using
        # ``output_sizes[0]`` alone counts the pre-shard logical width and
        # wrongly requires ``tp_size`` more channels than exist on this rank.
        gather_out = bool(getattr(module, "gather_output", False))
        oparts = getattr(module, "output_partition_sizes", None)
        if gather_out:
            q_ch = int(osizes[0])
        elif isinstance(oparts, (list, tuple)) and len(oparts) >= 1:
            q_ch = int(oparts[0])
        else:
            tp_sz = int(getattr(module, "tp_size", 1) or 1)
            q_ch = int(osizes[0]) // max(tp_sz, 1)
        if out.dim() < 2 or out.shape[-1] < q_ch:
            return
        q_flat = out[..., :q_ch]
        nh_i, hd_i = int(nh), int(hd)
        last_d = int(q_flat.shape[-1])
        if last_d == nh_i * hd_i:
            q = q_flat.reshape(q_flat.shape[0], nh_i, hd_i)
        elif hd_i > 0 and last_d % hd_i == 0:
            # Ascend/custom matmul may expose a different local head count than
            # ``self.num_heads`` on some paths; keep head_dim fixed.
            nh_eff = last_d // hd_i
            q = q_flat.reshape(q_flat.shape[0], nh_eff, hd_i)
        else:
            return
        _store_q_slices(q, module_name=module_name, ctx=ctx)

    hooked_qkv: set[str] = set()
    for name, mod in model.named_modules():
        # Fused path (Mistral-7B / Llama): only qkv_proj exists.
        if "qkv_proj" in name and hasattr(mod, "output_sizes"):
            try:
                h = mod.register_forward_hook(
                    lambda m, inp, out, mn=name: _hook_qkv_proj(
                        m, inp, out, module_name=mn
                    )
                )
                handles.append(h)
                hooked_qkv.add(name)
            except Exception:
                continue

    for name, mod in model.named_modules():
        # Separate q_proj (some architectures); skip if this layer uses qkv only.
        if "q_proj" not in name:
            continue
        prefix = name.rsplit(".", 1)[0]
        if any(existing.startswith(prefix + ".") for existing in hooked_qkv):
            continue
        try:
            h = mod.register_forward_hook(
                lambda m, inp, out, mn=name: _hook_q_proj(m, inp, out, module_name=mn)
            )
            handles.append(h)
        except Exception:
            continue

    logger.info(
        "[DynamicKV][offload] installed %d Q-capture hooks (qkv_proj + q_proj)",
        len(handles),
    )
    return handles


def run_offload_rewrite_and_build_updates(
    *,
    req_ids: list[str],
    seq_lens: list[int],
    block_tables: torch.Tensor,
    kv_caches: dict[str, Any],
    q_last_store: dict[str, dict[int, torch.Tensor]],
    max_capacity: int,
    window_size: int,
    pooling: str,
    kernel_size: int,
    radio_max: float,
    radio_min: float,
    num_layers: int,
) -> dict[str, dict[str, Any]]:
    """
    Returns kv_transfer_params_updates payload:
      { req_id: { "dynamic_kv": { per_layer_kv_lens, per_layer_keep_indices } } }
    """
    if not req_ids or num_layers <= 0:
        return {}
    B = len(req_ids)
    assert len(seq_lens) == B

    W = int(window_size)
    C = int(max_capacity)
    if C <= 0:
        return {}
    # Skip rewrite when compression delta is too small; this avoids triggering
    # potentially fragile packed-prefix semantics for negligible memory/transfer wins.
    # NOTE: 128 matches vLLM-Ascend supported paged-attn block size.
    min_rewrite_delta = 1

    def _full_prefix_update(rid: str, L: int) -> None:
        updates[rid] = {
            "dynamic_kv": _dynamic_kv_payload_full_prefix(
                num_layers=num_layers,
                layer_indices=layer_indices_all,
                L=L,
            )
        }

    def _validate_payload_or_fallback(
        *,
        rid: str,
        L: int,
        full_lens: list[int],
        full_idx: list[list[int]],
    ) -> bool:
        """Return True if payload is consistent; otherwise overwrite as full prefix."""
        try:
            if len(full_lens) != num_layers or len(full_idx) != num_layers:
                _full_prefix_update(rid, L)
                return False
            for li in layer_indices_all:
                kv_len = int(full_lens[li])
                idx = full_idx[li]
                if not isinstance(idx, list):
                    _full_prefix_update(rid, L)
                    return False
                # `per_layer_keep_indices` is optional in offload mode: decode uses
                # packed-prefix layout and only needs `per_layer_kv_lens`.
                # When per-head selection is enabled, there is no single
                # chronological keep list in original prompt space, so idx can be [].
                if idx and kv_len != len(idx):
                    _full_prefix_update(rid, L)
                    return False
                if kv_len <= 0:
                    _full_prefix_update(rid, L)
                    return False
                if kv_len > L:
                    _full_prefix_update(rid, L)
                    return False
                # If idx is provided (non-empty), it must be strictly increasing
                # and within [0, L).
                if idx:
                    last = -1
                    for x in idx:
                        xi = int(x)
                        if xi < 0 or xi >= L or xi <= last:
                            _full_prefix_update(rid, L)
                            return False
                        last = xi
            return True
        except Exception:
            _full_prefix_update(rid, L)
            return False

    cfg = DynamicKVConfig(
        num_hidden_layers=int(num_layers),
        window_size=int(W),
        max_capacity_prompt=int(C),
        pooling=str(pooling),
        kernel_size=int(kernel_size),
        radio_max=float(radio_max),
        radio_min=float(radio_min),
    )

    # We allocate per-layer budgets via update_and_reset_budget.
    # First compute per-layer old scores and indices for each request.
    updates: dict[str, dict[str, Any]] = {}

    # Normalize kv_caches entries to (k_cache, v_cache) tensors.
    # We assume each layer has its own k/v cache tensors.
    layer_items: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for layer_name, kv in kv_caches.items():
        li = try_extract_layer_idx_from_name(layer_name)
        if li is None or li < 0 or li >= num_layers:
            continue
        k_cache = None
        v_cache = None
        if isinstance(kv, (tuple, list)) and len(kv) >= 2:
            if isinstance(kv[0], torch.Tensor) and isinstance(kv[1], torch.Tensor):
                k_cache = kv[0]
                v_cache = kv[1]
        elif isinstance(kv, torch.Tensor) and kv.dim() >= 5:
            k_cache = kv[0]
            v_cache = kv[1]
        if k_cache is None or v_cache is None:
            continue
        # Expect [num_blocks, block_size, Hkv, D]
        if k_cache.dim() != 4 or v_cache.dim() != 4:
            continue
        layer_items.append((li, k_cache, v_cache))

    if not layer_items:
        return {}

    # Ensure stable order by layer index.
    layer_items.sort(key=lambda x: x[0])

    layer_indices_all = [li for li, _, _ in layer_items]

    # For each request, compute per-layer keep indices, then rewrite KV.
    for ridx, rid in enumerate(req_ids):
        L = int(seq_lens[ridx])
        if L <= 0:
            continue

        # Entire prompt fits in ``prompt_kv_len_budget``: no cross-layer budget / lift.
        if L <= C or (L - C) < min_rewrite_delta:
            _full_prefix_update(rid, L)
            continue

        per_layer_q = q_last_store.get(rid) or {}
        if len(per_layer_q) == 0:
            msg = (
                "[DynamicKV][offload] no Q capture: request_id=%s L=%d prompt_kv_len_budget=%d; "
                "using full-prefix metadata (compression skipped)."
            ) % (rid, L, C)
            logger.warning(msg)
            if os.environ.get("VLLM_ASCEND_DYNKV_STRICT", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                raise RuntimeError(msg)
            _full_prefix_update(rid, L)
            continue

        # Tail indices are always kept.
        tail_start = max(L - W, 0)
        tail_idx = torch.arange(tail_start, L, device=next(iter(per_layer_q.values())).device, dtype=torch.long)

        # Old-token candidate pool size (top-k before cross-layer reallocation).
        # Allow kv_len to exceed `prompt_kv_len_budget` by collecting a larger candidate
        # set; final per-layer budgets still come from `update_and_reset_budget`.
        old_len = max(L - W, 0)
        cand_old = min(int(cfg.radio_max * cfg.base), int(old_len))

        per_layer_scores_old: list[torch.Tensor] = []
        per_layer_indices_old: list[torch.Tensor] = []  # each [Hkv, cand_old]
        per_layer_k_full: list[torch.Tensor] = []
        per_layer_v_full: list[torch.Tensor] = []
        per_layer_slots_full: list[torch.Tensor] = []

        # Gather full KV once per layer (batched gather returns packed; slice by ridx).
        for li, k_cache, v_cache in layer_items:
            q_last = per_layer_q.get(li)
            if q_last is None or not isinstance(q_last, torch.Tensor):
                # Missing layer capture: abort this request (keep correctness).
                per_layer_scores_old = []
                break

            k_packed, v_packed, slots_packed, cu = gather_kv_from_paged_cache_batched(
                k_cache,
                v_cache,
                block_tables.to(device=k_cache.device),
                seq_lens,
            )
            start = 0 if ridx == 0 else int(cu[ridx - 1])
            end = int(cu[ridx])
            k_full = k_packed[start:end]
            v_full = v_packed[start:end]
            slots_full = slots_packed[start:end].to(torch.long)

            # Per-KV-head scores over old tokens + per-head topk indices.
            scores_old, idx_old = scores_and_indices_old_per_kv_head(
                query_last=q_last,
                key_full=k_full,
                cfg=cfg,
                budget_size=cand_old,
            )
            per_layer_scores_old.append(scores_old)
            per_layer_indices_old.append(idx_old)
            per_layer_k_full.append(k_full)
            per_layer_v_full.append(v_full)
            per_layer_slots_full.append(slots_full)

        if not per_layer_scores_old or len(per_layer_scores_old) != len(layer_items):
            logger.warning(
                "[DynamicKV][offload] missing layer scores/KV for request_id=%s "
                "(incomplete Q capture); skipping DynamicKV rewrite.",
                rid,
            )
            continue

        # Reallocate budgets across layers for this request.
        # Allocate per-layer budgets via per-head cross-layer density (upstream style).
        # Dummy shapes only used to cap max_old.
        dummy = []
        for s in per_layer_scores_old:
            old_len_i = int(s.shape[-1]) if isinstance(s, torch.Tensor) and s.dim() == 2 else int(s.numel())
            dummy.append(
                torch.empty((old_len_i + int(W), 1, 1), device=s.device, dtype=torch.float16)
            )
        per_layer_old_budget = update_and_reset_budget_per_kv_head(
            per_layer_scores_old=per_layer_scores_old,
            per_layer_k_budget=dummy,
            per_layer_v_budget=dummy,
            cfg=cfg,
            budget_size=cand_old,
        )
        if len(per_layer_old_budget) != num_layers:
            # Fallback: uniform.
            per_layer_old_budget = [cand_old] * num_layers

        per_layer_keep_indices: list[list[int]] = []
        per_layer_kv_lens: list[int] = []
        cap_trunc_any = False  # retained for logging compatibility

        # Rewrite each layer.
        # Track kv_len by layer index for later zero-padding to PD transfer upper bound.
        kv_len_by_layer: dict[int, int] = {}
        for local_i, (li, _k_cache, _v_cache) in enumerate(layer_items):
            old_budget = int(per_layer_old_budget[li])
            idx_old_h = per_layer_indices_old[local_i]  # [Hkv, cand_old]
            k_full = per_layer_k_full[local_i]
            v_full = per_layer_v_full[local_i]
            slots_full = per_layer_slots_full[local_i]
            Hkv = int(k_full.shape[1])
            W_eff = int(tail_idx.numel())
            old_budget_eff = max(0, min(int(old_budget), int(idx_old_h.shape[1])))
            kv_len = int(old_budget_eff + W_eff)
            if kv_len <= 0 or Hkv <= 0:
                per_layer_keep_indices.append([])
                per_layer_kv_lens.append(0)
                continue
            # Flatten paged cache to [num_blocks*block_size, H, D]
            # NOTE: Some kernels may read within the last block beyond `context_lens`
            # due to block-granularity; we zero-pad the remainder of the last block
            # to prevent stale KV from leaking into attention.
            block_size = int(_k_cache.shape[1])
            k_flat = _k_cache.reshape(-1, _k_cache.shape[-2], _k_cache.shape[-1])
            v_flat = _v_cache.reshape(-1, _v_cache.shape[-2], _v_cache.shape[-1])
            # Build packed KV tensor per KV head:
            # - old part: per-head top-k indices (order within old part does not matter)
            # - tail part: chronological tail window (shared across heads)
            # Final packed tensor has shape [kv_len, Hkv, D].
            idx_old_keep = idx_old_h[:, :old_budget_eff].to(torch.long) if old_budget_eff > 0 else idx_old_h[:, :0].to(torch.long)
            tail_keep = tail_idx.to(torch.long)
            k_heads: list[torch.Tensor] = []
            v_heads: list[torch.Tensor] = []
            for h in range(Hkv):
                if old_budget_eff > 0:
                    k_old = k_full.index_select(0, idx_old_keep[h])[:, h, :]
                    v_old = v_full.index_select(0, idx_old_keep[h])[:, h, :]
                else:
                    k_old = k_full[:0, h, :]
                    v_old = v_full[:0, h, :]
                if W_eff > 0:
                    k_tail = k_full.index_select(0, tail_keep)[:, h, :]
                    v_tail = v_full.index_select(0, tail_keep)[:, h, :]
                else:
                    k_tail = k_full[:0, h, :]
                    v_tail = v_full[:0, h, :]
                k_heads.append(torch.cat([k_old, k_tail], dim=0))
                v_heads.append(torch.cat([v_old, v_tail], dim=0))
            k_packed = torch.stack(k_heads, dim=0).transpose(0, 1).contiguous()
            v_packed = torch.stack(v_heads, dim=0).transpose(0, 1).contiguous()
            per_layer_kv_lens.append(kv_len)
            kv_len_by_layer[int(li)] = int(kv_len)
            slots_keep = slots_full[:kv_len]
            k_flat.index_copy_(0, slots_keep.to(k_flat.device), k_packed)
            v_flat.index_copy_(0, slots_keep.to(v_flat.device), v_packed)
            # For debugging/PD export: keep_indices in original prompt space is
            # no longer a single chronological list under per-head selection.
            # Keep it empty to avoid misinterpretation; decode uses kv_len only.
            per_layer_keep_indices.append([])

        # Zero-pad within PD transfer upper bound.
        #
        # PD transfer (and some kernels) operate in blocks. Even if decode uses per-layer
        # `context_lens`, some implementations may read within the transferred block
        # range beyond the logical kv_len. Clear all padding slots in the transferred
        # prefix up to `ceil(max(per_layer_kv_lens)/block_size)*block_size` so stale KV
        # cannot leak into attention.
        if kv_len_by_layer:
            max_kv_len = max(int(v) for v in kv_len_by_layer.values() if int(v) > 0) if kv_len_by_layer else 0
            if max_kv_len > 0:
                # Use any layer's cache for block_size; all layers share KV block_size.
                try:
                    any_cache = layer_items[0][1]
                    block_size = int(any_cache.shape[1])
                except Exception:
                    block_size = 0
                if block_size > 0:
                    pad_to_global = int(((int(max_kv_len) + block_size - 1) // block_size) * block_size)
                    pad_to_global = min(int(pad_to_global), int(L))
                    if pad_to_global > 0:
                        for local_i, (li, _k_cache, _v_cache) in enumerate(layer_items):
                            kv_len_i = int(kv_len_by_layer.get(int(li), 0))
                            if kv_len_i <= 0 or kv_len_i >= pad_to_global:
                                continue
                            slots_full = per_layer_slots_full[local_i]
                            slots_pad = slots_full[kv_len_i:pad_to_global]
                            if slots_pad.numel() <= 0:
                                continue
                            k_flat = _k_cache.reshape(-1, _k_cache.shape[-2], _k_cache.shape[-1])
                            v_flat = _v_cache.reshape(-1, _v_cache.shape[-2], _v_cache.shape[-1])
                            Hkv = int(k_flat.shape[1])
                            D = int(k_flat.shape[2])
                            zeros_k = k_flat.new_zeros((int(slots_pad.numel()), Hkv, D))
                            zeros_v = v_flat.new_zeros((int(slots_pad.numel()), Hkv, D))
                            k_flat.index_copy_(0, slots_pad.to(k_flat.device), zeros_k)
                            v_flat.index_copy_(0, slots_pad.to(v_flat.device), zeros_v)

        # Expand to full num_layers: for layers not present in kv_caches (rare),
        # fill with -1/[] so decode can ignore.
        full_lens: list[int] = [-1] * num_layers
        full_idx: list[list[int]] = [[] for _ in range(num_layers)]
        for local_i, (li, _, _) in enumerate(layer_items):
            full_lens[li] = per_layer_kv_lens[local_i]
            full_idx[li] = per_layer_keep_indices[local_i]

        pos_lens = [int(full_lens[li]) for li in layer_indices_all if int(full_lens[li]) > 0]
        if pos_lens:
            _do_log = True
            try:
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_rank,
                )

                _do_log = int(get_tensor_model_parallel_rank()) == 0
            except Exception:
                pass
            if _do_log:
                logger.info(
                    "[DynamicKV][offload] rewrite request_id=%s L=%d C=%d W=%d "
                    "kv_len min=%d max=%d mean=%.2f cap_truncated=%s",
                    rid,
                    L,
                    C,
                    W,
                    min(pos_lens),
                    max(pos_lens),
                    float(sum(pos_lens)) / float(len(pos_lens)),
                    cap_trunc_any,
                )
        updates[rid] = {
            "dynamic_kv": {
                "per_layer_kv_lens": full_lens,
                "per_layer_keep_indices": full_idx,
            }
        }
        _validate_payload_or_fallback(rid=rid, L=L, full_lens=full_lens, full_idx=full_idx)

    return updates

