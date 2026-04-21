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
                                              update_and_reset_budget)

_DYNKV_REWRITE_STRATEGY_ENV = "VLLM_ASCEND_DYNKV_REWRITE_STRATEGY"


def _get_dynkv_rewrite_strategy() -> str:
    """
    DynamicKV KV-rewrite strategy (offload impl).

    - "pack" (default): pack kept tokens into prefix slots 0..kv_len-1.
    - "zero_inplace": keep layout unchanged; dropped tokens' KV set to 0;
      per-layer kv_lens exported as full prompt length (no compute-side gain).
    - "pack_rebuild_full": pack for transfer compatibility, but instruct Decode
      to rebuild a full-length KV layout using keep_indices + zero-fill.
    """
    v = os.environ.get(_DYNKV_REWRITE_STRATEGY_ENV, "").strip().lower()
    if not v:
        return "pack"
    if v in ("pack", "zero_inplace", "pack_rebuild_full"):
        return v
    logger.warning(
        "[DynamicKV][offload] unknown %s=%r; falling back to 'pack'",
        _DYNKV_REWRITE_STRATEGY_ENV,
        v,
    )
    return "pack"



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
        strategy = _get_dynkv_rewrite_strategy()

        # Entire prompt fits in ``prompt_kv_len_budget``: no cross-layer budget / lift.
        if L <= C:
            updates[rid] = {
                "dynamic_kv": _dynamic_kv_payload_full_prefix(
                    num_layers=num_layers,
                    layer_indices=layer_indices_all,
                    L=L,
                )
            }
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
            updates[rid] = {
                "dynamic_kv": _dynamic_kv_payload_full_prefix(
                    num_layers=num_layers,
                    layer_indices=layer_indices_all,
                    L=L,
                )
            }
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
        per_layer_indices_old: list[torch.Tensor] = []
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

            # scores over old tokens + topk indices (sorted).
            scores_old, idx_old = scores_and_indices_old(
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
        dummy = [
            torch.empty((int(s.numel()) + int(W), 1, 1), device=s.device, dtype=torch.float16)
            for s in per_layer_scores_old
        ]
        per_layer_old_budget = update_and_reset_budget(
            per_layer_scores_old=per_layer_scores_old,
            per_layer_k_budget=dummy,
            per_layer_v_budget=dummy,
            cfg=cfg,
            budget_size=cand_old,
            head_factor=1,
        )
        if len(per_layer_old_budget) != num_layers:
            # Fallback: uniform.
            per_layer_old_budget = [cand_old] * num_layers

        per_layer_keep_indices: list[list[int]] = []
        per_layer_kv_lens: list[int] = []
        per_layer_keep_lens: list[int] = []
        cap_trunc_any = False

        # Rewrite each layer.
        for local_i, (li, _k_cache, _v_cache) in enumerate(layer_items):
            old_budget = int(per_layer_old_budget[li])
            idx_old = per_layer_indices_old[local_i]
        # NOTE: `dynamic_kv.prompt_kv_len_budget` is NOT treated as a hard kv_len cap.
            # We allow kv_len to exceed it (up to the prompt length L). The only
            # hard constraint is never selecting indices outside [0, L).
            keep_sorted, cap_trunc = cap_keep_indices_chronological(
                idx_old=idx_old,
                old_budget=old_budget,
                tail_idx=tail_idx,
                cap=L,
                device=idx_old.device,
            )
            if cap_trunc:
                cap_trunc_any = True
            if keep_sorted.numel() == 0:
                per_layer_keep_indices.append([])
                per_layer_kv_lens.append(0)
                continue
            # Decode paged attention reads cache slots ``0..kv_len-1`` as *causal*
            # time order: ``keep_sorted`` is ascending original token indices.
            keep_list = [int(x) for x in keep_sorted.tolist()]
            per_layer_keep_indices.append(keep_list)
            per_layer_keep_lens.append(int(keep_sorted.numel()))

            k_full = per_layer_k_full[local_i]
            v_full = per_layer_v_full[local_i]
            slots_full = per_layer_slots_full[local_i]
            kv_len = int(keep_sorted.shape[0])
            # Flatten paged cache to [num_blocks*block_size, H, D]
            k_flat = _k_cache.reshape(-1, _k_cache.shape[-2], _k_cache.shape[-1])
            v_flat = _v_cache.reshape(-1, _v_cache.shape[-2], _v_cache.shape[-1])

            if strategy == "zero_inplace":
                # Keep the original KV layout (positions 0..L-1). Zero out dropped tokens.
                keep_mask = torch.zeros((L,), device=keep_sorted.device, dtype=torch.bool)
                keep_mask.index_fill_(0, keep_sorted, True)
                drop_pos = torch.nonzero(~keep_mask, as_tuple=False).flatten()
                if drop_pos.numel() > 0:
                    slots_drop = slots_full.index_select(0, drop_pos).to(torch.long)
                    zeros_k = torch.zeros(
                        (int(slots_drop.shape[0]),) + tuple(k_flat.shape[1:]),
                        device=k_flat.device,
                        dtype=k_flat.dtype,
                    )
                    zeros_v = torch.zeros(
                        (int(slots_drop.shape[0]),) + tuple(v_flat.shape[1:]),
                        device=v_flat.device,
                        dtype=v_flat.dtype,
                    )
                    k_flat.index_copy_(0, slots_drop.to(k_flat.device), zeros_k)
                    v_flat.index_copy_(0, slots_drop.to(v_flat.device), zeros_v)
                per_layer_kv_lens.append(int(L))
            else:
                # "pack" / "pack_rebuild_full": pack kept KV into prefix slots for transfer.
                k_chrono = k_full.index_select(0, keep_sorted)
                v_chrono = v_full.index_select(0, keep_sorted)
                per_layer_kv_lens.append(kv_len)
                slots_keep = slots_full[:kv_len]
                k_flat.index_copy_(0, slots_keep.to(k_flat.device), k_chrono)
                v_flat.index_copy_(0, slots_keep.to(v_flat.device), v_chrono)

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
                # More intuitive stats: "effective" (kept) prompt tokens per layer.
                if per_layer_keep_lens:
                    try:
                        keep_pos = [int(x) for x in per_layer_keep_lens if int(x) > 0]
                        if keep_pos:
                            uniq = len(set(keep_pos))
                            mn = min(keep_pos)
                            mx = max(keep_pos)
                            sm = sum(keep_pos)
                            mean = float(sm) / float(len(keep_pos))
                            logger.info(
                                "[DynamicKV][offload] per_layer_keep_lens stats: request_id=%s strategy=%s "
                                "unique=%d min=%d max=%d sum=%d mean=%.2f (layers_len>0=%d; L=%d; C=%d; W=%d)",
                                rid,
                                strategy,
                                uniq,
                                mn,
                                mx,
                                sm,
                                mean,
                                len(keep_pos),
                                L,
                                C,
                                W,
                            )
                            # Also report implied dropped count for Step1 readability.
                            drop_pos = [int(L - x) for x in keep_pos]
                            logger.info(
                                "[DynamicKV][offload] per_layer_drop_lens(implied) stats: request_id=%s strategy=%s "
                                "unique=%d min=%d max=%d sum=%d mean=%.2f (layers_len>0=%d; L=%d)",
                                rid,
                                strategy,
                                len(set(drop_pos)),
                                min(drop_pos),
                                max(drop_pos),
                                sum(drop_pos),
                                float(sum(drop_pos)) / float(len(drop_pos)),
                                len(drop_pos),
                                L,
                            )
                    except Exception:
                        pass

        dyn_payload: dict[str, Any] = {"per_layer_kv_lens": full_lens}
        if strategy == "pack":
            dyn_payload["per_layer_keep_indices"] = full_idx
        elif strategy == "pack_rebuild_full":
            dyn_payload["per_layer_keep_indices"] = full_idx
            dyn_payload["rebuild_full"] = True
            dyn_payload["rebuild_full_prompt_len"] = int(L)
        elif strategy == "zero_inplace":
            # Intentionally omit keep_indices so Decode won't derive a shorter kv_len.
            dyn_payload["zero_inplace"] = True
            dyn_payload["original_prompt_len"] = int(L)

        updates[rid] = {"dynamic_kv": dyn_payload}

    return updates

