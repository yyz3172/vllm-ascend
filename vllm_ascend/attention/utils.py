import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.utils import (AscendDeviceType, get_ascend_config,
                               get_ascend_device_type)

def using_paged_attention(runtime_shape: int, vllm_config: VllmConfig) -> bool:
    from vllm.config.compilation import CUDAGraphMode

    if vllm_config.speculative_config is not None:
        return False
    if get_ascend_device_type() == AscendDeviceType.A5:
        return False
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False

    return runtime_shape in get_ascend_config().pa_shape_list


def _merge_dynamic_kv_lens_list(
    dyn_lens: list[int] | None,
    fallback_lens: list[int] | None,
) -> list[int] | None:
    """Merge per-request DynamicKV lens with fallback (e.g. padded slots)."""
    if dyn_lens is None:
        return fallback_lens
    if fallback_lens is None or len(fallback_lens) != len(dyn_lens):
        return list(dyn_lens)
    if not any(v < 0 for v in dyn_lens):
        return list(dyn_lens)
    return [d if d >= 0 else f for d, f in zip(dyn_lens, fallback_lens)]


def fia_dynamic_kv_seq_lens_list(attn_metadata: Any) -> list[int] | None:
    """FIA ``actual_seq_lengths_kv`` for PD DynamicKV decode graph updates."""
    return _merge_dynamic_kv_lens_list(
        getattr(attn_metadata, "dynamic_kv_seq_lens_list", None),
        getattr(attn_metadata, "seq_lens_list", None),
    )


def pa_dynamic_kv_context_lens(attn_metadata: Any) -> torch.Tensor:
    """Paged-attention ``context_lens`` for PD DynamicKV decode.

    Prefer ``dynamic_kv_seq_lens_tensor`` when set. For entries with negative
    lens (e.g. padded batch slots), fall back to ``seq_lens``.
    """
    dl = getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)
    if dl is None:
        return attn_metadata.seq_lens
    t = getattr(attn_metadata, "dynamic_kv_seq_lens_tensor", None)
    sl = attn_metadata.seq_lens
    if isinstance(t, torch.Tensor) and isinstance(sl, torch.Tensor):
        n_dl = len(dl)
        if (t.device == sl.device and t.dtype == sl.dtype
                and int(t.numel()) == int(sl.numel()) and int(t.numel()) >= n_dl):
            # FULL-graph pinned buffer (prepare-filled); avoid torch.tensor(dl).
            if not (t < 0).any():
                return t
            return torch.where(t >= 0, t, sl)
        if (int(t.numel()) == n_dl and t.device == sl.device
                and t.dtype == sl.dtype):
            if (t < 0).any():
                return torch.where(t >= 0, t, sl)
            return t
    ctx = torch.tensor(dl, device=sl.device, dtype=sl.dtype)
    if (ctx < 0).any():
        return torch.where(ctx >= 0, ctx, sl)
    return ctx


def dynkv_profile_pa_enabled() -> bool:
    """``VLLM_DYNKV_PROFILE_PA=1`` (+ FORWARD): graph replay NPU sync + eager PA layer timing."""
    return (
        os.environ.get("VLLM_DYNKV_PROFILE_FORWARD", "0") == "1"
        and os.environ.get("VLLM_DYNKV_PROFILE_PA", "0") == "1"
    )


def dynkv_pa_kv_tokens_avg_from_attn_metadata(
    attn_metadata: Any,
) -> float | None:
    """Mean of per-layer ``max(context_lens)`` for forward_profile (PA decode)."""
    if attn_metadata is None:
        return None
    metas = (attn_metadata.values()
             if isinstance(attn_metadata, dict) else [attn_metadata])
    maxes: list[float] = []
    for meta in metas:
        if meta is None:
            continue
        try:
            ctx = pa_dynamic_kv_context_lens(meta)
            if isinstance(ctx, torch.Tensor) and ctx.numel() > 0:
                maxes.append(float(ctx.max().item()))
        except Exception:
            continue
    if not maxes:
        return None
    return sum(maxes) / len(maxes)


def dynkv_fill_all_graph_context_lens_bufs(
    *,
    layer_names: list[str],
    context_lens_bufs: dict[str, torch.Tensor],
    stacked_dyn_lens_t: torch.Tensor,
    seq_lens: torch.Tensor,
    all_tmp_lens: list[list[int]],
    layer_idx_map: dict[str, int],
) -> None:
    """Batch-fill per-layer graph ``context_lens`` buffers (prepare P1)."""
    if not context_lens_bufs or stacked_dyn_lens_t is None:
        return
    if not isinstance(seq_lens, torch.Tensor):
        return
    n_layers = min(
        len(layer_names),
        int(stacked_dyn_lens_t.shape[0]),
        len(all_tmp_lens),
    )
    for li in range(n_layers):
        layer_name = layer_names[li]
        buf = context_lens_bufs.get(layer_name)
        if buf is None:
            continue
        tmp_lens_layer = all_tmp_lens[li]
        if (layer_idx_map.get(layer_name, -1) < 0
                or not tmp_lens_layer
                or all(v < 0 for v in tmp_lens_layer)):
            continue
        row = stacked_dyn_lens_t[li]
        n_buf = int(buf.numel())
        n_row = int(row.numel())
        if row.device != buf.device or row.dtype != buf.dtype:
            continue
        if n_row < n_buf and int(seq_lens.numel()) == n_buf:
            buf.copy_(seq_lens)
            part = buf[:n_row]
            if (row < 0).any():
                part.copy_(torch.where(row >= 0, row, seq_lens[:n_row]))
            else:
                part.copy_(row)
        elif n_row == n_buf:
            if (row < 0).any():
                buf.copy_(torch.where(row >= 0, row, seq_lens))
            else:
                buf.copy_(row)


def dynkv_fill_graph_context_lens_buf(
    attn_metadata: Any,
    context_lens_buf: torch.Tensor,
    *,
    stacked_row: torch.Tensor | None = None,
) -> None:
    """Write runtime lens into the tensor pinned at FULL-graph PA capture.

    Called from ``_prepare_inputs`` so ``model_acl`` can reuse ``context_lens_buf``
    without a per-layer ``copy_`` in ``pa_dynamic_kv_context_lens_for_graph_update``.
    """
    sl = attn_metadata.seq_lens
    if stacked_row is not None and isinstance(sl, torch.Tensor):
        row = stacked_row
        n_buf = int(context_lens_buf.numel())
        n_row = int(row.numel())
        if (row.device == context_lens_buf.device
                and row.dtype == context_lens_buf.dtype):
            if n_row < n_buf and int(sl.numel()) == n_buf:
                context_lens_buf.copy_(sl)
                part = context_lens_buf[:n_row]
                if (row < 0).any():
                    part.copy_(torch.where(row >= 0, row, sl[:n_row]))
                else:
                    part.copy_(row)
                return
            if n_row == n_buf:
                if (row < 0).any():
                    context_lens_buf.copy_(torch.where(row >= 0, row, sl))
                else:
                    context_lens_buf.copy_(row)
                return
    ctx = pa_dynamic_kv_context_lens(attn_metadata)
    if (isinstance(ctx, torch.Tensor)
            and context_lens_buf.data_ptr() != ctx.data_ptr()
            and int(ctx.numel()) == int(context_lens_buf.numel())
            and ctx.device == context_lens_buf.device
            and ctx.dtype == context_lens_buf.dtype):
        context_lens_buf.copy_(ctx)


def pa_dynamic_kv_context_lens_for_graph_update(
    attn_metadata: Any,
    context_lens_buf: torch.Tensor | None,
) -> torch.Tensor:
    """PA ``context_lens`` for ACL graph replay / ``graph_task_update``.

    Reuse the tensor captured into ``attn_params`` and copy runtime values
    in-place when shapes match. Passing a fresh tensor each step can leave the
    replayed op reading stale lengths on some Ascend builds.
    """
    pinned = getattr(attn_metadata, "dynamic_kv_seq_lens_tensor", None)
    if (
        context_lens_buf is not None
        and isinstance(pinned, torch.Tensor)
        and isinstance(context_lens_buf, torch.Tensor)
        and pinned.data_ptr() == context_lens_buf.data_ptr()
    ):
        return context_lens_buf
    ctx = pa_dynamic_kv_context_lens(attn_metadata)
    if (
        context_lens_buf is not None
        and isinstance(ctx, torch.Tensor)
        and isinstance(context_lens_buf, torch.Tensor)
        and context_lens_buf.device == ctx.device
        and context_lens_buf.dtype == ctx.dtype
        and int(context_lens_buf.numel()) == int(ctx.numel())
    ):
        if context_lens_buf.data_ptr() != ctx.data_ptr():
            context_lens_buf.copy_(ctx)
        return context_lens_buf
    return ctx


@lru_cache(maxsize=1)
def enable_cp():
    prefill_config = get_current_vllm_config().parallel_config
    return prefill_config.prefill_context_parallel_size > 1 \
                or prefill_config.decode_context_parallel_size > 1


@dataclass
# class AscendCommonLongSequenceMetadata:
class AscendPrefillContextParallelMetadata:
    pcp_allgather_restore_idx: torch.Tensor = None

    num_actual_tokens_pcp_padded: int = 0

    num_computed_tokens_of_pcp_dcp: Optional[list[list[list[int]]]] = None

    q_head_idx_tensor: torch.Tensor = None

    q_tail_idx_tensor: torch.Tensor = None

    kv_with_q_head_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_head_mask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_mask_idx_tensor: torch.Tensor = None

    attn_mask_seqlens: torch.Tensor = None

    head_attn_nomask_seqlens: torch.Tensor = None

    tail_attn_nomask_seqlens: torch.Tensor = None

    q_full_idx: torch.Tensor = None

    # original query_lens before pcp split
    query_lens_pcp_full_cpu: torch.Tensor = None

    # original max_query_len before pcp split
    max_query_len_pcp_full: int = 0


@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """
    seq_lens_cpu: torch.Tensor = None
    num_computed_tokens_cpu: torch.Tensor = None

    decode_token_per_req: int = 1
    """decode token number per request"""

    actual_seq_lengths_q: list[int] = field(default_factory=list)

    positions: torch.Tensor = None

    attn_state: Any = None

    graph_pad_size: int = -1

    # num_input_tokens refers to total number of tokens including
    # padding tokens. It is used to handle some padding operations.
    num_input_tokens: int = 0

    prefill_context_parallel_metadata: Optional[
        AscendPrefillContextParallelMetadata] = None

    # DynamicKV: whether this ChunkedPrefill step is the last chunk.
    # This is computed in the attn metadata builder and used by attention backend
    # to avoid compressing KV cache on every chunk.
    dynamic_kv_is_last_chunk: bool = False

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int,
                 num_actual_reqs: int) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[:num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[:num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.
            num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            # NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
            # there will be error about shape mismatch during reshape and cache.
            # This is really strange since vLLM slices them as well
            block_table_tensor=self.block_table_tensor,
            slot_mapping=self.slot_mapping,
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.
            prefill_context_parallel_metadata,
            dynamic_kv_is_last_chunk=self.dynamic_kv_is_last_chunk,
            max_seq_len=self.max_seq_len)


def filter_chunked_req_indices(
    seq_len: torch.Tensor,
    mask_for_non_zero_chunk: Optional[List[bool]],
) -> torch.Tensor:
    """
    filter the reqs which are doing real chunk_prefill.

    Args:
        seq_len: contains multi-req length: [req0_len, req1_len, ...]
        mask_for_non_zero_chunk: [True, False, True, False, ...]
    Returns:
        filtered_indices: the real chunked req's indices
    """
    assert mask_for_non_zero_chunk is not None and len(seq_len) == len(
        mask_for_non_zero_chunk)
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)
    filtered_indices = torch.cat([
        torch.arange(offsets[i], offsets[i] + seq_len[i])
        for i in range(len(mask_for_non_zero_chunk))
        if mask_for_non_zero_chunk[i]
    ])
    return filtered_indices


def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.
    While pcp > 1, query_lens is split across pcp ranks, so we pass in the
    original query_lens and max_query_len to distinguish prefills and decodes.

    Args:
        common_attn_metadata: AscendCommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
    query_lens_pcp_full = long_seq_metadata.query_lens_pcp_full_cpu \
        if long_seq_metadata else None
    max_query_len_pcp_full = long_seq_metadata.max_query_len_pcp_full \
        if long_seq_metadata else 0
    max_query_len = common_attn_metadata.max_query_len \
        if max_query_len_pcp_full == 0 else max_query_len_pcp_full
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = (query_start_loc[1:] - query_start_loc[:-1]) \
        if query_lens_pcp_full is None else query_lens_pcp_full
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: List[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def trans_rope_weight(weight, rope_dim):
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    reordered_rope_part = torch.cat(
        (rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, reordered_rope_part), dim=-2).contiguous()


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, (0, r_pad, 0, c_pad))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1],
             block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat,
        (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat
