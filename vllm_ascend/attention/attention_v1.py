#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Type

import time
import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.registry import (AttentionBackendEnum,
                                              register_backend)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.v1.worker.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import (
    AscendMetadataForDecode, AscendMetadataForPrefill)
from vllm_ascend.attention.dynamic_kv import (
    DynamicKVConfig,
    build_attention_mask_from_important_mask,
    cap_keep_indices_chronological,
    clear_validation_masks,
    gather_kv_from_paged_cache,
    gather_kv_from_paged_cache_batched,
    get_validation_mask,
    save_validation_mask,
    scores_and_indices_old,
    scores_and_indices_old_perhead_aggregated,
    update_and_reset_budget,
    update_and_reset_budget_per_kv_head,
)
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         enable_cp, split_decodes_and_prefills,
                                         using_paged_attention)
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params, get_graph_params,
    update_draft_graph_params_workspaces, update_graph_params_workspaces)
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               weak_ref_tensors)

# default max value of sliding window size
SWA_INT_MAX = 2147483647

# PD DynamicKV：prefill 内按层累计，最后一层做跨层预算并写回各层 KV。
# 算法与开源 DynamicKV 参考实现（code/DynamicKV）对齐。
# Structure:
#   - prefill_batch: 当前 batch 的 per-layer 中间态
#   - last_results: per-request 最终结果（供 PD kv_transfer_params）
_DYNKV_STATE: dict[str, object] = {"last_results": {}}
_ASCEND_DYNKV_LAYER_REGISTRY: dict[str, tuple["AscendAttentionBackendImpl", Tuple[torch.Tensor]]] = {}
# Keep only a bounded number of results to avoid unbounded growth.
_DYNKV_LAST_RESULTS_MAX = 2048
_DYNKV_LAST_RESULTS_TTL_S = 300.0


def _dynkv_prune_last_results(now_s: float | None = None) -> None:
    now_s = time.time() if now_s is None else float(now_s)
    last_results = _DYNKV_STATE.get("last_results")
    if not isinstance(last_results, dict) or not last_results:
        return

    # TTL prune.
    for rid, item in list(last_results.items()):
        try:
            ts = float(item.get("ts", 0.0)) if isinstance(item, dict) else 0.0
        except Exception:
            ts = 0.0
        if ts and (now_s - ts) > _DYNKV_LAST_RESULTS_TTL_S:
            last_results.pop(rid, None)

    # Size prune (oldest first).
    if len(last_results) <= _DYNKV_LAST_RESULTS_MAX:
        return
    items: list[tuple[float, str]] = []
    for rid, item in last_results.items():
        try:
            ts = float(item.get("ts", 0.0)) if isinstance(item, dict) else 0.0
        except Exception:
            ts = 0.0
        items.append((ts, rid))
    items.sort(key=lambda x: x[0])
    overflow = len(last_results) - _DYNKV_LAST_RESULTS_MAX
    for _, rid in items[:overflow]:
        last_results.pop(rid, None)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import \
                AscendAttentionCPImpl
            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import \
                AscendAttentionCPMetadataBuilder
            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    # **************************** Basic Properties ************************** #
    # The layer name this metadata instance is built for.
    # vLLM constructs per-layer metadata keyed by layer_name.
    layer_name: str = ""
    # Request ids in the batch (aligned with seq_lens_list order).
    # Used to export per-request DynamicKV results for PD transfer.
    req_ids: List[str] = None  # type: ignore
    # PD DynamicKV：本层各请求的 kv 长度（与 req_ids 对齐）。
    dynamic_kv_seq_lens_list: Optional[List[int]] = None
    # PD DynamicKV：本层各请求保留的 token 下标（与 req_ids 对齐）。
    # Each element is a python list of token indices in original prompt space.
    dynamic_kv_keep_indices_list: Optional[List[List[int]]] = None
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None  # type: ignore
    actual_seq_lengths_q: List[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # Optional: DynamicKV 写回后的各请求 KV 长度（PrefillNoCache 或 ChunkedPrefill 最后一块）。
    # - `seq_lens_list_kv`: per-request kv lengths after compression
    # - `actual_seq_lengths_kv`: cumulative kv lengths (same format as actual_seq_lengths_q)
    seq_lens_list_kv: Optional[List[int]] = None
    actual_seq_lengths_kv: Optional[List[int]] = None
    # DynamicKV: whether this ChunkedPrefill step is the last chunk.
    dynamic_kv_is_last_chunk: bool = False
    # pcp
    prefill: Optional[AscendMetadataForPrefill] = None
    # dcp
    decode_meta: Optional[AscendMetadataForDecode] = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # sliding window attention mask
    swa_mask: Optional[torch.Tensor] = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    # AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0])

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        AscendAttentionMetadataBuilder.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)

        block_table = common_attn_metadata.block_table_tensor
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        attn_state = common_attn_metadata.attn_state

        # Get attn_mask and swa_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(
            self.model_config)

        swa_mask = None
        is_swa = hasattr(self.model_config.hf_text_config, 'sliding_window')
        if self.model_config is not None and is_swa:
            swa_mask = self.attn_mask_builder.get_swa_mask(
                self.model_config.dtype,
                self.model_config.hf_text_config.sliding_window)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(
            self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            swa_mask=swa_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            dynamic_kv_is_last_chunk=getattr(
                common_attn_metadata, "dynamic_kv_is_last_chunk", False
            ),
            # Propagate request ids (required for PD DynamicKV export).
            req_ids=list(getattr(common_attn_metadata, "req_ids", []) or []),
            # Optional: propagate per-layer dynamic kv lens/indices if present.
            dynamic_kv_seq_lens_list=getattr(
                common_attn_metadata, "dynamic_kv_seq_lens_list", None
            ),
            dynamic_kv_keep_indices_list=getattr(
                common_attn_metadata, "dynamic_kv_keep_indices_list", None
            ),
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):

        if attn_state in (AscendAttentionState.DecodeOnly,
                          AscendAttentionState.ChunkedPrefill):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer

        # DynamicKV (experimental): PrefillNoCache or ChunkedPrefill last chunk.
        # Note: ascend config is a process-global singleton; initialize it from
        # the current vllm_config if needed.
        init_ascend_config(self.vllm_config)
        ascend_cfg = get_ascend_config()
        # DynamicKV has multiple implementations. Only the legacy "attn" mode
        # hooks Python attention forward; "offload" keeps the attention
        # execution path unchanged and rewrites KV after prefill elsewhere.
        self._dynamickv_impl = str(getattr(ascend_cfg, "dynamic_kv_impl", "offload"))
        self._dynamickv_enabled = bool(getattr(ascend_cfg, "dynamic_kv_enabled", False)) and (self._dynamickv_impl == "attn")
        self._dynamickv_window_size = int(getattr(ascend_cfg, "dynamic_kv_window_size", 16))
        self._dynamickv_prompt_kv_len_budget = int(getattr(ascend_cfg, "dynamic_kv_prompt_kv_len_budget", 512))
        self._dynamickv_pooling = getattr(ascend_cfg, "dynamic_kv_pooling", "none")
        self._dynamickv_kernel_size = int(getattr(ascend_cfg, "dynamic_kv_kernel_size", 1))
        self._dynamickv_softmax_chunk_size = int(getattr(ascend_cfg, "dynamic_kv_softmax_chunk_size", 1024) or 1024)
        self._dynamickv_radio_max = float(getattr(ascend_cfg, "dynamic_kv_radio_max", 10.0))
        self._dynamickv_radio_min = float(getattr(ascend_cfg, "dynamic_kv_radio_min", 0.1))
        self._dynamickv_validation_mode = str(getattr(ascend_cfg, "dynamic_kv_validation_mode", "none"))
        self._dynamickv_model_types = set(getattr(ascend_cfg, "dynamic_kv_model_types", ["mistral"]) or [])
        try:
            model_type = getattr(self.vllm_config.model_config.hf_config, "model_type", "")
        except Exception:
            model_type = ""
        self._dynamickv_model_type_ok = (model_type in self._dynamickv_model_types)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(self, query: torch.Tensor, key: torch.Tensor,
                       value: torch.Tensor, attn_metadata: AscendMetadata,
                       output: torch.Tensor) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv \
            = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        forward_context = get_forward_context()
        if forward_context.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
            )
            if forward_context.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (weak_ref_tensors(query), weak_ref_tensors(key),
             weak_ref_tensors(value), weak_ref_tensors(block_table),
             weak_ref_tensors(attn_metadata.attn_mask), block_size,
             actual_seq_lengths_kv, actual_seq_lengths_q, self.num_kv_heads,
             self.num_heads, self.scale, weak_ref_tensors(output),
             weak_ref_tensors(softmax_lse)))

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ):
        graph_params = get_graph_params()
        forward_context: ForwardContext = get_forward_context()
        num_tokens = query.shape[0]
        if forward_context.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=(
                        torch.tensor(
                            getattr(attn_metadata, "dynamic_kv_seq_lens_list"),
                            device=attn_metadata.seq_lens.device,
                            dtype=attn_metadata.seq_lens.dtype,
                        )
                        if getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)
                        is not None
                        else attn_metadata.seq_lens
                    ),
                    out=output)
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append((
                weak_ref_tensors(query),
                weak_ref_tensors(self.key_cache),
                weak_ref_tensors(self.value_cache),
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                attn_metadata.block_tables,
                (
                    torch.tensor(
                        getattr(attn_metadata, "dynamic_kv_seq_lens_list"),
                        device=attn_metadata.seq_lens.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    if getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)
                    is not None
                    else attn_metadata.seq_lens
                ),
                weak_ref_tensors(output),
            ))

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=(
                    torch.tensor(
                        getattr(attn_metadata, "dynamic_kv_seq_lens_list"),
                        device=attn_metadata.seq_lens.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    if getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)
                    is not None
                    else attn_metadata.seq_lens
                ),
                out=output,
                workspace=workspace)
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor,
                        attn_metadata: AscendMetadata):
        dyn_lens_list = getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = (
                attn_metadata.actual_seq_lengths_kv
                if attn_metadata.actual_seq_lengths_kv is not None
                else attn_metadata.actual_seq_lengths_q
            )
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens,
                                                     dim=0).tolist()
        elif attn_metadata.attn_state == \
                AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            actual_seq_lengths_kv = dyn_lens_list or attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = dyn_lens_list or attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = dyn_lens_list or attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor,
                                   attn_metadata: AscendMetadata,
                                   output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens)

        output = output.view(batch_size, self.num_heads, self.head_size)
        return output

    def forward_fused_infer_attention(self, query: torch.Tensor,
                                      key: torch.Tensor, value: torch.Tensor,
                                      attn_metadata: AscendMetadata,
                                      output: torch.Tensor):
        forward_context: ForwardContext = get_forward_context()
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if getattr(forward_context, "capturing", False):
            attn_output, num_tokens = self.full_graph_fia(
                query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        # Validation mode: decode with mask when using FIA (eager mode).
        if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and self._dynamickv_validation_mode == "mask"
                and attn_metadata.seq_lens.shape[0] == query.size(0)):
            dyn_lens_list = getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)
            if dyn_lens_list is not None:
                context_lens = torch.tensor(
                    dyn_lens_list,
                    device=attn_metadata.seq_lens.device,
                    dtype=attn_metadata.seq_lens.dtype,
                )
                return self._forward_decode_with_mask_validation(
                    query=query,
                    attn_metadata=attn_metadata,
                    context_lens=context_lens,
                    output=output,
                )
        if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and self.sliding_window is not None
                and attn_metadata.seq_lens.shape[0] == query.size(0)):
            return self._forward_fia_slidingwindow(query, attn_metadata,
                                                   output)
        key, value, block_size, block_table, actual_seq_lengths_kv \
            = self._get_fia_params(key, value, attn_metadata)
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache and self.attn_type != AttentionType.ENCODER_DECODER:
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )

        attn_output = attn_output.view(num_tokens, self.num_heads,
                                       self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        if forward_context.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        context_lens = attn_metadata.seq_lens
        dyn_keep = getattr(attn_metadata, "dynamic_kv_keep_indices_list", None)
        dyn_lens_list = getattr(attn_metadata, "dynamic_kv_seq_lens_list", None)

        # If keep indices exist, prefer to derive kv_len from indices.
        if dyn_keep is not None and isinstance(dyn_keep, list) and dyn_keep:
            try:
                dyn_lens_list = [len(x) for x in dyn_keep]
            except Exception:
                pass

        if dyn_lens_list is not None:
            if not getattr(attn_metadata, "_dynamic_kv_decode_logged", False):
                try:
                    layer_idx = int(extract_layer_index(attn_metadata.layer_name, num_attn_module=1))
                except Exception:
                    layer_idx = -1
                if dyn_keep is not None and isinstance(dyn_keep, list) and len(dyn_keep) == len(dyn_lens_list):
                    bad = 0
                    for idxs, kv_len in zip(dyn_keep, dyn_lens_list):
                        if not isinstance(idxs, list):
                            bad += 1
                            continue
                        if len(idxs) != int(kv_len):
                            bad += 1
                            continue
                        if any((not isinstance(t, int)) for t in idxs):
                            bad += 1
                            continue
                        if any(idxs[i] > idxs[i + 1] for i in range(len(idxs) - 1)):
                            bad += 1
                            continue
                        if idxs and (idxs[0] < 0):
                            bad += 1
                            continue
                    if bad:
                        logger.warning(
                            "[DynamicKV][Decode] keep_indices consistency warnings: layer=%s layer_idx=%d bad_reqs=%d/%d",
                            attn_metadata.layer_name,
                            layer_idx,
                            bad,
                            len(dyn_lens_list),
                        )
                setattr(attn_metadata, "_dynamic_kv_decode_logged", True)
            # Per-layer kv_len override for PD DynamicKV.
            # Expect dyn_lens_list to be ordered by request order in the batch.
            context_lens = torch.tensor(
                dyn_lens_list,
                device=attn_metadata.seq_lens.device,
                dtype=attn_metadata.seq_lens.dtype,
            )
        # ``mask`` validation: decode uses full KV + ``npu_fusion_attention`` mask
        # (indices from PD). ``zero`` validation: KV already zeroed on prefill; decode
        # uses normal paged attention.
        if self._dynamickv_validation_mode == "mask":
            return self._forward_decode_with_mask_validation(
                query=query,
                attn_metadata=attn_metadata,
                context_lens=context_lens,
                output=output,
            )
        
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=context_lens,
            out=output,
        )
        return output

    def _forward_decode_with_mask_validation(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        context_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode ``validation_mode=mask``: full KV + attention mask (fusion)."""
        mode = "mask"
        try:
            layer_idx = int(extract_layer_index(attn_metadata.layer_name, num_attn_module=1))
        except Exception:
            layer_idx = 0

        num_reqs = int(attn_metadata.block_tables.shape[0])
        num_tokens = int(query.shape[0])

        output_list = []
        token_offset = 0

        if not hasattr(self, "_dynkv_decode_validation_logged"):
            self._dynkv_decode_validation_logged = set()
        _log_set = self._dynkv_decode_validation_logged
        if len(_log_set) > 4096:
            _log_set.clear()

        for ridx in range(num_reqs):
            rid = (
                attn_metadata.req_ids[ridx]
                if hasattr(attn_metadata, "req_ids")
                and attn_metadata.req_ids
                and ridx < len(attn_metadata.req_ids)
                else None
            )

            full_seq_len = (
                int(attn_metadata.seq_lens[ridx].item())
                if hasattr(attn_metadata, "seq_lens")
                else int(context_lens[ridx].item())
            )

            block_table_row = attn_metadata.block_tables[ridx]
            k_full, v_full = gather_kv_from_paged_cache(
                self.key_cache,
                self.value_cache,
                block_table_row,
                full_seq_len,
            )

            important_mask = get_validation_mask(rid, layer_idx) if rid else None
            if important_mask is not None:
                important_mask = important_mask.to(device=k_full.device)
                ml = int(important_mask.shape[0])
                if ml < full_seq_len:
                    pad_n = full_seq_len - ml
                    important_mask = torch.cat(
                        (
                            important_mask,
                            torch.ones(
                                pad_n,
                                dtype=torch.bool,
                                device=k_full.device,
                            ),
                        ),
                        dim=0,
                    )
                elif ml > full_seq_len:
                    important_mask = important_mask[:full_seq_len]

            n_imp = -1
            if important_mask is not None and important_mask.shape[0] == full_seq_len:
                n_imp = int(important_mask.sum().item())

            log_key = (str(rid or ""), int(layer_idx), mode)
            if log_key not in _log_set:
                _log_set.add(log_key)
                try:
                    from vllm.distributed.parallel_state import (
                        get_tensor_model_parallel_rank,
                    )

                    _do = int(get_tensor_model_parallel_rank()) == 0
                except Exception:
                    _do = True
                if _do:
                    logger.debug(
                        "[DynamicKV][Decode][validation] mode=%s layer_idx=%d "
                        "request_id=%s important_tokens=%d seq_len=%d",
                        mode,
                        layer_idx,
                        rid,
                        n_imp,
                        full_seq_len,
                    )

            if important_mask is not None and important_mask.shape[0] == full_seq_len:
                attn_mask = build_attention_mask_from_important_mask(
                    important_mask,
                    query_len=1,
                )
            else:
                attn_mask = torch.zeros(
                    1, full_seq_len, device=query.device, dtype=torch.uint8
                )

            q = query[token_offset : token_offset + 1]

            fa_result = torch_npu.npu_fusion_attention(
                query=q,
                key=k_full,
                value=v_full,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                atten_mask=attn_mask,
                actual_seq_qlen=[1],
                actual_seq_kvlen=[full_seq_len],
            )
            attn_out = fa_result[0] if isinstance(fa_result, (list, tuple)) else fa_result

            output_list.append(attn_out)
            token_offset += 1

        if output_list:
            combined = torch.cat(output_list, dim=0)
            output[:num_tokens] = combined[:num_tokens]

        return output

    def _forward_encoder_attention(self, query: torch.Tensor,
                                   key: torch.Tensor, value: torch.Tensor,
                                   attn_metadata: AscendMetadata,
                                   _: torch.Tensor) -> torch.Tensor:
        assert attn_metadata is not None

        if attn_metadata.causal:
            # use sparse_mode 3 in causal scenario
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                sparse_mode=3,
                atten_mask=attn_metadata.attn_mask,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]
        else:
            # use default sparse_mode 0 in normal scenario, which means no mask works on it
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
    ):

        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = (self.attn_type == AttentionType.ENCODER_DECODER)
            if get_ascend_device_type() == AscendDeviceType.A5:
                # TODO: Once eagle running to here, it may has error because of the 0 dim of slot_mapping.
                # Should check if the 0 dim of slot_mapping must equal to the 0 dim of key.
                # If it's necessary, the slots should be sliced.
                torch_npu.npu_scatter_pa_kv_cache(
                    key=key[:attn_metadata.num_actual_tokens]
                    if not encoder_decoder else key,
                    value=value[:attn_metadata.num_actual_tokens].contiguous()
                    if not encoder_decoder else value,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_mapping=slots)
            else:
                torch_npu._npu_reshape_and_cache(
                    key=key[:attn_metadata.num_actual_tokens]
                    if not encoder_decoder else key,
                    value=value[:attn_metadata.num_actual_tokens]
                    if not encoder_decoder else value,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots[:attn_metadata.num_actual_tokens]
                    if not encoder_decoder else slots)
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return key, value

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and using_paged_attention(num_tokens, self.vllm_config)
                and self.sliding_window is None):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(
                query, key, value, attn_metadata, output)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Register this layer's impl + kv_cache handle so last layer can
        # rewrite other layers' KV for DynamicKV compression.
        try:
            if attn_metadata.layer_name:
                _ASCEND_DYNKV_LAYER_REGISTRY[attn_metadata.layer_name] = (self, kv_cache)
        except Exception:
            pass

        # NOTE: On prefix-cache hits, prefill engine may run the final step with
        # attn_state=PrefillCacheHit/DecodeOnly and without explicit key/value
        # tensors. DynamicKV export still needs to run from the already-cached
        # paged KV in that case.
        has_kv_inputs = (key is not None and value is not None)
        if has_kv_inputs or (
            self._dynamickv_enabled
            and self._dynamickv_model_type_ok
            and self.is_kv_producer
            and attn_metadata.attn_state
            in (
                AscendAttentionState.PrefillCacheHit,
                AscendAttentionState.DecodeOnly,
            )
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            # DynamicKV: compress KV in prefill paths.
            # NOTE: We only enable this for selected model types (default: Mistral)
            # and non-encoder-decoder attention.
            if (
                self._dynamickv_enabled
                and self._dynamickv_model_type_ok
                and attn_metadata.attn_state
                in (
                    AscendAttentionState.PrefillNoCache,
                    AscendAttentionState.ChunkedPrefill,
                    # Prefix-cache hits can schedule only the tail tokens and mark
                    # the step as PrefillCacheHit. We still want to export per-layer
                    # DynamicKV results for PD transfer on the last prefill step.
                    AscendAttentionState.PrefillCacheHit,
                    # On prefill (KV producer) engine, prefix-cache hits may run
                    # the final step as DecodeOnly. We still need DynamicKV export
                    # from the already-cached KV.
                    AscendAttentionState.DecodeOnly,
                )
                and self.attn_type != AttentionType.ENCODER_DECODER
            ):
                # DynamicKV compression:
                # - PrefillNoCache: always (single-shot full prefill).
                # - ChunkedPrefill: only when dynamic_kv_is_last_chunk (intermediate chunks
                #   only extend KV without compression).
                forward_context = get_forward_context()
                if getattr(forward_context, "capturing", False):
                    if has_kv_inputs:
                        key, value = self.reshape_and_cache(
                            key, value, kv_cache, attn_metadata)
                else:
                    is_chunked = (
                        attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill)
                    is_last_chunk = bool(
                        getattr(attn_metadata, "dynamic_kv_is_last_chunk", False))
                    # For PrefillCacheHit, treat this step as the last prefill step
                    # for DynamicKV export purposes (prefix cache already covers the
                    # leading tokens).
                    if attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
                        is_last_chunk = True
                    if (self.is_kv_producer
                            and attn_metadata.attn_state
                            == AscendAttentionState.DecodeOnly):
                        is_last_chunk = True

                    if is_chunked and not is_last_chunk:
                        if has_kv_inputs:
                            key, value = self.reshape_and_cache(
                                key, value, kv_cache, attn_metadata)
                    else:
                        # PrefillNoCache OR ChunkedPrefill last chunk.
                        if has_kv_inputs:
                            key, value = self.reshape_and_cache(
                                key, value, kv_cache, attn_metadata)

                        # Cache per-layer q_last for final-layer budget + rewrite.
                        try:
                            layer_idx = int(extract_layer_index(attn_metadata.layer_name, num_attn_module=1))
                        except Exception:
                            layer_idx = -1

                        model_cfg = self.vllm_config.model_config
                        hf_cfg = getattr(model_cfg, "hf_config", None)
                        num_layers = int(
                            getattr(model_cfg, "num_hidden_layers", 0)
                            or getattr(model_cfg, "num_layers", 0)
                            or getattr(hf_cfg, "num_hidden_layers", 0)
                            or getattr(hf_cfg, "num_layers", 0)
                            or 0)
                        if layer_idx < 0 or num_layers <= 0:
                            return output

                        # Prepare per-request q_last for this layer using query boundaries.
                        qsl_cpu = [0] + [int(x) for x in (attn_metadata.actual_seq_lengths_q or [])]
                        if len(qsl_cpu) != len(attn_metadata.seq_lens_list) + 1:
                            return output

                        q_last_list: list[torch.Tensor] = []
                        for i in range(len(attn_metadata.seq_lens_list)):
                            q0, q1 = int(qsl_cpu[i]), int(qsl_cpu[i + 1])
                            qi = query[q0:q1]
                            qi_last = qi[-min(int(qi.shape[0]), int(self._dynamickv_window_size)) :] if qi.numel() else qi
                            q_last_list.append(qi_last)

                        state = _DYNKV_STATE.setdefault("prefill_batch", {})
                        assert isinstance(state, dict)
                        if layer_idx == 0:
                            # Reset per-step state at the first layer.
                            state.clear()
                        per_layer = state.setdefault("per_layer", {})
                        assert isinstance(per_layer, dict)
                        per_layer[layer_idx] = {
                            "q_last_list": q_last_list,
                            "req_ids": list(attn_metadata.req_ids or []),
                        }

                        # Compute & cache per-layer scores/indices for this layer (batch).
                        try:
                            dynkv_cfg_layer = DynamicKVConfig(
                                num_hidden_layers=num_layers,
                                window_size=int(self._dynamickv_window_size),
                                max_capacity_prompt=int(self._dynamickv_prompt_kv_len_budget),
                                pooling=(
                                    "avgpool"
                                    if self._dynamickv_pooling == "avgpool"
                                    else ("maxpool" if self._dynamickv_pooling == "maxpool" else "none")
                                ),
                                kernel_size=int(self._dynamickv_kernel_size)
                                if int(self._dynamickv_kernel_size) > 1 else 7,
                                softmax_chunk_size=int(self._dynamickv_softmax_chunk_size),
                                radio_max=float(self._dynamickv_radio_max),
                                radio_min=float(self._dynamickv_radio_min),
                            )
                            seq_lens_list_int = [int(x) for x in attn_metadata.seq_lens_list]
                            # Use this layer's KV cache from registry (not implicit self.*),
                            # so scores/indices always match the layer being processed.
                            key_cache_layer = self.key_cache
                            value_cache_layer = self.value_cache
                            try:
                                reg_entry = _ASCEND_DYNKV_LAYER_REGISTRY.get(
                                    attn_metadata.layer_name)
                                if reg_entry is not None:
                                    impl_layer, _ = reg_entry
                                    if getattr(impl_layer, "key_cache", None) is not None:
                                        key_cache_layer = impl_layer.key_cache
                                    if getattr(impl_layer, "value_cache", None) is not None:
                                        value_cache_layer = impl_layer.value_cache
                            except Exception:
                                pass
                            k_packed, _, _, cu = gather_kv_from_paged_cache_batched(
                                key_cache_layer,
                                value_cache_layer,
                                attn_metadata.block_tables,
                                seq_lens_list_int,
                            )
                            scores_old_list: list[torch.Tensor] = []
                            indices_old_list: list[torch.Tensor] = []
                            for ridx in range(len(seq_lens_list_int)):
                                seq_len_i = int(seq_lens_list_int[ridx])
                                start = 0 if ridx == 0 else int(cu[ridx - 1])
                                end = int(cu[ridx])
                                k_full = k_packed[start:end]
                                budget_size = min(
                                    int(dynkv_cfg_layer.radio_max * dynkv_cfg_layer.base),
                                    max(seq_len_i - dynkv_cfg_layer.window_size, 0),
                                )
                                # Per-head scores + aggregated indices (open-source density style).
                                s_old, idx_old = scores_and_indices_old_perhead_aggregated(
                                    query_last=q_last_list[ridx],
                                    key_full=k_full,
                                    cfg=dynkv_cfg_layer,
                                    budget_size=budget_size,
                                )
                                scores_old_list.append(s_old)
                                indices_old_list.append(idx_old)
                                
                                # NOTE: For ``validation_mode=mask``, we build the final per-layer
                                # important_mask after cross-layer budgets are determined (last layer),
                                # so that the mask reflects the *effective* keep set rather than the
                                # pre-reallocation candidate pool.
                            per_layer[layer_idx]["scores_old_list"] = scores_old_list
                            per_layer[layer_idx]["indices_old_list"] = indices_old_list
                        except Exception as e:
                            # Hard debug: if we cannot compute scores/indices, the
                            # final-layer rewrite/export will never happen.
                            try:
                                rid0 = None
                                if isinstance(attn_metadata.req_ids, list) and attn_metadata.req_ids:
                                    rid0 = attn_metadata.req_ids[0]
                                key = (rid0, "scores_indices", int(layer_idx))
                                dbgfail = _DYNKV_STATE.setdefault("debug_fail", set())
                                if isinstance(dbgfail, set) and key not in dbgfail:
                                    dbgfail.add(key)
                                    logger.warning(
                                        "[DynamicKV] prefill_fail: stage=scores_indices request_id=%s layer_idx=%d err=%r",
                                        rid0,
                                        int(layer_idx),
                                        e,
                                    )
                            except Exception:
                                pass

                        # Periodic cross-layer budget reallocation (every 4 layers).
                        if (layer_idx % 4) == 3 and layer_idx < (num_layers - 1):
                            try:
                                per_req_old_budget = state.setdefault("per_req_old_budget", {})
                                assert isinstance(per_req_old_budget, dict)
                                for ridx in range(len(attn_metadata.seq_lens_list)):
                                    rid = (
                                        attn_metadata.req_ids[ridx]
                                        if attn_metadata.req_ids and ridx < len(attn_metadata.req_ids)
                                        else None
                                    )
                                    if not rid:
                                        continue
                                    scores_layers: list[torch.Tensor] = []
                                    for li in range(layer_idx + 1):
                                        item = per_layer.get(li)
                                        if not item or "scores_old_list" not in item:
                                            scores_layers = []
                                            break
                                        scores_list = item["scores_old_list"]
                                        if not isinstance(scores_list, list) or ridx >= len(scores_list):
                                            scores_layers = []
                                            break
                                        scores_layers.append(scores_list[ridx])
                                    if not scores_layers:
                                        continue
                                    cfg_sub = DynamicKVConfig(
                                        num_hidden_layers=len(scores_layers),
                                        window_size=int(self._dynamickv_window_size),
                                        max_capacity_prompt=int(self._dynamickv_prompt_kv_len_budget),
                                        pooling="none",
                                        kernel_size=1,
                                        softmax_chunk_size=int(self._dynamickv_softmax_chunk_size),
                                        radio_max=float(self._dynamickv_radio_max),
                                        radio_min=float(self._dynamickv_radio_min),
                                    )
                                    # Dummy shapes: scores are now [Hkv, old_len], use last dim for old_len.
                                    dummy = [
                                        torch.empty(
                                            (int(s.shape[-1]) + int(cfg_sub.window_size), 1, 1),
                                            device=s.device,
                                            dtype=torch.float16,
                                        )
                                        for s in scores_layers
                                    ]
                                    seq_len_i = int(attn_metadata.seq_lens_list[ridx])
                                    budget_size = min(
                                        int(cfg_sub.radio_max * cfg_sub.base),
                                        max(seq_len_i - cfg_sub.window_size, 0),
                                    )
                                    # Per-head mode: tk = base * H * layers (open-source style).
                                    budgets_sub = update_and_reset_budget_per_kv_head(
                                        per_layer_scores_old=scores_layers,
                                        per_layer_k_budget=dummy,
                                        per_layer_v_budget=dummy,
                                        cfg=cfg_sub,
                                        budget_size=budget_size,
                                    )
                                    # Expand budgets to full num_layers length so the final writeback
                                    # can directly consume it.
                                    full = [int(cfg_sub.base)] * int(num_layers)
                                    try:
                                        # If previous allocation exists, start from it.
                                        prev = per_req_old_budget.get(rid)
                                        if isinstance(prev, dict) and isinstance(prev.get("budgets"), list):
                                            pb = [int(x) for x in prev["budgets"]]
                                            if len(pb) == num_layers:
                                                full = pb
                                    except Exception:
                                        pass
                                    for j in range(min(len(budgets_sub), num_layers)):
                                        full[j] = int(budgets_sub[j])
                                    per_req_old_budget[rid] = {"upto": layer_idx, "budgets": full}

                                    # Apply the reallocated budgets immediately by truncating
                                    # cached indices for layers computed so far.
                                    try:
                                        for li2 in range(layer_idx + 1):
                                            li_item = per_layer.get(li2)
                                            if not li_item or "indices_old_list" not in li_item:
                                                continue
                                            idx_list2 = li_item["indices_old_list"]
                                            if not isinstance(idx_list2, list) or ridx >= len(idx_list2):
                                                continue
                                            idx_old2 = idx_list2[ridx]
                                            if not isinstance(idx_old2, torch.Tensor):
                                                continue
                                            keep_old = int(full[li2]) if li2 < len(full) else int(cfg_sub.base)
                                            if keep_old < 0:
                                                keep_old = 0
                                            idx_kept = idx_old2[:keep_old]
                                            idx_list2[ridx] = idx_kept
                                            # Sync scores: keep same [Hkv, old_len] shape but zero out
                                            # mass outside the kept index set so later realloc matches indices.
                                            scores_list2 = li_item.get("scores_old_list")
                                            if isinstance(scores_list2, list) and ridx < len(
                                                    scores_list2):
                                                s_old2 = scores_list2[ridx]
                                                if isinstance(s_old2, torch.Tensor) and s_old2.numel(
                                                ) > 0:
                                                    new_s = torch.zeros_like(s_old2)
                                                    if idx_kept.numel() > 0:
                                                        # s_old2 is [Hkv, old_len], apply mask per head.
                                                        if s_old2.dim() == 2:
                                                            old_len_s = int(s_old2.shape[-1])
                                                            idx_clamped = idx_kept.clamp(0, old_len_s - 1).to(torch.long)
                                                            for h in range(s_old2.shape[0]):
                                                                new_s[h].scatter_(
                                                                    0,
                                                                    idx_clamped,
                                                                    s_old2[h].index_select(0, idx_clamped),
                                                                )
                                                        else:
                                                            # Fallback for 1D scores (backward compat).
                                                            idx_clamped = idx_kept.clamp(0, s_old2.numel() - 1).to(torch.long)
                                                            new_s.scatter_(
                                                                0,
                                                                idx_clamped,
                                                                s_old2.index_select(0, idx_clamped),
                                                            )
                                                    scores_list2[ridx] = new_s
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                        # Trigger at last layer: compute/reallocate budgets and rewrite all layers.
                        if layer_idx != num_layers - 1:
                            return output

                        # Build per-layer intermediate budgets.
                        assert self.key_cache is not None and self.value_cache is not None
                        num_reqs = len(attn_metadata.seq_lens_list)
                        if num_reqs <= 0:
                            return output

                        dynkv_cfg = DynamicKVConfig(
                            num_hidden_layers=num_layers,
                            window_size=int(self._dynamickv_window_size),
                            max_capacity_prompt=int(self._dynamickv_prompt_kv_len_budget),
                            pooling=(
                                "avgpool"
                                if self._dynamickv_pooling == "avgpool"
                                else ("maxpool" if self._dynamickv_pooling == "maxpool" else "none")
                            ),
                            kernel_size=int(self._dynamickv_kernel_size)
                            if int(self._dynamickv_kernel_size) > 1 else 7,
                            softmax_chunk_size=int(self._dynamickv_softmax_chunk_size),
                            radio_max=float(self._dynamickv_radio_max),
                            radio_min=float(self._dynamickv_radio_min),
                        )

                        # Map layer_idx -> layer_name using registry (preferred).
                        layer_idx_to_name: dict[int, str] = {}
                        for name in list(_ASCEND_DYNKV_LAYER_REGISTRY.keys()):
                            try:
                                li = int(extract_layer_index(name, num_attn_module=1))
                            except Exception:
                                continue
                            if li not in layer_idx_to_name:
                                layer_idx_to_name[li] = name
                        if len(layer_idx_to_name) < num_layers:
                            return output

                        # For each request, compute per-layer budgets and rewrite caches.
                        results_by_req: dict[str, dict[str, object]] = {}
                        # Precompute window tail indices per request once.
                        seq_lens_list_int = [int(x) for x in attn_metadata.seq_lens_list]
                        per_req_old_budget = state.get("per_req_old_budget") if isinstance(state, dict) else None
                        tail_keep_idx_by_req: list[torch.Tensor] = [
                            torch.empty((0,), device=query.device, dtype=torch.long)
                            for _ in range(num_reqs)
                        ]
                        # PERF: pre-gather KV for each layer across the batch once.
                        gathered_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]] = {}
                        for ridx in range(num_reqs):
                            rid = (attn_metadata.req_ids[ridx] if attn_metadata.req_ids and ridx < len(attn_metadata.req_ids) else None)
                            if not rid:
                                continue
                            seq_len_i = int(seq_lens_list_int[ridx])
                            if seq_len_i <= 0:
                                continue
                            max_cap_cfg = int(dynkv_cfg.max_capacity_prompt)
                            if seq_len_i <= max_cap_cfg:
                                # Full prompt fits in prompt_kv_len_budget: export full prefix;
                                # do not rewrite cache (same semantics as offload path).
                                results_by_req[rid] = {
                                    "per_layer_kv_lens": [seq_len_i] * num_layers,
                                    "per_layer_keep_indices": [
                                        list(range(seq_len_i)) for _ in range(num_layers)
                                    ],
                                }
                                continue
                            W = int(dynkv_cfg.window_size)
                            tail_keep_idx_by_req[ridx] = (
                                torch.arange(
                                    max(seq_len_i - W, 0),
                                    seq_len_i,
                                    device=query.device,
                                    dtype=torch.long,
                                )
                                if (W > 0 and seq_len_i > 0)
                                else torch.empty((0,), device=query.device, dtype=torch.long)
                            )

                            block_row = attn_metadata.block_tables[ridx]

                            # Per-layer buffers for this request.
                            per_layer_scores: list[torch.Tensor] = []
                            per_layer_slots_full: list[torch.Tensor] = []
                            per_layer_indices_old: list[torch.Tensor] = []

                            budget_size = min(
                                int(dynkv_cfg.radio_max * dynkv_cfg.base),
                                max(seq_len_i - dynkv_cfg.window_size, 0),
                            )

                            # Iterate layers in order; require q_last_list captured.
                            for li in range(num_layers):
                                item = per_layer.get(li)
                                if not item or "q_last_list" not in item:
                                    try:
                                        key = (rid, "missing_q_last_list", int(li))
                                        dbgfail = _DYNKV_STATE.setdefault("debug_fail", set())
                                        if isinstance(dbgfail, set) and key not in dbgfail:
                                            dbgfail.add(key)
                                            logger.warning(
                                                "[DynamicKV] prefill_fail: stage=final_rewrite request_id=%s reason=missing_q_last_list layer_idx=%d",
                                                rid,
                                                int(li),
                                            )
                                    except Exception:
                                        pass
                                    return output
                                q_last_list_i = item["q_last_list"]
                                if not isinstance(q_last_list_i, list) or ridx >= len(q_last_list_i):
                                    return output
                                q_last = q_last_list_i[ridx]
                                if not isinstance(q_last, torch.Tensor) or q_last.numel() == 0:
                                    # If no queries for this request in this chunk, skip rewrite.
                                    q_last = q_last  # keep empty, handled below

                                layer_name = layer_idx_to_name.get(li)
                                if layer_name is None:
                                    return output
                                impl_i, kv_cache_i = _ASCEND_DYNKV_LAYER_REGISTRY[layer_name]
                                key_cache_i = impl_i.key_cache
                                value_cache_i = impl_i.value_cache
                                if key_cache_i is None or value_cache_i is None:
                                    return output

                                packed = gathered_by_layer.get(li)
                                if packed is None:
                                    packed = gather_kv_from_paged_cache_batched(
                                        key_cache_i,
                                        value_cache_i,
                                        attn_metadata.block_tables,
                                        seq_lens_list_int,
                                    )
                                    gathered_by_layer[li] = packed
                                k_packed, v_packed, slots_packed, cu = packed
                                start = 0 if ridx == 0 else int(cu[ridx - 1])
                                end = int(cu[ridx])
                                k_full = k_packed[start:end]
                                v_full = v_packed[start:end]
                                slots_full = slots_packed[start:end]

                                cached = per_layer.get(li)
                                if not cached or "scores_old_list" not in cached or "indices_old_list" not in cached:
                                    return output
                                scores_list = cached["scores_old_list"]
                                indices_list = cached["indices_old_list"]
                                if not isinstance(scores_list, list) or not isinstance(indices_list, list):
                                    return output
                                if ridx >= len(scores_list) or ridx >= len(indices_list):
                                    return output
                                per_layer_scores.append(scores_list[ridx])
                                per_layer_slots_full.append(slots_full)
                                per_layer_indices_old.append(indices_list[ridx])

                            # Dummy shapes: scores are now [Hkv, old_len], use last dim for old_len.
                            dummy = [
                                torch.empty(
                                    (int(s.shape[-1]) + int(dynkv_cfg.window_size), 1, 1),
                                    device=s.device,
                                    dtype=torch.float16,
                                )
                                for s in per_layer_scores
                            ]
                            # Use the latest periodic reallocation budget if present; otherwise fallback
                            # to final reallocation over all layers.
                            per_layer_old_budget: list[int] | None = None
                            try:
                                if rid and isinstance(per_req_old_budget, dict):
                                    b = per_req_old_budget.get(rid)
                                    if isinstance(b, dict) and isinstance(b.get("budgets"), list):
                                        per_layer_old_budget = [int(x) for x in b["budgets"]]
                            except Exception:
                                per_layer_old_budget = None
                            if per_layer_old_budget is None or len(per_layer_old_budget) != num_layers:
                                # Per-head mode: tk = base * H * layers (open-source style).
                                per_layer_old_budget = update_and_reset_budget_per_kv_head(
                                    per_layer_scores_old=per_layer_scores,
                                    per_layer_k_budget=dummy,
                                    per_layer_v_budget=dummy,
                                    cfg=dynkv_cfg,
                                    budget_size=budget_size,
                                )

                            per_layer_kv_lens: list[int] = []
                            per_layer_keep_indices: list[list[int]] = []

                            # Validation: skip KV pack; ``mask`` exports indices for decode;
                            # ``zero`` zeros unimportant slots in paged KV on this worker.
                            if self._dynamickv_validation_mode == "mask":
                                full_indices = list(range(seq_len_i))
                                for li in range(num_layers):
                                    per_layer_kv_lens.append(seq_len_i)
                                    per_layer_keep_indices.append(full_indices)
                                plii: list[list[int]] = []
                                for li in range(num_layers):
                                    old_budget = int(per_layer_old_budget[li])
                                    idx_old = per_layer_indices_old[li]
                                    tail_idx_m = tail_keep_idx_by_req[ridx].to(idx_old.device)
                                    old_budget_eff = max(
                                        0,
                                        min(int(old_budget), int(idx_old.shape[0])),
                                    )
                                    idx_old_keep = (
                                        idx_old[:old_budget_eff].to(torch.long)
                                        if old_budget_eff > 0
                                        else idx_old[:0].to(torch.long)
                                    )
                                    keep_sorted, _ = cap_keep_indices_chronological(
                                        idx_old=idx_old_keep,
                                        old_budget=old_budget_eff,
                                        tail_idx=tail_idx_m,
                                        cap=seq_len_i,
                                        device=idx_old.device,
                                    )
                                    # Save per-layer important_mask for in-proc decode validation,
                                    # and export sparse important indices for PD decode.
                                    if rid:
                                        important_mask = torch.zeros(
                                            seq_len_i,
                                            dtype=torch.bool,
                                            device=idx_old.device,
                                        )
                                        if keep_sorted.numel() > 0:
                                            important_mask[keep_sorted.to(torch.long)] = True
                                        save_validation_mask(rid, li, important_mask)
                                    plii.append([int(x) for x in keep_sorted.tolist()])
                                results_by_req[rid] = {
                                    "per_layer_kv_lens": per_layer_kv_lens,
                                    "per_layer_keep_indices": per_layer_keep_indices,
                                    "per_layer_important_indices": plii,
                                }
                                _val_log = True
                                try:
                                    from vllm.distributed.parallel_state import (
                                        get_tensor_model_parallel_rank,
                                    )

                                    _val_log = int(get_tensor_model_parallel_rank()) == 0
                                except Exception:
                                    pass
                                if _val_log:
                                    # Debug-level verbose log (full per-layer counts).
                                    # Enable DEBUG logging when needed; avoid spamming prefill.log by default.
                                    important_n_by_layer = [len(x) for x in plii] if plii else []
                                    pairs = ", ".join(
                                        f"L{lj}={nj}" for lj, nj in enumerate(important_n_by_layer)
                                    ) if important_n_by_layer else "<empty>"
                                    logger.debug(
                                        "[DynamicKV][attn] validation_mode=%s: skip KV pack, "
                                        "full_prefix metadata L=%d request_id=%s "
                                        "important_tokens_per_layer: %s (%s)",
                                        str(self._dynamickv_validation_mode),
                                        int(seq_len_i),
                                        rid,
                                        pairs,
                                        "PD mask indices attached",
                                    )
                                continue  # Skip KV rewrite for this request
                            if self._dynamickv_validation_mode == "zero":
                                full_indices = list(range(seq_len_i))
                                for li in range(num_layers):
                                    per_layer_kv_lens.append(seq_len_i)
                                    per_layer_keep_indices.append(full_indices)
                                for li in range(num_layers):
                                    old_budget = int(per_layer_old_budget[li])
                                    idx_old = per_layer_indices_old[li]
                                    tail_idx_z = tail_keep_idx_by_req[ridx].to(
                                        idx_old.device
                                    )
                                    old_budget_eff = max(
                                        0,
                                        min(int(old_budget), int(idx_old.shape[0])),
                                    )
                                    idx_old_keep = (
                                        idx_old[:old_budget_eff].to(torch.long)
                                        if old_budget_eff > 0
                                        else idx_old[:0].to(torch.long)
                                    )
                                    keep_sorted, _ = cap_keep_indices_chronological(
                                        idx_old=idx_old_keep,
                                        old_budget=old_budget_eff,
                                        tail_idx=tail_idx_z,
                                        cap=seq_len_i,
                                        device=idx_old.device,
                                    )
                                    important_mask = torch.zeros(
                                        seq_len_i,
                                        dtype=torch.bool,
                                        device=idx_old.device,
                                    )
                                    if keep_sorted.numel() > 0:
                                        important_mask[
                                            keep_sorted.to(torch.long)
                                        ] = True
                                    layer_name_z = layer_idx_to_name.get(li)
                                    if not layer_name_z:
                                        continue
                                    impl_z, _ = _ASCEND_DYNKV_LAYER_REGISTRY[
                                        layer_name_z
                                    ]
                                    kc = impl_z.key_cache
                                    vc = impl_z.value_cache
                                    if kc is None or vc is None:
                                        continue
                                    slots_full = per_layer_slots_full[li]
                                    k_fl = kc.reshape(
                                        -1, kc.shape[-2], kc.shape[-1]
                                    )
                                    v_fl = vc.reshape(
                                        -1, vc.shape[-2], vc.shape[-1]
                                    )
                                    unimportant = ~important_mask
                                    if unimportant.any():
                                        sb = slots_full[unimportant].to(
                                            torch.long
                                        ).flatten()
                                        sb = torch.unique(sb)
                                        if sb.numel() > 0:
                                            zblk = k_fl.new_zeros(
                                                (
                                                    int(sb.numel()),
                                                    int(k_fl.shape[-2]),
                                                    int(k_fl.shape[-1]),
                                                )
                                            )
                                            k_fl.index_copy_(
                                                0, sb.to(k_fl.device), zblk
                                            )
                                            v_fl.index_copy_(
                                                0, sb.to(v_fl.device), zblk
                                            )
                                results_by_req[rid] = {
                                    "per_layer_kv_lens": per_layer_kv_lens,
                                    "per_layer_keep_indices": per_layer_keep_indices,
                                }
                                continue

                            # Rewrite each layer for this request.
                            for li in range(num_layers):
                                old_budget = int(per_layer_old_budget[li])
                                # Merge old top-k + tail window, unique-sort, cap to prompt_kv_len_budget
                                # (align with ``impl=offload``).
                                #
                                # IMPORTANT: `dynamic_kv.prompt_kv_len_budget` is not treated as a hard kv_len cap.
                                # Some layers may keep more than `prompt_kv_len_budget` tokens; the real upper bound
                                # is the prompt length `seq_len_i`.
                                idx_old = per_layer_indices_old[li]
                                tail_idx = tail_keep_idx_by_req[ridx].to(idx_old.device)
                                keep_sorted, _ = cap_keep_indices_chronological(
                                    idx_old=idx_old,
                                    old_budget=old_budget,
                                    tail_idx=tail_idx,
                                    cap=seq_len_i,
                                    device=idx_old.device,
                                )
                                per_layer_keep_indices.append(
                                    [int(x) for x in keep_sorted.tolist()]
                                )

                                # Build compressed KV by gathering from full KV.
                                packed = gathered_by_layer.get(li)
                                if packed is None:
                                    return output
                                k_packed, v_packed, slots_packed, cu = packed
                                start = 0 if ridx == 0 else int(cu[ridx - 1])
                                end = int(cu[ridx])
                                k_full = k_packed[start:end]
                                v_full = v_packed[start:end]
                                k_final = k_full.index_select(0, keep_sorted.to(torch.long))
                                v_final = v_full.index_select(0, keep_sorted.to(torch.long))

                                kv_len_i = int(k_final.shape[0])
                                per_layer_kv_lens.append(kv_len_i)

                                slots_keep = per_layer_slots_full[li][:kv_len_i]

                                saved_num_actual_tokens = attn_metadata.num_actual_tokens
                                saved_slot_mapping = attn_metadata.slot_mapping
                                attn_metadata.num_actual_tokens = kv_len_i
                                attn_metadata.slot_mapping = slots_keep
                                attn_metadata.seq_lens_list_kv = [kv_len_i]
                                attn_metadata.actual_seq_lengths_kv = [kv_len_i]

                                layer_name = layer_idx_to_name[li]
                                impl_i, kv_cache_i = _ASCEND_DYNKV_LAYER_REGISTRY[layer_name]
                                _ = impl_i.reshape_and_cache(k_final, v_final, kv_cache_i, attn_metadata)

                                attn_metadata.num_actual_tokens = saved_num_actual_tokens
                                attn_metadata.slot_mapping = saved_slot_mapping

                            results_by_req[rid] = {
                                "per_layer_kv_lens": per_layer_kv_lens,
                                "per_layer_keep_indices": per_layer_keep_indices,
                            }

                        # Clear state after rewrite.
                        _DYNKV_STATE.pop("prefill_batch", None)
                        # Export results to shared state for Mooncake to attach.
                        try:
                            last_results = _DYNKV_STATE.get("last_results")
                            if isinstance(last_results, dict):
                                now_s = time.time()
                                for rid, item in results_by_req.items():
                                    last_results[rid] = dict(item)
                                    last_results[rid]["ts"] = now_s
                                _dynkv_prune_last_results(now_s)
                        except Exception:
                            pass
                        # Registry lifecycle: drop cached layer handles after rewrite.
                        try:
                            _ASCEND_DYNKV_LAYER_REGISTRY.clear()
                        except Exception:
                            pass
            else:
                if has_kv_inputs:
                    key, value = self.reshape_and_cache(
                        key, value, kv_cache, attn_metadata)
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(
                query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        output = self.forward_impl(query, key, value, kv_cache, attn_metadata,
                                   output)
        return output
