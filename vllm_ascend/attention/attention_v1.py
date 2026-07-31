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

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Type

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.ops.turboquant_kv_cache import (
    _try_8bit_decode_paged,
    bit_residual_attention_paged_k8v4,
    bit_residual_fia_paged_k8v4,
    bit_residual_k8v4_key_packed_width,
    turboquant_4bit_slab_cache_enabled,
    turboquant_attention_paged4bit,
    turboquant_attention_paged8bit,
    ensure_turboquant_pack_tables_registered,
    turboquant_decode_kv_cache_compact,
    turboquant_fused_infer_attention_score_8bit,
    turboquant_pack_kv_for_cache,
    turboquant_pack_kv_for_cache_to_cache,
    turboquant_packed_bytes_per_vector,
    turboquant_slab_block_size_or_none,
    warm_up_turboquant_4bit_pack_op,
    warm_up_turboquant_4bit_tables,
)
from vllm_ascend.ascend_config import get_ascend_config
import vllm_ascend.envs as envs_ascend

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_TURBOQUANT_FIA_WARMED: set[tuple[int, torch.dtype, int, int, int]] = set()


def _normalize_npu_device(device: torch.device) -> torch.device:
    if device.type not in ("npu", "privateuseone") or device.index is not None:
        return device
    try:
        return torch.device(device.type, torch.npu.current_device())
    except Exception:
        return torch.device(device.type, 0)


def _sync_npu(device: torch.device) -> None:
    if device.type not in ("npu", "privateuseone"):
        return
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _warm_up_turboquant_fia_op(
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> None:
    """Pre-launch FIA once so smoke profile does not include ACL cold start."""
    if dtype not in (torch.float16, torch.bfloat16):
        return
    device = _normalize_npu_device(device)
    device_index = device.index if device.index is not None else 0
    warm_key = (device_index, dtype, head_size, num_heads, num_kv_heads)
    if warm_key in _TURBOQUANT_FIA_WARMED:
        return

    query = torch.zeros((1, num_heads, head_size), device=device, dtype=dtype)
    key = torch.zeros((1, num_kv_heads, head_size), device=device, dtype=dtype)
    value = torch.zeros_like(key)
    torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        input_layout="TND",
        actual_seq_lengths=[1],
        actual_seq_lengths_kv=[1],
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=3,
    )
    _sync_npu(device)
    _TURBOQUANT_FIA_WARMED.add(warm_key)


def _turboquant_kv_packed_slot_width(head_size: int) -> int:
    """Per-vector packed byte width for K/V cache rows (``max(P_k, P_v)``)."""
    try:
        cfg = get_ascend_config()
        pk = turboquant_packed_bytes_per_vector(
            head_size, bits=cfg.turboquant_kv_bits_key
        )
        pv = turboquant_packed_bytes_per_vector(
            head_size, bits=cfg.turboquant_kv_bits_value
        )
        return max(pk, pv)
    except Exception:
        return turboquant_packed_bytes_per_vector(head_size, bits=4)


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
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        if cache_dtype_str == "turboquant":
            try:
                cfg = get_ascend_config()
                bits_key = cfg.turboquant_kv_bits_key
                bits_value = cfg.turboquant_kv_bits_value
            except Exception:
                bits_key = bits_value = None
            # BitResidual k8v4: asymmetric 3D slab layout.
            if bits_key == 8 and bits_value == 4 and head_size == 128:
                key_width = bit_residual_k8v4_key_packed_width(block_size)
                return (2, num_blocks, num_kv_heads, key_width)
            if turboquant_4bit_slab_cache_enabled(
                bits_key, bits_value
            ):
                packed = turboquant_packed_bytes_per_vector(head_size, bits=4)
                return (2, num_blocks, num_kv_heads, block_size * packed)
            packed = _turboquant_kv_packed_slot_width(head_size)
            return (2, num_blocks, block_size, num_kv_heads, packed)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
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
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_decodes_flatten: int = 0
    num_reqs: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

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
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # sliding window attention mask
    swa_mask: torch.Tensor | None = None

class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

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
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

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

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        pack_num_reqs = 0
        while (
            pack_num_reqs < num_reqs
            and query_start_loc_cpu[pack_num_reqs].item() < num_actual_tokens
        ):
            pack_num_reqs += 1

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)[
                :num_actual_tokens
            ]
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        # Pack-to-cache / _npu_reshape_and_cache expect dense int32 slot indices.
        # Normalize once here so hot paths do not call .to() / .contiguous() per layer.
        if slot_mapping.dtype != torch.int32:
            slot_mapping = slot_mapping.to(torch.int32)
        if not slot_mapping.is_contiguous():
            slot_mapping = slot_mapping.contiguous()

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask and swa_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        swa_mask = None
        is_swa = hasattr(self.model_config.hf_text_config, "sliding_window")
        if self.model_config is not None and is_swa:
            swa_mask = self.attn_mask_builder.get_swa_mask(
                self.model_config.dtype, self.model_config.hf_text_config.sliding_window
            )

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            num_reqs=pack_num_reqs,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
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
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
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
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype == "turboquant":
            try:
                cfg = get_ascend_config()
                self.turboquant_kv_bits_key = cfg.turboquant_kv_bits_key
                self.turboquant_kv_bits_value = cfg.turboquant_kv_bits_value
            except Exception:
                self.turboquant_kv_bits_key = 4
                self.turboquant_kv_bits_value = 4
            # Eager-register pack v2 tables (codebook + R^T) on the default NPU device.
            # Only for TurboQuant [8,8] registered-pack; BitResidual K8V4 does not use them.
            try:
                if (
                    self.turboquant_kv_bits_key == 8
                    and self.turboquant_kv_bits_value == 8
                ):
                    ensure_turboquant_pack_tables_registered(
                        torch.device("npu"),
                        head_size,
                        self.turboquant_kv_bits_key,
                    )
                warm_up_turboquant_4bit_tables(
                    torch.device("npu"),
                    head_size,
                    self.turboquant_kv_bits_key,
                    self.turboquant_kv_bits_value,
                    self.vllm_config.model_config.dtype,
                )
                block_size = self.vllm_config.cache_config.block_size
                warm_up_turboquant_4bit_pack_op(
                    device=torch.device("npu"),
                    dtype=self.vllm_config.model_config.dtype,
                    head_size=head_size,
                    num_kv_heads=self.num_kv_heads,
                    block_size=block_size,
                    bits_key=self.turboquant_kv_bits_key,
                    bits_value=self.turboquant_kv_bits_value,
                )
                if (
                    self.turboquant_kv_bits_key == 4
                    and self.turboquant_kv_bits_value == 4
                ):
                    _warm_up_turboquant_fia_op(
                        device=torch.device("npu"),
                        dtype=self.vllm_config.model_config.dtype,
                        head_size=head_size,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        scale=self.scale,
                    )
                # Warm up BitResidual k8v4 rotation tensors (lazy-init quantizer).
                if (
                    self.turboquant_kv_bits_key == 8
                    and self.turboquant_kv_bits_value == 4
                    and head_size == 128
                ):
                    from vllm_ascend.ops.turboquant_kv_cache import (
                        _bit_residual_k8v4_rotation_t,
                        _bit_residual_k8v4_rotation,
                    )
                    _bit_residual_k8v4_rotation_t(
                        torch.device("npu"), self.vllm_config.model_config.dtype
                    )
                    _bit_residual_k8v4_rotation(
                        torch.device("npu"), self.vllm_config.model_config.dtype
                    )
            except Exception:
                pass
        else:
            self.turboquant_kv_bits_key = 4
            self.turboquant_kv_bits_value = 4
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self._decoded_key_cache = None
        self._decoded_value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.sinks = sinks

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        block_tables = attn_metadata[key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=3,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
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
            if _EXTRA_CTX.is_draft_model:
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
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                actual_seq_lengths_q,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )

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
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
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
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata):
        def _cache_view_for_fia(cache: torch.Tensor) -> tuple[int, torch.Tensor]:
            if self.kv_cache_dtype == "turboquant":
                # BitResidual k8v4 uses its own attention op; cache is 3D slab.
                # If k8v4 attention returns None (fallback), we need a view for
                # standard FIA. For now, return a placeholder — k8v4 should not
                # fall through to standard FIA.
                if (
                    self.turboquant_kv_bits_key == 8
                    and self.turboquant_kv_bits_value == 4
                    and cache.ndim == 3
                ):
                    # k8v4 cache: [num_blocks, num_heads, packed_width]
                    # No valid FIA view; return cache directly as placeholder.
                    block_size = self.vllm_config.cache_config.block_size
                    return block_size, cache
                slab_block_size = turboquant_slab_block_size_or_none(
                    cache,
                    head_size=self.head_size,
                    bits=self.turboquant_kv_bits_key,
                )
                if slab_block_size is not None:
                    # Slab cache must be decoded before FIA; this view is only a
                    # shape-safe placeholder for paths that immediately decode.
                    return slab_block_size, cache.view(
                        cache.shape[0], slab_block_size, -1
                    )
            num_block, block_size, _, _ = cache.shape
            return block_size, cache.view(num_block, block_size, -1)

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            block_size, key = _cache_view_for_fia(self.key_cache)  # type: ignore[arg-type]
            _, value = _cache_view_for_fia(self.value_cache)  # type: ignore[arg-type]
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            block_size, key = _cache_view_for_fia(self.key_cache)  # type: ignore[arg-type]
            _, value = _cache_view_for_fia(self.value_cache)  # type: ignore[arg-type]
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            block_size, key = _cache_view_for_fia(self.key_cache)  # type: ignore[arg-type]
            _, value = _cache_view_for_fia(self.value_cache)  # type: ignore[arg-type]
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            if self.kv_cache_dtype == "turboquant":
                assert self._decoded_key_cache is not None
                assert self._decoded_value_cache is not None
                key = self._decoded_key_cache.flatten(2, 3).contiguous()
                value = self._decoded_value_cache.flatten(2, 3).contiguous()
            else:
                key = self.key_cache.flatten(2, 3).contiguous()
                value = self.value_cache.flatten(2, 3).contiguous()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
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
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
        output[:batch_size] = attn_output[:batch_size]
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
            and self.sinks is None
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]

        # PrefillNoCache normally uses float key/value FIA (block_table=None).
        # When VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA is ON, use already-packed
        # paged KV via bit_residual_fia_paged_k8v4; fall back to float FIA if
        # the op returns None. Do not mutate _get_fia_params float-path defaults.
        if (
            self.kv_cache_dtype == "turboquant"
            and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and envs_ascend.VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA
            and self.turboquant_kv_bits_key == 8
            and self.turboquant_kv_bits_value == 4
            and self.head_size == 128
            and self.key_cache is not None
            and self.value_cache is not None
            and attn_metadata.block_tables is not None
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            batch_size = attn_metadata.seq_lens.shape[0]
            nocache_bt = attn_metadata.block_tables[:batch_size, :]
            if int(nocache_bt.numel()) > 0:
                nocache_block_size = self.vllm_config.cache_config.block_size
                attn_output = bit_residual_fia_paged_k8v4(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=nocache_bt,
                    actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    block_size=nocache_block_size,
                    scale=self.scale,
                    atten_mask=attn_metadata.attn_mask,
                    pre_tokens=SWA_INT_MAX,
                    next_tokens=SWA_INT_MAX,
                    sparse_mode=3,
                    out=output[:num_tokens],
                )
                if attn_output is not None:
                    return output

        if self.kv_cache_dtype == "turboquant" and block_table is not None:
            assert self.key_cache is not None and self.value_cache is not None

            # BitResidual k8v4:
            # - PrefillNoCache: optional paged FIA via NOCACHE_FIA (handled above)
            # - PrefillCacheHit / ChunkedPrefill: FIA when VLLM_ASCEND_BIT_RESIDUAL_FIA
            # - DecodeOnly: FIA when VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA (A/B vs vector)
            # - else / fallback: bit_residual_attention_paged_k8v4
            if (
                self.turboquant_kv_bits_key == 8
                and self.turboquant_kv_bits_value == 4
                and self.head_size == 128
            ):
                bt_numel = int(block_table.numel())
                fia_env = envs_ascend.VLLM_ASCEND_BIT_RESIDUAL_FIA
                decode_fia_env = envs_ascend.VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA
                state = attn_metadata.attn_state
                prefill_ok = state in (
                    AscendAttentionState.PrefillCacheHit,
                    AscendAttentionState.ChunkedPrefill,
                )
                decode_ok = state == AscendAttentionState.DecodeOnly
                use_br_fia = (
                    bt_numel > 0
                    and ((fia_env and prefill_ok) or (decode_fia_env and decode_ok))
                )
                if use_br_fia:
                    is_decode = (
                        attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                    )
                    attn_output = bit_residual_fia_paged_k8v4(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        block_tables=block_table,
                        actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                        head_size=self.head_size,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        block_size=block_size,
                        scale=self.scale,
                        atten_mask=None if is_decode else attn_metadata.attn_mask,
                        pre_tokens=SWA_INT_MAX,
                        next_tokens=SWA_INT_MAX,
                        sparse_mode=0 if is_decode else 3,
                        out=output[:num_tokens],
                    )
                    if attn_output is not None:
                        return output
                if attn_metadata.attn_state in (
                    AscendAttentionState.DecodeOnly,
                    AscendAttentionState.ChunkedPrefill,
                    AscendAttentionState.PrefillCacheHit,
                ):
                    attn_output = bit_residual_attention_paged_k8v4(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        block_tables=block_table,
                        actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                        head_size=self.head_size,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        block_size=block_size,
                        scale=self.scale,
                        out=output[:num_tokens],
                    )
                    if attn_output is not None:
                        return output
                    # k8v4 layout is incompatible with MSE/slab decode+FIA; fail
                    # fast instead of silently producing wrong attention.
                    raise RuntimeError(
                        "BitResidual k8v4 attention returned None for "
                        "turboquant_kv_bits=[8,4]; refusing silent fallback to "
                        "MSE/paged paths."
                    )

            slab_block_size = turboquant_slab_block_size_or_none(
                self.key_cache,
                head_size=self.head_size,
                bits=self.turboquant_kv_bits_key,
            )
            use_slab_cache = slab_block_size is not None
            # Scheme B kernel supports DecodeOnly and ChunkedPrefill (mixed
            # prefill/decode) with per-token causal KV bounds.
            if not use_slab_cache and attn_metadata.attn_state in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.ChunkedPrefill,
            ):
                attn_output = turboquant_attention_paged8bit(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=block_table,
                    actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    block_size=block_size,
                    scale=self.scale,
                    bits_key=self.turboquant_kv_bits_key,
                    bits_value=self.turboquant_kv_bits_value,
                )
                if attn_output is not None:
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
            if use_slab_cache and attn_metadata.attn_state in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.ChunkedPrefill,
            ):
                attn_output = turboquant_attention_paged4bit(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=block_table,
                    actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    block_size=block_size,
                    scale=self.scale,
                    bits_key=self.turboquant_kv_bits_key,
                    bits_value=self.turboquant_kv_bits_value,
                )
                if attn_output is not None:
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
            # Phase 1 / scheme A: decode packed 8-bit KV -> compact fp16 via the
            # fused paged decode op (opt-in), then fall through to the standard
            # FIA below. Returns None (and we keep the old paths) when disabled or
            # the op/conditions are not met.
            fast = None
            if not use_slab_cache:
                fast = _try_8bit_decode_paged(
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_table=block_table,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    head_size=self.head_size,
                    dtype=query.dtype,
                    bits_key=self.turboquant_kv_bits_key,
                    bits_value=self.turboquant_kv_bits_value,
                )
            if fast is not None:
                key_ws, value_ws, block_table = fast
                key = key_ws.flatten(2, 3).contiguous()
                value = value_ws.flatten(2, 3).contiguous()
            elif (
                self.sinks is None
                and self.turboquant_kv_bits_key == 8
                and self.turboquant_kv_bits_value == 8
                and self.head_size == 128
                and query.dtype == torch.float16
            ):
                attn_output = turboquant_fused_infer_attention_score_8bit(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=block_table,
                    atten_mask=attn_metadata.attn_mask,
                    actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    block_size=block_size,
                    scale=self.scale,
                    bits_key=self.turboquant_kv_bits_key,
                    bits_value=self.turboquant_kv_bits_value,
                )
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                key_dec, value_dec, block_table = turboquant_decode_kv_cache_compact(
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=block_table,
                    head_size=self.head_size,
                    dtype=query.dtype,
                    bits_key=self.turboquant_kv_bits_key,
                    bits_value=self.turboquant_kv_bits_value,
                    decode_only_arange_fast_path=(
                        attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                    ),
                )
                # FIA expects key/value shaped like [num_blocks, block_size, hidden]
                # in TND layout flattening; keep consistent with existing path.
                key = key_dec.flatten(2, 3).contiguous()
                value = value_dec.flatten(2, 3).contiguous()

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                atten_mask = attn_metadata.swa_mask
                sparse_mode = 4
            else:
                atten_mask = attn_metadata.attn_mask
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=atten_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
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

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        block_table = attn_metadata.block_tables
        key_cache = self.key_cache
        value_cache = self.value_cache
        if self.kv_cache_dtype == "turboquant":
            assert key_cache is not None and value_cache is not None
            # BitResidual k8v4 Decode: default vector paged attn; optional FIA A/B.
            if (
                self.turboquant_kv_bits_key == 8
                and self.turboquant_kv_bits_value == 4
                and self.head_size == 128
            ):
                block_size = self.vllm_config.cache_config.block_size
                decode_fia_env = envs_ascend.VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA
                if decode_fia_env:
                    attn_output = bit_residual_fia_paged_k8v4(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_tables=block_table,
                        actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                        actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                        head_size=self.head_size,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        block_size=block_size,
                        scale=self.scale,
                        atten_mask=None,
                        pre_tokens=SWA_INT_MAX,
                        next_tokens=SWA_INT_MAX,
                        sparse_mode=0,
                        out=output,
                    )
                    if attn_output is not None:
                        return output
                attn_output = bit_residual_attention_paged_k8v4(
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_tables=block_table,
                    actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    block_size=block_size,
                    scale=self.scale,
                    out=output,
                )
                if attn_output is not None:
                    return output
                raise RuntimeError(
                    "BitResidual k8v4 paged attention returned None for "
                    "turboquant_kv_bits=[8,4]; refusing silent fallback to "
                    "MSE decode + stock paged attention."
                )
            key_cache, value_cache, block_table = turboquant_decode_kv_cache_compact(
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_table,
                head_size=self.head_size,
                dtype=query.dtype,
                bits_key=self.turboquant_kv_bits_key,
                bits_value=self.turboquant_kv_bits_value,
            )
        torch_npu._npu_paged_attention(query=query,
                                       key_cache=key_cache,
                                       value_cache=value_cache,
                                       num_kv_heads=self.num_kv_heads,
                                       num_heads=self.num_heads,
                                       scale_value=self.scale,
                                       block_table=block_table,
                                       context_lens=attn_metadata.seq_lens,
                                       out=output)
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                if self.kv_cache_dtype == "turboquant":
                    try:
                        block_size = turboquant_slab_block_size_or_none(
                            self.key_cache,
                            head_size=self.head_size,
                            bits=self.turboquant_kv_bits_key,
                        )
                        if block_size is not None:
                            warm_up_turboquant_4bit_pack_op(
                                device=key.device,
                                dtype=key.dtype,
                                head_size=self.head_size,
                                num_kv_heads=self.num_kv_heads,
                                block_size=block_size,
                                bits_key=self.turboquant_kv_bits_key,
                                bits_value=self.turboquant_kv_bits_value,
                            )
                    except Exception:
                        pass
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            
            if self.kv_cache_dtype == "turboquant":
                if attn_metadata.num_actual_tokens > 0:
                    # slot_mapping is int32+contiguous from metadata build().
                    # K/V are contiguous after Attention.forward view(-1, H, D).
                    num_tok = attn_metadata.num_actual_tokens
                    if not encoder_decoder:
                        key_in = key if key.shape[0] == num_tok else key[:num_tok]
                        value_in = (
                            value if value.shape[0] == num_tok else value[:num_tok]
                        )
                        cache_slots = (
                            slots if slots.shape[0] == num_tok else slots[:num_tok]
                        )
                    else:
                        key_in = key
                        value_in = value
                        cache_slots = slots
                    turboquant_pack_kv_for_cache_to_cache(
                        key=key_in,
                        value=value_in,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        slot_mapping=cache_slots,
                        query_start_loc=attn_metadata.query_start_loc,
                        num_reqs=attn_metadata.num_reqs,
                        bits_key=self.turboquant_kv_bits_key,
                        bits_value=self.turboquant_kv_bits_value,
                    )
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: ``(key_cache, value_cache)`` each
                ``[num_blocks, block_size, num_kv_heads, slot_dim]``.
                For fp16/bf16, ``slot_dim`` is ``head_size`` and both tensors match.
                For TurboQuant uint8, ``slot_dim`` is packed bytes per vector; with
                mixed K/V bits, ``key_cache.shape[-1]`` (``P_k``) and
                ``value_cache.shape[-1]`` (``P_v``) may differ—quantize/pack uses each
                tensor's own last dim (see ``turboquant_pack_kv_for_cache``).
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)
        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        # All eager Ascend attention paths accept and return the caller-provided
        # output buffer. Avoid dispatching an in-place copy when that exact
        # buffer is returned; preserve the copy for any future allocating path.
        if attn_output is not output:
            output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        float_key, float_value = None, None
        if key is not None and value is not None:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                float_key, float_value = key, value
            key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
            query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        self._prepare_c8_scales(layer, query.device)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_c8_decode(query, attn_metadata, output, layer)
        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
        else:
            return self._forward_c8_fused_infer_attention(
                query,
                float_key if float_key is not None else key,
                float_value if float_value is not None else value,
                attn_metadata,
                output,
                layer,
            )

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_v_aq_offset = layer._c8_v_offset.to(torch.bfloat16).view(bnsd).contiguous()

        layer._c8_k_inv_scale_bf16 = (1.0 / layer._c8_k_scale).to(torch.bfloat16)
        layer._c8_k_offset_bf16 = layer._c8_k_offset.to(torch.bfloat16)
        layer._c8_v_inv_scale_bf16 = (1.0 / layer._c8_v_scale).to(torch.bfloat16)
        layer._c8_v_offset_bf16 = layer._c8_v_offset.to(torch.bfloat16)

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize to target_dtype."""
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]

        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        k_scale = layer._c8_k_scale.to(target_dtype)
        k_offset = layer._c8_k_offset.to(target_dtype)
        v_scale = layer._c8_v_scale.to(target_dtype)
        v_offset = layer._c8_v_offset.to(target_dtype)
        dense_k = (dense_k.to(target_dtype) - k_offset) * k_scale
        dense_v = (dense_v.to(target_dtype) - v_offset) * v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        self._prepare_c8_scales(layer, key.device)

        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale_bf16 + layer._c8_k_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale_bf16 + layer._c8_v_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        key = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        value = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
            kv_v = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale,
                key_antiquant_offset=layer._c8_k_aq_offset,
                value_antiquant_scale=layer._c8_v_aq_scale,
                value_antiquant_offset=layer._c8_v_aq_offset,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self.key_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                paged_v = self.value_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        self._prepare_c8_scales(layer, query.device)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                qdt = query.dtype
                k_scale = layer._c8_k_scale.to(qdt)
                k_offset = layer._c8_k_offset.to(qdt)
                v_scale = layer._c8_v_scale.to(qdt)
                v_offset = layer._c8_v_offset.to(qdt)
                key = (key.to(qdt) - k_offset) * k_scale
                value = (value.to(qdt) - v_offset) * v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output
