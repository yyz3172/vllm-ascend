# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/attn_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.model import ModelDType
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, KVCacheConfig
from vllm.v1.worker.utils import extract_layer_index

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         AscendPrefillContextParallelMetadata)

_ATTENTION_MASK_BUILDER = None


def get_attn_mask_builder(device: torch.device):
    """Get attention mask builder which only have one instance."""
    global _ATTENTION_MASK_BUILDER
    if _ATTENTION_MASK_BUILDER is None:
        _ATTENTION_MASK_BUILDER = AttentionMaskBuilder(device)
    return _ATTENTION_MASK_BUILDER


def build_attn_metadata(
    attn_metadata_builders: list[AttentionMetadataBuilder],
    num_reqs: int,
    num_tokens: int,
    req_ids: list[str] | None = None,
    kv_transfer_params_list: list[dict[str, Any] | None] | None = None,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_computed_tokens_cpu: torch.Tensor,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    decode_token_per_req: int,
    actual_seq_lengths_q: list[int],
    positions: torch.Tensor | None = None,
    attn_state: Any | None = None,
    graph_pad_size: int = -1,
    num_input_tokens: int = 0,
    prefill_context_parallel_metadata: AscendPrefillContextParallelMetadata
    | None = None,
) -> dict[str, Any]:
    """Build attention metadata for Ascend NPUs."""
    # TODO(Ronald1995): optimize AscendCommonAttentionMetadata.
    max_query_len = int(query_start_loc_cpu.max())
    max_seq_len = int(seq_lens_cpu.max())

    # DynamicKV: ChunkedPrefill 仅在最后一块压缩；PrefillNoCache 在 attention 内始终走压缩路径。
    dynamic_kv_is_last_chunk = False
    try:
        if attn_state == AscendAttentionState.ChunkedPrefill:
            q_lens = (query_start_loc_cpu[1:num_reqs + 1] -
                      query_start_loc_cpu[:num_reqs]).to(torch.int64)
            seq_l = seq_lens_cpu[:num_reqs].to(torch.int64)
            comp = num_computed_tokens_cpu[:num_reqs].to(torch.int64)
            dynamic_kv_is_last_chunk = bool(torch.all((comp + q_lens) >= seq_l).item())
    except Exception:
        dynamic_kv_is_last_chunk = False

    attn_metadata: dict[str, Any] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for i, kv_cache_spec in enumerate(kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu[:num_reqs],
            seq_lens=seq_lens[:num_reqs],
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            decode_token_per_req=decode_token_per_req,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            actual_seq_lengths_q=actual_seq_lengths_q,
            positions=positions,
            attn_state=attn_state,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
            prefill_context_parallel_metadata=prefill_context_parallel_metadata,
            dynamic_kv_is_last_chunk=dynamic_kv_is_last_chunk,
            max_seq_len=max_seq_len)

        attn_metadata_builder = attn_metadata_builders[i]
        metadata = attn_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,  # type: ignore
        )
        for layer_name in kv_cache_spec.layer_names:
            dynamic_kv_seq_lens_list: list[int] | None = None
            dynamic_kv_keep_indices_list: list[list[int]] | None = None
            if req_ids and kv_transfer_params_list and len(kv_transfer_params_list) == len(req_ids):
                try:
                    layer_idx = int(extract_layer_index(layer_name, num_attn_module=1))
                except Exception:
                    layer_idx = -1
                if layer_idx >= 0:
                    tmp: list[int] = []
                    idx_tmp: list[list[int]] = []
                    for ridx, kvp in enumerate(kv_transfer_params_list):
                        if not kvp:
                            tmp.append(-1)
                            idx_tmp.append([])
                            continue
                        dyn = kvp.get("dynamic_kv") or {}
                        per_layer = dyn.get("per_layer_kv_lens")
                        per_layer_indices = dyn.get("per_layer_keep_indices")
                        # PD+DynamicKV: per-layer kv_len must grow with decode tokens.
                        # decode_extra = num_computed_tokens - transferred_tokens.
                        try:
                            ncomp = int(num_computed_tokens_cpu[ridx])
                        except Exception:
                            ncomp = 0
                        transferred = dyn.get("transferred_tokens")
                        use_shared_len = isinstance(
                            transferred, int) and transferred > 0
                        if use_shared_len:
                            # Scheduler starts at `transferred - 1`; include
                            # current decode token.
                            decode_extra = max(0, ncomp - int(transferred) + 2)
                        else:
                            decode_extra = 0
                        # Prefer kv_len derived from indices when available.
                        # NOTE: offload+none mode sets per_layer_keep_indices to empty []
                        # for all layers (packed-prefix layout). Only use indices if non-empty.
                        if isinstance(per_layer_indices, list) and layer_idx < len(per_layer_indices):
                            try:
                                idxs = per_layer_indices[layer_idx]
                                if isinstance(idxs, list) and idxs:  # Check non-empty
                                    idx_tmp.append([int(x) for x in idxs])
                                    if use_shared_len:
                                        tmp.append(int(transferred) + decode_extra)
                                    else:
                                        tmp.append(len(idxs) + decode_extra)
                                    continue
                            except Exception:
                                idx_tmp.append([])
                                tmp.append(-1)
                                continue
                        if isinstance(per_layer, list) and layer_idx < len(per_layer):
                            try:
                                Li = int(per_layer[layer_idx])
                                if Li > 0:
                                    if use_shared_len:
                                        tmp.append(int(transferred) + decode_extra)
                                    else:
                                        tmp.append(Li + decode_extra)
                                else:
                                    tmp.append(-1)
                            except Exception:
                                tmp.append(-1)
                        else:
                            tmp.append(-1)
                        idx_tmp.append([])
                    if tmp and not all(v < 0 for v in tmp):
                        dynamic_kv_seq_lens_list = tmp
                    if idx_tmp and any(len(x) > 0 for x in idx_tmp):
                        dynamic_kv_keep_indices_list = idx_tmp

            # vLLM expects a metadata dict keyed by layer name. We create a
            # per-layer copy so attention backends can branch on layer_name.
            attn_metadata[layer_name] = replace(
                metadata,
                layer_name=layer_name,
                # Propagate request ids so attention backends can export
                # per-request DynamicKV results for PD transfer.
                req_ids=req_ids or [],
                dynamic_kv_seq_lens_list=dynamic_kv_seq_lens_list,
                dynamic_kv_keep_indices_list=dynamic_kv_keep_indices_list,
            )
    return attn_metadata


def build_attn_state(
    vllm_config: VllmConfig,
    seq_lens_np: np.ndarray,
    num_reqs,
    num_scheduled_tokens,
    num_valid_tokens,
):
    """Build attention state for npu's attention backend."""
    if vllm_config.model_config.runner_type == "pooling":
        if isinstance(
                vllm_config.kv_cache_config.kv_cache_groups[0].kv_cache_spec,
                EncoderOnlyAttentionSpec,
        ):
            attn_state = AscendAttentionState.PrefillNoCache
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
    elif np.array_equal(seq_lens_np[:num_reqs], num_scheduled_tokens):
        attn_state = AscendAttentionState.PrefillNoCache
    # We assume it is the decode stage, where prefill occurs
    # but only one token is not hit in cache.
    elif np.all(num_scheduled_tokens == 1):
        attn_state = AscendAttentionState.DecodeOnly
        if (vllm_config.speculative_config
                and vllm_config.speculative_config.method == 'mtp'):
            # SpecDecoding now supports seq_len=1 and seq_len=2
            # In Prefilling Decoding Disaggregation scenario, SpecDecoding
            # need to supports seq_len=1
            attn_state = AscendAttentionState.SpecDecoding
    # Speculative decoding.
    elif np.all(num_valid_tokens == 1):
        if (vllm_config.speculative_config
                and vllm_config.speculative_config.method == 'mtp'):
            attn_state = AscendAttentionState.SpecDecoding
        else:
            attn_state = AscendAttentionState.ChunkedPrefill
    # splitfuse
    elif vllm_config.scheduler_config.enable_chunked_prefill:
        attn_state = AscendAttentionState.ChunkedPrefill
    else:
        attn_state = AscendAttentionState.PrefillCacheHit
    return attn_state


def make_attention_mask(
    vllm_config: VllmConfig,
    attn_state: AscendAttentionState,
    dtype: ModelDType | torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """make attention mask for npu's attention backend."""
    attn_mask_builder = get_attn_mask_builder(device)
    # pcp situation.
    if attn_mask_builder is None:
        raise ValueError("Attn mask builder is None")
    # Pooling situation.
    if vllm_config.model_config.runner_type == "pooling":
        return attn_mask_builder.get_attn_mask(2048, torch.bool)

    # TODO(Ronald1995) cosidering pcp.
    if vllm_config.model_config.use_mla:
        # mla prefill
        if attn_state != AscendAttentionState.DecodeOnly:
            return attn_mask_builder.get_mla_mask(dtype)
    return attn_mask_builder.get_splitfuse_attn_mask()
