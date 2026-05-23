#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import threading
from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.attention.layer import Attention, MLAAttention
from vllm.attention.selector import get_attn_backend
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_dcp_group, get_dp_group,
                                             get_pcp_group, get_pp_group,
                                             get_tensor_model_parallel_rank,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (AttentionSpec, CrossAttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput,
                             SamplerOutput,
                             make_empty_encoder_model_runner_output)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.worker.gpu_model_runner import (AsyncGPUModelRunnerOutput,
                                             GPUModelRunner)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.attention_v1 import (
    _DYNKV_STATE,
    dynkv_fia_profile_enabled,
    dynkv_fia_profile_reset,
    dynkv_fia_profile_snapshot,
    dynkv_pa_profile_enabled,
    dynkv_pa_profile_reset,
    dynkv_pa_profile_snapshot,
)
from vllm_ascend.attention.dynamic_kv import (
    gather_kv_from_paged_cache_batched,
    save_validation_mask,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    dynkv_fill_all_graph_context_lens_bufs,
    dynkv_fill_graph_context_lens_buf,
    dynkv_pa_kv_tokens_avg_from_attn_metadata,
    using_paged_attention,
)
from vllm_ascend.worker.dynamic_kv_offload import (
    OffloadCaptureContext,
    install_qproj_hooks,
    run_offload_rewrite_and_build_updates,
)
# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    dynkv_graph_npu_profile_take_ms,
    dynkv_graph_replay_profile_reset,
    dynkv_graph_replay_profile_take_ms,
    set_draft_graph_params,
    set_graph_params,
    update_attn_dcp_pcp_params,
    update_attn_params,
    update_mla_attn_dcp_pcp_params,
    update_mla_attn_params,
)
# yapf: enable
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.utils import (AscendDeviceType, ProfileExecuteDuration,
                               enable_sp, get_ascend_device_type,
                               is_drafter_moe_model, is_moe_model,
                               lmhead_tp_enable, maybe_trans_nz,
                               set_weight_prefetch_method, vllm_version_is)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType, get_mc2_tokens_capacity, select_moe_comm_method,
    set_ascend_forward_context, set_mc2_mask, set_mc2_tokens_capacity)
if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

if get_ascend_device_type() == AscendDeviceType._310P:
    torch_npu.npu.set_compile_mode(jit_compile=False)

SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144


_DYNKV_MERGE_LIST_KEYS: tuple[str, ...] = (
    "per_layer_kv_lens",
    "per_layer_keep_indices",
    "per_layer_important_indices",
    "prefix_remote_block_ids",
)


def _merge_dynamic_kv_drain_with_execute(
    dk_drain: dict[str, Any] | None,
    dk_execute: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge ``dynamic_kv`` from drain vs execute_model without dropping offload fields.

    Execute (prefill rewrite) normally wins key conflicts. If execute carries an
    empty list for a merge-listed field while drain has a non-empty list (e.g.
    ``per_layer_kv_lens`` populated only in ``_DYNKV_STATE``), keep the drain value
    so TPOT metadata stays consistent with attention.
    """
    d = dict(dk_drain) if isinstance(dk_drain, dict) else {}
    e = dict(dk_execute) if isinstance(dk_execute, dict) else {}
    out: dict[str, Any] = {**d, **e}
    for k in _DYNKV_MERGE_LIST_KEYS:
        v_e = e.get(k)
        v_d = d.get(k)
        e_ok = isinstance(v_e, list) and len(v_e) > 0
        d_ok = isinstance(v_d, list) and len(v_d) > 0
        if d_ok and not e_ok:
            out[k] = v_d
    return out


def _merge_kv_xfer_updates_drain(
    existing: dict[str, dict[str, Any]] | None,
    drain: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge ``drain_kv_transfer_params_updates`` into execute_model updates.

    Worker offload attaches ``kv_transfer_params_updates`` in ``execute_model``;
    draining the KV connector in ``sample_tokens`` must not replace those entries
    with scheduler-side fallbacks for the same keys.
    """
    if not drain:
        return existing
    merged: dict[str, dict[str, Any]] = dict(existing or {})
    for rid, d_req in drain.items():
        if rid not in merged:
            merged[rid] = dict(d_req)
            continue
        e_req = dict(merged[rid])
        u_req = dict(d_req)
        out_req = {**u_req, **e_req}
        dk_e = e_req.get("dynamic_kv")
        dk_u = u_req.get("dynamic_kv")
        if isinstance(dk_e, dict) or isinstance(dk_u, dict):
            out_req["dynamic_kv"] = _merge_dynamic_kv_drain_with_execute(
                dk_u if isinstance(dk_u, dict) else None,
                dk_e if isinstance(dk_e, dict) else None,
            )
        merged[rid] = out_req
    return merged


def _dynkv_decode_build_per_layer_tmp_lens_and_jobs(
    *,
    dyn_layer_names: list[str],
    rid_list: list[str],
    kv_list: list[dict[str, Any]],
    n_r: int,
    input_batch: Any,
    block_size: int,
) -> tuple[list[list[int]], list[list[tuple[int, int, int]]]]:
    """Build per-layer ``tmp_lens`` (length ``n_r``) and slot remap jobs on TP rank0.

    Mirrors the per-request loop previously executed once per layer inside
    ``_prepare_inputs``; factored out so TP can ``broadcast_object`` once per
    KV group instead of once per layer.
    """
    all_tmp_lens: list[list[int]] = []
    slot_jobs_all: list[list[tuple[int, int, int]]] = []
    bs_dyn = int(block_size)
    for layer_name in dyn_layer_names:
        tmp_row = [-1] * n_r
        jobs: list[tuple[int, int, int]] = []
        try:
            layer_idx = int(
                extract_layer_index(layer_name, num_attn_module=1))
        except Exception:
            layer_idx = -1
        if layer_idx < 0:
            all_tmp_lens.append(tmp_row)
            slot_jobs_all.append(jobs)
            continue
        kv_len = len(kv_list)
        for req_idx in range(n_r):
            # FULL_DECODE_ONLY uniform_decode pads ``num_reqs`` to the next
            # captured size while ``kv_list`` only contains real requests; the
            # extra padded slots have no kv_transfer_params, so treat them as
            # uncompressed.
            if req_idx >= kv_len:
                tmp_row[req_idx] = -1
                continue
            kvp = kv_list[req_idx]
            if not kvp:
                tmp_row[req_idx] = -1
                continue
            dyn = kvp.get("dynamic_kv") or {}
            per_layer = dyn.get("per_layer_kv_lens")
            pii = dyn.get("per_layer_important_indices")
            try:
                npt = int(
                    input_batch.num_prompt_tokens[req_idx])
                ncomp = int(
                    input_batch.num_computed_tokens_cpu[req_idx])
            except Exception:
                npt, ncomp = 0, 0
            base_tokens = npt
            transferred = dyn.get("transferred_tokens")
            use_dyn_base = isinstance(transferred, int) and transferred > 0
            if use_dyn_base:
                base_tokens = int(transferred)
            # PD decode: scheduler initializes num_computed_tokens at
            # ``transferred_tokens - 1``. The ``+2`` accounts for that base and
            # includes the current decode query in context_lens.
            # Example: transferred=2560, first step ncomp=2559 -> decode_extra=1.
            decode_extra = max(0, ncomp - base_tokens + 2)
            if (isinstance(per_layer, list) and layer_idx < len(per_layer)):
                try:
                    Li = int(per_layer[layer_idx])
                except Exception:
                    Li = -1
                if Li > 0:
                    tmp_row[req_idx] = Li + decode_extra
                    if use_dyn_base and bs_dyn > 0:
                        jobs.append(
                            (req_idx, int(base_tokens), int(Li)))
                    try:
                        rid_here = (
                            rid_list[req_idx]
                            if req_idx < len(rid_list) else None)
                        if (
                            rid_here
                            and isinstance(pii, list)
                            and layer_idx < len(pii)
                        ):
                            idxs_layer = pii[layer_idx]
                            if isinstance(idxs_layer, list):
                                li_tot = int(Li) + int(decode_extra)
                                mcpu = torch.zeros(
                                    li_tot, dtype=torch.bool)
                                for t in idxs_layer:
                                    ti = int(t)
                                    if 0 <= ti < int(Li):
                                        mcpu[ti] = True
                                if decode_extra > 0 and li_tot > int(Li):
                                    mcpu[int(Li):li_tot] = True
                                save_validation_mask(
                                    str(rid_here),
                                    int(layer_idx),
                                    mcpu,
                                )
                    except Exception:
                        pass
                else:
                    tmp_row[req_idx] = -1
            else:
                tmp_row[req_idx] = -1
        all_tmp_lens.append(tmp_row)
        slot_jobs_all.append(jobs)
    return all_tmp_lens, slot_jobs_all


# Per-step accumulators for VLLM_DYNKV_PROFILE_FORWARD model breakdown (perf_counter).
_DYNKV_FWD_MODEL_PROFILE_HOOKS: list[Any] = []
_DYNKV_FWD_MODEL_ACC: dict[str, float] = {
    "embed": 0.0,
    "norm": 0.0,
    "attn": 0.0,
    "attn_op": 0.0,
    "mlp": 0.0,
}


def _dynkv_fwd_model_profile_enabled() -> bool:
    return os.environ.get("VLLM_DYNKV_PROFILE_FORWARD", "0") == "1"


def _dynkv_fwd_model_profile_reset() -> None:
    for k in _DYNKV_FWD_MODEL_ACC:
        _DYNKV_FWD_MODEL_ACC[k] = 0.0


def _dynkv_fwd_model_profile_snapshot() -> dict[str, float]:
    return dict(_DYNKV_FWD_MODEL_ACC)


def _dynkv_resolve_decoder_model(model: nn.Module) -> nn.Module | None:
    """Unwrap ACLGraphWrapper / LlamaForCausalLM to LlamaModel-like module."""
    m: nn.Module = model
    if hasattr(m, "runnable"):
        m = m.runnable
    if hasattr(m, "model"):
        return m.model
    return m


def _dynkv_fwd_model_profile_install(model: nn.Module) -> None:
    """Register forward hooks once (Llama/Mistral: embed, layers, norm)."""
    global _DYNKV_FWD_MODEL_PROFILE_HOOKS
    if _DYNKV_FWD_MODEL_PROFILE_HOOKS:
        return

    if isinstance(model, ACLGraphWrapper):
        model = model.unwrap()

    def _add_hooks(module: nn.Module, key: str) -> None:
        def pre(_m: nn.Module, _in: Any) -> None:
            _m._dynkv_prof_t0 = time.perf_counter()

        def post(_m: nn.Module, _in: Any, _out: Any) -> None:
            t0 = getattr(_m, "_dynkv_prof_t0", None)
            if t0 is not None:
                _DYNKV_FWD_MODEL_ACC[key] += (time.perf_counter() - t0) * 1000

        _DYNKV_FWD_MODEL_PROFILE_HOOKS.append(
            module.register_forward_pre_hook(pre))
        _DYNKV_FWD_MODEL_PROFILE_HOOKS.append(
            module.register_forward_hook(post))

    inner = _dynkv_resolve_decoder_model(model)
    if inner is None:
        return

    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and isinstance(embed, nn.Module):
        _add_hooks(embed, "embed")

    layers = getattr(inner, "layers", None)
    if layers is not None:
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                _add_hooks(self_attn, "attn")
                attn_op = getattr(self_attn, "attn", None)
                if attn_op is not None:
                    _add_hooks(attn_op, "attn_op")
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                _add_hooks(mlp, "mlp")

    norm = getattr(inner, "norm", None)
    if norm is not None and isinstance(norm, nn.Module):
        _add_hooks(norm, "norm")

    if not _DYNKV_FWD_MODEL_PROFILE_HOOKS:
        logger.warning(
            "[DynamicKV][forward_profile] model hooks not installed "
            "(unsupported model layout)",
        )


def _dynkv_log_forward_profile(
    *,
    pa_decode: bool,
    fia_prof: bool,
    pa_prof: bool,
    ctx_setup: float,
    kv_setup: float,
    dynkv_pre: float,
    t_fwd_model: float,
    mp: dict[str, float],
    t_fwd_dynkv_post: float,
    t_fwd_total: float,
    pa_kv_tokens: float,
) -> None:
    """Emit forward_profile line; PA graph omits always-zero FIA/eager-PA fields."""
    model_acl = float(mp.get("model_acl", 0.0))
    model_core = float(mp.get("model_core", t_fwd_model))
    graph_replay_wall = float(mp.get("model_graph_replay", 0.0))
    graph_npu_ms = float(mp.get("graph_npu_ms", 0.0))
    model_npu_ms = float(mp.get("model_npu_ms", 0.0))
    fwd_block_npu_ms = float(mp.get("fwd_block_npu_ms", 0.0))

    if pa_decode:
        logger.info(
            "[DynamicKV][forward_profile][pa] ctx_setup=%.2fms kv_setup=%.2fms "
            "dynkv_pre=%.2fms model_cpu=%.2fms model_acl=%.2fms "
            "graph_replay_wall=%.2fms graph_npu_ms=%.3f model_npu_ms=%.3f "
            "fwd_block_npu_ms=%.3f pa_kv_tokens_avg=%.1f dynkv_post=%.2fms "
            "profile_cpu_total=%.2fms",
            ctx_setup,
            kv_setup,
            dynkv_pre,
            t_fwd_model,
            model_acl,
            graph_replay_wall,
            graph_npu_ms,
            model_npu_ms,
            fwd_block_npu_ms,
            pa_kv_tokens,
            t_fwd_dynkv_post,
            t_fwd_total,
        )
        return

    if pa_prof:
        logger.info(
            "[DynamicKV][forward_profile] ctx_setup=%.2fms kv_setup=%.2fms "
            "dynkv_pre=%.2fms model=%.2fms model_acl=%.2fms model_core=%.2fms "
            "graph_replay_wall=%.2fms graph_npu_ms=%.3f model_npu_ms=%.3f "
            "fwd_block_npu_ms=%.3f model_embed=%.2fms model_norm=%.2fms "
            "model_attn=%.2fms model_attn_op=%.2fms%s%s model_mlp=%.2fms "
            "model_layer_rms=%.2fms model_sp_pcp=%.2fms dynkv_post=%.2fms "
            "profile_cpu_total=%.2fms",
            ctx_setup,
            kv_setup,
            dynkv_pre,
            t_fwd_model,
            model_acl,
            model_core,
            graph_replay_wall,
            graph_npu_ms,
            model_npu_ms,
            fwd_block_npu_ms,
            float(mp.get("embed", 0.0)),
            float(mp.get("norm", 0.0)),
            float(mp.get("attn", 0.0)),
            float(mp.get("attn_op", 0.0)),
            (
                " fia_ms_total=%.3f fia_kv_tokens_avg=%.1f"
                % (float(mp.get("fia_ms_total", 0.0)),
                   float(mp.get("fia_kv_tokens_avg", 0.0)))
                if fia_prof else ""
            ),
            (
                " pa_ms_total=%.3f pa_kv_tokens_avg=%.1f"
                % (float(mp.get("pa_ms_total", 0.0)), pa_kv_tokens)
                if pa_prof else ""
            ),
            float(mp.get("mlp", 0.0)),
            max(0.0, model_core - float(mp.get("attn", 0.0))
                  - float(mp.get("mlp", 0.0))),
            float(mp.get("model_sp_pcp", 0.0)),
            t_fwd_dynkv_post,
            t_fwd_total,
        )
        return

    if fia_prof:
        logger.info(
            "[DynamicKV][forward_profile] ctx_setup=%.2fms kv_setup=%.2fms "
            "dynkv_pre=%.2fms model=%.2fms model_acl=%.2fms model_core=%.2fms "
            "model_embed=%.2fms model_norm=%.2fms model_attn=%.2fms "
            "model_attn_op=%.2fms fia_ms_total=%.3f fia_kv_tokens_avg=%.1f "
            "model_mlp=%.2fms model_layer_rms=%.2fms model_sp_pcp=%.2fms "
            "dynkv_post=%.2fms profile_cpu_total=%.2fms",
            ctx_setup,
            kv_setup,
            dynkv_pre,
            t_fwd_model,
            model_acl,
            model_core,
            float(mp.get("embed", 0.0)),
            float(mp.get("norm", 0.0)),
            float(mp.get("attn", 0.0)),
            float(mp.get("attn_op", 0.0)),
            float(mp.get("fia_ms_total", 0.0)),
            float(mp.get("fia_kv_tokens_avg", 0.0)),
            float(mp.get("mlp", 0.0)),
            max(0.0, model_core - float(mp.get("attn", 0.0))
                  - float(mp.get("mlp", 0.0))),
            float(mp.get("model_sp_pcp", 0.0)),
            t_fwd_dynkv_post,
            t_fwd_total,
        )
        return

    logger.info(
        "[DynamicKV][forward_profile] ctx_setup=%.2fms kv_setup=%.2fms "
        "dynkv_pre=%.2fms model=%.2fms model_acl=%.2fms model_core=%.2fms "
        "model_embed=%.2fms model_norm=%.2fms model_attn=%.2fms "
        "model_attn_op=%.2fms model_mlp=%.2fms model_layer_rms=%.2fms "
        "model_sp_pcp=%.2fms dynkv_post=%.2fms profile_cpu_total=%.2fms",
        ctx_setup,
        kv_setup,
        dynkv_pre,
        t_fwd_model,
        model_acl,
        model_core,
        float(mp.get("embed", 0.0)),
        float(mp.get("norm", 0.0)),
        float(mp.get("attn", 0.0)),
        float(mp.get("attn_op", 0.0)),
        float(mp.get("mlp", 0.0)),
        max(0.0, model_core - float(mp.get("attn", 0.0))
              - float(mp.get("mlp", 0.0))),
        float(mp.get("model_sp_pcp", 0.0)),
        t_fwd_dynkv_post,
        t_fwd_total,
    )


@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(
        torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


def get_tp_context(drafter):
    return getattr(drafter, "tp_group_context", nullcontext())


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: dict[str, Any]
    positions: torch.Tensor


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers, which need to be reverted after
        # the following PR is merged:
        # https://github.com/vllm-project/vllm/pull/28988
        max_pcp_pad_tokens = vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group(
            ).rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        max_buffer_num_tokens = self.max_num_tokens
        if self.pcp_size * self.dcp_size > 1:
            max_buffer_num_tokens = (self.max_num_tokens +
                                     self.max_num_reqs * 2 * self.pcp_size)
            self.pcp_manager = PCPManager(
                self.pcp_size,
                self.pcp_rank,
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.use_async_scheduling,
                self.pin_memory,
            )
            # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
            self.input_ids = self._make_buffer(max_buffer_num_tokens,
                                               dtype=torch.int32)
            self.positions = self._make_buffer(max_buffer_num_tokens,
                                               dtype=torch.int64)
        self.sampler = AscendSampler()
        self.attn_state = None

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)
        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger
                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                raise RuntimeError(
                    "Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # DynamicKV PD decode: reusable [layer, token] slot scratch (scheme 2).
        self._dynkv_slot_stack_buf: Optional[torch.Tensor] = None
        self._dynkv_slot_stack_cap_L: int = 0
        self._dynkv_slot_stack_cap_n: int = 0
        # FULL-graph PA decode: [L, n_sm] parent tensor pinned at capture; each
        # layer's ``slot_mapping`` is a row view (see ``_build_dummy_attn_metadata``).
        self._dynkv_slot_workspace_bufs: dict[int, torch.Tensor] = {}
        self._dynkv_slot_workspace_layer_row: dict[int, dict[str, int]] = {}
        # Per-layer slot_mapping pinned during FULL ACL graph capture
        # ({num_tokens: {layer_name: Tensor}}). Required for all FULL-graph decode,
        # not only when DynamicKV is enabled.
        self._dynkv_graph_slot_bufs: dict[int, dict[str, torch.Tensor]] = {}
        # Per-layer context_lens pinned during FULL PA graph capture.
        self._dynkv_graph_context_lens_bufs: dict[int, dict[str, torch.Tensor]] = (
            {})
        self._dynkv_prepare_step_acc: Optional[dict[str, float]] = None
        # Set up Attention
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config,
                                  "index_topk")
        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            self.block_size,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None
            and self.model_config.is_mm_prefix_lm)

        self._set_up_drafter()

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs,
                        self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs,
                                self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.decode_threshold = 1 + (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        self.dynamic_eplb = self.ascend_config.dynamic_eplb or self.ascend_config.expert_map_record_path
        if self.dynamic_eplb:
            EPLBParamUtils.check_dynamic_eplb(self.ascend_config.dynamic_eplb)
            EPLBParamUtils.check_expert_map_record_path(
                self.ascend_config.expert_map_record_path)
            self.is_eplb_warmuped = False
            self.policy_type = self.ascend_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({
                "expert_map": None,
                "moe_load": None,
                "expert_maps": None
            })
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict,
                                            policy_type=self.policy_type,
                                            enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            ascend_config = get_ascend_config()
            self.eplb_updator = EplbUpdator(ascend_config, self.eplb_loader,
                                            self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len,
                              self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens
                if self.vllm_config.speculative_config else 0),
            cp_kv_cache_interleave_size=self.parallel_config.
            cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                  dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        self.long_seq_metadata = None

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: Optional[Union[NgramProposer, EagleProposer, MtpProposer,
                                     SuffixDecodingProposer]] = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                if self.speculative_config.method == "eagle3":
                    assert isinstance(self.drafter, EagleProposer)
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state)
                self.rejection_sampler = RejectionSampler(self.sampler)
            self.actual_seq_lengths_q = list(
                range(self.decode_token_per_req, self.max_num_tokens + 1,
                      self.decode_token_per_req))
        self.discard_request_indices = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method,
                                      self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and self.compilation_config.mode == CompilationMode.VLLM_COMPILE and not self.model_config.enforce_eager

    def _skip_all_reduce_across_dp_group(self, is_draft_model=False) -> bool:
        """
        Decide whether to skip the all-reduce across the data-parallel (DP) group.

        Skipping is applicable for all dense models and for moe models only on ranks
        that act as KV consumers. We skip the DP all-reduce when either:
        - Both the prefill and decode communication methods are MC2 (or FUSED_MC2), or
        - Decode requires MC2 and ascend_config.recompute_scheduler_enable is True.
        """
        # For dense models, since we don't actually need dp communication, we simply skip it.
        # This usually happens when main model is moe while eagle draft model is dense.
        is_context_moe_model = is_drafter_moe_model(self.vllm_config) if is_draft_model \
            else is_moe_model(self.vllm_config)
        if not is_context_moe_model:
            return True

        # Only applicable to MoE models on KV consumer ranks.
        if not self.is_kv_consumer:
            return False

        def needs_mc2(num_tokens: int) -> bool:
            return select_moe_comm_method(num_tokens, self.vllm_config) in {
                MoECommType.MC2, MoECommType.FUSED_MC2
            }

        # Determine whether decode must use MC2. Use max cudagraph capture size
        # if available, otherwise use the maximal uniform decode token count.
        if self.compilation_config.cudagraph_capture_sizes:
            potential_max_tokens = self.compilation_config.max_cudagraph_capture_size
        else:
            potential_max_tokens = self.max_num_reqs * self.uniform_decode_query_len
        decode_must_use_mc2 = needs_mc2(potential_max_tokens)

        # For prefill, use the scheduler's max_num_batched_tokens for a single
        # batch.
        prefill_must_use_mc2 = needs_mc2(
            self.vllm_config.scheduler_config.max_num_batched_tokens)

        # Skip all-reduce if decode requires MC2 and either prefill also
        # requires MC2 or recompute-based scheduler is enabled.
        return decode_must_use_mc2 and (
            prefill_must_use_mc2
            or self.ascend_config.recompute_scheduler_enable)

    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_mode: int = 0,
        is_draft_model: bool = False
    ) -> tuple[int, Optional[torch.Tensor], bool, int]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill, cudagraph_mode

        if self._skip_all_reduce_across_dp_group(is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] *
                                                    self.dp_size,
                                                    device="cpu",
                                                    dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, with_prefill, cudagraph_mode

        # Sync num_tokens, with_prefill across dp ranks
        num_tokens_tensor = torch.tensor([
            num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)
        ],
                                         dtype=torch.int32,
                                         device="cpu")

        flags_tensor = torch.tensor([int(with_prefill)],
                                    dtype=torch.int32,
                                    device="cpu")

        cudagraph_mode_tensor = torch.tensor([
            cudagraph_mode if i == self.dp_rank else 0
            for i in range(self.dp_size)
        ],
                                             dtype=torch.int32,
                                             device="cpu")

        packed_tensor = torch.cat(
            [num_tokens_tensor, flags_tensor, cudagraph_mode_tensor])
        # use cpu_group to avoid cpu synchronization issue.
        # it can be overlapped with main moell execution on npu.
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[:self.dp_size]
        synced_flags = packed_tensor[self.dp_size:self.dp_size + 1]
        cudagraph_mode_across_dp = packed_tensor[self.dp_size + 1:]
        max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
        global_with_prefill = bool(synced_flags[0])
        synced_cudagraph_mode = torch.min(cudagraph_mode_across_dp).item()

        # Create a tensor for num_tokens_after_padding
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill, synced_cudagraph_mode

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    @staticmethod
    def _is_dynamic_kv_enabled() -> bool:
        return bool(getattr(get_ascend_config(), "dynamic_kv_enabled", False))

    def _get_graph_slot_bufs_for_tokens(
        self,
        num_input_tokens: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        bufs = self._dynkv_graph_slot_bufs.get(int(num_input_tokens))
        if bufs is not None:
            return bufs
        for cap_n in sorted(self._dynkv_graph_slot_bufs.keys()):
            if cap_n >= int(num_input_tokens):
                return self._dynkv_graph_slot_bufs[cap_n]
        return None

    def _get_graph_context_lens_bufs_for_tokens(
        self,
        num_input_tokens: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        bufs = self._dynkv_graph_context_lens_bufs.get(int(num_input_tokens))
        if bufs is not None:
            return bufs
        for cap_n in sorted(self._dynkv_graph_context_lens_bufs.keys()):
            if cap_n >= int(num_input_tokens):
                return self._dynkv_graph_context_lens_bufs[cap_n]
        return None

    @staticmethod
    def _dynkv_prepare_profile_enabled() -> bool:
        return os.environ.get("VLLM_DYNKV_PROFILE_PREPARE", "0") == "1"

    def _dynkv_prepare_step_acc_reset(self) -> None:
        if self._dynkv_prepare_profile_enabled():
            self._dynkv_prepare_step_acc = {
                "update_states_ms": 0.0,
                "prepare_core_ms": 0.0,
                "prepare_attn_build_ms": 0.0,
            }
        else:
            self._dynkv_prepare_step_acc = None

    def _dynkv_prepare_step_acc_add(self, key: str, ms: float) -> None:
        acc = getattr(self, "_dynkv_prepare_step_acc", None)
        if isinstance(acc, dict):
            acc[key] = float(acc.get(key, 0.0)) + float(ms)

    def _dynkv_log_prepare_profile_ext(self) -> None:
        acc = getattr(self, "_dynkv_prepare_step_acc", None)
        if not isinstance(acc, dict):
            return
        logger.info(
            "[DynamicKV][prepare_profile_ext] update_states=%.2fms "
            "prepare_core=%.2fms prepare_attn_build=%.2fms",
            float(acc.get("update_states_ms", 0.0)),
            float(acc.get("prepare_core_ms", 0.0)),
            float(acc.get("prepare_attn_build_ms", 0.0)),
        )

    @staticmethod
    def _dynkv_log_prepare_profile_loop(
        *,
        layers: int,
        stack_init: float,
        kv_list_build: float,
        build_helper: float,
        broadcast: float,
        stacked_tensor: float,
        layer_ctx_fill_batch: float,
        layer_slot_assign: float,
        layer_copy_meta: float,
        layer_slot_remap: float,
        layer_meta_assign: float,
    ) -> None:
        total_loop = (layer_copy_meta + layer_slot_remap + layer_meta_assign
                      + layer_slot_assign)
        logger.info(
            "[DynamicKV][prepare_profile] layers=%d stack_init=%.2fms "
            "kv_list_build=%.2fms build_helper=%.2fms broadcast=%.2fms "
            "stacked_tensor=%.2fms layer_ctx_fill_batch=%.2fms "
            "layer_slot_assign=%.2fms layer_copy_meta=%.2fms "
            "layer_slot_remap=%.2fms layer_meta_assign=%.2fms "
            "layer_other=%.2fms total_loop=%.2fms",
            layers,
            stack_init,
            kv_list_build,
            build_helper,
            broadcast,
            stacked_tensor,
            layer_ctx_fill_batch,
            layer_slot_assign,
            layer_copy_meta,
            layer_slot_remap,
            layer_meta_assign,
            layer_meta_assign,
            total_loop,
        )

    def _register_dynkv_graph_context_lens_bufs_from_capture(
        self,
        num_input_tokens: int,
        layer_names: list[str],
    ) -> Optional[dict[str, torch.Tensor]]:
        """Map layer_name -> PA ``context_lens`` tensor pinned in ``attn_params``.

        Must use tensors from graph capture (``attn_params[i][7]``), not pre-capture
        metadata clones, so prepare writes the same addresses replay reads.
        """
        existing = self._get_graph_context_lens_bufs_for_tokens(num_input_tokens)
        if existing is not None:
            return existing
        from vllm_ascend.compilation.acl_graph import get_graph_params

        graph_params = get_graph_params()
        if graph_params is None:
            return None
        cap_key = int(num_input_tokens)
        params_list = graph_params.attn_params.get(cap_key)
        if params_list is None:
            for cap_n in sorted(graph_params.attn_params.keys()):
                if cap_n >= cap_key:
                    cap_key = cap_n
                    params_list = graph_params.attn_params[cap_n]
                    break
        if not params_list:
            return None
        n_layers = len(layer_names)
        if len(params_list) != n_layers:
            logger.debug(
                "[DynamicKV][decode] context_lens_bufs layer count mismatch: "
                "attn_params=%d layer_names=%d (cap_key=%d)",
                len(params_list),
                n_layers,
                cap_key,
            )
        bufs: dict[str, torch.Tensor] = {}
        for i, layer_name in enumerate(layer_names):
            if i >= len(params_list):
                break
            param = params_list[i]
            if len(param) > 7 and isinstance(param[7], torch.Tensor):
                bufs[str(layer_name)] = param[7]
        if not bufs:
            return None
        self._dynkv_graph_context_lens_bufs[cap_key] = bufs
        return bufs

    def _resolve_slot_workspace_cap_key(self, num_input_tokens: int) -> int:
        cap_key = int(num_input_tokens)
        if cap_key in self._dynkv_slot_workspace_bufs:
            return cap_key
        for cap_n in sorted(self._dynkv_slot_workspace_bufs.keys()):
            if cap_n >= cap_key:
                return cap_n
        return cap_key

    def _get_slot_workspace_for_tokens(
        self,
        num_input_tokens: int,
    ) -> Optional[torch.Tensor]:
        cap_key = self._resolve_slot_workspace_cap_key(num_input_tokens)
        return self._dynkv_slot_workspace_bufs.get(cap_key)

    def _get_slot_workspace_row_map(
        self,
        num_input_tokens: int,
    ) -> Optional[dict[str, int]]:
        cap_key = self._resolve_slot_workspace_cap_key(num_input_tokens)
        row_map = self._dynkv_slot_workspace_layer_row.get(cap_key)
        return row_map if row_map else None

    def _get_or_create_slot_workspace(
        self,
        cap_n: int,
        num_layers: int,
        n_sm: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cap_n = int(cap_n)
        ws = self._dynkv_slot_workspace_bufs.get(cap_n)
        need_L = max(int(num_layers), int(ws.shape[0]) if ws is not None else 0)
        need_n = max(int(n_sm), int(ws.shape[1]) if ws is not None else 0)
        if (
            ws is None
            or ws.shape[0] < need_L
            or ws.shape[1] < need_n
            or ws.device != device
            or ws.dtype != dtype
        ):
            ws = torch.empty((need_L, need_n), device=device, dtype=dtype)
            self._dynkv_slot_workspace_bufs[cap_n] = ws
        self._dynkv_slot_stack_buf = ws
        self._dynkv_slot_stack_cap_L = int(ws.shape[0])
        self._dynkv_slot_stack_cap_n = int(ws.shape[1])
        return ws

    @staticmethod
    def _layer_rows_contiguous(rows: list[int]) -> bool:
        if not rows:
            return False
        start = int(rows[0])
        return rows == list(range(start, start + len(rows)))

    def _resolve_dynkv_stack_from_workspace(
        self,
        num_input_tokens: int,
        layer_names: list[str],
        n_sm: int,
    ) -> tuple[Optional[torch.Tensor], bool]:
        """Return (stack_view, graph_rows_alias_workspace)."""
        ws = self._get_slot_workspace_for_tokens(num_input_tokens)
        row_map = self._get_slot_workspace_row_map(num_input_tokens)
        if ws is None or not row_map or n_sm <= 0 or not layer_names:
            return None, False
        rows = [row_map.get(str(ln)) for ln in layer_names]
        if any(r is None for r in rows):
            return None, False
        rows_int = [int(r) for r in rows]
        if not self._layer_rows_contiguous(rows_int):
            return None, False
        row0 = rows_int[0]
        n_layers = len(layer_names)
        if row0 + n_layers > int(ws.shape[0]) or n_sm > int(ws.shape[1]):
            return None, False
        stack = ws[row0:row0 + n_layers, :n_sm]
        graph_bufs = self._get_graph_slot_bufs_for_tokens(num_input_tokens)
        if graph_bufs is None:
            return stack, False
        alias = True
        for li, ln in enumerate(layer_names):
            buf = graph_bufs.get(str(ln))
            if buf is None:
                alias = False
                break
            n_buf = min(int(buf.numel()), n_sm)
            if n_buf <= 0:
                continue
            expected = ws[row0 + li, :n_buf]
            if buf.data_ptr() != expected.data_ptr():
                alias = False
                break
        return stack, alias

    def _broadcast_base_sm_to_workspace_layers(
        self,
        num_input_tokens: int,
        layer_names: list[str],
        base_sm: torch.Tensor,
        slot_n: int,
    ) -> bool:
        ws = self._get_slot_workspace_for_tokens(num_input_tokens)
        row_map = self._get_slot_workspace_row_map(num_input_tokens)
        if ws is None or not row_map or slot_n <= 0 or not layer_names:
            return False
        rows = [row_map.get(str(ln)) for ln in layer_names]
        if any(r is None for r in rows):
            return False
        rows_int = [int(r) for r in rows]
        n_sm = min(int(slot_n), int(base_sm.numel()))
        if n_sm <= 0:
            return False
        if self._layer_rows_contiguous(rows_int):
            row0 = rows_int[0]
            ws[row0:row0 + len(rows_int), :n_sm] = base_sm[:n_sm].unsqueeze(0)
        else:
            for row in rows_int:
                ws[row, :n_sm].copy_(base_sm[:n_sm])
        return True

    def _prepare_standard_layer_attn_metadata(
        self,
        base_meta: Any,
        layer_name: str,
        graph_slot_bufs: Optional[dict[str, torch.Tensor]],
        slot_n: int,
        profile_acc: Optional[dict[str, float]] = None,
        *,
        skip_slot_remap_copy: bool = False,
    ) -> Any:
        """Default per-layer metadata (vLLM-style); pins FULL-graph slot buffers."""
        _t0_cm = time.perf_counter() if profile_acc is not None else 0
        try:
            meta_i = copy(base_meta)
        except Exception:
            meta_i = base_meta
        try:
            setattr(meta_i, "layer_name", layer_name)
        except Exception:
            pass
        if profile_acc is not None:
            profile_acc["layer_copy_meta"] = profile_acc.get(
                "layer_copy_meta", 0.0) + (
                    time.perf_counter() - _t0_cm) * 1000
        if graph_slot_bufs is not None and slot_n > 0:
            _t0_sr = time.perf_counter() if profile_acc is not None else 0
            captured_sm = graph_slot_bufs.get(layer_name)
            if captured_sm is not None:
                meta_i.slot_mapping = captured_sm
                if not skip_slot_remap_copy:
                    try:
                        n_sm = min(int(captured_sm.numel()), slot_n)
                        if n_sm > 0:
                            meta_i.slot_mapping[:n_sm].copy_(
                                base_meta.slot_mapping[:n_sm])
                    except Exception:
                        pass
            if profile_acc is not None:
                profile_acc["layer_slot_remap"] = profile_acc.get(
                    "layer_slot_remap", 0.0) + (
                        time.perf_counter() - _t0_sr) * 1000
        return meta_i

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[dict[str, Any], torch.Tensor, np.ndarray, int, torch.Tensor,
               int, torch.Tensor, SpecDecodeMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], int, int, dict[
                   str, Any]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        _prep_acc = getattr(self, "_dynkv_prepare_step_acc", None)
        _t0_prepare_core = (time.perf_counter()
                            if isinstance(_prep_acc, dict) else 0)

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)

        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(tokens, dtype=np.int32)
        else:
            num_valid_tokens = np.array([
                num_tokens -
                len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(tokens, req_ids)
            ],
                                        dtype=np.int32)
        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)
        # for pcp, prefill mtp should use origin scheduleroutput ,
        if self.speculative_config and self.pcp_size * self.dcp_size > 1:
            self.pcp_manager.generate_pcp_mtp_input(
                num_reqs,
                total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens,
                with_prefill,
                self.input_batch,
                self.arange_np,
                req_indices,
                positions_np,
                cu_num_tokens,
                self._draft_token_ids,  # type: ignore[has-type]
                scheduler_output,
                self.num_spec_tokens)

        if self.pcp_size > 1:
            num_scheduled_tokens[:
                                 num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                                     num_scheduled_tokens[:num_reqs],
                                     self.arange_np,
                                     self.input_batch.num_reqs,
                                     self.reorder_batch_threshold,
                                 )
            # Re-update after PCP split sequences.
            total_num_scheduled_tokens = sum(num_scheduled_tokens)
            req_indices = np.repeat(self.arange_np[:num_reqs],
                                    num_scheduled_tokens)
            cu_num_tokens, _ = self._get_cumsum_and_arange(
                num_scheduled_tokens)
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        # Keep a raw (cache-space) position snapshot for token-id lookup.
        # Slot-mapping and token indexing must remain in compressed cache space.
        token_positions_np = positions_np.copy()

        # DynamicKV PD decode: RoPE positions should stay in original prompt space
        # while slot mapping/token lookup stays in compressed cache space.
        if (not self.uses_mrope and self.uses_xdrope_dim == 0
                and getattr(self, "is_kv_consumer", False)):
            try:
                ascend_cfg = get_ascend_config()
                if self._is_dynamic_kv_enabled():
                    offset_by_req_idx: dict[int, int] = {}
                    for req_idx in range(num_reqs):
                        req_id = req_ids[req_idx]
                        req = self.requests.get(req_id)
                        kvp = getattr(req, "kv_transfer_params",
                                      None) if req is not None else None
                        if not isinstance(kvp, dict):
                            continue
                        dyn = kvp.get("dynamic_kv") if isinstance(
                            kvp.get("dynamic_kv"), dict) else {}
                        # Backward-compatible: prefer explicit field, else derive.
                        offset = dyn.get("decode_position_offset")
                        if not isinstance(offset, int):
                            opl = dyn.get("original_prompt_len")
                            transferred = dyn.get("transferred_tokens")
                            if (isinstance(opl, int) and opl > 0
                                    and isinstance(transferred, int)
                                    and transferred > 0):
                                # first decode query position should be `opl`
                                # when raw position starts at transferred-1.
                                offset = int(opl) - int(transferred) + 1
                        if isinstance(offset, int) and offset != 0:
                            offset_by_req_idx[req_idx] = int(offset)
                    if offset_by_req_idx:
                        per_req_off = np.zeros(num_reqs, dtype=positions_np.dtype)
                        for req_idx, offset in offset_by_req_idx.items():
                            ri = int(req_idx)
                            if 0 <= ri < num_reqs:
                                per_req_off[ri] = int(offset)
                        # Vectorized: same as ``positions_np[req==r]+=off`` per r (scheme A).
                        positions_np += per_req_off[req_indices]
            except Exception as e:
                logger.warning(
                    "[DynamicKV][decode] RoPE offset apply failed: %s", e)
        max_num_scheduled_tokens = max(tokens)
        uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) \
            and (total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs)
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        # the following process is corresponding to _pad_for_sequence_parallelism
        # in gpu_model_runner
        if enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            num_input_tokens = total_num_scheduled_tokens
        cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_input_tokens,
            uniform_decode=uniform_decode,
            has_lora=has_lora,
        )
        num_input_tokens = batch_descriptor.num_tokens
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         synced_cudagraph_mode) = self._sync_metadata_across_dp(
             num_input_tokens, with_prefill, cudagraph_mode.value)
        self.with_prefill = with_prefill
        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (token_positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or
                                                   self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens.copy_to_gpu()

        self.seq_lens.gpu[num_reqs:].fill_(0)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens,
                                cu_num_tokens)
        self.positions.cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions.copy_to_gpu()
        attn_metadata: dict[str, Any] = {}

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        # DynamicKV: ChunkedPrefill only compresses on the last chunk.
        # Determine if this step finishes the prompt for all requests.
        dynamic_kv_is_last_chunk = False
        dynkv_max_capacity = None
        if self._is_dynamic_kv_enabled():
            try:
                if self.attn_state in (AscendAttentionState.PrefillNoCache,
                                       AscendAttentionState.ChunkedPrefill):
                    comp = self.input_batch.num_computed_tokens_cpu[:num_reqs]
                    sched = num_scheduled_tokens[:num_reqs]
                    total = num_tokens_np[:num_reqs]
                    # `dynamic_kv.prompt_kv_len_budget` is a per-layer KV budget target, not a hard cap on
                    # how many prompt tokens get chunked prefill. Using
                    # min(prompt_len, max_capacity) here marks the *first* chunk as
                    # "last" for long prompts and breaks DynamicKV + PD metadata.
                    try:
                        dynkv_cfg = None
                        add_cfg = getattr(self.vllm_config, "additional_config", None)
                        if isinstance(add_cfg, dict):
                            dynkv_cfg = add_cfg.get("dynamic_kv")
                        if isinstance(dynkv_cfg, dict):
                            mc = dynkv_cfg.get("prompt_kv_len_budget")
                            if isinstance(mc, int) and mc > 0:
                                dynkv_max_capacity = mc
                    except Exception:
                        dynkv_max_capacity = None
                    total_eff = total
                    dynamic_kv_is_last_chunk = bool(
                        np.all((comp + sched) >= total_eff))
            except Exception:
                dynamic_kv_is_last_chunk = False
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            original_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                np.array(tokens_original, dtype=np.int32))
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.is_multimodal_model and not self.model_config.is_encoder_decoder:
            self.multimodal_cpu_fields = ["grid_thw"]
            self._prepare_multimodal_fields()
            with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
            ):
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)

                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(
                    scheduler_output)

            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the acl graph all the time. The v0
            # engine avoids this by "double compiling" the acl graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the acl graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:total_num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        # Run the encoder, just like we do with other multimodal inputs.
        if self.model_config.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
            positions = self.positions.gpu[:total_num_scheduled_tokens]
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        # type: ignore
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            # If both flashcomm1 and pp are used simultaneously,
            # the shape of the received data and the shape of the space to be copied to will not match,
            # requiring a recalculation of the incoming data's shape.
            tp_size = get_tensor_model_parallel_world_size()
            num_input_tokens_with_flashcomm1 = num_input_tokens
            if enable_sp():
                num_input_tokens_with_flashcomm1 = (num_input_tokens +
                                                    tp_size - 1) // tp_size
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[
                    k][:num_input_tokens_with_flashcomm1].copy_(
                        v[:num_input_tokens_with_flashcomm1],
                        non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k:
                v[:num_input_tokens_with_flashcomm1]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            if self.pcp_size * self.dcp_size > 1:
                logits_indices = self.pcp_manager.get_logits_indices(
                    cu_num_tokens, num_reqs)
                logits_indices = logits_indices.pin_memory().to(
                    self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1:num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs]
                if self.pcp_size > 1 else None)
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:
                                            num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Used in the below loop.
        self.spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # DynamicKV scheme A (PD decode): after ``compute_slot_mapping``, CPU layout
        # for ``token_positions_np`` / ``req_indices`` is final; upload once per
        # ``_prepare_inputs`` for on-device slot remap (not per KV group / per layer).
        dynkv_decode_token_pos_t: Optional[torch.Tensor] = None
        dynkv_decode_req_idx_t: Optional[torch.Tensor] = None
        if getattr(self, "is_kv_consumer", False):
            try:
                _ac_dyn = get_ascend_config()
                if self._is_dynamic_kv_enabled():
                    dynkv_decode_token_pos_t = torch.as_tensor(
                        token_positions_np,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    dynkv_decode_req_idx_t = torch.as_tensor(
                        req_indices,
                        device=self.device,
                        dtype=torch.int64,
                    )
            except Exception:
                dynkv_decode_token_pos_t = None
                dynkv_decode_req_idx_t = None

        if isinstance(_prep_acc, dict):
            self._dynkv_prepare_step_acc_add(
                "prepare_core_ms",
                (time.perf_counter() - _t0_prepare_core) * 1000,
            )

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            encoder_seq_lens, encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                scheduler_output.num_scheduled_tokens or {},
                kv_cache_group_spec.kv_cache_spec,
                self.input_batch.num_reqs,
            )
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                maybe_pcp_full_tokens = (
                    num_input_tokens if self.pcp_size == 1 else
                    total_num_scheduled_tokens * self.pcp_size -
                    sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs]))
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping.gpu[:
                                                          maybe_pcp_full_tokens]
                if self.pcp_size == 1:
                    slot_mapping[
                        total_num_scheduled_tokens:num_input_tokens].fill_(-1)
            if self.pcp_size * self.dcp_size > 1:
                self.long_seq_metadata = self.pcp_manager.generate_pcp_metadata(
                    total_num_scheduled_tokens, self.query_lens,
                    self.input_batch, num_scheduled_tokens)
                blk_table.slot_mapping.gpu[maybe_pcp_full_tokens:].fill_(-1)
                if self.pcp_size > 1:
                    slot_mapping_pcp = self.pcp_manager.get_padded_slot_mapping(
                        total_num_scheduled_tokens,
                        slot_mapping,
                    )
                    blk_table.slot_mapping.gpu[:self.pcp_manager.
                                               num_actual_tokens_pcp_padded] = slot_mapping_pcp
                    slot_mapping = blk_table.slot_mapping.gpu[:self.
                                                              pcp_manager.
                                                              num_actual_tokens_pcp_padded]

            # NOTE: This is a temporary hack, now in GPUModelRunner, this prepare_inputs
            # has been split to multiple parts, and there are 3 parts that is related to this
            # `num_reqs`, we'll take `query_start_loc` as an example:
            # 1. self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            # 2. get `num_reqs_padded`, this depends on dispatcher and which is why we have the
            #    following simplified `dispatch` logic here, we try to minimize the impact
            # 3. query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) \
                and (total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs)

            # TODO: We should make this official ASAP. Also note that if we pad here,
            # the builders won’t need to add any extra padding.
            if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                uniform_decode and synced_cudagraph_mode == CUDAGraphMode.FULL.value:
                max_decode_tokens = min(
                    self.scheduler_config.max_num_seqs *
                    self.uniform_decode_query_len,
                    self.cudagraph_batch_sizes[-1])
                if self.uniform_decode_query_len <= num_input_tokens <= max_decode_tokens:
                    num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
                    pad_size = num_reqs_padded - num_reqs
                    if pad_size > 0:
                        last_query_loc = self.query_start_loc.np[num_reqs]

                        self.query_start_loc.np[
                            num_reqs + 1:num_reqs_padded + 1] = self.arange_np[
                                1:pad_size +
                                1] * self.uniform_decode_query_len + last_query_loc
                        self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)
                        self.seq_lens.np[num_reqs:].fill(0)
                        self.seq_lens.copy_to_gpu(num_reqs_padded)

                    # So we are trying to simulate the behavior of GPUModelRunner's
                    # prepare_inputs for uniform decode mode by padding query_start_loc
                    num_reqs = num_reqs_padded

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                seq_lens=self.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=slot_mapping,
                num_computed_tokens_cpu=self.input_batch.
                num_computed_tokens_cpu_tensor[:num_reqs],
                positions=self.positions.gpu,
                attn_state=self.attn_state,
                dynamic_kv_is_last_chunk=dynamic_kv_is_last_chunk,
                max_query_len=max_num_scheduled_tokens,
                decode_token_per_req=self.decode_token_per_req,
                prefill_context_parallel_metadata=self.long_seq_metadata,
                max_seq_len=0,
                encoder_seq_lens=encoder_seq_lens,
                encoder_seq_lens_cpu=encoder_seq_lens_cpu)
            # DynamicKV: carry request ids through to attention metadata.
            # AscendCommonAttentionMetadata may not accept this as a ctor kwarg
            # across versions, so attach it dynamically.
            try:
                setattr(common_attn_metadata, "req_ids", list(req_ids))
            except Exception:
                pass

            if self.speculative_config and self.pcp_size * self.dcp_size > 1:
                # For pcp + spec decode, we flatten block_table
                # to avoid irregular attn_mask shape, e.g.,
                # num_decode_req=2, num_prefill_req=3, num_speculative_tokens=1,
                # ori block_table: # [d0, d1, p0, p1, p2]
                # (num_reqs_d + num_reqs_p, max_num_blocks),
                # flattened block_table: [d0, d0, d1, d1, p0, p1, p2]
                # (num_reqs_d * decode_threshold + num_reqs_p, max_num_blocks),
                ori_query_lens_cpu = self.pcp_manager.query_lens_pcp_full.cpu[:
                                                                              num_reqs]
                ori_query_lens = self.pcp_manager.query_lens_pcp_full.gpu[:
                                                                          num_reqs]
                num_prefill_reqs = (ori_query_lens
                                    > self.decode_threshold).sum().item()
                num_decode_reqs = num_reqs - num_prefill_reqs
                num_decode_reqs_flatten = \
                    ori_query_lens_cpu[:num_decode_reqs].sum().item()
                blk_table_tensor[
                    num_decode_reqs_flatten:num_decode_reqs_flatten +
                    num_prefill_reqs].copy_(
                        blk_table_tensor[num_decode_reqs:num_decode_reqs +
                                         num_prefill_reqs].clone())
                blk_table_tensor[:num_decode_reqs_flatten].copy_(
                    blk_table_tensor[:num_decode_reqs].repeat_interleave(
                        ori_query_lens[:num_decode_reqs], dim=0))
                common_attn_metadata.block_table_tensor = \
                    blk_table_tensor[:num_decode_reqs_flatten + num_prefill_reqs]
                assert self.long_seq_metadata is not None
                self.long_seq_metadata.query_lens_pcp_full_cpu = ori_query_lens_cpu

                if 'pad_size' in locals() and pad_size > 0:
                    ori_query_lens_cpu[-pad_size:] = \
                        torch.full([pad_size], ori_query_lens_cpu[-pad_size - 1].item())
                self.long_seq_metadata.max_query_len_pcp_full = \
                    ori_query_lens_cpu.max().item()



            if self.speculative_config and \
                self.spec_decode_common_attn_metadata is None:
                self.spec_decode_common_attn_metadata = common_attn_metadata
                if num_reqs != base_num_reqs or total_num_scheduled_tokens != num_input_tokens:
                    self.spec_decode_common_attn_metadata = \
                        self.spec_decode_common_attn_metadata.unpadded(
                            total_num_scheduled_tokens, base_num_reqs)

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    if use_spec_decode:
                        patch_torch_npu_argsort()
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.
                            num_decode_draft_tokens.cpu[:num_reqs],
                        )
                _t0_attn_build = (time.perf_counter()
                                 if isinstance(_prep_acc, dict) else 0)
                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args)
                if isinstance(_prep_acc, dict):
                    self._dynkv_prepare_step_acc_add(
                        "prepare_attn_build_ms",
                        (time.perf_counter() - _t0_attn_build) * 1000,
                    )
                if self._is_dynamic_kv_enabled():
                    try:
                        setattr(attn_metadata_i, "dynamic_kv_is_last_chunk",
                                getattr(common_attn_metadata,
                                        "dynamic_kv_is_last_chunk", False))
                    except Exception:
                        pass
                try:
                    if getattr(attn_metadata_i, "req_ids", None) is None:
                        setattr(attn_metadata_i, "req_ids",
                                getattr(common_attn_metadata, "req_ids", None))
                except Exception:
                    pass

                _run_dynkv_decode_prepare = (
                    self._is_dynamic_kv_enabled()
                    and getattr(self, "is_kv_consumer", False))
                if not _run_dynkv_decode_prepare:
                    _dynkv_profile_std = (
                        os.environ.get("VLLM_DYNKV_PROFILE_PREPARE", "0")
                        == "1")
                    _graph_slot_bufs_std = self._get_graph_slot_bufs_for_tokens(
                        int(num_input_tokens))
                    try:
                        _slot_n_std = int(attn_metadata_i.slot_mapping.numel())
                    except Exception:
                        _slot_n_std = 0
                    _std_L = len(attn_group.layer_names)
                    _std_profile_acc: Optional[dict[str, float]] = None
                    if _dynkv_profile_std and _std_L > 0:
                        _std_profile_acc = {
                            "layer_copy_meta": 0.0,
                            "layer_slot_remap": 0.0,
                        }
                    _t0_sr_std = (time.perf_counter()
                                  if (_dynkv_profile_std and _std_L > 0)
                                  else 0)
                    _std_ws_broadcast = self._broadcast_base_sm_to_workspace_layers(
                        int(num_input_tokens),
                        list(attn_group.layer_names),
                        attn_metadata_i.slot_mapping,
                        _slot_n_std,
                    )
                    if (_dynkv_profile_std and _std_ws_broadcast and _std_L > 0
                            and _std_profile_acc is not None):
                        _std_profile_acc["layer_slot_remap"] = (
                            (time.perf_counter() - _t0_sr_std) * 1000)
                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = (
                            self._prepare_standard_layer_attn_metadata(
                                attn_metadata_i,
                                layer_name,
                                _graph_slot_bufs_std,
                                _slot_n_std,
                                profile_acc=_std_profile_acc,
                                skip_slot_remap_copy=_std_ws_broadcast,
                            ))
                    if _dynkv_profile_std and _std_L > 0:
                        _t_cm = float(
                            _std_profile_acc.get("layer_copy_meta", 0.0)
                            if _std_profile_acc else 0.0)
                        _t_sr = float(
                            _std_profile_acc.get("layer_slot_remap", 0.0)
                            if _std_profile_acc else 0.0)
                        self._dynkv_log_prepare_profile_loop(
                            layers=_std_L,
                            stack_init=0.0,
                            kv_list_build=0.0,
                            build_helper=0.0,
                            broadcast=0.0,
                            stacked_tensor=0.0,
                            layer_ctx_fill_batch=0.0,
                            layer_slot_assign=0.0,
                            layer_copy_meta=_t_cm,
                            layer_slot_remap=_t_sr,
                            layer_meta_assign=0.0,
                        )
                else:
                    # DynamicKV PD decode: compressed context_lens + slot_remap.
                    # Scheme 1: (req_idx, base_tokens) -> (mask, rel); rel does not depend
                    # on layer ``Li``, reused across ``layer_names`` in this attn_group.
                    dynkv_mask_rel_cache: dict[tuple[int, int], tuple[torch.Tensor,
                                                                      torch.Tensor]] = {}
                    _dynkv_layer_names = list(attn_group.layer_names)
                    _dynkv_L = len(_dynkv_layer_names)
                    self._register_dynkv_graph_context_lens_bufs_from_capture(
                        int(num_input_tokens), _dynkv_layer_names)
                    _dynkv_n = int(attn_metadata_i.slot_mapping.numel())

                    # Timing accumulators (only used when VLLM_DYNKV_PROFILE_PREPARE=1)
                    _dynkv_profile = os.environ.get("VLLM_DYNKV_PROFILE_PREPARE", "0") == "1"
                    _t_stack_init = 0.0
                    _t_kv_list_build = 0.0
                    _t_build_helper = 0.0
                    _t_broadcast = 0.0
                    _t_stacked_tensor = 0.0
                    _t_layer_copy_meta = 0.0
                    _t_layer_slot_remap = 0.0
                    _t_layer_ctx_fill_batch = 0.0
                    _t_layer_slot_assign = 0.0
                    _t_layer_meta_assign = 0.0

                    _t0_stack = time.perf_counter() if _dynkv_profile else 0
                    _dynkv_stack: Optional[torch.Tensor] = None
                    _slot_workspace_alias = False
                    _slot_n_sm = 0
                    try:
                        _slot_n_sm = int(attn_metadata_i.slot_mapping.numel())
                    except Exception:
                        _slot_n_sm = 0
                    _graph_slot_bufs = self._get_graph_slot_bufs_for_tokens(
                        int(num_input_tokens))
                    _graph_context_lens_bufs = (
                        self._get_graph_context_lens_bufs_for_tokens(
                            int(num_input_tokens)))
                    _use_graph_slot_bufs = (
                        _graph_slot_bufs is not None and _slot_n_sm > 0)
                    if _dynkv_L > 0 and _dynkv_n > 0:
                        _dynkv_stack, _slot_workspace_alias = (
                            self._resolve_dynkv_stack_from_workspace(
                                int(num_input_tokens),
                                _dynkv_layer_names,
                                min(_slot_n_sm, _dynkv_n),
                            ))
                        if _dynkv_stack is None:
                            _need_L = max(_dynkv_L,
                                          self._dynkv_slot_stack_cap_L)
                            _need_n = max(_dynkv_n,
                                          self._dynkv_slot_stack_cap_n)
                            _sb = self._dynkv_slot_stack_buf
                            if (
                                _sb is None
                                or _sb.shape[0] < _need_L
                                or _sb.shape[1] < _need_n
                                or _sb.device
                                != attn_metadata_i.slot_mapping.device
                                or _sb.dtype
                                != attn_metadata_i.slot_mapping.dtype
                            ):
                                self._dynkv_slot_stack_buf = torch.empty(
                                    (_need_L, _need_n),
                                    device=attn_metadata_i.slot_mapping.device,
                                    dtype=attn_metadata_i.slot_mapping.dtype,
                                )
                                self._dynkv_slot_stack_cap_L = _need_L
                                self._dynkv_slot_stack_cap_n = _need_n
                                _sb = self._dynkv_slot_stack_buf
                            _dynkv_stack = _sb[:_dynkv_L, :_dynkv_n]
                            _slot_workspace_alias = False
                    if _dynkv_profile:
                        _t_stack_init = (time.perf_counter() - _t0_stack) * 1000

                    # PD DynamicKV (decode): build all layers' ``tmp_lens`` / slot jobs on
                    # rank-0 once, ``broadcast_object`` once per KV group (not per layer),
                    # then upload ``dynamic_kv_seq_lens`` as a single [L, R] tensor.
                    all_tmp_lens: list[list[int]] | None = None
                    slot_jobs_all: list[list[tuple[int, int, int]]] | None = None
                    stacked_dyn_lens_t: Optional[torch.Tensor] = None
                    if _dynkv_L > 0:
                        _t0_kvlist = time.perf_counter() if _dynkv_profile else 0
                        n_r_dyn = int(num_reqs)
                        rid_list_dyn = list(req_ids[:n_r_dyn])
                        kv_list_dyn: list[dict[str, Any]] = [
                            (lambda r: r if isinstance(r, dict) else {})(
                                getattr(self.requests.get(rid), "kv_transfer_params", None)
                            )
                            for rid in rid_list_dyn
                        ]
                        if _dynkv_profile:
                            _t_kv_list_build = (time.perf_counter() - _t0_kvlist) * 1000
                        if kv_list_dyn and len(kv_list_dyn) == len(rid_list_dyn):
                            tg_pre = get_tp_group()
                            built_dyn: tuple[
                                list[list[int]],
                                list[list[tuple[int, int, int]]],
                            ] | None = None
                            if tg_pre.world_size > 1:
                                if get_tensor_model_parallel_rank() == 0:
                                    _t0_bh = time.perf_counter() if _dynkv_profile else 0
                                    built_dyn = (
                                        _dynkv_decode_build_per_layer_tmp_lens_and_jobs(
                                            dyn_layer_names=_dynkv_layer_names,
                                            rid_list=rid_list_dyn,
                                            kv_list=kv_list_dyn,
                                            n_r=n_r_dyn,
                                            input_batch=self.input_batch,
                                            block_size=int(self.block_size),
                                        ))
                                    if _dynkv_profile:
                                        _t_build_helper = (time.perf_counter() - _t0_bh) * 1000
                                _t0_bc = time.perf_counter() if _dynkv_profile else 0
                                built_dyn = tg_pre.broadcast_object(
                                    built_dyn
                                    if get_tensor_model_parallel_rank() == 0 else None,
                                    src=0,
                                )
                                if _dynkv_profile:
                                    _t_broadcast = (time.perf_counter() - _t0_bc) * 1000
                            else:
                                _t0_bh = time.perf_counter() if _dynkv_profile else 0
                                built_dyn = (
                                    _dynkv_decode_build_per_layer_tmp_lens_and_jobs(
                                        dyn_layer_names=_dynkv_layer_names,
                                        rid_list=rid_list_dyn,
                                        kv_list=kv_list_dyn,
                                        n_r=n_r_dyn,
                                        input_batch=self.input_batch,
                                        block_size=int(self.block_size),
                                    ))
                                if _dynkv_profile:
                                    _t_build_helper = (time.perf_counter() - _t0_bh) * 1000
                            if built_dyn is not None:
                                all_tmp_lens, slot_jobs_all = built_dyn
                        else:
                            all_tmp_lens = [[-1] * n_r_dyn
                                            for _ in range(_dynkv_L)]
                            slot_jobs_all = [[] for _ in range(_dynkv_L)]
                        if all_tmp_lens is not None:
                            try:
                                _sl0 = attn_metadata_i.seq_lens
                                if isinstance(_sl0, torch.Tensor):
                                    _t0_st = time.perf_counter() if _dynkv_profile else 0
                                    stacked_dyn_lens_t = torch.tensor(
                                        all_tmp_lens,
                                        device=_sl0.device,
                                        dtype=_sl0.dtype,
                                    )
                                    if _dynkv_profile:
                                        _t_stacked_tensor = (time.perf_counter() - _t0_st) * 1000
                            except Exception:
                                stacked_dyn_lens_t = None

                    bs_dyn = int(self.block_size)

                    # Pre-build layer_name -> layer_idx map (used by ctx fill + remap)
                    _layer_idx_map: dict[str, int] = {}
                    for _ln in _dynkv_layer_names:
                        try:
                            _layer_idx_map[_ln] = int(
                                extract_layer_index(_ln, num_attn_module=1))
                        except Exception:
                            _layer_idx_map[_ln] = -1

                    # ============================================================
                    # Batched slot_remap on ``_dynkv_stack`` [L, n_sm] (one 2D kernel
                    # per job). Graph capture buffers are filled in layer_slot_assign.
                    # ============================================================
                    _slot_remap_done = False
                    if (
                        all_tmp_lens is not None
                        and slot_jobs_all is not None
                        and _dynkv_stack is not None
                        and dynkv_decode_token_pos_t is not None
                        and dynkv_decode_req_idx_t is not None
                        and _dynkv_L > 0
                    ):
                        _t0_sr = time.perf_counter() if _dynkv_profile else 0
                        try:
                            base_sm = attn_metadata_i.slot_mapping
                            _slot_n_sm = int(base_sm.numel())
                            bt_dev = attn_metadata_i.block_tables
                            if (_slot_n_sm > 0
                                    and _slot_n_sm <= _dynkv_stack.shape[1]):
                                # 1. Broadcast base_sm to all layers at once
                                _dynkv_stack[:_dynkv_L, :_slot_n_sm] = (
                                    base_sm.unsqueeze(0))

                                # 2. Collect per-(req_idx, base_tokens) -> Li per layer
                                # slot_jobs_all[li] = [(req_idx, base_tokens, Li), ...]
                                # Group by (req_idx, base_tokens), collect Li for each layer
                                job_key_to_Li_per_layer: dict[
                                    tuple[int, int], list[int]
                                ] = defaultdict(lambda: [-1] * _dynkv_L)
                                for li in range(_dynkv_L):
                                    for (job_req_idx, job_base_tokens, job_Li) in slot_jobs_all[li]:
                                        key = (int(job_req_idx), int(job_base_tokens))
                                        job_key_to_Li_per_layer[key][li] = int(job_Li)

                                # 2.5 Batch build all Li tensors in ONE CPU->NPU transfer
                                job_keys_list = list(job_key_to_Li_per_layer.keys())
                                n_jobs = len(job_keys_list)
                                if n_jobs > 0:
                                    all_Li_lists = [job_key_to_Li_per_layer[k] for k in job_keys_list]
                                    # Single CPU->NPU transfer for all requests
                                    _Li_device = dynkv_decode_token_pos_t.device
                                    _Li_dtype = dynkv_decode_token_pos_t.dtype
                                    all_Li_tensor = torch.tensor(
                                        all_Li_lists, device=_Li_device, dtype=_Li_dtype
                                    )  # [N_jobs, L]

                                # 3. For each (req_idx, base_tokens), batch compute new_slots
                                for _job_i, (job_req_idx, job_base_tokens) in enumerate(job_keys_list):
                                    _ck = (job_req_idx, job_base_tokens)
                                    _cached = dynkv_mask_rel_cache.get(_ck)
                                    if _cached is None:
                                        mask = (dynkv_decode_req_idx_t == job_req_idx)
                                        # token_pos starts at transferred-1 on first decode;
                                        # subtract (base-1) so rel=0 maps to slot Li.
                                        rel = (
                                            dynkv_decode_token_pos_t[mask]
                                            - (job_base_tokens - 1)
                                        )
                                        dynkv_mask_rel_cache[_ck] = (mask, rel)
                                    else:
                                        mask, rel = _cached

                                    n_masked = int(rel.numel())
                                    if n_masked == 0:
                                        continue

                                    # Li_tensor: use pre-built tensor slice (no CPU->NPU here)
                                    Li_tensor = all_Li_tensor[_job_i].unsqueeze(1)  # [L, 1]

                                    # tgt_pos_2d: [L, n_masked]
                                    tgt_pos_2d = Li_tensor + rel.unsqueeze(0)  # broadcast

                                    # For layers where Li == -1, we skip (handled by original base_sm)
                                    valid_layer_mask = (Li_tensor.squeeze(1) >= 0)  # [L,]

                                    bt_row = bt_dev[job_req_idx]  # [max_blocks,]
                                    idx_2d = tgt_pos_2d // bs_dyn  # [L, n_masked]
                                    idx_2d = idx_2d.clamp(min=0, max=bt_row.shape[0] - 1)
                                    block_ids_2d = bt_row[idx_2d].to(torch.int64)  # [L, n_masked]
                                    new_slots_2d = (
                                        block_ids_2d * bs_dyn + (tgt_pos_2d % bs_dyn)
                                    ).to(base_sm.dtype)  # [L, n_masked]

                                    base_masked = base_sm[mask]
                                    valid_layer_mask_2d = valid_layer_mask.unsqueeze(1)
                                    final_vals = torch.where(
                                        valid_layer_mask_2d, new_slots_2d, base_masked
                                    )
                                    stack_view = _dynkv_stack[:_dynkv_L, :_slot_n_sm]
                                    stack_view[:, mask] = final_vals

                                _slot_remap_done = True
                                logger.debug(
                                    "[DynamicKV][decode] batched slot_remap for %d layers, %d job_keys",
                                    _dynkv_L, len(job_key_to_Li_per_layer),
                                )
                        except Exception:
                            _slot_remap_done = False
                        if _dynkv_profile:
                            _t_layer_slot_remap = (time.perf_counter() - _t0_sr) * 1000

                    if (
                        _slot_workspace_alias
                        and not _slot_remap_done
                        and _slot_n_sm > 0
                    ):
                        self._broadcast_base_sm_to_workspace_layers(
                            int(num_input_tokens),
                            _dynkv_layer_names,
                            attn_metadata_i.slot_mapping,
                            _slot_n_sm,
                        )

                    if _slot_n_sm == 0 and _dynkv_L > 0:
                        try:
                            _slot_n_sm = int(attn_metadata_i.slot_mapping.numel())
                        except Exception:
                            _slot_n_sm = 0
                        _use_graph_slot_bufs = (
                            _graph_slot_bufs is not None and _slot_n_sm > 0)

                    if (
                        all_tmp_lens is not None
                        and _graph_context_lens_bufs is not None
                        and stacked_dyn_lens_t is not None
                        and isinstance(attn_metadata_i.seq_lens, torch.Tensor)
                    ):
                        _t0_ctx_batch = (
                            time.perf_counter() if _dynkv_profile else 0)
                        try:
                            dynkv_fill_all_graph_context_lens_bufs(
                                layer_names=_dynkv_layer_names,
                                context_lens_bufs=_graph_context_lens_bufs,
                                stacked_dyn_lens_t=stacked_dyn_lens_t,
                                seq_lens=attn_metadata_i.seq_lens,
                                all_tmp_lens=all_tmp_lens,
                                layer_idx_map=_layer_idx_map,
                            )
                        except Exception:
                            pass
                        if _dynkv_profile:
                            _t_layer_ctx_fill_batch = (
                                time.perf_counter() - _t0_ctx_batch) * 1000
                    if (
                        _slot_remap_done
                        and _use_graph_slot_bufs
                        and _dynkv_stack is not None
                        and _slot_n_sm > 0
                        and not _slot_workspace_alias
                    ):
                        _t0_slot_assign = (
                            time.perf_counter() if _dynkv_profile else 0)
                        try:
                            for _dyn_li, layer_name in enumerate(
                                    _dynkv_layer_names):
                                _captured_sm = _graph_slot_bufs.get(layer_name)
                                if _captured_sm is None:
                                    continue
                                _n_sm = min(int(_captured_sm.numel()),
                                            _slot_n_sm)
                                if _n_sm > 0:
                                    _captured_sm[:_n_sm].copy_(
                                        _dynkv_stack[_dyn_li, :_n_sm])
                        except Exception:
                            pass
                        if _dynkv_profile:
                            _t_layer_slot_assign = (
                                time.perf_counter() - _t0_slot_assign) * 1000
                    for _dyn_li, layer_name in enumerate(_dynkv_layer_names):
                        # vLLM will index attn_metadata by layer_name. We must ensure
                        # each layer sees its own metadata instance with `layer_name`
                        # populated, otherwise DynamicKV cannot resolve layer_idx.
                        _t0_cm = time.perf_counter() if _dynkv_profile else 0
                        try:
                            meta_i = copy(attn_metadata_i)
                        except Exception:
                            meta_i = attn_metadata_i
                        # FULL graph replay uses slot_mapping addresses from capture.
                        # Reuse those buffers and copy runtime slots in-place each step.
                        if _use_graph_slot_bufs:
                            _captured_sm = _graph_slot_bufs.get(layer_name)
                            if _captured_sm is not None:
                                meta_i.slot_mapping = _captured_sm
                            else:
                                try:
                                    meta_i.slot_mapping = meta_i.slot_mapping.clone()
                                except Exception:
                                    pass
                        elif not _use_graph_slot_bufs:
                            # Eager (no FULL graph): shallow ``copy()`` shares one
                            # ``slot_mapping`` across layers; per-layer slot_remap
                            # must not overwrite the same buffer each iteration.
                            try:
                                meta_i.slot_mapping = meta_i.slot_mapping.clone()
                            except Exception:
                                pass
                        try:
                            setattr(meta_i, "layer_name", layer_name)
                        except Exception:
                            pass
                        if _dynkv_profile:
                            _t_layer_copy_meta += (time.perf_counter() - _t0_cm) * 1000
                        if all_tmp_lens is not None and slot_jobs_all is not None:
                            _t0_ma = time.perf_counter() if _dynkv_profile else 0
                            layer_idx = _layer_idx_map.get(layer_name, -1)
                            tmp_lens_layer = all_tmp_lens[_dyn_li]
                            slot_remap_jobs = slot_jobs_all[_dyn_li]
                            if (layer_idx >= 0 and tmp_lens_layer
                                    and not all(v < 0 for v in tmp_lens_layer)):
                                setattr(meta_i, "dynamic_kv_seq_lens_list",
                                        tmp_lens_layer)
                                try:
                                    _ctx_buf = None
                                    if _graph_context_lens_bufs is not None:
                                        _ctx_buf = _graph_context_lens_bufs.get(
                                            layer_name)
                                    if _ctx_buf is not None:
                                        setattr(meta_i,
                                                "dynamic_kv_seq_lens_tensor",
                                                _ctx_buf)
                                    elif stacked_dyn_lens_t is not None:
                                        setattr(
                                            meta_i,
                                            "dynamic_kv_seq_lens_tensor",
                                            stacked_dyn_lens_t[_dyn_li],
                                        )
                                    else:
                                        _sl_kv = meta_i.seq_lens
                                        if isinstance(_sl_kv, torch.Tensor):
                                            setattr(
                                                meta_i,
                                                "dynamic_kv_seq_lens_tensor",
                                                torch.tensor(
                                                    tmp_lens_layer,
                                                    device=_sl_kv.device,
                                                    dtype=_sl_kv.dtype,
                                                ),
                                            )
                                except Exception:
                                    try:
                                        delattr(meta_i,
                                                "dynamic_kv_seq_lens_tensor")
                                    except Exception:
                                        pass
                                if _dynkv_profile:
                                    _t_layer_meta_assign += (
                                        time.perf_counter() - _t0_ma) * 1000

                                elif (
                                    slot_remap_jobs
                                    and dynkv_decode_token_pos_t is not None
                                    and dynkv_decode_req_idx_t is not None
                                ):
                                    # Fallback: per-layer remap (should not happen if batched succeeded)
                                    _t0_sr_fb = time.perf_counter() if _dynkv_profile else 0
                                    try:
                                        base_sm = meta_i.slot_mapping
                                        n_sm = int(base_sm.numel())
                                        if (
                                            _dynkv_stack is not None
                                            and _dyn_li < _dynkv_stack.shape[0]
                                            and n_sm <= _dynkv_stack.shape[1]
                                        ):
                                            dyn_slot_t = _dynkv_stack[
                                                _dyn_li, :n_sm]
                                            dyn_slot_t.copy_(base_sm)
                                        else:
                                            dyn_slot_t = base_sm.clone()
                                        bt_dev = meta_i.block_tables
                                        for (job_req_idx, job_base_tokens,
                                             job_Li) in slot_remap_jobs:
                                            _ck = (int(job_req_idx),
                                                   int(job_base_tokens))
                                            _cached = dynkv_mask_rel_cache.get(
                                                _ck)
                                            if _cached is None:
                                                mask = (
                                                    dynkv_decode_req_idx_t
                                                    == job_req_idx)
                                                rel = (
                                                    dynkv_decode_token_pos_t[mask]
                                                    - (job_base_tokens - 1))
                                                dynkv_mask_rel_cache[_ck] = (
                                                    mask, rel)
                                            else:
                                                mask, rel = _cached
                                            tgt_pos = int(job_Li) + rel
                                            bt_row = bt_dev[job_req_idx]
                                            idx = tgt_pos // bs_dyn
                                            block_ids = bt_row[idx].to(
                                                torch.int64)
                                            new_slots = (
                                                block_ids * bs_dyn
                                                + (tgt_pos % bs_dyn))
                                            dyn_slot_t[mask] = new_slots.to(
                                                base_sm.dtype)
                                        if dyn_slot_t.data_ptr() != base_sm.data_ptr():
                                            base_sm.copy_(dyn_slot_t)
                                        if layer_idx == 0:
                                            logger.debug(
                                                "[DynamicKV][decode] fallback per-layer slot_mapping override"
                                            )
                                    except Exception:
                                        pass
                                    if _dynkv_profile:
                                        _t_layer_slot_remap += (time.perf_counter() - _t0_sr_fb) * 1000
                        if (_use_graph_slot_bufs and not _slot_remap_done
                                and _slot_n_sm > 0 and not _slot_workspace_alias):
                            try:
                                _sm = meta_i.slot_mapping
                                _n_sm = min(int(_sm.numel()), _slot_n_sm)
                                if _n_sm > 0:
                                    _sm[:_n_sm].copy_(
                                        attn_metadata_i.slot_mapping[:_n_sm])
                            except Exception:
                                pass
                        attn_metadata[layer_name] = meta_i
                    # Print timing summary once per step (only layer 0 triggers print)
                    if _dynkv_profile and _dynkv_L > 0:
                        self._dynkv_log_prepare_profile_loop(
                            layers=_dynkv_L,
                            stack_init=_t_stack_init,
                            kv_list_build=_t_kv_list_build,
                            build_helper=_t_build_helper,
                            broadcast=_t_broadcast,
                            stacked_tensor=_t_stacked_tensor,
                            layer_ctx_fill_batch=_t_layer_ctx_fill_batch,
                            layer_slot_assign=_t_layer_slot_assign,
                            layer_copy_meta=_t_layer_copy_meta,
                            layer_slot_remap=_t_layer_slot_remap,
                            layer_meta_assign=_t_layer_meta_assign,
                        )

        # update global cos, sin
        update_cos_sin(positions)

        self._dynkv_log_prepare_profile_ext()

        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens, synced_cudagraph_mode, model_kwargs)

    # all-gather one hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states(hidden_states):
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            hidden_states = hidden_states[:-pad_size, :]

        return hidden_states

    # all-gather a list of hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_list(hidden_states_list):
        return [
            NPUModelRunner._all_gather_hidden_states(hidden_states)
            for hidden_states in hidden_states_list
        ]

    # all-gather hidden-states in last layer with aux-hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_and_aux(hidden_states):
        if isinstance(hidden_states, tuple):
            return (NPUModelRunner._all_gather_hidden_states(hidden_states[0]),
                    NPUModelRunner._all_gather_hidden_states_list(
                        hidden_states[1]))
        return NPUModelRunner._all_gather_hidden_states(hidden_states)

    def _update_aclgraph_attn_params(self, runtime_num_tokens: int) -> None:
        """Refresh captured attention op args on ``update_stream`` before replay.

        Must run after ``_prepare_inputs`` (metadata for this step is ready)
        and before ``self.model()`` (ACL graph replay). Calling this after
        ``model()`` makes PA ``context_lens`` lag by one decode step, which
        breaks PD DynamicKV (compressed kv_len vs block-aligned seq_lens).
        """
        forward_context = get_forward_context()
        if (forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or self.use_sparse or not hasattr(self, "update_stream")):
            return
        if self.vllm_config.model_config.use_mla:
            if self.pcp_size * self.dcp_size > 1:
                update_mla_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               runtime_num_tokens)
            else:
                update_mla_attn_params(self.update_stream, forward_context,
                                       runtime_num_tokens,
                                       self.speculative_config)
        elif self.pcp_size * self.dcp_size > 1:
            update_attn_dcp_pcp_params(self.update_stream, forward_context,
                                       runtime_num_tokens)
        else:
            update_attn_params(self.update_stream, forward_context,
                               runtime_num_tokens, self.vllm_config)

    def _generate_process_reqs_hidden_states(self, maybe_padded_num_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds, model_kwargs):
        assert self.model is not None
        _prof = _dynkv_fwd_model_profile_enabled()
        _fia_prof = dynkv_fia_profile_enabled()
        _pa_prof = dynkv_pa_profile_enabled()
        if _prof:
            _dynkv_fwd_model_profile_install(self.model)
            _dynkv_fwd_model_profile_reset()
            dynkv_graph_replay_profile_reset()
            if _fia_prof:
                dynkv_fia_profile_reset()
            if _pa_prof:
                dynkv_pa_profile_reset()
            _t0_acl = time.perf_counter()
        self._update_aclgraph_attn_params(maybe_padded_num_tokens)
        if _prof:
            _t_model_acl = (time.perf_counter() - _t0_acl) * 1000
            _t0_core = time.perf_counter()
        _model_npu_ev0 = None
        if _pa_prof:
            _model_npu_ev0 = torch_npu.npu.Event(enable_timing=True)
            _model_npu_ev0.record()
        hidden_states = self.model(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds,
                                   **model_kwargs)
        if _prof:
            _t_model_core = (time.perf_counter() - _t0_core) * 1000
            _model_acc = _dynkv_fwd_model_profile_snapshot()
            _last_mp: dict[str, float] = {
                "model_acl": _t_model_acl,
                "model_core": _t_model_core,
                **_model_acc,
            }
            if _fia_prof:
                _last_mp.update(dynkv_fia_profile_snapshot())
            if _pa_prof:
                _last_mp.update(dynkv_pa_profile_snapshot())
            _last_mp["model_graph_replay"] = dynkv_graph_replay_profile_take_ms()
            _last_mp["graph_npu_ms"] = dynkv_graph_npu_profile_take_ms()
            if _pa_prof and _model_npu_ev0 is not None:
                _model_npu_ev1 = torch_npu.npu.Event(enable_timing=True)
                _model_npu_ev1.record()
                _model_npu_ev1.synchronize()
                _last_mp["model_npu_ms"] = float(
                    _model_npu_ev0.elapsed_time(_model_npu_ev1))
            self._dynkv_last_model_profile = _last_mp
        else:
            self._dynkv_last_model_profile = None

        if _prof:
            _t0_sp = time.perf_counter()
        if get_forward_context().sp_enabled and not isinstance(
                hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(
                hidden_states)
        out = hidden_states if self.pcp_size == 1 else self.pcp_manager.get_restore_hidden_states(
            hidden_states)
        if _prof and self._dynkv_last_model_profile is not None:
            self._dynkv_last_model_profile["model_sp_pcp"] = (
                time.perf_counter() - _t0_sp) * 1000
        return out

    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
                          num_valid_tokens):
        if np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config and self.speculative_config.method == 'mtp':
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
        return attn_state

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray | None,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(
                cu_num_scheduled_tokens - num_sampled_tokens,
                num_sampled_tokens)
            logits_indices_pcp += arange
            logits_indices_pcp = torch.from_numpy(
                logits_indices_pcp).pin_memory().to(self.device,
                                                    non_blocking=True)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = (
            torch.from_numpy(cu_num_draft_tokens).pin_memory().to(
                self.device, non_blocking=True))
        cu_num_sampled_tokens = (
            torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(
                self.device, non_blocking=True))
        logits_indices = (torch.from_numpy(logits_indices).pin_memory().to(
            self.device, non_blocking=True))
        target_logits_indices = (
            torch.from_numpy(target_logits_indices).pin_memory().to(
                self.device, non_blocking=True))
        bonus_logits_indices = torch.from_numpy(
            bonus_logits_indices).pin_memory().to(self.device,
                                                  non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        if self.pcp_size > 1:
            logits_indices = logits_indices_pcp
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    # TODO: Once the PCP features are complete, it will fully inherit the classes from the VLLM community.
    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: dict[str, Any],
        aux_hidden_states: torch.Tensor = None,
    ) -> Optional[list[list[int]]]:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        else:
            if self.speculative_config.method in ("suffix", "ngram"):
                draft_token_ids = self.drafter.generate_token_ids(
                    valid_sampled_token_ids, sampling_metadata,
                    scheduler_output, spec_decode_metadata, positions,
                    num_scheduled_tokens, hidden_states, aux_hidden_states)

            elif self.speculative_config.use_eagle():
                common_attn_metadata = self.spec_decode_common_attn_metadata
                sampled_token_ids = valid_sampled_token_ids

                if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                    # When padded-batch is disabled, the sampled_token_ids should be
                    # the cpu-side list[list[int]] of valid sampled tokens for each
                    # request, with invalid requests having empty lists.
                    assert isinstance(sampled_token_ids, list), \
                        "sampled_token_ids should be a python list when" \
                        "padded-batch is disabled."
                    assert self.drafter is not None
                    next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                        sampled_token_ids, self.requests, self.input_batch,
                        scheduler_output.num_scheduled_tokens)
                else:
                    # When using padded-batch, the sampled_token_ids should be
                    # the gpu tensor of sampled tokens for each request, of shape
                    # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                    # value -1.
                    assert isinstance(sampled_token_ids, torch.Tensor), \
                        "sampled_token_ids should be a torch.Tensor when" \
                        "padded-batch is enabled."
                    assert self.drafter is not None
                    next_token_ids, valid_sampled_tokens_count = \
                        self.drafter.prepare_next_token_ids_padded(
                            common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_indices.gpu,
                            self.num_discarded_requests
                        )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count)

                req_scheduled_tokens = scheduler_output.num_scheduled_tokens
                if self.pcp_size * self.dcp_size > 1:
                    long_seq_metadata = self.long_seq_metadata  # type: ignore
                    input_ids_pcp_full = self.pcp_manager.input_ids_pcp_full.gpu
                    query_start_loc_pcp_full = self.pcp_manager.query_start_loc_pcp_full.gpu
                    query_start_loc_pcp_full_cpu = self.pcp_manager.query_start_loc_pcp_full.cpu
                    num_reqs = self.input_batch.num_reqs
                    ori_query_lens = query_start_loc_pcp_full_cpu[1:num_reqs+1] - \
                        query_start_loc_pcp_full_cpu[:num_reqs]
                    num_prefill_reqs = (ori_query_lens
                                        > self.decode_threshold).sum().item()
                    num_decode_reqs = num_reqs - num_prefill_reqs
                else:
                    long_seq_metadata = None  # type: ignore
                    num_prefill_reqs = 0
                    num_decode_reqs = 0
                if spec_decode_metadata is None:
                    # update pcp related params
                    if self.pcp_size > 1:
                        token_indices_to_sample = \
                            query_start_loc_pcp_full[1:num_reqs + 1] - 1
                        target_token_ids = input_ids_pcp_full[:
                                                              num_scheduled_tokens]
                        target_positions = positions[:num_scheduled_tokens]
                        target_hidden_states = hidden_states
                    else:
                        token_indices_to_sample = None
                        # input_ids can be None for multimodal models.
                        target_token_ids = self.input_ids.gpu[:
                                                              num_scheduled_tokens]
                        target_positions = positions[:num_scheduled_tokens]
                        if self.use_aux_hidden_state_outputs:
                            target_hidden_states = torch.cat([
                                h[:num_scheduled_tokens]
                                for h in aux_hidden_states
                            ],
                                                             dim=-1)
                        else:
                            target_hidden_states = hidden_states[:
                                                                 num_scheduled_tokens]
                else:
                    if self.pcp_size > 1:
                        assert common_attn_metadata is not None
                        common_attn_metadata.query_start_loc_cpu[:num_reqs + 1] = \
                            query_start_loc_pcp_full_cpu[:num_reqs + 1]
                        assert common_attn_metadata is not None
                        common_attn_metadata.query_start_loc[:num_reqs + 1] = \
                            query_start_loc_pcp_full[:num_reqs + 1]
                    if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                        # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                        token_indices_to_sample = None
                        assert self.drafter is not None
                        common_attn_metadata, token_indices =\
                            self.drafter.prepare_inputs(
                                common_attn_metadata,
                                sampled_token_ids,
                                spec_decode_metadata.num_draft_tokens)
                    else:
                        assert self.drafter is not None
                        common_attn_metadata, token_indices, \
                            token_indices_to_sample =\
                                self.drafter.prepare_inputs_padded(
                                    common_attn_metadata,
                                    spec_decode_metadata,
                                    valid_sampled_tokens_count)
                    if self.pcp_size > 1:
                        target_token_ids = input_ids_pcp_full[token_indices]
                        target_positions = positions
                        target_hidden_states = hidden_states
                    else:
                        target_token_ids = self.input_ids.gpu[token_indices]
                        target_positions = positions[token_indices]
                        if self.use_aux_hidden_state_outputs:
                            target_hidden_states = torch.cat(
                                [h[token_indices] for h in aux_hidden_states],
                                dim=-1)
                        else:
                            target_hidden_states = hidden_states[token_indices]
                assert self.drafter is not None
                draft_token_ids = self.drafter._propose(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                    last_token_indices=token_indices_to_sample,
                    common_attn_metadata=common_attn_metadata,
                    sampling_metadata=sampling_metadata,
                    req_scheduled_tokens=req_scheduled_tokens,
                    long_seq_metadata=long_seq_metadata,
                    num_prefill_reqs=num_prefill_reqs,
                    num_decode_reqs=num_decode_reqs,
                    scheduler_output=scheduler_output,
                    num_scheduled_tokens=num_scheduled_tokens,
                )

            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")

        return draft_token_ids

    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors] | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")

        with ProfileExecuteDuration().capture_async("prepare input"):
            self._dynkv_prepare_step_acc_reset()
            _prep_acc = getattr(self, "_dynkv_prepare_step_acc", None)
            _t0_update_states = (time.perf_counter()
                                 if isinstance(_prep_acc, dict) else 0)
            self._update_states(scheduler_output)
            if isinstance(_prep_acc, dict):
                self._dynkv_prepare_step_acc["update_states_ms"] = (
                    time.perf_counter() - _t0_update_states) * 1000
            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                ):
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors, max_query_len, synced_cudagraph_mode,
             model_kwargs) = (self._prepare_inputs(scheduler_output,
                                                   intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(
                getattr(dbg_cfg, "level",
                        "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora,
                                               disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value)
        num_input_tokens = batch_descriptor.num_tokens

        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators)

        dynkv_updates: dict[str, dict[str, Any]] = {}

        # Run forward pass
        _fwd_profile = os.environ.get("VLLM_DYNKV_PROFILE_FORWARD", "0") == "1"
        _fia_prof = dynkv_fia_profile_enabled()
        _pa_prof = dynkv_pa_profile_enabled()
        _pa_decode = using_paged_attention(num_input_tokens, self.vllm_config)
        _fwd_npu_ev0 = None
        _t0_fwd_total = time.perf_counter() if _fwd_profile else 0
        _t_fwd_ctx_setup = 0.0
        _t_fwd_kv_setup = 0.0
        _t_fwd_dynkv_pre = 0.0
        _t_fwd_model = 0.0
        _t_fwd_dynkv_post = 0.0

        with ProfileExecuteDuration().capture_async("forward"):
            _t0_ctx = time.perf_counter() if _fwd_profile else 0
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    model_instance=self.model,
                    is_multimodal_model=self.is_multimodal_model):
                if _fwd_profile:
                    _t_fwd_ctx_setup = (time.perf_counter() - _t0_ctx) * 1000
                    _t0_kv = time.perf_counter()
                if _pa_prof:
                    _fwd_npu_ev0 = torch_npu.npu.Event(enable_timing=True)
                    _fwd_npu_ev0.record()
                self.maybe_setup_kv_connector(scheduler_output)
                if _fwd_profile:
                    _t_fwd_kv_setup = (time.perf_counter() - _t0_kv) * 1000
                    _t0_dynkv_pre = time.perf_counter()

                # DynamicKV offload: publish capture context for q_proj hooks.
                # Only the KV producer (prefill) needs Q capture + post-prefill rewrite.
                # On the decode worker, ``should_enable()`` must stay false: otherwise hooks
                # keep writing ``offload_q_last`` every step while cleanup only runs on the
                # producer after ``run_offload_rewrite_and_build_updates``, leaking NPU
                # memory across finished requests (OOM after many requests).
                try:
                    ascend_cfg = get_ascend_config()
                    dyn_impl = str(getattr(ascend_cfg, "dynamic_kv_impl", "offload"))
                    if self._is_dynamic_kv_enabled() and dyn_impl == "offload" and getattr(
                            self, "is_kv_producer", False):
                        num_reqs = int(self.input_batch.num_reqs)
                        req_ids = list(self.input_batch.req_ids)
                        # query_start_loc is cumulative starts for each req in this step.
                        qsl = self.query_start_loc.np[:num_reqs + 1].astype(int).tolist()
                        q_start = qsl[:-1]
                        q_end = qsl[1:]
                        # Use prompt length for prefill completion; ``num_tokens`` can
                        # diverge once outputs/spec tokens are appended.
                        seq_lens: list[int] = []
                        for r in req_ids:
                            req = self.requests[r]
                            nprompt = int(getattr(req, "num_prompt_tokens", 0) or 0)
                            if nprompt <= 0:
                                nprompt = int(req.num_tokens)
                            seq_lens.append(nprompt)
                        comp = [int(x) for x in self.input_batch.num_computed_tokens_cpu[:num_reqs]]
                        nst = scheduler_output.num_scheduled_tokens
                        sched = [int(nst.get(r, 0)) if isinstance(nst, dict) else 0 for r in req_ids]
                        active_idx = [i for i in range(num_reqs) if seq_lens[i] > 0]
                        # Per-request completion: in a multi-request batch, some requests
                        # may finish prefill earlier than others. We must trigger the
                        # DynamicKV rewrite for each request when it completes.
                        finished_idx = [
                            i for i in active_idx if (comp[i] + sched[i]) >= seq_lens[i]
                        ]
                        _DYNKV_STATE["offload_ctx"] = OffloadCaptureContext(
                            req_ids=req_ids,
                            q_start=q_start,
                            q_end=q_end,
                            seq_lens=seq_lens,
                            finished_idx=finished_idx,
                        )
                    elif self._is_dynamic_kv_enabled() and dyn_impl == "offload":
                        _DYNKV_STATE["offload_ctx"] = None
                except Exception:
                    _DYNKV_STATE["offload_ctx"] = None

                if _fwd_profile:
                    _t_fwd_dynkv_pre = (time.perf_counter() - _t0_dynkv_pre) * 1000
                    _t0_model = time.perf_counter()

                hidden_states = self._generate_process_reqs_hidden_states(
                    maybe_padded_num_tokens, input_ids, positions,
                    intermediate_tensors, inputs_embeds, model_kwargs)

                if _fwd_profile:
                    _t_fwd_model = (time.perf_counter() - _t0_model) * 1000
                    _t0_dynkv_post = time.perf_counter()

            # DynamicKV (offload impl): run a post-prefill KV rewrite pass in
            # the worker process. This keeps the attention execution path
            # unchanged while allowing PD to receive per-layer kv_lens/indices.
            _dynkv_offload_q_cleanup_rids: list[str] = []
            try:
                ascend_cfg = get_ascend_config()
                dyn_impl = str(getattr(ascend_cfg, "dynamic_kv_impl", "offload"))
                if self._is_dynamic_kv_enabled() and dyn_impl == "offload" and has_kv_transfer_group():
                    # Only meaningful on KV producer (prefill) side.
                    kv_cfg = getattr(self.vllm_config, "kv_transfer_config", None)
                    if kv_cfg is not None and getattr(kv_cfg, "is_kv_producer", False):
                        ctx = _DYNKV_STATE.get("offload_ctx")
                        kv_by_layer = getattr(self, "_kv_caches_by_layer_name", None)
                        if (isinstance(ctx, OffloadCaptureContext) and ctx.finished_idx
                                and isinstance(kv_by_layer, dict) and kv_by_layer):
                            # Block table is shared; some layers' metadata may omit it—scan.
                            block_tables = None
                            if isinstance(attn_metadata, dict) and attn_metadata:
                                for _m in attn_metadata.values():
                                    if _m is None:
                                        continue
                                    bt = getattr(_m, "block_tables", None)
                                    if bt is None:
                                        bt = getattr(_m, "block_table_tensor", None)
                                    if bt is not None:
                                        block_tables = bt
                                        break
                            if block_tables is not None:
                                model_cfg = self.vllm_config.model_config
                                hf_text_cfg = getattr(model_cfg, "hf_text_config", None)
                                hf_cfg = getattr(model_cfg, "hf_config", None)
                                num_layers = int(
                                    getattr(hf_text_cfg, "num_hidden_layers", None)
                                    or getattr(model_cfg, "num_hidden_layers", None)
                                    or getattr(hf_cfg, "num_hidden_layers", None)
                                    or 0
                                )
                                q_last_store = _DYNKV_STATE.get("offload_q_last") or {}
                                if not isinstance(q_last_store, dict):
                                    q_last_store = {}
                                done_idx = [int(i) for i in ctx.finished_idx if int(i) >= 0]
                                # Subselect to only finished requests.
                                idx_one = torch.tensor(
                                    done_idx,
                                    device=block_tables.device,
                                    dtype=torch.long,
                                )
                                block_tables_done = block_tables.index_select(0, idx_one)
                                req_ids_done = [ctx.req_ids[i] for i in done_idx]
                                _dynkv_offload_q_cleanup_rids = list(req_ids_done)
                                seq_lens_done = [ctx.seq_lens[i] for i in done_idx]
                                dynkv_updates = run_offload_rewrite_and_build_updates(
                                    req_ids=req_ids_done,
                                    seq_lens=seq_lens_done,
                                    block_tables=block_tables_done,
                                    kv_caches=kv_by_layer,
                                    q_last_store=q_last_store,
                                    max_capacity=int(getattr(ascend_cfg, "dynamic_kv_prompt_kv_len_budget", 0) or 0),
                                    window_size=int(getattr(ascend_cfg, "dynamic_kv_window_size", 16) or 16),
                                    pooling=str(getattr(ascend_cfg, "dynamic_kv_pooling", "none")),
                                    kernel_size=int(getattr(ascend_cfg, "dynamic_kv_kernel_size", 1) or 1),
                                    softmax_chunk_size=int(getattr(ascend_cfg, "dynamic_kv_softmax_chunk_size", 1024) or 1024),
                                    radio_max=float(getattr(ascend_cfg, "dynamic_kv_radio_max", 10.0)),
                                    radio_min=float(getattr(ascend_cfg, "dynamic_kv_radio_min", 0.1)),
                                    num_layers=num_layers,
                                    validation_mode=str(
                                        getattr(ascend_cfg, "dynamic_kv_validation_mode", "none")
                                    ),
                                    min_rewrite_delta=int(
                                        getattr(ascend_cfg, "dynamic_kv_min_rewrite_delta", 128)
                                    ),
                                )
                                # Attach block_table-ordered prefix physical blocks for PD shrink.
                                # NOTE: Do NOT use allocator-ordered `block_ids[:n]` to shrink:
                                # only `block_tables` represents logical prefix order.
                                try:
                                    bs = int(getattr(self.vllm_config.cache_config, "block_size", 0) or 0)
                                    if bs > 0 and isinstance(dynkv_updates, dict) and dynkv_updates:
                                        # Normalize block_tables to CPU list-of-lists.
                                        bt = block_tables_done
                                        if isinstance(bt, torch.Tensor):
                                            bt_cpu = bt.detach().to("cpu")
                                            bt_rows = bt_cpu.tolist()
                                        else:
                                            bt_rows = None
                                        if isinstance(bt_rows, list) and bt_rows:
                                            for ridx, rid in enumerate(req_ids_done):
                                                upd = dynkv_updates.get(rid)
                                                if not isinstance(upd, dict):
                                                    continue
                                                dyn = upd.get("dynamic_kv")
                                                if not isinstance(dyn, dict):
                                                    continue
                                                pl = dyn.get("per_layer_kv_lens")
                                                if not isinstance(pl, list) or not pl:
                                                    continue
                                                try:
                                                    lens_pos = [int(x) for x in pl if int(x) > 0]
                                                except Exception:
                                                    lens_pos = []
                                                if not lens_pos:
                                                    continue
                                                max_len = int(max(lens_pos))
                                                n_transfer = int(math.ceil(max_len / bs)) if max_len > 0 else 0
                                                if n_transfer <= 0:
                                                    continue
                                                if ridx >= len(bt_rows) or not isinstance(bt_rows[ridx], list):
                                                    continue
                                                row = [int(x) for x in bt_rows[ridx] if int(x) >= 0]
                                                prefix = row[:n_transfer]
                                                if not prefix:
                                                    continue
                                                dyn["prefix_remote_block_ids"] = prefix
                                except Exception:
                                    logger.info(
                                        "[DynamicKV][offload] attach prefix_remote_block_ids skipped",
                                        exc_info=True,
                                    )
            except Exception:
                logger.exception("[DynamicKV][offload] post-prefill rewrite failed")
            finally:
                # Clear capture context to avoid accidental reuse.
                _DYNKV_STATE["offload_ctx"] = None
                # Captured q_last tensors are only for rewrite; drop them so
                # offload_q_last does not retain NPU memory across requests.
                if _dynkv_offload_q_cleanup_rids:
                    _store = _DYNKV_STATE.get("offload_q_last")
                    if isinstance(_store, dict):
                        for _rid in _dynkv_offload_q_cleanup_rids:
                            _store.pop(_rid, None)

                if _pa_prof and _fwd_npu_ev0 is not None:
                    _fwd_npu_ev1 = torch_npu.npu.Event(enable_timing=True)
                    _fwd_npu_ev1.record()
                    _fwd_npu_ev1.synchronize()
                    _fwd_block_npu_ms = float(
                        _fwd_npu_ev0.elapsed_time(_fwd_npu_ev1))
                    _mp_fwd = getattr(self, "_dynkv_last_model_profile", None)
                    if isinstance(_mp_fwd, dict):
                        _mp_fwd["fwd_block_npu_ms"] = _fwd_block_npu_ms

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states

            if _fwd_profile:
                _t_fwd_dynkv_post = (time.perf_counter() - _t0_dynkv_post) * 1000
                _t_fwd_total = (time.perf_counter() - _t0_fwd_total) * 1000
                _mp = getattr(self, "_dynkv_last_model_profile", None)
                if isinstance(_mp, dict):
                    _pa_kv_tokens = float(_mp.get("pa_kv_tokens_avg", 0.0))
                    if _pa_decode:
                        _pa_kv_meta = dynkv_pa_kv_tokens_avg_from_attn_metadata(
                            attn_metadata)
                        if _pa_kv_meta is not None:
                            _pa_kv_tokens = float(_pa_kv_meta)
                    _dynkv_log_forward_profile(
                        pa_decode=_pa_decode,
                        fia_prof=_fia_prof,
                        pa_prof=_pa_prof,
                        ctx_setup=_t_fwd_ctx_setup,
                        kv_setup=_t_fwd_kv_setup,
                        dynkv_pre=_t_fwd_dynkv_pre,
                        t_fwd_model=_t_fwd_model,
                        mp=_mp,
                        t_fwd_dynkv_post=_t_fwd_dynkv_post,
                        t_fwd_total=_t_fwd_total,
                        pa_kv_tokens=_pa_kv_tokens,
                    )
                else:
                    logger.info(
                        "[DynamicKV][forward_profile] ctx_setup=%.2fms "
                        "kv_setup=%.2fms dynkv_pre=%.2fms model=%.2fms "
                        "dynkv_post=%.2fms total=%.2fms",
                        _t_fwd_ctx_setup,
                        _t_fwd_kv_setup,
                        _t_fwd_dynkv_pre,
                        _t_fwd_model,
                        _t_fwd_dynkv_post,
                        _t_fwd_total,
                    )

        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        if dynkv_updates:
            try:
                kv_connector_output.kv_transfer_params_updates = dynkv_updates
            except Exception:
                pass
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    if vllm_version_is('0.13.0'):
                        pool_output = self._pool(
                            hidden_states,
                            scheduler_output.total_num_scheduled_tokens,
                            num_scheduled_tokens_np)
                    else:
                        pool_output = self._pool(
                            hidden_states,
                            scheduler_output.total_num_scheduled_tokens,
                            num_scheduled_tokens_np, kv_connector_output)
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return pool_output
                # Sometimes, after the model is compiled through the AOT backend,
                # the model output may become a list containing only one Tensor object.
                if isinstance(hidden_states, list) and \
                        len(hidden_states) == 1 and \
                        isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
            )
            self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Same step as GPU path's ``_get_kv_connector_output`` finally: drain
        # worker DynamicKV (etc.) into the output *before* the scheduler runs
        # ``request_finished``. Ascend splits execute_model/sample_tokens and
        # does not use that context manager, so without this drain the scheduler
        # may miss ``per_layer_kv_lens`` and hit Mooncake's
        # ``no_compress_fallback`` (full ``prompt_len`` per layer).
        if kv_connector_output is not None and has_kv_transfer_group():
            try:
                kc = get_kv_transfer_group()
                if hasattr(kc, "drain_kv_transfer_params_updates"):
                    req_ids_set: set[str] = set()
                    if kv_connector_output.finished_sending:
                        req_ids_set.update(kv_connector_output.finished_sending)
                    if kv_connector_output.finished_recving:
                        req_ids_set.update(kv_connector_output.finished_recving)
                    if not req_ids_set and scheduler_output.finished_req_ids:
                        req_ids_set.update(scheduler_output.finished_req_ids)
                    if not req_ids_set:
                        req_ids_set.update(
                            scheduler_output.num_scheduled_tokens.keys())
                    upd = kc.drain_kv_transfer_params_updates(list(req_ids_set))
                    if isinstance(upd, dict) and upd:
                        prev = getattr(
                            kv_connector_output,
                            "kv_transfer_params_updates",
                            None,
                        )
                        kv_connector_output.kv_transfer_params_updates = (
                            _merge_kv_xfer_updates_drain(prev, upd))
            except Exception:
                pass

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with ProfileExecuteDuration().capture_async("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert self.spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with ProfileExecuteDuration().capture_async("Draft"):
            if self.speculative_config:
                use_padded_batch_for_eagle = self.speculative_config and \
                    self.speculative_config.use_eagle() and \
                    not self.speculative_config.disable_padded_drafter_batch
                if use_padded_batch_for_eagle:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch_for_eagle:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        extra_args = ({"kv_connector_output": kv_connector_output})

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            if self.debugger is not None:
                assert self.debugger is not None
                self.debugger.stop()
                self.debugger.step()
            return model_runner_output

        if self.debugger is not None:
            assert self.debugger is not None
            self.debugger.stop()
            self.debugger.step()
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[:self.input_batch.num_reqs]
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[:len(spec_decode_metadata.logits_indices)]
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        if self.need_accepted_tokens:  # TODO remove this if
            self._update_states_after_model_execute(
                sampler_output.sampled_token_ids)
        return sampler_output

    # TODO: remove this func after eagle_proposer is refactored and
    #  _bookkeeping_sync is moved after propose_draft_token_ids
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
            LogprobsLists | None,
            list[list[int]],
            dict[str, LogprobsTensors | None],
            list[str],
            dict[str, int],
            list[int],
    ]:
        # TODO: implement PR 28597 from vllm
        discard_sampled_tokens_req_indices = \
            self.discard_request_indices.np[:self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    return_cu_num_tokens=logprobs_tensors is not None,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            if self.num_spec_tokens <= 0:
                assert sampled_token_ids.shape[-1] == 1
                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids

            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [
                    -1
                ] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (logprobs_tensors.tolists(cu_num_tokens)
                          if not self.use_async_scheduling
                          and logprobs_tensors is not None else None)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    def _build_dummy_attn_metadata(
        self,
        with_prefill: bool,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
        num_scheduled_tokens: np.ndarray,
        aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        is_graph_capturing: bool = False,
    ) -> Optional[dict[str, Any]]:
        attn_metadata: Optional[dict[str, Any]] = None

        if force_attention or aclgraph_runtime_mode == CUDAGraphMode.FULL:
            assert with_prefill is False, \
                "Full decode graph only supports uniform batch now."

            attn_metadata = {}

            use_pa_slot_workspace = (
                is_graph_capturing
                and using_paged_attention(int(num_tokens), self.vllm_config))
            _slot_workspace: Optional[torch.Tensor] = None
            _slot_workspace_row = 0
            _slot_workspace_cap_n = int(num_tokens)
            if use_pa_slot_workspace:
                self._dynkv_slot_workspace_layer_row[_slot_workspace_cap_n] = {}

            # The reason why we use a fixed seq_len rather than max_query_len is that
            # _npu_paged_attention_get_workspace only returns max workspace with specific
            # seq_lens. We use this seq_len only when capturing graph, and still use max_query_len
            # in inference. This will be removed once npu_fused_infer_attention_score
            # outperforms _npu_paged_attention on all cases.
            seq_lens = SEQ_LEN_WITH_MAX_PA_WORKSPACE if is_graph_capturing and using_paged_attention(
                num_tokens, self.vllm_config) else max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cu_num_tokens, arange = self._get_cumsum_and_arange(
                num_scheduled_tokens)

            self.query_start_loc.cpu[1:num_reqs +
                                     1] = torch.Tensor(cu_num_tokens)
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

            num_computed_tokens_cpu = (
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                block_table_tensor = self.input_batch.block_table[
                    kv_cache_group_id].get_device_tensor()
                slot_mapping = self.input_batch.block_table[
                    kv_cache_group_id].slot_mapping
                if (
                    use_pa_slot_workspace
                    and _slot_workspace is None
                ):
                    total_layers = sum(
                        len(g.layer_names)
                        for g in self.kv_cache_config.kv_cache_groups)
                    _slot_workspace = self._get_or_create_slot_workspace(
                        _slot_workspace_cap_n,
                        total_layers,
                        int(slot_mapping.gpu.numel()),
                        slot_mapping.gpu.device,
                        slot_mapping.gpu.dtype,
                    )
                long_seq_metadata = None if self.pcp_size * self.dcp_size == 1 else self.pcp_manager.generate_pcp_metadata(
                    num_tokens, self.query_lens, self.input_batch,
                    num_scheduled_tokens)
                if long_seq_metadata is not None:
                    pcp_world_size = get_pcp_group().world_size
                    dcp_world_size = get_dcp_group().world_size
                    num_computed_tokens_of_pcp_dcp = [[
                        [0] * dcp_world_size for _ in range(pcp_world_size)
                    ] for _ in range(num_tokens)]
                    long_seq_metadata.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp

                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens_cpu=self.seq_lens.cpu,
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    actual_seq_lengths_q=self.actual_seq_lengths_q,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    positions=self.positions.gpu,
                    attn_state=self.attn_state,
                    max_query_len=max_query_len,
                    decode_token_per_req=self.decode_token_per_req,
                    prefill_context_parallel_metadata=long_seq_metadata,
                    max_seq_len=0)
                if self.pcp_size * self.dcp_size > 1:
                    common_attn_metadata.block_table_tensor = \
                        block_table_tensor[:num_reqs * self.decode_threshold]
                attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and \
                        self.speculative_config.method == "mtp":
                    # `AscendAttentionState.SpecDecoding` is only designed for mla
                    if self.vllm_config.model_config.use_mla:
                        attn_state = AscendAttentionState.SpecDecoding
                    else:
                        attn_state = AscendAttentionState.ChunkedPrefill

                common_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    _seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    seq_lens=self.seq_lens.cpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    _num_computed_tokens_cpu=num_computed_tokens_cpu,
                    max_query_len=max_query_len,
                    max_seq_len=seq_lens)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    builder = attn_group.get_metadata_builder()
                    if isinstance(builder, GDNAttentionMetadataBuilder):
                        attn_metadata_gdn_attention = builder.build_for_cudagraph_capture(
                            common_metadata)
                    else:
                        attn_metadata_full_attention = builder.build_for_graph_capture(
                            common_attn_metadata, attn_state)
                    for layer_name in kv_cache_group_spec.layer_names:
                        if "linear_attn" in layer_name:
                            meta_src = attn_metadata_gdn_attention
                        else:
                            meta_src = attn_metadata_full_attention
                        try:
                            import copy as _copy

                            meta_i = _copy.copy(meta_src)
                        except Exception:
                            meta_i = meta_src
                        # FULL-graph reshape_and_cache pins distinct addresses per layer.
                        # PA decode: use one [L, n_sm] workspace row view per layer so
                        # runtime remap writes the same memory graph captured.
                        try:
                            if (
                                use_pa_slot_workspace
                                and _slot_workspace is not None
                            ):
                                meta_i.slot_mapping = _slot_workspace[
                                    _slot_workspace_row]
                                self._dynkv_slot_workspace_layer_row[
                                    _slot_workspace_cap_n][str(layer_name)] = (
                                        _slot_workspace_row)
                                _slot_workspace_row += 1
                            else:
                                meta_i.slot_mapping = meta_i.slot_mapping.clone()
                        except Exception:
                            pass
                        if (is_graph_capturing
                                and using_paged_attention(
                                    int(num_tokens), self.vllm_config)):
                            try:
                                _sl_cap = meta_i.seq_lens
                                if isinstance(_sl_cap, torch.Tensor):
                                    meta_i.seq_lens = _sl_cap.clone()
                            except Exception:
                                pass
                        try:
                            setattr(meta_i, "layer_name", layer_name)
                        except Exception:
                            pass
                        attn_metadata[layer_name] = meta_i

            if is_graph_capturing and attn_metadata:
                self._dynkv_graph_slot_bufs[int(num_tokens)] = {
                    str(ln): attn_metadata[ln].slot_mapping
                    for ln in attn_metadata
                    if getattr(attn_metadata[ln], "slot_mapping", None)
                    is not None
                }
                if using_paged_attention(int(num_tokens), self.vllm_config):
                    self._dynkv_graph_context_lens_bufs[int(num_tokens)] = {
                        str(ln): attn_metadata[ln].seq_lens
                        for ln in attn_metadata
                        if isinstance(
                            getattr(attn_metadata[ln], "seq_lens", None),
                            torch.Tensor)
                    }

        return attn_metadata

    def _generate_dummy_run_hidden_states(self, input_ids, positions,
                                          num_tokens, intermediate_tensors,
                                          inputs_embeds):
        forward_context = get_forward_context()
        assert forward_context is not None
        if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not forward_context.capturing and not self.use_sparse
                and hasattr(self, "update_stream")):
            self._update_aclgraph_attn_params(num_tokens)
        hidden_states = self.model(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds)

        if self.use_aux_hidden_state_outputs:
            hidden_states, _ = hidden_states
        else:
            hidden_states = hidden_states
        return hidden_states

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        # In multi-DP scenarios, there may be situations where all DP groups are executing dummy runs.
        # If sequence parallelism is enabled, it is essential to ensure that num_tokens is divisible by tp_size.
        if enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer and not self.is_kv_consumer:
            with_prefill = True

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill,
         synced_cudagraph_mode) = self._sync_metadata_across_dp(
             batch_descriptor.num_tokens, with_prefill, _ag_mode.value)

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.seperate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.max_num_reqs
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                num_reqs = (num_tokens + self.decode_token_per_req -
                            1) // self.decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        if num_tokens_across_dp is not None:
            _ag_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
                disable_full=synced_cudagraph_mode
                <= CUDAGraphMode.PIECEWISE.value)

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = (batch_descriptor.num_reqs if
                           batch_descriptor.num_reqs is not None else num_reqs)
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        # filter out the valid batch descriptor
        if cudagraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if cudagraph_runtime_mode != CUDAGraphMode.NONE and cudagraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {cudagraph_runtime_mode}.")
        else:
            cudagraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            force_attention=force_attention,
            is_graph_capturing=is_graph_capturing,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens,
                                            num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by tp_size;
                # otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading to incorrect memory estimation and potentially causing OOM.
                actual_tokens = num_tokens
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    actual_tokens = num_tokens // tp_size
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=actual_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k:
                    v[:num_tokens_padded]
                    for k, v in self.intermediate_tensors.items()
                })

            need_dummy_logits = (not is_profile and lmhead_tp_enable())
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                        dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(
                        self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(
                        hidden_states[dummy_indices])

            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    in_profile_run=is_profile,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    model_instance=self.model,
                    is_multimodal_model=self.is_multimodal_model):
                hidden_states = self._generate_dummy_run_hidden_states(
                    input_ids, positions, num_tokens_padded,
                    intermediate_tensors, inputs_embeds)
                dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile)
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()
            return hidden_states, hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = None

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * self.max_num_reqs
        num_scheduled_tokens_list[
            -1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        # Sometimes, after the model is compiled through the AOT backend,
        # the model output may become a list containing only one Tensor object.
        if isinstance(hidden_states, list) and \
            len(hidden_states) == 1 and \
            isinstance(hidden_states[0], torch.Tensor):
            hidden_states = hidden_states[0]
            hidden_states = hidden_states[logit_indices]
            output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and \
            select_moe_comm_method(mc2_tokens_capacity, self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity,
                            with_prefill=True,
                            is_profile=True)
        origin_max_num_tokens = self.max_num_tokens
        # in the pcp scenario, the split sequence needs to be used for profile run
        # TODO: after the vllm pcp function is launched, this logic needs to be brought up to the community
        if self.pcp_size > 1:
            self.max_num_tokens = math.ceil(self.max_num_tokens /
                                            (self.pcp_size * 2)) * 2
        super().profile_run()
        self.eplb_warmup()
        self.max_num_tokens = origin_max_num_tokens

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.dynamic_eplb:
                model_register(self.model, self.model_config)
            if self.drafter:
                logger.info("Loading drafter model...")
                with get_tp_context(self.drafter):
                    self.drafter.load_model(self.model)
                if self.use_aux_hidden_state_outputs:
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # DynamicKV offload: install q_proj hooks BEFORE ACLGraphWrapper.
        # Hooks must attach to real ``qkv_proj`` modules and capture pre-RoPE Q.
        # Installing after graph wrap can leave handles registered but not firing
        # during prefill forward; attention-side post-RoPE Q capture compresses but
        # breaks decode quality.
        try:
            ascend_cfg = get_ascend_config()
            dyn_impl = str(getattr(ascend_cfg, "dynamic_kv_impl", "offload"))
            if self._is_dynamic_kv_enabled() and dyn_impl == "offload":
                _DYNKV_STATE.setdefault("offload_q_last", {})
                _DYNKV_STATE.setdefault("offload_ctx", None)
                _DYNKV_STATE.setdefault("offload_hooks", [])
                _DYNKV_STATE.setdefault("offload_lock", threading.Lock())

                def should_enable() -> bool:
                    try:
                        ctx = _DYNKV_STATE.get("offload_ctx")
                        return ctx is not None and isinstance(ctx, OffloadCaptureContext)
                    except Exception:
                        return False

                def get_ctx() -> OffloadCaptureContext | None:
                    try:
                        ctx = _DYNKV_STATE.get("offload_ctx")
                        return ctx if isinstance(ctx, OffloadCaptureContext) else None
                    except Exception:
                        return None

                q_last_store = _DYNKV_STATE["offload_q_last"]
                if not isinstance(q_last_store, dict):
                    q_last_store = {}
                    _DYNKV_STATE["offload_q_last"] = q_last_store

                if not _DYNKV_STATE.get("offload_hooks"):
                    handles = install_qproj_hooks(
                        model=self.model,
                        should_enable=should_enable,
                        get_capture_ctx=get_ctx,
                        q_last_store=q_last_store,
                        window_size=int(getattr(ascend_cfg, "dynamic_kv_window_size", 16)),
                    )
                    _DYNKV_STATE["offload_hooks"] = handles
                    if not handles:
                        logger.warning(
                            "[DynamicKV][offload] no Q-capture hooks installed; "
                            "compression will be skipped"
                        )
        except Exception:
            logger.exception("[DynamicKV][offload] failed to install q_proj hooks")

        # wrap the model with full graph wrapper if needed (after hook install).
        use_full_aclgraph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        if use_full_aclgraph:
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = (len(self.attn_groups) > 1)
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any([
            isinstance(attn_group[0].kv_cache_spec, MambaSpec)
            for attn_group in self.attn_groups
        ])

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def _align_memory(self, tensor: torch.Tensor,
                      alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache
        num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches, num_attn_module)
        # ``self.kv_caches`` is a per-layer list after bind; DynamicKV offload needs
        # the name->(k,v) dict for ``run_offload_rewrite_and_build_updates``.
        self._kv_caches_by_layer_name = kv_caches
        return kv_caches

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cahce and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
         """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, Union[torch.Tensor,
                                              Optional[torch.Tensor]]] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # for mamba linear attention
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size,
                                             dtype=torch.int8,
                                             device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned,
                                             dtype=torch.int8,
                                             device=self.device)
                        tensor = self._align_memory(
                            tensor, alignment)[:kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    if self.model_config.use_mla:
                        head_size = self.model_config.hf_text_config.qk_rope_head_dim + \
                            self.model_config.hf_text_config.kv_lora_rank

                    dsa_k_cache_factor = None
                    dsa_k_cache_size = None
                    if not self.model_config.use_mla:
                        # for non-mla model, use FullAttentionSpec
                        k_tensor_split_factor = 2
                        v_tensor_split_factor = 2
                    elif self.use_sparse:
                        # for deepseek v3.2, DSA use FullAttentionSpec
                        # FullAttentionSpec allocate 2 * mla page size bytes,
                        # and we use half of that for k cache in DSA
                        dsa_k_cache_factor = 2
                        k_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.qk_rope_head_dim
                        dsa_k_cache_size = int(kv_cache_tensor.size //
                                               dsa_k_cache_factor)
                    else:
                        # for other deepseek models, use MLAAttentionSpec
                        k_tensor_split_factor = head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = head_size / self.model_config.hf_text_config.qk_rope_head_dim

                    k_tensor_size = int(kv_cache_tensor.size //
                                        k_tensor_split_factor)
                    v_tensor_size = int(kv_cache_tensor.size //
                                        v_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size,
                                dtype=torch.int8,
                                device=self.device)
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        k_tensor = self._align_memory(
                            k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(
                            v_tensor, alignment)[:v_tensor_size]
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None and dsa_k_cache_size is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size + alignment,
                                dtype=torch.int8,
                                device=self.device)
                            dsa_k_cache_tensor = self._align_memory(
                                dsa_k_cache_tensor,
                                alignment)[:dsa_k_cache_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if ("attn" in layer_name_inner
                                and "linear_attn" not in layer_name_inner):
                            kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor) if \
                                not self.use_sparse else (k_tensor, v_tensor, dsa_k_cache_tensor)

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, AttentionSpec):
                    raw_dsa_k_tensor = None
                    if self.use_sparse:
                        raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        assert raw_dsa_k_tensor is not None
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_block_size"
                               ) and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_block_size()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk, block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    if not self.model_config.use_mla:
                        k_shape = kv_cache_shape[1:]
                        v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.kv_lora_rank
                        ]
                        v_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.qk_rope_head_dim
                        ]
                    k_cache = raw_k_tensor.view(dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(dtype).view(v_shape)
                    if get_ascend_device_type() == AscendDeviceType._310P:
                        k_cache = maybe_trans_nz(k_cache)
                        v_cache = maybe_trans_nz(v_cache)
                    if self.use_sparse and raw_dsa_k_tensor is not None:
                        dsa_k_cache_shape = (num_blocks,
                                             kv_cache_spec.block_size, 1, 128)
                        dsa_k_cache_size = (
                            num_blocks
                        ) * kv_cache_spec.block_size * 128 * dtype.itemsize
                        dsa_k_cache = raw_dsa_k_tensor[:dsa_k_cache_size].view(
                            dtype).view(dsa_k_cache_shape)
                        kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel(
                    ) % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel(
                    ) // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(kv_cache_spec.shapes,
                                            kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += torch.prod(
                            torch.tensor(target_shape)).item()
                        tensor = raw_tensor.view(
                            dtype)[start_idx:target_idx].view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec,
                              EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(
                kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(
                    iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # the backend.
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                except IndexError:
                    attn_groups = None
                if attn_groups and self.use_hybrid_blocks:
                    # Use the backend's supported block size list
                    backend = attn_groups[0].backend
                    supported_sizes = backend.get_supported_block_size()
                    # If no specific sizes supported, use cache config
                    # block_size
                    kernel_block_size_list = (supported_sizes
                                              if supported_sizes else
                                              [self.cache_config.block_size])
                else:
                    # Fallback to cache config block_size if no backend found
                    kernel_block_size_list = [self.cache_config.block_size]
                kernel_block_sizes.append(kernel_block_size_list)
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                kernel_block_sizes.append([0])
        if block_sizes != [
                self.cache_config.block_size
        ] or kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.model_config.max_model_len,
                                  self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
                kernel_block_sizes=kernel_block_sizes,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]],
                   set[type[AttentionBackend]]]:
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,
                kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                        layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend,
                                                       layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return (
                {
                    attn_backends[k]: v
                    for k, v in attn_backend_layers.items()
                },
                set(group_key.attn_backend
                    for group_key in attn_backends.values()),
            )

        def create_attn_groups(attn_backends_map: dict[AttentionBackend,
                                                       list[str]],
                               kv_cache_group_id: int) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend,
                 kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = []
                attn_metadata_builders.append(attn_backend.get_builder_cls()(
                    kv_cache_spec,
                    layer_names,
                    self.vllm_config,
                    self.device,
                ))
                attn_group = AttentionGroup(attn_backend, layer_names,
                                            kv_cache_spec, kv_cache_group_id,
                                            attn_metadata_builders)
                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        self._check_and_update_cudagraph_mode(attention_backend_list,
                                              kv_cache_config.kv_cache_groups)

        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            attn_backends = get_attn_backends_for_group(  # type: ignore
                kv_cache_group_spec)
            groups = create_attn_groups(attn_backends[0], i)
            self.attn_groups.append(groups)

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()
            if hasattr(attn_metadata_builder_i,
                       "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = (
                    attn_metadata_builder_i.reorder_batch_threshold)
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != \
                            self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}")
                    else:
                        self.reorder_batch_threshold = reorder_batch_threshold_i  # noqa

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                  AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                # TODO: Support other attention modules, e.g., cross-attention
                # TODO(lucas): move the attention specs into the model layers like
                # the attention backends
                if attn_module.attn_type == AttentionType.DECODER:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                elif attn_module.attn_type in (AttentionType.ENCODER,
                                               AttentionType.ENCODER_ONLY):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    kv_cache_spec[layer_name] = CrossAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")

            elif isinstance(attn_module, MLAAttention):
                if use_mla and not self.use_sparse:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.cache_config.cache_dtype)
                else:
                    # TODO(cmq): This is a hack way to fix deepseek kvcache when
                    # using DSA. Fix the spec in vLLM is a finnal way.
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)

        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            if (self.vllm_config.speculative_config is not None
                    and self.vllm_config.model_config.hf_text_config.model_type
                    not in ["qwen3_next"]):
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len

            page_size_padded = (
                self.vllm_config.cache_config.mamba_page_size_padded)

            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtypes=mamba_module.get_state_dtype(),
                    block_size=max_model_len,
                    page_size_padded=page_size_padded,
                    mamba_type=mamba_module.mamba_type,
                    num_speculative_blocks=(
                        self.speculative_config.num_speculative_tokens
                        if self.speculative_config else 0),
                )

        return kv_cache_spec

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        with update_pass_config(self):
            super()._check_and_update_cudagraph_mode(attention_backends,
                                                     kv_cache_groups)

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        if self.use_aclgraph:
            set_graph_params(self.cudagraph_batch_sizes)
            if self.speculative_config:
                set_draft_graph_params(self.cudagraph_batch_sizes)

    def capture_model(self) -> None:
        gpu_model_runner_cls = next((cls for cls in self.__class__.__mro__
                                     if cls.__name__ == "GPUModelRunner"),
                                    None)
        if gpu_model_runner_cls is None:
            raise TypeError("Could not find GPUModelRunner in the MRO. "
                            "The class hierarchy may have changed.")
        parent_module_name = gpu_model_runner_cls.__module__
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(
                parent_module_name):
            GPUModelRunner.capture_model(self)

    def _prepare_multimodal_fields(self):
        """
        Ensures specific multimodal tensors are on CPU.
        This is necessary for fields like 'grid_thw' which are converted to numpy 
        inside the model's forward pass.
        """
        if not self.multimodal_cpu_fields:
            return

        req_ids = self.input_batch.req_ids
        for req_id in req_ids:
            req = self.requests.get(req_id)
            if req is None:
                continue

            mm_data = getattr(req, 'multimodal_data', None)
            if not mm_data:
                continue

            for field in self.multimodal_cpu_fields:
                if field in mm_data:
                    tensor = mm_data[field]
                    if isinstance(
                            tensor,
                            torch.Tensor) and tensor.device.type != 'cpu':
                        mm_data[field] = tensor.cpu()


@contextmanager
def _torch_cuda_wrapper():

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        yield
    except Exception as e:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
        torch.cuda.synchronize = _StreamPlaceholder
        torch.cuda.mem_get_info = _StreamPlaceholder
        raise RuntimeError(f"NPUModelRunner init failed, error is {e}")
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info


# TODO: This method will be removed subsequently and implemented in platform.
@contextmanager
def _replace_gpu_model_runner_function_wrapper(target_module_name):
    try:
        target_module = sys.modules[target_module_name]
        setattr(target_module, "graph_capture", graph_capture)
        yield
    finally:
        setattr(target_module, "graph_capture", graph_capture)


# TODO: remove it when flash_common1 is removed
@contextmanager
def update_pass_config(model_runner):
    try:
        original_pass_config_sp = model_runner.compilation_config.pass_config.enable_sp
        model_runner.compilation_config.pass_config.enable_sp = enable_sp(
            model_runner.vllm_config)
        yield
    finally:
        model_runner.compilation_config.pass_config.enable_sp = original_pass_config_sp
