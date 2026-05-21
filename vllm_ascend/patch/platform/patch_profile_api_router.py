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

"""Expose /start_profile and /stop_profile when Ascend torch profiler dir is set.

Upstream vLLM only mounts profile routes if ``--profiler-config`` sets
``profiler_config.profiler``. On Ascend, ``NPUWorker`` also enables
``torch_npu.profiler`` via ``VLLM_TORCH_PROFILER_DIR`` alone; this patch
registers the HTTP endpoints in that case so users need not duplicate CLI flags.
"""

from __future__ import annotations

import os

import vllm.envs as envs_vllm
from fastapi import FastAPI
from vllm.config import ProfilerConfig
from vllm.entrypoints.serve.profile import api_router as profile_api_router
from vllm.logger import logger

_orig_attach_router = profile_api_router.attach_router


def _ascend_torch_profiler_dir_configured() -> bool:
    trace_dir = envs_vllm.VLLM_TORCH_PROFILER_DIR
    return bool(trace_dir and str(trace_dir).strip())


def attach_router(app: FastAPI) -> None:
    profiler_config = getattr(app.state.args, "profiler_config", None)
    assert profiler_config is None or isinstance(profiler_config,
                                                 ProfilerConfig)

    if profiler_config is not None and profiler_config.profiler is not None:
        _orig_attach_router(app)
        return

    if _ascend_torch_profiler_dir_configured():
        logger.warning_once(
            "Ascend PyTorch Profiler: VLLM_TORCH_PROFILER_DIR is set; "
            "registering /start_profile and /stop_profile without "
            "--profiler-config. For local development only.",
        )
        app.include_router(profile_api_router.router)
        return

    _orig_attach_router(app)


profile_api_router.attach_router = attach_router
