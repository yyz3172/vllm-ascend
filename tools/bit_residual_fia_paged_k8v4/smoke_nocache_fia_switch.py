#!/usr/bin/env python3
"""Smoke: PrefillNoCache br_fia switch on/off."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(
    _REPO / "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENGINE_CORE_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ASCEND_TURBOQUANT_MSE_IMPL", "v1")
os.environ["VLLM_ASCEND_BIT_RESIDUAL_FIA"] = "1"
os.environ["VLLM_ASCEND_BIT_RESIDUAL_DECODE_FIA"] = "1"

from vllm import LLM, SamplingParams
from vllm_ascend.ascend_config import clear_ascend_config


def run(tag: str, nocache_on: bool) -> tuple[dict, str]:
    clear_ascend_config()
    os.environ["VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA"] = "1" if nocache_on else "0"
    llm = LLM(
        model=os.getenv("MODEL_PATH", "/root/cyl/model/Qwen3-0.6B"),
        seed=0,
        trust_remote_code=True,
        max_model_len=256,
        block_size=128,
        kv_cache_dtype="turboquant",
        gpu_memory_utilization=0.05,
        additional_config={"turboquant_kv_bits": [8, 4]},
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    def install_counter(worker):
        import vllm_ascend.attention.attention_v1 as m

        if getattr(m, "_nocache_smoke_patched", False):
            return {"already": True}
        orig = m.bit_residual_fia_paged_k8v4
        counts = {"total": 0}

        def wrapped(*args, **kwargs):
            counts["total"] += 1
            return orig(*args, **kwargs)

        m.bit_residual_fia_paged_k8v4 = wrapped
        m._nocache_smoke_patched = True
        m._nocache_smoke_counts = counts
        return {"ok": True}

    def read_counts(worker):
        import vllm_ascend.attention.attention_v1 as m

        c = getattr(m, "_nocache_smoke_counts", None)
        return dict(c) if c else {"missing": True}

    print(llm.collective_rpc(install_counter), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    text = llm.generate(
        ["介绍turboquant技术"], sampling_params=sp, use_tqdm=False
    )[0].outputs[0].text
    c1 = llm.collective_rpc(read_counts)[0]
    print(f"{tag}: text={text!r} counts={c1}", flush=True)
    del llm
    return c1, text


def main() -> int:
    import vllm_ascend.envs as envs

    assert "VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA" in dir(envs) or hasattr(
        envs, "VLLM_ASCEND_BIT_RESIDUAL_NOCACHE_FIA"
    )
    print("=== NOCACHE_FIA=0 ===", flush=True)
    c_off, t_off = run("off", False)
    print("=== NOCACHE_FIA=1 ===", flush=True)
    c_on, t_on = run("on", True)
    print(
        "SUMMARY",
        {"off": c_off, "on": c_on, "t_off": t_off, "t_on": t_on},
        flush=True,
    )
    # PrefillNoCache ON should call paged FIA at least as often as OFF
    # (OFF: decode-only FIA; ON: prefill + decode FIA).
    if c_on.get("total", 0) < c_off.get("total", 0):
        print("WARN: expected on.total >= off.total", flush=True)
        return 1
    if c_on.get("total", 0) <= 0:
        print("ERROR: NOCACHE_FIA=1 made zero FIA calls", flush=True)
        return 1
    print("SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
