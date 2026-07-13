import contextlib
import os

from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.ops.turboquant_kv_cache import refresh_turboquant_env_cache


def spawn(extra):
    inherited = {}
    for k, v in os.environ.items():
        if k.startswith(("ASCEND_", "LD_", "PYTHON", "HCCL_", "GE_", "TOOLCHAIN_")) or k in (
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
        ):
            inherited[k] = v
    inherited.update(extra)
    return inherited


@contextlib.contextmanager
def patched(env):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def main():
    encode = os.getenv("ENC", "1")
    fia = os.getenv("FIA", "1")
    prompts = os.getenv("PROMPTS", "1")
    env = spawn({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_CORE_MULTIPROC_METHOD": "spawn",
        "VLLM_ASCEND_TURBOQUANT_4BIT_SLAB_CACHE": "0",
        "VLLM_ASCEND_TURBOQUANT_MSE_IMPL": "v1",
        "VLLM_ASCEND_TURBOQUANT_DECODE_OP": "1",
        "VLLM_ASCEND_TURBOQUANT_ENCODE_OP": encode,
        "VLLM_ASCEND_TURBOQUANT_FUSED_FIA_K8V4": fia,
    })
    texts = ["Hello, my name is"]
    if prompts == "2":
        texts.append("介绍turboquant技术")
    with patched(env):
        clear_ascend_config()
        refresh_turboquant_env_cache()
        kwargs = dict(
            model="/root/yyz/models/Qwen3-0.6B",
            seed=0,
            trust_remote_code=True,
            max_model_len=256,
            block_size=16,
            kv_cache_dtype="turboquant",
            gpu_memory_utilization=0.05,
            additional_config={"turboquant_kv_bits": [8, 4]},
            enforce_eager=True,
            enable_chunked_prefill=True,
            disable_log_stats=True,
        )
        llm = LLM(**kwargs)
        outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=40))
    print(f"RESULT encode={encode} fia={fia} prompts={prompts}")
    for o in outs:
        print(repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
