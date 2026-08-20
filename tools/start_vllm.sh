#!/bin/bash
export PYTHONPATH="/root/l00856060/code/vllm018/vllm-ascend:/root/l00856060/code/vllm018/vllm:$PYTHONPATH"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=1
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_VERSION=0.18.0
export VLLM_ASCEND_TURBOQUANT_ENCODE_OP=1
export VLLM_ASCEND_TURBOQUANT_DECODE_OP=0
export VLLM_ASCEND_TURBOQUANT_CODEBOOK_METHOD=fast
export VLLM_ASCEND_TURBOQUANT_MSE_IMPL=v1

vllm serve /root/l00856060/model/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 12345 \
    --kv_cache_dtype turboquant \
    --served-model-name qwen3 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --additional-config '{"turboquant_kv_bits": [8, 8]}' > /root/l00856060/perflog2/vllm.log 2>&1 &
