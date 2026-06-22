#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

init_pd_env
ensure_model_exists
setup_common_runtime_env
if [[ "${TYPE}" == "TURBOQUANT" ]]; then
    setup_turboquant_runtime_env
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES_PREFILL}"
export HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-256}"

export VLLM_DP_SIZE=1
export VLLM_DP_MASTER_IP="${PREFILL_DP_MASTER_IP:-127.0.0.1}"
export VLLM_DP_MASTER_PORT="${PREFILL_DP_MASTER_PORT:-13395}"
export VLLM_DP_RANK_LOCAL=0
export VLLM_DP_RANK=0
export VLLM_DP_SIZE_LOCAL=1

ARGS=(
    "${MODEL_PATH}"
    --host 0.0.0.0
    --port "${PREFILL_PORT}"
    --enable-prefix-caching
    --tensor-parallel-size 1
    --seed 1024
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}"
    --trust-remote-code
    --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}"
    --enforce-eager
    --kv-transfer-config "{
        \"kv_connector\": \"MooncakeConnectorV1\",
        \"kv_buffer_device\": \"npu\",
        \"kv_role\": \"kv_producer\",
        \"kv_parallel_size\": \"1\",
        \"kv_port\": \"${PREFILL_KV_PORT}\",
        \"engine_id\": \"0\",
        \"kv_connector_extra_config\": {
            \"prefill\": { \"dp_size\": 1, \"tp_size\": 1 },
            \"decode\": { \"dp_size\": 1, \"tp_size\": 1 }
        },
        \"kv_connector_module_path\": \"vllm_ascend.distributed.mooncake_connector\"
    }"
)

if [[ "${TYPE}" == "TURBOQUANT" ]]; then
    ARGS+=(
        --kv-cache-dtype turboquant
        --additional-config "{\"turboquant_kv_bits\": ${TURBOQUANT_KV_BITS:-[8, 8]}}"
    )
fi

mkdir -p "${LOG_DIR}"
exec >> "${LOG_DIR}/prefill.log" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') [prefill] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "$(date '+%Y-%m-%d %H:%M:%S') [prefill] MODEL_PATH=${MODEL_PATH}"
echo "$(date '+%Y-%m-%d %H:%M:%S') [prefill] TYPE=${TYPE}"
echo "$(date '+%Y-%m-%d %H:%M:%S') [prefill] TURBOQUANT_KV_BITS=${TURBOQUANT_KV_BITS:-N/A}"

exec vllm serve "${ARGS[@]}"
