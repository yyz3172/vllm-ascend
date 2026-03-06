#!/bin/sh
# 单机 4 卡 Decode（TP=4），与 launch_decode_single_node_4cards.py 配合使用。
# 使用前请修改下方 nic_name、local_ip、MODEL_PATH。
# 参数 $8 由启动脚本传入 ASCEND_RT_VISIBLE_DEVICES（单副本时为 "4,5,6,7"）。

nic_name="eth0"
local_ip="127.0.0.1"
MODEL_PATH="/ds/models/Qwen3-32B"

export ASCEND_RT_VISIBLE_DEVICES=$8

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024

export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port $6 \
  --tensor-parallel-size 4 \
  --nnodes 1 \
  --seed 1024 \
  --served-model-name qwen3_32b \
  --max-model-len 8192 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 256 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '
    {
      "kv_connector": "MooncakeConnectorV1",
      "kv_buffer_device": "npu",
      "kv_role": "kv_consumer",
      "kv_parallel_size": "1",
      "kv_port": "20001",
      "engine_id": "0",
      "kv_connector_extra_config": {
        "prefill": { "dp_size": 1, "tp_size": 4 },
        "decode": { "dp_size": 1, "tp_size": 4 }
      },
      "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
