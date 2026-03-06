#!/bin/sh
# 单机 2 卡 Decode（TP=2），Qwen3-32B。基于 1card 版本，直接执行：bash run_decode_2cards.sh
# ========== 配置区（仅改此处即可，须与 run_prefill_2cards.sh 中 model_path/nic_name/local_ip 一致）==========
nic_name="eth0"
local_ip="172.17.0.2"
model_path="/root/autodl-tmp/models/Qwen3-32B"
# Mooncake libtransfer_engine.so 所在目录（若报 libtransfer_engine.so 找不到，请填写并取消下一行注释）
# transfer_engine_lib_path="/usr/local/Ascend/ascend-toolkit/latest/lib64"
transfer_engine_lib_path="/usr/local/lib"
# 当前 Python 的 lib 目录（若报 libpython3.11.so.1.0 找不到，请填写 venv 的 lib 或 uv 的 Python lib）
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"
# DP / 端口 / 设备（dp_port 须与 Prefill 的 dp_port 不同；Decode 用另两张卡）
dp_size=1
dp_ip="127.0.0.1"
dp_port=13495
engine_port=9010
visible_devices="2,3"
# ==========================================

export ASCEND_RT_VISIBLE_DEVICES=$visible_devices

# 使用系统 libstdc++，满足 Ascend Mooncake engine 对 GLIBCXX_3.4.30 的要求（conda 自带版本较旧）
# LD_LIBRARY_PATH 对 fork 出的 worker 可能被 RPATH 覆盖，故用 LD_PRELOAD 强制加载系统 libstdc++
if [ -f /usr/lib/aarch64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6
elif [ -f /usr/lib64/libstdc++.so.6 ]; then
  export LD_PRELOAD=/usr/lib64/libstdc++.so.6
elif [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
fi
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH="${python_lib_path:+$python_lib_path:}${transfer_engine_lib_path:+$transfer_engine_lib_path:}/usr/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib:${LD_LIBRARY_PATH:-}"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024

export VLLM_DP_SIZE=$dp_size
export VLLM_DP_MASTER_IP=$dp_ip
export VLLM_DP_MASTER_PORT=$dp_port
export VLLM_DP_RANK_LOCAL=0
export VLLM_DP_RANK=0
export VLLM_DP_SIZE_LOCAL=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $engine_port \
    --tensor-parallel-size 2 \
    --nnodes 1 \
    --seed 1024 \
    --served-model-name qwen3_32b \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": "1",
        "kv_port": "20001",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "prefill": { "dp_size": 1, "tp_size": 2 },
            "decode": { "dp_size": 1, "tp_size": 2 }
        },
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
