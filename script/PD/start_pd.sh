#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PROXY_PORT=9878
export MODEL_PATH="/root/l00856060/model/Qwen3-0.6B/"
export ASCEND_RT_VISIBLE_DEVICES_PREFILL=3
export ASCEND_RT_VISIBLE_DEVICES_DECODE=4
export VLLM_VENV=/root/devcommon/w30022782/env/.venv/
export TYPE="BASE"

ACTION="$1"

if [ "$2" == "turboquant" ]; then
    TYPE="TURBOQUANT"
fi

if [ "$ACTION" == "start" ]; then
    python "${SCRIPT_DIR}/pd_service_ctl.py" start \
        --pd_mode 1P1_1D1 \
        --vllm_venv $VLLM_VENV
else
    python "${SCRIPT_DIR}/pd_service_ctl.py" stop
fi