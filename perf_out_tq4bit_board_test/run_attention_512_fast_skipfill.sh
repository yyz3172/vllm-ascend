#!/usr/bin/env bash
set -euo pipefail
cd /root/x00827378/vllm-ascend
source xrx_infoenvs
exec tools/run_turboquant4bit_op_quick.sh -- \
    --attention-only \
    --skip-cache-fill \
    --seq-len 512 \
    --query-tokens 1 \
    --heads 16 \
    --kv-heads 8 \
    --block-size 128 \
    --warmup 1 \
    --repeat 5
