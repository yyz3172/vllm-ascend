#!/usr/bin/env bash
set -euo pipefail
cd /root/x00827378/vllm-ascend
export ASCEND_CUSTOM_OPP_PATH=/root/x00827378/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:/root/x00827378/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
exec /root/x00827378/vllm-ascend/ztmp/flex_tq_4bit_perf/flex_tq_4bit_perf --attention-only --skip-cache-fill --q-lens 1 --kv-lens 512 --heads 16 --kv-heads 8 --block-size 128 --warmup 1 --repeat 5
