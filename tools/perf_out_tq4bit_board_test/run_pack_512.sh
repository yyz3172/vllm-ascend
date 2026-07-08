#!/usr/bin/env bash
set -euo pipefail
cd /root/x00827378/vllm-ascend
source xrx_infoenvs
exec ../../mytmp/flex_tq_4bit_perf/flex_tq_4bit_perf \
    --pack-only \
    --q-lens 1 \
    --kv-lens 512 \
    --pack-tokens 512 \
    --heads 16 \
    --kv-heads 8 \
    --block-size 128 \
    --warmup 1 \
    --repeat 5
