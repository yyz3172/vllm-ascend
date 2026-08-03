#!/usr/bin/env bash
set -euo pipefail
cd .
source xrx_infoenvs
exec ztmp/flex_tq_4bit_perf/flex_tq_4bit_perf \
    --attention-only \
    --skip-cache-fill \
    --q-lens 1 \
    --kv-lens 512 \
    --heads 16 \
    --kv-heads 8 \
    --block-size 128 \
    --warmup 1 \
    --repeat 5
