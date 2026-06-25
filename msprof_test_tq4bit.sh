#!/bin/bash
set -x
source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
export TQ4BIT_SEQ_LEN=128
export TQ4BIT_QUERY_TOKENS=1
export TQ4BIT_NUM_HEADS=16
export TQ4BIT_NUM_KV_HEADS=8
export TQ4BIT_BLOCK_SIZE=128
msprof --output=/root/l00856060/perflog2 --application=./test_tq4bit
