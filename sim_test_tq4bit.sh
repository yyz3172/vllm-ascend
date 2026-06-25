#!/bin/bash
set -x
source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash

KEY="${1:-0}"
case "$KEY" in
  0) TQ4BIT_NUM_HEADS=16; TQ4BIT_NUM_KV_HEADS=8; TQ4BIT_QUERY_TOKENS=1  ;;
  4) TQ4BIT_NUM_HEADS=16; TQ4BIT_NUM_KV_HEADS=2; TQ4BIT_QUERY_TOKENS=24 ;;
  5) TQ4BIT_NUM_HEADS=4;  TQ4BIT_NUM_KV_HEADS=2; TQ4BIT_QUERY_TOKENS=24 ;;
  *) echo "Unsupported KEY: $KEY (supported: 0, 4, 5)"; exit 1 ;;
esac

export TQ4BIT_SEQ_LEN=128
export TQ4BIT_QUERY_TOKENS=$TQ4BIT_QUERY_TOKENS
export TQ4BIT_NUM_HEADS=$TQ4BIT_NUM_HEADS
export TQ4BIT_NUM_KV_HEADS=$TQ4BIT_NUM_KV_HEADS
export TQ4BIT_BLOCK_SIZE=128
msprof op simulator --kernel-name=TurboquantAttentionPaged4bit --soc-version=Ascend910B3 --launch-count=10 --output=/root/l00856060/perflog2 --application=./test_tq4bit
