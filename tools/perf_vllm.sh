#!/bin/bash

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 12345 \
  --model /root/l00856060/model/Qwen3-0.6B \
  --served-model-name qwen3 \
  --dataset-name random \
  --input-len 10 \
  --output-len 10 \
  --num-prompts 100