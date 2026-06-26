#!/bin/bash

curl http://127.0.0.1:12345/v1/completions \
   -H "Content-Type: application/json" \
   -d "{\"model\": \"qwen3\", \
        \"prompt\": \"Hello, my name is\", \
        \"max_completion_tokens\": \"128\", \
        \"temperature\": \"0.0\"}"