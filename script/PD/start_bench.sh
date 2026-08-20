vllm bench serve \
--backend vllm \
--model /root/l00856060/model/Qwen3-0.6B/ \
--port 9878 \
--endpoint /v1/completions \
--dataset-name random \
--input-len 10 \
--output-len 200 \
--num_prompt 1000