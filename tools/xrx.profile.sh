source vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash

msprof op ${ROOT_DIR}/ztmp/turboquant4bit_perf --seq-len 1024 --heads 16 --kv-heads 2 --block-size 128 --pack-tokens 1 --warmup 10 --repeat 50
