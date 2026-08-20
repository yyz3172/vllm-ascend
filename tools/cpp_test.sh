SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUR_VLLM_ASCEND_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_VERSION=0.18.0
VLLM_ASCEND_TURBOQUANT_MSE_IMPL=v1
ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
VLLM_ASCEND_TURBOQUANT_CODEBOOK_METHOD=fast
ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
ASCEND_RT_VISIBLE_DEVICES=0
ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
VLLM_TORCH_PROFILER_DIR=${SCRIPT_DIR}../ztmp/logs
ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:
ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
VLLM_CONFIGURE_LOGGING=1
CMAKE_PREFIX_PATH=/usr/local/Ascend/cann-8.5.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-8.5.1/lib64/cmake:/usr/local/Ascend/cann-8.5.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-8.5.1/lib64/cmake:/usr/local/Ascend/cann-8.5.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-8.5.1/lib64/cmake
PLAT=linux-aarch64
TOOLCHAIN_HOME=/usr/local/Ascend/cann-8.5.1/toolkit
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1
COMPILE_CUSTOM_KERNELS=1
PYTHONPATH=/usr/local/Ascend/cann-8.5.1/python/site-packages:/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe:/vllm-workspace/vllm:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/vllm-workspace/vllm:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:
VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
LCCL_DETERMINISTIC=0
VLLM_ASCEND_TURBOQUANT_DECODE_OP=0
TASK_QUEUE_ENABLE=1
SHLVL=3
ATB_COMPARE_TILING_EVERY_KERNEL=0
ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.1/opp
LD_LIBRARY_PATH=${SCRIPT_DIR}/../csrc/build:/usr/local/Ascend/cann-8.5.1/lib64:/usr/local/Ascend/cann-8.5.1/lib64/plugin/opskernel:/usr/local/Ascend/cann-8.5.1/lib64/plugin/nnengine:/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/cann-8.5.1/tools/aml/lib64:/usr/local/Ascend/cann-8.5.1/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/python3.11.14/lib:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/python3.11.14/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/python3.11.14/lib::/usr/local/lib
SOC_VERSION=ascend910b3
LCCL_PARALLEL=0
LC_CTYPE=C.UTF-8
ASCEND_AICPU_PATH=/usr/local/Ascend/cann-8.5.1
OMP_NUM_THREADS=1
VLLM_LOGGING_LEVEL=INFO
VLLM_ASCEND_TURBOQUANT_ENCODE_OP=0
ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
PATH=/usr/local/Ascend/cann-8.5.1/bin:/usr/local/Ascend/cann-8.5.1/tools/ccec_compiler/bin:/usr/local/Ascend/cann-8.5.1/tools/profiler/bin:/usr/local/Ascend/cann-8.5.1/tools/ascend_system_advisor/asys:/usr/local/Ascend/cann-8.5.1/tools/show_kernel_debug_data:/usr/local/Ascend/cann-8.5.1/tools/msobjdump:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.11.14/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.11.14/bin:/root/.local/bin:/usr/local/go/bin:/root/usr/node-v24.16.0-linux-arm64/bin:/root/.npm-global/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.11.14/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
CUSTOM_OPP_PATH=${PWD}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
export ASCEND_CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}
ATB_MATMUL_SHUFFLE_K_ENABLE=1
DEBIAN_FRONTEND=noninteractive
ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1

g++ -std=c++17 test_tq_decode.cpp \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/aarch64-linux/include \
    -DVLLM_ASCEND_CUSTOM_OPP_PATH=\"${CUSTOM_OPP_PATH}\" \
    -I${CUSTOM_OPP_PATH}/op_api/include \
    -L${ASCEND_HOME_PATH}/lib64 \
    -L${CUSTOM_OPP_PATH}/op_api/lib \
    -lascendcl -lacl_op_compiler -lcust_opapi -lnnopbase \
    -Wl,-rpath,${CUSTOM_OPP_PATH}/op_api/lib \
    -o ${ROOT_DIR}/ztmp/test_turboquant_decode_paged8bit


