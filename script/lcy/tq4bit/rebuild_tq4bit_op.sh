#!/usr/bin/env bash
set -euo pipefail

# Build and test turboquant_attention_paged4bit (including Key 1 Cube QK/PV)
# inside the vllm-ascend container.
#
# Usage:
#   bash script/lcy/tq4bit/rebuild_tq4bit_op.sh                            # rebuild op only
#   bash script/lcy/tq4bit/rebuild_tq4bit_op.sh --sim 0                    # rebuild op + compile test_tq4bit + run sim
#   bash script/lcy/tq4bit/rebuild_tq4bit_op.sh --run 0                    # rebuild op + compile test_tq4bit + run binary
#   bash script/lcy/tq4bit/rebuild_tq4bit_op.sh --run 0 --no-build         # skip op rebuild, compile test_tq4bit + run binary
#   bash script/lcy/tq4bit/rebuild_tq4bit_op.sh --run 0 --no-compile-test  # skip op rebuild and test compile, run binary only

CONTAINER="vllm-ascend-lcy-v018"
WORKDIR="/root/l00856060/code/vllm018/vllm-ascend"
SOC_VERSION="ascend910b"

# Environment init commands run inside the container
INIT_ENV="source /root/l00856060/venv/lcyvllm/.venv/bin/activate && source ${WORKDIR}/infoenvs"

REBUILD_OP=true
BUILD_TEST=true
TEST_MODE=""
TEST_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim)
      TEST_MODE="sim"
      TEST_KEY="${2:-0}"
      shift 2
      ;;
    --run)
      TEST_MODE="run"
      TEST_KEY="${2:-0}"
      shift 2
      ;;
    --no-build)
      REBUILD_OP=false
      shift
      ;;
    --no-compile-test)
      BUILD_TEST=false
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ "${REBUILD_OP}" == "false" && "${BUILD_TEST}" == "false" && -z "${TEST_MODE}" ]]; then
  echo "[rebuild] ERROR: --no-build/--no-compile-test requires --sim or --run."
  exit 1
fi

if [[ -n "${TEST_MODE}" && "${BUILD_TEST}" == "false" && ! -f "${WORKDIR}/test_tq4bit" ]]; then
  echo "[rebuild] ERROR: test_tq4bit binary not found at ${WORKDIR}/test_tq4bit"
  echo "[rebuild]        Remove --no-compile-test or compile it first with script/lcy/tq4bit/build_test_tq4bit.sh"
  exit 1
fi

# Step 1: Rebuild op (optional)
if [[ "${REBUILD_OP}" == "true" ]]; then
  echo "[rebuild] === Step 1: Rebuilding op in container ${CONTAINER} ==="
  docker exec -w "${WORKDIR}" "${CONTAINER}" bash -c "
      ${INIT_ENV} &&
      cd csrc &&
      rm -rf build output &&
      bash build.sh -n 'turboquant_pack_kv_for_cache4bit;turboquant_attention_paged4bit' -c ${SOC_VERSION} &&
      ./output/CANN-custom_ops-*.run --install-path='${WORKDIR}/vllm_ascend/_cann_ops_custom' &&
      echo '[rebuild] Op rebuilt and installed.'
  " || { echo "[rebuild] ERROR: Op rebuild failed."; exit 1; }
fi

# Step 2: Compile test_tq4bit.cpp (optional)
if [[ -n "${TEST_MODE}" && "${BUILD_TEST}" == "true" ]]; then
  echo "[rebuild] === Step 2: Compiling test_tq4bit.cpp ==="
  docker exec -w "${WORKDIR}" "${CONTAINER}" bash -c "
      ${INIT_ENV} &&
      bash script/lcy/tq4bit/build_test_tq4bit.sh &&
      echo '[rebuild] test_tq4bit binary compiled.'
  " || { echo "[rebuild] ERROR: Test binary compilation failed."; exit 1; }
fi

# Step 3: Run test (optional)
if [[ -n "${TEST_MODE}" ]]; then
  echo "[rebuild] === Step 3: Running sim_test_tq4bit for key ${TEST_KEY} (mode=${TEST_MODE}) ==="
  docker exec -w "${WORKDIR}" "${CONTAINER}" bash -c "
      ${INIT_ENV} &&
      bash script/lcy/tq4bit/sim_test_tq4bit.sh ${TEST_KEY} ${TEST_MODE}
  "
fi

echo "[rebuild] Done."
