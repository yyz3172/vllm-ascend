#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -d build ]]; then
    find build -type d -name vllm_ascend_kernels_preprocess-prefix \
        -prune -print -exec rm -rf {} +
fi

export COMPILE_CUSTOM_KERNELS="${COMPILE_CUSTOM_KERNELS:-1}"
export MAX_JOBS="${MAX_JOBS:-64}"

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" setup.py build_ext --inplace "$@"
