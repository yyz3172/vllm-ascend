/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// BitResidual K8V4 decode primitives for fused attention.
//
// Key decode (sign-reversal quantization, no normalization):
//   sign_bit = code >> 7 (0=positive, 1=negative)
//   q7 = code & 0x7f
//   sig_vec[d] = 1 - 2*sign_bit → {+1, -1}
//   err = base + q7 * step (all positive residual)
//   decoded_K = err * sig_vec (restore original sign per dimension)
//
// Value decode (4-bit uniform, raw vmin/vstep without norm folding):
//   y = vmin + idx4 * vstep

#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace bit_residual_attn {

using namespace AscendC;

// BitResidual K8V4 cache layout constants (must match pack kernel).
static constexpr uint32_t TQ_BR_HEAD_SIZE = 128;
static constexpr uint32_t TQ_BR_BLOCK_ROWS = 16;
static constexpr uint32_t TQ_BR_KEY_GROUP_ROWS = 2;
static constexpr uint32_t TQ_BR_VAL_GROUP_ROWS = 4;
static constexpr uint32_t TQ_BR_GROUP_INDEX_WORDS = TQ_BR_HEAD_SIZE;
static constexpr uint32_t TQ_BR_GROUP_INDEX_BYTES =
    TQ_BR_GROUP_INDEX_WORDS * sizeof(uint16_t);

// 16-row sub-block layout:
static constexpr uint32_t TQ_BR_KEY_BLOCK_CODE_BYTES =
    (TQ_BR_BLOCK_ROWS / TQ_BR_KEY_GROUP_ROWS) * TQ_BR_GROUP_INDEX_BYTES;
static constexpr uint32_t TQ_BR_KEY_BLOCK_BASE_BYTES = TQ_BR_BLOCK_ROWS * sizeof(uint16_t);
static constexpr uint32_t TQ_BR_KEY_BLOCK_STEP_BYTES = TQ_BR_BLOCK_ROWS * sizeof(uint16_t);
static constexpr uint32_t TQ_BR_KEY_BLOCK_BASE_OFFSET = TQ_BR_KEY_BLOCK_CODE_BYTES;
static constexpr uint32_t TQ_BR_KEY_BLOCK_STEP_OFFSET = TQ_BR_KEY_BLOCK_BASE_OFFSET + TQ_BR_KEY_BLOCK_BASE_BYTES;
static constexpr uint32_t TQ_BR_KEY_BLOCK_STRIDE = TQ_BR_KEY_BLOCK_CODE_BYTES + TQ_BR_KEY_BLOCK_BASE_BYTES + TQ_BR_KEY_BLOCK_STEP_BYTES;  // 2176

static constexpr uint32_t TQ_BR_VAL_BLOCK_CODE_BYTES =
    (TQ_BR_BLOCK_ROWS / TQ_BR_VAL_GROUP_ROWS) * TQ_BR_GROUP_INDEX_BYTES;
static constexpr uint32_t TQ_BR_VAL_BLOCK_VMIN_BYTES = TQ_BR_BLOCK_ROWS * sizeof(uint16_t);
static constexpr uint32_t TQ_BR_VAL_BLOCK_VSTEP_BYTES = TQ_BR_BLOCK_ROWS * sizeof(uint16_t);
static constexpr uint32_t TQ_BR_VAL_BLOCK_VMIN_OFFSET = TQ_BR_VAL_BLOCK_CODE_BYTES;
static constexpr uint32_t TQ_BR_VAL_BLOCK_VSTEP_OFFSET = TQ_BR_VAL_BLOCK_VMIN_OFFSET + TQ_BR_VAL_BLOCK_VMIN_BYTES;
static constexpr uint32_t TQ_BR_VAL_BLOCK_STRIDE = TQ_BR_VAL_BLOCK_CODE_BYTES + TQ_BR_VAL_BLOCK_VMIN_BYTES + TQ_BR_VAL_BLOCK_VSTEP_BYTES;  // 1152

// KFC matmul type five-tuple for Q/output rotation.
template <typename T>
using TqRotateAT = AscendC::MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, T>;
template <typename T>
using TqRotateBT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
template <typename T>
using TqRotateCT = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, T>;
template <typename T>
using TqRotateBiasT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;

__aicore__ inline constexpr MatmulConfig TqRotateMatmulConfig() {
    constexpr MatmulShapeParams shapeParams = {
        64, TQ_BR_HEAD_SIZE, TQ_BR_HEAD_SIZE,
        64, TQ_BR_HEAD_SIZE, TQ_BR_HEAD_SIZE};
    constexpr MatmulBiasParams biasParams = {false};
    return GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams, biasParams);
}

template <typename T>
__aicore__ inline constexpr MatmulApiStaticTiling TqRotateMatmulTiling() {
    MatmulApiStaticTiling tiling =
        GetMatmulApiTiling<TqRotateAT<T>, TqRotateBT<T>,
                           TqRotateCT<T>, TqRotateBiasT<T>>(
            TqRotateMatmulConfig());
    tiling.usedCoreNum = 1;
    return tiling;
}

template <typename T>
static constexpr MatmulApiStaticTiling TQ_BR_ROTATE_MATMUL_TILING =
    TqRotateMatmulTiling<T>();

template <typename T>
using TqRotateMatmulOp =
    AscendC::Matmul<TqRotateAT<T>, TqRotateBT<T>, TqRotateCT<T>,
                    TqRotateBiasT<T>, TQ_BR_ROTATE_MATMUL_TILING<T>>;

// Prefill qTile Cube QK reuses TqRotateMatmulOp (same KFC object as rotation):
//   C[M,N] = A[M,K] @ B[K,N] with B = physical K^T in GM (SetTensorB false).
// M covers qTile*GQA (16*2); rotate MatmulConfig singleM max is 64.
static constexpr uint32_t TQ_BR_QK_CUBE_MAX_M = 32;
static constexpr uint32_t TQ_BR_QK_CUBE_MAX_N = 64;

// Pipe sync helper (reuses TPipe for event ID, matching the pack kernel pattern).
template <AscendC::HardEvent EVT>
__aicore__ inline void TqBrSync() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(e);
    AscendC::WaitFlag<EVT>(e);
}

// Read a float32 scalar from a byte region in UB (for base/step/vmin/vstep).
__aicore__ inline float TqBrReadFloatFromU8(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<float>& floatScratch,
    uint32_t byteOffset) {
    (void)floatScratch;
    auto packedF32 = packed.template ReinterpretCast<float>();
    return packedF32.GetValue(byteOffset / sizeof(float));
}

template <typename T>
__aicore__ inline float TqBrRead16FromU8(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<float>& floatScratch,
    uint32_t byteOffset) {
    auto packedT = packed.template ReinterpretCast<T>();
    AscendC::Cast(floatScratch, packedT[byteOffset / sizeof(T)],
                  AscendC::RoundMode::CAST_NONE, 1);
    TqBrSync<AscendC::HardEvent::V_S>();
    return floatScratch.GetValue(0);
}

}  // namespace bit_residual_attn
