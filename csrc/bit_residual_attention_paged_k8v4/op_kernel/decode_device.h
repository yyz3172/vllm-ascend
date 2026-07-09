/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You can obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// BitResidual K8V4 decode primitives for fused attention.
//
// Key decode: code = (q7 << 1) | sign
//   sign_val = sign ? +1/sqrt(D) : -1/sqrt(D)   (D=128, INV_SQRT_D = 0.08839)
//   err = base + q7 * step                        (7-bit residual)
//   y = sign_val + err
//   decoded_K = y * norm                          (broadcast per-row)
//
// Value decode: idx4 ∈ [0,15]
//   y = vmin + idx4 * vstep                       (4-bit uniform)
//   No norm for value (value cache does not store norm).

#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace bit_residual_attn {

using namespace AscendC;

// BitResidual K8V4 cache layout constants (must match pack kernel IMPL_DESIGN.md).
static constexpr uint32_t TQ_BR_HEAD_SIZE = 128;
static constexpr uint32_t TQ_BR_KEY_GROUP_ROWS = 2;
static constexpr uint32_t TQ_BR_VALUE_GROUP_ROWS = 4;
static constexpr uint32_t TQ_BR_GROUP_STRIDE = 288;
static constexpr uint32_t TQ_BR_GROUP_INDEX_WORDS = TQ_BR_HEAD_SIZE;
static constexpr uint32_t TQ_BR_GROUP_INDEX_BYTES = TQ_BR_HEAD_SIZE * sizeof(uint16_t);  // 256

static constexpr float TQ_BR_INV_SQRT_D = 0.08838834764831845f;  // 1/sqrt(128)
static constexpr float TQ_BR_NORM_EPS = 1e-10f;

// Key group layout (288 bytes per 2-row group):
//   byte[0..255]   = uint16[128] packed code, 2 rows
//   byte[256..259] = uint16[2] norms (fp16/bf16), row0, row1
//   byte[260..267] = float[2] base, row0, row1
//   byte[268..275] = float[2] step, row0, row1
//   byte[276..287] = padding

// Value group layout (288 bytes per 4-row group):
//   byte[0..255]   = uint16[128] packed idx4, 4 rows
//   byte[256..271] = float[4] vmin, row0..row3
//   byte[272..287] = float[4] vstep, row0..row3

// Offsets within a 288-byte group for metadata fields.
static constexpr uint32_t TQ_BR_KEY_NORM_BYTE_OFFSET = 256;
static constexpr uint32_t TQ_BR_KEY_BASE_BYTE_OFFSET = 260;
static constexpr uint32_t TQ_BR_KEY_STEP_BYTE_OFFSET = 268;

static constexpr uint32_t TQ_BR_VAL_VMIN_BYTE_OFFSET = 256;
static constexpr uint32_t TQ_BR_VAL_VSTEP_BYTE_OFFSET = 272;

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

// Read a fp16/bf16 norm scalar from a byte region in UB and return as float.
template <typename T>
__aicore__ inline float TqBrReadNormToFloat(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& normScratch,
    const AscendC::LocalTensor<float>& floatScratch,
    uint32_t byteOffset) {
    const uint16_t lo = static_cast<uint16_t>(packed.GetValue(byteOffset));
    const uint16_t hi = static_cast<uint16_t>(packed.GetValue(byteOffset + 1));
    normScratch.template ReinterpretCast<uint16_t>().SetValue(
        0, static_cast<uint16_t>(lo | (hi << 8)));
    TqBrSync<AscendC::HardEvent::S_V>();
    AscendC::Cast(floatScratch, normScratch, AscendC::RoundMode::CAST_NONE, 1);
    AscendC::PipeBarrier<PIPE_V>();
    TqBrSync<AscendC::HardEvent::V_S>();
    const float norm = floatScratch.GetValue(0);
    TqBrSync<AscendC::HardEvent::S_V>();
    return norm;
}

}  // namespace bit_residual_attn
