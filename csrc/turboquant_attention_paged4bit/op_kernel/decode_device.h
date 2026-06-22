/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Phase 1 / Phase 2 reuse contract — design doc §2.6.9.
//
// DecodeRows4bit: given compact unpacked UB rows, emit xHat (M, 128) fp16/bf16 UB by
//   y_hat = codebook[idx]                 (16-entry LUT, vectorized Gather)
//   y_hat = y_hat * norm                  (per-row broadcast)
//   x_hat = y_hat @ R                     (Cube, KFC, R stationary in L1)
// Multiplying ``norm`` (a per-row scalar) before the matmul is equivalent to the
// PyTorch reference's post-matmul scaling because the matmul is linear in each
// row: (c * y) @ R == c * (y @ R). This lets the Cube input already carry the
// norm so no extra pass over x_hat is needed.
//
// Phase 1 (decode_paged_4bit): the kernel does GM<->UB movement + compact block
//   addressing, then calls DecodeRows4bit and writes xHat to the workspace.
// Phase 2 (fused attention, scheme B): the attention kernel calls DecodeRows4bit
//   per KV tile and feeds xHat straight into the Q*K / P*V Cube — no workspace.
//
// A fully scalar reference (DecodeRowsScalar) backs the AIV-only tiling key for
// numeric bring-up / golden comparison (design doc §2.6.5.8).

#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace turboquant {

static constexpr uint32_t TQ_DECODE_HEAD_SIZE = 128;
static constexpr uint32_t TQ_DECODE_INDEX_BYTES = TQ_DECODE_HEAD_SIZE / 2;  // uint4 packed indices in GM
static constexpr uint32_t TQ_DECODE_PACKED_BYTES = TQ_DECODE_INDEX_BYTES + 2;  // 66-byte slab row
static constexpr uint32_t TQ_DECODE_COMPACT_BYTES = TQ_DECODE_HEAD_SIZE + 2;  // UB: 128 idx + 2 norm
static constexpr uint32_t TQ_DECODE_CODEBOOK_SIZE = 16;
static constexpr uint32_t TQ_DECODE_NORM_OFFSET = TQ_DECODE_INDEX_BYTES;  // byte 64..65 in GM row

static constexpr uint32_t TQ_DECODE_DTYPE_BYTES = sizeof(uint16_t);
static constexpr float TQ_DECODE_FY_LINEAR = 0.020799f;
static constexpr float TQ_DECODE_FY_CUBIC = 0.0001926f;

template <typename T>
__aicore__ inline void TqDecodeDuplicateZero(
    const AscendC::LocalTensor<T>& dst, uint32_t count) {
    AscendC::Duplicate(dst.template ReinterpretCast<uint16_t>(),
                       static_cast<uint16_t>(0), count);
}

// KFC matmul type five-tuple — identical to the validated pack v2 op:
// A in VECOUT UB, B in GM, C in VECIN UB. Keep the rotate result local so the
// attention path does not round-trip through GM scratch.
template <typename T>
using TqDecodeRotateAT = AscendC::MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, T>;
template <typename T>
using TqDecodeRotateBT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
template <typename T>
using TqDecodeRotateCT = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, T>;
template <typename T>
using TqDecodeRotateBiasT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;

template <typename T>
using TqDecodeRotateMatmulOp =
    AscendC::Matmul<TqDecodeRotateAT<T>, TqDecodeRotateBT<T>, TqDecodeRotateCT<T>, TqDecodeRotateBiasT<T>>;

// csrc/kernels build has no PipeSync helpers; use SetFlag+WaitFlag (pack op style).
// HardEvent is a compile-time template arg for SetFlag/WaitFlag.
template <AscendC::HardEvent EVT>
__aicore__ inline void TqDecodeSync() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(e);
    AscendC::WaitFlag<EVT>(e);
}

// Reinterpret 2 little-endian bytes in UB as the decode dtype.
template <typename T>
__aicore__ inline T TqDecodeReadNorm(const AscendC::LocalTensor<uint8_t>& packed, uint32_t normByteOffset) {
    const uint16_t lo = static_cast<uint16_t>(packed.GetValue(normByteOffset));
    const uint16_t hi = static_cast<uint16_t>(packed.GetValue(normByteOffset + 1));
    union {
        uint16_t u;
        T v;
    } bits {};
    bits.u = static_cast<uint16_t>(lo | (hi << 8));
    return bits.v;
}

template <typename T>
__aicore__ inline float TqDecodeNormToFloat(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& normScratch,
    const AscendC::LocalTensor<float>& floatScratch,
    uint32_t normByteOffset) {
    const uint16_t lo = static_cast<uint16_t>(packed.GetValue(normByteOffset));
    const uint16_t hi = static_cast<uint16_t>(packed.GetValue(normByteOffset + 1));
    normScratch.template ReinterpretCast<uint16_t>().SetValue(
        0, static_cast<uint16_t>(lo | (hi << 8)));
    TqDecodeSync<AscendC::HardEvent::S_V>();
    AscendC::Cast(floatScratch, normScratch, AscendC::RoundMode::CAST_NONE, 1);
    AscendC::PipeBarrier<PIPE_V>();
    TqDecodeSync<AscendC::HardEvent::V_S>();
    const float norm = floatScratch.GetValue(0);
    TqDecodeSync<AscendC::HardEvent::S_V>();
    return norm;
}

__aicore__ inline float TqDecodeFyScalar(int32_t idx) {
    const float x = static_cast<float>(idx) - 7.5f;
    return TQ_DECODE_FY_CUBIC * x * x * x + TQ_DECODE_FY_LINEAR * x;
}

template <typename T>
__aicore__ inline void TqDecodeFyVectorFromFloat(
    const AscendC::LocalTensor<T>& yHat,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    uint32_t n) {
    auto x = idxFloat;
    auto x2 = idxS32.template ReinterpretCast<float>();
    AscendC::Adds(x, x, -7.5f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x2, x, x, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(x2, x2, TQ_DECODE_FY_CUBIC, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(x2, x2, TQ_DECODE_FY_LINEAR, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x, x, x2, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(yHat, x, AscendC::RoundMode::CAST_RINT, n);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void TqDecodeFyVectorFloatInPlace(
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    uint32_t n)
{
    auto x = idxFloat;
    auto x2 = idxS32.template ReinterpretCast<float>();
    AscendC::Adds(x, x, -7.5f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x2, x, x, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(x2, x2, TQ_DECODE_FY_CUBIC, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(x2, x2, TQ_DECODE_FY_LINEAR, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x, x, x2, n);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TqDecodeFyVector(
    const AscendC::LocalTensor<T>& yHat,
    const AscendC::LocalTensor<half>& idxHalf,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    uint32_t n) {
    (void)idxHalf;
    TqDecodeFyVectorFromFloat(yHat, idxFloat, idxS32, n);
}

// Vectorized 4-bit decode for M packed rows. Caller must have:
//   - Unpacked ``packed`` into UB as [M*128 index bytes][M*2 norm bytes],
//     with the index region DataCopied and ready for MTE2->V sync,
//   - ``codebook`` (16 fp16/bf16) resident in UB,
//   - ``rotationGm`` (128x128 fp16/bf16) in GM with ``rotateMm`` REGIST_MATMUL_OBJ'd.
// Scratch tensors (idxHalf/idxFloat/idxS32/yHat[VECOUT]/xHat[VECIN]) sized M*128.
// On return ``xHat`` (UB fp16/bf16, M*128) holds the decoded rows row-major.
template <typename T>
__aicore__ inline void DecodeRows4bit(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& codebook,
    AscendC::GlobalTensor<T>& rotationGm,
    TqDecodeRotateMatmulOp<T>& rotateMm,
    const AscendC::LocalTensor<half>& idxHalf,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    AscendC::TQue<AscendC::TPosition::VECOUT, 1>& yHatQue,
    const AscendC::LocalTensor<uint8_t>& rotateWork,
    const AscendC::LocalTensor<T>& xHat,
    uint32_t M,
    uint32_t mPad) {
    (void)codebook;
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t n = M * D;

    // (1) idx bytes are compacted by LoadPackedTile, so V can widen the full
    //     [M, 128] region directly without scalar GetValue/SetValue loops.
    TqDecodeSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(idxHalf, packed, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();

    // (2) idx -> fy(idx - 7.5). This experiment replaces the 16-entry
    //     codebook LUT with the analytic cubic used to generate it.
    AscendC::Cast(idxFloat, idxHalf, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();

    auto yHat = yHatQue.template AllocTensor<T>();
    TqDecodeFyVector(yHat, idxHalf, idxFloat, idxS32, n);

    // (4) per-row norms read from the compact norm tail (scalar; packed already in UB).
    float normArr[64];  // >= max T_rows
    const uint32_t normBase = M * D;
    auto normScratch = rotateWork.template ReinterpretCast<T>();
    auto normFloat = idxFloat;
    for (uint32_t i = 0; i < M; ++i) {
        normArr[i] = TqDecodeNormToFloat<T>(
            packed, normScratch, normFloat, normBase + i * sizeof(T));
    }

    // (5) y_hat *= norm (broadcast over the feature dim). The Gather output and
    //     this Muls are both on PIPE_V so the prior barrier orders them.
    for (uint32_t i = 0; i < M; ++i) {
        auto rowFloat = idxFloat[i * D];
        AscendC::Cast(rowFloat, yHat[i * D], AscendC::RoundMode::CAST_NONE, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(rowFloat, rowFloat, normArr[i], D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yHat[i * D], rowFloat, AscendC::RoundMode::CAST_RINT, D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Zero-pad the Cube A tail when M is not 16-aligned (compact blocks keep M ==
    // T_rows so this is normally a no-op, but stay safe for a ragged last tile).
    if (mPad > M) {
        TqDecodeDuplicateZero(yHat[n], (mPad - M) * D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // (6) Cube: x_hat = y_hat @ R, with local C output consumed directly by attention.
    // SetSingleShape(singleM, singleN, singleK): A[M,K] @ B[K,N], doc requires
    // tensorA >= singleM*singleK, tensorB >= singleK*singleN (in elements).
    yHatQue.EnQue(yHat);
    auto yHatReady = yHatQue.template DeQue<T>();
    rotateMm.SetOrgShape(mPad, D, D);
    rotateMm.SetSingleShape(M, D, D);
    rotateMm.SetTensorA(yHatReady, false);
    rotateMm.SetTensorB(rotationGm, false);
    rotateMm.SetLocalWorkspace(rotateWork);
    rotateMm.IterateAll(xHat);
    rotateMm.End();
    yHatQue.FreeTensor(yHatReady);
}

template <typename T>
__aicore__ inline void DecodeRows4bitFromIdxHalf(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& codebook,
    AscendC::GlobalTensor<T>& rotationGm,
    TqDecodeRotateMatmulOp<T>& rotateMm,
    const AscendC::LocalTensor<half>& idxHalf,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    AscendC::TQue<AscendC::TPosition::VECOUT, 1>& yHatQue,
    const AscendC::LocalTensor<uint8_t>& rotateWork,
    const AscendC::LocalTensor<T>& xHat,
    uint32_t M,
    uint32_t mPad) {
    (void)codebook;
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t n = M * D;

    AscendC::Cast(idxFloat, idxHalf, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();

    auto yHat = yHatQue.template AllocTensor<T>();
    TqDecodeFyVector(yHat, idxHalf, idxFloat, idxS32, n);

    float normArr[64];
    const uint32_t normBase = M * D;
    auto normScratch = rotateWork.template ReinterpretCast<T>();
    auto normFloat = idxFloat;
    for (uint32_t i = 0; i < M; ++i) {
        normArr[i] = TqDecodeNormToFloat<T>(
            packed, normScratch, normFloat, normBase + i * sizeof(T));
    }
    for (uint32_t i = 0; i < M; ++i) {
        auto rowFloat = idxFloat[i * D];
        AscendC::Cast(rowFloat, yHat[i * D], AscendC::RoundMode::CAST_NONE, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(rowFloat, rowFloat, normArr[i], D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yHat[i * D], rowFloat, AscendC::RoundMode::CAST_RINT, D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    if (mPad > M) {
        TqDecodeDuplicateZero(yHat[n], (mPad - M) * D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    yHatQue.EnQue(yHat);
    auto yHatReady = yHatQue.template DeQue<T>();
    rotateMm.SetOrgShape(mPad, D, D);
    rotateMm.SetSingleShape(M, D, D);
    rotateMm.SetTensorA(yHatReady, false);
    rotateMm.SetTensorB(rotationGm, false);
    rotateMm.SetLocalWorkspace(rotateWork);
    rotateMm.IterateAll(xHat);
    rotateMm.End();
    yHatQue.FreeTensor(yHatReady);
}

template <typename T, bool RESTORE_NORM>
__aicore__ inline void DecodeRows4bitFromIdxHalfNoRotate(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& codebook,
    const AscendC::LocalTensor<half>& idxHalf,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    const AscendC::LocalTensor<uint8_t>& rotateWork,
    const AscendC::LocalTensor<float>& normOut,
    const AscendC::LocalTensor<T>& yOut,
    uint32_t M) {
    (void)codebook;
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t n = M * D;

    AscendC::Cast(idxFloat, idxHalf, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();
    TqDecodeFyVector(yOut, idxHalf, idxFloat, idxS32, n);

    const uint32_t normBase = M * D;
    (void)rotateWork;
    auto normLocal = packed[normBase].template ReinterpretCast<T>();
    AscendC::Cast(normOut, normLocal, AscendC::RoundMode::CAST_NONE, M);
    AscendC::PipeBarrier<PIPE_V>();
    if (RESTORE_NORM) {
        for (uint32_t i = 0; i < M; ++i) {
            auto rowFloat = idxFloat[i * D];
            AscendC::Cast(rowFloat, yOut[i * D], AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(rowFloat, rowFloat, normOut.GetValue(i), D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yOut[i * D], rowFloat, AscendC::RoundMode::CAST_RINT, D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T, bool RESTORE_NORM>
__aicore__ inline void DecodeRows4bitFromIdxFloatNoRotate(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& codebook,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    const AscendC::LocalTensor<uint8_t>& rotateWork,
    const AscendC::LocalTensor<float>& normOut,
    const AscendC::LocalTensor<T>& yOut,
    uint32_t M) {
    (void)codebook;
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t n = M * D;

    TqDecodeFyVectorFromFloat(yOut, idxFloat, idxS32, n);

    const uint32_t normBase = M * D;
    (void)rotateWork;
    auto normLocal = packed[normBase].template ReinterpretCast<T>();
    AscendC::Cast(normOut, normLocal, AscendC::RoundMode::CAST_NONE, M);
    AscendC::PipeBarrier<PIPE_V>();
    if (RESTORE_NORM) {
        for (uint32_t i = 0; i < M; ++i) {
            auto rowFloat = idxFloat[i * D];
            AscendC::Cast(rowFloat, yOut[i * D], AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(rowFloat, rowFloat, normOut.GetValue(i), D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yOut[i * D], rowFloat, AscendC::RoundMode::CAST_RINT, D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

// Fully scalar reference decode (AIV-only). No Gather, no Cube — used as the
// numeric golden / debug fallback (design doc §2.6.5.8). codebook + rotation are
// read scalar from UB; caller must issue S->MTE3 before copying xHat to GM.
template <typename T>
__aicore__ inline void DecodeRowsScalar(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<T>& codebook,
    const AscendC::LocalTensor<T>& rotation,
    const AscendC::LocalTensor<T>& xHat,
    const AscendC::LocalTensor<float>& scalarFloat,
    uint32_t M) {
    (void)codebook;
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t normBase = M * D;
    auto rotationRowFloat = scalarFloat;
    auto outScalar = scalarFloat[D];
    auto normScratch = scalarFloat[D * 2].template ReinterpretCast<T>();
    auto outRowFloat = scalarFloat[D * 3];
    for (uint32_t i = 0; i < M; ++i) {
        const uint32_t rowByteBase = i * D;
        const float norm = TqDecodeNormToFloat<T>(
            packed, normScratch, outScalar, normBase + i * sizeof(T));

        float acc[TQ_DECODE_HEAD_SIZE];
        for (uint32_t nn = 0; nn < D; ++nn) {
            acc[nn] = 0.f;
        }
        // x_hat[n] = (sum_k y_hat[k] * R[k, n]) * norm
        for (uint32_t k = 0; k < D; ++k) {
            const uint8_t idx = packed.GetValue(rowByteBase + k);
            const int32_t idxSigned = static_cast<int32_t>(idx);
            const float y = TqDecodeFyScalar(idxSigned);
            AscendC::Cast(rotationRowFloat, rotation[k * D],
                          AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
            TqDecodeSync<AscendC::HardEvent::V_S>();
            for (uint32_t nn = 0; nn < D; ++nn) {
                acc[nn] += y * rotationRowFloat.GetValue(nn);
            }
            TqDecodeSync<AscendC::HardEvent::S_V>();
        }
        for (uint32_t nn = 0; nn < D; ++nn) {
            outRowFloat.SetValue(nn, acc[nn] * norm);
        }
        TqDecodeSync<AscendC::HardEvent::S_V>();
        AscendC::Cast(xHat[i * D], outRowFloat,
                      AscendC::RoundMode::CAST_RINT, D);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

}  // namespace turboquant
