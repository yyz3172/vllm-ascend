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
// DecodeRows8bit: given packed (M, 130) UB rows, emit xHat (M, 128) fp16 UB by
//   y_hat = codebook[idx]                 (256-entry LUT, vectorized Gather)
//   y_hat = y_hat * norm                  (per-row broadcast)
//   x_hat = y_hat @ R                     (Cube, KFC, R stationary in L1)
// Multiplying ``norm`` (a per-row scalar) before the matmul is equivalent to the
// PyTorch reference's post-matmul scaling because the matmul is linear in each
// row: (c * y) @ R == c * (y @ R). This lets the Cube input already carry the
// norm so no extra pass over x_hat is needed.
//
// Phase 1 (decode_paged_8bit): the kernel does GM<->UB movement + compact block
//   addressing, then calls DecodeRows8bit and writes xHat to the workspace.
// Phase 2 (fused attention, scheme B): the attention kernel calls DecodeRows8bit
//   per KV tile and feeds xHat straight into the Q*K / P*V Cube — no workspace.
//
// A fully scalar reference (DecodeRowsScalar) backs the AIV-only tiling key for
// numeric bring-up / golden comparison (design doc §2.6.5.8).

#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace turboquant {

static constexpr uint32_t TQ_DECODE_HEAD_SIZE = 128;
static constexpr uint32_t TQ_DECODE_PACKED_BYTES = TQ_DECODE_HEAD_SIZE + 2;  // 130
static constexpr uint32_t TQ_DECODE_CODEBOOK_SIZE = 256;
static constexpr uint32_t TQ_DECODE_NORM_OFFSET = TQ_DECODE_HEAD_SIZE;       // byte 128..129 in GM row

// KFC matmul type five-tuple — identical to the (validated) pack op
// (csrc/turboquant_pack_kv_for_cache_fused): A in VECOUT UB, B/C in GM.
// fp32 C accumulate (then cast to fp16) avoids fp16 K=128 accumulation error.
using TqDecodeRotateAT = AscendC::MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, half>;
using TqDecodeRotateBT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using TqDecodeRotateCT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using TqDecodeRotateBiasT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using TqDecodeRotateMatmulOp =
    AscendC::Matmul<TqDecodeRotateAT, TqDecodeRotateBT, TqDecodeRotateCT, TqDecodeRotateBiasT>;

// csrc/kernels build has no PipeSync helpers; use SetFlag+WaitFlag (pack op style).
// HardEvent is a compile-time template arg for SetFlag/WaitFlag.
template <AscendC::HardEvent EVT>
__aicore__ inline void TqDecodeSync() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(e);
    AscendC::WaitFlag<EVT>(e);
}

// Reinterpret 2 little-endian bytes in UB as fp16.
__aicore__ inline half TqDecodeReadNorm(const AscendC::LocalTensor<uint8_t>& packed, uint32_t normByteOffset) {
    const uint16_t lo = static_cast<uint16_t>(packed.GetValue(normByteOffset));
    const uint16_t hi = static_cast<uint16_t>(packed.GetValue(normByteOffset + 1));
    union {
        uint16_t u;
        half h;
    } bits {};
    bits.u = static_cast<uint16_t>(lo | (hi << 8));
    return bits.h;
}

// Vectorized 8-bit decode for M packed rows. Caller must have:
//   - Compacted ``packed`` into UB as [M*128 index bytes][M*2 norm bytes],
//     with the index region DataCopied and ready for MTE2->V sync,
//   - ``codebook`` (256 fp16) resident in UB,
//   - ``rotationGm`` (128x128 fp16) in GM with ``rotateMm`` REGIST_MATMUL_OBJ'd,
//   - ``cubeCGm`` a fp32 GM scratch of >= M*128 floats.
// Scratch tensors (idxHalf/idxFloat/idxS32/yHat[VECOUT]/cubeFp32/xHat) sized M*128.
// On return ``xHat`` (UB fp16, M*128) holds the decoded rows row-major.
__aicore__ inline void DecodeRows8bit(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<half>& codebook,
    AscendC::GlobalTensor<half>& rotationGm,
    AscendC::GlobalTensor<float>& cubeCGm,
    TqDecodeRotateMatmulOp& rotateMm,
    const AscendC::LocalTensor<half>& idxHalf,
    const AscendC::LocalTensor<float>& idxFloat,
    const AscendC::LocalTensor<int32_t>& idxS32,
    const AscendC::LocalTensor<half>& yHat,
    const AscendC::LocalTensor<uint8_t>& rotateWork,
    const AscendC::LocalTensor<float>& cubeFp32,
    const AscendC::LocalTensor<half>& xHat,
    uint32_t M,
    uint32_t mPad) {
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t n = M * D;

    // (1) idx bytes are compacted by LoadPackedTile, so V can widen the full
    //     [M, 128] region directly without scalar GetValue/SetValue loops.
    TqDecodeSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(idxHalf, packed, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();

    // (2) idx -> int32 byte offsets into the fp16 codebook (idx * sizeof(half)).
    AscendC::Cast(idxFloat, idxHalf, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(idxS32, idxFloat, AscendC::RoundMode::CAST_RINT, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(idxS32, idxS32, static_cast<int32_t>(sizeof(half)), n);
    AscendC::PipeBarrier<PIPE_V>();

    // (3) codebook LUT: y_hat = codebook[idx] (256-entry, fits Vector LUT).
    AscendC::Gather(yHat, codebook, idxS32.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), n);
    AscendC::PipeBarrier<PIPE_V>();

    // (4) per-row norms read from the compact norm tail (scalar; packed already in UB).
    half normArr[64];  // >= max T_rows
    const uint32_t normBase = M * D;
    for (uint32_t i = 0; i < M; ++i) {
        normArr[i] = TqDecodeReadNorm(packed, normBase + i * sizeof(half));
    }

    // (5) y_hat *= norm (broadcast over the feature dim). The Gather output and
    //     this Muls are both on PIPE_V so the prior barrier orders them.
    for (uint32_t i = 0; i < M; ++i) {
        AscendC::Muls(yHat[i * D], yHat[i * D], normArr[i], D);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // Zero-pad the Cube A tail when M is not 16-aligned (compact blocks keep M ==
    // T_rows so this is normally a no-op, but stay safe for a ragged last tile).
    if (mPad > M) {
        AscendC::Duplicate(yHat[n], static_cast<half>(0), (mPad - M) * D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // (6) Cube: x_hat = y_hat @ R, fp32 accumulate to GM, then back to UB + fp16.
    // SetSingleShape(singleM, singleN, singleK): A[M,K] @ B[K,N], doc requires
    // tensorA >= singleM*singleK, tensorB >= singleK*singleN (in elements).
    rotateMm.SetOrgShape(mPad, D, D);
    rotateMm.SetSingleShape(M, D, D);
    rotateMm.SetTensorA(yHat, false);
    rotateMm.SetTensorB(rotationGm, false);
    rotateMm.SetLocalWorkspace(rotateWork);
    // Match pack op: non-sequential GetTensorC per baseN tile assembles full [mPad,N]
    // in cubeCGm; linear DataCopy(m*D) is valid. IterateAll does not reproduce this
    // layout on MIX KFC (CANN: IterateAll expects continuous GM).
    // while (rotateMm.Iterate()) {
    //     rotateMm.GetTensorC(cubeCGm);
    //     iterCount++;
    // }
    rotateMm.IterateAll(cubeCGm);
    rotateMm.End();

    AscendC::DataCopy(cubeFp32, cubeCGm, n);
    TqDecodeSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(xHat, cubeFp32, AscendC::RoundMode::CAST_NONE, n);
    AscendC::PipeBarrier<PIPE_V>();
}

// Fully scalar reference decode (AIV-only). No Gather, no Cube — used as the
// numeric golden / debug fallback (design doc §2.6.5.8). codebook + rotation are
// read scalar from UB; caller must issue S->MTE3 before copying xHat to GM.
__aicore__ inline void DecodeRowsScalar(
    const AscendC::LocalTensor<uint8_t>& packed,
    const AscendC::LocalTensor<half>& codebook,
    const AscendC::LocalTensor<half>& rotation,
    const AscendC::LocalTensor<half>& xHat,
    uint32_t M) {
    const uint32_t D = TQ_DECODE_HEAD_SIZE;
    const uint32_t normBase = M * D;
    for (uint32_t i = 0; i < M; ++i) {
        const uint32_t rowByteBase = i * D;
        const float norm = static_cast<float>(TqDecodeReadNorm(packed, normBase + i * sizeof(half)));
        // y_hat[k] = codebook[idx[k]]
        half yhat[TQ_DECODE_HEAD_SIZE];
        for (uint32_t k = 0; k < D; ++k) {
            const uint8_t idx = packed.GetValue(rowByteBase + k);
            yhat[k] = codebook.GetValue(idx);
        }
        // x_hat[n] = (sum_k y_hat[k] * R[k, n]) * norm
        for (uint32_t nn = 0; nn < D; ++nn) {
            float acc = 0.f;
            for (uint32_t k = 0; k < D; ++k) {
                acc += static_cast<float>(yhat[k]) *
                       static_cast<float>(rotation.GetValue(k * D + nn));
            }
            xHat.SetValue(i * D + nn, static_cast<half>(acc * norm));
        }
    }
}

}  // namespace turboquant
