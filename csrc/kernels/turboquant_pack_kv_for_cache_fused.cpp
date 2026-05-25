/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused TurboQuant pack for KV cache (fp16, aligned with TurboQuantMSE v1).
//
// Reference (Python):
//   norms = vector_norm(x, dim=-1, keepdim=True)
//   y = (x / (norms + eps)) @ R^T        # batched [M, D] @ [D, D]
//   d = (y.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
//   idx = d.argmin(dim=-1)
//
// Per AICore: normalize all assigned rows, then ONE matmul [M,128]@[128,128],
// not M separate M=1 matmuls (avoids Cube padding waste).

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "types.h"

using namespace AscendC;

namespace {

static constexpr int TQ_PACK_D = 128;
static constexpr int TQ_PACK_K = 256;
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
// Max rows per core for static Matmul tiling / UB batch buffers.
static constexpr uint32_t TQ_MAX_BATCH_M = 128;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_AIV_SUB_BLOCKS = 2;
static constexpr uint32_t TQ_ROT_N_PER_SUB = TQ_ROT_N / TQ_AIV_SUB_BLOCKS;
static constexpr uint32_t TQ_SINGLE_ROT_M_PAD = 16;
// Broadcast tile: [D, K_TILE] = [128, 128] fits in 32 KiB UB (half).
static constexpr uint32_t TQ_ARGMIN_K_TILE = 128;
static constexpr uint32_t TQ_TILE_DIFF_ELEMS = TQ_PACK_D * TQ_ARGMIN_K_TILE;
static constexpr uint32_t TQ_REDUCE_MIN_WORK = 64;
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr uint32_t TQ_ROT_LOCAL_WORKSPACE_BYTES = TQ_MAX_BATCH_M * TQ_ROT_K * sizeof(half);

__aicore__ inline uint32_t AlignUp16(uint32_t x) {
    return (x + TQ_CUBE_M_ALIGN - 1) / TQ_CUBE_M_ALIGN * TQ_CUBE_M_ALIGN;
}

// csrc/kernels build has no PipeSync / SetWaitFlag; use SetFlag+WaitFlag (moe_gating_top_k).
__aicore__ inline void TqSyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void TqSyncVToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
}

__aicore__ inline void TqSyncSToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
}

__aicore__ inline bool TqDebugEnabled(uint32_t debugLog) {
    if ASCEND_IS_AIV {
        return debugLog != 0;
    }
    return false;
}

__aicore__ inline uint8_t ReadReduceMinIndexHalf(const AscendC::LocalTensor<half>& minPair, uint32_t pairOff) {
    const half idxHalf = minPair.GetValue(pairOff + 1);
    const uint16_t idxU16 = *(const uint16_t*)(&idxHalf);
    return (uint8_t)idxU16;
}

__aicore__ inline void write_norm_fp16_le(
    AscendC::GlobalTensor<uint8_t>& packedGm, uint64_t out_base, half norm_h) {
    union {
        half h;
        uint16_t u;
    } normBits {};
    normBits.h = norm_h;
    packedGm.SetValue(out_base + (uint32_t)TQ_PACK_D, (uint8_t)(normBits.u & 0xFFu));
    packedGm.SetValue(out_base + (uint32_t)TQ_PACK_D + 1, (uint8_t)((normBits.u >> 8) & 0xFFu));
}

__aicore__ inline void CopyTqRotateTiling(TCubeTiling* tiling, __gm__ uint8_t* tilingGm) {
    auto dst = reinterpret_cast<uint32_t*>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

using TqRotateAT = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
using TqRotateBT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using TqRotateCT = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
using TqRotateBiasT = MatmulType<TPosition::GM, CubeFormat::ND, half>;

using TqRotateMatmulOp =
    AscendC::Matmul<TqRotateAT, TqRotateBT, TqRotateCT, TqRotateBiasT>;

class TurboquantPackKVForCacheFused {
public:
    __aicore__ inline explicit TurboquantPackKVForCacheFused(AscendC::TPipe* pipe, TqRotateMatmulOp* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm) {}

    __aicore__ inline void Init(
        __gm__ half* key,
        __gm__ half* value,
        __gm__ half* codebook,
        __gm__ half* rotation_t,
        __gm__ uint8_t* packed_k,
        __gm__ uint8_t* packed_v,
        uint32_t nVec,
        uint32_t slot_w_k,
        uint32_t slot_w_v,
        uint32_t vecPerCore,
        uint32_t debugLog) {
        nVec_ = nVec;
        slot_w_k_ = slot_w_k;
        slot_w_v_ = slot_w_v;
        debugLog_ = debugLog;
        vecPerCore_ = vecPerCore;
        if (vecPerCore_ > TQ_MAX_BATCH_M) {
            vecPerCore_ = TQ_MAX_BATCH_M;
        }

        keyGm_.SetGlobalBuffer(key, (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(value, (uint64_t)nVec_ * TQ_PACK_D);
        codebookGm_.SetGlobalBuffer(codebook, TQ_PACK_K);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        packedKGm_.SetGlobalBuffer(packed_k, (uint64_t)nVec_ * slot_w_k_);
        packedVGm_.SetGlobalBuffer(packed_v, (uint64_t)nVec_ * slot_w_v_);

        const uint32_t batchElems = TQ_MAX_BATCH_M * TQ_PACK_D;
        pipe_->InitBuffer(xBatchQue_, 1, batchElems * sizeof(half));
        pipe_->InitBuffer(yBatchQue_, 1, batchElems * sizeof(half));
        pipe_->InitBuffer(sqBuf_, TQ_PACK_D * sizeof(half));
        pipe_->InitBuffer(reduceWorkBuf_, TQ_PACK_D * sizeof(half));
        pipe_->InitBuffer(normScalarBuf_, TQ_UB_ALIGN);
        pipe_->InitBuffer(normsBuf_, TQ_MAX_BATCH_M * sizeof(half));
        pipe_->InitBuffer(codebookBuf_, TQ_PACK_K * sizeof(half));
        pipe_->InitBuffer(diffTileBuf_, TQ_TILE_DIFF_ELEMS * sizeof(half));
        pipe_->InitBuffer(cbTileBuf_, TQ_TILE_DIFF_ELEMS * sizeof(half));
        pipe_->InitBuffer(reduceOutBuf_, TQ_PACK_D * 2 * sizeof(half));
        pipe_->InitBuffer(argminWorkBuf_, TQ_REDUCE_MIN_WORK * sizeof(half));
        pipe_->InitBuffer(idxBuf_, TQ_PACK_D * sizeof(uint8_t));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_LOCAL_WORKSPACE_BYTES);

        matmulReady_ = GetSysWorkSpacePtr() != nullptr;
    }

    __aicore__ inline void Process() {
        if (!matmulReady_) {
            return;
        }
        const uint32_t core = AscendC::GetBlockIdx() / TQ_AIV_SUB_BLOCKS;
        const uint32_t start = core * vecPerCore_;
        uint32_t end = start + vecPerCore_;
        if (end > nVec_) {
            end = nVec_;
        }
        if (start >= end) {
            return;
        }
        if (TqDebugEnabled(debugLog_)) {
            const uint32_t blockDim = (nVec_ + vecPerCore_ - 1) / vecPerCore_;
            if (core < 8 || core + 1 == blockDim) {
                AscendC::printf(
                    "[TQ_PACK] block=%u/%u start=%u end=%u rows=%u vecPerCore=%u nVec=%u\n",
                    core, blockDim, start, end, end - start, vecPerCore_, nVec_);
            }
        }

        PackBatch(keyGm_, packedKGm_, start, end, slot_w_k_);
        PackBatch(valueGm_, packedVGm_, start, end, slot_w_v_);
    }

private:
    // norms[i] = ||x[i]||; xBatch rows unitized in-place (matches x / (norm + eps)).
    __aicore__ inline void NormalizeBatch(
        AscendC::LocalTensor<half>& xBatch, AscendC::LocalTensor<half>& norms, uint32_t m) {
        auto sqLocal = sqBuf_.Get<half>();
        auto reduceWork = reduceWorkBuf_.Get<half>();
        auto normLocal = normScalarBuf_.Get<half>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            AscendC::Mul(sqLocal, xBatch[rowOff], xBatch[rowOff], TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(normLocal, sqLocal, reduceWork, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(normLocal, normLocal, 1);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            // Scalar half arithmetic is disallowed on AIC; use float on S-pipe then vector Muls.
            const float normF = static_cast<float>(normLocal.GetValue(0));
            TqSyncSToV();

            norms.SetValue(i, normLocal.GetValue(0));
            const half invH = static_cast<half>(1.0f / (normF + TQ_NORM_EPS_F));
            AscendC::Muls(xBatch[rowOff], xBatch[rowOff], invH, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // One batched matmul for all rows on this core (M may be padded to 16 for Cube).
    __aicore__ inline void RotateBatchMatmul(
        AscendC::LocalTensor<half>& xUnitBatch, AscendC::LocalTensor<half>& yBatch, uint32_t mPad) {
        rotateMm_->SetOrgShape(mPad, TQ_ROT_N, TQ_ROT_K);
        rotateMm_->SetTensorA(xUnitBatch, false);
        rotateMm_->SetTensorB(rotationTGm_, false);
        auto rotateWorkspace = rotateWorkBuf_.Get<uint8_t>();
        rotateMm_->SetLocalWorkspace(rotateWorkspace);
        while (rotateMm_->template Iterate<true>()) {
            rotateMm_->template GetTensorC<true>(yBatch, false, true);
        }
        rotateMm_->End();
        AscendC::PipeBarrier<PIPE_V>();
    }

    // d = |y.unsqueeze(-1) - codebook|, idx = argmin(d, dim=-1); K tiled in chunks of 128.
    __aicore__ inline void EncodeRowBroadcast(
        const AscendC::LocalTensor<half>& yRow,
        AscendC::LocalTensor<half>& codebookLocal,
        AscendC::LocalTensor<uint8_t>& idxOut,
        uint32_t dCount) {
        auto diffTile = diffTileBuf_.Get<half>();
        auto cbTile = cbTileBuf_.Get<half>();
        auto reduceOut = reduceOutBuf_.Get<half>();
        auto argminWork = argminWorkBuf_.Get<half>();
        auto bestDist = sqBuf_.Get<half>();

        const uint32_t ySrcShape[2] = {dCount, 1};
        const uint32_t yDstShape[2] = {dCount, TQ_ARGMIN_K_TILE};
        const uint32_t cbSrcShape[2] = {1, TQ_ARGMIN_K_TILE};
        const uint32_t cbDstShape[2] = {dCount, TQ_ARGMIN_K_TILE};
        constexpr uint32_t kRepStride = TQ_ARGMIN_K_TILE / 16;
        const uint32_t diffElems = dCount * TQ_ARGMIN_K_TILE;

        for (uint32_t kBase = 0; kBase < TQ_PACK_K; kBase += TQ_ARGMIN_K_TILE) {
            AscendC::BroadCast<half, 2, 1>(diffTile, yRow, yDstShape, ySrcShape);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BroadCast<half, 2, 0>(cbTile, codebookLocal[kBase], cbDstShape, cbSrcShape);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(diffTile, diffTile, cbTile, diffElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(diffTile, diffTile, diffElems);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceMin<half>(
                reduceOut, diffTile, argminWork, TQ_ARGMIN_K_TILE, dCount, kRepStride, true);
            AscendC::PipeBarrier<PIPE_V>();

            if (kBase == 0) {
                for (uint32_t d = 0; d < dCount; ++d) {
                    const uint32_t pairOff = d * 2;
                    bestDist.SetValue(d, reduceOut.GetValue(pairOff));
                    idxOut.SetValue(d, ReadReduceMinIndexHalf(reduceOut, pairOff));
                }
            } else {
                for (uint32_t d = 0; d < dCount; ++d) {
                    const uint32_t pairOff = d * 2;
                    const float distNewF = static_cast<float>(reduceOut.GetValue(pairOff));
                    const uint8_t idxNew =
                        (uint8_t)(kBase + (uint32_t)ReadReduceMinIndexHalf(reduceOut, pairOff));
                    if (distNewF < static_cast<float>(bestDist.GetValue(d))) {
                        bestDist.SetValue(d, reduceOut.GetValue(pairOff));
                        idxOut.SetValue(d, idxNew);
                    }
                }
            }
        }
    }

    __aicore__ inline void EncodeBatch(
        AscendC::LocalTensor<half>& yBatch,
        AscendC::LocalTensor<half>& norms,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        uint32_t start,
        uint32_t m,
        uint32_t slot_w,
        uint32_t dBase,
        uint32_t dCount,
        bool writeMeta) {
        auto codebookLocal = codebookBuf_.Get<half>();
        AscendC::DataCopy(codebookLocal, codebookGm_[0], TQ_PACK_K);
        TqSyncMte2ToV();

        auto idxLocal = idxBuf_.Get<uint8_t>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t vecIdx = start + i;
            const uint64_t out_base = (uint64_t)vecIdx * slot_w;
            const uint32_t yOff = i * dCount;

            EncodeRowBroadcast(yBatch[yOff], codebookLocal, idxLocal, dCount);

            for (uint32_t j = 0; j < dCount; ++j) {
                packedGm.SetValue(out_base + dBase + j, idxLocal.GetValue(j));
            }
            if (writeMeta) {
                write_norm_fp16_le(packedGm, out_base, norms.GetValue(i));

                for (uint32_t k = (uint32_t)TQ_PACK_D + 2; k < slot_w; ++k) {
                    packedGm.SetValue(out_base + k, (uint8_t)0);
                }
            }
        }
    }

    __aicore__ inline void PackBatch(
        const AscendC::GlobalTensor<half>& xGm,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        uint32_t start,
        uint32_t end,
        uint32_t slot_w) {
        const uint32_t m = end - start;
        const uint32_t mPad = AlignUp16(m);
        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] PackBatch enter start=%u end=%u m=%u mPad=%u slot_w=%u\n",
                            start, end, m, mPad, slot_w);
        }

        auto xBatch = xBatchQue_.AllocTensor<half>();
        auto yBatch = yBatchQue_.AllocTensor<half>();
        auto norms = normsBuf_.Get<half>();

        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] before DataCopy x\n");
        }
        AscendC::DataCopy(xBatch, xGm[(uint64_t)start * TQ_PACK_D], m * TQ_PACK_D);
        TqSyncMte2ToV();
        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] after DataCopy x\n");
        }

        if (mPad > m) {
            AscendC::Duplicate(xBatch[m * TQ_PACK_D], (half)0, (mPad - m) * TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] before Normalize\n");
        }
        NormalizeBatch(xBatch, norms, m);
        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] after Normalize before Matmul\n");
        }
        RotateBatchMatmul(xBatch, yBatch, mPad);
        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] after Matmul before Encode\n");
        }
        const uint32_t subIdx = AscendC::GetSubBlockIdx() % TQ_AIV_SUB_BLOCKS;
        const uint32_t dBase = subIdx * TQ_ROT_N_PER_SUB;
        EncodeBatch(yBatch, norms, packedGm, start, m, slot_w, dBase, TQ_ROT_N_PER_SUB, subIdx == 0);
        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK] after Encode\n");
        }

        yBatchQue_.FreeTensor(yBatch);
        xBatchQue_.FreeTensor(xBatch);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    TqRotateMatmulOp* rotateMm_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t slot_w_k_ = 0;
    uint32_t slot_w_v_ = 0;
    uint32_t vecPerCore_ = 1;
    uint32_t debugLog_ = 0;
    bool matmulReady_ = false;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> xBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yBatchQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cbTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> argminWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateWorkBuf_;

    AscendC::GlobalTensor<half> keyGm_;
    AscendC::GlobalTensor<half> valueGm_;
    AscendC::GlobalTensor<half> codebookGm_;
    AscendC::GlobalTensor<half> rotationTGm_;
    AscendC::GlobalTensor<uint8_t> packedKGm_;
    AscendC::GlobalTensor<uint8_t> packedVGm_;
};

class TurboquantPackKVForCacheNoKfc {
public:
    __aicore__ inline explicit TurboquantPackKVForCacheNoKfc(AscendC::TPipe* pipe, TqRotateMatmulOp* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm) {}

    __aicore__ inline void Init(
        __gm__ half* key,
        __gm__ half* value,
        __gm__ half* codebook,
        __gm__ half* rotation_t,
        __gm__ uint8_t* packed_k,
        __gm__ uint8_t* packed_v,
        uint32_t nVec,
        uint32_t slot_w_k,
        uint32_t slot_w_v,
        uint32_t vecPerCore,
        uint32_t debugLog) {
        nVec_ = nVec;
        slot_w_k_ = slot_w_k;
        slot_w_v_ = slot_w_v;
        vecPerCore_ = vecPerCore;
        debugLog_ = debugLog;

        keyGm_.SetGlobalBuffer(key, (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(value, (uint64_t)nVec_ * TQ_PACK_D);
        codebookGm_.SetGlobalBuffer(codebook, TQ_PACK_K);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        packedKGm_.SetGlobalBuffer(packed_k, (uint64_t)nVec_ * slot_w_k_);
        packedVGm_.SetGlobalBuffer(packed_v, (uint64_t)nVec_ * slot_w_v_);

        pipe_->InitBuffer(xQue_, 1, TQ_SINGLE_ROT_M_PAD * TQ_PACK_D * sizeof(half));
        pipe_->InitBuffer(yQue_, 1, TQ_SINGLE_ROT_M_PAD * TQ_ROT_N * sizeof(half));
        pipe_->InitBuffer(sqBuf_, TQ_PACK_D * sizeof(half));
        pipe_->InitBuffer(reduceWorkBuf_, TQ_PACK_D * sizeof(half));
        pipe_->InitBuffer(normScalarBuf_, TQ_UB_ALIGN);
        pipe_->InitBuffer(codebookBuf_, TQ_PACK_K * sizeof(half));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_LOCAL_WORKSPACE_BYTES);

        matmulReady_ = GetSysWorkSpacePtr() != nullptr;
    }

    __aicore__ inline void Process() {
        if (!matmulReady_) {
            return;
        }
        const uint32_t core = AscendC::GetBlockIdx() / TQ_AIV_SUB_BLOCKS;
        const uint32_t start = core * vecPerCore_;
        uint32_t end = start + vecPerCore_;
        if (end > nVec_) {
            end = nVec_;
        }
        if (start >= end) {
            return;
        }

        auto codebookLocal = codebookBuf_.Get<half>();
        AscendC::DataCopy(codebookLocal, codebookGm_[0], TQ_PACK_K);
        TqSyncMte2ToV();

        if (TqDebugEnabled(debugLog_)) {
            AscendC::printf("[TQ_PACK_NOKFC] block=%u start=%u end=%u vecPerCore=%u nVec=%u\n",
                            core, start, end, vecPerCore_, nVec_);
        }

        const uint32_t subIdx = AscendC::GetSubBlockIdx() % TQ_AIV_SUB_BLOCKS;
        const uint32_t dBase = subIdx * TQ_ROT_N_PER_SUB;
        for (uint32_t row = start; row < end; ++row) {
            PackOneRow(keyGm_, packedKGm_, codebookLocal, row, slot_w_k_, dBase, TQ_ROT_N_PER_SUB, subIdx == 0);
            PackOneRow(valueGm_, packedVGm_, codebookLocal, row, slot_w_v_, dBase, TQ_ROT_N_PER_SUB, subIdx == 0);
        }
    }

private:
    __aicore__ inline half LoadAndNormalizeRow(
        const AscendC::GlobalTensor<half>& xGm,
        AscendC::LocalTensor<half>& xLocal,
        uint32_t row) {
        auto sqLocal = sqBuf_.Get<half>();
        auto reduceWork = reduceWorkBuf_.Get<half>();
        auto normLocal = normScalarBuf_.Get<half>();

        AscendC::DataCopy(xLocal, xGm[(uint64_t)row * TQ_PACK_D], TQ_PACK_D);
        TqSyncMte2ToV();
        AscendC::Mul(sqLocal, xLocal, xLocal, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(normLocal, sqLocal, reduceWork, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(normLocal, normLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();
        TqSyncVToS();
        const float normF = static_cast<float>(normLocal.GetValue(0));
        TqSyncSToV();
        const half normH = normLocal.GetValue(0);
        const half invH = static_cast<half>(1.0f / (normF + TQ_NORM_EPS_F));
        AscendC::Muls(xLocal, xLocal, invH, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        return normH;
    }

    __aicore__ inline void RotateRowMatmul(
        AscendC::LocalTensor<half>& xLocal,
        AscendC::LocalTensor<half>& yLocal) {
        AscendC::Duplicate(xLocal[TQ_PACK_D], (half)0, (TQ_SINGLE_ROT_M_PAD - 1) * TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        rotateMm_->SetOrgShape(TQ_SINGLE_ROT_M_PAD, TQ_ROT_N, TQ_ROT_K);
        rotateMm_->SetTensorA(xLocal, false);
        rotateMm_->SetTensorB(rotationTGm_, false);
        auto rotateWorkspace = rotateWorkBuf_.Get<uint8_t>();
        rotateMm_->SetLocalWorkspace(rotateWorkspace);
        while (rotateMm_->template Iterate<true>()) {
            rotateMm_->template GetTensorC<true>(yLocal, false, true);
        }
        rotateMm_->End();
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantizeAndWrite(
        const AscendC::LocalTensor<half>& yLocal,
        const AscendC::LocalTensor<half>& codebookLocal,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        uint64_t outBase,
        uint32_t dBase,
        uint32_t dCount) {
        for (uint32_t d = 0; d < dCount; ++d) {
            const float y = static_cast<float>(yLocal.GetValue(d));
            float bestDist = 3.402823466e38f;
            uint8_t bestIdx = 0;
            for (uint32_t c = 0; c < TQ_PACK_K; ++c) {
                const float diff = y - static_cast<float>(codebookLocal.GetValue(c));
                const float dist = diff * diff;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<uint8_t>(c);
                }
            }
            packedGm.SetValue(outBase + dBase + d, bestIdx);
        }
    }

    __aicore__ inline void PackOneRow(
        const AscendC::GlobalTensor<half>& xGm,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const AscendC::LocalTensor<half>& codebookLocal,
        uint32_t row,
        uint32_t slot_w,
        uint32_t dBase,
        uint32_t dCount,
        bool writeMeta) {
        auto xLocal = xQue_.AllocTensor<half>();
        auto yLocal = yQue_.AllocTensor<half>();
        const half normH = LoadAndNormalizeRow(xGm, xLocal, row);
        RotateRowMatmul(xLocal, yLocal);

        const uint64_t outBase = (uint64_t)row * slot_w;
        QuantizeAndWrite(yLocal, codebookLocal, packedGm, outBase, dBase, dCount);
        if (writeMeta) {
            write_norm_fp16_le(packedGm, outBase, normH);
            for (uint32_t k = (uint32_t)TQ_PACK_D + 2; k < slot_w; ++k) {
                packedGm.SetValue(outBase + k, (uint8_t)0);
            }
        }
        yQue_.FreeTensor(yLocal);
        xQue_.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    TqRotateMatmulOp* rotateMm_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t slot_w_k_ = 0;
    uint32_t slot_w_v_ = 0;
    uint32_t vecPerCore_ = 1;
    uint32_t debugLog_ = 0;
    bool matmulReady_ = false;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> xQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateWorkBuf_;

    AscendC::GlobalTensor<half> keyGm_;
    AscendC::GlobalTensor<half> valueGm_;
    AscendC::GlobalTensor<half> codebookGm_;
    AscendC::GlobalTensor<half> rotationTGm_;
    AscendC::GlobalTensor<uint8_t> packedKGm_;
    AscendC::GlobalTensor<uint8_t> packedVGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache_fused_fp16_8bit_128(
    __gm__ half* key,
    __gm__ half* value,
    __gm__ half* codebook,
    __gm__ half* rotation_t,
    __gm__ uint8_t* packed_k,
    __gm__ uint8_t* packed_v,
    uint32_t nVec,
    uint32_t slot_w_k,
    uint32_t slot_w_v,
    uint32_t vecPerCore,
    uint32_t debugLog,
    GM_ADDR workspace,
    __gm__ uint8_t* rotate_tiling) {
    // vllm-ascend device build uses __aicore__ + KERNEL_TASK_TYPE for 1C2V MIX (not __mix__).
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // REGIST is a collective KFC handshake: AIC enters as server; both AIV sub-blocks
    // enter as clients. After REGIST, each AIV consumes its own N/2 slice.
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] kernel enter block=%u sub=%u nVec=%u vecPerCore=%u\n",
                        AscendC::GetBlockIdx(), AscendC::GetSubBlockIdx(), nVec, vecPerCore);
    }
    AscendC::SetSysWorkspace(workspace);
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] after kfc workspace bind\n");
    }
    AscendC::TPipe pipe;
    TqRotateMatmulOp rotateMm;
    if (GetSysWorkSpacePtr() == nullptr) {
        if (TqDebugEnabled(debugLog)) {
            AscendC::printf("[TQ_PACK] sys workspace null\n");
        }
        return;
    }
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] before REGIST_MATMUL_OBJ\n");
    }
    // REGIST_MATMUL_OBJ requires a normal TCubeTiling*; each MIX participant keeps its own stack copy.
    TCubeTiling tiling;
    CopyTqRotateTiling(&tiling, rotate_tiling);
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf(
            "[TQ_PACK] rotate tiling usedCore=%d M=%d N=%d Ka=%d Kb=%d singleM=%d singleN=%d singleK=%d\n",
            tiling.usedCoreNum, tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.singleCoreM,
            tiling.singleCoreN, tiling.singleCoreK);
    }
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &tiling);
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] after REGIST_MATMUL_OBJ\n");
    }
    // AIC path returns inside REGIST_MATMUL_OBJ; both AIV sub-blocks reach the fused body.
    TurboquantPackKVForCacheFused op(&pipe, &rotateMm);
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] before op.Init\n");
    }
    op.Init(key, value, codebook, rotation_t, packed_k, packed_v, nVec, slot_w_k, slot_w_v, vecPerCore, debugLog);
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] after op.Init before Process\n");
    }
    op.Process();
    if (TqDebugEnabled(debugLog)) {
        AscendC::printf("[TQ_PACK] after Process\n");
    }
}

extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache_fused_fp16_8bit_128_nokfc(
    __gm__ half* key,
    __gm__ half* value,
    __gm__ half* codebook,
    __gm__ half* rotation_t,
    __gm__ uint8_t* packed_k,
    __gm__ uint8_t* packed_v,
    uint32_t nVec,
    uint32_t slot_w_k,
    uint32_t slot_w_v,
    uint32_t vecPerCore,
    uint32_t debugLog,
    GM_ADDR workspace,
    __gm__ uint8_t* rotate_tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspace(workspace);
    AscendC::TPipe pipe;
    TqRotateMatmulOp rotateMm;
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    TCubeTiling tiling;
    CopyTqRotateTiling(&tiling, rotate_tiling);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &tiling);
    TurboquantPackKVForCacheNoKfc op(&pipe, &rotateMm);
    op.Init(key, value, codebook, rotation_t, packed_k, packed_v, nVec, slot_w_k, slot_w_v, vecPerCore, debugLog);
    op.Process();
}

namespace vllm_ascend {

extern void turboquant_pack_kv_for_cache_fused_fp16_8bit_128_impl(
    void* stream,
    void* key,
    void* value,
    void* codebook,
    void* rotation_t,
    void* packed_k,
    void* packed_v,
    uint32_t nVec,
    uint32_t slot_w_k,
    uint32_t slot_w_v,
    uint32_t vecPerCore,
    uint32_t debugLog,
    void* workspace,
    void* rotate_tiling) {
    uint32_t blockDim = (nVec + vecPerCore - 1) / vecPerCore;
    turboquant_pack_kv_for_cache_fused_fp16_8bit_128<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<__gm__ half*>(key),
        reinterpret_cast<__gm__ half*>(value),
        reinterpret_cast<__gm__ half*>(codebook),
        reinterpret_cast<__gm__ half*>(rotation_t),
        reinterpret_cast<__gm__ uint8_t*>(packed_k),
        reinterpret_cast<__gm__ uint8_t*>(packed_v),
        nVec,
        slot_w_k,
        slot_w_v,
        vecPerCore,
        debugLog,
        reinterpret_cast<uint8_t*>(workspace),
        reinterpret_cast<__gm__ uint8_t*>(rotate_tiling));
}

extern void turboquant_pack_kv_for_cache_fused_fp16_8bit_128_nokfc_impl(
    void* stream,
    void* key,
    void* value,
    void* codebook,
    void* rotation_t,
    void* packed_k,
    void* packed_v,
    uint32_t nVec,
    uint32_t slot_w_k,
    uint32_t slot_w_v,
    uint32_t vecPerCore,
    uint32_t debugLog,
    void* workspace,
    void* rotate_tiling) {
    uint32_t blockDim = (nVec + vecPerCore - 1) / vecPerCore;
    turboquant_pack_kv_for_cache_fused_fp16_8bit_128_nokfc<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<__gm__ half*>(key),
        reinterpret_cast<__gm__ half*>(value),
        reinterpret_cast<__gm__ half*>(codebook),
        reinterpret_cast<__gm__ half*>(rotation_t),
        reinterpret_cast<__gm__ uint8_t*>(packed_k),
        reinterpret_cast<__gm__ uint8_t*>(packed_v),
        nVec,
        slot_w_k,
        slot_w_v,
        vecPerCore,
        debugLog,
        reinterpret_cast<uint8_t*>(workspace),
        reinterpret_cast<__gm__ uint8_t*>(rotate_tiling));
}

}  // namespace vllm_ascend
