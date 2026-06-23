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

// Fused TurboQuant pack for KV cache (fp16/bf16, aligned with TurboQuantMSE v1).
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

using namespace AscendC;

namespace {

static constexpr int TQ_PACK_D = 128;
static constexpr int TQ_PACK_K = 16;
static constexpr uint32_t TQ_PACKED_INDEX_BYTES = TQ_PACK_D / 2;
static constexpr uint32_t TQ_ROW_BYTES = TQ_PACKED_INDEX_BYTES + 2;
static constexpr uint32_t TQ_GROUP_ROWS = 4;
static constexpr uint32_t TQ_GROUP_INDEX_BYTES = TQ_PACK_D * sizeof(uint16_t);
static constexpr uint32_t TQ_GROUP_BYTES = TQ_GROUP_ROWS * TQ_ROW_BYTES;
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_GROUP_STRIDE =
    (TQ_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_ENCODED_ROW_WORDS = TQ_PACK_D + 1;
static constexpr uint32_t TQ_ENCODED_ROW_BYTES = TQ_ENCODED_ROW_WORDS * sizeof(uint16_t);
static constexpr uint32_t TQ_ENCODED_ROW_STRIDE_BYTES =
    (TQ_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_ENCODED_ROW_STRIDE_WORDS =
    TQ_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
// Match the v2 pack kernel: keeps UB pressure low while still amortizing Cube.
static constexpr uint32_t TQ_MAX_BATCH_M = 32;
static constexpr uint32_t TQ_QUEUE_DEPTH = 2;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_DTYPE_BYTES = sizeof(uint16_t);
static constexpr uint32_t TQ_NORM_STRIDE = TQ_UB_ALIGN / TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_ROT_LOCAL_WORKSPACE_BYTES = TQ_MAX_BATCH_M * TQ_ROT_K * TQ_DTYPE_BYTES;
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr uint32_t TQ_DIST_TILE_ELEMS = TQ_PACK_D;
static constexpr uint32_t TQ_ARGMIN_INDEX_BYTES = TQ_PACK_D * sizeof(int32_t);
static constexpr uint32_t TQ_ARGMIN_INDEX_U16_BYTES = TQ_PACK_D * sizeof(int16_t);
static constexpr uint32_t TQ_AIV_SUB_BLOCKS = 2;
static constexpr uint32_t TQ_COMPARE_MASK_BYTES = 256;
static constexpr float TQ_FY_LINEAR = 0.020799f;
static constexpr float TQ_FY_CUBIC = 0.0001926f;
static constexpr uint32_t TQ_PACK_MODE_GENERAL = 0;
static constexpr uint32_t TQ_PACK_MODE_DECODE_DIRECT = 1;
static constexpr uint32_t TQ_PACK_MODE_LOGICAL_FAST_FALLBACK = 2;

#if defined(ORIG_DTYPE_KEY)
#if (ORIG_DTYPE_KEY == DT_BF16)
using TqDataT = bfloat16_t;
#else
using TqDataT = half;
#endif
#elif defined(DTYPE_KEY)
#if (DTYPE_KEY == DT_BF16)
using TqDataT = bfloat16_t;
#else
using TqDataT = half;
#endif
#else
using TqDataT = half;
#endif

__aicore__ inline uint32_t AlignUp16(uint32_t x) {
    return (x + TQ_CUBE_M_ALIGN - 1) / TQ_CUBE_M_ALIGN * TQ_CUBE_M_ALIGN;
}

// csrc/kernels build has no PipeSync / SetWaitFlag; use SetFlag+WaitFlag (moe_gating_top_k).
__aicore__ inline void TqSyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void TqSyncMte2ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e);
    WaitFlag<HardEvent::MTE2_S>(e);
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

__aicore__ inline void TqSyncSToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

__aicore__ inline void TqSyncVToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void TqSyncMte3ToMte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

__aicore__ inline float TqFyScalar(float x) {
    return TQ_FY_CUBIC * x * x * x + TQ_FY_LINEAR * x;
}

__aicore__ inline float TqQuantThreshold(uint32_t code) {
    switch (code) {
        case 1:
            return TqFyScalar(-7.0f);
        case 2:
            return TqFyScalar(-6.0f);
        case 3:
            return TqFyScalar(-5.0f);
        case 4:
            return TqFyScalar(-4.0f);
        case 5:
            return TqFyScalar(-3.0f);
        case 6:
            return TqFyScalar(-2.0f);
        case 7:
            return TqFyScalar(-1.0f);
        case 8:
            return TqFyScalar(0.0f);
        case 9:
            return TqFyScalar(1.0f);
        case 10:
            return TqFyScalar(2.0f);
        case 11:
            return TqFyScalar(3.0f);
        case 12:
            return TqFyScalar(4.0f);
        case 13:
            return TqFyScalar(5.0f);
        case 14:
            return TqFyScalar(6.0f);
        case 15:
            return TqFyScalar(7.0f);
        default:
            return TqFyScalar(0.0f);
    }
}

__aicore__ inline float TqCodeAsFloat(uint32_t code) {
    switch (code) {
        case 1:
            return 1.0f;
        case 2:
            return 2.0f;
        case 3:
            return 3.0f;
        case 4:
            return 4.0f;
        case 5:
            return 5.0f;
        case 6:
            return 6.0f;
        case 7:
            return 7.0f;
        case 8:
            return 8.0f;
        case 9:
            return 9.0f;
        case 10:
            return 10.0f;
        case 11:
            return 11.0f;
        case 12:
            return 12.0f;
        case 13:
            return 13.0f;
        case 14:
            return 14.0f;
        case 15:
            return 15.0f;
        default:
            return 0.0f;
    }
}

template <typename T>
__aicore__ inline void write_tq16_le_local(
    AscendC::LocalTensor<uint8_t>& packedLocal, uint32_t byte_offset, T value) {
    union {
        T v;
        uint16_t u;
    } bits {};
    bits.v = value;
    packedLocal.SetValue(byte_offset, (uint8_t)(bits.u & 0xFFu));
    packedLocal.SetValue(byte_offset + 1, (uint8_t)((bits.u >> 8) & 0xFFu));
}

// 66B rows are not 32B-aligned; use DataCopyPad for GM writes.
__aicore__ inline void copy_packed_ub_to_gm(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t>& packedLocal,
    uint32_t nbytes) {
    TqSyncVToMte3();
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
    TqSyncMte3ToMte2();
}

__aicore__ inline void copy_packed_gm_to_ub(
    AscendC::LocalTensor<uint8_t>& packedLocal,
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    uint32_t nbytes) {
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(packedLocal, packedGm[gm_offset], copyParams, padParams);
    TqSyncMte2ToV();
    TqSyncMte2ToS();
}

template <typename T>
using TqRotateAT = MatmulType<TPosition::VECOUT, CubeFormat::ND, T>;
template <typename T>
using TqRotateBT = MatmulType<TPosition::GM, CubeFormat::ND, T>;
template <typename T>
using TqRotateCT = MatmulType<TPosition::VECIN, CubeFormat::ND, T>;
template <typename T>
using TqRotateBiasT = MatmulType<TPosition::GM, CubeFormat::ND, T>;

template <typename T>
using TqRotateMatmulOp =
    AscendC::Matmul<TqRotateAT<T>, TqRotateBT<T>, TqRotateCT<T>, TqRotateBiasT<T>>;

template <typename T, typename RotateMmT>
class TurboquantPackKVForCache4bitToCache {
public:
    __aicore__ inline explicit TurboquantPackKVForCache4bitToCache(AscendC::TPipe* pipe, RotateMmT* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm) {}

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ T* codebook,
        __gm__ T* rotation_t,
        __gm__ int32_t* slot_mapping,
        __gm__ uint8_t* key_cache,
        __gm__ uint8_t* value_cache,
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t vecPerCore,
        uint32_t numHeads,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t dataCores,
        uint32_t packMode) {
        nVec_ = nVec;
        vecPerCore_ = vecPerCore;
        numHeads_ = numHeads == 0 ? 1 : numHeads;
        tokenCount_ = (nVec_ + numHeads_ - 1) / numHeads_;
        blockSize_ = blockSize == 0 ? 1 : blockSize;
        numBlocks_ = numBlocks;
        cacheSlots_ = blockSize_ * numBlocks_;
        dataCores_ = dataCores == 0 ? 1 : dataCores;
        packMode_ = packMode;
        if (vecPerCore_ > TQ_MAX_BATCH_M) {
            vecPerCore_ = TQ_MAX_BATCH_M;
        }

        (void)codebook;
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key), (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), (uint64_t)nVec_ * TQ_PACK_D);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        slotMappingGm_.SetGlobalBuffer(slot_mapping, tokenCount_);
        keyCacheGm_.SetGlobalBuffer(key_cache, (uint64_t)numBlocks_ * numHeads_ * blockSize_ * TQ_ROW_BYTES);
        valueCacheGm_.SetGlobalBuffer(value_cache, (uint64_t)numBlocks_ * numHeads_ * blockSize_ * TQ_ROW_BYTES);

        const uint32_t batchElems = TQ_MAX_BATCH_M * TQ_PACK_D;
        pipe_->InitBuffer(xBatchQue_, TQ_QUEUE_DEPTH, batchElems * sizeof(T));
        pipe_->InitBuffer(aBatchQue_, TQ_QUEUE_DEPTH, batchElems * sizeof(T));
        pipe_->InitBuffer(yBatchQue_, TQ_QUEUE_DEPTH, batchElems * sizeof(T));
        pipe_->InitBuffer(normScalarBuf_, TQ_UB_ALIGN);
        pipe_->InitBuffer(normsBuf_, TQ_MAX_BATCH_M * TQ_NORM_STRIDE * sizeof(T));
        pipe_->InitBuffer(quantMaskBuf_, TQ_COMPARE_MASK_BYTES);
        pipe_->InitBuffer(distBuf_, TQ_DIST_TILE_ELEMS * sizeof(float));
        // yFp32Buf: fp32 y row [D=128]
        pipe_->InitBuffer(yFp32Buf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(argminIndexBuf_, TQ_ARGMIN_INDEX_BYTES);
        pipe_->InitBuffer(argminIndexU16Buf_, TQ_ARGMIN_INDEX_U16_BYTES);
        // NormalizeBatch: fp32 row + fp32 squared row + fp32 ReduceSum tmp.
        pipe_->InitBuffer(reduceOutBuf_, TQ_PACK_D * 3 * sizeof(float));
        pipe_->InitBuffer(packedRowBuf_, TQ_GROUP_STRIDE * sizeof(uint8_t));
        pipe_->InitBuffer(packMergeBuf_, TQ_GROUP_INDEX_BYTES);
        pipe_->InitBuffer(packMaskBuf_, TQ_GROUP_INDEX_BYTES);
        pipe_->InitBuffer(encodedBatchQue_, TQ_QUEUE_DEPTH,
                          TQ_MAX_BATCH_M * TQ_ENCODED_ROW_STRIDE_BYTES);
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_LOCAL_WORKSPACE_BYTES);

        matmulReady_ = (rawWorkspace != nullptr);
    }

    __aicore__ inline void Process() {
        if (!matmulReady_) {
            return;
        }
        if ASCEND_IS_AIC {
            return;
        }
        // Assign each physical 4-row cache group to exactly one worker.  A
        // worker then gathers all token-major input rows for its groups, so no
        // other core can race on the same packed group RMW.
        const uint32_t worker = AscendC::GetBlockIdx();
        if (worker >= GetActiveWorkers()) {
            return;
        }
        if (packMode_ == TQ_PACK_MODE_DECODE_DIRECT) {
            ProcessCachePairByDirectTasks(worker);
            return;
        }
        if (packMode_ == TQ_PACK_MODE_LOGICAL_FAST_FALLBACK) {
            ProcessCachePairByLogicalFastFallback(worker);
            return;
        }
        ProcessCachePairByOwnedGroups(worker);
    }

private:

    // norms[i] = ||x[i]||; xBatch rows unitized in-place (matches x / (norm + eps)).
    // Inner dim: Cast + Mul + ReduceSum (vector), not scalar loop; fp32 acc avoids 16-bit overflow.
    __aicore__ inline void NormalizeBatch(uint32_t m) {
        auto xBatch = xBatchQue_.DeQue<T>();
        auto aBatch = aBatchQue_.AllocTensor<T>();
        auto norms = normsBuf_.Get<T>();
        auto fp32Row = reduceOutBuf_.Get<float>();
        auto fp32Square = reduceOutBuf_.Get<float>()[TQ_PACK_D];
        auto fp32Tmp = reduceOutBuf_.Get<float>()[TQ_PACK_D * 2];
        auto normAcc = normScalarBuf_.Get<float>();

        TqSyncMte2ToV();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;

            AscendC::Cast(fp32Row, xBatch[rowOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(fp32Square, fp32Row, fp32Row, TQ_PACK_D);
            AscendC::ReduceSum<float>(normAcc, fp32Square, fp32Tmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(normAcc, normAcc, 1);
            TqSyncVToS();
            const float normF = normAcc.GetValue(0);
            TqSyncSToV();

            AscendC::Cast(norms[i * TQ_NORM_STRIDE], normAcc, AscendC::RoundMode::CAST_RINT, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(fp32Row, fp32Row, 1.0f / (normF + TQ_NORM_EPS_F), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(xBatch[rowOff], fp32Row, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::DataCopy(aBatch, xBatch, m * TQ_PACK_D);

        aBatchQue_.EnQue(aBatch);
        xBatchQue_.FreeTensor(xBatch);
    }

    // Cube Matmul: T A(VECOUT) x T B(GM) -> T C(VECIN).
    // Keep rotate output in local memory, aligned with the v2 pack kernel.
    __aicore__ inline void RotateBatchMatmul(
        uint32_t m,
        uint32_t dBase,
        uint32_t dCount) {
        auto aBatch = aBatchQue_.DeQue<T>();
        auto yBatch = yBatchQue_.AllocTensor<T>();
        uint32_t mPad = AlignUp16(m);
        if (mPad < TQ_CUBE_M_ALIGN) {
            mPad = TQ_CUBE_M_ALIGN;
        }
        if (mPad > m) {
            AscendC::Duplicate(
                aBatch[m * TQ_PACK_D].template ReinterpretCast<uint16_t>(),
                static_cast<uint16_t>(0),
                (mPad - m) * TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        rotateMm_->SetOrgShape(mPad, TQ_ROT_N, TQ_ROT_K);
        rotateMm_->SetSingleShape(m, dCount, TQ_ROT_K);
        rotateMm_->SetTensorA(aBatch, false);
        rotateMm_->SetTensorB(rotationTGm_[dBase], false);
        auto rotateWorkspace = rotateWorkBuf_.Get<uint8_t>();
        rotateMm_->SetLocalWorkspace(rotateWorkspace);
        // IterateAll: single atomic KFC message per AIV worker.
        rotateMm_->IterateAll(yBatch);
        rotateMm_->End();

        yBatchQue_.EnQue(yBatch);
        aBatchQue_.FreeTensor(aBatch);
    }

    // Encoded local row layout used between Compute and CopyOut:
    //   uint16[0..127] = uint4 index widened to uint16 for vector pack merge
    //   uint16[128] = norm bits matching T
    __aicore__ inline void EncodeBatch(uint32_t m) {
        auto yBatch = yBatchQue_.DeQue<T>();
        auto encodedBatch = encodedBatchQue_.AllocTensor<uint16_t>();
        auto norms = normsBuf_.Get<T>();
        auto yFp32 = yFp32Buf_.Get<float>();
        auto qFloat = distBuf_.Get<float>();
        auto argminIndex = argminIndexBuf_.Get<int32_t>();
        auto qCandidate = argminIndex.template ReinterpretCast<float>();
        auto argminIndexU16 = argminIndexU16Buf_.Get<int16_t>();
        auto quantMask = quantMaskBuf_.Get<uint8_t>();
        auto argminMask = packMaskBuf_.Get<int16_t>();

        AscendC::Duplicate(argminMask, static_cast<int16_t>(0x000F), TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_ENCODED_ROW_STRIDE_WORDS;

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(qFloat, 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t code = 1; code < TQ_PACK_K; ++code) {
                AscendC::CompareScalar(
                    quantMask,
                    yFp32,
                    TqQuantThreshold(code),
                    AscendC::CMPMODE::GT,
                    TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(qCandidate, TqCodeAsFloat(code), TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select(
                    qFloat,
                    quantMask,
                    qCandidate,
                    qFloat,
                    SELMODE::VSEL_TENSOR_TENSOR_MODE,
                    TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(
                argminIndex,
                qFloat,
                AscendC::RoundMode::CAST_RINT,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                argminIndexU16,
                argminIndex,
                AscendC::RoundMode::CAST_NONE,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(argminIndexU16, argminIndexU16, argminMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(
                encodedBatch[encodedOff].template ReinterpretCast<int16_t>(),
                argminIndexU16,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        TqSyncVToS();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t encodedOff = i * TQ_ENCODED_ROW_STRIDE_WORDS;
            auto normWords = norms.template ReinterpretCast<uint16_t>();
            const uint32_t normOff = i * TQ_NORM_STRIDE;
            encodedBatch.SetValue(encodedOff + TQ_PACK_D, normWords.GetValue(normOff));
        }
        TqSyncSToV();

        encodedBatchQue_.EnQue(encodedBatch);
        yBatchQue_.FreeTensor(yBatch);
    }

    __aicore__ inline uint32_t MakeVecIndex(
        uint32_t tokenIdx,
        uint32_t headIdx) const {
        return tokenIdx * numHeads_ + headIdx;
    }

    __aicore__ inline void CopyInIndexedTask(
        const AscendC::GlobalTensor<T>& xGm,
        const uint32_t* vecIndices,
        uint32_t rows,
        uint32_t& m) {
        if (rows == 0) {
            m = 0;
            return;
        }
        auto xBatch = xBatchQue_.AllocTensor<T>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::DataCopy(
                xBatch[row * TQ_PACK_D],
                xGm[static_cast<uint64_t>(vecIndices[row]) * TQ_PACK_D],
                TQ_PACK_D);
        }
        m = rows;
        xBatchQue_.EnQue(xBatch);
    }

    __aicore__ inline void ComputeBatch(uint32_t m) {
        NormalizeBatch(m);
        RotateBatchMatmul(m, 0, TQ_PACK_D);
        EncodeBatch(m);
    }

    // Physical 4-row cache group layout:
    //   uint16 word[d] bits  0.. 3 = row0 idx[d]
    //                  bits  4.. 7 = row1 idx[d]
    //                  bits  8..11 = row2 idx[d]
    //                  bits 12..15 = row3 idx[d]
    //   norm bytes are appended as row0,row1,row2,row3 after the 256B index group.
    __aicore__ inline void ClearPackedGroup(AscendC::LocalTensor<uint8_t>& packedGroup) {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        AscendC::Duplicate(
            packedU16,
            static_cast<uint16_t>(0),
            TQ_GROUP_STRIDE / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeEncodedRowToGroup(
        AscendC::LocalTensor<uint8_t>& packedGroup,
        const AscendC::LocalTensor<uint16_t>& encodedBatch,
        uint32_t encodedRow,
        uint32_t groupRow,
        bool clearOldNibble) {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        auto shiftedIdx = packMergeBuf_.Get<uint16_t>();
        const uint32_t encodedOff = encodedRow * TQ_ENCODED_ROW_STRIDE_WORDS;

        if (clearOldNibble) {
            auto clearMask = packMaskBuf_.Get<uint16_t>();
            const uint16_t mask = static_cast<uint16_t>(
                ~static_cast<uint16_t>(0x0Fu << (groupRow * 4)));
            AscendC::Duplicate(clearMask, mask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(packedU16, packedU16, clearMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::ShiftLeft(
            shiftedIdx,
            encodedBatch[encodedOff],
            static_cast<uint16_t>(groupRow * 4),
            TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(packedU16, packedU16, shiftedIdx, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();

        packedU16.SetValue(
            TQ_GROUP_INDEX_BYTES / sizeof(uint16_t) + groupRow,
            encodedBatch.GetValue(encodedOff + TQ_PACK_D));
    }

    __aicore__ inline void FlushResolvedGroups(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        AscendC::LocalTensor<uint16_t>& encodedBatch,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        const uint8_t* validRows,
        uint32_t m) {
        auto packedGroup = packedRowBuf_.Get<uint8_t>();
        uint32_t sortedRows[TQ_MAX_BATCH_M];
        uint32_t rowForGroup[TQ_GROUP_ROWS];
        uint32_t validCount = 0;

        for (uint32_t row = 0; row < m; ++row) {
            if (validRows[row] == 0) {
                continue;
            }
            sortedRows[validCount] = row;
            ++validCount;
        }

        for (uint32_t i = 1; i < validCount; ++i) {
            const uint32_t row = sortedRows[i];
            uint32_t pos = i;
            while (pos > 0) {
                const uint32_t prevRow = sortedRows[pos - 1];
                if (groupBases[prevRow] < groupBases[row] ||
                    (groupBases[prevRow] == groupBases[row] && groupRows[prevRow] <= groupRows[row])) {
                    break;
                }
                sortedRows[pos] = prevRow;
                --pos;
            }
            sortedRows[pos] = row;
        }

        uint32_t row = 0;
        while (row < validCount) {
            const uint64_t groupBase = groupBases[sortedRows[row]];
            uint32_t rowMask = 0;
            for (uint32_t groupRow = 0; groupRow < TQ_GROUP_ROWS; ++groupRow) {
                rowForGroup[groupRow] = TQ_MAX_BATCH_M;
            }
            do {
                const uint32_t srcRow = sortedRows[row];
                const uint32_t groupRow = groupRows[srcRow];
                rowForGroup[groupRow] = srcRow;
                rowMask |= (1u << groupRow);
                ++row;
            } while (row < validCount && groupBases[sortedRows[row]] == groupBase);

            if (rowMask == ((1u << TQ_GROUP_ROWS) - 1u)) {
                ClearPackedGroup(packedGroup);
                for (uint32_t groupRow = 0; groupRow < TQ_GROUP_ROWS; ++groupRow) {
                    MergeEncodedRowToGroup(
                        packedGroup, encodedBatch, rowForGroup[groupRow], groupRow, false);
                }
            } else {
                copy_packed_gm_to_ub(packedGroup, packedGm, groupBase, TQ_GROUP_BYTES);
                for (uint32_t groupRow = 0; groupRow < TQ_GROUP_ROWS; ++groupRow) {
                    if ((rowMask & (1u << groupRow)) == 0) {
                        continue;
                    }
                    MergeEncodedRowToGroup(
                        packedGroup, encodedBatch, rowForGroup[groupRow], groupRow, true);
                }
            }

            copy_packed_ub_to_gm(packedGm, groupBase, packedGroup, TQ_GROUP_BYTES);
        }
    }

    __aicore__ inline void CopyOutResolvedTask(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        uint32_t m) {
        auto encodedBatch = encodedBatchQue_.DeQue<uint16_t>();
        uint8_t validRows[TQ_MAX_BATCH_M];

        for (uint32_t row = 0; row < m; ++row) {
            validRows[row] = 1;
        }

        FlushResolvedGroups(packedGm, encodedBatch, groupBases, groupRows, validRows, m);
        encodedBatchQue_.FreeTensor(encodedBatch);
    }

    __aicore__ inline void PackCacheIndexedTask(
        const AscendC::GlobalTensor<T>& xGm,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const uint32_t* vecIndices,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        uint32_t rows) {
        uint32_t m = 0;
        CopyInIndexedTask(xGm, vecIndices, rows, m);
        if (m == 0) {
            return;
        }

        ComputeBatch(m);
        CopyOutResolvedTask(packedGm, groupBases, groupRows, m);
    }

    __aicore__ inline uint32_t GetActiveWorkers() const {
        return dataCores_ * TQ_AIV_SUB_BLOCKS;
    }

    __aicore__ inline bool ResolveTokenSlot(
        uint32_t tokenIdx,
        uint32_t& blockIdx,
        uint32_t& groupInBlock,
        uint32_t& groupRow) {
        if (tokenIdx >= tokenCount_) {
            return false;
        }
        const int32_t slot = slotMappingGm_.GetValue(tokenIdx);
        if (slot < 0 || static_cast<uint32_t>(slot) >= cacheSlots_) {
            return false;
        }
        const uint32_t slotU = static_cast<uint32_t>(slot);
        blockIdx = slotU / blockSize_;
        const uint32_t blockOff = slotU - blockIdx * blockSize_;
        groupInBlock = blockOff / TQ_GROUP_ROWS;
        groupRow = blockOff - groupInBlock * TQ_GROUP_ROWS;
        return true;
    }

    __aicore__ inline bool ResolveTokenSlotU(
        uint32_t tokenIdx,
        uint32_t& slotU) {
        if (tokenIdx >= tokenCount_) {
            return false;
        }
        const int32_t slot = slotMappingGm_.GetValue(tokenIdx);
        if (slot < 0 || static_cast<uint32_t>(slot) >= cacheSlots_) {
            return false;
        }
        slotU = static_cast<uint32_t>(slot);
        return true;
    }

    __aicore__ inline bool TryResolveFullPhysicalGroupStart(
        uint32_t tokenIdx,
        uint32_t& blockIdx,
        uint32_t& groupInBlock) {
        if (tokenIdx + TQ_GROUP_ROWS > tokenCount_ || cacheSlots_ < TQ_GROUP_ROWS) {
            return false;
        }

        uint32_t slot0 = 0;
        if (!ResolveTokenSlotU(tokenIdx, slot0) ||
            slot0 > cacheSlots_ - TQ_GROUP_ROWS) {
            return false;
        }
        blockIdx = slot0 / blockSize_;
        const uint32_t blockOff = slot0 - blockIdx * blockSize_;
        if ((blockOff % TQ_GROUP_ROWS) != 0 ||
            blockOff + TQ_GROUP_ROWS > blockSize_) {
            return false;
        }

        for (uint32_t groupRow = 1; groupRow < TQ_GROUP_ROWS; ++groupRow) {
            uint32_t slot = 0;
            if (!ResolveTokenSlotU(tokenIdx + groupRow, slot) ||
                slot != slot0 + groupRow) {
                return false;
            }
        }
        groupInBlock = blockOff / TQ_GROUP_ROWS;
        return true;
    }

    __aicore__ inline uint32_t MakePhysicalGroupTask(
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t headIdx) const {
        return ((blockIdx * (blockSize_ / TQ_GROUP_ROWS)) + groupInBlock) * numHeads_ + headIdx;
    }

    __aicore__ inline uint64_t MakeGroupBase(
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t headIdx) const {
        return ((uint64_t)blockIdx * numHeads_ + headIdx) * blockSize_ * TQ_ROW_BYTES
               + static_cast<uint64_t>(groupInBlock) * TQ_GROUP_BYTES;
    }

    __aicore__ inline void FlushVecBatch(
        const uint32_t* vecIndices,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        uint32_t rows) {
        if (rows == 0) {
            return;
        }
        PackCacheIndexedTask(keyGm_, keyCacheGm_, vecIndices, groupBases, groupRows, rows);
        PackCacheIndexedTask(valueGm_, valueCacheGm_, vecIndices, groupBases, groupRows, rows);
    }

    __aicore__ inline void GetWorkerTokenRange(
        uint32_t worker,
        uint32_t activeWorkers,
        uint32_t& tokenStart,
        uint32_t& tokenEnd) const {
        const uint32_t baseTokens = tokenCount_ / activeWorkers;
        const uint32_t extraTokens = tokenCount_ - baseTokens * activeWorkers;
        const uint32_t priorExtra = worker < extraTokens ? worker : extraTokens;
        tokenStart = worker * baseTokens + priorExtra;
        tokenEnd = tokenStart + baseTokens + (worker < extraTokens ? 1 : 0);
    }

    __aicore__ inline void AppendFullGroupHeadTask(
        uint32_t tokenIdx,
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t headIdx,
        uint32_t* vecIndices,
        uint64_t* groupBases,
        uint32_t* groupRows,
        uint32_t& rows) {
        const uint32_t fullGroupRows = TQ_GROUP_ROWS;
        if (rows + fullGroupRows > TQ_MAX_BATCH_M) {
            FlushVecBatch(vecIndices, groupBases, groupRows, rows);
            rows = 0;
        }

        const uint64_t groupBase = MakeGroupBase(blockIdx, groupInBlock, headIdx);
        for (uint32_t groupRow = 0; groupRow < TQ_GROUP_ROWS; ++groupRow) {
            const uint32_t vecIdx = MakeVecIndex(tokenIdx + groupRow, headIdx);
            if (vecIdx >= nVec_) {
                continue;
            }
            vecIndices[rows] = vecIdx;
            groupBases[rows] = groupBase;
            groupRows[rows] = groupRow;
            ++rows;
        }
    }

    __aicore__ inline void ProcessLogicalFullGroupsByRange(
        uint32_t worker,
        uint32_t activeWorkers) {
        uint32_t tokenStart = 0;
        uint32_t tokenEnd = 0;
        GetWorkerTokenRange(worker, activeWorkers, tokenStart, tokenEnd);

        uint32_t vecIndices[TQ_MAX_BATCH_M];
        uint64_t groupBases[TQ_MAX_BATCH_M];
        uint32_t groupRows[TQ_MAX_BATCH_M];
        uint32_t rows = 0;
        uint32_t tokenIdx = tokenStart;
        while (tokenIdx < tokenEnd) {
            uint32_t blockIdx = 0;
            uint32_t groupInBlock = 0;
            if (!TryResolveFullPhysicalGroupStart(tokenIdx, blockIdx, groupInBlock)) {
                ++tokenIdx;
                continue;
            }

            for (uint32_t headIdx = 0; headIdx < numHeads_; ++headIdx) {
                AppendFullGroupHeadTask(
                    tokenIdx,
                    blockIdx,
                    groupInBlock,
                    headIdx,
                    vecIndices,
                    groupBases,
                    groupRows,
                    rows);
                FlushVecBatch(vecIndices, groupBases, groupRows, rows);
                rows = 0;
            }
            tokenIdx += TQ_GROUP_ROWS;
        }
        FlushVecBatch(vecIndices, groupBases, groupRows, rows);
    }

    __aicore__ inline void ProcessFallbackByOwnedGroupsSkippingFullGroups(uint32_t worker) {
        const uint32_t activeWorkers = GetActiveWorkers();
        uint32_t vecIndices[TQ_MAX_BATCH_M];
        uint64_t groupBases[TQ_MAX_BATCH_M];
        uint32_t groupRows[TQ_MAX_BATCH_M];
        uint32_t rows = 0;
        uint32_t tokenIdx = 0;
        while (tokenIdx < tokenCount_) {
            uint32_t fullBlockIdx = 0;
            uint32_t fullGroupInBlock = 0;
            if (TryResolveFullPhysicalGroupStart(tokenIdx, fullBlockIdx, fullGroupInBlock)) {
                tokenIdx += TQ_GROUP_ROWS;
                continue;
            }

            uint32_t blockIdx = 0;
            uint32_t groupInBlock = 0;
            uint32_t groupRow = 0;
            if (!ResolveTokenSlot(tokenIdx, blockIdx, groupInBlock, groupRow)) {
                ++tokenIdx;
                continue;
            }

            for (uint32_t headIdx = 0; headIdx < numHeads_; ++headIdx) {
                const uint32_t physicalGroupTask =
                    MakePhysicalGroupTask(blockIdx, groupInBlock, headIdx);
                if ((physicalGroupTask % activeWorkers) != worker) {
                    continue;
                }

                const uint32_t vecIdx = MakeVecIndex(tokenIdx, headIdx);
                if (vecIdx >= nVec_) {
                    continue;
                }
                vecIndices[rows] = vecIdx;
                groupBases[rows] = MakeGroupBase(blockIdx, groupInBlock, headIdx);
                groupRows[rows] = groupRow;
                ++rows;
                if (rows == vecPerCore_) {
                    FlushVecBatch(vecIndices, groupBases, groupRows, rows);
                    rows = 0;
                }
            }
            ++tokenIdx;
        }
        FlushVecBatch(vecIndices, groupBases, groupRows, rows);
    }

    __aicore__ inline void ProcessCachePairByLogicalFastFallback(uint32_t worker) {
        const uint32_t activeWorkers = GetActiveWorkers();
        if (activeWorkers == 0 || worker >= activeWorkers) {
            return;
        }

        ProcessLogicalFullGroupsByRange(worker, activeWorkers);
        ProcessFallbackByOwnedGroupsSkippingFullGroups(worker);
    }

    __aicore__ inline void ProcessCachePairByOwnedGroups(uint32_t worker) {
        const uint32_t activeWorkers = GetActiveWorkers();
        if (activeWorkers == 0 || worker >= activeWorkers) {
            return;
        }

        uint32_t vecIndices[TQ_MAX_BATCH_M];
        uint64_t groupBases[TQ_MAX_BATCH_M];
        uint32_t groupRows[TQ_MAX_BATCH_M];
        uint32_t rows = 0;
        for (uint32_t tokenIdx = 0; tokenIdx < tokenCount_; ++tokenIdx) {
            uint32_t blockIdx = 0;
            uint32_t groupInBlock = 0;
            uint32_t groupRow = 0;
            if (!ResolveTokenSlot(tokenIdx, blockIdx, groupInBlock, groupRow)) {
                continue;
            }

            for (uint32_t headIdx = 0; headIdx < numHeads_; ++headIdx) {
                const uint32_t physicalGroupTask =
                    MakePhysicalGroupTask(blockIdx, groupInBlock, headIdx);
                if ((physicalGroupTask % activeWorkers) != worker) {
                    continue;
                }

                const uint32_t vecIdx = MakeVecIndex(tokenIdx, headIdx);
                if (vecIdx >= nVec_) {
                    continue;
                }
                vecIndices[rows] = vecIdx;
                groupBases[rows] = MakeGroupBase(blockIdx, groupInBlock, headIdx);
                groupRows[rows] = groupRow;
                ++rows;
                if (rows == vecPerCore_) {
                    FlushVecBatch(vecIndices, groupBases, groupRows, rows);
                    rows = 0;
                }
            }
        }
        FlushVecBatch(vecIndices, groupBases, groupRows, rows);
    }

    __aicore__ inline void ProcessCachePairByDirectTasks(uint32_t worker) {
        const uint32_t activeWorkers = GetActiveWorkers();
        if (activeWorkers == 0 || worker >= activeWorkers) {
            return;
        }

        uint32_t vecIndices[TQ_MAX_BATCH_M];
        uint64_t groupBases[TQ_MAX_BATCH_M];
        uint32_t groupRows[TQ_MAX_BATCH_M];
        uint32_t rows = 0;
        const uint32_t taskCount = tokenCount_ * numHeads_;
        for (uint32_t taskIdx = worker; taskIdx < taskCount; taskIdx += activeWorkers) {
            const uint32_t tokenIdx = taskIdx / numHeads_;
            const uint32_t headIdx = taskIdx - tokenIdx * numHeads_;
            uint32_t blockIdx = 0;
            uint32_t groupInBlock = 0;
            uint32_t groupRow = 0;
            if (!ResolveTokenSlot(tokenIdx, blockIdx, groupInBlock, groupRow)) {
                continue;
            }

            const uint32_t vecIdx = MakeVecIndex(tokenIdx, headIdx);
            if (vecIdx >= nVec_) {
                continue;
            }
            vecIndices[rows] = vecIdx;
            groupBases[rows] = MakeGroupBase(blockIdx, groupInBlock, headIdx);
            groupRows[rows] = groupRow;
            ++rows;
            if (rows == vecPerCore_) {
                FlushVecBatch(vecIndices, groupBases, groupRows, rows);
                rows = 0;
            }
        }
        FlushVecBatch(vecIndices, groupBases, groupRows, rows);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    RotateMmT* rotateMm_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t vecPerCore_ = 1;
    uint32_t numHeads_ = 1;
    uint32_t tokenCount_ = 0;
    uint32_t blockSize_ = 1;
    uint32_t numBlocks_ = 0;
    uint32_t cacheSlots_ = 0;
    uint32_t dataCores_ = 1;
    uint32_t packMode_ = TQ_PACK_MODE_GENERAL;
    bool matmulReady_ = false;

    AscendC::TQue<AscendC::TPosition::VECIN, TQ_QUEUE_DEPTH> xBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, TQ_QUEUE_DEPTH> aBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, TQ_QUEUE_DEPTH> yBatchQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> quantMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packMergeBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packMaskBuf_;
    AscendC::TQue<AscendC::TPosition::VECOUT, TQ_QUEUE_DEPTH> encodedBatchQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> distBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> argminIndexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> argminIndexU16Buf_;
    AscendC::GlobalTensor<T> keyGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> rotationTGm_;
    AscendC::GlobalTensor<int32_t> slotMappingGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache4bit(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR codebook,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    auto* codebookPtr = reinterpret_cast<__gm__ TqDataT*>(codebook);
    auto* rotationPtr = reinterpret_cast<__gm__ TqDataT*>(rotation_t);
    auto* slotMappingPtr = reinterpret_cast<__gm__ int32_t*>(slot_mapping);
    auto* keyCachePtr = reinterpret_cast<__gm__ uint8_t*>(key_cache);
    auto* valueCachePtr = reinterpret_cast<__gm__ uint8_t*>(value_cache);

    if (!TILING_KEY_IS(0)) {
        return;
    }
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
    AscendC::SetSysWorkspace(wsPtr);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    TqRotateMatmulOp<TqDataT> rotateMm;
    TCubeTiling cubeTiling = tilingData.cubeTiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &cubeTiling);
    TurboquantPackKVForCache4bitToCache<TqDataT, TqRotateMatmulOp<TqDataT>> op(&pipe, &rotateMm);
    op.Init(
        key,
        value,
        codebookPtr,
        rotationPtr,
        slotMappingPtr,
        keyCachePtr,
        valueCachePtr,
        wsPtr,
        tilingData.nVec,
        tilingData.vecPerCore,
        tilingData.numHeads,
        tilingData.blockSize,
        tilingData.numBlocks,
        tilingData.dataCores,
        tilingData.packMode);
    op.Process();
}
