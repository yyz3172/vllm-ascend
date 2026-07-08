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

// Fused BitResidual pack for KV cache (fp16/bf16, head_size=128).
//
// Reference (Python):
//   norms = vector_norm(x, dim=-1, keepdim=True)
//   y = (x / (norms + eps)) @ R^T        # batched [M, D] @ [D, D]
//   sign = y >= 0
//   err = y - (+/- 1/sqrt(D))
//   base = min(err), step = (max(err) - base) / 127
//   q7 = round((err - base) / step)
//   code = (q7 << 1) | sign
//
// Per AICore: normalize all assigned rows, then ONE matmul [M,128]@[128,128],
// not M separate M=1 matmuls (avoids Cube padding waste).

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

static constexpr uint64_t TQ_INVALID_OFFSET = ~static_cast<uint64_t>(0);
static constexpr int TQ_PACK_D = 128;
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
static constexpr uint32_t BR_CODE_BYTES = TQ_PACK_D;
static constexpr uint32_t BR_SCALAR_BYTES = sizeof(uint16_t);
static constexpr uint32_t BR_ROW_BYTES = BR_CODE_BYTES + 3 * BR_SCALAR_BYTES;
static constexpr uint32_t BR_NORM_LOCAL_OFFSET = BR_CODE_BYTES;
static constexpr uint32_t BR_BASE_LOCAL_OFFSET = BR_NORM_LOCAL_OFFSET + BR_SCALAR_BYTES;
static constexpr uint32_t BR_STEP_LOCAL_OFFSET = BR_BASE_LOCAL_OFFSET + BR_SCALAR_BYTES;
// Max rows per batch for static Matmul tiling / UB batch buffers.
static constexpr uint32_t TQ_MAX_BATCH_M = 32;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_SINGLE_ROT_M_PAD = 16;
static constexpr uint32_t TQ_ROT_LOCAL_WORKSPACE_BYTES = TQ_MAX_BATCH_M * TQ_ROT_K * sizeof(half);
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr float BR_INV_SQRT_D = 0.08838834764831845f;
static constexpr float BR_Q7_MAX = 127.0f;
static constexpr float BR_STEP_EPS_F = 1e-10f;

#if defined(ORIG_DTYPE_KEY)
#if (ORIG_DTYPE_KEY == DT_BF16)
using TqInputT = bfloat16_t;
#else
using TqInputT = half;
#endif
#elif defined(DTYPE_KEY)
#if (DTYPE_KEY == DT_BF16)
using TqInputT = bfloat16_t;
#else
using TqInputT = half;
#endif
#else
using TqInputT = half;
#endif

template <typename T>
struct TqInputTraits {
    static constexpr bool isBf16 = false;
};

template <>
struct TqInputTraits<bfloat16_t> {
    static constexpr bool isBf16 = true;
};

__aicore__ inline uint32_t AlignUp16(uint32_t x) {
    return (x + TQ_CUBE_M_ALIGN - 1) / TQ_CUBE_M_ALIGN * TQ_CUBE_M_ALIGN;
}

__aicore__ inline uint32_t AlignUp32(uint32_t x) {
    return (x + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
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

__aicore__ inline void TqSyncVToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void TqSyncSToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

__aicore__ inline bool TqIsAiv() {
    if ASCEND_IS_AIV {
        return true;
    }
    return false;
}

__aicore__ inline void write_half_le_local(
    AscendC::LocalTensor<uint8_t>& packedLocal, uint32_t out_offset, half value) {
    union {
        half h;
        uint16_t u;
    } bits {};
    bits.h = value;
    packedLocal.SetValue(out_offset, (uint8_t)(bits.u & 0xFFu));
    packedLocal.SetValue(out_offset + 1, (uint8_t)((bits.u >> 8) & 0xFFu));
}

// Field slices are not always 32B-aligned; use DataCopyPad for GM writes.
__aicore__ inline void copy_packed_ub_to_gm(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t> packedLocal,
    uint32_t nbytes) {
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
}

using TqRotateAT = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
using TqRotateBT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using TqRotateCT = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
using TqRotateBiasT = MatmulType<TPosition::GM, CubeFormat::ND, half>;

using TqRotateMatmulOp =
    AscendC::Matmul<TqRotateAT, TqRotateBT, TqRotateCT, TqRotateBiasT>;

class BitResidualPackKVForCache {
public:
    __aicore__ inline explicit BitResidualPackKVForCache(AscendC::TPipe* pipe, TqRotateMatmulOp* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm) {}

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ half* rotation_t,
        __gm__ int32_t* slot_mapping,
        __gm__ uint8_t* key_cache,
        __gm__ uint8_t* value_cache,
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t vecPerCore,
        uint32_t numHeads,
        uint32_t blockSize,
        uint32_t numBlocks) {
        nVec_ = nVec;
        numHeads_ = numHeads;
        blockSize_ = blockSize;
        numBlocks_ = numBlocks;
        cacheSlots_ = blockSize_ * numBlocks_;
        tokenCount_ = (nVec + numHeads - 1) / numHeads;
        blockBytes_ = blockSize_ * BR_ROW_BYTES;
        packedStride_ = AlignUp32(BR_ROW_BYTES);
        vecPerCore_ = vecPerCore;
        if (vecPerCore_ > TQ_MAX_BATCH_M) {
            vecPerCore_ = TQ_MAX_BATCH_M;
        }

        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TqInputT*>(key), (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TqInputT*>(value), (uint64_t)nVec_ * TQ_PACK_D);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        slotMappingGm_.SetGlobalBuffer(slot_mapping, tokenCount_);
        keyCacheGm_.SetGlobalBuffer(key_cache, (uint64_t)numBlocks_ * numHeads_ * blockBytes_);
        valueCacheGm_.SetGlobalBuffer(value_cache, (uint64_t)numBlocks_ * numHeads_ * blockBytes_);

        const uint32_t batchElems = TQ_MAX_BATCH_M * TQ_PACK_D;
        pipe_->InitBuffer(xBatchQue_, 2, batchElems * sizeof(half));
        pipe_->InitBuffer(aBatchQue_, 2, batchElems * sizeof(half));
        pipe_->InitBuffer(yBatchQue_, 2, batchElems * sizeof(half));
        if constexpr (TqInputTraits<TqInputT>::isBf16) {
            pipe_->InitBuffer(xBatchInputQue_, 2, batchElems * sizeof(TqInputT));
            pipe_->InitBuffer(bf16SquareBuf_, TQ_PACK_D * sizeof(float));
        }
        pipe_->InitBuffer(normScalarBuf_, TQ_UB_ALIGN);
        pipe_->InitBuffer(normsBuf_, TQ_MAX_BATCH_M * sizeof(half));
        pipe_->InitBuffer(yFp32Buf_, TQ_PACK_D * sizeof(float));
        // NormalizeBatch: fp32 row + fp32 ReduceSum tmp (2 * TQ_PACK_D floats).
        pipe_->InitBuffer(reduceOutBuf_, TQ_PACK_D * 2 * sizeof(float));
        pipe_->InitBuffer(packedRowQue_, 2, TQ_MAX_BATCH_M * packedStride_ * sizeof(uint8_t));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_LOCAL_WORKSPACE_BYTES);

        matmulReady_ = (rawWorkspace != nullptr);
    }

    __aicore__ inline void Process() {
        if (!matmulReady_ || !TqIsAiv()) {
            return;
        }
        // On 910B3 each blockIdx is one AIV (sub-block indices alternate
        // 0,1,0,1… — all are independent workers).  Use blockIdx directly.
        // dataCores is always 16 for KFC mode; K/V rows are distributed evenly
        // and each core loops in sub-batches of vecPerCore_ rows.
        const uint32_t dataCores = 16;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t totalRows = nVec_ * 2;
        const uint32_t rowsPerCore = (totalRows + dataCores - 1) / dataCores;
        const uint32_t coreStart = core * rowsPerCore;
        if (coreStart >= totalRows) {
            return;
        }
        const uint32_t coreEnd = coreStart + rowsPerCore > totalRows
            ? totalRows : coreStart + rowsPerCore;

        // Process assigned rows in sub-batches of up to vecPerCore_ (32).
        for (uint32_t batchStart = coreStart; batchStart < coreEnd;
             batchStart += vecPerCore_) {
            const uint32_t batchEnd = batchStart + vecPerCore_ > coreEnd
                ? coreEnd : batchStart + vecPerCore_;

            PackMergedBatch(batchStart, batchEnd);
        }
    }

private:
    __aicore__ inline uint64_t CacheBlockHeadBase(uint32_t vecIdx, uint32_t& blockOffset) {
        const uint32_t tokenIdx = vecIdx / numHeads_;
        if (tokenIdx >= tokenCount_) {
            return TQ_INVALID_OFFSET;
        }
        const int32_t slot = slotMappingGm_.GetValue(tokenIdx);
        if (slot < 0 || static_cast<uint32_t>(slot) >= cacheSlots_) {
            return TQ_INVALID_OFFSET;
        }
        const uint32_t slotU = static_cast<uint32_t>(slot);
        const uint32_t blockIdx = slotU / blockSize_;
        blockOffset = slotU - blockIdx * blockSize_;
        const uint32_t headIdx = vecIdx - tokenIdx * numHeads_;
        return ((uint64_t)blockIdx * numHeads_ + headIdx) * blockBytes_;
    }

    // norms[i] = ||x[i]||; xBatch rows unitized in-place (matches x / (norm + eps)).
    // Inner dim: Cast + Mul + ReduceSum (vector), not scalar loop; fp32 acc avoids fp16 overflow.
    __aicore__ inline void NormalizeBatch(
        AscendC::LocalTensor<half>& xBatch, AscendC::LocalTensor<half>& norms, uint32_t m) {
        auto fp32Row = reduceOutBuf_.Get<float>();
        auto fp32Tmp = reduceOutBuf_.Get<float>()[TQ_PACK_D];
        auto normAcc = normScalarBuf_.Get<float>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;

            AscendC::Cast(fp32Row, xBatch[rowOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::Mul(fp32Row, fp32Row, fp32Row, TQ_PACK_D);
            AscendC::ReduceSum<float>(normAcc, fp32Row, fp32Tmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(normAcc, normAcc, 1);
            TqSyncVToS();
            const float normF = normAcc.GetValue(0);
            TqSyncSToV();

            norms.SetValue(i, static_cast<half>(normF));
            const half invH = static_cast<half>(1.0f / (normF + TQ_NORM_EPS_F));
            AscendC::Muls(xBatch[rowOff], xBatch[rowOff], invH, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    template <typename InputT>
    __aicore__ inline void CopyMergedRowsToLocal(
        AscendC::LocalTensor<InputT>& dstBatch,
        uint32_t start,
        uint32_t end) {
        uint32_t dstOffset = 0;
        uint32_t linear = start;
        while (linear < end) {
            const bool isKey = linear < nVec_;
            const uint32_t srcRow = isKey ? linear : (linear - nVec_);
            const uint32_t srcRemain = nVec_ - srcRow;
            const uint32_t take = (end - linear) < srcRemain ? (end - linear) : srcRemain;
            auto src = isKey ? keyGm_[(uint64_t)srcRow * TQ_PACK_D]
                             : valueGm_[(uint64_t)srcRow * TQ_PACK_D];
            AscendC::DataCopy(dstBatch[dstOffset], src, take * TQ_PACK_D);
            dstOffset += take * TQ_PACK_D;
            linear += take;
        }
    }

    // BF16 input path: cast one row to fp32, normalize in fp32, then write fp16 xBatch once.
    template <typename InputT>
    __aicore__ inline void NormalizeInputBatch(
        AscendC::LocalTensor<InputT>& inputBatch,
        AscendC::LocalTensor<half>& xBatch,
        AscendC::LocalTensor<half>& norms,
        uint32_t m) {
        if constexpr (TqInputTraits<InputT>::isBf16) {
            auto fp32Row = reduceOutBuf_.Get<float>();
            auto reduceTmp = reduceOutBuf_.Get<float>()[TQ_PACK_D];
            auto squareRow = bf16SquareBuf_.Get<float>();
            auto normAcc = normScalarBuf_.Get<float>();

            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t rowOff = i * TQ_PACK_D;

                AscendC::Cast(fp32Row, inputBatch[rowOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(squareRow, fp32Row, fp32Row, TQ_PACK_D);
                AscendC::ReduceSum<float>(normAcc, squareRow, reduceTmp, TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sqrt(normAcc, normAcc, 1);
                TqSyncVToS();
                const float normF = normAcc.GetValue(0);
                TqSyncSToV();

                norms.SetValue(i, static_cast<half>(normF));
                AscendC::Muls(fp32Row, fp32Row, 1.0f / (normF + TQ_NORM_EPS_F), TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(xBatch[rowOff], fp32Row, AscendC::RoundMode::CAST_ROUND, TQ_PACK_D);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            NormalizeBatch(xBatch, norms, m);
        }
    }

    template <typename InputT>
    __aicore__ inline void NormalizeDequeuedBatch(
        AscendC::LocalTensor<half>& xBatch,
        AscendC::LocalTensor<half>& norms,
        uint32_t m) {
        if constexpr (!TqInputTraits<InputT>::isBf16) {
            NormalizeBatch(xBatch, norms, m);
        }
    }

    template <typename InputT>
    __aicore__ inline void NormalizeCopiedInputIfNeeded(
        AscendC::LocalTensor<half>& xBatch,
        AscendC::LocalTensor<half>& norms,
        uint32_t m) {
        if constexpr (TqInputTraits<InputT>::isBf16) {
            auto inputReady = xBatchInputQue_.DeQue<InputT>();
            NormalizeInputBatch(inputReady, xBatch, norms, m);
            xBatchInputQue_.FreeTensor(inputReady);
        }
    }

    // Cube Matmul: half A(VECOUT) × half B(GM) → half C(VECIN).
    // Keep the rotate result in local memory and let EncodeBatch consume it
    // directly, avoiding the previous explicit GM scratch + GM->UB readback.
    __aicore__ inline void RotateBatchMatmul(
        AscendC::LocalTensor<half>& xUnitBatch,
        AscendC::LocalTensor<half>& yBatch,
        uint32_t m,
        uint32_t mPad,
        uint32_t dBase,
        uint32_t dCount) {
        rotateMm_->SetOrgShape(mPad, TQ_ROT_N, TQ_ROT_K);
        rotateMm_->SetSingleShape(m, dCount, TQ_ROT_K);
        rotateMm_->SetTensorA(xUnitBatch, false);
        rotateMm_->SetTensorB(rotationTGm_[dBase], false);
        auto rotateWorkspace = rotateWorkBuf_.Get<uint8_t>();
        rotateMm_->SetLocalWorkspace(rotateWorkspace);
        // IterateAll: single atomic KFC message — safe when 2 AIVs share one AIC
        // in MIX mode (no tile interleaving across concurrent AIVs).
        rotateMm_->IterateAll(yBatch);
        rotateMm_->End();
    }

    __aicore__ inline void EncodeBatch(
        AscendC::LocalTensor<half>& yBatch,
        AscendC::LocalTensor<half>& norms,
        AscendC::LocalTensor<uint8_t>& packedBatch,
        uint32_t m) {
        auto yFp32 = yFp32Buf_.Get<float>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            auto packedRow = packedBatch[i * packedStride_];

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            float yVals[TQ_PACK_D];
            float errVals[TQ_PACK_D];
            float errMin = 3.402823466e+38f;
            float errMax = -3.402823466e+38f;
            for (uint32_t d = 0; d < TQ_PACK_D; ++d) {
                yVals[d] = yFp32.GetValue(d);
                const float bitVec = (yVals[d] >= 0.0f) ? BR_INV_SQRT_D : -BR_INV_SQRT_D;
                const float err = yVals[d] - bitVec;
                errVals[d] = err;
                if (err < errMin) {
                    errMin = err;
                }
                if (err > errMax) {
                    errMax = err;
                }
            }

            float range = errMax - errMin;
            float step = range / BR_Q7_MAX;
            if (range < BR_STEP_EPS_F) {
                step = BR_STEP_EPS_F;
            }
            for (uint32_t d = 0; d < TQ_PACK_D; ++d) {
                int32_t q = static_cast<int32_t>((errVals[d] - errMin) / step + 0.5f);
                if (q < 0) {
                    q = 0;
                } else if (q > 127) {
                    q = 127;
                }
                const uint8_t sign = yVals[d] >= 0.0f ? 1u : 0u;
                packedRow.SetValue(d, static_cast<uint8_t>((q << 1) | sign));
            }
            for (uint32_t k = BR_ROW_BYTES; k < packedStride_; ++k) {
                packedRow.SetValue(k, (uint8_t)0);
            }
            write_half_le_local(packedRow, BR_NORM_LOCAL_OFFSET, norms.GetValue(i));
            write_half_le_local(packedRow, BR_BASE_LOCAL_OFFSET, static_cast<half>(errMin));
            write_half_le_local(packedRow, BR_STEP_LOCAL_OFFSET, static_cast<half>(step));
            TqSyncSToV();
        }
    }

    __aicore__ inline void CopyInMergedBatch(uint32_t start, uint32_t end) {
        auto xBatch = xBatchQue_.AllocTensor<half>();
        if constexpr (TqInputTraits<TqInputT>::isBf16) {
            auto xBatchInput = xBatchInputQue_.AllocTensor<TqInputT>();
            CopyMergedRowsToLocal(xBatchInput, start, end);
            xBatchInputQue_.EnQue(xBatchInput);
        } else {
            CopyMergedRowsToLocal(xBatch, start, end);
        }
        xBatchQue_.EnQue(xBatch);
    }

    __aicore__ inline void CopyPackedRowToFieldMajorCache(
        AscendC::GlobalTensor<uint8_t>& cacheGm,
        uint64_t blockHeadBase,
        uint32_t blockOffset,
        AscendC::LocalTensor<uint8_t> packedRow) {
        const uint64_t codeOffset = blockHeadBase + (uint64_t)blockOffset * BR_CODE_BYTES;
        const uint64_t normOffset = blockHeadBase + (uint64_t)blockSize_ * BR_CODE_BYTES +
            (uint64_t)blockOffset * BR_SCALAR_BYTES;
        const uint64_t baseOffset = normOffset + (uint64_t)blockSize_ * BR_SCALAR_BYTES;
        const uint64_t stepOffset = baseOffset + (uint64_t)blockSize_ * BR_SCALAR_BYTES;
        copy_packed_ub_to_gm(cacheGm, codeOffset, packedRow, BR_CODE_BYTES);
        copy_packed_ub_to_gm(cacheGm, normOffset, packedRow[BR_NORM_LOCAL_OFFSET], BR_SCALAR_BYTES);
        copy_packed_ub_to_gm(cacheGm, baseOffset, packedRow[BR_BASE_LOCAL_OFFSET], BR_SCALAR_BYTES);
        copy_packed_ub_to_gm(cacheGm, stepOffset, packedRow[BR_STEP_LOCAL_OFFSET], BR_SCALAR_BYTES);
    }

    __aicore__ inline void CopyOutPackedBatch(uint32_t start, uint32_t m) {
        auto packedBatch = packedRowQue_.DeQue<uint8_t>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t linear = start + i;
            auto packedRow = packedBatch[i * packedStride_];
            uint32_t blockOffset = 0;
            if (linear < nVec_) {
                const uint64_t out_base = CacheBlockHeadBase(linear, blockOffset);
                if (out_base != TQ_INVALID_OFFSET) {
                    CopyPackedRowToFieldMajorCache(keyCacheGm_, out_base, blockOffset, packedRow);
                }
            } else {
                const uint32_t valueRow = linear - nVec_;
                const uint64_t out_base = CacheBlockHeadBase(valueRow, blockOffset);
                if (out_base != TQ_INVALID_OFFSET) {
                    CopyPackedRowToFieldMajorCache(valueCacheGm_, out_base, blockOffset, packedRow);
                }
            }
        }
        packedRowQue_.FreeTensor(packedBatch);
    }

    __aicore__ inline void Compute(uint32_t m, uint32_t mPad) {
        auto xBatch = xBatchQue_.DeQue<half>();
        auto yBatch = yBatchQue_.AllocTensor<half>();
        auto packedBatch = packedRowQue_.AllocTensor<uint8_t>();
        auto norms = normsBuf_.Get<half>();

        NormalizeCopiedInputIfNeeded<TqInputT>(xBatch, norms, m);
        if (mPad > m) {
            AscendC::Duplicate(xBatch[m * TQ_PACK_D], (half)0, (mPad - m) * TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        NormalizeDequeuedBatch<TqInputT>(xBatch, norms, m);
        auto aBatch = aBatchQue_.AllocTensor<half>();
        AscendC::Adds(aBatch, xBatch, static_cast<half>(0.0), mPad * TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        aBatchQue_.EnQue(aBatch);
        auto aReady = aBatchQue_.DeQue<half>();

        RotateBatchMatmul(aReady, yBatch, m, mPad, 0, TQ_PACK_D);
        aBatchQue_.FreeTensor(aReady);
        yBatchQue_.EnQue(yBatch);
        auto yReady = yBatchQue_.DeQue<half>();
        EncodeBatch(yReady, norms, packedBatch, m);
        yBatchQue_.FreeTensor(yReady);

        packedRowQue_.EnQue(packedBatch);
        xBatchQue_.FreeTensor(xBatch);
    }

    __aicore__ inline void PackMergedBatch(uint32_t start, uint32_t end) {
        const uint32_t m = end - start;
        const uint32_t mPad = AlignUp16(m);

        CopyInMergedBatch(start, end);
        Compute(m, mPad);
        CopyOutPackedBatch(start, m);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    TqRotateMatmulOp* rotateMm_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t cacheSlots_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t blockBytes_ = 0;
    uint32_t tokenCount_ = 0;
    uint32_t vecPerCore_ = 1;
    bool matmulReady_ = false;

    AscendC::TQue<AscendC::TPosition::VECIN, 2> xBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> aBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> yBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> xBatchInputQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> packedRowQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf16SquareBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateWorkBuf_;
    uint32_t packedStride_ = 0;

    AscendC::GlobalTensor<TqInputT> keyGm_;
    AscendC::GlobalTensor<TqInputT> valueGm_;
    AscendC::GlobalTensor<half> rotationTGm_;
    AscendC::GlobalTensor<int32_t> slotMappingGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void bit_residual_pack_kv_for_cache(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    auto* rotationPtr = reinterpret_cast<__gm__ half*>(rotation_t);
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
    TqRotateMatmulOp rotateMm;
    TCubeTiling cubeTiling = tilingData.cubeTiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &cubeTiling);
    BitResidualPackKVForCache op(&pipe, &rotateMm);
    op.Init(
        key,
        value,
        rotationPtr,
        slotMappingPtr,
        keyCachePtr,
        valueCachePtr,
        wsPtr,
        tilingData.nVec,
        tilingData.vecPerCore,
        tilingData.numHeads,
        tilingData.blockSize,
        tilingData.numBlocks);
    op.Process();
}
