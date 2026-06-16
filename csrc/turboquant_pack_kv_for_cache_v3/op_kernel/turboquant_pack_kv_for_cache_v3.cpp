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

// Fused TurboQuant pack for KV cache v3 (fp16/bf16, AIV-only vector rotate).
//
// Reference (Python):
//   norms = vector_norm(x, dim=-1, keepdim=True)
//   y = (x / (norms + eps)) @ R^T        # batched [M, D] @ [D, D]
//   d = (y.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
//   idx = d.argmin(dim=-1)
//
// Per AIV: normalize assigned rows, compute rotate by vector instructions, then
// encode to packed KV rows.  This variant intentionally does not use Cube/KFC.

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace {

static constexpr int TQ_PACK_D = 128;
static constexpr int TQ_PACK_K = 256;
static constexpr uint32_t TQ_UB_ALIGN = 32;
// Max rows per batch for UB batch buffers.
static constexpr uint32_t TQ_MAX_BATCH_M = 64;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr uint32_t TQ_REDUCE_MASK = 64;  // max mask for float WholeReduceMin
static constexpr uint32_t TQ_REDUCE_BATCHES = (TQ_PACK_K + TQ_REDUCE_MASK - 1) / TQ_REDUCE_MASK;  // 4
static constexpr int32_t TQ_REDUCE_SRC_REP_STRIDE = TQ_REDUCE_MASK / 8;  // 8 dataBlocks per repeat
// Dimensions processed per tile — balance between UB usage and sync reduction.
// distBuf = D_TILE * K * 4 bytes.  D_TILE=16 → 16 KiB.
static constexpr uint32_t TQ_D_TILE = 16;

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

__aicore__ inline float TqAbsF32(float x) {
    return x < 0.f ? -x : x;
}

// Scalar argmin: find k in [0,TQ_PACK_K) minimizing |yf - cb[k]|.
__aicore__ inline uint8_t ArgminAbsL1Scalar(float yf, const AscendC::LocalTensor<half>& cbLocal) {
    float best = TqAbsF32(yf - static_cast<float>(cbLocal.GetValue(0)));
    uint8_t bestIdx = 0;
    for (uint32_t k = 1; k < TQ_PACK_K; ++k) {
        const float d = TqAbsF32(yf - static_cast<float>(cbLocal.GetValue(k)));
        if (d < best) {
            best = d;
            bestIdx = static_cast<uint8_t>(k);
        }
    }
    return bestIdx;
}

__aicore__ inline void write_norm_fp16_le_local(
    AscendC::LocalTensor<uint8_t>& packedLocal, uint32_t out_offset, half norm_h) {
    union {
        half h;
        uint16_t u;
    } normBits {};
    normBits.h = norm_h;
    packedLocal.SetValue(out_offset + (uint32_t)TQ_PACK_D, (uint8_t)(normBits.u & 0xFFu));
    packedLocal.SetValue(out_offset + (uint32_t)TQ_PACK_D + 1, (uint8_t)((normBits.u >> 8) & 0xFFu));
}

// slot_w (e.g. 130) is not 32B-aligned; use DataCopyPad for GM writes.
__aicore__ inline void copy_packed_ub_to_gm(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t> packedLocal,
    uint32_t nbytes) {
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
}

class TurboquantPackKVForCacheV3 {
public:
    __aicore__ inline explicit TurboquantPackKVForCacheV3(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ half* codebook,
        __gm__ half* rotation_t,
        __gm__ uint8_t* packed_k,
        __gm__ uint8_t* packed_v,
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t slot_w_k,
        uint32_t slot_w_v,
        uint32_t vecPerCore) {
        (void)rawWorkspace;
        nVec_ = nVec;
        slot_w_k_ = slot_w_k;
        slot_w_v_ = slot_w_v;
        maxSlotW_ = slot_w_k_ > slot_w_v_ ? slot_w_k_ : slot_w_v_;
        packedStride_ = AlignUp32(maxSlotW_);
        vecPerCore_ = vecPerCore;
        if (vecPerCore_ > TQ_MAX_BATCH_M) {
            vecPerCore_ = TQ_MAX_BATCH_M;
        }

        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TqInputT*>(key), (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TqInputT*>(value), (uint64_t)nVec_ * TQ_PACK_D);
        codebookGm_.SetGlobalBuffer(codebook, TQ_PACK_K);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        packedKGm_.SetGlobalBuffer(packed_k, (uint64_t)nVec_ * slot_w_k_);
        packedVGm_.SetGlobalBuffer(packed_v, (uint64_t)nVec_ * slot_w_v_);

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
        pipe_->InitBuffer(codebookBuf_, TQ_PACK_K * sizeof(half));
        pipe_->InitBuffer(rotationTBuf_, TQ_PACK_D * TQ_PACK_D * sizeof(half));
        // distBuf: tiled distance vectors [D_TILE][K] (16×256×4 = 16 KiB)
        pipe_->InitBuffer(distBuf_, TQ_D_TILE * TQ_PACK_K * sizeof(float) + 256);
        // yFp32Buf: fp32 y row [D=128]
        pipe_->InitBuffer(yFp32Buf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(cbTileBuf_, TQ_PACK_K * sizeof(float));
        // argminResultBuf: results for WholeReduceMin [4 reduce batches][index,value].
        pipe_->InitBuffer(argminResultBuf_, TQ_REDUCE_BATCHES * 2 * sizeof(float));
        // NormalizeBatch: fp32 row + fp32 ReduceSum tmp (2 * TQ_PACK_D floats).
        pipe_->InitBuffer(reduceOutBuf_, TQ_PACK_D * 2 * sizeof(float));
        pipe_->InitBuffer(packedRowQue_, 2, TQ_MAX_BATCH_M * packedStride_ * sizeof(uint8_t));
        if (TqIsAiv()) {
            LoadStaticTablesToUb();
        }
    }

    __aicore__ inline void Process() {
        if (!TqIsAiv()) {
            return;
        }
        // On 910B3 each blockIdx is one AIV (sub-block indices alternate
        // 0,1,0,1… — all are independent workers).  Use blockIdx directly.
        // K/V rows are distributed evenly and each core loops in sub-batches
        // of vecPerCore_ rows.
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

        // Process assigned rows in sub-batches of up to vecPerCore_ (64).
        for (uint32_t batchStart = coreStart; batchStart < coreEnd;
             batchStart += vecPerCore_) {
            const uint32_t batchEnd = batchStart + vecPerCore_ > coreEnd
                ? coreEnd : batchStart + vecPerCore_;

            PackMergedBatch(batchStart, batchEnd);
        }
    }

private:
    __aicore__ inline void LoadStaticTablesToUb() {
        auto codebookLocal = codebookBuf_.Get<half>();
        auto rotationTLocal = rotationTBuf_.Get<half>();
        AscendC::DataCopy(codebookLocal, codebookGm_[0], TQ_PACK_K);
        AscendC::DataCopy(rotationTLocal, rotationTGm_[0], TQ_PACK_D * TQ_PACK_D);
        TqSyncMte2ToV();

        auto cbFp32 = cbTileBuf_.Get<float>();
        AscendC::Cast(cbFp32, codebookLocal, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(cbFp32[TQ_PACK_D], codebookLocal[TQ_PACK_D], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
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
    __aicore__ inline void NormalizeMergedBatch(
        AscendC::LocalTensor<half>& xBatch,
        AscendC::LocalTensor<half>& norms,
        uint32_t m) {
        if constexpr (TqInputTraits<InputT>::isBf16) {
            auto inputReady = xBatchInputQue_.DeQue<InputT>();
            NormalizeInputBatch(inputReady, xBatch, norms, m);
            xBatchInputQue_.FreeTensor(inputReady);
        } else {
            NormalizeBatch(xBatch, norms, m);
        }
    }

    // Vector rotate matmul: half A(VECOUT) x half B(UB) -> half C(VECIN).
    // It keeps the same output layout as the v2 Cube path for comparison.
    __aicore__ inline void RotateBatchMatmulVector(
        AscendC::LocalTensor<half>& xUnitBatch,
        AscendC::LocalTensor<half>& yBatch,
        uint32_t m,
        uint32_t dBase,
        uint32_t dCount) {
        auto rotationTLocal = rotationTBuf_.Get<half>();
        auto acc = yFp32Buf_.Get<float>();
        auto rotFp32 = reduceOutBuf_.Get<float>();
        auto term = reduceOutBuf_.Get<float>()[TQ_PACK_D];

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t xOff = i * TQ_PACK_D;
            float xVals[TQ_PACK_D];
            TqSyncVToS();
            for (uint32_t k = 0; k < TQ_PACK_D; ++k) {
                xVals[k] = static_cast<float>(xUnitBatch.GetValue(xOff + k));
            }
            TqSyncSToV();

            AscendC::Duplicate(acc, 0.0f, dCount);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t k = 0; k < TQ_PACK_D; ++k) {
                const uint32_t rotOff = k * TQ_PACK_D + dBase;
                AscendC::Cast(rotFp32, rotationTLocal[rotOff], AscendC::RoundMode::CAST_NONE, dCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(term, rotFp32, xVals[k], dCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(acc, acc, term, dCount);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(yBatch[i * TQ_ROT_N], acc, AscendC::RoundMode::CAST_NONE, dCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Vectorized argmin per dimension using Duplicate/Sub/Abs + WholeReduceMin.
    // Python equivalent: d = (y.unsqueeze(-1) - codebook.view(1,1,-1)).abs(); idx = d.argmin(dim=-1)
    //
    // Optimization: tile D_TILE dimensions at a time to amortize sync overhead.
    //   - Batch-read all y values into stack array (1 V→S + 1 S→V sync pair)
    //   - Per tile: Duplicate/Sub/Abs for D_TILE rows, then WholeReduceMin per row
    //   - Batch-read results, batch-write indices
    //   - No PipeBarrier between consecutive V ops (Duplicate→Sub→Abs)
    __aicore__ inline void EncodeBatch(
        AscendC::LocalTensor<half>& yBatch,
        AscendC::LocalTensor<half>& norms,
        AscendC::LocalTensor<uint8_t>& packedBatch,
        uint32_t m,
        uint32_t dBase,
        uint32_t dCount,
        bool writeMeta) {
        // codebook fp32 table is prepared once in UB during Init.
        auto cbFp32 = cbTileBuf_.Get<float>();
        auto yFp32 = yFp32Buf_.Get<float>();
        auto distTile = distBuf_.Get<float>();    // [D_TILE][K]
        auto argminRes = argminResultBuf_.Get<float>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            auto packedRow = packedBatch[i * packedStride_];

            // Cast y row half → float, then batch-read all values
            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, dCount);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            float yVals[TQ_PACK_D];
            for (uint32_t d = 0; d < dCount; ++d) {
                yVals[d] = yFp32.GetValue(d);
            }
            TqSyncSToV();

            // Process dimensions in tiles of D_TILE
            for (uint32_t tileStart = 0; tileStart < dCount; tileStart += TQ_D_TILE) {
                const uint32_t tileCnt = (tileStart + TQ_D_TILE <= dCount)
                    ? TQ_D_TILE : (dCount - tileStart);

                // Phase 1: compute distance vectors for this tile
                for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                    auto distRow = distTile[dl * TQ_PACK_K];
                    AscendC::Duplicate(distRow, yVals[tileStart + dl], TQ_PACK_K);
                    AscendC::Sub(distRow, distRow, cbFp32, TQ_PACK_K);
                    AscendC::Abs(distRow, distRow, TQ_PACK_K);
                }

                // Phase 2+3: reduce each dimension, merge reduce-batch results
                // on Scalar, and write the final uint8 index directly to the packed row.
                for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                    auto distRow = distTile[dl * TQ_PACK_K];
                    AscendC::WholeReduceMin<float>(
                        argminRes, distRow, TQ_REDUCE_MASK,
                        TQ_REDUCE_BATCHES, 1, 1, TQ_REDUCE_SRC_REP_STRIDE,
                        AscendC::ReduceOrder::ORDER_INDEX_VALUE);
                    AscendC::PipeBarrier<PIPE_V>();
                    TqSyncVToS();

                    int32_t globalIdx = 0;
                    float globalMin = argminRes.GetValue(1);
                    float idxF = argminRes.GetValue(0);
                    int32_t rawIdx = *reinterpret_cast<int32_t *>(&idxF);
                    globalIdx = rawIdx;
                    for (uint32_t b = 1; b < TQ_REDUCE_BATCHES; ++b) {
                        idxF = argminRes.GetValue(b * 2);
                        rawIdx = *reinterpret_cast<int32_t *>(&idxF);
                        float batchMin = argminRes.GetValue(b * 2 + 1);
                        if (batchMin < globalMin) {
                            globalMin = batchMin;
                            globalIdx = rawIdx + static_cast<int32_t>(b * TQ_REDUCE_MASK);
                        }
                    }
                    packedRow.SetValue(tileStart + dl, static_cast<uint8_t>(globalIdx));
                    TqSyncSToV();
                }
            }

            for (uint32_t k = dCount; k < packedStride_; ++k) {
                packedRow.SetValue(k, (uint8_t)0);
            }
            if (writeMeta) {
                write_norm_fp16_le_local(packedRow, 0, norms.GetValue(i));
            }
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

    __aicore__ inline void CopyOutPackedBatch(uint32_t start, uint32_t m) {
        auto packedBatch = packedRowQue_.DeQue<uint8_t>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t linear = start + i;
            auto packedRow = packedBatch[i * packedStride_];
            if (linear < nVec_) {
                copy_packed_ub_to_gm(packedKGm_, (uint64_t)linear * slot_w_k_, packedRow, slot_w_k_);
            } else {
                const uint32_t valueRow = linear - nVec_;
                copy_packed_ub_to_gm(packedVGm_, (uint64_t)valueRow * slot_w_v_, packedRow, slot_w_v_);
            }
        }
        packedRowQue_.FreeTensor(packedBatch);
    }

    __aicore__ inline void Compute(uint32_t m) {
        auto xBatch = xBatchQue_.DeQue<half>();
        auto yBatch = yBatchQue_.AllocTensor<half>();
        auto packedBatch = packedRowQue_.AllocTensor<uint8_t>();
        auto norms = normsBuf_.Get<half>();

        NormalizeMergedBatch<TqInputT>(xBatch, norms, m);
        auto aBatch = aBatchQue_.AllocTensor<half>();
        AscendC::Adds(aBatch, xBatch, static_cast<half>(0.0), m * TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        aBatchQue_.EnQue(aBatch);
        auto aReady = aBatchQue_.DeQue<half>();

        RotateBatchMatmulVector(aReady, yBatch, m, 0, TQ_PACK_D);
        aBatchQue_.FreeTensor(aReady);
        yBatchQue_.EnQue(yBatch);
        auto yReady = yBatchQue_.DeQue<half>();
        EncodeBatch(yReady, norms, packedBatch, m, 0, TQ_PACK_D, true);
        yBatchQue_.FreeTensor(yReady);

        packedRowQue_.EnQue(packedBatch);
        xBatchQue_.FreeTensor(xBatch);
    }

    __aicore__ inline void PackMergedBatch(uint32_t start, uint32_t end) {
        const uint32_t m = end - start;

        CopyInMergedBatch(start, end);
        Compute(m);
        CopyOutPackedBatch(start, m);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t slot_w_k_ = 0;
    uint32_t slot_w_v_ = 0;
    uint32_t vecPerCore_ = 1;

    AscendC::TQue<AscendC::TPosition::VECIN, 2> xBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> aBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> yBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> xBatchInputQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> packedRowQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotationTBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> distBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cbTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> argminResultBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf16SquareBuf_;
    uint32_t maxSlotW_ = 0;
    uint32_t packedStride_ = 0;

    AscendC::GlobalTensor<TqInputT> keyGm_;
    AscendC::GlobalTensor<TqInputT> valueGm_;
    AscendC::GlobalTensor<half> codebookGm_;
    AscendC::GlobalTensor<half> rotationTGm_;
    AscendC::GlobalTensor<uint8_t> packedKGm_;
    AscendC::GlobalTensor<uint8_t> packedVGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache_v3(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR codebook,
    GM_ADDR rotation_t,
    GM_ADDR packed_k,
    GM_ADDR packed_v,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    auto* codebookPtr = reinterpret_cast<__gm__ half*>(codebook);
    auto* rotationPtr = reinterpret_cast<__gm__ half*>(rotation_t);
    auto* packedKPtr = reinterpret_cast<__gm__ uint8_t*>(packed_k);
    auto* packedVPtr = reinterpret_cast<__gm__ uint8_t*>(packed_v);

    if (!TILING_KEY_IS(0)) {
        return;
    }
    (void)workspace;
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    TurboquantPackKVForCacheV3 op(&pipe);
    op.Init(
        key,
        value,
        codebookPtr,
        rotationPtr,
        packedKPtr,
        packedVPtr,
        nullptr,
        tilingData.nVec,
        tilingData.slotWK,
        tilingData.slotWV,
        tilingData.vecPerCore);
    op.Process();
}
