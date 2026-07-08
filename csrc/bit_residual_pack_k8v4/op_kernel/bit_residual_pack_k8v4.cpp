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

// Fused bit-residual pack for KV cache (fp16/bf16).
//
// Reference (Python):
//   norms = vector_norm(x, dim=-1, keepdim=True)
//   y = (x / (norms + eps)) @ R^T        # batched [M, D] @ [D, D]
//   key:   sign + 7-bit residual, 2 rows packed per uint16
//   value: 4-bit uniform quantization, 4 rows packed per uint16
//
// Per AICore: normalize all assigned rows, then ONE matmul [M,128]@[128,128],
// not M separate M=1 matmuls (avoids Cube padding waste).

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

static constexpr int TQ_PACK_D = 128;
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_KEY_GROUP_ROWS = 2;
static constexpr uint32_t TQ_VAL_GROUP_ROWS = 4;
static constexpr uint32_t TQ_GROUP_INDEX_BYTES = TQ_PACK_D * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_GROUP_NORM_BYTES = TQ_KEY_GROUP_ROWS * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_GROUP_BASE_BYTES = TQ_KEY_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_KEY_GROUP_STEP_BYTES = TQ_KEY_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_KEY_GROUP_NORM_OFFSET = TQ_GROUP_INDEX_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_BASE_OFFSET =
    TQ_KEY_GROUP_NORM_OFFSET + TQ_KEY_GROUP_NORM_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_STEP_OFFSET =
    TQ_KEY_GROUP_BASE_OFFSET + TQ_KEY_GROUP_BASE_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_BYTES =
    TQ_GROUP_INDEX_BYTES + TQ_KEY_GROUP_NORM_BYTES +
    TQ_KEY_GROUP_BASE_BYTES + TQ_KEY_GROUP_STEP_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_STRIDE =
    (TQ_KEY_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_VAL_GROUP_VMIN_BYTES = TQ_VAL_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_VAL_GROUP_VSTEP_BYTES = TQ_VAL_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_VAL_GROUP_VMIN_OFFSET = TQ_GROUP_INDEX_BYTES;
static constexpr uint32_t TQ_VAL_GROUP_VSTEP_OFFSET =
    TQ_VAL_GROUP_VMIN_OFFSET + TQ_VAL_GROUP_VMIN_BYTES;
static constexpr uint32_t TQ_VAL_GROUP_BYTES =
    TQ_GROUP_INDEX_BYTES + TQ_VAL_GROUP_VMIN_BYTES + TQ_VAL_GROUP_VSTEP_BYTES;
static constexpr uint32_t TQ_VAL_GROUP_STRIDE =
    (TQ_VAL_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_PACKED_GROUP_STRIDE =
    TQ_KEY_GROUP_STRIDE > TQ_VAL_GROUP_STRIDE ? TQ_KEY_GROUP_STRIDE : TQ_VAL_GROUP_STRIDE;
static constexpr uint32_t TQ_PACKED_GROUP_BUFFER_COUNT = 64;
static constexpr uint32_t TQ_KEY_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(uint16_t) + 2 * sizeof(float);
static constexpr uint32_t TQ_KEY_ENCODED_NORM_WORD_OFFSET = TQ_PACK_D;
static constexpr uint32_t TQ_KEY_ENCODED_BASE_BYTE_OFFSET =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_ENCODED_STEP_BYTE_OFFSET =
    TQ_KEY_ENCODED_BASE_BYTE_OFFSET + sizeof(float);
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_BYTES =
    (TQ_KEY_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_WORDS =
    TQ_KEY_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(float);
static constexpr uint32_t TQ_VAL_ENCODED_VMIN_BYTE_OFFSET = TQ_PACK_D * sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET =
    TQ_VAL_ENCODED_VMIN_BYTE_OFFSET + sizeof(float);
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_BYTES =
    (TQ_VAL_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_WORDS =
    TQ_VAL_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
static constexpr uint32_t TQ_MAX_BATCH_M = 64;
static constexpr uint32_t TQ_QUEUE_DEPTH = 2;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_DTYPE_BYTES = sizeof(uint16_t);
static constexpr uint32_t TQ_NORM_STRIDE = TQ_UB_ALIGN / TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_ROT_LOCAL_WORKSPACE_BYTES = TQ_MAX_BATCH_M * TQ_ROT_K * TQ_DTYPE_BYTES;
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr float TQ_INV_SQRT_D_F = 0.08838834764831845f;
static constexpr float TQ_KEY_QUANT_LEVELS_F = 127.0f;
static constexpr float TQ_VAL_QUANT_LEVELS_F = 15.0f;
static constexpr uint32_t TQ_SIGN_MASK_BYTES = 256;
static constexpr uint32_t TQ_QUANT_INDEX_BYTES = TQ_PACK_D * sizeof(int32_t);
static constexpr uint32_t TQ_QUANT_INDEX_U16_BYTES = TQ_PACK_D * sizeof(int16_t);
static constexpr uint32_t TQ_AIV_SUB_BLOCKS = 2;

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

__aicore__ inline void TqSyncMte3ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e);
    WaitFlag<HardEvent::MTE3_V>(e);
}

__aicore__ inline void TqSyncMte3ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(e);
    WaitFlag<HardEvent::MTE3_S>(e);
}

// Group rows are not necessarily 32B-aligned; use DataCopyPad for GM writes.
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
    TqSyncMte3ToV();
    TqSyncMte3ToS();
}

__aicore__ inline void copy_packed_ub_to_gm_async(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t>& packedLocal,
    uint32_t nbytes) {
    TqSyncVToMte3();
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
}

__aicore__ inline void wait_packed_ub_to_gm_async() {
    TqSyncMte3ToMte2();
    TqSyncMte3ToV();
    TqSyncMte3ToS();
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

__aicore__ inline constexpr MatmulConfig TqRotateMatmulConfig() {
    constexpr MatmulShapeParams shapeParams = {
        TQ_MAX_BATCH_M, TQ_ROT_N, TQ_ROT_K,
        TQ_MAX_BATCH_M, TQ_ROT_N, TQ_ROT_K};
    constexpr MatmulBiasParams biasParams = {false};
    return GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams, biasParams);
}

template <typename T>
__aicore__ inline constexpr MatmulApiStaticTiling TqRotateMatmulTiling() {
    MatmulApiStaticTiling tiling =
        GetMatmulApiTiling<TqRotateAT<T>, TqRotateBT<T>, TqRotateCT<T>, TqRotateBiasT<T>>(
            TqRotateMatmulConfig());
    // Each AIV worker issues its own KFC matmul request.
    tiling.usedCoreNum = 1;
    return tiling;
}

template <typename T>
static constexpr MatmulApiStaticTiling TQ_ROTATE_MATMUL_TILING = TqRotateMatmulTiling<T>();

template <typename T>
using TqRotateMatmulOp =
    AscendC::Matmul<TqRotateAT<T>, TqRotateBT<T>, TqRotateCT<T>, TqRotateBiasT<T>,
                    TQ_ROTATE_MATMUL_TILING<T>>;

template <typename T, typename RotateMmT>
class BitResidualPackK8v4 {
public:
    __aicore__ inline explicit BitResidualPackK8v4(
        AscendC::TPipe* pipe,
        RotateMmT* rotateMm,
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t vecPerCore,
        uint32_t numHeads,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t dataCores,
        uint32_t numReqs,
        uint32_t keyStrideToken,
        uint32_t keyStrideHead,
        uint32_t valueStrideToken,
        uint32_t valueStrideHead,
        uint64_t keyStorageOffset,
        uint64_t valueStorageOffset)
        : pipe_(pipe),
          rotateMm_(rotateMm),
          nVec_(nVec),
          vecPerCore_(vecPerCore > TQ_MAX_BATCH_M ? TQ_MAX_BATCH_M : vecPerCore),
          numHeads_(numHeads == 0 ? 1 : numHeads),
          tokenCount_((nVec + (numHeads == 0 ? 1 : numHeads) - 1) / (numHeads == 0 ? 1 : numHeads)),
          blockSize_(blockSize == 0 ? 1 : blockSize),
          numBlocks_(numBlocks),
          cacheSlots_((blockSize == 0 ? 1 : blockSize) * numBlocks),
          dataCores_(dataCores == 0 ? 1 : dataCores),
          numReqs_(numReqs),
          keyStrideToken_(keyStrideToken),
          keyStrideHead_(keyStrideHead),
          valueStrideToken_(valueStrideToken),
          valueStrideHead_(valueStrideHead),
          keyStorageOffset_(keyStorageOffset),
          valueStorageOffset_(valueStorageOffset),
          keySpan_(MakeInputSpan(keyStorageOffset, keyStrideToken, keyStrideHead)),
          valueSpan_(MakeInputSpan(valueStorageOffset, valueStrideToken, valueStrideHead)),
          matmulReady_(rawWorkspace != nullptr),
          packedWritePending_(false),
          packedGroupSlot_(0) {}

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ T* rotation_t,
        __gm__ int32_t* slot_mapping,
        __gm__ int32_t* query_start_loc,
        __gm__ uint8_t* key_cache,
        __gm__ uint8_t* value_cache) {
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key), keySpan_);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), valueSpan_);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        slotMappingGm_.SetGlobalBuffer(slot_mapping, tokenCount_);
        queryStartLocGm_.SetGlobalBuffer(query_start_loc, static_cast<uint64_t>(numReqs_) + 1);
        keyCacheGm_.SetGlobalBuffer(
            key_cache,
            (uint64_t)numBlocks_ * numHeads_ *
                (blockSize_ / TQ_KEY_GROUP_ROWS) * TQ_KEY_GROUP_STRIDE);
        valueCacheGm_.SetGlobalBuffer(
            value_cache,
            (uint64_t)numBlocks_ * numHeads_ *
                (blockSize_ / TQ_VAL_GROUP_ROWS) * TQ_VAL_GROUP_STRIDE);

        const uint32_t batchElems = TQ_MAX_BATCH_M * TQ_PACK_D;
        pipe_->InitBuffer(xBatchQue_, TQ_QUEUE_DEPTH, batchElems * sizeof(T));
        pipe_->InitBuffer(aBatchQue_, TQ_QUEUE_DEPTH, batchElems * sizeof(T));
        pipe_->InitBuffer(yBatchQue_, TQ_QUEUE_DEPTH, batchElems * sizeof(T));
        pipe_->InitBuffer(normScalarBuf_, TQ_UB_ALIGN);
        pipe_->InitBuffer(normsBuf_, TQ_MAX_BATCH_M * TQ_NORM_STRIDE * sizeof(T));
        pipe_->InitBuffer(signMaskBuf_, TQ_SIGN_MASK_BYTES);
        // yFp32Buf: fp32 y row [D=128]
        pipe_->InitBuffer(yFp32Buf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(signValBuf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(errBuf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(quantIndexBuf_, TQ_QUANT_INDEX_BYTES);
        pipe_->InitBuffer(quantIndexU16Buf_, TQ_QUANT_INDEX_U16_BYTES);
        pipe_->InitBuffer(reduceScalarBuf_, TQ_UB_ALIGN);
        // NormalizeBatch: fp32 row + fp32 squared row + fp32 ReduceSum tmp.
        pipe_->InitBuffer(reduceOutBuf_, TQ_PACK_D * 3 * sizeof(float));
        pipe_->InitBuffer(reduceTmpBuf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(packedRowBuf_, TQ_PACKED_GROUP_BUFFER_COUNT * TQ_PACKED_GROUP_STRIDE * sizeof(uint8_t));
        pipe_->InitBuffer(packMergeBuf_, TQ_GROUP_INDEX_BYTES);
        pipe_->InitBuffer(packMaskBuf_, TQ_GROUP_INDEX_BYTES);
        pipe_->InitBuffer(keyEncodedQue_, TQ_QUEUE_DEPTH,
                          TQ_MAX_BATCH_M * TQ_KEY_ENCODED_ROW_STRIDE_BYTES);
        pipe_->InitBuffer(valEncodedQue_, TQ_QUEUE_DEPTH,
                          TQ_MAX_BATCH_M * TQ_VAL_ENCODED_ROW_STRIDE_BYTES);
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_LOCAL_WORKSPACE_BYTES);
    }

    __aicore__ inline void Process() {
        if (!matmulReady_) {
            return;
        }
        if ASCEND_IS_AIC {
            return;
        }
        const uint32_t worker = AscendC::GetBlockIdx();
        if (worker >= GetActiveWorkers()) {
            return;
        }
        ProcessCacheBySequenceContiguousSegments<true>(worker);
        WaitPendingPackedWrites();
        ProcessCacheBySequenceContiguousSegments<false>(worker);
        WaitPendingPackedWrites();
    }

private:

    // norms[i] = ||x[i]||; xBatch rows unitized in-place (matches x / (norm + eps)).
    // Inner dim: Cast + Mul + ReduceSum (vector), not scalar loop; fp32 acc avoids 16-bit overflow.
    __aicore__ inline void NormalizeBatch(uint32_t m) {
        if constexpr (TILING_KEY_IS(1)) {
            NormalizeBatchBrcbScale(m);
        } else {
            NormalizeBatchScalarScale(m);
        }
    }

    __aicore__ inline void NormalizeBatchScalarScale(uint32_t m) {
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
            AscendC::Cast(aBatch[rowOff], fp32Row, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        aBatchQue_.EnQue(aBatch);
        xBatchQue_.FreeTensor(xBatch);
    }

    __aicore__ inline void NormalizeBatchBrcbScale(uint32_t m) {
        auto xBatch = xBatchQue_.DeQue<T>();
        auto aBatch = aBatchQue_.AllocTensor<T>();
        auto norms = normsBuf_.Get<T>();
        auto fp32Row = reduceOutBuf_.Get<float>();
        auto scaleBlock = reduceOutBuf_.Get<float>()[TQ_PACK_D];
        auto fp32Tmp = reduceOutBuf_.Get<float>()[TQ_PACK_D * 2];
        auto normAcc = normScalarBuf_.Get<float>();

        TqSyncMte2ToV();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;

            AscendC::Cast(fp32Row, xBatch[rowOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(scaleBlock, fp32Row, fp32Row, TQ_PACK_D);
            AscendC::ReduceSum<float>(normAcc, scaleBlock, fp32Tmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(normAcc, normAcc, 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(norms[i * TQ_NORM_STRIDE], normAcc, AscendC::RoundMode::CAST_RINT, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(normAcc, normAcc, TQ_NORM_EPS_F, 1);
            AscendC::Duplicate(fp32Tmp, 1.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(normAcc, fp32Tmp, normAcc, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(scaleBlock, normAcc, 1, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(
                fp32Row,
                fp32Row,
                scaleBlock,
                static_cast<uint64_t>(TQ_PACK_D / 2),
                2,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(aBatch[rowOff], fp32Row, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

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

    __aicore__ inline void EncodeKeyBatch(uint32_t m) {
        auto yBatch = yBatchQue_.DeQue<T>();
        auto encodedBatch = keyEncodedQue_.AllocTensor<uint16_t>();
        auto norms = normsBuf_.Get<T>();
        auto yFp32 = yFp32Buf_.Get<float>();
        auto signMask = signMaskBuf_.Get<uint8_t>();
        auto signVal = signValBuf_.Get<float>();
        auto err = errBuf_.Get<float>();
        auto qI32 = quantIndexBuf_.Get<int32_t>();
        auto qI16 = quantIndexU16Buf_.Get<int16_t>();
        auto codeU16 = packMergeBuf_.Get<uint16_t>();
        auto signI16 = packMaskBuf_.Get<int16_t>();
        auto reduceScalar = reduceScalarBuf_.Get<float>();
        auto baseAcc = reduceScalar;
        auto maxAcc = reduceScalar[1];
        auto reduceTmp = reduceTmpBuf_.Get<float>();
        auto normWords = norms.template ReinterpretCast<uint16_t>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_KEY_ENCODED_ROW_STRIDE_WORDS;
            const uint32_t normOff = i * TQ_NORM_STRIDE;

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(signMask, yFp32, -0.0f, AscendC::CMPMODE::GT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(signVal, TQ_INV_SQRT_D_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                signVal,
                signMask,
                signVal,
                -TQ_INV_SQRT_D_F,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(err, yFp32, signVal, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceMin<float>(baseAcc, err, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(maxAcc, err, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            const float baseF = baseAcc.GetValue(0);
            const float maxF = maxAcc.GetValue(0);
            const float rangeF = maxF - baseF;
            const float stepF = rangeF > 0.0f ? rangeF / TQ_KEY_QUANT_LEVELS_F : 1.0f;
            const float invStepF = rangeF > 0.0f ? 1.0f / stepF : 0.0f;
            TqSyncSToV();

            AscendC::Adds(err, err, -baseF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(err, err, invStepF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(err, err, 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(err, err, TQ_KEY_QUANT_LEVELS_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI32, err, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, qI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(
                codeU16,
                qI16.template ReinterpretCast<uint16_t>(),
                static_cast<uint16_t>(1),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Duplicate(signVal, 1.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                signVal,
                signMask,
                signVal,
                0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI32, signVal, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signI16, qI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(
                codeU16,
                codeU16,
                signI16.template ReinterpretCast<uint16_t>(),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(encodedBatch[encodedOff], codeU16, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            encodedBatch.SetValue(
                encodedOff + TQ_KEY_ENCODED_NORM_WORD_OFFSET,
                normWords.GetValue(normOff));
            auto baseOut = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_BASE_BYTE_OFFSET / sizeof(uint16_t)]
                               .template ReinterpretCast<float>();
            auto stepOut = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_STEP_BYTE_OFFSET / sizeof(uint16_t)]
                               .template ReinterpretCast<float>();
            baseOut.SetValue(0, baseF);
            stepOut.SetValue(0, stepF);
        }
        TqSyncSToV();

        keyEncodedQue_.EnQue(encodedBatch);
        yBatchQue_.FreeTensor(yBatch);
    }

    __aicore__ inline void EncodeValueBatch(uint32_t m) {
        auto yBatch = yBatchQue_.DeQue<T>();
        auto encodedBatch = valEncodedQue_.AllocTensor<uint16_t>();
        auto yFp32 = yFp32Buf_.Get<float>();
        auto qFp32 = errBuf_.Get<float>();
        auto qI32 = quantIndexBuf_.Get<int32_t>();
        auto qI16 = quantIndexU16Buf_.Get<int16_t>();
        auto reduceScalar = reduceScalarBuf_.Get<float>();
        auto vminAcc = reduceScalar;
        auto vmaxAcc = reduceScalar[1];
        auto reduceTmp = reduceTmpBuf_.Get<float>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_VAL_ENCODED_ROW_STRIDE_WORDS;

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMin<float>(vminAcc, yFp32, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(vmaxAcc, yFp32, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            const float vminF = vminAcc.GetValue(0);
            const float vmaxF = vmaxAcc.GetValue(0);
            const float rangeF = vmaxF - vminF;
            const float vstepF = rangeF > 0.0f ? rangeF / TQ_VAL_QUANT_LEVELS_F : 1.0f;
            const float invStepF = rangeF > 0.0f ? 1.0f / vstepF : 0.0f;
            TqSyncSToV();

            AscendC::Adds(qFp32, yFp32, -vminF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(qFp32, qFp32, invStepF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(qFp32, qFp32, 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(qFp32, qFp32, TQ_VAL_QUANT_LEVELS_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI32, qFp32, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, qI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(
                encodedBatch[encodedOff].template ReinterpretCast<int16_t>(),
                qI16,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            auto vminOut = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VMIN_BYTE_OFFSET / sizeof(uint16_t)]
                               .template ReinterpretCast<float>();
            auto vstepOut = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET / sizeof(uint16_t)]
                                .template ReinterpretCast<float>();
            vminOut.SetValue(0, vminF);
            vstepOut.SetValue(0, vstepF);
        }
        TqSyncSToV();

        valEncodedQue_.EnQue(encodedBatch);
        yBatchQue_.FreeTensor(yBatch);
    }

    __aicore__ inline uint32_t MakeVecIndex(
        uint32_t tokenIdx,
        uint32_t headIdx) const {
        return tokenIdx * numHeads_ + headIdx;
    }

    __aicore__ inline uint64_t MakeInputSpan(
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) const {
        if (tokenCount_ == 0 || numHeads_ == 0) {
            return storageOffset;
        }
        return storageOffset +
               static_cast<uint64_t>(tokenCount_ - 1) * strideToken +
               static_cast<uint64_t>(numHeads_ - 1) * strideHead +
               TQ_PACK_D;
    }

    __aicore__ inline uint64_t MakeInputOffset(
        uint32_t vecIdx,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) const {
        const uint32_t tokenIdx = vecIdx / numHeads_;
        const uint32_t headIdx = vecIdx - tokenIdx * numHeads_;
        return storageOffset +
               static_cast<uint64_t>(tokenIdx) * strideToken +
               static_cast<uint64_t>(headIdx) * strideHead;
    }

    __aicore__ inline bool IsContiguousVecBatch(
        const uint32_t* vecIndices,
        uint32_t rows,
        uint32_t strideToken,
        uint32_t strideHead) const {
        if (rows == 0 ||
            strideHead != TQ_PACK_D ||
            strideToken != numHeads_ * TQ_PACK_D) {
            return false;
        }
        const uint32_t firstVec = vecIndices[0];
        for (uint32_t row = 1; row < rows; ++row) {
            if (vecIndices[row] != firstVec + row) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void CopyInIndexedTask(
        const AscendC::GlobalTensor<T>& xGm,
        const uint32_t* vecIndices,
        uint32_t rows,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead,
        uint32_t& m) {
        if (rows == 0) {
            m = 0;
            return;
        }
        auto xBatch = xBatchQue_.AllocTensor<T>();
        if (IsContiguousVecBatch(vecIndices, rows, strideToken, strideHead)) {
            AscendC::DataCopy(
                xBatch,
                xGm[MakeInputOffset(vecIndices[0], storageOffset, strideToken, strideHead)],
                rows * TQ_PACK_D);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                AscendC::DataCopy(
                    xBatch[row * TQ_PACK_D],
                    xGm[MakeInputOffset(vecIndices[row], storageOffset, strideToken, strideHead)],
                    TQ_PACK_D);
            }
        }
        m = rows;
        xBatchQue_.EnQue(xBatch);
    }

    __aicore__ inline void CopyInPhysicalGroupRowsTask(
        const AscendC::GlobalTensor<T>& xGm,
        uint32_t tokenStart,
        uint32_t rowCount,
        uint32_t headStart,
        uint32_t headCount,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead,
        uint32_t& m) {
        if (rowCount == 0 || headCount == 0) {
            m = 0;
            return;
        }

        auto xBatch = xBatchQue_.AllocTensor<T>();
        if (headStart == 0 && headCount == numHeads_ &&
            strideHead == TQ_PACK_D && strideToken == numHeads_ * TQ_PACK_D) {
            AscendC::DataCopy(
                xBatch,
                xGm[storageOffset + static_cast<uint64_t>(tokenStart) * strideToken],
                rowCount * headCount * TQ_PACK_D);
        } else {
            uint32_t dstRow = 0;
            for (uint32_t row = 0; row < rowCount; ++row) {
                for (uint32_t headOff = 0; headOff < headCount; ++headOff) {
                    const uint32_t vecIdx = MakeVecIndex(tokenStart + row, headStart + headOff);
                    AscendC::DataCopy(
                        xBatch[dstRow * TQ_PACK_D],
                        xGm[MakeInputOffset(vecIdx, storageOffset, strideToken, strideHead)],
                        TQ_PACK_D);
                    ++dstRow;
                }
            }
        }
        m = rowCount * headCount;
        xBatchQue_.EnQue(xBatch);
    }

    __aicore__ inline void ComputeKeyBatch(uint32_t m) {
        NormalizeBatch(m);
        RotateBatchMatmul(m, 0, TQ_PACK_D);
        EncodeKeyBatch(m);
    }

    __aicore__ inline void ComputeValueBatch(uint32_t m) {
        NormalizeBatch(m);
        RotateBatchMatmul(m, 0, TQ_PACK_D);
        EncodeValueBatch(m);
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t GroupRows() const {
        return IS_KEY ? TQ_KEY_GROUP_ROWS : TQ_VAL_GROUP_ROWS;
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t GroupBytes() const {
        return IS_KEY ? TQ_KEY_GROUP_BYTES : TQ_VAL_GROUP_BYTES;
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t GroupStride() const {
        return IS_KEY ? TQ_KEY_GROUP_STRIDE : TQ_VAL_GROUP_STRIDE;
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t EncodedRowStrideWords() const {
        return IS_KEY ? TQ_KEY_ENCODED_ROW_STRIDE_WORDS : TQ_VAL_ENCODED_ROW_STRIDE_WORDS;
    }

    __aicore__ inline void ClearPackedGroup(AscendC::LocalTensor<uint8_t>& packedGroup) const {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        AscendC::Duplicate(
            packedU16,
            static_cast<uint16_t>(0),
            TQ_PACKED_GROUP_STRIDE / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool IS_KEY>
    __aicore__ inline void MergeEncodedRowToGroup(
        AscendC::LocalTensor<uint8_t>& packedGroup,
        const AscendC::LocalTensor<uint16_t>& encodedBatch,
        uint32_t encodedRow,
        uint32_t groupRow,
        bool clearOldBits) {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        auto shiftedIdx = packMergeBuf_.Get<uint16_t>();
        const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
        const uint32_t shiftBits = groupRow * (IS_KEY ? 8 : 4);

        if (clearOldBits) {
            auto clearMask = packMaskBuf_.Get<uint16_t>();
            const uint16_t mask = static_cast<uint16_t>(
                IS_KEY
                    ? ~static_cast<uint16_t>(0x00FFu << shiftBits)
                    : ~static_cast<uint16_t>(0x000Fu << shiftBits));
            AscendC::Duplicate(clearMask, mask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(packedU16, packedU16, clearMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::ShiftLeft(
            shiftedIdx,
            encodedBatch[encodedOff],
            static_cast<uint16_t>(shiftBits),
            TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(packedU16, packedU16, shiftedIdx, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (IS_KEY) {
            const uint32_t normWord = TQ_KEY_GROUP_NORM_OFFSET / sizeof(uint16_t) + groupRow;
            packedU16.SetValue(
                normWord,
                encodedBatch.GetValue(encodedOff + TQ_KEY_ENCODED_NORM_WORD_OFFSET));
            auto packedBase = packedGroup[TQ_KEY_GROUP_BASE_OFFSET].template ReinterpretCast<float>();
            auto packedStep = packedGroup[TQ_KEY_GROUP_STEP_OFFSET].template ReinterpretCast<float>();
            auto encodedBase = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_BASE_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            auto encodedStep = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_STEP_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            packedBase.SetValue(groupRow, encodedBase.GetValue(0));
            packedStep.SetValue(groupRow, encodedStep.GetValue(0));
        } else {
            auto packedVmin = packedGroup[TQ_VAL_GROUP_VMIN_OFFSET].template ReinterpretCast<float>();
            auto packedVstep = packedGroup[TQ_VAL_GROUP_VSTEP_OFFSET].template ReinterpretCast<float>();
            auto encodedVmin = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VMIN_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            auto encodedVstep = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET / sizeof(uint16_t)]
                                    .template ReinterpretCast<float>();
            packedVmin.SetValue(groupRow, encodedVmin.GetValue(0));
            packedVstep.SetValue(groupRow, encodedVstep.GetValue(0));
        }
    }

    template <bool IS_KEY>
    __aicore__ inline void InitFullGroupFromRow0(
        AscendC::LocalTensor<uint8_t>& packedGroup,
        const AscendC::LocalTensor<uint16_t>& encodedBatch,
        uint32_t encodedRow) {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
        AscendC::DataCopy(packedU16, encodedBatch[encodedOff], TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        MergeEncodedRowToGroup<IS_KEY>(packedGroup, encodedBatch, encodedRow, 0, true);
    }

    template <bool IS_KEY>
    __aicore__ inline void FlushResolvedGroups(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        AscendC::LocalTensor<uint16_t>& encodedBatch,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        const uint8_t* preserveRows,
        const uint8_t* validRows,
        uint32_t m) {
        auto packedGroup = packedRowBuf_.Get<uint8_t>();
        uint32_t sortedRows[TQ_MAX_BATCH_M];
        uint32_t rowForGroup[TQ_VAL_GROUP_ROWS];
        uint32_t validCount = 0;
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();

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
            bool preserveExisting = false;
            for (uint32_t groupRow = 0; groupRow < groupRowsPerGroup; ++groupRow) {
                rowForGroup[groupRow] = TQ_MAX_BATCH_M;
            }
            do {
                const uint32_t srcRow = sortedRows[row];
                const uint32_t groupRow = groupRows[srcRow];
                rowForGroup[groupRow] = srcRow;
                preserveExisting = preserveExisting || preserveRows[srcRow] != 0;
                rowMask |= (1u << groupRow);
                ++row;
            } while (row < validCount && groupBases[sortedRows[row]] == groupBase);

            if (rowMask == ((1u << groupRowsPerGroup) - 1u) || !preserveExisting) {
                ClearPackedGroup(packedGroup);
                for (uint32_t groupRow = 0; groupRow < groupRowsPerGroup; ++groupRow) {
                    if ((rowMask & (1u << groupRow)) == 0) {
                        continue;
                    }
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, rowForGroup[groupRow], groupRow, false);
                }
            } else {
                copy_packed_gm_to_ub(packedGroup, packedGm, groupBase, GroupBytes<IS_KEY>());
                for (uint32_t groupRow = 0; groupRow < groupRowsPerGroup; ++groupRow) {
                    if ((rowMask & (1u << groupRow)) == 0) {
                        continue;
                    }
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, rowForGroup[groupRow], groupRow, true);
                }
            }

            copy_packed_ub_to_gm(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
        }
    }

    __aicore__ inline void WaitPendingPackedWrites() {
        if (!packedWritePending_) {
            return;
        }
        wait_packed_ub_to_gm_async();
        packedWritePending_ = false;
        packedGroupSlot_ = 0;
    }

    __aicore__ inline void MarkPackedWritePending() {
        packedWritePending_ = true;
    }

    template <bool IS_KEY>
    __aicore__ inline AscendC::LocalTensor<uint16_t> DeQueEncodedBatch() {
        if constexpr (IS_KEY) {
            return keyEncodedQue_.template DeQue<uint16_t>();
        } else {
            return valEncodedQue_.template DeQue<uint16_t>();
        }
    }

    template <bool IS_KEY>
    __aicore__ inline void FreeEncodedBatch(AscendC::LocalTensor<uint16_t>& encodedBatch) {
        if constexpr (IS_KEY) {
            keyEncodedQue_.FreeTensor(encodedBatch);
        } else {
            valEncodedQue_.FreeTensor(encodedBatch);
        }
    }

    template <bool IS_KEY>
    __aicore__ inline uint64_t MakeCacheGroupBaseOffset(
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t headIdx) const {
        return ((uint64_t)blockIdx * numHeads_ + headIdx) *
                   (blockSize_ / GroupRows<IS_KEY>()) * GroupStride<IS_KEY>() +
               static_cast<uint64_t>(groupInBlock) * GroupStride<IS_KEY>();
    }

    template <bool IS_KEY>
    __aicore__ inline bool CopyOutPhysicalGroupRowsFast(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        AscendC::LocalTensor<uint16_t>& encodedBatch,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        const uint8_t* preserveRows,
        uint32_t m) {
        if (m == 0) {
            return true;
        }

        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();
        const uint32_t firstGroupRow = groupRows[0];
        if (firstGroupRow >= groupRowsPerGroup) {
            return false;
        }

        const uint64_t firstGroupBase = groupBases[0];
        const uint64_t headGroupStride =
            static_cast<uint64_t>(blockSize_ / groupRowsPerGroup) * GroupStride<IS_KEY>();
        const uint8_t preserveExisting = preserveRows[0];
        uint32_t headCount = 1;
        while (headCount < m && groupRows[headCount] == firstGroupRow &&
               preserveRows[headCount] == preserveExisting &&
               groupBases[headCount] == firstGroupBase + static_cast<uint64_t>(headCount) * headGroupStride) {
            ++headCount;
        }
        if (headCount == 0 || m % headCount != 0) {
            return false;
        }

        const uint32_t rowCount = m / headCount;
        if (rowCount == 0 || rowCount > groupRowsPerGroup ||
            firstGroupRow + rowCount > groupRowsPerGroup) {
            return false;
        }

        for (uint32_t row = 0; row < rowCount; ++row) {
            for (uint32_t headOff = 0; headOff < headCount; ++headOff) {
                const uint32_t srcRow = row * headCount + headOff;
                if (groupRows[srcRow] != firstGroupRow + row ||
                    preserveRows[srcRow] != preserveExisting ||
                    groupBases[srcRow] != firstGroupBase + static_cast<uint64_t>(headOff) * headGroupStride) {
                    return false;
                }
            }
        }

        const uint32_t rowMask = ((1u << rowCount) - 1u) << firstGroupRow;
        const bool fullGroup = rowMask == ((1u << groupRowsPerGroup) - 1u);
        auto packedGroups = packedRowBuf_.Get<uint8_t>();
        for (uint32_t headOff = 0; headOff < headCount; ++headOff) {
            const uint64_t groupBase = firstGroupBase + static_cast<uint64_t>(headOff) * headGroupStride;
            if (packedWritePending_ && packedGroupSlot_ == 0) {
                WaitPendingPackedWrites();
            }
            const uint32_t groupSlot = packedGroupSlot_;
            ++packedGroupSlot_;
            if (packedGroupSlot_ >= TQ_PACKED_GROUP_BUFFER_COUNT) {
                packedGroupSlot_ = 0;
            }
            auto packedGroup = packedGroups[groupSlot * TQ_PACKED_GROUP_STRIDE];
            if (fullGroup && firstGroupRow == 0 && rowCount == groupRowsPerGroup) {
                InitFullGroupFromRow0<IS_KEY>(packedGroup, encodedBatch, headOff);
                for (uint32_t row = 1; row < groupRowsPerGroup; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, row * headCount + headOff, row, false);
                }
            } else if (fullGroup || preserveExisting == 0) {
                ClearPackedGroup(packedGroup);
                for (uint32_t row = 0; row < rowCount; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, row * headCount + headOff, firstGroupRow + row, false);
                }
            } else {
                WaitPendingPackedWrites();
                copy_packed_gm_to_ub(packedGroup, packedGm, groupBase, GroupBytes<IS_KEY>());
                for (uint32_t row = 0; row < rowCount; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, row * headCount + headOff, firstGroupRow + row, true);
                }
                copy_packed_ub_to_gm(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
                continue;
            }
            copy_packed_ub_to_gm_async(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
            MarkPackedWritePending();
        }
        return true;
    }

    template <bool IS_KEY>
    __aicore__ inline void CopyOutPhysicalGroupRowsKnown(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t firstGroupRow,
        uint32_t headStart,
        uint32_t headCount,
        bool preserveExisting) {
        auto encodedBatch = DeQueEncodedBatch<IS_KEY>();
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();
        const uint32_t rowMask = ((1u << rowCount) - 1u) << firstGroupRow;
        const bool fullGroup = rowMask == ((1u << groupRowsPerGroup) - 1u);
        auto packedGroups = packedRowBuf_.Get<uint8_t>();

        for (uint32_t headOff = 0; headOff < headCount; ++headOff) {
            const uint32_t headIdx = headStart + headOff;
            const uint64_t groupBase = MakeCacheGroupBaseOffset<IS_KEY>(blockIdx, groupInBlock, headIdx);
            if (packedWritePending_ && packedGroupSlot_ == 0) {
                WaitPendingPackedWrites();
            }
            const uint32_t groupSlot = packedGroupSlot_;
            ++packedGroupSlot_;
            if (packedGroupSlot_ >= TQ_PACKED_GROUP_BUFFER_COUNT) {
                packedGroupSlot_ = 0;
            }
            auto packedGroup = packedGroups[groupSlot * TQ_PACKED_GROUP_STRIDE];
            if (fullGroup && firstGroupRow == 0 && rowCount == groupRowsPerGroup) {
                InitFullGroupFromRow0<IS_KEY>(packedGroup, encodedBatch, headOff);
                for (uint32_t row = 1; row < groupRowsPerGroup; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, row * headCount + headOff, row, false);
                }
                copy_packed_ub_to_gm_async(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
                MarkPackedWritePending();
            } else if (fullGroup || !preserveExisting) {
                ClearPackedGroup(packedGroup);
                for (uint32_t row = 0; row < rowCount; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, row * headCount + headOff, firstGroupRow + row, false);
                }
                copy_packed_ub_to_gm_async(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
                MarkPackedWritePending();
            } else {
                WaitPendingPackedWrites();
                copy_packed_gm_to_ub(packedGroup, packedGm, groupBase, GroupBytes<IS_KEY>());
                for (uint32_t row = 0; row < rowCount; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, row * headCount + headOff, firstGroupRow + row, true);
                }
                copy_packed_ub_to_gm(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
            }
        }

        FreeEncodedBatch<IS_KEY>(encodedBatch);
    }

    template <bool IS_KEY>
    __aicore__ inline void CopyOutPhysicalFullGroupRunKnown(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t firstGroupInBlock,
        uint32_t headStart,
        uint32_t headCount) {
        auto encodedBatch = DeQueEncodedBatch<IS_KEY>();
        auto packedGroups = packedRowBuf_.Get<uint8_t>();
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();
        const uint32_t groupCount = rowCount / groupRowsPerGroup;

        for (uint32_t groupOff = 0; groupOff < groupCount; ++groupOff) {
            const uint32_t rowBase = groupOff * groupRowsPerGroup;
            const uint32_t groupInBlock = firstGroupInBlock + groupOff;
            for (uint32_t headOff = 0; headOff < headCount; ++headOff) {
                const uint32_t headIdx = headStart + headOff;
                const uint64_t groupBase = MakeCacheGroupBaseOffset<IS_KEY>(blockIdx, groupInBlock, headIdx);
                if (packedWritePending_ && packedGroupSlot_ == 0) {
                    WaitPendingPackedWrites();
                }
                const uint32_t groupSlot = packedGroupSlot_;
                ++packedGroupSlot_;
                if (packedGroupSlot_ >= TQ_PACKED_GROUP_BUFFER_COUNT) {
                    packedGroupSlot_ = 0;
                }

                auto packedGroup = packedGroups[groupSlot * TQ_PACKED_GROUP_STRIDE];
                const uint32_t encodedRow0 = rowBase * headCount + headOff;
                InitFullGroupFromRow0<IS_KEY>(packedGroup, encodedBatch, encodedRow0);
                for (uint32_t row = 1; row < groupRowsPerGroup; ++row) {
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch,
                        encodedRow0 + row * headCount, row, false);
                }
                copy_packed_ub_to_gm_async(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
                MarkPackedWritePending();
            }
        }

        FreeEncodedBatch<IS_KEY>(encodedBatch);
    }

    template <bool IS_KEY>
    __aicore__ inline void CopyOutResolvedTask(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        const uint8_t* preserveRows,
        uint32_t m) {
        auto encodedBatch = DeQueEncodedBatch<IS_KEY>();
        if (CopyOutPhysicalGroupRowsFast<IS_KEY>(packedGm, encodedBatch, groupBases, groupRows, preserveRows, m)) {
            FreeEncodedBatch<IS_KEY>(encodedBatch);
            return;
        }

        WaitPendingPackedWrites();
        uint8_t validRows[TQ_MAX_BATCH_M];

        for (uint32_t row = 0; row < m; ++row) {
            validRows[row] = 1;
        }

        FlushResolvedGroups<IS_KEY>(packedGm, encodedBatch, groupBases, groupRows, preserveRows, validRows, m);
        FreeEncodedBatch<IS_KEY>(encodedBatch);
    }

    template <bool IS_KEY>
    __aicore__ inline void PackCacheIndexedTask(
        const uint32_t* vecIndices,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        const uint8_t* preserveRows,
        uint32_t rows) {
        uint32_t m = 0;
        if constexpr (IS_KEY) {
            CopyInIndexedTask(
                keyGm_, vecIndices, rows,
                keyStorageOffset_, keyStrideToken_, keyStrideHead_, m);
        } else {
            CopyInIndexedTask(
                valueGm_, vecIndices, rows,
                valueStorageOffset_, valueStrideToken_, valueStrideHead_, m);
        }
        if (m == 0) {
            return;
        }

        if constexpr (IS_KEY) {
            ComputeKeyBatch(m);
            CopyOutResolvedTask<IS_KEY>(keyCacheGm_, groupBases, groupRows, preserveRows, m);
        } else {
            ComputeValueBatch(m);
            CopyOutResolvedTask<IS_KEY>(valueCacheGm_, groupBases, groupRows, preserveRows, m);
        }
    }

    template <bool IS_KEY>
    __aicore__ inline void PackCachePhysicalGroupRowsTask(
        uint32_t tokenStart,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t firstGroupRow,
        uint32_t headStart,
        uint32_t headCount,
        bool preserveExisting) {
        uint32_t m = 0;
        if constexpr (IS_KEY) {
            CopyInPhysicalGroupRowsTask(
                keyGm_, tokenStart, rowCount, headStart, headCount,
                keyStorageOffset_, keyStrideToken_, keyStrideHead_, m);
        } else {
            CopyInPhysicalGroupRowsTask(
                valueGm_, tokenStart, rowCount, headStart, headCount,
                valueStorageOffset_, valueStrideToken_, valueStrideHead_, m);
        }
        if (m == 0) {
            return;
        }

        if constexpr (IS_KEY) {
            ComputeKeyBatch(m);
            CopyOutPhysicalGroupRowsKnown<IS_KEY>(
                keyCacheGm_, rowCount, blockIdx, groupInBlock, firstGroupRow,
                headStart, headCount, preserveExisting);
        } else {
            ComputeValueBatch(m);
            CopyOutPhysicalGroupRowsKnown<IS_KEY>(
                valueCacheGm_, rowCount, blockIdx, groupInBlock, firstGroupRow,
                headStart, headCount, preserveExisting);
        }
    }

    template <bool IS_KEY>
    __aicore__ inline void PackCachePhysicalGroupRows(
        uint32_t tokenStart,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t firstGroupRow,
        bool preserveExisting) {
        uint32_t headStart = 0;
        const uint32_t batchCapacity = GetBatchCapacity();
        while (headStart < numHeads_) {
            uint32_t headCount = numHeads_ - headStart;
            const uint32_t maxHeadsPerBatch = batchCapacity / rowCount;
            if (maxHeadsPerBatch == 0) {
                headCount = 1;
            } else if (headCount > maxHeadsPerBatch) {
                headCount = maxHeadsPerBatch;
            }

            PackCachePhysicalGroupRowsTask<IS_KEY>(
                tokenStart, rowCount, blockIdx, groupInBlock,
                firstGroupRow, headStart, headCount, preserveExisting);
            headStart += headCount;
        }
    }

    template <bool IS_KEY>
    __aicore__ inline void PackCachePhysicalFullGroupRunTask(
        uint32_t tokenStart,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t firstGroupInBlock,
        uint32_t headStart,
        uint32_t headCount) {
        uint32_t m = 0;
        if constexpr (IS_KEY) {
            CopyInPhysicalGroupRowsTask(
                keyGm_, tokenStart, rowCount, headStart, headCount,
                keyStorageOffset_, keyStrideToken_, keyStrideHead_, m);
        } else {
            CopyInPhysicalGroupRowsTask(
                valueGm_, tokenStart, rowCount, headStart, headCount,
                valueStorageOffset_, valueStrideToken_, valueStrideHead_, m);
        }
        if (m == 0) {
            return;
        }

        if constexpr (IS_KEY) {
            ComputeKeyBatch(m);
            CopyOutPhysicalFullGroupRunKnown<IS_KEY>(
                keyCacheGm_, rowCount, blockIdx, firstGroupInBlock, headStart, headCount);
        } else {
            ComputeValueBatch(m);
            CopyOutPhysicalFullGroupRunKnown<IS_KEY>(
                valueCacheGm_, rowCount, blockIdx, firstGroupInBlock, headStart, headCount);
        }
    }

    template <bool IS_KEY>
    __aicore__ inline void PackCachePhysicalFullGroupRun(
        uint32_t tokenStart,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t firstGroupInBlock) {
        uint32_t consumedRows = 0;
        const uint32_t batchCapacity = GetBatchCapacity();
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();

        while (consumedRows < rowCount) {
            uint32_t rowsThisBatch = 0;
            if (numHeads_ <= batchCapacity) {
                rowsThisBatch = (batchCapacity / numHeads_) / groupRowsPerGroup * groupRowsPerGroup;
            }
            if (rowsThisBatch < groupRowsPerGroup) {
                PackCachePhysicalGroupRows<IS_KEY>(
                    tokenStart + consumedRows,
                    groupRowsPerGroup,
                    blockIdx,
                    firstGroupInBlock + consumedRows / groupRowsPerGroup,
                    0,
                    false);
                consumedRows += groupRowsPerGroup;
                continue;
            }

            uint32_t remainingRows = rowCount - consumedRows;
            if (rowsThisBatch > remainingRows) {
                rowsThisBatch = remainingRows / groupRowsPerGroup * groupRowsPerGroup;
            }
            if (rowsThisBatch == 0) {
                rowsThisBatch = groupRowsPerGroup;
            }

            PackCachePhysicalFullGroupRunTask<IS_KEY>(
                tokenStart + consumedRows,
                rowsThisBatch, blockIdx,
                firstGroupInBlock + consumedRows / groupRowsPerGroup,
                0, numHeads_);
            consumedRows += rowsThisBatch;
        }
    }

    __aicore__ inline uint32_t GetActiveWorkers() const {
        return dataCores_ * TQ_AIV_SUB_BLOCKS;
    }

    __aicore__ inline bool ResolveTokenSlotU(
        uint32_t tokenIdx,
        uint32_t& slotU) const {
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

    template <bool IS_KEY>
    __aicore__ inline void FlushVecBatch(
        const uint32_t* vecIndices,
        const uint64_t* groupBases,
        const uint32_t* groupRows,
        const uint8_t* preserveRows,
        uint32_t rows) {
        if (rows == 0) {
            return;
        }
        PackCacheIndexedTask<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
    }

    template <bool IS_KEY>
    __aicore__ inline void FlushSequenceBatch(
        uint32_t* vecIndices,
        uint64_t* groupBases,
        uint32_t* groupRows,
        uint8_t* preserveRows,
        uint32_t& rows) {
        FlushVecBatch<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
        rows = 0;
    }

    __aicore__ inline uint32_t GetBatchCapacity() const {
        return vecPerCore_ == 0 ? TQ_MAX_BATCH_M : vecPerCore_;
    }

    template <bool IS_KEY>
    __aicore__ inline void EnsureSequenceBatchCapacity(
        uint32_t requiredRows,
        uint32_t* vecIndices,
        uint64_t* groupBases,
        uint32_t* groupRows,
        uint8_t* preserveRows,
        uint32_t& rows) {
        if (rows + requiredRows <= GetBatchCapacity()) {
            return;
        }
        FlushSequenceBatch<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
    }

    template <bool IS_KEY>
    __aicore__ inline void AppendPhysicalCacheGroupRows(
        uint32_t tokenStart,
        uint32_t rowCount,
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t firstGroupRow,
        bool preserveExisting,
        uint32_t* vecIndices,
        uint64_t* groupBases,
        uint32_t* groupRows,
        uint8_t* preserveRows,
        uint32_t& rows) {
        if (rowCount == 0) {
            return;
        }

        const uint32_t batchCapacity = GetBatchCapacity();
        uint32_t headStart = 0;
        while (headStart < numHeads_) {
            uint32_t headCount = numHeads_ - headStart;
            const uint32_t maxHeadsPerBatch = batchCapacity / rowCount;
            if (maxHeadsPerBatch == 0) {
                headCount = 1;
            } else if (headCount > maxHeadsPerBatch) {
                headCount = maxHeadsPerBatch;
            }

            const uint32_t requiredRows = rowCount * headCount;
            EnsureSequenceBatchCapacity<IS_KEY>(
                requiredRows, vecIndices, groupBases, groupRows, preserveRows, rows);
            for (uint32_t row = 0; row < rowCount; ++row) {
                if (rows >= batchCapacity) {
                    FlushSequenceBatch<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
                }
                for (uint32_t headOff = 0; headOff < headCount; ++headOff) {
                    if (rows >= batchCapacity) {
                        FlushSequenceBatch<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
                    }
                    const uint32_t headIdx = headStart + headOff;
                    const uint32_t vecIdx = MakeVecIndex(tokenStart + row, headIdx);
                    if (vecIdx >= nVec_) {
                        continue;
                    }
                    vecIndices[rows] = vecIdx;
                    groupBases[rows] = MakeCacheGroupBaseOffset<IS_KEY>(blockIdx, groupInBlock, headIdx);
                    groupRows[rows] = firstGroupRow + row;
                    preserveRows[rows] = preserveExisting ? 1 : 0;
                    ++rows;
                }
            }
            headStart += headCount;
        }
    }

    __aicore__ inline bool ResolveRequestTokenRange(
        uint32_t reqIdx,
        uint32_t& tokenStart,
        uint32_t& tokenEnd) const {
        if (reqIdx >= numReqs_) {
            return false;
        }
        const int32_t start = queryStartLocGm_.GetValue(reqIdx);
        const int32_t end = queryStartLocGm_.GetValue(reqIdx + 1);
        if (start < 0 || end < start) {
            return false;
        }
        tokenStart = static_cast<uint32_t>(start);
        tokenEnd = static_cast<uint32_t>(end);
        if (tokenStart > tokenCount_ || tokenEnd > tokenCount_) {
            return false;
        }
        return tokenStart < tokenEnd;
    }

    template <bool IS_KEY>
    __aicore__ inline void ProcessSequenceBlockRange(
        uint32_t tokenStart,
        uint32_t blockIdx,
        uint32_t blockOffset,
        uint32_t rowsInBlock,
        uint32_t* vecIndices,
        uint64_t* groupBases,
        uint32_t* groupRows,
        uint8_t* preserveRows,
        uint32_t& rows) {
        uint32_t consumedRows = 0;
        uint32_t currentOffset = blockOffset;
        uint32_t remainingRows = rowsInBlock;
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();

        const uint32_t leadingGroupRow = currentOffset % groupRowsPerGroup;
        if (leadingGroupRow != 0 && remainingRows > 0) {
            uint32_t leadingRows = groupRowsPerGroup - leadingGroupRow;
            if (leadingRows > remainingRows) {
                leadingRows = remainingRows;
            }
            const uint32_t groupInBlock = currentOffset / groupRowsPerGroup;
            AppendPhysicalCacheGroupRows<IS_KEY>(
                tokenStart,
                leadingRows,
                blockIdx,
                groupInBlock,
                leadingGroupRow,
                true,
                vecIndices,
                groupBases,
                groupRows,
                preserveRows,
                rows);
            consumedRows += leadingRows;
            currentOffset += leadingRows;
            remainingRows -= leadingRows;
        }

        if (remainingRows >= groupRowsPerGroup) {
            uint32_t fullRows = remainingRows / groupRowsPerGroup * groupRowsPerGroup;
            const uint32_t groupInBlock = currentOffset / groupRowsPerGroup;
            if (rows != 0) {
                FlushSequenceBatch<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
            }
            PackCachePhysicalFullGroupRun<IS_KEY>(
                tokenStart + consumedRows,
                fullRows,
                blockIdx,
                groupInBlock);
            consumedRows += fullRows;
            currentOffset += fullRows;
            remainingRows -= fullRows;
        }

        if (remainingRows == 0) {
            return;
        }
        const uint32_t groupInBlock = currentOffset / groupRowsPerGroup;
        AppendPhysicalCacheGroupRows<IS_KEY>(
            tokenStart + consumedRows,
            remainingRows,
            blockIdx,
            groupInBlock,
            0,
            false,
            vecIndices,
            groupBases,
            groupRows,
            preserveRows,
            rows);
    }

    template <bool IS_KEY>
    __aicore__ inline void ProcessCacheBySequenceContiguousSegments(uint32_t worker) {
        const uint32_t activeWorkers = GetActiveWorkers();
        if (activeWorkers == 0 || worker >= activeWorkers) {
            return;
        }

        const uint32_t tokensPerWorker = (tokenCount_ + activeWorkers - 1) / activeWorkers;
        const uint32_t rawBegin = worker * tokensPerWorker;
        if (rawBegin >= tokenCount_) {
            return;
        }
        uint32_t rawEnd = rawBegin + tokensPerWorker;
        if (rawEnd > tokenCount_) {
            rawEnd = tokenCount_;
        }
        const bool lastWorkerRange = rawEnd == tokenCount_ || worker == activeWorkers - 1;

        uint32_t vecIndices[TQ_MAX_BATCH_M];
        uint64_t groupBases[TQ_MAX_BATCH_M];
        uint32_t groupRows[TQ_MAX_BATCH_M];
        uint8_t preserveRows[TQ_MAX_BATCH_M];
        uint32_t rows = 0;
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();

        for (uint32_t reqIdx = 0; reqIdx < numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!ResolveRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                continue;
            }

            uint32_t firstSlot = 0;
            if (!ResolveTokenSlotU(seqStart, firstSlot)) {
                continue;
            }

            uint32_t segmentStart = rawBegin > seqStart ? rawBegin : seqStart;
            uint32_t segmentEnd = rawEnd < seqEnd ? rawEnd : seqEnd;
            if (segmentStart >= segmentEnd) {
                continue;
            }

            if (segmentStart > seqStart) {
                const uint32_t slotAtBegin = firstSlot + (segmentStart - seqStart);
                uint32_t backRows = slotAtBegin % groupRowsPerGroup;
                const uint32_t availableRows = segmentStart - seqStart;
                if (backRows > availableRows) {
                    backRows = availableRows;
                }
                segmentStart -= backRows;
            }
            if (!lastWorkerRange && segmentEnd < seqEnd) {
                const uint32_t slotAtEnd = firstSlot + (segmentEnd - seqStart);
                segmentEnd -= slotAtEnd % groupRowsPerGroup;
            }
            if (segmentStart >= segmentEnd) {
                continue;
            }

            uint32_t tokenIdx = segmentStart;
            while (tokenIdx < segmentEnd) {
                uint32_t slot = firstSlot + (tokenIdx - seqStart);
                if (slot >= cacheSlots_) {
                    break;
                }
                const uint32_t blockIdx = slot / blockSize_;
                const uint32_t blockOffset = slot - blockIdx * blockSize_;
                uint32_t rowsInBlock = blockSize_ - blockOffset;
                const uint32_t remainingSegmentRows = segmentEnd - tokenIdx;
                if (rowsInBlock > remainingSegmentRows) {
                    rowsInBlock = remainingSegmentRows;
                }
                const uint32_t remainingCacheRows = cacheSlots_ - slot;
                if (rowsInBlock > remainingCacheRows) {
                    rowsInBlock = remainingCacheRows;
                }
                ProcessSequenceBlockRange<IS_KEY>(
                    tokenIdx,
                    blockIdx,
                    blockOffset,
                    rowsInBlock,
                    vecIndices,
                    groupBases,
                    groupRows,
                    preserveRows,
                    rows);
                tokenIdx += rowsInBlock;
            }
        }
        FlushSequenceBatch<IS_KEY>(vecIndices, groupBases, groupRows, preserveRows, rows);
        WaitPendingPackedWrites();
    }

private:
    AscendC::TPipe* const pipe_;
    RotateMmT* const rotateMm_;
    const uint32_t nVec_;
    const uint32_t vecPerCore_;
    const uint32_t numHeads_;
    const uint32_t tokenCount_;
    const uint32_t blockSize_;
    const uint32_t numBlocks_;
    const uint32_t cacheSlots_;
    const uint32_t dataCores_;
    const uint32_t numReqs_;
    const uint32_t keyStrideToken_;
    const uint32_t keyStrideHead_;
    const uint32_t valueStrideToken_;
    const uint32_t valueStrideHead_;
    const uint64_t keyStorageOffset_;
    const uint64_t valueStorageOffset_;
    const uint64_t keySpan_;
    const uint64_t valueSpan_;
    const bool matmulReady_;
    bool packedWritePending_;
    uint32_t packedGroupSlot_;

    AscendC::TQue<AscendC::TPosition::VECIN, TQ_QUEUE_DEPTH> xBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, TQ_QUEUE_DEPTH> aBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, TQ_QUEUE_DEPTH> yBatchQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> signMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> signValBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> errBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> quantIndexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> quantIndexU16Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packMergeBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packMaskBuf_;
    AscendC::TQue<AscendC::TPosition::VECOUT, TQ_QUEUE_DEPTH> keyEncodedQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, TQ_QUEUE_DEPTH> valEncodedQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateWorkBuf_;
    AscendC::GlobalTensor<T> keyGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> rotationTGm_;
    AscendC::GlobalTensor<int32_t> slotMappingGm_;
    AscendC::GlobalTensor<int32_t> queryStartLocGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void bit_residual_pack_k8v4(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR query_start_loc,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    auto* rotationPtr = reinterpret_cast<__gm__ TqDataT*>(rotation_t);
    auto* slotMappingPtr = reinterpret_cast<__gm__ int32_t*>(slot_mapping);
    auto* queryStartLocPtr = reinterpret_cast<__gm__ int32_t*>(query_start_loc);
    auto* keyCachePtr = reinterpret_cast<__gm__ uint8_t*>(key_cache);
    auto* valueCachePtr = reinterpret_cast<__gm__ uint8_t*>(value_cache);

    auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
    AscendC::SetSysWorkspace(wsPtr);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    TqRotateMatmulOp<TqDataT> rotateMm;
    REGIST_MATMUL_OBJ_STATIC(&pipe, GetSysWorkSpacePtr(), rotateMm, (TCubeTiling*)nullptr);
    BitResidualPackK8v4<TqDataT, TqRotateMatmulOp<TqDataT>> op(
        &pipe,
        &rotateMm,
        wsPtr,
        tilingData.nVec,
        tilingData.vecPerCore,
        tilingData.numHeads,
        tilingData.blockSize,
        tilingData.numBlocks,
        tilingData.dataCores,
        tilingData.numReqs,
        tilingData.keyStrideToken,
        tilingData.keyStrideHead,
        tilingData.valueStrideToken,
        tilingData.valueStrideHead,
        tilingData.keyStorageOffset,
        tilingData.valueStorageOffset);
    op.Init(
        key,
        value,
        rotationPtr,
        slotMappingPtr,
        queryStartLocPtr,
        keyCachePtr,
        valueCachePtr);
    op.Process();
}
