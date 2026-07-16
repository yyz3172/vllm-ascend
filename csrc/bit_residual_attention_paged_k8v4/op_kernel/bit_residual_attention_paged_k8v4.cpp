/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// BitResidual K8V4: fused sign-reversal decode + paged attention.
// Reads packed KV cache in bit_residual format and performs attention:
//   K: 8-bit code = q7 | (sign<<7) → sig_vec=±1, err=base+q7*step → decoded=err*sig_vec
//   V: 4-bit idx4 → vmin + idx4*vstep (raw, no norm folding)
// Rotation: Q @ R^T (pre-rotate) and out @ R (post-rotate) via Cube KFC.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "decode_device.h"
#include "attention_device.h"

using namespace AscendC;
using namespace bit_residual_attn;

// ---------------------------------------------------------------
// Constants
// ---------------------------------------------------------------
static constexpr uint32_t TQ_BR_UB_KV_TILE_CAP = 64;
static constexpr uint32_t TQ_BR_UB_GQA_CAP = 8;
static constexpr uint32_t TQ_BR_SEQ_BINSEARCH_MIN_BATCH = 16;
static constexpr uint32_t TQ_BR_MIX_AIV_SUB = 2;
// Packed sub-block copy bytes (32B-aligned).
// Key sub-block = 2176B, Val sub-block = 1152B. Use max for shared buffer.
static constexpr uint32_t TQ_BR_BLOCK_COPY_BYTES = ((TQ_BR_KEY_BLOCK_STRIDE + 31) / 32) * 32;  // 2176 aligned to 32

static constexpr uint32_t TQ_BR_DTYPE_BYTES = sizeof(uint16_t);

#if defined(ORIG_DTYPE_QUERY)
#if (ORIG_DTYPE_QUERY == DT_BF16)
using TqQueryT = bfloat16_t;
#else
using TqQueryT = half;
#endif
#elif defined(DTYPE_QUERY)
#if (DTYPE_QUERY == DT_BF16)
using TqQueryT = bfloat16_t;
#else
using TqQueryT = half;
#endif
#else
using TqQueryT = half;
#endif

// ---------------------------------------------------------------
// LocalTensor static offsets for UB buffers
// ---------------------------------------------------------------
// We use a layout similar to the TQ 4-bit attention kernel but adapted for
// bit_residual format. Key decode needs more scratch space (sign + q7 + base + step).

// UB layout (total fits in 192 KiB on 910B):
//   packedRawBuf: holds one 16-row sub-block at a time (2176B max, 32B-aligned → 2176)
//   We time-share: load K sub-block → extract → decode → load V sub-block → extract → decode

constexpr uint32_t TQ_BR_UB_PACKED_RAW_OFFSET = 0;
constexpr uint32_t TQ_BR_UB_PACKED_RAW_BYTES = TQ_BR_BLOCK_COPY_BYTES;  // 2176

// codeI16 (extracted key code/value idx, kept as int16 for AscendC bit ops)
constexpr uint32_t TQ_BR_UB_CODE_I16_OFFSET = TQ_BR_UB_PACKED_RAW_OFFSET + TQ_BR_UB_PACKED_RAW_BYTES;
constexpr uint32_t TQ_BR_UB_CODE_I16_BYTES = TQ_BR_UB_KV_TILE_CAP * TQ_BR_HEAD_SIZE * sizeof(int16_t);  // 16384

// codeFloat (q7/idx4 widened to fp32 for decode)
constexpr uint32_t TQ_BR_UB_CODE_FLOAT_OFFSET = TQ_BR_UB_CODE_I16_OFFSET + TQ_BR_UB_CODE_I16_BYTES;
constexpr uint32_t TQ_BR_UB_CODE_FLOAT_BYTES = TQ_BR_UB_KV_TILE_CAP * TQ_BR_HEAD_SIZE * sizeof(float);  // 32768

// qGroupFloat (Q in float, gqaCap * 128)
constexpr uint32_t TQ_BR_UB_QGROUP_FLOAT_OFFSET = TQ_BR_UB_CODE_FLOAT_OFFSET + TQ_BR_UB_CODE_FLOAT_BYTES;
constexpr uint32_t TQ_BR_UB_QGROUP_FLOAT_BYTES = TQ_BR_UB_GQA_CAP * TQ_BR_HEAD_SIZE * sizeof(float);  // 4096

// scoreBuf (QK scores, gqaCap * kvTileRows)
constexpr uint32_t TQ_BR_UB_SCORE_OFFSET = TQ_BR_UB_QGROUP_FLOAT_OFFSET + TQ_BR_UB_QGROUP_FLOAT_BYTES;
constexpr uint32_t TQ_BR_UB_SCORE_BYTES = TQ_BR_UB_GQA_CAP * TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 2048

// kBaseBuf (key base, kvTileRows floats)
constexpr uint32_t TQ_BR_UB_KBASE_OFFSET = TQ_BR_UB_SCORE_OFFSET + TQ_BR_UB_SCORE_BYTES;
constexpr uint32_t TQ_BR_UB_KBASE_BYTES = TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 256

// kStepBuf (key step, kvTileRows floats)
constexpr uint32_t TQ_BR_UB_KSTEP_OFFSET = TQ_BR_UB_KBASE_OFFSET + TQ_BR_UB_KBASE_BYTES;
constexpr uint32_t TQ_BR_UB_KSTEP_BYTES = TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 256

// valVminBuf (value vmin, kvTileRows floats)
constexpr uint32_t TQ_BR_UB_VMIN_OFFSET = TQ_BR_UB_KSTEP_OFFSET + TQ_BR_UB_KSTEP_BYTES;
constexpr uint32_t TQ_BR_UB_VMIN_BYTES = TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 256

// valVstepBuf (value vstep, kvTileRows floats)
constexpr uint32_t TQ_BR_UB_VSTEP_OFFSET = TQ_BR_UB_VMIN_OFFSET + TQ_BR_UB_VMIN_BYTES;
constexpr uint32_t TQ_BR_UB_VSTEP_BYTES = TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 256

// mStateBuf / sStateBuf (softmax running max/sum, gqaCap floats each)
constexpr uint32_t TQ_BR_UB_MSTATE_OFFSET = TQ_BR_UB_VSTEP_OFFSET + TQ_BR_UB_VSTEP_BYTES;
constexpr uint32_t TQ_BR_UB_MSTATE_BYTES = TQ_BR_UB_GQA_CAP * sizeof(float);  // 32

constexpr uint32_t TQ_BR_UB_SSTATE_OFFSET = TQ_BR_UB_MSTATE_OFFSET + TQ_BR_UB_MSTATE_BYTES;
constexpr uint32_t TQ_BR_UB_SSTATE_BYTES = TQ_BR_UB_GQA_CAP * sizeof(float);  // 32

// outAccBuf (softmax accumulator, gqaCap * 128 floats)
constexpr uint32_t TQ_BR_UB_OUTACC_OFFSET = TQ_BR_UB_SSTATE_OFFSET + TQ_BR_UB_SSTATE_BYTES;
constexpr uint32_t TQ_BR_UB_OUTACC_BYTES = TQ_BR_UB_GQA_CAP * TQ_BR_HEAD_SIZE * sizeof(float);  // 4096

// decodedKFloat (decoded K float, kvTileRows * 128 floats) — time-shared with decodedV
constexpr uint32_t TQ_BR_UB_DECODED_OFFSET = TQ_BR_UB_OUTACC_OFFSET + TQ_BR_UB_OUTACC_BYTES;
constexpr uint32_t TQ_BR_UB_DECODED_BYTES = TQ_BR_UB_KV_TILE_CAP * TQ_BR_HEAD_SIZE * sizeof(float);  // 32768

// floatScratch for scalar reads (base/step/vmin/vstep)
constexpr uint32_t TQ_BR_UB_FLOAT_SCRATCH_OFFSET = TQ_BR_UB_DECODED_OFFSET + TQ_BR_UB_DECODED_BYTES;
constexpr uint32_t TQ_BR_UB_FLOAT_SCRATCH_BYTES = TQ_BR_HEAD_SIZE * 2 * sizeof(float);

// packedMaskBuf (uint16 mask constants, filled once)
constexpr uint32_t TQ_BR_UB_MASK_OFFSET = TQ_BR_UB_FLOAT_SCRATCH_OFFSET + TQ_BR_UB_FLOAT_SCRATCH_BYTES;
constexpr uint32_t TQ_BR_UB_MASK_BYTES = TQ_BR_HEAD_SIZE * sizeof(uint16_t);

constexpr uint32_t TQ_BR_UB_VAL_MASK_OFFSET = TQ_BR_UB_MASK_OFFSET + TQ_BR_UB_MASK_BYTES;
constexpr uint32_t TQ_BR_UB_VAL_MASK_BYTES = TQ_BR_HEAD_SIZE * sizeof(uint16_t);

// rotateWorkBuf (for Cube matmul workspace)
constexpr uint32_t TQ_BR_UB_ROTATE_WORK_OFFSET = TQ_BR_UB_VAL_MASK_OFFSET + TQ_BR_UB_VAL_MASK_BYTES;
constexpr uint32_t TQ_BR_UB_ROTATE_WORK_BYTES = TQ_BR_HEAD_SIZE * TQ_BR_HEAD_SIZE * TQ_BR_DTYPE_BYTES;  // 32768

// Prefill qTile buffers (GQA<=2): Cap=8 tokens × GqaCap=2 heads
constexpr uint32_t TQ_BR_UB_QTILE_CAP = 8;
constexpr uint32_t TQ_BR_UB_QTILE_GQA_CAP = 2;
constexpr uint32_t TQ_BR_UB_QTILE_Q_OFFSET =
    TQ_BR_UB_ROTATE_WORK_OFFSET + TQ_BR_UB_ROTATE_WORK_BYTES;
constexpr uint32_t TQ_BR_UB_QTILE_Q_BYTES =
    TQ_BR_UB_QTILE_CAP * TQ_BR_UB_QTILE_GQA_CAP * TQ_BR_HEAD_SIZE * sizeof(float);  // 8192
constexpr uint32_t TQ_BR_UB_QTILE_SCORE_OFFSET =
    TQ_BR_UB_QTILE_Q_OFFSET + TQ_BR_UB_QTILE_Q_BYTES;
constexpr uint32_t TQ_BR_UB_QTILE_SCORE_BYTES =
    TQ_BR_UB_QTILE_CAP * TQ_BR_UB_QTILE_GQA_CAP * TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 4096
constexpr uint32_t TQ_BR_UB_QTILE_OUTACC_OFFSET =
    TQ_BR_UB_QTILE_SCORE_OFFSET + TQ_BR_UB_QTILE_SCORE_BYTES;
constexpr uint32_t TQ_BR_UB_QTILE_OUTACC_BYTES =
    TQ_BR_UB_QTILE_CAP * TQ_BR_UB_QTILE_GQA_CAP * TQ_BR_HEAD_SIZE * sizeof(float);  // 8192

constexpr uint32_t TQ_BR_UB_TOTAL_BYTES =
    TQ_BR_UB_QTILE_OUTACC_OFFSET + TQ_BR_UB_QTILE_OUTACC_BYTES;
static_assert(TQ_BR_UB_TOTAL_BYTES <= 192 * 1024, "UB budget exceeded");

__aicore__ inline uint32_t TqBrAlignUp16(uint32_t x)
{
    return (x + 15U) / 16U * 16U;
}

template <typename T>
__aicore__ inline void TqBrDuplicateZero(LocalTensor<T> dst, uint32_t count)
{
    if (count == 0) {
        return;
    }
    Duplicate(dst, static_cast<T>(0), count);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline LocalTensor<T> TqMakeVecCalcLocalTensor(
    uint32_t address,
    uint32_t bytes)
{
    TBuffAddr tensorAddr {};
    tensorAddr.dataLen = bytes;
    tensorAddr.bufferAddr = address;
    tensorAddr.bufferHandle = nullptr;
    tensorAddr.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    tensorAddr.absAddr =
        GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(TPosition::VECCALC)) +
        address;
#endif
    LocalTensor<T> tensor;
    tensor.SetAddr(tensorAddr);
    return tensor;
}

// ---------------------------------------------------------------
// Kernel class
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
class BitResidualAttentionPagedK8v4Kernel {
public:
    __aicore__ inline BitResidualAttentionPagedK8v4Kernel() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache,
                                GM_ADDR blockTable, GM_ADDR actualSeqLenQ,
                                GM_ADDR actualSeqLenKv, GM_ADDR rotationKey,
                                GM_ADDR rotationValue, GM_ADDR out, GM_ADDR workspace,
                                const TilingT* tilingData, TPipe* pipe,
                                TqRotateMatmulOp<QueryT>* rotateMm);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSplitBn(uint32_t coreIdx);
    __aicore__ inline void ProcessSplitBns(uint32_t coreIdx);
    __aicore__ inline void CombineFlashDecode(uint32_t coreIdx, uint32_t combineCoreNum);

    // ---- KV tile loading ----
    // Per-row decode params (base/step, vmin/vstep) are passed as scalar arrays
    // instead of UB LocalTensor round-trips, mirroring the Scalar m/s pattern
    // (eliminates the per-row GetValue/SetValue + implicit V<->S sync).
    __aicore__ inline void LoadPackedKeyTileRows(uint32_t seqIdx, uint32_t kvHead,
                                                  uint32_t absStart, uint32_t mRows,
                                                  float* kBaseArr, float* kStepArr);
    __aicore__ inline void LoadPackedValueTileRows(uint32_t seqIdx, uint32_t kvHead,
                                                    uint32_t absStart, uint32_t mRows,
                                                    float* vminArr, float* vstepArr);

    // ---- Decode ----
    __aicore__ inline void DecodeKeyTileToFloat(uint32_t mRows,
                                                const float* kBaseArr, const float* kStepArr);
    __aicore__ inline void DecodeValueTileToFloat(uint32_t mRows,
                                                  const float* vminArr, const float* vstepArr);

    // ---- Attention compute ----
    __aicore__ inline void ComputeAttention(uint32_t tokenIdx, uint32_t kvHead,
                                            uint32_t gqaStart, uint32_t gqaCount,
                                            uint32_t kvLoopStart = 0,
                                            uint32_t kvLoopEnd = 0xFFFFFFFFu,
                                            bool writePartial = false,
                                            uint32_t segIdx = 0);
    __aicore__ inline void ComputeAttentionQTile(uint32_t tokenStartIdx, uint32_t qRows,
                                                 uint32_t kvHead, uint32_t gqaStart,
                                                 uint32_t gqaCount);
    __aicore__ inline bool CubeQkQTile(LocalTensor<float> qTile, LocalTensor<float> decodedK,
                                       LocalTensor<float> scoreTile, uint32_t compactRows,
                                       uint32_t mRows);
    __aicore__ inline void WriteFinalOutputQTile(uint32_t tokenStartIdx, uint32_t qRows,
                                                 uint32_t kvHead, float* sStateScalar,
                                                 LocalTensor<float> outAcc, uint32_t gqaStart,
                                                 uint32_t gqaCount);
    __aicore__ inline void WriteEmptyPartial(uint32_t tokenIdx, uint32_t kvHead,
                                             uint32_t segIdx, uint32_t gqaStart,
                                             uint32_t gqaCount);
    __aicore__ inline void WritePartial(uint32_t tokenIdx, uint32_t kvHead, uint32_t segIdx,
                                        float* mStateScalar, float* sStateScalar,
                                        LocalTensor<float> outAcc, uint32_t gqaStart,
                                        uint32_t gqaCount);
    __aicore__ inline void RotateRowsFloatScalar(LocalTensor<float> rows,
                                              GlobalTensor<QueryT>& rotationGm,
                                              uint32_t rowCount);
    __aicore__ inline void RotateRowsInPlace(LocalTensor<float> rows,
                                             GlobalTensor<QueryT>& rotationGm,
                                             uint32_t rowCount);

    // ---- Helpers ----
    __aicore__ inline uint32_t GetCausalKvEnd(uint32_t tokenIdx, uint32_t seqIdx);
    __aicore__ inline void InitPackedMask();
    __aicore__ inline uint32_t FindSeqIdx(uint32_t tokenIdx);
    __aicore__ inline float ReadSeqLenKv(uint32_t seqIdx);

    // ---- UB buffers (LocalTensor at static offsets) ----
    __aicore__ inline LocalTensor<uint8_t> PackedRawBuf() {
        return TqMakeVecCalcLocalTensor<uint8_t>(
            TQ_BR_UB_PACKED_RAW_OFFSET, TQ_BR_UB_PACKED_RAW_BYTES);
    }
    __aicore__ inline LocalTensor<int16_t> CodeI16Buf() {
        return TqMakeVecCalcLocalTensor<int16_t>(
            TQ_BR_UB_CODE_I16_OFFSET, TQ_BR_UB_CODE_I16_BYTES);
    }
    __aicore__ inline LocalTensor<float> CodeFloatBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_CODE_FLOAT_OFFSET, TQ_BR_UB_CODE_FLOAT_BYTES);
    }
    __aicore__ inline LocalTensor<float> QGroupFloatBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_QGROUP_FLOAT_OFFSET, TQ_BR_UB_QGROUP_FLOAT_BYTES);
    }
    __aicore__ inline LocalTensor<float> ScoreBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_SCORE_OFFSET, TQ_BR_UB_SCORE_BYTES);
    }
    __aicore__ inline LocalTensor<float> KBaseBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_KBASE_OFFSET, TQ_BR_UB_KBASE_BYTES);
    }
    __aicore__ inline LocalTensor<float> KStepBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_KSTEP_OFFSET, TQ_BR_UB_KSTEP_BYTES);
    }
    __aicore__ inline LocalTensor<float> VminBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_VMIN_OFFSET, TQ_BR_UB_VMIN_BYTES);
    }
    __aicore__ inline LocalTensor<float> VstepBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_VSTEP_OFFSET, TQ_BR_UB_VSTEP_BYTES);
    }
    __aicore__ inline LocalTensor<float> MStateBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_MSTATE_OFFSET, TQ_BR_UB_MSTATE_BYTES);
    }
    __aicore__ inline LocalTensor<float> SStateBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_SSTATE_OFFSET, TQ_BR_UB_SSTATE_BYTES);
    }
    __aicore__ inline LocalTensor<float> OutAccBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_OUTACC_OFFSET, TQ_BR_UB_OUTACC_BYTES);
    }
    __aicore__ inline LocalTensor<float> DecodedBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_DECODED_OFFSET, TQ_BR_UB_DECODED_BYTES);
    }
    __aicore__ inline LocalTensor<float> FloatScratchBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_FLOAT_SCRATCH_OFFSET, TQ_BR_UB_FLOAT_SCRATCH_BYTES);
    }
    __aicore__ inline LocalTensor<uint16_t> MaskBuf() {
        return TqMakeVecCalcLocalTensor<uint16_t>(
            TQ_BR_UB_MASK_OFFSET, TQ_BR_UB_MASK_BYTES);
    }
    __aicore__ inline LocalTensor<uint16_t> ValMaskBuf() {
        return TqMakeVecCalcLocalTensor<uint16_t>(
            TQ_BR_UB_VAL_MASK_OFFSET, TQ_BR_UB_VAL_MASK_BYTES);
    }
    __aicore__ inline LocalTensor<uint8_t> RotateWorkBuf() {
        return TqMakeVecCalcLocalTensor<uint8_t>(
            TQ_BR_UB_ROTATE_WORK_OFFSET, TQ_BR_UB_ROTATE_WORK_BYTES);
    }
    __aicore__ inline LocalTensor<float> QTileGroupBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_QTILE_Q_OFFSET, TQ_BR_UB_QTILE_Q_BYTES);
    }
    __aicore__ inline LocalTensor<float> QTileScoreBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_QTILE_SCORE_OFFSET, TQ_BR_UB_QTILE_SCORE_BYTES);
    }
    __aicore__ inline LocalTensor<float> QTileOutAccBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_QTILE_OUTACC_OFFSET, TQ_BR_UB_QTILE_OUTACC_BYTES);
    }

    // ---- Tiling data members ----
    TilingT tiling_;
    uint32_t numTokens_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t gqaGroupSize_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t totalCacheBlocks_ = 0;
    uint32_t maxKvLen_ = 0;
    uint32_t kvTileRows_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t splitMode_ = 0;
    uint32_t qTileMode_ = 0;
    uint32_t qkPvMode_ = 0;
    uint32_t kvSplitPart_ = 1;
    uint32_t kvSegmentLen_ = 0;
    uint32_t qkWorkspaceStride_ = 0;
    float scaleValue_ = 0.f;
    __gm__ uint8_t* partialWorkspace_ = nullptr;

    // ---- GM tensors ----
    GlobalTensor<QueryT> queryGm_;
    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int64_t> actualSeqLenQGm_;
    GlobalTensor<int64_t> actualSeqLenKvGm_;
    GlobalTensor<QueryT> rotationKeyGm_;
    GlobalTensor<QueryT> rotationValueGm_;
    GlobalTensor<QueryT> outGm_;
    GlobalTensor<float> partialOutGm_;
    GlobalTensor<float> partialLseGm_;
    GlobalTensor<QueryT> qkKGm_;
    bool qkGmReady_ = false;

    // ---- Cube matmul ----
    TPipe* pipe_ = nullptr;
    TqRotateMatmulOp<QueryT>* rotateMm_ = nullptr;
    bool matmulReady_ = false;
};

// ---------------------------------------------------------------
// Init()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::Init(
    GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache,
    GM_ADDR blockTable, GM_ADDR actualSeqLenQ,
    GM_ADDR actualSeqLenKv, GM_ADDR rotationKey,
    GM_ADDR rotationValue, GM_ADDR out, GM_ADDR workspace,
    const TilingT* tilingData, TPipe* pipe,
    TqRotateMatmulOp<QueryT>* rotateMm)
{
    pipe_ = pipe;
    rotateMm_ = rotateMm;
    tiling_ = *tilingData;

    numTokens_ = tiling_.numTokens;
    batchSize_ = tiling_.batchSize;
    numHeads_ = tiling_.numHeads;
    numKvHeads_ = tiling_.numKvHeads;
    gqaGroupSize_ = tiling_.gqaGroupSize;
    headSize_ = tiling_.headSize;
    blockSize_ = tiling_.blockSize;
    maxBlocksPerSeq_ = tiling_.maxBlocksPerSeq;
    totalCacheBlocks_ = tiling_.totalCacheBlocks;
    maxKvLen_ = tiling_.maxKvLen;
    kvTileRows_ = tiling_.kvTileRows;
    usedCoreNum_ = tiling_.usedCoreNum;
    splitMode_ = tiling_.splitMode;
    qTileMode_ = tiling_.qTileMode;
    qkPvMode_ = tiling_.qkPvMode;
    kvSplitPart_ = tiling_.kvSplitPart;
    kvSegmentLen_ = tiling_.kvSegmentLen;
    qkWorkspaceStride_ = tiling_.qkWorkspaceStride;
    scaleValue_ = tiling_.scaleValue;

    // GM buffers
    queryGm_.SetGlobalBuffer((__gm__ QueryT*)query, numTokens_ * numHeads_ * headSize_);
    keyCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)keyCache,
        totalCacheBlocks_ * numKvHeads_ * (blockSize_ / TQ_BR_BLOCK_ROWS) * TQ_BR_KEY_BLOCK_STRIDE);
    valueCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)valueCache,
        totalCacheBlocks_ * numKvHeads_ * (blockSize_ / TQ_BR_BLOCK_ROWS) * TQ_BR_VAL_BLOCK_STRIDE);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable, batchSize_ * maxBlocksPerSeq_);
    actualSeqLenQGm_.SetGlobalBuffer((__gm__ int64_t*)actualSeqLenQ, batchSize_);
    actualSeqLenKvGm_.SetGlobalBuffer((__gm__ int64_t*)actualSeqLenKv, batchSize_);
    rotationKeyGm_.SetGlobalBuffer((__gm__ QueryT*)rotationKey, headSize_ * headSize_);
    rotationValueGm_.SetGlobalBuffer((__gm__ QueryT*)rotationValue, headSize_ * headSize_);
    outGm_.SetGlobalBuffer((__gm__ QueryT*)out, numTokens_ * numHeads_ * headSize_);

    if (splitMode_ == 1 && tiling_.partialWorkspaceOffset > 0 && workspace != nullptr) {
        partialWorkspace_ =
            reinterpret_cast<__gm__ uint8_t*>(workspace) + tiling_.partialWorkspaceOffset;
        const size_t accumBytes =
            static_cast<size_t>(tiling_.accumOutSize) * sizeof(float);
        partialOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partialWorkspace_),
                                      tiling_.accumOutSize);
        partialLseGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(partialWorkspace_ + accumBytes),
            tiling_.logSumExpSize);
    }

    qkGmReady_ = false;
    if (qkPvMode_ == 1 && tiling_.qkWorkspaceOffset > 0 && qkWorkspaceStride_ > 0) {
        // Prefer GetSysWorkSpacePtr (same base SetSysWorkspace uses).
        __gm__ uint8_t* wsBase =
            reinterpret_cast<__gm__ uint8_t*>(GetSysWorkSpacePtr());
        if (wsBase == nullptr && workspace != nullptr) {
            wsBase = reinterpret_cast<__gm__ uint8_t*>(workspace);
        }
        if (wsBase != nullptr) {
            qkKGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ QueryT*>(wsBase + tiling_.qkWorkspaceOffset),
                static_cast<uint64_t>(20) * qkWorkspaceStride_);
            qkGmReady_ = true;
        }
    }

    matmulReady_ = (GetSysWorkSpacePtr() != nullptr);
    InitPackedMask();
}

// ---------------------------------------------------------------
// InitPackedMask() - fill MaskBuf with key/value extraction masks
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::InitPackedMask() {
    auto keyMask = MaskBuf();
    Duplicate(keyMask, static_cast<uint16_t>(0x00FF), TQ_BR_HEAD_SIZE);
    auto valMask = ValMaskBuf();
    Duplicate(valMask, static_cast<uint16_t>(0x000F), TQ_BR_HEAD_SIZE);
    PipeBarrier<PIPE_V>();
}

// ---------------------------------------------------------------
// FindSeqIdx() - map token index to batch sequence
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline uint32_t BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::FindSeqIdx(
    uint32_t tokenIdx)
{
    TqBrSync<HardEvent::S_MTE2>();
    if (batchSize_ <= TQ_BR_SEQ_BINSEARCH_MIN_BATCH) {
        for (uint32_t i = 0; i < batchSize_; ++i) {
            const int64_t end = actualSeqLenQGm_.GetValue(i);
            if (static_cast<int64_t>(tokenIdx) < end) {
                TqBrSync<HardEvent::MTE2_S>();
                return i;
            }
        }
        TqBrSync<HardEvent::MTE2_S>();
        return batchSize_ > 0 ? batchSize_ - 1 : 0;
    }

    uint32_t lo = 0;
    uint32_t hi = batchSize_;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        const int64_t end = actualSeqLenQGm_.GetValue(mid);
        if (static_cast<int64_t>(tokenIdx) < end) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    TqBrSync<HardEvent::MTE2_S>();
    return lo < batchSize_ ? lo : (batchSize_ > 0 ? batchSize_ - 1 : 0);
}

// ---------------------------------------------------------------
// ReadSeqLenKv()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline float BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::ReadSeqLenKv(
    uint32_t seqIdx) {
    // Read actual_seq_len_kv[seqIdx] from GM scalar.
    // The tensor is int64, but we only need it as a uint32 for KV range.
    TqBrSync<HardEvent::S_MTE2>();
    const int64_t val = actualSeqLenKvGm_.GetValue(seqIdx);
    TqBrSync<HardEvent::MTE2_S>();
    return static_cast<float>(val);
}

// ---------------------------------------------------------------
// GetCausalKvEnd()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline uint32_t BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::GetCausalKvEnd(
    uint32_t tokenIdx, uint32_t seqIdx) {
    TqBrSync<HardEvent::S_MTE2>();
    const int64_t seqLenQ = actualSeqLenQGm_.GetValue(seqIdx);
    const int64_t seqLenKv = actualSeqLenKvGm_.GetValue(seqIdx);
    TqBrSync<HardEvent::MTE2_S>();
    const int64_t qStart = (seqIdx == 0) ? 0 : actualSeqLenQGm_.GetValue(seqIdx - 1);
    const int64_t numQInSeq = seqLenQ - qStart;
    const int64_t qPos = tokenIdx - qStart;
    const int64_t causalEnd = seqLenKv - numQInSeq + qPos + 1;
    int64_t clampedEnd = causalEnd;
    if (clampedEnd < 0) {
        clampedEnd = 0;
    }
    if (clampedEnd > seqLenKv) {
        clampedEnd = seqLenKv;
    }
    return static_cast<uint32_t>(clampedEnd);
}

// ---------------------------------------------------------------
// LoadPackedKeyTileRows()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::LoadPackedKeyTileRows(
    uint32_t seqIdx, uint32_t kvHead, uint32_t absStart, uint32_t mRows,
    float* kBaseArr, float* kStepArr)
{
    // Independent-row key cache layout:
    //   codes: blockSize * 128 uint8
    //   base / step: blockSize * sizeof(QueryT) each
    // Per row: load code + base + step, then widen uint8→int16 for decode.

    const uint32_t seqBlockBase = seqIdx * maxBlocksPerSeq_;
    auto packedRaw = PackedRawBuf();
    auto codeI16 = CodeI16Buf();
    auto floatScratch = FloatScratchBuf();

    const uint32_t D = TQ_BR_HEAD_SIZE;

    // Cache block_table lookups: consecutive rows in the same page share blockId.
    int32_t cachedBlockId = -1;
    uint32_t cachedBlockOffset = 0xFFFFFFFFu;

    for (uint32_t row = 0; row < mRows; ++row) {
        const uint32_t absPos = absStart + row;
        const uint32_t blockOffset = absPos / blockSize_;
        const uint32_t posInBlock = absPos % blockSize_;

        if (blockOffset != cachedBlockOffset) {
            TqBrSync<HardEvent::S_MTE2>();
            cachedBlockId = blockTableGm_.GetValue(seqBlockBase + blockOffset);
            TqBrSync<HardEvent::MTE2_S>();
            cachedBlockOffset = blockOffset;
        }
        const int32_t blockId = cachedBlockId;

        const uint32_t headStride = blockSize_ * (TQ_BR_HEAD_SIZE + 2 * sizeof(QueryT));
        const uint64_t headBase =
            (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * headStride;
        DataCopyExtParams copyParams{1, TQ_BR_HEAD_SIZE, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(packedRaw, keyCacheGm_[headBase + posInBlock * TQ_BR_HEAD_SIZE], copyParams, padParams);
        DataCopyExtParams metaParams{1, sizeof(QueryT), 0, 0, 0};
        DataCopyPad(packedRaw[TQ_BR_HEAD_SIZE],
            keyCacheGm_[headBase + blockSize_ * TQ_BR_HEAD_SIZE + posInBlock * sizeof(QueryT)],
            metaParams, padParams);
        DataCopyPad(packedRaw[TQ_BR_HEAD_SIZE + 32],
            keyCacheGm_[headBase + blockSize_ * (TQ_BR_HEAD_SIZE + sizeof(QueryT)) +
                        posInBlock * sizeof(QueryT)], metaParams, padParams);
        TqBrSync<HardEvent::MTE2_V>();
        TqBrSync<HardEvent::MTE2_S>();

        // Vectorized uint8→int16 zero-extend (mirrors V's Cast path).
        // Intermediate half keeps 0..255 exact; then cast to int16.
        auto extractI16 = codeI16[row * D];
        auto tmpHalf = floatScratch.template ReinterpretCast<half>();
        AscendC::Cast(tmpHalf, packedRaw, AscendC::RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(extractI16, tmpHalf, AscendC::RoundMode::CAST_RINT, D);
        PipeBarrier<PIPE_V>();

        const float base = TqBrRead16FromU8<QueryT>(packedRaw, floatScratch, TQ_BR_HEAD_SIZE);
        kBaseArr[row] = base;

        const float step = TqBrRead16FromU8<QueryT>(packedRaw, floatScratch, TQ_BR_HEAD_SIZE + 32);
        kStepArr[row] = step;
    }
    TqBrSync<HardEvent::V_S>();
}

// ---------------------------------------------------------------
// LoadPackedValueTileRows()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::LoadPackedValueTileRows(
    uint32_t seqIdx, uint32_t kvHead, uint32_t absStart, uint32_t mRows,
    float* vminArr, float* vstepArr)
{
    // Independent-row value cache layout:
    //   codes: blockSize * 64 uint8 (packed int4)
    //   vmin / vstep: blockSize * sizeof(QueryT) each

    const uint32_t seqBlockBase = seqIdx * maxBlocksPerSeq_;
    auto packedRaw = PackedRawBuf();
    auto codeI16 = CodeI16Buf();  // reuse for value idx4 extraction
    auto codeFloat = CodeFloatBuf();
    auto floatScratch = FloatScratchBuf();

    const uint32_t D = TQ_BR_HEAD_SIZE;

    int32_t cachedBlockId = -1;
    uint32_t cachedBlockOffset = 0xFFFFFFFFu;

    for (uint32_t row = 0; row < mRows; ++row) {
        const uint32_t absPos = absStart + row;
        const uint32_t blockOffset = absPos / blockSize_;
        const uint32_t posInBlock = absPos % blockSize_;

        if (blockOffset != cachedBlockOffset) {
            TqBrSync<HardEvent::S_MTE2>();
            cachedBlockId = blockTableGm_.GetValue(seqBlockBase + blockOffset);
            TqBrSync<HardEvent::MTE2_S>();
            cachedBlockOffset = blockOffset;
        }
        const int32_t blockId = cachedBlockId;

        const uint32_t rowBytes = TQ_BR_HEAD_SIZE / 2;
        const uint32_t headStride = blockSize_ * (rowBytes + 2 * sizeof(QueryT));
        const uint64_t headBase =
            (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * headStride;
        DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(packedRaw, valueCacheGm_[headBase + posInBlock * rowBytes], copyParams, padParams);
        DataCopyExtParams metaParams{1, sizeof(QueryT), 0, 0, 0};
        DataCopyPad(packedRaw[rowBytes],
            valueCacheGm_[headBase + blockSize_ * rowBytes + posInBlock * sizeof(QueryT)],
            metaParams, padParams);
        DataCopyPad(packedRaw[rowBytes + 32],
            valueCacheGm_[headBase + blockSize_ * (rowBytes + sizeof(QueryT)) +
                          posInBlock * sizeof(QueryT)], metaParams, padParams);
        TqBrSync<HardEvent::MTE2_V>();
        TqBrSync<HardEvent::MTE2_S>();

        auto extractI16 = codeI16[row * D].template ReinterpretCast<half>();
        AscendC::Cast(extractI16, packedRaw.template ReinterpretCast<int4b_t>(),
            AscendC::RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();
        // Keep signed idx4 in [-8, 7]; fold the +8 unsigned bias into vmin below.
        Cast(codeFloat[row * D], extractI16, RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();

        const float vmin = TqBrRead16FromU8<QueryT>(packedRaw, floatScratch, rowBytes);
        const float vstep = TqBrRead16FromU8<QueryT>(packedRaw, floatScratch, rowBytes + 32);
        // y = vmin + (idx4_s + 8) * vstep = (vmin + 8*vstep) + idx4_s * vstep
        vminArr[row] = vmin + 8.f * vstep;
        vstepArr[row] = vstep;
    }

    // Restore key mask for next K tile load.
    auto mask = MaskBuf();
    Duplicate(mask, static_cast<uint16_t>(0x00FF), TQ_BR_HEAD_SIZE);
    PipeBarrier<PIPE_V>();

    TqBrSync<HardEvent::V_S>();
}

// ---------------------------------------------------------------
// DecodeKeyTileToFloat()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::DecodeKeyTileToFloat(
    uint32_t mRows, const float* kBaseArr, const float* kStepArr)
{
    // Input: CodeI16Buf()[row*128] has the 8-bit code for each dimension.
    // kBaseArr[k], kStepArr[k] are per-row scalar floats (no UB GetValue).
    //
    // Decode: code = q7 | (sign << 7)
    //   sign_bit = code >> 7       → 0=positive, 1=negative
    //   q7       = code & 0x7f     → [0, 127]
    //   sig_vec  = 1 - 2*sign_bit  → {+1.0, -1.0}
    //   err      = base + q7 * step (positive residual)
    //   decoded  = err * sig_vec

    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t n = mRows * D;
    auto codeI16 = CodeI16Buf();
    auto codeFloat = CodeFloatBuf();
    auto decoded = DecodedBuf();
    // RotateWork is idle during DecodeKey (Cube QK staging uses it later).
    auto signBits = RotateWorkBuf().template ReinterpretCast<float>();

    auto signStorage = decoded.template ReinterpretCast<int16_t>();
    auto signStorageU16 = signStorage.template ReinterpretCast<uint16_t>();
    auto codeU16 = codeI16.template ReinterpretCast<uint16_t>();

    // Full-width And: isolate sign (0x00/0x80) then mask q7.
    Duplicate(signStorageU16, static_cast<uint16_t>(0x80), n);
    PipeBarrier<PIPE_V>();
    And(signStorageU16, codeU16, signStorageU16, n);
    PipeBarrier<PIPE_V>();

    auto q7MaskU16 = codeFloat.template ReinterpretCast<uint16_t>();
    Duplicate(q7MaskU16, static_cast<uint16_t>(0x7f), n);
    PipeBarrier<PIPE_V>();
    And(codeU16, codeU16, q7MaskU16, n);
    PipeBarrier<PIPE_V>();

    // sign_bit → sig_vec = ±1.0 in RotateWork (keeps codeFloat free for q7).
    ShiftRight(signStorage, signStorage, static_cast<int16_t>(7), n);
    PipeBarrier<PIPE_V>();
    Cast(signBits, signStorage, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Muls(signBits, signBits, -2.0f, n);
    PipeBarrier<PIPE_V>();
    Adds(signBits, signBits, 1.0f, n);
    PipeBarrier<PIPE_V>();

    // q7 → float in codeFloat; err = base + q7*step in decoded via Duplicate+Axpy.
    Cast(codeFloat, codeI16, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < mRows; ++row) {
        const float base = kBaseArr[row];
        const float step = kStepArr[row];
        auto rowFloat = decoded[row * D];
        Duplicate(rowFloat, base, D);
        PipeBarrier<PIPE_V>();
        Axpy(rowFloat, codeFloat[row * D], step, D);
        PipeBarrier<PIPE_V>();
    }

    Mul(decoded, decoded, signBits, n);
    PipeBarrier<PIPE_V>();
}

// ---------------------------------------------------------------
// DecodeValueTileToFloat()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::DecodeValueTileToFloat(
    uint32_t mRows, const float* vminArr, const float* vstepArr)
{
    // Input: CodeFloatBuf()[row*128] has signed idx4 in [-8, 7] as float.
    // vminArr[k] already includes the +8*vstep bias from LoadPackedValueTileRows.
    // Decode: y = vmin' + idx4_s * vstep

    const uint32_t D = TQ_BR_HEAD_SIZE;
    auto decoded = DecodedBuf();
    auto codeFloat = CodeFloatBuf();

    for (uint32_t row = 0; row < mRows; ++row) {
        const float vmin = vminArr[row];
        const float vstep = vstepArr[row];
        auto rowFloat = decoded[row * D];
        Duplicate(rowFloat, vmin, D);
        PipeBarrier<PIPE_V>();
        Axpy(rowFloat, codeFloat[row * D], vstep, D);
        PipeBarrier<PIPE_V>();
    }
}

// ---------------------------------------------------------------
// RotateRowsFloatScalar()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::RotateRowsFloatScalar(
    LocalTensor<float> rows, GlobalTensor<QueryT>& rotationGm, uint32_t rowCount)
{
    auto rotationLocal = RotateWorkBuf().template ReinterpretCast<QueryT>();
    auto rotationRow = FloatScratchBuf();
    auto outRow = FloatScratchBuf()[TQ_BR_HEAD_SIZE];
    DataCopy(rotationLocal, rotationGm, TQ_BR_HEAD_SIZE * TQ_BR_HEAD_SIZE);
    TqBrSync<HardEvent::MTE2_V>();

    for (uint32_t row = 0; row < rowCount; ++row) {
        Duplicate(outRow, 0.f, TQ_BR_HEAD_SIZE);
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < TQ_BR_HEAD_SIZE; ++k) {
            Cast(rotationRow, rotationLocal[k * TQ_BR_HEAD_SIZE],
                 RoundMode::CAST_NONE, TQ_BR_HEAD_SIZE);
            PipeBarrier<PIPE_V>();
            TqBrSync<HardEvent::V_S>();
            const float alpha = rows.GetValue(row * TQ_BR_HEAD_SIZE + k);
            TqBrSync<HardEvent::S_V>();
            Axpy(outRow, rotationRow, alpha, TQ_BR_HEAD_SIZE);
            PipeBarrier<PIPE_V>();
        }
        DataCopy(rows[row * TQ_BR_HEAD_SIZE], outRow, TQ_BR_HEAD_SIZE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::RotateRowsInPlace(
    LocalTensor<float> rows, GlobalTensor<QueryT>& rotationGm, uint32_t rowCount)
{
    if (rowCount == 0) {
        return;
    }
    if (!matmulReady_ || rotateMm_ == nullptr) {
        RotateRowsFloatScalar(rows, rotationGm, rowCount);
        return;
    }

    const uint32_t n = rowCount * TQ_BR_HEAD_SIZE;
    const uint32_t mPad = TqBrAlignUp16(rowCount);
    auto rotateHalf = DecodedBuf().template ReinterpretCast<QueryT>();
    auto inputHalf = rotateHalf;
    auto outputHalf = rotateHalf[mPad * TQ_BR_HEAD_SIZE];

    Cast(inputHalf, rows, RoundMode::CAST_RINT, n);
    PipeBarrier<PIPE_V>();
    if (mPad > rowCount) {
        TqBrDuplicateZero(inputHalf[n], (mPad - rowCount) * TQ_BR_HEAD_SIZE);
    }

    rotateMm_->SetOrgShape(mPad, TQ_BR_HEAD_SIZE, TQ_BR_HEAD_SIZE);
    rotateMm_->SetSingleShape(rowCount, TQ_BR_HEAD_SIZE, TQ_BR_HEAD_SIZE);
    rotateMm_->SetTensorA(inputHalf, false);
    rotateMm_->SetTensorB(rotationGm, false);
    rotateMm_->SetLocalWorkspace(RotateWorkBuf());
    rotateMm_->IterateAll(outputHalf);
    rotateMm_->End();

    Cast(rows, outputHalf, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
}

// ---------------------------------------------------------------
// ComputeAttention() - single-token decode
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::ComputeAttention(
    uint32_t tokenIdx, uint32_t kvHead, uint32_t gqaStart, uint32_t gqaCount,
    uint32_t kvLoopStart, uint32_t kvLoopEnd, bool writePartial, uint32_t segIdx)
{
    const uint32_t D = TQ_BR_HEAD_SIZE;

    // Determine KV range.
    const uint32_t seqIdx = FindSeqIdx(tokenIdx);

    const uint32_t causalKvEnd = GetCausalKvEnd(tokenIdx, seqIdx);
    uint32_t kvStart = kvLoopStart;
    uint32_t kvEnd = causalKvEnd;
    if (kvLoopEnd < causalKvEnd) {
        kvEnd = kvLoopEnd;
    }
    if (kvStart >= kvEnd) {
        if (writePartial) {
            WriteEmptyPartial(tokenIdx, kvHead, segIdx, gqaStart, gqaCount);
        } else if (causalKvEnd == 0) {
            auto outTmp = DecodedBuf().template ReinterpretCast<QueryT>();
            Duplicate(outTmp, static_cast<QueryT>(0), gqaCount * D);
            PipeBarrier<PIPE_V>();
            TqBrSync<HardEvent::V_MTE3>();
            for (uint32_t g = 0; g < gqaCount; ++g) {
                const uint32_t headIdx = kvHead * gqaGroupSize_ + gqaStart + g;
                auto outRow = outGm_[tokenIdx * numHeads_ * D + headIdx * D];
                DataCopy(outRow, outTmp[g * D], D);
            }
            TqBrSync<HardEvent::MTE3_V>();
        }
        return;
    }

    // Load and rotate Q group for gqaStart..gqaStart+gqaCount heads.
    auto qGroupFloat = QGroupFloatBuf();
    auto qGroupHalfBuf = DecodedBuf().template ReinterpretCast<QueryT>();  // time-share before K/V decode
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const uint32_t headIdx = kvHead * gqaGroupSize_ + gqaStart + g;
        auto qRow = queryGm_[tokenIdx * numHeads_ * D + headIdx * D];
        DataCopy(qGroupHalfBuf[g * D], qRow, D);
    }
    TqBrSync<HardEvent::MTE2_V>();
    Cast(qGroupFloat, qGroupHalfBuf, RoundMode::CAST_NONE, gqaCount * D);
    PipeBarrier<PIPE_V>();
    RotateRowsInPlace(qGroupFloat, rotationKeyGm_, gqaCount);
    PipeBarrier<PIPE_V>();
    Muls(qGroupFloat, qGroupFloat, scaleValue_, gqaCount * D);
    PipeBarrier<PIPE_V>();

    // m/s states as Scalar arrays to avoid LocalTensor GetValue/SetValue
    // V<->S sync in the KV tile loop.
    float mStateScalar[TQ_BR_UB_GQA_CAP];
    float sStateScalar[TQ_BR_UB_GQA_CAP];
    for (uint32_t g = 0; g < gqaCount; ++g) {
        mStateScalar[g] = -3.402823466e+38f;
        sStateScalar[g] = 0.f;
    }
    auto outAcc = OutAccBuf();
    Duplicate(outAcc, 0.f, gqaCount * D);
    PipeBarrier<PIPE_V>();

    // Per-tile decode params kept as scalar arrays (no UB round-trip).
    float kBaseArr[TQ_BR_UB_KV_TILE_CAP];
    float kStepArr[TQ_BR_UB_KV_TILE_CAP];
    float vminArr[TQ_BR_UB_KV_TILE_CAP];
    float vstepArr[TQ_BR_UB_KV_TILE_CAP];

    // Main KV tile loop.
    for (uint32_t pos = kvStart; pos < kvEnd; pos += kvTileRows_) {
        const uint32_t remainRows = kvEnd - pos;
        const uint32_t mRows = (kvTileRows_ < remainRows) ? kvTileRows_ : remainRows;

        // Load and decode K tile.
        LoadPackedKeyTileRows(seqIdx, kvHead, pos, mRows, kBaseArr, kStepArr);
        DecodeKeyTileToFloat(mRows, kBaseArr, kStepArr);

        auto decodedK = DecodedBuf();  // decoded K in float

        // QK scores: score[g][m] = sum_d Q[g][d] * K[m][d] * scale
        auto scoreBuf = ScoreBuf();
        auto reduceTmp = FloatScratchBuf();
        auto mulTmp = reduceTmp[TQ_BR_HEAD_SIZE];
        auto reduceScalar = KStepBuf();  // reuse KStep buffer as ReduceSum scalar temp
        turboquant_attn::VectorQkFloatPreScaled(
            qGroupFloat, decodedK, scoreBuf, reduceTmp, mulTmp, reduceScalar,
            gqaCount, mRows);

        // Load and decode V tile.
        LoadPackedValueTileRows(seqIdx, kvHead, pos, mRows, vminArr, vstepArr);
        DecodeValueTileToFloat(mRows, vminArr, vstepArr);

        auto decodedV = DecodedBuf();  // decoded V in float

        auto expBuf = FloatScratchBuf();
        auto softmaxReduceTmp = expBuf[TQ_BR_HEAD_SIZE];
        auto weightedValue = mulTmp;
        turboquant_attn::OnlineSoftmaxUpdateTileFloatPreScaledScalar(
            scoreBuf, decodedV, mStateScalar, sStateScalar, outAcc, expBuf,
            softmaxReduceTmp, weightedValue,
            gqaCount, mRows);
    }

    // Normalize: outAcc /= sState
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const float s = sStateScalar[g];
        auto outAccRow = outAcc[g * D];
        Muls(outAccRow, outAccRow, 1.0f / s, D);
        PipeBarrier<PIPE_V>();
    }

    if (writePartial) {
        WritePartial(tokenIdx, kvHead, segIdx, mStateScalar, sStateScalar, outAcc,
                     gqaStart, gqaCount);
        return;
    }

    // Post-rotate: out @ rotationValue (R).
    RotateRowsInPlace(outAcc, rotationValueGm_, gqaCount);
    PipeBarrier<PIPE_V>();

    // Write output to GM.
    auto outLocal = DecodedBuf().template ReinterpretCast<QueryT>();
    Cast(outLocal, outAcc, RoundMode::CAST_RINT, gqaCount * D);
    PipeBarrier<PIPE_V>();
    TqBrSync<HardEvent::V_MTE3>();
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const uint32_t headIdx = kvHead * gqaGroupSize_ + gqaStart + g;
        auto outRow = outGm_[tokenIdx * numHeads_ * D + headIdx * D];
        DataCopy(outRow, outLocal[g * D], D);
    }
    TqBrSync<HardEvent::MTE3_V>();
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::WriteEmptyPartial(
    uint32_t tokenIdx, uint32_t kvHead, uint32_t segIdx, uint32_t gqaStart, uint32_t gqaCount)
{
    if (partialWorkspace_ == nullptr) {
        return;
    }
    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t qBaseHead = kvHead * gqaGroupSize_ + gqaStart;
    auto zero = FloatScratchBuf();
    Duplicate(zero, 0.f, D);
    PipeBarrier<PIPE_V>();
    TqBrSync<HardEvent::V_MTE3>();
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const uint32_t headIdx = qBaseHead + g;
        const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
        partialLseGm_.SetValue(partIdx * 2, -3.402823466e+38f);
        partialLseGm_.SetValue(partIdx * 2 + 1, 0.f);
        DataCopy(partialOutGm_[partIdx * D], zero, D);
    }
    TqBrSync<HardEvent::MTE3_V>();
    TqBrSync<HardEvent::MTE3_MTE2>();
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::WritePartial(
    uint32_t tokenIdx, uint32_t kvHead, uint32_t segIdx, float* mStateScalar,
    float* sStateScalar, LocalTensor<float> outAcc, uint32_t gqaStart, uint32_t gqaCount)
{
    if (partialWorkspace_ == nullptr) {
        return;
    }
    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t qBaseHead = kvHead * gqaGroupSize_ + gqaStart;
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const uint32_t headIdx = qBaseHead + g;
        const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
        partialLseGm_.SetValue(partIdx * 2, mStateScalar[g]);
        partialLseGm_.SetValue(partIdx * 2 + 1, sStateScalar[g]);
    }
    TqBrSync<HardEvent::V_MTE3>();
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const uint32_t headIdx = qBaseHead + g;
        const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
        DataCopy(partialOutGm_[partIdx * D], outAcc[g * D], D);
    }
    TqBrSync<HardEvent::MTE3_V>();
    TqBrSync<HardEvent::MTE3_MTE2>();
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::WriteFinalOutputQTile(
    uint32_t tokenStartIdx, uint32_t qRows, uint32_t kvHead, float* sStateScalar,
    LocalTensor<float> outAcc, uint32_t gqaStart, uint32_t gqaCount)
{
    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t qBaseHead = kvHead * gqaGroupSize_ + gqaStart;
    const uint32_t compactRows = qRows * gqaCount;
    for (uint32_t q = 0; q < qRows; ++q) {
        const uint32_t stateBase = q * gqaCount;
        const uint32_t outBaseQ = q * gqaCount * D;
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const float sum = sStateScalar[stateBase + g];
            const float invS = (sum > 0.f) ? (1.f / sum) : 0.f;
            AscendC::Muls(outAcc[outBaseQ + g * D], outAcc[outBaseQ + g * D], invS, D);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();

    RotateRowsInPlace(outAcc, rotationValueGm_, compactRows);
    PipeBarrier<PIPE_V>();

    auto outLocal = DecodedBuf().template ReinterpretCast<QueryT>();
    for (uint32_t q = 0; q < qRows; ++q) {
        AscendC::Cast(outLocal, outAcc[q * gqaCount * D], RoundMode::CAST_RINT,
                      gqaCount * D);
        PipeBarrier<PIPE_V>();
        TqBrSync<HardEvent::V_MTE3>();
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint32_t headIdx = qBaseHead + g;
            DataCopy(outGm_[(tokenStartIdx + q) * numHeads_ * D + headIdx * D],
                     outLocal[g * D], D);
        }
        TqBrSync<HardEvent::MTE3_V>();
        TqBrSync<HardEvent::MTE3_MTE2>();
    }
}

template <typename TilingT, typename QueryT>
__aicore__ inline bool BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::CubeQkQTile(
    LocalTensor<float> qTile, LocalTensor<float> decodedK, LocalTensor<float> scoreTile,
    uint32_t compactRows, uint32_t mRows)
{
    // Reuse rotate KFC with physical K^T in GM (SetTensorB false):
    //   C[M,N] = Q[M,128] @ KT[128,N], N<=64, score stride 64.
    if (!matmulReady_ || !qkGmReady_ || rotateMm_ == nullptr || qkWorkspaceStride_ == 0 ||
        compactRows == 0 || mRows == 0) {
        return false;
    }
    if (compactRows > TQ_BR_QK_CUBE_MAX_M || mRows > TQ_BR_QK_CUBE_MAX_N) {
        return false;
    }

    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t mPad = TqBrAlignUp16(compactRows);
    const uint32_t nPad = TQ_BR_QK_CUBE_MAX_N;  // 64
    constexpr uint32_t kTile = 16;
    const uint32_t coreIdx = GetBlockIdx();
    if (coreIdx >= usedCoreNum_ || qkWorkspaceStride_ < D * nPad) {
        return false;
    }

    auto kHalf = CodeI16Buf().template ReinterpretCast<QueryT>();
    Duplicate(kHalf, static_cast<QueryT>(0), nPad * D);
    PipeBarrier<PIPE_V>();
    Cast(kHalf, decodedK, RoundMode::CAST_RINT, mRows * D);
    PipeBarrier<PIPE_V>();

    // 16x16 vtranspose: K[64,128] → KT[128,64] then stage to per-core GM.
    auto ktHalf = CodeFloatBuf().template ReinterpretCast<QueryT>();
    Duplicate(ktHalf, static_cast<QueryT>(0), D * nPad);
    PipeBarrier<PIPE_V>();
    auto tileSrc = RotateWorkBuf().template ReinterpretCast<QueryT>();
    auto tileDst = tileSrc[kTile * kTile];
    for (uint32_t r = 0; r < nPad; r += kTile) {
        for (uint32_t c = 0; c < D; c += kTile) {
            for (uint32_t i = 0; i < kTile; ++i) {
                DataCopy(tileSrc[i * kTile], kHalf[(r + i) * D + c], kTile);
            }
            PipeBarrier<PIPE_V>();
            Transpose(tileDst, tileSrc);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < kTile; ++i) {
                DataCopy(ktHalf[(c + i) * nPad + r], tileDst[i * kTile], kTile);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    TqBrSync<HardEvent::V_MTE3>();
    DataCopy(qkKGm_[static_cast<uint64_t>(coreIdx) * qkWorkspaceStride_], ktHalf, D * nPad);
    TqBrSync<HardEvent::MTE3_V>();
    TqBrSync<HardEvent::MTE3_MTE2>();

    auto qHalf = DecodedBuf().template ReinterpretCast<QueryT>();
    Cast(qHalf, qTile, RoundMode::CAST_RINT, compactRows * D);
    PipeBarrier<PIPE_V>();
    if (mPad > compactRows) {
        TqBrDuplicateZero(qHalf[compactRows * D], (mPad - compactRows) * D);
    }

    auto scoreHalf = CodeI16Buf().template ReinterpretCast<QueryT>();
    Duplicate(scoreHalf, static_cast<QueryT>(0), mPad * nPad);
    PipeBarrier<PIPE_V>();

    rotateMm_->SetOrgShape(mPad, nPad, D);
    rotateMm_->SetSingleShape(compactRows, mRows, D);
    rotateMm_->SetTensorA(qHalf, false);
    rotateMm_->SetTensorB(qkKGm_[static_cast<uint64_t>(coreIdx) * qkWorkspaceStride_], false);
    rotateMm_->SetLocalWorkspace(RotateWorkBuf());
    rotateMm_->IterateAll(scoreHalf);
    rotateMm_->End();

    Cast(scoreTile, scoreHalf, RoundMode::CAST_NONE, mPad * nPad);
    PipeBarrier<PIPE_V>();
    return true;
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::ComputeAttentionQTile(
    uint32_t tokenStartIdx, uint32_t qRows, uint32_t kvHead, uint32_t gqaStart,
    uint32_t gqaCount)
{
    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t seqIdx = FindSeqIdx(tokenStartIdx);
    const uint32_t maxCausalKvEnd = GetCausalKvEnd(tokenStartIdx + qRows - 1, seqIdx);
    if (maxCausalKvEnd == 0) {
        auto outTmp = DecodedBuf().template ReinterpretCast<QueryT>();
        Duplicate(outTmp, static_cast<QueryT>(0), gqaCount * D);
        PipeBarrier<PIPE_V>();
        TqBrSync<HardEvent::V_MTE3>();
        for (uint32_t q = 0; q < qRows; ++q) {
            for (uint32_t g = 0; g < gqaCount; ++g) {
                const uint32_t headIdx = kvHead * gqaGroupSize_ + gqaStart + g;
                DataCopy(outGm_[(tokenStartIdx + q) * numHeads_ * D + headIdx * D],
                         outTmp[g * D], D);
            }
        }
        TqBrSync<HardEvent::MTE3_V>();
        return;
    }

    auto qTile = QTileGroupBuf();
    auto qHalf = DecodedBuf().template ReinterpretCast<QueryT>();
    for (uint32_t q = 0; q < qRows; ++q) {
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint32_t headIdx = kvHead * gqaGroupSize_ + gqaStart + g;
            DataCopy(qHalf[(q * gqaCount + g) * D],
                     queryGm_[(tokenStartIdx + q) * numHeads_ * D + headIdx * D], D);
        }
    }
    TqBrSync<HardEvent::MTE2_V>();
    Cast(qTile, qHalf, RoundMode::CAST_NONE, qRows * gqaCount * D);
    PipeBarrier<PIPE_V>();
    RotateRowsInPlace(qTile, rotationKeyGm_, qRows * gqaCount);
    PipeBarrier<PIPE_V>();
    Muls(qTile, qTile, scaleValue_, qRows * gqaCount * D);
    PipeBarrier<PIPE_V>();

    float mStateScalar[TQ_BR_UB_QTILE_CAP * TQ_BR_UB_QTILE_GQA_CAP];
    float sStateScalar[TQ_BR_UB_QTILE_CAP * TQ_BR_UB_QTILE_GQA_CAP];
    const uint32_t compactStateSize = qRows * gqaCount;
    for (uint32_t i = 0; i < compactStateSize; ++i) {
        mStateScalar[i] = -3.402823466e+38f;
        sStateScalar[i] = 0.f;
    }
    auto outAccTile = QTileOutAccBuf();
    Duplicate(outAccTile, 0.f, qRows * gqaCount * D);
    PipeBarrier<PIPE_V>();

    auto scoreTile = QTileScoreBuf();
    auto reduceTmp = FloatScratchBuf();
    auto mulTmp = reduceTmp[TQ_BR_HEAD_SIZE];
    auto reduceScalar = KStepBuf();
    // Cube QK: physical K^T in GM + rotateMm SetTensorB(..., false).
    const bool preferCubeQk = (qkPvMode_ == 1) && (qRows > 1);

    float kBaseArr[TQ_BR_UB_KV_TILE_CAP];
    float kStepArr[TQ_BR_UB_KV_TILE_CAP];
    float vminArr[TQ_BR_UB_KV_TILE_CAP];
    float vstepArr[TQ_BR_UB_KV_TILE_CAP];

    // Cache per-q causal ends once (seqIdx fixed inside a qTile).
    uint32_t causalEnds[TQ_BR_UB_QTILE_CAP];
    for (uint32_t q = 0; q < qRows; ++q) {
        causalEnds[q] = GetCausalKvEnd(tokenStartIdx + q, seqIdx);
    }

    for (uint32_t pos = 0; pos < maxCausalKvEnd; pos += kvTileRows_) {
        const uint32_t remainRows = maxCausalKvEnd - pos;
        const uint32_t mRows = (kvTileRows_ < remainRows) ? kvTileRows_ : remainRows;

        LoadPackedKeyTileRows(seqIdx, kvHead, pos, mRows, kBaseArr, kStepArr);
        DecodeKeyTileToFloat(mRows, kBaseArr, kStepArr);
        auto decodedK = DecodedBuf();

        bool usedCube = false;
        if (preferCubeQk) {
            usedCube = CubeQkQTile(qTile, decodedK, scoreTile, qRows * gqaCount, mRows);
        }
        if (!usedCube) {
            for (uint32_t q = 0; q < qRows; ++q) {
                const uint32_t causalKvEnd = causalEnds[q];
                if (pos >= causalKvEnd) {
                    continue;
                }
                const uint32_t activeRows =
                    (pos + mRows <= causalKvEnd) ? mRows : (causalKvEnd - pos);
                turboquant_attn::VectorQkFloatPreScaled(
                    qTile[q * gqaCount * D], decodedK,
                    scoreTile[q * gqaCount * TQ_BR_UB_KV_TILE_CAP], reduceTmp, mulTmp,
                    reduceScalar, gqaCount, activeRows);
            }
        }

        LoadPackedValueTileRows(seqIdx, kvHead, pos, mRows, vminArr, vstepArr);
        DecodeValueTileToFloat(mRows, vminArr, vstepArr);
        auto decodedV = DecodedBuf();
        auto expBuf = FloatScratchBuf();
        auto softmaxReduceTmp = expBuf[TQ_BR_HEAD_SIZE];
        auto weightedValue = mulTmp;

        for (uint32_t q = 0; q < qRows; ++q) {
            const uint32_t causalKvEnd = causalEnds[q];
            if (pos >= causalKvEnd) {
                continue;
            }
            const uint32_t activeRows =
                (pos + mRows <= causalKvEnd) ? mRows : (causalKvEnd - pos);
            turboquant_attn::OnlineSoftmaxUpdateTileFloatPreScaledScalar(
                scoreTile[q * gqaCount * TQ_BR_UB_KV_TILE_CAP], decodedV,
                mStateScalar + q * gqaCount, sStateScalar + q * gqaCount,
                outAccTile[q * gqaCount * D], expBuf, softmaxReduceTmp, weightedValue,
                gqaCount, activeRows);
        }
    }

    WriteFinalOutputQTile(tokenStartIdx, qRows, kvHead, sStateScalar, outAccTile, gqaStart,
                          gqaCount);
}

// ---------------------------------------------------------------
// ProcessSplitBn() - round-robin by default; contiguous + qTile for prefill
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::ProcessSplitBn(
    uint32_t coreIdx)
{
    if (coreIdx >= usedCoreNum_) {
        return;
    }

    const uint32_t gqaGroup = numHeads_ / numKvHeads_;
    const uint32_t gqaChunkCount = (gqaGroup + TQ_BR_UB_GQA_CAP - 1) / TQ_BR_UB_GQA_CAP;
    const uint32_t headChunkScale = numKvHeads_ * gqaChunkCount;

    // Decode / short-Q: keep validated round-robin ownership.
    if (qTileMode_ == 0 || gqaGroup > TQ_BR_UB_QTILE_GQA_CAP) {
        uint32_t taskIdx = 0;
        for (uint32_t tokenIdx = 0; tokenIdx < numTokens_; ++tokenIdx) {
            for (uint32_t kvHead = 0; kvHead < numKvHeads_; ++kvHead) {
                for (uint32_t gqaChunk = 0; gqaChunk < gqaChunkCount; ++gqaChunk) {
                    if (taskIdx % usedCoreNum_ == coreIdx) {
                        const uint32_t gqaStart = gqaChunk * TQ_BR_UB_GQA_CAP;
                        const uint32_t gqaTileEnd = gqaStart + TQ_BR_UB_GQA_CAP;
                        const uint32_t gqaEnd =
                            (gqaTileEnd < gqaGroup) ? gqaTileEnd : gqaGroup;
                        ComputeAttention(tokenIdx, kvHead, gqaStart, gqaEnd - gqaStart);
                    }
                    ++taskIdx;
                }
            }
        }
        return;
    }

    // Prefill qTile: partition by token so consecutive Q rows land on one core
    // and can share a single KV decode stream per (kvHead, gqaChunk).
    (void)headChunkScale;
    const uint32_t tokensPerCore =
        (numTokens_ + usedCoreNum_ - 1) / usedCoreNum_;
    const uint32_t tokenStart = coreIdx * tokensPerCore;
    if (tokenStart >= numTokens_) {
        return;
    }
    uint32_t tokenEnd = tokenStart + tokensPerCore;
    if (tokenEnd > numTokens_) {
        tokenEnd = numTokens_;
    }

    for (uint32_t kvHead = 0; kvHead < numKvHeads_; ++kvHead) {
        for (uint32_t gqaChunk = 0; gqaChunk < gqaChunkCount; ++gqaChunk) {
            const uint32_t gqaStart = gqaChunk * TQ_BR_UB_GQA_CAP;
            const uint32_t gqaTileEnd = gqaStart + TQ_BR_UB_GQA_CAP;
            const uint32_t gqaEnd = (gqaTileEnd < gqaGroup) ? gqaTileEnd : gqaGroup;
            const uint32_t gqaCount = gqaEnd - gqaStart;

            uint32_t tokenIdx = tokenStart;
            while (tokenIdx < tokenEnd) {
                const uint32_t tileStart = tokenIdx;
                uint32_t tileEnd = tileStart + 1;
                while (tileEnd < tokenEnd &&
                       tileEnd - tileStart < TQ_BR_UB_QTILE_CAP) {
                    if (FindSeqIdx(tileEnd) != FindSeqIdx(tileStart)) {
                        break;
                    }
                    ++tileEnd;
                }
                const uint32_t qRows = tileEnd - tileStart;
                if (qRows > 1) {
                    ComputeAttentionQTile(tileStart, qRows, kvHead, gqaStart, gqaCount);
                } else {
                    ComputeAttention(tileStart, kvHead, gqaStart, gqaCount);
                }
                tokenIdx = tileEnd;
            }
        }
    }
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::ProcessSplitBns(
    uint32_t coreIdx)
{
    if (coreIdx >= usedCoreNum_) {
        return;
    }

    const uint32_t gqaGroup = numHeads_ / numKvHeads_;
    const uint32_t gqaChunkCount = (gqaGroup + TQ_BR_UB_GQA_CAP - 1) / TQ_BR_UB_GQA_CAP;
    const uint32_t headChunkScale = numKvHeads_ * gqaChunkCount;
    const uint32_t seqTaskScale = headChunkScale * kvSplitPart_;

    uint32_t taskIdx = 0;
    for (uint32_t tokenIdx = 0; tokenIdx < numTokens_; ++tokenIdx) {
        for (uint32_t kvHead = 0; kvHead < numKvHeads_; ++kvHead) {
            for (uint32_t gqaChunk = 0; gqaChunk < gqaChunkCount; ++gqaChunk) {
                for (uint32_t segIdx = 0; segIdx < kvSplitPart_; ++segIdx) {
                    if (taskIdx % usedCoreNum_ == coreIdx) {
                        const uint32_t gqaStart = gqaChunk * TQ_BR_UB_GQA_CAP;
                        const uint32_t gqaTileEnd = gqaStart + TQ_BR_UB_GQA_CAP;
                        const uint32_t gqaEnd = (gqaTileEnd < gqaGroup) ? gqaTileEnd : gqaGroup;
                        const uint32_t gqaCount = gqaEnd - gqaStart;
                        const uint32_t kvStart = segIdx * kvSegmentLen_;
                        const uint32_t kvEnd = (segIdx + 1) * kvSegmentLen_;
                        const bool writePartial = kvSplitPart_ > 1;
                        ComputeAttention(tokenIdx, kvHead, gqaStart, gqaCount, kvStart, kvEnd,
                                         writePartial, segIdx);
                    }
                    ++taskIdx;
                }
            }
        }
    }
}

template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::CombineFlashDecode(
    uint32_t coreIdx, uint32_t combineCoreNum)
{
    const uint32_t D = TQ_BR_HEAD_SIZE;
    // FloatScratch has 256 fp32: [0:128) for LSE copy, [128:256) for Exp workspace.
    auto lseLocal = FloatScratchBuf();
    auto expLocal = FloatScratchBuf()[TQ_BR_HEAD_SIZE];
    auto globalOut = QGroupFloatBuf();
    auto partFloat = OutAccBuf();
    auto outLocal = DecodedBuf().template ReinterpretCast<QueryT>();
    const uint32_t parts = kvSplitPart_;
    if (parts == 0 || parts * 2 > TQ_BR_HEAD_SIZE || combineCoreNum == 0) {
        return;
    }
    // OutAcc holds up to GQA_CAP * HEAD floats; reuse as contiguous partial buffer.
    const uint32_t maxPartsInOutAcc = TQ_BR_UB_GQA_CAP;
    const bool bulkPartialLoad = (parts <= maxPartsInOutAcc);

    const uint32_t totalTH = numTokens_ * numHeads_;
    for (uint32_t th = coreIdx; th < totalTH; th += combineCoreNum) {
        const uint32_t tokenIdx = th / numHeads_;
        const uint32_t headIdx = th % numHeads_;
        const uint32_t lseBase = th * parts * 2;
        DataCopy(lseLocal, partialLseGm_[lseBase], parts * 2);
        TqBrSync<HardEvent::MTE2_V>();
        TqBrSync<HardEvent::V_S>();

        float bestM = -3.402823466e+38f;
        for (uint32_t p = 0; p < parts; ++p) {
            const float partM = lseLocal.GetValue(p * 2);
            const float partS = lseLocal.GetValue(p * 2 + 1);
            if (partS > 0.f && partM > bestM) {
                bestM = partM;
            }
        }

        // Batch Exp(partM - bestM) for all parts (one S→V sync for SetValue loop).
        for (uint32_t p = 0; p < parts; ++p) {
            const float partM = lseLocal.GetValue(p * 2);
            const float partS = lseLocal.GetValue(p * 2 + 1);
            const float delta = (partS > 0.f) ? (partM - bestM) : -1000.f;
            expLocal.SetValue(p, delta);
        }
        TqBrSync<HardEvent::S_V>();
        Exp(expLocal, expLocal, parts);
        PipeBarrier<PIPE_V>();
        TqBrSync<HardEvent::V_S>();

        float globalS = 0.f;
        Duplicate(globalOut, 0.f, D);
        PipeBarrier<PIPE_V>();

        if (bulkPartialLoad) {
            const uint32_t partBase = th * parts;
            DataCopy(partFloat, partialOutGm_[partBase * D], parts * D);
            TqBrSync<HardEvent::MTE2_V>();
            for (uint32_t p = 0; p < parts; ++p) {
                const float partS = lseLocal.GetValue(p * 2 + 1);
                if (partS <= 0.f) {
                    continue;
                }
                const float weight = expLocal.GetValue(p) * partS;
                globalS += weight;
                Axpy(globalOut, partFloat[p * D], weight, D);
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t p = 0; p < parts; ++p) {
                const float partS = lseLocal.GetValue(p * 2 + 1);
                if (partS <= 0.f) {
                    continue;
                }
                const float weight = expLocal.GetValue(p) * partS;
                globalS += weight;
                const uint32_t partIdx = th * parts + p;
                DataCopy(partFloat, partialOutGm_[partIdx * D], D);
                TqBrSync<HardEvent::MTE2_V>();
                Axpy(globalOut, partFloat, weight, D);
                PipeBarrier<PIPE_V>();
            }
        }
        const float invS = (globalS > 0.f) ? (1.f / globalS) : 0.f;
        Muls(globalOut, globalOut, invS, D);
        PipeBarrier<PIPE_V>();
        RotateRowsInPlace(globalOut, rotationValueGm_, 1);
        Cast(outLocal, globalOut, RoundMode::CAST_RINT, D);
        PipeBarrier<PIPE_V>();
        TqBrSync<HardEvent::V_MTE3>();
        DataCopy(outGm_[th * D], outLocal, D);
        TqBrSync<HardEvent::MTE3_V>();
        TqBrSync<HardEvent::MTE3_MTE2>();
    }
}

// ---------------------------------------------------------------
// Process()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::Process()
{
    if (splitMode_ == 1) {
        const uint32_t coreIdx = GetBlockIdx() / TQ_BR_MIX_AIV_SUB;
        if ASCEND_IS_AIC {
            SyncAll();
            return;
        }
        const bool isPrimaryAiv = ((GetSubBlockIdx() % TQ_BR_MIX_AIV_SUB) == 0);
        if (isPrimaryAiv && coreIdx < usedCoreNum_) {
            ProcessSplitBns(coreIdx);
        }
        SyncAll();
        // Combine is embarrassingly parallel over (token, head); spread across
        // the same primary AIVs that produced partials.
        if (isPrimaryAiv && coreIdx < usedCoreNum_) {
            CombineFlashDecode(coreIdx, usedCoreNum_);
        }
        return;
    }

    // AIC returns — SplitBN mode is AIV-only.
    if ASCEND_IS_AIC {
        return;
    }
    const uint32_t coreIdx = GetBlockIdx();
    if (coreIdx >= usedCoreNum_) {
        return;
    }
    ProcessSplitBn(coreIdx);
}

// ---------------------------------------------------------------
// Global kernel entry point
// ---------------------------------------------------------------
extern "C" __global__ __aicore__ void bit_residual_attention_paged_k8v4(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR block_table, GM_ADDR actual_seq_len_q,
    GM_ADDR actual_seq_len_kv, GM_ADDR rotation_key,
    GM_ADDR rotation_value, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }

    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    using TilingT = BitResidualAttentionPagedK8v4TilingData;
    using QueryT = TqQueryT;
    TCubeTiling decodeTiling = tilingData.decodeRotateTiling;
    TqRotateMatmulOp<QueryT> rotateMm;
    // Single KFC object: rotation and qTile Cube QK time-share rotateMm
    // (Cube QK stages physical K^T in GM; SetTensorB(..., false)).
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &decodeTiling);
    BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT> op;
    op.Init(query, key_cache, value_cache, block_table, actual_seq_len_q,
            actual_seq_len_kv, rotation_key, rotation_value, out, workspace,
            &tilingData, &pipe, &rotateMm);
    op.Process();
}
