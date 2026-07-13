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
//   K: 8-bit code = (q7 << 1) | sign → sig_vec=±1, err=base+q7*step → decoded=err*sig_vec
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

// packedMaskBuf (uint16 mask constant, filled once)
constexpr uint32_t TQ_BR_UB_MASK_OFFSET = TQ_BR_UB_FLOAT_SCRATCH_OFFSET + TQ_BR_UB_FLOAT_SCRATCH_BYTES;
constexpr uint32_t TQ_BR_UB_MASK_BYTES = TQ_BR_HEAD_SIZE * sizeof(uint16_t);

// rotateWorkBuf (for Cube matmul workspace)
constexpr uint32_t TQ_BR_UB_ROTATE_WORK_OFFSET = TQ_BR_UB_MASK_OFFSET + TQ_BR_UB_MASK_BYTES;
constexpr uint32_t TQ_BR_UB_ROTATE_WORK_BYTES = TQ_BR_HEAD_SIZE * TQ_BR_HEAD_SIZE * TQ_BR_DTYPE_BYTES;  // 32768

constexpr uint32_t TQ_BR_UB_TOTAL_BYTES = TQ_BR_UB_ROTATE_WORK_OFFSET + TQ_BR_UB_ROTATE_WORK_BYTES;
static_assert(TQ_BR_UB_TOTAL_BYTES <= 192 * 1024, "UB budget exceeded");

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
                                GM_ADDR rotationValue, GM_ADDR out,
                                const TilingT* tilingData, TPipe* pipe,
                                TqRotateMatmulOp<QueryT>* rotateMm);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSplitBn(uint32_t coreIdx);

    // ---- KV tile loading ----
    __aicore__ inline void LoadPackedKeyTileRows(uint32_t seqIdx, uint32_t kvHead,
                                                  uint32_t absStart, uint32_t mRows);
    __aicore__ inline void LoadPackedValueTileRows(uint32_t seqIdx, uint32_t kvHead,
                                                    uint32_t absStart, uint32_t mRows);

    // ---- Decode ----
    __aicore__ inline void DecodeKeyTileToFloat(uint32_t mRows);
    __aicore__ inline void DecodeValueTileToFloat(uint32_t mRows);

    // ---- Attention compute ----
    __aicore__ inline void ComputeAttention(uint32_t tokenIdx, uint32_t kvHead,
                                            uint32_t gqaStart, uint32_t gqaCount);
    __aicore__ inline void RotateRowsFloatScalar(LocalTensor<float> rows,
                                              GlobalTensor<QueryT>& rotationGm,
                                              uint32_t rowCount);

    // ---- Helpers ----
    __aicore__ inline uint32_t GetCausalKvEnd(uint32_t tokenIdx, uint32_t seqIdx);
    __aicore__ inline void InitPackedMask();
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
    __aicore__ inline LocalTensor<uint8_t> RotateWorkBuf() {
        return TqMakeVecCalcLocalTensor<uint8_t>(
            TQ_BR_UB_ROTATE_WORK_OFFSET, TQ_BR_UB_ROTATE_WORK_BYTES);
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
    float scaleValue_ = 0.f;

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

    // ---- Cube matmul ----
    TPipe* pipe_ = nullptr;
    TqRotateMatmulOp<QueryT>* rotateMm_ = nullptr;
};

// ---------------------------------------------------------------
// Init()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::Init(
    GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache,
    GM_ADDR blockTable, GM_ADDR actualSeqLenQ,
    GM_ADDR actualSeqLenKv, GM_ADDR rotationKey,
    GM_ADDR rotationValue, GM_ADDR out,
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

    InitPackedMask();
}

// ---------------------------------------------------------------
// InitPackedMask() - fill MaskBuf with key/value extraction masks
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::InitPackedMask() {
    // Fill mask buffer with 0x00FF for key code byte extraction (8-bit code in uint16)
    auto mask = MaskBuf();
    Duplicate(mask, static_cast<uint16_t>(0x00FF), TQ_BR_HEAD_SIZE);
    PipeBarrier<PIPE_V>();
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
    uint32_t seqIdx, uint32_t kvHead, uint32_t absStart, uint32_t mRows)
{
    // Key cache: 16-row sub-block layout
    // [8 packed 2-row code groups] [16 bases] [16 steps] = 2176 bytes.
    // blockSize/16 sub-blocks per block.
    //
    // For each row, find its 16-row sub-block in the block, load it, extract code + base + step.

    const uint32_t keySubBlocksPerBlock = blockSize_ / TQ_BR_BLOCK_ROWS;
    const uint32_t seqBlockBase = seqIdx * maxBlocksPerSeq_;
    auto packedRaw = PackedRawBuf();
    auto codeI16 = CodeI16Buf();
    auto floatScratch = FloatScratchBuf();
    auto kBase = KBaseBuf();
    auto kStep = KStepBuf();

    const uint32_t D = TQ_BR_HEAD_SIZE;

    for (uint32_t row = 0; row < mRows; ++row) {
        const uint32_t absPos = absStart + row;
        const uint32_t blockOffset = absPos / blockSize_;
        const uint32_t posInBlock = absPos % blockSize_;
        const uint32_t rowInSubBlock = posInBlock % TQ_BR_BLOCK_ROWS;

        // Look up physical block from block_table.
        TqBrSync<HardEvent::S_MTE2>();
        const int32_t blockId = blockTableGm_.GetValue(seqBlockBase + blockOffset);
        TqBrSync<HardEvent::MTE2_S>();

        const uint32_t headStride = blockSize_ * (TQ_BR_HEAD_SIZE + 2 * sizeof(float));
        const uint64_t headBase =
            (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * headStride;
        DataCopyExtParams copyParams{1, TQ_BR_HEAD_SIZE, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(packedRaw, keyCacheGm_[headBase + posInBlock * TQ_BR_HEAD_SIZE], copyParams, padParams);
        DataCopyExtParams metaParams{1, sizeof(float), 0, 0, 0};
        DataCopyPad(packedRaw[TQ_BR_HEAD_SIZE],
            keyCacheGm_[headBase + blockSize_ * TQ_BR_HEAD_SIZE + posInBlock * sizeof(float)],
            metaParams, padParams);
        DataCopyPad(packedRaw[TQ_BR_HEAD_SIZE + 32],
            keyCacheGm_[headBase + blockSize_ * (TQ_BR_HEAD_SIZE + sizeof(float)) +
                        posInBlock * sizeof(float)], metaParams, padParams);
        TqBrSync<HardEvent::MTE2_V>();
        TqBrSync<HardEvent::MTE2_S>();

        auto extractI16 = codeI16[row * D];
        for (uint32_t d = 0; d < D; ++d) {
            extractI16.SetValue(d, static_cast<int16_t>(packedRaw.GetValue(d)));
        }

        // Read base and step at new offsets (base at rowInSubBlock*4 within base zone,
        // step at rowInSubBlock*4 within step zone).
        const float base = TqBrReadFloatFromU8(packedRaw, floatScratch, TQ_BR_HEAD_SIZE);
        kBase.SetValue(row, base);

        const float step = TqBrReadFloatFromU8(packedRaw, floatScratch, TQ_BR_HEAD_SIZE + 32);
        kStep.SetValue(row, step);
    }
    TqBrSync<HardEvent::V_S>();
}

// ---------------------------------------------------------------
// LoadPackedValueTileRows()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::LoadPackedValueTileRows(
    uint32_t seqIdx, uint32_t kvHead, uint32_t absStart, uint32_t mRows)
{
    // Value cache: 16-row sub-block layout
    // [4 packed 4-row code groups] [16 vmins] [16 vsteps] = 1152 bytes.

    const uint32_t valSubBlocksPerBlock = blockSize_ / TQ_BR_BLOCK_ROWS;
    const uint32_t seqBlockBase = seqIdx * maxBlocksPerSeq_;
    auto packedRaw = PackedRawBuf();
    auto codeI16 = CodeI16Buf();  // reuse for value idx4 extraction
    auto codeFloat = CodeFloatBuf();
    auto floatScratch = FloatScratchBuf();
    auto vminBuf = VminBuf();
    auto vstepBuf = VstepBuf();

    const uint32_t D = TQ_BR_HEAD_SIZE;

    for (uint32_t row = 0; row < mRows; ++row) {
        const uint32_t absPos = absStart + row;
        const uint32_t blockOffset = absPos / blockSize_;
        const uint32_t posInBlock = absPos % blockSize_;
        const uint32_t rowInSubBlock = posInBlock % TQ_BR_BLOCK_ROWS;

        TqBrSync<HardEvent::S_MTE2>();
        const int32_t blockId = blockTableGm_.GetValue(seqBlockBase + blockOffset);
        TqBrSync<HardEvent::MTE2_S>();

        const uint32_t rowBytes = TQ_BR_HEAD_SIZE / 2;
        const uint32_t headStride = blockSize_ * (rowBytes + 2 * sizeof(float));
        const uint64_t headBase =
            (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * headStride;
        DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(packedRaw, valueCacheGm_[headBase + posInBlock * rowBytes], copyParams, padParams);
        DataCopyExtParams metaParams{1, sizeof(float), 0, 0, 0};
        DataCopyPad(packedRaw[rowBytes],
            valueCacheGm_[headBase + blockSize_ * rowBytes + posInBlock * sizeof(float)],
            metaParams, padParams);
        DataCopyPad(packedRaw[rowBytes + 32],
            valueCacheGm_[headBase + blockSize_ * (rowBytes + sizeof(float)) +
                          posInBlock * sizeof(float)], metaParams, padParams);
        TqBrSync<HardEvent::MTE2_V>();
        TqBrSync<HardEvent::MTE2_S>();

        auto extractI16 = codeI16[row * D];
        for (uint32_t d = 0; d < D / 2; ++d) {
            const uint8_t code = packedRaw.GetValue(d);
            extractI16.SetValue(2 * d, static_cast<int16_t>(code & 0x0f));
            extractI16.SetValue(2 * d + 1, static_cast<int16_t>(code >> 4));
        }

        // Cast extracted idx4 to float for decode.
        Cast(codeFloat[row * D], extractI16, RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();

        // Read vmin and vstep at new offsets (rowInSubBlock*4 within vmin/vstep zone).
        const float vmin = TqBrReadFloatFromU8(packedRaw, floatScratch, rowBytes);
        vminBuf.SetValue(row, vmin);

        const float vstep = TqBrReadFloatFromU8(packedRaw, floatScratch, rowBytes + 32);
        vstepBuf.SetValue(row, vstep);
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
    uint32_t mRows)
{
    // Input: CodeI16Buf()[row*128] has the 8-bit code for each dimension.
    // kBase[k], kStep[k] are per-row scalar floats.
    //
    // Decode: code = (q7 << 1) | sign
    //   sign_bit = code & 1        → 0=positive, 1=negative
    //   q7       = code >> 1       → [0, 127]
    //   sig_vec  = 1 - 2*sign_bit  → {+1.0, -1.0}
    //   err      = base + q7 * step (positive residual)
    //   decoded  = err * sig_vec   (restore original sign per dimension)
    //   NO norm multiplication!

    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t n = mRows * D;
    auto codeI16 = CodeI16Buf();
    auto codeFloat = CodeFloatBuf();
    auto decoded = DecodedBuf();
    auto kBase = KBaseBuf();
    auto kStep = KStepBuf();

    // Step 1: Extract sign_bit (code & 1) and q7 (code >> 1).
    auto signStorage = decoded.template ReinterpretCast<int16_t>();
    auto signStorageU16 = signStorage.template ReinterpretCast<uint16_t>();
    auto codeU16 = codeI16.template ReinterpretCast<uint16_t>();
    auto signMask = MaskBuf();
    Duplicate(signMask, static_cast<uint16_t>(1), D);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < mRows; ++row) {
        And(signStorageU16[row * D], codeU16[row * D], signMask, D);
    }
    PipeBarrier<PIPE_V>();

    ShiftRight(codeI16, codeI16, static_cast<int16_t>(1), n);
    PipeBarrier<PIPE_V>();

    // Step 2: sign_bit → sig_vec = ±1.0
    // sign_bit = 0 (positive) → sig_vec = +1.0
    // sign_bit = 1 (negative) → sig_vec = -1.0
    // sig_vec = 1 - 2*sign_bit = Cast(sign) * (-2) + 1
    auto signF32Final = codeFloat;
    Cast(signF32Final, signStorage, RoundMode::CAST_NONE, n);  // sign_bit: 0.0 or 1.0
    PipeBarrier<PIPE_V>();
    Muls(signF32Final, signF32Final, -2.0f, n);   // 0→0, 1→-2
    PipeBarrier<PIPE_V>();
    Adds(signF32Final, signF32Final, 1.0f, n);    // 0→+1, 1→-1 = sig_vec
    PipeBarrier<PIPE_V>();

    // Step 3: q7 → float, then err = base + q7 * step (broadcast per-row)
    auto q7F32 = decoded;
    Cast(q7F32, codeI16, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();

    // Broadcast base and step per-row: err[row][d] = base[row] + q7[row][d] * step[row]
    for (uint32_t row = 0; row < mRows; ++row) {
        const float base = kBase.GetValue(row);
        const float step = kStep.GetValue(row);
        auto rowFloat = q7F32[row * D];
        Muls(rowFloat, rowFloat, step, D);
        PipeBarrier<PIPE_V>();
        Adds(rowFloat, rowFloat, base, D);
        PipeBarrier<PIPE_V>();
    }

    // Step 4: decoded = err * sig_vec (restore sign per dimension)
    Mul(decoded, q7F32, signF32Final, n);
    PipeBarrier<PIPE_V>();
}

// ---------------------------------------------------------------
// DecodeValueTileToFloat()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::DecodeValueTileToFloat(
    uint32_t mRows)
{
    // Input: CodeFloatBuf()[row*128] has the 4-bit idx4 widened to float.
    // vmin[k], vstep[k] are per-row scalar floats.
    //
    // Decode: y = vmin + idx4 * vstep (broadcast per-row, no norm)

    const uint32_t D = TQ_BR_HEAD_SIZE;
    auto decoded = DecodedBuf();
    auto vminBuf = VminBuf();
    auto vstepBuf = VstepBuf();

    auto codeFloat = CodeFloatBuf();

    // Broadcast vmin and vstep per-row.
    for (uint32_t row = 0; row < mRows; ++row) {
        const float vmin = vminBuf.GetValue(row);
        const float vstep = vstepBuf.GetValue(row);
        auto rowFloat = decoded[row * D];
        auto idx4Float = codeFloat[row * D];
        Muls(rowFloat, idx4Float, vstep, D);
        PipeBarrier<PIPE_V>();
        Adds(rowFloat, rowFloat, vmin, D);
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

// ---------------------------------------------------------------
// ComputeAttention() - single-token decode
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::ComputeAttention(
    uint32_t tokenIdx, uint32_t kvHead, uint32_t gqaStart, uint32_t gqaCount)
{
    const uint32_t D = TQ_BR_HEAD_SIZE;

    // Determine KV range.
    uint32_t seqIdx = 0;
    // Find which sequence this token belongs to.
    {
        int64_t prev = 0;
        TqBrSync<HardEvent::S_MTE2>();
        for (uint32_t i = 0; i < batchSize_; ++i) {
            const int64_t end = actualSeqLenQGm_.GetValue(i);
            if (static_cast<int64_t>(tokenIdx) < end) {
                seqIdx = i;
                break;
            }
            prev = end;
        }
        TqBrSync<HardEvent::MTE2_S>();
    }

    const uint32_t causalKvEnd = GetCausalKvEnd(tokenIdx, seqIdx);
    if (causalKvEnd == 0) {
        // Zero-length KV: output is zero.
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
    RotateRowsFloatScalar(qGroupFloat, rotationKeyGm_, gqaCount);
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

    // Main KV tile loop.
    for (uint32_t pos = 0; pos < causalKvEnd; pos += kvTileRows_) {
        const uint32_t remainRows = causalKvEnd - pos;
        const uint32_t mRows = (kvTileRows_ < remainRows) ? kvTileRows_ : remainRows;

        // Load and decode K tile.
        LoadPackedKeyTileRows(seqIdx, kvHead, pos, mRows);
        DecodeKeyTileToFloat(mRows);

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
        LoadPackedValueTileRows(seqIdx, kvHead, pos, mRows);
        DecodeValueTileToFloat(mRows);

        auto decodedV = DecodedBuf();  // decoded V in float

        auto expBuf = FloatScratchBuf();
        auto softmaxReduceTmp = expBuf[TQ_BR_HEAD_SIZE];
        auto weightedValue = expBuf[TQ_BR_HEAD_SIZE];
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

    // Post-rotate: out @ rotationValue (R).
    RotateRowsFloatScalar(outAcc, rotationValueGm_, gqaCount);
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

// ---------------------------------------------------------------
// ProcessSplitBn() - SplitBN mode (decode only, no FlashDecode)
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
    const uint32_t taskCount = numTokens_ * headChunkScale;

    uint32_t taskIdx = 0;
    for (uint32_t tokenIdx = 0; tokenIdx < numTokens_; ++tokenIdx) {
        for (uint32_t kvHead = 0; kvHead < numKvHeads_; ++kvHead) {
            for (uint32_t gqaChunk = 0; gqaChunk < gqaChunkCount; ++gqaChunk) {
                if (taskIdx % usedCoreNum_ == coreIdx) {
                    const uint32_t gqaStart = gqaChunk * TQ_BR_UB_GQA_CAP;
                    const uint32_t gqaTileEnd = gqaStart + TQ_BR_UB_GQA_CAP;
                    const uint32_t gqaEnd = (gqaTileEnd < gqaGroup) ? gqaTileEnd : gqaGroup;
                    const uint32_t gqaCount = gqaEnd - gqaStart;
                    ComputeAttention(tokenIdx, kvHead, gqaStart, gqaCount);
                }
                ++taskIdx;
            }
        }
    }
}

// ---------------------------------------------------------------
// Process()
// ---------------------------------------------------------------
template <typename TilingT, typename QueryT>
__aicore__ inline void BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT>::Process()
{
    // AIC returns — this kernel is AIV-only for SplitBN mode.
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
    TqRotateMatmulOp<QueryT> rotateMm;

    // Register matmul object (required for KFC).
    REGIST_MATMUL_OBJ_STATIC(&pipe, GetSysWorkSpacePtr(), rotateMm, (TCubeTiling*)nullptr);

    BitResidualAttentionPagedK8v4Kernel<TilingT, QueryT> op;
    op.Init(query, key_cache, value_cache, block_table,
            actual_seq_len_q, actual_seq_len_kv, rotation_key,
            rotation_value, out, &tilingData, &pipe, &rotateMm);
    op.Process();
}
