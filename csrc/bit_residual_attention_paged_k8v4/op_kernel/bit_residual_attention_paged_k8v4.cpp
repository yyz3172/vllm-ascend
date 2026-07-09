/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// BitResidual K8V4: fused sign+residual decode + paged attention.
// Reads packed KV cache in bit_residual format and performs attention:
//   K: 8-bit code = (q7 << 1) | sign → sign_val + base + q7*step → y * norm
//   V: 4-bit idx4 → vmin + idx4*vstep (no norm)
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
static constexpr uint32_t TQ_BR_GROUP_COPY_BYTES = ((TQ_BR_GROUP_STRIDE + 31) / 32) * 32;  // 288 aligned to 32

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

// UB layout (total ~115 KiB, fits in 192 KiB on 910B):
//   packedRawBuf:       KV_TILE_CAP / KEY_GROUP_ROWS * TQ_BR_GROUP_COPY_BYTES = 64/2 * 288 = 9216 bytes (key) + same for val
//   But we time-share: load K group → extract → decode → load V group → extract → decode
//   So only one set of raw group buffers needed.
//   packedRawBuf:       TQ_BR_UB_KV_TILE_CAP * TQ_BR_GROUP_COPY_BYTES bytes  (but we load at most kvTileRows groups)
//                       Upper bound: 64 * 288 = 18432 bytes

// Simplified approach: allocate fixed-size buffers for worst-case kvTileRows=64.

constexpr uint32_t TQ_BR_UB_PACKED_RAW_OFFSET = 0;
constexpr uint32_t TQ_BR_UB_PACKED_RAW_BYTES = TQ_BR_UB_KV_TILE_CAP * TQ_BR_GROUP_COPY_BYTES;  // 18432

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

// kNormBuf (key norms, kvTileRows floats)
constexpr uint32_t TQ_BR_UB_KNORM_OFFSET = TQ_BR_UB_SCORE_OFFSET + TQ_BR_UB_SCORE_BYTES;
constexpr uint32_t TQ_BR_UB_KNORM_BYTES = TQ_BR_UB_KV_TILE_CAP * sizeof(float);  // 256

// kBaseBuf (key base, kvTileRows floats)
constexpr uint32_t TQ_BR_UB_KBASE_OFFSET = TQ_BR_UB_KNORM_OFFSET + TQ_BR_UB_KNORM_BYTES;
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

// floatScratch for scalar reads (base/step/norm)
constexpr uint32_t TQ_BR_UB_FLOAT_SCRATCH_OFFSET = TQ_BR_UB_DECODED_OFFSET + TQ_BR_UB_DECODED_BYTES;
constexpr uint32_t TQ_BR_UB_FLOAT_SCRATCH_BYTES = TQ_BR_HEAD_SIZE * 2 * sizeof(float);

// packedMaskBuf (uint16 mask constant, filled once)
constexpr uint32_t TQ_BR_UB_MASK_OFFSET = TQ_BR_UB_FLOAT_SCRATCH_OFFSET + TQ_BR_UB_FLOAT_SCRATCH_BYTES;
constexpr uint32_t TQ_BR_UB_MASK_BYTES = TQ_BR_GROUP_INDEX_BYTES;  // 256

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
    __aicore__ inline LocalTensor<float> KNormBuf() {
        return TqMakeVecCalcLocalTensor<float>(
            TQ_BR_UB_KNORM_OFFSET, TQ_BR_UB_KNORM_BYTES);
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

    // Norm scratch (reuses floatScratch)
    __aicore__ inline LocalTensor<QueryT> NormScratchBuf() {
        return TqMakeVecCalcLocalTensor<QueryT>(
            TQ_BR_UB_FLOAT_SCRATCH_OFFSET + TQ_BR_HEAD_SIZE * sizeof(float),
            TQ_BR_HEAD_SIZE * sizeof(float));
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
        totalCacheBlocks_ * numKvHeads_ * (blockSize_ / TQ_BR_KEY_GROUP_ROWS) * TQ_BR_GROUP_STRIDE);
    valueCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)valueCache,
        totalCacheBlocks_ * numKvHeads_ * (blockSize_ / TQ_BR_VALUE_GROUP_ROWS) * TQ_BR_GROUP_STRIDE);
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
    Duplicate(mask, static_cast<uint16_t>(0x00FF), TQ_BR_GROUP_INDEX_WORDS);
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
    // Key cache: [num_blocks, num_kv_heads, (block_size/2) * 288]
    // Each 2-row group occupies 288 bytes.
    // group_base = (block_id * num_kv_heads + kv_head) * (block_size/2) * 288 + group_in_block * 288

    const uint32_t keyBlockStride = (blockSize_ / TQ_BR_KEY_GROUP_ROWS) * TQ_BR_GROUP_STRIDE;
    const uint32_t seqBlockBase = seqIdx * maxBlocksPerSeq_;
    auto packedRaw = PackedRawBuf();
    auto codeI16 = CodeI16Buf();
    auto normScratch = NormScratchBuf();
    auto floatScratch = FloatScratchBuf();
    auto kNorm = KNormBuf();
    auto kBase = KBaseBuf();
    auto kStep = KStepBuf();
    auto mask = MaskBuf();

    const uint32_t D = TQ_BR_HEAD_SIZE;

    // Extract code bytes, norms, base, step per row.
    // We load each group individually since the layout is different from TQ 4bit.
    for (uint32_t row = 0; row < mRows; ++row) {
        const uint32_t absPos = absStart + row;
        const uint32_t blockOffset = absPos / blockSize_;
        const uint32_t posInBlock = absPos % blockSize_;
        const uint32_t groupIdx = posInBlock / TQ_BR_KEY_GROUP_ROWS;
        const uint32_t groupRow = posInBlock % TQ_BR_KEY_GROUP_ROWS;

        // Look up physical block from block_table.
        TqBrSync<HardEvent::S_MTE2>();
        const int32_t blockId = blockTableGm_.GetValue(seqBlockBase + blockOffset);
        TqBrSync<HardEvent::MTE2_S>();

        // Compute GM offset for this group.
        const uint64_t groupBase = static_cast<uint64_t>(blockId) * numKvHeads_ * keyBlockStride +
                                   static_cast<uint64_t>(kvHead) * keyBlockStride +
                                   static_cast<uint64_t>(groupIdx) * TQ_BR_GROUP_STRIDE;

        DataCopyExtParams copyParams{1, TQ_BR_GROUP_STRIDE, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(packedRaw, keyCacheGm_[groupBase], copyParams, padParams);
        TqBrSync<HardEvent::MTE2_V>();

        // Reinterpret as uint16 for code extraction.
        auto groupU16 = packedRaw.template ReinterpretCast<uint16_t>();

        // Extract code byte for this group_row.
        // Key uint16[d] = (code_row1[d] << 8) | code_row0[d]
        // group_row == 0: low byte → And with 0x00FF
        // group_row == 1: high byte → ShiftRight 8 then And with 0x00FF
        auto extractI16 = codeI16[row * D];
        auto extractU16 = extractI16.template ReinterpretCast<uint16_t>();

        if (groupRow == 0) {
            And(extractU16, groupU16, mask, D);
        } else {
            ShiftRight(extractI16, groupU16.template ReinterpretCast<int16_t>(),
                       static_cast<int16_t>(groupRow * 8), D);
            PipeBarrier<PIPE_V>();
            And(extractU16, extractI16.template ReinterpretCast<uint16_t>(), mask, D);
        }
        PipeBarrier<PIPE_V>();

        // Read norm (fp16/bf16 word at byte offset 256 + groupRow*2).
        const uint32_t normByteOff = TQ_BR_KEY_NORM_BYTE_OFFSET + groupRow * sizeof(QueryT);
        const float norm = TqBrReadNormToFloat<QueryT>(packedRaw, normScratch, floatScratch, normByteOff);
        kNorm.SetValue(row, norm);

        // Read base (float at byte offset 260 + groupRow*4).
        const uint32_t baseByteOff = TQ_BR_KEY_BASE_BYTE_OFFSET + groupRow * sizeof(float);
        const float base = TqBrReadFloatFromU8(packedRaw, floatScratch, baseByteOff);
        kBase.SetValue(row, base);

        // Read step (float at byte offset 268 + groupRow*4).
        const uint32_t stepByteOff = TQ_BR_KEY_STEP_BYTE_OFFSET + groupRow * sizeof(float);
        const float step = TqBrReadFloatFromU8(packedRaw, floatScratch, stepByteOff);
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
    // Value cache: [num_blocks, num_kv_heads, (block_size/4) * 288]
    // Each 4-row group occupies 288 bytes.
    // uint16[d] = (idx4_row3 << 12) | (idx4_row2 << 8) | (idx4_row1 << 4) | idx4_row0

    const uint32_t valBlockStride = (blockSize_ / TQ_BR_VALUE_GROUP_ROWS) * TQ_BR_GROUP_STRIDE;
    const uint32_t seqBlockBase = seqIdx * maxBlocksPerSeq_;
    auto packedRaw = PackedRawBuf();
    auto codeI16 = CodeI16Buf();  // reuse for value idx4 extraction
    auto codeFloat = CodeFloatBuf();
    auto floatScratch = FloatScratchBuf();
    auto vminBuf = VminBuf();
    auto vstepBuf = VstepBuf();

    const uint32_t D = TQ_BR_HEAD_SIZE;

    auto valMask = MaskBuf();
    Duplicate(valMask, static_cast<uint16_t>(0x000F), D);
    PipeBarrier<PIPE_V>();

    for (uint32_t row = 0; row < mRows; ++row) {
        const uint32_t absPos = absStart + row;
        const uint32_t blockOffset = absPos / blockSize_;
        const uint32_t posInBlock = absPos % blockSize_;
        const uint32_t groupIdx = posInBlock / TQ_BR_VALUE_GROUP_ROWS;
        const uint32_t groupRow = posInBlock % TQ_BR_VALUE_GROUP_ROWS;

        TqBrSync<HardEvent::S_MTE2>();
        const int32_t blockId = blockTableGm_.GetValue(seqBlockBase + blockOffset);
        TqBrSync<HardEvent::MTE2_S>();

        const uint64_t groupBase = static_cast<uint64_t>(blockId) * numKvHeads_ * valBlockStride +
                                   static_cast<uint64_t>(kvHead) * valBlockStride +
                                   static_cast<uint64_t>(groupIdx) * TQ_BR_GROUP_STRIDE;

        DataCopyExtParams copyParams{1, TQ_BR_GROUP_STRIDE, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(packedRaw, valueCacheGm_[groupBase], copyParams, padParams);
        TqBrSync<HardEvent::MTE2_V>();

        auto groupI16 = packedRaw.template ReinterpretCast<int16_t>();

        // Extract 4-bit index for this group_row.
        // idx4 = (uint16 >> (groupRow * 4)) & 0x0F
        auto extractI16 = codeI16[row * D];
        auto extractU16 = extractI16.template ReinterpretCast<uint16_t>();
        ShiftRight(extractI16, groupI16, static_cast<int16_t>(groupRow * 4), D);
        PipeBarrier<PIPE_V>();

        And(extractU16, extractU16, valMask, D);
        PipeBarrier<PIPE_V>();

        // Cast extracted idx4 to float for decode.
        Cast(codeFloat[row * D], extractI16, RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();

        // Read vmin (float at byte offset 256 + groupRow*4).
        const uint32_t vminByteOff = TQ_BR_VAL_VMIN_BYTE_OFFSET + groupRow * sizeof(float);
        const float vmin = TqBrReadFloatFromU8(packedRaw, floatScratch, vminByteOff);
        vminBuf.SetValue(row, vmin);

        // Read vstep (float at byte offset 272 + groupRow*4).
        const uint32_t vstepByteOff = TQ_BR_VAL_VSTEP_BYTE_OFFSET + groupRow * sizeof(float);
        const float vstep = TqBrReadFloatFromU8(packedRaw, floatScratch, vstepByteOff);
        vstepBuf.SetValue(row, vstep);
    }

    // Restore key mask for next K tile load.
    auto mask = MaskBuf();
    Duplicate(mask, static_cast<uint16_t>(0x00FF), TQ_BR_GROUP_INDEX_WORDS);
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
    // kNorm[k], kBase[k], kStep[k] are per-row scalar floats.
    //
    // Decode: code = (q7 << 1) | sign
    //   sign = code & 1        → 0 or 1
    //   q7   = code >> 1       → [0, 127]
    //   sign_val = sign ? +INV_SQRT_D : -INV_SQRT_D
    //   err = base + q7 * step (broadcast per-row)
    //   y = sign_val + err     (in rotated space)
    //   decoded = y * norm     (broadcast per-row)

    const uint32_t D = TQ_BR_HEAD_SIZE;
    const uint32_t n = mRows * D;
    auto codeI16 = CodeI16Buf();
    auto codeFloat = CodeFloatBuf();
    auto decoded = DecodedBuf();
    auto kNorm = KNormBuf();
    auto kBase = KBaseBuf();
    auto kStep = KStepBuf();

    // Step 1: Extract sign (code & 1) and q7 (code >> 1).
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

    // Step 2: sign → sign_val = ±INV_SQRT_D
    // sign = 0 → sign_val = -INV_SQRT_D
    // sign = 1 → sign_val = +INV_SQRT_D
    // sign_val = (2*sign - 1) * INV_SQRT_D = sign * 2*INV_SQRT_D - INV_SQRT_D
    auto signF32Final = codeFloat;  // float tensor for sign_val
    Cast(signF32Final, signStorage, RoundMode::CAST_NONE, n);  // sign: 0.0 or 1.0
    PipeBarrier<PIPE_V>();
    Muls(signF32Final, signF32Final, 2.0f * TQ_BR_INV_SQRT_D, n);
    PipeBarrier<PIPE_V>();
    Adds(signF32Final, signF32Final, -TQ_BR_INV_SQRT_D, n);  // 0→-0.08839, 1→+0.08839
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

    // Step 4: y = sign_val + err
    Add(decoded, signF32Final, q7F32, n);
    PipeBarrier<PIPE_V>();

    // Step 5: y * norm (broadcast per-row)
    for (uint32_t row = 0; row < mRows; ++row) {
        const float norm = kNorm.GetValue(row);
        auto rowFloat = decoded[row * D];
        Muls(rowFloat, rowFloat, norm, D);
        PipeBarrier<PIPE_V>();
    }
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

    // Init mState and sState.
    auto mState = MStateBuf();
    auto sState = SStateBuf();
    auto outAcc = OutAccBuf();
    for (uint32_t g = 0; g < gqaCount; ++g) {
        mState.SetValue(g, -1e30f);  // -inf
        sState.SetValue(g, 0.f);
    }
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
        auto reduceScalar = KNormBuf();
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
        turboquant_attn::OnlineSoftmaxUpdateTileFloatPreScaled(
            scoreBuf, decodedV, mState, sState, outAcc, expBuf,
            softmaxReduceTmp, weightedValue,
            gqaCount, mRows);
    }

    // Normalize: outAcc /= sState
    for (uint32_t g = 0; g < gqaCount; ++g) {
        const float s = sState.GetValue(g);
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
