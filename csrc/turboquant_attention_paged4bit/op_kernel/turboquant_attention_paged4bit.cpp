/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// TurboQuant Scheme B: fused packed KV decode + paged attention.
// Supports DecodeOnly and ChunkedPrefill (mixed prefill/decode) via per-token
// causal KV bounds derived from actual_seq_len_q / actual_seq_len_kv.
//
// Tiling keys (host-selected):
//   0 SplitBN  + Vector QK/PV
//   1 SplitBN  + Cube QK/PV   (G >= 8, kvTile >= 32)
//   2 SplitBNS + Vector QK/PV (FlashDecode)
//   3 SplitBNS + Cube QK/PV

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#include "attention_device.h"
#include "decode_device.h"

namespace {

using namespace AscendC;

constexpr uint32_t TQ_HEAD = turboquant_attn::TQ_ATTN_HEAD;
constexpr uint32_t TQ_PACKED = turboquant::TQ_DECODE_PACKED_BYTES;
constexpr uint32_t TQ_CODEBOOK = turboquant::TQ_DECODE_CODEBOOK_SIZE;
constexpr uint32_t TQ_GROUP_ROWS = 4;
constexpr uint32_t TQ_GROUP_INDEX_BYTES = TQ_HEAD * sizeof(uint16_t);
constexpr uint32_t TQ_GROUP_INDEX_WORDS = TQ_HEAD;
constexpr uint32_t TQ_GROUP_BYTES = TQ_GROUP_ROWS * TQ_PACKED;
constexpr uint32_t TQ_GROUP_COPY_BYTES = ((TQ_GROUP_BYTES + 31U) / 32U) * 32U;
constexpr uint32_t TQ_AIV_SUB = 2;
constexpr uint32_t TQ_DTYPE_BYTES = sizeof(uint16_t);
constexpr uint32_t TQ_ROT_WS = TQ_HEAD * TQ_HEAD * TQ_DTYPE_BYTES;
// Host PickKvTileRows() caps at TQ_UB_KV_TILE_CAP; PA block_size may be 128.
// CANN GetWorkspaceSize static analysis uses InitBuffer sizes from the kernel
// binary; runtime tiling fields here must not be used as InitBuffer extents.
// Compile-time UB caps for InitBuffer; Mc2 workspace analysis sums these statically.
constexpr uint32_t TQ_UB_KV_TILE_CAP = 64;
constexpr uint32_t TQ_UB_GQA_CAP = 8;
constexpr uint32_t TQ_UB_Q_TILE_CAP = 4;
constexpr uint32_t TQ_UB_Q_TILE_GQA2_CAP = 16;
constexpr uint32_t TQ_UB_Q_TILE_SMALL_GQA_CAP = 2;

__aicore__ inline uint32_t AlignUp16(uint32_t x)
{
    return (x + 15U) / 16U * 16U;
}

__aicore__ inline uint32_t AlignUpGroupRows(uint32_t x)
{
    return (x + TQ_GROUP_ROWS - 1U) / TQ_GROUP_ROWS * TQ_GROUP_ROWS;
}

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}

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

using TqDataT = TqQueryT;
using TqRotateMm = turboquant::TqDecodeRotateMatmulOp<TqDataT>;

template <bool EnableQTile, uint32_t QTileCap, uint32_t QTileGqaCap>
struct TqQTileScratch {
    __aicore__ inline void InitQTileBuffers(TPipe*) {}
};

template <uint32_t QTileCap, uint32_t QTileGqaCap>
struct TqQTileScratch<true, QTileCap, QTileGqaCap> {
    __aicore__ inline void InitQTileBuffers(TPipe* pipe)
    {
        pipe->InitBuffer(qTileGroupFloatBuf_,
                         QTileCap * QTileGqaCap * TQ_HEAD * sizeof(float));
        pipe->InitBuffer(qTileScoreBuf_,
                         QTileCap * QTileGqaCap * TQ_UB_KV_TILE_CAP * sizeof(float));
        pipe->InitBuffer(qTileMStateBuf_,
                         QTileCap * QTileGqaCap * sizeof(float));
        pipe->InitBuffer(qTileSStateBuf_,
                         QTileCap * QTileGqaCap * sizeof(float));
        pipe->InitBuffer(qTileOutAccBuf_,
                         QTileCap * QTileGqaCap * TQ_HEAD * sizeof(float));
    }

    TBuf<TPosition::VECCALC> qTileGroupFloatBuf_;
    TBuf<TPosition::VECCALC> qTileScoreBuf_;
    TBuf<TPosition::VECCALC> qTileMStateBuf_;
    TBuf<TPosition::VECCALC> qTileSStateBuf_;
    TBuf<TPosition::VECCALC> qTileOutAccBuf_;
};

template <typename TilingT, typename QueryT, bool EnableQTile, uint32_t QTileCap,
          uint32_t QTileGqaCap>
class TurboquantAttentionPaged4bitKernel
    : private TqQTileScratch<EnableQTile, QTileCap, QTileGqaCap> {
public:
    __aicore__ inline TurboquantAttentionPaged4bitKernel(TPipe* pipe, TqRotateMm* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm)
    {}

    __aicore__ inline void Init(
        __gm__ QueryT* query,
        __gm__ uint8_t* keyCache,
        __gm__ uint8_t* valueCache,
        __gm__ int32_t* blockTable,
        __gm__ int64_t* actualSeqLenQ,
        __gm__ int64_t* actualSeqLenKv,
        __gm__ TqDataT* codebook,
        __gm__ TqDataT* rotation,
        __gm__ TqDataT* codebookValue,
        __gm__ TqDataT* rotationValue,
        __gm__ QueryT* out,
        __gm__ uint8_t* workspace,
        const TilingT* tiling)
    {
        tiling_ = tiling;
        numTokens_ = tiling_->numTokens;
        batchSize_ = tiling_->batchSize;
        numHeads_ = tiling_->numHeads;
        numKvHeads_ = tiling_->numKvHeads;
        gqaGroup_ = tiling_->gqaGroupSize;
        gqaChunkCount_ = CeilDiv(gqaGroup_, TQ_UB_GQA_CAP);
        headSize_ = tiling_->headSize;
        blockSize_ = tiling_->blockSize;
        maxBlocksPerSeq_ = tiling_->maxBlocksPerSeq;
        totalCacheBlocks_ = tiling_->totalCacheBlocks;
        kvTileRows_ = tiling_->kvTileRows;
        splitMode_ = tiling_->splitMode;
        kvSplitPart_ = tiling_->kvSplitPart;
        qkPvMode_ = tiling_->qkPvMode;
        kvSegmentLen_ = tiling_->kvSegmentLen;
        scaleValue_ = tiling_->scaleValue;
        usedCoreNum_ = tiling_->usedCoreNum;

        queryGm_.SetGlobalBuffer(query, static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);
        const uint64_t cacheBytes = static_cast<uint64_t>(totalCacheBlocks_) * blockSize_ *
                                    numKvHeads_ * TQ_PACKED;
        keyCacheGm_.SetGlobalBuffer(keyCache, cacheBytes);
        valueCacheGm_.SetGlobalBuffer(valueCache, cacheBytes);
        blockTableGm_.SetGlobalBuffer(blockTable,
                                      static_cast<uint64_t>(batchSize_) * maxBlocksPerSeq_);
        actualSeqLenQGm_.SetGlobalBuffer(actualSeqLenQ, batchSize_);
        actualSeqLenKvGm_.SetGlobalBuffer(actualSeqLenKv, batchSize_);
        codebookGm_.SetGlobalBuffer(codebook, TQ_CODEBOOK);
        codebookVGm_.SetGlobalBuffer(codebookValue, TQ_CODEBOOK);
        // rotation is pack-side R^T for Q pre-rotation; rotationValue is
        // decode-side R for the final output rotation.
        rotationGm_.SetGlobalBuffer(rotation, static_cast<uint64_t>(TQ_HEAD) * TQ_HEAD);
        rotationVGm_.SetGlobalBuffer(rotationValue, static_cast<uint64_t>(TQ_HEAD) * TQ_HEAD);
        outGm_.SetGlobalBuffer(out, static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);

        pipe_->InitBuffer(codebookBuf_, TQ_CODEBOOK * sizeof(TqDataT));
        pipe_->InitBuffer(codebookVBuf_, TQ_CODEBOOK * sizeof(TqDataT));
        // Compact decode layout: [M*128 unpacked uint4 idx bytes][M*2 norm bytes].
        pipe_->InitBuffer(packedBuf_,
                          TQ_UB_KV_TILE_CAP * TQ_HEAD +
                              TQ_UB_KV_TILE_CAP * TQ_DTYPE_BYTES);
        pipe_->InitBuffer(xHatQue_, 1, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(TqDataT));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_WS);
        pipe_->InitBuffer(expBuf_, TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(qGroupBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(QueryT));
        pipe_->InitBuffer(qGroupFloatBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(outFloatBuf_, TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, TQ_UB_GQA_CAP * TQ_UB_KV_TILE_CAP * sizeof(float));
        pipe_->InitBuffer(kNormBuf_, TQ_UB_KV_TILE_CAP * sizeof(float));
        pipe_->InitBuffer(mStateBuf_, TQ_UB_GQA_CAP * sizeof(float));
        pipe_->InitBuffer(sStateBuf_, TQ_UB_GQA_CAP * sizeof(float));
        pipe_->InitBuffer(outAccBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        this->InitQTileBuffers(pipe_);
        if (splitMode_ == 1) {
            pipe_->InitBuffer(combineOutBuf_, TQ_HEAD * sizeof(float));
            pipe_->InitBuffer(partialOutBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        }
        pipe_->InitBuffer(idxFloatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(idxS32Buf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(int32_t));
        pipe_->InitBuffer(attnTmpBuf_, TQ_HEAD * 2 * sizeof(float));
        pipe_->InitBuffer(yHatQue_, 1, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(TqDataT));
        pipe_->InitBuffer(packedRawBuf_, TQ_GROUP_COPY_BYTES * sizeof(uint8_t));
        pipe_->InitBuffer(packedMaskBuf_, TQ_GROUP_INDEX_BYTES * sizeof(uint8_t));

        auto* wsBase = reinterpret_cast<__gm__ uint8_t*>(GetSysWorkSpacePtr());
        matmulReady_ = (wsBase != nullptr);
        if (splitMode_ == 1 && workspace != nullptr) {
            partialWorkspace_ = workspace + tiling_->partialWorkspaceOffset;
            const size_t accumBytes =
                static_cast<size_t>(tiling_->accumOutSize) * sizeof(float);
            const size_t lseBytes =
                static_cast<size_t>(tiling_->logSumExpSize) * sizeof(float);
            partialOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partialWorkspace_),
                                          accumBytes / sizeof(float));
            partialLseGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(partialWorkspace_ + accumBytes),
                lseBytes / sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (splitMode_ == 1) {
            // SplitBNS + FlashDecode: every launched MIX core (AIC + both AIV
            // sub-blocks, including idle ones) must enter SyncAll exactly once.
            const uint32_t coreIdx = GetBlockIdx() / TQ_AIV_SUB;
            if ASCEND_IS_AIC {
                SyncAll();
                return;
            }
            const bool isPrimaryAiv = ((GetSubBlockIdx() % TQ_AIV_SUB) == 0);
            if (isPrimaryAiv && coreIdx < usedCoreNum_) {
                InitPackedMask();
                ProcessSplitBns(coreIdx);
            }
            SyncAll();
            if (isPrimaryAiv && coreIdx == 0) {
                CombineFlashDecode();
            }
            return;
        }

        if ASCEND_IS_AIC {
            return;
        }
        // Match the validated v2 pack KFC path: on 910B each AIV worker is
        // addressed by blockIdx directly. Let all launched AIV workers enter the
        // KFC matmul path instead of keeping one sub-block idle.
        const uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }

        InitPackedMask();
        ProcessSplitBn(coreIdx);
    }

private:
    __aicore__ inline void InitPackedMask()
    {
        auto lowMask = packedMaskBuf_.Get<uint16_t>();
        AscendC::Duplicate(lowMask, static_cast<uint16_t>(0x000F), TQ_GROUP_INDEX_WORDS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadCodebook()
    {
        auto codebookLocal = codebookBuf_.Get<TqDataT>();
        auto codebookVLocal = codebookVBuf_.Get<TqDataT>();
        DataCopy(codebookLocal, codebookGm_, TQ_CODEBOOK);
        DataCopy(codebookVLocal, codebookVGm_, TQ_CODEBOOK);
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline uint32_t TaskStart(uint32_t coreIdx) const
    {
        if (splitMode_ == 1) {
            return coreIdx;
        }
        const uint32_t former = tiling_->formerCoreNum;
        const uint32_t blockRange = tiling_->blockSplitRange;
        const uint32_t tailRange = tiling_->tailSplitRange;
        if (coreIdx < former) {
            return blockRange * coreIdx;
        }
        return former * blockRange + tailRange * (coreIdx - former);
    }

    __aicore__ inline uint32_t TaskEnd(uint32_t coreIdx) const
    {
        if (coreIdx + 1 >= usedCoreNum_) {
            const uint32_t taskScale = numKvHeads_ * gqaChunkCount_;
            return (splitMode_ == 1) ? (numTokens_ * taskScale * kvSplitPart_)
                                     : (numTokens_ * taskScale);
        }
        return TaskStart(coreIdx + 1);
    }

    __aicore__ inline bool SplitBnTaskInRange(
        uint32_t localToken,
        uint32_t kvHead,
        uint32_t gqaChunkIdx,
        uint32_t relStart,
        uint32_t relEnd) const
    {
        const uint32_t taskScale = numKvHeads_ * gqaChunkCount_;
        const uint32_t task = localToken * taskScale + kvHead * gqaChunkCount_ + gqaChunkIdx;
        return task >= relStart && task < relEnd;
    }

    __aicore__ inline uint32_t GqaChunkStart(uint32_t gqaChunkIdx) const
    {
        return gqaChunkIdx * TQ_UB_GQA_CAP;
    }

    __aicore__ inline uint32_t GqaChunkCount(uint32_t gqaChunkStart) const
    {
        const uint32_t remain = gqaGroup_ - gqaChunkStart;
        return (remain > TQ_UB_GQA_CAP) ? TQ_UB_GQA_CAP : remain;
    }

    __aicore__ inline void ProcessSplitBnScalarRange(
        uint32_t seqIdx,
        uint32_t qStart,
        uint32_t qEnd,
        uint32_t kvLen,
        uint32_t relStart,
        uint32_t relEnd,
        uint32_t localTokenStart,
        uint32_t localTokenEnd)
    {
        const uint32_t taskScale = numKvHeads_ * gqaChunkCount_;
        for (uint32_t localToken = localTokenStart;
             localToken < localTokenEnd; ++localToken) {
            const uint32_t tokenTaskStart = localToken * taskScale;
            const uint32_t headChunkStart =
                (relStart > tokenTaskStart) ? (relStart - tokenTaskStart) : 0U;
            uint32_t headChunkEnd = relEnd - tokenTaskStart;
            if (headChunkEnd > taskScale) {
                headChunkEnd = taskScale;
            }
            const uint32_t tokenIdx = qStart + localToken;
            for (uint32_t headChunk = headChunkStart; headChunk < headChunkEnd; ++headChunk) {
                const uint32_t kvHead = headChunk / gqaChunkCount_;
                const uint32_t gqaChunkIdx = headChunk - kvHead * gqaChunkCount_;
                const uint32_t gqaStart = GqaChunkStart(gqaChunkIdx);
                const uint32_t gqaCount = GqaChunkCount(gqaStart);
                ComputeAttention(tokenIdx, kvHead, seqIdx, qStart, qEnd, kvLen,
                                 gqaStart, gqaCount, 0, kvSplitPart_, false);
            }
        }
    }

    __aicore__ inline void ProcessSplitBn(uint32_t coreIdx)
    {
        const uint32_t start = TaskStart(coreIdx);
        const uint32_t end = TaskEnd(coreIdx);
        uint32_t qStart = 0;
        const uint32_t taskScale = numKvHeads_ * gqaChunkCount_;
        for (uint32_t seqIdx = 0; seqIdx < batchSize_; ++seqIdx) {
            const uint32_t qEnd = static_cast<uint32_t>(actualSeqLenQGm_.GetValue(seqIdx));
            const uint32_t kvLen = static_cast<uint32_t>(actualSeqLenKvGm_.GetValue(seqIdx));
            const uint32_t seqTaskStart = qStart * taskScale;
            const uint32_t seqTaskEnd = qEnd * taskScale;
            const uint32_t overlapStart = (start > seqTaskStart) ? start : seqTaskStart;
            const uint32_t overlapEnd = (end < seqTaskEnd) ? end : seqTaskEnd;
            if (overlapStart < overlapEnd) {
                const uint32_t relStart = overlapStart - seqTaskStart;
                const uint32_t relEnd = overlapEnd - seqTaskStart;
                const uint32_t localTokenStart = relStart / taskScale;
                const uint32_t localTokenEnd = CeilDiv(relEnd, taskScale);
                if constexpr (!EnableQTile) {
                    ProcessSplitBnScalarRange(seqIdx, qStart, qEnd, kvLen, relStart, relEnd,
                                              localTokenStart, localTokenEnd);
                } else {
                    if (relEnd - relStart <= taskScale) {
                        ProcessSplitBnScalarRange(seqIdx, qStart, qEnd, kvLen, relStart, relEnd,
                                                  localTokenStart, localTokenEnd);
                    } else {
                        for (uint32_t kvHead = 0; kvHead < numKvHeads_; ++kvHead) {
                            for (uint32_t gqaChunkIdx = 0; gqaChunkIdx < gqaChunkCount_;
                                 ++gqaChunkIdx) {
                                uint32_t localToken = localTokenStart;
                                while (localToken < localTokenEnd) {
                                    while (localToken < localTokenEnd &&
                                           !SplitBnTaskInRange(localToken, kvHead, gqaChunkIdx,
                                                               relStart, relEnd)) {
                                        ++localToken;
                                    }
                                    if (localToken >= localTokenEnd) {
                                        break;
                                    }
                                    const uint32_t tileStart = localToken;
                                    uint32_t tileEnd = tileStart + 1;
                                    while (tileEnd < localTokenEnd &&
                                           tileEnd - tileStart < QTileCap &&
                                           SplitBnTaskInRange(tileEnd, kvHead, gqaChunkIdx,
                                                              relStart, relEnd)) {
                                        ++tileEnd;
                                    }

                                    const uint32_t gqaStart = GqaChunkStart(gqaChunkIdx);
                                    const uint32_t gqaCount = GqaChunkCount(gqaStart);
                                    const uint32_t tokenIdx = qStart + tileStart;
                                    const uint32_t qRows = tileEnd - tileStart;
                                    if (qRows > 1) {
                                        ComputeAttentionQTile(tokenIdx, qRows, kvHead, seqIdx,
                                                              qStart, qEnd, kvLen, gqaStart,
                                                              gqaCount);
                                    } else {
                                        ComputeAttention(tokenIdx, kvHead, seqIdx, qStart, qEnd,
                                                         kvLen, gqaStart, gqaCount, 0,
                                                         kvSplitPart_, false);
                                    }
                                    localToken = tileEnd;
                                }
                            }
                        }
                    }
                }
            }
            qStart = qEnd;
        }
    }

    __aicore__ inline void ProcessSplitBns(uint32_t coreIdx)
    {
        const uint32_t start = TaskStart(coreIdx);
        const uint32_t end = TaskEnd(coreIdx);
        uint32_t qStart = 0;
        const uint32_t headChunkScale = numKvHeads_ * gqaChunkCount_;
        const uint32_t seqTaskScale = headChunkScale * kvSplitPart_;
        for (uint32_t seqIdx = 0; seqIdx < batchSize_; ++seqIdx) {
            const uint32_t qEnd = static_cast<uint32_t>(actualSeqLenQGm_.GetValue(seqIdx));
            const uint32_t kvLen = static_cast<uint32_t>(actualSeqLenKvGm_.GetValue(seqIdx));
            const uint32_t seqTaskStart = qStart * seqTaskScale;
            const uint32_t seqTaskEnd = qEnd * seqTaskScale;
            const uint32_t overlapStart = (start > seqTaskStart) ? start : seqTaskStart;
            const uint32_t overlapEnd = (end < seqTaskEnd) ? end : seqTaskEnd;
            if (overlapStart < overlapEnd) {
                const uint32_t relStart = overlapStart - seqTaskStart;
                const uint32_t relEnd = overlapEnd - seqTaskStart;
                const uint32_t localTokenStart = relStart / seqTaskScale;
                const uint32_t localTokenEnd = CeilDiv(relEnd, seqTaskScale);
                for (uint32_t localToken = localTokenStart;
                     localToken < localTokenEnd; ++localToken) {
                    const uint32_t tokenTaskStart = localToken * seqTaskScale;
                    const uint32_t tokenRelStart =
                        (relStart > tokenTaskStart) ? (relStart - tokenTaskStart) : 0U;
                    uint32_t tokenRelEnd = relEnd - tokenTaskStart;
                    if (tokenRelEnd > seqTaskScale) {
                        tokenRelEnd = seqTaskScale;
                    }
                    const uint32_t tokenIdx = qStart + localToken;
                    for (uint32_t headSegTask = tokenRelStart; headSegTask < tokenRelEnd;
                         ++headSegTask) {
                        const uint32_t headChunkTask = headSegTask / kvSplitPart_;
                        const uint32_t segIdx = headSegTask - headChunkTask * kvSplitPart_;
                        const uint32_t kvHead = headChunkTask / gqaChunkCount_;
                        const uint32_t gqaChunkIdx =
                            headChunkTask - kvHead * gqaChunkCount_;
                        const uint32_t gqaStart = GqaChunkStart(gqaChunkIdx);
                        const uint32_t gqaCount = GqaChunkCount(gqaStart);
                        ComputeAttention(tokenIdx, kvHead, seqIdx, qStart, qEnd, kvLen,
                                         gqaStart, gqaCount, segIdx, kvSplitPart_, true);
                    }
                }
            }
            qStart = qEnd;
        }
    }

    __aicore__ inline void LoadPackedTileRows(
        GlobalTensor<uint8_t>& cache,
        const LocalTensor<uint8_t>& packed,
        uint32_t mRows,
        uint32_t seqIdx,
        uint32_t kvHead,
        uint32_t kvStart)
    {
        auto rawGroup = packedRawBuf_.Get<uint8_t>();
        auto rawGroupU16 = rawGroup.template ReinterpretCast<uint16_t>();
        auto rawGroupI16 = rawGroup.template ReinterpretCast<int16_t>();
        auto packedNormU16 = packed.template ReinterpretCast<uint16_t>();
        auto extractI16 = idxS32Buf_.Get<int16_t>();
        auto extractU16 = extractI16.template ReinterpretCast<uint16_t>();
        auto idxFloat = idxFloatBuf_.Get<float>();
        auto lowMask = packedMaskBuf_.Get<uint16_t>();
        AscendC::DataCopyExtParams copyParams{1, TQ_GROUP_BYTES, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};

        const uint64_t seqBlockBase = static_cast<uint64_t>(seqIdx) * maxBlocksPerSeq_;
        const uint64_t blockStride = static_cast<uint64_t>(blockSize_) * TQ_PACKED;
        const uint32_t firstBlockOffset = kvStart / blockSize_;
        const uint32_t lastBlockOffset = (kvStart + mRows - 1U) / blockSize_;
        if (firstBlockOffset == lastBlockOffset) {
            const uint32_t blockId = static_cast<uint32_t>(
                blockTableGm_.GetValue(seqBlockBase + firstBlockOffset));
            const uint64_t blockBase =
                (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * blockStride;
            const uint32_t basePosInBlock = kvStart - firstBlockOffset * blockSize_;
            if ((basePosInBlock % TQ_GROUP_ROWS) == 0 && (mRows % TQ_GROUP_ROWS) == 0) {
                const uint32_t firstGroupInBlock = basePosInBlock / TQ_GROUP_ROWS;
                const uint32_t groupCount = mRows / TQ_GROUP_ROWS;
                const uint64_t srcBase =
                    blockBase + static_cast<uint64_t>(firstGroupInBlock) * TQ_GROUP_BYTES;
                auto rawTile = packed;
                for (uint32_t copyGroup = 0; copyGroup < groupCount; ++copyGroup) {
                    AscendC::DataCopyPad(rawTile[copyGroup * TQ_GROUP_COPY_BYTES],
                                         cache[srcBase + static_cast<uint64_t>(copyGroup) *
                                                          TQ_GROUP_BYTES],
                                         copyParams, padParams);
                }
                turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();

                for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
                    auto groupRaw = rawTile[groupIdx * TQ_GROUP_COPY_BYTES];
                    auto groupU16 = groupRaw.template ReinterpretCast<uint16_t>();
                    auto groupI16 = groupRaw.template ReinterpretCast<int16_t>();
                    for (uint32_t groupRow = 0; groupRow < TQ_GROUP_ROWS; ++groupRow) {
                        const uint32_t dstRow = groupIdx * TQ_GROUP_ROWS + groupRow;
                        const uint32_t dstBase = dstRow * TQ_HEAD;
                        if (groupRow == 0) {
                            AscendC::And(extractU16, groupU16, lowMask, TQ_GROUP_INDEX_WORDS);
                        } else {
                            AscendC::ShiftRight(
                                extractI16, groupI16,
                                static_cast<int16_t>(groupRow * 4), TQ_GROUP_INDEX_WORDS);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::And(extractU16, extractU16, lowMask, TQ_GROUP_INDEX_WORDS);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(idxFloat[dstBase], extractI16,
                                      AscendC::RoundMode::CAST_NONE, TQ_HEAD);
                        AscendC::PipeBarrier<PIPE_V>();

                        const uint32_t dstNorm = (mRows * TQ_HEAD) / TQ_DTYPE_BYTES + dstRow;
                        const uint32_t srcNorm = TQ_GROUP_INDEX_BYTES / TQ_DTYPE_BYTES + groupRow;
                        packedNormU16.SetValue(dstNorm, groupU16.GetValue(srcNorm));
                    }
                }
                turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
                return;
            }

            for (uint32_t t = 0; t < mRows;) {
                const uint32_t posInBlock = basePosInBlock + t;
                const uint32_t groupInBlock = posInBlock / TQ_GROUP_ROWS;
                const uint32_t firstGroupRow = posInBlock - groupInBlock * TQ_GROUP_ROWS;
                uint32_t rowsInGroup = TQ_GROUP_ROWS - firstGroupRow;
                if (rowsInGroup > mRows - t) {
                    rowsInGroup = mRows - t;
                }
                const uint64_t srcOff =
                    blockBase + static_cast<uint64_t>(groupInBlock) * TQ_GROUP_BYTES;

                AscendC::DataCopyPad(rawGroup, cache[srcOff], copyParams, padParams);
                turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();

                for (uint32_t localRow = 0; localRow < rowsInGroup; ++localRow) {
                    const uint32_t groupRow = firstGroupRow + localRow;
                    const uint32_t dstRow = t + localRow;
                    const uint32_t dstBase = dstRow * TQ_HEAD;
                    if (groupRow == 0) {
                        AscendC::And(extractU16, rawGroupU16, lowMask, TQ_GROUP_INDEX_WORDS);
                    } else {
                        AscendC::ShiftRight(
                            extractI16, rawGroupI16,
                            static_cast<int16_t>(groupRow * 4), TQ_GROUP_INDEX_WORDS);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::And(extractU16, extractU16, lowMask, TQ_GROUP_INDEX_WORDS);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(idxFloat[dstBase], extractI16,
                                  AscendC::RoundMode::CAST_NONE, TQ_HEAD);
                    AscendC::PipeBarrier<PIPE_V>();

                    const uint32_t dstNorm = (mRows * TQ_HEAD) / TQ_DTYPE_BYTES + dstRow;
                    const uint32_t srcNorm = TQ_GROUP_INDEX_BYTES / TQ_DTYPE_BYTES + groupRow;
                    packedNormU16.SetValue(dstNorm, rawGroupU16.GetValue(srcNorm));
                }
                t += rowsInGroup;
            }
            turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
            return;
        }

        for (uint32_t t = 0; t < mRows;) {
            const uint32_t absPos = kvStart + t;
            const uint32_t blockOffset = absPos / blockSize_;
            const uint32_t blockId = static_cast<uint32_t>(
                blockTableGm_.GetValue(seqBlockBase + blockOffset));
            const uint64_t blockBase =
                (static_cast<uint64_t>(blockId) * numKvHeads_ + kvHead) * blockStride;
            const uint32_t posInBlock = absPos - blockOffset * blockSize_;
            const uint32_t groupInBlock = posInBlock / TQ_GROUP_ROWS;
            const uint32_t firstGroupRow = posInBlock - groupInBlock * TQ_GROUP_ROWS;
            uint32_t rowsInGroup = TQ_GROUP_ROWS - firstGroupRow;
            if (rowsInGroup > mRows - t) {
                rowsInGroup = mRows - t;
            }
            const uint64_t srcOff =
                blockBase + static_cast<uint64_t>(groupInBlock) * TQ_GROUP_BYTES;

            AscendC::DataCopyPad(rawGroup, cache[srcOff], copyParams, padParams);
            turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();

            for (uint32_t localRow = 0; localRow < rowsInGroup; ++localRow) {
                const uint32_t groupRow = firstGroupRow + localRow;
                const uint32_t dstRow = t + localRow;
                const uint32_t dstBase = dstRow * TQ_HEAD;
                if (groupRow == 0) {
                    AscendC::And(extractU16, rawGroupU16, lowMask, TQ_GROUP_INDEX_WORDS);
                } else {
                    AscendC::ShiftRight(
                        extractI16, rawGroupI16,
                        static_cast<int16_t>(groupRow * 4), TQ_GROUP_INDEX_WORDS);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::And(extractU16, extractU16, lowMask, TQ_GROUP_INDEX_WORDS);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(idxFloat[dstBase], extractI16,
                              AscendC::RoundMode::CAST_NONE, TQ_HEAD);
                AscendC::PipeBarrier<PIPE_V>();

                const uint32_t dstNorm = (mRows * TQ_HEAD) / TQ_DTYPE_BYTES + dstRow;
                const uint32_t srcNorm = TQ_GROUP_INDEX_BYTES / TQ_DTYPE_BYTES + groupRow;
                packedNormU16.SetValue(dstNorm, rawGroupU16.GetValue(srcNorm));
            }
            t += rowsInGroup;
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void LoadRotation(GlobalTensor<TqDataT>& rotationGm)
    {
        auto rotation = rotateWorkBuf_.Get<TqDataT>();
        DataCopy(rotation, rotationGm, TQ_HEAD * TQ_HEAD);
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_S>();
    }

    __aicore__ inline void RotateRowsScalar(
        LocalTensor<float> rowsFloat,
        GlobalTensor<TqDataT>& rotationGm,
        uint32_t rowCount)
    {
        LoadRotation(rotationGm);
        auto rotation = rotateWorkBuf_.Get<TqDataT>();
        auto rotationRowFloat = idxFloatBuf_.Get<float>();
        for (uint32_t row = 0; row < rowCount; ++row) {
            const uint32_t rowBase = row * TQ_HEAD;
            float acc[TQ_HEAD];
            for (uint32_t n = 0; n < TQ_HEAD; ++n) {
                acc[n] = 0.f;
            }
            for (uint32_t k = 0; k < TQ_HEAD; ++k) {
                const float a = rowsFloat.GetValue(rowBase + k);
                AscendC::Cast(rotationRowFloat, rotation[k * TQ_HEAD],
                              AscendC::RoundMode::CAST_NONE, TQ_HEAD);
                AscendC::PipeBarrier<PIPE_V>();
                turboquant::TqDecodeSync<AscendC::HardEvent::V_S>();
                for (uint32_t n = 0; n < TQ_HEAD; ++n) {
                    acc[n] += a * rotationRowFloat.GetValue(n);
                }
                turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
            }
            for (uint32_t n = 0; n < TQ_HEAD; ++n) {
                rowsFloat.SetValue(rowBase + n, acc[n]);
            }
            turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
        }
    }

    __aicore__ inline void RotateRowsInPlace(
        LocalTensor<float> rowsFloat,
        GlobalTensor<TqDataT>& rotationGm,
        uint32_t rowCount)
    {
        if (rowCount == 0) {
            return;
        }
        if (!matmulReady_ || rotateMm_ == nullptr) {
            RotateRowsScalar(rowsFloat, rotationGm, rowCount);
            return;
        }

        const uint32_t n = rowCount * TQ_HEAD;
        const uint32_t mPad = AlignUp16(rowCount);
        auto input = yHatQue_.AllocTensor<TqDataT>();
        auto output = xHatQue_.AllocTensor<TqDataT>();
        AscendC::Cast(input, rowsFloat, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        if (mPad > rowCount) {
            turboquant::TqDecodeDuplicateZero(input[n], (mPad - rowCount) * TQ_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }

        yHatQue_.EnQue(input);
        auto inputReady = yHatQue_.DeQue<TqDataT>();
        rotateMm_->SetOrgShape(mPad, TQ_HEAD, TQ_HEAD);
        rotateMm_->SetSingleShape(rowCount, TQ_HEAD, TQ_HEAD);
        rotateMm_->SetTensorA(inputReady, false);
        rotateMm_->SetTensorB(rotationGm, false);
        rotateMm_->SetLocalWorkspace(rotateWorkBuf_.Get<uint8_t>());
        rotateMm_->IterateAll(output);
        rotateMm_->End();
        yHatQue_.FreeTensor(inputReady);

        AscendC::Cast(rowsFloat, output, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_V>();
        xHatQue_.FreeTensor(output);
    }

    template <bool RESTORE_NORM>
    __aicore__ inline LocalTensor<TqDataT> DecodeTile(
        LocalTensor<uint8_t> packedLocal,
        LocalTensor<TqDataT> codebookLocal,
        LocalTensor<float> normOut,
        uint32_t mRows)
    {
        auto xHat = xHatQue_.AllocTensor<TqDataT>();
        turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
        turboquant::DecodeRows4bitFromIdxFloatNoRotate<TqDataT, RESTORE_NORM>(
            packedLocal, codebookLocal,
            idxFloatBuf_.Get<float>(), idxS32Buf_.Get<int32_t>(),
            rotateWorkBuf_.Get<uint8_t>(), normOut, xHat, mRows);
        turboquant::TqDecodeSync<AscendC::HardEvent::V_S>();
        xHatQue_.EnQue(xHat);
        return xHatQue_.DeQue<TqDataT>();
    }

    __aicore__ inline void FreeDecodeTile(LocalTensor<TqDataT> xHat)
    {
        xHatQue_.FreeTensor(xHat);
    }

    __aicore__ inline void DecodeTileToFloat(
        LocalTensor<uint8_t> packedLocal,
        LocalTensor<TqDataT> codebookLocal,
        LocalTensor<float> normOut,
        LocalTensor<float> tileFloat,
        uint32_t mRows,
        bool syncNormToScalar)
    {
        (void)codebookLocal;
        const uint32_t n = mRows * TQ_HEAD;
        turboquant::TqDecodeFyVectorFloatInPlace(
            tileFloat, idxS32Buf_.Get<int32_t>(), n);

        const uint32_t normBase = mRows * TQ_HEAD;
        auto normLocal = packedLocal[normBase].template ReinterpretCast<TqDataT>();
        AscendC::Cast(normOut, normLocal, AscendC::RoundMode::CAST_NONE, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        if (syncNormToScalar) {
            turboquant::TqDecodeSync<AscendC::HardEvent::V_S>();
        }
    }

    __aicore__ inline void CastTileToFloat(
        LocalTensor<float> dst, const LocalTensor<TqDataT>& src, uint32_t mRows)
    {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, mRows * TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadQGroup(
        uint32_t tokenIdx,
        uint32_t kvHead,
        uint32_t gqaStart,
        uint32_t gqaCount,
        LocalTensor<float> qGroup)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_ + gqaStart;
        auto qLocal = qGroupBuf_.Get<QueryT>();
        const uint64_t off =
            (static_cast<uint64_t>(tokenIdx) * numHeads_ + qBaseHead) * TQ_HEAD;
        DataCopy(qLocal, queryGm_[off], gqaCount * TQ_HEAD);
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(qGroup, qLocal, AscendC::RoundMode::CAST_NONE, gqaCount * TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutputRow(uint64_t off, LocalTensor<QueryT> outLocal)
    {
        DataCopy(outGm_[off], outLocal, TQ_HEAD);
    }

    __aicore__ inline uint32_t GetCausalKvEnd(
        uint32_t tokenIdx,
        uint32_t qChunkStart,
        uint32_t qChunkEnd,
        uint32_t kvLen) const
    {
        const uint32_t numQInChunk = qChunkEnd - qChunkStart;
        const uint32_t qPosInChunk = tokenIdx - qChunkStart;
        const uint32_t absQPos = kvLen - numQInChunk + qPosInChunk;
        return absQPos + 1U;
    }

    __aicore__ inline uint32_t QTileStateOffset(uint32_t q) const
    {
        return q * QTileGqaCap;
    }

    __aicore__ inline uint32_t QTileCompactStateOffset(uint32_t q, uint32_t gqaCount) const
    {
        return q * gqaCount;
    }

    __aicore__ inline uint32_t QTileGroupOffset(uint32_t q) const
    {
        return q * QTileGqaCap * TQ_HEAD;
    }

    __aicore__ inline uint32_t QTileCompactGroupOffset(uint32_t q, uint32_t gqaCount) const
    {
        return q * gqaCount * TQ_HEAD;
    }

    __aicore__ inline uint32_t QTileScoreOffset(uint32_t q) const
    {
        return q * QTileGqaCap * TQ_UB_KV_TILE_CAP;
    }

    __aicore__ inline uint32_t QTileCompactScoreOffset(uint32_t q, uint32_t gqaCount) const
    {
        return q * gqaCount * turboquant_attn::TQ_ATTN_TILE_STRIDE;
    }

    __aicore__ inline void WriteFinalOutputQTile(
        uint32_t tokenStartIdx,
        uint32_t qRows,
        uint32_t kvHead,
        float* sStateScalar,
        LocalTensor<float> outAcc,
        uint32_t gqaStart,
        uint32_t gqaCount)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_ + gqaStart;
        const uint32_t compactRows = qRows * gqaCount;
        for (uint32_t q = 0; q < qRows; ++q) {
            const uint32_t stateBase = QTileCompactStateOffset(q, gqaCount);
            const uint32_t outBaseQ = QTileCompactGroupOffset(q, gqaCount);
            for (uint32_t g = 0; g < gqaCount; ++g) {
                const float sum = sStateScalar[stateBase + g];
                const float invS = (sum > 0.f) ? (1.f / sum) : 0.f;
                const uint32_t outBase = outBaseQ + g * TQ_HEAD;
                AscendC::Muls(outAcc[outBase], outAcc[outBase], invS, TQ_HEAD);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        RotateRowsInPlace(outAcc, rotationVGm_, compactRows);

        auto outLocal = qGroupBuf_.Get<QueryT>();
        for (uint32_t q = 0; q < qRows; ++q) {
            AscendC::Cast(outLocal, outAcc[QTileCompactGroupOffset(q, gqaCount)],
                          AscendC::RoundMode::CAST_RINT, gqaCount * TQ_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            turboquant::TqDecodeSync<AscendC::HardEvent::V_MTE3>();
            for (uint32_t g = 0; g < gqaCount; ++g) {
                const uint64_t off =
                    (static_cast<uint64_t>(tokenStartIdx + q) * numHeads_ +
                     qBaseHead + g) *
                    TQ_HEAD;
                CopyOutputRow(off, outLocal[g * TQ_HEAD]);
            }
            turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_V>();
            turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void ComputeAttentionQTile(
        uint32_t tokenStartIdx,
        uint32_t qRows,
        uint32_t kvHead,
        uint32_t seqIdx,
        uint32_t qChunkStart,
        uint32_t qChunkEnd,
        uint32_t kvLen,
        uint32_t gqaStart,
        uint32_t gqaCount)
    {
        const uint32_t maxCausalKvEnd =
            GetCausalKvEnd(tokenStartIdx + qRows - 1, qChunkStart, qChunkEnd, kvLen);
        if (kvLen == 0 || maxCausalKvEnd == 0) {
            for (uint32_t q = 0; q < qRows; ++q) {
                WriteZeroOutput(tokenStartIdx + q, kvHead, gqaStart, gqaCount);
            }
            return;
        }

        auto qTile = this->qTileGroupFloatBuf_.template Get<float>();
        for (uint32_t q = 0; q < qRows; ++q) {
            auto qGroup = qTile[QTileCompactGroupOffset(q, gqaCount)];
            LoadQGroup(tokenStartIdx + q, kvHead, gqaStart, gqaCount, qGroup);
        }
        RotateRowsInPlace(qTile, rotationGm_, qRows * gqaCount);

        // m/s states as Scalar arrays — eliminates GetValue/SetValue V↔S sync.
        const uint32_t compactStateSize = qRows * gqaCount;
        float mStateScalar[QTileCap * QTileGqaCap];
        float sStateScalar[QTileCap * QTileGqaCap];
        for (uint32_t i = 0; i < compactStateSize; ++i) {
            mStateScalar[i] = -3.402823466e+38f;
            sStateScalar[i] = 0.f;
        }
        auto outAccTile = this->qTileOutAccBuf_.template Get<float>();
        AscendC::Duplicate(outAccTile, 0.f, qRows * gqaCount * TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();

        auto packedLocal = packedBuf_.Get<uint8_t>();
        auto codebookLocal = codebookBuf_.Get<TqDataT>();
        auto codebookVLocal = codebookVBuf_.Get<TqDataT>();
        auto expLocal = expBuf_.Get<float>();
        auto attnTmp0 = attnTmpBuf_.Get<float>();
        auto attnTmp1 = attnTmp0[TQ_HEAD];
        auto kNorm = kNormBuf_.Get<float>();
        auto tileFloat = idxFloatBuf_.Get<float>();
        auto qTileScore = this->qTileScoreBuf_.template Get<float>();
        const bool preScaleTile = gqaCount >= 8U;
        if (!preScaleTile) {
            AscendC::Muls(qTile, qTile, scaleValue_, qRows * gqaCount * TQ_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t pos = 0; pos < maxCausalKvEnd;) {
            const uint32_t computeRows =
                (pos + kvTileRows_ <= maxCausalKvEnd) ? kvTileRows_ : (maxCausalKvEnd - pos);
            const uint32_t loadRows = AlignUpGroupRows(computeRows);
            LoadPackedTileRows(keyCacheGm_, packedLocal, loadRows, seqIdx, kvHead, pos);
            DecodeTileToFloat(packedLocal, codebookLocal, kNorm, tileFloat, loadRows,
                              preScaleTile);
            if (preScaleTile) {
                turboquant_attn::ScaleRowsFloat(tileFloat, kNorm, loadRows, scaleValue_);
            }

            for (uint32_t q = 0; q < qRows; ++q) {
                const uint32_t causalKvEnd =
                    GetCausalKvEnd(tokenStartIdx + q, qChunkStart, qChunkEnd, kvLen);
                if (pos >= causalKvEnd) {
                    continue;
                }
                const uint32_t activeRows =
                    (pos + computeRows <= causalKvEnd) ? computeRows : (causalKvEnd - pos);
                if (preScaleTile) {
                    turboquant_attn::VectorQkFloatPreScaled(
                        qTile[QTileCompactGroupOffset(q, gqaCount)], tileFloat,
                        qTileScore[QTileCompactScoreOffset(q, gqaCount)], attnTmp0, attnTmp1,
                        expLocal, gqaCount, activeRows);
                } else {
                    if constexpr (QTileGqaCap == TQ_UB_Q_TILE_SMALL_GQA_CAP) {
                        if (gqaCount == TQ_UB_Q_TILE_SMALL_GQA_CAP) {
                            turboquant_attn::VectorQkFloatNormVectorGqa2(
                                qTile[QTileCompactGroupOffset(q, gqaCount)], tileFloat,
                                qTileScore[QTileCompactScoreOffset(q, gqaCount)], kNorm,
                                attnTmp0, attnTmp1, expLocal, activeRows);
                        } else {
                            turboquant_attn::VectorQkFloatNormVector(
                                qTile[QTileCompactGroupOffset(q, gqaCount)], tileFloat,
                                qTileScore[QTileCompactScoreOffset(q, gqaCount)], kNorm,
                                attnTmp0, attnTmp1, expLocal, gqaCount, activeRows, 1.f);
                        }
                    } else {
                        turboquant_attn::VectorQkFloatNormVector(
                            qTile[QTileCompactGroupOffset(q, gqaCount)], tileFloat,
                            qTileScore[QTileCompactScoreOffset(q, gqaCount)], kNorm, attnTmp0,
                            attnTmp1, expLocal, gqaCount, activeRows, 1.f);
                    }
                }
            }

            LoadPackedTileRows(valueCacheGm_, packedLocal, loadRows, seqIdx, kvHead, pos);
            DecodeTileToFloat(packedLocal, codebookVLocal, kNorm, tileFloat, loadRows,
                              preScaleTile);
            if (preScaleTile) {
                turboquant_attn::ScaleRowsFloat(tileFloat, kNorm, loadRows, 1.f);
            }

            for (uint32_t q = 0; q < qRows; ++q) {
                const uint32_t causalKvEnd =
                    GetCausalKvEnd(tokenStartIdx + q, qChunkStart, qChunkEnd, kvLen);
                if (pos >= causalKvEnd) {
                    continue;
                }
                const uint32_t activeRows =
                    (pos + computeRows <= causalKvEnd) ? computeRows : (causalKvEnd - pos);
                if (preScaleTile) {
                    turboquant_attn::OnlineSoftmaxUpdateTileFloatPreScaledScalar(
                        qTileScore[QTileCompactScoreOffset(q, gqaCount)], tileFloat,
                        mStateScalar + QTileCompactStateOffset(q, gqaCount),
                        sStateScalar + QTileCompactStateOffset(q, gqaCount),
                        outAccTile[QTileCompactGroupOffset(q, gqaCount)],
                        expLocal, attnTmp0, attnTmp1,
                        gqaCount, activeRows);
                } else {
                    if constexpr (QTileGqaCap == TQ_UB_Q_TILE_SMALL_GQA_CAP) {
                        if (gqaCount == TQ_UB_Q_TILE_SMALL_GQA_CAP) {
                            turboquant_attn::OnlineSoftmaxUpdateTileFloatGqa2Scalar(
                                qTileScore[QTileCompactScoreOffset(q, gqaCount)], tileFloat,
                                kNorm,
                                mStateScalar + QTileCompactStateOffset(q, gqaCount),
                                sStateScalar + QTileCompactStateOffset(q, gqaCount),
                                outAccTile[QTileCompactGroupOffset(q, gqaCount)],
                                expLocal, attnTmp0, activeRows);
                        } else {
                            turboquant_attn::OnlineSoftmaxUpdateTileFloatScalar(
                                qTileScore[QTileCompactScoreOffset(q, gqaCount)], tileFloat,
                                kNorm,
                                mStateScalar + QTileCompactStateOffset(q, gqaCount),
                                sStateScalar + QTileCompactStateOffset(q, gqaCount),
                                outAccTile[QTileCompactGroupOffset(q, gqaCount)],
                                expLocal, attnTmp0, attnTmp1, gqaCount, activeRows);
                        }
                    } else {
                        turboquant_attn::OnlineSoftmaxUpdateTileFloatScalar(
                            qTileScore[QTileCompactScoreOffset(q, gqaCount)], tileFloat, kNorm,
                            mStateScalar + QTileCompactStateOffset(q, gqaCount),
                            sStateScalar + QTileCompactStateOffset(q, gqaCount),
                            outAccTile[QTileCompactGroupOffset(q, gqaCount)],
                            expLocal, attnTmp0, attnTmp1, gqaCount, activeRows);
                    }
                }
            }
            pos += computeRows;
        }

        WriteFinalOutputQTile(tokenStartIdx, qRows, kvHead, sStateScalar, outAccTile,
                              gqaStart, gqaCount);
    }

    __aicore__ inline void ComputeAttention(
        uint32_t tokenIdx,
        uint32_t kvHead,
        uint32_t seqIdx,
        uint32_t qChunkStart,
        uint32_t qChunkEnd,
        uint32_t kvLen,
        uint32_t gqaStart,
        uint32_t gqaCount,
        uint32_t segIdx,
        uint32_t numSegs,
        bool writePartial)
    {
        const uint32_t causalKvEnd = GetCausalKvEnd(tokenIdx, qChunkStart, qChunkEnd, kvLen);
        if (kvLen == 0 || causalKvEnd == 0) {
            if (writePartial) {
                WriteEmptyPartial(tokenIdx, kvHead, segIdx, gqaStart, gqaCount);
            } else {
                WriteZeroOutput(tokenIdx, kvHead, gqaStart, gqaCount);
            }
            return;
        }

        uint32_t kvStart = 0;
        uint32_t kvEnd = causalKvEnd;
        if (numSegs > 1) {
            kvStart = segIdx * kvSegmentLen_;
            kvEnd = (segIdx + 1) * kvSegmentLen_;
            if (kvEnd > causalKvEnd) {
                kvEnd = causalKvEnd;
            }
            if (kvStart >= causalKvEnd) {
                if (writePartial) {
                    WriteEmptyPartial(tokenIdx, kvHead, segIdx, gqaStart, gqaCount);
                }
                return;
            }
        }

        auto qGroup = qGroupFloatBuf_.Get<float>();
        LoadQGroup(tokenIdx, kvHead, gqaStart, gqaCount, qGroup);
        RotateRowsInPlace(qGroup, rotationGm_, gqaCount);

        // m/s states as Scalar arrays — eliminates GetValue/SetValue V↔S sync.
        float mStateScalar[TQ_UB_GQA_CAP];
        float sStateScalar[TQ_UB_GQA_CAP];
        for (uint32_t g = 0; g < gqaCount; ++g) {
            mStateScalar[g] = -3.402823466e+38f;
            sStateScalar[g] = 0.f;
        }
        auto outAcc = outAccBuf_.Get<float>();
        AscendC::Duplicate(outAcc, 0.f, gqaCount * TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();

        auto packedLocal = packedBuf_.Get<uint8_t>();
        auto codebookLocal = codebookBuf_.Get<TqDataT>();
        auto codebookVLocal = codebookVBuf_.Get<TqDataT>();
        auto expLocal = expBuf_.Get<float>();
        auto attnTmp0 = attnTmpBuf_.Get<float>();
        auto attnTmp1 = attnTmp0[TQ_HEAD];
        auto kNorm = kNormBuf_.Get<float>();
        auto tileFloat = idxFloatBuf_.Get<float>();
        const bool preScaleTile = gqaCount >= 8U;
        if (!preScaleTile) {
            AscendC::Muls(qGroup, qGroup, scaleValue_, gqaCount * TQ_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t pos = kvStart; pos < kvEnd;) {
            const uint32_t computeRows =
                (pos + kvTileRows_ <= kvEnd) ? kvTileRows_ : (kvEnd - pos);
            const uint32_t loadRows = AlignUpGroupRows(computeRows);
            LoadPackedTileRows(keyCacheGm_, packedLocal, loadRows, seqIdx, kvHead, pos);
            DecodeTileToFloat(packedLocal, codebookLocal, kNorm, tileFloat, loadRows,
                              preScaleTile);
            if (preScaleTile) {
                turboquant_attn::ScaleRowsFloat(tileFloat, kNorm, loadRows, scaleValue_);
            }

            auto scoreTile = scoreBuf_.Get<float>();
            if (preScaleTile) {
                turboquant_attn::VectorQkFloatPreScaled(qGroup, tileFloat, scoreTile,
                                                        attnTmp0, attnTmp1, expLocal,
                                                        gqaCount, computeRows);
            } else {
#if TQ_ATTN_QK_NORM_VECTOR
                if (gqaCount == 2U) {
                    turboquant_attn::VectorQkFloatNormVectorGqa2(
                        qGroup, tileFloat, scoreTile, kNorm, attnTmp0, attnTmp1, expLocal,
                        computeRows);
                } else {
                    turboquant_attn::VectorQkFloatNormVector(
                        qGroup, tileFloat, scoreTile, kNorm, attnTmp0, attnTmp1, expLocal,
                        gqaCount, computeRows, 1.f);
                }
#else
                turboquant_attn::VectorQkFloat(qGroup, tileFloat, scoreTile, kNorm,
                                               attnTmp0, attnTmp1, expLocal,
                                               gqaCount, computeRows, 1.f);
#endif
            }

            LoadPackedTileRows(valueCacheGm_, packedLocal, loadRows, seqIdx, kvHead, pos);
            DecodeTileToFloat(packedLocal, codebookVLocal, kNorm, tileFloat, loadRows,
                              preScaleTile);
            if (preScaleTile) {
                turboquant_attn::ScaleRowsFloat(tileFloat, kNorm, loadRows, 1.f);
            }

            if (preScaleTile) {
                turboquant_attn::OnlineSoftmaxUpdateTileFloatPreScaledScalar(
                    scoreTile, tileFloat, mStateScalar, sStateScalar, outAcc, expLocal,
                    attnTmp0, attnTmp1, gqaCount, computeRows);
            } else {
                turboquant_attn::OnlineSoftmaxUpdateTileFloatScalar(
                    scoreTile, tileFloat, kNorm, mStateScalar, sStateScalar, outAcc, expLocal,
                    attnTmp0, attnTmp1, gqaCount, computeRows);
            }
            pos += computeRows;
        }

        if (writePartial) {
            WritePartial(tokenIdx, kvHead, segIdx, mStateScalar, sStateScalar, outAcc, gqaStart, gqaCount);
        } else {
            WriteFinalOutput(tokenIdx, kvHead, sStateScalar, outAcc, gqaStart, gqaCount);
        }
    }

    __aicore__ inline void WriteZeroOutput(
        uint32_t tokenIdx, uint32_t kvHead, uint32_t gqaStart, uint32_t gqaCount)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_ + gqaStart;
        auto outLocal = qGroupBuf_.Get<QueryT>();
        AscendC::Duplicate(outLocal.template ReinterpretCast<uint16_t>(),
                           static_cast<uint16_t>(0), TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::V_MTE3>();
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint64_t off =
                (static_cast<uint64_t>(tokenIdx) * numHeads_ + qBaseHead + g) * TQ_HEAD;
            CopyOutputRow(off, outLocal);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void WriteFinalOutput(
        uint32_t tokenIdx,
        uint32_t kvHead,
        float* sStateScalar,
        LocalTensor<float> outAcc,
        uint32_t gqaStart,
        uint32_t gqaCount)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_ + gqaStart;
        auto outLocal = qGroupBuf_.Get<QueryT>();
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const float sum = sStateScalar[g];
            const float invS = (sum > 0.f) ? (1.f / sum) : 0.f;
            const uint32_t outBase = g * TQ_HEAD;
            AscendC::Muls(outAcc[outBase], outAcc[outBase], invS, TQ_HEAD);
        }
        AscendC::PipeBarrier<PIPE_V>();
        RotateRowsInPlace(outAcc, rotationVGm_, gqaCount);
        AscendC::Cast(outLocal, outAcc, AscendC::RoundMode::CAST_RINT,
                      gqaCount * TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::V_MTE3>();
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint64_t off =
                (static_cast<uint64_t>(tokenIdx) * numHeads_ + qBaseHead + g) * TQ_HEAD;
            CopyOutputRow(off, outLocal[g * TQ_HEAD]);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void WriteEmptyPartial(
        uint32_t tokenIdx,
        uint32_t kvHead,
        uint32_t segIdx,
        uint32_t gqaStart,
        uint32_t gqaCount)
    {
        if (partialWorkspace_ == nullptr) {
            return;
        }
        const uint32_t qBaseHead = kvHead * gqaGroup_ + gqaStart;
        auto zero = outAccBuf_.Get<float>();
        AscendC::Duplicate(zero, 0.f, TQ_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::V_MTE3>();
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint32_t headIdx = qBaseHead + g;
            const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
            partialLseGm_.SetValue(partIdx * 2, -3.402823466e+38f);
            partialLseGm_.SetValue(partIdx * 2 + 1, 0.f);
            DataCopy(partialOutGm_[partIdx * TQ_HEAD], zero, TQ_HEAD);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void WritePartial(
        uint32_t tokenIdx,
        uint32_t kvHead,
        uint32_t segIdx,
        float* mStateScalar,
        float* sStateScalar,
        LocalTensor<float> outAcc,
        uint32_t gqaStart,
        uint32_t gqaCount)
    {
        if (partialWorkspace_ == nullptr) {
            return;
        }
        const uint32_t qBaseHead = kvHead * gqaGroup_ + gqaStart;
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint32_t headIdx = qBaseHead + g;
            const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
            const float maxVal = mStateScalar[g];
            const float sumVal = sStateScalar[g];
            partialLseGm_.SetValue(partIdx * 2, maxVal);
            partialLseGm_.SetValue(partIdx * 2 + 1, sumVal);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::V_MTE3>();
        for (uint32_t g = 0; g < gqaCount; ++g) {
            const uint32_t headIdx = qBaseHead + g;
            const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
            DataCopy(partialOutGm_[partIdx * TQ_HEAD], outAcc[g * TQ_HEAD], TQ_HEAD);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_V>();
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void CombineFlashDecode()
    {
        auto expLocal = expBuf_.Get<float>();
        auto globalOut = combineOutBuf_.Get<float>();
        auto partFloat = partialOutBuf_.Get<float>();
        for (uint32_t tokenIdx = 0; tokenIdx < numTokens_; ++tokenIdx) {
            for (uint32_t headIdx = 0; headIdx < numHeads_; ++headIdx) {
                float bestM = -3.402823466e+38f;
                for (uint32_t p = 0; p < kvSplitPart_; ++p) {
                    const uint32_t partIdx =
                        (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + p;
                    const float partM = partialLseGm_.GetValue(partIdx * 2);
                    const float partS = partialLseGm_.GetValue(partIdx * 2 + 1);
                    if (partS > 0.f && partM > bestM) {
                        bestM = partM;
                    }
                }
                float globalS = 0.f;
                AscendC::Duplicate(globalOut, 0.f, TQ_HEAD);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t p = 0; p < kvSplitPart_; ++p) {
                    const uint32_t partIdx =
                        (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + p;
                    const float partM = partialLseGm_.GetValue(partIdx * 2);
                    const float partS = partialLseGm_.GetValue(partIdx * 2 + 1);
                    if (partS <= 0.f) {
                        continue;
                    }
                    const float weight = turboquant_attn::ExpScalar(expLocal, partM - bestM);
                    globalS += weight * partS;
                    DataCopy(partFloat, partialOutGm_[partIdx * TQ_HEAD], TQ_HEAD);
                    turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();
                    AscendC::Muls(partFloat, partFloat, weight, TQ_HEAD);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(globalOut, globalOut, partFloat, TQ_HEAD);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                const float invS = (globalS > 0.f) ? (1.f / globalS) : 0.f;
                const uint64_t off =
                    (static_cast<uint64_t>(tokenIdx) * numHeads_ + headIdx) * TQ_HEAD;
                AscendC::Muls(globalOut, globalOut, invS, TQ_HEAD);
                AscendC::PipeBarrier<PIPE_V>();
                RotateRowsInPlace(globalOut, rotationVGm_, 1);
                auto partLocal = qGroupBuf_.Get<QueryT>();
                AscendC::Cast(partLocal, globalOut, AscendC::RoundMode::CAST_RINT, TQ_HEAD);
                AscendC::PipeBarrier<PIPE_V>();
                turboquant::TqDecodeSync<AscendC::HardEvent::V_MTE3>();
                CopyOutputRow(off, partLocal);
                turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_V>();
                turboquant::TqDecodeSync<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
    }

    TPipe* pipe_ = nullptr;
    TqRotateMm* rotateMm_ = nullptr;
    const TilingT* tiling_ = nullptr;
    bool matmulReady_ = false;
    __gm__ uint8_t* partialWorkspace_ = nullptr;

    uint32_t numTokens_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t gqaGroup_ = 0;
    uint32_t gqaChunkCount_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t totalCacheBlocks_ = 0;
    uint32_t kvTileRows_ = 0;
    uint32_t splitMode_ = 0;
    uint32_t kvSplitPart_ = 1;
    uint32_t qkPvMode_ = 0;
    uint32_t kvSegmentLen_ = 0;
    uint32_t usedCoreNum_ = 0;
    float scaleValue_ = 1.f;

    GlobalTensor<QueryT> queryGm_;
    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int64_t> actualSeqLenQGm_;
    GlobalTensor<int64_t> actualSeqLenKvGm_;
    GlobalTensor<TqDataT> codebookGm_;
    GlobalTensor<TqDataT> codebookVGm_;
    GlobalTensor<TqDataT> rotationGm_;
    GlobalTensor<TqDataT> rotationVGm_;
    GlobalTensor<QueryT> outGm_;
    GlobalTensor<float> partialOutGm_;
    GlobalTensor<float> partialLseGm_;
    TBuf<TPosition::VECCALC> codebookBuf_;
    TBuf<TPosition::VECCALC> codebookVBuf_;
    TBuf<TPosition::VECCALC> packedBuf_;
    TQue<TPosition::VECIN, 1> xHatQue_;
    TBuf<TPosition::VECCALC> rotateWorkBuf_;
    TBuf<TPosition::VECCALC> expBuf_;
    TBuf<TPosition::VECCALC> qGroupBuf_;
    TBuf<TPosition::VECCALC> qGroupFloatBuf_;
    TBuf<TPosition::VECCALC> outFloatBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> kNormBuf_;
    TBuf<TPosition::VECCALC> mStateBuf_;
    TBuf<TPosition::VECCALC> sStateBuf_;
    TBuf<TPosition::VECCALC> outAccBuf_;
    TBuf<TPosition::VECCALC> combineOutBuf_;
    TBuf<TPosition::VECCALC> partialOutBuf_;
    TBuf<TPosition::VECCALC> idxFloatBuf_;
    TBuf<TPosition::VECCALC> idxS32Buf_;
    TBuf<TPosition::VECCALC> attnTmpBuf_;
    TBuf<TPosition::VECCALC> packedRawBuf_;
    TBuf<TPosition::VECCALC> packedMaskBuf_;
    TQue<TPosition::VECOUT, 1> yHatQue_;
};

#define INVOKE_TQ_ATTN_KERNEL(ENABLE_Q_TILE, Q_TILE_CAP, Q_TILE_GQA_CAP)                         \
    do {                                                                                        \
        TurboquantAttentionPaged4bitKernel<TurboquantAttentionPaged4bitTilingData, TqQueryT,     \
                                           ENABLE_Q_TILE, Q_TILE_CAP, Q_TILE_GQA_CAP> op(        \
            &pipe, &rotateMm);                                                                   \
        op.Init(reinterpret_cast<__gm__ TqQueryT*>(query),                                      \
                reinterpret_cast<__gm__ uint8_t*>(key_cache),                                   \
                reinterpret_cast<__gm__ uint8_t*>(value_cache),                                 \
                reinterpret_cast<__gm__ int32_t*>(block_table),                                 \
                reinterpret_cast<__gm__ int64_t*>(actual_seq_len_q),                          \
                reinterpret_cast<__gm__ int64_t*>(actual_seq_len_kv),                         \
                reinterpret_cast<__gm__ TqDataT*>(codebook),                                   \
                reinterpret_cast<__gm__ TqDataT*>(rotation),                                   \
                reinterpret_cast<__gm__ TqDataT*>(codebook_value),                             \
                reinterpret_cast<__gm__ TqDataT*>(rotation_value),                             \
                reinterpret_cast<__gm__ TqQueryT*>(out),                                        \
                reinterpret_cast<__gm__ uint8_t*>(workspace), &tilingData);                      \
        op.Process();                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void turboquant_attention_paged4bit(
    GM_ADDR query,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR block_table,
    GM_ADDR actual_seq_len_q,
    GM_ADDR actual_seq_len_kv,
    GM_ADDR codebook,
    GM_ADDR rotation,
    GM_ADDR codebook_value,
    GM_ADDR rotation_value,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    TqRotateMm rotateMm;
    REGIST_MATMUL_OBJ_STATIC(&pipe, GetSysWorkSpacePtr(), rotateMm, (TCubeTiling*)nullptr);
    if constexpr (TILING_KEY_IS(5)) {
        INVOKE_TQ_ATTN_KERNEL(true, TQ_UB_Q_TILE_GQA2_CAP, TQ_UB_Q_TILE_SMALL_GQA_CAP);
    } else if constexpr (TILING_KEY_IS(4)) {
        INVOKE_TQ_ATTN_KERNEL(true, TQ_UB_Q_TILE_CAP, TQ_UB_GQA_CAP);
    } else {
        INVOKE_TQ_ATTN_KERNEL(false, 1, 1);
    }
}

}  // namespace
