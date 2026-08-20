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
constexpr uint32_t TQ_AIV_SUB = 2;
constexpr uint32_t TQ_ROT_WS = TQ_HEAD * TQ_HEAD * sizeof(half);
constexpr uint64_t TQ_CUBE_C_OFF = 256 * 1024;
// Host PickKvTileRows() caps at TQ_UB_KV_TILE_CAP (32); PA block_size may be 128.
// CANN GetWorkspaceSize static analysis uses InitBuffer sizes from the kernel
// binary; runtime tiling fields here must not be used as InitBuffer extents.
// Compile-time UB caps for InitBuffer; Mc2 workspace analysis sums these statically.
constexpr uint32_t TQ_UB_KV_TILE_CAP = 32;
constexpr uint32_t TQ_UB_GQA_CAP = 8;

__aicore__ inline uint32_t AlignUp16(uint32_t x)
{
    return (x + 15U) / 16U * 16U;
}

__aicore__ inline uint32_t FindSeqForToken(
    uint32_t tokenIdx,
    const GlobalTensor<int64_t>& actualSeqLenQ,
    uint32_t batch)
{
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t end = static_cast<uint32_t>(actualSeqLenQ.GetValue(b));
        if (tokenIdx < end) {
            return b;
        }
    }
    return (batch > 0) ? (batch - 1) : 0;
}

using TqRotateMm = turboquant::TqDecodeRotateMatmulOp;

template <typename TilingT>
class TurboquantAttentionPaged8bitKernel {
public:
    __aicore__ inline TurboquantAttentionPaged8bitKernel(TPipe* pipe, TqRotateMm* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm)
    {}

    __aicore__ inline void Init(
        __gm__ half* query,
        __gm__ uint8_t* keyCache,
        __gm__ uint8_t* valueCache,
        __gm__ int32_t* blockTable,
        __gm__ int64_t* actualSeqLenQ,
        __gm__ int64_t* actualSeqLenKv,
        __gm__ half* codebook,
        __gm__ half* rotation,
        __gm__ half* codebookValue,
        __gm__ half* rotationValue,
        __gm__ half* out,
        __gm__ uint8_t* workspace,
        const TilingT* tiling)
    {
        tiling_ = tiling;
        numTokens_ = tiling_->numTokens;
        batchSize_ = tiling_->batchSize;
        numHeads_ = tiling_->numHeads;
        numKvHeads_ = tiling_->numKvHeads;
        gqaGroup_ = tiling_->gqaGroupSize;
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
        rotationGm_.SetGlobalBuffer(rotation, static_cast<uint64_t>(TQ_HEAD) * TQ_HEAD);
        rotationVGm_.SetGlobalBuffer(rotationValue, static_cast<uint64_t>(TQ_HEAD) * TQ_HEAD);
        outGm_.SetGlobalBuffer(out, static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);

        pipe_->InitBuffer(codebookBuf_, 2 * TQ_CODEBOOK * sizeof(half));
        // Compact decode layout: [M*128 idx bytes][M*2 norm bytes] (matches decode op).
        pipe_->InitBuffer(packedBuf_,
                          TQ_UB_KV_TILE_CAP * TQ_HEAD +
                              TQ_UB_KV_TILE_CAP * static_cast<uint32_t>(sizeof(half)));
        pipe_->InitBuffer(xHatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(half));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_WS);
        pipe_->InitBuffer(expBuf_, 4 * sizeof(float));
        pipe_->InitBuffer(qGroupBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(half));
        pipe_->InitBuffer(qGroupFloatBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, TQ_UB_GQA_CAP * TQ_UB_KV_TILE_CAP * sizeof(float));
        pipe_->InitBuffer(mStateBuf_, TQ_UB_GQA_CAP * sizeof(float));
        pipe_->InitBuffer(sStateBuf_, TQ_UB_GQA_CAP * sizeof(float));
        pipe_->InitBuffer(outAccBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        if (splitMode_ == 1) {
            pipe_->InitBuffer(combineOutBuf_, TQ_HEAD * sizeof(float));
        }
        pipe_->InitBuffer(idxHalfBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(half));
        pipe_->InitBuffer(idxFloatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(idxS32Buf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(int32_t));
        pipe_->InitBuffer(yHatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(half));

        auto* wsBase = reinterpret_cast<__gm__ uint8_t*>(GetSysWorkSpacePtr());
        matmulReady_ = (wsBase != nullptr);
        if (matmulReady_) {
            // wsBase is shared across blocks; stride by logical MIX core (coreIdx), not
            // GetBlockIdx() (primary+secondary AIV share one MIX core). Must match host
            // TQ_ATTN_MAX_PARALLEL_CORES cap on usedCoreNum / blockDim.
            const uint32_t coreIdx = GetBlockIdx() / TQ_AIV_SUB;
            constexpr uint64_t kCubeCStride =
                static_cast<uint64_t>(TQ_UB_KV_TILE_CAP) * TQ_HEAD * sizeof(float);
            const uint64_t cubeCOff = TQ_CUBE_C_OFF + static_cast<uint64_t>(coreIdx) * kCubeCStride;
            cubeCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsBase + cubeCOff),
                                     static_cast<uint64_t>(TQ_UB_KV_TILE_CAP) * TQ_HEAD);
        }

        if (splitMode_ == 1 && workspace != nullptr) {
            partialWorkspace_ = workspace;
            const size_t accumBytes =
                static_cast<size_t>(tiling_->accumOutSize) * sizeof(half);
            const size_t lseBytes =
                static_cast<size_t>(tiling_->logSumExpSize) * sizeof(float);
            partialOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace), accumBytes / sizeof(half));
            partialLseGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(workspace + accumBytes), lseBytes / sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx() / TQ_AIV_SUB;
        if (splitMode_ == 1) {
            // SplitBNS + FlashDecode: every launched MIX core (AIC + both AIV
            // sub-blocks, including idle ones) must enter SyncAll exactly once.
            if ASCEND_IS_AIC {
                SyncAll();
                return;
            }
            const bool isPrimaryAiv = ((GetSubBlockIdx() % TQ_AIV_SUB) == 0);
            if (isPrimaryAiv && coreIdx < usedCoreNum_) {
                LoadCodebook();
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
        const bool isPrimaryAiv = ((GetSubBlockIdx() % TQ_AIV_SUB) == 0);
        if (!isPrimaryAiv || coreIdx >= usedCoreNum_) {
            return;
        }

        LoadCodebook();
        ProcessSplitBn(coreIdx);
    }

private:
    __aicore__ inline void LoadCodebook()
    {
        auto codebookLocal = codebookBuf_.Get<half>();
        DataCopy(codebookLocal, codebookGm_, TQ_CODEBOOK);
        DataCopy(codebookLocal[TQ_CODEBOOK], codebookVGm_, TQ_CODEBOOK);
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
            return (splitMode_ == 1) ? (numTokens_ * numKvHeads_ * kvSplitPart_)
                                     : (numTokens_ * numKvHeads_);
        }
        return TaskStart(coreIdx + 1);
    }

    __aicore__ inline void ProcessSplitBn(uint32_t coreIdx)
    {
        const uint32_t start = TaskStart(coreIdx);
        const uint32_t end = TaskEnd(coreIdx);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t tokenIdx = task / numKvHeads_;
            const uint32_t kvHead = task % numKvHeads_;
            ComputeAttention(tokenIdx, kvHead, 0, kvSplitPart_, false);
        }
    }

    __aicore__ inline void ProcessSplitBns(uint32_t coreIdx)
    {
        const uint32_t start = TaskStart(coreIdx);
        const uint32_t end = TaskEnd(coreIdx);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t bnTask = task / kvSplitPart_;
            const uint32_t segIdx = task % kvSplitPart_;
            const uint32_t tokenIdx = bnTask / numKvHeads_;
            const uint32_t kvHead = bnTask % numKvHeads_;
            ComputeAttention(tokenIdx, kvHead, segIdx, kvSplitPart_, true);
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
        for (uint32_t t = 0; t < mRows; ++t) {
            const uint32_t absPos = kvStart + t;
            const uint32_t blockId = static_cast<uint32_t>(blockTableGm_.GetValue(
                static_cast<uint64_t>(seqIdx) * maxBlocksPerSeq_ + (absPos / blockSize_)));
            const uint32_t posInBlock = absPos % blockSize_;
            const uint64_t srcOff = (((static_cast<uint64_t>(blockId) * blockSize_ + posInBlock) *
                                      numKvHeads_ +
                                      kvHead) *
                                     TQ_PACKED);
            DataCopy(packed[t * TQ_HEAD], cache[srcOff], TQ_HEAD);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_S>();

        const uint32_t normBase = mRows * TQ_HEAD;
        for (uint32_t t = 0; t < mRows; ++t) {
            const uint32_t absPos = kvStart + t;
            const uint32_t blockId = static_cast<uint32_t>(blockTableGm_.GetValue(
                static_cast<uint64_t>(seqIdx) * maxBlocksPerSeq_ + (absPos / blockSize_)));
            const uint32_t posInBlock = absPos % blockSize_;
            const uint64_t srcOff = (((static_cast<uint64_t>(blockId) * blockSize_ + posInBlock) *
                                      numKvHeads_ +
                                      kvHead) *
                                     TQ_PACKED);
            const uint32_t dstNorm = normBase + t * sizeof(half);
            packed.SetValue(dstNorm, cache.GetValue(srcOff + TQ_HEAD));
            packed.SetValue(dstNorm + 1, cache.GetValue(srcOff + TQ_HEAD + 1));
        }
    }

    __aicore__ inline void LoadRotation(GlobalTensor<half>& rotationGm)
    {
        auto rotation = rotateWorkBuf_.Get<half>();
        DataCopy(rotation, rotationGm, TQ_HEAD * TQ_HEAD);
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_S>();
    }

    __aicore__ inline void DecodeTile(
        LocalTensor<uint8_t> packedLocal,
        LocalTensor<half> codebookLocal,
        GlobalTensor<half>& rotationGm,
        LocalTensor<half>& xHat,
        uint32_t mRows,
        uint32_t decodePhase)
    {
        turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
        if (!matmulReady_ || rotateMm_ == nullptr) {
            LoadRotation(rotationGm);
            turboquant::DecodeRowsScalar(packedLocal, codebookLocal, rotateWorkBuf_.Get<half>(),
                                         xHat, mRows);
            turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
            return;
        }
        turboquant::DecodeRows8bit(
            packedLocal, codebookLocal, rotationGm, cubeCGm_, *rotateMm_,
            idxHalfBuf_.Get<half>(), idxFloatBuf_.Get<float>(), idxS32Buf_.Get<int32_t>(),
            yHatBuf_.Get<half>(), rotateWorkBuf_.Get<uint8_t>(), idxFloatBuf_.Get<float>(), xHat,
            mRows, AlignUp16(mRows));
        turboquant::TqDecodeSync<AscendC::HardEvent::V_S>();
    }

    __aicore__ inline void LoadQGroup(uint32_t tokenIdx, uint32_t kvHead, LocalTensor<float> qGroup)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_;
        auto qLocal = qGroupBuf_.Get<half>();
        for (uint32_t g = 0; g < gqaGroup_; ++g) {
            const uint64_t off =
                (static_cast<uint64_t>(tokenIdx) * numHeads_ + qBaseHead + g) * TQ_HEAD;
            DataCopy(qLocal[g * TQ_HEAD], queryGm_[off], TQ_HEAD);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_S>();
        for (uint32_t g = 0; g < gqaGroup_; ++g) {
            for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                qGroup.SetValue(g * TQ_HEAD + d, static_cast<float>(qLocal.GetValue(g * TQ_HEAD + d)));
            }
        }
    }

    __aicore__ inline uint32_t GetCausalKvEnd(uint32_t tokenIdx, uint32_t seqIdx) const
    {
        const uint32_t qChunkEnd = static_cast<uint32_t>(actualSeqLenQGm_.GetValue(seqIdx));
        const uint32_t qChunkStart =
            (seqIdx == 0) ? 0U : static_cast<uint32_t>(actualSeqLenQGm_.GetValue(seqIdx - 1));
        const uint32_t numQInChunk = qChunkEnd - qChunkStart;
        const uint32_t qPosInChunk = tokenIdx - qChunkStart;
        const uint32_t kvLen = static_cast<uint32_t>(actualSeqLenKvGm_.GetValue(seqIdx));
        const uint32_t absQPos = kvLen - numQInChunk + qPosInChunk;
        return absQPos + 1U;
    }

    __aicore__ inline void ComputeAttention(
        uint32_t tokenIdx,
        uint32_t kvHead,
        uint32_t segIdx,
        uint32_t numSegs,
        bool writePartial)
    {
        const uint32_t seqIdx = FindSeqForToken(tokenIdx, actualSeqLenQGm_, batchSize_);
        const uint32_t kvLen = static_cast<uint32_t>(actualSeqLenKvGm_.GetValue(seqIdx));
        const uint32_t causalKvEnd = GetCausalKvEnd(tokenIdx, seqIdx);
        if (kvLen == 0 || causalKvEnd == 0) {
            WriteZeroOutput(tokenIdx, kvHead);
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
                return;
            }
        }

        auto qGroup = qGroupFloatBuf_.Get<float>();
        LoadQGroup(tokenIdx, kvHead, qGroup);

        auto mState = mStateBuf_.Get<float>();
        auto sState = sStateBuf_.Get<float>();
        auto outAcc = outAccBuf_.Get<float>();
        for (uint32_t g = 0; g < gqaGroup_; ++g) {
            mState.SetValue(g, -3.402823466e+38f);
            sState.SetValue(g, 0.f);
            for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                outAcc.SetValue(g * TQ_HEAD + d, 0.f);
            }
        }

        auto packedLocal = packedBuf_.Get<uint8_t>();
        auto xHat = xHatBuf_.Get<half>();
        auto codebookLocal = codebookBuf_.Get<half>();
        auto expLocal = expBuf_.Get<float>();

        for (uint32_t pos = kvStart; pos < kvEnd;) {
            const uint32_t tileRows = (pos + kvTileRows_ <= kvEnd) ? kvTileRows_ : (kvEnd - pos);
            LoadPackedTileRows(keyCacheGm_, packedLocal, tileRows, seqIdx, kvHead, pos);
            LocalTensor<half> kTile = xHat;
            DecodeTile(packedLocal, codebookLocal, rotationGm_, kTile, tileRows, 0);

            auto scoreTile = scoreBuf_.Get<float>();
            turboquant_attn::VectorQk(qGroup, kTile, scoreTile, gqaGroup_, tileRows, scaleValue_);

            LoadPackedTileRows(valueCacheGm_, packedLocal, tileRows, seqIdx, kvHead, pos);
            LocalTensor<half> vTile = xHat;
            DecodeTile(packedLocal, codebookLocal[TQ_CODEBOOK], rotationVGm_, vTile, tileRows, 1);

            turboquant_attn::OnlineSoftmaxUpdateTile(
                scoreTile, vTile, mState, sState, outAcc, expLocal, gqaGroup_, tileRows);
            pos += tileRows;
        }

        if (writePartial) {
            WritePartial(tokenIdx, kvHead, segIdx, mState, sState, outAcc);
        } else {
            WriteFinalOutput(tokenIdx, kvHead, sState, outAcc);
        }
    }

    __aicore__ inline void WriteZeroOutput(uint32_t tokenIdx, uint32_t kvHead)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_;
        for (uint32_t g = 0; g < gqaGroup_; ++g) {
            const uint64_t off =
                (static_cast<uint64_t>(tokenIdx) * numHeads_ + qBaseHead + g) * TQ_HEAD;
            for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                outGm_.SetValue(off + d, static_cast<half>(0));
            }
        }
    }

    __aicore__ inline void WriteFinalOutput(
        uint32_t tokenIdx, uint32_t kvHead, LocalTensor<float> sState, LocalTensor<float> outAcc)
    {
        const uint32_t qBaseHead = kvHead * gqaGroup_;
        for (uint32_t g = 0; g < gqaGroup_; ++g) {
            const float sum = sState.GetValue(g);
            const float invS = (sum > 0.f) ? (1.f / sum) : 0.f;
            const uint64_t off =
                (static_cast<uint64_t>(tokenIdx) * numHeads_ + qBaseHead + g) * TQ_HEAD;
            for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                outGm_.SetValue(off + d,
                                static_cast<half>(outAcc.GetValue(g * TQ_HEAD + d) * invS));
            }
        }
    }

    __aicore__ inline void WritePartial(
        uint32_t tokenIdx,
        uint32_t kvHead,
        uint32_t segIdx,
        LocalTensor<float> mState,
        LocalTensor<float> sState,
        LocalTensor<float> outAcc)
    {
        if (partialWorkspace_ == nullptr) {
            return;
        }
        const uint32_t qBaseHead = kvHead * gqaGroup_;
        for (uint32_t g = 0; g < gqaGroup_; ++g) {
            const uint32_t headIdx = qBaseHead + g;
            const uint32_t partIdx = (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + segIdx;
            const float maxVal = mState.GetValue(g);
            const float sumVal = sState.GetValue(g);
            partialLseGm_.SetValue(partIdx * 2, maxVal);
            partialLseGm_.SetValue(partIdx * 2 + 1, sumVal);
            const float invS = (sumVal > 0.f) ? (1.f / sumVal) : 0.f;
            for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                partialOutGm_.SetValue(partIdx * TQ_HEAD + d,
                                       static_cast<half>(outAcc.GetValue(g * TQ_HEAD + d) * invS));
            }
        }
    }

    __aicore__ inline void CombineFlashDecode()
    {
        auto expLocal = expBuf_.Get<float>();
        auto globalOut = combineOutBuf_.Get<float>();
        for (uint32_t tokenIdx = 0; tokenIdx < numTokens_; ++tokenIdx) {
            for (uint32_t headIdx = 0; headIdx < numHeads_; ++headIdx) {
                float bestM = -3.402823466e+38f;
                for (uint32_t p = 0; p < kvSplitPart_; ++p) {
                    const uint32_t partIdx =
                        (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + p;
                    const float partM = partialLseGm_.GetValue(partIdx * 2);
                    if (partM > bestM) {
                        bestM = partM;
                    }
                }
                float globalS = 0.f;
                for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                    globalOut.SetValue(d, 0.f);
                }
                for (uint32_t p = 0; p < kvSplitPart_; ++p) {
                    const uint32_t partIdx =
                        (tokenIdx * numHeads_ + headIdx) * kvSplitPart_ + p;
                    const float partM = partialLseGm_.GetValue(partIdx * 2);
                    const float partS = partialLseGm_.GetValue(partIdx * 2 + 1);
                    const float weight = turboquant_attn::ExpScalar(expLocal, partM - bestM) * partS;
                    globalS += weight;
                    for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                        const float partOut =
                            static_cast<float>(partialOutGm_.GetValue(partIdx * TQ_HEAD + d));
                        globalOut.SetValue(d, globalOut.GetValue(d) + weight * partOut);
                    }
                }
                const float invS = (globalS > 0.f) ? (1.f / globalS) : 0.f;
                const uint64_t off =
                    (static_cast<uint64_t>(tokenIdx) * numHeads_ + headIdx) * TQ_HEAD;
                for (uint32_t d = 0; d < TQ_HEAD; ++d) {
                    outGm_.SetValue(off + d, static_cast<half>(globalOut.GetValue(d) * invS));
                }
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

    GlobalTensor<half> queryGm_;
    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int64_t> actualSeqLenQGm_;
    GlobalTensor<int64_t> actualSeqLenKvGm_;
    GlobalTensor<half> codebookGm_;
    GlobalTensor<half> codebookVGm_;
    GlobalTensor<half> rotationGm_;
    GlobalTensor<half> rotationVGm_;
    GlobalTensor<half> outGm_;
    GlobalTensor<half> partialOutGm_;
    GlobalTensor<float> partialLseGm_;
    GlobalTensor<float> cubeCGm_;

    TBuf<TPosition::VECCALC> codebookBuf_;
    TBuf<TPosition::VECCALC> packedBuf_;
    TBuf<TPosition::VECCALC> xHatBuf_;
    TBuf<TPosition::VECCALC> rotateWorkBuf_;
    TBuf<TPosition::VECCALC> expBuf_;
    TBuf<TPosition::VECCALC> qGroupBuf_;
    TBuf<TPosition::VECCALC> qGroupFloatBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> mStateBuf_;
    TBuf<TPosition::VECCALC> sStateBuf_;
    TBuf<TPosition::VECCALC> outAccBuf_;
    TBuf<TPosition::VECCALC> combineOutBuf_;
    TBuf<TPosition::VECCALC> idxHalfBuf_;
    TBuf<TPosition::VECCALC> idxFloatBuf_;
    TBuf<TPosition::VECCALC> idxS32Buf_;
    TBuf<TPosition::VECOUT> yHatBuf_;
};

#define INVOKE_TQ_ATTN_KERNEL()                                                                   \
    do {                                                                                        \
        GET_TILING_DATA(tilingData, tiling);                                                    \
        TPipe pipe;                                                                             \
        TqRotateMm rotateMm;                                                                    \
        TCubeTiling decodeTiling = tilingData.decodeRotateTiling;                               \
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &decodeTiling);                \
        TurboquantAttentionPaged8bitKernel<TurboquantAttentionPaged8bitTilingData> op(          \
            &pipe, &rotateMm);                                                                  \
        op.Init(reinterpret_cast<__gm__ half*>(query),                                          \
                reinterpret_cast<__gm__ uint8_t*>(key_cache),                                   \
                reinterpret_cast<__gm__ uint8_t*>(value_cache),                                 \
                reinterpret_cast<__gm__ int32_t*>(block_table),                                 \
                reinterpret_cast<__gm__ int64_t*>(actual_seq_len_q),                          \
                reinterpret_cast<__gm__ int64_t*>(actual_seq_len_kv),                         \
                reinterpret_cast<__gm__ half*>(codebook),                                     \
                reinterpret_cast<__gm__ half*>(rotation),                                     \
                reinterpret_cast<__gm__ half*>(codebook_value),                                 \
                reinterpret_cast<__gm__ half*>(rotation_value),                                 \
                reinterpret_cast<__gm__ half*>(out),                                            \
                reinterpret_cast<__gm__ uint8_t*>(workspace), &tilingData);                      \
        op.Process();                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void turboquant_attention_paged8bit(
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
    // Single MIX entry: splitMode/qkPvMode are runtime tiling fields. Additional
    // TILING_KEY_IS branches each register matmul and trigger Mc2 workspace blow-up.
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    INVOKE_TQ_ATTN_KERNEL();
}

}  // namespace
