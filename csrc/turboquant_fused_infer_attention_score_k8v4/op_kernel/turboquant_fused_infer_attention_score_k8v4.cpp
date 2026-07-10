/*
 * TurboQuant fused attention (K8V4): key 8-bit + value 4-bit packed KV.
 *
 * Performance path aligned with turboquant_attention_paged8bit:
 * - KV tiled GM->UB loads
 * - Vectorized Gather + Cube rotate decode (decode_device.h)
 * - GQA-grouped online softmax (attention_device.h)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#include "decode_device.h"
#include "attention_device.h"

namespace {

using namespace AscendC;

constexpr uint32_t TQ_HEAD = turboquant_attn::TQ_ATTN_HEAD;
constexpr uint32_t TQ_K_CODEBOOK = turboquant::TQ_DECODE_CODEBOOK_SIZE;
constexpr uint32_t TQ_V_CODEBOOK = 16;
constexpr uint32_t TQ_K_PACKED = turboquant::TQ_DECODE_PACKED_BYTES;
constexpr uint32_t TQ_V_INDEX_BYTES = 64;
constexpr uint32_t TQ_V_PACKED = TQ_V_INDEX_BYTES + 2;
constexpr uint32_t TQ_V_PACKED_STRIDE = (TQ_V_PACKED + 31U) / 32U * 32U;  // 66 -> 96 for MTE
constexpr uint32_t TQ_AIV_SUB = 2;
constexpr uint32_t TQ_ROT_WS = TQ_HEAD * TQ_HEAD * sizeof(half);
constexpr uint64_t TQ_CUBE_C_OFF = 256 * 1024;
constexpr uint32_t TQ_UB_KV_TILE_CAP = 32;
constexpr uint32_t TQ_UB_GQA_CAP = 8;

__aicore__ inline uint32_t AlignUp16(uint32_t x)
{
    return (x + 15U) / 16U * 16U;
}

__aicore__ inline uint32_t FindSeqForToken(
    uint32_t tokenIdx,
    const GlobalTensor<int32_t>& actualSeqLenQ,
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
class TurboquantFusedFiaK8v4Kernel {
public:
    __aicore__ inline TurboquantFusedFiaK8v4Kernel(TPipe* pipe, TqRotateMm* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm)
    {}

    __aicore__ inline void Init(
        __gm__ half* query,
        __gm__ uint8_t* keyCache,
        __gm__ uint8_t* valueCache,
        __gm__ int32_t* blockTable,
        __gm__ int32_t* actualSeqLenQ,
        __gm__ int32_t* actualSeqLenKv,
        __gm__ half* codebook,
        __gm__ half* rotation,
        __gm__ half* codebookValue,
        __gm__ half* rotationValue,
        __gm__ half* out,
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
        keyRowBytes_ = tiling_->keyRowBytes;
        valueRowBytes_ = tiling_->valueRowBytes;
        kvTileRows_ = tiling_->kvTileRows;
        usedCoreNum_ = tiling_->usedCoreNum;
        scaleValue_ = tiling_->scaleValue;
        (void)tiling_->cacheNormBf16;

        queryGm_.SetGlobalBuffer(query, static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);
        keyCacheGm_.SetGlobalBuffer(keyCache, 0);
        valueCacheGm_.SetGlobalBuffer(valueCache, 0);
        blockTableGm_.SetGlobalBuffer(blockTable,
                                      static_cast<uint64_t>(batchSize_) * maxBlocksPerSeq_);
        actualSeqLenQGm_.SetGlobalBuffer(actualSeqLenQ, batchSize_);
        actualSeqLenKvGm_.SetGlobalBuffer(actualSeqLenKv, batchSize_);
        codebookGm_.SetGlobalBuffer(codebook, TQ_K_CODEBOOK);
        codebookVGm_.SetGlobalBuffer(codebookValue, TQ_V_CODEBOOK);
        rotationGm_.SetGlobalBuffer(rotation, static_cast<uint64_t>(TQ_HEAD) * TQ_HEAD);
        rotationVGm_.SetGlobalBuffer(rotationValue, static_cast<uint64_t>(TQ_HEAD) * TQ_HEAD);
        outGm_.SetGlobalBuffer(out, static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);

        pipe_->InitBuffer(codebookBuf_, (TQ_K_CODEBOOK + TQ_V_CODEBOOK) * sizeof(half));
        pipe_->InitBuffer(packedBuf_,
                          TQ_UB_KV_TILE_CAP * TQ_HEAD +
                              TQ_UB_KV_TILE_CAP * static_cast<uint32_t>(sizeof(half)));
        pipe_->InitBuffer(valueRawBuf_, TQ_UB_KV_TILE_CAP * TQ_V_PACKED_STRIDE);
        pipe_->InitBuffer(xHatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(half));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_WS);
        pipe_->InitBuffer(expBuf_, 4 * sizeof(float));
        pipe_->InitBuffer(qGroupBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(half));
        pipe_->InitBuffer(qGroupFloatBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, TQ_UB_GQA_CAP * TQ_UB_KV_TILE_CAP * sizeof(float));
        pipe_->InitBuffer(mStateBuf_, TQ_UB_GQA_CAP * sizeof(float));
        pipe_->InitBuffer(sStateBuf_, TQ_UB_GQA_CAP * sizeof(float));
        pipe_->InitBuffer(outAccBuf_, TQ_UB_GQA_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(idxHalfBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(half));
        pipe_->InitBuffer(idxFloatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(float));
        pipe_->InitBuffer(idxS32Buf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(int32_t));
        pipe_->InitBuffer(yHatBuf_, TQ_UB_KV_TILE_CAP * TQ_HEAD * sizeof(half));

        auto* wsBase = reinterpret_cast<__gm__ uint8_t*>(GetSysWorkSpacePtr());
        matmulReady_ = (wsBase != nullptr);
        if (matmulReady_) {
            const uint32_t coreIdx = GetBlockIdx() / TQ_AIV_SUB;
            constexpr uint64_t kCubeCStride =
                static_cast<uint64_t>(TQ_UB_KV_TILE_CAP) * TQ_HEAD * sizeof(float);
            const uint64_t cubeCOff = TQ_CUBE_C_OFF + static_cast<uint64_t>(coreIdx) * kCubeCStride;
            cubeCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsBase + cubeCOff),
                                     static_cast<uint64_t>(TQ_UB_KV_TILE_CAP) * TQ_HEAD);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            return;
        }
        const uint32_t coreIdx = GetBlockIdx() / TQ_AIV_SUB;
        const bool isPrimaryAiv = ((GetSubBlockIdx() % TQ_AIV_SUB) == 0);
        if (!isPrimaryAiv || coreIdx >= usedCoreNum_) {
            return;
        }

        LoadCodebooks();
        const uint32_t start = TaskStart(coreIdx);
        const uint32_t end = TaskEnd(coreIdx);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t tokenIdx = task / numKvHeads_;
            const uint32_t kvHead = task % numKvHeads_;
            ComputeAttention(tokenIdx, kvHead);
        }
    }

private:
    __aicore__ inline void LoadCodebooks()
    {
        auto codebookLocal = codebookBuf_.Get<half>();
        DataCopy(codebookLocal, codebookGm_, TQ_K_CODEBOOK);
        DataCopy(codebookLocal[TQ_K_CODEBOOK], codebookVGm_, TQ_V_CODEBOOK);
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline uint32_t TaskStart(uint32_t coreIdx) const
    {
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
            return numTokens_ * numKvHeads_;
        }
        return TaskStart(coreIdx + 1);
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

    __aicore__ inline void LoadKeyTileRows(
        const LocalTensor<uint8_t>& packed,
        uint32_t mRows,
        uint32_t seqIdx,
        uint32_t kvHead,
        uint32_t kvStart)
    {
        AscendC::DataCopyExtParams copyParams{1, TQ_HEAD, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        for (uint32_t t = 0; t < mRows; ++t) {
            const uint32_t absPos = kvStart + t;
            const uint32_t blockId = static_cast<uint32_t>(blockTableGm_.GetValue(
                static_cast<uint64_t>(seqIdx) * maxBlocksPerSeq_ + (absPos / blockSize_)));
            const uint32_t posInBlock = absPos % blockSize_;
            const uint64_t srcOff = (((static_cast<uint64_t>(blockId) * blockSize_ + posInBlock) *
                                      numKvHeads_ +
                                      kvHead) *
                                     keyRowBytes_);
            AscendC::DataCopyPad(packed[t * TQ_HEAD], keyCacheGm_[srcOff], copyParams, padParams);
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
                                     keyRowBytes_);
            const uint32_t dstNorm = normBase + t * sizeof(half);
            packed.SetValue(dstNorm, keyCacheGm_.GetValue(srcOff + TQ_HEAD));
            packed.SetValue(dstNorm + 1, keyCacheGm_.GetValue(srcOff + TQ_HEAD + 1));
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void LoadValueTileRows(
        const LocalTensor<uint8_t>& packed,
        uint32_t mRows,
        uint32_t seqIdx,
        uint32_t kvHead,
        uint32_t kvStart)
    {
        auto valueRaw = valueRawBuf_.Get<uint8_t>();
        AscendC::DataCopyExtParams copyParams{1, TQ_V_PACKED, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        for (uint32_t t = 0; t < mRows; ++t) {
            const uint32_t absPos = kvStart + t;
            const uint32_t blockId = static_cast<uint32_t>(blockTableGm_.GetValue(
                static_cast<uint64_t>(seqIdx) * maxBlocksPerSeq_ + (absPos / blockSize_)));
            const uint32_t posInBlock = absPos % blockSize_;
            const uint64_t srcOff = (((static_cast<uint64_t>(blockId) * blockSize_ + posInBlock) *
                                      numKvHeads_ +
                                      kvHead) *
                                     valueRowBytes_);
            AscendC::DataCopyPad(valueRaw[t * TQ_V_PACKED_STRIDE], valueCacheGm_[srcOff],
                                 copyParams, padParams);
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_S>();

        const uint32_t normBase = mRows * TQ_HEAD;
        for (uint32_t t = 0; t < mRows; ++t) {
            const uint32_t rowBase = t * TQ_V_PACKED_STRIDE;
            const uint32_t dstBase = t * TQ_HEAD;
            for (uint32_t b = 0; b < TQ_V_INDEX_BYTES; ++b) {
                const uint8_t byteVal = valueRaw.GetValue(rowBase + b);
                packed.SetValue(dstBase + b, static_cast<uint8_t>(byteVal & 0x0Fu));
                packed.SetValue(dstBase + b + TQ_V_INDEX_BYTES,
                                static_cast<uint8_t>((byteVal >> 4) & 0x0Fu));
            }
            const uint32_t dstNorm = normBase + t * sizeof(half);
            packed.SetValue(dstNorm, valueRaw.GetValue(rowBase + TQ_V_INDEX_BYTES));
            packed.SetValue(dstNorm + 1, valueRaw.GetValue(rowBase + TQ_V_INDEX_BYTES + 1));
        }
        turboquant::TqDecodeSync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void LoadRotation(GlobalTensor<half>& rotationGm)
    {
        auto rotation = rotateWorkBuf_.Get<half>();
        DataCopy(rotation, rotationGm, TQ_HEAD * TQ_HEAD);
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_S>();
    }

    __aicore__ inline void DecodeKeyTile(
        LocalTensor<uint8_t> packedLocal,
        LocalTensor<half> codebookLocal,
        GlobalTensor<half>& rotationGm,
        LocalTensor<half>& xHat,
        uint32_t mRows)
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

    __aicore__ inline void DecodeValueTile(
        LocalTensor<uint8_t> packedLocal,
        LocalTensor<half> codebookLocal,
        GlobalTensor<half>& rotationGm,
        LocalTensor<half>& xHat,
        uint32_t mRows)
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
        turboquant::TqDecodeSync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(qGroup, qLocal, AscendC::RoundMode::CAST_NONE, gqaGroup_ * TQ_HEAD);
        turboquant::TqDecodeSync<AscendC::HardEvent::V_S>();
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

    __aicore__ inline void ComputeAttention(uint32_t tokenIdx, uint32_t kvHead)
    {
        const uint32_t seqIdx = FindSeqForToken(tokenIdx, actualSeqLenQGm_, batchSize_);
        const uint32_t kvLen = static_cast<uint32_t>(actualSeqLenKvGm_.GetValue(seqIdx));
        const uint32_t causalKvEnd = GetCausalKvEnd(tokenIdx, seqIdx);
        if (kvLen == 0 || causalKvEnd == 0) {
            WriteZeroOutput(tokenIdx, kvHead);
            return;
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

        for (uint32_t pos = 0; pos < causalKvEnd;) {
            const uint32_t tileRows =
                (pos + kvTileRows_ <= causalKvEnd) ? kvTileRows_ : (causalKvEnd - pos);
            LoadKeyTileRows(packedLocal, tileRows, seqIdx, kvHead, pos);
            LocalTensor<half> kTile = xHat;
            DecodeKeyTile(packedLocal, codebookLocal, rotationGm_, kTile, tileRows);

            auto scoreTile = scoreBuf_.Get<float>();
            turboquant_attn::VectorQk(qGroup, kTile, scoreTile, gqaGroup_, tileRows, scaleValue_);

            LoadValueTileRows(packedLocal, tileRows, seqIdx, kvHead, pos);
            LocalTensor<half> vTile = xHat;
            DecodeValueTile(packedLocal, codebookLocal[TQ_K_CODEBOOK], rotationVGm_, vTile, tileRows);

            turboquant_attn::OnlineSoftmaxUpdateTile(
                scoreTile, vTile, mState, sState, outAcc, expLocal, gqaGroup_, tileRows);
            pos += tileRows;
        }

        WriteFinalOutput(tokenIdx, kvHead, sState, outAcc);
    }

    TPipe* pipe_ = nullptr;
    TqRotateMm* rotateMm_ = nullptr;
    const TilingT* tiling_ = nullptr;
    bool matmulReady_ = false;

    uint32_t numTokens_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t gqaGroup_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t keyRowBytes_ = TQ_K_PACKED;
    uint32_t valueRowBytes_ = TQ_V_PACKED;
    uint32_t kvTileRows_ = 0;
    uint32_t usedCoreNum_ = 0;
    float scaleValue_ = 1.f;

    GlobalTensor<half> queryGm_;
    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> actualSeqLenQGm_;
    GlobalTensor<int32_t> actualSeqLenKvGm_;
    GlobalTensor<half> codebookGm_;
    GlobalTensor<half> codebookVGm_;
    GlobalTensor<half> rotationGm_;
    GlobalTensor<half> rotationVGm_;
    GlobalTensor<half> outGm_;
    GlobalTensor<float> cubeCGm_;

    TBuf<TPosition::VECCALC> codebookBuf_;
    TBuf<TPosition::VECCALC> packedBuf_;
    TBuf<TPosition::VECCALC> valueRawBuf_;
    TBuf<TPosition::VECCALC> xHatBuf_;
    TBuf<TPosition::VECCALC> rotateWorkBuf_;
    TBuf<TPosition::VECCALC> expBuf_;
    TBuf<TPosition::VECCALC> qGroupBuf_;
    TBuf<TPosition::VECCALC> qGroupFloatBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> mStateBuf_;
    TBuf<TPosition::VECCALC> sStateBuf_;
    TBuf<TPosition::VECCALC> outAccBuf_;
    TBuf<TPosition::VECCALC> idxHalfBuf_;
    TBuf<TPosition::VECCALC> idxFloatBuf_;
    TBuf<TPosition::VECCALC> idxS32Buf_;
    TBuf<TPosition::VECOUT> yHatBuf_;
};

#define INVOKE_TQ_K8V4_FIA_KERNEL()                                                               \
    do {                                                                                        \
        GET_TILING_DATA(tilingData, tiling);                                                    \
        TPipe pipe;                                                                             \
        TqRotateMm rotateMm;                                                                    \
        TCubeTiling decodeTiling = tilingData.decodeRotateTiling;                               \
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &decodeTiling);                \
        TurboquantFusedFiaK8v4Kernel<TurboquantFusedInferAttentionScoreK8v4TilingData> op(      \
            &pipe, &rotateMm);                                                                  \
        op.Init(reinterpret_cast<__gm__ half*>(query),                                          \
                reinterpret_cast<__gm__ uint8_t*>(key_cache),                                 \
                reinterpret_cast<__gm__ uint8_t*>(value_cache),                               \
                reinterpret_cast<__gm__ int32_t*>(block_table),                               \
                reinterpret_cast<__gm__ int32_t*>(actual_seq_len_q),                          \
                reinterpret_cast<__gm__ int32_t*>(actual_seq_len_kv),                         \
                reinterpret_cast<__gm__ half*>(codebook),                                     \
                reinterpret_cast<__gm__ half*>(rotation),                                       \
                reinterpret_cast<__gm__ half*>(codebook_value),                                 \
                reinterpret_cast<__gm__ half*>(rotation_value),                               \
                reinterpret_cast<__gm__ half*>(out),                                            \
                &tilingData);                                                                   \
        op.Process();                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void turboquant_fused_infer_attention_score_k8v4(
    GM_ADDR query,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR block_table,
    GM_ADDR actual_seq_len_q,
    GM_ADDR actual_seq_len_kv,
    GM_ADDR atten_mask,
    GM_ADDR codebook,
    GM_ADDR rotation,
    GM_ADDR codebook_value,
    GM_ADDR rotation_value,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)atten_mask;
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    INVOKE_TQ_K8V4_FIA_KERNEL();
}

}  // namespace
