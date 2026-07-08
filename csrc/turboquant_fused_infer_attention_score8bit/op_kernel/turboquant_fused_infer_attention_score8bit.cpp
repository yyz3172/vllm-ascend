/*
 * TurboQuant true fused attention (8-bit packed KV).
 *
 * Correctness-first baseline:
 * - Decode packed uint8 indices + fp16 norm -> y
 * - Apply rotation in-kernel
 * - Compute causal scaled dot-product attention + online softmax
 *
 * Constraints (v1):
 * - DecodeOnly
 * - fp16 query/output
 * - head_size == 128
 * - packed bytes per vector == head_size + 2 (uint8 indices + fp16 norm)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace {

static constexpr uint32_t TQ_CODEBOOK_SIZE = 256;
static constexpr uint32_t TQ_HEAD_SIZE = 128;
static constexpr uint32_t TQ_PACKED_BYTES = TQ_HEAD_SIZE + 2;
static constexpr uint32_t TQ_COPY_STRIDE = 160U;  // align_up(130, 32)

template <AscendC::HardEvent EVT>
__aicore__ inline void SyncHardEvent()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(event);
    AscendC::WaitFlag<EVT>(event);
}

__aicore__ inline void CopyPackedRowToLocal(AscendC::LocalTensor<uint8_t>& packedLocal,
                                            const AscendC::GlobalTensor<uint8_t>& cacheGm,
                                            uint64_t gmIdx,
                                            uint32_t logicalBytes)
{
    AscendC::DataCopyExtParams copyParams{1, logicalBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(packedLocal, cacheGm[gmIdx], copyParams, padParams);
    SyncHardEvent<AscendC::HardEvent::MTE2_S>();
}

__aicore__ inline uint32_t FindSeqForToken(uint32_t tokenIdx,
                                          const AscendC::GlobalTensor<int32_t>& actualSeqLenQ,
                                          uint32_t batch)
{
    // actualSeqLenQ is a prefix-sum (cum) array of length batch.
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t end = static_cast<uint32_t>(actualSeqLenQ.GetValue(b));
        if (tokenIdx < end) return b;
    }
    return batch - 1;
}

__aicore__ inline uint32_t SeqStartToken(uint32_t seqIdx,
                                        const AscendC::GlobalTensor<int32_t>& actualSeqLenQ)
{
    if (seqIdx == 0) return 0;
    return static_cast<uint32_t>(actualSeqLenQ.GetValue(seqIdx - 1));
}

template <typename TilingT>
class TurboquantFusedFia8bitKernel {
public:
    __aicore__ inline explicit TurboquantFusedFia8bitKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ half* query,
                               __gm__ int8_t* keyCache,
                               __gm__ int8_t* valueCache,
                               __gm__ int32_t* blockTable,
                               __gm__ int32_t* actualSeqLenQ,
                               __gm__ int32_t* actualSeqLenKv,
                               __gm__ half* attenMask,
                               __gm__ half* codebook,
                               __gm__ half* rotation,
                               __gm__ half* codebookValue,
                               __gm__ half* rotationValue,
                               __gm__ half* out,
                               const TilingT* tilingData,
                               __gm__ uint8_t* tilingGm)
    {
        (void)attenMask;   // v1: rely on causal bound from seq lens
        (void)tilingGm;
        tiling_ = tilingData;
        numTokens_ = tiling_->numTokens;
        batchSize_ = tiling_->batchSize;
        numHeads_ = tiling_->numHeads;
        numKvHeads_ = tiling_->numKvHeads;
        headSize_ = tiling_->headSize;
        blockSize_ = tiling_->blockSize;
        maxBlocksPerSeq_ = tiling_->maxBlocksPerSeq;
        scaleValue_ = tiling_->scaleValue;

        queryGm_.SetGlobalBuffer((__gm__ half*)query,
                                 static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);
        keyCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)keyCache, 0);
        valueCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)valueCache, 0);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable,
                                      static_cast<uint64_t>(batchSize_) * maxBlocksPerSeq_);
        actualSeqLenQGm_.SetGlobalBuffer((__gm__ int32_t*)actualSeqLenQ, batchSize_);
        actualSeqLenKvGm_.SetGlobalBuffer((__gm__ int32_t*)actualSeqLenKv, batchSize_);
        codebookGm_.SetGlobalBuffer((__gm__ half*)codebook, TQ_CODEBOOK_SIZE);
        codebookVGm_.SetGlobalBuffer((__gm__ half*)codebookValue, TQ_CODEBOOK_SIZE);
        rotationGm_.SetGlobalBuffer((__gm__ half*)rotation, TQ_HEAD_SIZE * TQ_HEAD_SIZE);
        rotationVGm_.SetGlobalBuffer((__gm__ half*)rotationValue, TQ_HEAD_SIZE * TQ_HEAD_SIZE);
        outGm_.SetGlobalBuffer((__gm__ half*)out,
                               static_cast<uint64_t>(numTokens_) * numHeads_ * headSize_);

        pipe_->InitBuffer(packedBuf_, TQ_COPY_STRIDE * sizeof(uint8_t));
        pipe_->InitBuffer(queryBuf_, TQ_HEAD_SIZE * sizeof(half));
        pipe_->InitBuffer(outBuf_, TQ_HEAD_SIZE * sizeof(half));
        pipe_->InitBuffer(expBuf_, 4 * sizeof(float));
        pipe_->InitBuffer(normU16Buf_, sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        // Each task maps to one (token, head).
        const uint32_t totalTasks = numTokens_ * numHeads_;
        const uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t start = (totalTasks * coreIdx) / coreNum;
        const uint32_t end = (totalTasks * (coreIdx + 1)) / coreNum;
        for (uint32_t task = start; task < end; ++task) {
            ComputeOne(task);
        }
    }

private:
    __aicore__ inline float ExpScalar(float x)
    {
        auto expLocal = expBuf_.Get<float>();
        expLocal.SetValue(0, x);
        SyncHardEvent<AscendC::HardEvent::S_V>();
        AscendC::Exp(expLocal, expLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();
        SyncHardEvent<AscendC::HardEvent::V_S>();
        return expLocal.GetValue(0);
    }

    __aicore__ inline void StoreOutput(uint64_t outOff, const float* values)
    {
        auto outLocal = outBuf_.Get<half>();
        for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) {
            outLocal.SetValue(d, static_cast<half>(values[d]));
        }
        SyncHardEvent<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(outGm_[outOff], outLocal, TQ_HEAD_SIZE);
        SyncHardEvent<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void LoadQuery(uint32_t tokenIdx, uint32_t headIdx, float qf[TQ_HEAD_SIZE])
    {
        auto qLocal = queryBuf_.Get<half>();
        const uint64_t qOff = (static_cast<uint64_t>(tokenIdx) * numHeads_ + headIdx) * headSize_;
        AscendC::DataCopy(qLocal, queryGm_[qOff], headSize_);
        SyncHardEvent<AscendC::HardEvent::MTE2_S>();
        for (uint32_t d = 0; d < headSize_; ++d) {
            qf[d] = static_cast<float>(qLocal.GetValue(d));
        }
    }

    __aicore__ inline void LoadPackedRow(const AscendC::GlobalTensor<uint8_t>& cache,
                                        uint32_t blockId,
                                        uint32_t posInBlock,
                                        uint32_t kvHead,
                                        AscendC::LocalTensor<uint8_t> packedLocal)
    {
        // Assume cache is laid out as [num_blocks, blockSize, numKvHeads, packedBytes].
        // packedBytes == headSize + 2.
        const uint64_t idx =
            (((static_cast<uint64_t>(blockId) * blockSize_ + posInBlock) * numKvHeads_ + kvHead) * TQ_PACKED_BYTES);
        CopyPackedRowToLocal(packedLocal, cache, idx, TQ_PACKED_BYTES);
    }

    __aicore__ inline void DecodeRotateDot(const AscendC::LocalTensor<uint8_t>& packedLocal,
                                          const AscendC::GlobalTensor<half>& codebook,
                                          const AscendC::GlobalTensor<half>& rotation,
                                          const float qf[TQ_HEAD_SIZE],
                                          float& scoreOut)
    {
        // Decode y[j] = codebook[idx] * norm.
        auto normU16Local = normU16Buf_.Get<uint16_t>();
        uint8_t b0 = packedLocal.GetValue(TQ_HEAD_SIZE);
        uint8_t b1 = packedLocal.GetValue(TQ_HEAD_SIZE + 1);
        uint16_t bits = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
        normU16Local.SetValue(0, bits);
        auto normHalfTensor = normU16Local.ReinterpretCast<half>();
        const float norm = static_cast<float>(normHalfTensor.GetValue(0));

        // Compute k = y @ rotation; then dot(q, k).
        // k[d] = sum_j (codebook[idx_j] * norm) * rotation[j, d]
        float dot = 0.0f;
        for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) {
            float kd = 0.0f;
            for (uint32_t j = 0; j < TQ_HEAD_SIZE; ++j) {
                const uint8_t idx = packedLocal.GetValue(j);
                const float yj = static_cast<float>(codebook.GetValue(idx)) * norm;
                const float r = static_cast<float>(rotation.GetValue(j * TQ_HEAD_SIZE + d));
                kd += yj * r;
            }
            dot += qf[d] * kd;
        }
        scoreOut = dot * scaleValue_;
    }

    __aicore__ inline void DecodeRotateValue(const AscendC::LocalTensor<uint8_t>& packedLocal,
                                            const AscendC::GlobalTensor<half>& codebook,
                                            const AscendC::GlobalTensor<half>& rotation,
                                            float vOut[TQ_HEAD_SIZE])
    {
        auto normU16Local = normU16Buf_.Get<uint16_t>();
        uint8_t b0 = packedLocal.GetValue(TQ_HEAD_SIZE);
        uint8_t b1 = packedLocal.GetValue(TQ_HEAD_SIZE + 1);
        uint16_t bits = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
        normU16Local.SetValue(0, bits);
        auto normHalfTensor = normU16Local.ReinterpretCast<half>();
        const float norm = static_cast<float>(normHalfTensor.GetValue(0));

        for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) {
            float vd = 0.0f;
            for (uint32_t j = 0; j < TQ_HEAD_SIZE; ++j) {
                const uint8_t idx = packedLocal.GetValue(j);
                const float yj = static_cast<float>(codebook.GetValue(idx)) * norm;
                const float r = static_cast<float>(rotation.GetValue(j * TQ_HEAD_SIZE + d));
                vd += yj * r;
            }
            vOut[d] = vd;
        }
    }

    __aicore__ inline void ComputeOne(uint32_t taskIdx)
    {
        const uint32_t tokenIdx = taskIdx / numHeads_;
        const uint32_t headIdx = taskIdx % numHeads_;
        const uint32_t seqIdx = FindSeqForToken(tokenIdx, actualSeqLenQGm_, batchSize_);
        const uint32_t tokenStart = SeqStartToken(seqIdx, actualSeqLenQGm_);
        (void)tokenStart;

        // DecodeOnly: one query token per seq; tokenIdx corresponds to seq order.
        const uint32_t kvLen = static_cast<uint32_t>(actualSeqLenKvGm_.GetValue(seqIdx));
        const uint64_t outOff =
            (static_cast<uint64_t>(tokenIdx) * numHeads_ + headIdx) * headSize_;
        if (kvLen == 0) {
            float zeros[TQ_HEAD_SIZE];
            for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) zeros[d] = 0.0f;
            StoreOutput(outOff, zeros);
            return;
        }

        float qf[TQ_HEAD_SIZE];
        LoadQuery(tokenIdx, headIdx, qf);

        float m = -3.402823466e+38f;  // -FLT_MAX
        float s = 0.0f;
        float outAcc[TQ_HEAD_SIZE];
        for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) outAcc[d] = 0.0f;

        const uint32_t gqaGroup = numHeads_ / numKvHeads_;
        const uint32_t kvHead = headIdx / gqaGroup;
        const uint32_t lastPos = kvLen - 1;  // causal bound for decode token

        auto packedLocal = packedBuf_.Get<uint8_t>();

        for (uint32_t pos = 0; pos <= lastPos; ++pos) {
            const uint32_t blockId = static_cast<uint32_t>(
                blockTableGm_.GetValue(static_cast<uint64_t>(seqIdx) * maxBlocksPerSeq_ + (pos / blockSize_)));
            const uint32_t posInBlock = pos % blockSize_;

            LoadPackedRow(keyCacheGm_, blockId, posInBlock, kvHead, packedLocal);
            float score;
            DecodeRotateDot(packedLocal, codebookGm_, rotationGm_, qf, score);

            const float mNew = (score > m) ? score : m;
            const float alpha = (mNew > m) ? ExpScalar(m - mNew) : 1.0f;
            const float beta = ExpScalar(score - mNew);

            // Decode V and update output accumulator in a single streaming pass.
            LoadPackedRow(valueCacheGm_, blockId, posInBlock, kvHead, packedLocal);
            float vVec[TQ_HEAD_SIZE];
            DecodeRotateValue(packedLocal, codebookVGm_, rotationVGm_, vVec);

            for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) {
                outAcc[d] = outAcc[d] * alpha + vVec[d] * beta;
            }
            s = s * alpha + beta;
            m = mNew;
        }

        if (!(s > 0.0f)) {
            for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) outAcc[d] = 0.0f;
            StoreOutput(outOff, outAcc);
            return;
        }
        const float invS = 1.0f / s;
        for (uint32_t d = 0; d < TQ_HEAD_SIZE; ++d) {
            outAcc[d] *= invS;
        }
        StoreOutput(outOff, outAcc);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    const TilingT* tiling_ = nullptr;
    uint32_t numTokens_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    float scaleValue_ = 1.0f;

    AscendC::GlobalTensor<half> queryGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> actualSeqLenQGm_;
    AscendC::GlobalTensor<int32_t> actualSeqLenKvGm_;
    AscendC::GlobalTensor<half> codebookGm_;
    AscendC::GlobalTensor<half> rotationGm_;
    AscendC::GlobalTensor<half> codebookVGm_;
    AscendC::GlobalTensor<half> rotationVGm_;
    AscendC::GlobalTensor<half> outGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> queryBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normU16Buf_;
};

extern "C" __global__ __aicore__ void turboquant_fused_infer_attention_score8bit(
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
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    TurboquantFusedFia8bitKernel<TurboquantFusedInferAttentionScore8bitTilingData> op(&pipe);
    op.Init(reinterpret_cast<__gm__ half*>(query),
            reinterpret_cast<__gm__ int8_t*>(key_cache),
            reinterpret_cast<__gm__ int8_t*>(value_cache),
            reinterpret_cast<__gm__ int32_t*>(block_table),
            reinterpret_cast<__gm__ int32_t*>(actual_seq_len_q),
            reinterpret_cast<__gm__ int32_t*>(actual_seq_len_kv),
            reinterpret_cast<__gm__ half*>(atten_mask),
            reinterpret_cast<__gm__ half*>(codebook),
            reinterpret_cast<__gm__ half*>(rotation),
            reinterpret_cast<__gm__ half*>(codebook_value),
            reinterpret_cast<__gm__ half*>(rotation_value),
            reinterpret_cast<__gm__ half*>(out),
            &tilingData,
            reinterpret_cast<__gm__ uint8_t*>(tiling));
    op.Process();
}

}  // namespace

