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

#include "kernel_operator.h"
#include "types.h"

// Kernel1: packed uint8 -> y_hat fp16/bf16.
// y_hat[i] = codebook[idx_i] * norm, where idx_i are unpacked uint4.

namespace {

__aicore__ inline uint32_t TqAlignUp32(uint32_t x)
{
    return (x + 31U) / 32U * 32U;
}

template <AscendC::HardEvent EVT>
__aicore__ inline void TqUnpackSync()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(event);
    AscendC::WaitFlag<EVT>(event);
}

template <typename T>
class TurboquantUnpackLookupScale {
public:
    using PACKED_T = uint8_t;
    using CB_T = T;
    using OUT_T = T;

    static constexpr uint32_t CODEBOOK_SIZE = 16;

    __aicore__ inline explicit TurboquantUnpackLookupScale(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* packed, __gm__ void* codebook, __gm__ void* y_hat,
                                uint32_t nVec, uint32_t headSize, uint32_t packedBytes, uint32_t vecPerCore)
    {
        nVec_ = nVec;
        headSize_ = headSize;
        packedBytes_ = packedBytes;
        packedStride_ = TqAlignUp32(packedBytes_);
        vecPerCore_ = vecPerCore;

        packedGm_.SetGlobalBuffer((__gm__ PACKED_T*)packed, (uint64_t)nVec_ * packedBytes_);
        codebookGm_.SetGlobalBuffer((__gm__ CB_T*)codebook, CODEBOOK_SIZE);
        yHatGm_.SetGlobalBuffer((__gm__ OUT_T*)y_hat, (uint64_t)nVec_ * headSize_);

        pipe_->InitBuffer(codebookBuf_, CODEBOOK_SIZE * sizeof(CB_T));
        // Store fp16 norm bits as uint16, then reinterpret to half.
        pipe_->InitBuffer(normU16Buf_, 1 * sizeof(uint16_t));
        // packedBytes is 66 for D=128 4-bit rows; pad the UB buffer for DataCopyPad.
        pipe_->InitBuffer(packedBuf_, packedStride_ * sizeof(PACKED_T));
        pipe_->InitBuffer(yBuf_, headSize_ * sizeof(OUT_T));
        pipe_->InitBuffer(yFloatBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(normFloatBuf_, sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto codebookLocal = codebookBuf_.Get<T>();
        AscendC::DataCopy(codebookLocal, codebookGm_[0], CODEBOOK_SIZE);

        uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        uint32_t start = blockIdx * vecPerCore_;
        uint32_t end = start + vecPerCore_;
        if (end > nVec_) end = nVec_;

        for (uint32_t v = start; v < end; ++v) {
            DecodeOne(v, codebookLocal);
        }
    }

private:
    __aicore__ inline void DecodeOne(uint32_t vecIdx, AscendC::LocalTensor<T> codebookLocal)
    {
        auto packedLocal = packedBuf_.Get<uint8_t>();
        AscendC::DataCopyExtParams copyParams{1, packedBytes_, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(
            packedLocal, packedGm_[vecIdx * packedBytes_], copyParams, padParams);
        TqUnpackSync<AscendC::HardEvent::MTE2_S>();

        // Load norm bytes from local packed buffer and decode as T.
        // The 4-bit fused pack writes the norm in the same dtype as codebook/y_hat.
        uint8_t b0 = packedLocal.GetValue(headSize_ / 2);
        uint8_t b1 = packedLocal.GetValue(headSize_ / 2 + 1);
        uint16_t bits = (uint16_t)b0 | ((uint16_t)b1 << 8);
        auto normU16Local = normU16Buf_.Get<uint16_t>();
        normU16Local.SetValue(0, bits);
        auto normTensor = normU16Local.ReinterpretCast<T>();
        auto normFloat = normFloatBuf_.Get<float>();
        TqUnpackSync<AscendC::HardEvent::S_V>();
        AscendC::Cast(normFloat, normTensor, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::PipeBarrier<PIPE_V>();
        TqUnpackSync<AscendC::HardEvent::V_S>();
        const float norm = normFloat.GetValue(0);
        TqUnpackSync<AscendC::HardEvent::S_V>();

        auto yLocal = yBuf_.Get<T>();
        // Unpack uint4 indices from bytes and lookup codebook.
        const uint32_t halfHead = headSize_ / 2;
        for (uint32_t b = 0; b < halfHead; ++b) {
            uint8_t byte = packedLocal.GetValue(b);
            uint8_t lo = byte & 0x0F;
            uint8_t hi = (byte >> 4) & 0x0F;
            yLocal.SetValue(b, codebookLocal.GetValue(lo));
            yLocal.SetValue(b + halfHead, codebookLocal.GetValue(hi));
        }

        auto yFloat = yFloatBuf_.Get<float>();
        TqUnpackSync<AscendC::HardEvent::S_V>();
        AscendC::Cast(yFloat, yLocal, AscendC::RoundMode::CAST_NONE, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(yFloat, yFloat, norm, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        TqUnpackSync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(yHatGm_[vecIdx * headSize_], yLocal, headSize_);
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t headSize_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t packedStride_ = 0;
    uint32_t vecPerCore_ = 1;

    AscendC::GlobalTensor<PACKED_T> packedGm_;
    AscendC::GlobalTensor<CB_T> codebookGm_;
    AscendC::GlobalTensor<OUT_T> yHatGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normU16Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normFloatBuf_;
};

extern "C" __global__ __aicore__ void turboquant_unpack_lookup_scale_half(
    __gm__ void* packed,
    __gm__ void* codebook,
    __gm__ void* y_hat,
    uint32_t nVec,
    uint32_t headSize,
    uint32_t packedBytes,
    uint32_t vecPerCore)
{
    AscendC::TPipe pipe;
    TurboquantUnpackLookupScale<half> op(&pipe);
    op.Init(packed, codebook, y_hat, nVec, headSize, packedBytes, vecPerCore);
    op.Process();
}

extern "C" __global__ __aicore__ void turboquant_unpack_lookup_scale_bf16(
    __gm__ void* packed,
    __gm__ void* codebook,
    __gm__ void* y_hat,
    uint32_t nVec,
    uint32_t headSize,
    uint32_t packedBytes,
    uint32_t vecPerCore)
{
    AscendC::TPipe pipe;
    TurboquantUnpackLookupScale<bfloat16_t> op(&pipe);
    op.Init(packed, codebook, y_hat, nVec, headSize, packedBytes, vecPerCore);
    op.Process();
}

} // namespace

namespace vllm_ascend {

extern void turboquant_unpack_lookup_scale_impl(
    void *stream,
    void *packed,
    void *codebook,
    void *y_hat,
    uint32_t nVec,
    uint32_t headSize,
    uint32_t packedBytes,
    uint32_t vecPerCore,
    uint32_t dtypeCode)
{
    uint32_t blockDim = (nVec + vecPerCore - 1) / vecPerCore;
    if (dtypeCode == static_cast<uint32_t>(AscendType::BF16)) {
        turboquant_unpack_lookup_scale_bf16<<<blockDim, nullptr, stream>>>(
            packed, codebook, y_hat, nVec, headSize, packedBytes, vecPerCore);
    } else {
        turboquant_unpack_lookup_scale_half<<<blockDim, nullptr, stream>>>(
            packed, codebook, y_hat, nVec, headSize, packedBytes, vecPerCore);
    }
}

extern void turboquant_unpack_lookup_scale_impl(
    void *stream,
    void *packed,
    void *codebook,
    void *y_hat,
    uint32_t nVec,
    uint32_t headSize,
    uint32_t packedBytes,
    uint32_t vecPerCore)
{
    turboquant_unpack_lookup_scale_impl(
        stream, packed, codebook, y_hat, nVec, headSize, packedBytes, vecPerCore,
        static_cast<uint32_t>(AscendType::FP16));
}

} // namespace vllm_ascend
