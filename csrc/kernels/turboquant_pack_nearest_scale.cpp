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

// AscendC aicore: libc `fabsf` is often unavailable / wrong overload for device code.
#define TURBOQUANT_ABS_F32(x) ((x) < 0.f ? -(x) : (x))

// Encode TurboQuant head vectors (MSE quant path): rotated unit vector y in fp16,
// nearest scalar centroid per dimension (L1 / absolute error), pack indices,
// append fp16 norm as little-endian bytes (matches Python pack layout).
// 4-bit: nibble-packed indices; 8-bit: one uint8 index per dimension.
//
// Layout mirrors turboquant_unpack_lookup_scale.cpp: anonymous namespace + single
// `extern "C"` kernel entry. 4- vs 8-bit share one launch symbol; `pack_mode` selects path.

namespace {

class TurboquantPackNearestScale {
public:
    using Y_T = half;
    using CB_T = half;
    using NORM_T = half;
    using OUT_T = uint8_t;

    static constexpr uint32_t CODEBOOK_SIZE = 16;

    __aicore__ inline explicit TurboquantPackNearestScale(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ half *y,
                                __gm__ half *codebook,
                                __gm__ half *norms_fp16,
                                __gm__ uint8_t *packed,
                                uint32_t nVec,
                                uint32_t headSize,
                                uint32_t packedBytes,
                                uint32_t vecPerCore)
    {
        nVec_ = nVec;
        headSize_ = headSize;
        packedBytes_ = packedBytes;
        vecPerCore_ = vecPerCore;

        yGm_.SetGlobalBuffer(y, (uint64_t)nVec_ * headSize_);
        codebookGm_.SetGlobalBuffer(codebook, CODEBOOK_SIZE);
        normsGm_.SetGlobalBuffer(norms_fp16, (uint64_t)nVec_);
        packedGm_.SetGlobalBuffer(packed, (uint64_t)nVec_ * packedBytes_);

        pipe_->InitBuffer(codebookBuf_, CODEBOOK_SIZE * sizeof(CB_T));
        pipe_->InitBuffer(normHalfBuf_, sizeof(half));
        pipe_->InitBuffer(yBuf_, headSize_ * sizeof(Y_T));
        pipe_->InitBuffer(packedBuf_, packedBytes_ * sizeof(OUT_T));
    }

    __aicore__ inline void Process()
    {
        auto codebookLocal = codebookBuf_.Get<half>();
        AscendC::DataCopy(codebookLocal, codebookGm_[0], CODEBOOK_SIZE);

        uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        uint32_t start = blockIdx * vecPerCore_;
        uint32_t end = start + vecPerCore_;
        if (end > nVec_) {
            end = nVec_;
        }

        for (uint32_t v = start; v < end; ++v) {
            EncodeOne(v, codebookLocal);
        }
    }

private:
    __aicore__ inline static uint8_t ArgminAbsL1(float yf, AscendC::LocalTensor<half> cbLocal)
    {
        float best = TURBOQUANT_ABS_F32(yf - static_cast<float>(cbLocal.GetValue(0)));
        uint8_t bestIdx = 0;
        for (uint32_t k = 1; k < CODEBOOK_SIZE; ++k) {
            float d = TURBOQUANT_ABS_F32(yf - static_cast<float>(cbLocal.GetValue(k)));
            if (d < best) {
                best = d;
                bestIdx = static_cast<uint8_t>(k);
            }
        }
        return bestIdx;
    }

    __aicore__ inline void EncodeOne(uint32_t vecIdx, AscendC::LocalTensor<half> codebookLocal)
    {
        auto yLocal = yBuf_.Get<half>();
        AscendC::DataCopy(yLocal, yGm_[vecIdx * headSize_], headSize_);

        auto packedLocal = packedBuf_.Get<uint8_t>();

        for (uint32_t b = 0; b < headSize_ / 2; ++b) {
            float y0 = (float)yLocal.GetValue(b * 2);
            float y1 = (float)yLocal.GetValue(b * 2 + 1);
            uint8_t hi = ArgminAbsL1(y0, codebookLocal);
            uint8_t lo = ArgminAbsL1(y1, codebookLocal);
            packedLocal.SetValue(b, static_cast<uint8_t>((hi << 4) | (lo & 0x0F)));
        }

        auto normHalfLocal = normHalfBuf_.Get<half>();
        AscendC::DataCopy(normHalfLocal, normsGm_[vecIdx], 1);
        union {
            half h;
            uint16_t u;
        } normBits {};
        normBits.h = normHalfLocal.GetValue(0);
        packedLocal.SetValue(headSize_ / 2, static_cast<uint8_t>(normBits.u & 0xFFu));
        packedLocal.SetValue(headSize_ / 2 + 1, static_cast<uint8_t>((normBits.u >> 8) & 0xFFu));

        AscendC::DataCopy(packedGm_[vecIdx * packedBytes_], packedLocal, packedBytes_);
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t headSize_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t vecPerCore_ = 1;

    AscendC::GlobalTensor<Y_T> yGm_;
    AscendC::GlobalTensor<CB_T> codebookGm_;
    AscendC::GlobalTensor<NORM_T> normsGm_;
    AscendC::GlobalTensor<OUT_T> packedGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
};

class TurboquantPackNearestScale8 {
public:
    using Y_T = half;
    using CB_T = half;
    using NORM_T = half;
    using OUT_T = uint8_t;

    static constexpr uint32_t CODEBOOK_SIZE = 256;

    __aicore__ inline explicit TurboquantPackNearestScale8(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ half *y,
                                __gm__ half *codebook,
                                __gm__ half *norms_fp16,
                                __gm__ uint8_t *packed,
                                uint32_t nVec,
                                uint32_t headSize,
                                uint32_t packedBytes,
                                uint32_t vecPerCore)
    {
        nVec_ = nVec;
        headSize_ = headSize;
        packedBytes_ = packedBytes;
        vecPerCore_ = vecPerCore;

        yGm_.SetGlobalBuffer(y, (uint64_t)nVec_ * headSize_);
        codebookGm_.SetGlobalBuffer(codebook, CODEBOOK_SIZE);
        normsGm_.SetGlobalBuffer(norms_fp16, (uint64_t)nVec_);
        packedGm_.SetGlobalBuffer(packed, (uint64_t)nVec_ * packedBytes_);

        pipe_->InitBuffer(codebookBuf_, CODEBOOK_SIZE * sizeof(CB_T));
        pipe_->InitBuffer(normHalfBuf_, sizeof(half));
        pipe_->InitBuffer(yBuf_, headSize_ * sizeof(Y_T));
        pipe_->InitBuffer(packedBuf_, packedBytes_ * sizeof(OUT_T));
    }

    __aicore__ inline void Process()
    {
        auto codebookLocal = codebookBuf_.Get<half>();
        AscendC::DataCopy(codebookLocal, codebookGm_[0], CODEBOOK_SIZE);

        uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        uint32_t start = blockIdx * vecPerCore_;
        uint32_t end = start + vecPerCore_;
        if (end > nVec_) {
            end = nVec_;
        }

        for (uint32_t v = start; v < end; ++v) {
            EncodeOne(v, codebookLocal);
        }
    }

private:
    __aicore__ inline static uint8_t ArgminAbsL1(float yf, AscendC::LocalTensor<half> cbLocal)
    {
        float best = TURBOQUANT_ABS_F32(yf - static_cast<float>(cbLocal.GetValue(0)));
        uint8_t bestIdx = 0;
        for (uint32_t k = 1; k < CODEBOOK_SIZE; ++k) {
            float d = TURBOQUANT_ABS_F32(yf - static_cast<float>(cbLocal.GetValue(k)));
            if (d < best) {
                best = d;
                bestIdx = static_cast<uint8_t>(k);
            }
        }
        return bestIdx;
    }

    __aicore__ inline void EncodeOne(uint32_t vecIdx, AscendC::LocalTensor<half> codebookLocal)
    {
        auto yLocal = yBuf_.Get<half>();
        AscendC::DataCopy(yLocal, yGm_[vecIdx * headSize_], headSize_);

        auto packedLocal = packedBuf_.Get<uint8_t>();

        for (uint32_t j = 0; j < headSize_; ++j) {
            float yf = (float)yLocal.GetValue(j);
            packedLocal.SetValue(j, ArgminAbsL1(yf, codebookLocal));
        }

        auto normHalfLocal = normHalfBuf_.Get<half>();
        AscendC::DataCopy(normHalfLocal, normsGm_[vecIdx], 1);
        union {
            half h;
            uint16_t u;
        } normBits {};
        normBits.h = normHalfLocal.GetValue(0);
        packedLocal.SetValue(headSize_, static_cast<uint8_t>(normBits.u & 0xFFu));
        packedLocal.SetValue(headSize_ + 1, static_cast<uint8_t>((normBits.u >> 8) & 0xFFu));

        AscendC::DataCopy(packedGm_[vecIdx * packedBytes_], packedLocal, packedBytes_);
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t headSize_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t vecPerCore_ = 1;

    AscendC::GlobalTensor<Y_T> yGm_;
    AscendC::GlobalTensor<CB_T> codebookGm_;
    AscendC::GlobalTensor<NORM_T> normsGm_;
    AscendC::GlobalTensor<OUT_T> packedGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
};

// pack_mode: 0 = 4-bit nibble pack, 1 = 8-bit one index per dim.
extern "C" __global__ __aicore__ void turboquant_pack_nearest_scale_half(__gm__ half* y,
                                                                         __gm__ half* codebook,
                                                                         __gm__ half* norms_fp16,
                                                                         __gm__ uint8_t* packed,
                                                                         uint32_t nVec,
                                                                         uint32_t headSize,
                                                                         uint32_t packedBytes,
                                                                         uint32_t vecPerCore,
                                                                         uint32_t pack_mode)
{
    if (pack_mode == 0) {
        AscendC::TPipe pipe;
        TurboquantPackNearestScale op(&pipe);
        op.Init(y, codebook, norms_fp16, packed, nVec, headSize, packedBytes, vecPerCore);
        op.Process();
    } else {
        AscendC::TPipe pipe8;
        TurboquantPackNearestScale8 op8(&pipe8);
        op8.Init(y, codebook, norms_fp16, packed, nVec, headSize, packedBytes, vecPerCore);
        op8.Process();
    }
}

} // namespace

namespace vllm_ascend {

extern void turboquant_pack_nearest_scale_impl(void *stream,
                                               void *y,
                                               void *codebook,
                                               void *norms_fp16,
                                               void *packed,
                                               uint32_t nVec,
                                               uint32_t headSize,
                                               uint32_t packedBytes,
                                               uint32_t vecPerCore)
{
    uint32_t blockDim = (nVec + vecPerCore - 1) / vecPerCore;
    turboquant_pack_nearest_scale_half<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<__gm__ half*>(y),
        reinterpret_cast<__gm__ half*>(codebook),
        reinterpret_cast<__gm__ half*>(norms_fp16),
        reinterpret_cast<__gm__ uint8_t*>(packed),
        nVec,
        headSize,
        packedBytes,
        vecPerCore,
        0u);
}

extern void turboquant_pack_nearest_scale_8bit_impl(void *stream,
                                                    void *y,
                                                    void *codebook,
                                                    void *norms_fp16,
                                                    void *packed,
                                                    uint32_t nVec,
                                                    uint32_t headSize,
                                                    uint32_t packedBytes,
                                                    uint32_t vecPerCore)
{
    uint32_t blockDim = (nVec + vecPerCore - 1) / vecPerCore;
    turboquant_pack_nearest_scale_half<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<__gm__ half*>(y),
        reinterpret_cast<__gm__ half*>(codebook),
        reinterpret_cast<__gm__ half*>(norms_fp16),
        reinterpret_cast<__gm__ uint8_t*>(packed),
        nVec,
        headSize,
        packedBytes,
        vecPerCore,
        1u);
}

} // namespace vllm_ascend
