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

// Fused TurboQuant pack for KV cache (correctness-first reference kernel).
//
// Scope (initial):
// - bits_key == bits_value == 8
// - head_size == 128
// - key/value dtype fp16
//
// Stack budget: do NOT cache full rotation [128,128] on stack (32 KiB).
// Read codebook / rotation from GM on demand.

#include "kernel_operator.h"
#include "types.h"

namespace {

static constexpr int TQ_PACK_D = 128;
static constexpr int TQ_PACK_K = 256;

__aicore__ inline float half_to_float(half h) { return (float)h; }

__aicore__ inline half float_to_half(float f) { return (half)f; }

__aicore__ inline float abs_f32(float x) { return x < 0.f ? -x : x; }

__aicore__ inline float sqrt_f32(float x) {
    if (x <= 0.f) {
        return 0.f;
    }
    float g = x;
    for (int i = 0; i < 6; ++i) {
        g = 0.5f * (g + x / g);
    }
    return g;
}

__aicore__ inline uint8_t nearest_codebook_index_gm(
    const AscendC::GlobalTensor<half> &codebookGm, half y) {
    float best = 1e30f;
    uint8_t best_k = 0;
    float yf = half_to_float(y);
    for (int k = 0; k < TQ_PACK_K; ++k) {
        float ck = half_to_float(codebookGm.GetValue(k));
        float d = abs_f32(yf - ck);
        if (d < best) {
            best = d;
            best_k = (uint8_t)k;
        }
    }
    return best_k;
}

// y[j] = sum_i x_unit[i] * rotation_t[i, j]; rotation_t in GM row-major [D,D].
__aicore__ inline void rotate_unit_vector_gm(
    const half* x_unit,
    const AscendC::GlobalTensor<half>& rotationTGm,
    half* y_out) {
    for (int j = 0; j < TQ_PACK_D; ++j) {
        float acc = 0.f;
        for (int i = 0; i < TQ_PACK_D; ++i) {
            acc += half_to_float(x_unit[i]) *
                   half_to_float(rotationTGm.GetValue((uint32_t)(i * TQ_PACK_D + j)));
        }
        y_out[j] = float_to_half(acc);
    }
}

// Pack one vector from GM input row -> GM packed row (indices + norm bytes + pad).
__aicore__ inline void pack_vector_gm_to_gm(
    const AscendC::GlobalTensor<half> &xGm,
    uint64_t x_base,
    const AscendC::GlobalTensor<half> &codebookGm,
    const AscendC::GlobalTensor<half> &rotationTGm,
    AscendC::GlobalTensor<uint8_t> &packedGm,
    uint64_t out_base,
    uint32_t slot_w) {
    float sumsq = 0.f;
    for (int i = 0; i < TQ_PACK_D; ++i) {
        float xf = half_to_float(xGm.GetValue(x_base + (uint32_t)i));
        sumsq += xf * xf;
    }
    float norm = sqrt_f32(sumsq);
    float inv = 1.0f / (norm + 1e-10f);

    half x_unit[TQ_PACK_D];
    for (int i = 0; i < TQ_PACK_D; ++i) {
        float xf = half_to_float(xGm.GetValue(x_base + (uint32_t)i));
        x_unit[i] = float_to_half(xf * inv);
    }

    half y[TQ_PACK_D];
    rotate_unit_vector_gm(x_unit, rotationTGm, y);

    for (int j = 0; j < TQ_PACK_D; ++j) {
        packedGm.SetValue(out_base + (uint32_t)j, nearest_codebook_index_gm(codebookGm, y[j]));
    }

    half norm_h = float_to_half(norm);
    uint8_t* nb = reinterpret_cast<uint8_t*>(&norm_h);
    packedGm.SetValue(out_base + (uint32_t)TQ_PACK_D, nb[0]);
    packedGm.SetValue(out_base + (uint32_t)TQ_PACK_D + 1, nb[1]);

    for (uint32_t k = (uint32_t)TQ_PACK_D + 2; k < slot_w; ++k) {
        packedGm.SetValue(out_base + k, (uint8_t)0);
    }
}

class TurboquantPackKVForCacheFused {
public:
    __aicore__ inline explicit TurboquantPackKVForCacheFused(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ half* key,
        __gm__ half* value,
        __gm__ half* codebook,
        __gm__ half* rotation_t,
        __gm__ uint8_t* packed_k,
        __gm__ uint8_t* packed_v,
        uint32_t nVec,
        uint32_t slot_w_k,
        uint32_t slot_w_v,
        uint32_t vecPerCore) {
        nVec_ = nVec;
        slot_w_k_ = slot_w_k;
        slot_w_v_ = slot_w_v;
        vecPerCore_ = vecPerCore;

        keyGm_.SetGlobalBuffer(key, (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(value, (uint64_t)nVec_ * TQ_PACK_D);
        codebookGm_.SetGlobalBuffer(codebook, TQ_PACK_K);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        packedKGm_.SetGlobalBuffer(packed_k, (uint64_t)nVec_ * slot_w_k_);
        packedVGm_.SetGlobalBuffer(packed_v, (uint64_t)nVec_ * slot_w_v_);
    }

    __aicore__ inline void Process() {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t start = core * vecPerCore_;
        const uint32_t end = start + vecPerCore_;

        for (uint32_t v = start; v < end && v < nVec_; ++v) {
            const uint64_t x_base = (uint64_t)v * TQ_PACK_D;
            const uint64_t out_k = (uint64_t)v * slot_w_k_;
            const uint64_t out_v = (uint64_t)v * slot_w_v_;

            pack_vector_gm_to_gm(
                keyGm_, x_base, codebookGm_, rotationTGm_, packedKGm_, out_k, slot_w_k_);
            pack_vector_gm_to_gm(
                valueGm_, x_base, codebookGm_, rotationTGm_, packedVGm_, out_v, slot_w_v_);
        }
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t slot_w_k_ = 0;
    uint32_t slot_w_v_ = 0;
    uint32_t vecPerCore_ = 1;

    AscendC::GlobalTensor<half> keyGm_;
    AscendC::GlobalTensor<half> valueGm_;
    AscendC::GlobalTensor<half> codebookGm_;
    AscendC::GlobalTensor<half> rotationTGm_;
    AscendC::GlobalTensor<uint8_t> packedKGm_;
    AscendC::GlobalTensor<uint8_t> packedVGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache_fused_fp16_8bit_128(
    __gm__ half* key,
    __gm__ half* value,
    __gm__ half* codebook,
    __gm__ half* rotation_t,
    __gm__ uint8_t* packed_k,
    __gm__ uint8_t* packed_v,
    uint32_t nVec,
    uint32_t slot_w_k,
    uint32_t slot_w_v,
    uint32_t vecPerCore) {
    AscendC::TPipe pipe;
    TurboquantPackKVForCacheFused op(&pipe);
    op.Init(key, value, codebook, rotation_t, packed_k, packed_v, nVec, slot_w_k, slot_w_v, vecPerCore);
    op.Process();
}

namespace vllm_ascend {

extern void turboquant_pack_kv_for_cache_fused_fp16_8bit_128_impl(
    void* stream,
    void* key,
    void* value,
    void* codebook,
    void* rotation_t,
    void* packed_k,
    void* packed_v,
    uint32_t nVec,
    uint32_t slot_w_k,
    uint32_t slot_w_v,
    uint32_t vecPerCore) {
    uint32_t blockDim = (nVec + vecPerCore - 1) / vecPerCore;
    turboquant_pack_kv_for_cache_fused_fp16_8bit_128<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<__gm__ half*>(key),
        reinterpret_cast<__gm__ half*>(value),
        reinterpret_cast<__gm__ half*>(codebook),
        reinterpret_cast<__gm__ half*>(rotation_t),
        reinterpret_cast<__gm__ uint8_t*>(packed_k),
        reinterpret_cast<__gm__ uint8_t*>(packed_v),
        nVec,
        slot_w_k,
        slot_w_v,
        vecPerCore);
}

}  // namespace vllm_ascend
