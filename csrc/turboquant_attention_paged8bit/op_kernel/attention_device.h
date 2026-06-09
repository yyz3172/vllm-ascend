/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "kernel_operator.h"

namespace turboquant_attn {

static constexpr uint32_t TQ_ATTN_HEAD = 128;

__aicore__ inline float ExpScalar(AscendC::LocalTensor<float>& expBuf, float x)
{
    expBuf.SetValue(0, x);
    AscendC::Exp(expBuf, expBuf, 1);
    return expBuf.GetValue(0);
}

// Vector QK: Q_group[G,128] dot K_tile[M,128] -> score[G,M]
__aicore__ inline void VectorQk(
    AscendC::LocalTensor<float> qGroup,
    const AscendC::LocalTensor<half>& kTile,
    AscendC::LocalTensor<float> scoreOut,
    uint32_t gqaGroup,
    uint32_t mRows,
    float scale)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        for (uint32_t m = 0; m < mRows; ++m) {
            float dot = 0.f;
            for (uint32_t d = 0; d < TQ_ATTN_HEAD; ++d) {
                dot += qGroup.GetValue(g * TQ_ATTN_HEAD + d) *
                       static_cast<float>(kTile.GetValue(m * TQ_ATTN_HEAD + d));
            }
            scoreOut.SetValue(g * mRows + m, dot * scale);
        }
    }
}

// Vector PV: prob[G,M] dot V_tile[M,128] -> out[G,128]
__aicore__ inline void VectorPv(
    AscendC::LocalTensor<float> prob,
    const AscendC::LocalTensor<half>& vTile,
    AscendC::LocalTensor<float> outGroup,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        for (uint32_t d = 0; d < TQ_ATTN_HEAD; ++d) {
            float acc = 0.f;
            for (uint32_t m = 0; m < mRows; ++m) {
                acc += prob.GetValue(g * mRows + m) *
                       static_cast<float>(vTile.GetValue(m * TQ_ATTN_HEAD + d));
            }
            outGroup.SetValue(g * TQ_ATTN_HEAD + d, acc);
        }
    }
}

// Online softmax update for one KV tile (per G head).
__aicore__ inline void OnlineSoftmaxUpdateTile(
    AscendC::LocalTensor<float> scoreTile,
    const AscendC::LocalTensor<half>& vTile,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        for (uint32_t m = 0; m < mRows; ++m) {
            const float score = scoreTile.GetValue(g * mRows + m);
            const float oldM = mState.GetValue(g);
            const float oldS = sState.GetValue(g);
            const float mNew = (score > oldM) ? score : oldM;
            const float alpha = (mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f;
            const float beta = ExpScalar(expBuf, score - mNew);

            for (uint32_t d = 0; d < TQ_ATTN_HEAD; ++d) {
                const float vVal =
                    static_cast<float>(vTile.GetValue(m * TQ_ATTN_HEAD + d));
                const uint32_t outIdx = g * TQ_ATTN_HEAD + d;
                outAcc.SetValue(outIdx, outAcc.GetValue(outIdx) * alpha + vVal * beta);
            }
            sState.SetValue(g, oldS * alpha + beta);
            mState.SetValue(g, mNew);
        }
    }
}

// FlashDecode combine: merge partial (max, sum, accum) across kv segments.
__aicore__ inline void FlashDecodeCombineHead(
    float* globalM,
    float* globalS,
    float* globalOut,
    const float* partialM,
    const float* partialS,
    const float* partialOut,
    uint32_t numParts,
    AscendC::LocalTensor<float>& expBuf)
{
    float bestM = partialM[0];
    for (uint32_t p = 1; p < numParts; ++p) {
        if (partialM[p] > bestM) {
            bestM = partialM[p];
        }
    }
    float sumS = 0.f;
    float outAcc[TQ_ATTN_HEAD];
    for (uint32_t d = 0; d < TQ_ATTN_HEAD; ++d) {
        outAcc[d] = 0.f;
    }
    for (uint32_t p = 0; p < numParts; ++p) {
        const float w = ExpScalar(expBuf, partialM[p] - bestM) * partialS[p];
        sumS += w;
        for (uint32_t d = 0; d < TQ_ATTN_HEAD; ++d) {
            outAcc[d] += w * partialOut[p * TQ_ATTN_HEAD + d];
        }
    }
    const float invS = (sumS > 0.f) ? (1.f / sumS) : 0.f;
    globalM[0] = bestM;
    globalS[0] = sumS;
    for (uint32_t d = 0; d < TQ_ATTN_HEAD; ++d) {
        globalOut[d] = outAcc[d] * invS;
    }
}

}  // namespace turboquant_attn
