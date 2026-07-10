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
static constexpr uint32_t TQ_ATTN_TILE_STRIDE = 32;

template <AscendC::HardEvent EVT>
__aicore__ inline void AttnSync()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(e);
    AscendC::WaitFlag<EVT>(e);
}

__aicore__ inline float ExpScalar(AscendC::LocalTensor<float>& expBuf, float x)
{
    expBuf.SetValue(0, x);
    AttnSync<AscendC::HardEvent::S_V>();
    AscendC::Exp(expBuf, expBuf, 1);
    AttnSync<AscendC::HardEvent::V_S>();
    return expBuf.GetValue(0);
}

// Vector QK: Q[G,128] dot K[M,128] -> score[G,M] with Mul+ReduceSum.
__aicore__ inline void VectorQkFloatPreScaled(
    AscendC::LocalTensor<float> qGroup,
    AscendC::LocalTensor<float> kTileFloat,
    AscendC::LocalTensor<float> scoreOut,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> mulTmp,
    uint32_t gqaGroup,
    uint32_t mRows,
    float scale)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreOut[g * TQ_ATTN_TILE_STRIDE];
        for (uint32_t m = 0; m < mRows; ++m) {
            AscendC::Mul(mulTmp, qGroup[g * TQ_ATTN_HEAD], kTileFloat[m * TQ_ATTN_HEAD],
                         TQ_ATTN_HEAD);
            AscendC::ReduceSum<float>(scoreVec[m], mulTmp, reduceTmp, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (scale != 1.f) {
            AscendC::Muls(scoreVec, scoreVec, scale, mRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

// Tile-level online softmax + vector PV with fp32 V tile.
__aicore__ inline void OnlineSoftmaxUpdateTileFloatPreScaled(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreTile[g * TQ_ATTN_TILE_STRIDE];
        AscendC::ReduceMax<float>(expBuf, scoreVec, reduceTmp, mRows, false);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        const float oldM = mState.GetValue(g);
        const float oldS = sState.GetValue(g);
        const float tileM = expBuf.GetValue(0);
        const float mNew = (tileM > oldM) ? tileM : oldM;
        const float alpha =
            (oldS <= 0.f) ? 0.f : ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

        AscendC::Adds(scoreVec, scoreVec, -mNew, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreVec, scoreVec, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(expBuf, scoreVec, reduceTmp, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        const uint32_t outBase = g * TQ_ATTN_HEAD;
        if (oldS > 0.f) {
            AscendC::Muls(outAcc[outBase], outAcc[outBase], alpha, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t m = 0; m < mRows; ++m) {
            const float beta = scoreVec.GetValue(m);
            AscendC::Axpy(outAcc[outBase], vTileFloat[m * TQ_ATTN_HEAD], beta, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const float tileS = expBuf.GetValue(0);
        sState.SetValue(g, oldS * alpha + tileS);
        mState.SetValue(g, mNew);
    }
}

}  // namespace turboquant_attn
