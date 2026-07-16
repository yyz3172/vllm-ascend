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

#ifndef TQ_ATTN_QK_NORM_VECTOR
#define TQ_ATTN_QK_NORM_VECTOR 1
#endif

namespace turboquant_attn {

static constexpr uint32_t TQ_ATTN_HEAD = 128;
static constexpr uint32_t TQ_ATTN_TILE_STRIDE = 64;

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

__aicore__ inline void ScaleRowsFloat(
    AscendC::LocalTensor<float> tileFloat,
    AscendC::LocalTensor<float> rowScale,
    uint32_t mRows,
    float extraScale)
{
    for (uint32_t m = 0; m < mRows; ++m) {
        const float scale = rowScale.GetValue(m) * extraScale;
        AscendC::Muls(tileFloat[m * TQ_ATTN_HEAD], tileFloat[m * TQ_ATTN_HEAD],
                      scale, TQ_ATTN_HEAD);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// Vector QK: Q_group[G,128] dot K_tile[M,128] -> score[G,M]
template <typename KvT>
__aicore__ inline void VectorQk(
    AscendC::LocalTensor<float> qGroup,
    const AscendC::LocalTensor<KvT>& kTile,
    AscendC::LocalTensor<float> scoreOut,
    AscendC::LocalTensor<float> kNorm,
    AscendC::LocalTensor<float> kFloat,
    AscendC::LocalTensor<float> mulTmp,
    AscendC::LocalTensor<float>& reduceScalar,
    uint32_t gqaGroup,
    uint32_t mRows,
    float scale)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        for (uint32_t m = 0; m < mRows; ++m) {
            AscendC::Cast(kFloat, kTile[m * TQ_ATTN_HEAD],
                          AscendC::RoundMode::CAST_NONE, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(mulTmp, qGroup[g * TQ_ATTN_HEAD], kFloat, TQ_ATTN_HEAD);
            AscendC::ReduceSum<float>(reduceScalar, mulTmp, kFloat, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AttnSync<AscendC::HardEvent::V_S>();
            const float dot = reduceScalar.GetValue(0);
            scoreOut.SetValue(g * TQ_ATTN_TILE_STRIDE + m,
                              dot * kNorm.GetValue(m) * scale);
        }
    }
}

// Vector QK with the K tile already widened to fp32 once per tile.
__aicore__ inline void VectorQkFloat(
    AscendC::LocalTensor<float> qGroup,
    AscendC::LocalTensor<float> kTileFloat,
    AscendC::LocalTensor<float> scoreOut,
    AscendC::LocalTensor<float> kNorm,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> mulTmp,
    AscendC::LocalTensor<float>& reduceScalar,
    uint32_t gqaGroup,
    uint32_t mRows,
    float scale)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreOut[g * TQ_ATTN_TILE_STRIDE];
        for (uint32_t m = 0; m < mRows; ++m) {
            AscendC::Mul(mulTmp, qGroup[g * TQ_ATTN_HEAD],
                         kTileFloat[m * TQ_ATTN_HEAD], TQ_ATTN_HEAD);
            AscendC::ReduceSum<float>(scoreVec[m], mulTmp, reduceTmp, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AttnSync<AscendC::HardEvent::V_S>();
            const float dot = scoreVec.GetValue(m);
            scoreVec.SetValue(m, dot * kNorm.GetValue(m) * scale);
        }
    }
}

// Vector QK with K row norm applied via vector Mul over scoreVec[M]. Avoids per-row
// GetValue/SetValue and V↔S sync in VectorQkFloat.
__aicore__ inline void VectorQkFloatNormVector(
    AscendC::LocalTensor<float> qGroup,
    AscendC::LocalTensor<float> kTileFloat,
    AscendC::LocalTensor<float> scoreOut,
    AscendC::LocalTensor<float> kNorm,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> mulTmp,
    AscendC::LocalTensor<float>& reduceScalar,
    uint32_t gqaGroup,
    uint32_t mRows,
    float scale)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreOut[g * TQ_ATTN_TILE_STRIDE];
        for (uint32_t m = 0; m < mRows; ++m) {
            AscendC::Mul(mulTmp, qGroup[g * TQ_ATTN_HEAD],
                         kTileFloat[m * TQ_ATTN_HEAD], TQ_ATTN_HEAD);
            AscendC::ReduceSum<float>(scoreVec[m], mulTmp, reduceTmp, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(scoreVec, scoreVec, kNorm, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        if (scale != 1.f) {
            AscendC::Muls(scoreVec, scoreVec, scale, mRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline void VectorQkFloatNormVectorGqa2(
    AscendC::LocalTensor<float> qGroup,
    AscendC::LocalTensor<float> kTileFloat,
    AscendC::LocalTensor<float> scoreOut,
    AscendC::LocalTensor<float> kNorm,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> mulTmp,
    AscendC::LocalTensor<float>& reduceScalar,
    uint32_t mRows)
{
    auto score0 = scoreOut;
    auto score1 = scoreOut[TQ_ATTN_TILE_STRIDE];
    auto q0 = qGroup;
    auto q1 = qGroup[TQ_ATTN_HEAD];
    for (uint32_t m = 0; m < mRows; ++m) {
        const uint32_t kBase = m * TQ_ATTN_HEAD;
        AscendC::Mul(mulTmp, q0, kTileFloat[kBase], TQ_ATTN_HEAD);
        AscendC::ReduceSum<float>(score0[m], mulTmp, reduceTmp, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(mulTmp, q1, kTileFloat[kBase], TQ_ATTN_HEAD);
        AscendC::ReduceSum<float>(score1[m], mulTmp, reduceTmp, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Mul(score0, score0, kNorm, mRows);
    AscendC::Mul(score1, score1, kNorm, mRows);
    AscendC::PipeBarrier<PIPE_V>();
}

// Vector QK with K rows already scaled by row norm and attention scale.
// Writes ReduceSum directly into scoreVec[m] to avoid per-row V↔S GetValue/SetValue.
__aicore__ inline void VectorQkFloatPreScaled(
    AscendC::LocalTensor<float> qGroup,
    AscendC::LocalTensor<float> kTileFloat,
    AscendC::LocalTensor<float> scoreOut,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> mulTmp,
    AscendC::LocalTensor<float>& /*reduceScalar*/,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreOut[g * TQ_ATTN_TILE_STRIDE];
        for (uint32_t m = 0; m < mRows; ++m) {
            AscendC::Mul(mulTmp, qGroup[g * TQ_ATTN_HEAD],
                         kTileFloat[m * TQ_ATTN_HEAD], TQ_ATTN_HEAD);
            AscendC::ReduceSum<float>(scoreVec[m], mulTmp, reduceTmp, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

// Vector PV: prob[G,M] dot V_tile[M,128] -> out[G,128]
template <typename KvT>
__aicore__ inline void VectorPv(
    AscendC::LocalTensor<float> prob,
    const AscendC::LocalTensor<KvT>& vTile,
    AscendC::LocalTensor<float> outGroup,
    AscendC::LocalTensor<float> valueFloat,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        AscendC::Duplicate(outGroup[g * TQ_ATTN_HEAD], 0.f, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t outBase = g * TQ_ATTN_HEAD;
        for (uint32_t m = 0; m < mRows; ++m) {
            const float beta = prob.GetValue(g * mRows + m);
            AscendC::Cast(valueFloat, vTile[m * TQ_ATTN_HEAD],
                          AscendC::RoundMode::CAST_NONE, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(valueFloat, valueFloat, beta, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(outGroup[outBase], outGroup[outBase], valueFloat, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

// Online softmax update with V rows already scaled by their row norm.
__aicore__ inline void OnlineSoftmaxUpdateTileFloatPreScaled(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> weightedValue,
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
        const float alpha = (oldS <= 0.f) ? 0.f :
                            ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

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
            AscendC::Axpy(outAcc[outBase], vTileFloat[m * TQ_ATTN_HEAD], beta,
                          TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const float tileS = expBuf.GetValue(0);
        sState.SetValue(g, oldS * alpha + tileS);
        mState.SetValue(g, mNew);
    }
}

// Online softmax update with fp32 V tile. The V row norm is applied to the
// probability scalar here so value decode can skip row-wise norm restoration.
__aicore__ inline void OnlineSoftmaxUpdateTileFloat(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> vNorm,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> weightedValue,
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
        const float alpha = (oldS <= 0.f) ? 0.f :
                            ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

        AscendC::Adds(scoreVec, scoreVec, -mNew, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreVec, scoreVec, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(expBuf, scoreVec, reduceTmp, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        AscendC::Mul(scoreVec, scoreVec, vNorm, mRows);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t outBase = g * TQ_ATTN_HEAD;
        if (oldS > 0.f) {
            AscendC::Muls(outAcc[outBase], outAcc[outBase], alpha, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t m = 0; m < mRows; ++m) {
            const float beta = scoreVec.GetValue(m);
            AscendC::Axpy(outAcc[outBase], vTileFloat[m * TQ_ATTN_HEAD], beta,
                          TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const float tileS = expBuf.GetValue(0);
        sState.SetValue(g, oldS * alpha + tileS);
        mState.SetValue(g, mNew);
    }
}

__aicore__ inline void OnlineSoftmaxUpdateTileFloatOne(
    AscendC::LocalTensor<float> scoreVec,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> vNorm,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    uint32_t stateIdx,
    uint32_t outBase,
    uint32_t mRows)
{
    AscendC::ReduceMax<float>(expBuf, scoreVec, reduceTmp, mRows, false);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();

    const float oldM = mState.GetValue(stateIdx);
    const float oldS = sState.GetValue(stateIdx);
    const float tileM = expBuf.GetValue(0);
    const float mNew = (tileM > oldM) ? tileM : oldM;
    const float alpha = (oldS <= 0.f) ? 0.f :
                        ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

    AscendC::Adds(scoreVec, scoreVec, -mNew, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(scoreVec, scoreVec, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(expBuf, scoreVec, reduceTmp, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();

    AscendC::Mul(scoreVec, scoreVec, vNorm, mRows);
    AscendC::PipeBarrier<PIPE_V>();

    if (oldS > 0.f) {
        AscendC::Muls(outAcc[outBase], outAcc[outBase], alpha, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t m = 0; m < mRows; ++m) {
        const float beta = scoreVec.GetValue(m);
        AscendC::Axpy(outAcc[outBase], vTileFloat[m * TQ_ATTN_HEAD], beta,
                      TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const float tileS = expBuf.GetValue(0);
    sState.SetValue(stateIdx, oldS * alpha + tileS);
    mState.SetValue(stateIdx, mNew);
}

__aicore__ inline void OnlineSoftmaxUpdateTileFloatGqa2(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> vNorm,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    uint32_t mRows)
{
    auto score0 = scoreTile;
    auto score1 = scoreTile[TQ_ATTN_TILE_STRIDE];

    AscendC::ReduceMax<float>(expBuf, score0, reduceTmp, mRows, false);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float oldM0 = mState.GetValue(0);
    const float oldS0 = sState.GetValue(0);
    const float tileM0 = expBuf.GetValue(0);
    const float mNew0 = (tileM0 > oldM0) ? tileM0 : oldM0;
    const float alpha0 = (oldS0 <= 0.f) ? 0.f :
                         ((mNew0 > oldM0) ? ExpScalar(expBuf, oldM0 - mNew0) : 1.f);

    AscendC::Adds(score0, score0, -mNew0, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(score0, score0, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(expBuf, score0, reduceTmp, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float tileS0 = expBuf.GetValue(0);
    AscendC::Mul(score0, score0, vNorm, mRows);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::ReduceMax<float>(expBuf, score1, reduceTmp, mRows, false);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float oldM1 = mState.GetValue(1);
    const float oldS1 = sState.GetValue(1);
    const float tileM1 = expBuf.GetValue(0);
    const float mNew1 = (tileM1 > oldM1) ? tileM1 : oldM1;
    const float alpha1 = (oldS1 <= 0.f) ? 0.f :
                         ((mNew1 > oldM1) ? ExpScalar(expBuf, oldM1 - mNew1) : 1.f);

    AscendC::Adds(score1, score1, -mNew1, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(score1, score1, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(expBuf, score1, reduceTmp, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float tileS1 = expBuf.GetValue(0);
    AscendC::Mul(score1, score1, vNorm, mRows);
    AscendC::PipeBarrier<PIPE_V>();

    if (oldS0 > 0.f) {
        AscendC::Muls(outAcc, outAcc, alpha0, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (oldS1 > 0.f) {
        AscendC::Muls(outAcc[TQ_ATTN_HEAD], outAcc[TQ_ATTN_HEAD], alpha1,
                      TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t m = 0; m < mRows; ++m) {
        const uint32_t vBase = m * TQ_ATTN_HEAD;
        const float beta0 = score0.GetValue(m);
        AscendC::Axpy(outAcc, vTileFloat[vBase], beta0, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        const float beta1 = score1.GetValue(m);
        AscendC::Axpy(outAcc[TQ_ATTN_HEAD], vTileFloat[vBase], beta1,
                      TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    sState.SetValue(0, oldS0 * alpha0 + tileS0);
    mState.SetValue(0, mNew0);
    sState.SetValue(1, oldS1 * alpha1 + tileS1);
    mState.SetValue(1, mNew1);
}

// --- Scalar m/s state variants ---
// These use float* Scalar arrays instead of LocalTensor<float> for mState/sState,
// eliminating all GetValue/SetValue implicit V↔S sync overhead (the primary
// bottleneck in decode-only scenarios).

// Vector PV for fp32 V tiles: prob[m] * V[m,:] accumulated into out row.
__aicore__ inline void VectorPvFp32PreScaled(
    AscendC::LocalTensor<float> scoreVec,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> outAccRow,
    AscendC::LocalTensor<float> weightedValue,
    uint32_t mRows)
{
    for (uint32_t m = 0; m < mRows; ++m) {
        const float beta = scoreVec.GetValue(m);
        AscendC::Muls(weightedValue, vTileFloat[m * TQ_ATTN_HEAD], beta, TQ_ATTN_HEAD);
        AscendC::Add(outAccRow, outAccRow, weightedValue, TQ_ATTN_HEAD);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// Online softmax update with V rows already scaled by their row norm (Scalar m/s).
__aicore__ inline void OnlineSoftmaxUpdateTileFloatPreScaledScalar(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    float* mState,
    float* sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> weightedValue,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreTile[g * TQ_ATTN_TILE_STRIDE];
        AscendC::ReduceMax<float>(expBuf, scoreVec, reduceTmp, mRows, false);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        const float oldM = mState[g];
        const float oldS = sState[g];
        const float tileM = expBuf.GetValue(0);
        const float mNew = (tileM > oldM) ? tileM : oldM;
        const float alpha = (oldS <= 0.f) ? 0.f :
                            ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

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
        VectorPvFp32PreScaled(
            scoreVec, vTileFloat, outAcc[outBase], weightedValue, mRows);
        const float tileS = expBuf.GetValue(0);
        sState[g] = oldS * alpha + tileS;
        mState[g] = mNew;
    }
}

// Online softmax update with fp32 V tile and vNorm (Scalar m/s).
__aicore__ inline void OnlineSoftmaxUpdateTileFloatScalar(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> vNorm,
    float* mState,
    float* sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    AscendC::LocalTensor<float> weightedValue,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreTile[g * TQ_ATTN_TILE_STRIDE];
        AscendC::ReduceMax<float>(expBuf, scoreVec, reduceTmp, mRows, false);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        const float oldM = mState[g];
        const float oldS = sState[g];
        const float tileM = expBuf.GetValue(0);
        const float mNew = (tileM > oldM) ? tileM : oldM;
        const float alpha = (oldS <= 0.f) ? 0.f :
                            ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

        AscendC::Adds(scoreVec, scoreVec, -mNew, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreVec, scoreVec, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(expBuf, scoreVec, reduceTmp, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        AscendC::Mul(scoreVec, scoreVec, vNorm, mRows);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t outBase = g * TQ_ATTN_HEAD;
        if (oldS > 0.f) {
            AscendC::Muls(outAcc[outBase], outAcc[outBase], alpha, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t m = 0; m < mRows; ++m) {
            const float beta = scoreVec.GetValue(m);
            AscendC::Axpy(outAcc[outBase], vTileFloat[m * TQ_ATTN_HEAD], beta,
                          TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const float tileS = expBuf.GetValue(0);
        sState[g] = oldS * alpha + tileS;
        mState[g] = mNew;
    }
}

// Online softmax update for GQA=2 with vNorm (Scalar m/s).
__aicore__ inline void OnlineSoftmaxUpdateTileFloatGqa2Scalar(
    AscendC::LocalTensor<float> scoreTile,
    AscendC::LocalTensor<float> vTileFloat,
    AscendC::LocalTensor<float> vNorm,
    float* mState,
    float* sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> reduceTmp,
    uint32_t mRows)
{
    auto score0 = scoreTile;
    auto score1 = scoreTile[TQ_ATTN_TILE_STRIDE];

    AscendC::ReduceMax<float>(expBuf, score0, reduceTmp, mRows, false);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float oldM0 = mState[0];
    const float oldS0 = sState[0];
    const float tileM0 = expBuf.GetValue(0);
    const float mNew0 = (tileM0 > oldM0) ? tileM0 : oldM0;
    const float alpha0 = (oldS0 <= 0.f) ? 0.f :
                         ((mNew0 > oldM0) ? ExpScalar(expBuf, oldM0 - mNew0) : 1.f);

    AscendC::Adds(score0, score0, -mNew0, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(score0, score0, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(expBuf, score0, reduceTmp, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float tileS0 = expBuf.GetValue(0);
    AscendC::Mul(score0, score0, vNorm, mRows);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::ReduceMax<float>(expBuf, score1, reduceTmp, mRows, false);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float oldM1 = mState[1];
    const float oldS1 = sState[1];
    const float tileM1 = expBuf.GetValue(0);
    const float mNew1 = (tileM1 > oldM1) ? tileM1 : oldM1;
    const float alpha1 = (oldS1 <= 0.f) ? 0.f :
                         ((mNew1 > oldM1) ? ExpScalar(expBuf, oldM1 - mNew1) : 1.f);

    AscendC::Adds(score1, score1, -mNew1, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(score1, score1, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(expBuf, score1, reduceTmp, mRows);
    AscendC::PipeBarrier<PIPE_V>();
    AttnSync<AscendC::HardEvent::V_S>();
    const float tileS1 = expBuf.GetValue(0);
    AscendC::Mul(score1, score1, vNorm, mRows);
    AscendC::PipeBarrier<PIPE_V>();

    if (oldS0 > 0.f) {
        AscendC::Muls(outAcc, outAcc, alpha0, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (oldS1 > 0.f) {
        AscendC::Muls(outAcc[TQ_ATTN_HEAD], outAcc[TQ_ATTN_HEAD], alpha1,
                      TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t m = 0; m < mRows; ++m) {
        const uint32_t vBase = m * TQ_ATTN_HEAD;
        const float beta0 = score0.GetValue(m);
        AscendC::Axpy(outAcc, vTileFloat[vBase], beta0, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        const float beta1 = score1.GetValue(m);
        AscendC::Axpy(outAcc[TQ_ATTN_HEAD], vTileFloat[vBase], beta1,
                      TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
    }
    sState[0] = oldS0 * alpha0 + tileS0;
    mState[0] = mNew0;
    sState[1] = oldS1 * alpha1 + tileS1;
    mState[1] = mNew1;
}

// --- End Scalar variants ---

// Online softmax update for one KV tile (per G head).
template <typename KvT>
__aicore__ inline void OnlineSoftmaxUpdateTile(
    AscendC::LocalTensor<float> scoreTile,
    const AscendC::LocalTensor<KvT>& vTile,
    AscendC::LocalTensor<float> mState,
    AscendC::LocalTensor<float> sState,
    AscendC::LocalTensor<float> outAcc,
    AscendC::LocalTensor<float>& expBuf,
    AscendC::LocalTensor<float> valueFloat,
    AscendC::LocalTensor<float> weightedValue,
    uint32_t gqaGroup,
    uint32_t mRows)
{
    for (uint32_t g = 0; g < gqaGroup; ++g) {
        auto scoreVec = scoreTile[g * TQ_ATTN_TILE_STRIDE];
        AscendC::ReduceMax<float>(expBuf, scoreVec, valueFloat, mRows, false);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        const float oldM = mState.GetValue(g);
        const float oldS = sState.GetValue(g);
        const float tileM = expBuf.GetValue(0);
        const float mNew = (tileM > oldM) ? tileM : oldM;
        const float alpha = (oldS <= 0.f) ? 0.f :
                            ((mNew > oldM) ? ExpScalar(expBuf, oldM - mNew) : 1.f);

        AscendC::Adds(scoreVec, scoreVec, -mNew, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreVec, scoreVec, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(expBuf, scoreVec, valueFloat, mRows);
        AscendC::PipeBarrier<PIPE_V>();
        AttnSync<AscendC::HardEvent::V_S>();

        const uint32_t outBase = g * TQ_ATTN_HEAD;
        AscendC::Muls(outAcc[outBase], outAcc[outBase], alpha, TQ_ATTN_HEAD);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t m = 0; m < mRows; ++m) {
            const float beta = scoreVec.GetValue(m);
            AscendC::Cast(valueFloat, vTile[m * TQ_ATTN_HEAD],
                          AscendC::RoundMode::CAST_NONE, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(weightedValue, valueFloat, beta, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(outAcc[outBase], outAcc[outBase], weightedValue, TQ_ATTN_HEAD);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const float tileS = expBuf.GetValue(0);
        sState.SetValue(g, oldS * alpha + tileS);
        mState.SetValue(g, mNew);
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
