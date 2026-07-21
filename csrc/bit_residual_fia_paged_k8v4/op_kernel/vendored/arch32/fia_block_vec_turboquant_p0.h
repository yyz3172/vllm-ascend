/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_block_vec_turboquant_p0.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_TURBOQUANT_P0_H
#define FIA_BLOCK_VEC_TURBOQUANT_P0_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../memory_copy.h"
#include "../vector_common.h"
#include "fia_turboquant_pi_bisect.h"
#include "br_pack_layout.h"
#include "br_dequant_device.h"

constexpr uint32_t TQ_VEC_DEQ_K0_READY_VEC = 14U;
constexpr uint32_t TQ_VEC_DEQ_V0_READY_VEC = 16U;
constexpr float TQ_MSE_SCALE = 0.0026f;
constexpr float TQ_MSE_CENTER = 127.5f;

using namespace AttentionCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
// Some call sites use unqualified RowMuls/RowDivs (ops-transformer style).
using fa_base_vector::RowMuls;
using fa_base_vector::RowDivs;

template <typename FIAT> class FiaBlockVecTurboQuantP0 {
public:
    __aicore__ inline FiaBlockVecTurboQuantP0() {}
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using ORIGIN_T = typename FIAT::orginalType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;
    static constexpr bool SOFTMAX_WITH_BRC = FIAT::softmaxWithBrc;
    static constexpr GmFormat KV_FORMAT = GetKVFormat<KV_LAYOUT_T, PAGE_ATTENTION>();

    // Dequant WS / vec1Res must match Cube Q dtype (Mmad rejects bf16×half).
    using WS_T = Q_T;
    using UPDATE_T = T;
    using TMP_T = T;
    using COMPUTE_T = T;
    using SOFTMAX_TYPE = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using SINK_T = bfloat16_t;
    using PSE_T = typename AscendC::Conditional<IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    // =================================设置参数=================================
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<WS_T> vec1ResGm, GlobalTensor<MM1_OUT_T> mm1ResGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm);
    __aicore__ inline void InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<T> lseMaxFdGm,
        GlobalTensor<T> lseSumFdGm);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void InitDequantWorkspace(__gm__ uint8_t *dequantKeyWsBase, __gm__ uint8_t *dequantValueWsBase);
    __aicore__ inline void DequantK(const RunInfo &info);
    __aicore__ inline void DequantV(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void SetMSplitInfo(uint32_t mDealSize);
    // V1
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, uint32_t startRow, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb, TBuf<> &tmpBuf,
        uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb,
        LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<COMPUTE_T> &softmaxSumUb, LocalTensor<COMPUTE_T> &softmaxMaxUb);
    // V2
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    // TurboQuant: 在 DealBmm2ResBaseBlock 的 isLastS2Loop 分支,RowDivs 之后、Bmm2ResCopyOut 之前,
    // 对 acc 做 O = Π^T@acc (= acc@Π)。覆盖普通/FD-split/FD-nonsplit 三条出口(在统一分发口之前插)。
    // P0 stub:piApplyEnabled_=false(Π=I)时直接 return,零回归。真实现见 plan §3.2。
    __aicore__ inline void ApplyPiTransposeToRows(LocalTensor<MM2_OUT_T> &accUb, uint32_t dealRowCount,
                                                  uint32_t headDim, uint32_t headDimAlign);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                              uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidMaskRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &attenOutUb, uint32_t wsMStart,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Vec1SinkCompute(const RunInfo &info, uint32_t idx, uint32_t wsMStart, uint32_t dealRowCount);
    __aicore__ inline void Vec1SinkSoftmaxProc(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                            uint32_t offset, uint32_t dealRowCountBrcb);
    __aicore__ inline void Vec1GetSinkValue(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                            uint32_t wsMStart, uint32_t dealRowCount);
    __aicore__ inline void SinkCopyIn(const RunInfo &info, LocalTensor<COMPUTE_T> &sinkBuf);
    __aicore__ inline void SinkValueNoBrc(LocalTensor<COMPUTE_T> tmpSinkResUb,
                                            LocalTensor<COMPUTE_T> tmpSinkResUbBrcb, uint32_t dealRowCount);
    __aicore__ inline void SinkInvalidRow(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                            int64_t s1Idx, int64_t row);
private:
    __aicore__ inline uint64_t GetKvCacheTokenIdx(uint32_t bIdx, uint32_t globalS2);
    __aicore__ inline uint64_t GetBrPackHeadBase(int32_t physBlock, uint32_t n2Idx, bool isKey);
    __aicore__ inline void BindKvCacheGm(uint32_t bIdx);
    __aicore__ inline void DequantKvImpl(const RunInfo &info, bool isKey);

    OffsetCalculator<KV_FORMAT> kvOffsetCalculator_;
    __gm__ uint8_t *keyListPtr_ = nullptr;
    __gm__ uint8_t *valueListPtr_ = nullptr;
    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    // BitResidual: Π = rotation_value, same dtype as query (half / bf16).
    // Vec2 末做 O = Π^T@acc (= acc@Π)。未传则 Π=I。
    GlobalTensor<Q_T> piGm_;
    bool piApplyEnabled_ = false;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<WS_T> dequantKeyWsGm_;
    GlobalTensor<WS_T> dequantValueWsGm_;
    TBuf<> dequantInt8Buf_;
    TBuf<> dequantFp32Buf_;
    TBuf<> dequantFp16Buf_;  // half scratch + Q_T out (same 2B stride)

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<WS_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<PSE_T> pseShiftGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ; // 需确认后续是否会用到
    GlobalTensor<uint64_t> actualSeqLengthsGm; // 需确认后续是否会用到
    GlobalTensor<SINK_T> sinkGm;

    __gm__ uint8_t *actualSequenceLengthsQ = nullptr;

    // =================================常量区=================================
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    T SOFTMAX_MIN_NUM = T(-1.0/0.0); // -inf
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_2K;
    static constexpr uint32_t LSE_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_8K;
    static constexpr uint32_t DATA_BLOCK_NUM = 8;
    static constexpr uint16_t brcbNum = (fa_base_vector::BYTE_BLOCK / sizeof(COMPUTE_T));

    // ================================Local Buffer区====================================
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    TQue<QuePosition::VECOUT, 1> outputQue2;

    // 临时tbuf
    TBuf<> tmpBuff1;
    TBuf<> softmaxMaxBuff;
    TBuf<> softmaxExpBuff;
    TBuf<> softmaxSumBuff;
    TBuf<> softmaxMaxDefaultBuff;
    TBuf<> softmaxSumDefaultBuff;

    // ================================LocalTensor区====================================
    // 常驻
    LocalTensor<COMPUTE_T> softmaxMaxDefaultUb;
    LocalTensor<COMPUTE_T> softmaxSumDefaultUb;
    LocalTensor<COMPUTE_T> softmaxMaxUb;
    LocalTensor<COMPUTE_T> softmaxSumUb;
    LocalTensor<COMPUTE_T> softmaxExpUb;

    // ================================其他成员区========================================
    ConstInfo constInfo = {};
    MSplitInfo mSplitInfo = {};

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>(); 
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>(); 
    ActualSeqLensParser<Q_MODE> qActSeqLensParser; 
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // PSE仅在Q的lauot为BSH/BSND/BNSD时支持
    static constexpr bool IS_SUPPORT_PSE = IsSupportPse<LAYOUT_T>();
    static constexpr UbFormat PSE_UB_FORMAT = GetPseUbFormat<LAYOUT_T>();
    FaGmTensor<PSE_T, GmFormat::BN2GS1S2> pseShiftGmTensor;
    CopyPSEGmToUb<PSE_T, GmFormat::BN2GS1S2, PSE_UB_FORMAT> copyPSEGmToUb;
    bool pseHasBatch = true;

    bool learnableSinkFlag = false;
};

template <typename FIAT>
__aicore__ inline void
FiaBlockVecTurboQuantP0<FIAT>::InitParams(const struct ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse)
{
    keyListPtr_ = key;
    valueListPtr_ = value;
    blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
    BindKvCacheGm(0U);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if (constInfo.softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
    attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
    this->actualSequenceLengthsQ = actualSeqLengthsQ;
    if (learnableSink != nullptr) {
        learnableSinkFlag = true;
        sinkGm.SetGlobalBuffer((__gm__ SINK_T *)learnableSink);
    }
    if (valueAntiquantScale != nullptr) {
        piGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T *>(valueAntiquantScale));
        piApplyEnabled_ = true;
    }
    qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
    kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);

    if constexpr (IS_SUPPORT_PSE) {
        if (constInfo.pseShiftFlag) {
            pseShiftGm.SetGlobalBuffer((__gm__ PSE_T *)pseShift);
            pseShiftGmTensor.gmTensor = pseShiftGm;
            pseShiftGmTensor.offsetCalculator.Init(constInfo.pseShiftByBatch ? constInfo.batchSize : 1,
                constInfo.kvHeadNum, constInfo.gSize, constInfo.pseShiftS1, constInfo.pseShiftS2);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::InitVec1GlobalTensor(GlobalTensor<WS_T> vec1ResGm, GlobalTensor<MM1_OUT_T> mm1ResGm)
{
    this->vec1ResGm = vec1ResGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm)
{
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm,
    GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm)
{
    this->accumOutGm = accumOutGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::InitBuffers(TPipe *pipe)
{
    const uint32_t softmaxPingPongBytes = SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum;
    const uint32_t softmaxDefaultBytes = SOFTMAX_TMP_BUFFER_SIZE;

    // in queue
    pipe->InitBuffer(inputQue1, 2, ConstInfo::BUFFER_SIZE_BYTE_32K); // 2:pingpong
    pipe->InitBuffer(inputQue2, 2, ConstInfo::BUFFER_SIZE_BYTE_16K);  // 2:pingpong

    // out queue
    pipe->InitBuffer(outputQue1, 1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue2, 1, ConstInfo::BUFFER_SIZE_BYTE_8K);

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);

    // 1. [M,8]场景: 2K/32B = 64, 即单个VEC上可以缓存64行, 整个AICORE上有2个VEC，所以此时分核的MBaseSize<=128
    // 2. [M,1]场景: 2K/sizeof(float) = 512, 即单个VEC上可以缓存512行, 所以此时分核的MBaseSize<=512*2=1024
    pipe->InitBuffer(softmaxMaxBuff, softmaxPingPongBytes);
    pipe->InitBuffer(softmaxExpBuff, softmaxPingPongBytes);
    pipe->InitBuffer(softmaxSumBuff, softmaxPingPongBytes);

    pipe->InitBuffer(softmaxMaxDefaultBuff, softmaxDefaultBytes);
    pipe->InitBuffer(softmaxSumDefaultBuff, softmaxDefaultBytes);

    softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
    softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();
    softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<COMPUTE_T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<COMPUTE_T>();

    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T));
    Duplicate(softmaxSumDefaultUb, (COMPUTE_T)0.0, SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T));

    uint32_t elemCount = constInfo.headDimAlign;
    // codes + 64B meta staging (legacy; Key/Value runs stage in tmpBuff1).
    uint32_t int8Bytes = ((br_dequant::BR_DEQUANT_UB_BYTES + 31U) / 32U) * 32U;
    // 1-row dedicated scratch only. Tile=8 Key decode scratch overlays tmpBuff1 tail.
    uint32_t fp32Bytes = ((elemCount * 3U * static_cast<uint32_t>(sizeof(COMPUTE_T)) + 31U) / 32U) * 32U;
    uint32_t fp16Bytes = ((elemCount * static_cast<uint32_t>(sizeof(half)) + 31U) / 32U) * 32U;
    pipe->InitBuffer(dequantInt8Buf_, int8Bytes);
    pipe->InitBuffer(dequantFp32Buf_, fp32Bytes);
    pipe->InitBuffer(dequantFp16Buf_, fp16Bytes);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::AllocEventID()
{
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::FreeEventID()
{
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::SetMSplitInfo(uint32_t mDealSize)
{
    mSplitInfo.nBufferIdx = 0U;
    mSplitInfo.nBufferStartM = 0U;
    mSplitInfo.nBufferDealM = mDealSize;
    // VEC0处理的M大小
    if (mSplitInfo.nBufferDealM <= 16) {
        mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
    } else {
        mSplitInfo.vecDealM = ((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16;
    }
    mSplitInfo.vecStartM = 0;
    if (GetBlockIdx() % 2 == 1) {
        // VEC1处理的M大小
        mSplitInfo.vecStartM = mSplitInfo.vecDealM;
        mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ProcessVec1SingleBuf(const RunInfo &info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = fa_base_vector::BYTE_BLOCK / sizeof(COMPUTE_T);
        // 向下8/16对齐是因为UB操作起始地址需32B对齐
        mSplitSize = mSplitSize / alignVal * alignVal;
    }
    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
    }

    if (info.isLastS2Loop) {
        uint32_t outIdx = info.loop % (constInfo.preLoadNum);
        if (unlikely(learnableSinkFlag)) {
            Vec1SinkCompute(info, outIdx, mSplitInfo.nBufferStartM + mSplitInfo.vecStartM, mSplitInfo.vecDealM);
        }

        auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)];
        auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)];
        if (info.tndIsS2SplitCore) {
            if constexpr (FLASH_DECODE) {
                ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
        } else if (constInfo.softmaxLseFlag) {
            LocalTensor<COMPUTE_T> totalLseUb = tmpBuff1.Get<COMPUTE_T>(LSE_TMP_BUFFER_SIZE);
            if constexpr (!SOFTMAX_WITH_BRC) {
                LocalTensor<COMPUTE_T> lseSumUb = tmpBuff1.GetWithOffset<COMPUTE_T>(LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE);
                LocalTensor<COMPUTE_T> lseMaxUb = tmpBuff1.GetWithOffset<COMPUTE_T>(
                    LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE  *2);
                Brcb(lseSumUb, sumTensor[mSplitInfo.nBufferStartM / 2], (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum, 
                    {1, brcbNum});
                AscendC::PipeBarrier<PIPE_V>();
                Brcb(lseMaxUb, maxTensor[mSplitInfo.nBufferStartM / 2], (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum, 
                    {1, brcbNum});
                AscendC::PipeBarrier<PIPE_V>();
                fa_base_vector::ComputeSoftMaxLse(totalLseUb, lseSumUb, lseMaxUb, mSplitInfo.vecDealM);
            } else {
                fa_base_vector::ComputeSoftMaxLse(totalLseUb, sumTensor, maxTensor, mSplitInfo.vecDealM);
            }

            bool isInvalidRows = fa_base_vector::IsExistInvalidRows(info.nextTokensPerBatch, info.preTokensPerBatch, 
                constInfo.sparseMode, constInfo.attenMaskFlag, constInfo.isRowInvalid);

            if (isInvalidRows) { // 存在行无效场景
                SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(mSplitInfo.vecDealM), static_cast<uint32_t>(brcbNum),
                static_cast<uint32_t>(mSplitInfo.vecDealM), static_cast<uint32_t>(brcbNum)};

                if constexpr (SOFTMAX_WITH_BRC) {
                    AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T>(totalLseUb, maxTensor, negativeIntScalar, 
                        (COMPUTE_T)3e+99, softmaxShapeInfo);
                } else {
                    AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T, false, 1>(totalLseUb, maxTensor, negativeIntScalar, 
                        (COMPUTE_T)3e+99, softmaxShapeInfo);
                }
            }

            LocalTensor<T> tmpLseResCastTensor = outputQue2.AllocTensor<T>();
            DataCopy(tmpLseResCastTensor, totalLseUb, mSplitInfo.vecDealM * brcbNum);
            outputQue2.EnQue(tmpLseResCastTensor);
            outputQue2.DeQue<T>();
            uint32_t mOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
            if (LAYOUT_T == FIA_LAYOUT::TND) {
                uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
                uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + info.n2Idx * constInfo.gSize;
                DataCopySoftmaxLseTND(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo);
            } else if (LAYOUT_T == FIA_LAYOUT::NTD) {
                uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
                uint32_t s1Size = info.bIdx == 0U ? 
                        actualSeqLengthsGmQ.GetValue(0U) : actualSeqLengthsGmQ.GetValue(info.bIdx) - actualSeqLengthsGmQ.GetValue(info.bIdx - 1U);
                uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + info.n2Idx * constInfo.gSize;
                DataCopySoftmaxLseNTD(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo, s1Size);
            } else if (LAYOUT_T == FIA_LAYOUT::BSND || LAYOUT_T == FIA_LAYOUT::BSH) {
                uint64_t bN2Offset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize + info.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                DataCopySoftmaxLseBSND(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo, qActSeqLensParser, info.bIdx);
            } else { // BNSD
                uint64_t bN2Offset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize + info.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                DataCopySoftmaxLseBNSD<T, Q_MODE>(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo, qActSeqLensParser, info.bIdx);
            }
            outputQue2.FreeTensor(tmpLseResCastTensor);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputQue1.AllocTensor<MM1_OUT_T>();
    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    inputQue1.EnQue(mmResUb);
    inputQue1.DeQue<MM1_OUT_T>();

    ElewiseCompute(info, mmResUb, tmpBuff1, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpBuff1.Get<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<WS_T> vec1ResUb = outputQue1.AllocTensor<WS_T>();
    Cast(vec1ResUb, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    outputQue1.EnQue(vec1ResUb);
    outputQue1.DeQue<WS_T>();
    DataCopy(vec1ResGm[inOutGmOffset], vec1ResUb, computeSize);
    outputQue1.FreeTensor(vec1ResUb);

    inputQue1.FreeTensor(mmResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ElewiseCompute(
    const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<MM1_OUT_T>(constInfo.scaleValue), dealRowCount * columnCount);

    if constexpr (IS_SUPPORT_PSE) {
        if (constInfo.pseShiftFlag) {
            LocalTensor<PSE_T> pseShiftB16 = inputQue2.AllocTensor<PSE_T>();
            FaUbTensor<PSE_T> pseShiftUbTensor {
                .tensor = pseShiftB16,
                .rowCount = dealRowCount,
                .colCount = columnCount
            };
            GmPseCoord pseCoord = {
                .bIdx = constInfo.pseShiftByBatch ? info.bIdx : 0,
                .n2Idx = info.n2Idx,
                .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow,
                .s2Idx = info.s2Idx * constInfo.s2BaseSize,
                .gS1DealSize = dealRowCount,
                .s2DealSize = actualColumnCount,
                .s1LeftPaddingSize = info.qPaddingBeginOffset,
                .s2LeftPaddingSize = info.kvPaddingBeginOffset
            };
            copyPSEGmToUb(pseShiftUbTensor, pseShiftGmTensor, pseCoord);
            inputQue2.EnQue(pseShiftB16);
            inputQue2.DeQue<PSE_T>();
            LocalTensor<T> pseShiftUbFP32 = tmpBuf.Get<T>();
            AscendC::Cast(pseShiftUbFP32, pseShiftB16, AscendC::RoundMode::CAST_NONE, dealRowCount * columnCount);
            inputQue2.FreeTensor(pseShiftB16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(mmResUb, mmResUb, pseShiftUbFP32, dealRowCount * columnCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    if (constInfo.attenMaskFlag == 1) {
        AscendC::PipeBarrier<PIPE_V>();
        fa_base_vector::MaskInfo maskInfo;
        maskInfo.gs1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
        maskInfo.gs1dealNum = dealRowCount;
        maskInfo.s1Size = info.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = info.s2Idx * constInfo.s2BaseSize;
        maskInfo.s2dealNum = info.actualSingleProcessSInnerSize;
        maskInfo.s2Size = info.actS2Size;
        maskInfo.preToken = constInfo.preToken;
        maskInfo.nextToken = constInfo.nextToken;
        maskInfo.sparseMode = static_cast<fa_base_vector::SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = info.bIdx;
        maskInfo.batchOffset = constInfo.attenMaskBatchStride;
        maskInfo.attenMaskStride = constInfo.attenMaskStride;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = info.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = info.kvPaddingBeginOffset;

        if (constInfo.qSeqSize == 1) {
            maskInfo.layout = fa_base_vector::S1_EQUAL1;
        } else if (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::BSH) {
            maskInfo.layout = fa_base_vector::SG;
        } else {
            maskInfo.layout = fa_base_vector::GS;
        }
        maskInfo.attenMaskType = fa_base_vector::MASK_BOOL; // compatible with int8/uint8
        LocalTensor<bool> maskUb;
        LocalTensor<bool> attenMaskTmpUb;
        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
        if (!fa_base_vector::IsSkipAttentionmask(maskInfo)) {
            maskUb = inputQue2.AllocTensor<bool>();
            attenMaskTmpUb = maskUb[BUFFER_SIZE_BYTE_16K / 2];
            fa_base_vector::AttentionmaskCopyIn(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo);
            AscendC::PipeBarrier<PIPE_V>();
            fa_base_vector::AttentionmaskCompute<MM1_OUT_T>(mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo);
            inputQue2.FreeTensor(maskUb);
        }
        if (!fa_base_vector::IsSkipAttentionmaskForPre(maskInfo)) {
            maskUb = inputQue2.AllocTensor<bool>();
            attenMaskTmpUb = maskUb[BUFFER_SIZE_BYTE_16K / 2];
            fa_base_vector::AttentionmaskCopyIn(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo, true);
            fa_base_vector::AttentionmaskCompute<MM1_OUT_T>(mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo, true);
            inputQue2.FreeTensor(maskUb);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(COMPUTE_T), sizeof(COMPUTE_T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<COMPUTE_T> inSumTensor;
    LocalTensor<COMPUTE_T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * this->brcbNum;
    }
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
    }
    if constexpr (SOFTMAX_WITH_BRC) {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ProcessVec2SingleBuf(const RunInfo &info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = fa_base_vector::BYTE_BLOCK / sizeof(COMPUTE_T);
        // 向下8/16对齐是因为UB操作起始地址需32B对齐
        mSplitSize = mSplitSize / alignVal * alignVal;
    }
    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, i * mSplitSize, dealSize, constInfo.headDimAlign, constInfo.headDim);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DealBmm2ResBaseBlock(
    const RunInfo &info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * this->brcbNum;
    }
    uint64_t inOutBaseOffset = mStart * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM2_OUT_T> bmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(bmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    inputQue1.EnQue(bmm2ResUb);
    inputQue1.DeQue<MM2_OUT_T>();

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<COMPUTE_T> bmm2ResPreUb = inputQue1.AllocTensor<COMPUTE_T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
        inputQue1.EnQue(bmm2ResPreUb);

        inputQue1.DeQue<COMPUTE_T>();
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % (constInfo.preLoadNum);

        if constexpr (SOFTMAX_WITH_BRC) {
            RowMuls<COMPUTE_T>(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpExpBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpExpBrcbResUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            RowMuls<COMPUTE_T>(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue1.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.isLastS2Loop) {
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % (constInfo.preLoadNum);

        if constexpr (SOFTMAX_WITH_BRC) {
            fa_base_vector::RowDivs<COMPUTE_T>(bmm2ResUb, bmm2ResUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpSumBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            fa_base_vector::RowDivs<COMPUTE_T>(bmm2ResUb, bmm2ResUb, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        ApplyPiTransposeToRows(bmm2ResUb, dealRowCount, actualColumnCount, columnCount);
        AscendC::PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<COMPUTE_T> tmpBmm2Res = outputQue1.AllocTensor<COMPUTE_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<COMPUTE_T>();
        uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }

    inputQue1.FreeTensor(bmm2ResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ApplyPiTransposeToRows(
    LocalTensor<MM2_OUT_T> &accUb, uint32_t dealRowCount, uint32_t headDim, uint32_t headDimAlign)
{
    // TurboQuant: O = Π^T@acc → O[m,d] = Σ_k Π[k,d]*acc[m,k]。仅 last S2 出口。
    // Row-prefetch: 每个 k 一次 GM→UB 整行 Π[k,:]（D elems），再按 dBlock tile 乘加。
    // 相对旧实现 m*(D/tile)*D 次 32 元小 DataCopy，MTE2 事务数降为 m*D。
    if (!piApplyEnabled_ || (TQ_PI_BISECT_VEC == 0)) {
        return;
    }

    constexpr uint32_t PI_COL_TILE = 32U;
    // Layout (bytes): accRow | outRow | piRowHalf[D] | piRowFp32[D] | mulTmp[tile]
    LocalTensor<COMPUTE_T> accRowCopy = tmpBuff1.Get<COMPUTE_T>();
    LocalTensor<COMPUTE_T> outRow =
        tmpBuff1.GetWithOffset<COMPUTE_T>(headDimAlign, headDimAlign * sizeof(COMPUTE_T));
    const uint32_t piQOffset = 2U * headDimAlign * sizeof(COMPUTE_T);
    LocalTensor<Q_T> piRowHalf = tmpBuff1.GetWithOffset<Q_T>(headDim, piQOffset);
    const uint32_t piFp32Offset = piQOffset + headDim * static_cast<uint32_t>(sizeof(Q_T));
    LocalTensor<COMPUTE_T> piRowFp32 =
        tmpBuff1.GetWithOffset<COMPUTE_T>(headDim, piFp32Offset);
    LocalTensor<COMPUTE_T> mulTmp = tmpBuff1.GetWithOffset<COMPUTE_T>(
        PI_COL_TILE, piFp32Offset + headDim * static_cast<uint32_t>(sizeof(COMPUTE_T)));

    event_t eventIdVWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdSWaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIdVWaitS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    for (uint32_t m = 0U; m < dealRowCount; ++m) {
        DataCopy(accRowCopy, accUb[m * headDimAlign], headDimAlign);
        SetFlag<HardEvent::V_S>(eventIdSWaitV);
        WaitFlag<HardEvent::V_S>(eventIdSWaitV);

        Duplicate(outRow, static_cast<COMPUTE_T>(0.0f), headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t k = 0U; k < headDim; ++k) {
            // One contiguous GM load of Π row k (headDim elems).
            DataCopy(piRowHalf, piGm_[k * headDim], headDim);
            SetFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);
            WaitFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);
            Cast(piRowFp32, piRowHalf, AscendC::RoundMode::CAST_NONE, headDim);
            AscendC::PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_S>(eventIdSWaitV);
            WaitFlag<HardEvent::V_S>(eventIdSWaitV);
            const COMPUTE_T accMk = accRowCopy.GetValue(k);
            SetFlag<HardEvent::S_V>(eventIdVWaitS);
            WaitFlag<HardEvent::S_V>(eventIdVWaitS);

            for (uint32_t dBlock = 0U; dBlock < headDim; dBlock += PI_COL_TILE) {
                Muls(mulTmp, piRowFp32[dBlock], accMk, PI_COL_TILE);
                AscendC::PipeBarrier<PIPE_V>();
                Add(outRow[dBlock], outRow[dBlock], mulTmp, PI_COL_TILE);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        DataCopy(accUb[m * headDimAlign], outRow, headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
    uint32_t wsMStart, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
    uint32_t wsMStart, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<MM2_OUT_T> tmp = outputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    outputQue1.EnQue(tmp);
    outputQue1.DeQue<T>();
    uint64_t offset = info.accumTmpOutNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
                      info.tndCoreStartKVSplitPos * constInfo.mBaseSize * constInfo.headDim + // 份数offset
                      wsMStart * actualColumnCount;                                             // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (fa_base_vector::BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(dst, tmp, dataCopyParams);
    outputQue1.FreeTensor(tmp);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DealInvalidMaskRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
    uint32_t wsMStart, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if (!constInfo.isRowInvalid || !constInfo.attenMaskFlag) {
        return;
    }
    if (constInfo.sparseMode != fa_base_vector::DEFAULT_MASK && constInfo.sparseMode != fa_base_vector::ALL_MASK) {
        return;
    }
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * (fa_base_vector::BYTE_BLOCK / sizeof(T));
    }

    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(T) + baseOffset;

    fa_base_vector::InvalidMaskRows<MM2_OUT_T, T, SOFTMAX_WITH_BRC>(softmaxOutOffset, dealRowCount, columnCount,
        softmaxMaxUb, negativeIntScalar, bmm2ResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Bmm2CastAndCopyOut(const RunInfo &info, 
    LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    DealInvalidMaskRows(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }
    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecTurboQuantP0<FIAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    FaUbTensor<OUT_T> ubTensor {
        .tensor = attenOutUb,
        .rowCount = dealRowCount,
        .colCount = columnCount,
    };
    GmCoord gmCoord {
        .bIdx = info.bIdx,
        .n2Idx = info.n2Idx,
        .gS1Idx = info.gS1Idx + wsMStart,
        .dIdx = 0,
        .gS1DealSize = dealRowCount,
        .dDealSize = (uint32_t)constInfo.headDim
    };

    if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
            constInfo.qSeqSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims,
            constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
            constInfo.qSeqSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims,
            constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
            constInfo.qSeqSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
        constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim,
            actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim,
            actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<COMPUTE_T> &softmaxSumUb,
                                                                                         LocalTensor<COMPUTE_T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    //  src-Shape  { gsizeV, S1, fa_base_vector::FP32_BLOCK_ELEMENT_NUM }
    //  dst-Shape  { B  N2, splitKV s1, G, fa_base_vector::FP32_BLOCK_ELEMENT_NUM}
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * brcbNum;
    uint64_t offset = (info.accumTmpOutNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                      fa_base_vector::FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if constexpr (SOFTMAX_WITH_BRC) {              
        DataCopy(lseSumFdGm[offset], softmaxSumUb[baseOffset], size);
        DataCopy(lseMaxFdGm[offset], softmaxMaxUb[baseOffset], size);       
    } else {
        LocalTensor<T> tmp = outputQue2.AllocTensor<T>();   
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseSumFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
        tmp = outputQue2.AllocTensor<T>(); 
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseMaxFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DealInvalidRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &attenOutUb,
                                                      uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                      uint32_t actualColumnCount)
{
    if (!constInfo.attenMaskFlag) {
        return;
    }

    if (constInfo.sparseMode == fa_base_vector::ALL_MASK || constInfo.sparseMode == fa_base_vector::LEFT_UP_CAUSAL) {
        return;
    }

    fa_base_vector::InvalidRowParams params {
        .actS1Size = info.actS1Size,
        .gSize = constInfo.gSize,
        .gS1Idx = info.gS1Idx + wsMStart,
        .dealRowCount = dealRowCount,
        .columnCount = columnCount,
        .preTokensPerBatch = info.preTokensPerBatch,
        .nextTokensPerBatch = info.nextTokensPerBatch,
    };

    fa_base_vector::InvalidRows<T, fa_base_vector::GeInputUbFormat<LAYOUT_T>()> invalidRows;
    invalidRows(attenOutUb, params);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::SinkCopyIn(const RunInfo &info, LocalTensor<COMPUTE_T> &sinkBuf)
{
    uint32_t copySize = constInfo.gSize + 16; //DataCopy函数搬运量要求是32字节整数倍，不对齐时，将向下取整，因此该处多搬运16*sizeof(bf16)占位
    uint64_t sinkGmOffset = info.n2Idx * constInfo.gSize;
    LocalTensor<SINK_T> sinkCopyInBuf = inputQue1.AllocTensor<SINK_T>();
    DataCopy(sinkCopyInBuf, sinkGm[sinkGmOffset], copySize);
    inputQue1.EnQue(sinkCopyInBuf);
    inputQue1.DeQue<SINK_T>();

    LocalTensor<COMPUTE_T> tmpSinkCastUb = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, BUFFER_SIZE_BYTE_8K * 3);
    Cast(tmpSinkCastUb, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo.gSize);
    AscendC::PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(sinkCopyInBuf);

    Brcb(sinkBuf, tmpSinkCastUb, (constInfo.gSize + brcbNum - 1) / brcbNum, {1, brcbNum});
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::SinkInvalidRow(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
    int64_t s1Idx, int64_t row)
{
    int64_t s1BottomTok = info.actS1Size + info.preTokensPerBatch;
    int64_t s1Tok = -info.nextTokensPerBatch;
    const COMPUTE_T minValue = *((COMPUTE_T *)&negativeIntScalar);

    if (unlikely(info.nextTokensPerBatch < 0)) { // 上方存在行无效
        if (s1Idx < s1Tok) {
            Duplicate(tmpSinkResUbBrcb[row * brcbNum], minValue, brcbNum);
        }
    }

    if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) { // sparse = 3时，不存在下方行无效，直接返回
        return;
    }

    if (unlikely(info.preTokensPerBatch < 0)) { // 下方存在行无效
        if (s1Idx >= s1BottomTok && s1Idx < info.actS1Size) {
            Duplicate(tmpSinkResUbBrcb[row * brcbNum], minValue, brcbNum);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Vec1GetSinkValue(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
    uint32_t wsMStart, uint32_t dealRowCount)
{
    constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    int64_t gIdx = 0;
    int64_t s1Idx = 0;

    LocalTensor<COMPUTE_T> sinkBuf = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, BUFFER_SIZE_BYTE_8K * 2);
    SinkCopyIn(info, sinkBuf);

    bool isInvalidRows = fa_base_vector::IsExistInvalidRows(info.nextTokensPerBatch, info.preTokensPerBatch, 
        constInfo.sparseMode, constInfo.attenMaskFlag, constInfo.isRowInvalid);

    for (int64_t row = 0; row < dealRowCount; ++row) {
        if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { //内存按照S1G排布
            gIdx = (info.gS1Idx + wsMStart + row) % constInfo.gSize;
            s1Idx = (info.gS1Idx + wsMStart + row) / constInfo.gSize;
        } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { //内存按照GS1排布
            gIdx = (info.gS1Idx + wsMStart + row) / info.actS1Size;
            s1Idx = (info.gS1Idx + wsMStart + row) % info.actS1Size;
        }
        DataCopy(tmpSinkResUbBrcb[row * brcbNum], sinkBuf[gIdx * brcbNum], brcbNum);

        if (unlikely(isInvalidRows)) { // 行无效处理
            SinkInvalidRow(info, tmpSinkResUbBrcb, s1Idx, row);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Vec1SinkSoftmaxProc(const RunInfo &info,
        LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb, uint32_t offset, uint32_t dealRowCountBrcb)
{
    AscendC::PipeBarrier<PIPE_V>();
    Sub(tmpSinkResUbBrcb, tmpSinkResUbBrcb, softmaxMaxUb[offset], dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tmpSinkResUbBrcb, tmpSinkResUbBrcb, dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxSumUb[offset], softmaxSumUb[offset], tmpSinkResUbBrcb, dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::SinkValueNoBrc(LocalTensor<COMPUTE_T> tmpSinkResUb,
        LocalTensor<COMPUTE_T> tmpSinkResUbBrcb, uint32_t dealRowCount)
{
    // 不带brcb，需要把sink按行取最大值，RowMax后变为1*m的shape
    uint64_t repeatTimesOnce = 128;  //由于WholeReduceMax接口中repeatTimes支持范围（0,255），因此需要分多次调用WholeReduceMax，这里就使用每次repeatTime=128
    uint64_t loopTimes = (dealRowCount + repeatTimesOnce - 1) / repeatTimesOnce;
    uint64_t repeatTimes = repeatTimesOnce;

    for (uint64_t loop = 0; loop < loopTimes; ++loop) {
        if (loop == loopTimes - 1) {
            repeatTimes = dealRowCount - loop * repeatTimesOnce;
        }
        WholeReduceMax(tmpSinkResUb[loop * repeatTimesOnce], tmpSinkResUbBrcb[loop * brcbNum * repeatTimesOnce],
            brcbNum, repeatTimes, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::Vec1SinkCompute(const RunInfo &info, uint32_t idx, uint32_t wsMStart, uint32_t dealRowCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {  // sink叠加FD规约场景，在FD规约流程中处理，该处不处理
            return;
        }
    }

    LocalTensor<COMPUTE_T> tmpSinkResUb = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, 0);
    LocalTensor<COMPUTE_T> tmpSinkResUbBrcb = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, BUFFER_SIZE_BYTE_8K);

    Vec1GetSinkValue(info, tmpSinkResUbBrcb, wsMStart, dealRowCount);

    uint32_t offset = idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + mSplitInfo.nBufferStartM / 2;
    if constexpr (SOFTMAX_WITH_BRC) {
        uint32_t dealRowCountBrcb = dealRowCount * brcbNum;
        Vec1SinkSoftmaxProc(info, tmpSinkResUbBrcb, offset, dealRowCountBrcb);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        SinkValueNoBrc(tmpSinkResUb, tmpSinkResUbBrcb, dealRowCount);
        Vec1SinkSoftmaxProc(info, tmpSinkResUb, offset, dealRowCount);
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ComputeVec1(const RunInfo &info)
{
    DequantV(info);
    SetMSplitInfo(info.actMBaseSize);
    CrossCoreWaitFlag(constInfo.syncC1V1);
    ProcessVec1SingleBuf(info);
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::ComputeVec2(const RunInfo &info)
{
    SetMSplitInfo(info.actMBaseSize);
    CrossCoreWaitFlag(constInfo.syncC2V2);
    ProcessVec2SingleBuf(info);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::InitDequantWorkspace(__gm__ uint8_t *dequantKeyWsBase,
    __gm__ uint8_t *dequantValueWsBase)
{
    dequantKeyWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ WS_T *>(dequantKeyWsBase));
    dequantValueWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ WS_T *>(dequantValueWsBase));
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DequantK(const RunInfo &info)
{
    // Dual-AIV S2 split: both subcores run DequantKvImpl on disjoint token halves of the
    // shared WS (si is a global index → no GM race). FIA_SYNC_MODE2 still needs both flags.
    DequantKvImpl(info, true);
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(TQ_VEC_DEQ_K0_READY_VEC + (info.loop % 2));
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DequantV(const RunInfo &info)
{
    DequantKvImpl(info, false);
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(TQ_VEC_DEQ_V0_READY_VEC + (info.loop % 2));
}

template <typename FIAT>
__aicore__ inline uint64_t FiaBlockVecTurboQuantP0<FIAT>::GetKvCacheTokenIdx(uint32_t bIdx, uint32_t globalS2)
{
    uint32_t blockInBatch = globalS2 / constInfo.kvCacheBlockSize;
    uint32_t posInBlock = globalS2 % constInfo.kvCacheBlockSize;
    int32_t physBlock = blockTableGm_.GetValue(bIdx * constInfo.maxBlockNumPerBatch + blockInBatch);
    return static_cast<uint64_t>(physBlock) * static_cast<uint64_t>(constInfo.kvCacheBlockSize) +
        static_cast<uint64_t>(posInBlock);
}

template <typename FIAT>
__aicore__ inline uint64_t FiaBlockVecTurboQuantP0<FIAT>::GetBrPackHeadBase(int32_t physBlock, uint32_t n2Idx,
    bool isKey)
{
    const uint32_t bs = constInfo.kvCacheBlockSize;
    if (isKey) {
        return br_pack::BrKeyHeadBase(static_cast<uint32_t>(physBlock), n2Idx, constInfo.kvHeadNum, bs);
    }
    return br_pack::BrValHeadBase(static_cast<uint32_t>(physBlock), n2Idx, constInfo.kvHeadNum, bs);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::BindKvCacheGm(uint32_t bIdx)
{
#if defined(TQ_FIA_KV_TENSOR_LIST)
    uint32_t listIdx = constInfo.batchContinuous ? 0U : bIdx;
    ListTensorDesc keyListDesc(reinterpret_cast<__gm__ void *>(keyListPtr_));
    ListTensorDesc valueListDesc(reinterpret_cast<__gm__ void *>(valueListPtr_));
    keyCacheGm_.SetGlobalBuffer(keyListDesc.GetDataPtr<__gm__ uint8_t>(listIdx));
    valueCacheGm_.SetGlobalBuffer(valueListDesc.GetDataPtr<__gm__ uint8_t>(listIdx));
#else
    (void)bIdx;
    keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(keyListPtr_));
    valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(valueListPtr_));
#endif
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecTurboQuantP0<FIAT>::DequantKvImpl(const RunInfo &info, bool isKey)
{
    BindKvCacheGm(info.bIdx);

    uint32_t s2Count = info.actualSingleProcessSInnerSize;
    uint32_t headDimAlign = constInfo.headDimAlign;
    uint32_t headDim = static_cast<uint32_t>(constInfo.headDim);
    uint64_t wsStride = static_cast<uint64_t>(constInfo.s2BaseSize) * static_cast<uint64_t>(headDimAlign);
    uint64_t wsOffset = static_cast<uint64_t>(info.loop % 2) * wsStride;

    // Dual-AIV S2 split (ops-transformer TQ pattern): sub0=[0,half), sub1=[half,s2Count).
    // Each writes disjoint WS rows via global si → no race; slot size unchanged.
    uint32_t subCoreId = GetBlockIdx() % 2U;
    uint32_t siStart = (s2Count * subCoreId) / 2U;
    uint32_t siEnd = (s2Count * (subCoreId + 1U)) / 2U;
    if (siStart >= siEnd) {
        return;
    }

    // A1 dual-buffer staging in tmpBuff1 front; tile=8 decode scratch in the tail
    // (shared). Vec1/Vec2 reclaim the full 32KB after dequant returns.
    constexpr uint32_t kTmp1Bytes = ConstInfo::BUFFER_SIZE_BYTE_32K;
    constexpr uint32_t kTileMax = br_dequant::BR_DECODE_TILE_MAX;
    constexpr uint32_t kScratchElems = br_dequant::BR_DECODE_TILE_ELEMS;
    constexpr uint32_t kFp32ScratchBytes =
        kScratchElems * 3U * static_cast<uint32_t>(sizeof(float));
    constexpr uint32_t kFp16ScratchBytes =
        kScratchElems * static_cast<uint32_t>(sizeof(half));
    constexpr uint32_t kScratchBytes = kFp32ScratchBytes + kFp16ScratchBytes; // 14336
    constexpr uint32_t kStageBytes = kTmp1Bytes - kScratchBytes;              // 18432
    constexpr uint32_t kHalfBytes = kStageBytes / 2U;                         // 9216

    LocalTensor<float> fp32UbA =
        tmpBuff1.GetWithOffset<float>(kScratchElems, kStageBytes);
    LocalTensor<float> fp32UbB =
        tmpBuff1.GetWithOffset<float>(kScratchElems, kStageBytes + kScratchElems * sizeof(float));
    LocalTensor<float> fp32UbC = tmpBuff1.GetWithOffset<float>(
        kScratchElems, kStageBytes + kScratchElems * 2U * sizeof(float));
    LocalTensor<half> halfScratch =
        tmpBuff1.GetWithOffset<half>(kScratchElems, kStageBytes + kFp32ScratchBytes);
    // Scalar meta reads reuse fp32UbA[0] before each Key tile clobbers scratch.

    GlobalTensor<uint8_t> srcGm = isKey ? keyCacheGm_ : valueCacheGm_;
    GlobalTensor<WS_T> dstWsGm = isKey ? dequantKeyWsGm_ : dequantValueWsGm_;

    event_t eventIdVWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdVWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMte3WaitV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdMte3WaitV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdMte2WaitS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));

    const uint32_t bs = constInfo.kvCacheBlockSize;
    const uint32_t codeRowBytes =
        isKey ? br_pack::BR_KEY_CODE_BYTES : br_pack::BR_VAL_CODE_BYTES;
    const uint32_t outRowBytes = headDimAlign * static_cast<uint32_t>(sizeof(WS_T));
    constexpr uint32_t kMetaSlot = br_dequant::BR_META_SLOT_BYTES;
    uint32_t maxSub = br_dequant::BR_S2_SUB_MAX;
    {
        uint32_t perRow = codeRowBytes + outRowBytes + 2U * kMetaSlot;
        uint32_t byUb = (kHalfBytes > 128U) ? ((kHalfBytes - 128U) / perRow) : 1U;
        if (byUb < maxSub) {
            maxSub = byUb;
        }
        if (maxSub < 1U) {
            maxSub = 1U;
        }
    }

    uint32_t si = siStart;
    uint32_t runId = 0U;
    while (si < siEnd) {
        uint32_t globalS2 = info.s2Idx * constInfo.s2BaseSize + si;
        uint32_t blockInBatch = globalS2 / bs;
        uint32_t pos0 = globalS2 % bs;
        uint32_t btIdx = info.bIdx * constInfo.maxBlockNumPerBatch + blockInBatch;

        SetFlag<HardEvent::S_MTE2>(eventIdMte2WaitS);
        WaitFlag<HardEvent::S_MTE2>(eventIdMte2WaitS);
        int32_t physBlockVal = blockTableGm_.GetValue(btIdx);
        uint64_t headBase = GetBrPackHeadBase(physBlockVal, info.n2Idx, isKey);

        // Extend run while same PA block and within this subcore's [si, siEnd).
        uint32_t n = 1U;
        uint32_t maxInBlock = bs - pos0;
        uint32_t nCap = maxSub < maxInBlock ? maxSub : maxInBlock;
        while (si + n < siEnd && n < nCap) {
            uint32_t gNext = info.s2Idx * constInfo.s2BaseSize + si + n;
            if ((gNext / bs) != blockInBatch) {
                break;
            }
            ++n;
        }

        const uint32_t bufIdx = runId % 2U;
        // Reuse this half only after its previous MTE3 store finished (runId >= 2).
        if (runId >= 2U) {
            event_t reuseEv = (bufIdx == 0U) ? eventIdMte3WaitV0 : eventIdMte3WaitV1;
            WaitFlag<HardEvent::MTE3_V>(reuseEv);
        }

        LocalTensor<uint8_t> batchUb =
            tmpBuff1.GetWithOffset<uint8_t>(kHalfBytes, bufIdx * kHalfBytes);

        const uint32_t codesBytes = n * codeRowBytes;
        const uint32_t meta0Off = br_dequant::BrAlignUp32(codesBytes);
        const uint32_t meta1Off = br_dequant::BrAlignUp32(meta0Off + n * kMetaSlot);
        const uint32_t outOff = br_dequant::BrAlignUp32(meta1Off + n * kMetaSlot);

        if (isKey) {
            uint64_t codeOff = br_pack::BrKeyCodeOffset(headBase, pos0);
            DataCopy(batchUb, srcGm[codeOff], codesBytes);
            br_dequant::BrCopyMetaRun(srcGm, batchUb[meta0Off],
                br_pack::BrKeyBaseOffset(headBase, bs, pos0), n);
            br_dequant::BrCopyMetaRun(srcGm, batchUb[meta1Off],
                br_pack::BrKeyStepOffset(headBase, bs, pos0), n);
        } else {
            uint64_t codeOff = br_pack::BrValCodeOffset(headBase, pos0);
            DataCopy(batchUb, srcGm[codeOff], codesBytes);
            br_dequant::BrCopyMetaRun(srcGm, batchUb[meta0Off],
                br_pack::BrValVminOffset(headBase, bs, pos0), n);
            br_dequant::BrCopyMetaRun(srcGm, batchUb[meta1Off],
                br_pack::BrValVstepOffset(headBase, bs, pos0), n);
        }

        SetFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);

        LocalTensor<WS_T> outBatch =
            batchUb[outOff].template ReinterpretCast<WS_T>();

        if (isKey) {
            for (uint32_t j0 = 0U; j0 < n; j0 += kTileMax) {
                uint32_t nb = n - j0;
                if (nb > kTileMax) {
                    nb = kTileMax;
                }
                float bases[kTileMax];
                float steps[kTileMax];
                for (uint32_t jj = 0U; jj < nb; ++jj) {
                    const uint32_t j = j0 + jj;
                    bases[jj] = br_dequant::BrReadMeta16FromUb<Q_T>(
                        batchUb, fp32UbA, meta0Off + j * kMetaSlot);
                    steps[jj] = br_dequant::BrReadMeta16FromUb<Q_T>(
                        batchUb, fp32UbA, meta1Off + j * kMetaSlot);
                }
                br_dequant::BrDecodeKeyTile(
                    batchUb[j0 * codeRowBytes], halfScratch, fp32UbA, fp32UbB, fp32UbC,
                    outBatch[j0 * headDimAlign], bases, steps, nb, headDim, headDimAlign);
            }
        } else {
            for (uint32_t j0 = 0U; j0 < n; j0 += kTileMax) {
                uint32_t nb = n - j0;
                if (nb > kTileMax) {
                    nb = kTileMax;
                }
                float vmins[kTileMax];
                float vsteps[kTileMax];
                for (uint32_t jj = 0U; jj < nb; ++jj) {
                    const uint32_t j = j0 + jj;
                    vmins[jj] = br_dequant::BrReadMeta16FromUb<Q_T>(
                        batchUb, fp32UbA, meta0Off + j * kMetaSlot);
                    vsteps[jj] = br_dequant::BrReadMeta16FromUb<Q_T>(
                        batchUb, fp32UbA, meta1Off + j * kMetaSlot);
                }
                br_dequant::BrDecodeValueTile(
                    batchUb[j0 * codeRowBytes], halfScratch, fp32UbA, fp32UbB,
                    outBatch[j0 * headDimAlign], vmins, vsteps, nb, headDim, headDimAlign);
            }
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVWaitMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVWaitMte3);
        uint64_t dstWsElemIdx = wsOffset + static_cast<uint64_t>(si) * headDimAlign;
        DataCopy(dstWsGm[dstWsElemIdx], outBatch, n * headDimAlign);
        event_t curEv = (bufIdx == 0U) ? eventIdMte3WaitV0 : eventIdMte3WaitV1;
        SetFlag<HardEvent::MTE3_V>(curEv);

        si += n;
        ++runId;
    }

    // Drain both ping-pong store halves.
    if (runId == 1U) {
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3WaitV0);
    } else if (runId >= 2U) {
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3WaitV0);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3WaitV1);
    }
}

#endif // FIA_BLOCK_VEC_TURBOQUANT_P0_H
