/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Fused TurboQuant pack for K8V4 KV cache (key 8-bit, value 4-bit).
//
// Reference (Python, per side):
//   norms = vector_norm(x, dim=-1, keepdim=True)
//   y = (x / (norms + eps)) @ R^T          # [M, D] @ [D, D]
//   idx = argmin_k |y - codebook[k]|       # K: 256 entries, V: 16 entries
// Key  packs one uint8 index per dim (128 bytes) + fp16 norm  -> 130-byte row.
// Value packs uint4 indices, nibble layout matching ``pack_uint4``:
//   byte b (0..63): low nibble = idx[b]; high nibble = idx[b + 64], + fp16 norm
//   -> 66-byte row.
//
// Baseline shares the proven Cube rotate + normalize with the 8-bit pack op; the
// value side uses a scalar 16-entry argmin (cheap) plus nibble packing.

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

static constexpr int TQ_PACK_D = 128;
static constexpr int TQ_PACK_K_KEY = 256;   // 8-bit key codebook
static constexpr int TQ_PACK_K_VALUE = 16;  // 4-bit value codebook
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
static constexpr uint32_t TQ_MAX_BATCH_M = 128;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_LOCAL_WORKSPACE_BYTES = TQ_MAX_BATCH_M * TQ_ROT_K * sizeof(half);
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr uint64_t TQ_INVALID_OFFSET = ~static_cast<uint64_t>(0);
static constexpr uint32_t TQ_VALUE_INDEX_BYTES = TQ_PACK_D / 2;  // 64
static constexpr uint32_t TQ_REDUCE_MASK = 64;
static constexpr uint32_t TQ_REDUCE_BATCHES = (TQ_PACK_K_KEY + TQ_REDUCE_MASK - 1) / TQ_REDUCE_MASK;
static constexpr int32_t TQ_REDUCE_SRC_REP_STRIDE = TQ_REDUCE_MASK / 8;
static constexpr uint32_t TQ_D_TILE = 32;
static constexpr uint32_t TQ_V_D_TILE = 128;  // 4-bit: all dims in one sync (K=16 fits UB)
static constexpr uint32_t TQ_V_REDUCE_MASK = TQ_PACK_K_VALUE;
static constexpr uint32_t TQ_V_REDUCE_BATCHES = 1;
static constexpr int32_t TQ_V_REDUCE_SRC_REP_STRIDE = TQ_V_REDUCE_MASK / 8;

#if defined(ORIG_DTYPE_KEY)
#if (ORIG_DTYPE_KEY == DT_BF16)
using TqInputT = bfloat16_t;
#define TQ_INPUT_IS_BF16 1
#else
using TqInputT = half;
#define TQ_INPUT_IS_BF16 0
#endif
#elif defined(DTYPE_KEY)
#if (DTYPE_KEY == DT_BF16)
using TqInputT = bfloat16_t;
#define TQ_INPUT_IS_BF16 1
#else
using TqInputT = half;
#define TQ_INPUT_IS_BF16 0
#endif
#else
using TqInputT = half;
#define TQ_INPUT_IS_BF16 0
#endif

constexpr uint64_t TQ_PER_CORE_CUBEC_BYTES = static_cast<uint64_t>(TQ_MAX_BATCH_M) * TQ_ROT_N * sizeof(float);
constexpr uint64_t TQ_PER_CORE_SCRATCH = TQ_PER_CORE_CUBEC_BYTES;
constexpr uint64_t TQ_PER_CORE_SCRATCH_BASE = 512 * 1024;

__aicore__ inline uint32_t AlignUp16(uint32_t x) {
    return (x + TQ_CUBE_M_ALIGN - 1) / TQ_CUBE_M_ALIGN * TQ_CUBE_M_ALIGN;
}

__aicore__ inline void TqSyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void TqSyncVToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
}

__aicore__ inline void TqSyncSToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
}

__aicore__ inline void TqSyncSToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

__aicore__ inline bool TqIsAiv() {
    if ASCEND_IS_AIV {
        return true;
    }
    return false;
}

__aicore__ inline float TqAbsF32(float x) {
    return x < 0.f ? -x : x;
}

// Scalar argmin over the first ``kSize`` codebook entries (debug / fallback).
__aicore__ inline uint8_t ArgminAbsL1Scalar(
    float yf, const float* cb, uint32_t kSize) {
    float best = TqAbsF32(yf - cb[0]);
    uint8_t bestIdx = 0;
    for (uint32_t k = 1; k < kSize; ++k) {
        const float d = TqAbsF32(yf - cb[k]);
        if (d < best) {
            best = d;
            bestIdx = static_cast<uint8_t>(k);
        }
    }
    return bestIdx;
}

__aicore__ inline uint8_t MergeWholeReduceMinIdx(
    AscendC::LocalTensor<float>& argminRes, uint32_t numBatches, uint32_t maskPerBatch)
{
    float idxF = argminRes.GetValue(0);
    int32_t rawIdx = *reinterpret_cast<int32_t*>(&idxF);
    int32_t globalIdx = rawIdx;
    float globalMin = argminRes.GetValue(1);
    for (uint32_t b = 1; b < numBatches; ++b) {
        idxF = argminRes.GetValue(b * 2);
        rawIdx = *reinterpret_cast<int32_t*>(&idxF);
        const float batchMin = argminRes.GetValue(b * 2 + 1);
        if (batchMin < globalMin) {
            globalMin = batchMin;
            globalIdx = rawIdx + static_cast<int32_t>(b * maskPerBatch);
        }
    }
    return static_cast<uint8_t>(globalIdx);
}

__aicore__ inline uint8_t ArgminAbsL1Vector16(
    float yf,
    AscendC::LocalTensor<float>& cbFp32,
    AscendC::LocalTensor<float>& distRow,
    AscendC::LocalTensor<float>& argminRes)
{
    AscendC::Duplicate(distRow, yf, TQ_PACK_K_VALUE);
    AscendC::Sub(distRow, distRow, cbFp32, TQ_PACK_K_VALUE);
    AscendC::Abs(distRow, distRow, TQ_PACK_K_VALUE);
    AscendC::WholeReduceMin<float>(
        argminRes, distRow, TQ_V_REDUCE_MASK, TQ_V_REDUCE_BATCHES, 1, 1,
        TQ_V_REDUCE_SRC_REP_STRIDE, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
    AscendC::PipeBarrier<PIPE_V>();
    TqSyncVToS();
    const uint8_t idx = MergeWholeReduceMinIdx(argminRes, TQ_V_REDUCE_BATCHES, TQ_V_REDUCE_MASK);
    TqSyncSToV();
    return idx;
}

__aicore__ inline void write_norm_fp16_le_local(
    AscendC::LocalTensor<uint8_t>& packedLocal, uint32_t norm_off, half norm_h) {
    union {
        half h;
        uint16_t u;
    } normBits {};
    normBits.h = norm_h;
    packedLocal.SetValue(norm_off, (uint8_t)(normBits.u & 0xFFu));
    packedLocal.SetValue(norm_off + 1, (uint8_t)((normBits.u >> 8) & 0xFFu));
}

#if TQ_INPUT_IS_BF16
__aicore__ inline void write_norm_le_local(
    AscendC::LocalTensor<uint8_t>& packedLocal, uint32_t norm_off, float norm_f) {
    union {
        float f;
        uint32_t u;
    } bits {};
    bits.f = norm_f;
    const uint16_t bf16Bits = static_cast<uint16_t>(bits.u >> 16);
    packedLocal.SetValue(norm_off, (uint8_t)(bf16Bits & 0xFFu));
    packedLocal.SetValue(norm_off + 1, (uint8_t)((bf16Bits >> 8) & 0xFFu));
}
#else
__aicore__ inline void write_norm_le_local(
    AscendC::LocalTensor<uint8_t>& packedLocal, uint32_t norm_off, float norm_f) {
    write_norm_fp16_le_local(packedLocal, norm_off, static_cast<half>(norm_f));
}
#endif

__aicore__ inline void copy_packed_ub_to_gm(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t>& packedLocal,
    uint32_t nbytes) {
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
}

using TqRotateAT = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
using TqRotateBT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using TqRotateCT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using TqRotateBiasT = MatmulType<TPosition::GM, CubeFormat::ND, half>;

using TqRotateMatmulOp =
    AscendC::Matmul<TqRotateAT, TqRotateBT, TqRotateCT, TqRotateBiasT>;

class TurboquantPackKVForCacheK8v4 {
public:
    __aicore__ inline explicit TurboquantPackKVForCacheK8v4(AscendC::TPipe* pipe, TqRotateMatmulOp* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm) {}

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ half* codebook_k,
        __gm__ half* rotation_t_k,
        __gm__ half* codebook_v,
        __gm__ half* rotation_t_v,
        __gm__ int32_t* slot_mapping,
        __gm__ uint8_t* key_cache,
        __gm__ uint8_t* value_cache,
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t slot_w_k,
        uint32_t slot_w_v,
        uint32_t vecPerCore,
        uint32_t numHeads,
        uint32_t cacheSlots) {
        nVec_ = nVec;
        slot_w_k_ = slot_w_k;
        slot_w_v_ = slot_w_v;
        vecPerCore_ = vecPerCore;
        numHeads_ = numHeads == 0 ? 1 : numHeads;
        tokenCount_ = (nVec_ + numHeads_ - 1) / numHeads_;
        cacheSlots_ = cacheSlots;
        if (vecPerCore_ > TQ_MAX_BATCH_M) {
            vecPerCore_ = TQ_MAX_BATCH_M;
        }

        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TqInputT*>(key), (uint64_t)nVec_ * TQ_PACK_D);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TqInputT*>(value), (uint64_t)nVec_ * TQ_PACK_D);
        codebookKGm_.SetGlobalBuffer(codebook_k, TQ_PACK_K_KEY);
        rotationTKGm_.SetGlobalBuffer(rotation_t_k, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        codebookVGm_.SetGlobalBuffer(codebook_v, TQ_PACK_K_VALUE);
        rotationTVGm_.SetGlobalBuffer(rotation_t_v, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        slotMappingGm_.SetGlobalBuffer(slot_mapping, tokenCount_);
        keyCacheGm_.SetGlobalBuffer(key_cache, (uint64_t)cacheSlots_ * numHeads_ * slot_w_k_);
        valueCacheGm_.SetGlobalBuffer(value_cache, (uint64_t)cacheSlots_ * numHeads_ * slot_w_v_);

        const uint32_t batchElems = TQ_MAX_BATCH_M * TQ_PACK_D;
        pipe_->InitBuffer(xBatchQue_, 1, batchElems * sizeof(half));
        pipe_->InitBuffer(yBatchQue_, 1, batchElems * sizeof(half));
#if TQ_INPUT_IS_BF16
        pipe_->InitBuffer(xBatchInputQue_, 1, batchElems * sizeof(TqInputT));
#endif
        pipe_->InitBuffer(normScalarBuf_, TQ_UB_ALIGN);
        pipe_->InitBuffer(normsBuf_, TQ_MAX_BATCH_M * sizeof(float));
        pipe_->InitBuffer(codebookBuf_, TQ_PACK_K_KEY * sizeof(half));
        pipe_->InitBuffer(yFp32Buf_, TQ_PACK_D * sizeof(float));
        pipe_->InitBuffer(cbTileBuf_, TQ_PACK_K_KEY * sizeof(float));
        pipe_->InitBuffer(distBuf_, TQ_D_TILE * TQ_PACK_K_KEY * sizeof(float) + 256);
        // 4-bit encode may batch TQ_V_D_TILE results; 8-bit uses TQ_D_TILE * batches.
        const uint32_t argminElems =
            (TQ_V_D_TILE > TQ_D_TILE * TQ_REDUCE_BATCHES)
                ? (TQ_V_D_TILE * 2)
                : (TQ_D_TILE * TQ_REDUCE_BATCHES * 2);
        pipe_->InitBuffer(argminResultBuf_, argminElems * sizeof(float));
        pipe_->InitBuffer(reduceOutBuf_, TQ_PACK_D * 2 * sizeof(float));
        const uint32_t maxSlotW = slot_w_k_ > slot_w_v_ ? slot_w_k_ : slot_w_v_;
        pipe_->InitBuffer(packedRowBuf_, (maxSlotW + TQ_UB_ALIGN) * sizeof(uint8_t));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_LOCAL_WORKSPACE_BYTES);
        pipe_->InitBuffer(yCubeFp32Buf_, TQ_MAX_BATCH_M * TQ_ROT_N * sizeof(float));

        matmulReady_ = (rawWorkspace != nullptr);
        if (matmulReady_) {
            // Match Process(): logical MIX core is blockIdx/2 (primary AIV only).
            constexpr uint32_t kAivSub = 2;
            const uint32_t core = GetBlockIdx() / kAivSub;
            auto* coreScratch = rawWorkspace + TQ_PER_CORE_SCRATCH_BASE
                                + static_cast<uint64_t>(core) * TQ_PER_CORE_SCRATCH;
            cubeCGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(coreScratch),
                (uint64_t)TQ_MAX_BATCH_M * TQ_ROT_N);
        }
    }

    __aicore__ inline void ProcessNoKfc() {
        if ASCEND_IS_AIC {
            return;
        }
        if (!TqIsAiv()) {
            return;
        }
        // Match Process() core split so packMode=1 uses all AIV workers evenly.
        const uint32_t batchGroups =
            (nVec_ + vecPerCore_ - 1) / vecPerCore_;
        const uint32_t dataCores = batchGroups > 16U ? 16U : batchGroups;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t rowsPerCore = (nVec_ + dataCores - 1) / dataCores;
        const uint32_t coreStart = core * rowsPerCore;
        if (coreStart >= nVec_) {
            return;
        }
        const uint32_t coreEnd = coreStart + rowsPerCore > nVec_
            ? nVec_ : coreStart + rowsPerCore;

        for (uint32_t batchStart = coreStart; batchStart < coreEnd;
             batchStart += vecPerCore_) {
            const uint32_t batchEnd = batchStart + vecPerCore_ > coreEnd
                ? coreEnd : batchStart + vecPerCore_;

            PackBatch(keyGm_, keyCacheGm_, batchStart, batchEnd,
                      slot_w_k_, /*fourBit=*/false, true);
            PackBatch(valueGm_, valueCacheGm_, batchStart, batchEnd,
                      slot_w_v_, /*fourBit=*/true, true);
        }
    }

    __aicore__ inline void Process() {
        if (!matmulReady_ || !TqIsAiv()) {
            return;
        }
        // MIX_AIC_1_2: each AIV has its own UB — no buffer doubling needed.
        // sub0 = Key with Cube KFC rotate; sub1 = Value with AIV vector rotate
        // (avoids concurrent IterateAll on the same Matmul/KFC client).
        constexpr uint32_t kAivSub = 2;
        const uint32_t sub = AscendC::GetSubBlockIdx() % kAivSub;
        const uint32_t batchGroups =
            (nVec_ + vecPerCore_ - 1) / vecPerCore_;
        const uint32_t dataCores = batchGroups > 16U ? 16U : batchGroups;
        const uint32_t core = AscendC::GetBlockIdx() / kAivSub;
        const uint32_t rowsPerCore = (nVec_ + dataCores - 1) / dataCores;
        const uint32_t coreStart = core * rowsPerCore;
        if (coreStart >= nVec_) {
            return;
        }
        const uint32_t coreEnd = coreStart + rowsPerCore > nVec_
            ? nVec_ : coreStart + rowsPerCore;

        for (uint32_t batchStart = coreStart; batchStart < coreEnd;
             batchStart += vecPerCore_) {
            const uint32_t batchEnd = batchStart + vecPerCore_ > coreEnd
                ? coreEnd : batchStart + vecPerCore_;

            if (sub == 0) {
                PackBatch(keyGm_, keyCacheGm_, batchStart, batchEnd,
                          slot_w_k_, /*fourBit=*/false, /*manualRotate=*/false);
            } else {
                PackBatch(valueGm_, valueCacheGm_, batchStart, batchEnd,
                          slot_w_v_, /*fourBit=*/true, /*manualRotate=*/true);
            }
        }
    }

private:
    __aicore__ inline uint64_t CacheOutBase(uint32_t vecIdx, uint32_t slot_w) {
        const uint32_t tokenIdx = vecIdx / numHeads_;
        if (tokenIdx >= tokenCount_) {
            return TQ_INVALID_OFFSET;
        }
        const int32_t slot = slotMappingGm_.GetValue(tokenIdx);
        if (slot < 0 || static_cast<uint32_t>(slot) >= cacheSlots_) {
            return TQ_INVALID_OFFSET;
        }
        const uint32_t headIdx = vecIdx - tokenIdx * numHeads_;
        return ((uint64_t)static_cast<uint32_t>(slot) * numHeads_ + headIdx) * slot_w;
    }

    __aicore__ inline void NormalizeBatch(
        AscendC::LocalTensor<half>& xBatch, AscendC::LocalTensor<float>& norms, uint32_t m) {
        auto fp32Row = reduceOutBuf_.Get<float>();
        auto fp32Tmp = reduceOutBuf_.Get<float>()[TQ_PACK_D];
        auto normAcc = normScalarBuf_.Get<float>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            AscendC::Cast(fp32Row, xBatch[rowOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::Mul(fp32Row, fp32Row, fp32Row, TQ_PACK_D);
            AscendC::ReduceSum<float>(normAcc, fp32Row, fp32Tmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(normAcc, normAcc, 1);
            TqSyncVToS();
            const float normF = normAcc.GetValue(0);
            TqSyncSToV();

            norms.SetValue(i, normF);
            const half invH = static_cast<half>(1.0f / (normF + TQ_NORM_EPS_F));
            AscendC::Muls(xBatch[rowOff], xBatch[rowOff], invH, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Vector rotate: x[m,D] @ R^T[D,D] on AIV (packMode=1). Matches v3 pack path.
    __aicore__ inline void LoadRotationToUb(AscendC::GlobalTensor<half>& rotationTGm) {
        auto rotationTLocal = rotateWorkBuf_.Get<half>();
        AscendC::DataCopy(rotationTLocal, rotationTGm, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        TqSyncMte2ToV();
    }

    __aicore__ inline void RotateBatchMatmulVector(
        AscendC::LocalTensor<half>& xUnitBatch,
        AscendC::LocalTensor<half>& yBatch,
        uint32_t m) {
        auto rotationTLocal = rotateWorkBuf_.Get<half>();
        auto acc = yFp32Buf_.Get<float>();
        auto rotFp32 = reduceOutBuf_.Get<float>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t xOff = i * TQ_PACK_D;
            auto xBlock = xUnitBatch[xOff];

            AscendC::Duplicate(acc, 0.0f, TQ_PACK_D);
            for (uint32_t k = 0; k < TQ_PACK_D; ++k) {
                const uint32_t rotOff = k * TQ_PACK_D;
                const float xVal = static_cast<float>(xBlock.GetValue(k));
                AscendC::Cast(rotFp32, rotationTLocal[rotOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
                AscendC::Muls(rotFp32, rotFp32, xVal, TQ_PACK_D);
                AscendC::Add(acc, acc, rotFp32, TQ_PACK_D);
            }
            AscendC::Cast(yBatch[i * TQ_ROT_N], acc, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RotateBatchMatmul(
        AscendC::LocalTensor<half>& xUnitBatch,
        AscendC::LocalTensor<half>& yBatch,
        AscendC::GlobalTensor<half>& rotationTGm,
        uint32_t m,
        uint32_t mPad) {
        rotateMm_->SetOrgShape(mPad, TQ_ROT_N, TQ_ROT_K);
        rotateMm_->SetSingleShape(m, TQ_ROT_N, TQ_ROT_K);
        rotateMm_->SetTensorA(xUnitBatch, false);
        rotateMm_->SetTensorB(rotationTGm, false);
        auto rotateWorkspace = rotateWorkBuf_.Get<uint8_t>();
        rotateMm_->SetLocalWorkspace(rotateWorkspace);
        rotateMm_->IterateAll(cubeCGm_);
        rotateMm_->End();

        auto yFp32 = yCubeFp32Buf_.Get<float>();
        AscendC::DataCopy(yFp32, cubeCGm_, m * TQ_ROT_N);
        TqSyncMte2ToV();
        AscendC::Cast(yBatch, yFp32, AscendC::RoundMode::CAST_NONE, m * TQ_ROT_N);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RotateBatchMatmulManual(
        AscendC::LocalTensor<half>& xUnitBatch,
        AscendC::LocalTensor<half>& yBatch,
        AscendC::GlobalTensor<half>& rotationTGm,
        uint32_t m) {
        LoadRotationToUb(rotationTGm);
        RotateBatchMatmulVector(xUnitBatch, yBatch, m);
    }

    // Encode one batch of rotated rows to packed cache rows.
    //   fourBit=false: 256-codebook, vectorized WholeReduceMin (key, 130-byte row)
    //   fourBit=true : 16-codebook, vectorized WholeReduceMin + nibble pack (value, 66-byte row)
    __aicore__ inline void EncodeBatch(
        AscendC::LocalTensor<half>& yBatch,
        AscendC::LocalTensor<float>& norms,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        AscendC::GlobalTensor<half>& codebookGm,
        uint32_t start,
        uint32_t m,
        uint32_t slot_w,
        bool fourBit,
        bool writeMeta) {
        const uint32_t kSize = fourBit ? TQ_PACK_K_VALUE : TQ_PACK_K_KEY;
        const uint32_t normOff = fourBit ? TQ_VALUE_INDEX_BYTES : (uint32_t)TQ_PACK_D;

        auto codebookLocal = codebookBuf_.Get<half>();
        AscendC::DataCopy(codebookLocal, codebookGm[0], kSize);
        TqSyncMte2ToV();
        auto cbFp32 = cbTileBuf_.Get<float>();
        if (fourBit) {
            AscendC::Cast(cbFp32, codebookLocal, AscendC::RoundMode::CAST_NONE, kSize);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(cbFp32, codebookLocal, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(cbFp32[TQ_PACK_D], codebookLocal[TQ_PACK_D], AscendC::RoundMode::CAST_NONE,
                          TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        auto yFp32 = yFp32Buf_.Get<float>();
        auto distTile = distBuf_.Get<float>();
        auto argminRes = argminResultBuf_.Get<float>();
        auto packedRow = packedRowBuf_.Get<uint8_t>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t vecIdx = start + i;
            const uint64_t out_base = CacheOutBase(vecIdx, slot_w);
            if (out_base == TQ_INVALID_OFFSET) {
                continue;
            }
            const uint32_t yOff = i * TQ_ROT_N;

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            float yVals[TQ_PACK_D];
            for (uint32_t d = 0; d < TQ_PACK_D; ++d) {
                yVals[d] = yFp32.GetValue(d);
            }
            TqSyncSToV();

            if (fourBit) {
                // Batch TQ_V_D_TILE dims per sync. distBuf holds TQ_D_TILE*256 floats;
                // with K=16 that fits TQ_V_D_TILE rows (32*16 < 16*256).
                uint8_t idxs[TQ_PACK_D];
                for (uint32_t tileStart = 0; tileStart < TQ_PACK_D; tileStart += TQ_V_D_TILE) {
                    const uint32_t tileCnt = (tileStart + TQ_V_D_TILE <= TQ_PACK_D)
                                                 ? TQ_V_D_TILE
                                                 : (TQ_PACK_D - tileStart);
                    for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                        auto distRow = distTile[dl * TQ_PACK_K_VALUE];
                        AscendC::Duplicate(distRow, yVals[tileStart + dl], TQ_PACK_K_VALUE);
                        AscendC::Sub(distRow, distRow, cbFp32, TQ_PACK_K_VALUE);
                        AscendC::Abs(distRow, distRow, TQ_PACK_K_VALUE);
                    }
                    for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                        auto distRow = distTile[dl * TQ_PACK_K_VALUE];
                        auto resRow = argminRes[dl * 2];
                        AscendC::WholeReduceMin<float>(
                            resRow, distRow, TQ_V_REDUCE_MASK, TQ_V_REDUCE_BATCHES, 1, 1,
                            TQ_V_REDUCE_SRC_REP_STRIDE, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    TqSyncVToS();
                    for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                        auto resRow = argminRes[dl * 2];
                        idxs[tileStart + dl] =
                            MergeWholeReduceMinIdx(resRow, TQ_V_REDUCE_BATCHES, TQ_V_REDUCE_MASK);
                    }
                    TqSyncSToV();
                }
                for (uint32_t b = 0; b < TQ_VALUE_INDEX_BYTES; ++b) {
                    packedRow.SetValue(
                        b, static_cast<uint8_t>((idxs[b] & 0x0Fu) | ((idxs[b + TQ_VALUE_INDEX_BYTES] & 0x0Fu) << 4)));
                }
            } else {
                // 8-bit key: build + reduce per tile, one V<->S sync per tile (not per dim).
                for (uint32_t tileStart = 0; tileStart < TQ_PACK_D; tileStart += TQ_D_TILE) {
                    const uint32_t tileCnt = (tileStart + TQ_D_TILE <= TQ_PACK_D)
                                                 ? TQ_D_TILE
                                                 : (TQ_PACK_D - tileStart);
                    for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                        auto distRow = distTile[dl * TQ_PACK_K_KEY];
                        AscendC::Duplicate(distRow, yVals[tileStart + dl], TQ_PACK_K_KEY);
                        AscendC::Sub(distRow, distRow, cbFp32, TQ_PACK_K_KEY);
                        AscendC::Abs(distRow, distRow, TQ_PACK_K_KEY);
                    }
                    for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                        auto distRow = distTile[dl * TQ_PACK_K_KEY];
                        auto resRow = argminRes[dl * TQ_REDUCE_BATCHES * 2];
                        AscendC::WholeReduceMin<float>(
                            resRow, distRow, TQ_REDUCE_MASK, TQ_REDUCE_BATCHES, 1, 1,
                            TQ_REDUCE_SRC_REP_STRIDE, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    TqSyncVToS();
                    for (uint32_t dl = 0; dl < tileCnt; ++dl) {
                        auto resRow = argminRes[dl * TQ_REDUCE_BATCHES * 2];
                        packedRow.SetValue(
                            tileStart + dl,
                            MergeWholeReduceMinIdx(resRow, TQ_REDUCE_BATCHES, TQ_REDUCE_MASK));
                    }
                    TqSyncSToV();
                }
            }

            for (uint32_t k = normOff + 2; k < slot_w; ++k) {
                packedRow.SetValue(k, (uint8_t)0);
            }
            if (writeMeta) {
                TqSyncVToS();
                const float normF = norms.GetValue(i);
                TqSyncSToV();
                write_norm_le_local(packedRow, normOff, normF);
            }
            TqSyncSToV();
            copy_packed_ub_to_gm(packedGm, out_base, packedRow, slot_w);
        }
    }

    __aicore__ inline void PackBatch(
        const AscendC::GlobalTensor<TqInputT>& xGm,
        AscendC::GlobalTensor<uint8_t>& packedGm,
        uint32_t start,
        uint32_t end,
        uint32_t slot_w,
        bool fourBit,
        bool manualRotate) {
        const uint32_t m = end - start;
        const uint32_t mPad = AlignUp16(m);

        auto xBatch = xBatchQue_.AllocTensor<half>();
        auto yBatch = yBatchQue_.AllocTensor<half>();
        auto norms = normsBuf_.Get<float>();

        auto& rotationTGm = fourBit ? rotationTVGm_ : rotationTKGm_;
        auto& codebookGm = fourBit ? codebookVGm_ : codebookKGm_;

#if TQ_INPUT_IS_BF16
        auto xBatchInput = xBatchInputQue_.AllocTensor<TqInputT>();
        auto xBatchFp32 = yCubeFp32Buf_.Get<float>();
        AscendC::DataCopy(xBatchInput, xGm[(uint64_t)start * TQ_PACK_D], m * TQ_PACK_D);
        TqSyncMte2ToV();
        AscendC::Cast(xBatchFp32, xBatchInput, AscendC::RoundMode::CAST_NONE, m * TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(xBatch, xBatchFp32, AscendC::RoundMode::CAST_ROUND, m * TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        xBatchInputQue_.FreeTensor(xBatchInput);
#else
        AscendC::DataCopy(xBatch, xGm[(uint64_t)start * TQ_PACK_D], m * TQ_PACK_D);
        TqSyncMte2ToV();
#endif

        if (mPad > m) {
            AscendC::Duplicate(xBatch[m * TQ_PACK_D], (half)0, (mPad - m) * TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        NormalizeBatch(xBatch, norms, m);
        if (manualRotate) {
            RotateBatchMatmulManual(xBatch, yBatch, rotationTGm, m);
        } else {
            RotateBatchMatmul(xBatch, yBatch, rotationTGm, m, mPad);
        }
        EncodeBatch(yBatch, norms, packedGm, codebookGm, start, m, slot_w, fourBit, true);

        yBatchQue_.FreeTensor(yBatch);
        xBatchQue_.FreeTensor(xBatch);
    }

private:
    AscendC::TPipe* pipe_ = nullptr;
    TqRotateMatmulOp* rotateMm_ = nullptr;
    uint32_t nVec_ = 0;
    uint32_t slot_w_k_ = 0;
    uint32_t slot_w_v_ = 0;
    uint32_t vecPerCore_ = 1;
    uint32_t numHeads_ = 1;
    uint32_t tokenCount_ = 0;
    uint32_t cacheSlots_ = 0;
    bool matmulReady_ = false;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> xBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yBatchQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xBatchInputQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> normsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> codebookBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cbTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> distBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> argminResultBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> yCubeFp32Buf_;

    AscendC::GlobalTensor<float> cubeCGm_;
    AscendC::GlobalTensor<TqInputT> keyGm_;
    AscendC::GlobalTensor<TqInputT> valueGm_;
    AscendC::GlobalTensor<half> codebookKGm_;
    AscendC::GlobalTensor<half> rotationTKGm_;
    AscendC::GlobalTensor<half> codebookVGm_;
    AscendC::GlobalTensor<half> rotationTVGm_;
    AscendC::GlobalTensor<int32_t> slotMappingGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void turboquant_pack_kv_for_cache_k8v4(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR codebook,
    GM_ADDR rotation_t,
    GM_ADDR codebook_value,
    GM_ADDR rotation_t_value,
    GM_ADDR slot_mapping,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    auto* codebookKPtr = reinterpret_cast<__gm__ half*>(codebook);
    auto* rotationKPtr = reinterpret_cast<__gm__ half*>(rotation_t);
    auto* codebookVPtr = reinterpret_cast<__gm__ half*>(codebook_value);
    auto* rotationVPtr = reinterpret_cast<__gm__ half*>(rotation_t_value);
    auto* slotMappingPtr = reinterpret_cast<__gm__ int32_t*>(slot_mapping);
    auto* keyCachePtr = reinterpret_cast<__gm__ uint8_t*>(key_cache);
    auto* valueCachePtr = reinterpret_cast<__gm__ uint8_t*>(value_cache);

    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
        AscendC::SetSysWorkspace(wsPtr);
        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }
        AscendC::TPipe pipe;
        TqRotateMatmulOp rotateMm;
        TCubeTiling cubeTiling = tilingData.cubeTiling;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &cubeTiling);
        TurboquantPackKVForCacheK8v4 op(&pipe, &rotateMm);
        op.Init(
            key, value, codebookKPtr, rotationKPtr, codebookVPtr, rotationVPtr,
            slotMappingPtr, keyCachePtr, valueCachePtr, wsPtr,
            tilingData.nVec, tilingData.slotWK, tilingData.slotWV,
            tilingData.vecPerCore, tilingData.numHeads, tilingData.cacheSlots);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        TurboquantPackKVForCacheK8v4 op(&pipe, nullptr);
        op.Init(
            key, value, codebookKPtr, rotationKPtr, codebookVPtr, rotationVPtr,
            slotMappingPtr, keyCachePtr, valueCachePtr, nullptr,
            tilingData.nVec, tilingData.slotWK, tilingData.slotWV,
            tilingData.vecPerCore, tilingData.numHeads, tilingData.cacheSlots);
        op.ProcessNoKfc();
    }
}
