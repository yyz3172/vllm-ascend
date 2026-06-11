/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// TurboquantDecodePaged8bit AscendC kernel — Phase 1 / scheme A (design doc §2.6).
//
// The op folds block_table addressing into the kernel: it consumes the *full*
// packed KV cache plus a compact physical-block-id list (gather_block_ids,
// host-side ceil+cumsum) and writes fp16 K/V laid out compactly in seq order.
//
// Decode semantics (per packed row, design doc §2.6.4):
//   y_hat = 0.0026 * (idx - 127.5)   (V3 closed-form affine, 8-bit)
//   y_hat = y_hat * norm              (norm from the 2-byte fp16 slot)
//   x_hat = y_hat @ R                 (Haar rotation, fp16)
//
// Two tiling keys (design doc §2.6.5.8):
//   key 0  KFC mix (1 AIC + 2 AIV): V3 affine + Cube y_hat@R.             [perf]
//   key 1  AIV-only: fully scalar V3 affine + scalar matmul.               [golden]
//
// Work is split by *compact block* across cores so a tile never straddles a
// block boundary (BS*H is a multiple of T_rows), keeping addressing and the
// Cube M dimension static.

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#include "decode_device.h"

namespace {

using namespace AscendC;

constexpr uint32_t TQ_HEAD_SIZE = turboquant::TQ_DECODE_HEAD_SIZE;       // 128
constexpr uint32_t TQ_PACKED_BYTES = turboquant::TQ_DECODE_PACKED_BYTES;  // 130
constexpr uint32_t TQ_T_ROWS = 32;                                       // tile size (§2.6.3)
constexpr uint32_t TQ_ROT_WORKSPACE_BYTES = TQ_HEAD_SIZE * TQ_HEAD_SIZE * sizeof(half);  // 32 KB
// Per-core scratch layout within the workspace region (after KFC lib internals):
//   [cube C fp32: TQ_T_ROWS * HEAD_SIZE * sizeof(float) = 16 KB]
constexpr uint64_t TQ_PER_CORE_CUBEC_BYTES = static_cast<uint64_t>(TQ_T_ROWS) * TQ_HEAD_SIZE * sizeof(float);  // 16 KB
constexpr uint64_t TQ_PER_CORE_SCRATCH = TQ_PER_CORE_CUBEC_BYTES;  // 16 KB per core
// Offset from workspace base where per-core scratch begins. KFC lib internals use
// ~256 KB at the start; we start per-core allocation at 512 KB to be safe.
constexpr uint64_t TQ_PER_CORE_SCRATCH_BASE = 512 * 1024;

__aicore__ inline uint32_t AlignUp16(uint32_t x) {
    return (x + 15U) / 16U * 16U;
}

template <typename TilingT>
class TurboquantDecodePaged8bitKernel {
public:
    __aicore__ inline TurboquantDecodePaged8bitKernel(TPipe* pipe, turboquant::TqDecodeRotateMatmulOp* rotateMm)
        : pipe_(pipe), rotateMm_(rotateMm) {}

    __aicore__ inline void Init(
        __gm__ uint8_t* keyCache,
        __gm__ uint8_t* valueCache,
        __gm__ int32_t* gatherBlockIds,
        __gm__ half* rotation,
        __gm__ half* keyOut,
        __gm__ half* valueOut,
        __gm__ uint8_t* rawWorkspace,
        const TilingT* tiling) {
        totalBlocks_ = tiling->totalBlocks;
        blockSize_ = tiling->blockSize;
        numKvHeads_ = tiling->numKvHeads;
        headSize_ = tiling->headSize;
        packedBytes_ = tiling->packedBytes;
        blocksPerCore_ = tiling->blocksPerCore;
        mode_ = tiling->mode;
        kfcMixBlockDim_ = tiling->kfcMixBlockDim;

        rowsPerBlock_ = blockSize_ * numKvHeads_;  // BS * H packed rows per block
        const uint64_t totalRows = static_cast<uint64_t>(totalBlocks_) * rowsPerBlock_;
        const uint64_t cacheBytes = totalRows * packedBytes_;

        keyCacheGm_.SetGlobalBuffer(keyCache, cacheBytes);
        valueCacheGm_.SetGlobalBuffer(valueCache, cacheBytes);
        gatherBlockIdsGm_.SetGlobalBuffer(gatherBlockIds, totalBlocks_);
        rotationGm_.SetGlobalBuffer(rotation, static_cast<uint64_t>(TQ_HEAD_SIZE) * TQ_HEAD_SIZE);
        keyOutGm_.SetGlobalBuffer(keyOut, totalRows * headSize_);
        valueOutGm_.SetGlobalBuffer(valueOut, totalRows * headSize_);

        // UB layout: [TQ_T_ROWS * 128 compact index bytes][TQ_T_ROWS * 2 norm bytes].
        pipe_->InitBuffer(packedBuf_, TQ_T_ROWS * packedBytes_ * sizeof(uint8_t));
        pipe_->InitBuffer(xHatBuf_, TQ_T_ROWS * headSize_ * sizeof(half));
        pipe_->InitBuffer(rotateWorkBuf_, TQ_ROT_WORKSPACE_BYTES);
        if (mode_ == 0) {
            pipe_->InitBuffer(idxHalfBuf_, TQ_T_ROWS * headSize_ * sizeof(half));
            pipe_->InitBuffer(yHatBuf_, TQ_T_ROWS * headSize_ * sizeof(half));
            pipe_->InitBuffer(cubeFp32Buf_, TQ_T_ROWS * headSize_ * sizeof(float));

            matmulReady_ = (rawWorkspace != nullptr);
            if (matmulReady_) {
                // Per-core Cube C scratch: each blockIdx gets its own 16KB region
                // within the shared workspace.
                const uint32_t core = GetBlockIdx();
                auto* coreScratch = rawWorkspace + TQ_PER_CORE_SCRATCH_BASE
                                    + static_cast<uint64_t>(core) * TQ_PER_CORE_SCRATCH;
                cubeCGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(coreScratch),
                    static_cast<uint64_t>(TQ_T_ROWS) * headSize_);
            }
        }
    }

    // KFC path: V3 affine formula + Cube y_hat @ R.
    // On 910B3 each blockIdx corresponds to one AIV (sub-block indices alternate
    // 0,1,0,1... across blocks — all are independent workers). Only the AIC exits.
    __aicore__ inline void ProcessKfc() {
        if (!matmulReady_) {
            return;
        }
        if ASCEND_IS_AIC {
            return;
        }
        // Each blockIdx is its own data-parallel unit on 910B3
        // (CalcTschBlockDim returns 1, but hardware routes 2 consecutive
        // blockIds to the same MIX group — both AIVs are independent workers).
        const uint32_t core = GetBlockIdx();
        uint32_t blkStart = 0;
        uint32_t blkEnd = 0;
        if (!CoreBlockRange(core, blkStart, blkEnd)) {
            return;
        }

        auto packed = packedBuf_.Get<uint8_t>();
        auto idxHalf = idxHalfBuf_.Get<half>();
        auto yHat = yHatBuf_.Get<half>();
        auto cubeFp32 = cubeFp32Buf_.Get<float>();
        auto rotateWork = rotateWorkBuf_.Get<uint8_t>();
        auto xHat = xHatBuf_.Get<half>();

        // K segment then V segment: R stays in L1 across both (no re-handshake).
        DecodeSegmentKfc(keyCacheGm_, keyOutGm_, blkStart, blkEnd,
                         packed, idxHalf, yHat, rotateWork, cubeFp32, xHat);
        DecodeSegmentKfc(valueCacheGm_, valueOutGm_, blkStart, blkEnd,
                         packed, idxHalf, yHat, rotateWork, cubeFp32, xHat);
    }

    // AIV-only scalar reference path (golden / debug).
    __aicore__ inline void ProcessAivOnly() {
        if ASCEND_IS_AIC {
            return;
        }
        const uint32_t core = GetBlockIdx();
        uint32_t blkStart = 0;
        uint32_t blkEnd = 0;
        if (!CoreBlockRange(core, blkStart, blkEnd)) {
            return;
        }

        auto rotation = rotateWorkBuf_.Get<half>();
        DataCopy(rotation, rotationGm_, static_cast<uint32_t>(TQ_HEAD_SIZE * TQ_HEAD_SIZE));
        TqDecodeSyncMte2ToS();
        auto packed = packedBuf_.Get<uint8_t>();
        auto xHat = xHatBuf_.Get<half>();

        DecodeSegmentScalar(keyCacheGm_, keyOutGm_, blkStart, blkEnd, packed, rotation, xHat);
        DecodeSegmentScalar(valueCacheGm_, valueOutGm_, blkStart, blkEnd, packed, rotation, xHat);
    }

private:
    __aicore__ inline bool CoreBlockRange(uint32_t core, uint32_t& blkStart, uint32_t& blkEnd) {
        blkStart = core * blocksPerCore_;
        if (blkStart >= totalBlocks_) {
            return false;
        }
        blkEnd = blkStart + blocksPerCore_;
        if (blkEnd > totalBlocks_) {
            blkEnd = totalBlocks_;
        }
        return true;
    }

    __aicore__ inline void TqDecodeSyncMte2ToS() {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(e);
        WaitFlag<HardEvent::MTE2_S>(e);
    }

    __aicore__ inline void TqDecodeSyncVToMte3() {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
    }

    __aicore__ inline void TqDecodeSyncSToV() {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
    }

    __aicore__ inline void TqDecodeSyncSToMte3() {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(e);
        WaitFlag<HardEvent::S_MTE3>(e);
    }

    __aicore__ inline void TqDecodeSyncMte3ToS() {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(e);
        WaitFlag<HardEvent::MTE3_S>(e);
    }

    __aicore__ inline void LoadPackedTile(
        GlobalTensor<uint8_t>& srcCacheGm,
        const LocalTensor<uint8_t>& packed,
        uint64_t srcOff,
        uint32_t M) {
        // Compact 130-byte GM rows into a vector-friendly UB layout. The index
        // region becomes contiguous so DecodeRows8bit can widen it with Cast
        // instead of scalar GetValue/SetValue loops.
        for (uint32_t i = 0; i < M; ++i) {
            const uint64_t srcRow = srcOff + static_cast<uint64_t>(i) * packedBytes_;
            const uint32_t dstRow = i * TQ_HEAD_SIZE;
            DataCopy(packed[dstRow], srcCacheGm[srcRow], TQ_HEAD_SIZE);
        }
        TqDecodeSyncMte2ToS();

        // Keep fp16 norms in a compact tail after the index region.
        const uint32_t normBase = M * TQ_HEAD_SIZE;
        for (uint32_t i = 0; i < M; ++i) {
            const uint64_t srcRow = srcOff + static_cast<uint64_t>(i) * packedBytes_;
            const uint32_t dstNorm = normBase + i * sizeof(half);
            packed.SetValue(dstNorm, srcCacheGm.GetValue(srcRow + TQ_HEAD_SIZE));
            packed.SetValue(dstNorm + 1, srcCacheGm.GetValue(srcRow + TQ_HEAD_SIZE + 1));
        }
    }

    __aicore__ inline void DecodeSegmentKfc(
        GlobalTensor<uint8_t>& srcCacheGm,
        const GlobalTensor<half>& dstOutGm,
        uint32_t blkStart,
        uint32_t blkEnd,
        const LocalTensor<uint8_t>& packed,
        const LocalTensor<half>& idxHalf,
        const LocalTensor<half>& yHat,
        const LocalTensor<uint8_t>& rotateWork,
        const LocalTensor<float>& cubeFp32,
        const LocalTensor<half>& xHat) {
        for (uint32_t blk = blkStart; blk < blkEnd; ++blk) {
            const uint32_t phys = static_cast<uint32_t>(gatherBlockIdsGm_.GetValue(blk));
            const uint64_t srcBlockBase = static_cast<uint64_t>(phys) * rowsPerBlock_ * packedBytes_;
            const uint64_t dstBlockBase = static_cast<uint64_t>(blk) * rowsPerBlock_ * headSize_;

            for (uint32_t t = 0; t < rowsPerBlock_; t += TQ_T_ROWS) {
                const uint32_t M = (t + TQ_T_ROWS <= rowsPerBlock_) ? TQ_T_ROWS : (rowsPerBlock_ - t);
                const uint64_t srcOff = srcBlockBase + static_cast<uint64_t>(t) * packedBytes_;
                const uint64_t dstOff = dstBlockBase + static_cast<uint64_t>(t) * headSize_;

                LoadPackedTile(srcCacheGm, packed, srcOff, M);
                turboquant::DecodeRows8bit(
                    packed, rotationGm_, cubeCGm_, *rotateMm_,
                    idxHalf, yHat, rotateWork, cubeFp32, xHat,
                    M, AlignUp16(M));

                TqDecodeSyncVToMte3();
                DataCopy(dstOutGm[dstOff], xHat, M * headSize_);
            }
        }
    }

    __aicore__ inline void DecodeSegmentScalar(
        GlobalTensor<uint8_t>& srcCacheGm,
        GlobalTensor<half>& dstOutGm,
        uint32_t blkStart,
        uint32_t blkEnd,
        const LocalTensor<uint8_t>& packed,
        const LocalTensor<half>& rotation,
        const LocalTensor<half>& xHat) {
        for (uint32_t blk = blkStart; blk < blkEnd; ++blk) {
            const uint32_t phys = static_cast<uint32_t>(gatherBlockIdsGm_.GetValue(blk));
            const uint64_t srcBlockBase = static_cast<uint64_t>(phys) * rowsPerBlock_ * packedBytes_;
            const uint64_t dstBlockBase = static_cast<uint64_t>(blk) * rowsPerBlock_ * headSize_;

            for (uint32_t t = 0; t < rowsPerBlock_; t += TQ_T_ROWS) {
                const uint32_t M = (t + TQ_T_ROWS <= rowsPerBlock_) ? TQ_T_ROWS : (rowsPerBlock_ - t);
                const uint64_t srcOff = srcBlockBase + static_cast<uint64_t>(t) * packedBytes_;
                const uint64_t dstOff = dstBlockBase + static_cast<uint64_t>(t) * headSize_;

                LoadPackedTile(srcCacheGm, packed, srcOff, M);

                turboquant::DecodeRowsScalar(packed, rotation, xHat, M);

                // xHat is produced by scalar SetValue; order S writes before MTE3
                // reads it, then wait before reusing the UB tile on the next row.
                TqDecodeSyncSToMte3();
                DataCopy(dstOutGm[dstOff], xHat, M * headSize_);
                TqDecodeSyncMte3ToS();
            }
        }
    }

    TPipe* pipe_ = nullptr;
    turboquant::TqDecodeRotateMatmulOp* rotateMm_ = nullptr;
    bool matmulReady_ = false;

    uint32_t totalBlocks_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t blocksPerCore_ = 0;
    uint32_t mode_ = 0;
    uint32_t rowsPerBlock_ = 0;
    uint32_t kfcMixBlockDim_ = 1;

    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    GlobalTensor<int32_t> gatherBlockIdsGm_;
    GlobalTensor<half> rotationGm_;
    GlobalTensor<half> keyOutGm_;
    GlobalTensor<half> valueOutGm_;
    GlobalTensor<float> cubeCGm_;

    TBuf<TPosition::VECCALC> packedBuf_;
    TBuf<TPosition::VECCALC> idxHalfBuf_;
    TBuf<TPosition::VECOUT> yHatBuf_;     // Cube A input (KFC-readable UB)
    TBuf<TPosition::VECIN> cubeFp32Buf_;  // Cube C readback
    TBuf<TPosition::VECCALC> rotateWorkBuf_;
    TBuf<TPosition::VECCALC> xHatBuf_;
};

}  // namespace

extern "C" __global__ __aicore__ void turboquant_decode_paged8bit(
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR gather_block_ids,
    GM_ADDR codebook,
    GM_ADDR rotation,
    GM_ADDR key_out,
    GM_ADDR value_out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    auto* keyCachePtr = reinterpret_cast<__gm__ uint8_t*>(key_cache);
    auto* valueCachePtr = reinterpret_cast<__gm__ uint8_t*>(value_cache);
    auto* gatherPtr = reinterpret_cast<__gm__ int32_t*>(gather_block_ids);
    auto* rotationPtr = reinterpret_cast<__gm__ half*>(rotation);
    auto* keyOutPtr = reinterpret_cast<__gm__ half*>(key_out);
    auto* valueOutPtr = reinterpret_cast<__gm__ half*>(value_out);

    if (TILING_KEY_IS(0)) {
        // KFC mix: V3 affine + Cube y_hat @ R.
        // All MIX data-parallel groups share one workspace for KFC internals
        // (ClearWorkspaceImpl already indexes per-block via GetBlockIdxImpl).
        // Per-core scratch (cube C output) is allocated at fixed offsets.
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);

        auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
        AscendC::SetSysWorkspace(wsPtr);
        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }

        AscendC::TPipe pipe;
        turboquant::TqDecodeRotateMatmulOp rotateMm;
        TCubeTiling cubeTiling = tilingData.cubeTiling;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), rotateMm, &cubeTiling);
        TurboquantDecodePaged8bitKernel<TurboquantDecodePaged8bitTilingData> op(&pipe, &rotateMm);
        op.Init(keyCachePtr, valueCachePtr, gatherPtr, rotationPtr,
                keyOutPtr, valueOutPtr, wsPtr, &tilingData);
        op.ProcessKfc();
    } else if (TILING_KEY_IS(1)) {
        // AIV-only scalar reference.
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        TurboquantDecodePaged8bitKernel<TurboquantDecodePaged8bitTilingData> op(&pipe, nullptr);
        op.Init(keyCachePtr, valueCachePtr, gatherPtr, rotationPtr,
                keyOutPtr, valueOutPtr, nullptr, &tilingData);
        op.ProcessAivOnly();
    }
}
