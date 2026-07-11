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

#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"

namespace bit_residual {

using namespace AscendC;

static constexpr int TQ_PACK_D = 128;
static constexpr uint32_t TQ_UB_ALIGN = 32;
static constexpr uint32_t TQ_KEY_GROUP_ROWS = 2;
static constexpr uint32_t TQ_VAL_GROUP_ROWS = 4;
static constexpr uint32_t TQ_GROUP_INDEX_BYTES = TQ_PACK_D * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_GROUP_NORM_BYTES = TQ_KEY_GROUP_ROWS * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_GROUP_BASE_BYTES = TQ_KEY_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_KEY_GROUP_STEP_BYTES = TQ_KEY_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_KEY_GROUP_NORM_OFFSET = TQ_GROUP_INDEX_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_BASE_OFFSET =
    TQ_KEY_GROUP_NORM_OFFSET + TQ_KEY_GROUP_NORM_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_STEP_OFFSET =
    TQ_KEY_GROUP_BASE_OFFSET + TQ_KEY_GROUP_BASE_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_BYTES =
    TQ_GROUP_INDEX_BYTES + TQ_KEY_GROUP_NORM_BYTES +
    TQ_KEY_GROUP_BASE_BYTES + TQ_KEY_GROUP_STEP_BYTES;
static constexpr uint32_t TQ_KEY_GROUP_STRIDE =
    (TQ_KEY_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_VAL_GROUP_VMIN_BYTES = TQ_VAL_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_VAL_GROUP_VSTEP_BYTES = TQ_VAL_GROUP_ROWS * sizeof(float);
static constexpr uint32_t TQ_VAL_GROUP_VMIN_OFFSET = TQ_GROUP_INDEX_BYTES;
static constexpr uint32_t TQ_VAL_GROUP_VSTEP_OFFSET =
    TQ_VAL_GROUP_VMIN_OFFSET + TQ_VAL_GROUP_VMIN_BYTES;
static constexpr uint32_t TQ_VAL_GROUP_BYTES =
    TQ_GROUP_INDEX_BYTES + TQ_VAL_GROUP_VMIN_BYTES + TQ_VAL_GROUP_VSTEP_BYTES;
static constexpr uint32_t TQ_VAL_GROUP_STRIDE =
    (TQ_VAL_GROUP_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_PACKED_GROUP_STRIDE =
    TQ_KEY_GROUP_STRIDE > TQ_VAL_GROUP_STRIDE ? TQ_KEY_GROUP_STRIDE : TQ_VAL_GROUP_STRIDE;
static constexpr uint32_t TQ_PACKED_GROUP_BUFFER_COUNT = 64;
static constexpr uint32_t TQ_KEY_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(uint16_t) + 2 * sizeof(float);
static constexpr uint32_t TQ_KEY_ENCODED_NORM_WORD_OFFSET = TQ_PACK_D;
static constexpr uint32_t TQ_KEY_ENCODED_BASE_BYTE_OFFSET =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_ENCODED_STEP_BYTE_OFFSET =
    TQ_KEY_ENCODED_BASE_BYTE_OFFSET + sizeof(float);
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_BYTES =
    (TQ_KEY_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_WORDS =
    TQ_KEY_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t) + 2 * sizeof(float);
static constexpr uint32_t TQ_VAL_ENCODED_VMIN_BYTE_OFFSET = TQ_PACK_D * sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET =
    TQ_VAL_ENCODED_VMIN_BYTE_OFFSET + sizeof(float);
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_BYTES =
    (TQ_VAL_ENCODED_ROW_BYTES + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_WORDS =
    TQ_VAL_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
static constexpr uint32_t TQ_MAX_BATCH_M = 64;
static constexpr uint32_t TQ_BATCH_ELEMS = TQ_MAX_BATCH_M * TQ_PACK_D;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_DTYPE_BYTES = sizeof(uint16_t);
static constexpr uint32_t TQ_NORM_STRIDE = TQ_UB_ALIGN / TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_MANUAL_WORKSPACE_BYTE_OFFSET = 512 * 1024;
static constexpr uint32_t TQ_MANUAL_GROUP_ROWS = TQ_VAL_GROUP_ROWS;
static constexpr uint32_t TQ_MANUAL_ROT_TILE_M = 32;
static constexpr uint32_t TQ_MANUAL_ROT_TILE_ELEMS = TQ_MANUAL_ROT_TILE_M * TQ_ROT_K;
static constexpr uint32_t TQ_MANUAL_HEADS_PER_TILE =
    TQ_MANUAL_ROT_TILE_M / TQ_MANUAL_GROUP_ROWS;
static constexpr uint32_t TQ_MANUAL_AIV_SLICE_M = TQ_MANUAL_ROT_TILE_M / 2;
static constexpr uint32_t TQ_MANUAL_HEADS_PER_AIV =
    TQ_MANUAL_AIV_SLICE_M / TQ_MANUAL_GROUP_ROWS;
static constexpr uint32_t TQ_MANUAL_AIV_SLICE_ELEMS =
    TQ_MANUAL_AIV_SLICE_M * TQ_ROT_K;
static constexpr uint32_t TQ_MANUAL_ROT_A_L1_OFFSET = 0;
static constexpr uint32_t TQ_MANUAL_ROT_B_L1_OFFSET =
    TQ_MANUAL_ROT_A_L1_OFFSET + TQ_MANUAL_ROT_TILE_ELEMS * TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_MANUAL_NORM_B_L0_BYTE_OFFSET =
    TQ_ROT_K * TQ_ROT_N * TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_MANUAL_NORM_A_L0_BYTE_OFFSET =
    TQ_MANUAL_ROT_TILE_M * TQ_ROT_K * TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_MANUAL_NORM_C_L0_BYTE_OFFSET =
    TQ_MANUAL_ROT_TILE_M * TQ_ROT_N * sizeof(float);
static constexpr uint32_t TQ_MANUAL_NORM_A_L0_EVENT = EVENT_ID0;
static constexpr uint32_t TQ_MANUAL_ROT_A_L0_EVENT = EVENT_ID3;
static constexpr uint32_t TQ_MANUAL_NORM_B_L0_EVENT = EVENT_ID2; 
static constexpr uint32_t TQ_MANUAL_ROT_B_L0_EVENT = EVENT_ID1; 
static constexpr uint32_t TQ_MANUAL_NORM_C_L0_EVENT = EVENT_ID0; 
static constexpr uint32_t TQ_MANUAL_ROT_C_L0_EVENT = EVENT_ID1; 
static constexpr uint32_t TQ_MANUAL_WORKSPACE_BUFFER_COUNT = 2;
static constexpr uint32_t TQ_MANUAL_WORKSPACE_STRIDE_ELEMS = TQ_BATCH_ELEMS;
static constexpr uint32_t TQ_MANUAL_NORM_MATRIX_DIM = TQ_MANUAL_AIV_SLICE_M;
static constexpr uint32_t TQ_MANUAL_NORM_MATRIX_ELEMS =
    TQ_MANUAL_NORM_MATRIX_DIM * TQ_MANUAL_NORM_MATRIX_DIM;
static constexpr uint32_t TQ_MANUAL_NORM_WORKSPACE_STRIDE_FLOATS =
    TQ_MANUAL_WORKSPACE_STRIDE_ELEMS * TQ_DTYPE_BYTES / sizeof(float);
static constexpr uint32_t TQ_MANUAL_WORKSPACE_ELEMS_PER_CORE =
    TQ_MANUAL_WORKSPACE_BUFFER_COUNT * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS * 2;
static constexpr uint32_t TQ_MANUAL_C_WORKSPACE_ELEM_OFFSET =
    TQ_MANUAL_WORKSPACE_BUFFER_COUNT * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS;
static constexpr uint32_t TQ_MANUAL_STREAM_KIND_COUNT = 2;
static constexpr uint32_t TQ_MANUAL_NORM_BUFFER_COUNT = TQ_MANUAL_STREAM_KIND_COUNT;
static constexpr uint32_t TQ_MANUAL_NORM_SLOT_ELEMS =
    TQ_MANUAL_AIV_SLICE_M * TQ_NORM_STRIDE;
static_assert(TQ_MANUAL_NORM_BUFFER_COUNT * TQ_MANUAL_NORM_SLOT_ELEMS <=
              TQ_MAX_BATCH_M * TQ_NORM_STRIDE,
              "Manual key1 norm ping-pong must fit in normsBuf_.");
static constexpr uint16_t TQ_MANUAL_SYNC_A_FREE = 0;
static constexpr uint16_t TQ_MANUAL_SYNC_A_READY = 1;
static constexpr uint16_t TQ_MANUAL_SYNC_C_FREE = 2;
static constexpr uint16_t TQ_MANUAL_SYNC_C_READY = 3;
static constexpr uint16_t TQ_MANUAL_SYNC_PP_STRIDE = 4;
static constexpr float TQ_NORM_EPS_F = 1e-10f;
static constexpr float TQ_INV_SQRT_D_F = 0.08838834764831845f;
static constexpr float TQ_KEY_QUANT_LEVELS_F = 127.0f;
static constexpr float TQ_VAL_QUANT_LEVELS_F = 15.0f;
static constexpr uint32_t TQ_SIGN_MASK_BYTES = 256;
static constexpr uint32_t TQ_QUANT_INDEX_BYTES = TQ_PACK_D * sizeof(int32_t);
static constexpr uint32_t TQ_QUANT_INDEX_U16_BYTES = TQ_PACK_D * sizeof(int16_t);
static constexpr uint32_t TQ_AIV_SUB_BLOCKS = 2;

struct ManualKey1StreamDesc {
    uint32_t tokenStart;
    uint32_t slotStart;
    uint32_t blockIdx;
    uint32_t valueGroupInBlock;
    uint32_t headTileStart;
    uint32_t startGroupRow;
    uint32_t validRows;
    uint32_t streamOrdinal;
    bool preserveValue;
    bool isValue;
};

// ── LocalTensor static buffer utilities (following turboquant_pack_kv_for_cache4bit) ──────
static constexpr uint32_t TqAlignUp32(uint32_t x) {
    return (x + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
}

template <typename T>
__aicore__ inline AscendC::LocalTensor<T> TqMakeLocalTensor(
    TPosition pos,
    uint32_t address,
    uint32_t elems) {
    AscendC::TBuffAddr tensorAddr {};
    tensorAddr.dataLen = elems * sizeof(T);
    tensorAddr.bufferAddr = address;
    tensorAddr.bufferHandle = nullptr;
    tensorAddr.logicPos = static_cast<uint8_t>(pos);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    tensorAddr.absAddr = GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(pos)) + address;
#endif
    AscendC::LocalTensor<T> tensor;
    tensor.SetAddr(tensorAddr);
    return tensor;
}

template <TPosition Pos>
struct TqStaticLocalBuffer {
    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> GetBufferByByte(
        uint32_t address,
        uint32_t bytes) const {
        return TqMakeLocalTensor<T>(Pos, address, bytes / sizeof(T));
    }
};

struct TqStaticLocalResource {
    TqStaticLocalBuffer<TPosition::VECIN> vecIn;
    TqStaticLocalBuffer<TPosition::VECOUT> vecOut;
    TqStaticLocalBuffer<TPosition::VECCALC> vecCalc;
};

// ── VECCALC (UB) static layout ──────────────────────────────────────────────────────
//
// Hand-built LocalTensor buffer addresses live in one logical UB address space.
// Do not reuse the same byte offsets across VECIN/VECOUT/VECCALC positions.
// The pack pipeline only needs two large row buffers at the same time:
//   normalize: XBatch + ABatch
//   encode:    YBatch + EncodedBatch
// Therefore X/Y share one slot, and A/encoded rows share another slot.
static constexpr uint32_t TQ_UB_XY_BATCH_OFFSET = 0;
static constexpr uint32_t TQ_UB_A_ENCODED_BATCH_OFFSET =
    TqAlignUp32(TQ_UB_XY_BATCH_OFFSET + TQ_BATCH_ELEMS * TQ_DTYPE_BYTES);
static constexpr uint32_t TQ_UB_ENCODED_BATCH_BYTES =
    (TQ_MAX_BATCH_M * TQ_KEY_ENCODED_ROW_STRIDE_BYTES >
     TQ_MAX_BATCH_M * TQ_VAL_ENCODED_ROW_STRIDE_BYTES)
        ? TQ_MAX_BATCH_M * TQ_KEY_ENCODED_ROW_STRIDE_BYTES
        : TQ_MAX_BATCH_M * TQ_VAL_ENCODED_ROW_STRIDE_BYTES;
static constexpr uint32_t TQ_UB_A_ENCODED_BYTES =
    (TQ_BATCH_ELEMS * TQ_DTYPE_BYTES > TQ_UB_ENCODED_BATCH_BYTES)
        ? TQ_BATCH_ELEMS * TQ_DTYPE_BYTES
        : TQ_UB_ENCODED_BATCH_BYTES;
static constexpr uint32_t TQ_UB_NORM_SCALAR_OFFSET =
    TqAlignUp32(TQ_UB_A_ENCODED_BATCH_OFFSET + TQ_UB_A_ENCODED_BYTES);
static constexpr uint32_t TQ_UB_NORMS_OFFSET =
    TqAlignUp32(TQ_UB_NORM_SCALAR_OFFSET + TQ_UB_ALIGN);
static constexpr uint32_t TQ_UB_SIGN_MASK_OFFSET =
    TqAlignUp32(TQ_UB_NORMS_OFFSET + TQ_MAX_BATCH_M * TQ_NORM_STRIDE * TQ_DTYPE_BYTES);
static constexpr uint32_t TQ_UB_Y_FP32_OFFSET =
    TqAlignUp32(TQ_UB_SIGN_MASK_OFFSET + TQ_SIGN_MASK_BYTES);
static constexpr uint32_t TQ_UB_SIGN_VAL_OFFSET =
    TqAlignUp32(TQ_UB_Y_FP32_OFFSET + TQ_PACK_D * sizeof(float));
static constexpr uint32_t TQ_UB_ERR_OFFSET =
    TqAlignUp32(TQ_UB_SIGN_VAL_OFFSET + TQ_PACK_D * sizeof(float));
static constexpr uint32_t TQ_UB_QUANT_INDEX_OFFSET =
    TqAlignUp32(TQ_UB_ERR_OFFSET + TQ_PACK_D * sizeof(float));
static constexpr uint32_t TQ_UB_QUANT_INDEX_U16_OFFSET =
    TqAlignUp32(TQ_UB_QUANT_INDEX_OFFSET + TQ_QUANT_INDEX_BYTES);
static constexpr uint32_t TQ_UB_REDUCE_SCALAR_OFFSET =
    TqAlignUp32(TQ_UB_QUANT_INDEX_U16_OFFSET + TQ_QUANT_INDEX_U16_BYTES);
static constexpr uint32_t TQ_UB_REDUCE_OUT_OFFSET =
    TqAlignUp32(TQ_UB_REDUCE_SCALAR_OFFSET + TQ_UB_ALIGN);
static constexpr uint32_t TQ_UB_REDUCE_TMP_OFFSET =
    TqAlignUp32(TQ_UB_REDUCE_OUT_OFFSET + TQ_PACK_D * 3 * sizeof(float));
static constexpr uint32_t TQ_UB_PACKED_ROW_OFFSET =
    TqAlignUp32(TQ_UB_REDUCE_TMP_OFFSET + TQ_PACK_D * sizeof(float));
static constexpr uint32_t TQ_UB_PACK_MERGE_OFFSET =
    TqAlignUp32(TQ_UB_PACKED_ROW_OFFSET + TQ_PACKED_GROUP_BUFFER_COUNT * TQ_PACKED_GROUP_STRIDE);
static constexpr uint32_t TQ_UB_PACK_MASK_OFFSET =
    TqAlignUp32(TQ_UB_PACK_MERGE_OFFSET + TQ_GROUP_INDEX_BYTES);
static constexpr uint32_t TQ_UB_TOTAL_BYTES =
    TqAlignUp32(TQ_UB_PACK_MASK_OFFSET + TQ_GROUP_INDEX_BYTES);
static_assert(TQ_UB_TOTAL_BYTES <= TOTAL_UB_SIZE,
              "Bit-residual K8v4 static UB slices exceed UB size.");

#if defined(ORIG_DTYPE_KEY)
#if (ORIG_DTYPE_KEY == DT_BF16)
using TqDataT = bfloat16_t;
#else
using TqDataT = half;
#endif
#elif defined(DTYPE_KEY)
#if (DTYPE_KEY == DT_BF16)
using TqDataT = bfloat16_t;
#else
using TqDataT = half;
#endif
#else
using TqDataT = half;
#endif

__aicore__ inline uint32_t TqManualRawBlockIdx() {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if ASCEND_IS_AIV {
        return static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
    }
    return static_cast<uint32_t>(AscendC::GetBlockIdx());
#else
    return static_cast<uint32_t>(get_block_idx());
#endif
}

using TqManualMmadResource = Catlass::Arch::Resource<Catlass::Arch::AtlasA2>;


template <AscendC::HardEvent EVT>
__aicore__ inline void TqSyncFixed(uint32_t eventId = EVENT_ID0) {
    SetFlag<EVT>(eventId);
    WaitFlag<EVT>(eventId);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreSet(uint16_t flag) {
    AscendC::CrossCoreSetFlag<0x2, PIPE>(flag);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreWait(uint16_t flag) {
    AscendC::CrossCoreWaitFlag<0x2, PIPE>(flag);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreWaitForBothAiv(uint16_t flag) {
    TqCrossCoreWait<PIPE>(flag);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreSetForBothAiv(uint16_t flag) {
    TqCrossCoreSet<PIPE>(flag);
}

// csrc/kernels build has no PipeSync / SetWaitFlag; use SetFlag+WaitFlag (moe_gating_top_k).
__aicore__ inline void TqSyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void TqSyncMte2ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e);
    WaitFlag<HardEvent::MTE2_S>(e);
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

__aicore__ inline void TqSyncVToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void TqSyncMte3ToMte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

__aicore__ inline void TqSyncMte3ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e);
    WaitFlag<HardEvent::MTE3_V>(e);
}

__aicore__ inline void TqSyncMte3ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(e);
    WaitFlag<HardEvent::MTE3_S>(e);
}


// Group rows are not necessarily 32B-aligned; use DataCopyPad for GM writes.
__aicore__ inline void copy_packed_ub_to_gm(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t>& packedLocal,
    uint32_t nbytes) {
    TqSyncVToMte3();
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
    TqSyncMte3ToMte2();
    TqSyncMte3ToV();
    TqSyncMte3ToS();
}

__aicore__ inline void copy_packed_gm_to_ub(
    AscendC::LocalTensor<uint8_t>& packedLocal,
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    uint32_t nbytes) {
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(packedLocal, packedGm[gm_offset], copyParams, padParams);
    TqSyncMte2ToV();
    TqSyncMte2ToS();
}

template <typename T>
class BitResidualPackK8v4Resource {
public:
    __aicore__ inline void Init() {}

    // ── Row buffers ───────────────────────────────────────────────────────────
    __aicore__ inline AscendC::LocalTensor<T> XBatch() {
        return local_.vecCalc.GetBufferByByte<T>(
            TQ_UB_XY_BATCH_OFFSET,
            TQ_BATCH_ELEMS * TQ_DTYPE_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<T> YBatch() {
        return local_.vecCalc.GetBufferByByte<T>(
            TQ_UB_XY_BATCH_OFFSET,
            TQ_BATCH_ELEMS * TQ_DTYPE_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<T> ABatch() {
        return local_.vecCalc.GetBufferByByte<T>(
            TQ_UB_A_ENCODED_BATCH_OFFSET,
            TQ_BATCH_ELEMS * TQ_DTYPE_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<uint16_t> KeyEncodedBatch() {
        return local_.vecCalc.GetBufferByByte<uint16_t>(
            TQ_UB_A_ENCODED_BATCH_OFFSET,
            TQ_MAX_BATCH_M * TQ_KEY_ENCODED_ROW_STRIDE_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<uint16_t> ValEncodedBatch() {
        return local_.vecCalc.GetBufferByByte<uint16_t>(
            TQ_UB_A_ENCODED_BATCH_OFFSET,
            TQ_MAX_BATCH_M * TQ_VAL_ENCODED_ROW_STRIDE_BYTES);
    }

    // ── VECCALC (UB compute scratch) ─────────────────────────────────────────
    __aicore__ inline AscendC::LocalTensor<float> NormScalar() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_NORM_SCALAR_OFFSET, TQ_UB_ALIGN);
    }
    __aicore__ inline AscendC::LocalTensor<T> Norms() {
        return local_.vecCalc.GetBufferByByte<T>(
            TQ_UB_NORMS_OFFSET,
            TQ_MAX_BATCH_M * TQ_NORM_STRIDE * TQ_DTYPE_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> SignMask() {
        return local_.vecCalc.GetBufferByByte<uint8_t>(
            TQ_UB_SIGN_MASK_OFFSET, TQ_SIGN_MASK_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<float> YFp32() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_Y_FP32_OFFSET, TQ_PACK_D * sizeof(float));
    }
    __aicore__ inline AscendC::LocalTensor<float> SignVal() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_SIGN_VAL_OFFSET, TQ_PACK_D * sizeof(float));
    }
    __aicore__ inline AscendC::LocalTensor<float> Err() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_ERR_OFFSET, TQ_PACK_D * sizeof(float));
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> QuantIndex() {
        return local_.vecCalc.GetBufferByByte<int32_t>(
            TQ_UB_QUANT_INDEX_OFFSET, TQ_QUANT_INDEX_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<int16_t> QuantIndexU16() {
        return local_.vecCalc.GetBufferByByte<int16_t>(
            TQ_UB_QUANT_INDEX_U16_OFFSET, TQ_QUANT_INDEX_U16_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<float> ReduceScalar() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_REDUCE_SCALAR_OFFSET, TQ_UB_ALIGN);
    }
    __aicore__ inline AscendC::LocalTensor<float> ReduceOut() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_REDUCE_OUT_OFFSET, TQ_PACK_D * 3 * sizeof(float));
    }
    __aicore__ inline AscendC::LocalTensor<float> ReduceTmp() {
        return local_.vecCalc.GetBufferByByte<float>(
            TQ_UB_REDUCE_TMP_OFFSET, TQ_PACK_D * sizeof(float));
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> PackedRow() {
        return local_.vecCalc.GetBufferByByte<uint8_t>(
            TQ_UB_PACKED_ROW_OFFSET,
            TQ_PACKED_GROUP_BUFFER_COUNT * TQ_PACKED_GROUP_STRIDE);
    }
    __aicore__ inline AscendC::LocalTensor<uint16_t> PackMerge() {
        return local_.vecCalc.GetBufferByByte<uint16_t>(
            TQ_UB_PACK_MERGE_OFFSET, TQ_GROUP_INDEX_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<uint16_t> PackMask() {
        return local_.vecCalc.GetBufferByByte<uint16_t>(
            TQ_UB_PACK_MASK_OFFSET, TQ_GROUP_INDEX_BYTES);
    }

private:
    TqStaticLocalResource local_;
};


template <typename T>
class BitResidualPackK8v4Context {
public:
    __aicore__ inline explicit BitResidualPackK8v4Context(
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t vecPerCore,
        uint32_t numHeads,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t dataCores,
        uint32_t numReqs,
        uint32_t keyStrideToken,
        uint32_t keyStrideHead,
        uint32_t valueStrideToken,
        uint32_t valueStrideHead,
        uint64_t keyStorageOffset,
        uint64_t valueStorageOffset)
        : nVec_(nVec),
          numHeads_(numHeads == 0 ? 1 : numHeads),
          tokenCount_((nVec + (numHeads == 0 ? 1 : numHeads) - 1) / (numHeads == 0 ? 1 : numHeads)),
          blockSize_(blockSize == 0 ? 1 : blockSize),
          numBlocks_(numBlocks),
          cacheSlots_((blockSize == 0 ? 1 : blockSize) * numBlocks),
          dataCores_(dataCores == 0 ? 1 : dataCores),
          numReqs_(numReqs),
          keyStrideToken_(keyStrideToken),
          keyStrideHead_(keyStrideHead),
          valueStrideToken_(valueStrideToken),
          valueStrideHead_(valueStrideHead),
          keyStorageOffset_(keyStorageOffset),
          valueStorageOffset_(valueStorageOffset),
          keySpan_(MakeInputSpan(keyStorageOffset, keyStrideToken, keyStrideHead)),
          valueSpan_(MakeInputSpan(valueStorageOffset, valueStrideToken, valueStrideHead)),
          workspaceReady_(rawWorkspace != nullptr),
          manualWorkspace_(rawWorkspace == nullptr
                               ? nullptr
                               : reinterpret_cast<__gm__ T*>(
                                     rawWorkspace + TQ_MANUAL_WORKSPACE_BYTE_OFFSET)) {
        (void)vecPerCore;
    }

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ T* rotation_t,
        __gm__ int32_t* slot_mapping,
        __gm__ int32_t* query_start_loc,
        __gm__ uint8_t* key_cache,
        __gm__ uint8_t* value_cache) {
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key), keySpan_);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), valueSpan_);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        slotMappingGm_.SetGlobalBuffer(slot_mapping, tokenCount_);
        queryStartLocGm_.SetGlobalBuffer(query_start_loc, static_cast<uint64_t>(numReqs_) + 1);
        keyCacheGm_.SetGlobalBuffer(
            key_cache,
            (uint64_t)numBlocks_ * numHeads_ *
                (blockSize_ / TQ_KEY_GROUP_ROWS) * TQ_KEY_GROUP_STRIDE);
        valueCacheGm_.SetGlobalBuffer(
            value_cache,
            (uint64_t)numBlocks_ * numHeads_ *
                (blockSize_ / TQ_VAL_GROUP_ROWS) * TQ_VAL_GROUP_STRIDE);

        resource_.Init();
    }

public:

    __aicore__ inline void InitManualWorkspaceTensors(
        uint32_t manualGroupId,
        AscendC::GlobalTensor<float>& normWorkGm,
        AscendC::GlobalTensor<T>& cWorkGm) const {
        __gm__ T* groupWork =
            manualWorkspace_ + static_cast<uint64_t>(manualGroupId) *
                                   TQ_MANUAL_WORKSPACE_ELEMS_PER_CORE;
        normWorkGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(groupWork),
            TQ_MANUAL_WORKSPACE_BUFFER_COUNT * TQ_MANUAL_NORM_WORKSPACE_STRIDE_FLOATS);
        cWorkGm.SetGlobalBuffer(
            groupWork + TQ_MANUAL_C_WORKSPACE_ELEM_OFFSET,
            TQ_MANUAL_WORKSPACE_BUFFER_COUNT * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS);
    }

    __aicore__ inline uint32_t ManualStreamBufferIndex(uint32_t streamOrdinal) const {
        return streamOrdinal & 1u;
    }

    __aicore__ inline uint16_t ManualFlagBase(uint32_t streamOrdinal) const {
        return static_cast<uint16_t>((streamOrdinal & 1u) * TQ_MANUAL_SYNC_PP_STRIDE);
    }

    __aicore__ inline uint32_t ManualBufferOffset(uint32_t streamOrdinal) const {
        return ManualStreamBufferIndex(streamOrdinal) * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS;
    }

    __aicore__ inline uint32_t ManualNormBufferOffset(uint32_t streamOrdinal) const {
        return ManualStreamBufferIndex(streamOrdinal) * TQ_MANUAL_NORM_WORKSPACE_STRIDE_FLOATS;
    }

    __aicore__ inline AscendC::LocalTensor<T> ManualNorms(uint32_t streamOrdinal) {
        return resource_.Norms()[ManualStreamBufferIndex(streamOrdinal) * TQ_MANUAL_NORM_SLOT_ELEMS];
    }

    __aicore__ inline uint64_t MakeInputSpan(
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) const {
        if (tokenCount_ == 0 || numHeads_ == 0) {
            return storageOffset;
        }
        return storageOffset +
               static_cast<uint64_t>(tokenCount_ - 1) * strideToken +
               static_cast<uint64_t>(numHeads_ - 1) * strideHead +
               TQ_PACK_D;
    }

    __aicore__ inline bool ReadRequestTokenRange(
        uint32_t reqIdx,
        uint32_t& tokenStart,
        uint32_t& tokenEnd) const {
        if (reqIdx >= numReqs_) {
            return false;
        }
        const int32_t start = queryStartLocGm_.GetValue(reqIdx);
        const int32_t end = queryStartLocGm_.GetValue(reqIdx + 1);
        if (start < 0 || end < start) {
            return false;
        }
        tokenStart = static_cast<uint32_t>(start);
        tokenEnd = static_cast<uint32_t>(end);
        if (tokenStart > tokenCount_ || tokenEnd > tokenCount_) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool CanUseManualKey1() const {
        if (!workspaceReady_ || manualWorkspace_ == nullptr ||
            dataCores_ == 0 || numHeads_ < TQ_MANUAL_HEADS_PER_TILE ||
            numHeads_ % TQ_MANUAL_HEADS_PER_TILE != 0 ||
            tokenCount_ == 0 || tokenCount_ * numHeads_ != nVec_ ||
            blockSize_ % TQ_VAL_GROUP_ROWS != 0 ||
            blockSize_ % TQ_KEY_GROUP_ROWS != 0 ||
            keyStrideHead_ != TQ_PACK_D ||
            valueStrideHead_ != TQ_PACK_D ||
            keyStrideToken_ == 0 ||
            valueStrideToken_ == 0) {
            return false;
        }

        uint32_t totalTiles = 0;
        for (uint32_t reqIdx = 0; reqIdx < numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!ReadRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                return false;
            }
            const uint32_t rowCount = seqEnd - seqStart;
            if (rowCount == 0) {
                continue;
            }
            uint32_t firstSlot = 0;
            if (!ResolveTokenSlotU(seqStart, firstSlot) ||
                firstSlot + rowCount > cacheSlots_) {
                return false;
            }
            totalTiles += ((rowCount + TQ_MANUAL_GROUP_ROWS - 1) / TQ_MANUAL_GROUP_ROWS) *
                          (numHeads_ / TQ_MANUAL_HEADS_PER_TILE);
        }
        return totalTiles != 0;
    }


    __aicore__ inline bool ResolveTokenSlotU(
        uint32_t tokenIdx,
        uint32_t& slotU) const {
        if (tokenIdx >= tokenCount_) {
            return false;
        }
        const int32_t slot = slotMappingGm_.GetValue(tokenIdx);
        if (slot < 0 || static_cast<uint32_t>(slot) >= cacheSlots_) {
            return false;
        }
        slotU = static_cast<uint32_t>(slot);
        return true;
    }

public:
    BitResidualPackK8v4Resource<T> resource_;
    const uint32_t nVec_;
    const uint32_t numHeads_;
    const uint32_t tokenCount_;
    const uint32_t blockSize_;
    const uint32_t numBlocks_;
    const uint32_t cacheSlots_;
    const uint32_t dataCores_;
    const uint32_t numReqs_;
    const uint32_t keyStrideToken_;
    const uint32_t keyStrideHead_;
    const uint32_t valueStrideToken_;
    const uint32_t valueStrideHead_;
    const uint64_t keyStorageOffset_;
    const uint64_t valueStorageOffset_;
    const uint64_t keySpan_;
    const uint64_t valueSpan_;
    const bool workspaceReady_;
    __gm__ T* const manualWorkspace_;

    AscendC::GlobalTensor<T> keyGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> rotationTGm_;
    AscendC::GlobalTensor<int32_t> slotMappingGm_;
    AscendC::GlobalTensor<int32_t> queryStartLocGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
};

}  // namespace bit_residual
