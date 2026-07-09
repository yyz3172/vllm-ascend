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

// Fused bit-residual pack for KV cache (fp16/bf16).
//
// Reference (Python):
//   norms = vector_norm(x, dim=-1, keepdim=True)
//   y = (x / (norms + eps)) @ R^T        # batched [M, D] @ [D, D]
//   key:   sign + 7-bit residual, 2 rows packed per uint16
//   value: 4-bit uniform quantization, 4 rows packed per uint16
//
// Per AICore: normalize all assigned rows, then ONE matmul [M,128]@[128,128],
// not M separate M=1 matmuls (avoids Cube padding waste).

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"

using namespace AscendC;

namespace {

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
static constexpr uint32_t TQ_MANUAL_WORKSPACE_BUFFER_COUNT = 2;
static constexpr uint32_t TQ_MANUAL_WORKSPACE_STRIDE_ELEMS = TQ_BATCH_ELEMS;
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

template <typename T>
__aicore__ inline constexpr QuantMode_t TqManualFixpipeQuantMode() {
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    }
    return QuantMode_t::F322F16;
}

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

template <typename T>
__aicore__ inline void TqManualLoadResidentRotation(
    TqManualMmadResource& resource,
    AscendC::GlobalTensor<T>& rotationTGm) {
    auto bL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_B_L1_OFFSET);
    AscendC::Nd2NzParams bNd2Nz;
    bNd2Nz.ndNum = TQ_ROT_K / TQ_CUBE_M_ALIGN;
    bNd2Nz.nValue = TQ_CUBE_M_ALIGN;
    bNd2Nz.dValue = TQ_ROT_N;
    bNd2Nz.srcDValue = TQ_ROT_N;
    bNd2Nz.srcNdMatrixStride = TQ_CUBE_M_ALIGN * TQ_ROT_N;
    bNd2Nz.dstNzC0Stride = TQ_CUBE_M_ALIGN;
    bNd2Nz.dstNzNStride = 1;
    bNd2Nz.dstNzMatrixStride = TQ_CUBE_M_ALIGN * TQ_ROT_N;
    AscendC::DataCopy(bL1, rotationTGm, bNd2Nz);
    TqSyncFixed<AscendC::HardEvent::MTE2_MTE1>();
}

template <typename T>
__aicore__ inline void TqManualLoadATileToL1(
    TqManualMmadResource& resource,
    AscendC::GlobalTensor<T>& aWorkGm,
    uint32_t aOffset) {
    auto aL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
    for (uint32_t rowBase = 0; rowBase < TQ_MANUAL_ROT_TILE_M; rowBase += TQ_CUBE_M_ALIGN) {
        AscendC::Nd2NzParams aNd2Nz;
        aNd2Nz.ndNum = 1;
        aNd2Nz.nValue = TQ_CUBE_M_ALIGN;
        aNd2Nz.dValue = TQ_ROT_K;
        aNd2Nz.srcDValue = TQ_ROT_K;
        aNd2Nz.dstNzC0Stride = TQ_CUBE_M_ALIGN;
        aNd2Nz.dstNzNStride = 1;
        aNd2Nz.srcNdMatrixStride = 0;
        aNd2Nz.dstNzMatrixStride = 0;
        AscendC::DataCopy(
            aL1[(rowBase / TQ_CUBE_M_ALIGN) * TQ_MANUAL_AIV_SLICE_ELEMS],
            aWorkGm[aOffset + rowBase * TQ_ROT_K],
            aNd2Nz);
    }
    TqSyncFixed<AscendC::HardEvent::MTE2_MTE1>();
}

template <typename T>
__aicore__ inline void TqManualComputeLoadedTile(
    TqManualMmadResource& resource,
    AscendC::GlobalTensor<T>& cWorkGm,
    uint32_t cOffset) {
    auto aL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
    auto bL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_B_L1_OFFSET);
    auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(0);
    auto bL0 = resource.l0BBuf.template GetBufferByByte<T>(0);
    auto cL0Base = resource.l0CBuf.template GetBufferByByte<float>(0);

    AscendC::LoadData2DParams bLoad;
    bLoad.startIndex = 0;
    bLoad.repeatTimes = TQ_ROT_N / TQ_CUBE_M_ALIGN;
    bLoad.srcStride = 1;
    bLoad.sid = 0;
    bLoad.dstGap = 0;
    bLoad.ifTranspose = true;
    bLoad.addrMode = 0;
    for (uint32_t i = 0; i < TQ_ROT_K / TQ_CUBE_M_ALIGN; ++i) {
        AscendC::LoadData(
            bL0[i * TQ_ROT_N * TQ_CUBE_M_ALIGN],
            bL1[i * TQ_ROT_N * TQ_CUBE_M_ALIGN],
            bLoad);
    }
    for (uint32_t rowBase = 0; rowBase < TQ_MANUAL_ROT_TILE_M; rowBase += TQ_CUBE_M_ALIGN) {
        auto aL0Tile = aL0[(rowBase / TQ_CUBE_M_ALIGN) * TQ_CUBE_M_ALIGN * TQ_ROT_K];
        auto cL0 = cL0Base[(rowBase / TQ_CUBE_M_ALIGN) * TQ_CUBE_M_ALIGN * TQ_ROT_N];
        TqSyncFixed<AscendC::HardEvent::M_MTE1>();
        AscendC::LoadData2DParams aLoad;
        aLoad.startIndex = 0;
        aLoad.repeatTimes = TQ_ROT_K / TQ_CUBE_M_ALIGN;
        aLoad.srcStride = 1;
        aLoad.sid = 0;
        aLoad.dstGap = 0;
        aLoad.ifTranspose = false;
        aLoad.addrMode = 0;
        AscendC::LoadData(
            aL0Tile,
            aL1[(rowBase / TQ_CUBE_M_ALIGN) * TQ_MANUAL_AIV_SLICE_ELEMS],
            aLoad);
        TqSyncFixed<AscendC::HardEvent::MTE1_M>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = TQ_CUBE_M_ALIGN;
        mmadParams.n = TQ_ROT_N;
        mmadParams.k = TQ_ROT_K;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0b11;
        TqSyncFixed<AscendC::HardEvent::FIX_M>();
        AscendC::Mmad(cL0, aL0Tile, bL0, mmadParams);
        AscendC::PipeBarrier<PIPE_M>();
        TqSyncFixed<AscendC::HardEvent::M_FIX>();

        AscendC::FixpipeParamsV220 fixParams;
        fixParams.nSize = TQ_ROT_N;
        fixParams.mSize = TQ_CUBE_M_ALIGN;
        fixParams.srcStride = TQ_CUBE_M_ALIGN;
        fixParams.dstStride = TQ_ROT_N;
        fixParams.ndNum = 1;
        fixParams.unitFlag = 0b11;
        fixParams.quantPre = TqManualFixpipeQuantMode<T>();
        fixParams.reluEn = false;
        AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(
            cWorkGm[cOffset + rowBase * TQ_ROT_N],
            cL0,
            fixParams);
        TqSyncFixed<AscendC::HardEvent::FIX_M>();
    }
}

template <typename T>
__aicore__ inline void TqCopyManualAUbToGm(
    AscendC::GlobalTensor<T>& aWorkGm,
    uint32_t elemOffset,
    AscendC::LocalTensor<T> inputLocal) {
    TqSyncVToMte3();
    AscendC::DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(TQ_MANUAL_AIV_SLICE_ELEMS * sizeof(T)),
        0,
        0,
        0};
    AscendC::DataCopyPad(aWorkGm[elemOffset], inputLocal, copyParams);
    TqSyncMte3ToMte2();
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
class BitResidualPackK8v4 {
public:
    __aicore__ inline explicit BitResidualPackK8v4(
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

    __aicore__ inline void Process() {
        if (!CanUseManualKey1()) {
            return;
        }
        if ASCEND_IS_AIV {
            const uint32_t manualGroupId = TqManualRawBlockIdx();
            if (manualGroupId >= dataCores_) {
                return;
            }
            const uint32_t aivSlice = AscendC::GetSubBlockIdx() % TQ_AIV_SUB_BLOCKS;
            ProcessManualKey1Aiv(manualGroupId, aivSlice);
            return;
        }
        const uint32_t manualGroupId = AscendC::GetBlockIdx();
        if (manualGroupId >= dataCores_) {
            return;
        }
        ProcessManualKey1Aic(manualGroupId);
    }

private:

    // norms[i] = ||x[i]||; xBatch rows unitized in-place (matches x / (norm + eps)).
    // Inner dim: Cast + Mul + ReduceSum (vector), not scalar loop; fp32 acc avoids 16-bit overflow.
    __aicore__ inline void NormalizeBatchBrcbScaleToNorms(
        uint32_t m,
        AscendC::LocalTensor<T> norms) {
        auto xBatch = resource_.XBatch();
        auto aBatch = resource_.ABatch();
        auto fp32Row = resource_.ReduceOut();
        auto scaleBlock = resource_.ReduceOut()[TQ_PACK_D];
        auto fp32Tmp = resource_.ReduceOut()[TQ_PACK_D * 2];
        auto normAcc = resource_.NormScalar();

        TqSyncMte2ToV();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;

            AscendC::Cast(fp32Row, xBatch[rowOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(scaleBlock, fp32Row, fp32Row, TQ_PACK_D);
            AscendC::ReduceSum<float>(normAcc, scaleBlock, fp32Tmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(normAcc, normAcc, 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(norms[i * TQ_NORM_STRIDE], normAcc, AscendC::RoundMode::CAST_RINT, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(normAcc, normAcc, TQ_NORM_EPS_F, 1);
            AscendC::Duplicate(fp32Tmp, 1.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(normAcc, fp32Tmp, normAcc, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(scaleBlock, normAcc, 1, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(
                fp32Row,
                fp32Row,
                scaleBlock,
                static_cast<uint64_t>(TQ_PACK_D / 2),
                2,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(aBatch[rowOff], fp32Row, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void EncodeKeyBatchWithNorms(
        uint32_t m,
        AscendC::LocalTensor<T> norms) {
        auto yBatch = resource_.YBatch();
        auto encodedBatch = resource_.KeyEncodedBatch();
        auto yFp32 = resource_.YFp32();
        auto signVal = resource_.SignVal();
        auto err = resource_.Err();
        auto qI16 = resource_.QuantIndexU16();
        auto codeU16 = resource_.PackMerge();
        auto codeMask = resource_.PackMask();
        auto reduceScalar = resource_.ReduceScalar();
        auto baseAcc = reduceScalar;
        auto maxAcc = reduceScalar[1];
        auto reduceTmp = resource_.ReduceTmp();
        auto normWords = norms.template ReinterpretCast<uint16_t>();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_KEY_ENCODED_ROW_STRIDE_WORDS;
            const uint32_t normOff = i * TQ_NORM_STRIDE;

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            auto signBits = qI16.template ReinterpretCast<uint16_t>();
            AscendC::Duplicate(codeMask, static_cast<uint16_t>(1), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(
                qI16,
                yBatch[yOff].template ReinterpretCast<int16_t>(),
                static_cast<int16_t>(15),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(signBits, signBits, codeMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signVal, qI16, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(signVal, signVal, -1.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(signVal, signVal, 1.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, signVal, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(codeMask, static_cast<uint16_t>(0), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(
                codeU16,
                qI16.template ReinterpretCast<uint16_t>(),
                codeMask,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(signVal, signVal, 2.0f * TQ_INV_SQRT_D_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(signVal, signVal, -TQ_INV_SQRT_D_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(err, yFp32, signVal, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceMin<float>(baseAcc, err, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(maxAcc, err, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            const float baseF = baseAcc.GetValue(0);
            const float maxF = maxAcc.GetValue(0);
            const float rangeF = maxF - baseF;
            const float stepF = rangeF > 0.0f ? rangeF / TQ_KEY_QUANT_LEVELS_F : 1.0f;
            const float invStepF = rangeF > 0.0f ? 1.0f / stepF : 0.0f;
            TqSyncSToV();

            AscendC::Adds(err, err, -baseF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(err, err, invStepF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(err, err, 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(err, err, TQ_KEY_QUANT_LEVELS_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, err, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(
                codeMask,
                qI16.template ReinterpretCast<uint16_t>(),
                static_cast<uint16_t>(1),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(
                codeU16,
                codeU16,
                codeMask,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(encodedBatch[encodedOff], codeU16, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            encodedBatch.SetValue(
                encodedOff + TQ_KEY_ENCODED_NORM_WORD_OFFSET,
                normWords.GetValue(normOff));
            auto baseOut = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_BASE_BYTE_OFFSET / sizeof(uint16_t)]
                               .template ReinterpretCast<float>();
            auto stepOut = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_STEP_BYTE_OFFSET / sizeof(uint16_t)]
                               .template ReinterpretCast<float>();
            baseOut.SetValue(0, baseF);
            stepOut.SetValue(0, stepF);
        }
        TqSyncSToV();
    }

    __aicore__ inline void EncodeValueBatch(uint32_t m,
                                               AscendC::LocalTensor<T> norms) {
        auto yBatch = resource_.YBatch();
        auto encodedBatch = resource_.ValEncodedBatch();
        auto yFp32 = resource_.YFp32();
        auto qFp32 = resource_.Err();
        auto qI32 = resource_.QuantIndex();
        auto qI16 = resource_.QuantIndexU16();
        auto reduceScalar = resource_.ReduceScalar();
        auto vminAcc = reduceScalar;
        auto vmaxAcc = reduceScalar[1];
        auto reduceTmp = resource_.ReduceTmp();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_VAL_ENCODED_ROW_STRIDE_WORDS;
            const uint32_t normOff = i * TQ_NORM_STRIDE;

            AscendC::Cast(yFp32, yBatch[yOff], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMin<float>(vminAcc, yFp32, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(vmaxAcc, yFp32, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            const float vminF = vminAcc.GetValue(0);
            const float vmaxF = vmaxAcc.GetValue(0);
            const float rangeF = vmaxF - vminF;
            const float vstepF = rangeF > 0.0f ? rangeF / TQ_VAL_QUANT_LEVELS_F : 1.0f;
            const float invStepF = rangeF > 0.0f ? 1.0f / vstepF : 0.0f;
            TqSyncSToV();

            // Fold norm into vmin/vstep so that decode reconstructs
            // values with original magnitude without a separate norm:
            //   y = vmin_scaled + idx4 * vstep_scaled
            //     = norm * (vmin + idx4 * vstep)
            //     ≈ norm * x_unit @ R^T = x @ R^T
            auto normToFloat = resource_.ReduceOut();  // float32 scratch
            AscendC::Cast(normToFloat, norms[normOff], AscendC::RoundMode::CAST_NONE, 1);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            const float normF = normToFloat.GetValue(0);
            TqSyncSToV();

            const float vminScaledF = vminF * normF;
            const float vstepScaledF = vstepF * normF;

            AscendC::Adds(qFp32, yFp32, -vminF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(qFp32, qFp32, invStepF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(qFp32, qFp32, 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(qFp32, qFp32, TQ_VAL_QUANT_LEVELS_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI32, qFp32, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, qI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(
                encodedBatch[encodedOff].template ReinterpretCast<int16_t>(),
                qI16,
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            auto vminOut = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VMIN_BYTE_OFFSET / sizeof(uint16_t)]
                               .template ReinterpretCast<float>();
            auto vstepOut = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET / sizeof(uint16_t)]
                                .template ReinterpretCast<float>();
            vminOut.SetValue(0, vminScaledF);
            vstepOut.SetValue(0, vstepScaledF);
        }
        TqSyncSToV();
    }

    __aicore__ inline void CopyManualSliceToInput(
        const AscendC::GlobalTensor<T>& xGm,
        uint32_t tokenStart,
        uint32_t headTileStart,
        uint32_t aivSlice,
        uint32_t startGroupRow,
        uint32_t validRows,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) {
        auto xBatch = resource_.XBatch();
        const uint32_t firstHead = headTileStart + aivSlice * TQ_MANUAL_HEADS_PER_AIV;
        bool clearedInvalidRows = false;
        for (uint32_t localHead = 0; localHead < TQ_MANUAL_HEADS_PER_AIV; ++localHead) {
            const uint32_t headIdx = firstHead + localHead;
            for (uint32_t groupRow = 0; groupRow < TQ_MANUAL_GROUP_ROWS; ++groupRow) {
                const uint32_t dstRow = localHead * TQ_MANUAL_GROUP_ROWS + groupRow;
                if (groupRow < startGroupRow || groupRow >= startGroupRow + validRows) {
                    AscendC::Duplicate(
                        xBatch[dstRow * TQ_PACK_D].template ReinterpretCast<uint16_t>(),
                        static_cast<uint16_t>(0),
                        TQ_PACK_D);
                    clearedInvalidRows = true;
                    continue;
                }
                const uint32_t tokenRow = groupRow - startGroupRow;
                const uint64_t srcOffset =
                    storageOffset +
                    static_cast<uint64_t>(tokenStart + tokenRow) * strideToken +
                    static_cast<uint64_t>(headIdx) * strideHead;
                AscendC::DataCopy(xBatch[dstRow * TQ_PACK_D], xGm[srcOffset], TQ_PACK_D);
            }
        }
        if (clearedInvalidRows) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyManualCToYBatch(
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t bufferOffset,
        uint32_t aivSlice) {
        auto yBatch = resource_.YBatch();
        const uint32_t sliceElemOffset = aivSlice * TQ_MANUAL_AIV_SLICE_ELEMS;
        for (uint32_t row = 0; row < TQ_MANUAL_AIV_SLICE_M; ++row) {
            const uint32_t rowOffset = row * TQ_PACK_D;
            AscendC::DataCopy(
                yBatch[rowOffset],
                cWorkGm[bufferOffset + sliceElemOffset + rowOffset],
                TQ_PACK_D);
        }
        TqSyncMte2ToV();
    }

    __aicore__ inline void InitManualWorkspaceTensors(
        uint32_t manualGroupId,
        AscendC::GlobalTensor<T>& aWorkGm,
        AscendC::GlobalTensor<T>& cWorkGm) const {
        __gm__ T* groupWork =
            manualWorkspace_ + static_cast<uint64_t>(manualGroupId) *
                                   TQ_MANUAL_WORKSPACE_ELEMS_PER_CORE;
        aWorkGm.SetGlobalBuffer(
            groupWork,
            TQ_MANUAL_WORKSPACE_BUFFER_COUNT * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS);
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

    __aicore__ inline AscendC::LocalTensor<T> ManualNorms(uint32_t streamOrdinal) {
        return resource_.Norms()[ManualStreamBufferIndex(streamOrdinal) * TQ_MANUAL_NORM_SLOT_ELEMS];
    }

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

    template <bool IS_KEY>
    __aicore__ inline void EncodeManualSliceToCache(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice,
        AscendC::LocalTensor<T> norms) {
        const uint32_t firstHead = desc.headTileStart + aivSlice * TQ_MANUAL_HEADS_PER_AIV;
        if constexpr (IS_KEY) {
            EncodeKeyBatchWithNorms(TQ_MANUAL_AIV_SLICE_M, norms);
        } else {
            EncodeValueBatch(TQ_MANUAL_AIV_SLICE_M, norms);
        }
        auto encodedBatch = IS_KEY ? resource_.KeyEncodedBatch() : resource_.ValEncodedBatch();
        auto packedGroups = resource_.PackedRow();
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();
        for (uint32_t localHead = 0; localHead < TQ_MANUAL_HEADS_PER_AIV; ++localHead) {
            const uint32_t headIdx = firstHead + localHead;
            uint32_t rowOff = 0;
            while (rowOff < desc.validRows) {
                const uint32_t slot = desc.slotStart + rowOff;
                const uint32_t blockIdx = slot / blockSize_;
                const uint32_t blockOffset = slot - blockIdx * blockSize_;
                const uint32_t groupInBlock = blockOffset / groupRowsPerGroup;
                const uint32_t firstGroupRow = blockOffset % groupRowsPerGroup;
                uint32_t rowsThisGroup = groupRowsPerGroup - firstGroupRow;
                if (rowsThisGroup > desc.validRows - rowOff) {
                    rowsThisGroup = desc.validRows - rowOff;
                }
                const uint64_t groupBase =
                    MakeCacheGroupBaseOffset<IS_KEY>(blockIdx, groupInBlock, headIdx);
                const bool preserveExisting =
                    firstGroupRow != 0 || rowsThisGroup < groupRowsPerGroup;

                auto packedGroup = packedGroups[localHead * TQ_PACKED_GROUP_STRIDE];
                if (preserveExisting) {
                    copy_packed_gm_to_ub(packedGroup, packedGm, groupBase, GroupBytes<IS_KEY>());
                } else {
                    ClearPackedGroup(packedGroup);
                }
                for (uint32_t r = 0; r < rowsThisGroup; ++r) {
                    const uint32_t manualGroupRow = desc.startGroupRow + rowOff + r;
                    const uint32_t encodedRow = localHead * TQ_MANUAL_GROUP_ROWS + manualGroupRow;
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup,
                        encodedBatch,
                        encodedRow,
                        firstGroupRow + r,
                        preserveExisting);
                }
                copy_packed_ub_to_gm(packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
                rowOff += rowsThisGroup;
            }
        }
    }

    __aicore__ inline void SubmitManualKey1AivA(
        AscendC::GlobalTensor<T>& aWorkGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        const uint16_t flagBase = ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset = ManualBufferOffset(desc.streamOrdinal);
        const uint32_t sliceElemOffset = aivSlice * TQ_MANUAL_AIV_SLICE_ELEMS;
        auto norms = ManualNorms(desc.streamOrdinal);

        if (desc.isValue) {
            CopyManualSliceToInput(
                valueGm_, desc.tokenStart, desc.headTileStart, aivSlice,
                desc.startGroupRow, desc.validRows,
                valueStorageOffset_, valueStrideToken_, valueStrideHead_);
        } else {
            CopyManualSliceToInput(
                keyGm_, desc.tokenStart, desc.headTileStart, aivSlice,
                desc.startGroupRow, desc.validRows,
                keyStorageOffset_, keyStrideToken_, keyStrideHead_);
        }
        NormalizeBatchBrcbScaleToNorms(TQ_MANUAL_AIV_SLICE_M, norms);

        TqCrossCoreWait<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_A_FREE);
        auto aBatch = resource_.ABatch();
        TqCopyManualAUbToGm(
            aWorkGm,
            bufferOffset + sliceElemOffset,
            aBatch);
        TqCrossCoreSet<PIPE_MTE3>(flagBase + TQ_MANUAL_SYNC_A_READY);
        TqCrossCoreWait<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_A_FREE);
    }

    __aicore__ inline void ReleaseManualKey1AivCWrite(uint32_t streamOrdinal) const {
        TqCrossCoreSet<PIPE_MTE2>(ManualFlagBase(streamOrdinal) + TQ_MANUAL_SYNC_C_FREE);
    }

    __aicore__ inline void FinishManualKey1AivC(
        AscendC::GlobalTensor<T>& cWorkGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        const uint16_t flagBase = ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset = ManualBufferOffset(desc.streamOrdinal);
        auto norms = ManualNorms(desc.streamOrdinal);

        TqCrossCoreWait<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_READY);
        CopyManualCToYBatch(cWorkGm, bufferOffset, aivSlice);
        TqCrossCoreSet<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_FREE);
        if (desc.isValue) {
            EncodeManualSliceToCache<false>(valueCacheGm_, desc, aivSlice, norms);
        } else {
            EncodeManualSliceToCache<true>(keyCacheGm_, desc, aivSlice, norms);
        }
    }

    __aicore__ inline void ProcessManualKey1AivPipelineStream(
        AscendC::GlobalTensor<T>& aWorkGm,
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t aivSlice,
        const ManualKey1StreamDesc& desc,
        bool& hasPending,
        ManualKey1StreamDesc& pending) {
        if (hasPending) {
            ReleaseManualKey1AivCWrite(pending.streamOrdinal);
            SubmitManualKey1AivA(aWorkGm, desc, aivSlice);
            FinishManualKey1AivC(cWorkGm, pending, aivSlice);
        } else {
            SubmitManualKey1AivA(aWorkGm, desc, aivSlice);
            hasPending = true;
        }
        pending = desc;
    }

    __aicore__ inline void DrainManualKey1AivPipeline(
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t aivSlice,
        bool& hasPending,
        ManualKey1StreamDesc& pending) {
        if (!hasPending) {
            return;
        }
        ReleaseManualKey1AivCWrite(pending.streamOrdinal);
        FinishManualKey1AivC(cWorkGm, pending, aivSlice);
        hasPending = false;
    }

    __aicore__ inline void StartManualKey1AicA(uint32_t streamOrdinal) const {
        TqCrossCoreSetForBothAiv<PIPE_FIX>(
            ManualFlagBase(streamOrdinal) + TQ_MANUAL_SYNC_A_FREE);
    }

    __aicore__ inline void WaitAndLoadManualKey1AicA(
        TqManualMmadResource& manualResource,
        AscendC::GlobalTensor<T>& aWorkGm,
        uint32_t streamOrdinal) const {
        const uint16_t flagBase = ManualFlagBase(streamOrdinal);
        const uint32_t bufferOffset = ManualBufferOffset(streamOrdinal);
        TqCrossCoreWaitForBothAiv<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_A_READY);
        TqManualLoadATileToL1(manualResource, aWorkGm, bufferOffset);
        TqCrossCoreSetForBothAiv<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_A_FREE);
    }

    __aicore__ inline void LoadManualKey1AicA(
        TqManualMmadResource& manualResource,
        AscendC::GlobalTensor<T>& aWorkGm,
        uint32_t streamOrdinal) const {
        StartManualKey1AicA(streamOrdinal);
        WaitAndLoadManualKey1AicA(manualResource, aWorkGm, streamOrdinal);
    }

    __aicore__ inline void ComputeManualKey1AicC(
        TqManualMmadResource& manualResource,
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t streamOrdinal) const {
        const uint16_t flagBase = ManualFlagBase(streamOrdinal);
        const uint32_t bufferOffset = ManualBufferOffset(streamOrdinal);
        TqCrossCoreWaitForBothAiv<PIPE_FIX>(flagBase + TQ_MANUAL_SYNC_C_FREE);
        TqManualComputeLoadedTile(manualResource, cWorkGm, bufferOffset);
        TqSyncFixed<AscendC::HardEvent::FIX_MTE3>();
        TqCrossCoreSetForBothAiv<PIPE_FIX>(flagBase + TQ_MANUAL_SYNC_C_READY);
    }

    __aicore__ inline void FinishManualKey1AicC(uint32_t streamOrdinal) const {
        TqCrossCoreWaitForBothAiv<PIPE_FIX>(
            ManualFlagBase(streamOrdinal) + TQ_MANUAL_SYNC_C_FREE);
    }

    __aicore__ inline void ProcessManualKey1AicPipelineStream(
        TqManualMmadResource& manualResource,
        AscendC::GlobalTensor<T>& aWorkGm,
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t streamOrdinal,
        bool& hasPending,
        uint32_t& pendingStream) const {
        if (!hasPending) {
            LoadManualKey1AicA(manualResource, aWorkGm, streamOrdinal);
            pendingStream = streamOrdinal;
            hasPending = true;
            return;
        }

        StartManualKey1AicA(streamOrdinal);
        ComputeManualKey1AicC(manualResource, cWorkGm, pendingStream);
        WaitAndLoadManualKey1AicA(manualResource, aWorkGm, streamOrdinal);
        FinishManualKey1AicC(pendingStream);
        pendingStream = streamOrdinal;
    }

    __aicore__ inline void DrainManualKey1AicPipeline(
        TqManualMmadResource& manualResource,
        AscendC::GlobalTensor<T>& cWorkGm,
        bool& hasPending,
        uint32_t& pendingStream) const {
        if (!hasPending) {
            return;
        }
        ComputeManualKey1AicC(manualResource, cWorkGm, pendingStream);
        FinishManualKey1AicC(pendingStream);
        hasPending = false;
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

    __aicore__ inline void ProcessManualKey1Aiv(uint32_t manualGroupId, uint32_t aivSlice) {
        if (aivSlice >= TQ_AIV_SUB_BLOCKS) {
            return;
        }

        AscendC::GlobalTensor<T> aWorkGm;
        AscendC::GlobalTensor<T> cWorkGm;
        InitManualWorkspaceTensors(manualGroupId, aWorkGm, cWorkGm);

        uint32_t tileOrdinal = 0;
        uint32_t manualTileOrdinal = 0;
        bool hasPending = false;
        ManualKey1StreamDesc pending {};
        for (uint32_t reqIdx = 0; reqIdx < numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!ReadRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                return;
            }
            if (seqStart == seqEnd) {
                continue;
            }
            uint32_t firstSlot = 0;
            if (!ResolveTokenSlotU(seqStart, firstSlot)) {
                return;
            }
            const uint32_t rowCount = seqEnd - seqStart;
            const uint32_t leadingGroupRow = firstSlot % TQ_MANUAL_GROUP_ROWS;
            for (uint32_t rowOff = 0; rowOff < rowCount; ) {
                const uint32_t startGroupRow = (rowOff == 0) ? leadingGroupRow : 0;
                const uint32_t rowsInThisGroup = TQ_MANUAL_GROUP_ROWS - startGroupRow;
                const uint32_t validRows = (rowCount - rowOff > rowsInThisGroup)
                    ? rowsInThisGroup
                    : rowCount - rowOff;
                const bool preserveValue =
                    startGroupRow != 0 || validRows < TQ_MANUAL_GROUP_ROWS;
                const uint32_t tokenStart = seqStart + rowOff;
                const uint32_t slot = firstSlot + rowOff;
                const uint32_t blockIdx = slot / blockSize_;
                const uint32_t blockOffset = slot - blockIdx * blockSize_;
                const uint32_t valueGroupInBlock = blockOffset / TQ_VAL_GROUP_ROWS;
                for (uint32_t headTileStart = 0;
                     headTileStart < numHeads_;
                     headTileStart += TQ_MANUAL_HEADS_PER_TILE) {
                    if (tileOrdinal % dataCores_ == manualGroupId) {
                        const uint32_t keyStream = manualTileOrdinal * TQ_MANUAL_STREAM_KIND_COUNT;
                        const uint32_t valueStream = keyStream + 1;
                        ManualKey1StreamDesc keyDesc {
                            tokenStart,
                            slot,
                            blockIdx,
                            valueGroupInBlock,
                            headTileStart,
                            startGroupRow,
                            validRows,
                            keyStream,
                            preserveValue,
                            false};
                        ManualKey1StreamDesc valueDesc {
                            tokenStart,
                            slot,
                            blockIdx,
                            valueGroupInBlock,
                            headTileStart,
                            startGroupRow,
                            validRows,
                            valueStream,
                            preserveValue,
                            true};
                        ProcessManualKey1AivPipelineStream(
                            aWorkGm, cWorkGm, aivSlice, keyDesc, hasPending, pending);
                        ProcessManualKey1AivPipelineStream(
                            aWorkGm, cWorkGm, aivSlice, valueDesc, hasPending, pending);
                        ++manualTileOrdinal;
                    }
                    ++tileOrdinal;
                }
                rowOff += validRows;
            }
        }
        DrainManualKey1AivPipeline(cWorkGm, aivSlice, hasPending, pending);
    }

    __aicore__ inline void ProcessManualKey1Aic(uint32_t manualGroupId) {
        AscendC::GlobalTensor<T> aWorkGm;
        AscendC::GlobalTensor<T> cWorkGm;
        InitManualWorkspaceTensors(manualGroupId, aWorkGm, cWorkGm);

        TqManualMmadResource manualResource;
        TqManualLoadResidentRotation(manualResource, rotationTGm_);

        uint32_t tileOrdinal = 0;
        uint32_t manualTileOrdinal = 0;
        bool hasPending = false;
        uint32_t pendingStream = 0;
        for (uint32_t reqIdx = 0; reqIdx < numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!ReadRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                return;
            }
            if (seqStart == seqEnd) {
                continue;
            }
            uint32_t firstSlot = 0;
            if (!ResolveTokenSlotU(seqStart, firstSlot)) {
                return;
            }
            const uint32_t rowCount = seqEnd - seqStart;
            const uint32_t leadingGroupRow = firstSlot % TQ_MANUAL_GROUP_ROWS;
            for (uint32_t rowOff = 0; rowOff < rowCount; ) {
                const uint32_t startGroupRow = (rowOff == 0) ? leadingGroupRow : 0;
                const uint32_t rowsInThisGroup = TQ_MANUAL_GROUP_ROWS - startGroupRow;
                const uint32_t validRows = (rowCount - rowOff > rowsInThisGroup)
                    ? rowsInThisGroup
                    : rowCount - rowOff;
                for (uint32_t headTileStart = 0;
                     headTileStart < numHeads_;
                     headTileStart += TQ_MANUAL_HEADS_PER_TILE) {
                    if (tileOrdinal % dataCores_ == manualGroupId) {
                        const uint32_t keyStream = manualTileOrdinal * TQ_MANUAL_STREAM_KIND_COUNT;
                        const uint32_t valueStream = keyStream + 1;
                        ProcessManualKey1AicPipelineStream(
                            manualResource, aWorkGm, cWorkGm,
                            keyStream, hasPending, pendingStream);
                        ProcessManualKey1AicPipelineStream(
                            manualResource, aWorkGm, cWorkGm,
                            valueStream, hasPending, pendingStream);
                        ++manualTileOrdinal;
                    }
                    ++tileOrdinal;
                }
                rowOff += validRows;
            }
        }
        DrainManualKey1AicPipeline(manualResource, cWorkGm, hasPending, pendingStream);
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t GroupRows() const {
        return IS_KEY ? TQ_KEY_GROUP_ROWS : TQ_VAL_GROUP_ROWS;
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t GroupBytes() const {
        return IS_KEY ? TQ_KEY_GROUP_BYTES : TQ_VAL_GROUP_BYTES;
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t GroupStride() const {
        return IS_KEY ? TQ_KEY_GROUP_STRIDE : TQ_VAL_GROUP_STRIDE;
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t EncodedRowStrideWords() const {
        return IS_KEY ? TQ_KEY_ENCODED_ROW_STRIDE_WORDS : TQ_VAL_ENCODED_ROW_STRIDE_WORDS;
    }

    __aicore__ inline void ClearPackedGroup(AscendC::LocalTensor<uint8_t>& packedGroup) const {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        AscendC::Duplicate(
            packedU16,
            static_cast<uint16_t>(0),
            TQ_PACKED_GROUP_STRIDE / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool IS_KEY>
    __aicore__ inline void MergeEncodedRowToGroup(
        AscendC::LocalTensor<uint8_t>& packedGroup,
        const AscendC::LocalTensor<uint16_t>& encodedBatch,
        uint32_t encodedRow,
        uint32_t groupRow,
        bool clearOldBits) {
        auto packedU16 = packedGroup.template ReinterpretCast<uint16_t>();
        auto shiftedIdx = resource_.PackMerge();
        const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
        const uint32_t shiftBits = groupRow * (IS_KEY ? 8 : 4);

        if (clearOldBits) {
            auto clearMask = resource_.PackMask();
            const uint16_t mask = static_cast<uint16_t>(
                IS_KEY
                    ? ~static_cast<uint16_t>(0x00FFu << shiftBits)
                    : ~static_cast<uint16_t>(0x000Fu << shiftBits));
            AscendC::Duplicate(clearMask, mask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(packedU16, packedU16, clearMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::ShiftLeft(
            shiftedIdx,
            encodedBatch[encodedOff],
            static_cast<uint16_t>(shiftBits),
            TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(packedU16, packedU16, shiftedIdx, TQ_PACK_D);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (IS_KEY) {
            const uint32_t normWord = TQ_KEY_GROUP_NORM_OFFSET / sizeof(uint16_t) + groupRow;
            packedU16.SetValue(
                normWord,
                encodedBatch.GetValue(encodedOff + TQ_KEY_ENCODED_NORM_WORD_OFFSET));
            auto packedBase = packedGroup[TQ_KEY_GROUP_BASE_OFFSET].template ReinterpretCast<float>();
            auto packedStep = packedGroup[TQ_KEY_GROUP_STEP_OFFSET].template ReinterpretCast<float>();
            auto encodedBase = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_BASE_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            auto encodedStep = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_STEP_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            packedBase.SetValue(groupRow, encodedBase.GetValue(0));
            packedStep.SetValue(groupRow, encodedStep.GetValue(0));
        } else {
            auto packedVmin = packedGroup[TQ_VAL_GROUP_VMIN_OFFSET].template ReinterpretCast<float>();
            auto packedVstep = packedGroup[TQ_VAL_GROUP_VSTEP_OFFSET].template ReinterpretCast<float>();
            auto encodedVmin = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VMIN_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            auto encodedVstep = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET / sizeof(uint16_t)]
                                    .template ReinterpretCast<float>();
            packedVmin.SetValue(groupRow, encodedVmin.GetValue(0));
            packedVstep.SetValue(groupRow, encodedVstep.GetValue(0));
        }
    }

    template <bool IS_KEY>
    __aicore__ inline uint64_t MakeCacheGroupBaseOffset(
        uint32_t blockIdx,
        uint32_t groupInBlock,
        uint32_t headIdx) const {
        return ((uint64_t)blockIdx * numHeads_ + headIdx) *
                   (blockSize_ / GroupRows<IS_KEY>()) * GroupStride<IS_KEY>() +
               static_cast<uint64_t>(groupInBlock) * GroupStride<IS_KEY>();
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

private:
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

}  // namespace

extern "C" __global__ __aicore__ void bit_residual_pack_k8v4(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR query_start_loc,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    auto* rotationPtr = reinterpret_cast<__gm__ TqDataT*>(rotation_t);
    auto* slotMappingPtr = reinterpret_cast<__gm__ int32_t*>(slot_mapping);
    auto* queryStartLocPtr = reinterpret_cast<__gm__ int32_t*>(query_start_loc);
    auto* keyCachePtr = reinterpret_cast<__gm__ uint8_t*>(key_cache);
    auto* valueCachePtr = reinterpret_cast<__gm__ uint8_t*>(value_cache);

    auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
    AscendC::SetSysWorkspace(wsPtr);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    BitResidualPackK8v4<TqDataT> op(
        wsPtr,
        tilingData.nVec,
        tilingData.vecPerCore,
        tilingData.numHeads,
        tilingData.blockSize,
        tilingData.numBlocks,
        tilingData.dataCores,
        tilingData.numReqs,
        tilingData.keyStrideToken,
        tilingData.keyStrideHead,
        tilingData.valueStrideToken,
        tilingData.valueStrideHead,
        tilingData.keyStorageOffset,
        tilingData.valueStorageOffset);
    op.Init(
        key,
        value,
        rotationPtr,
        slotMappingPtr,
        queryStartLocPtr,
        keyCachePtr,
        valueCachePtr);
    op.Process();
}
