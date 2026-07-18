/**
 * BitResidual K8/V4 AIV dequant helpers (head_size=128).
 *
 * AscendC on 910B does not support Cast<int16, uint8>; widen via half/float.
 * Metadata (base/step/vmin/vstep) must be brought into UB via DataCopyPad —
 * raw scalar GM half reads are unreliable on AIV.
 *
 * P2: batch copy contiguous same-PA-block rows (codes + meta runs), then
 * per-row decode into a staged out tile (see DequantKvImpl).
 */
#ifndef BR_DEQUANT_DEVICE_H
#define BR_DEQUANT_DEVICE_H

#include "kernel_operator.h"
#include "br_pack_layout.h"

namespace br_dequant {

using namespace AscendC;
using br_pack::BR_HEAD_SIZE;

// Staging layout inside dequantInt8Buf_ (uint8):
//   [0, headDim)           : key codes OR value nibble bytes
//   [headDim, headDim+32)  : meta0 (base/vmin), 2B used
//   [headDim+32, headDim+64): meta1 (step/vstep), 2B used
static constexpr uint32_t BR_META0_UB_OFF = BR_HEAD_SIZE;
static constexpr uint32_t BR_META1_UB_OFF = BR_HEAD_SIZE + 32U;
static constexpr uint32_t BR_DEQUANT_UB_BYTES = BR_HEAD_SIZE + 64U;

// Pack stores base/step/vmin/vstep as 2-byte floats matching the pack input
// dtype (half or bfloat16). FIA must decode with the same type — reading bf16
// metadata as half (or vice versa) produces catastrophic dequant values.
__aicore__ inline float BrReadFp16FromUb(LocalTensor<uint8_t> ub, LocalTensor<float> scratch,
    uint32_t byteOffset)
{
    auto asHalf = ub.template ReinterpretCast<half>();
    Cast(scratch, asHalf[byteOffset / sizeof(half)], RoundMode::CAST_NONE, 1);
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
    return scratch.GetValue(0);
}

__aicore__ inline float BrReadBf16FromUb(LocalTensor<uint8_t> ub, LocalTensor<float> scratch,
    uint32_t byteOffset)
{
    auto asBf16 = ub.template ReinterpretCast<bfloat16_t>();
    Cast(scratch, asBf16[byteOffset / sizeof(bfloat16_t)], RoundMode::CAST_NONE, 1);
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
    return scratch.GetValue(0);
}

template <typename MetaT>
__aicore__ inline float BrReadMeta16FromUb(LocalTensor<uint8_t> ub, LocalTensor<float> scratch,
    uint32_t byteOffset)
{
    if constexpr (IsSameType<MetaT, bfloat16_t>::value) {
        return BrReadBf16FromUb(ub, scratch, byteOffset);
    } else {
        return BrReadFp16FromUb(ub, scratch, byteOffset);
    }
}

__aicore__ inline void BrCopyMetaPair(GlobalTensor<uint8_t> srcGm, LocalTensor<uint8_t> ub,
    uint64_t meta0Off, uint64_t meta1Off)
{
    // Both half and bfloat16 metadata are 2 bytes.
    DataCopyExtParams metaParams{1, sizeof(half), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    DataCopyPad(ub[BR_META0_UB_OFF], srcGm[meta0Off], metaParams, padParams);
    DataCopyPad(ub[BR_META1_UB_OFF], srcGm[meta1Off], metaParams, padParams);
}

// Copy a contiguous GM meta run into UB with one 32B-aligned slot per row.
// VEC Cast of half/bf16 requires 32B-aligned UB addresses (packed 2B stride faults).
static constexpr uint32_t BR_META_SLOT_BYTES = 32U;

__aicore__ inline void BrCopyMetaRun(GlobalTensor<uint8_t> srcGm, LocalTensor<uint8_t> ubDst,
    uint64_t metaGmOff, uint32_t numRows)
{
    DataCopyExtParams metaParams{1, sizeof(half), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    for (uint32_t j = 0U; j < numRows; ++j) {
        DataCopyPad(ubDst[j * BR_META_SLOT_BYTES], srcGm[metaGmOff + j * sizeof(half)],
            metaParams, padParams);
    }
}

static constexpr uint32_t BR_S2_SUB_MAX = 64U;

__aicore__ inline uint32_t BrAlignUp32(uint32_t x)
{
    return (x + 31U) & ~31U;
}

// Key: y = sign * (base + q7 * step), code = q7 | (sign << 7)
// scratchA/scratchB/outFp32 each hold headDim floats.
// halfScratch: temp for uint8→half widen (AscendC lacks uint8→float cast).
template <typename OutT>
__aicore__ inline void BrDecodeKeyRow(
    LocalTensor<uint8_t> codeUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<float> outFp32,
    LocalTensor<OutT> outUb,
    float base,
    float step,
    uint32_t headDim)
{
    const uint32_t n = headDim;

    // uint8 [0,255] → half → float (supported cast chain).
    Cast(halfScratch, codeUb, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Cast(scratchA, halfScratch, RoundMode::CAST_NONE, n);  // code_f
    PipeBarrier<PIPE_V>();

    // sign_bit = floor(code / 128) ∈ {0,1}
    Muls(scratchB, scratchA, 1.0f / 128.0f, n);
    PipeBarrier<PIPE_V>();
    auto signI16 = outFp32.template ReinterpretCast<int16_t>();
    Cast(signI16, scratchB, RoundMode::CAST_FLOOR, n);
    PipeBarrier<PIPE_V>();
    Cast(scratchB, signI16, RoundMode::CAST_NONE, n);  // sign_f
    PipeBarrier<PIPE_V>();

    // q7 = code - sign * 128
    Muls(outFp32, scratchB, 128.0f, n);
    PipeBarrier<PIPE_V>();
    Sub(outFp32, scratchA, outFp32, n);  // q7_f
    PipeBarrier<PIPE_V>();

    // sign_val = 1 - 2 * sign
    Muls(scratchB, scratchB, -2.0f, n);
    PipeBarrier<PIPE_V>();
    Adds(scratchB, scratchB, 1.0f, n);
    PipeBarrier<PIPE_V>();

    // err = base + q7 * step
    Muls(outFp32, outFp32, step, n);
    PipeBarrier<PIPE_V>();
    Adds(outFp32, outFp32, base, n);
    PipeBarrier<PIPE_V>();

    // decoded = err * sign_val
    Mul(outFp32, outFp32, scratchB, n);
    PipeBarrier<PIPE_V>();
    Cast(outUb, outFp32, RoundMode::CAST_RINT, n);
    PipeBarrier<PIPE_V>();
}

// Value: V = vmin + idx4 * vstep (nibble-packed codes)
template <typename OutT>
__aicore__ inline void BrDecodeValueRow(
    LocalTensor<uint8_t> nibbleUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> idx4F32,
    LocalTensor<float> outFp32,
    LocalTensor<OutT> outUb,
    float vmin,
    float vstep,
    uint32_t headDim)
{
    // int4 → half unpack (same pattern as bit_residual_attention_paged_k8v4).
    Cast(halfScratch, nibbleUb.template ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, headDim);
    PipeBarrier<PIPE_V>();
    Adds(halfScratch, halfScratch, static_cast<half>(8.0f), headDim);
    PipeBarrier<PIPE_V>();
    Cast(idx4F32, halfScratch, RoundMode::CAST_NONE, headDim);
    PipeBarrier<PIPE_V>();
    Muls(outFp32, idx4F32, vstep, headDim);
    PipeBarrier<PIPE_V>();
    Adds(outFp32, outFp32, vmin, headDim);
    PipeBarrier<PIPE_V>();
    Cast(outUb, outFp32, RoundMode::CAST_RINT, headDim);
    PipeBarrier<PIPE_V>();
}

}  // namespace br_dequant

#endif
