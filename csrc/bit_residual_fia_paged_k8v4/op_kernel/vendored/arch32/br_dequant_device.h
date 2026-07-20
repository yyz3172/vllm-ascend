/**
 * BitResidual K8/V4 AIV dequant helpers (head_size=128).
 *
 * AscendC on 910B does not support Cast<int16, uint8>; widen via half then int16.
 * Metadata (base/step/vmin/vstep) must be brought into UB via DataCopyPad —
 * raw scalar GM half reads are unreliable on AIV.
 *
 * P2: batch copy contiguous same-PA-block rows (codes + meta runs), then
 * per-row / tile decode into a staged out tile (see DequantKvImpl).
 * A1: DequantKvImpl dual-buffers tmpBuff1 so MTE2/MTE3 overlap.
 * Key/Value decode: run-level BrDecodeKeyTile / BrDecodeValueTile;
 * tile=8 scratch overlays tmpBuff1 tail (dedicated dequantFp* stays 1-row).
 * Dual-AIV S2 split: both subcores dequant disjoint [0,half)/[half,s2) WS rows.
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
// Run-level Key decode tile: mask Duplicate amortized across this many rows.
// tile=8 scratch (~14KB) overlays tmpBuff1 tail (see DequantKvImpl); dedicated
// dequantFp* stays 1-row so total AIV UB remains within ~192KB.
static constexpr uint32_t BR_DECODE_TILE_MAX = 8U;
static constexpr uint32_t BR_DECODE_TILE_ELEMS = BR_DECODE_TILE_MAX * BR_HEAD_SIZE;

__aicore__ inline uint32_t BrAlignUp32(uint32_t x)
{
    return (x + 31U) & ~31U;
}

// Key tile: y = sign * (base + q7 * step), code = q7 | (sign << 7).
// codesUb: contiguous numRows * headDim uint8.
// halfScratch / scratchA/B/C: each numRows * headDim elements.
// outUb: numRows * headDimAlign (CAST writes headDim elems per row).
// bases/steps: length numRows.
template <typename OutT>
__aicore__ inline void BrDecodeKeyTile(
    LocalTensor<uint8_t> codesUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<float> scratchC,
    LocalTensor<OutT> outUb,
    const float *bases,
    const float *steps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    const uint32_t N = numRows * headDim;

    Cast(halfScratch, codesUb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    auto codeI16 = scratchA.template ReinterpretCast<int16_t>();
    Cast(codeI16, halfScratch, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    auto codeU16 = codeI16.template ReinterpretCast<uint16_t>();
    auto signStorage = scratchC.template ReinterpretCast<int16_t>();
    auto signStorageU16 = signStorage.template ReinterpretCast<uint16_t>();

    // One mask build for the whole tile.
    Duplicate(signStorageU16, static_cast<uint16_t>(0x80), N);
    PipeBarrier<PIPE_V>();
    And(signStorageU16, codeU16, signStorageU16, N);
    PipeBarrier<PIPE_V>();

    auto q7MaskU16 = halfScratch.template ReinterpretCast<uint16_t>();
    Duplicate(q7MaskU16, static_cast<uint16_t>(0x7f), N);
    PipeBarrier<PIPE_V>();
    And(codeU16, codeU16, q7MaskU16, N);
    PipeBarrier<PIPE_V>();

    ShiftRight(signStorage, signStorage, static_cast<int16_t>(7), N);
    PipeBarrier<PIPE_V>();
    Cast(scratchB, signStorage, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Muls(scratchB, scratchB, -2.0f, N);
    PipeBarrier<PIPE_V>();
    Adds(scratchB, scratchB, 1.0f, N);
    PipeBarrier<PIPE_V>();

    // q7 → scratchC; err = base + q7*step → scratchA (overwrites codeI16).
    Cast(scratchC, codeI16, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0U; row < numRows; ++row) {
        const uint32_t off = row * headDim;
        Duplicate(scratchA[off], bases[row], headDim);
        PipeBarrier<PIPE_V>();
        Axpy(scratchA[off], scratchC[off], steps[row], headDim);
        PipeBarrier<PIPE_V>();
    }

    Mul(scratchA, scratchA, scratchB, N);
    PipeBarrier<PIPE_V>();
    if (headDimAlign == headDim) {
        Cast(outUb, scratchA, RoundMode::CAST_RINT, N);
        PipeBarrier<PIPE_V>();
    } else {
        for (uint32_t row = 0U; row < numRows; ++row) {
            Cast(outUb[row * headDimAlign], scratchA[row * headDim], RoundMode::CAST_RINT, headDim);
            PipeBarrier<PIPE_V>();
        }
    }
}

// Single-row wrapper.
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
    float bases[1];
    float steps[1];
    bases[0] = base;
    steps[0] = step;
    BrDecodeKeyTile(codeUb, halfScratch, scratchA, scratchB, outFp32, outUb, bases, steps, 1U, headDim,
        headDim);
}

// Value tile: y = vmin + idx4 * vstep (nibble-packed codes).
// nibbleUb: contiguous numRows * (headDim/2) uint8 (int4 packed).
// halfScratch / scratchA/B: each numRows * headDim elements.
template <typename OutT>
__aicore__ inline void BrDecodeValueTile(
    LocalTensor<uint8_t> nibbleUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<OutT> outUb,
    const float *vmins,
    const float *vsteps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    const uint32_t N = numRows * headDim;

    Cast(halfScratch, nibbleUb.template ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Adds(halfScratch, halfScratch, static_cast<half>(8.0f), N);
    PipeBarrier<PIPE_V>();
    Cast(scratchA, halfScratch, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    for (uint32_t row = 0U; row < numRows; ++row) {
        const uint32_t off = row * headDim;
        Duplicate(scratchB[off], vmins[row], headDim);
        PipeBarrier<PIPE_V>();
        Axpy(scratchB[off], scratchA[off], vsteps[row], headDim);
        PipeBarrier<PIPE_V>();
    }

    if (headDimAlign == headDim) {
        Cast(outUb, scratchB, RoundMode::CAST_RINT, N);
        PipeBarrier<PIPE_V>();
    } else {
        for (uint32_t row = 0U; row < numRows; ++row) {
            Cast(outUb[row * headDimAlign], scratchB[row * headDim], RoundMode::CAST_RINT, headDim);
            PipeBarrier<PIPE_V>();
        }
    }
}

// Single-row wrapper.
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
    float vmins[1];
    float vsteps[1];
    vmins[0] = vmin;
    vsteps[0] = vstep;
    BrDecodeValueTile(nibbleUb, halfScratch, idx4F32, outFp32, outUb, vmins, vsteps, 1U, headDim,
        headDim);
}

}  // namespace br_dequant

#endif
