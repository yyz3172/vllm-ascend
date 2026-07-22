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
 *
 * Meta P1: each decode tile copies contiguous GM meta0/meta1 runs into two
 * aligned 32B UB buffers, then batch-Casts both runs.
 * This replaces 2*numRows 2B DataCopyPad transactions and 32B-per-row slots
 * with two bulk DataCopyPad transactions and 64B fixed UB staging per tile.
 * Meta P4: metadata stays in UB after Cast. Brcb expands each row scalar to
 * one FP32 block and row-broadcast Mul/Add consumes it directly, avoiding
 * GetValue and all metadata V_S synchronization.
 */
#ifndef BR_DEQUANT_DEVICE_H
#define BR_DEQUANT_DEVICE_H

#include "kernel_operator.h"
#include "br_pack_layout.h"

namespace br_dequant {

using namespace AscendC;
using br_pack::BR_HEAD_SIZE;

static constexpr uint32_t BR_S2_SUB_MAX = 64U;
// Legacy 32B packed slots (16 rows) + FP32 cast at +64/+128.
static constexpr uint32_t BR_PACKED_META_TILE_ROWS = 16U;
static constexpr uint32_t BR_PACKED_META_BYTES =
    BR_PACKED_META_TILE_ROWS * static_cast<uint32_t>(sizeof(half));
static constexpr uint32_t BR_META_FP32_0_UB_OFF = 64U;
static constexpr uint32_t BR_META_FP32_1_UB_OFF = 128U;
static constexpr uint32_t BR_DEQUANT_UB_BYTES = 256U;

// Legacy single-row staging offsets (BrCopyMetaPair only).
static constexpr uint32_t BR_META0_UB_OFF = BR_HEAD_SIZE;
static constexpr uint32_t BR_META1_UB_OFF = BR_HEAD_SIZE + 32U;

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

// Copy packed meta0/meta1 runs. GM already uses SoA layout, so each run is
// contiguous; DataCopyPad handles sub-32B tails and potentially unaligned GM.
__aicore__ inline void BrCopyPackedMetaTile(GlobalTensor<uint8_t> srcGm,
    LocalTensor<uint8_t> dst, uint64_t meta0GmOff, uint64_t meta1GmOff,
    uint32_t numRows)
{
    DataCopyExtParams metaParams{
        1, numRows * static_cast<uint32_t>(sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    DataCopyPad(dst, srcGm[meta0GmOff], metaParams, padParams);
    DataCopyPad(dst[BR_PACKED_META_BYTES], srcGm[meta1GmOff], metaParams, padParams);
}

template <typename MetaT>
__aicore__ inline void BrCastPackedMetaToFp32(LocalTensor<uint8_t> src,
    uint32_t numRows, LocalTensor<float> dst0, LocalTensor<float> dst1)
{
    if constexpr (IsSameType<MetaT, bfloat16_t>::value) {
        auto src0 = src.template ReinterpretCast<bfloat16_t>();
        auto src1 = src[BR_PACKED_META_BYTES].template ReinterpretCast<bfloat16_t>();
        Cast(dst0, src0, RoundMode::CAST_NONE, numRows);
        Cast(dst1, src1, RoundMode::CAST_NONE, numRows);
    } else {
        auto src0 = src.template ReinterpretCast<half>();
        auto src1 = src[BR_PACKED_META_BYTES].template ReinterpretCast<half>();
        Cast(dst0, src0, RoundMode::CAST_NONE, numRows);
        Cast(dst1, src1, RoundMode::CAST_NONE, numRows);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BrApplyRowAffine(LocalTensor<float> dst,
    LocalTensor<float> src, LocalTensor<float> offsets,
    LocalTensor<float> scales, LocalTensor<float> broadcast,
    uint32_t numRows, uint32_t headDim)
{
    constexpr uint32_t FP32_BLOCK_ELEMS = 8U;
    constexpr uint32_t FP32_REPEAT_ELEMS = 64U;
    const uint32_t rowStrideBlocks = headDim / FP32_BLOCK_ELEMS;
    const uint32_t columnLoops =
        (headDim + FP32_REPEAT_ELEMS - 1U) / FP32_REPEAT_ELEMS;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1U;
    repeatParams.src0BlkStride = 1U;
    repeatParams.src1BlkStride = 0U;
    repeatParams.dstRepStride = rowStrideBlocks;
    repeatParams.src0RepStride = rowStrideBlocks;
    repeatParams.src1RepStride = 1U;

    Brcb(broadcast, scales, (numRows + FP32_BLOCK_ELEMS - 1U) /
        FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
    PipeBarrier<PIPE_V>();
    for (uint32_t columnLoop = 0U; columnLoop < columnLoops; ++columnLoop) {
        const uint32_t columnOffset = columnLoop * FP32_REPEAT_ELEMS;
        uint32_t columnCount = headDim - columnOffset;
        if (columnCount > FP32_REPEAT_ELEMS) {
            columnCount = FP32_REPEAT_ELEMS;
        }
        Mul(dst[columnOffset], src[columnOffset], broadcast, columnCount,
            numRows, repeatParams);
    }
    PipeBarrier<PIPE_V>();

    Brcb(broadcast, offsets, (numRows + FP32_BLOCK_ELEMS - 1U) /
        FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
    PipeBarrier<PIPE_V>();
    for (uint32_t columnLoop = 0U; columnLoop < columnLoops; ++columnLoop) {
        const uint32_t columnOffset = columnLoop * FP32_REPEAT_ELEMS;
        uint32_t columnCount = headDim - columnOffset;
        if (columnCount > FP32_REPEAT_ELEMS) {
            columnCount = FP32_REPEAT_ELEMS;
        }
        Add(dst[columnOffset], dst[columnOffset], broadcast, columnCount,
            numRows, repeatParams);
    }
    PipeBarrier<PIPE_V>();
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

static constexpr uint32_t BR_KEY_DECODE_TILE_MAX = 8U;
static constexpr uint32_t BR_VALUE_DECODE_TILE_MAX = 13U;

__aicore__ inline uint32_t BrAlignUp32(uint32_t x)
{
    return (x + 31U) & ~31U;
}

// Key tile: y = sign * (base + q7 * step), code = q7 | (sign << 7).
// codesUb: contiguous numRows * headDim uint8.
// halfScratch / scratchA/B/C: each numRows * headDim elements.
// outUb: numRows * headDimAlign (CAST writes headDim elems per row).
// bases/steps: UB tensors containing numRows FP32 values.
template <typename OutT>
__aicore__ inline void BrDecodeKeyTile(
    LocalTensor<uint8_t> codesUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<float> scratchC,
    LocalTensor<OutT> outUb,
    LocalTensor<float> bases,
    LocalTensor<float> steps,
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

    // Codes are widened uint8 values in [0,255], so shifting directly
    // extracts the encoded sign bit without constructing a 0x80 mask tile.
    ShiftRight(signStorageU16, codeU16, static_cast<uint16_t>(7), N);
    PipeBarrier<PIPE_V>();

    auto q7MaskU16 = halfScratch.template ReinterpretCast<uint16_t>();
    Duplicate(q7MaskU16, static_cast<uint16_t>(0x7f), N);
    PipeBarrier<PIPE_V>();
    And(codeU16, codeU16, q7MaskU16, N);
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
    auto metaBroadcast = halfScratch.template ReinterpretCast<float>();
    BrApplyRowAffine(
        scratchA, scratchC, bases, steps, metaBroadcast, numRows, headDim);

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
    LocalTensor<float> vmins,
    LocalTensor<float> vsteps,
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

    auto metaBroadcast = halfScratch.template ReinterpretCast<float>();
    BrApplyRowAffine(
        scratchB, scratchA, vmins, vsteps, metaBroadcast, numRows, headDim);

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

}  // namespace br_dequant

#endif
