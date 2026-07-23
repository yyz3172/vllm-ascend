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
 * Meta P9b: dequantInt8Buf_ holds up to BR_S2_SUB_MAX packed meta rows so
 * each PA run does one meta DMA+cast, then tiles only decode.
 * P14: BrApplyRowAffine — one column loop, fewer PipeBarriers.
 * P13: BrDecodeKeyTile — merge safe Cast/Muls+Adds chains, fewer barriers.
 * P13b: BrDecodeValueTile Cast→Adds→Cast chain; Key drop non-RAW barriers
 * (LSB-sign layout: And→ShiftRight / ShiftRight→Cast(sign)).
 * P17a: BrApplyRowAffine — unroll headDim=128 (columnLoops=2) Mul/Add.
 * P17b-B: Value +8 folded once per PA run (vmin'=vmin+8*vstep on n rows);
 * BrDecodeValueTile consumes signed Cast(int4) with pre-folded vmin'.
 */
#ifndef BR_DEQUANT_DEVICE_H
#define BR_DEQUANT_DEVICE_H

#include "kernel_operator.h"
#include "br_pack_layout.h"

namespace br_dequant {

using namespace AscendC;
using br_pack::BR_HEAD_SIZE;

static constexpr uint32_t BR_S2_SUB_MAX = 64U;
// P9b: packed meta0/meta1 runs then FP32 cast rows (up to BR_S2_SUB_MAX).
// Layout: [0, 128) meta0 half, [128, 256) meta1 half, [256, 512) meta0 fp32,
// [512, 768) meta1 fp32. Cast sources stay 32B-aligned.
static constexpr uint32_t BR_PACKED_META_TILE_ROWS = BR_S2_SUB_MAX;
static constexpr uint32_t BR_PACKED_META_BYTES =
    BR_PACKED_META_TILE_ROWS * static_cast<uint32_t>(sizeof(half));
static constexpr uint32_t BR_META_FP32_ROW_BYTES =
    BR_PACKED_META_TILE_ROWS * static_cast<uint32_t>(sizeof(float));
static constexpr uint32_t BR_META_FP32_0_UB_OFF =
    ((2U * BR_PACKED_META_BYTES + 31U) / 32U) * 32U;
static constexpr uint32_t BR_META_FP32_1_UB_OFF =
    BR_META_FP32_0_UB_OFF + BR_META_FP32_ROW_BYTES;
static constexpr uint32_t BR_DEQUANT_UB_BYTES =
    BR_META_FP32_1_UB_OFF + BR_META_FP32_ROW_BYTES;

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

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1U;
    repeatParams.src0BlkStride = 1U;
    repeatParams.src1BlkStride = 0U;
    repeatParams.src1RepStride = 1U;

    // P17a: headDim=128 → exactly two 64-wide column chunks. Unroll Mul/Add
    // to drop the column for/if control path seen in L6 source OTHER (~25%).
    if (headDim == BR_HEAD_SIZE) {
        constexpr uint32_t ROW_STRIDE_BLOCKS = BR_HEAD_SIZE / FP32_BLOCK_ELEMS;
        repeatParams.dstRepStride = ROW_STRIDE_BLOCKS;
        repeatParams.src0RepStride = ROW_STRIDE_BLOCKS;

        Brcb(broadcast, scales, (numRows + FP32_BLOCK_ELEMS - 1U) /
            FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        Mul(dst[0], src[0], broadcast, FP32_REPEAT_ELEMS, numRows, repeatParams);
        Mul(dst[FP32_REPEAT_ELEMS], src[FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);

        Brcb(broadcast, offsets, (numRows + FP32_BLOCK_ELEMS - 1U) /
            FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        Add(dst[0], dst[0], broadcast, FP32_REPEAT_ELEMS, numRows, repeatParams);
        Add(dst[FP32_REPEAT_ELEMS], dst[FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        PipeBarrier<PIPE_V>();
        return;
    }

    const uint32_t rowStrideBlocks = headDim / FP32_BLOCK_ELEMS;
    const uint32_t columnLoops =
        (headDim + FP32_REPEAT_ELEMS - 1U) / FP32_REPEAT_ELEMS;
    repeatParams.dstRepStride = rowStrideBlocks;
    repeatParams.src0RepStride = rowStrideBlocks;

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

// Key tile: y = sign * (base + q7 * step), code = (q7 << 1) | sign.
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

    // P13: merge back-to-back Cast; keep barriers on RAW deps only.
    // P13b (LSB-sign layout): And→ShiftRight and ShiftRight→Cast(sign) are
    // disjoint-UB; Duplicate→And and ShiftRight→Cast(q7) stay RAW-protected
    // (q7 Cast after the Adds barrier).
    Cast(halfScratch, codesUb, RoundMode::CAST_NONE, N);
    auto codeI16 = scratchA.template ReinterpretCast<int16_t>();
    Cast(codeI16, halfScratch, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    auto codeU16 = codeI16.template ReinterpretCast<uint16_t>();
    auto signStorage = scratchC.template ReinterpretCast<int16_t>();
    auto signStorageU16 = signStorage.template ReinterpretCast<uint16_t>();

    // sign lives in bit0 of the code (code = (q7<<1)|sign).  Isolate it with
    // a 0x01 mask; no shift needed since it is already the low bit.
    auto signMaskU16 = halfScratch.template ReinterpretCast<uint16_t>();
    Duplicate(signMaskU16, static_cast<uint16_t>(0x01), N);
    PipeBarrier<PIPE_V>();
    And(signStorageU16, codeU16, signMaskU16, N);

    // q7 lives in bits 1..7; shift right by 1 to drop the sign and align q7
    // to bits 0..6 (0..127).  In-place on codeU16 (scratchA).
    ShiftRight(codeU16, codeU16, static_cast<uint16_t>(1), N);

    Cast(scratchB, signStorage, RoundMode::CAST_NONE, N);
    Muls(scratchB, scratchB, -2.0f, N);
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

// Value tile: y = vmin' + s*vstep, s = Cast(int4) in [-8,7].
// Caller must pre-fold vmin' = vmin + 8*vstep (P17b-B: once per PA run).
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

    // Signed int4 → fp32; +8 already absorbed into vmin' by the caller.
    Cast(halfScratch, nibbleUb.template ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, N);
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
