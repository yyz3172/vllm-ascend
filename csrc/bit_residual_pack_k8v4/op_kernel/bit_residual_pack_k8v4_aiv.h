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
#include "bit_residual_pack_k8v4_common.h"

namespace bit_residual {

using namespace AscendC;

template <typename T>
class BitResidualPackK8v4VectorService {
public:
    __aicore__ inline explicit BitResidualPackK8v4VectorService(
        BitResidualPackK8v4Context<T>& context)
        : context_(context) {}

    __aicore__ inline void Process(uint32_t manualGroupId, uint32_t aivSlice) {
        if (aivSlice >= TQ_AIV_SUB_BLOCKS) {
            return;
        }

        auto& op = context_;
        AscendC::GlobalTensor<half> cWorkGm;
        op.InitManualWorkspaceTensors(manualGroupId, cWorkGm);

        for (uint32_t buffer = 0; buffer < TQ_MANUAL_WORKSPACE_BUFFER_COUNT; ++buffer) {
            TqCrossCoreSet<PIPE_MTE2>(
                static_cast<uint16_t>(buffer * TQ_MANUAL_SYNC_PP_STRIDE +
                                      TQ_MANUAL_SYNC_C_FREE));
        }

        uint32_t manualTileOrdinal = 0;
        const uint32_t subBlocksPerBlock = op.blockSize_ / TQ_BLOCK_ROWS;
        const uint32_t headTileCount =
            (op.numHeads_ + TQ_MANUAL_HEADS_PER_TILE - 1) /
            TQ_MANUAL_HEADS_PER_TILE;
        bool hasPending = false;
        uint32_t pendingInputBuffer = 0;
        ManualKey1StreamDesc pendingDesc {};
        for (uint32_t reqIdx = 0; reqIdx < op.numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!op.ReadRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                return;
            }
            if (seqStart == seqEnd) {
                continue;
            }
            uint32_t firstSlot = 0;
            if (!op.ResolveTokenSlotU(seqStart, firstSlot)) {
                return;
            }
            const uint32_t rowCount = seqEnd - seqStart;
            const uint32_t leadingGroupRow = firstSlot % TQ_MANUAL_GROUP_ROWS;
            for (uint32_t rowOff = 0; rowOff < rowCount;) {
                const uint32_t startGroupRow = (rowOff == 0) ? leadingGroupRow : 0;
                const uint32_t rowsInThisGroup = TQ_MANUAL_GROUP_ROWS - startGroupRow;
                const uint32_t validRows = (rowCount - rowOff > rowsInThisGroup)
                    ? rowsInThisGroup
                    : rowCount - rowOff;
                const bool preserveValue =
                    startGroupRow != 0 || validRows < TQ_MANUAL_GROUP_ROWS;
                const uint32_t tokenStart = seqStart + rowOff;
                const uint32_t slot = firstSlot + rowOff;
                const uint32_t blockIdx = slot / op.blockSize_;
                const uint32_t blockOffset = slot - blockIdx * op.blockSize_;
                const uint32_t subBlockInBlock = blockOffset / TQ_BLOCK_ROWS;
                const uint32_t valueGroupInBlock = blockOffset / TQ_MANUAL_GROUP_ROWS;
                for (uint32_t headTileStart = 0;
                     headTileStart < op.numHeads_;
                     headTileStart += TQ_MANUAL_HEADS_PER_TILE) {
                    const uint64_t workOrdinal =
                        (static_cast<uint64_t>(blockIdx) * subBlocksPerBlock +
                         subBlockInBlock) * headTileCount +
                        headTileStart / TQ_MANUAL_HEADS_PER_TILE;
                    if (workOrdinal % op.dataCores_ == manualGroupId) {
                        const uint32_t keyStream =
                            manualTileOrdinal * TQ_MANUAL_STREAM_KIND_COUNT;
                        ManualKey1StreamDesc keyDesc {
                            tokenStart, slot, blockIdx, valueGroupInBlock,
                            headTileStart, startGroupRow, validRows, keyStream,
                            preserveValue, false};
                        ManualKey1StreamDesc valueDesc {
                            tokenStart, slot, blockIdx, valueGroupInBlock,
                            headTileStart, startGroupRow, validRows, keyStream + 1,
                            preserveValue, true};
                        SubmitStream(cWorkGm, keyDesc, aivSlice, hasPending,
                                     pendingDesc, pendingInputBuffer);
                        SubmitStream(cWorkGm, valueDesc, aivSlice, hasPending,
                                     pendingDesc, pendingInputBuffer);
                        ++manualTileOrdinal;
                    }
                }
                rowOff += validRows;
            }
        }
        DrainStream(aivSlice, hasPending, pendingDesc, pendingInputBuffer);
    }

private:
    /*
        Key 7-bit + sign quantization, half precision, no normalization.

        Three physical UB slots carry every live value:
          yBatch (XY_BATCH, 4 KB half)  = the AIC Fixpipe F322F16 half output
            read directly as half — IS origBatch16 (no fp32→half Cast).
          fat   (EncodeBuffer1, 4 KB half)
            absVec → quant_16 → quant_i16 → final → pack_u8
          tmpB  (EncodeTmpB, 4 KB half)  — carved from the 4 KB freed by
            halving each XY slot from 8 KB fp32 to 4 KB half.
            signVec → shiftedSign → pack_half
          scal (EncodeBuffer2, sub-blocked; slots are TQ_ENCODE_SCALAR_TILE_ELEMS half apart)
            slot0 [0×]   : maxVec → baseBlk   (max dead after gap; reused)
            slot1 [1×]   : minVec              (long-lived → metadata base)
            slot2 [2×]   : gap → step (in-place, long-lived → metadata step)
            slot3 [3×]   : stepBlk             (freed after Div)

        origin16 = yBatch                              (half, direct) [yBatch]
        sign     = ShiftRight(origin16, 15)            (bit15)        [tmpB]
        abs      = Abs(origin16)                                      [fat]
        // origin16 dead; yBatch free (unused below)
        max      = WholeReduceMax(abs, per row)                       [scal slot0]
        min      = WholeReduceMin(abs, per row)       (=base)         [scal slot1]
        gap      = Sub(max, min)                                      [scal slot2]
        step     = Maxs(Muls(gap, 1/127), eps)        (in-place)      [scal slot2]
        // max dead; slot0 free
        baseBlk  = Brcb(min, 16)                      (reuse slot0)
        stepBlk  = Brcb(step, 16)                                     [scal slot3]
        quant    = Sub(abs, baseBlk)                                  [fat]
        // abs dead
        quant    = Div(quant, stepBlk)               (in-place)       [fat]
        // stepBlk dead; slot3 free
        q_i16    = Cast(quant, RINT)                 (in-place)       [fat]
        shSign   = ShiftLeft(sign, 7)                                [tmpB]
        // sign dead; tmpB free for pack_half
        final    = Or(q_i16, shSign)                  (in-place)       [fat]
        // shSign dead
        pack_h   = Cast(final, half)                                 [tmpB]
        pack_u8  = Cast(pack_h, u8)                  (in-place)       [fat]
        DataCopy(encodedBatch, pack_u8)
        // base/step metadata already half-domain — no scale-back.
    */
    __aicore__ inline void EncodeKeyBatch(
        uint32_t m,
        AscendC::LocalTensor<half>& yBatch) {
        static constexpr float TQ_KEY_QUANT_LEVELS_F = 1.0f / 127.0f;
        using ComputeT = half;
        static_assert(TQ_MANUAL_AIV_SLICE_M == TQ_VECTOR_BATCH,
                      "key encoder expects one physical 16-row tile");
        constexpr uint32_t typePerBlock = ASCEND_BLOCK_BYTES / sizeof(ComputeT);
        const ComputeT safeDivisor = static_cast<ComputeT>(1.0e-6f);
        const uint32_t totalElems = m * TQ_PACK_D;
        constexpr uint32_t binaryRepeatBatchF16 = 128;
        constexpr uint32_t repeatsOfRows = TQ_VECTOR_BATCH / 8;

        auto encodedBatch = context_.resource_.KeyEncodedBatch();

        // ── Three physical slots, reused as the pipeline progresses ────────
        auto fat = context_.resource_.EncodeBuffer1().template ReinterpretCast<half>();
        auto tmpB = context_.resource_.EncodeTmpB().template ReinterpretCast<half>();
        auto scal = context_.resource_.EncodeBuffer2().template ReinterpretCast<half>();
        // scal sub-blocks: one Brcb/reduce tile per slot
        // (TQ_ENCODE_SCALAR_TILE_ELEMS half apart).
        auto sMaxVec = scal;            // slot0: max → baseBlk (reused after max dead)
        auto minVec = scal[TQ_ENCODE_SCALAR_TILE_ELEMS];         // slot1: min (long-lived → base meta)
        auto stepVec = scal[2 * TQ_ENCODE_SCALAR_TILE_ELEMS];    // slot2: gap → step (in-place, long-lived)
        auto stepBlk = scal[3 * TQ_ENCODE_SCALAR_TILE_ELEMS];    // slot3: stepBlk (freed after Div)
        auto baseBlk = sMaxVec;         // slot0 reused

        // yBatch IS the half input (AIC Fixpipe F322F16 output) — no Cast.
        auto origBatch16 = yBatch;       // [yBatch]
        // fat = absVec → quant chain; tmpB = signVec → shiftedSign → pack_half.
        auto signVec = tmpB.template ReinterpretCast<uint16_t>();  // [tmpB]
        auto absVec = fat;                                            // [fat]

        // ── Step 1: origBatch16 = yBatch (half input, direct) ──

        // ── Step 2: signVec(u16) in tmpB = sign bit (bit 15) of origBatch16 ──
        //    Hoisted before Abs so origBatch16 dies earlier, freeing yBatch
        //    (unused below) while sign lives in tmpB and abs lands in fat.
        AscendC::ShiftRight(signVec,
                            origBatch16.template ReinterpretCast<uint16_t>(),
                            static_cast<uint16_t>(15), totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 3: absVec(half) in fat = |origBatch16| ──
        AscendC::Abs(absVec, origBatch16, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // origBatch16 (yBatch) is dead → yBatch free (unused below); fat holds abs.

        // ── Step 4: maxVec(half) in scal slot0 = WholeReduceMax(absVec) ──
        AscendC::WholeReduceMax<half, false>(
            sMaxVec, absVec, static_cast<int32_t>(TQ_PACK_D), m, 1, 1,
            TQ_PACK_D / typePerBlock, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 5: minVec(half, =base) in scal slot1 = WholeReduceMin(absVec) ──
        AscendC::WholeReduceMin<half, false>(
            minVec, absVec, static_cast<int32_t>(TQ_PACK_D), m, 1, 1,
            TQ_PACK_D / typePerBlock, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 6: gap→step in scal slot2 (in-place) = (max-min)*1/127 ──
        //    maxVec dies here (last use in gap).  stepVec must survive to the
        //    metadata write (Step 16).
        AscendC::Sub(stepVec, sMaxVec, minVec, m);
        AscendC::PipeBarrier<PIPE_V>();
        // sMaxVec (slot0) is dead → reusable for baseBlk.
        AscendC::Muls(stepVec, stepVec,
                      static_cast<half>(TQ_KEY_QUANT_LEVELS_F), m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(stepVec, stepVec, safeDivisor, m);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 7a: baseBlk(half) in scal slot0 (reuse) = Brcb(minVec) ──
        //    Each repeat takes 8 src scalars → 8 datablocks (16 half each).
        //    2 repeats → 16 datablocks (one per row), 256 half total.
        AscendC::Brcb(baseBlk, minVec, repeatsOfRows, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 7b: stepBlk(half) in scal slot3 = Brcb(stepVec) ──
        AscendC::Brcb(stepBlk, stepVec, repeatsOfRows, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 8: quant_16(half) in fat = Sub(absVec, baseBlk) ──
        //    Quantize |x| (not x): base/step live in the |x| domain; sign is
        //    recombined later from signVec.  absVec is consumed here and dies.
        //    mask=128 (8 blocks), repeatTimes=16 (16 rows).
        //    BinaryRepeatParams(1,1,0, 8,8,1): per-row src1 broadcast.
        auto quant_16 = fat;
        AscendC::Sub<half, false>(
            quant_16, absVec, baseBlk,
            binaryRepeatBatchF16, m,
            AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        AscendC::PipeBarrier<PIPE_V>();
        // baseBlk (slot0), absVec (fat input) are dead; fat holds quant_16.

        // ── Step 9: quant_16 = Div(quant_16, stepBlk) (in-place in fat) ──
        AscendC::Div<half, false>(
            quant_16, quant_16, stepBlk,
            binaryRepeatBatchF16, m,
            AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        AscendC::PipeBarrier<PIPE_V>();
        // stepBlk (slot3) is dead.

        // ── Step 10: quant_i16(i16) in fat (in-place) = Cast(quant_16, RINT) ──
        auto quant_i16 = fat.template ReinterpretCast<int16_t>();
        AscendC::Cast(quant_i16, quant_16, AscendC::RoundMode::CAST_RINT, totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 11: shiftedSign(i16) in tmpB = ShiftLeft(signVec, 7) ──
        auto shiftedSign = tmpB.template ReinterpretCast<int16_t>();
        AscendC::ShiftLeft(shiftedSign, signVec.template ReinterpretCast<int16_t>(),
                           static_cast<int16_t>(7), totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // signVec (tmpB input) is dead → tmpB free for pack_half.

        // ── Step 12: final_quant_16(i16) in fat (in-place)
        //            = Or(quant_i16, shiftedSign) — bit 7 = sign, bits 0..6 =
        //            7-bit quant code (0..127).
        AscendC::Or(quant_i16, quant_i16, shiftedSign, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // shiftedSign (tmpB) is dead.
        auto final_quant_16 = quant_i16;

        // ── Step 13: pack_half(half) in tmpB = Cast(final_quant_16 → half) ──
        //    Numerical cast: uint16 value 0..255 → half 0.0..255.0.
        auto pack_half = tmpB;
        AscendC::Cast(pack_half, final_quant_16, AscendC::RoundMode::CAST_NONE, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // final_quant_16 (fat) is dead → fat free for pack_u8.

        // ── Step 14: pack_u8(uint8) in fat = Cast(pack_half → u8) ──
        auto pack_u8 = fat.template ReinterpretCast<uint8_t>();
        AscendC::Cast(pack_u8, pack_half, AscendC::RoundMode::CAST_NONE, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // pack_half (tmpB) is dead.

        // ── Step 15: DataCopy 2048 code bytes to encodedBatch[0..2047] ──
        auto encodedBytes = encodedBatch.template ReinterpretCast<uint8_t>();
        AscendC::DataCopy(encodedBytes, pack_u8, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // pack_u8 (fat) is dead.

        // ── Step 16: write base/step metadata to encodedBatch ──
        //    base/step are already in the original (half) domain — no scale-
        //    back needed because we quantized origBatch16 directly (no prior
        //    normalization).  quant chain is done → fat is free as fp32 scratch
        //    for the bf16 metadata path; minVec/stepVec live in scal slots 1/2.
        auto metadataBaseT = encodedBatch[
            TQ_KEY_ENCODED_BASE_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        auto metadataStepT = encodedBatch[
            TQ_KEY_ENCODED_STEP_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            auto metadataBaseFp32 = fat.template ReinterpretCast<float>();
            auto metadataStepFp32 = metadataBaseFp32[TQ_BLOCK_ROWS];
            AscendC::Cast(metadataBaseFp32, minVec,
                          AscendC::RoundMode::CAST_NONE, m);
            AscendC::Cast(metadataStepFp32, stepVec,
                          AscendC::RoundMode::CAST_NONE, m);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(metadataBaseT, metadataBaseFp32,
                          AscendC::RoundMode::CAST_RINT, m);
            AscendC::Cast(metadataStepT, metadataStepFp32,
                          AscendC::RoundMode::CAST_RINT, m);
        } else {
            AscendC::Adds(metadataBaseT, minVec, static_cast<half>(0.0f), m);
            AscendC::Adds(metadataStepT, stepVec, static_cast<half>(0.0f), m);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // minVec (scal slot1), stepVec (scal slot2) are dead.
    }
    // ── Value quantization ────────────────────────────────────────────────
    // Process all 16 rows as one vector batch.  yBatch IS the half input
    // (AIC Fixpipe F322F16 output) — compute per-row max/min directly on it
    // and quantize (x - vmin) / vstep into 4-bit.  No normalization, no abs,
    // no sign extraction — vmin/vstep are already in the original domain.
    //
    // Two physical UB slots (no third needed: value has no sign/abs branch):
    //   yBatch (XY_BATCH, 4 KB half) = origBatch16 until Step 6's Sub, then dead.
    //   fat    (EncodeBuffer1, 4 KB half): quant_16 → range16; then free for
    //          the bf16 metadata fp32 scratch.
    //   scal (EncodeBuffer2, sub-blocked; slots are TQ_ENCODE_SCALAR_TILE_ELEMS half apart):
    //        slot0 [0×]  : max → minBlk (reused after max dead)
    //        slot1 [1×]  : min (long-lived → vmin meta)
    //        slot2 [2×]  : range → step (in-place, long-lived → vstep meta)
    //        slot3 [3×]  : stepBlk (freed after Div)
    __aicore__ inline void EncodeValueBatch(
        uint32_t m,
        AscendC::LocalTensor<half>& yBatch) {
        static constexpr float TQ_VAL_QUANT_LEVELS_F = 1.0f / 15.0f;
        static_assert(TQ_MANUAL_AIV_SLICE_M == TQ_VECTOR_BATCH,
                      "value encoder expects one physical 16-row tile");
        constexpr uint32_t typePerBlock = ASCEND_BLOCK_BYTES / sizeof(half);
        const half safeDivisor = static_cast<half>(1.0e-6f);
        const uint32_t totalElems = m * TQ_PACK_D;
        constexpr uint32_t binaryRepeatBatchF16 = 128;
        constexpr uint32_t repeatsOfRows = TQ_VECTOR_BATCH / 8;

        auto encodedBatch = context_.resource_.KeyEncodedBatch();
        // ── Two physical slots, reused as the pipeline progresses ────────
        auto fat = context_.resource_.EncodeBuffer1().template ReinterpretCast<half>();
        auto scal = context_.resource_.EncodeBuffer2().template ReinterpretCast<half>();
        // scal sub-blocks: one Brcb/reduce tile per slot
        // (TQ_ENCODE_SCALAR_TILE_ELEMS half apart).
        auto sMaxVec = scal;            // slot0: max → minBlk (reused after max dead)
        auto minVec = scal[TQ_ENCODE_SCALAR_TILE_ELEMS];         // slot1: min (long-lived → vmin meta)
        auto stepVec = scal[2 * TQ_ENCODE_SCALAR_TILE_ELEMS];    // slot2: range → step (in-place, long-lived)
        auto stepBlk = scal[3 * TQ_ENCODE_SCALAR_TILE_ELEMS];    // slot3: stepBlk (freed after Div)
        auto minBlk = sMaxVec;          // slot0 reused

        // ── Step 1: origBatch16 = yBatch (half input, direct — no Cast) ──
        auto origBatch16 = yBatch;

        // ── Step 2: maxVec(half) in scal slot0 = WholeReduceMax(origBatch16) ──
        //    No Abs — value quantizes the raw (signed) data directly.
        AscendC::WholeReduceMax<half, false>(
            sMaxVec, origBatch16, static_cast<int32_t>(TQ_PACK_D), m, 1, 1,
            TQ_PACK_D / typePerBlock, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 3: minVec(half) in scal slot1 = WholeReduceMin(origBatch16) ──
        AscendC::WholeReduceMin<half, false>(
            minVec, origBatch16, static_cast<int32_t>(TQ_PACK_D), m, 1, 1,
            TQ_PACK_D / typePerBlock, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 4: range→step in scal slot2 (in-place) = (max-min)*1/15 ──
        //    maxVec dies here (last use in range).  stepVec must survive to
        //    the metadata write (Step 11).
        AscendC::Sub(stepVec, sMaxVec, minVec, m);
        AscendC::PipeBarrier<PIPE_V>();
        // sMaxVec (slot0) is dead → reusable for minBlk.
        AscendC::Muls(stepVec, stepVec,
                      static_cast<half>(TQ_VAL_QUANT_LEVELS_F), m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(stepVec, stepVec, safeDivisor, m);
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 5a: minBlk(half) in scal slot0 (reuse) = Brcb(minVec) ──
        AscendC::Brcb(minBlk, minVec, repeatsOfRows, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 5b: stepBlk(half) in scal slot3 = Brcb(stepVec) ──
        AscendC::Brcb(stepBlk, stepVec, repeatsOfRows, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // ── Step 6: quant_16(half) in fat = Sub(origBatch16, minBlk) ──
        //    origBatch16 lives in yBatch and is consumed here (then dies).
        //    BinaryRepeatParams(1,1,0, 8,8,1): per-row src1 broadcast.
        auto quant_16 = fat;
        AscendC::Sub<half, false>(
            quant_16, origBatch16, minBlk,
            binaryRepeatBatchF16, m,
            AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        AscendC::PipeBarrier<PIPE_V>();
        // minBlk (slot0), origBatch16 (yBatch) are dead → yBatch free; fat holds quant_16.

        // ── Step 7: quant_16 = Div(quant_16, stepBlk) (in-place in fat) ──
        AscendC::Div<half, false>(
            quant_16, quant_16, stepBlk,
            binaryRepeatBatchF16, m,
            AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        AscendC::PipeBarrier<PIPE_V>();
        // stepBlk (slot3) is dead.

        // ── Step 8: range16(half) in fat (in-place) = Adds(quant_16, -8) ──
        //    Shift [0, 15] → [-8, 7] for signed int4 Cast.
        AscendC::Adds(quant_16, quant_16, static_cast<half>(-8.0f), totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        auto range16 = quant_16;

        // ── Step 9: Cast range16 → int4b_t directly into encodedBatch ──
        auto quantI4 = encodedBatch.template ReinterpretCast<int4b_t>();
        AscendC::Cast(quantI4, range16, AscendC::RoundMode::CAST_NONE, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // range16 (fat) is dead.

        // ── Step 10: write vmin/vstep metadata to encodedBatch ──
        //    vmin/vstep are already in the original (half) domain — no scale-
        //    back needed because we quantized origBatch16 directly (no prior
        //    normalization).  fat is free as fp32 scratch for the bf16 path.
        auto metadataMinT = encodedBatch[
            TQ_VAL_ENCODED_VMIN_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                .template ReinterpretCast<T>();
        auto metadataStepT = encodedBatch[
            TQ_VAL_ENCODED_VSTEP_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            auto metadataMinFp32 = fat.template ReinterpretCast<float>();
            auto metadataStepFp32 = metadataMinFp32[typePerBlock];
            AscendC::Cast(metadataMinFp32, minVec,
                          AscendC::RoundMode::CAST_NONE, m);
            AscendC::Cast(metadataStepFp32, stepVec,
                          AscendC::RoundMode::CAST_NONE, m);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(metadataMinT, metadataMinFp32,
                          AscendC::RoundMode::CAST_RINT, m);
            AscendC::Cast(metadataStepT, metadataStepFp32,
                          AscendC::RoundMode::CAST_RINT, m);
        } else {
            AscendC::Adds(metadataMinT, minVec, static_cast<half>(0.0f), m);
            AscendC::Adds(metadataStepT, stepVec, static_cast<half>(0.0f), m);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // minVec (scal slot1), stepVec (scal slot2) are dead.
    }

    __aicore__ inline void CopyManualCToYBatch(
        AscendC::GlobalTensor<half>& cWorkGm,
        uint32_t bufferOffset,
        uint32_t aivSlice,
        uint32_t inputBuffer) {
        auto yBatch = context_.resource_.YBatchHalf(inputBuffer);
        static constexpr uint32_t SLICE_ELEMS =
            TQ_MANUAL_AIV_SLICE_M * TQ_ROT_N;  // 16×128 = 2048 half elems
        const uint32_t sliceElemOffset = aivSlice * SLICE_ELEMS;
        AscendC::DataCopy(
            yBatch,
            cWorkGm[bufferOffset + sliceElemOffset],
            SLICE_ELEMS);
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t EncodedRowStrideWords() const {
        return IS_KEY ? TQ_KEY_ENCODED_ROW_STRIDE_WORDS : TQ_VAL_ENCODED_ROW_STRIDE_WORDS;
    }

    // ── Independent-row cache layout helpers ──────────────────────────────
    // Per head: [all row codes] [all row base/vmin] [all row step/vstep].

    template <bool IS_KEY>
    __aicore__ inline uint32_t HeadStride() const {
        return context_.blockSize_ *
            (IS_KEY ? TQ_KEY_BYTES_PER_ROW : TQ_VAL_BYTES_PER_ROW);
    }

    template <bool IS_KEY>
    __aicore__ inline void WriteEncodedSliceToCache(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        auto& op = context_;
        const uint32_t headIdx = desc.headTileStart + aivSlice;
        if (headIdx >= op.numHeads_) {
            return;
        }
        TqSyncVToS();
        auto encoded = op.resource_.KeyEncodedBatch();
        auto scratch = op.resource_.PackedRow();
        const uint32_t codeBytes = IS_KEY ? TQ_KEY_ROW_CODE_BYTES : TQ_VAL_ROW_CODE_BYTES;
        const uint32_t headStride = HeadStride<IS_KEY>();
        uint32_t rowOff = 0;
        while (rowOff < desc.validRows) {
            const uint32_t slot = desc.slotStart + rowOff;
            const uint32_t blockIdx = slot / op.blockSize_;
            const uint32_t blockRow = slot % op.blockSize_;
            const uint32_t tileRow = blockRow % TQ_BLOCK_ROWS;
            uint32_t rows = TQ_BLOCK_ROWS - tileRow;
            rows = rows > desc.validRows - rowOff ? desc.validRows - rowOff : rows;
            const uint32_t tileStart = blockRow - tileRow;
            const uint64_t headBase =
                (static_cast<uint64_t>(blockIdx) * op.numHeads_ + headIdx) * headStride;
            const uint64_t meta0Base = headBase + op.blockSize_ * codeBytes;
            const uint64_t meta1Base = meta0Base + op.blockSize_ * TQ_ROW_META_BYTES;
            auto meta0 = scratch.template ReinterpretCast<T>();
            auto meta1 = scratch[TQ_META_TILE_BYTES].template ReinterpretCast<T>();
            copy_packed_gm_to_ub(scratch, packedGm,
                meta0Base + tileStart * TQ_ROW_META_BYTES, TQ_META_TILE_BYTES);
            auto meta1Bytes = scratch[TQ_META_TILE_BYTES];
            auto codeBytesUb = scratch[2 * TQ_META_TILE_BYTES];
            copy_packed_gm_to_ub(meta1Bytes, packedGm,
                meta1Base + tileStart * TQ_ROW_META_BYTES, TQ_META_TILE_BYTES);
            TqSyncMte2ToS();
            // ── Code bytes: DataCopy (32B aligned) + Meta: Copy (non-aligned UB→UB) ──
            auto encodedBytes = encoded.template ReinterpretCast<uint8_t>();
            const uint32_t codeRowBytes = IS_KEY ? TQ_KEY_ROW_CODE_BYTES : TQ_VAL_ROW_CODE_BYTES;
            const uint32_t codeSrcOff = (desc.startGroupRow + rowOff) * codeRowBytes;
            auto codeOutAll = scratch[2 * TQ_META_TILE_BYTES];
            DataCopy(codeOutAll, encodedBytes[codeSrcOff], rows * codeRowBytes);
            PipeBarrier<PIPE_V>();
            // Meta0/meta1: Copy (V-pipe, supports non-aligned offsets)
            auto encodedT = encoded.template ReinterpretCast<T>();
            const uint32_t metaEncodedRow = desc.startGroupRow + rowOff;
            if constexpr (IS_KEY) {
                auto encodedBytes = encoded.template ReinterpretCast<uint8_t>();
                const uint32_t srcOff = (desc.startGroupRow + rowOff) * TQ_KEY_ROW_CODE_BYTES;
                auto codeOutAll = scratch[2 * TQ_META_TILE_BYTES];
                // rows×128 bytes contiguous in both src and dst
                DataCopy(codeOutAll, encodedBytes[srcOff], rows * TQ_KEY_ROW_CODE_BYTES);
                PipeBarrier<PIPE_V>();
                // Metadata only: 2 scalar SetValue per row
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t encodedRow = desc.startGroupRow + rowOff + r;
                    meta0.SetValue(tileRow + r, encoded[
                        TQ_KEY_ENCODED_BASE_BATCH_BYTE_OFFSET / sizeof(uint16_t) +
                        encodedRow]
                            .template ReinterpretCast<T>().GetValue(0));
                    meta1.SetValue(tileRow + r, encoded[
                        TQ_KEY_ENCODED_STEP_BATCH_BYTE_OFFSET / sizeof(uint16_t) +
                        encodedRow]
                            .template ReinterpretCast<T>().GetValue(0));
                }
            } else {
                auto encodedBytes = encoded.template ReinterpretCast<uint8_t>();
                const uint32_t srcOff = (desc.startGroupRow + rowOff) * TQ_VAL_ROW_CODE_BYTES;
                auto codeOutAll = scratch[2 * TQ_META_TILE_BYTES];
                DataCopy(codeOutAll, encodedBytes[srcOff], rows * TQ_VAL_ROW_CODE_BYTES);
                PipeBarrier<PIPE_V>();
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t encodedRow = desc.startGroupRow + rowOff + r;
                    meta0.SetValue(tileRow + r, encoded[
                        TQ_VAL_ENCODED_VMIN_BATCH_BYTE_OFFSET / sizeof(uint16_t) +
                        encodedRow]
                            .template ReinterpretCast<T>().GetValue(0));
                    meta1.SetValue(tileRow + r, encoded[
                        TQ_VAL_ENCODED_VSTEP_BATCH_BYTE_OFFSET / sizeof(uint16_t) +
                        encodedRow]
                            .template ReinterpretCast<T>().GetValue(0));
                }
            }
            TqSyncSToMte3();
            copy_packed_ub_to_gm(
                packedGm,
                headBase + static_cast<uint64_t>(blockRow) * codeBytes,
                codeBytesUb, rows * codeBytes);
            copy_packed_ub_to_gm(packedGm, meta0Base + tileStart * TQ_ROW_META_BYTES,
                scratch, TQ_META_TILE_BYTES);
            copy_packed_ub_to_gm(packedGm, meta1Base + tileStart * TQ_ROW_META_BYTES,
                meta1Bytes, TQ_META_TILE_BYTES);
            rowOff += rows;
        }
    }

    __aicore__ inline void PrefetchStream(
        AscendC::GlobalTensor<half>& cWorkGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice,
        uint32_t inputBuffer) {
        const uint16_t flagBase = context_.ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset = context_.ManualBufferOffset(desc.streamOrdinal);

        TqCrossCoreWait<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_READY);
        CopyManualCToYBatch(cWorkGm, bufferOffset, aivSlice, inputBuffer);
        // Cross-core flag operations are not a completion fence for the GM->UB
        // transfer.  Wait on MTE2 from S before releasing the workspace buffer,
        // otherwise AIC may overwrite it while the copy is still in flight.
        // For steady-state streams this scalar wait overlaps the already-issued
        // vector encoding of the previous UB buffer.
        TqSyncMte2ToS();
        TqCrossCoreSet<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_FREE);
        // The dependency is inserted after the copy on MTE2 and after any
        // already-issued work on V.  It therefore protects the prefetched UB
        // slot without draining the current stream's vector pipeline.
        TqSyncMte2ToV();
    }

    __aicore__ inline void EncodeStream(
        const ManualKey1StreamDesc& desc,
        uint32_t inputBuffer) {
        auto yBatch = context_.resource_.YBatchHalf(inputBuffer);
        if (desc.isValue) {
            EncodeValueBatch(TQ_MANUAL_AIV_SLICE_M, yBatch);
        } else {
            EncodeKeyBatch(TQ_MANUAL_AIV_SLICE_M, yBatch);
        }
    }

    __aicore__ inline void WriteStream(
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        if (desc.isValue) {
            WriteEncodedSliceToCache<false>(
                context_.valueCacheGm_, desc, aivSlice);
        } else {
            WriteEncodedSliceToCache<true>(
                context_.keyCacheGm_, desc, aivSlice);
        }
    }

    __aicore__ inline void SubmitStream(
        AscendC::GlobalTensor<half>& cWorkGm,
        const ManualKey1StreamDesc& nextDesc,
        uint32_t aivSlice,
        bool& hasPending,
        ManualKey1StreamDesc& pendingDesc,
        uint32_t& pendingInputBuffer) {
        if (!hasPending) {
            pendingInputBuffer = 0;
            PrefetchStream(cWorkGm, nextDesc, aivSlice, pendingInputBuffer);
            pendingDesc = nextDesc;
            hasPending = true;
            return;
        }

        EncodeStream(pendingDesc, pendingInputBuffer);
        const uint32_t nextInputBuffer = pendingInputBuffer ^ 1u;
        PrefetchStream(cWorkGm, nextDesc, aivSlice, nextInputBuffer);
        WriteStream(pendingDesc, aivSlice);
        pendingDesc = nextDesc;
        pendingInputBuffer = nextInputBuffer;
    }

    __aicore__ inline void DrainStream(
        uint32_t aivSlice,
        bool hasPending,
        const ManualKey1StreamDesc& pendingDesc,
        uint32_t pendingInputBuffer) {
        if (!hasPending) {
            return;
        }
        EncodeStream(pendingDesc, pendingInputBuffer);
        WriteStream(pendingDesc, aivSlice);
    }

    BitResidualPackK8v4Context<T>& context_;
};

}  // namespace bit_residual
