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
        AscendC::GlobalTensor<float> cWorkGm;
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
                        FinishC(cWorkGm, keyDesc, aivSlice);
                        FinishC(cWorkGm, valueDesc, aivSlice);
                        ++manualTileOrdinal;
                    }
                }
                rowOff += validRows;
            }
        }
    }

private:
    // ── Sign-reversal quantization for key rows ───────────────────────────
    // After rotation y = x @ R^T:
    //   sig_vec[d] = sign(y[d]) ? +1 : -1  (positive→+1, negative→-1)
    //   rev_vec = y * sig_vec  (all dims become positive)
    //   base/step from min/max of rev_vec
    //   q7 = (rev_vec - base) * invStep, clamped [0, 127]
    //   code = (q7 << 1) | sign_bit
    //
    __aicore__ inline void EncodeKeyBatch(uint32_t m) {
        auto yBatch = context_.resource_.YBatchFloat();
        auto encodedBatch = context_.resource_.KeyEncodedBatch();
        auto yFp32 = context_.resource_.YFp32();
        auto signVal = context_.resource_.SignVal();
        auto revVec = context_.resource_.RevVec();
        auto signI32 = context_.resource_.QuantIndex();
        auto qI16 = context_.resource_.QuantIndexU16();
        auto codeU16 = context_.resource_.PackMerge();
        auto codeMask = context_.resource_.PackMask();
        auto signSave = context_.resource_.SignMask().template ReinterpretCast<uint16_t>();
        auto reduceScalar = context_.resource_.ReduceScalar();
        auto baseAcc = reduceScalar;
        auto maxAcc = reduceScalar[1];
        auto reduceTmp = context_.resource_.ReduceTmp();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_KEY_ENCODED_ROW_STRIDE_WORDS;

            // Step 1: Extract sign bits from rotated y
            auto signBits = qI16.template ReinterpretCast<uint16_t>();
            auto signMaskI32 = yFp32.template ReinterpretCast<int32_t>();
            AscendC::Duplicate(signMaskI32, static_cast<int32_t>(1), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(
                signI32,
                yBatch[yOff].template ReinterpretCast<int32_t>(),
                static_cast<int32_t>(31),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(signI32, signI32, signMaskI32, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, signI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            // Step 2: Generate sig_vec: 0→+1.0, 1→-1.0
            // sign_bit=0 (positive) → +1.0, sign_bit=1 (negative) → -1.0
            AscendC::Cast(signVal, signI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(signVal, signVal, -2.0f, TQ_PACK_D);  // 0→0, 1→-2
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(signVal, signVal, 1.0f, TQ_PACK_D);   // 0→+1, 1→-1 = sig_vec
            AscendC::PipeBarrier<PIPE_V>();

            // Step 3: rotation output is already FP32.
            AscendC::Adds(yFp32, yBatch[yOff], 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            // Step 4: rev_vec = yFp32 * sig_vec (all dims become positive)
            AscendC::Mul(revVec, yFp32, signVal, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            // Step 5: base/step from rev_vec (all positive → min ≥ 0)
            AscendC::ReduceMin<float>(baseAcc, revVec, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(maxAcc, revVec, reduceTmp, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            TqSyncVToS();
            const float baseF = baseAcc.GetValue(0);
            const float maxF = maxAcc.GetValue(0);
            const float rangeF = maxF - baseF;
            const float stepF = rangeF > 0.0f ? rangeF / TQ_KEY_QUANT_LEVELS_F : 1.0f;
            const float invStepF = rangeF > 0.0f ? 1.0f / stepF : 0.0f;
            TqSyncSToV();

            // Step 6: Quantize rev_vec: q7 = (rev_vec - baseF) * invStepF, clamped [0,127]
            AscendC::Adds(revVec, revVec, -baseF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(revVec, revVec, invStepF, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(revVec, revVec, 0.0f, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(revVec, revVec, TQ_KEY_QUANT_LEVELS_F, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signI32, revVec, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qI16, signI32, AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            // Step 7: Pack code = (q7 << 1) | sign_bit
            // qI16 has quantized index (0..127)
            // Re-extract sign bits here to avoid an asynchronous UB copy.
            AscendC::Duplicate(signMaskI32, static_cast<int32_t>(1), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(
                signI32,
                yBatch[yOff].template ReinterpretCast<int32_t>(),
                static_cast<int32_t>(31),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(signI32, signI32, signMaskI32, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(signSave.template ReinterpretCast<int16_t>(), signI32,
                AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(codeMask, static_cast<uint16_t>(1), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(
                codeMask,
                qI16.template ReinterpretCast<uint16_t>(),
                static_cast<uint16_t>(1),
                TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(codeU16, signSave, codeMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(encodedBatch[encodedOff], codeU16, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();

            // Step 8: Store base and step at new offsets (no norm word)
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

    // ── Value quantization (no norm folding) ──────────────────────────────
    // vmin/vstep stored as raw values (not multiplied by norm).
    // Decode: y = vmin + idx4 * vstep
    //
    __aicore__ inline void EncodeValueBatch(uint32_t m) {
        auto yBatch = context_.resource_.YBatchFloat();
        auto encodedBatch = context_.resource_.ValEncodedBatch();
        auto yFp32 = context_.resource_.YFp32();
        auto qFp32 = context_.resource_.RevVec();  // reuse RevVec buffer for value quant
        auto qI32 = context_.resource_.QuantIndex();
        auto qI16 = context_.resource_.QuantIndexU16();
        auto reduceScalar = context_.resource_.ReduceScalar();
        auto vminAcc = reduceScalar;
        auto vmaxAcc = reduceScalar[1];
        auto reduceTmp = context_.resource_.ReduceTmp();

        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t yOff = i * TQ_ROT_N;
            const uint32_t encodedOff = i * TQ_VAL_ENCODED_ROW_STRIDE_WORDS;

            AscendC::Adds(yFp32, yBatch[yOff], 0.0f, TQ_PACK_D);
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

            // Quantize: idx4 = (yFp32 - vminF) * invStepF, clamped [0, 15]
            // Store raw vmin/vstep (no norm folding)
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
            vminOut.SetValue(0, vminF);
            vstepOut.SetValue(0, vstepF);
        }
        TqSyncSToV();
    }

    __aicore__ inline void CopyManualCToYBatch(
        AscendC::GlobalTensor<float>& cWorkGm,
        uint32_t bufferOffset,
        uint32_t aivSlice) {
        auto yBatch = context_.resource_.YBatchFloat();
        static constexpr uint32_t SLICE_ELEMS =
            TQ_MANUAL_AIV_SLICE_M * TQ_ROT_N;  // 16×128 = 2048
        const uint32_t sliceElemOffset = aivSlice * SLICE_ELEMS;
        AscendC::DataCopy(
            yBatch,
            cWorkGm[bufferOffset + sliceElemOffset],
            SLICE_ELEMS);
        TqSyncMte2ToV();
        TqSyncMte2ToS();
    }

    template <bool IS_KEY>
    __aicore__ inline constexpr uint32_t EncodedRowStrideWords() const {
        return IS_KEY ? TQ_KEY_ENCODED_ROW_STRIDE_WORDS : TQ_VAL_ENCODED_ROW_STRIDE_WORDS;
    }

    // ── 16-row sub-block layout helpers ───────────────────────────────────
    // Key block: [8 packed 2-row code groups] [16 bases] [16 steps]
    // Val block: [4 packed 4-row code groups] [16 vmins] [16 vsteps]

    template <bool IS_KEY>
    __aicore__ inline uint32_t HeadStride() const {
        return context_.blockSize_ *
            (IS_KEY ? TQ_KEY_BYTES_PER_ROW : TQ_VAL_BYTES_PER_ROW);
    }

    // ── Merge one encoded row into a 16-row sub-block ─────────────────────
    // Key pairs place their sign bits in the middle of the uint16 word:
    // first=(sign<<7)|q7 in the low byte, second=(q7<<1)|sign in the high byte.
    //
    #if 0
    template <bool IS_KEY>
    __aicore__ inline void MergeEncodedRowToBlock(
        AscendC::LocalTensor<uint8_t>& packedBlock,
        const AscendC::LocalTensor<uint16_t>& encodedBatch,
        uint32_t encodedRow,
        uint32_t rowInBlock) {
        const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
        const uint32_t groupRows = IS_KEY ? TQ_KEY_GROUP_ROWS : TQ_VAL_GROUP_ROWS;
        const uint32_t groupInBlock = rowInBlock / groupRows;
        const uint32_t groupRow = rowInBlock % groupRows;
        const uint32_t codeWordOff =
            groupInBlock * TQ_GROUP_INDEX_BYTES / sizeof(uint16_t);
        auto packedU16 = packedBlock.template ReinterpretCast<uint16_t>();
        if constexpr (IS_KEY) {
            for (uint32_t d = 0; d < TQ_PACK_D; ++d) {
                const uint16_t genericCode = encodedBatch.GetValue(encodedOff + d);
                const uint16_t oldWord = packedU16.GetValue(codeWordOff + d);
                if (groupRow == 0) {
                    const uint16_t q7 = genericCode >> 1;
                    const uint16_t sign = genericCode & 1;
                    const uint16_t firstCode = (sign << 7) | q7;
                    packedU16.SetValue(
                        codeWordOff + d,
                        static_cast<uint16_t>((oldWord & 0xff00u) |
                                              firstCode));
                } else {
                    packedU16.SetValue(
                        codeWordOff + d,
                        static_cast<uint16_t>((oldWord & 0x00ffu) |
                                              (genericCode << 8)));
                }
            }
        } else {
            const uint32_t shiftBits = groupRow * 4;
            auto shifted = context_.resource_.PackMerge();
            auto clearMask = context_.resource_.PackMask();
            const uint16_t rowMask = static_cast<uint16_t>(0x000fu << shiftBits);
            AscendC::Duplicate(clearMask, static_cast<uint16_t>(~rowMask), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(
                packedU16[codeWordOff], packedU16[codeWordOff], clearMask, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(
                shifted, encodedBatch[encodedOff],
                static_cast<uint16_t>(shiftBits), TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(
                packedU16[codeWordOff], packedU16[codeWordOff], shifted, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // ── Scalar zone: write base/step/vmin/vstep per row ───────────────
        if constexpr (IS_KEY) {
            auto packedBase = packedBlock[KeyBaseOffset(rowInBlock)].template ReinterpretCast<float>();
            auto packedStep = packedBlock[KeyStepOffset(rowInBlock)].template ReinterpretCast<float>();
            auto encodedBase = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_BASE_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            auto encodedStep = encodedBatch[
                encodedOff + TQ_KEY_ENCODED_STEP_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            packedBase.SetValue(0, encodedBase.GetValue(0));
            packedStep.SetValue(0, encodedStep.GetValue(0));
        } else {
            auto packedVmin = packedBlock[ValVminOffset(rowInBlock)].template ReinterpretCast<float>();
            auto packedVstep = packedBlock[ValVstepOffset(rowInBlock)].template ReinterpretCast<float>();
            auto encodedVmin = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VMIN_BYTE_OFFSET / sizeof(uint16_t)]
                                   .template ReinterpretCast<float>();
            auto encodedVstep = encodedBatch[
                encodedOff + TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET / sizeof(uint16_t)]
                                    .template ReinterpretCast<float>();
            packedVmin.SetValue(0, encodedVmin.GetValue(0));
            packedVstep.SetValue(0, encodedVstep.GetValue(0));
        }
    }

    // ── Cache block offset: blockIdx × numHeads × subBlocks × stride ──────
    template <bool IS_KEY>
    __aicore__ inline uint64_t MakeCacheBlockBaseOffset(
        uint32_t blockIdx,
        uint32_t subBlockInBlock,
        uint32_t headIdx) const {
        const uint32_t subBlocksPerBlock = context_.blockSize_ / TQ_BLOCK_ROWS;
        return ((uint64_t)blockIdx * context_.numHeads_ + headIdx) *
                   subBlocksPerBlock * BlockStride<IS_KEY>() +
               static_cast<uint64_t>(subBlockInBlock) * BlockStride<IS_KEY>();
    }

    template <bool IS_KEY>
    __aicore__ inline void EncodeSliceToCache(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        auto& op = context_;
        const uint32_t firstHead =
            desc.headTileStart + aivSlice * TQ_MANUAL_HEADS_PER_AIV;
        if constexpr (IS_KEY) {
            EncodeKeyBatch(TQ_MANUAL_AIV_SLICE_M);
        } else {
            EncodeValueBatch(TQ_MANUAL_AIV_SLICE_M);
        }
        TqSyncVToS();
        auto encodedBatch =
            IS_KEY ? op.resource_.KeyEncodedBatch() : op.resource_.ValEncodedBatch();
        auto packedBlocks = op.resource_.PackedRow();

        for (uint32_t localHead = 0; localHead < TQ_MANUAL_HEADS_PER_AIV;
             ++localHead) {
            const uint32_t headIdx = firstHead + localHead;
            uint32_t rowOff = 0;
            while (rowOff < desc.validRows) {
                const uint32_t slot = desc.slotStart + rowOff;
                const uint32_t blockIdx = slot / op.blockSize_;
                const uint32_t blockOffset = slot - blockIdx * op.blockSize_;
                const uint32_t subBlockInBlock = blockOffset / TQ_BLOCK_ROWS;
                const uint32_t rowInSubBlock = blockOffset % TQ_BLOCK_ROWS;
                uint32_t rowsThisSubBlock = TQ_BLOCK_ROWS - rowInSubBlock;
                if (rowsThisSubBlock > desc.validRows - rowOff) {
                    rowsThisSubBlock = desc.validRows - rowOff;
                }
                // Maximum rows we can process in this sub-block
                uint32_t rowsThisRound = rowsThisSubBlock;

                const uint64_t subBlockBase = MakeCacheBlockBaseOffset<IS_KEY>(
                    blockIdx, subBlockInBlock, headIdx);
                const bool preserveExisting =
                    rowInSubBlock != 0 || rowsThisRound < TQ_BLOCK_ROWS;

                auto packedBlock = packedBlocks[localHead * TQ_PACKED_BLOCK_STRIDE];
                if (preserveExisting) {
                    copy_packed_gm_to_ub(
                        packedBlock, packedGm, subBlockBase, BlockTotalBytes<IS_KEY>());
                } else {
                    ClearPackedBlock(packedBlock);
                }
                for (uint32_t r = 0; r < rowsThisRound; ++r) {
                    const uint32_t manualGroupRow = desc.startGroupRow + rowOff + r;
                    const uint32_t encodedRow =
                        localHead * TQ_MANUAL_GROUP_ROWS + manualGroupRow;
                    const uint32_t currentRowInSubBlock = rowInSubBlock + r;
                    MergeEncodedRowToBlock<IS_KEY>(
                        packedBlock, encodedBatch, encodedRow, currentRowInSubBlock);
                }
                copy_packed_ub_to_gm(
                    packedGm, subBlockBase, packedBlock, BlockTotalBytes<IS_KEY>());
                rowOff += rowsThisRound;
            }
        }
    }

    #endif

    template <bool IS_KEY>
    __aicore__ inline void EncodeSliceToCache(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        auto& op = context_;
        const uint32_t headIdx = desc.headTileStart + aivSlice;
        if (headIdx >= op.numHeads_) {
            return;
        }
        if constexpr (IS_KEY) {
            EncodeKeyBatch(TQ_MANUAL_AIV_SLICE_M);
        } else {
            EncodeValueBatch(TQ_MANUAL_AIV_SLICE_M);
        }
        TqSyncVToS();
        auto encoded = IS_KEY ? op.resource_.KeyEncodedBatch() : op.resource_.ValEncodedBatch();
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
            const uint64_t meta1Base = meta0Base + op.blockSize_ * sizeof(float);
            auto meta0 = scratch.template ReinterpretCast<float>();
            auto meta1 = scratch[TQ_META_TILE_BYTES].template ReinterpretCast<float>();
            copy_packed_gm_to_ub(scratch, packedGm,
                meta0Base + tileStart * sizeof(float), TQ_META_TILE_BYTES);
            auto meta1Bytes = scratch[TQ_META_TILE_BYTES];
            auto codeBytesUb = scratch[2 * TQ_META_TILE_BYTES];
            copy_packed_gm_to_ub(meta1Bytes, packedGm,
                meta1Base + tileStart * sizeof(float), TQ_META_TILE_BYTES);
            TqSyncMte2ToS();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t encodedRow = desc.startGroupRow + rowOff + r;
                const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
                auto codeOut = scratch[2 * TQ_META_TILE_BYTES + r * codeBytes];
                if constexpr (IS_KEY) {
                    for (uint32_t d = 0; d < TQ_PACK_D; ++d) {
                        codeOut.SetValue(d, static_cast<uint8_t>(encoded.GetValue(encodedOff + d)));
                    }
                    meta0.SetValue(tileRow + r, encoded[
                        encodedOff + TQ_KEY_ENCODED_BASE_BYTE_OFFSET / sizeof(uint16_t)]
                            .template ReinterpretCast<float>().GetValue(0));
                    meta1.SetValue(tileRow + r, encoded[
                        encodedOff + TQ_KEY_ENCODED_STEP_BYTE_OFFSET / sizeof(uint16_t)]
                            .template ReinterpretCast<float>().GetValue(0));
                } else {
                    for (uint32_t d = 0; d < TQ_PACK_D / 2; ++d) {
                        const uint8_t lo = static_cast<uint8_t>(encoded.GetValue(encodedOff + 2 * d));
                        const uint8_t hi = static_cast<uint8_t>(encoded.GetValue(encodedOff + 2 * d + 1));
                        codeOut.SetValue(d, static_cast<uint8_t>(lo | (hi << 4)));
                    }
                    meta0.SetValue(tileRow + r, encoded[
                        encodedOff + TQ_VAL_ENCODED_VMIN_BYTE_OFFSET / sizeof(uint16_t)]
                            .template ReinterpretCast<float>().GetValue(0));
                    meta1.SetValue(tileRow + r, encoded[
                        encodedOff + TQ_VAL_ENCODED_VSTEP_BYTE_OFFSET / sizeof(uint16_t)]
                            .template ReinterpretCast<float>().GetValue(0));
                }
            }
            TqSyncSToMte3();
            copy_packed_ub_to_gm(packedGm,
                headBase + static_cast<uint64_t>(blockRow) * codeBytes,
                codeBytesUb, rows * codeBytes);
            copy_packed_ub_to_gm(packedGm, meta0Base + tileStart * sizeof(float),
                scratch, TQ_META_TILE_BYTES);
            copy_packed_ub_to_gm(packedGm, meta1Base + tileStart * sizeof(float),
                meta1Bytes, TQ_META_TILE_BYTES);
            rowOff += rows;
        }
    }

    __aicore__ inline void FinishC(
        AscendC::GlobalTensor<float>& cWorkGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        const uint16_t flagBase = context_.ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset = context_.ManualBufferOffset(desc.streamOrdinal);

        TqCrossCoreWait<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_READY);
        CopyManualCToYBatch(cWorkGm, bufferOffset, aivSlice);
        TqCrossCoreSet<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_FREE);
        if (desc.isValue) {
            EncodeSliceToCache<false>(context_.valueCacheGm_, desc, aivSlice);
        } else {
            EncodeSliceToCache<true>(context_.keyCacheGm_, desc, aivSlice);
        }
    }

    BitResidualPackK8v4Context<T>& context_;
};

}  // namespace bit_residual
