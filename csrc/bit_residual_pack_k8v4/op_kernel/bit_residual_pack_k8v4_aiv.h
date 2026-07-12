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
        AscendC::GlobalTensor<float> normWorkGm;
        AscendC::GlobalTensor<T> cWorkGm;
        op.InitManualWorkspaceTensors(manualGroupId, normWorkGm, cWorkGm);

        for (uint32_t buffer = 0; buffer < TQ_MANUAL_WORKSPACE_BUFFER_COUNT; ++buffer) {
            TqCrossCoreSet<PIPE_MTE2>(
                static_cast<uint16_t>(buffer * TQ_MANUAL_SYNC_PP_STRIDE +
                                      TQ_MANUAL_SYNC_C_FREE));
        }

        uint32_t tileOrdinal = 0;
        uint32_t manualTileOrdinal = 0;
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
                const uint32_t valueGroupInBlock = blockOffset / TQ_VAL_GROUP_ROWS;
                for (uint32_t headTileStart = 0;
                     headTileStart < op.numHeads_;
                     headTileStart += TQ_MANUAL_HEADS_PER_TILE) {
                    if (tileOrdinal % op.dataCores_ == manualGroupId) {
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
                        FinishC(normWorkGm, cWorkGm, keyDesc, aivSlice);
                        FinishC(normWorkGm, cWorkGm, valueDesc, aivSlice);
                        ++manualTileOrdinal;
                    }
                    ++tileOrdinal;
                }
                rowOff += validRows;
            }
        }
    }

private:
    __aicore__ inline void LoadNormsAndNormalizeRotatedBatch(
        AscendC::GlobalTensor<float>& normWorkGm,
        uint32_t streamOrdinal,
        uint32_t aivSlice,
        AscendC::LocalTensor<T> norms) {
        const uint32_t normOffset =
            context_.ManualNormBufferOffset(streamOrdinal) +
            aivSlice * TQ_MANUAL_NORM_MATRIX_ELEMS;
        auto gram = context_.resource_.ReduceOut();
        auto yBatch = context_.resource_.YBatch();
        auto yFp32 = context_.resource_.YFp32();
        auto scaleBlock = context_.resource_.ReduceTmp();
        auto normAcc = context_.resource_.NormScalar();

        AscendC::DataCopy(gram, normWorkGm[normOffset], TQ_MANUAL_NORM_MATRIX_ELEMS);
        TqSyncMte2ToV();
        TqSyncMte2ToS();
        for (uint32_t row = 0; row < TQ_MANUAL_AIV_SLICE_M; ++row) {
            normAcc.SetValue(
                0, gram.GetValue(row * (TQ_MANUAL_NORM_MATRIX_DIM + 1)));
            AscendC::Maxs(normAcc, normAcc, 0.0f, 1);
            AscendC::Sqrt(normAcc, normAcc, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                norms[row * TQ_NORM_STRIDE],
                normAcc,
                AscendC::RoundMode::CAST_RINT,
                1);
            AscendC::Adds(normAcc, normAcc, TQ_NORM_EPS_F, 1);
            AscendC::Duplicate(scaleBlock, 1.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(normAcc, scaleBlock, normAcc, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(scaleBlock, normAcc, 1, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();

            const uint32_t rowOffset = row * TQ_PACK_D;
            AscendC::Cast(
                yFp32, yBatch[rowOffset], AscendC::RoundMode::CAST_NONE, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(
                yFp32,
                yFp32,
                scaleBlock,
                static_cast<uint64_t>(TQ_PACK_D / 2),
                2,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                yBatch[rowOffset], yFp32, AscendC::RoundMode::CAST_RINT, TQ_PACK_D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        TqSyncSToV();
    }

    __aicore__ inline void EncodeKeyBatchWithNorms(
        uint32_t m,
        AscendC::LocalTensor<T> norms) {
        auto yBatch = context_.resource_.YBatch();
        auto encodedBatch = context_.resource_.KeyEncodedBatch();
        auto yFp32 = context_.resource_.YFp32();
        auto signVal = context_.resource_.SignVal();
        auto err = context_.resource_.Err();
        auto qI16 = context_.resource_.QuantIndexU16();
        auto codeU16 = context_.resource_.PackMerge();
        auto codeMask = context_.resource_.PackMask();
        auto reduceScalar = context_.resource_.ReduceScalar();
        auto baseAcc = reduceScalar;
        auto maxAcc = reduceScalar[1];
        auto reduceTmp = context_.resource_.ReduceTmp();
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
        auto yBatch = context_.resource_.YBatch();
        auto encodedBatch = context_.resource_.ValEncodedBatch();
        auto yFp32 = context_.resource_.YFp32();
        auto qFp32 = context_.resource_.Err();
        auto qI32 = context_.resource_.QuantIndex();
        auto qI16 = context_.resource_.QuantIndexU16();
        auto reduceScalar = context_.resource_.ReduceScalar();
        auto vminAcc = reduceScalar;
        auto vmaxAcc = reduceScalar[1];
        auto reduceTmp = context_.resource_.ReduceTmp();

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
            auto normToFloat = context_.resource_.ReduceOut();  // float32 scratch
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

    __aicore__ inline void CopyManualCToYBatch(
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t bufferOffset,
        uint32_t aivSlice) {
        auto yBatch = context_.resource_.YBatch();
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
        auto shiftedIdx = context_.resource_.PackMerge();
        const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
        const uint32_t shiftBits = groupRow * (IS_KEY ? 8 : 4);

        if (clearOldBits) {
            auto clearMask = context_.resource_.PackMask();
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
        return ((uint64_t)blockIdx * context_.numHeads_ + headIdx) *
                   (context_.blockSize_ / GroupRows<IS_KEY>()) * GroupStride<IS_KEY>() +
               static_cast<uint64_t>(groupInBlock) * GroupStride<IS_KEY>();
    }

    template <bool IS_KEY>
    __aicore__ inline void EncodeSliceToCache(
        AscendC::GlobalTensor<uint8_t>& packedGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice,
        AscendC::LocalTensor<T> norms) {
        auto& op = context_;
        const uint32_t firstHead =
            desc.headTileStart + aivSlice * TQ_MANUAL_HEADS_PER_AIV;
        if constexpr (IS_KEY) {
            EncodeKeyBatchWithNorms(TQ_MANUAL_AIV_SLICE_M, norms);
        } else {
            EncodeValueBatch(TQ_MANUAL_AIV_SLICE_M, norms);
        }
        auto encodedBatch =
            IS_KEY ? op.resource_.KeyEncodedBatch() : op.resource_.ValEncodedBatch();
        auto packedGroups = op.resource_.PackedRow();
        const uint32_t groupRowsPerGroup = GroupRows<IS_KEY>();
        for (uint32_t localHead = 0; localHead < TQ_MANUAL_HEADS_PER_AIV;
             ++localHead) {
            const uint32_t headIdx = firstHead + localHead;
            uint32_t rowOff = 0;
            while (rowOff < desc.validRows) {
                const uint32_t slot = desc.slotStart + rowOff;
                const uint32_t blockIdx = slot / op.blockSize_;
                const uint32_t blockOffset = slot - blockIdx * op.blockSize_;
                const uint32_t groupInBlock = blockOffset / groupRowsPerGroup;
                const uint32_t firstGroupRow = blockOffset % groupRowsPerGroup;
                uint32_t rowsThisGroup = groupRowsPerGroup - firstGroupRow;
                if (rowsThisGroup > desc.validRows - rowOff) {
                    rowsThisGroup = desc.validRows - rowOff;
                }
                const uint64_t groupBase = MakeCacheGroupBaseOffset<IS_KEY>(
                    blockIdx, groupInBlock, headIdx);
                const bool preserveExisting =
                    firstGroupRow != 0 || rowsThisGroup < groupRowsPerGroup;

                auto packedGroup = packedGroups[localHead * TQ_PACKED_GROUP_STRIDE];
                if (preserveExisting) {
                    copy_packed_gm_to_ub(
                        packedGroup, packedGm, groupBase, GroupBytes<IS_KEY>());
                } else {
                    ClearPackedGroup(packedGroup);
                }
                for (uint32_t r = 0; r < rowsThisGroup; ++r) {
                    const uint32_t manualGroupRow = desc.startGroupRow + rowOff + r;
                    const uint32_t encodedRow =
                        localHead * TQ_MANUAL_GROUP_ROWS + manualGroupRow;
                    MergeEncodedRowToGroup<IS_KEY>(
                        packedGroup, encodedBatch, encodedRow, firstGroupRow + r,
                        preserveExisting);
                }
                copy_packed_ub_to_gm(
                    packedGm, groupBase, packedGroup, GroupBytes<IS_KEY>());
                rowOff += rowsThisGroup;
            }
        }
    }

    __aicore__ inline void FinishC(
        AscendC::GlobalTensor<float>& normWorkGm,
        AscendC::GlobalTensor<T>& cWorkGm,
        const ManualKey1StreamDesc& desc,
        uint32_t aivSlice) {
        const uint16_t flagBase = context_.ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset = context_.ManualBufferOffset(desc.streamOrdinal);
        auto norms = context_.ManualNorms(desc.streamOrdinal);

        TqCrossCoreWait<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_READY);
        CopyManualCToYBatch(cWorkGm, bufferOffset, aivSlice);
        LoadNormsAndNormalizeRotatedBatch(
            normWorkGm, desc.streamOrdinal, aivSlice, norms);
        TqCrossCoreSet<PIPE_MTE2>(flagBase + TQ_MANUAL_SYNC_C_FREE);
        if (desc.isValue) {
            EncodeSliceToCache<false>(
                context_.valueCacheGm_, desc, aivSlice, norms);
        } else {
            EncodeSliceToCache<true>(
                context_.keyCacheGm_, desc, aivSlice, norms);
        }
    }

    BitResidualPackK8v4Context<T>& context_;
};

}  // namespace bit_residual
