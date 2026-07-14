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
    
        buf0=yBatch
        原始数据 row_vec = buf0
        计算绝对值
        abs_vec = buf2 = abs(row_vec)
        计算16个最大值
        for (i:m) {
        max_vec = ResuceMax(abs_vec, buf3, call_index=false)
        }
        把每个最大值填充乘一个Block
        max_vec_blk = brcb(max_vec)
        把abs_vec缩小max倍，降低取值范围到[0,1]
        buf3 = div(abs_vec, max_vec_blk)
        // 降低精度到16位
        norm_vec_16 = buf4 = Cast(buf3)
        // 提取首位符号位
        buf2 = ShiftRight(row_vec, 31)
        转化为u16
        bit_vec_16=Cast(buf2)
        // 计算16个最小值
        for (i:m) {
        base[i] = ResuceMax(norm_vec_16, , call_index=false)
        }
        max = {1,1,...}
        gap = max - base
        step = Divs(gap, 127)
        // 每个base，step填充一个block
        base_blk = brcb(base, 16)
        step_blk = brcb(step，16)
        对norm_vec_16进行量化 (x - base)/step
        buf = sub(norm_vec_16, base_blk, repeatParams)
        quant_16 = div(buf, step_blk, repeatParams)
        quant_u16 = cast(quant_16)
        buf = ShiftLeft(bit_vec_16, 7)
        final_quant_16= or(buf, quant_u16)


        base *= max_vec
        step *= max_vec

    */
    __aicore__ inline void EncodeKeyBatch(
        uint32_t m,
        AscendC::LocalTensor<float>& yBatch) {
        using ComputeT = half;
        static_assert(TQ_MANUAL_AIV_SLICE_M == TQ_VECTOR_BATCH,
                      "key encoder expects one physical 16-row tile");
        constexpr uint32_t typePerBlock = 32 / sizeof(ComputeT);
        constexpr float safeFp32Divisor = 1.0e-12f;
        const ComputeT quantStep =
            static_cast<ComputeT>(1.0f / TQ_KEY_QUANT_LEVELS_F);
        const ComputeT safeTDivisor = static_cast<ComputeT>(1.0e-6f);
        const uint32_t totalElems = m * TQ_PACK_D;
        constexpr uint32_t codeBatchElems =
            (TQ_VECTOR_BATCH / 2) * TQ_PACK_D;
        constexpr uint32_t binaryRepeatBatchF16 = 128;
        constexpr uint32_t binaryRepeatBatchF32 = 64;

        auto encodedBatch = context_.resource_.KeyEncodedBatch();
        auto buf1 = context_.resource_.EncodeBuffer1().template ReinterpretCast<float>();
        auto buf2 = context_.resource_.EncodeBuffer2().template ReinterpretCast<float>();
        auto buf3 = context_.resource_.EncodeBuffer3().template ReinterpretCast<float>();
        auto buf4 = context_.resource_.EncodeBuffer4().template ReinterpretCast<ComputeT>();
        auto buf5 = context_.resource_.EncodeBuffer5().template ReinterpretCast<int16_t>();
        auto buf6 = context_.resource_.EncodeBuffer6().template ReinterpretCast<ComputeT>();
        auto buf7 = context_.resource_.EncodeBuffer7().template ReinterpretCast<ComputeT>();
        auto buf8 = context_.resource_.EncodeBuffer8().template ReinterpretCast<float>();

        // Normalize all 16 FP32 rows together.  ReduceMax still emits one
        // scalar per row, then Brcb turns the 16 scalars into row divisors.
        auto absVec = buf1;
        auto normFp32 = buf2;
        auto maxFold = buf3;
        auto maxVec = buf8;
        AscendC::Abs(absVec, yBatch, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        // FP32 WholeReduce handles 64 elements per repeat.  Fold each row's
        // two halves first, then reduce all 16 rows in one repeat sequence.
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_ROT_N;
            AscendC::Max(maxFold[i * TQ_ROT_N / 2], absVec[rowOff],
                         absVec[rowOff + TQ_ROT_N / 2], TQ_ROT_N / 2);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<float, false>(
            maxVec, maxFold, static_cast<int32_t>(TQ_ROT_N / 2), m, 1, 1,
            8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        auto maxVecT = buf6;
        auto baseVec = buf6[m];
        auto stepVec = buf6[2 * m];
        auto gapVec = buf6[3 * m];
        auto safeStepVec = buf6[4 * m];
        auto oneVec = buf6[5 * m];
        AscendC::Cast(maxVecT, maxVec, AscendC::RoundMode::CAST_RINT, m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(maxVec, maxVec, safeFp32Divisor, m);
        AscendC::PipeBarrier<PIPE_V>();

        auto maxVecBlk = buf3;
        AscendC::Brcb(maxVecBlk, maxVec, 2, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            AscendC::Div<float, false>(
                normFp32[rowOff], absVec[rowOff], maxVecBlk[i * 8],
                binaryRepeatBatchF32, 2,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto normVec16 = buf4;
        AscendC::Cast(normVec16, normFp32, AscendC::RoundMode::CAST_RINT,
                      totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        // Extract all sign bits once; buf1 is free after the normalized cast.
        auto signVec = buf1.template ReinterpretCast<uint32_t>();
        auto signMask = buf3.template ReinterpretCast<uint32_t>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(signVec, yBits, static_cast<int32_t>(31), totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buf5, signVec, AscendC::RoundMode::CAST_NONE, totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        // One T-domain minimum per row.  max(normVec16) is exactly 1 for every
        // nonzero row, so step=(1-base)/127; a safe divisor handles equal rows.
        AscendC::WholeReduceMin<ComputeT, false>(
            baseVec, normVec16, static_cast<int32_t>(TQ_ROT_N), m, 1, 1,
            TQ_ROT_N / typePerBlock,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(oneVec, static_cast<ComputeT>(1.0f), m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(gapVec, oneVec, baseVec, m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(stepVec, gapVec, quantStep, m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(safeStepVec, stepVec, static_cast<ComputeT>(0.0f), m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(safeStepVec, safeStepVec, safeTDivisor, m);
        AscendC::PipeBarrier<PIPE_V>();

        // Brcb produces one 32-byte scalar block per row.  Each row call keeps
        // src1RepStrideIn=0 so its repeat(s) reuse that same scalar block.
        auto baseBlk = buf7;
        auto stepBlk = buf7[m * typePerBlock];
        AscendC::Brcb(baseBlk, baseVec, 2, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(stepBlk, safeStepVec, 2,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            const uint32_t blockOff = i * typePerBlock;
            AscendC::Sub<ComputeT, false>(
                normVec16[rowOff], normVec16[rowOff], baseBlk[blockOff],
                binaryRepeatBatchF16, 1,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            const uint32_t blockOff = i * typePerBlock;
            AscendC::Div<ComputeT, false>(
                normVec16[rowOff], normVec16[rowOff], stepBlk[blockOff],
                binaryRepeatBatchF16, 1,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t batch = 0; batch < 2; ++batch) {
            const uint32_t batchOff = batch * codeBatchElems;
            AscendC::Maxs(normVec16[batchOff], normVec16[batchOff],
                          static_cast<ComputeT>(0.0f), codeBatchElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(normVec16[batchOff], normVec16[batchOff],
                          static_cast<ComputeT>(TQ_KEY_QUANT_LEVELS_F),
                          codeBatchElems);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // C220 does not provide the required T->int16 conversion directly.
        // Convert through FP32/int32 before vectorized code assembly.
        auto quantI32 = buf1.template ReinterpretCast<int32_t>();
        auto quantI16 = buf4.template ReinterpretCast<int16_t>();
        for (uint32_t batch = 0; batch < 2; ++batch) {
            const uint32_t batchOff = batch * codeBatchElems;
            AscendC::Cast(normFp32[batchOff], normVec16[batchOff],
                          AscendC::RoundMode::CAST_NONE, codeBatchElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(quantI32[batchOff], normFp32[batchOff],
                          AscendC::RoundMode::CAST_RINT, codeBatchElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(quantI16[batchOff], quantI32[batchOff],
                          AscendC::RoundMode::CAST_NONE, codeBatchElems);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(baseVec, baseVec, maxVecT, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(stepVec, stepVec, maxVecT, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        auto metadataBaseT = encodedBatch[
            TQ_KEY_ENCODED_BASE_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        auto metadataStepT = encodedBatch[
            TQ_KEY_ENCODED_STEP_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            auto metadataBaseFp32 = buf2;
            auto metadataStepFp32 = buf2[typePerBlock];
            AscendC::Cast(metadataBaseFp32, baseVec,
                          AscendC::RoundMode::CAST_NONE, m);
            AscendC::Cast(metadataStepFp32, stepVec,
                          AscendC::RoundMode::CAST_NONE, m);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(metadataBaseT, metadataBaseFp32,
                          AscendC::RoundMode::CAST_RINT, m);
            AscendC::Cast(metadataStepT, metadataStepFp32,
                          AscendC::RoundMode::CAST_RINT, m);
        } else {
            AscendC::Adds(metadataBaseT, baseVec, static_cast<half>(0.0f), m);
            AscendC::Adds(metadataStepT, stepVec, static_cast<half>(0.0f), m);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // First assemble the key's 128 q7/sign bytes as uint16 elements.  Pack
        // adjacent dimensions into each uint16 so its byte view is exactly
        // dim0, dim1, dim2, ... and can be copied directly to cache.
        auto signVecU16 = buf5.template ReinterpretCast<uint16_t>();
        auto quantU16 = quantI16.template ReinterpretCast<uint16_t>();
        auto shiftedSign = context_.resource_.EncodeBuffer6()
                               .template ReinterpretCast<uint16_t>();
        auto finalCodes = context_.resource_.EncodeBuffer8()
                              .template ReinterpretCast<uint16_t>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            AscendC::ShiftLeft(shiftedSign[rowOff], signVecU16[rowOff],
                               static_cast<uint16_t>(7), TQ_PACK_D);
            AscendC::Or(finalCodes[rowOff], quantU16[rowOff],
                        shiftedSign[rowOff], TQ_PACK_D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 0;
        uint64_t gatheredCount = 0;
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            const uint32_t encodedOff = i * TQ_KEY_ENCODED_ROW_STRIDE_WORDS;
            // Built-in patterns 1 and 2 select even and odd elements.
            AscendC::GatherMask(encodedBatch[encodedOff], finalCodes[rowOff],
                                static_cast<uint8_t>(1), false,
                                static_cast<uint32_t>(0), gatherParams,
                                gatheredCount);
            AscendC::GatherMask(shiftedSign[encodedOff], finalCodes[rowOff],
                                static_cast<uint8_t>(2), false,
                                static_cast<uint32_t>(0), gatherParams,
                                gatheredCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(shiftedSign, shiftedSign,
                           static_cast<uint16_t>(8),
                           m * TQ_KEY_ENCODED_ROW_STRIDE_WORDS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(encodedBatch, encodedBatch, shiftedSign,
                    m * TQ_KEY_ENCODED_ROW_STRIDE_WORDS);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // ── Value quantization ────────────────────────────────────────────────
    // Process all 16 rows as one vector batch.  Values are normalized through
    // FP16 before 4-bit min/max quantization; vmin/vstep are scaled back to the
    // original domain and stored in the input dtype.
    __aicore__ inline void EncodeValueBatch(
        uint32_t m,
        AscendC::LocalTensor<float>& yBatch) {
        using ComputeT = half;
        static_assert(TQ_MANUAL_AIV_SLICE_M == TQ_VECTOR_BATCH,
                      "value encoder expects one physical 16-row tile");
        constexpr uint32_t typePerBlock = 32 / sizeof(ComputeT);
        constexpr float safeFp32Divisor = 1.0e-12f;
        const ComputeT quantStep =
            static_cast<ComputeT>(1.0f / TQ_VAL_QUANT_LEVELS_F);
        const ComputeT safeTDivisor = static_cast<ComputeT>(1.0e-6f);
        const uint32_t totalElems = m * TQ_PACK_D;
        constexpr uint32_t binaryRepeatBatchF16 = 128;
        constexpr uint32_t binaryRepeatBatchF32 = 64;

        auto encodedBatch = context_.resource_.KeyEncodedBatch();
        auto buf1 = context_.resource_.EncodeBuffer1().template ReinterpretCast<float>();
        auto buf2 = context_.resource_.EncodeBuffer2().template ReinterpretCast<float>();
        auto buf3 = context_.resource_.EncodeBuffer3().template ReinterpretCast<float>();
        auto buf4 = context_.resource_.EncodeBuffer4().template ReinterpretCast<ComputeT>();
        auto buf6 = context_.resource_.EncodeBuffer6().template ReinterpretCast<ComputeT>();
        auto buf7 = context_.resource_.EncodeBuffer7().template ReinterpretCast<ComputeT>();
        auto buf8 = context_.resource_.EncodeBuffer8().template ReinterpretCast<float>();

        auto absVec = buf1;
        auto normFp32 = buf2;
        auto maxFold = buf3;
        auto maxVec = buf8;
        AscendC::Abs(absVec, yBatch, totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_ROT_N;
            AscendC::Max(maxFold[i * TQ_ROT_N / 2], absVec[rowOff],
                         absVec[rowOff + TQ_ROT_N / 2], TQ_ROT_N / 2);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<float, false>(
            maxVec, maxFold, static_cast<int32_t>(TQ_ROT_N / 2), m, 1, 1,
            TQ_ROT_N / 16, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        auto maxVecT = buf6;
        auto minVec = buf6[m];
        auto maxNormVec = buf6[2 * m];
        auto rangeVec = buf6[3 * m];
        auto stepVec = buf6[4 * m];
        auto safeStepVec = buf6[5 * m];
        AscendC::Cast(maxVecT, maxVec, AscendC::RoundMode::CAST_RINT, m);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(maxVec, maxVec, safeFp32Divisor, m);
        AscendC::PipeBarrier<PIPE_V>();

        auto maxVecBlk = buf3;
        AscendC::Brcb(maxVecBlk.template ReinterpretCast<uint32_t>(),
                      maxVec.template ReinterpretCast<uint32_t>(), 2,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            AscendC::Div<float, false>(
                normFp32[rowOff], yBatch[rowOff], maxVecBlk[i * 8],
                binaryRepeatBatchF32, 2,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto normVec16 = buf4;
        AscendC::Cast(normVec16, normFp32, AscendC::RoundMode::CAST_RINT,
                      totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceMin<ComputeT, false>(
            minVec, normVec16, static_cast<int32_t>(TQ_ROT_N), m, 1, 1,
            TQ_ROT_N / typePerBlock,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<ComputeT, false>(
            maxNormVec, normVec16, static_cast<int32_t>(TQ_ROT_N), m, 1, 1,
            TQ_ROT_N / typePerBlock,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(rangeVec, maxNormVec, minVec, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(stepVec, rangeVec, quantStep, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(safeStepVec, stepVec, static_cast<ComputeT>(0.0f),
                      typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(safeStepVec, safeStepVec, safeTDivisor, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        auto minBlk = buf7;
        auto stepBlk = buf7[16 * typePerBlock];
        AscendC::Brcb(minBlk, minVec, 2, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(stepBlk, safeStepVec, 2,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            const uint32_t blockOff = i * typePerBlock;
            AscendC::Sub<ComputeT, false>(
                normVec16[rowOff], normVec16[rowOff], minBlk[blockOff],
                binaryRepeatBatchF16, 1,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t rowOff = i * TQ_PACK_D;
            const uint32_t blockOff = i * typePerBlock;
            AscendC::Div<ComputeT, false>(
                normVec16[rowOff], normVec16[rowOff], stepBlk[blockOff],
                binaryRepeatBatchF16, 1,
                AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(normVec16, normVec16, static_cast<ComputeT>(0.0f),
                      totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(normVec16, normVec16,
                      static_cast<ComputeT>(TQ_VAL_QUANT_LEVELS_F), totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(normFp32, normVec16, AscendC::RoundMode::CAST_NONE,
                      totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        auto quantI32 = buf1.template ReinterpretCast<int32_t>();
        AscendC::Cast(quantI32, normFp32, AscendC::RoundMode::CAST_RINT,
                      totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        auto quantI16 = buf4.template ReinterpretCast<int16_t>();
        AscendC::Cast(quantI16, quantI32, AscendC::RoundMode::CAST_NONE,
                      totalElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(
            encodedBatch.template ReinterpretCast<int16_t>(), quantI16,
            totalElems);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(minVec, minVec, maxVecT, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(stepVec, stepVec, maxVecT, typePerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        auto metadataMinT = encodedBatch[
            TQ_VAL_ENCODED_VMIN_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                .template ReinterpretCast<T>();
        auto metadataStepT = encodedBatch[
            TQ_VAL_ENCODED_VSTEP_BATCH_BYTE_OFFSET / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            auto metadataMinFp32 = buf2;
            auto metadataStepFp32 = buf2[typePerBlock];
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
    }

    __aicore__ inline void CopyManualCToYBatch(
        AscendC::GlobalTensor<float>& cWorkGm,
        uint32_t bufferOffset,
        uint32_t aivSlice,
        uint32_t inputBuffer) {
        auto yBatch = context_.resource_.YBatchFloat(inputBuffer);
        static constexpr uint32_t SLICE_ELEMS =
            TQ_MANUAL_AIV_SLICE_M * TQ_ROT_N;  // 16×128 = 2048
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
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t encodedRow = desc.startGroupRow + rowOff + r;
                const uint32_t encodedOff = encodedRow * EncodedRowStrideWords<IS_KEY>();
                auto codeOut = scratch[2 * TQ_META_TILE_BYTES + r * codeBytes];
                if constexpr (IS_KEY) {
                    auto encodedBytes = encoded.template ReinterpretCast<uint8_t>();
                    auto encodedRowBytes =
                        encodedBytes[encodedRow * TQ_KEY_ENCODED_ROW_BYTES];
                    for (uint32_t d = 0; d < TQ_KEY_ROW_CODE_BYTES; ++d) {
                        codeOut.SetValue(d, encodedRowBytes.GetValue(d));
                    }
                    meta0.SetValue(tileRow + r, encoded[
                        TQ_KEY_ENCODED_BASE_BATCH_BYTE_OFFSET / sizeof(uint16_t) +
                        encodedRow]
                            .template ReinterpretCast<T>().GetValue(0));
                    meta1.SetValue(tileRow + r, encoded[
                        TQ_KEY_ENCODED_STEP_BATCH_BYTE_OFFSET / sizeof(uint16_t) +
                        encodedRow]
                            .template ReinterpretCast<T>().GetValue(0));
                } else {
                    for (uint32_t d = 0; d < TQ_PACK_D / 2; ++d) {
                        const uint8_t lo = static_cast<uint8_t>(encoded.GetValue(encodedOff + 2 * d));
                        const uint8_t hi = static_cast<uint8_t>(encoded.GetValue(encodedOff + 2 * d + 1));
                        codeOut.SetValue(d, static_cast<uint8_t>(lo | (hi << 4)));
                    }
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
            if constexpr (IS_KEY) {
                constexpr uint32_t maxRowsPerCopy =
                    TQ_MAX_UINT8_DATACOPY_BYTES / TQ_KEY_ROW_CODE_BYTES;
                uint32_t copiedRows = 0;
                while (copiedRows < rows) {
                    const uint32_t remainingRows = rows - copiedRows;
                    const uint32_t copyRows = remainingRows > maxRowsPerCopy
                        ? maxRowsPerCopy
                        : remainingRows;
                    auto scratchCodeBytes =
                        codeBytesUb[copiedRows * TQ_KEY_ROW_CODE_BYTES];
                    copy_packed_ub_to_gm(
                        packedGm,
                        headBase + static_cast<uint64_t>(blockRow + copiedRows) *
                                           codeBytes,
                        scratchCodeBytes, copyRows * codeBytes);
                    copiedRows += copyRows;
                }
            } else {
                copy_packed_ub_to_gm(
                    packedGm,
                    headBase + static_cast<uint64_t>(blockRow) * codeBytes,
                    codeBytesUb, rows * codeBytes);
            }
            copy_packed_ub_to_gm(packedGm, meta0Base + tileStart * TQ_ROW_META_BYTES,
                scratch, TQ_META_TILE_BYTES);
            copy_packed_ub_to_gm(packedGm, meta1Base + tileStart * TQ_ROW_META_BYTES,
                meta1Bytes, TQ_META_TILE_BYTES);
            rowOff += rows;
        }
    }

    __aicore__ inline void PrefetchStream(
        AscendC::GlobalTensor<float>& cWorkGm,
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
        auto yBatch = context_.resource_.YBatchFloat(inputBuffer);
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
        AscendC::GlobalTensor<float>& cWorkGm,
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
