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
class BitResidualPackK8v4CubeService {
public:
    __aicore__ inline explicit BitResidualPackK8v4CubeService(
        BitResidualPackK8v4Context<T>& context)
        : context_(context) {}

    __aicore__ inline void Process(uint32_t manualGroupId) {
        auto& op = context_;
        AscendC::GlobalTensor<float> normWorkGm;
        AscendC::GlobalTensor<T> cWorkGm;
        op.InitManualWorkspaceTensors(manualGroupId, normWorkGm, cWorkGm);

        TqManualMmadResource manualResource;
        LoadResidentRotation(manualResource, op.rotationTGm_);

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
                const uint32_t tokenStart = seqStart + rowOff;
                const uint32_t slot = firstSlot + rowOff;
                const uint32_t blockIdx = slot / op.blockSize_;
                const uint32_t blockOffset = slot - blockIdx * op.blockSize_;
                const uint32_t valueGroupInBlock = blockOffset / TQ_VAL_GROUP_ROWS;
                const bool preserveValue =
                    startGroupRow != 0 || validRows < TQ_MANUAL_GROUP_ROWS;
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
                        ComputeStream(
                            manualResource, normWorkGm, cWorkGm, keyDesc);
                        ComputeStream(
                            manualResource, normWorkGm, cWorkGm, valueDesc);
                        ++manualTileOrdinal;
                    }
                    ++tileOrdinal;
                }
                rowOff += validRows;
            }
        }
    }

private:
    __aicore__ inline constexpr QuantMode_t FixpipeQuantMode() const {
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            return QuantMode_t::F322BF16;
        }
        return QuantMode_t::F322F16;
    }

    __aicore__ inline void LoadResidentRotation(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& rotationTGm) {
        auto bL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_B_L1_OFFSET);
        AscendC::Nd2NzParams params;
        params.ndNum = TQ_ROT_K / TQ_CUBE_M_ALIGN;
        params.nValue = TQ_CUBE_M_ALIGN;
        params.dValue = TQ_ROT_N;
        params.srcDValue = TQ_ROT_N;
        params.srcNdMatrixStride = TQ_CUBE_M_ALIGN * TQ_ROT_N;
        params.dstNzC0Stride = TQ_CUBE_M_ALIGN;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = TQ_CUBE_M_ALIGN * TQ_ROT_N;
        AscendC::DataCopy(bL1, rotationTGm, params);
        TqSyncFixed<AscendC::HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void LoadRawTensorToL1(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& xGm,
        const ManualKey1StreamDesc& desc,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) {
        auto aL1 =
            resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        for (uint32_t aivSlice = 0; aivSlice < TQ_AIV_SUB_BLOCKS; ++aivSlice) {
            const uint32_t firstHead =
                desc.headTileStart + aivSlice * TQ_MANUAL_HEADS_PER_AIV;
            const uint32_t l1SliceOffset =
                aivSlice * TQ_MANUAL_AIV_SLICE_ELEMS;
            for (uint32_t localHead = 0;
                 localHead < TQ_MANUAL_HEADS_PER_AIV;
                 ++localHead) {
                const uint32_t headIdx = firstHead + localHead;
                const uint64_t srcOffset =
                    storageOffset +
                    static_cast<uint64_t>(desc.tokenStart) * strideToken +
                    static_cast<uint64_t>(headIdx) * strideHead;
                const uint32_t dstRow =
                    localHead * TQ_MANUAL_GROUP_ROWS + desc.startGroupRow;
                AscendC::Nd2NzParams params;
                params.ndNum = 1;
                params.nValue = desc.validRows;
                params.dValue = TQ_ROT_K;
                params.srcDValue = strideToken;
                params.dstNzC0Stride = TQ_MANUAL_AIV_SLICE_M;
                params.dstNzNStride = 1;
                params.srcNdMatrixStride = 0;
                params.dstNzMatrixStride = 0;
                AscendC::DataCopy(
                    aL1[l1SliceOffset + dstRow * TQ_CUBE_M_ALIGN],
                    xGm[srcOffset],
                    params);
            }
        }
        TqSyncFixed<AscendC::HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void LoadRawInputToL1(
        TqManualMmadResource& resource,
        const ManualKey1StreamDesc& desc) {
        if (desc.isValue) {
            LoadRawTensorToL1(
                resource,
                context_.valueGm_,
                desc,
                context_.valueStorageOffset_,
                context_.valueStrideToken_,
                context_.valueStrideHead_);
        } else {
            LoadRawTensorToL1(
                resource,
                context_.keyGm_,
                desc,
                context_.keyStorageOffset_,
                context_.keyStrideToken_,
                context_.keyStrideHead_);
        }
    }

    __aicore__ inline void ComputeNormMatrices(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<float>& normWorkGm,
        uint32_t streamOrdinal) {
        auto aL1 =
            resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto bL0 = resource.l0BBuf.template GetBufferByByte<T>(
            TQ_MANUAL_NORM_B_L0_BYTE_OFFSET);
        auto cL0 = resource.l0CBuf.template GetBufferByByte<float>(0);
        const uint32_t normBase = context_.ManualNormBufferOffset(streamOrdinal);

        for (uint32_t slice = 0; slice < TQ_AIV_SUB_BLOCKS; ++slice) {
            const uint32_t l1SliceOffset =
                slice * TQ_MANUAL_AIV_SLICE_ELEMS;
            TqSyncFixed<AscendC::HardEvent::M_MTE1>();

            AscendC::LoadData2DParams aLoad;
            aLoad.startIndex = 0;
            aLoad.repeatTimes = TQ_ROT_K / TQ_CUBE_M_ALIGN;
            aLoad.srcStride = 1;
            aLoad.sid = 0;
            aLoad.dstGap = 0;
            aLoad.ifTranspose = false;
            aLoad.addrMode = 0;
            AscendC::LoadData(aL0, aL1[l1SliceOffset], aLoad);

            AscendC::LoadData2DParams bLoad;
            bLoad.startIndex = 0;
            bLoad.repeatTimes = 1;
            bLoad.srcStride = 1;
            bLoad.sid = 0;
            bLoad.dstGap = 0;
            bLoad.ifTranspose = true;
            bLoad.addrMode = 0;
            for (uint32_t kBlock = 0;
                 kBlock < TQ_ROT_K / TQ_CUBE_M_ALIGN;
                 ++kBlock) {
                const uint32_t blockOffset =
                    kBlock * TQ_MANUAL_AIV_SLICE_M * TQ_CUBE_M_ALIGN;
                AscendC::LoadData(
                    bL0[blockOffset],
                    aL1[l1SliceOffset + blockOffset],
                    bLoad);
            }
            TqSyncFixed<AscendC::HardEvent::MTE1_M>();

            AscendC::MmadParams mmParams;
            mmParams.m = TQ_MANUAL_AIV_SLICE_M;
            mmParams.n = TQ_MANUAL_AIV_SLICE_M;
            mmParams.k = TQ_ROT_K;
            mmParams.cmatrixInitVal = true;
            mmParams.cmatrixSource = false;
            mmParams.unitFlag = 0b11;
            TqSyncFixed<AscendC::HardEvent::FIX_M>();
            AscendC::Mmad(cL0, aL0, bL0, mmParams);
            AscendC::PipeBarrier<PIPE_M>();
            TqSyncFixed<AscendC::HardEvent::M_FIX>();

            AscendC::DataCopyCO12DstParams fixParams;
            fixParams.mSize = TQ_MANUAL_AIV_SLICE_M;
            fixParams.nSize = TQ_MANUAL_AIV_SLICE_M;
            fixParams.dstStride = TQ_MANUAL_AIV_SLICE_M;
            fixParams.srcStride = TQ_MANUAL_AIV_SLICE_M;
            fixParams.quantPre = QuantMode_t::NoQuant;
            fixParams.nz2ndEn = true;
            fixParams.unitFlag = 0b11;
            fixParams.reluPre = 0;
            AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
            AscendC::DataCopy(
                normWorkGm[
                    normBase + slice * TQ_MANUAL_NORM_MATRIX_ELEMS],
                cL0,
                fixParams);
            TqSyncFixed<AscendC::HardEvent::FIX_M>();
        }
    }

    __aicore__ inline void ComputeLoadedTile(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t cOffset) {
        auto aL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        auto bL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_B_L1_OFFSET);
        auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto bL0 = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto cL0Base = resource.l0CBuf.template GetBufferByByte<float>(0);

        // The preceding Gram MMAD also consumes L0B.  Wait for PIPE_M before
        // replacing that buffer with the resident rotation matrix.
        TqSyncFixed<AscendC::HardEvent::M_MTE1>();
        AscendC::LoadData2DParams bLoad;
        bLoad.startIndex = 0;
        bLoad.repeatTimes = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.srcStride = 1;
        bLoad.sid = 0;
        bLoad.dstGap = 0;
        bLoad.ifTranspose = true;
        bLoad.addrMode = 0;
        for (uint32_t i = 0; i < TQ_ROT_K / TQ_CUBE_M_ALIGN; ++i) {
            AscendC::LoadData(bL0[i * TQ_ROT_N * TQ_CUBE_M_ALIGN],
                              bL1[i * TQ_ROT_N * TQ_CUBE_M_ALIGN], bLoad);
        }
        for (uint32_t rowBase = 0; rowBase < TQ_MANUAL_ROT_TILE_M;
             rowBase += TQ_CUBE_M_ALIGN) {
            auto aL0Tile =
                aL0[(rowBase / TQ_CUBE_M_ALIGN) * TQ_CUBE_M_ALIGN * TQ_ROT_K];
            auto cL0 =
                cL0Base[(rowBase / TQ_CUBE_M_ALIGN) * TQ_CUBE_M_ALIGN * TQ_ROT_N];
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

            AscendC::MmadParams mmParams;
            mmParams.m = TQ_CUBE_M_ALIGN;
            mmParams.n = TQ_ROT_N;
            mmParams.k = TQ_ROT_K;
            mmParams.cmatrixInitVal = true;
            mmParams.cmatrixSource = false;
            mmParams.unitFlag = 0b11;
            TqSyncFixed<AscendC::HardEvent::FIX_M>();
            AscendC::Mmad(cL0, aL0Tile, bL0, mmParams);
            AscendC::PipeBarrier<PIPE_M>();
            TqSyncFixed<AscendC::HardEvent::M_FIX>();

            AscendC::FixpipeParamsV220 fixParams;
            fixParams.nSize = TQ_ROT_N;
            fixParams.mSize = TQ_CUBE_M_ALIGN;
            fixParams.srcStride = TQ_CUBE_M_ALIGN;
            fixParams.dstStride = TQ_ROT_N;
            fixParams.ndNum = 1;
            fixParams.unitFlag = 0b11;
            fixParams.quantPre = FixpipeQuantMode();
            fixParams.reluEn = false;
            AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(
                cWorkGm[cOffset + rowBase * TQ_ROT_N], cL0, fixParams);
            TqSyncFixed<AscendC::HardEvent::FIX_M>();
        }
    }

    __aicore__ inline void ComputeStream(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<float>& normWorkGm,
        AscendC::GlobalTensor<T>& cWorkGm,
        const ManualKey1StreamDesc& desc) {
        const uint16_t flagBase = context_.ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset =
            context_.ManualBufferOffset(desc.streamOrdinal);
        TqCrossCoreWaitForBothAiv<PIPE_FIX>(
            flagBase + TQ_MANUAL_SYNC_C_FREE);
        LoadRawInputToL1(resource, desc);
        ComputeNormMatrices(resource, normWorkGm, desc.streamOrdinal);
        ComputeLoadedTile(resource, cWorkGm, bufferOffset);
        TqSyncFixed<AscendC::HardEvent::FIX_MTE3>();
        TqCrossCoreSetForBothAiv<PIPE_FIX>(
            flagBase + TQ_MANUAL_SYNC_C_READY);
    }

    BitResidualPackK8v4Context<T>& context_;
};

}  // namespace bit_residual
