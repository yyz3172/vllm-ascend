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

    __aicore__ inline void Init() {
        eventListMte1M[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
        eventListMte1M[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
        eventListMmte1[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
        eventListMmte1[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
        eventListMte1Mte2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
        eventListMte2Mte1 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void Finite() {
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_M>(eventListMte1M[1]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(eventListMmte1[1]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
    }

    __aicore__ inline void Process(uint32_t manualGroupId) {
        auto& op = context_;
        AscendC::GlobalTensor<float> normWorkGm;
        AscendC::GlobalTensor<T> cWorkGm;
        op.InitManualWorkspaceTensors(manualGroupId, normWorkGm, cWorkGm);
        Init();

        TqManualMmadResource manualResource;
        
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
        LoadResidentRotation(manualResource, op.rotationTGm_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[1]);

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

        // ── Drain remaining hardware event flags ─────────────────────────
        // M_MTE1: last loop iteration re-sets "buffer free" flags that no
        // subsequent iteration consumes.  Must drain before kernel exits.
        //
        // FIX_M: with unitFlag=0b11, hardware signals FIX_M after each
        // Fixpipe.  The last Fixpipe's FIX_M signal remains pending
        // (no subsequent MMAD's hardware-internal wait to consume it).
        // Must drain with software Wait<FIX_M> (following reference
        // destructor pattern).
        //
        // M_FIX: NOT drained.  With unitFlag=0b11, hardware-internal
        // M_FIX signals are consumed by Fixpipe's hardware-internal wait.
        // No residual M_FIX remains after the last Fixpipe.
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[1]);
        Finite();
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
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);

        auto rotationBL0 = resource.l0BBuf.template GetBufferByByte<T>(0);
        // Wait until M engine has finished consuming any prior L0B data
        // before MTE1 overwrites it with the resident rotation B matrix.
        AscendC::LoadData2DParams bLoad;
        bLoad.startIndex = 0;
        bLoad.repeatTimes = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.srcStride = 1;
        bLoad.sid = 0;
        bLoad.dstGap = 0;
        bLoad.ifTranspose = true;
        bLoad.addrMode = 0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
        for (uint32_t i = 0; i < TQ_ROT_K / TQ_CUBE_M_ALIGN; ++i) {
            AscendC::LoadData(
                rotationBL0[i * TQ_ROT_N * TQ_CUBE_M_ALIGN],
                bL1[i * TQ_ROT_N * TQ_CUBE_M_ALIGN], bLoad);
        }
        // MTE1 finished writing L0B.  Signal M that L0B is ready.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);
    }

    __aicore__ inline void LoadRawTensorToL1(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& xGm,
        const ManualKey1StreamDesc& desc,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
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
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
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

    // ── Gram (norm) MMAD: A * A^T → norm²  ──────────────────────────────
    //
    // Per-slice async event sequence (per-buffer M_MTE1, MTE1_M;
    // unitFlag=0b11 handles M_FIX/FIX_M at hardware level):
    //
    //   Wait<M_MTE1>(A)  — M done with prev L0A; MTE1 can write
    //   Wait<M_MTE1>(B)  — M done with prev L0B; MTE1 can write
    //   LoadData L1→L0A
    //   LoadData L1→L0B
    //   Set<MTE1_M>(A)   — L0A loaded; M can read
    //   Set<MTE1_M>(B)   — L0B loaded; M can read
    //   Wait<MTE1_M>(A)  — L0A ready for M
    //   Wait<MTE1_M>(B)  — L0B ready for M
    //   Mmad L0A×L0B→L0C (unitFlag=0b11: hw waits FIX_M, signals M_FIX)
    //   PipeBarrier<PIPE_M>
    //   Set<M_MTE1>(A)   — M freed L0A; next MTE1 can write
    //   Set<M_MTE1>(B)   — M freed L0B; next MTE1 can write
    //   Fixpipe<CFG_NZ> L0C→GM (unitFlag=0b11: hw waits M_FIX, signals FIX_M;
    //                 NoQuant, no L0B borrow)
    //
    __aicore__ inline void ComputeNormMatrices(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<float>& normWorkGm,
        uint32_t streamOrdinal) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
        auto aL1 =
            resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(
            TQ_MANUAL_NORM_A_L0_BYTE_OFFSET);
        auto bL0 = resource.l0BBuf.template GetBufferByByte<T>(
            TQ_MANUAL_NORM_B_L0_BYTE_OFFSET);
        auto cL0Base = resource.l0CBuf.template GetBufferByByte<float>(
            TQ_MANUAL_NORM_C_L0_BYTE_OFFSET);
        const uint32_t normBase = context_.ManualNormBufferOffset(streamOrdinal);
        for (uint32_t slice = 0; slice < TQ_AIV_SUB_BLOCKS; ++slice) {
            const uint32_t l1SliceOffset =
                slice * TQ_MANUAL_AIV_SLICE_ELEMS;
            const uint8_t unitFlag = 0b11;

            // ── MTE1: load A and B from L1 to L0 ────────────────────────
            // Wait until M engine has finished consuming the previous
            // L0A/L0B data (M_MTE1 direction).
            AscendC::LoadData2DParams bLoad;
            bLoad.startIndex = 0;
            bLoad.repeatTimes = TQ_ROT_K / TQ_CUBE_M_ALIGN;
            bLoad.srcStride = 1;
            bLoad.sid = 0;
            bLoad.dstGap = 0;
            bLoad.ifTranspose = false;
            bLoad.addrMode = 0;

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
            AscendC::LoadData(bL0, aL1[l1SliceOffset], bLoad);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);
            
            AscendC::LoadData2DParams aLoad;
            aLoad.startIndex = 0;
            aLoad.repeatTimes = TQ_ROT_K / TQ_CUBE_M_ALIGN;
            aLoad.srcStride = 1;
            aLoad.sid = 0;
            aLoad.dstGap = 0;
            aLoad.ifTranspose = false;
            aLoad.addrMode = 0;

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[1]);
            AscendC::LoadData(aL0, aL1[l1SliceOffset], aLoad);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[1]);

            // ── M: MMAD L0A × L0B → L0C ────────────────────────────────
            // Wait until MTE1 has loaded L0A/L0B (MTE1_M direction).

            AscendC::MmadParams mmParams;
            mmParams.m = TQ_MANUAL_AIV_SLICE_M;
            mmParams.n = TQ_MANUAL_AIV_SLICE_M;
            mmParams.k = TQ_ROT_K;
            mmParams.cmatrixInitVal = true;
            mmParams.cmatrixSource = false;
            mmParams.unitFlag = unitFlag;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[1]);
            auto cL0 = cL0Base[
                slice * TQ_MANUAL_NORM_MATRIX_ELEMS];
            AscendC::Mmad(cL0, aL0, bL0, mmParams);

            AscendC::FixpipeParamsV220 fixParams;
            fixParams.nSize = TQ_MANUAL_AIV_SLICE_M;
            fixParams.mSize = TQ_MANUAL_AIV_SLICE_M;
            fixParams.srcStride = TQ_MANUAL_AIV_SLICE_M;
            fixParams.dstStride = TQ_MANUAL_AIV_SLICE_M;
            fixParams.ndNum = 1;
            fixParams.unitFlag = unitFlag;
            fixParams.quantPre = QuantMode_t::NoQuant;
            fixParams.reluEn = false;
            AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(
                normWorkGm[
                    normBase + slice * TQ_MANUAL_NORM_MATRIX_ELEMS],
                cL0,
                fixParams);

            // M finished consuming L0A and L0B.  Release them so the next
            // slice's MTE1 LoadData can overwrite them (M_MTE1 direction).
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[1]);
        }
    }

    // ── Rotation MMAD: A * B → C ────────────────────────────────────────
    //
    // Per-rowBase tile flow (16 rows × 128 cols):
    //   Wait<M_MTE1>(A)   — M done with prev L0A; MTE1 can write
    //   LoadData L1→L0A
    //   Set<MTE1_M>(A)    — L0A loaded; M can read
    //   Wait<MTE1_M>(A)   — L0A ready for M
    //   Mmad L0A×L0B→L0C  (resident B, unitFlag=0b11: hw FIX_M/M_FIX)
    //   PipeBarrier<PIPE_M>
    //   Set<M_MTE1>(A)    — M freed L0A; next MTE1 can write
    //   Fixpipe<CFG_NZ>   (unitFlag=0b11: hw M_FIX/FIX_M, no L0B borrow)
    //
    // Resident B in L0B[0..32KB]:
    //   Loaded once by LoadResidentRotation → MTE1_M(ROT_B) flag consumed
    //   once in Process() after LoadResidentRotation returns.  No per-call
    //   or per-iteration MTE1_M(ROT_B) sync needed — resident B stays in
    //   L0B and MTE1 never re-loads it.  (Moving this Wait inside
    //   ComputeLoadedTile would cause a hang on the 2nd call: the flag is
    //   consumed on the 1st call, and subsequent calls block forever because
    //   no new Set is produced.)
    //
    __aicore__ inline void ComputeLoadedTile(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& cWorkGm,
        uint32_t cOffset) {
        // TODO: wait id1
        auto aL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto bL0 = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto cL0Base = resource.l0CBuf.template GetBufferByByte<float>(0);

        // Resident B's MTE1_M(ROT_B) flag was consumed in Process() once
        // after LoadResidentRotation.  B stays resident in L0B for all
        // iterations — no per-call or per-iteration MTE1_M(ROT_B) sync
        // is needed.

        for (uint32_t rowBase = 0; rowBase < TQ_MANUAL_ROT_TILE_M;
             rowBase += TQ_CUBE_M_ALIGN) {
            const uint8_t unitFlag = 0b11;
            auto aL0Tile =
                aL0[(rowBase / TQ_CUBE_M_ALIGN) * TQ_CUBE_M_ALIGN * TQ_ROT_K];
            auto cL0 =
                cL0Base[(rowBase / TQ_CUBE_M_ALIGN) * TQ_CUBE_M_ALIGN * TQ_ROT_N];

            // ── MTE1: load A from L1 to L0A ─────────────────────────────

            AscendC::LoadData2DParams aLoad;
            aLoad.startIndex = 0;
            aLoad.repeatTimes = TQ_ROT_K / TQ_CUBE_M_ALIGN;
            aLoad.srcStride = 1;
            aLoad.sid = 0;
            aLoad.dstGap = 0;
            aLoad.ifTranspose = false;
            aLoad.addrMode = 0;
            
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
            AscendC::LoadData(
                aL0Tile,
                aL1[(rowBase / TQ_CUBE_M_ALIGN) * TQ_MANUAL_AIV_SLICE_ELEMS],
                aLoad);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);

            // ── M: MMAD L0A × L0B → L0C ────────────────────────────────
            // With unitFlag=0b11, hardware handles FIX_M/M_FIX internally:
            //   - MMAD hw-internal: wait FIX_M before MMAD, signal M_FIX
            //     after MMAD
            //   - No software Wait<FIX_M> needed (conflicts with hardware
            //     internal signal — depth-1 flag conflict → deadlock).
            // Resident B already consumed above — no per-iter Wait needed.

            AscendC::MmadParams mmParams;
            mmParams.m = TQ_CUBE_M_ALIGN;
            mmParams.n = TQ_ROT_N;
            mmParams.k = TQ_ROT_K;
            mmParams.cmatrixInitVal = true;
            mmParams.cmatrixSource = false;
            mmParams.unitFlag = unitFlag;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M[0]);
            AscendC::Mmad(cL0, aL0Tile, bL0, mmParams);
            AscendC::FixpipeParamsV220 fixParams;
            fixParams.nSize = TQ_ROT_N;
            fixParams.mSize = TQ_CUBE_M_ALIGN;
            fixParams.srcStride = TQ_CUBE_M_ALIGN;
            fixParams.dstStride = TQ_ROT_N;
            fixParams.ndNum = 1;
            fixParams.unitFlag = unitFlag;
            fixParams.quantPre = FixpipeQuantMode();
            fixParams.reluEn = false;
            AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(
                cWorkGm[cOffset + rowBase * TQ_ROT_N], cL0, fixParams);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1[0]);
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
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        TqSyncFixed<AscendC::HardEvent::FIX_MTE3>();
        TqCrossCoreSetForBothAiv<PIPE_FIX>(
            flagBase + TQ_MANUAL_SYNC_C_READY);
    }

    BitResidualPackK8v4Context<T>& context_;
    int8_t eventListMte1M[2];
    int8_t eventListMmte1[2];
    int8_t eventListMte1Mte2;
    int8_t eventListMte2Mte1;
};

}  // namespace bit_residual
