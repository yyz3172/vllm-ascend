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

using namespace AscendC;

namespace tq_mmad {

using TqManualMmadResource = Catlass::Arch::Resource<Catlass::Arch::AtlasA2>;

template <AscendC::HardEvent EVT>
__aicore__ inline void TqSyncFixed(uint32_t eventId = EVENT_ID0) {
    SetFlag<EVT>(eventId);
    WaitFlag<EVT>(eventId);
}

__aicore__ inline void TqSyncVToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void TqSyncMte3ToMte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

template <typename T>
class TqAivDataManager {
public:
    __aicore__ inline explicit TqAivDataManager(TqManualMmadResource& resource)
        : resource_(resource) {}

    __aicore__ inline void CopyGmToL1(
        uint32_t l1Offset,
        AscendC::GlobalTensor<T>& srcGm,
        uint32_t gmOffset,
        uint32_t rows,
        uint32_t cols,
        uint32_t gmStride) {
        auto dstL1 = resource_.l1Buf.GetBufferByByte<T>(l1Offset);
        for (uint32_t rowBase = 0; rowBase < rows; rowBase += 16) {
            AscendC::Nd2NzParams params;
            params.ndNum = 1;
            params.nValue = 16;
            params.dValue = cols;
            params.srcDValue = gmStride;
            params.dstNzC0Stride = 16;
            params.dstNzNStride = 1;
            params.srcNdMatrixStride = 0;
            params.dstNzMatrixStride = 0;
            AscendC::DataCopy(
                dstL1[rowBase * cols],
                srcGm[gmOffset + rowBase * gmStride],
                params);
        }
        TqSyncFixed<AscendC::HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void LoadL1ToL0A(
        uint32_t l1Offset,
        uint32_t l0Offset,
        uint32_t rows,
        uint32_t cols) {
        auto srcL1 = resource_.l1Buf.GetBufferByByte<T>(l1Offset);
        auto dstL0 = resource_.l0ABuf.GetBufferByByte<T>(l0Offset);

        AscendC::LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = cols / 16;
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = false;
        loadParams.addrMode = 0;

        for (uint32_t rowBase = 0; rowBase < rows; rowBase += 16) {
            AscendC::LoadData(
                dstL0[rowBase * cols],
                srcL1[rowBase * cols],
                loadParams);
        }
    }

    __aicore__ inline void LoadL1ToL0B(
        uint32_t l1Offset,
        uint32_t l0Offset,
        uint32_t rows,
        uint32_t cols) {
        auto srcL1 = resource_.l1Buf.GetBufferByByte<T>(l1Offset);
        auto dstL0 = resource_.l0BBuf.GetBufferByByte<T>(l0Offset);

        AscendC::LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = cols / 16;
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = true;
        loadParams.addrMode = 0;

        for (uint32_t i = 0; i < rows / 16; ++i) {
            AscendC::LoadData(
                dstL0[i * cols * 16],
                srcL1[i * cols * 16],
                loadParams);
        }
    }

    __aicore__ inline void DuplicateToL1(
        uint32_t l1Offset,
        T value,
        uint32_t count) {
        auto dstL1 = resource_.l1Buf.GetBufferByByte<T>(l1Offset);
        AscendC::LocalTensor<T> local = dstL1;
        AscendC::Duplicate(local, value, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyUbToGm(
        AscendC::GlobalTensor<T>& dstGm,
        uint32_t gmOffset,
        AscendC::LocalTensor<T> srcUb,
        uint32_t elemCount) {
        TqSyncVToMte3();
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(elemCount * sizeof(T)),
            0,
            0,
            0};
        AscendC::DataCopyPad(dstGm[gmOffset], srcUb, copyParams);
        TqSyncMte3ToMte2();
    }

private:
    TqManualMmadResource& resource_;
};

}  // namespace tq_mmad
