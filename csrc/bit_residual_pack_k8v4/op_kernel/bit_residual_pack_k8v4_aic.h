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

static constexpr uint32_t TQ_NORM_MMAD_M = 64;
static constexpr uint32_t TQ_NORM_MMAD_K = 128;
static constexpr uint32_t TQ_NORM_MMAD_N = 1;
static constexpr uint32_t TQ_NORM_MMAD_A_L1_OFFSET = 0;
static constexpr uint32_t TQ_NORM_MMAD_B_L1_OFFSET =
    TQ_NORM_MMAD_A_L1_OFFSET + TQ_NORM_MMAD_M * TQ_NORM_MMAD_K * sizeof(uint16_t);

using TqManualMmadResource = Catlass::Arch::Resource<Catlass::Arch::AtlasA2>;

template <typename T>
__aicore__ inline constexpr QuantMode_t TqManualFixpipeQuantMode() {
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    }
    return QuantMode_t::F322F16;
}

template <typename T>
class BitResidualPackK8v4AicService {
public:
    __aicore__ inline explicit BitResidualPackK8v4AicService(TqManualMmadResource& resource)
        : resource_(resource) {}

    __aicore__ inline void Mmad(
        uint32_t cL0Offset,
        uint32_t aL0Offset,
        uint32_t bL0Offset,
        uint32_t m,
        uint32_t n,
        uint32_t k) {
        auto aL0 = resource_.l0ABuf.GetBufferByByte<T>(aL0Offset);
        auto bL0 = resource_.l0BBuf.GetBufferByByte<T>(bL0Offset);
        auto cL0 = resource_.l0CBuf.GetBufferByByte<float>(cL0Offset);

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0b11;

        AscendC::Mmad(cL0, aL0, bL0, mmadParams);
        AscendC::PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void Fixpipe(
        AscendC::GlobalTensor<float>& dstGm,
        uint32_t gmOffset,
        uint32_t cL0Offset,
        uint32_t m,
        uint32_t n) {
        auto cL0 = resource_.l0CBuf.GetBufferByByte<float>(cL0Offset);

        AscendC::FixpipeParamsV220 fixParams;
        fixParams.nSize = n;
        fixParams.mSize = m;
        fixParams.srcStride = m;
        fixParams.dstStride = n;
        fixParams.ndNum = 1;
        fixParams.unitFlag = 0b11;
        fixParams.quantPre = TqManualFixpipeQuantMode<T>();
        fixParams.reluEn = false;

        AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(
            dstGm[gmOffset],
            cL0,
            fixParams);
    }

private:
    TqManualMmadResource& resource_;
};

}  // namespace tq_mmad
