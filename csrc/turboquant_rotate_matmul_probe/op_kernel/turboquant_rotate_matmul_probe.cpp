/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * TurboQuant rotate matmul probe:
 *   probe_mode=0: MIX 1C2V KFC REGIST_MATMUL_OBJ handshake only (matches fused pack path)
 *   probe_mode=1: pure AIC GM matmul C=A@B via mm.Init (no KFC / no REGIST)
 *   probe_mode=2: MIX 1C2V KFC matmul A(VECOUT)@B(GM)->C(VECIN), then copy C to GM
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

static constexpr uint32_t TQ_PROBE_K = 128;
static constexpr uint32_t TQ_PROBE_N = 128;
static constexpr uint32_t TQ_PROBE_KFC_MAX_M = 32;
static constexpr uint32_t TQ_PROBE_KFC_MAX_ELEMS = TQ_PROBE_KFC_MAX_M * TQ_PROBE_K;

template <AscendC::HardEvent EVT>
__aicore__ inline void ProbeSync()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVT));
    AscendC::SetFlag<EVT>(event);
    AscendC::WaitFlag<EVT>(event);
}

// Fused rotate matmul types (VECOUT/GM/VECIN) for KFC registration.
using ProbeKfcAT = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
using ProbeKfcBT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using ProbeKfcCT = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
using ProbeKfcBiasT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using ProbeKfcMatmulOp = Matmul<ProbeKfcAT, ProbeKfcBT, ProbeKfcCT, ProbeKfcBiasT>;

// Pure GM cube matmul for correctness check (no KFC), aligned with moe_grouped_matmul.
constexpr MatmulConfig ProbeGmMatmulCfg{false, false, true, 0, 0, 0, false, false, false, false, false,
                                        0, 0, 0, 0, 0, 0, 0, true};
using ProbeGmAT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using ProbeGmBT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using ProbeGmCT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using ProbeGmBiasT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using ProbeGmMatmulOp = matmul::MatmulImpl<ProbeGmAT, ProbeGmBT, ProbeGmCT, ProbeGmBiasT, ProbeGmMatmulCfg>;

#define PROBE_KFC_REGIST_ONLY()                                                                                        \
    do {                                                                                                               \
        GET_TILING_DATA(tilingData, tiling);                                                                           \
        AscendC::SetSysWorkspace(workspace);                                                                           \
        if (GetSysWorkSpacePtr() == nullptr) {                                                                         \
            return;                                                                                                    \
        }                                                                                                              \
        AscendC::TPipe pipe;                                                                                           \
        ProbeKfcMatmulOp mm;                                                                                           \
        TCubeTiling cubeTiling = tilingData.cubeTiling;                                                                \
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &cubeTiling);                                               \
    } while (0)

#define PROBE_KFC_MATMUL_VECIN()                                                                                       \
    do {                                                                                                               \
        GET_TILING_DATA(tilingData, tiling);                                                                           \
        AscendC::SetSysWorkspace(workspace);                                                                           \
        if (GetSysWorkSpacePtr() == nullptr) {                                                                         \
            return;                                                                                                    \
        }                                                                                                              \
        AscendC::TPipe pipe;                                                                                           \
        ProbeKfcMatmulOp mm;                                                                                           \
        TCubeTiling cubeTiling = tilingData.cubeTiling;                                                                \
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &cubeTiling);                                               \
        if ASCEND_IS_AIC {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        if ((AscendC::GetSubBlockIdx() % 2) != 0) {                                                                    \
            return;                                                                                                    \
        }                                                                                                              \
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQue;                                                          \
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> aQue;                                                             \
        AscendC::TQue<AscendC::TPosition::VECIN, 1> cQue;                                                              \
        AscendC::TBuf<AscendC::TPosition::VECCALC> mmWorkspace;                                                        \
        pipe.InitBuffer(inputQue, 1, TQ_PROBE_KFC_MAX_ELEMS * sizeof(half));                                           \
        pipe.InitBuffer(aQue, 1, TQ_PROBE_KFC_MAX_ELEMS * sizeof(half));                                               \
        pipe.InitBuffer(cQue, 1, TQ_PROBE_KFC_MAX_ELEMS * sizeof(half));                                               \
        pipe.InitBuffer(mmWorkspace, TQ_PROBE_KFC_MAX_ELEMS * sizeof(half));                                           \
        GlobalTensor<half> aGm;                                                                                        \
        GlobalTensor<half> bGm;                                                                                        \
        GlobalTensor<half> cGm;                                                                                        \
        const uint32_t mPad = static_cast<uint32_t>(tilingData.cubeTiling.M);                                          \
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a),                                                         \
                            (uint64_t)mPad * TQ_PROBE_K);                                                             \
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), (uint64_t)TQ_PROBE_K * TQ_PROBE_N);                    \
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c),                                                         \
                            (uint64_t)mPad * TQ_PROBE_N);                                                             \
        auto inputLocal = inputQue.AllocTensor<half>();                                                                \
        DataCopy(inputLocal, aGm, mPad * TQ_PROBE_K);                                                                  \
        inputQue.EnQue(inputLocal);                                                                                    \
        auto inputReady = inputQue.DeQue<half>();                                                                      \
        auto aLocal = aQue.AllocTensor<half>();                                                                        \
        AscendC::Adds(aLocal, inputReady, static_cast<half>(0), mPad * TQ_PROBE_K);                                    \
        AscendC::PipeBarrier<PIPE_V>();                                                                                \
        inputQue.FreeTensor(inputReady);                                                                               \
        aQue.EnQue(aLocal);                                                                                            \
        auto aReady = aQue.DeQue<half>();                                                                              \
        auto cLocal = cQue.AllocTensor<half>();                                                                        \
        mm.SetOrgShape(mPad, TQ_PROBE_N, TQ_PROBE_K);                                                                  \
        mm.SetSingleShape(tilingData.m, TQ_PROBE_N, TQ_PROBE_K);                                                       \
        mm.SetTensorA(aReady, false);                                                                                  \
        mm.SetTensorB(bGm, false);                                                                                     \
        mm.SetLocalWorkspace(mmWorkspace.Get<uint8_t>());                                                              \
        mm.IterateAll(cLocal);                                                                                         \
        mm.End();                                                                                                      \
        aQue.FreeTensor(aReady);                                                                                       \
        cQue.EnQue(cLocal);                                                                                            \
        auto cReady = cQue.DeQue<half>();                                                                              \
        ProbeSync<AscendC::HardEvent::V_MTE3>();                                                                       \
        DataCopy(cGm, cReady, tilingData.m * TQ_PROBE_N);                                                              \
        ProbeSync<AscendC::HardEvent::MTE3_MTE2>();                                                                    \
        cQue.FreeTensor(cReady);                                                                                       \
    } while (0)

#define PROBE_GM_MATMUL()                                                                                              \
    do {                                                                                                               \
        if ASCEND_IS_AIV {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        GET_TILING_DATA(tilingData, tiling);                                                                           \
        if (AscendC::GetBlockIdx() != 0) {                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        AscendC::TPipe pipe;                                                                                           \
        ProbeGmMatmulOp mm;                                                                                            \
        mm.Init(&tilingData.cubeTiling, &pipe);                                                                          \
        GlobalTensor<half> aGm;                                                                                        \
        GlobalTensor<half> bGm;                                                                                        \
        GlobalTensor<half> cGm;                                                                                        \
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a),                                                         \
                            (uint64_t)tilingData.cubeTiling.M * TQ_PROBE_K);                                           \
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), (uint64_t)TQ_PROBE_K * TQ_PROBE_N);                    \
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c),                                                         \
                            (uint64_t)tilingData.cubeTiling.M * TQ_PROBE_N);                                           \
        mm.SetOrgShape(tilingData.m, TQ_PROBE_N, TQ_PROBE_K);                                                          \
        mm.SetSingleShape(tilingData.m, TQ_PROBE_N, TQ_PROBE_K);                                                       \
        mm.SetTensorA(aGm, false);                                                                                     \
        mm.SetTensorB(bGm, false);                                                                                     \
        mm.template IterateAll<false>(cGm, 0);                                                                         \
        mm.End();                                                                                                      \
    } while (0)

}  // namespace

extern "C" __global__ __aicore__ void turboquant_rotate_matmul_probe(
    GM_ADDR a,
    GM_ADDR b,
    GM_ADDR c,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) {
        // Match grouped_matmul / dispatch_ffn_combine: tiling key must pair with KERNEL_TASK_TYPE.
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        PROBE_KFC_REGIST_ONLY();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIC_ONLY);
        PROBE_GM_MATMUL();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        PROBE_KFC_MATMUL_VECIN();
    }
}
