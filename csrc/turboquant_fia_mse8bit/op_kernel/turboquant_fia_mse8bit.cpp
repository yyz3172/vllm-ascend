/*
 * TurboQuant FIA MSE-8bit kernel entry for vllm-ascend.
 *
 * Wraps ops-transformer FiaKernelTurboQuantP0 with a vllm-style IO signature:
 *   query, key_cache, value_cache, block_table, actual_seq_len_q/kv,
 *   atten_mask (optional), key_scale, value_scale, rotation(=Pi), out
 *
 * Tiling keys:
 *   0 = TND+PA, no FlashDecode
 *   1 = TND+PA, FlashDecode
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

// Device-safe tiling layout (do NOT include host tilingdata_base.h).
// GET_TILING_DATA below generates a layout matching host SaveToBuffer;
// this POD must stay field-compatible for the FIA kernel typedef.
#include "tq_fia_device_tiling.h"

#include "vendored/arch32/fia_kernel_turboquant_p0.h"
#include "vendored/arch32/fia_block_cube_turboquant_p0.h"
#include "vendored/arch32/fia_block_vec_turboquant_p0.h"
#include "vendored/arch32/fia_block_vec_flashdecode.h"

using namespace AscendC;
using namespace AttentionCommon;

// Prefer framework-generated tiling copy (matches host REGISTER padding).
// Hand-written WITH_STRUCT POD can disagree with CheckAlignAndGenPlaceHolder
// and poison s2BaseSize / outerSplit → infinite loop or CrossCore hang.
#define TQ_FIA_COPY_TILING(tiling)                                                                 \
    GET_TILING_DATA(tiling_data_gen, tiling);                                                      \
    const FusedInferAttentionScoreTilingData *__restrict tiling_data =                             \
        reinterpret_cast<const FusedInferAttentionScoreTilingData *>(&tiling_data_gen)

#define INVOKE_TQ_FIA(FLASH_DECODE)                                                                \
    do {                                                                                           \
        using FIAT = FIAType<half, half, half, half, true, FLASH_DECODE, FIA_LAYOUT::TND,          \
                             FIA_TQ_MSE_8BIT_MODE, false, FIA_LAYOUT::BSH, false>;                 \
        using CubeT = FiaBlockCubeTurboQuantP0<FIAT>;                                               \
        using VecT = FiaBlockVecTurboQuantP0<FIAT>;                                                 \
        using FdT = FiaBlockVecFlashDecode<FIAT>;                                                   \
        FiaKernelTurboQuantP0<FIAT, CubeT, VecT, FdT> op;                                          \
        TQ_FIA_COPY_TILING(tiling);                                                                \
        op.Init(query, key_cache, value_cache, /*pse*/ nullptr, atten_mask, actual_seq_len_q,      \
                actual_seq_len_kv, /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,                 \
                /*deqScale2*/ nullptr, /*quantScale2*/ nullptr, /*quantOffset2*/ nullptr,          \
                rotation, /*antiquantOffset*/ nullptr, block_table, /*qPad*/ nullptr,              \
                /*kvPad*/ nullptr, key_scale, /*keyAntiquantOffset*/ nullptr, value_scale,         \
                /*valueAntiquantOffset*/ nullptr, /*keySharedPrefix*/ nullptr,                    \
                /*valueSharedPrefix*/ nullptr, /*actualSharedPrefixLen*/ nullptr,                  \
                /*queryRope*/ nullptr, /*keyRope*/ nullptr, /*keyRopeAntiquantScale*/ nullptr,    \
                /*learnableSink*/ nullptr, attention_out, /*softmaxLse*/ nullptr, workspace,       \
                tiling_data, tiling, &tPipe);                                                      \
        op.Process();                                                                              \
    } while (0)

extern "C" __global__ __aicore__ void turboquant_fia_mse8bit(
    GM_ADDR query,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR block_table,
    GM_ADDR actual_seq_len_q,
    GM_ADDR actual_seq_len_kv,
    GM_ADDR atten_mask,
    GM_ADDR key_scale,
    GM_ADDR value_scale,
    GM_ADDR rotation,
    GM_ADDR attention_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    TPipe tPipe;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        INVOKE_TQ_FIA(false);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        INVOKE_TQ_FIA(true);
    }
}
