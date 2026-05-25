#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "turboquant_rotate_matmul_probe_tiling.h"

namespace {
constexpr uint32_t TQ_PROBE_M_ALIGN = 16;
constexpr uint32_t TQ_PROBE_N = 128;
constexpr uint32_t TQ_PROBE_K = 128;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t TQ_PROBE_TILING_KEY_KFC = 0;
constexpr uint32_t TQ_PROBE_TILING_KEY_GM = 1;
constexpr int32_t TQ_PROBE_BEST_BASEN = 256;
constexpr uint64_t TQ_PROBE_DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t TQ_PROBE_DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t TQ_PROBE_FP32_SIZE = 4;
constexpr int32_t TQ_PROBE_MAX_BASEM = 256;

uint32_t AlignUp16(uint32_t x)
{
    return (x + TQ_PROBE_M_ALIGN - 1) / TQ_PROBE_M_ALIGN * TQ_PROBE_M_ALIGN;
}

ge::graphStatus FillKfcCubeTiling(
    matmul_tiling::MultiCoreMatmulTiling& tilingApi,
    uint32_t mPad,
    optiling::TCubeTiling& cubeTiling)
{
    tilingApi.SetOrgShape(mPad, TQ_PROBE_N, TQ_PROBE_K);
    tilingApi.SetShape(mPad, TQ_PROBE_N, TQ_PROBE_K);
    tilingApi.EnableBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillGmCubeTiling(
    const platform_ascendc::PlatformAscendC& platform,
    uint32_t mPad,
    optiling::TCubeTiling& cubeTiling)
{
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    matmul_tiling::PlatformInfo platformInfo;
    platformInfo.socVersion = platform.GetSocVersion();
    platformInfo.l1Size = l1Size;
    platformInfo.l0CSize = l0cSize;
    platformInfo.ubSize = ubSize;
    platformInfo.l0ASize = l0aSize;
    platformInfo.l0BSize = l0bSize;

    const int32_t baseN = std::min<int32_t>(TQ_PROBE_BEST_BASEN, static_cast<int32_t>(TQ_PROBE_N));
    constexpr uint32_t mmDataTypeSize = 2;
    int32_t baseK = static_cast<int32_t>(
        (l0bSize / TQ_PROBE_DOUBLE_BUFFER_L0A_L0B) / (static_cast<uint64_t>(baseN) * mmDataTypeSize));
    baseK = static_cast<int32_t>(AlignUp16(static_cast<uint32_t>(baseK)));
    const uint32_t maxBaseM = static_cast<uint32_t>(l0cSize / (baseN * TQ_PROBE_FP32_SIZE));
    int32_t baseM = static_cast<int32_t>(std::min<uint64_t>(
        (l0aSize / TQ_PROBE_DOUBLE_BUFFER_L0A_L0B) / (static_cast<uint64_t>(baseK) * mmDataTypeSize),
        maxBaseM));
    baseM = static_cast<int32_t>(
        baseM > static_cast<int32_t>(mPad) ? AlignUp16(mPad) : AlignUp16(static_cast<uint32_t>(baseM)));
    if (baseM > TQ_PROBE_MAX_BASEM) {
        baseM = TQ_PROBE_MAX_BASEM;
    }
    if (baseM <= 0 || baseK <= 0) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MultiCoreMatmulTiling tilingApi(platformInfo);
    tilingApi.SetDim(1);
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16,
        false);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16,
        false);
    tilingApi.SetCType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetOrgShape(mPad, TQ_PROBE_N, TQ_PROBE_K);
    tilingApi.SetShape(mPad, TQ_PROBE_N, TQ_PROBE_K);
    tilingApi.EnableBias(false);
    tilingApi.SetFixSplit(baseM, baseN, baseK);
    tilingApi.SetBufferSpace(l1Size, l0cSize, ubSize);
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepN = 1;
    const uint32_t mmStepKa = 1;
    const uint32_t mmStepKb = 1;
    const uint32_t mmDepthA1 = mmStepKa * TQ_PROBE_DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    const uint32_t mmDepthB1 = mmStepKb * TQ_PROBE_DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
    cubeTiling.set_shareMode(0);
    cubeTiling.set_dbL0C(1);
    cubeTiling.set_baseM(baseM);
    cubeTiling.set_baseN(baseN);
    cubeTiling.set_baseK(baseK);
    cubeTiling.set_stepKa(mmStepKa);
    cubeTiling.set_depthA1(mmDepthA1);
    cubeTiling.set_stepKb(mmStepKb);
    cubeTiling.set_depthB1(mmDepthB1);
    cubeTiling.set_stepM(stepM);
    cubeTiling.set_stepN(stepN);
    cubeTiling.set_usedCoreNum(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace

namespace optiling {

static ge::graphStatus TurboquantRotateMatmulProbeTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto aShape = context->GetInputShape(0);
    if (aShape == nullptr) {
        OPS_LOG_E(nodeName, "input a shape is null");
        return ge::GRAPH_FAILED;
    }

    const auto& storageShape = aShape->GetStorageShape();
    if (storageShape.GetDimNum() != 2 || storageShape.GetDim(1) != TQ_PROBE_K) {
        OPS_LOG_E(nodeName, "input a must be [M, 128]");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* probeModePtr = attrs->GetAttrPointer<int64_t>(0);
    if (probeModePtr == nullptr) {
        OPS_LOG_E(nodeName, "probe_mode attr is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* mAttrPtr = attrs->GetAttrPointer<int64_t>(1);
    if (mAttrPtr == nullptr) {
        OPS_LOG_E(nodeName, "m attr is null");
        return ge::GRAPH_FAILED;
    }
    const uint32_t probeMode = static_cast<uint32_t>(*probeModePtr);
    if (probeMode > 1) {
        OPS_LOG_E(nodeName, "probe_mode must be 0 (KFC regist) or 1 (GM matmul)");
        return ge::GRAPH_FAILED;
    }

    const uint32_t m = static_cast<uint32_t>(*mAttrPtr);
    if (m < 1 || m > 128) {
        OPS_LOG_E(nodeName, "m must be in [1, 128]");
        return ge::GRAPH_FAILED;
    }
    const uint32_t mPad = AlignUp16(m);
    if (static_cast<uint32_t>(storageShape.GetDim(0)) != mPad) {
        OPS_LOG_E(nodeName, "input a rows must be M_pad=%u (got %ld)", mPad, storageShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tilingApi(ascendcPlatform);

    if (probeMode == 0) {
        // Match fused TurboQuant rotate: 2 AIV clients + AIC for KFC handshake.
        tilingApi.SetDim(2);
        tilingApi.SetAType(
            matmul_tiling::TPosition::VECOUT,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16,
            false);
        tilingApi.SetBType(
            matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16,
            false);
        tilingApi.SetCType(
            matmul_tiling::TPosition::VECIN,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16);
    } else {
        // Pure GM cube matmul, no KFC participants.
        tilingApi.SetDim(1);
        tilingApi.SetAType(
            matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16,
            false);
        tilingApi.SetBType(
            matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16,
            false);
        tilingApi.SetCType(
            matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16);
    }
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);

    TurboquantRotateMatmulProbeTilingData tilingData {};
    ge::graphStatus tilingStatus = ge::GRAPH_FAILED;
    if (probeMode == 0) {
        tilingStatus = FillKfcCubeTiling(tilingApi, mPad, tilingData.cubeTiling);
    } else {
        tilingStatus = FillGmCubeTiling(ascendcPlatform, mPad, tilingData.cubeTiling);
    }
    if (tilingStatus != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "failed to get cube tiling");
        return ge::GRAPH_FAILED;
    }
    tilingData.set_m(m);
    tilingData.set_probeMode(probeMode);

    uint32_t blockDim = 1;
    if (probeMode == 0) {
        // MIX 1C2V: one AIC + two AIV (SetDim(2)). TSCH block dim must reflect AIC/AIV ratio.
        const uint32_t kfcAicNum = 1;
        const uint32_t kfcAivNum = 2;
        blockDim = ascendcPlatform.CalcTschBlockDim(kfcAivNum, kfcAicNum, kfcAivNum);
    }

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tilingData.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tilingData.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = SYSTEM_NEED_WORKSPACE;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(probeMode == 0 ? TQ_PROBE_TILING_KEY_KFC : TQ_PROBE_TILING_KEY_GM);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantRotateMatmulProbe)
    .Tiling(TurboquantRotateMatmulProbeTilingFunc);

}  // namespace optiling
