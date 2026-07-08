#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "turboquant_pack_kv_for_cache_k8v4_tiling.h"

namespace {
constexpr uint32_t TQ_PACK_M_ALIGN = 16;
constexpr uint32_t TQ_PACK_N = 128;
constexpr uint32_t TQ_PACK_K = 128;
constexpr uint32_t TQ_PACK_SINGLE_ROT_M_PAD = 16;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t TQ_PACK_TILING_KEY_KFC = 0;
constexpr uint32_t TQ_PACK_TILING_KEY_AIC_GM = 1;
constexpr int32_t TQ_PACK_BEST_BASEN = 256;
constexpr uint64_t TQ_PACK_DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t TQ_PACK_DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t TQ_PACK_FP32_SIZE = 4;
constexpr int32_t TQ_PACK_MAX_BASEM = 256;

uint32_t AlignUp16(uint32_t x)
{
    return (x + TQ_PACK_M_ALIGN - 1) / TQ_PACK_M_ALIGN * TQ_PACK_M_ALIGN;
}

ge::graphStatus FillKfcCubeTiling(
    const platform_ascendc::PlatformAscendC& platform,
    matmul_tiling::MultiCoreMatmulTiling& tilingApi,
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

    constexpr uint32_t mmDataTypeSize = 2;  // fp16
    const uint32_t l1Usable = static_cast<uint32_t>(l1Size);
    int32_t baseM = static_cast<int32_t>(std::min<uint64_t>(
        l1Usable / (TQ_PACK_K + TQ_PACK_N) / mmDataTypeSize,
        static_cast<uint64_t>(TQ_PACK_MAX_BASEM)));
    baseM = static_cast<int32_t>(AlignUp16(static_cast<uint32_t>(baseM)));
    if (baseM > static_cast<int32_t>(mPad)) {
        baseM = static_cast<int32_t>(AlignUp16(mPad));
    }
    if (baseM <= 0) {
        baseM = static_cast<int32_t>(AlignUp16(mPad));
    }

    tilingApi.SetOrgShape(mPad, TQ_PACK_N, TQ_PACK_K);
    tilingApi.SetShape(mPad, TQ_PACK_N, TQ_PACK_K);
    tilingApi.EnableBias(false);
    tilingApi.SetBufferSpace(l1Size, l0cSize, ubSize);
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }

    cubeTiling.set_baseM(static_cast<uint32_t>(baseM));
    cubeTiling.set_usedCoreNum(1);
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

    const int32_t baseN = std::min<int32_t>(TQ_PACK_BEST_BASEN, static_cast<int32_t>(TQ_PACK_N));
    constexpr uint32_t mmDataTypeSize = 2;
    int32_t baseK = static_cast<int32_t>(
        (l0bSize / TQ_PACK_DOUBLE_BUFFER_L0A_L0B) / (static_cast<uint64_t>(baseN) * mmDataTypeSize));
    baseK = static_cast<int32_t>(AlignUp16(static_cast<uint32_t>(baseK)));
    const uint32_t maxBaseM = static_cast<uint32_t>(l0cSize / (baseN * TQ_PACK_FP32_SIZE));
    int32_t baseM = static_cast<int32_t>(std::min<uint64_t>(
        (l0aSize / TQ_PACK_DOUBLE_BUFFER_L0A_L0B) / (static_cast<uint64_t>(baseK) * mmDataTypeSize),
        maxBaseM));
    baseM = static_cast<int32_t>(
        baseM > static_cast<int32_t>(mPad) ? AlignUp16(mPad) : AlignUp16(static_cast<uint32_t>(baseM)));
    if (baseM > TQ_PACK_MAX_BASEM) {
        baseM = TQ_PACK_MAX_BASEM;
    }
    if (baseM <= 0 || baseK <= 0) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MultiCoreMatmulTiling tilingApi(platformInfo);
    tilingApi.SetDim(1);
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetOrgShape(mPad, TQ_PACK_N, TQ_PACK_K);
    tilingApi.SetShape(mPad, TQ_PACK_N, TQ_PACK_K);
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
    const uint32_t mmDepthA1 = mmStepKa * TQ_PACK_DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    const uint32_t mmDepthB1 = mmStepKb * TQ_PACK_DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
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

static ge::graphStatus TurboquantPackKvForCacheK8v4TilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* packModePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* nVecPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* slotWKPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* slotWVPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t* vecPerCorePtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t* cacheSlotsPtr = attrs->GetAttrPointer<int64_t>(6);
    if (packModePtr == nullptr || nVecPtr == nullptr || slotWKPtr == nullptr || slotWVPtr == nullptr ||
        vecPerCorePtr == nullptr || numHeadsPtr == nullptr || cacheSlotsPtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }

    const uint32_t packMode = static_cast<uint32_t>(*packModePtr);
    const uint32_t nVec = static_cast<uint32_t>(*nVecPtr);
    const uint32_t slotWK = static_cast<uint32_t>(*slotWKPtr);
    const uint32_t slotWV = static_cast<uint32_t>(*slotWVPtr);
    uint32_t vecPerCore = static_cast<uint32_t>(*vecPerCorePtr);
    const uint32_t numHeads = static_cast<uint32_t>(*numHeadsPtr);
    const uint32_t cacheSlots = static_cast<uint32_t>(*cacheSlotsPtr);
    if (packMode > 1 || nVec < 1 || vecPerCore < 1 || numHeads < 1 || cacheSlots < 1) {
        OPS_LOG_E(nodeName, "invalid pack attrs");
        return ge::GRAPH_FAILED;
    }
    if (vecPerCore > 128) {
        vecPerCore = 128;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tilingApi(ascendcPlatform);

    uint32_t mPad = AlignUp16(vecPerCore);
    if (packMode == 0) {
        tilingApi.SetDim(2);
        tilingApi.SetAType(
            matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16, false);
        tilingApi.SetBType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16, false);
        tilingApi.SetCType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT);
    } else {
        mPad = TQ_PACK_SINGLE_ROT_M_PAD;
        tilingApi.SetDim(1);
        tilingApi.SetAType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16, false);
        tilingApi.SetBType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16, false);
        tilingApi.SetCType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16);
    }
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);

    TurboquantPackKvForCacheK8v4TilingData tilingData {};
    ge::graphStatus tilingStatus = ge::GRAPH_FAILED;
    if (packMode == 0) {
        tilingStatus = FillKfcCubeTiling(ascendcPlatform, tilingApi, mPad, tilingData.cubeTiling);
    } else {
        tilingStatus = FillGmCubeTiling(ascendcPlatform, mPad, tilingData.cubeTiling);
    }
    if (tilingStatus != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "failed to get cube tiling");
        return ge::GRAPH_FAILED;
    }

    tilingData.set_nVec(nVec);
    tilingData.set_vecPerCore(vecPerCore);
    tilingData.set_slotWK(slotWK);
    tilingData.set_slotWV(slotWV);
    tilingData.set_packMode(packMode);
    tilingData.set_numHeads(numHeads);
    tilingData.set_cacheSlots(cacheSlots);

    uint32_t dataCores = std::max<uint32_t>(1, (nVec + vecPerCore - 1) / vecPerCore);
    uint32_t blockDim = dataCores;
    if (packMode == 0) {
        const uint32_t kfcAicNum = 1;
        const uint32_t kfcAivNum = 2;
        const uint32_t mixBlockDim =
            ascendcPlatform.CalcTschBlockDim(kfcAivNum, kfcAicNum, kfcAivNum);
        constexpr uint32_t kMaxMixGroups = 16;
        dataCores = std::min(dataCores, kMaxMixGroups);
        blockDim = dataCores * mixBlockDim;
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
    if (packMode == 0) {
        constexpr uint64_t kPerCoreScratchBase = 512 * 1024;
        constexpr uint64_t kPerCoreScratch = 64 * 1024;
        workspaces[0] = kPerCoreScratchBase + dataCores * kPerCoreScratch;
        if (workspaces[0] < SYSTEM_NEED_WORKSPACE) {
            workspaces[0] = SYSTEM_NEED_WORKSPACE;
        }
    } else {
        workspaces[0] = SYSTEM_NEED_WORKSPACE;
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(packMode == 0 ? TQ_PACK_TILING_KEY_KFC : TQ_PACK_TILING_KEY_AIC_GM);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantPackKvForCacheK8v4)
    .Tiling(TurboquantPackKvForCacheK8v4TilingFunc);

}  // namespace optiling
