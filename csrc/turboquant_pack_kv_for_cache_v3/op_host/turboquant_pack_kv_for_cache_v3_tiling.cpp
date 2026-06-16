#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "turboquant_pack_kv_for_cache_v3_tiling.h"

namespace {
constexpr uint32_t TQ_PACK_TILING_KEY_AIV = 0;
constexpr uint32_t TQ_PACK_AIV_CORES = 16;
constexpr uint32_t TQ_PACK_MAX_BATCH_M = 64;
}  // namespace

namespace optiling {

static ge::graphStatus TurboquantPackKvForCacheV3TilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* nVecPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* slotWKPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* slotWVPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* vecPerCorePtr = attrs->GetAttrPointer<int64_t>(3);
    if (nVecPtr == nullptr || slotWKPtr == nullptr || slotWVPtr == nullptr || vecPerCorePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }
    if (*nVecPtr < 1 || *slotWKPtr < 1 || *slotWVPtr < 1 || *vecPerCorePtr < 1) {
        OPS_LOG_E(nodeName, "invalid pack attrs");
        return ge::GRAPH_FAILED;
    }

    const uint32_t nVec = static_cast<uint32_t>(*nVecPtr);
    const uint32_t slotWK = static_cast<uint32_t>(*slotWKPtr);
    const uint32_t slotWV = static_cast<uint32_t>(*slotWVPtr);
    uint32_t vecPerCore = static_cast<uint32_t>(*vecPerCorePtr);
    if (vecPerCore > TQ_PACK_MAX_BATCH_M) {
        vecPerCore = TQ_PACK_MAX_BATCH_M;
    }

    TurboquantPackKvForCacheV3TilingData tilingData {};
    tilingData.set_nVec(nVec);
    tilingData.set_vecPerCore(vecPerCore);
    tilingData.set_slotWK(slotWK);
    tilingData.set_slotWV(slotWV);

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
    workspaces[0] = 0;

    context->SetBlockDim(TQ_PACK_AIV_CORES);
    context->SetTilingKey(TQ_PACK_TILING_KEY_AIV);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantPackKvForCacheV3)
    .Tiling(TurboquantPackKvForCacheV3TilingFunc);

}  // namespace optiling
