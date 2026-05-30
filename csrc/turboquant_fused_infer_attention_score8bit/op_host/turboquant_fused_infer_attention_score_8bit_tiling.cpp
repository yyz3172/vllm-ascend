#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "turboquant_fused_infer_attention_score_8bit_tiling.h"
#include <algorithm>

namespace optiling {

static ge::graphStatus TurboquantFusedInferAttentionScore8bitTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OPS_LOG_E(nodeName, "platformInfo is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());

    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* headSizePtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(3);
    const float* scaleValuePtr = attrs->GetAttrPointer<float>(4);
    if (numHeadsPtr == nullptr || numKvHeadsPtr == nullptr || headSizePtr == nullptr ||
        blockSizePtr == nullptr || scaleValuePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }

    auto qShapePtr = context->GetInputShape(0);
    auto btShapePtr = context->GetInputShape(3);
    if (qShapePtr == nullptr || btShapePtr == nullptr) {
        OPS_LOG_E(nodeName, "required input shapes are null");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (qShape.GetDimNum() < 2 || btShape.GetDimNum() < 2) {
        OPS_LOG_E(nodeName, "unexpected input ranks: query=%zu block_table=%zu",
                  qShape.GetDimNum(), btShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens64 = qShape.GetDim(0);
    const int64_t numHeadsShape64 = qShape.GetDim(1);
    const int64_t headSize = *headSizePtr;
    if (headSize != 128) {
        OPS_LOG_E(nodeName, "only head_size=128 supported, got %ld", headSize);
        return ge::GRAPH_FAILED;
    }
    if (numTokens64 <= 0 || numHeadsShape64 <= 0) {
        OPS_LOG_E(nodeName, "invalid query shape");
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize64 = btShape.GetDim(0);
    const int64_t maxBlocksPerSeq64 = btShape.GetDim(1);
    if (batchSize64 <= 0) {
        OPS_LOG_E(nodeName, "invalid block_table batch size");
        return ge::GRAPH_FAILED;
    }
    if (maxBlocksPerSeq64 <= 0) {
        OPS_LOG_E(nodeName, "invalid block_table shape");
        return ge::GRAPH_FAILED;
    }

    TurboquantFusedInferAttentionScore8bitTilingData tiling {};
    const uint64_t totalTasks = static_cast<uint64_t>(numTokens64) * static_cast<uint64_t>(*numHeadsPtr);
    const uint32_t blockDim = (coreNum == 0) ? 1U : static_cast<uint32_t>(std::min<uint64_t>(coreNum, totalTasks));

    tiling.set_blockDim(blockDim);
    tiling.set_numTokens(static_cast<uint32_t>(numTokens64));
    tiling.set_batchSize(static_cast<uint32_t>(batchSize64));
    tiling.set_numHeads(static_cast<uint32_t>(*numHeadsPtr));
    tiling.set_numKvHeads(static_cast<uint32_t>(*numKvHeadsPtr));
    tiling.set_headSize(static_cast<uint32_t>(*headSizePtr));
    tiling.set_blockSize(static_cast<uint32_t>(*blockSizePtr));
    tiling.set_maxBlocksPerSeq(static_cast<uint32_t>(maxBlocksPerSeq64));
    tiling.set_scaleValue(*scaleValuePtr);

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tiling.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0;

    context->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantFusedInferAttentionScore8bit)
    .Tiling(TurboquantFusedInferAttentionScore8bitTilingFunc);

}  // namespace optiling

