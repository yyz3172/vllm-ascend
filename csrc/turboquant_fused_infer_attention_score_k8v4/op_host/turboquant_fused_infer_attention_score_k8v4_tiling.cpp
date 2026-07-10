#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "turboquant_fused_infer_attention_score_k8v4_tiling.h"
#include <algorithm>

namespace {

constexpr uint32_t TQ_M_ALIGN = 16;
constexpr uint32_t TQ_KFC_AIC_NUM = 1;
constexpr uint32_t TQ_KFC_AIV_NUM = 2;
constexpr size_t MAX_USER_WORKSPACE = 256ULL * 1024ULL * 1024ULL;

uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return (x + align - 1) / align * align;
}

uint32_t PickKvTileRows(uint32_t blockSize)
{
    uint32_t tile = optiling::TQ_K8V4_KV_TILE_ROWS;
    if (blockSize > 0 && blockSize < tile) {
        tile = blockSize;
    }
    tile = std::min(tile, optiling::TQ_K8V4_UB_KV_TILE_CAP);
    return std::max(tile, 1U);
}

ge::graphStatus FillDecodeRotateTiling(
    const char* nodeName,
    const platform_ascendc::PlatformAscendC& platform,
    uint32_t kvTileRows,
    optiling::TCubeTiling& cubeTiling)
{
    uint64_t l1Size = 0, l0aSize = 0, l0bSize = 0, l0cSize = 0, ubSize = 0;
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

    matmul_tiling::MultiCoreMatmulTiling tilingApi(platformInfo);
    tilingApi.SetDim(2);
    tilingApi.SetAType(matmul_tiling::TPosition::VECOUT,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM,
                          matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_FLOAT16);
    const uint32_t mPad = AlignUp(kvTileRows, TQ_M_ALIGN);
    tilingApi.SetOrgShape(mPad, optiling::TQ_K8V4_HEAD_SIZE, optiling::TQ_K8V4_HEAD_SIZE);
    tilingApi.SetShape(mPad, optiling::TQ_K8V4_HEAD_SIZE, optiling::TQ_K8V4_HEAD_SIZE);
    tilingApi.EnableBias(false);
    tilingApi.SetBufferSpace(l1Size, l0cSize, ubSize);
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        OPS_LOG_E(nodeName, "decode rotate GetTiling failed");
        return ge::GRAPH_FAILED;
    }
    cubeTiling.set_baseM(mPad);
    cubeTiling.set_usedCoreNum(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillQkTiling(
    const char* nodeName,
    const platform_ascendc::PlatformAscendC& platform,
    uint32_t gqaGroup,
    uint32_t kvTileRows,
    optiling::TCubeTiling& cubeTiling)
{
    uint64_t l1Size = 0, l0aSize = 0, l0bSize = 0, l0cSize = 0, ubSize = 0;
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

    matmul_tiling::MatmulApiTiling bmm1(platformInfo);
    const uint32_t m = AlignUp(gqaGroup, TQ_M_ALIGN);
    bmm1.SetShape(m, kvTileRows, optiling::TQ_K8V4_HEAD_SIZE);
    bmm1.SetOrgShape(m, kvTileRows, optiling::TQ_K8V4_HEAD_SIZE, optiling::TQ_K8V4_HEAD_SIZE);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT);
    bmm1.SetBias(false);
    const uint32_t baseN = std::min(AlignUp(kvTileRows, TQ_M_ALIGN), 256U);
    if (bmm1.SetFixSplit(m, baseN) == -1) {
        OPS_LOG_E(nodeName, "qk SetFixSplit failed");
        return ge::GRAPH_FAILED;
    }
    if (bmm1.GetTiling(cubeTiling) == -1) {
        OPS_LOG_E(nodeName, "qk GetTiling failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillPvTiling(
    const char* nodeName,
    const platform_ascendc::PlatformAscendC& platform,
    uint32_t gqaGroup,
    uint32_t kvTileRows,
    optiling::TCubeTiling& cubeTiling)
{
    uint64_t l1Size = 0, l0aSize = 0, l0bSize = 0, l0cSize = 0, ubSize = 0;
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

    matmul_tiling::MatmulApiTiling bmm2(platformInfo);
    const uint32_t m = AlignUp(gqaGroup, TQ_M_ALIGN);
    bmm2.SetShape(m, optiling::TQ_K8V4_HEAD_SIZE, kvTileRows);
    bmm2.SetOrgShape(m, optiling::TQ_K8V4_HEAD_SIZE, kvTileRows, kvTileRows);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                  matmul_tiling::DataType::DT_FLOAT);
    bmm2.SetBias(false);
    if (bmm2.SetFixSplit(m) == -1) {
        OPS_LOG_E(nodeName, "pv SetFixSplit failed");
        return ge::GRAPH_FAILED;
    }
    if (bmm2.GetTiling(cubeTiling) == -1) {
        OPS_LOG_E(nodeName, "pv GetTiling failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t PickQkPvMode(uint32_t gqaGroup, uint32_t kvTileRows)
{
    if (gqaGroup >= optiling::TQ_K8V4_CUBE_MIN_G &&
        kvTileRows >= optiling::TQ_K8V4_CUBE_MIN_TILE) {
        return optiling::TQ_K8V4_QKPV_CUBE;
    }
    return optiling::TQ_K8V4_QKPV_VECTOR;
}

size_t CalcWorkspaceSize(const platform_ascendc::PlatformAscendC& platform)
{
    size_t ws = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    if (ws < 16U * 1024U * 1024U) {
        ws = 16U * 1024U * 1024U;
    }
    // Per-core GM scratch: decode fp32 C + QK/PV cube operands (see kernel offsets).
    constexpr size_t kDecodeStride =
        static_cast<size_t>(optiling::TQ_K8V4_UB_KV_TILE_CAP) * optiling::TQ_K8V4_HEAD_SIZE *
        sizeof(float);
    constexpr size_t kQkPvStride =
        static_cast<size_t>(optiling::TQ_K8V4_UB_GQA_CAP) * optiling::TQ_K8V4_UB_KV_TILE_CAP *
            sizeof(float) +
        static_cast<size_t>(optiling::TQ_K8V4_UB_GQA_CAP) * optiling::TQ_K8V4_HEAD_SIZE *
            sizeof(uint16_t) * 2 +
        static_cast<size_t>(optiling::TQ_K8V4_UB_KV_TILE_CAP) * optiling::TQ_K8V4_HEAD_SIZE *
            sizeof(uint16_t) +
        static_cast<size_t>(optiling::TQ_K8V4_UB_GQA_CAP) * optiling::TQ_K8V4_HEAD_SIZE *
            sizeof(float);
    constexpr size_t kPerCore = kDecodeStride + kQkPvStride;
    ws += static_cast<size_t>(256) * 1024U;
    ws += static_cast<size_t>(optiling::TQ_K8V4_MAX_PARALLEL_CORES) * kPerCore;
    return ws;
}

void SplitBn(uint32_t bn, uint32_t coreNum, uint32_t& usedCoreNum,
             uint32_t& formerCoreNum, uint32_t& blockSplitRange, uint32_t& tailSplitRange)
{
    usedCoreNum = (bn > coreNum) ? coreNum : bn;
    if (usedCoreNum == 0) {
        formerCoreNum = 0;
        blockSplitRange = 0;
        tailSplitRange = 0;
        return;
    }
    formerCoreNum = bn % usedCoreNum;
    if (formerCoreNum == 0) {
        blockSplitRange = bn / usedCoreNum;
        tailSplitRange = blockSplitRange;
    } else {
        blockSplitRange = bn / usedCoreNum + 1;
        tailSplitRange = blockSplitRange - 1;
    }
}

// actual_seq_len tensors live on NPU; host tiling must not call GetData() on them
// (device pointers are not host-readable and will segfault). Shape/dtype only.
ge::graphStatus CheckActualSeqLenShape(const char* nodeName,
                                       const char* tensorName,
                                       const gert::StorageShape* shapePtr,
                                       const gert::CompileTimeTensorDesc* descPtr,
                                       uint32_t batchSize)
{
    if (shapePtr == nullptr || descPtr == nullptr) {
        OPS_LOG_E(nodeName, "%s shape or desc is null", tensorName);
        return ge::GRAPH_FAILED;
    }
    if (descPtr->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(nodeName, "%s dtype must be int32", tensorName);
        return ge::GRAPH_FAILED;
    }
    const auto shape = shapePtr->GetStorageShape();
    if (shape.GetDimNum() != 1 || shape.GetDim(0) != static_cast<int64_t>(batchSize)) {
        OPS_LOG_E(nodeName, "%s size must equal batch_size(%u)", tensorName, batchSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace optiling {

static ge::graphStatus TurboquantFusedInferAttentionScoreK8v4TilingFunc(gert::TilingContext* context)
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

    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* headSizePtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(3);
    const float* scaleValuePtr = attrs->GetAttrPointer<float>(4);
    const int64_t* cacheNormBf16Ptr = attrs->GetAttrPointer<int64_t>(5);
    if (numHeadsPtr == nullptr || numKvHeadsPtr == nullptr || headSizePtr == nullptr ||
        blockSizePtr == nullptr || scaleValuePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }

    auto qShapePtr = context->GetInputShape(0);
    auto kcShapePtr = context->GetInputShape(1);
    auto vcShapePtr = context->GetInputShape(2);
    auto btShapePtr = context->GetInputShape(3);
    if (qShapePtr == nullptr || kcShapePtr == nullptr || vcShapePtr == nullptr ||
        btShapePtr == nullptr) {
        OPS_LOG_E(nodeName, "required input shapes are null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape kcShape = kcShapePtr->GetStorageShape();
    const gert::Shape vcShape = vcShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (qShape.GetDimNum() < 2 || btShape.GetDimNum() < 2 || kcShape.GetDimNum() < 1) {
        OPS_LOG_E(nodeName, "unexpected input ranks");
        return ge::GRAPH_FAILED;
    }

    const uint32_t numTokens = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t numHeads = static_cast<uint32_t>(*numHeadsPtr);
    const uint32_t numKvHeads = static_cast<uint32_t>(*numKvHeadsPtr);
    const uint32_t headSize = static_cast<uint32_t>(*headSizePtr);
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t batchSize = static_cast<uint32_t>(btShape.GetDim(0));
    const uint32_t maxBlocksPerSeq = static_cast<uint32_t>(btShape.GetDim(1));

    if (headSize != TQ_K8V4_HEAD_SIZE) {
        OPS_LOG_E(nodeName, "only head_size=128 supported, got %u", headSize);
        return ge::GRAPH_FAILED;
    }
    if (numHeads == 0 || numKvHeads == 0 || numHeads % numKvHeads != 0) {
        OPS_LOG_E(nodeName, "invalid head configuration");
        return ge::GRAPH_FAILED;
    }

    uint32_t keyRowBytes = TQ_K8V4_K_PACKED_BYTES;
    uint32_t valueRowBytes = TQ_K8V4_V_PACKED_BYTES;
    if (kcShape.GetDimNum() >= 1) {
        keyRowBytes = static_cast<uint32_t>(kcShape.GetDim(kcShape.GetDimNum() - 1));
    }
    if (vcShape.GetDimNum() >= 1) {
        valueRowBytes = static_cast<uint32_t>(vcShape.GetDim(vcShape.GetDimNum() - 1));
    }
    if (keyRowBytes < TQ_K8V4_K_PACKED_BYTES || valueRowBytes < TQ_K8V4_V_PACKED_BYTES) {
        OPS_LOG_E(nodeName, "invalid cache row bytes: key=%u value=%u", keyRowBytes, valueRowBytes);
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalCacheBlocks =
        (kcShape.GetDimNum() >= 4) ? static_cast<uint32_t>(kcShape.GetDim(0)) : 0U;
    const uint32_t maxKvLen = maxBlocksPerSeq * blockSize;
    const uint32_t gqaGroup = numHeads / numKvHeads;
    if (gqaGroup > TQ_K8V4_UB_GQA_CAP) {
        OPS_LOG_E(nodeName, "gqa_group %u exceeds kernel UB cap %u", gqaGroup, TQ_K8V4_UB_GQA_CAP);
        return ge::GRAPH_FAILED;
    }

    if (CheckActualSeqLenShape(nodeName, "actual_seq_len_q", context->GetInputShape(4),
                               context->GetInputDesc(4), batchSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckActualSeqLenShape(nodeName, "actual_seq_len_kv", context->GetInputShape(5),
                               context->GetInputDesc(5), batchSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    (void)numTokens;
    (void)maxKvLen;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t coreNum = std::max(1U, aicNum);
    const uint32_t parallelCoreNum = std::min(coreNum, TQ_K8V4_MAX_PARALLEL_CORES);
    const uint32_t bn = numTokens * numKvHeads;
    const uint32_t kvTileRows = PickKvTileRows(blockSize);

    const uint32_t qkPvMode = PickQkPvMode(gqaGroup, kvTileRows);

    TurboquantFusedInferAttentionScoreK8v4TilingData tiling{};
    if (FillDecodeRotateTiling(nodeName, ascendcPlatform, kvTileRows, tiling.decodeRotateTiling) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qkPvMode == optiling::TQ_K8V4_QKPV_CUBE) {
        if (FillQkTiling(nodeName, ascendcPlatform, gqaGroup, kvTileRows, tiling.qkTiling) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (FillPvTiling(nodeName, ascendcPlatform, gqaGroup, kvTileRows, tiling.pvTiling) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t usedCoreNum = 0;
    uint32_t formerCoreNum = 0;
    uint32_t blockSplitRange = 0;
    uint32_t tailSplitRange = 0;
    SplitBn(bn, parallelCoreNum, usedCoreNum, formerCoreNum, blockSplitRange, tailSplitRange);

    tiling.set_numTokens(numTokens);
    tiling.set_batchSize(batchSize);
    tiling.set_numHeads(numHeads);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_gqaGroupSize(gqaGroup);
    tiling.set_headSize(headSize);
    tiling.set_blockSize(blockSize);
    tiling.set_maxBlocksPerSeq(maxBlocksPerSeq);
    tiling.set_totalCacheBlocks(totalCacheBlocks);
    tiling.set_keyRowBytes(keyRowBytes);
    tiling.set_valueRowBytes(valueRowBytes);
    tiling.set_kvTileRows(kvTileRows);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_blockSplitRange(blockSplitRange);
    tiling.set_tailSplitRange(tailSplitRange);
    tiling.set_qkPvMode(qkPvMode);
    tiling.set_scaleValue(*scaleValuePtr);
    const uint32_t cacheNormBf16 =
        (cacheNormBf16Ptr != nullptr && *cacheNormBf16Ptr != 0) ? 1U : 0U;
    tiling.set_cacheNormBf16(cacheNormBf16);

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
    workspaces[0] = CalcWorkspaceSize(ascendcPlatform);
    if (workspaces[0] > MAX_USER_WORKSPACE) {
        OPS_LOG_E(nodeName, "workspace size %zu exceeds cap", workspaces[0]);
        return ge::GRAPH_FAILED;
    }

    const uint32_t mixBlockDim =
        ascendcPlatform.CalcTschBlockDim(TQ_KFC_AIV_NUM, TQ_KFC_AIC_NUM, TQ_KFC_AIV_NUM);
    const uint32_t blockDim = std::max(1U, usedCoreNum) * mixBlockDim;
    context->SetBlockDim(blockDim);
    // Cube QK/PV kernel path uses a separate tiling key; keep key=0 until multi-matmul
    // registration is split into its own TU (CANN Mc2 workspace constraint).
    context->SetTilingKey(optiling::TQ_K8V4_KEY_VECTOR);

    OPS_LOG_I(nodeName,
              "TurboquantFusedInferAttentionScoreK8v4 tiling: tokens=%u bn=%u kvTile=%u "
              "usedCore=%u qkpv=%u key=%lu keyRow=%u valueRow=%u ws=%zu",
              numTokens, bn, kvTileRows, usedCoreNum, qkPvMode,
              static_cast<unsigned long>(optiling::TQ_K8V4_KEY_VECTOR), keyRowBytes,
              valueRowBytes, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantFusedInferAttentionScoreK8v4)
    .Tiling(TurboquantFusedInferAttentionScoreK8v4TilingFunc);

}  // namespace optiling
