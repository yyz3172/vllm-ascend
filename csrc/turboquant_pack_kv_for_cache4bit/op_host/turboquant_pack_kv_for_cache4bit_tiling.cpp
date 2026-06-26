#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "turboquant_pack_kv_for_cache4bit_tiling.h"

namespace {
constexpr uint32_t TQ_PACK_M_ALIGN = 16;
constexpr uint32_t TQ_PACK_N = 128;
constexpr uint32_t TQ_PACK_K = 128;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t TQ_PACK_TILING_KEY_DEFAULT = 0;
constexpr int32_t TQ_PACK_MAX_BASEM = 32;
constexpr uint32_t TQ_PACK_MAX_BATCH_M = 64;

uint32_t AlignUp16(uint32_t x)
{
    return (x + TQ_PACK_M_ALIGN - 1) / TQ_PACK_M_ALIGN * TQ_PACK_M_ALIGN;
}

matmul_tiling::DataType ToMatmulDtype(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 ? matmul_tiling::DataType::DT_BF16
                                : matmul_tiling::DataType::DT_FLOAT16;
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

    // KFC mode: A from VECOUT (L1), B from GM, C to VECIN (L1).
    // L1 holds both xBatch (A input) and yBatch (C output), so constrain baseM.
    constexpr uint32_t mmDataTypeSize = 2;  // fp16/bf16
    const uint32_t l1Usable = static_cast<uint32_t>(l1Size);
    // baseM*baseK + baseM*baseN <= l1Usable (A + C share L1)
    // For N=128, K=128: baseM*128*2*2 <= l1Usable => baseM <= l1Usable / 512
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

    // Override with L1-aware baseM; keep auto baseN/baseK from GetTiling.
    // Force usedCoreNum=1: each AIV worker calls IterateAll independently.
    cubeTiling.set_baseM(static_cast<uint32_t>(baseM));
    cubeTiling.set_usedCoreNum(1);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace optiling {

static ge::graphStatus TurboquantPackKvForCache4bitTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* nVecPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* vecPerCorePtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t* numBlocksPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t* numReqsPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t* keyStrideTokenPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t* keyStrideHeadPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t* valueStrideTokenPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t* valueStrideHeadPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t* keyStorageOffsetPtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t* valueStorageOffsetPtr = attrs->GetAttrPointer<int64_t>(11);
    if (nVecPtr == nullptr || vecPerCorePtr == nullptr || numHeadsPtr == nullptr ||
        blockSizePtr == nullptr || numBlocksPtr == nullptr || numReqsPtr == nullptr ||
        keyStrideTokenPtr == nullptr || keyStrideHeadPtr == nullptr ||
        valueStrideTokenPtr == nullptr || valueStrideHeadPtr == nullptr ||
        keyStorageOffsetPtr == nullptr || valueStorageOffsetPtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }

    const uint32_t nVec = static_cast<uint32_t>(*nVecPtr);
    uint32_t vecPerCore = static_cast<uint32_t>(*vecPerCorePtr);
    const uint32_t numHeads = static_cast<uint32_t>(*numHeadsPtr);
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t numBlocks = static_cast<uint32_t>(*numBlocksPtr);
    const uint32_t numReqs = static_cast<uint32_t>(*numReqsPtr);
    const uint32_t keyStrideToken = static_cast<uint32_t>(*keyStrideTokenPtr);
    const uint32_t keyStrideHead = static_cast<uint32_t>(*keyStrideHeadPtr);
    const uint32_t valueStrideToken = static_cast<uint32_t>(*valueStrideTokenPtr);
    const uint32_t valueStrideHead = static_cast<uint32_t>(*valueStrideHeadPtr);
    if (*keyStorageOffsetPtr < 0 || *valueStorageOffsetPtr < 0) {
        OPS_LOG_E(nodeName, "key/value storage offsets must be non-negative");
        return ge::GRAPH_FAILED;
    }
    const uint64_t keyStorageOffset = static_cast<uint64_t>(*keyStorageOffsetPtr);
    const uint64_t valueStorageOffset = static_cast<uint64_t>(*valueStorageOffsetPtr);
    if (nVec < 1 || vecPerCore < 1 || numHeads < 1 ||
        blockSize < 1 || numBlocks < 1 || numReqs < 1 ||
        keyStrideToken < 1 || keyStrideHead < 1 ||
        valueStrideToken < 1 || valueStrideHead < 1) {
        OPS_LOG_E(nodeName,
                  "invalid pack attrs: n_vec/vec_per_core/num_heads/block_size/"
                  "num_blocks/num_reqs/strides must be positive");
        return ge::GRAPH_FAILED;
    }
    if (blockSize % 4 != 0) {
        OPS_LOG_E(nodeName, "4-bit group cache layout requires block_size multiple of 4, got %u",
                  blockSize);
        return ge::GRAPH_FAILED;
    }
    if (vecPerCore > TQ_PACK_MAX_BATCH_M) {
        vecPerCore = TQ_PACK_MAX_BATCH_M;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tilingApi(ascendcPlatform);
    auto keyDesc = context->GetInputDesc(0);
    auto valueDesc = context->GetInputDesc(1);
    auto codebookDesc = context->GetInputDesc(2);
    auto rotationDesc = context->GetInputDesc(3);
    if (keyDesc == nullptr || valueDesc == nullptr || codebookDesc == nullptr ||
        rotationDesc == nullptr) {
        OPS_LOG_E(nodeName, "input desc is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dataType = keyDesc->GetDataType();
    if ((dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16) ||
        valueDesc->GetDataType() != dataType ||
        codebookDesc->GetDataType() != dataType ||
        rotationDesc->GetDataType() != dataType) {
        OPS_LOG_E(nodeName, "key/value/codebook/rotation_t must share fp16 or bf16 dtype");
        return ge::GRAPH_FAILED;
    }
    const matmul_tiling::DataType mmDataType = ToMatmulDtype(dataType);

    const uint32_t mPad = AlignUp16(vecPerCore);
    tilingApi.SetDim(2);
    tilingApi.SetAType(
        matmul_tiling::TPosition::VECOUT,
        matmul_tiling::CubeFormat::ND,
        mmDataType,
        false);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        mmDataType,
        false);
    tilingApi.SetCType(
        matmul_tiling::TPosition::VECIN,
        matmul_tiling::CubeFormat::ND,
        mmDataType);
    tilingApi.SetBiasType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        mmDataType);

    TurboquantPackKvForCache4bitTilingData tilingData {};
    ge::graphStatus tilingStatus = FillKfcCubeTiling(
        ascendcPlatform, tilingApi, mPad, tilingData.cubeTiling);
    if (tilingStatus != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "failed to get cube tiling");
        return ge::GRAPH_FAILED;
    }

    tilingData.set_nVec(nVec);
    tilingData.set_vecPerCore(vecPerCore);
    tilingData.set_numHeads(numHeads);
    tilingData.set_blockSize(blockSize);
    tilingData.set_numBlocks(numBlocks);
    tilingData.set_numReqs(numReqs);
    tilingData.set_keyStrideToken(keyStrideToken);
    tilingData.set_keyStrideHead(keyStrideHead);
    tilingData.set_valueStrideToken(valueStrideToken);
    tilingData.set_valueStrideHead(valueStrideHead);
    tilingData.set_keyStorageOffset(keyStorageOffset);
    tilingData.set_valueStorageOffset(valueStorageOffset);

    // MIX 1C2V per data-parallel group.  Use all physical groups that the
    // current SOC can provide instead of fixing the kernel to 16 groups.
    const uint32_t kfcAicNum = 1;
    const uint32_t kfcAivNum = 2;
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aicNum == 0 || aivNum < kfcAivNum) {
        OPS_LOG_E(nodeName, "invalid core count for MIX 1C2V: aic=%u, aiv=%u", aicNum, aivNum);
        return ge::GRAPH_FAILED;
    }
    const uint32_t dataCores = std::min(aicNum / kfcAicNum, aivNum / kfcAivNum);
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(
        dataCores * kfcAivNum, dataCores * kfcAicNum, dataCores * kfcAivNum);
    tilingData.set_dataCores(dataCores);

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
    // KFC message queues and CANN internal workspace.
    workspaces[0] = SYSTEM_NEED_WORKSPACE;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(TQ_PACK_TILING_KEY_DEFAULT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantPackKvForCache4bit)
    .Tiling(TurboquantPackKvForCache4bitTilingFunc);

}  // namespace optiling
