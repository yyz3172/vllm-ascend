/*
 * Host tiling for TurboQuant FIA MSE-8bit (TND + PA + GQA).
 *
 * Uses ops-transformer SplitCore for outer split + FlashDecode tables.
 * Tiling keys: 0 = no FD, 1 = FlashDecode (numOfFdHead > 0).
 */

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "turboquant_fia_mse8bit_tiling.h"
#include "vendored/split_core.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace optiling {
namespace {

constexpr uint32_t kPreLoadNum = 2;
constexpr uint32_t kS2BaseSize = 512;
constexpr uint32_t kMBaseSize = 256;
constexpr uint32_t kFdGS1BaseSize = 8;
constexpr uint32_t kByteBlock = 32;
constexpr uint32_t kBlockTableElemByte = 4;
constexpr uint32_t kLibApiWorkspaceFallback = 16 * 1024 * 1024;
constexpr uint64_t kTilingKeyNoFd = 0;
constexpr uint64_t kTilingKeyFd = 1;

inline uint32_t AlignUp(uint32_t a, uint32_t align)
{
    return align == 0 ? a : ((a + align - 1) / align) * align;
}

void ApplySplitResult(TurboquantFiaMse8bitTilingData& tiling, const SplitResult& res,
                      uint32_t aicNum, uint32_t cvRatio)
{
    uint32_t bN2End[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1End[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2End[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t bN2IdxOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1IdxOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2SplitNumOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2SplitStartIdxOfCore[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1SplitNumOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1LastPartSizeOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1IdxEndOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM * 2] = {0};
    uint32_t gS1IdxEndOfFdHeadSplit[TQ_FIA_MAX_AIC_CORE_NUM * 2] = {0};

    const uint32_t n = std::min(res.usedCoreNum, TQ_FIA_MAX_AIC_CORE_NUM);
    for (uint32_t i = 0; i < n; ++i) {
        bN2End[i] = res.bN2End[i];
        gS1End[i] = res.gS1End[i];
        s2End[i] = res.s2End[i];
        bN2IdxOfFdHead[i] = res.fdRes.bN2IdxOfFdHead[i];
        gS1IdxOfFdHead[i] = res.fdRes.gS1IdxOfFdHead[i];
        s2SplitNumOfFdHead[i] = res.fdRes.s2SplitNumOfFdHead[i];
        s2SplitStartIdxOfCore[i] = res.fdRes.s2SplitStartIdxOfCore[i];
        gS1SplitNumOfFdHead[i] = res.fdRes.gS1SplitNumOfFdHead[i];
        gS1LastPartSizeOfFdHead[i] = res.fdRes.gS1LastPartSizeOfFdHead[i];
    }
    const uint32_t nVec = std::min(res.usedCoreNum * cvRatio, TQ_FIA_MAX_AIC_CORE_NUM * 2U);
    for (uint32_t i = 0; i < nVec; ++i) {
        gS1IdxEndOfFdHead[i] = res.fdRes.gS1IdxEndOfFdHead[i];
        gS1IdxEndOfFdHeadSplit[i] = res.fdRes.gS1IdxEndOfFdHeadSplit[i];
    }

    tiling.outerSplitParams.set_bN2End(bN2End);
    tiling.outerSplitParams.set_gS1End(gS1End);
    tiling.outerSplitParams.set_s2End(s2End);

    tiling.fdParams.set_numOfFdHead(res.numOfFdHead);
    tiling.fdParams.set_reserved(0U);
    tiling.fdParams.set_gS1BaseSizeOfFd(kFdGS1BaseSize);
    tiling.fdParams.set_usedVecNumOfFd(res.numOfFdHead > 0U ? res.usedVecNumOfFd : 0U);
    tiling.fdParams.set_bN2IdxOfFdHead(bN2IdxOfFdHead);
    tiling.fdParams.set_gS1IdxOfFdHead(gS1IdxOfFdHead);
    tiling.fdParams.set_s2SplitNumOfFdHead(s2SplitNumOfFdHead);
    tiling.fdParams.set_s2SplitStartIdxOfCore(s2SplitStartIdxOfCore);
    tiling.fdParams.set_gS1SplitNumOfFdHead(gS1SplitNumOfFdHead);
    tiling.fdParams.set_gS1LastPartSizeOfFdHead(gS1LastPartSizeOfFdHead);
    tiling.fdParams.set_gS1IdxEndOfFdHead(gS1IdxEndOfFdHead);
    tiling.fdParams.set_gS1IdxEndOfFdHeadSplit(gS1IdxEndOfFdHeadSplit);

    (void)aicNum;
}

}  // namespace

static ge::graphStatus TurboquantFiaMse8bitTilingFunc(gert::TilingContext* context)
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
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (aicNum == 0 || aivNum == 0) {
        OPS_LOG_E(nodeName, "invalid core nums aic=%u aiv=%u", aicNum, aivNum);
        return ge::GRAPH_FAILED;
    }
    const uint32_t cvRatio = aivNum / aicNum;
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (libapiSize == 0) {
        libapiSize = kLibApiWorkspaceFallback;
    }

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

    const int64_t numHeads = *numHeadsPtr;
    const int64_t numKvHeads = *numKvHeadsPtr;
    const int64_t headSize = *headSizePtr;
    const int64_t blockSize = *blockSizePtr;
    if (headSize != 128) {
        OPS_LOG_E(nodeName, "only head_size=128 supported, got %ld", headSize);
        return ge::GRAPH_FAILED;
    }
    if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0 || blockSize <= 0) {
        OPS_LOG_E(nodeName, "invalid head/block attrs");
        return ge::GRAPH_FAILED;
    }
    const uint32_t gSize = static_cast<uint32_t>(numHeads / numKvHeads);

    auto qShapePtr = context->GetInputShape(0);
    auto btShapePtr = context->GetInputShape(3);
    auto seqQTensor = context->GetInputTensor(4);
    auto seqKvTensor = context->GetInputTensor(5);
    if (qShapePtr == nullptr || btShapePtr == nullptr || seqQTensor == nullptr || seqKvTensor == nullptr) {
        OPS_LOG_E(nodeName, "required inputs missing");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (qShape.GetDimNum() < 3 || btShape.GetDimNum() < 2) {
        OPS_LOG_E(nodeName, "unexpected ranks query=%zu block_table=%zu",
                  qShape.GetDimNum(), btShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens = qShape.GetDim(0);
    const int64_t batchSize = btShape.GetDim(0);
    const int64_t maxBlocksPerSeq = btShape.GetDim(1);
    if (numTokens <= 0 || batchSize <= 0 || maxBlocksPerSeq <= 0) {
        OPS_LOG_E(nodeName, "invalid query/block_table shape");
        return ge::GRAPH_FAILED;
    }

    const int64_t* seqQHost = seqQTensor->GetData<int64_t>();
    const int64_t* seqKvHost = seqKvTensor->GetData<int64_t>();
    if (seqQHost == nullptr || seqKvHost == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len tensors must be host-value-dependent");
        return ge::GRAPH_FAILED;
    }

    int64_t maxKvSeq = 0;
    for (int64_t b = 0; b < batchSize; ++b) {
        maxKvSeq = std::max(maxKvSeq, seqKvHost[b]);
    }
    if (maxKvSeq <= 0) {
        OPS_LOG_E(nodeName, "invalid max kv seq");
        return ge::GRAPH_FAILED;
    }

    const bool accumQ = (seqQHost[batchSize - 1] == numTokens);
    // SplitCore uses per-batch S1 via actualSeqS1Size; s1Size is the max / fallback.
    const uint32_t s1Size = accumQ ? 1U : static_cast<uint32_t>(numTokens);
    const uint32_t s2Size = static_cast<uint32_t>(maxKvSeq);
    const uint32_t s2BaseSize = kS2BaseSize;
    const uint32_t sInnerSizeAlign = AlignUp(std::min(s2Size, s2BaseSize), 16U);
    const uint32_t headDimAlign = AlignUp(static_cast<uint32_t>(headSize), 16U);

    BaseInfo baseInfo {};
    SplitParam splitParam {};
    baseInfo.bSize = static_cast<uint32_t>(batchSize);
    baseInfo.n2Size = static_cast<uint32_t>(numKvHeads);
    baseInfo.gSize = gSize;
    baseInfo.s1Size = s1Size;
    baseInfo.s2Size = s2Size;
    baseInfo.isS1G = true;  // TND
    baseInfo.isAccumSeqS1 = accumQ;
    baseInfo.isAccumSeqS2 = false;
    baseInfo.actualLenQDims = static_cast<uint32_t>(batchSize);
    baseInfo.actualLenKvDims = static_cast<uint32_t>(batchSize);
    baseInfo.attenMaskFlag = false;
    baseInfo.sparseMode = 0;
    baseInfo.preToken = 2147483647;
    baseInfo.nextToken = 2147483647;
    baseInfo.actualSeqS1Size.assign(seqQHost, seqQHost + batchSize);
    baseInfo.actualSeqS2Size.assign(seqKvHost, seqKvHost + batchSize);

    splitParam.mBaseSize = kMBaseSize;
    splitParam.s2BaseSize = s2BaseSize;
    splitParam.gS1BaseSizeOfFd = kFdGS1BaseSize;

    SplitResult splitRes {aicNum, cvRatio};
    SplitCore(aicNum, baseInfo, splitParam, splitRes);
    if (splitRes.usedCoreNum == 0U) {
        OPS_LOG_E(nodeName, "SplitCore returned usedCoreNum=0");
        return ge::GRAPH_FAILED;
    }
    if (splitRes.numOfFdHead > aicNum || splitRes.usedCoreNum > aicNum ||
        splitRes.maxS2SplitNum > aicNum + 1U) {
        OPS_LOG_E(nodeName,
                  "SplitCore out of range used=%u fdHeads=%u maxS2Split=%u aic=%u",
                  splitRes.usedCoreNum, splitRes.numOfFdHead, splitRes.maxS2SplitNum, aicNum);
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum = splitRes.usedCoreNum;
    const bool enableFd = (splitRes.numOfFdHead > 0U);

    TurboquantFiaMse8bitTilingData tiling {};
    auto& base = tiling.baseParams;
    base.set_bSize(static_cast<uint32_t>(batchSize));
    base.set_n2Size(static_cast<uint32_t>(numKvHeads));
    base.set_gSize(gSize);
    base.set_s1Size(s1Size);
    base.set_s2Size(s2Size);
    base.set_headDim(static_cast<uint32_t>(headSize));
    base.set_headDimRope(0);
    base.set_actualSeqS1Dims(static_cast<uint32_t>(batchSize));
    base.set_actualSeqS2Dims(static_cast<uint32_t>(batchSize));
    base.set_accumQSeqFlag(accumQ ? 1U : 0U);
    base.set_accumKVSeqFlag(0U);
    base.set_scaleValue(*scaleValuePtr);
    base.set_usedCoreNum(usedCoreNum);
    base.set_outputLayout(3U);  // FIA_LAYOUT::TND
    base.set_batchContinuous(1U);
    base.set_softmaxLseFlag(0U);
    base.set_needInit(0U);
    base.set_slidingFlag(0U);
    base.set_l2CacheOffFlag(0U);
    base.set_isLegacyIfa(0U);

    tiling.pageAttenParams.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.pageAttenParams.set_maxBlockNumPerBatch(static_cast<uint32_t>(maxBlocksPerSeq));

    tiling.maskParams.set_attenMaskFlag(0U);
    tiling.maskParams.set_attenMaskBatchStride(0U);
    tiling.maskParams.set_attenMaskStride(0U);
    tiling.maskParams.set_preToken(2147483647);
    tiling.maskParams.set_nextToken(2147483647);
    tiling.maskParams.set_isRowInvalid(0U);
    tiling.maskParams.set_isExistRowInvalid(0U);
    tiling.maskParams.set_sparseMode(0U);

    const uint32_t mm1ResSize = kMBaseSize * sInnerSizeAlign;
    const uint32_t mm2ResSize = kMBaseSize * headDimAlign;
    tiling.workspaceParams.set_mm1ResSize(mm1ResSize);
    tiling.workspaceParams.set_mm2ResSize(mm2ResSize);

    tiling.innerSplitParams.set_mBaseSize(kMBaseSize);
    tiling.innerSplitParams.set_s2BaseSize(s2BaseSize);

    ApplySplitResult(tiling, splitRes, aicNum, cvRatio);

    // FD workspace element counts (mirror FiaTilingNonQuant::FillTilingWorkspaceParams).
    const uint64_t fdAccumOutSize =
        enableFd ? (static_cast<uint64_t>(aicNum) * 2ULL * kMBaseSize * headDimAlign) : 0ULL;
    const uint64_t fdLogSumExpSize =
        enableFd ? (2ULL * aicNum * 2ULL * kMBaseSize * (kByteBlock / kBlockTableElemByte)) : 0ULL;
    tiling.workspaceParams.set_fdAccumOutSize(static_cast<uint32_t>(fdAccumOutSize));
    tiling.workspaceParams.set_fdLogSumExpSize(static_cast<uint32_t>(fdLogSumExpSize));

    tiling.prefixParams.set_prefixMaxLen(0);
    tiling.prefixParams.set_prefixLen(0);
    tiling.prefixParams.set_prefixFlag(false);
    tiling.pseParams.set_pseShiftFlag(0U);
    tiling.pseParams.set_pseShiftByBatch(0U);
    tiling.pseParams.set_pseShiftS1(0U);
    tiling.pseParams.set_pseShiftS2(0U);
    tiling.leftPaddingParams.set_qPaddingFlag(0U);
    tiling.leftPaddingParams.set_kvPaddingFlag(0U);

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tiling.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small need=%zu cap=%zu",
                  tiling.GetDataSize(),
                  rawTiling == nullptr ? 0UL : rawTiling->GetCapacity());
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    const uint64_t normalWs =
        static_cast<uint64_t>(kPreLoadNum) * usedCoreNum *
        (static_cast<uint64_t>(mm1ResSize) * 4ULL +
         static_cast<uint64_t>(mm1ResSize) * 2ULL +
         static_cast<uint64_t>(mm2ResSize) * 4ULL +
         static_cast<uint64_t>(mm2ResSize) * 4ULL);
    // FD bytes: (accumOut + logSumExp) * sizeof(float) with paramNums = aic*2*mBase
    const uint64_t fdParamNums = static_cast<uint64_t>(aicNum) * 2ULL * kMBaseSize;
    const uint64_t fdWs =
        enableFd ? ((fdParamNums * headDimAlign +
                     2ULL * fdParamNums * (kByteBlock / kBlockTableElemByte)) *
                    static_cast<uint64_t>(kBlockTableElemByte))
                 : 0ULL;
    const uint64_t tqWs =
        static_cast<uint64_t>(kPreLoadNum) * usedCoreNum * 2ULL *
        static_cast<uint64_t>(sInnerSizeAlign) * headDimAlign * 2ULL;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(libapiSize + normalWs + fdWs + tqWs);

    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(usedCoreNum * cvRatio, usedCoreNum,
                                                               usedCoreNum * cvRatio);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(enableFd ? kTilingKeyFd : kTilingKeyNoFd);

    OPS_LOG_I(nodeName,
              "TQ-FIA tiling: B=%ld N2=%ld G=%u S1=%u S2=%u cores=%u fd=%d fdHeads=%u "
              "maxS2Split=%u ws=%zu key=%lu",
              batchSize, numKvHeads, gSize, s1Size, s2Size, usedCoreNum, enableFd ? 1 : 0,
              splitRes.numOfFdHead, splitRes.maxS2SplitNum, workspaces[0],
              static_cast<unsigned long>(enableFd ? kTilingKeyFd : kTilingKeyNoFd));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantFiaMse8bit)
    .Tiling(TurboquantFiaMse8bitTilingFunc);

}  // namespace optiling
