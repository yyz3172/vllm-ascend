/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You can obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// BitResidual K8V4: fused sign+residual decode + paged attention on packed KV cache.

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "bit_residual_attention_paged_k8v4_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t TQ_M_ALIGN = 16;
constexpr uint32_t TQ_KFC_AIC_NUM = 1;
constexpr uint32_t TQ_KFC_AIV_NUM = 2;
constexpr uint32_t TQ_ROT_N = optiling::TQ_BR_HEAD_SIZE;
constexpr uint32_t TQ_ROT_K = optiling::TQ_BR_HEAD_SIZE;
constexpr uint32_t TQ_ROT_MAX_BASE_M = optiling::TQ_BR_ATTN_KV_TILE_CAP;
constexpr size_t MAX_USER_WORKSPACE = 128UL * 1024 * 1024;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024 * 1024;

uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return (x + align - 1) / align * align;
}

uint32_t AlignDown(uint32_t x, uint32_t align)
{
    return (x / align) * align;
}

matmul_tiling::DataType ToMatmulDtype(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 ? matmul_tiling::DataType::DT_BF16
                                : matmul_tiling::DataType::DT_FLOAT16;
}

uint32_t PickKvTileRows(uint32_t blockSize)
{
    uint32_t tile = std::min(optiling::TQ_BR_ATTN_KV_TILE_CAP, optiling::TQ_BR_ATTN_KV_TILE_CAP);
    if (blockSize == 0) {
        return tile;
    }
    if (blockSize <= tile && tile % blockSize != 0) {
        tile = AlignDown(tile, blockSize);
        if (tile == 0) {
            tile = blockSize;
        }
    }
    tile = std::min(tile, optiling::TQ_BR_ATTN_KV_TILE_CAP);
    return tile;
}

ge::graphStatus FillDecodeRotateTiling(
    const char* nodeName,
    const platform_ascendc::PlatformAscendC& platform,
    uint32_t kvTileRows,
    matmul_tiling::DataType dataType,
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
                       dataType, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       dataType, false);
    tilingApi.SetCType(matmul_tiling::TPosition::VECIN,
                       matmul_tiling::CubeFormat::ND,
                       dataType);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM,
                          matmul_tiling::CubeFormat::ND,
                          dataType);
    const uint32_t mPad = AlignUp(kvTileRows, TQ_M_ALIGN);
    tilingApi.SetOrgShape(mPad, optiling::TQ_BR_HEAD_SIZE, optiling::TQ_BR_HEAD_SIZE);
    tilingApi.SetShape(mPad, optiling::TQ_BR_HEAD_SIZE, optiling::TQ_BR_HEAD_SIZE);
    tilingApi.EnableBias(false);
    tilingApi.SetBufferSpace(l1Size, l0cSize, ubSize);
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        OPS_LOG_E(nodeName, "decode rotate GetTiling failed");
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t mmDataTypeSize = sizeof(uint16_t);
    uint32_t baseM = static_cast<uint32_t>(std::min<uint64_t>(
        l1Size / (TQ_ROT_K + TQ_ROT_N) / mmDataTypeSize,
        static_cast<uint64_t>(TQ_ROT_MAX_BASE_M)));
    baseM = AlignUp(baseM, TQ_M_ALIGN);
    if (baseM > mPad) {
        baseM = mPad;
    }
    if (baseM == 0) {
        baseM = mPad;
    }
    cubeTiling.set_baseM(baseM);
    cubeTiling.set_usedCoreNum(1);
    return ge::GRAPH_SUCCESS;
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

size_t CalcSystemWorkspaceSize(const platform_ascendc::PlatformAscendC& platform)
{
    size_t ws = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return std::max(ws, SYSTEM_NEED_WORKSPACE);
}

}  // namespace

namespace optiling {

static ge::graphStatus BitResidualAttentionPagedK8v4TilingFunc(gert::TilingContext* context)
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
    const int64_t* maxActualSeqLenPtr = attrs->GetAttrPointer<int64_t>(4);
    const float* scaleValuePtr = attrs->GetAttrPointer<float>(5);
    if (numHeadsPtr == nullptr || numKvHeadsPtr == nullptr || headSizePtr == nullptr ||
        blockSizePtr == nullptr || maxActualSeqLenPtr == nullptr || scaleValuePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }

    auto qShapePtr = context->GetInputShape(0);
    auto keyCacheShapePtr = context->GetInputShape(1);
    auto valCacheShapePtr = context->GetInputShape(2);
    auto btShapePtr = context->GetInputShape(3);
    auto actualSeqLenQShapePtr = context->GetInputShape(4);
    auto actualSeqLenKvShapePtr = context->GetInputShape(5);
    auto actualSeqLenQDesc = context->GetInputDesc(4);
    auto actualSeqLenKvDesc = context->GetInputDesc(5);
    auto queryDesc = context->GetInputDesc(0);
    auto rotKeyDesc = context->GetInputDesc(6);
    auto rotValDesc = context->GetInputDesc(7);
    auto actualSeqLenQTensor = context->GetInputTensor(4);
    auto actualSeqLenKvTensor = context->GetInputTensor(5);
    if (qShapePtr == nullptr || keyCacheShapePtr == nullptr || valCacheShapePtr == nullptr ||
        btShapePtr == nullptr) {
        OPS_LOG_E(nodeName, "required input shapes are null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape keyCacheShape = keyCacheShapePtr->GetStorageShape();
    const gert::Shape valCacheShape = valCacheShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (qShape.GetDimNum() < 2 || keyCacheShape.GetDimNum() < 3 ||
        valCacheShape.GetDimNum() < 3 || btShape.GetDimNum() < 2) {
        OPS_LOG_E(nodeName, "unexpected input ranks");
        return ge::GRAPH_FAILED;
    }
    if (queryDesc == nullptr || rotKeyDesc == nullptr || rotValDesc == nullptr) {
        OPS_LOG_E(nodeName, "dtype input desc is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dataType = queryDesc->GetDataType();
    if ((dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16) ||
        rotKeyDesc->GetDataType() != dataType ||
        rotValDesc->GetDataType() != dataType) {
        OPS_LOG_E(nodeName, "query/rotation_key/rotation_value must share fp16 or bf16 dtype");
        return ge::GRAPH_FAILED;
    }
    const matmul_tiling::DataType mmDataType = ToMatmulDtype(dataType);

    const uint32_t numTokens = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t numHeads = static_cast<uint32_t>(*numHeadsPtr);
    const uint32_t numKvHeads = static_cast<uint32_t>(*numKvHeadsPtr);
    const uint32_t headSize = static_cast<uint32_t>(*headSizePtr);
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t attrMaxActualSeqLen = static_cast<uint32_t>(*maxActualSeqLenPtr);
    const uint32_t batchSize = static_cast<uint32_t>(btShape.GetDim(0));
    const uint32_t maxBlocksPerSeq = static_cast<uint32_t>(btShape.GetDim(1));
    const uint32_t totalCacheBlocks = static_cast<uint32_t>(keyCacheShape.GetDim(0));

    // BitResidual K8V4 cache shape validation (16-row sub-block layout).
    const uint32_t subBlocksPerBlock = blockSize / TQ_BR_BLOCK_ROWS;
    const uint32_t keyPackedBytes = subBlocksPerBlock * TQ_BR_KEY_BLOCK_STRIDE;
    const uint32_t valuePackedBytes = subBlocksPerBlock * TQ_BR_VAL_BLOCK_STRIDE;
    if (keyCacheShape.GetDim(1) != static_cast<int64_t>(numKvHeads) ||
        keyCacheShape.GetDim(2) != static_cast<int64_t>(keyPackedBytes)) {
        OPS_LOG_E(nodeName,
                  "key_cache shape must be [num_blocks, num_kv_heads, (block_size/%u)*%u], "
                  "got [%ld, %ld, %ld]",
                  TQ_BR_BLOCK_ROWS, TQ_BR_KEY_BLOCK_STRIDE,
                  keyCacheShape.GetDim(0), keyCacheShape.GetDim(1), keyCacheShape.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    if (valCacheShape.GetDim(1) != static_cast<int64_t>(numKvHeads) ||
        valCacheShape.GetDim(2) != static_cast<int64_t>(valuePackedBytes)) {
        OPS_LOG_E(nodeName,
                  "value_cache shape must be [num_blocks, num_kv_heads, (block_size/%u)*%u], "
                  "got [%ld, %ld, %ld]",
                  TQ_BR_BLOCK_ROWS, TQ_BR_VAL_BLOCK_STRIDE,
                  valCacheShape.GetDim(0), valCacheShape.GetDim(1), valCacheShape.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    if (totalCacheBlocks == 0) {
        OPS_LOG_E(nodeName, "key_cache num_blocks must be > 0");
        return ge::GRAPH_FAILED;
    }

    if (headSize != TQ_BR_HEAD_SIZE) {
        OPS_LOG_E(nodeName, "only head_size=128 supported, got %u", headSize);
        return ge::GRAPH_FAILED;
    }
    if (blockSize % TQ_BR_BLOCK_ROWS != 0) {
        OPS_LOG_E(nodeName, "block_size must be a multiple of %u (sub-block rows), got %u",
                  TQ_BR_BLOCK_ROWS, blockSize);
        return ge::GRAPH_FAILED;
    }
    if (numHeads == 0 || numKvHeads == 0 || numHeads % numKvHeads != 0) {
        OPS_LOG_E(nodeName, "invalid head configuration");
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxKvLen = maxBlocksPerSeq * blockSize;
    uint32_t maxActualSeqLen = maxKvLen;

    // Validate actual_seq_len_q/kv tensors.
    if (actualSeqLenQShapePtr != nullptr && actualSeqLenQTensor != nullptr) {
        const auto actualSeqLenQStorageShape = actualSeqLenQShapePtr->GetStorageShape();
        const int64_t* actualSeqLenQ = actualSeqLenQTensor->GetData<int64_t>();
        if (actualSeqLenQ != nullptr && actualSeqLenQStorageShape.GetDimNum() == 1) {
            int64_t prev = 0;
            for (uint32_t i = 0; i < static_cast<uint32_t>(actualSeqLenQStorageShape.GetDim(0)); ++i) {
                if (actualSeqLenQ[i] < prev || actualSeqLenQ[i] > static_cast<int64_t>(numTokens)) {
                    OPS_LOG_E(nodeName, "actual_seq_len_q[%u]=%ld invalid", i, actualSeqLenQ[i]);
                    return ge::GRAPH_FAILED;
                }
                prev = actualSeqLenQ[i];
            }
        }
    }
    if (actualSeqLenKvShapePtr != nullptr && actualSeqLenKvTensor != nullptr) {
        const auto actualSeqLenKvStorageShape = actualSeqLenKvShapePtr->GetStorageShape();
        const int64_t* actualSeqLenKv = actualSeqLenKvTensor->GetData<int64_t>();
        if (actualSeqLenKv != nullptr && actualSeqLenKvStorageShape.GetDimNum() == 1) {
            uint32_t observedMax = 0;
            for (uint32_t i = 0; i < static_cast<uint32_t>(actualSeqLenKvStorageShape.GetDim(0)); ++i) {
                if (actualSeqLenKv[i] < 0 || actualSeqLenKv[i] > static_cast<int64_t>(maxKvLen)) {
                    OPS_LOG_E(nodeName, "actual_seq_len_kv[%u]=%ld invalid", i, actualSeqLenKv[i]);
                    return ge::GRAPH_FAILED;
                }
                observedMax = std::max(observedMax, static_cast<uint32_t>(actualSeqLenKv[i]));
            }
            if (attrMaxActualSeqLen < observedMax) {
                OPS_LOG_E(nodeName, "max_actual_seq_len attr %u < observed max %u",
                          attrMaxActualSeqLen, observedMax);
                return ge::GRAPH_FAILED;
            }
        }
    }
    maxActualSeqLen = attrMaxActualSeqLen;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t coreNum = std::max(1U, aicNum);
    const uint32_t parallelCoreNum = std::min(coreNum, optiling::TQ_BR_ATTN_MAX_PARALLEL_CORES);
    const uint32_t gqaGroup = numHeads / numKvHeads;
    const uint32_t gqaChunkCount = (gqaGroup + TQ_BR_ATTN_GQA_CAP - 1) / TQ_BR_ATTN_GQA_CAP;
    const uint32_t headChunkScale = numKvHeads * gqaChunkCount;
    const uint32_t taskCount = numTokens * headChunkScale;
    const uint32_t kvTileRows = PickKvTileRows(blockSize);

    BitResidualAttentionPagedK8v4TilingData tiling{};
    if (FillDecodeRotateTiling(nodeName, ascendcPlatform, kvTileRows, mmDataType,
                               tiling.decodeRotateTiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Initial release: only SplitBN Vector mode.
    const uint32_t splitMode = TQ_BR_ATTN_SPLIT_BN;
    const uint32_t qkPvMode = TQ_BR_ATTN_QKPV_VECTOR;

    uint32_t usedCoreNum = 0;
    uint32_t formerCoreNum = 0;
    uint32_t blockSplitRange = 0;
    uint32_t tailSplitRange = 0;
    SplitBn(taskCount, parallelCoreNum, usedCoreNum, formerCoreNum, blockSplitRange,
            tailSplitRange);

    tiling.set_numTokens(numTokens);
    tiling.set_batchSize(batchSize);
    tiling.set_numHeads(numHeads);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_gqaGroupSize(gqaGroup);
    tiling.set_headSize(headSize);
    tiling.set_blockSize(blockSize);
    tiling.set_maxBlocksPerSeq(maxBlocksPerSeq);
    tiling.set_totalCacheBlocks(totalCacheBlocks);
    tiling.set_maxKvLen(maxKvLen);
    tiling.set_maxActualSeqLen(maxActualSeqLen);
    tiling.set_kvTileRows(kvTileRows);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_splitMode(splitMode);
    tiling.set_qkPvMode(qkPvMode);
    tiling.set_kvSegmentLen(maxActualSeqLen);
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_blockSplitRange(blockSplitRange);
    tiling.set_tailSplitRange(tailSplitRange);
    tiling.set_scaleValue(*scaleValuePtr);

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tiling.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    const uint32_t mixBlockDim =
        ascendcPlatform.CalcTschBlockDim(TQ_KFC_AIV_NUM, TQ_KFC_AIC_NUM, TQ_KFC_AIV_NUM);
    const uint32_t dataCores = parallelCoreNum;
    const uint32_t blockDim = dataCores * mixBlockDim;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = CalcSystemWorkspaceSize(ascendcPlatform);
    if (workspaces[0] > MAX_USER_WORKSPACE) {
        OPS_LOG_E(nodeName, "workspace size %zu exceeds cap %zu", workspaces[0], MAX_USER_WORKSPACE);
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(blockDim);
    const uint64_t tilingKey = TQ_BR_ATTN_KEY_SPLITBN_VECTOR;
    context->SetTilingKey(tilingKey);
    OPS_LOG_I(nodeName,
              "BitResidualAttentionPagedK8v4 tiling: tokens=%u tasks=%u gqa=%u "
              "maxKv=%u maxActual=%u split=%u qkpv=%u kvTile=%u usedCore=%u "
              "dataCores=%u blockRange=%u tailRange=%u key=%lu ws=%zu",
              numTokens, taskCount, gqaGroup, maxKvLen, maxActualSeqLen,
              splitMode, qkPvMode, kvTileRows, usedCoreNum, dataCores,
              blockSplitRange, tailSplitRange, tilingKey, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BitResidualAttentionPagedK8v4)
    .Tiling(BitResidualAttentionPagedK8v4TilingFunc);

}  // namespace optiling
