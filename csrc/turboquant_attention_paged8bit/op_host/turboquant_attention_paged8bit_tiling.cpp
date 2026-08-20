/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "turboquant_attention_paged8bit_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t TQ_M_ALIGN = 16;
constexpr uint32_t TQ_KFC_AIC_NUM = 1;
constexpr uint32_t TQ_KFC_AIV_NUM = 2;
constexpr size_t MAX_USER_WORKSPACE = 128UL * 1024 * 1024;
constexpr float FLASH_DECODE_BN_RATIO = 0.5f;
constexpr uint32_t FLASH_DECODE_MIN_KV_LEN = 2048;
constexpr uint32_t FLASH_DECODE_MIN_SEGMENT = 512;

uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return (x + align - 1) / align * align;
}

uint32_t AlignDown(uint32_t x, uint32_t align)
{
    return (x / align) * align;
}

bool IsFlashDecode(uint32_t numTokens, uint32_t numKvHeads, uint32_t maxKvLen, uint32_t aicNum)
{
    const uint32_t bn = numTokens * numKvHeads;
    if (maxKvLen < FLASH_DECODE_MIN_KV_LEN) {
        return false;
    }
    if (bn > static_cast<uint32_t>(FLASH_DECODE_BN_RATIO * aicNum)) {
        return false;
    }
    return true;
}

uint32_t PickKvTileRows(uint32_t blockSize)
{
    uint32_t tile = std::min(optiling::TQ_ATTN_KV_TILE_ROWS, optiling::TQ_ATTN_UB_KV_TILE_CAP);
    if (blockSize == 0) {
        return tile;
    }
    // When block_size fits in UB, prefer a tile that is a multiple of block_size.
    // For block_size > UB cap (e.g. PA block 128), keep tile at UB cap and let the
    // kernel walk KV positions with absPos / blockSize addressing.
    if (blockSize <= tile && tile % blockSize != 0) {
        tile = AlignDown(tile, blockSize);
        if (tile == 0) {
            tile = blockSize;
        }
    }
    tile = std::min(tile, optiling::TQ_ATTN_UB_KV_TILE_CAP);
    const uint32_t cubeMin = std::min(optiling::TQ_ATTN_CUBE_MIN_TILE, optiling::TQ_ATTN_UB_KV_TILE_CAP);
    return std::max(tile, cubeMin);
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
    tilingApi.SetOrgShape(mPad, optiling::TQ_ATTN_HEAD_SIZE, optiling::TQ_ATTN_HEAD_SIZE);
    tilingApi.SetShape(mPad, optiling::TQ_ATTN_HEAD_SIZE, optiling::TQ_ATTN_HEAD_SIZE);
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
    bmm1.SetShape(m, kvTileRows, optiling::TQ_ATTN_HEAD_SIZE);
    bmm1.SetOrgShape(m, kvTileRows, optiling::TQ_ATTN_HEAD_SIZE, optiling::TQ_ATTN_HEAD_SIZE);
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
    bmm2.SetShape(m, optiling::TQ_ATTN_HEAD_SIZE, kvTileRows);
    bmm2.SetOrgShape(m, optiling::TQ_ATTN_HEAD_SIZE, kvTileRows, kvTileRows);
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

void SplitBns(uint32_t bn, uint32_t maxKvLen, uint32_t blockSize, uint32_t coreNum,
              uint32_t& usedCoreNum, uint32_t& kvSplitPart, uint32_t& kvSegmentLen,
              uint32_t& formerCoreNum, uint32_t& blockSplitRange, uint32_t& tailSplitRange)
{
    formerCoreNum = 0;
    blockSplitRange = 1;
    tailSplitRange = 1;
    kvSplitPart = (bn > 0) ? (coreNum / bn) : 1;
    if (kvSplitPart == 0) {
        kvSplitPart = 1;
    }
    while (((maxKvLen / kvSplitPart) < FLASH_DECODE_MIN_SEGMENT) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }
    usedCoreNum = bn * kvSplitPart;
    kvSegmentLen = (maxKvLen + kvSplitPart - 1) / kvSplitPart;
    if (blockSize > 0) {
        kvSegmentLen = AlignUp(kvSegmentLen, blockSize);
    }
}

uint32_t PickQkPvMode(uint32_t gqaGroup, uint32_t kvTileRows)
{
    if (gqaGroup >= optiling::TQ_ATTN_CUBE_MIN_G &&
        kvTileRows >= optiling::TQ_ATTN_CUBE_MIN_TILE) {
        return optiling::TQ_ATTN_QKPV_CUBE;
    }
    return optiling::TQ_ATTN_QKPV_VECTOR;
}

uint64_t PickTilingKey(uint32_t splitMode, uint32_t qkPvMode)
{
    if (splitMode == optiling::TQ_ATTN_SPLIT_BN) {
        return (qkPvMode == optiling::TQ_ATTN_QKPV_CUBE) ? optiling::TQ_ATTN_KEY_SPLITBN_CUBE
                                                         : optiling::TQ_ATTN_KEY_SPLITBN_VECTOR;
    }
    return (qkPvMode == optiling::TQ_ATTN_QKPV_CUBE) ? optiling::TQ_ATTN_KEY_SPLITBNS_CUBE
                                                     : optiling::TQ_ATTN_KEY_SPLITBNS_VECTOR;
}

ge::graphStatus CheckActualSeqLenQ(const char* nodeName,
                                   const gert::StorageShape* actualSeqLenQShape,
                                   const gert::CompileTimeTensorDesc* actualSeqLenQDesc,
                                   const gert::Tensor* actualSeqLenQTensor,
                                   uint32_t batchSize,
                                   uint32_t numTokens)
{
    if (actualSeqLenQShape == nullptr || actualSeqLenQDesc == nullptr || actualSeqLenQTensor == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len_q shape, desc, or tensor is null");
        return ge::GRAPH_FAILED;
    }

    if (actualSeqLenQDesc->GetDataType() != ge::DT_INT64) {
        OPS_LOG_E(nodeName, "actual_seq_len_q dtype must be int64");
        return ge::GRAPH_FAILED;
    }

    const auto actualSeqLenQStorageShape = actualSeqLenQShape->GetStorageShape();
    if (actualSeqLenQStorageShape.GetDimNum() != 1) {
        OPS_LOG_E(nodeName, "actual_seq_len_q must be 1D");
        return ge::GRAPH_FAILED;
    }
    const int64_t actualLenElems = actualSeqLenQStorageShape.GetDim(0);
    if (actualLenElems != static_cast<int64_t>(batchSize)) {
        OPS_LOG_E(nodeName, "actual_seq_len_q size(%ld) must equal batch_size(%u)",
                  actualLenElems, batchSize);
        return ge::GRAPH_FAILED;
    }

    const int64_t* actualSeqLenQ = actualSeqLenQTensor->GetData<int64_t>();
    if (actualSeqLenQ == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len_q data is null");
        return ge::GRAPH_FAILED;
    }

    int64_t prev = 0;
    for (uint32_t i = 0; i < batchSize; ++i) {
        const int64_t cur = actualSeqLenQ[i];
        if (cur < prev || cur > static_cast<int64_t>(numTokens)) {
            OPS_LOG_E(nodeName,
                      "actual_seq_len_q[%u]=%ld must be non-decreasing and in [0, %u]",
                      i, cur, numTokens);
            return ge::GRAPH_FAILED;
        }
        prev = cur;
    }
    if (batchSize > 0 && actualSeqLenQ[batchSize - 1] != static_cast<int64_t>(numTokens)) {
        OPS_LOG_E(nodeName, "actual_seq_len_q last value(%ld) must equal num_tokens(%u)",
                  actualSeqLenQ[batchSize - 1], numTokens);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndGetMaxActualSeqLenKv(const char* nodeName,
                                             const gert::StorageShape* actualSeqLenKvShape,
                                             const gert::CompileTimeTensorDesc* actualSeqLenKvDesc,
                                             const gert::Tensor* actualSeqLenKvTensor,
                                             uint32_t batchSize,
                                             uint32_t maxKvLen,
                                             uint32_t attrMaxActualSeqLen,
                                             uint32_t& maxActualSeqLen)
{
    maxActualSeqLen = maxKvLen;
    if (actualSeqLenKvShape == nullptr || actualSeqLenKvDesc == nullptr || actualSeqLenKvTensor == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len_kv shape, desc, or tensor is null");
        return ge::GRAPH_FAILED;
    }

    if (actualSeqLenKvDesc->GetDataType() != ge::DT_INT64) {
        OPS_LOG_E(nodeName, "actual_seq_len_kv dtype must be int64");
        return ge::GRAPH_FAILED;
    }

    const auto actualSeqLenKvStorageShape = actualSeqLenKvShape->GetStorageShape();
    if (actualSeqLenKvStorageShape.GetDimNum() != 1) {
        OPS_LOG_E(nodeName, "actual_seq_len_kv must be 1D");
        return ge::GRAPH_FAILED;
    }
    const int64_t actualLenElems = actualSeqLenKvStorageShape.GetDim(0);
    if (actualLenElems != static_cast<int64_t>(batchSize)) {
        OPS_LOG_E(nodeName, "actual_seq_len_kv size(%ld) must equal batch_size(%u)",
                  actualLenElems, batchSize);
        return ge::GRAPH_FAILED;
    }
    const int64_t* actualSeqLenKv = actualSeqLenKvTensor->GetData<int64_t>();
    if (actualSeqLenKv == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len_kv data is null");
        return ge::GRAPH_FAILED;
    }

    uint32_t observedMaxActualSeqLen = 0;
    for (uint32_t i = 0; i < batchSize; ++i) {
        const int64_t cur = actualSeqLenKv[i];
        if (cur < 0 || cur > static_cast<int64_t>(maxKvLen)) {
            OPS_LOG_E(nodeName,
                      "actual_seq_len_kv[%u]=%ld must be in [0, %u]",
                      i, cur, maxKvLen);
            return ge::GRAPH_FAILED;
        }
        observedMaxActualSeqLen = std::max(observedMaxActualSeqLen, static_cast<uint32_t>(cur));
    }

    if (attrMaxActualSeqLen == 0 || attrMaxActualSeqLen > maxKvLen) {
        OPS_LOG_E(nodeName, "max_actual_seq_len attr %u must be in (0, %u]",
                  attrMaxActualSeqLen, maxKvLen);
        return ge::GRAPH_FAILED;
    }
    if (attrMaxActualSeqLen < observedMaxActualSeqLen) {
        OPS_LOG_E(nodeName,
                  "max_actual_seq_len attr %u must be >= max(actual_seq_len_kv) %u",
                  attrMaxActualSeqLen, observedMaxActualSeqLen);
        return ge::GRAPH_FAILED;
    }
    maxActualSeqLen = attrMaxActualSeqLen;
    return ge::GRAPH_SUCCESS;
}

size_t CalcWorkspaceSize(const platform_ascendc::PlatformAscendC& platform, uint32_t splitMode,
                         uint32_t numHeads, uint32_t headSize, uint32_t kvSplitPart)
{
    // Match IFA: lib-api scratch for the decode/QK/PV matmul path plus optional
    // user buffers for FlashDecode partial outputs.
    size_t ws = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    if (splitMode == optiling::TQ_ATTN_SPLIT_BNS) {
        const size_t accumOutElems = static_cast<size_t>(numHeads) * kvSplitPart * headSize;
        const size_t lseBytes = static_cast<size_t>(numHeads) * kvSplitPart * 2 * sizeof(float);
        ws += accumOutElems * sizeof(uint16_t) + lseBytes;
    }
    return ws;
}

}  // namespace

namespace optiling {

static ge::graphStatus TurboquantAttentionPaged8bitTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        (nullptr, "enter context-null");
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
    auto cacheShapePtr = context->GetInputShape(1);
    auto btShapePtr = context->GetInputShape(3);
    auto actualSeqLenQShapePtr = context->GetInputShape(4);
    auto actualSeqLenKvShapePtr = context->GetInputShape(5);
    auto actualSeqLenQDesc = context->GetInputDesc(4);
    auto actualSeqLenKvDesc = context->GetInputDesc(5);
    auto actualSeqLenQTensor = context->GetInputTensor(4);
    auto actualSeqLenKvTensor = context->GetInputTensor(5);
    if (qShapePtr == nullptr || cacheShapePtr == nullptr || btShapePtr == nullptr) {
        OPS_LOG_E(nodeName, "required input shapes are null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape cacheShape = cacheShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (qShape.GetDimNum() < 2 || cacheShape.GetDimNum() < 4 || btShape.GetDimNum() < 2) {
        OPS_LOG_E(nodeName, "unexpected input ranks");
        return ge::GRAPH_FAILED;
    }

    const uint32_t numTokens = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t numHeads = static_cast<uint32_t>(*numHeadsPtr);
    const uint32_t numKvHeads = static_cast<uint32_t>(*numKvHeadsPtr);
    const uint32_t headSize = static_cast<uint32_t>(*headSizePtr);
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t attrMaxActualSeqLen = static_cast<uint32_t>(*maxActualSeqLenPtr);
    const uint32_t batchSize = static_cast<uint32_t>(btShape.GetDim(0));
    const uint32_t maxBlocksPerSeq = static_cast<uint32_t>(btShape.GetDim(1));
    const uint32_t totalCacheBlocks = static_cast<uint32_t>(cacheShape.GetDim(0));
    if (cacheShape.GetDim(1) != static_cast<int64_t>(blockSize) ||
        cacheShape.GetDim(2) != static_cast<int64_t>(numKvHeads) ||
        cacheShape.GetDim(3) != static_cast<int64_t>(TQ_ATTN_PACKED_BYTES)) {
        OPS_LOG_E(nodeName,
                  "key_cache shape must be [num_blocks, block_size, num_kv_heads, %u]",
                  TQ_ATTN_PACKED_BYTES);
        return ge::GRAPH_FAILED;
    }
    if (totalCacheBlocks == 0) {
        OPS_LOG_E(nodeName, "key_cache num_blocks must be > 0");
        return ge::GRAPH_FAILED;
    }

    if (headSize != TQ_ATTN_HEAD_SIZE) {
        OPS_LOG_E(nodeName, "only head_size=128 supported, got %u", headSize);
        return ge::GRAPH_FAILED;
    }
    if (numHeads == 0 || numKvHeads == 0 || numHeads % numKvHeads != 0) {
        OPS_LOG_E(nodeName, "invalid head configuration");
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxKvLen = maxBlocksPerSeq * blockSize;
    uint32_t maxActualSeqLen = maxKvLen;
    if (CheckActualSeqLenQ(nodeName, actualSeqLenQShapePtr, actualSeqLenQDesc, actualSeqLenQTensor,
                           batchSize, numTokens) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAndGetMaxActualSeqLenKv(nodeName, actualSeqLenKvShapePtr, actualSeqLenKvDesc, actualSeqLenKvTensor,
                                     batchSize, maxKvLen, attrMaxActualSeqLen, maxActualSeqLen) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t coreNum = std::max(1U, aicNum);
    const uint32_t parallelCoreNum = std::min(coreNum, optiling::TQ_ATTN_MAX_PARALLEL_CORES);
    const uint32_t gqaGroup = numHeads / numKvHeads;
    const uint32_t bn = numTokens * numKvHeads;
    const uint32_t kvTileRows = PickKvTileRows(blockSize);
    if (gqaGroup > TQ_ATTN_UB_GQA_CAP) {
        OPS_LOG_E(nodeName, "gqa_group %u exceeds kernel UB cap %u", gqaGroup, TQ_ATTN_UB_GQA_CAP);
        return ge::GRAPH_FAILED;
    }
    if (kvTileRows > TQ_ATTN_UB_KV_TILE_CAP) {
        OPS_LOG_E(nodeName, "kv_tile_rows %u exceeds kernel UB cap %u", kvTileRows,
                  TQ_ATTN_UB_KV_TILE_CAP);
        return ge::GRAPH_FAILED;
    }

    TurboquantAttentionPaged8bitTilingData tiling{};
    if (FillDecodeRotateTiling(nodeName, ascendcPlatform, kvTileRows, tiling.decodeRotateTiling) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t splitMode = IsFlashDecode(numTokens, numKvHeads, maxActualSeqLen, coreNum)
                                   ? TQ_ATTN_SPLIT_BNS
                                   : TQ_ATTN_SPLIT_BN;
    const uint32_t qkPvMode = PickQkPvMode(gqaGroup, kvTileRows);
    if (qkPvMode == TQ_ATTN_QKPV_CUBE) {
        // Validate cube QK/PV shapes on host; tiling payload stays decode-only until
        // kernel registers qk/pv matmul objects (separate TU or key-specific path).
        TCubeTiling qkProbe {};
        TCubeTiling pvProbe {};
        if (FillQkTiling(nodeName, ascendcPlatform, gqaGroup, kvTileRows, qkProbe) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (FillPvTiling(nodeName, ascendcPlatform, gqaGroup, kvTileRows, pvProbe) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t usedCoreNum = 0;
    uint32_t formerCoreNum = 0;
    uint32_t blockSplitRange = 0;
    uint32_t tailSplitRange = 0;
    uint32_t kvSplitPart = 1;
    uint32_t kvSegmentLen = maxActualSeqLen;

    if (splitMode == TQ_ATTN_SPLIT_BNS) {
        SplitBns(bn, maxActualSeqLen, blockSize, parallelCoreNum, usedCoreNum, kvSplitPart,
                 kvSegmentLen, formerCoreNum, blockSplitRange, tailSplitRange);
    } else {
        SplitBn(bn, parallelCoreNum, usedCoreNum, formerCoreNum, blockSplitRange, tailSplitRange);
    }

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
    tiling.set_kvSplitPart(kvSplitPart);
    tiling.set_qkPvMode(qkPvMode);
    tiling.set_kvSegmentLen(kvSegmentLen);
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_blockSplitRange(blockSplitRange);
    tiling.set_tailSplitRange(tailSplitRange);
    tiling.set_scaleValue(*scaleValuePtr);

    if (splitMode == TQ_ATTN_SPLIT_BNS) {
        tiling.set_accumOutSize(numHeads * kvSplitPart * headSize);
        tiling.set_logSumExpSize(numHeads * kvSplitPart * 2);
    } else {
        tiling.set_accumOutSize(0);
        tiling.set_logSumExpSize(0);
    }

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
    workspaces[0] = CalcWorkspaceSize(ascendcPlatform, splitMode, numHeads, headSize, kvSplitPart);
    if (workspaces[0] > MAX_USER_WORKSPACE) {
        OPS_LOG_E(nodeName, "workspace size %zu exceeds cap %zu", workspaces[0], MAX_USER_WORKSPACE);
        return ge::GRAPH_FAILED;
    }

    const uint32_t mixBlockDim =
        ascendcPlatform.CalcTschBlockDim(TQ_KFC_AIV_NUM, TQ_KFC_AIC_NUM, TQ_KFC_AIV_NUM);
    const uint32_t blockDim = std::max(1U, usedCoreNum) * mixBlockDim;
    context->SetBlockDim(blockDim);
    // Single MIX tiling key for now. splitMode/qkPvMode stay in tiling payload; extra
    // TILING_KEY_IS branches each register matmul and blow up Mc2 workspace analysis.
    const uint64_t tilingKey = optiling::TQ_ATTN_KEY_SPLITBN_VECTOR;
    context->SetTilingKey(tilingKey);
    OPS_LOG_I(nodeName,
              "TurboquantAttentionPaged8bit tiling: tokens=%u bn=%u maxKv=%u maxActual=%u "
              "split=%u qkpv=%u kvSplit=%u kvTile=%u usedCore=%u key=%lu logical=%lu ws=%zu",
              numTokens, bn, maxKvLen, maxActualSeqLen, splitMode, qkPvMode, kvSplitPart,
              kvTileRows, usedCoreNum, tilingKey, PickTilingKey(splitMode, qkPvMode),
              workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantAttentionPaged8bit)
    .Tiling(TurboquantAttentionPaged8bitTilingFunc);

}  // namespace optiling
