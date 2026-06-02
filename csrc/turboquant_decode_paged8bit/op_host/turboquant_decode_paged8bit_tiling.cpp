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
#include "turboquant_decode_paged8bit_tiling.h"

#include <algorithm>

namespace {

constexpr uint32_t TQ_DECODE_HEAD_SIZE = 128;
constexpr uint32_t TQ_DECODE_PACKED_BYTES = TQ_DECODE_HEAD_SIZE + 2;  // 130
constexpr uint32_t TQ_DECODE_M_ALIGN = 16;
constexpr uint32_t TQ_DECODE_T_ROWS = 32;             // tile size (§2.6.3)
constexpr uint32_t TQ_DECODE_MODE_KFC = 0;
constexpr uint32_t TQ_DECODE_MODE_AIV = 1;
constexpr uint32_t TQ_DECODE_KFC_AIC_NUM = 1;
constexpr uint32_t TQ_DECODE_KFC_AIV_NUM = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;

uint32_t AlignUp16(uint32_t x) {
    return (x + TQ_DECODE_M_ALIGN - 1) / TQ_DECODE_M_ALIGN * TQ_DECODE_M_ALIGN;
}

// KFC Cube tiling for y_hat @ R: A fp16 VECOUT, B fp16 GM (rotation), C fp32 GM.
// Same five-tuple as the validated pack op; M fixed to the tile size T_rows.
ge::graphStatus FillKfcCubeTiling(
    const platform_ascendc::PlatformAscendC& platform,
    optiling::TCubeTiling& cubeTiling) {
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
    tilingApi.SetDim(1);
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
    const uint32_t mPad = AlignUp16(TQ_DECODE_T_ROWS);
    tilingApi.SetOrgShape(mPad, TQ_DECODE_HEAD_SIZE, TQ_DECODE_HEAD_SIZE);
    tilingApi.SetShape(mPad, TQ_DECODE_HEAD_SIZE, TQ_DECODE_HEAD_SIZE);
    tilingApi.EnableBias(false);
    tilingApi.SetBufferSpace(l1Size, l0cSize, ubSize);
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    cubeTiling.set_baseM(mPad);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace optiling {

static ge::graphStatus TurboquantDecodePaged8bitTilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }

    const int64_t* headSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* totalBlocksPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t* rowsPerCorePtr = attrs->GetAttrPointer<int64_t>(4);  // legacy, unused
    const int64_t* outDtypePtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t* modePtr = attrs->GetAttrPointer<int64_t>(6);
    if (headSizePtr == nullptr || blockSizePtr == nullptr || numKvHeadsPtr == nullptr ||
        totalBlocksPtr == nullptr || rowsPerCorePtr == nullptr || outDtypePtr == nullptr ||
        modePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }
    (void)rowsPerCorePtr;

    const uint32_t headSize = static_cast<uint32_t>(*headSizePtr);
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t numKvHeads = static_cast<uint32_t>(*numKvHeadsPtr);
    const uint32_t totalBlocks = static_cast<uint32_t>(*totalBlocksPtr);
    const uint32_t outDtype = static_cast<uint32_t>(*outDtypePtr);
    const uint32_t mode = static_cast<uint32_t>(*modePtr);

    if (headSize != TQ_DECODE_HEAD_SIZE) {
        OPS_LOG_E(nodeName, "only head_size=128 supported, got %u", headSize);
        return ge::GRAPH_FAILED;
    }
    if (totalBlocks == 0) {
        OPS_LOG_E(nodeName, "totalBlocks must be > 0");
        return ge::GRAPH_FAILED;
    }
    if (numKvHeads == 0 || blockSize == 0) {
        OPS_LOG_E(nodeName, "blockSize/numKvHeads must be > 0");
        return ge::GRAPH_FAILED;
    }
    if (mode > TQ_DECODE_MODE_AIV) {
        OPS_LOG_E(nodeName, "mode must be 0 (KFC) or 1 (AIV-only), got %u", mode);
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    TurboquantDecodePaged8bitTilingData tiling{};
    if (FillKfcCubeTiling(ascendcPlatform, tiling.cubeTiling) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "failed to fill cube tiling");
        return ge::GRAPH_FAILED;
    }

    // Split compact blocks across cores; a block (BS*H packed rows) is the work
    // unit so a tile never straddles a block boundary.
    const uint32_t coreNum = (mode == TQ_DECODE_MODE_KFC)
        ? static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic())
        : static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    const uint32_t usableCores = std::max<uint32_t>(1, coreNum);
    const uint32_t blocksPerCore = std::max<uint32_t>(1, (totalBlocks + usableCores - 1) / usableCores);
    const uint32_t dataCores = (totalBlocks + blocksPerCore - 1) / blocksPerCore;

    tiling.set_totalBlocks(totalBlocks);
    tiling.set_blockSize(blockSize);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_headSize(headSize);
    tiling.set_packedBytes(TQ_DECODE_PACKED_BYTES);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_outDtype(outDtype);
    tiling.set_mode(mode);

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tiling.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    uint32_t blockDim = dataCores;
    if (mode == TQ_DECODE_MODE_KFC) {
        // MIX 1 AIC + 2 AIV per data-parallel group (same as the pack op).
        const uint32_t mixBlockDim = ascendcPlatform.CalcTschBlockDim(
            TQ_DECODE_KFC_AIV_NUM, TQ_DECODE_KFC_AIC_NUM, TQ_DECODE_KFC_AIV_NUM);
        blockDim = dataCores * mixBlockDim;
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = SYSTEM_NEED_WORKSPACE;

    context->SetBlockDim(blockDim);
    context->SetTilingKey(mode);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboquantDecodePaged8bit)
    .Tiling(TurboquantDecodePaged8bitTilingFunc);

}  // namespace optiling
