/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TurboquantDecodePaged8bitTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);  // KFC (mode 0) Cube y_hat @ R
TILING_DATA_FIELD_DEF(uint32_t, totalBlocks);           // 紧凑块数 = gather_block_ids.numel()
TILING_DATA_FIELD_DEF(uint32_t, blockSize);             // BS
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);            // H
TILING_DATA_FIELD_DEF(uint32_t, headSize);              // 128
TILING_DATA_FIELD_DEF(uint32_t, packedBytes);           // 130 (= headSize + 2)
TILING_DATA_FIELD_DEF(uint32_t, blocksPerCore);         // 每个逻辑核处理多少紧凑块
TILING_DATA_FIELD_DEF(uint32_t, outDtype);              // 0=fp16
TILING_DATA_FIELD_DEF(uint32_t, mode);                  // 0=KFC Cube, 1=AIV-only scalar
TILING_DATA_FIELD_DEF(uint32_t, kfcMixBlockDim);        // CalcTschBlockDim result (1=no MIX, 3=MIX_AIC_1_2)
TILING_DATA_FIELD_DEF(uint32_t, dataCores);              // 数据并行核数（用于 per-core workspace 偏移）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantDecodePaged8bit, TurboquantDecodePaged8bitTilingData)

}  // namespace optiling
