#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BitResidualPackKvForCacheTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, nVec);
TILING_DATA_FIELD_DEF(uint32_t, vecPerCore);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BitResidualPackKvForCache, BitResidualPackKvForCacheTilingData)

}  // namespace optiling
