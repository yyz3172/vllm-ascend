#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TurboquantPackKvForCacheV3ToCacheTilingData)
TILING_DATA_FIELD_DEF(uint32_t, nVec);
TILING_DATA_FIELD_DEF(uint32_t, vecPerCore);
TILING_DATA_FIELD_DEF(uint32_t, slotWK);
TILING_DATA_FIELD_DEF(uint32_t, slotWV);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, cacheSlots);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantPackKvForCacheV3ToCache, TurboquantPackKvForCacheV3ToCacheTilingData)

}  // namespace optiling
