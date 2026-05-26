#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TurboquantPackKvForCacheFusedTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, nVec);
TILING_DATA_FIELD_DEF(uint32_t, vecPerCore);
TILING_DATA_FIELD_DEF(uint32_t, slotWK);
TILING_DATA_FIELD_DEF(uint32_t, slotWV);
TILING_DATA_FIELD_DEF(uint32_t, packMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantPackKvForCacheFused, TurboquantPackKvForCacheFusedTilingData)

}  // namespace optiling
