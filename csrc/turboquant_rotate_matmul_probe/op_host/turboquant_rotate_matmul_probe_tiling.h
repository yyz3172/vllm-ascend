#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TurboquantRotateMatmulProbeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, probeMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantRotateMatmulProbe, TurboquantRotateMatmulProbeTilingData)

}  // namespace optiling
