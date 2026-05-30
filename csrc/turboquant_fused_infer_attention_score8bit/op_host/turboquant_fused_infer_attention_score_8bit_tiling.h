#ifndef TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_8BIT_TILING_H
#define TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_8BIT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TurboquantFusedInferAttentionScore8bitTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, headSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq);
TILING_DATA_FIELD_DEF(float, scaleValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantFusedInferAttentionScore8bit, TurboquantFusedInferAttentionScore8bitTilingData)

}  // namespace optiling

#endif

