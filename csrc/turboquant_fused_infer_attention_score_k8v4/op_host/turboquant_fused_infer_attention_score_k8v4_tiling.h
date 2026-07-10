#ifndef TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_K8V4_TILING_H
#define TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_K8V4_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t TQ_K8V4_HEAD_SIZE = 128;
constexpr uint32_t TQ_K8V4_K_PACKED_BYTES = 130;
constexpr uint32_t TQ_K8V4_V_PACKED_BYTES = 66;
constexpr uint32_t TQ_K8V4_KV_TILE_ROWS = 16;
constexpr uint32_t TQ_K8V4_UB_KV_TILE_CAP = 32;
constexpr uint32_t TQ_K8V4_UB_GQA_CAP = 8;
constexpr uint32_t TQ_K8V4_MAX_PARALLEL_CORES = 8;
constexpr uint32_t TQ_K8V4_CUBE_MIN_G = 8;
constexpr uint32_t TQ_K8V4_CUBE_MIN_TILE = 16;
constexpr uint32_t TQ_K8V4_QKPV_VECTOR = 0;
constexpr uint32_t TQ_K8V4_QKPV_CUBE = 1;

// Tiling keys:
//   0 = Vector QK/PV + KFC decode rotate
//   1 = Cube QK/PV   + KFC decode rotate
constexpr uint64_t TQ_K8V4_KEY_VECTOR = 0;
constexpr uint64_t TQ_K8V4_KEY_CUBE = 1;

BEGIN_TILING_DATA_DEF(TurboquantFusedInferAttentionScoreK8v4TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, decodeRotateTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, pvTiling);
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, gqaGroupSize);
TILING_DATA_FIELD_DEF(uint32_t, headSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq);
TILING_DATA_FIELD_DEF(uint32_t, totalCacheBlocks);
TILING_DATA_FIELD_DEF(uint32_t, keyRowBytes);
TILING_DATA_FIELD_DEF(uint32_t, valueRowBytes);
TILING_DATA_FIELD_DEF(uint32_t, kvTileRows);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitRange);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitRange);
TILING_DATA_FIELD_DEF(uint32_t, qkPvMode);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, cacheNormBf16);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantFusedInferAttentionScoreK8v4, TurboquantFusedInferAttentionScoreK8v4TilingData)

}  // namespace optiling

#endif
