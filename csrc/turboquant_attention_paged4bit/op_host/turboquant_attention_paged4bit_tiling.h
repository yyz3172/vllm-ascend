#ifndef TURBOQUANT_ATTENTION_PAGED4BIT_TILING_H
#define TURBOQUANT_ATTENTION_PAGED4BIT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// Matmul scope (plan todo: confirm-matmul-scope):
// - decode rotate: always Cube KFC (y_hat[M,128] @ R[128,128])
// - QK/PV: Cube only when G >= TQ_ATTN_CUBE_MIN_G and kvTileRows >= TQ_ATTN_CUBE_MIN_TILE;
//          otherwise Vector VMLA fallback (better for M=1 / small G decode).
constexpr uint32_t TQ_ATTN_HEAD_SIZE = 128;
constexpr uint32_t TQ_ATTN_PACKED_BYTES = 66;
constexpr uint32_t TQ_ATTN_KV_TILE_ROWS = 64;
constexpr uint32_t TQ_ATTN_CUBE_MIN_G = 8;
constexpr uint32_t TQ_ATTN_CUBE_MIN_TILE = 32;
// Must match kernel UB caps in turboquant_attention_paged4bit.cpp.
// CANN Mc2 GetWorkspaceSize static analysis uses these compile-time InitBuffer
// extents (not runtime kvTileRows/gqa).
constexpr uint32_t TQ_ATTN_UB_KV_TILE_CAP = 64;
constexpr uint32_t TQ_ATTN_UB_GQA_CAP = 8;
constexpr uint32_t TQ_ATTN_QTILE_SMALL_GQA_CAP = 2;
// Mc2 sys workspace is shared across launched MIX blocks (same wsBase); cubeC
// scratch lives at a fixed +256KB offset. Limit concurrent KFC cores to avoid hang.
constexpr uint32_t TQ_ATTN_MAX_PARALLEL_CORES = 20;
constexpr uint32_t TQ_ATTN_SPLIT_BN = 0;
constexpr uint32_t TQ_ATTN_SPLIT_BNS = 1;
constexpr uint32_t TQ_ATTN_QKPV_VECTOR = 0;
constexpr uint32_t TQ_ATTN_QKPV_CUBE = 1;

// Tiling keys:
//   0 = SplitBN  + Vector QK/PV + KFC decode
//   1 = SplitBN  + Cube QK/PV   + KFC decode
//   2 = SplitBNS + Vector QK/PV + KFC decode
//   3 = SplitBNS + Cube QK/PV   + KFC decode
//   4 = SplitBN  + Vector QK/PV + KFC decode + qTile
//   5 = SplitBN  + Vector QK/PV + KFC decode + qTile, GQA<=2 scratch
constexpr uint64_t TQ_ATTN_KEY_SPLITBN_VECTOR = 0;
constexpr uint64_t TQ_ATTN_KEY_SPLITBN_CUBE = 1;
constexpr uint64_t TQ_ATTN_KEY_SPLITBNS_VECTOR = 2;
constexpr uint64_t TQ_ATTN_KEY_SPLITBNS_CUBE = 3;
constexpr uint64_t TQ_ATTN_KEY_SPLITBN_VECTOR_QTILE = 4;
constexpr uint64_t TQ_ATTN_KEY_SPLITBN_VECTOR_QTILE_GQA2 = 5;

BEGIN_TILING_DATA_DEF(TurboquantAttentionPaged4bitTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, decodeRotateTiling);
// QK/PV cube tilings are computed on host when qkPvMode==CUBE but are not stored in
// tiling data yet. Keeping only decodeRotateTiling here avoids CANN Mc2 static
// workspace analysis treating this op as a 3-matmul kernel before host tiling runs.
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, gqaGroupSize);
TILING_DATA_FIELD_DEF(uint32_t, headSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq);
TILING_DATA_FIELD_DEF(uint32_t, totalCacheBlocks);
TILING_DATA_FIELD_DEF(uint32_t, maxKvLen);
TILING_DATA_FIELD_DEF(uint32_t, maxActualSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, kvTileRows);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, splitMode);
TILING_DATA_FIELD_DEF(uint32_t, kvSplitPart);
TILING_DATA_FIELD_DEF(uint32_t, qkPvMode);
TILING_DATA_FIELD_DEF(uint32_t, kvSegmentLen);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitRange);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitRange);
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize);
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize);
TILING_DATA_FIELD_DEF(uint64_t, partialWorkspaceOffset);
TILING_DATA_FIELD_DEF(float, scaleValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboquantAttentionPaged4bit, TurboquantAttentionPaged4bitTilingData)

}  // namespace optiling

#endif
