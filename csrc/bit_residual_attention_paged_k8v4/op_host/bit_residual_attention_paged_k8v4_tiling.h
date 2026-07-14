#ifndef BIT_RESIDUAL_ATTENTION_PAGED_K8V4_TILING_H
#define BIT_RESIDUAL_ATTENTION_PAGED_K8V4_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// BitResidual K8V4 cache layout constants (sign-reversal, 16-row sub-blocks).
// Must match decode_device.h and pack kernel common.h.
constexpr uint32_t TQ_BR_HEAD_SIZE = 128;
constexpr uint32_t TQ_BR_BLOCK_ROWS = 16;  // rows per sub-block
constexpr uint32_t TQ_BR_KEY_BLOCK_STRIDE = 2176;  // 8*256 code + 16*4 base + 16*4 step
constexpr uint32_t TQ_BR_VAL_BLOCK_STRIDE = 1152;  // 4*256 code + 16*4 vmin + 16*4 vstep

// Attention UB caps.
constexpr uint32_t TQ_BR_ATTN_KV_TILE_CAP = 64;
constexpr uint32_t TQ_BR_ATTN_GQA_CAP = 8;

// Cube/QFC constraints.
constexpr uint32_t TQ_BR_ATTN_MAX_PARALLEL_CORES = 20;

// Split mode constants.
constexpr uint32_t TQ_BR_ATTN_SPLIT_BN = 0;
constexpr uint32_t TQ_BR_ATTN_SPLIT_BNS = 1;
constexpr uint32_t TQ_BR_ATTN_QKPV_VECTOR = 0;
constexpr uint32_t TQ_BR_ATTN_QKPV_CUBE = 1;

// Tiling keys (initial release: only key 0 = SplitBN Vector).
constexpr uint32_t TQ_BR_ATTN_KEY_SPLITBN_VECTOR = 0;
constexpr uint64_t TQ_BR_ATTN_KEY_SPLITBNS_VECTOR = 1;

BEGIN_TILING_DATA_DEF(BitResidualAttentionPagedK8v4TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, decodeRotateTiling);
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
TILING_DATA_FIELD_DEF(uint32_t, qkPvMode);
TILING_DATA_FIELD_DEF(uint32_t, kvSegmentLen);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitRange);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitRange);
TILING_DATA_FIELD_DEF(uint32_t, kvSplitPart);
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize);
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize);
TILING_DATA_FIELD_DEF(uint64_t, partialWorkspaceOffset);
TILING_DATA_FIELD_DEF(float, scaleValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BitResidualAttentionPagedK8v4, BitResidualAttentionPagedK8v4TilingData)

}  // namespace optiling

#endif
