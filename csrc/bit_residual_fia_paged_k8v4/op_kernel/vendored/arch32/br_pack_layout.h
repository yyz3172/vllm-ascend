/**
 * BitResidual K8V4 pack cache layout (must match bit_residual_pack_k8v4/IMPL_DESIGN.md).
 */
#ifndef BR_PACK_LAYOUT_H
#define BR_PACK_LAYOUT_H

namespace br_pack {

static constexpr uint32_t BR_HEAD_SIZE = 128U;
static constexpr uint32_t BR_BLOCK_ROWS = 16U;
static constexpr uint32_t BR_KEY_CODE_BYTES = BR_HEAD_SIZE;
static constexpr uint32_t BR_VAL_CODE_BYTES = BR_HEAD_SIZE / 2U;
static constexpr uint32_t BR_META_BYTES = 4U;  // base+step or vmin+vstep per row
static constexpr uint32_t BR_KEY_ROW_BYTES = BR_KEY_CODE_BYTES + BR_META_BYTES;
static constexpr uint32_t BR_VAL_ROW_BYTES = BR_VAL_CODE_BYTES + BR_META_BYTES;
static constexpr uint32_t BR_KEY_TILE_BYTES = BR_BLOCK_ROWS * BR_KEY_ROW_BYTES;   // 2112
static constexpr uint32_t BR_VAL_TILE_BYTES = BR_BLOCK_ROWS * BR_VAL_ROW_BYTES;   // 1088

__aicore__ inline uint32_t BrKeyHeadStride(uint32_t blockSize)
{
    return blockSize * BR_KEY_ROW_BYTES;
}

__aicore__ inline uint32_t BrValHeadStride(uint32_t blockSize)
{
    return blockSize * BR_VAL_ROW_BYTES;
}

__aicore__ inline uint64_t BrKeyHeadBase(uint32_t physBlock, uint32_t kvHead, uint32_t numKvHeads,
                                         uint32_t blockSize)
{
    return (static_cast<uint64_t>(physBlock) * numKvHeads + kvHead) *
           static_cast<uint64_t>(BrKeyHeadStride(blockSize));
}

__aicore__ inline uint64_t BrValHeadBase(uint32_t physBlock, uint32_t kvHead, uint32_t numKvHeads,
                                         uint32_t blockSize)
{
    return (static_cast<uint64_t>(physBlock) * numKvHeads + kvHead) *
           static_cast<uint64_t>(BrValHeadStride(blockSize));
}

__aicore__ inline uint64_t BrKeyCodeOffset(uint64_t headBase, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(posInBlock) * BR_KEY_CODE_BYTES;
}

__aicore__ inline uint64_t BrKeyBaseOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * BR_KEY_CODE_BYTES +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t BrKeyStepOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * (BR_KEY_CODE_BYTES + 2U) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t BrValCodeOffset(uint64_t headBase, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(posInBlock) * BR_VAL_CODE_BYTES;
}

__aicore__ inline uint64_t BrValVminOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * BR_VAL_CODE_BYTES +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t BrValVstepOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * (BR_VAL_CODE_BYTES + 2U) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

}  // namespace br_pack

#endif
