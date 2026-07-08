#ifndef ACLNN_BIT_RESIDUAL_PACK_KV_FOR_CACHE_H_
#define ACLNN_BIT_RESIDUAL_PACK_KV_FOR_CACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnBitResidualPackKvForCacheGetWorkspaceSize(
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* rotationT,
    const aclTensor* slotMapping,
    int64_t nVec,
    int64_t vecPerCore,
    int64_t numHeads,
    int64_t blockSize,
    int64_t numBlocks,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnBitResidualPackKvForCache(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
