#ifndef ACLNN_TURBOQUANT_PACK_KV_FOR_CACHE_TO_CACHE_H_
#define ACLNN_TURBOQUANT_PACK_KV_FOR_CACHE_TO_CACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantPackKvForCacheToCacheGetWorkspaceSize(
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* codebook,
    const aclTensor* rotationT,
    const aclTensor* slotMapping,
    int64_t packMode,
    int64_t nVec,
    int64_t slotWK,
    int64_t slotWV,
    int64_t vecPerCore,
    int64_t numHeads,
    int64_t cacheSlots,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantPackKvForCacheToCache(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
