#ifndef ACLNN_TURBOQUANT_PACK_KV_FOR_CACHE4BIT_H_
#define ACLNN_TURBOQUANT_PACK_KV_FOR_CACHE4BIT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantPackKvForCache4bitGetWorkspaceSize(
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* codebook,
    const aclTensor* rotationT,
    const aclTensor* slotMapping,
    const aclTensor* queryStartLoc,
    int64_t nVec,
    int64_t vecPerCore,
    int64_t numHeads,
    int64_t blockSize,
    int64_t numBlocks,
    int64_t numReqs,
    int64_t keyStrideToken,
    int64_t keyStrideHead,
    int64_t valueStrideToken,
    int64_t valueStrideHead,
    int64_t keyStorageOffset,
    int64_t valueStorageOffset,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantPackKvForCache4bit(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
