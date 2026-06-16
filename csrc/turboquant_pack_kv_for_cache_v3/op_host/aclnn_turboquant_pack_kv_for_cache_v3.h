#ifndef ACLNN_TURBOQUANT_PACK_KV_FOR_CACHE_V3_H_
#define ACLNN_TURBOQUANT_PACK_KV_FOR_CACHE_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantPackKvForCacheV3GetWorkspaceSize(
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* codebook,
    const aclTensor* rotationT,
    int64_t nVec,
    int64_t slotWK,
    int64_t slotWV,
    int64_t vecPerCore,
    const aclTensor* packedK,
    const aclTensor* packedV,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantPackKvForCacheV3(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
