#ifndef ACLNN_TURBOQUANT_DECODE_PAGED_8BIT_H_
#define ACLNN_TURBOQUANT_DECODE_PAGED_8BIT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantDecodePaged8bitGetWorkspaceSize(
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    const aclTensor* gatherBlockIds,
    const aclTensor* codebook,
    const aclTensor* rotation,
    int64_t headSize,
    int64_t blockSize,
    int64_t numKvHeads,
    int64_t totalBlocks,
    int64_t rowsPerCore,
    int64_t outDtype,
    int64_t mode,
    const aclTensor* keyOut,
    const aclTensor* valueOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantDecodePaged8bit(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
