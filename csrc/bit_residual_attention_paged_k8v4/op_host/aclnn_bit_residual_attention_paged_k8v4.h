#ifndef ACLNN_BIT_RESIDUAL_ATTENTION_PAGED_K8V4_H_
#define ACLNN_BIT_RESIDUAL_ATTENTION_PAGED_K8V4_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnBitResidualAttentionPagedK8v4GetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    const aclTensor* blockTable,
    const aclIntArray* actualSeqLenQ,
    const aclIntArray* actualSeqLenKv,
    const aclTensor* rotationKey,
    const aclTensor* rotationValue,
    int64_t numHeads,
    int64_t numKvHeads,
    int64_t headSize,
    int64_t blockSize,
    int64_t maxActualSeqLen,
    double scaleValue,
    const aclTensor* attentionOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnBitResidualAttentionPagedK8v4(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
