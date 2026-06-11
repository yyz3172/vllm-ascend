#ifndef ACLNN_TURBOQUANT_ATTENTION_PAGED8BIT_H_
#define ACLNN_TURBOQUANT_ATTENTION_PAGED8BIT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantAttentionPaged8bitGetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    const aclTensor* blockTable,
    const aclIntArray* actualSeqLenQ,      // 修改为 aclIntArray*
    const aclIntArray* actualSeqLenKv,     // 修改为 aclIntArray*
    const aclTensor* codebook,
    const aclTensor* rotation,
    const aclTensor* codebookValue,
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
aclnnStatus aclnnTurboquantAttentionPaged8bit(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
