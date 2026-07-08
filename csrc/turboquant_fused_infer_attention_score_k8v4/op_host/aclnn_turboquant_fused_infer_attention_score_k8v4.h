#ifndef ACLNN_TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_K8V4_H_
#define ACLNN_TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_K8V4_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantFusedInferAttentionScoreK8v4GetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    const aclTensor* blockTable,
    const aclTensor* actualSeqLenQ,
    const aclTensor* actualSeqLenKv,
    const aclTensor* attenMask,
    const aclTensor* codebook,
    const aclTensor* rotation,
    const aclTensor* codebookValue,
    const aclTensor* rotationValue,
    int64_t numHeads,
    int64_t numKvHeads,
    int64_t headSize,
    int64_t blockSize,
    double scaleValue,
    const aclTensor* attentionOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantFusedInferAttentionScoreK8v4(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
