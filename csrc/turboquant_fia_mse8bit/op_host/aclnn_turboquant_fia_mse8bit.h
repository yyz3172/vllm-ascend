#ifndef ACLNN_TURBOQUANT_FIA_MSE8BIT_H_
#define ACLNN_TURBOQUANT_FIA_MSE8BIT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FIA TQ MSE-8bit fused attention (vllm-ascend custom op).
 *
 * key/value_cache : int8 [BN, BS, KV*D]
 * key/value_scale : fp16 gamma [phys_tokens, KV]
 * rotation        : fp16 Pi [D, D]
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantFiaMse8bitGetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    const aclTensor* blockTable,
    const aclIntArray* actualSeqLenQ,
    const aclIntArray* actualSeqLenKv,
    const aclTensor* attenMask,
    const aclTensor* keyScale,
    const aclTensor* valueScale,
    const aclTensor* rotation,
    int64_t numHeads,
    int64_t numKvHeads,
    int64_t headSize,
    int64_t blockSize,
    double scaleValue,
    const aclTensor* attentionOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantFiaMse8bit(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
