#ifndef ACLNN_TURBOQUANT_ROTATE_MATMUL_PROBE_H_
#define ACLNN_TURBOQUANT_ROTATE_MATMUL_PROBE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantRotateMatmulProbeGetWorkspaceSize(
    const aclTensor* a,
    const aclTensor* b,
    int64_t probeMode,
    int64_t m,
    const aclTensor* c,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnTurboquantRotateMatmulProbe(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
