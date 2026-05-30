#ifndef TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_8BIT_TORCH_ADPT_H
#define TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_8BIT_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch/torch.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

at::Tensor turboquant_fused_infer_attention_score_8bit(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_len_q,
    const at::Tensor& actual_seq_len_kv,
    const at::Tensor& atten_mask,
    const at::Tensor& codebook,
    const at::Tensor& rotation,
    const at::Tensor& codebook_value,
    const at::Tensor& rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value)
{
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "fp16 only in initial version");
    TORCH_CHECK(head_size == 128, "initial version only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");

    at::Tensor out = at::empty(query.sizes(), query.options().dtype(at::kHalf));

    EXEC_NPU_CMD(
        aclnnTurboquantFusedInferAttentionScore8bit,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_len_q,
        actual_seq_len_kv,
        atten_mask,
        codebook,
        rotation,
        codebook_value,
        rotation_value,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        scale_value,
        out);

    return out;
}

}  // namespace vllm_ascend

#endif

