#ifndef TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_K8V4_TORCH_ADPT_H
#define TURBOQUANT_FUSED_INFER_ATTENTION_SCORE_K8V4_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch/torch.h>
#include "../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline at::Tensor _fp16_contiguous_for_fia(const at::Tensor& tensor)
{
    if (tensor.scalar_type() == at::kHalf) {
        return tensor.contiguous();
    }
    return tensor.to(at::kHalf).contiguous();
}

at::Tensor turboquant_fused_infer_attention_score_k8v4(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& atten_mask,
    const at::Tensor& actual_seq_len_q,
    const at::Tensor& actual_seq_len_kv,
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
    TORCH_CHECK(
        query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query must be fp16 or bf16");
    TORCH_CHECK(head_size == 128, "initial version only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(atten_mask.defined(), "atten_mask is required");

    // bf16 norm decode in-kernel is still being validated; rewrite norms in Python
    // when query is bf16 (see turboquant_fused_infer_attention_score_k8v4).
    const int64_t cache_norm_bf16 = 0;
    const at::Tensor query_fp16 = _fp16_contiguous_for_fia(query);
    const at::Tensor atten_mask_fp16 = _fp16_contiguous_for_fia(atten_mask);

    at::Tensor out = at::empty(query_fp16.sizes(), query_fp16.options());

    EXEC_NPU_CMD(
        aclnnTurboquantFusedInferAttentionScoreK8v4,
        query_fp16,
        key_cache,
        value_cache,
        block_table,
        actual_seq_len_q,
        actual_seq_len_kv,
        atten_mask_fp16,
        codebook,
        rotation,
        codebook_value,
        rotation_value,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        scale_value,
        cache_norm_bf16,
        out);

    if (query.scalar_type() == at::kBFloat16) {
        return out.to(at::kBFloat16);
    }
    return out;
}

}  // namespace vllm_ascend

#endif
