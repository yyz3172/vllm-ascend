#ifndef TURBOQUANT_ATTENTION_PAGED8BIT_TORCH_ADPT_H
#define TURBOQUANT_ATTENTION_PAGED8BIT_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch/torch.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "turboquant_attention_paged8bit/op_host/aclnn_turboquant_attention_paged8bit.h"

namespace vllm_ascend {

at::Tensor turboquant_attention_paged8bit(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    at::IntArrayRef actual_seq_len_q,
    at::IntArrayRef actual_seq_len_kv,
    const at::Tensor& codebook,
    const at::Tensor& rotation,
    const at::Tensor& codebook_value,
    const at::Tensor& rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t max_actual_seq_len,
    double scale_value)
{
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "fp16 only in initial version");
    TORCH_CHECK(head_size == 128, "initial version only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(max_actual_seq_len > 0, "max_actual_seq_len must be > 0");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(actual_seq_len_q.size() > 0, "actual_seq_len_q must not be empty");
    TORCH_CHECK(actual_seq_len_q.size() == actual_seq_len_kv.size(),
                "actual_seq_len_q and actual_seq_len_kv must have the same length");

    at::Tensor out = at::empty(query.sizes(), query.options().dtype(at::kHalf));

    EXEC_NPU_CMD(
        aclnnTurboquantAttentionPaged8bit,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_len_q,
        actual_seq_len_kv,
        codebook,
        rotation,
        codebook_value,
        rotation_value,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_actual_seq_len,
        scale_value,
        out);

    return out;
}

}  // namespace vllm_ascend

#endif
