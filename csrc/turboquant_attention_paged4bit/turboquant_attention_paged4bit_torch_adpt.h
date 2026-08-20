#ifndef TURBOQUANT_ATTENTION_PAGED4BIT_TORCH_ADPT_H
#define TURBOQUANT_ATTENTION_PAGED4BIT_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch/torch.h>
#include "../aclnn_torch_adapter/op_api_common.h"
#include "op_host/aclnn_turboquant_attention_paged4bit.h"

namespace vllm_ascend {

constexpr int64_t TQ_ATTN_4BIT_HEAD_SIZE = 128;
constexpr int64_t TQ_ATTN_4BIT_ROW_BYTES = TQ_ATTN_4BIT_HEAD_SIZE / 2 + 2;

at::Tensor turboquant_attention_paged4bit(
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
    TORCH_CHECK(query.scalar_type() == at::kHalf ||
                    query.scalar_type() == at::kBFloat16,
                "4-bit attention accepts fp16/bf16 query");
    TORCH_CHECK(head_size == TQ_ATTN_4BIT_HEAD_SIZE,
                "initial version only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(max_actual_seq_len > 0, "max_actual_seq_len must be > 0");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(actual_seq_len_q.size() > 0, "actual_seq_len_q must not be empty");
    TORCH_CHECK(actual_seq_len_q.size() == actual_seq_len_kv.size(),
                "actual_seq_len_q and actual_seq_len_kv must have the same length");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(codebook.scalar_type() == query.scalar_type() && codebook.numel() == 16,
                "codebook must have 16 entries and match query dtype");
    TORCH_CHECK(codebook_value.scalar_type() == query.scalar_type() && codebook_value.numel() == 16,
                "codebook_value must have 16 entries and match query dtype");
    TORCH_CHECK(rotation.scalar_type() == query.scalar_type() && rotation.numel() == head_size * head_size,
                "rotation must be [128, 128] and match query dtype");
    TORCH_CHECK(rotation_value.scalar_type() == query.scalar_type() &&
                    rotation_value.numel() == head_size * head_size,
                "rotation_value must be [128, 128] and match query dtype");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "4-bit slab caches must be [num_blocks, num_kv_heads, block_size * 66]");
    TORCH_CHECK(key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads,
                "slab cache num_kv_heads mismatch");
    TORCH_CHECK(key_cache.size(2) == block_size * TQ_ATTN_4BIT_ROW_BYTES &&
                    value_cache.size(2) == block_size * TQ_ATTN_4BIT_ROW_BYTES,
                "slab cache row span mismatch");

    at::Tensor out = at::empty(query.sizes(), query.options());

    EXEC_NPU_CMD(
        aclnnTurboquantAttentionPaged4bit,
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
