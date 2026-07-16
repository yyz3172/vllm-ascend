#ifndef BIT_RESIDUAL_FIA_PAGED_K8V4_TORCH_ADPT_H
#define BIT_RESIDUAL_FIA_PAGED_K8V4_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch/torch.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

/**
 * Torch adapter for BitResidual FIA Paged K8V4.
 *
 * key/value_cache : uint8 pack layout [num_blocks, num_kv_heads, packed_bytes]
 * rotation_key/value : fp16 [D, D]
 */
at::Tensor bit_residual_fia_paged_k8v4(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_len_q,
    const at::Tensor& actual_seq_len_kv,
    const c10::optional<at::Tensor>& atten_mask,
    const at::Tensor& rotation_key,
    const at::Tensor& rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value,
    int64_t pre_tokens = 2147483647,
    int64_t next_tokens = 2147483647,
    int64_t sparse_mode = 0)
{
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "fp16 query only in P0");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "uint8 key_cache");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "uint8 value_cache");
    TORCH_CHECK(head_size == 128, "head_size=128 only in P0");

    at::Tensor out = at::empty(query.sizes(), query.options().dtype(at::kHalf));
    at::Tensor mask = atten_mask.has_value() && atten_mask->defined()
                          ? atten_mask.value()
                          : at::empty({0}, query.options().dtype(at::kChar));

    EXEC_NPU_CMD(
        aclnnBitResidualFiaPagedK8v4,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_len_q,
        actual_seq_len_kv,
        mask,
        rotation_key,
        rotation_value,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        scale_value,
        pre_tokens,
        next_tokens,
        sparse_mode,
        out);

    return out;
}

}  // namespace vllm_ascend

#endif
