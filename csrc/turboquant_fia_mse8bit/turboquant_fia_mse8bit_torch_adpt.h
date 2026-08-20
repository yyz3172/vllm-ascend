#ifndef TURBOQUANT_FIA_MSE8BIT_TORCH_ADPT_H
#define TURBOQUANT_FIA_MSE8BIT_TORCH_ADPT_H

#include <torch/extension.h>
#include <torch/torch.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

/**
 * Torch adapter for FIA TQ MSE-8bit.
 *
 * key/value_cache : int8 [BN, BS, KV*D]
 * key/value_scale : fp16 gamma
 * rotation        : fp16 Pi [D, D]
 */
at::Tensor turboquant_fia_mse8bit(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_len_q,
    const at::Tensor& actual_seq_len_kv,
    const c10::optional<at::Tensor>& atten_mask,
    const at::Tensor& key_scale,
    const at::Tensor& value_scale,
    const at::Tensor& rotation,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value)
{
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "fp16 query only");
    TORCH_CHECK(key_cache.scalar_type() == at::kChar, "int8 key_cache (FIA idx)");
    TORCH_CHECK(head_size == 128, "head_size=128 only in P0");

    at::Tensor out = at::empty(query.sizes(), query.options().dtype(at::kHalf));
    at::Tensor mask = atten_mask.has_value() && atten_mask->defined()
                          ? atten_mask.value()
                          : at::empty({0}, query.options().dtype(at::kChar));

    EXEC_NPU_CMD(
        aclnnTurboquantFiaMse8bit,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_len_q,
        actual_seq_len_kv,
        mask,
        key_scale,
        value_scale,
        rotation,
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
