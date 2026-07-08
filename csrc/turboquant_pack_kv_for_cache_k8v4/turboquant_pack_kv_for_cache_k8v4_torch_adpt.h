#ifndef TURBOQUANT_PACK_KV_FOR_CACHE_K8V4_TORCH_ADPT_H
#define TURBOQUANT_PACK_KV_FOR_CACHE_K8V4_TORCH_ADPT_H

#include <cstdlib>
#include <torch/extension.h>
#include <torch/torch.h>
#include "../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

// K8V4 fused pack-to-cache. Key uses (codebook, rotation_t) at 8-bit, value uses
// (codebook_value, rotation_t_value) at 4-bit. Tables are passed explicitly
// because the 4-bit value tables are not part of the registered 8-bit table set.
inline void turboquant_pack_kv_for_cache_k8v4(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& codebook,
    const at::Tensor& rotation_t,
    const at::Tensor& codebook_value,
    const at::Tensor& rotation_t_value,
    const at::Tensor& slot_mapping,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    const bool no_kfc = std::getenv("VLLM_ASCEND_TURBOQUANT_NO_KFC") != nullptr;

    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack-to-cache accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "key_cache/value_cache must be uint8 views");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "key_cache/value_cache must be [num_blocks, block_size, num_heads, slot_w]");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack-to-cache supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit key");
    TORCH_CHECK(slot_w_v >= head_size / 2 + 2, "slot_w_v must be >= head_size/2+2 for 4-bit value");
    TORCH_CHECK(key_cache.size(3) == slot_w_k && value_cache.size(3) == slot_w_v,
                "cache last dim must match slot_w_k/slot_w_v");

    const int64_t n_vec = key.numel() / head_size;
    const int64_t num_heads = key_cache.size(2);
    TORCH_CHECK(num_heads > 0 && key.size(-2) == num_heads && value.size(-2) == num_heads,
                "key/value num_heads must match cache num_heads");

    at::Tensor key_work = key.is_contiguous() ? key : key.contiguous();
    at::Tensor value_work = value.is_contiguous() ? value : value.contiguous();
    at::Tensor slot_work = slot_mapping.is_contiguous() ? slot_mapping : slot_mapping.contiguous();
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
                "key_cache/value_cache uint8 views must be contiguous");

    const c10_npu::OptionalNPUGuard npuGuard(key_work.device());

    uint32_t vec_per_core = 128;
    if (n_vec < 128) {
        vec_per_core = static_cast<uint32_t>(((n_vec + 15) / 16) * 16);
        if (vec_per_core == 0) {
            vec_per_core = 16;
        }
    }
    const int64_t pack_mode = no_kfc ? 1 : 0;
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    const int64_t cache_slots = key_cache.size(0) * key_cache.size(1);

    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheK8v4,
        key_work,
        value_work,
        codebook,
        rotation_t,
        codebook_value,
        rotation_t_value,
        slot_work,
        pack_mode,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        num_heads,
        cache_slots,
        key_cache,
        value_cache);
}

}  // namespace vllm_ascend

#endif
