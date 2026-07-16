/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "ops.h"
#include "utils.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#include "apply_top_k_top_p_custom/apply_top_k_top_p_custom_torch_adpt.h"
#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
#include "dispatch_gmm_combine_decode/dispatch_gmm_combine_decode_torch_adpt.h"
#include "dispatch_layout/dispatch_layout_torch_adpt.h"
#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
#include "lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h"
#include "matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h"
#include "mla_preprocess/mla_preprocess_torch_adpt.h"
#include "moe_combine_normal/moe_combine_normal_torch_adpt.h"
#include "moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"
#include "sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
#include "turboquant_rotate_matmul_probe/op_host/aclnn_turboquant_rotate_matmul_probe.h"
#include "turboquant_pack_kv_for_cache_fused/op_host/aclnn_turboquant_pack_kv_for_cache_fused.h"
#include "turboquant_pack_kv_for_cache_v2/op_host/aclnn_turboquant_pack_kv_for_cache_v2.h"
#include "turboquant_pack_kv_for_cache_v2_to_cache/op_host/aclnn_turboquant_pack_kv_for_cache_v2_to_cache.h"
#include "turboquant_pack_kv_for_cache_v3/op_host/aclnn_turboquant_pack_kv_for_cache_v3.h"
#include "bit_residual_pack_k8v4/op_host/aclnn_bit_residual_pack_k8v4.h"
#include "bit_residual_attention_paged_k8v4/op_host/aclnn_bit_residual_attention_paged_k8v4.h"
#include "bit_residual_fia_paged_k8v4/op_host/aclnn_bit_residual_fia_paged_k8v4.h"
#include "turboquant_pack_kv_for_cache4bit/op_host/aclnn_turboquant_pack_kv_for_cache4bit.h"
#include <mutex>
#include <unordered_map>
#include "turboquant_fused_infer_attention_score8bit/op_host/aclnn_turboquant_fused_infer_attention_score8bit.h"
#include "turboquant_attention_paged8bit/op_host/aclnn_turboquant_attention_paged8bit.h"
#include "turboquant_attention_paged4bit/turboquant_attention_paged4bit_torch_adpt.h"
#include "turboquant_decode_paged8bit/op_host/aclnn_turboquant_decode_paged8bit.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace vllm_ascend {

namespace {

struct TurboquantPackTableEntry {
    at::Tensor codebook;
    at::Tensor rotation_t;
    bool valid = false;
};

std::mutex g_turboquant_pack_tables_mu;
std::unordered_map<c10::DeviceIndex, TurboquantPackTableEntry> g_turboquant_pack_tables;

TurboquantPackTableEntry &GetTurboquantPackTables(const at::Tensor &ref) {
    std::lock_guard<std::mutex> lock(g_turboquant_pack_tables_mu);
    return g_turboquant_pack_tables[ref.device().index()];
}

at::Tensor MaybeContiguous(const at::Tensor &tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

}  // namespace

void turboquant_pack_register_tables(
    const at::Tensor &codebook,
    const at::Tensor &rotation_t) {
    TORCH_CHECK(codebook.is_privateuseone(), "codebook must be on NPU");
    TORCH_CHECK(rotation_t.is_privateuseone(), "rotation_t must be on NPU");
    TORCH_CHECK(codebook.scalar_type() == at::kHalf, "codebook must be fp16");
    TORCH_CHECK(rotation_t.scalar_type() == at::kHalf, "rotation_t must be fp16");
    TORCH_CHECK(codebook.numel() == 256, "8-bit codebook must have 256 entries");
    TORCH_CHECK(rotation_t.dim() == 2 && rotation_t.size(0) == 128 && rotation_t.size(1) == 128,
                "rotation_t must be [128,128]");
    TORCH_CHECK(rotation_t.device() == codebook.device(),
                "codebook and rotation_t must be on the same NPU device");

    std::lock_guard<std::mutex> lock(g_turboquant_pack_tables_mu);
    auto &entry = g_turboquant_pack_tables[codebook.device().index()];
    entry.codebook = codebook.contiguous();
    entry.rotation_t = rotation_t.contiguous();
    entry.valid = true;
}

// TurboQuant pack: registered tables + in-kernel bf16->fp16 cast to avoid host aten::to.
std::tuple<at::Tensor, at::Tensor> turboquant_pack_kv_for_cache(
    const at::Tensor &key,
    const at::Tensor &value,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    const bool no_kfc = std::getenv("VLLM_ASCEND_TURBOQUANT_NO_KFC") != nullptr;

    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(key.dim() >= 1 && value.dim() >= 1, "key/value must have at least 1 dim");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack accepts fp16/bf16 key/value");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= head_size + 2, "slot_w_v must be >= head_size+2 for 8-bit");

    auto &tables = GetTurboquantPackTables(key);
    TORCH_CHECK(tables.valid,
                "turboquant pack tables not registered; call turboquant_pack_register_tables first");

    // Keep original shape to avoid extra reshape launches in hot path.
    at::Tensor key_work = key;
    at::Tensor value_work = value;
    if (!key_work.is_contiguous()) {
        key_work = key_work.contiguous();
    }
    if (!value_work.is_contiguous()) {
        value_work = value_work.contiguous();
    }
    TORCH_CHECK(key_work.numel() % head_size == 0, "key numel must align with head_size");
    TORCH_CHECK(value_work.numel() % head_size == 0, "value numel must align with head_size");
    const int64_t n_vec = key_work.numel() / head_size;
    TORCH_CHECK(value_work.numel() / head_size == n_vec, "key/value row count mismatch");

    std::vector<int64_t> shape_k(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> shape_v(value.sizes().begin(), value.sizes().end());
    shape_k.back() = slot_w_k;
    shape_v.back() = slot_w_v;
    at::Tensor packed_k = at::empty(shape_k, key.options().dtype(at::kByte));
    at::Tensor packed_v = at::empty(shape_v, key.options().dtype(at::kByte));

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
    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheFused,
        key_work,
        value_work,
        tables.codebook,
        tables.rotation_t,
        pack_mode,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        packed_k,
        packed_v);

    return std::make_tuple(packed_k, packed_v);
}

// TurboQuant pack v2: registered tables + 1C2V KFC path only.
std::tuple<at::Tensor, at::Tensor> turboquant_pack_kv_for_cache_v2(
    const at::Tensor &key,
    const at::Tensor &value,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(key.dim() >= 1 && value.dim() >= 1, "key/value must have at least 1 dim");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack v2 accepts fp16/bf16 key/value");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack v2 supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= head_size + 2, "slot_w_v must be >= head_size+2 for 8-bit");

    auto &tables = GetTurboquantPackTables(key);
    TORCH_CHECK(tables.valid,
                "turboquant pack tables not registered; call turboquant_pack_register_tables first");

    at::Tensor key_work = key;
    at::Tensor value_work = value;
    if (!key_work.is_contiguous()) {
        key_work = key_work.contiguous();
    }
    if (!value_work.is_contiguous()) {
        value_work = value_work.contiguous();
    }
    TORCH_CHECK(key_work.numel() % head_size == 0, "key numel must align with head_size");
    TORCH_CHECK(value_work.numel() % head_size == 0, "value numel must align with head_size");
    const int64_t n_vec = key_work.numel() / head_size;
    TORCH_CHECK(value_work.numel() / head_size == n_vec, "key/value row count mismatch");

    std::vector<int64_t> shape_k(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> shape_v(value.sizes().begin(), value.sizes().end());
    shape_k.back() = slot_w_k;
    shape_v.back() = slot_w_v;
    at::Tensor packed_k = at::empty(shape_k, key.options().dtype(at::kByte));
    at::Tensor packed_v = at::empty(shape_v, key.options().dtype(at::kByte));

    const c10_npu::OptionalNPUGuard npuGuard(key_work.device());

    uint32_t vec_per_core = 64;
    if (n_vec < 64) {
        vec_per_core = static_cast<uint32_t>(((n_vec + 15) / 16) * 16);
        if (vec_per_core == 0) {
            vec_per_core = 16;
        }
    }
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheV2,
        key_work,
        value_work,
        tables.codebook,
        tables.rotation_t,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        packed_k,
        packed_v);

    return std::make_tuple(packed_k, packed_v);
}

// TurboQuant pack v3: registered tables + AIV-only vector rotate path.
std::tuple<at::Tensor, at::Tensor> turboquant_pack_kv_for_cache_v3(
    const at::Tensor &key,
    const at::Tensor &value,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(key.dim() >= 1 && value.dim() >= 1, "key/value must have at least 1 dim");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack v3 accepts fp16/bf16 key/value");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack v3 supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= head_size + 2, "slot_w_v must be >= head_size+2 for 8-bit");

    auto &tables = GetTurboquantPackTables(key);
    TORCH_CHECK(tables.valid,
                "turboquant pack tables not registered; call turboquant_pack_register_tables first");

    at::Tensor key_work = key;
    at::Tensor value_work = value;
    if (!key_work.is_contiguous()) {
        key_work = key_work.contiguous();
    }
    if (!value_work.is_contiguous()) {
        value_work = value_work.contiguous();
    }
    TORCH_CHECK(key_work.numel() % head_size == 0, "key numel must align with head_size");
    TORCH_CHECK(value_work.numel() % head_size == 0, "value numel must align with head_size");
    const int64_t n_vec = key_work.numel() / head_size;
    TORCH_CHECK(value_work.numel() / head_size == n_vec, "key/value row count mismatch");

    std::vector<int64_t> shape_k(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> shape_v(value.sizes().begin(), value.sizes().end());
    shape_k.back() = slot_w_k;
    shape_v.back() = slot_w_v;
    at::Tensor packed_k = at::empty(shape_k, key.options().dtype(at::kByte));
    at::Tensor packed_v = at::empty(shape_v, key.options().dtype(at::kByte));

    const c10_npu::OptionalNPUGuard npuGuard(key_work.device());

    // v3 uses an AIV-only rotate path. Queue depth is one in the kernel, so
    // a 64-row micro-batch fits the bf16 UB budget without falling back to
    // row-by-row processing.
    constexpr uint32_t kPackV3RowsPerBatch = 64;
    uint32_t vec_per_core = kPackV3RowsPerBatch;
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheV3,
        key_work,
        value_work,
        tables.codebook,
        tables.rotation_t,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        packed_k,
        packed_v);

    return std::make_tuple(packed_k, packed_v);
}

namespace {

void turboquant_pack_kv_for_cache_v2_to_cache_impl(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.device() == value.device() && key.device() == slot_mapping.device() &&
                    key.device() == key_cache.device() && key.device() == value_cache.device(),
                "key/value/slot_mapping/caches must be on the same NPU device");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack v2-to-cache accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "key_cache/value_cache must be uint8 views");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "key_cache/value_cache must be [num_blocks, block_size, num_heads, slot_w]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) && key_cache.size(1) == value_cache.size(1) &&
                    key_cache.size(2) == value_cache.size(2),
                "key_cache/value_cache leading dims must match");
    TORCH_CHECK(key_cache.size(3) == slot_w_k && value_cache.size(3) == slot_w_v,
                "cache last dim must match slot_w_k/slot_w_v");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack v2-to-cache supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= head_size + 2, "slot_w_v must be >= head_size+2 for 8-bit");
    TORCH_CHECK(key.numel() % head_size == 0 && value.numel() % head_size == 0,
                "key/value numel must align with head_size");
    const int64_t n_vec = key.numel() / head_size;
    TORCH_CHECK(value.numel() / head_size == n_vec, "key/value row count mismatch");
    const int64_t num_heads = key_cache.size(2);
    TORCH_CHECK(num_heads > 0 && key.size(-2) == num_heads && value.size(-2) == num_heads,
                "key/value num_heads must match cache num_heads");
    const int64_t token_count = (n_vec + num_heads - 1) / num_heads;
    TORCH_CHECK(slot_mapping.numel() >= token_count, "slot_mapping length is smaller than token count");

    auto &tables = GetTurboquantPackTables(key);
    TORCH_CHECK(tables.valid,
                "turboquant pack tables not registered; call turboquant_pack_register_tables first");

    at::Tensor key_work = key;
    at::Tensor value_work = value;
    at::Tensor slot_work = slot_mapping;
    if (!key_work.is_contiguous()) {
        key_work = key_work.contiguous();
    }
    if (!value_work.is_contiguous()) {
        value_work = value_work.contiguous();
    }
    if (!slot_work.is_contiguous()) {
        slot_work = slot_work.contiguous();
    }
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
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    const int64_t cache_slots = key_cache.size(0) * key_cache.size(1);

    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheV2ToCache,
        key_work,
        value_work,
        tables.codebook,
        tables.rotation_t,
        slot_work,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        num_heads,
        cache_slots,
        key_cache,
        value_cache);
}

void turboquant_pack_kv_for_cache_v3_to_cache_impl(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.device() == value.device() && key.device() == slot_mapping.device() &&
                    key.device() == key_cache.device() && key.device() == value_cache.device(),
                "key/value/slot_mapping/caches must be on the same NPU device");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack v3-to-cache accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "key_cache/value_cache must be uint8 views");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "key_cache/value_cache must be [num_blocks, block_size, num_heads, slot_w]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) && key_cache.size(1) == value_cache.size(1) &&
                    key_cache.size(2) == value_cache.size(2),
                "key_cache/value_cache leading dims must match");
    TORCH_CHECK(key_cache.size(3) == slot_w_k && value_cache.size(3) == slot_w_v,
                "cache last dim must match slot_w_k/slot_w_v");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack v3-to-cache supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= head_size + 2, "slot_w_v must be >= head_size+2 for 8-bit");
    TORCH_CHECK(key.numel() % head_size == 0 && value.numel() % head_size == 0,
                "key/value numel must align with head_size");
    const int64_t n_vec = key.numel() / head_size;
    TORCH_CHECK(value.numel() / head_size == n_vec, "key/value row count mismatch");
    const int64_t num_heads = key_cache.size(2);
    TORCH_CHECK(num_heads > 0 && key.size(-2) == num_heads && value.size(-2) == num_heads,
                "key/value num_heads must match cache num_heads");
    const int64_t token_count = (n_vec + num_heads - 1) / num_heads;
    TORCH_CHECK(slot_mapping.numel() >= token_count, "slot_mapping length is smaller than token count");

    auto &tables = GetTurboquantPackTables(key);
    TORCH_CHECK(tables.valid,
                "turboquant pack tables not registered; call turboquant_pack_register_tables first");

    at::Tensor key_work = key;
    at::Tensor value_work = value;
    at::Tensor slot_work = slot_mapping;
    if (!key_work.is_contiguous()) {
        key_work = key_work.contiguous();
    }
    if (!value_work.is_contiguous()) {
        value_work = value_work.contiguous();
    }
    if (!slot_work.is_contiguous()) {
        slot_work = slot_work.contiguous();
    }
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
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    const int64_t cache_slots = key_cache.size(0) * key_cache.size(1);

    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheV3ToCache,
        key_work,
        value_work,
        tables.codebook,
        tables.rotation_t,
        slot_work,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        num_heads,
        cache_slots,
        key_cache,
        value_cache);
}

}  // namespace

void turboquant_pack_kv_for_cache_v2_to_cache(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    turboquant_pack_kv_for_cache_v2_to_cache_impl(
        key, value, slot_mapping, key_cache, value_cache, slot_w_k, slot_w_v);
}

void turboquant_pack_kv_for_cache_v3_to_cache(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    turboquant_pack_kv_for_cache_v3_to_cache_impl(
        key, value, slot_mapping, key_cache, value_cache, slot_w_k, slot_w_v);
}

void bit_residual_pack_k8v4(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    const at::Tensor &query_start_loc,
    const at::Tensor &rotation_t,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t num_reqs,
    int64_t block_size) {
    constexpr int64_t kHeadSize = 128;
    constexpr int64_t kBlockRows = 16;
    constexpr int64_t kKeyBlockStride = 2112;
    constexpr int64_t kValBlockStride = 1088;
    constexpr int64_t kKeyGroupRows = 2;
    constexpr int64_t kValueGroupRows = 4;
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(query_start_loc.is_privateuseone(), "query_start_loc must be on NPU");
    TORCH_CHECK(rotation_t.is_privateuseone(), "rotation_t must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.device() == value.device() && key.device() == slot_mapping.device() &&
                    key.device() == query_start_loc.device() &&
                    key.device() == rotation_t.device() &&
                    key.device() == key_cache.device() && key.device() == value_cache.device(),
                "all tensors must be on the same NPU device");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "BitResidual pack accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc must be int32");
    TORCH_CHECK(rotation_t.scalar_type() == key.scalar_type() &&
                    rotation_t.dim() == 2 && rotation_t.size(0) == kHeadSize &&
                    rotation_t.size(1) == kHeadSize,
                "rotation_t must be [128,128] and match key/value dtype");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "key_cache/value_cache must be uint8");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "BitResidual k8v4 caches must be [num_blocks, num_heads, packed_bytes]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
                    key_cache.size(1) == value_cache.size(1),
                "key/value cache num_blocks and num_heads must match");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(num_reqs > 0, "num_reqs must be > 0");
    TORCH_CHECK(block_size % kBlockRows == 0,
                "BitResidual k8v4 cache layout requires block_size to be a multiple of 16");
    TORCH_CHECK(key_cache.size(2) == (block_size / kBlockRows) * kKeyBlockStride,
                "key_cache last dim must equal (block_size / 16) * 2112");
    TORCH_CHECK(value_cache.size(2) == (block_size / kBlockRows) * kValBlockStride,
                "value_cache last dim must equal (block_size / 16) * 1088");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
                "BitResidual caches must be contiguous");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == kHeadSize, "BitResidual pack supports head_size=128 only");
    TORCH_CHECK(key.numel() % head_size == 0 && value.numel() % head_size == 0,
                "key/value numel must align with head_size");
    const int64_t n_vec = key.numel() / head_size;
    TORCH_CHECK(value.numel() / head_size == n_vec, "key/value row count mismatch");
    const int64_t num_heads = key_cache.size(1);
    TORCH_CHECK(num_heads > 0 && key.dim() == 3 && value.dim() == 3,
                "key/value must be [num_tokens, num_heads, 128]");
    TORCH_CHECK(key.size(1) == num_heads && value.size(1) == num_heads,
                "key/value num_heads must match cache num_heads");
    const int64_t token_count = key.size(0);
    TORCH_CHECK(value.size(0) == token_count, "key/value token count mismatch");
    TORCH_CHECK(n_vec == token_count * num_heads,
                "key/value row count must equal num_tokens * num_heads");
    TORCH_CHECK(slot_mapping.numel() >= token_count, "slot_mapping length is smaller than token count");
    TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be a 1-D tensor");
    TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1,
                "query_start_loc length must be at least num_reqs + 1");

    const bool key_strided_copy_supported =
        key.stride(2) == 1 && key.stride(0) > 0 && key.stride(1) > 0;
    const bool value_strided_copy_supported =
        value.stride(2) == 1 && value.stride(0) > 0 && value.stride(1) > 0;
    at::Tensor key_work = key;
    at::Tensor value_work = value;
    at::Tensor slot_work = slot_mapping;
    at::Tensor query_start_work = query_start_loc;
    at::Tensor rotation_work = rotation_t;
    if (!key_strided_copy_supported) {
        if (!key_work.is_contiguous()) {
            key_work = key_work.contiguous();
        }
        if (key_work.storage_offset() != 0) {
            key_work = key_work.clone(at::MemoryFormat::Contiguous);
        }
    }
    if (!value_strided_copy_supported) {
        if (!value_work.is_contiguous()) {
            value_work = value_work.contiguous();
        }
        if (value_work.storage_offset() != 0) {
            value_work = value_work.clone(at::MemoryFormat::Contiguous);
        }
    }
    if (!slot_work.is_contiguous()) {
        slot_work = slot_work.contiguous();
    }
    if (!query_start_work.is_contiguous()) {
        query_start_work = query_start_work.contiguous();
    }
    if (!rotation_work.is_contiguous()) {
        rotation_work = rotation_work.contiguous();
    }

    uint32_t vec_per_core = 128;
    if (n_vec < 128) {
        // Decode/chunked-prefill paths usually pack a handful of KV rows per
        // step; vecPerCore=2 matches the ori TurboQuant tuning (~41 us vs ~44).
        vec_per_core = static_cast<uint32_t>(n_vec <= 16 ? 2 : ((n_vec + 15) / 16) * 16);
        if (vec_per_core == 0) {
            vec_per_core = 16;
        }
    }
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    const int64_t num_blocks = key_cache.size(0);
    const int64_t key_stride_token = key_work.stride(0);
    const int64_t key_stride_head = key_work.stride(1);
    const int64_t value_stride_token = value_work.stride(0);
    const int64_t value_stride_head = value_work.stride(1);
    // ConvertType passes tensor storage_offset to ACL, and the GM_ADDR seen by
    // the AscendC kernel is already at the logical tensor view. Keep the
    // kernel-side offset at zero to avoid applying storage_offset twice.
    const int64_t key_storage_offset = 0;
    const int64_t value_storage_offset = 0;
    constexpr int64_t kMaxKernelStride = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(key_stride_token > 0 && key_stride_head > 0 &&
                    value_stride_token > 0 && value_stride_head > 0 &&
                    key_stride_token <= kMaxKernelStride &&
                    key_stride_head <= kMaxKernelStride &&
                    value_stride_token <= kMaxKernelStride &&
                    value_stride_head <= kMaxKernelStride,
                "key/value strides must be positive and fit uint32_t");
    const c10_npu::OptionalNPUGuard npuGuard(key_work.device());
    EXEC_NPU_CMD(
        aclnnBitResidualPackK8v4,
        key_work,
        value_work,
        rotation_work,
        slot_work,
        query_start_work,
        n_vec,
        vec_per_core_i64,
        num_heads,
        block_size,
        num_blocks,
        num_reqs,
        key_stride_token,
        key_stride_head,
        value_stride_token,
        value_stride_head,
        key_storage_offset,
        value_storage_offset,
        key_cache,
        value_cache);
}

at::Tensor bit_residual_attention_paged_k8v4(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    at::IntArrayRef actual_seq_len_q,
    at::IntArrayRef actual_seq_len_kv,
    const at::Tensor& rotation_key,
    const at::Tensor& rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t max_actual_seq_len,
    double scale_value,
    c10::optional<at::Tensor> out_opt)
{
    constexpr int64_t kHeadSize = 128;
    constexpr int64_t kBlockRows = 16;
    constexpr int64_t kKeyGroupRows = 2;
    constexpr int64_t kValueGroupRows = 4;
    constexpr int64_t kKeyBlockStride = 2112;
    constexpr int64_t kValBlockStride = 1088;
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(rotation_key.is_privateuseone(), "rotation_key must be on NPU");
    TORCH_CHECK(rotation_value.is_privateuseone(), "rotation_value must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "BitResidual attention accepts fp16/bf16 query");
    TORCH_CHECK(head_size == kHeadSize, "BitResidual attention only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(block_size % kBlockRows == 0,
                "block_size must be a multiple of 16 (sub-block rows)");
    TORCH_CHECK(max_actual_seq_len > 0, "max_actual_seq_len must be > 0");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(rotation_key.scalar_type() == query.scalar_type() &&
                rotation_key.dim() == 2 && rotation_key.size(0) == kHeadSize &&
                rotation_key.size(1) == kHeadSize,
                "rotation_key must be [128,128] and match query dtype");
    TORCH_CHECK(rotation_value.scalar_type() == query.scalar_type() &&
                rotation_value.dim() == 2 && rotation_value.size(0) == kHeadSize &&
                rotation_value.size(1) == kHeadSize,
                "rotation_value must be [128,128] and match query dtype");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "BitResidual caches must be [num_blocks, num_kv_heads, packed_bytes]");
    TORCH_CHECK(key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads,
                "cache num_kv_heads mismatch");
    TORCH_CHECK(key_cache.size(2) == (block_size / kBlockRows) * kKeyBlockStride,
                "key_cache last dim must equal (block_size / 16) * 2112");
    TORCH_CHECK(value_cache.size(2) == (block_size / kBlockRows) * kValBlockStride,
                "value_cache last dim must equal (block_size / 16) * 1088");
    TORCH_CHECK(actual_seq_len_q.size() > 0, "actual_seq_len_q must not be empty");
    TORCH_CHECK(actual_seq_len_q.size() == actual_seq_len_kv.size(),
                "actual_seq_len_q and actual_seq_len_kv must have the same length");

    const at::Tensor query_c = MaybeContiguous(query);
    const at::Tensor key_cache_c = MaybeContiguous(key_cache);
    const at::Tensor value_cache_c = MaybeContiguous(value_cache);
    const at::Tensor block_table_c = MaybeContiguous(block_table);
    const at::Tensor rotation_key_c = MaybeContiguous(rotation_key);
    const at::Tensor rotation_value_c = MaybeContiguous(rotation_value);

    at::Tensor out;
    if (out_opt.has_value() && out_opt->defined()) {
        TORCH_CHECK(out_opt->is_privateuseone(), "out must be on NPU");
        TORCH_CHECK(out_opt->sizes().equals(query_c.sizes()), "out shape must match query");
        TORCH_CHECK(out_opt->scalar_type() == query_c.scalar_type(), "out dtype must match query");
        TORCH_CHECK(out_opt->device() == query_c.device(), "out device must match query");
        TORCH_CHECK(out_opt->is_contiguous(), "out must be contiguous for in-place write");
        out = *out_opt;
    } else {
        out = at::empty(query_c.sizes(), query_c.options());
    }

    const c10_npu::OptionalNPUGuard npuGuard(query_c.device());
    EXEC_NPU_CMD(
        aclnnBitResidualAttentionPagedK8v4,
        query_c,
        key_cache_c,
        value_cache_c,
        block_table_c,
        actual_seq_len_q,
        actual_seq_len_kv,
        rotation_key_c,
        rotation_value_c,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_actual_seq_len,
        scale_value,
        out);

    return out;
}

at::Tensor bit_residual_fia_paged_k8v4(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    at::IntArrayRef actual_seq_len_q,
    at::IntArrayRef actual_seq_len_kv,
    const c10::optional<at::Tensor>& atten_mask,
    const at::Tensor& rotation_key,
    const at::Tensor& rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t sparse_mode)
{
    constexpr int64_t kHeadSize = 128;
    constexpr int64_t kBlockRows = 16;
    constexpr int64_t kKeyBlockStride = 2112;
    constexpr int64_t kValBlockStride = 1088;
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(rotation_key.is_privateuseone(), "rotation_key must be on NPU");
    TORCH_CHECK(rotation_value.is_privateuseone(), "rotation_value must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "BitResidual FIA accepts fp16 query in P0");
    TORCH_CHECK(head_size == kHeadSize, "BitResidual FIA only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(block_size % kBlockRows == 0,
                "block_size must be a multiple of 16 (sub-block rows)");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(rotation_key.scalar_type() == at::kHalf &&
                rotation_key.dim() == 2 && rotation_key.size(0) == kHeadSize &&
                rotation_key.size(1) == kHeadSize,
                "rotation_key must be [128,128] fp16");
    TORCH_CHECK(rotation_value.scalar_type() == at::kHalf &&
                rotation_value.dim() == 2 && rotation_value.size(0) == kHeadSize &&
                rotation_value.size(1) == kHeadSize,
                "rotation_value must be [128,128] fp16");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "BitResidual caches must be [num_blocks, num_kv_heads, packed_bytes]");
    TORCH_CHECK(key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads,
                "cache num_kv_heads mismatch");
    TORCH_CHECK(key_cache.size(2) == (block_size / kBlockRows) * kKeyBlockStride,
                "key_cache last dim must equal (block_size / 16) * 2112");
    TORCH_CHECK(value_cache.size(2) == (block_size / kBlockRows) * kValBlockStride,
                "value_cache last dim must equal (block_size / 16) * 1088");
    TORCH_CHECK(actual_seq_len_q.size() > 0, "actual_seq_len_q must not be empty");
    TORCH_CHECK(actual_seq_len_q.size() == actual_seq_len_kv.size(),
                "actual_seq_len_q and actual_seq_len_kv must have the same length");

    const at::Tensor query_c = query.contiguous();
    const at::Tensor key_cache_c = key_cache.contiguous();
    const at::Tensor value_cache_c = value_cache.contiguous();
    const at::Tensor block_table_c = block_table.contiguous();
    const at::Tensor rotation_key_c = rotation_key.contiguous();
    const at::Tensor rotation_value_c = rotation_value.contiguous();
    at::Tensor mask = atten_mask.has_value() && atten_mask->defined()
                          ? atten_mask->contiguous()
                          : at::empty({0}, query.options().dtype(at::kChar));

    at::Tensor out = at::empty(query_c.sizes(), query_c.options());
    const c10_npu::OptionalNPUGuard npuGuard(query_c.device());
    EXEC_NPU_CMD(
        aclnnBitResidualFiaPagedK8v4,
        query_c,
        key_cache_c,
        value_cache_c,
        block_table_c,
        actual_seq_len_q,
        actual_seq_len_kv,
        mask,
        rotation_key_c,
        rotation_value_c,
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

void turboquant_pack_kv_for_cache_to_cache(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    const bool no_kfc = std::getenv("VLLM_ASCEND_TURBOQUANT_NO_KFC") != nullptr;

    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.device() == value.device() && key.device() == slot_mapping.device() &&
                    key.device() == key_cache.device() && key.device() == value_cache.device(),
                "key/value/slot_mapping/caches must be on the same NPU device");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "pack-to-cache accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "key_cache/value_cache must be uint8 views");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "key_cache/value_cache must be [num_blocks, block_size, num_heads, slot_w]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) && key_cache.size(1) == value_cache.size(1) &&
                    key_cache.size(2) == value_cache.size(2),
                "key_cache/value_cache leading dims must match");
    TORCH_CHECK(key_cache.size(3) == slot_w_k && value_cache.size(3) == slot_w_v,
                "cache last dim must match slot_w_k/slot_w_v");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == 128, "pack-to-cache supports head_size=128 only");
    TORCH_CHECK(slot_w_k >= head_size + 2, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= head_size + 2, "slot_w_v must be >= head_size+2 for 8-bit");
    TORCH_CHECK(key.numel() % head_size == 0 && value.numel() % head_size == 0,
                "key/value numel must align with head_size");
    const int64_t n_vec = key.numel() / head_size;
    TORCH_CHECK(value.numel() / head_size == n_vec, "key/value row count mismatch");
    const int64_t num_heads = key_cache.size(2);
    TORCH_CHECK(num_heads > 0 && key.size(-2) == num_heads && value.size(-2) == num_heads,
                "key/value num_heads must match cache num_heads");
    const int64_t token_count = (n_vec + num_heads - 1) / num_heads;
    TORCH_CHECK(slot_mapping.numel() >= token_count, "slot_mapping length is smaller than token count");

    auto &tables = GetTurboquantPackTables(key);
    TORCH_CHECK(tables.valid,
                "turboquant pack tables not registered; call turboquant_pack_register_tables first");

    at::Tensor key_work = key;
    at::Tensor value_work = value;
    at::Tensor slot_work = slot_mapping;
    if (!key_work.is_contiguous()) {
        key_work = key_work.contiguous();
    }
    if (!value_work.is_contiguous()) {
        value_work = value_work.contiguous();
    }
    if (!slot_work.is_contiguous()) {
        slot_work = slot_work.contiguous();
    }
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
        aclnnTurboquantPackKvForCacheToCache,
        key_work,
        value_work,
        tables.codebook,
        tables.rotation_t,
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

void turboquant_pack_kv_for_cache_4bit(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    const at::Tensor &query_start_loc,
    const at::Tensor &codebook,
    const at::Tensor &rotation_t,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t num_reqs,
    int64_t block_size) {
    constexpr int64_t kHeadSize = 128;
    constexpr int64_t kRowBytes = kHeadSize / 2 + 2;
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(query_start_loc.is_privateuseone(), "query_start_loc must be on NPU");
    TORCH_CHECK(codebook.is_privateuseone() && rotation_t.is_privateuseone(),
                "codebook/rotation_t must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.device() == value.device() && key.device() == slot_mapping.device() &&
                    key.device() == query_start_loc.device() &&
                    key.device() == codebook.device() && key.device() == rotation_t.device() &&
                    key.device() == key_cache.device() && key.device() == value_cache.device(),
                "all tensors must be on the same NPU device");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "4-bit pack-to-cache accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc must be int32");
    TORCH_CHECK(codebook.scalar_type() == key.scalar_type() && codebook.numel() == 16,
                "4-bit codebook must have 16 entries and match key/value dtype");
    TORCH_CHECK(rotation_t.scalar_type() == key.scalar_type() &&
                    rotation_t.dim() == 2 && rotation_t.size(0) == kHeadSize &&
                    rotation_t.size(1) == kHeadSize,
                "rotation_t must be [128,128] and match key/value dtype");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "4-bit slab caches must be uint8");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "4-bit slab caches must be [num_blocks, num_heads, block_size * 66]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
                    key_cache.size(1) == value_cache.size(1) &&
                    key_cache.size(2) == value_cache.size(2),
                "key/value slab cache shapes must match");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(num_reqs > 0, "num_reqs must be > 0");
    TORCH_CHECK(key_cache.size(2) == block_size * kRowBytes,
                "slab cache last dim must equal block_size * 66");
    TORCH_CHECK(block_size % 4 == 0,
                "4-bit group slab cache requires block_size to be a multiple of 4");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
                "4-bit slab caches must be contiguous");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == kHeadSize, "4-bit pack-to-cache supports head_size=128 only");
    TORCH_CHECK(key.numel() % head_size == 0 && value.numel() % head_size == 0,
                "key/value numel must align with head_size");
    const int64_t n_vec = key.numel() / head_size;
    TORCH_CHECK(value.numel() / head_size == n_vec, "key/value row count mismatch");
    const int64_t num_heads = key_cache.size(1);
    TORCH_CHECK(num_heads > 0, "slab cache num_heads must be > 0");
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "4-bit pack-to-cache expects token-major key/value layout "
                "[num_tokens, num_heads, 128]");
    TORCH_CHECK(key.size(1) == num_heads && value.size(1) == num_heads,
                "token-major key/value dim1 must match slab cache num_heads");
    const int64_t token_count = key.size(0);
    TORCH_CHECK(value.size(0) == token_count, "key/value token count mismatch");
    TORCH_CHECK(n_vec == token_count * num_heads,
                "token-major key/value row count must equal num_tokens * num_heads");
    TORCH_CHECK(slot_mapping.numel() >= token_count, "slot_mapping length is smaller than token count");
    TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be a 1-D tensor");
    TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1,
                "query_start_loc length must be at least num_reqs + 1");

    const bool key_strided_copy_supported =
        key.stride(2) == 1 && key.stride(0) > 0 && key.stride(1) > 0;
    const bool value_strided_copy_supported =
        value.stride(2) == 1 && value.stride(0) > 0 && value.stride(1) > 0;
    at::Tensor key_work = key;
    at::Tensor value_work = value;
    at::Tensor slot_work = slot_mapping;
    at::Tensor query_start_work = query_start_loc;
    at::Tensor codebook_work = codebook;
    at::Tensor rotation_work = rotation_t;
    if (!key_strided_copy_supported) {
        if (!key_work.is_contiguous()) {
            key_work = key_work.contiguous();
        }
        if (key_work.storage_offset() != 0) {
            key_work = key_work.clone(at::MemoryFormat::Contiguous);
        }
    }
    if (!value_strided_copy_supported) {
        if (!value_work.is_contiguous()) {
            value_work = value_work.contiguous();
        }
        if (value_work.storage_offset() != 0) {
            value_work = value_work.clone(at::MemoryFormat::Contiguous);
        }
    }
    if (!slot_work.is_contiguous()) {
        slot_work = slot_work.contiguous();
    }
    if (!query_start_work.is_contiguous()) {
        query_start_work = query_start_work.contiguous();
    }
    if (!codebook_work.is_contiguous()) {
        codebook_work = codebook_work.contiguous();
    }
    if (!rotation_work.is_contiguous()) {
        rotation_work = rotation_work.contiguous();
    }

    uint32_t vec_per_core = 128;
    if (n_vec < 128) {
        vec_per_core = static_cast<uint32_t>(((n_vec + 15) / 16) * 16);
        if (vec_per_core == 0) {
            vec_per_core = 16;
        }
    }
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    const int64_t num_blocks = key_cache.size(0);
    const int64_t key_stride_token = key_work.stride(0);
    const int64_t key_stride_head = key_work.stride(1);
    const int64_t value_stride_token = value_work.stride(0);
    const int64_t value_stride_head = value_work.stride(1);
    // ConvertType passes tensor storage_offset to ACL, and the GM_ADDR seen by
    // the AscendC kernel is already at the logical tensor view. Keep the
    // kernel-side offset at zero to avoid applying storage_offset twice.
    const int64_t key_storage_offset = 0;
    const int64_t value_storage_offset = 0;
    constexpr int64_t kMaxKernelStride = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(key_stride_token > 0 && key_stride_head > 0 &&
                    value_stride_token > 0 && value_stride_head > 0 &&
                    key_stride_token <= kMaxKernelStride &&
                    key_stride_head <= kMaxKernelStride &&
                    value_stride_token <= kMaxKernelStride &&
                    value_stride_head <= kMaxKernelStride,
                "key/value strides must be positive and fit uint32_t");
    const c10_npu::OptionalNPUGuard npuGuard(key_work.device());
    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCache4bit,
        key_work,
        value_work,
        codebook_work,
        rotation_work,
        slot_work,
        query_start_work,
        n_vec,
        vec_per_core_i64,
        num_heads,
        block_size,
        num_blocks,
        num_reqs,
        key_stride_token,
        key_stride_head,
        value_stride_token,
        value_stride_head,
        key_storage_offset,
        value_storage_offset,
        key_cache,
        value_cache);
}

// TurboQuant 4-bit decode (packed uint8 -> fp16/bf16).
// Two-stage implementation:
//  1) AscendC kernel: unpack uint4 + codebook lookup + scale by norm -> y_hat [N, D].
//  2) Use optimized NPU matmul: out = y_hat @ rotation.
//
// packed: [N, P] uint8, where P = D/2 + 2 (uint4 indices packed + 16-bit norm)
// codebook: [16] fp16/bf16, rotation: [D, D] fp16/bf16; both match output dtype.
at::Tensor turboquant_decode_packed_blocks(
    const at::Tensor &packed,
    const at::Tensor &codebook,
    const at::Tensor &rotation,
    int64_t head_size,
    int64_t out_dtype_code) {
    TORCH_CHECK(packed.is_privateuseone(), "packed must be on NPU");
    TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must be uint8");
    TORCH_CHECK(codebook.is_privateuseone(), "codebook must be on NPU");
    TORCH_CHECK(rotation.is_privateuseone(), "rotation must be on NPU");
    TORCH_CHECK(codebook.scalar_type() == at::kHalf || codebook.scalar_type() == at::kBFloat16,
                "codebook must be fp16/bf16");
    TORCH_CHECK(rotation.scalar_type() == codebook.scalar_type(),
                "rotation dtype must match codebook dtype");
    TORCH_CHECK(codebook.numel() == 16, "codebook must have 16 entries");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == head_size && rotation.size(1) == head_size,
                "rotation must be [D, D]");
    TORCH_CHECK(head_size % 2 == 0, "head_size must be even for 4-bit packing");

    const int64_t packed_bytes = head_size / 2 + 2;
    TORCH_CHECK(packed.dim() == 2 && packed.size(1) == packed_bytes,
                "packed must be [N, P] with P=head_size/2+2");

    at::ScalarType out_dtype = (out_dtype_code == 1) ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(codebook.scalar_type() == out_dtype,
                "4-bit decode codebook/rotation dtype must match requested output dtype");
    // IMPORTANT: The AscendC stage-1 kernel treats `packed` as contiguous row-major bytes.
    // If `packed` is a view with non-trivial strides, raw pointer indexing will read
    // the wrong bytes and corrupt decode (often showing up as garbled text output).
    const at::Tensor packed_c = packed.contiguous();
    const at::Tensor codebook_c = codebook.contiguous();
    const at::Tensor rotation_c = rotation.contiguous();

    at::Tensor y_hat = at::empty({packed_c.size(0), head_size}, packed_c.options().dtype(out_dtype));
    at::Tensor out = at::empty({packed_c.size(0), head_size}, packed_c.options().dtype(out_dtype));

    const c10_npu::OptionalNPUGuard npuGuard(packed_c.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Choose per-core work granularity. Keep it small to reduce tail latency.
    // This can be tuned after profiling.
    uint32_t vec_per_core = 4;
    if (packed.size(0) >= 8192) vec_per_core = 16;
    else if (packed.size(0) >= 2048) vec_per_core = 8;

    // Stage 1: unpack+lookup+scale.
    turboquant_unpack_lookup_scale_impl(
        stream,
        const_cast<void *>(packed_c.data_ptr()),
        const_cast<void *>(codebook_c.data_ptr()),
        y_hat.data_ptr(),
        static_cast<uint32_t>(packed_c.size(0)),
        static_cast<uint32_t>(head_size),
        static_cast<uint32_t>(packed_bytes),
        vec_per_core,
        static_cast<uint32_t>(out_dtype == at::kBFloat16 ? AscendType::BF16 : AscendType::FP16));
    // Stage 2: matmul on NPU.
    //
    // IMPORTANT: Keep Stage2 numerically aligned with the PyTorch reference path
    // (`at::matmul(y_hat, rotation)`), otherwise greedy decoding can diverge wildly
    // even when Stage1 is only slightly off.
    at::Tensor x_hat = at::matmul(y_hat, rotation_c);
    out.copy_(x_hat);
    return out;
}

// TurboQuant decode for compact KV cache blocks.
// packed: [U, BS, H, P] uint8, rotation_batched: [BS*H, D, D] fp16/bf16.
// Output: [U, BS, H, D] fp16/bf16.
at::Tensor turboquant_decode_packed_blocks_compact(
    const at::Tensor &packed,
    const at::Tensor &codebook,
    const at::Tensor &rotation_batched,
    int64_t head_size,
    int64_t out_dtype_code) {
    TORCH_CHECK(packed.is_privateuseone(), "packed must be on NPU");
    TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must be uint8");
    TORCH_CHECK(codebook.is_privateuseone(), "codebook must be on NPU");
    TORCH_CHECK(rotation_batched.is_privateuseone(), "rotation_batched must be on NPU");
    TORCH_CHECK(codebook.scalar_type() == at::kHalf || codebook.scalar_type() == at::kBFloat16,
                "codebook must be fp16/bf16");
    TORCH_CHECK(codebook.numel() == 16, "codebook must have 16 entries");
    TORCH_CHECK(packed.dim() == 4, "packed must be [U, BS, H, P]");
    TORCH_CHECK(head_size % 2 == 0, "head_size must be even for 4-bit packing");

    const int64_t U = packed.size(0);
    const int64_t BS = packed.size(1);
    const int64_t H = packed.size(2);
    const int64_t packed_bytes = head_size / 2 + 2;
    TORCH_CHECK(packed.size(3) == packed_bytes, "packed last dim must be head_size/2+2");

    const int64_t batch = BS * H;
    TORCH_CHECK(rotation_batched.dim() == 3, "rotation_batched must be [batch, D, D]");
    TORCH_CHECK(rotation_batched.size(0) == batch, "rotation_batched batch mismatch");
    TORCH_CHECK(rotation_batched.size(1) == head_size && rotation_batched.size(2) == head_size,
                "rotation_batched must be [batch, D, D]");
    TORCH_CHECK(rotation_batched.scalar_type() == codebook.scalar_type(),
                "rotation_batched dtype must match codebook dtype");

    at::ScalarType out_dtype = (out_dtype_code == 1) ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(codebook.scalar_type() == out_dtype,
                "4-bit compact decode codebook/rotation dtype must match requested output dtype");
    const at::Tensor codebook_c = codebook.contiguous();
    const at::Tensor rotation_batched_c = rotation_batched.contiguous();
    at::Tensor y_hat = at::empty({U * batch, head_size}, packed.options().dtype(out_dtype));
    at::Tensor out = at::empty({U, BS, H, head_size}, packed.options().dtype(out_dtype));

    const c10_npu::OptionalNPUGuard npuGuard(packed.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Stage 1: unpack+lookup+scale over flattened vectors.
    const uint32_t n_vec = static_cast<uint32_t>(U * batch);
    uint32_t vec_per_core = 8;
    if (n_vec >= 8192) vec_per_core = 16;
    else if (n_vec < 2048) vec_per_core = 4;
    at::Tensor packed_flat = packed.reshape({U * batch, packed_bytes}).contiguous();
    turboquant_unpack_lookup_scale_impl(
        stream,
        const_cast<void *>(packed_flat.data_ptr()),
        const_cast<void *>(codebook_c.data_ptr()),
        y_hat.data_ptr(),
        n_vec,
        static_cast<uint32_t>(head_size),
        static_cast<uint32_t>(packed_bytes),
        vec_per_core,
        static_cast<uint32_t>(out_dtype == at::kBFloat16 ? AscendType::BF16 : AscendType::FP16));

    // Stage 2: batched matmul.
    //
    // NOTE: `at::bmm` requires A/B to have the same leading batch dim.
    // Here we want per-(bs,h) rotation: for each batch index t in [0, batch),
    //   out[u, t, :] = y_hat[u, t, :] @ rotation_batched[t, :, :]
    //
    // Use `at::matmul` batch-broadcasting rules:
    //   y3: [batch, U, D]
    //   r3: [batch, D, D]
    // => x3: [batch, U, D]
    at::Tensor y3 = y_hat.view({U, batch, head_size}).transpose(0, 1).contiguous();
    at::Tensor x3 = at::matmul(y3, rotation_batched_c);
    at::Tensor x_hat = x3.transpose(0, 1).contiguous().view({U, BS, H, head_size});
    out.copy_(x_hat);
    return out;
}

namespace {

std::tuple<at::Tensor, at::Tensor> TurboquantCompactSelectBlocks(
    const at::Tensor &cache, const at::Tensor &block_tables) {
    at::Tensor bt = block_tables.to(torch::kInt32);
    at::Tensor valid = bt.ge(0);
    if (!valid.any().item<bool>()) {
        at::Tensor empty = cache.index({0}).slice(0, 0);
        return {empty, bt};
    }
    at::Tensor used = std::get<0>(at::_unique(bt.masked_select(valid), true, false));
    at::Tensor selected = cache.index_select(0, used);
    at::Tensor bt_compact = bt.clone();
    at::Tensor used_vals = bt.masked_select(valid);
    at::Tensor compact_idx = at::searchsorted(used, used_vals);
    bt_compact.masked_scatter_(valid, compact_idx);
    return {selected, bt_compact};
}

at::Tensor TurboquantMakeRotationBatched(const at::Tensor &rotation, int64_t block_size, int64_t num_kv_heads) {
    const int64_t batch = block_size * num_kv_heads;
    return rotation.unsqueeze(0).expand({batch, rotation.size(0), rotation.size(1)}).contiguous();
}

}  // namespace

// True fused TurboQuant 8-bit KV attention: dispatch to custom aclnn op.
at::Tensor turboquant_fused_infer_attention_score_8bit(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &block_table,
    const at::Tensor &atten_mask,
    const at::Tensor &actual_seq_len_q,
    const at::Tensor &actual_seq_len_kv,
    const at::Tensor &codebook,
    const at::Tensor &rotation,
    const at::Tensor &codebook_value,
    const at::Tensor &rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value) {
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "fp16 only in initial version");
    TORCH_CHECK(head_size == 128, "initial version only supports head_size=128");
    TORCH_CHECK(atten_mask.defined(), "atten_mask is required");

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

// TurboQuant Scheme B: fused packed decode + paged attention (DecodeOnly).
// actual_seq_len_q and actual_seq_len_kv are passed as aclIntArray inputs.
at::Tensor turboquant_attention_paged8bit(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &block_table,
    at::IntArrayRef actual_seq_len_q,
    at::IntArrayRef actual_seq_len_kv,
    const at::Tensor &codebook,
    const at::Tensor &rotation,
    const at::Tensor &codebook_value,
    const at::Tensor &rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t max_actual_seq_len,
    double scale_value) {
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

// TurboQuant 8-bit paged decode (设计文档 §2.6 方案 X / Phase 1).
//
// 算子吞掉 block_table 寻址 — host 侧只做 ceil + cumsum 算紧凑物理块号列表 gather_block_ids,
// 算子吃完整 packed KV cache + gather_block_ids,输出按 seq 顺序紧凑排布的 fp16 K/V workspace。
//
// S1 阶段语义: kernel 只验证寻址链路,把 packed 行的 idx 字节(uint8) zero-extend 成 fp16 写出。
// S2/S3 阶段会替换为 codebook 查表 + ×norm + Cube y_hat @ R(KFC).
//
// Inputs:
//   key_cache:        [num_blocks_total, BS, H, P=130] uint8
//   value_cache:      同 key_cache shape
//   gather_block_ids: [total_blocks] int32, host 侧 prefix-sum 生成
//   codebook:         [256] fp16  (S1 不读,占位)
//   rotation:         [128, 128] fp16  (S1 不读,占位)
// Outputs:
//   key_out / value_out: [total_blocks, BS, H, 128] fp16
std::tuple<at::Tensor, at::Tensor> turboquant_decode_paged_8bit(
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &gather_block_ids,
    const at::Tensor &codebook,
    const at::Tensor &rotation,
    int64_t head_size,
    int64_t block_size,
    int64_t out_dtype_code,
    int64_t mode) {
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(gather_block_ids.is_privateuseone(), "gather_block_ids must be on NPU");
    TORCH_CHECK(codebook.is_privateuseone(), "codebook must be on NPU");
    TORCH_CHECK(rotation.is_privateuseone(), "rotation must be on NPU");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(gather_block_ids.scalar_type() == at::kInt, "gather_block_ids must be int32");
    TORCH_CHECK(codebook.scalar_type() == at::kHalf, "codebook must be fp16");
    TORCH_CHECK(rotation.scalar_type() == at::kHalf, "rotation must be fp16");
    TORCH_CHECK(head_size == 128, "Phase 1 only supports head_size=128");
    TORCH_CHECK(out_dtype_code == 0, "Phase 1 only supports fp16 output (out_dtype=0)");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (KFC) or 1 (AIV-only), got ", mode);
    TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [num_blocks, BS, H, P]");
    TORCH_CHECK(value_cache.dim() == 4, "value_cache must be [num_blocks, BS, H, P]");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
                "key_cache and value_cache must have identical shape");
    const int64_t packed_bytes = head_size + 2;
    TORCH_CHECK(key_cache.size(3) == packed_bytes,
                "key_cache last dim must be head_size + 2 (=130 for 8-bit)");
    TORCH_CHECK(key_cache.size(1) == block_size, "key_cache.size(1) must == block_size");
    TORCH_CHECK(codebook.numel() == 256, "8-bit codebook must have 256 entries");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == head_size && rotation.size(1) == head_size,
                "rotation must be [128, 128]");
    TORCH_CHECK(gather_block_ids.dim() == 1, "gather_block_ids must be 1D");

    const int64_t total_blocks = gather_block_ids.size(0);
    const int64_t num_kv_heads = key_cache.size(2);

    const at::Tensor key_cache_c = key_cache.contiguous();
    const at::Tensor value_cache_c = value_cache.contiguous();
    const at::Tensor gather_ids_c = gather_block_ids.contiguous();
    const at::Tensor codebook_c = codebook.contiguous();
    const at::Tensor rotation_c = rotation.contiguous();

    at::Tensor key_out = at::empty({total_blocks, block_size, num_kv_heads, head_size},
                                   key_cache.options().dtype(at::kHalf));
    at::Tensor value_out = at::empty({total_blocks, block_size, num_kv_heads, head_size},
                                     key_cache.options().dtype(at::kHalf));

    const c10_npu::OptionalNPUGuard npuGuard(key_cache_c.device());

    // rows_per_core: 当前简化为常量 32(与 kernel 内 TQ_T_ROWS 对齐)。
    // tiling 函数会按平台 AIV 数算 blockDim,无需 host 提供更多信息。
    const int64_t rows_per_core = 32;

    EXEC_NPU_CMD(
        aclnnTurboquantDecodePaged8bit,
        key_cache_c,
        value_cache_c,
        gather_ids_c,
        codebook_c,
        rotation_c,
        head_size,
        block_size,
        num_kv_heads,
        total_blocks,
        rows_per_core,
        out_dtype_code,
        mode,
        key_out,
        value_out);

    return std::make_tuple(key_out, value_out);
}

// TurboQuant encode (fp16 rotated unit vectors -> packed uint8), 4- or 8-bit MSE path.
// y: [N, D] fp16 where y = (x/||x||) @ R^T in fp16; norms_fp16: [N,1] fp16 (||x||).
// bits 4: codebook [16] fp16, packed width D/2+2; bits 8: codebook [256] fp16, width D+2.
at::Tensor turboquant_encode_packed_blocks(
    const at::Tensor &y,
    const at::Tensor &codebook,
    const at::Tensor &norms_fp16,
    int64_t head_size,
    int64_t bits) {
    TORCH_CHECK(y.is_privateuseone(), "y must be on NPU");
    TORCH_CHECK(y.scalar_type() == at::kHalf, "y must be fp16");
    TORCH_CHECK(y.dim() == 2 && y.size(1) == head_size, "y must be [N, head_size]");
    TORCH_CHECK(codebook.is_privateuseone(), "codebook must be on NPU");
    TORCH_CHECK(codebook.scalar_type() == at::kHalf, "codebook must be fp16");
    TORCH_CHECK(bits == 4 || bits == 8, "TurboQuant encode supports bits 4 or 8 only, got ", bits);
    if (bits == 4) {
        TORCH_CHECK(codebook.numel() == 16, "4-bit encode: codebook must have 16 entries");
        TORCH_CHECK(head_size % 2 == 0, "head_size must be even for 4-bit packing");
    } else {
        TORCH_CHECK(codebook.numel() == 256, "8-bit encode: codebook must have 256 entries");
    }
    TORCH_CHECK(norms_fp16.is_privateuseone(), "norms_fp16 must be on NPU");
    TORCH_CHECK(norms_fp16.scalar_type() == at::kHalf, "norms_fp16 must be fp16");
    TORCH_CHECK(norms_fp16.dim() == 2 && norms_fp16.size(1) == 1,
                "norms_fp16 must be [N, 1]");
    TORCH_CHECK(norms_fp16.size(0) == y.size(0), "norms_fp16 batch must match y");

    const int64_t packed_bytes = bits == 8 ? head_size + 2 : head_size / 2 + 2;
    const at::Tensor y_c = y.contiguous();
    const at::Tensor codebook_c = codebook.contiguous();
    const at::Tensor norms_c = norms_fp16.contiguous();

    at::Tensor packed = at::empty({y_c.size(0), packed_bytes}, y_c.options().dtype(at::kByte));

    const c10_npu::OptionalNPUGuard npuGuard(y_c.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    uint32_t vec_per_core = 4;
    if (bits == 8) {
        vec_per_core = 2;
        if (y_c.size(0) >= 8192) {
            vec_per_core = 8;
        } else if (y_c.size(0) >= 2048) {
            vec_per_core = 4;
        }
    } else if (y_c.size(0) >= 8192) {
        vec_per_core = 16;
    } else if (y_c.size(0) >= 2048) {
        vec_per_core = 8;
    }

    if (bits == 4) {
        // aclrtStream is an opaque handle (typically void*); impl uses void* for C linkage.
        turboquant_pack_nearest_scale_impl(
            static_cast<void *>(stream),
            const_cast<void *>(y_c.data_ptr()),
            const_cast<void *>(codebook_c.data_ptr()),
            const_cast<void *>(norms_c.data_ptr()),
            packed.data_ptr(),
            static_cast<uint32_t>(y_c.size(0)),
            static_cast<uint32_t>(head_size),
            static_cast<uint32_t>(packed_bytes),
            vec_per_core);
    } else {
        turboquant_pack_nearest_scale_8bit_impl(
            static_cast<void *>(stream),
            const_cast<void *>(y_c.data_ptr()),
            const_cast<void *>(codebook_c.data_ptr()),
            const_cast<void *>(norms_c.data_ptr()),
            packed.data_ptr(),
            static_cast<uint32_t>(y_c.size(0)),
            static_cast<uint32_t>(head_size),
            static_cast<uint32_t>(packed_bytes),
            vec_per_core);
    }
    return packed;
}

namespace {

at::Tensor turboquant_pad_packed_row_to_slot(const at::Tensor &packed, int64_t slot_w) {
    const int64_t w = packed.size(-1);
    if (w == slot_w) {
        return packed;
    }
    TORCH_CHECK(w < slot_w, "packed width ", w, " exceeds cache slot width ", slot_w);
    return at::constant_pad_nd(packed, {0, slot_w - w});
}

}  // namespace

// Minimal standard custom-op Cube matmul probe: C = A @ B.
at::Tensor turboquant_rotate_matmul_probe(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t probe_mode) {
    TORCH_CHECK(a.is_privateuseone() && b.is_privateuseone(), "inputs must be on NPU");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, "fp16 only");
    TORCH_CHECK(a.dim() == 2 && a.size(1) == 128, "a must be [M, 128]");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == 128 && b.size(1) == 128, "b must be [128, 128]");
    TORCH_CHECK(probe_mode == 0 || probe_mode == 1 || probe_mode == 2,
                "probe_mode must be 0 (regist only), 1 (GM matmul), or 2 (KFC matmul)");

    const at::Tensor a_c = a.contiguous();
    const at::Tensor b_c = b.contiguous();
    const int64_t m = a_c.size(0);
    TORCH_CHECK(m >= 1 && m <= 128, "M must be in [1, 128]");
    TORCH_CHECK(probe_mode != 2 || m <= 32, "probe_mode=2 supports M <= 32");

    const uint32_t m_pad = static_cast<uint32_t>(((m + 15) / 16) * 16);
    at::Tensor a_mat = a_c;
    if (static_cast<uint32_t>(m) != m_pad) {
        a_mat = at::zeros({static_cast<int64_t>(m_pad), 128}, a.options());
        a_mat.slice(0, 0, m).copy_(a_c);
    }
    at::Tensor c_pad = at::zeros({static_cast<int64_t>(m_pad), 128}, a.options());

    EXEC_NPU_CMD(aclnnTurboquantRotateMatmulProbe, a_mat, b_c, probe_mode, m, c_pad);

    return c_pad.slice(0, 0, m).contiguous();
}

void swap_blocks_impl(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping, aclrtStream stream)
{
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    aclrtMemcpyKind memcpy_type;

    if ((!src_device.is_cpu()) && (!dst_device.is_cpu())) {
        TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same npu");
        memcpy_type = ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else if ((!src_device.is_cpu()) && dst_device.is_cpu()) {
        memcpy_type = ACL_MEMCPY_DEVICE_TO_HOST;
    } else if (src_device.is_cpu() && (!dst_device.is_cpu())) {
        memcpy_type = ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
        TORCH_CHECK(false, "Invalid device combination, src tensor device: ", src_device, ", dst tensor device: ", dst_device);
    }

    TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
    
    const int64_t num_blocks = block_mapping.size(0);
    const int64_t max_src_block = src.size(0);
    const int64_t max_dst_block = dst.size(0);
    for (size_t i = 0; i < num_blocks; i++) {
        int64_t src_block_number = block_mapping[i][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
        TORCH_CHECK(src_block_number >= 0 && src_block_number <= max_src_block,
                    "src block index ", src_block_number, " out of range (max: ", max_src_block, ")");
        TORCH_CHECK(dst_block_number >= 0 && dst_block_number <= max_dst_block,
                    "dst block index ", dst_block_number, " out of range (max: ", max_dst_block, ")");
        
        int64_t src_offset = src_block_number * block_size_in_bytes;
        int64_t dst_offset = dst_block_number * block_size_in_bytes;

        aclrtMemcpyAsync(dst_ptr + dst_offset, block_size_in_bytes,
                         src_ptr + src_offset, block_size_in_bytes,
                         memcpy_type, stream);
    }
}

void swap_blocks(torch::Tensor &x, torch::Tensor &y, const torch::Tensor &z)
{    
  
    const c10_npu::OptionalNPUGuard npuGuard(
        (!x.device().is_cpu()) ? x.device() : y.device()
    );
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();                       
    swap_blocks_impl(x, y, z, stream);           
    return;
}

AscendType get_dtype_from_torch(at::ScalarType scalarType)
{
    if (scalarType == at::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == at::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else {
        return AscendType::FP16;
    }
}

std::tuple<at::Tensor, at::Tensor> get_masked_input_and_mask(
    at::Tensor &input,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index)
    /*
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py#L161-L198
    Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |
    Parameters:
        org_vocab_start_index //base embeddings start
        org_vocab_end_index //base embeddings end
        num_org_vocab_padding //base embeddings padding
        added_vocab_start_index //LoRA embeddings start
        added_vocab_end_index //LoRA embeddings end
    */
{
    // Input validation
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(org_vocab_start_index >= 0, "org_vocab_start_index must be non-negative");
    TORCH_CHECK(org_vocab_end_index >= org_vocab_start_index, "org_vocab_end_index must be greater than org_vocab_start_index");
    TORCH_CHECK(num_org_vocab_padding >= 0, "num_org_vocab_padding must be non-negative");
    TORCH_CHECK(added_vocab_start_index >= org_vocab_end_index, "added_vocab_start_index must be greater than org_vocab_end_index");
    TORCH_CHECK(added_vocab_end_index >= added_vocab_start_index, "added_vocab_end_index must be greater than added_vocab_start_index");

    // Get total number of elements
    int64_t size = input.numel();

    // Create output tensors
    at::Tensor masked_input = at::empty_like(input);
	at::Tensor mask = at::empty_like(input).to(at::kBool);

    // Get data pointers
    void *input_ptr = input.data_ptr();
    void *masked_input_ptr = masked_input.data_ptr();
    void *mask_ptr = mask.data_ptr();

    // Get current stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Get scalar type
    at::ScalarType scalar_type = input.scalar_type();

    // Create and configure OpCommand
    at_npu::native::OpCommand cmd;
    cmd.Name("get_masked_input_and_mask");
    cmd.SetCustomHandler([scalar_type, size, stream,
                         input_ptr, masked_input_ptr, mask_ptr,
                         org_vocab_start_index, org_vocab_end_index,
                         num_org_vocab_padding, added_vocab_start_index,
                         added_vocab_end_index]() -> int {
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (size + aiv_num - 1) / aiv_num;

        // Call implementation
        get_masked_input_and_mask_impl(
            stream,
            input_ptr,
            masked_input_ptr,
            mask_ptr,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            size,
            loop_cnt,
            aiv_num);

        return 0;
    });
    cmd.Run();
    return {masked_input, mask};
}

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, input_hidden_token,
                          lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, num_tokens_per_core,
                         input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                       int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size, lora_rank,
                          slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size,
                         num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                 at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size,
                          seq_len_ptr, seq_len_size, y_ptr,
                          batch_size, input_hidden_token, lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size,
                         y_ptr, batch_size,
                         num_tokens_per_core, input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                          batch_size, lora_rank, slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                         batch_size, num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_prefill(
    const at::Tensor& x, const at::Tensor& topk_idx, const at::Tensor& topk_weights,
    const at::Tensor& num_tokens_per_rank, const at::Tensor& is_token_in_rank, at::Tensor& num_tokens_per_expert,
    int64_t num_worst_tokens, c10::string_view groupEp, int64_t rank, int64_t num_ranks) {
    std::vector<char> group_ep_chrs(groupEp.begin(), groupEp.end());
    group_ep_chrs.push_back('\0');
    char* group_ep_ptr = &group_ep_chrs[0];
    at::Tensor new_x = x;

    // Type checks
    TORCH_BIND_ASSERT(is_token_in_rank.scalar_type() == at::kBool);
    TORCH_BIND_ASSERT(num_tokens_per_expert.scalar_type() == at::kInt);
    TORCH_BIND_ASSERT(num_tokens_per_rank.scalar_type() == at::kInt);

    // Shape and contiguous checks
    TORCH_BIND_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    // TORCH_BIND_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    TORCH_BIND_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    TORCH_BIND_ASSERT(is_token_in_rank.size(0) == new_x.size(0) and is_token_in_rank.size(1) == num_ranks);
    TORCH_BIND_ASSERT(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens_per_expert.size(0) % num_ranks == 0);
    TORCH_BIND_ASSERT(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens_per_rank.size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0));
    auto hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert.size(0));
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);

    // Top-k checks
    int num_topk = 0;
    num_topk = static_cast<int>(topk_idx.size(1));
    TORCH_BIND_ASSERT(num_experts > 0);
    TORCH_BIND_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    TORCH_BIND_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens == topk_idx.size(0));
    TORCH_BIND_ASSERT(num_topk == topk_weights.size(1));
    TORCH_BIND_ASSERT(topk_weights.scalar_type() == at::kFloat);

    int send_per_group = 3;  // (send_to_expert_num, send_to_expert_offset, send_rank_tokens)

    auto send_data = at::empty({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    int64_t send_count = send_per_group * num_local_experts * num_ranks;

    auto send_data_offset = at::empty({num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_data = at::empty({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;

    EXEC_NPU_CMD(aclnnNotifyDispatch,
        send_data,
        num_tokens_per_expert, 
        send_count,
        num_tokens,
        group_ep_ptr,  // commGroup
        num_ranks,     // rankSize
        rank,          // rankId
        local_rank_size,
        local_rank_id,
        send_data_offset,
        recv_data);

    auto options_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    std::vector<int32_t> local_expert_acc(num_experts, 0);
    auto send_token_idx_cpu = at::empty({num_tokens, num_topk}, options_cpu);
    auto send_token_idx_ptr = send_token_idx_cpu.data_ptr<int>();

    auto topk_idx_cpu = topk_idx.to(at::kCPU);
    auto topk_idx_ptr = topk_idx_cpu.data_ptr<int64_t>();
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_topk; ++j) {
            int64_t expert_idx = topk_idx_ptr[i * num_topk + j];
            if (expert_idx >= 0) {
                int32_t cnt = local_expert_acc[expert_idx];
                send_token_idx_ptr[i * num_topk + j] = cnt;
                local_expert_acc[expert_idx]++;
            }
        }
    }

    TORCH_BIND_ASSERT(recv_data.dim() == 1 and recv_data.is_contiguous());
    TORCH_BIND_ASSERT(recv_data.size(0) % num_experts == 0);
    at::Tensor recv_offset_cpu = at::empty({num_experts}, options_cpu);
    at::Tensor recv_count_cpu = at::empty({num_experts}, options_cpu);
    auto recv_data_cpu = recv_data.to(at::kCPU);
    auto recv_data_ptr = recv_data_cpu.data_ptr<int>();
    auto recv_count_ptr = recv_count_cpu.data_ptr<int>();
    auto recv_offset_ptr = recv_offset_cpu.data_ptr<int>();
    int64_t total_recv_tokens = 0;
    int64_t num_max_dispatch_tokens_per_rank = 0;
    std::vector<int64_t> num_recv_tokens_per_expert_list;

    for (int64_t local_e = 0; local_e < num_local_experts; ++local_e) {
        int64_t local_expert_recv_tokens = 0;
        for (int64_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
            int64_t index = local_e * num_ranks + src_rank;
            int64_t pair_idx = send_per_group * (src_rank * num_local_experts + local_e);

            int recv_cnt = recv_data_ptr[pair_idx];                 // count from this src_rank for
                                                                    // this global_expert
            int recv_off = recv_data_ptr[pair_idx + 1];             // offset in that src_rank's window
            int64_t send_num_tokens = recv_data_ptr[pair_idx + 2];  // all bs from rank

            total_recv_tokens += recv_cnt;
            recv_count_ptr[index] = total_recv_tokens;
            recv_offset_ptr[index] = recv_off;
            num_max_dispatch_tokens_per_rank = std::max(num_max_dispatch_tokens_per_rank, send_num_tokens);

            local_expert_recv_tokens += recv_cnt;
        }
        num_recv_tokens_per_expert_list.push_back(local_expert_recv_tokens);
    }
    auto option = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    at::Tensor num_recv_tokens_per_expert = torch::from_blob(
        num_recv_tokens_per_expert_list.data(), {static_cast<int64_t>(num_recv_tokens_per_expert_list.size())}, option)
        .clone();

    at::Tensor expert_ids = topk_idx.to(at::kInt);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = 0;
    int64_t global_bs = static_cast<int64_t>(
        std::max(num_max_dispatch_tokens_per_rank * num_ranks, static_cast<int64_t>(num_worst_tokens)));

    auto send_token_idx = send_token_idx_cpu.to(x.device());
    auto recv_offset = recv_offset_cpu.to(x.device());
    auto recv_count = recv_count_cpu.to(x.device());

    int total_cnt = total_recv_tokens;
    if (total_cnt == 0) {
        total_cnt = 1;
    }
    auto expandx_out = at::empty({total_cnt, hidden}, x.options());
    auto dynamic_scales_out = at::empty({total_cnt}, at::dtype(at::kFloat).device(x.device()));
    auto expand_idx_out = at::empty({total_cnt * 3}, at::dtype(at::kInt).device(x.device()));

    EXEC_NPU_CMD(aclnnMoeDispatchNormal,
        new_x,
        expert_ids,
        send_data_offset,
        send_token_idx,
        recv_offset,
        recv_count,
        group_ep_ptr,  // commGroup
        num_ranks,     // rankSize
        rank,          // rankId
        group_ep_ptr,
        tp_size,
        tp_rank,
        num_experts,
        quant_mode,
        global_bs,
        expandx_out,
        dynamic_scales_out,
        expand_idx_out);

    // Return values
    return {expandx_out, expand_idx_out, recv_count, num_recv_tokens_per_expert};
}

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    int64_t dim_x = x.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    std::vector<int64_t> new_shape;
    at::Tensor rstd;
    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x_sizes = x.sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(1);
        }
    } else {
        new_shape.assign(dim_x, 1);
    }
    rstd = at::empty(new_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
    return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}

void transpose_kv_cache_by_block(
    const at::TensorList &kCache,
    const at::TensorList &vCache,
    const at::Tensor &blockIDs,
    int64_t blockSize,
    int64_t headNum,
    int64_t headDim,
    int64_t splitNum,
    int64_t layerNum)
{

    EXEC_NPU_CMD(aclnnTransposeKvCacheByBlock, kCache, vCache, blockIDs,
                 blockSize, headNum, headDim, splitNum, layerNum);

}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_copy_and_expand_eagle_inputs(
    const at::Tensor &target_token_ids,
    const at::Tensor &target_positions,
    const at::Tensor &next_token_ids,
    const at::Tensor &query_start_loc,
    const at::Tensor &query_end_loc,
    int64_t padding_token_id,
    int64_t parallel_drafting_token_id,
    int64_t num_padding_slots_per_request,
    bool shift_input_ids,
    int64_t total_draft_tokens)
{
    int64_t total_input_tokens = target_token_ids.size(0);
    int64_t num_reqs = query_start_loc.size(0) - 1;

    auto device = target_token_ids.device();
    at::Tensor out_input_ids = at::empty({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_positions = at::empty({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_is_rejected_token_mask = at::empty({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_is_masked_token_mask = at::empty({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_new_token_indices = at::empty({num_reqs * num_padding_slots_per_request}, at::dtype(at::kInt).device(device));
    at::Tensor out_hidden_state_mapping = at::empty({total_input_tokens}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnCopyAndExpandEagleInputs,
        target_token_ids, target_positions, next_token_ids, query_start_loc, query_end_loc,
        padding_token_id, parallel_drafting_token_id, num_padding_slots_per_request,
        shift_input_ids, total_input_tokens,
        out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
        out_new_token_indices, out_hidden_state_mapping);

    return {out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
            out_new_token_indices, out_hidden_state_mapping};
}

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    at::IntArrayRef num_accepted_tokens_opt,
    int64_t  activation_mode,
    int64_t  pad_slot_id,
    int64_t  run_mode)
{
    at::Tensor output = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnCausalConv1d,
                    x,
                    weight,
                    bias_opt,
                    conv_state,
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode,
                    output
                );

    return output;
}
  
// It is expected that further improvements will be made after it is incorporated into CANN on June 30th.
std::vector<at::Tensor> moe_grouped_matmul(
    at::Tensor x,
    at::Tensor weight,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type
)
{
    bool transpose_weight = false;
    bool weight_nz = true;

    at::TensorList x_list = at::TensorList(x);
    at::TensorList weight_list = at::TensorList(weight);
    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_list[0].options().dtype(x[0].scalar_type());
    auto m = x_list[0].sizes()[0];
    auto n = weight_list[0].sizes()[1];
    if (!transpose_weight) {
        n = weight_list[0].sizes()[2];
    }
    at::Tensor y_0 = at::empty(at::IntArrayRef{m, n}, options);
    y.emplace_back(y_0);
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnMoeGroupedMatmulWeightNz,
                x_list, weight_list, group_list, transpose_weight, result);

    return y;
}

} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{

    // vLLM-Ascend custom ops
    // Gemma RmsNorm
    ops.def(
        "npu_gemma_rms_norm(Tensor x, "
                            "Tensor gamma, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd)"
        );
    ops.impl("npu_gemma_rms_norm", torch::kPrivateUse1, &vllm_ascend::npu_gemma_rms_norm);
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", torch::kPrivateUse1, &vllm_ascend::weak_ref_tensor);

    ops.def(
        "get_masked_input_and_mask(Tensor input, "
        "                         int org_vocab_start_index, "
        "                         int org_vocab_end_index, "
        "                         int num_org_vocab_padding, "
        "                         int added_vocab_start_index, "
        "                         int added_vocab_end_index) -> (Tensor masked_input, Tensor mask)");
    ops.impl("get_masked_input_and_mask", torch::kPrivateUse1, &vllm_ascend::get_masked_input_and_mask);

    ops.def("bgmv_shrink(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y, float scale) -> ()");
    ops.impl("bgmv_shrink", torch::kPrivateUse1, &vllm_ascend::bgmv_shrink);

    ops.def(
        "bgmv_expand(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("bgmv_expand", torch::kPrivateUse1, &vllm_ascend::bgmv_expand);

    ops.def("sgmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y, float scale) -> ()");
    ops.impl("sgmv_shrink", torch::kPrivateUse1, &vllm_ascend::sgmv_shrink);

    ops.def(
        "sgmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("sgmv_expand", torch::kPrivateUse1, &vllm_ascend::sgmv_expand);

    ops.def(
        "mla_preprocess(Tensor hiddenState, Tensor wdqkv,"
        "               Tensor? descale0, Tensor gamma1, Tensor? beta1, Tensor wuq, Tensor? descale1,"
        "               Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache,"
        "               Tensor kv_cache_rope, Tensor slotmapping, Tensor? quant_scale0,"
        "               Tensor? quant_offset0, Tensor? bias0, Tensor? quant_scale1, Tensor? quant_offset1,"
        "               Tensor? bias1, Tensor? ctkv_scale, Tensor? q_nope_scale, str? cache_mode,"
        "               str? quant_mode, bool? enable_inner_out, Tensor! q_out0, Tensor! kv_cache_out0, Tensor! q_out1,"
        "               Tensor! kv_cache_out1, Tensor! inner_out) -> (Tensor q_out0, Tensor kv_cache_out0,"
        "                                          Tensor q_out1, Tensor kv_cache_out1, Tensor inner_out)"
    );
    ops.impl("mla_preprocess", torch::kPrivateUse1, &vllm_ascend::mla_preprocess);

    //batch_matmul ops refer to sgl-kernel-npu
    ops.def(
            "batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b, Tensor tensor_c, str? format_mode=None, str? quant_mode=None) -> ()");    
    ops.impl("batch_matmul_transpose", torch::kPrivateUse1, &vllm_ascend::batch_matmul_transpose);

    ops.def("swap_blocks(Tensor! x, Tensor! y, Tensor z) -> ()");    
    ops.impl("swap_blocks", torch::kPrivateUse1, &vllm_ascend::swap_blocks);

    // TurboQuant packed decode (MVP).
    ops.def(
        "turboquant_decode_packed_blocks(Tensor packed, Tensor codebook, Tensor rotation, int head_size, int out_dtype_code) -> Tensor"
    );
    ops.impl("turboquant_decode_packed_blocks", torch::kPrivateUse1, &vllm_ascend::turboquant_decode_packed_blocks);

    // TurboQuant packed decode for compact KV blocks: packed [U, BS, H, P], rotation_batched [BS*H, D, D].
    ops.def(
        "turboquant_decode_packed_blocks_compact(Tensor packed, Tensor codebook, Tensor rotation_batched, int head_size, int out_dtype_code) -> Tensor"
    );
    ops.impl("turboquant_decode_packed_blocks_compact", torch::kPrivateUse1, &vllm_ascend::turboquant_decode_packed_blocks_compact);

    ops.def(
        "turboquant_encode_packed_blocks(Tensor y, Tensor codebook, Tensor norms_fp16, int head_size, int bits) -> Tensor");
    ops.impl("turboquant_encode_packed_blocks", torch::kPrivateUse1, &vllm_ascend::turboquant_encode_packed_blocks);

    ops.def(
        "turboquant_pack_kv_for_cache(Tensor key, Tensor value, int slot_w_k, int slot_w_v) -> (Tensor, Tensor)");
    ops.impl("turboquant_pack_kv_for_cache", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache);

    ops.def(
        "turboquant_pack_kv_for_cache_v2(Tensor key, Tensor value, int slot_w_k, int slot_w_v) -> (Tensor, Tensor)");
    ops.impl("turboquant_pack_kv_for_cache_v2", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_v2);

    ops.def(
        "turboquant_pack_kv_for_cache_v3(Tensor key, Tensor value, int slot_w_k, int slot_w_v) -> (Tensor, Tensor)");
    ops.impl("turboquant_pack_kv_for_cache_v3", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_v3);

    ops.def(
        "turboquant_pack_kv_for_cache_v2_to_cache(Tensor key, Tensor value, Tensor slot_mapping, "
        "Tensor! key_cache, Tensor! value_cache, int slot_w_k, int slot_w_v) -> ()");
    ops.impl("turboquant_pack_kv_for_cache_v2_to_cache", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_v2_to_cache);

    ops.def(
        "turboquant_pack_kv_for_cache_v3_to_cache(Tensor key, Tensor value, Tensor slot_mapping, "
        "Tensor! key_cache, Tensor! value_cache, int slot_w_k, int slot_w_v) -> ()");
    ops.impl("turboquant_pack_kv_for_cache_v3_to_cache", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_v3_to_cache);

    ops.def(
        "bit_residual_pack_k8v4(Tensor key, Tensor value, Tensor slot_mapping, "
        "Tensor query_start_loc, Tensor rotation_t, Tensor! key_cache, "
        "Tensor! value_cache, int num_reqs, int block_size) -> ()");
    ops.impl("bit_residual_pack_k8v4", torch::kPrivateUse1,
             &vllm_ascend::bit_residual_pack_k8v4);

    ops.def(
        "turboquant_pack_kv_for_cache_to_cache(Tensor key, Tensor value, Tensor slot_mapping, "
        "Tensor! key_cache, Tensor! value_cache, int slot_w_k, int slot_w_v) -> ()");
    ops.impl("turboquant_pack_kv_for_cache_to_cache", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_to_cache);

    ops.def(
        "turboquant_pack_kv_for_cache_4bit(Tensor key, Tensor value, Tensor slot_mapping, "
        "Tensor query_start_loc, Tensor codebook, Tensor rotation_t, Tensor! key_cache, "
        "Tensor! value_cache, int num_reqs, int block_size) -> ()");
    ops.impl("turboquant_pack_kv_for_cache_4bit", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_4bit);

    ops.def("turboquant_pack_register_tables(Tensor codebook, Tensor rotation_t) -> ()");
    ops.impl("turboquant_pack_register_tables", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_register_tables);

    ops.def(
        "turboquant_rotate_matmul_probe(Tensor a, Tensor b, int probe_mode) -> Tensor");
    ops.impl("turboquant_rotate_matmul_probe", torch::kPrivateUse1,
             &vllm_ascend::turboquant_rotate_matmul_probe);

    ops.def(
        "turboquant_fused_infer_attention_score_8bit("
        "Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, "
        "Tensor atten_mask, Tensor actual_seq_len_q, Tensor actual_seq_len_kv, "
        "Tensor codebook, Tensor rotation, Tensor codebook_value, Tensor rotation_value, "
        "int num_heads, int num_kv_heads, int head_size, int block_size, float scale_value"
        ") -> Tensor"
    );
    ops.impl(
        "turboquant_fused_infer_attention_score_8bit",
        torch::kPrivateUse1,
        &vllm_ascend::turboquant_fused_infer_attention_score_8bit);

    ops.def(
        "turboquant_attention_paged8bit("
        "Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, "
        "int[] actual_seq_len_q, int[] actual_seq_len_kv, "
        "Tensor codebook, Tensor rotation, Tensor codebook_value, Tensor rotation_value, "
        "int num_heads, int num_kv_heads, int head_size, int block_size, "
        "int max_actual_seq_len, float scale_value"
        ") -> Tensor");
    ops.impl(
        "turboquant_attention_paged8bit",
        torch::kPrivateUse1,
        &vllm_ascend::turboquant_attention_paged8bit);

    ops.def(
        "turboquant_attention_paged4bit("
        "Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, "
        "int[] actual_seq_len_q, int[] actual_seq_len_kv, "
        "Tensor codebook, Tensor rotation, Tensor codebook_value, Tensor rotation_value, "
        "int num_heads, int num_kv_heads, int head_size, int block_size, "
        "int max_actual_seq_len, float scale_value"
        ") -> Tensor");
    ops.impl(
        "turboquant_attention_paged4bit",
        torch::kPrivateUse1,
        &vllm_ascend::turboquant_attention_paged4bit);

    ops.def(
        "bit_residual_attention_paged_k8v4("
        "Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, "
        "int[] actual_seq_len_q, int[] actual_seq_len_kv, "
        "Tensor rotation_key, Tensor rotation_value, "
        "int num_heads, int num_kv_heads, int head_size, int block_size, "
        "int max_actual_seq_len, float scale_value, Tensor? out=None"
        ") -> Tensor");
    ops.impl(
        "bit_residual_attention_paged_k8v4",
        torch::kPrivateUse1,
        &vllm_ascend::bit_residual_attention_paged_k8v4);

    ops.def(
        "bit_residual_fia_paged_k8v4("
        "Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, "
        "int[] actual_seq_len_q, int[] actual_seq_len_kv, "
        "Tensor? atten_mask, Tensor rotation_key, Tensor rotation_value, "
        "int num_heads, int num_kv_heads, int head_size, int block_size, "
        "float scale_value, int pre_tokens, int next_tokens, int sparse_mode"
        ") -> Tensor");
    ops.impl(
        "bit_residual_fia_paged_k8v4",
        torch::kPrivateUse1,
        &vllm_ascend::bit_residual_fia_paged_k8v4);

    // TurboQuant 8-bit paged decode (设计文档 §2.6 方案 X / Phase 1).
    // 算子吞掉 block_table 寻址,host 侧只需 ceil + cumsum 算 gather_block_ids。
    ops.def(
        "turboquant_decode_paged_8bit("
        "Tensor key_cache, Tensor value_cache, Tensor gather_block_ids, "
        "Tensor codebook, Tensor rotation, "
        "int head_size, int block_size, int out_dtype_code, int mode"
        ") -> (Tensor, Tensor)");
    ops.impl("turboquant_decode_paged_8bit", torch::kPrivateUse1,
             &vllm_ascend::turboquant_decode_paged_8bit);

    ops.def(
        "grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "                            Tensor group_list, *, Tensor? bias=None,"
        "                            Tensor? offset=None) -> (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant);

    ops.def(
        "dispatch_gmm_combine_decode(Tensor x, Tensor expert_ids, Tensor[] gmm1_permuted_weight,"
        "                            Tensor[] gmm1_permuted_weight_scale,"
        "                            Tensor[] gmm2_weight, Tensor[] gmm2_weight_scale,"
        "                            Tensor expert_scales, Tensor? expert_smooth_scales=None,"
        "                            Tensor? x_active_mask=None,"
        "                            str group_ep='',"
        "                            int ep_rank_size=0, int ep_rank_id=0, int moe_expert_num=0,"
        "                            int shared_expert_num=1, int shared_expert_rank_num=0,"
        "                            int quant_mode=0,"
        "                            int global_bs=0) -> (Tensor output, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_gmm_combine_decode", torch::kPrivateUse1, &vllm_ascend::dispatch_gmm_combine_decode);

    ops.def(
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,"
        "                                                  Tensor group_list, *,"
        "                                                  Tensor? bias=None, Tensor? offset=None) ->"
        "                                                  (Tensor output, Tensor output_scale, Tensor output_offset)"
    );
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz_tensor_list);

    ops.def(
        "npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *,"
        "                      Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None,"
        "                      Tensor? block_table=None, str layout_query='BSND', str layout_key='BSND',"
        "                      int sparse_count=2048, int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer);

    ops.def(
        "npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                           Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
        "                           Tensor? block_table=None, Tensor? actual_seq_lengths_query=None,"
        "                           Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None,"
        "                           Tensor? key_rope=None, str layout_query='BSND', str layout_kv='BSND',"
        "                           int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_sparse_flash_attention", torch::kPrivateUse1, &vllm_ascend::npu_sparse_flash_attention);

    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,"
        "                     Tensor[] scale1, Tensor[] scale2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out, Tensor! expert_token_nums) -> (Tensor out, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_ffn_combine", torch::kPrivateUse1, &vllm_ascend::dispatch_ffn_combine);

    ops.def("matmul_allreduce_add_rmsnorm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, \
        str groupTp, int tpRankSize, int tpRankId, float epsilon, bool isTransB, bool isGatherAddOut) -> (Tensor output, Tensor add_out)");
    ops.impl("matmul_allreduce_add_rmsnorm", torch::kPrivateUse1, &vllm_ascend::matmul_allreduce_add_rmsnorm);

    ops.def("get_dispatch_layout(Tensor topk_idx, int num_experts, int "
            "num_ranks) -> (Tensor num_tokens_per_rank, Tensor "
            "num_tokens_per_expert, Tensor is_token_in_rank_bool)");
    ops.impl("get_dispatch_layout", torch::kPrivateUse1,
             &vllm_ascend::get_dispatch_layout);

    ops.def(
        "dispatch_prefill(Tensor x, Tensor topk_idx, Tensor topk_weights, "
        "Tensor num_tokens_per_rank, Tensor is_token_in_rank, Tensor "
        "num_tokens_per_expert, int num_worst_tokens, str groupEp, int rank, "
        "int num_ranks) -> (Tensor expandx_out, Tensor expand_idx_out, Tensor "
        "recv_count, Tensor num_recv_tokens_per_expert)");
    ops.impl("dispatch_prefill", torch::kPrivateUse1,
             &vllm_ascend::dispatch_prefill);

    ops.def("combine_prefill(Tensor x, Tensor topk_idx, Tensor topk_weights, "
            "Tensor src_idx, Tensor send_head, str grouEp, int rank, int "
            "num_ranks) -> Tensor");
    ops.impl("combine_prefill", torch::kPrivateUse1,
             &vllm_ascend::combine_prefill);
    
    ops.def(
        "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "                            int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, "
        "                            bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], "
        "                            int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)"
    );
    ops.impl("npu_moe_init_routing_custom", torch::kPrivateUse1, &vllm_ascend::npu_moe_init_routing_custom);
    // vLLM-Ascend custom ops
    ops.def(
        "moe_gating_top_k(Tensor x, "
                            "int k, "
                            "int k_group, "
                            "int group_count, "
                            "int group_select_mode, "
                            "int renorm, "
                            "int norm_type, "
                            "bool out_flag, "
                            "float routed_scaling_factor, "
                            "float eps,"
                            "Tensor? bias_opt=None)"
                            
        "-> (Tensor y ,Tensor expert_idx, Tensor out)"
        );
    ops.impl("moe_gating_top_k", torch::kPrivateUse1,&vllm_ascend::moe_gating_top_k);

    ops.def(
        "npu_add_rms_norm_bias(Tensor x1, "
                            "Tensor x2, "
                            "Tensor gamma, "
                            "Tensor? beta=None, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd, Tensor x)"
        );
    ops.impl("npu_add_rms_norm_bias", torch::kPrivateUse1, &vllm_ascend::npu_add_rms_norm_bias);

    ops.def("npu_apply_top_k_top_p(Tensor logits, Tensor? p=None, Tensor? k=None) -> Tensor");
    ops.impl("npu_apply_top_k_top_p", torch::kPrivateUse1, &vllm_ascend::npu_apply_top_k_top_p);
    ops.def(
        "transpose_kv_cache_by_block(Tensor[] kCache, Tensor[] vCache, Tensor blockIDs, int blockSize, int headNum, int headDim, int splitNum, int layerNum) -> ()"
    );
    ops.impl("transpose_kv_cache_by_block", torch::kPrivateUse1, &vllm_ascend::transpose_kv_cache_by_block);

    ops.def(
        "npu_copy_and_expand_eagle_inputs(Tensor target_token_ids, Tensor target_positions, "
        "Tensor next_token_ids, Tensor query_start_loc, Tensor query_end_loc, "
        "int padding_token_id, int parallel_drafting_token_id, int num_padding_slots_per_request, "
        "bool shift_input_ids, int total_draft_tokens) -> "
        "(Tensor out_input_ids, Tensor out_positions, Tensor out_is_rejected_token_mask, "
        "Tensor out_is_masked_token_mask, Tensor out_new_token_indices, Tensor out_hidden_state_mapping)"
    );
    ops.impl("npu_copy_and_expand_eagle_inputs", torch::kPrivateUse1, &vllm_ascend::npu_copy_and_expand_eagle_inputs);
    ops.def(
        "npu_causal_conv1d_custom(Tensor x, "
        "                         Tensor weight, "
        "                         Tensor conv_state, "
        "                         Tensor? bias_opt, "
        "                         int[] query_start_loc_opt, "
        "                         int[] cache_indices_opt, "
        "                         int[] initial_state_mode_opt, "
        "                         int[] num_accepted_tokens_opt, "
        "                         int activation_mode, "
        "                         int pad_slot_id, "
        "                         int run_mode"
        ") -> (Tensor output)");
    ops.impl("npu_causal_conv1d_custom", torch::kPrivateUse1, &vllm_ascend::npu_causal_conv1d_custom);
    ops.def(
        "moe_grouped_matmul("
            "Tensor x,"
            "Tensor weight,"
            "Tensor group_list,"
            "int split_item,"
            "int group_type,"
            "int group_list_type)"

        "-> Tensor[]"
    );
    ops.impl("moe_grouped_matmul", torch::kPrivateUse1,&vllm_ascend::moe_grouped_matmul);

    // This operator is planned to be integrated into PTA in the near future.
    // Once that happens, the implementation in csrc will be removed.
    ops.def(
        "npu_lightning_indexer_quant(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
        "                            Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, "
        "                            Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, "
        "                            int query_quant_mode=0, int key_quant_mode=0, "
        "                            str layout_query='BSND', str layout_key='BSND',"
        "                            int sparse_count=2048, int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_lightning_indexer_quant", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer_quant);
}
