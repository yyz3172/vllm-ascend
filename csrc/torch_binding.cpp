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
#include <ATen/core/Formatting.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include <cstdio>
#include <cstdlib>
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
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "mla_preprocess/mla_preprocess_torch_adpt.h"
#endif
#include "mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
#include "mc2/dispatch_gmm_combine_decode/dispatch_gmm_combine_decode_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h"
#include "attention/lightning_indexer/lightning_indexer_torch_adpt.h"
#include "mc2/matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"
#include "attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "attention/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_torch_adpt.h"
#include "attention/lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
#include "attention/ngram_spec_decode/ngram_spec_decode_torch_adpt.h"
#include "moe/causal_conv1d_v310/causal_conv1d_310_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule_v310/recurrent_gated_delta_rule_310_torch_adpt.h"
#include "attention/store_kv_block/store_kv_block_torch_adpt.h"
#include "attention/store_kv_block_metadata/store_kv_block_metadata_torch_adpt.cpp"
#include "attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h"
#include "moe_combine_normal/moe_combine_normal_torch_adpt.h"
#include "moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"
#include "sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
#include "turboquant_rotate_matmul_probe/op_host/aclnn_turboquant_rotate_matmul_probe.h"
#include "turboquant_pack_kv_for_cache_fused/op_host/aclnn_turboquant_pack_kv_for_cache_fused.h"
#include "turboquant_pack_kv_for_cache_to_cache/op_host/aclnn_turboquant_pack_kv_for_cache_to_cache.h"
#include <mutex>
#include <unordered_map>
#include "turboquant_fused_infer_attention_score8bit/op_host/aclnn_turboquant_fused_infer_attention_score8bit.h"
#include "turboquant_attention_paged8bit/op_host/aclnn_turboquant_attention_paged8bit.h"
#include "turboquant_decode_paged8bit/op_host/aclnn_turboquant_decode_paged8bit.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#include <c10/core/Device.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace vllm_ascend {

namespace {

constexpr int64_t DSA_SLOT_MAPPING_FLAT = 1;
constexpr int64_t DSA_SLOT_MAPPING_BLOCK_OFFSET = 2;

struct DevicePrintPayload {
    std::string message;
    at::Tensor host_tensor_snapshot;
};

std::mutex& get_device_print_mutex()
{
    static std::mutex device_print_mutex;
    return device_print_mutex;
}

void device_print_callback(void* args)
{
    // device_print is a debug-only helper. We intentionally do not reclaim the
    // callback payload here because aclgraph replay may re-execute the same host
    // callback payload multiple times. Freeing it on first execution would make
    // later replays dereference a dangling pointer.
    auto* payload = static_cast<DevicePrintPayload*>(args);
    if (payload == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> guard(get_device_print_mutex());
    if (!payload->message.empty()) {
        std::cout << payload->message;
    }

    if (payload->host_tensor_snapshot.defined()) {
        if (!payload->message.empty()) {
            std::cout << std::endl;
        }
        at::print(std::cout, payload->host_tensor_snapshot.contiguous(), 120);
    }

    std::cout << std::endl;
    std::cout.flush();
}

void enqueue_device_print(std::unique_ptr<DevicePrintPayload> payload,
                          aclrtStream stream)
{
    auto* raw_payload = payload.release();
    const aclError ret = aclrtLaunchHostFunc(stream, device_print_callback,
                                             raw_payload);
    if (ret != ACL_SUCCESS) {
        delete raw_payload;
    }
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtLaunchHostFunc failed, error code: ", ret);
}

}

void swap_blocks_batch(const torch::Tensor& src_ptrs,
                       const torch::Tensor& dst_ptrs,
                       const torch::Tensor& sizes,
                       int64_t direction) {

    TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
    TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
    TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
    TORCH_CHECK(src_ptrs.dtype() == torch::kInt64, "src_ptrs must be int64");
    TORCH_CHECK(dst_ptrs.dtype() == torch::kInt64, "dst_ptrs must be int64");
    TORCH_CHECK(sizes.dtype() == torch::kInt64, "sizes must be int64");

    const int64_t n = src_ptrs.size(0);
    TORCH_CHECK(dst_ptrs.size(0) == n, "dst_ptrs length must match src_ptrs");
    TORCH_CHECK(sizes.size(0) == n, "sizes length must match src_ptrs");

    if (n == 0) return;

    const int64_t* src_data = src_ptrs.data_ptr<int64_t>();
    const int64_t* dst_data = dst_ptrs.data_ptr<int64_t>();
    const int64_t* size_data = sizes.data_ptr<int64_t>();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    aclrtMemcpyKind memcpy_kind;
    switch (direction) {
        case 0:
            memcpy_kind = ACL_MEMCPY_HOST_TO_DEVICE;
            break;
        case 1:
            memcpy_kind = ACL_MEMCPY_DEVICE_TO_HOST;
            break;
        case 2:
            memcpy_kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            break;
        default:
            TORCH_CHECK(false,
                        "swap_blocks_batch: invalid direction ", direction,
                        " (expected 0=H2D, 1=D2H, 2=D2D)");
    }

    // =========================================================================
    // path 1: aclrtMemcpyBatchAsync (CANN 8.5+)
    // =========================================================================
#if defined(CANN_MEMCPY_BATCH_ASYNC)
    if (memcpy_kind != ACL_MEMCPY_DEVICE_TO_DEVICE) {
        static_assert(sizeof(void*) == sizeof(int64_t),
                      "void* and int64_t must be the same size");
        static_assert(sizeof(size_t) == sizeof(int64_t),
                      "size_t and int64_t must be the same size");

        void** dst_arr = reinterpret_cast<void**>(
            const_cast<int64_t*>(dst_data));
        void** src_arr = reinterpret_cast<void**>(
            const_cast<int64_t*>(src_data));
        size_t* size_arr = reinterpret_cast<size_t*>(
            const_cast<int64_t*>(size_data));
        size_t* dest_maxs = size_arr;

        // aclrtMemcpyBatchAttr uses srcLoc/dstLoc (aclrtMemLocation)
        // to specify memory locations, not aclrtMemcpyKind.
        int32_t device_id = 0;
        aclrtGetDevice(&device_id);

        aclrtMemLocation host_loc = {};
        host_loc.type = ACL_MEM_LOCATION_TYPE_HOST;
        host_loc.id = 0;

        aclrtMemLocation device_loc = {};
        device_loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        device_loc.id = device_id;

        aclrtMemcpyBatchAttr attr = {};
        if (memcpy_kind == ACL_MEMCPY_HOST_TO_DEVICE) {
            attr.srcLoc = host_loc;
            attr.dstLoc = device_loc;
        } else {  // ACL_MEMCPY_DEVICE_TO_HOST
            attr.srcLoc = device_loc;
            attr.dstLoc = host_loc;
        }

        size_t attrs_index = 0;
        size_t fail_index = 0;

        aclError result = aclrtMemcpyBatchAsync(
            dst_arr, dest_maxs, src_arr, size_arr,
            static_cast<size_t>(n),
            &attr, &attrs_index, 1,
            &fail_index, stream);

        TORCH_CHECK(result == ACL_SUCCESS,
                    "aclrtMemcpyBatchAsync failed at index ", fail_index,
                    " with error code ", result);
        return;
    }
#endif

    // =========================================================================
    // path 2: aclrtMemcpyAsync
    // =========================================================================
    for (int64_t i = 0; i < n; i++) {
        void* dst = reinterpret_cast<void*>(dst_data[i]);
        const void* src = reinterpret_cast<const void*>(src_data[i]);
        size_t copy_size = static_cast<size_t>(size_data[i]);

        aclError ret = aclrtMemcpyAsync(
            dst,
            copy_size,
            src,
            copy_size,
            memcpy_kind,
            stream);

        TORCH_CHECK(ret == ACL_SUCCESS,
                    "aclrtMemcpyAsync failed at index ", i,
                    " with error code ", ret,
                    ", src=", src_data[i],
                    ", dst=", dst_data[i],
                    ", size=", size_data[i]);
    }
}
namespace {

constexpr uint32_t TQ_SINGLE_ROT_M_PAD = 16;

std::vector<uint8_t> GenerateTurboQuantRotateTiling(
    platform_ascendc::PlatformAscendC* ascendc_platform,
    uint32_t block_dim,
    uint32_t vec_per_core) {
    std::vector<uint8_t> tiling_buf(sizeof(TCubeTiling), 0);
    optiling::TCubeTiling tiling_data;
    matmul_tiling::MultiCoreMatmulTiling tiling_api(*ascendc_platform);

    // SetDim is the number of AIV clients participating in the KFC matmul handshake.
    // One MIX launch block expands to AIV-0/AIV-1 plus AIC-0, so keep this local
    // matmul at two AIV participants. Do not scale it by <<<blockDim>>>.
    (void)block_dim;
    tiling_api.SetDim(2);
    tiling_api.SetAType(
        matmul_tiling::TPosition::VECOUT,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16,
        false);
    tiling_api.SetBType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16,
        false);
    tiling_api.SetCType(
        matmul_tiling::TPosition::VECIN,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    tiling_api.SetBiasType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
    tiling_api.SetOrgShape(vec_per_core, 128, 128);
    tiling_api.SetShape(vec_per_core, 128, 128);
    tiling_api.EnableBias(false);
    tiling_api.SetBufferSpace(-1, -1, -1);

    const int64_t ret = tiling_api.GetTiling(tiling_data);
    TORCH_CHECK(ret != -1, "failed to generate TurboQuant rotate matmul tiling");
    const uint32_t tiling_size = tiling_data.GetDataSize();
    TORCH_CHECK(tiling_size <= tiling_buf.size(), "TurboQuant rotate tiling buffer is too small");
    tiling_data.SaveToBuffer(tiling_buf.data(), tiling_size);
    return tiling_buf;
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

// TurboQuant decode (packed uint8 -> fp16/bf16).
// Two-stage implementation:
//  1) AscendC kernel: unpack uint4 + codebook lookup + scale by norm -> y_hat [N, D] fp16
//  2) Use optimized NPU matmul: out = y_hat @ rotation -> fp16/bf16
//
// packed: [N, P] uint8, where P = D/2 + 2 (uint4 indices packed + fp16 norm)
// codebook: [16] fp16, rotation: [D, D] fp16.
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
    TORCH_CHECK(codebook.scalar_type() == at::kHalf, "codebook must be fp16");
    TORCH_CHECK(rotation.scalar_type() == at::kHalf, "rotation must be fp16");
    TORCH_CHECK(codebook.numel() == 16, "codebook must have 16 entries");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == head_size && rotation.size(1) == head_size,
                "rotation must be [D, D]");
    TORCH_CHECK(head_size % 2 == 0, "head_size must be even for 4-bit packing");

    const int64_t packed_bytes = head_size / 2 + 2;
    TORCH_CHECK(packed.dim() == 2 && packed.size(1) == packed_bytes,
                "packed must be [N, P] with P=head_size/2+2");

    at::ScalarType out_dtype = (out_dtype_code == 1) ? at::kBFloat16 : at::kHalf;
    // Stage 1 output always fp16 for matmul compatibility/perf.
    // IMPORTANT: The AscendC stage-1 kernel treats `packed` as contiguous row-major bytes.
    // If `packed` is a view with non-trivial strides, raw pointer indexing will read
    // the wrong bytes and corrupt decode (often showing up as garbled text output).
    const at::Tensor packed_c = packed.contiguous();
    const at::Tensor codebook_c = codebook.contiguous();
    const at::Tensor rotation_c = rotation.contiguous();

    at::Tensor y_hat = at::empty({packed_c.size(0), head_size}, packed_c.options().dtype(at::kHalf));
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
        vec_per_core);
    // Stage 2: matmul on NPU.
    //
    // IMPORTANT: Keep Stage2 numerically aligned with the PyTorch reference path
    // (`at::matmul(y_hat, rotation)`), otherwise greedy decoding can diverge wildly
    // even when Stage1 is only slightly off.
    at::Tensor x_hat = at::matmul(y_hat, rotation_c);
    if (out_dtype == at::kHalf) {
        out.copy_(x_hat);
    } else {
        out.copy_(x_hat.to(out_dtype));
    }
    return out;
}

// TurboQuant decode for compact KV cache blocks.
// packed: [U, BS, H, P] uint8, rotation_batched: [BS*H, D, D] fp16.
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
    TORCH_CHECK(codebook.scalar_type() == at::kHalf, "codebook must be fp16");
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
    TORCH_CHECK(rotation_batched.scalar_type() == at::kHalf, "rotation_batched must be fp16");

    at::ScalarType out_dtype = (out_dtype_code == 1) ? at::kBFloat16 : at::kHalf;
    const at::Tensor codebook_c = codebook.contiguous();
    const at::Tensor rotation_batched_c = rotation_batched.contiguous();
    at::Tensor y_hat = at::empty({U * batch, head_size}, packed.options().dtype(at::kHalf));
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
        vec_per_core);

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
    if (out_dtype == at::kHalf) {
        out.copy_(x_hat);
    } else {
        out.copy_(x_hat.to(out_dtype));
    }
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
    TORCH_CHECK(probe_mode == 0 || probe_mode == 1, "probe_mode must be 0 (regist only) or 1 (matmul)");

    const at::Tensor a_c = a.contiguous();
    const at::Tensor b_c = b.contiguous();
    const int64_t m = a_c.size(0);
    TORCH_CHECK(m >= 1 && m <= 128, "M must be in [1, 128]");

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

// TurboQuant pack K/V for paged cache: norm + rotate + nearest-neighbor encode.
// When bits_key == bits_value, K and V are encoded in one batched kernel launch.
// key/value: [T, H, D] fp16/bf16 on NPU; codebook_*: [2^bits] fp16; rotation_t_*: [D, D] fp16 (R^T).
std::tuple<at::Tensor, at::Tensor> turboquant_pack_kv_for_cache(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &codebook_key,
    const at::Tensor &rotation_t_key,
    const at::Tensor &codebook_value,
    const at::Tensor &rotation_t_value,
    int64_t bits_key,
    int64_t bits_value,
    int64_t slot_w_k,
    int64_t slot_w_v) {
    const uint32_t debug_blocks =
        std::getenv("VLLM_ASCEND_TURBOQUANT_DEBUG_BLOCKS") != nullptr ? 1U : 0U;
    const bool no_kfc = std::getenv("VLLM_ASCEND_TURBOQUANT_NO_KFC") != nullptr;
    if (debug_blocks != 0U) {
        std::printf(
            "[TQ_PACK] enter key_dim=%ld value_dim=%ld bits_key=%ld bits_value=%ld slot_w_k=%ld slot_w_v=%ld no_kfc=%d\n",
            static_cast<long>(key.dim()), static_cast<long>(value.dim()), static_cast<long>(bits_key),
            static_cast<long>(bits_value), static_cast<long>(slot_w_k), static_cast<long>(slot_w_v),
            no_kfc ? 1 : 0);
        std::fflush(stdout);
    }
    TORCH_CHECK(key.is_privateuseone(), "key must be on NPU");
    TORCH_CHECK(value.is_privateuseone(), "value must be on NPU");
    TORCH_CHECK(codebook_key.is_privateuseone() && codebook_value.is_privateuseone(),
                "codebooks must be on NPU");
    TORCH_CHECK(rotation_t_key.is_privateuseone() && rotation_t_value.is_privateuseone(),
                "rotations must be on NPU");
    TORCH_CHECK(value.device() == key.device() && codebook_key.device() == key.device() &&
                    codebook_value.device() == key.device() && rotation_t_key.device() == key.device() &&
                    rotation_t_value.device() == key.device(),
                "key/value/codebooks/rotations must be on the same NPU device");
    TORCH_CHECK(key.dim() >= 1 && value.dim() >= 1, "key/value must have at least 1 dim");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(bits_key == 4 || bits_key == 8, "bits_key must be 4 or 8");
    TORCH_CHECK(bits_value == 4 || bits_value == 8, "bits_value must be 4 or 8");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    if (bits_key == 4 || bits_value == 4) {
        TORCH_CHECK(head_size % 2 == 0, "head_size must be even for 4-bit packing");
    }

    // Initial fused implementation supports only: fp16 + bits_key==bits_value==8 + head_size==128.
    TORCH_CHECK(key.scalar_type() == at::kHalf, "initial fused pack supports fp16 only");
    TORCH_CHECK(bits_key == 8 && bits_value == 8, "initial fused pack supports 8-bit only");
    TORCH_CHECK(head_size == 128, "initial fused pack supports head_size=128 only");
    TORCH_CHECK(codebook_key.scalar_type() == at::kHalf && codebook_value.scalar_type() == at::kHalf,
                "initial fused pack expects fp16 codebooks");
    TORCH_CHECK(rotation_t_key.scalar_type() == at::kHalf && rotation_t_value.scalar_type() == at::kHalf,
                "initial fused pack expects fp16 rotations");
    TORCH_CHECK(codebook_key.numel() == 256, "8-bit codebook must have 256 entries");
    TORCH_CHECK(rotation_t_key.dim() == 2 && rotation_t_key.size(0) == 128 && rotation_t_key.size(1) == 128,
                "rotation_t_key must be [128,128]");
    TORCH_CHECK(codebook_value.numel() == 256, "8-bit codebook must have 256 entries");
    TORCH_CHECK(rotation_t_value.dim() == 2 && rotation_t_value.size(0) == 128 && rotation_t_value.size(1) == 128,
                "rotation_t_value must be [128,128]");
    if (debug_blocks != 0U) {
        std::printf("[TQ_PACK] before equal checks\n");
        std::fflush(stdout);
    }
    if (debug_blocks != 0U) {
        std::printf("[TQ_PACK] codebook equal check done\n");
        std::fflush(stdout);
    }
    if (debug_blocks != 0U) {
        std::printf("[TQ_PACK] rotation equal check done\n");
        std::fflush(stdout);
    }

    if (debug_blocks != 0U) {
        std::printf("[TQ_PACK] before reshape/contiguous\n");
        std::fflush(stdout);
    }
    at::Tensor key_flat = key.reshape({-1, head_size}).contiguous();
    at::Tensor value_flat = value.reshape({-1, head_size}).contiguous();
    const int64_t n_vec = key_flat.size(0);
    TORCH_CHECK(value_flat.size(0) == n_vec, "key/value row count mismatch");

    const int64_t packed_bytes = head_size + 2;
    TORCH_CHECK(slot_w_k >= packed_bytes, "slot_w_k must be >= head_size+2 for 8-bit");
    TORCH_CHECK(slot_w_v >= packed_bytes, "slot_w_v must be >= head_size+2 for 8-bit");

    const at::Tensor key_c = key_flat.contiguous();
    const at::Tensor value_c = value_flat.contiguous();
    const at::Tensor codebook_c = codebook_key.contiguous();
    const at::Tensor rot_c = rotation_t_key.contiguous();
    if (debug_blocks != 0U) {
        std::printf("[TQ_PACK] contiguous done nVec=%ld\n", static_cast<long>(n_vec));
        std::fflush(stdout);
    }

    at::Tensor packed_k = at::empty({n_vec, slot_w_k}, key.options().dtype(at::kByte));
    at::Tensor packed_v = at::empty({n_vec, slot_w_v}, key.options().dtype(at::kByte));

    const c10_npu::OptionalNPUGuard npuGuard(key_c.device());

    // Batch rows per core for one [M,128]@[128,128] matmul (Cube M aligned to 16).
    uint32_t vec_per_core = 128;
    if (n_vec < 128) {
        vec_per_core = static_cast<uint32_t>(((n_vec + 15) / 16) * 16);
        if (vec_per_core == 0) {
            vec_per_core = 16;
        }
    }
    const int64_t pack_mode = no_kfc ? 1 : 0;
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    if (debug_blocks != 0U) {
        std::printf(
            "[TQ_PACK] launch aclnn pack_mode=%ld nVec=%ld vecPerCore=%ld slot_w_k=%ld slot_w_v=%ld\n",
            static_cast<long>(pack_mode), static_cast<long>(n_vec), static_cast<long>(vec_per_core_i64),
            static_cast<long>(slot_w_k), static_cast<long>(slot_w_v));
        std::fflush(stdout);
    }

    // Same launch path as turboquant_rotate_matmul_probe: aclnn custom op + SetTilingKey +
    // KERNEL_TASK_TYPE(pack_mode, MIX or AIC_ONLY). Avoids <<<>>> direct launch REGIST hang.
    EXEC_NPU_CMD(
        aclnnTurboquantPackKvForCacheFused,
        key_c,
        value_c,
        codebook_c,
        rot_c,
        pack_mode,
        n_vec,
        slot_w_k,
        slot_w_v,
        vec_per_core_i64,
        packed_k,
        packed_v);

    std::vector<int64_t> shape_k(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> shape_v(value.sizes().begin(), value.sizes().end());
    shape_k.back() = slot_w_k;
    shape_v.back() = slot_w_v;

    return std::make_tuple(
        packed_k.reshape(shape_k).to(at::kChar),
        packed_v.reshape(shape_v).to(at::kChar));
}

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
// Direct kernel wrappers depend on vllm_ascend_kernels, which is skipped on
// 310P and A5 builds.
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
#endif

at::Tensor convert_hamming_dist_top_k_output(const at::Tensor &hashq,
                                             const at::Tensor &hashkCache,
                                             const c10::optional<at::Tensor>& indices) {
    if (indices.has_value()) {
        return indices.value();
    }
    uint32_t MAX_BLOCK_PER_REQ_INHSA = 512;

    auto n_bs = hashq.size(0);
    auto n_kv_heads = hashkCache.size(1);
    auto n_max_kv = MAX_BLOCK_PER_REQ_INHSA;
    at::Tensor res = at::empty({n_bs, n_kv_heads, n_max_kv}, torch::TensorOptions().dtype(torch::kInt32).device(hashq.device()));
    return res;
}

at::Tensor npu_hamming_dist_top_k(const at::Tensor &hashq,
                                       const at::Tensor &hashkCache,
                                       const at::Tensor& hashkCacheRope,
                                       const at::Tensor &topN,
                                       const at::Tensor &seqLen,
                                       const c10::optional<at::Tensor> &chunkSize,
                                       const c10::optional<int64_t> maxSeqLen,
                                       const c10::optional<int64_t> sink,
                                       const c10::optional<int64_t> recent,
                                       const c10::optional<int64_t> supportOffload,
                                       const c10::optional<at::Tensor> &blockTable,
                                       const c10::optional<at::Tensor> &mask,
                                       const c10::optional<at::Tensor>& indices) {

    auto&& maxSeqLen_ = maxSeqLen.value_or(0);
    auto&& sink_ = sink.value_or(0);
    auto&& recent_ = recent.value_or(0);
    auto&& supportOffload_ = supportOffload.value_or(0);

    at::Tensor out = convert_hamming_dist_top_k_output(hashq, hashkCache, indices);
    EXEC_NPU_CMD(aclnnHammingDistTopK, hashq, hashkCache, topN, seqLen, chunkSize, blockTable, indices, hashkCacheRope, mask, maxSeqLen_, sink_, recent_, supportOffload_, out);
    return out;
}

at::Tensor npu_reshape_and_cache_bnsd(const at::Tensor& hashq,
                                           const at::Tensor& hashkCache,
                                           const at::Tensor& slotMapping,
                                           const at::Tensor& seqLen,
                                           const at::Tensor& hashkCacheOut) {
    EXEC_NPU_CMD(aclnnReshapeAndCacheBnsd, hashq, hashkCache, slotMapping, seqLen, hashkCacheOut);
    return hashkCacheOut;
}

at::Tensor npu_sign_bits_pack(const at::Tensor& input,
                                   const int64_t size) {
    int64_t ySize = (input.size(0) + 7) / 8;
    int64_t outDim = 0;
    if (size != 0) {
        outDim = ySize / size;
    }

    at::Tensor out = torch::empty({size, outDim}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    EXEC_NPU_CMD(aclnnSignBitsPack, input, size, out);
    return out;
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

void device_print(c10::string_view msg)
{
    auto payload = std::make_unique<DevicePrintPayload>();
    payload->message = std::string(msg);
    enqueue_device_print(std::move(payload), c10_npu::getCurrentNPUStream().stream());
}

void device_print(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.defined(), "tensor must be defined");
    TORCH_CHECK(
        tensor.device().is_cpu() ||
            tensor.device().type() == c10::DeviceType::PrivateUse1,
        "device_print only supports CPU and NPU tensors, but got device ",
        tensor.device());

    auto payload = std::make_unique<DevicePrintPayload>();
    if (tensor.device().is_cpu()) {
        payload->host_tensor_snapshot = tensor.contiguous().clone();
        enqueue_device_print(std::move(payload),
                             c10_npu::getCurrentNPUStream().stream());
        return;
    }

    const c10_npu::OptionalNPUGuard npu_guard(tensor.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at::Tensor contiguous_tensor = tensor.contiguous();
    payload->host_tensor_snapshot = at::empty_like(
        contiguous_tensor,
        contiguous_tensor.options().device(at::kCPU).pinned_memory(true));

    const size_t num_bytes = contiguous_tensor.numel() *
                             contiguous_tensor.element_size();
    const aclError memcpy_ret = aclrtMemcpyAsync(
        payload->host_tensor_snapshot.data_ptr(), num_bytes,
        contiguous_tensor.data_ptr(), num_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    TORCH_CHECK(memcpy_ret == ACL_SUCCESS,
                "aclrtMemcpyAsync failed, error code: ", memcpy_ret);

    // The D2H copy and host callback are queued on the same stream so the
    // callback prints only after the host snapshot is ready.
    enqueue_device_print(std::move(payload), stream);
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
    at::Tensor out_input_ids = at::zeros({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_positions = at::zeros({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_is_rejected_token_mask = at::zeros({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_is_masked_token_mask = at::zeros({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_new_token_indices = at::zeros({num_reqs * num_padding_slots_per_request}, at::dtype(at::kInt).device(device));
    at::Tensor out_hidden_state_mapping = at::zeros({total_input_tokens}, at::dtype(at::kInt).device(device));

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
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& query_start_loc_opt,
    const c10::optional<at::Tensor>& cache_indices_opt,
    const c10::optional<at::Tensor>& initial_state_mode_opt,
    const c10::optional<at::Tensor>& num_accepted_tokens_opt,
    int64_t  activation_mode,
    int64_t  pad_slot_id,
    int64_t  run_mode)
{
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& tid2eid_opt,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag)
{

    TORCH_CHECK(x.dim() == 2, "x must be 2D, but got dim=", x.dim());
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "x dtype must be float16/float32/bfloat16, but got ", x.scalar_type());

    TORCH_CHECK(k > 0, "k must be > 0, but got k=", k);
    TORCH_CHECK(k_group >= 1, "k_group must be >= 1, but got k_group=", k_group);
    TORCH_CHECK(group_count >= 1, "group_count must be >= 1, but got group_count=", group_count);

    TORCH_CHECK(group_select_mode == 0 || group_select_mode == 1,
                "group_select_mode must be 0 or 1, but got ", group_select_mode);
    TORCH_CHECK(renorm == 0,
                "renorm can only be 0 currently, but got ", renorm);
    TORCH_CHECK(norm_type == 0 || norm_type == 1 || norm_type ==2,
                "norm_type must be 0 (softmax) or 1 (sigmoid) or 2 (softplus), but got ", norm_type);

    TORCH_CHECK(eps > 0.0, "eps must be > 0, but got ", eps);
    TORCH_CHECK(routed_scaling_factor > 0.0,
                "routed_scaling_factor must be > 0, but got ", routed_scaling_factor);

    const auto sizes = x.sizes();
    const int64_t rows = sizes[0];
    const int64_t expert_num = sizes[1];

    TORCH_CHECK(expert_num > 0, "expert_num must be > 0");
    TORCH_CHECK(expert_num <= 2048,
                "expert_num (E) must be <= 2048, but got ", expert_num);

    if (bias_opt.has_value() && bias_opt->defined()) {
        const auto& bias = *bias_opt;
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D, but got dim=", bias.dim());
        TORCH_CHECK(bias.size(0) == expert_num,
                    "bias.size(0) must equal expert_num. bias.size(0)=",
                    bias.size(0), ", expert_num=", expert_num);
        TORCH_CHECK(bias.scalar_type() == x.scalar_type(),
                    "bias dtype must equal x dtype. x=", x.scalar_type(),
                    ", bias=", bias.scalar_type());
    }

    if (input_ids_opt.has_value() && input_ids_opt->defined()) {
        const auto& input_ids = *input_ids_opt;
        TORCH_CHECK(input_ids.scalar_type() == at::kInt || input_ids.scalar_type() == at::kLong,
                    "input_ids dtype must be int32 or int64, but got ", input_ids.scalar_type());
        TORCH_CHECK(input_ids.numel() == rows,
                    "input_ids.numel() must equal x.size(0). input_ids.numel()=",
                    input_ids.numel(), ", rows=", rows);
    }

    if (tid2eid_opt.has_value() && tid2eid_opt->defined()) {
        const auto& tid2eid = *tid2eid_opt;
        TORCH_CHECK(tid2eid.scalar_type() == at::kInt || tid2eid.scalar_type() == at::kLong,
                    "tid2eid dtype must be int32 or int64, but got ", tid2eid.scalar_type());
        TORCH_CHECK(tid2eid.dim() >= 1, "tid2eid must have dim>=1, but got dim=", tid2eid.dim());
    }

    const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor& input_ids = c10::value_or_else(input_ids_opt, [] { return at::Tensor(); });
    const at::Tensor& tid2eid = c10::value_or_else(tid2eid_opt, [] { return at::Tensor(); });

    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnMoeGatingTopKHash,
                 x,
                 bias,
                 input_ids,
                 tid2eid,
                 k,
                 k_group,
                 group_count,
                 routed_scaling_factor,
                 eps,
                 group_select_mode,
                 renorm,
                 norm_type,
                 out_flag,
                 y,
                 expert_idx,
                 out);

    return {y, expert_idx, out};
}

std::vector<bool> is_contiguous_axes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();

    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }


    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguous_stride[i]);
    }
    return result;
}

std::tuple<at::Tensor> construct_compressor_output_tensor(const at::Tensor &x, const at::Tensor &norm_weight,
                                                          const at::Tensor &rope_sin, int64_t cmp_ratio, int64_t coff)
{
    constexpr int DIM_3 = 3;
    auto x_dim = x.dim();
    at::SmallVector<int64_t, 8> cmp_kv_size;
    at::Tensor cmp_kv;
    auto cmp_s = 0;
    if (x_dim == DIM_3) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    } else {
        cmp_s = rope_sin.size(0);
        cmp_kv_size = {cmp_s, norm_weight.size(0)};
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor>(cmp_kv);
}


std::tuple<at::Tensor> compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                                  at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                                  const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                                  const c10::optional<at::Tensor> &state_block_table,
                                  const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                                  const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio,
                                  int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode)
{
    constexpr int CONTINUOUS = 1;
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;
    constexpr int32_t DIM_3 = 3;
    constexpr int32_t VALUE_0 = 0;
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_2 || x_dim == DIM_3, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(norm_weight.defined(), "Check norm_weight != nullptr failed");
    auto norm_weight_dim = norm_weight.dim();
    TORCH_CHECK(norm_weight_dim == DIM_1, "norm_weight dim num[", norm_weight_dim, "] should be 1");

    TORCH_CHECK(rope_sin.defined(), "Check rope_sin != nullptr failed");
    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == x_dim, "rope_sin dim num[", rope_sin_dim, "] should be equal to x dim num[", x_dim,
                "]");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");

    std::tuple<at::Tensor> output = construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);
    at::Tensor cmp_kv = std::get<0>(output);

    auto state_cache_dim = state_cache.dim();
    TORCH_CHECK(state_cache_dim == DIM_3, "state_cache dim num[", state_cache_dim, "] should be 3");
    auto contiguous_axes_result = is_contiguous_axes(state_cache);
    // if (cache_mode == CONTINUOUS) {
    //     TORCH_CHECK(contiguous_axes_result[0] && contiguous_axes_result[1] && contiguous_axes_result[2],
    //                 "when cache_mode == ", cache_mode, ", state_cache must be contiguous on all axes");
    // }
    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    EXEC_NPU_CMD(aclnnCompressor, x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                    state_block_table, cu_seqlens, seqused, start_pos, rope_head_dim, cmp_ratio, coff, norm_eps,
                    rotary_mode, cache_mode, state_cache_stride_dim0, cmp_kv);

    return std::tuple<at::Tensor>(cmp_kv);
}

void check_compressor_metadata_common(
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &cu_seqlens,
    const at::Tensor &start_pos, const at::Tensor &kv_block_table, int64_t kv_block_size,
    int64_t slot_mapping_format, int64_t compress_ratio, int64_t num_reqs_actual)
{
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t VALUE_0 = 0;

    TORCH_CHECK(rope_cos.defined() && rope_sin.defined(), "rope_cos and rope_sin should be defined");
    TORCH_CHECK(rope_cos.dim() == DIM_2 && rope_sin.dim() == DIM_2,
                "rope_cos and rope_sin should be 2D tensors");
    TORCH_CHECK(rope_cos.scalar_type() == rope_sin.scalar_type(),
                "rope_cos and rope_sin should have same dtype");
    TORCH_CHECK(rope_cos.size(0) == rope_sin.size(0) && rope_cos.size(1) == rope_sin.size(1),
                "rope_cos and rope_sin should have same shape");
    TORCH_CHECK(rope_cos.size(0) > VALUE_0 && rope_cos.size(1) > VALUE_0,
                "rope_cos shape should be non-empty");
    TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.dim() == 1, "cu_seqlens should be a 1D tensor");
    TORCH_CHECK(start_pos.defined() && start_pos.dim() == 1, "start_pos should be a 1D tensor");
    TORCH_CHECK(kv_block_table.defined() && kv_block_table.dim() == DIM_2, "kv_block_table should be a 2D tensor");
    TORCH_CHECK(kv_block_size > VALUE_0, "kv_block_size should be greater than 0");
    TORCH_CHECK(compress_ratio > VALUE_0, "compress_ratio should be greater than 0");
    TORCH_CHECK(slot_mapping_format == DSA_SLOT_MAPPING_BLOCK_OFFSET || slot_mapping_format == DSA_SLOT_MAPPING_FLAT,
                "slot_mapping_format should be 1(flat) or 2(block_offset), but got ", slot_mapping_format);
    TORCH_CHECK(num_reqs_actual > VALUE_0, "num_reqs_actual should be greater than 0");
    TORCH_CHECK(cu_seqlens.size(0) > num_reqs_actual,
                "cu_seqlens dim0 should be greater than num_reqs_actual");
    TORCH_CHECK(start_pos.size(0) >= num_reqs_actual,
                "start_pos dim0 should be greater than or equal to num_reqs_actual");
    TORCH_CHECK(kv_block_table.size(0) >= num_reqs_actual,
                "kv_block_table dim0 should be greater than or equal to num_reqs_actual");
}

void check_compressor_metadata_outputs(
    const at::Tensor &rope_cos, const at::Tensor &compress_cos, const at::Tensor &compress_sin,
    const at::Tensor &slot_mapping, int64_t slot_mapping_format)
{
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t VALUE_0 = 0;

    TORCH_CHECK(compress_cos.defined() && compress_sin.defined() && slot_mapping.defined(),
                "compress_cos, compress_sin, and slot_mapping should be defined");
    TORCH_CHECK(compress_cos.dim() >= DIM_2, "compress_cos dim num should be at least 2");
    TORCH_CHECK(compress_sin.dim() == compress_cos.dim(), "compress_cos and compress_sin should have same dim num");
    TORCH_CHECK(compress_cos.size(0) > VALUE_0, "compress_cos dim0 should be greater than 0");
    TORCH_CHECK(compress_cos.size(compress_cos.dim() - 1) == rope_cos.size(1),
                "compress_cos last dim should match rope dim");
    for (int64_t dim_idx = 0; dim_idx < compress_cos.dim(); ++dim_idx) {
        TORCH_CHECK(compress_sin.size(dim_idx) == compress_cos.size(dim_idx),
                    "compress_cos and compress_sin should have same shape");
    }
    TORCH_CHECK(compress_cos.scalar_type() == rope_cos.scalar_type() &&
                    compress_sin.scalar_type() == rope_cos.scalar_type(),
                "compress outputs should have same dtype as rope_cos");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping dtype should be int32");
    if (slot_mapping_format == DSA_SLOT_MAPPING_BLOCK_OFFSET) {
        TORCH_CHECK(slot_mapping.dim() == DIM_2 && slot_mapping.size(0) == compress_cos.size(0) &&
                        slot_mapping.size(1) == DIM_2,
                    "block_offset slot_mapping should have shape [num_rows, 2]");
    } else {
        TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == compress_cos.size(0),
                    "flat slot_mapping should have shape [num_rows]");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compressor_metadata(
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &cu_seqlens,
    const at::Tensor &start_pos, const at::Tensor &kv_block_table, int64_t kv_block_size,
    int64_t slot_mapping_format, int64_t compress_ratio, int64_t num_compressed_tokens, int64_t num_reqs_actual)
{
    constexpr int64_t VALUE_0 = 0;

    check_compressor_metadata_common(
        rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table, kv_block_size, slot_mapping_format, compress_ratio,
        num_reqs_actual);
    TORCH_CHECK(num_compressed_tokens > VALUE_0, "num_compressed_tokens should be greater than 0");

    at::SmallVector<int64_t, 4> rope_output_size = {num_compressed_tokens, 1, 1, rope_cos.size(1)};
    at::Tensor compress_cos = at::empty(rope_output_size, rope_cos.options());
    at::Tensor compress_sin = at::empty(rope_output_size, rope_sin.options());

    at::SmallVector<int64_t, 2> slot_mapping_size;
    if (slot_mapping_format == DSA_SLOT_MAPPING_BLOCK_OFFSET) {
        slot_mapping_size = {num_compressed_tokens, 2};
    } else {
        slot_mapping_size = {num_compressed_tokens};
    }
    at::Tensor slot_mapping = at::empty(slot_mapping_size, kv_block_table.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnCompressorMetadata, rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table,
                 kv_block_size, slot_mapping_format, compress_ratio, num_reqs_actual, compress_cos, compress_sin,
                 slot_mapping);
    return std::make_tuple(compress_cos, compress_sin, slot_mapping);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compressor_metadata_out(
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &cu_seqlens,
    const at::Tensor &start_pos, const at::Tensor &kv_block_table, int64_t kv_block_size,
    int64_t slot_mapping_format, int64_t compress_ratio, int64_t num_reqs_actual, at::Tensor &compress_cos,
    at::Tensor &compress_sin, at::Tensor &slot_mapping)
{
    check_compressor_metadata_common(
        rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table, kv_block_size, slot_mapping_format, compress_ratio,
        num_reqs_actual);
    check_compressor_metadata_outputs(rope_cos, compress_cos, compress_sin, slot_mapping, slot_mapping_format);

    EXEC_NPU_CMD(aclnnCompressorMetadata, rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table,
                 kv_block_size, slot_mapping_format, compress_ratio, num_reqs_actual, compress_cos, compress_sin,
                 slot_mapping);
    return std::make_tuple(compress_cos, compress_sin, slot_mapping);
}

std::tuple<at::Tensor, at::Tensor> construct_quant_lightning_indexer_output_tensor(const at::Tensor& query, const at::Tensor& key,
                                                           int64_t sparse_count, std::string query_layout_str,
                                                           std::string key_layout_str, bool return_value)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = at::empty(output_size, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_vllm_quant_lightning_indexer_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &metadata,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_value)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer_output = construct_quant_lightning_indexer_output_tensor(
            query, key, sparse_count, query_layout_str, key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(quant_lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(quant_lightning_indexer_output);
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());
    int64_t stride = key.stride(0);
    int64_t scale_stride = key_dequant_scale.stride(0);

    if (key_layout_str == "PA_BSND") {
        auto contiguous_axes_result_key = is_contiguous_axes(key);
        TORCH_CHECK(contiguous_axes_result_key[1] && contiguous_axes_result_key[2],
                    "key must be contiguous on all axes except axis 0");
        auto contiguous_axes_result_key_scale = is_contiguous_axes(key_dequant_scale);
        TORCH_CHECK(contiguous_axes_result_key_scale[1] && contiguous_axes_result_key_scale[2],
                    "key_dequant_scale must be contiguous on all axes except axis 0");
    }

    EXEC_NPU_CMD(aclnnVllmQuantLightningIndexer, query,
        key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key,
        block_table, metadata, query_quant_mode, key_quant_mode, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
        pre_tokens, next_tokens, cmp_ratio, return_value, stride, scale_stride, sparse_indices_out, sparse_values_out);


    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> construct_output_tensor(const at::Tensor &q, std::string layout,
    bool return_softmax_lse)
{
    for (size_t i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0,
            "All values within query's shape should be greater "
            "than 0, but shape[",
            i,
            "] is ",
            q.size(i));
    }
    at::Tensor output = at::empty(q.sizes(), q.options().dtype(q.dtype()));
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        std::vector<int64_t> lse_sizes(q.sizes().begin(), q.sizes().end());
        lse_sizes.back() = 1;
        softmax_lse = at::empty(lse_sizes, q.options().dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv_npu(const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    std::tuple<at::Tensor, at::Tensor> output = construct_output_tensor(q, layout_q_str, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(output);
    at::Tensor softmax_lse = std::get<1>(output);
    int64_t ori_kv_stride = 0;
    int64_t cmp_kv_stride = 0;
    if (ori_kv.has_value()){
        const at::Tensor& tmp_kv = *ori_kv;
        ori_kv_stride = tmp_kv.stride(0);
    }
    if (cmp_kv.has_value()){
        const at::Tensor& tmp_kv = *cmp_kv;
        cmp_kv_stride = tmp_kv.stride(0);
    }

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    EXEC_NPU_CMD(aclnnSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv, sinks,
        metadata, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_kv_stride, cmp_kv_stride, ori_win_left, ori_win_right, layout_q_ptr,
        layout_kv_ptr, return_softmax_lse, attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}

auto get_valid_tensor = [](const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_sparse_attn_sharedkv_metadata_npu(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    }
    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(aclnnSparseAttnSharedkvMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val, cu_seqlens_cmp_kv_val, seqused_q_val,
                    seqused_kv_val, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk,
                    cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr,
                    layout_kv_ptr, has_ori_kv, has_cmp_kv, output);
    return output;
}

at::Tensor npu_vllm_quant_lightning_indexer_metadata_npu(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, const c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (actual_seq_lengths_query.has_value()) {
        output_device = actual_seq_lengths_query.value().device();
    } else if (actual_seq_lengths_key.has_value()) {
        output_device = actual_seq_lengths_key.value().device();
    }

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto actual_seq_lengths_query_val = get_valid_tensor(actual_seq_lengths_query, output_device);
    auto actual_seq_lengths_key_val = get_valid_tensor(actual_seq_lengths_key, output_device);

    std::string layout_query_str = std::string(layout_query);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    std::string layout_key_str = std::string(layout_key);
    char *layout_key_ptr = const_cast<char *>(layout_key_str.c_str());

    EXEC_NPU_CMD(aclnnVllmQuantLightningIndexerMetadata, actual_seq_lengths_query_val, actual_seq_lengths_key_val,
                    num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, batch_size,
                    max_seqlen_q, max_seqlen_k, layout_query_ptr, layout_key_ptr, sparse_count,
                    sparse_mode, pre_tokens, next_tokens, cmp_ratio, output);

    return output;
}

at::Tensor construct_hc_post_output_tensor(const at::Tensor& residual)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    at::SmallVector<int64_t, SIZE> output_size = {residual.size(DIM_0), residual.size(DIM_1), residual.size(DIM_2), residual.size(DIM_3)};
    at::Tensor out = at::empty(output_size, residual.options().dtype(residual.dtype()));
    return out;
}

// step1，工具函数，检查输入shape
void check_hc_post_shape_and_dtype(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post, const at::Tensor& com) {
    // check x shape: [b, s, d]
    TORCH_CHECK(x.dim() == 3, "Input tensor x's dim num should be 3, actual ", x.dim(), ".");
    for (size_t i = 0; i < 3; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is :", x.size(i), ".");
    }
    auto batch = x.size(0);
    auto sequence = x.size(1);
    auto d = x.size(2);
    // check residual: [b, s, hc, d]
    TORCH_CHECK(residual.dim() == 4, "Input tensor residual's dim num should be 4, actual ", residual.dim(), ".");
    auto hc = residual.size(2);
    TORCH_CHECK(hc > 0, "The hc of residual should be positive, actual ", hc, ".");
    TORCH_CHECK(residual.size(0) == batch, "The residual.shape[0] should be batch, actual residual.shape[0] is ", residual.size(0), ", batch is ", batch, ".");
    TORCH_CHECK(residual.size(1) == sequence, "The residual.shape[1] should be sequence, actual residual.shape[1] is ", residual.size(1), ", sequence is ", sequence, ".");
    TORCH_CHECK(residual.size(3) == d, "The residual.shape[3] should be d, actual residual.shape[3] is ", residual.size(3), ", d is ", d, ".");
    // check post [b, s, hc]
    TORCH_CHECK(post.dim() == 3, "Input tensor post's dim num should be 3, actual ", post.dim(), ".");
    TORCH_CHECK(post.size(0) == batch, "The post.shape[0] should be batch, actual post.shape[0] is ", post.size(0), ", batch is ", batch, ".");
    TORCH_CHECK(post.size(1) == sequence, "The post.shape[1] should be sequence, actual post.shape[1] is ", post.size(1), ", sequence is ", sequence, ".");
    TORCH_CHECK(post.size(2) == hc, "The post.shape[2] should be hc, actual post.shape[2] is ", post.size(2), ", hc is ", hc, ".");
    // check com: [b, s, hc, hc]
    TORCH_CHECK(com.dim() == 4, "Input tensor com's dim num should be 4, actual ", com.dim(), ".");
    TORCH_CHECK(com.size(0) == batch, "The com.shape[0] should be batch, actual com.shape[0] is ", com.size(0), ", batch is ", batch, ".");
    TORCH_CHECK(com.size(1) == sequence, "The com.shape[1] should be sequence, actual com.shape[1] is ", com.size(1), ", sequence is ", sequence, ".");
    TORCH_CHECK(com.size(2) == hc, "The com.shape[2] should be hc, actual com.shape[2] is ", com.size(2), ", hc is ", hc, ".");
    TORCH_CHECK(com.size(3) == hc, "The com.shape[3] should be hc, actual com.shape[3] is ", com.size(3), ", hc is ", hc, ".");
    // check dtype
    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");
    TORCH_CHECK(residual.dtype() == x.dtype(), "x's dtype should be equal to residual's dtype.");
    TORCH_CHECK(post.dtype() == at::kFloat || post.dtype() == at::kHalf || post.dtype() == at::kBFloat16,
                "post should be FLOAT16, BFLOAT16, or FLOAT32.");
    TORCH_CHECK(com.dtype() == post.dtype(), "com's dtype should be equal to post's dtype.");
}

at::Tensor npu_hc_post_npu(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    check_hc_post_shape_and_dtype(x, residual, post, comb);
    // construct the output tensor
    at::Tensor out = construct_hc_post_output_tensor(residual);
    EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out);
    return out;
}

constexpr int64_t HC_PRE_HC_LIMIT = 4;
constexpr int64_t HC_PRE_D_LIMIT = 4096;
constexpr int64_t HC_PRE_D_LIMIT_EXTEND = 7168;
constexpr int64_t HC_PRE_MIX_HC_LIMIT = 24;

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(const at::Tensor& x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3){
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_rsqrt_output_tensor(const at::Tensor& x, float epsilon=1e-6)
{
    constexpr int64_t SIZE = 8;
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    auto options = x.options();
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (size_t i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

void check_hc_pre_shape_and_dtype(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    int64_t hc_mult)
{
    constexpr int64_t HC_SCALE_SIZE = 3;
    auto x_dims = x.dim();
    TORCH_CHECK(x_dims == 3 || x_dims == 4, "Input tensor x's dim num should be 3 or 4, actual ", x_dims, ".");
    for (auto i = 0; i < x_dims; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is ",
                    x.size(i), ".");
    }

    auto hc = x_dims == 4 ? x.size(2) : x.size(1);
    auto d = x_dims == 4 ? x.size(3) : x.size(2);
    TORCH_CHECK(hc_mult == HC_PRE_HC_LIMIT, "hc_mult only supports ", HC_PRE_HC_LIMIT, ", actual ", hc_mult, ".");
    TORCH_CHECK(hc == HC_PRE_HC_LIMIT, "The hc of x only supports ", HC_PRE_HC_LIMIT, ", actual ", hc, ".");
    TORCH_CHECK(d == HC_PRE_D_LIMIT || d == HC_PRE_D_LIMIT_EXTEND, "The d of x only supports ", HC_PRE_D_LIMIT,
                " or ", HC_PRE_D_LIMIT_EXTEND, ", actual ", d, ".");
    TORCH_CHECK(hc_fn.dim() == 2, "Input tensor hc_fn's dim num should be 2, actual ", hc_fn.dim(), ".");
    TORCH_CHECK(hc_fn.size(0) == HC_PRE_MIX_HC_LIMIT, "The hc_fn.shape[0] only supports ",
                HC_PRE_MIX_HC_LIMIT, ", actual ", hc_fn.size(0), ".");
    TORCH_CHECK(hc_fn.size(1) == hc * d, "The hc_fn.shape[1] should be hc * d, actual hc_fn.shape[1] is ",
                hc_fn.size(1), ", hc is ", hc, ", d is ", d, ".");
    TORCH_CHECK(hc_scale.dim() == 1, "Input tensor hc_scale's dim num should be 1, actual ", hc_scale.dim(), ".");
    TORCH_CHECK(hc_scale.size(0) == HC_SCALE_SIZE, "Input tensor hc_scale's shape should be [", HC_SCALE_SIZE,
                "], actual [", hc_scale.size(0), "].");
    TORCH_CHECK(hc_base.dim() == 1, "Input tensor hc_base's dim num should be 1, actual ", hc_base.dim(), ".");
    TORCH_CHECK(hc_base.size(0) == HC_PRE_MIX_HC_LIMIT, "The hc_base.shape[0] only supports ",
                HC_PRE_MIX_HC_LIMIT, ", actual ", hc_base.size(0), ".");

    TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
    TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
    TORCH_CHECK(hc_scale.dtype() == at::kFloat, "hc_scale's dtype should be FLOAT32.");
    TORCH_CHECK(hc_base.dtype() == at::kFloat, "hc_base's dtype should be FLOAT32.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_hc_pre_composite(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto xDims = x.dim();
    auto rsqrt = construct_hc_pre_rsqrt_output_tensor(x, norm_eps);
    EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt);

    auto original_type = x.dtype();
    at::Tensor x_float = x.to(at::kFloat);
    at::Tensor x_flattened = x_float.flatten(2, -1);
    if (xDims == 3) {
        x_flattened = x_float.flatten(1, -1);
    }
    auto mixes = at::linear(x_flattened, hc_fn);

    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps,
                    y, post, comb_frag);
    y = y.to(original_type);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_hc_pre_fusion(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    EXEC_NPU_CMD(aclnnHcPre, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps,
                 y, post, comb_frag);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    return run_hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x, float epsilon=1e-20)
{
    constexpr int64_t SIZE = 8;
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    auto options = x.options();
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (auto i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

at::Tensor npu_hc_pre_inv_rms_npu(const at::Tensor& x, double epsilon=1e-20)
{
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");

    at::Tensor yOut;
    yOut = construct_hc_pre_inv_rms_output_tensor(x, epsilon);

    EXEC_NPU_CMD(aclnnHcPreInvRms, x, epsilon, yOut);

    return yOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_sinkhorn_output_tensor(const at::Tensor& mixes, const at::Tensor& x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3){
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_sinkhorn_npu(
    const at::Tensor& mixes, const at::Tensor& rsqrt, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    const at::Tensor& x, int64_t hc_mult, int64_t hc_sinkhorn_iters, double hc_eps)
{
    auto output_tensors = construct_hc_pre_sinkhorn_output_tensor(mixes, x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);

    EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps,
                    y, post, comb_frag);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

void inplace_partial_rotary_mul_npu(at::Tensor & x, const at::Tensor &r1, const at::Tensor &r2, c10::string_view rotary_mode, at::IntArrayRef partial_slice)
{
    constexpr int BSND_DIM_NUM = 4;
    static const std::unordered_map<std::string, int> mode_map = {
        {"half", 0},
        {"interleave", 1},
        {"quarter", 2},
        {"interleave-half", 3}
    };
    std::string rotary_mode_str = std::string(rotary_mode);
    auto it = mode_map.find(rotary_mode_str);
    if (it == mode_map.end())
    {
        return;
    }
    auto origin_dim_num = x.dim();
    TORCH_CHECK(origin_dim_num == BSND_DIM_NUM, "Input tensor x's dim num should be 4, actual ", origin_dim_num, ".");
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice);
}

std::tuple<at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_npu(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    constexpr int32_t SIZE = 8;
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(gamma.numel() > 0, "Input tensor gamma should not be empty.");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x.size(-1), "gamma dim are not equal to last dim of x shape.");
    TORCH_CHECK(epsilon > 0, "epsilon should be greater than 0.");
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16, "x should be FLOAT16, BFLOAT16.");

    at::Tensor smooth_scale2{nullptr};
    auto options = x.options();
    at::Tensor y_out = at::empty_like(x, options.dtype(at::kChar));
    at::Tensor y2_out = at::empty({1}, options.dtype(at::kChar));

    c10::SmallVector<int64_t, SIZE> scale_out_shape;
    for (size_t i = 0; i < x.sizes().size() - 1; i++) {
        scale_out_shape.push_back(x.sizes()[i]);
    }
    at::Tensor scale_out = at::empty(scale_out_shape, options.dtype(at::kFloat));
    at::Tensor scale2_out = at::empty_like(scale_out);
    std::array<bool, 2>* output_mask = nullptr;
    int64_t* dst_type = nullptr;

    EXEC_NPU_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth_scale, smooth_scale2, beta, epsilon, output_mask, dst_type,
                 y_out, y2_out, scale_out, scale2_out);

    return std::make_tuple(y_out, scale_out);
}

void indexer_compress_epilog_npu(
    at::Tensor& indexer_compress_cache,
    at::Tensor& indexer_compress_cache_scale,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_mode = 1,
    bool round_scale = true)
{
    EXEC_NPU_CMD(aclnnIndexerCompressEpilog, indexer_compress_cache, indexer_compress_cache_scale, x,
                 slot_mapping, quant_mode, round_scale);
}

void validate_kv_compress_epilog_inputs(
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_compress_cache)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D tensor, but got dimensions: ", x.dim());
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
                "x dimensions must be positive, but got: [", x.size(0), ", ", x.size(1), "]");
    TORCH_CHECK(slot_mapping.dim() == 1,
                "slot_mapping must be 1D tensor, but got dimensions: ", slot_mapping.dim());
    TORCH_CHECK(slot_mapping.size(0) == x.size(0),
                "slot_mapping size must equal x's first dimension, but got slot_mapping_size=",
                slot_mapping.size(0), ", x.dim(0)=", x.size(0));
    if (kv_compress_cache.dim() == 4) {
        TORCH_CHECK(kv_compress_cache.size(2) == 1,
                    "kv_compress_cache 4D tensor requires headnum (dim 2) == 1, but got ",
                    kv_compress_cache.size(2));
    }
    TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be BF16, but got ", x.dtype());
    TORCH_CHECK(slot_mapping.dtype() == at::kInt || slot_mapping.dtype() == at::kLong,
                "slot_mapping must be INT32 or INT64, but got ", slot_mapping.dtype());
    TORCH_CHECK(kv_compress_cache.dtype() == at::ScalarType::Float8_e5m2 ||
                kv_compress_cache.dtype() == at::ScalarType::Float8_e4m3fn,
                "kv_compress_cache must be FP8_E5M2 or FP8_E4M3, but got ", kv_compress_cache.dtype());
}

void kv_compress_epilog_npu(
    at::Tensor& kv_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_group_size,
    int64_t quant_mode,
    bool round_scale_flag,
    int64_t layout)
{
    validate_kv_compress_epilog_inputs(x, slot_mapping, kv_compress_cache);

    at::Tensor cache = kv_compress_cache;
    if (cache.dim() == 4) {
        cache = cache.squeeze(2);
    }

    int64_t round_scale = round_scale_flag ? 1 : 0;
    int64_t cache_stride = cache.stride(0);
    EXEC_NPU_CMD(aclnnKvCompressEpilog, cache, x, slot_mapping, quant_group_size, quant_mode, round_scale,
                 layout, cache_stride);
}

std::tuple<at::Tensor, at::Tensor> npu_kv_quant_sparse_attn_sharedkv_npu(
    const at::Tensor& q,
    int64_t kv_quant_mode,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    int64_t tile_size,
    int64_t rope_head_dim,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    auto output = construct_output_tensor(q, layout_q_str, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(output);
    at::Tensor softmax_lse = std::get<1>(output);

    char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
    char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());
    int64_t ori_kv_stride0 = 0;
    int64_t cmp_kv_stride0 = 0;
    if (ori_kv.has_value() && ori_kv.value().defined()) {
        ori_kv_stride0 = ori_kv.value().stride(0);
    }
    if (cmp_kv.has_value() && cmp_kv.value().defined()) {
        cmp_kv_stride0 = cmp_kv.value().stride(0);
    }

    EXEC_NPU_CMD(aclnnKvQuantSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
                 ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                 seqused_q, seqused_kv, sinks, metadata, kv_quant_mode, tile_size, rope_head_dim,
                 softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                 layout_q_ptr, layout_kv_ptr, ori_kv_stride0, cmp_kv_stride0, return_softmax_lse,
                 attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}

at::Tensor npu_kv_quant_sparse_attn_sharedkv_metadata_npu(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    int64_t kv_quant_mode,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t tile_size,
    int64_t rope_head_dim,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    }
    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
    char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());

    EXEC_NPU_CMD(aclnnKvQuantSparseAttnSharedkvMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val,
                 cu_seqlens_cmp_kv_val, seqused_q_val, seqused_kv_val, num_heads_q, num_heads_kv,
                 head_dim, batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk, kv_quant_mode,
                 tile_size, rope_head_dim, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
                 ori_win_right, layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv, output);
    return output;
}

int64_t get_type_code(at::ScalarType dst_type)
{
    switch (dst_type) {
        case at::ScalarType::Float8_e5m2:
            return 35;
        case at::ScalarType::Float8_e4m3fn:
            return 36;
        case at::ScalarType::Half:
            return 1;
        case at::ScalarType::BFloat16:
            return 27;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dst_type);
    }
    return 0;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_swiglu_group_quant_output_tensor(
    const at::Tensor& x,
    int64_t dst_type,
    int64_t quant_mode,
    bool ue8m0_scale)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t SWIGLU_FACTOR = 2;
    constexpr int64_t PER_BLOCK_FP16 = 128;
    constexpr int64_t PER_MX_FP16 = 32;
    constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;
    constexpr int64_t GROUP_QUANT = 1;
    constexpr int64_t MX_QUANT = 2;
    constexpr int64_t FP8_QUANT = 3;

    at::SmallVector<int64_t, SIZE> y_size(x.sizes().begin(), x.sizes().end());
    for (size_t i = 0; i < x.sizes().size(); i++) {
        TORCH_CHECK(x.size(i) >= 0, "All values within x's shape should be non-negative, but shape[",
                    i, "] is ", x.size(i));
    }
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16 or BFLOAT16.");
    int64_t x_last_dim = x.sizes().back();
    TORCH_CHECK(quant_mode == GROUP_QUANT || quant_mode == MX_QUANT || quant_mode == FP8_QUANT,
                "Unsupported quant mode, only support ", GROUP_QUANT, " or ", MX_QUANT, " or ", FP8_QUANT, ".");
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        TORCH_CHECK(x_last_dim % 256 == 0,
                    "In group quant, the last dim of x should be divisible by 256, actual ", x_last_dim, ".");
    } else {
        TORCH_CHECK(x_last_dim % 128 == 0,
                    "In mx quant, the last dim of x should be divisible by 128, actual ", x_last_dim, ".");
    }

    y_size.back() = y_size.back() / SWIGLU_FACTOR;
    int64_t y_last_dim = y_size.back();
    auto y_dtype = dst_type == 35 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));

    at::SmallVector<int64_t, SIZE> scale_size(y_size.begin(), y_size.end());
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        scale_size.back() = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
    } else if (quant_mode == MX_QUANT) {
        int64_t scale_last_dim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_last_dim = (scale_last_dim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_size.back() = scale_last_dim;
        scale_size.push_back(MX_SCALE_ALIGN_FACTOR);
    }

    auto scale_type = at::kFloat;
    if (quant_mode == MX_QUANT || (quant_mode == FP8_QUANT && ue8m0_scale)) {
        scale_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale = at::empty(scale_size, x.options().dtype(scale_type));
    at::Tensor y_origin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swiglu_group_quant_npu(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& topk_weight,
    const c10::optional<at::Tensor>& group_index,
    at::ScalarType dst_type = at::ScalarType::Float8_e4m3fn,
    int64_t quant_mode = 1,
    int64_t group_size = 128,
    bool round_scale = false,
    bool ue8m0_scale = false,
    bool output_origin = false,
    int64_t group_list_type = 0,
    double clamp_value = 0.0)
{
    int64_t dst_type_code = get_type_code(dst_type);
    auto output_tensors = construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode, ue8m0_scale);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor scale = std::get<1>(output_tensors);
    at::Tensor y_origin = std::get<2>(output_tensors);

    EXEC_NPU_CMD(aclnnSwigluGroupQuant, x, topk_weight, group_index, dst_type_code, quant_mode, group_size,
                 round_scale, ue8m0_scale, output_origin, group_list_type, clamp_value, y, scale, y_origin);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

std::tuple<at::Tensor, at::Tensor> construct_load_index_kv_cache_output_tensor(
    const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping)
{
    constexpr int64_t KV_LAST_DIM = 128;
    int64_t n = slot_mapping.size(0);

    at::Tensor kv = at::empty({n, KV_LAST_DIM}, kv_cache.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor kv_scale = at::empty({n}, kv_cache.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
}

std::tuple<at::Tensor, at::Tensor> npu_load_index_kv_cache_npu(
    const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping)
{
    auto output_tensors = construct_load_index_kv_cache_output_tensor(kv_cache, slot_mapping);
    at::Tensor kv = std::get<0>(output_tensors);
    at::Tensor kv_scale = std::get<1>(output_tensors);

    int64_t kv_cache_stride = kv_cache.stride(0);
    EXEC_NPU_CMD(aclnnLoadIndexKvCache, kv_cache, slot_mapping, kv_cache_stride, kv, kv_scale);

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
}

void indexer_compress_epilog_v2_npu(
    at::Tensor& indexer_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t layout = 2)
{
    int64_t indexer_compress_cache_stride = indexer_compress_cache.stride(0);
    EXEC_NPU_CMD(aclnnIndexerCompressEpilogV2, indexer_compress_cache, x, slot_mapping, layout,
                 indexer_compress_cache_stride);
}

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode,
    int64_t swiglu_mode,
    double clamp_limit,
    double glu_alpha,
    double glu_bias)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1");
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1, but got ", quant_mode);
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode);
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit >= 0.0, "clamp_limit should be positive finite");
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite");
    TORCH_CHECK(std::isfinite(glu_bias), "glu_bias should be finite");
    TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even");

    c10::SmallVector<int64_t, 8> y_size;
    c10::SmallVector<int64_t, 8> scale_size;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(x.size(x.dim() - 1) / 2);

    at::Tensor y = at::empty(y_size, x.options().dtype(c10::ScalarType::Char));
    at::Tensor scale = at::empty(scale_size, x.options().dtype(c10::ScalarType::Float));

    std::string quant_mode_str = quant_mode == 1 ? "dynamic" : "static";
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& weight_scale_value = c10::value_or_else(weight_scale, [] { return at::Tensor(); });
    const at::Tensor& activation_scale_opt = c10::value_or_else(activation_scale, [] { return at::Tensor(); });
    const at::Tensor& bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });
    const at::Tensor& quant_scale_opt = c10::value_or_else(quant_scale, [] { return at::Tensor(); });
    const at::Tensor& quant_offset_opt = c10::value_or_else(quant_offset, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    static const bool is_v2_available =
        GetOpApiFuncAddr("aclnnDequantSwigluQuantV2") != nullptr &&
        GetOpApiFuncAddr("aclnnDequantSwigluQuantV2GetWorkspaceSize") != nullptr;

    if (swiglu_mode == 0 && !is_v2_available) {
        EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_value, activation_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y, scale);
    } else {
        int64_t dst_type = 2;
        char* round_mode = const_cast<char*>("rint");
        int64_t activate_dim = -1;
        EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_value, activation_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, dst_type, round_mode,
                     activate_dim, swiglu_mode, clamp_limit, glu_alpha, glu_bias, y, scale);
    }

    return std::make_tuple(y, scale);
}

void npu_scatter_nd_update_v2(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    // construct the output tensor
    at::IntArrayRef var_stride = var.strides();
    EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, update, var_stride);
    return;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h(
    const at::Tensor & k,
    const at::Tensor & w,
    const at::Tensor & u,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & gk,
    const c10::optional<at::Tensor> & initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    bool output_final_state_ = output_final_state.has_value() ? output_final_state.value() : false;
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &gk_ = c10::value_or_else(gk, [] { return at::Tensor(); });

    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int K = k_sizes[3];
    int B = k_sizes[0];
    int T = k_sizes[2];
    int HV = u_sizes[1];
    int V = u_sizes[3];

    int NT = 0;
    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();
        NT = chunk_indices_ref.size() / 2;
    } else {
        NT = (T + chunk_size_ - 1) / chunk_size_;
    }

    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::zeros(u.sizes(), u.options());
    at::Tensor final_state_out;
    if (output_final_state_) {
        int N = cu_seqlens.has_value() ? cu_seqlens->size() - 1 : B;
        auto state_options = initial_state.has_value() ? initial_state->options() : h_out.options();
        final_state_out = at::empty({N, HV, K, V}, state_options);
    } else {
        final_state_out = at::empty({1}, k.options());
    }

    bool save_new_value_ = save_new_value.value_or(true);
    bool use_exp2_ = use_exp2.value_or(false);
    bool transpose_state_layout_ = transpose_state_layout.value_or(false);

    EXEC_NPU_CMD(
        aclnnChunkGatedDeltaRuleFwdH,
        k, w, u, g_,
        gk_, initial_state_, output_final_state_, chunk_size_, save_new_value_,
        cu_seqlens, chunk_indices, use_exp2_, transpose_state_layout_,
        h_out, v_new_out, final_state_out
    );

    if (output_final_state_) {
        return std::make_tuple(h_out, v_new_out, final_state_out);
    } else {
        return std::make_tuple(h_out, v_new_out, at::Tensor());
    }
}

at::Tensor chunk_fwd_o(
    const at::Tensor & q,
    const at::Tensor & k,
    const at::Tensor & v,
    const at::Tensor & h,
    double scale,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & g_gamma,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout)
{
    at::Tensor o = at::zeros(v.sizes(), v.options());
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    (void)g_gamma;
    (void)transpose_state_layout;

    EXEC_NPU_CMD(
        aclnnChunkFwdO,
        q, k, v, h, g_,
        cu_seqlens, chunk_indices, scale, chunk_size_,
        o
    );
    return o;
}

std::vector<int64_t> get_npu_storage_shape(const at::Tensor& tensor)
{
    TORCH_CHECK(
        tensor.is_privateuseone(),
        "get_npu_storage_shape only supports NPU tensors, but got device ",
        tensor.device());
    const auto& desc = NPUBridge::GetNpuStorageImplDesc(tensor);
    return std::vector<int64_t>(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
}


} // namespace vllm_ascend

#ifdef ASCEND_PLATFORM_310P
// Pybind on Ascend 310P
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
    ops.def(
        "npu_causal_conv1d_310(Tensor x, "
        "                         Tensor weight, "
        "                         Tensor? bias, "
        "                         Tensor conv_states, "
        "                         Tensor? query_start_loc, "
        "                         Tensor? cache_indices, "
        "                         Tensor? initial_state_mode, "
        "                         Tensor? num_accepted_tokens, "
        "                         int activation_mode, "
        "                         int pad_slot_id, "
        "                         int run_mode) -> (Tensor output)");
    ops.impl("npu_causal_conv1d_310", torch::kPrivateUse1, &vllm_ascend::npu_causal_conv1d_310);

    ops.def(
        "npu_recurrent_gated_delta_rule_310(Tensor query, "
        "                                   Tensor key, "
        "                                   Tensor value, "
        "                                   Tensor beta, "
        "                                   Tensor state, "
        "                                   Tensor actual_seq_lengths, "
        "                                   Tensor ssm_state_indices, "
        "                                   Tensor? g, "
        "                                   Tensor? gk, "
        "                                   Tensor? num_accepted_tokens, "
        "                                   float scale_value=1.0) -> (Tensor output)");
    ops.impl("npu_recurrent_gated_delta_rule_310", torch::kPrivateUse1, &vllm_ascend::npu_recurrent_gated_delta_rule_310);

    ops.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h);

    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o);
}
#else
// Pybind on other platform
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

    ops.def(
        "npu_recurrent_gated_delta_rule(Tensor query, "
        "                               Tensor key, "
        "                               Tensor value, "
        "                               Tensor(a!) state, "
        "                               *, "
        "                               Tensor? beta=None, "
        "                               float? scale=None, "
        "                               Tensor? actual_seq_lengths=None, "
        "                               Tensor? ssm_state_indices=None, "
        "                               Tensor? num_accepted_tokens=None, "
        "                               Tensor? g=None, "
        "                               Tensor? gk=None) -> Tensor");
    ops.impl("npu_recurrent_gated_delta_rule", torch::kPrivateUse1, &vllm_ascend::npu_recurrent_gated_delta_rule);

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
    // Direct kernel custom ops
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
#endif

    // swap_blocks_batch takes CPU tensors (int64 pointer/size arrays), not NPU
    // tensors, so dispatch must be registered on the CPU backend. The function
    // internally submits async memcpy on the current NPU stream.
    ops.def("swap_blocks_batch(Tensor x, Tensor y, Tensor z, int direction) -> ()");
    ops.impl("swap_blocks_batch", torch::kCPU, &vllm_ascend::swap_blocks_batch);
    ops.def("device_print(str msg) -> ()");
    ops.impl("device_print", c10::DispatchKey::CompositeExplicitAutograd,
             static_cast<void (*)(c10::string_view)>(&vllm_ascend::device_print));

    ops.def("device_print_tensor(Tensor tensor) -> ()");
    ops.impl("device_print_tensor", c10::DispatchKey::CompositeExplicitAutograd,
             static_cast<void (*)(const at::Tensor&)>(&vllm_ascend::device_print));

    ops.def("get_npu_storage_shape(Tensor tensor) -> int[]");
    ops.impl("get_npu_storage_shape", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_npu_storage_shape);

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
        "turboquant_pack_kv_for_cache_to_cache(Tensor key, Tensor value, Tensor slot_mapping, "
        "Tensor! key_cache, Tensor! value_cache, int slot_w_k, int slot_w_v) -> ()");
    ops.impl("turboquant_pack_kv_for_cache_to_cache", torch::kPrivateUse1,
             &vllm_ascend::turboquant_pack_kv_for_cache_to_cache);

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
        "                            Tensor? offset=None, float swiglu_limit=0.0) ->"
        "                            (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant);

    ops.def(
        "grouped_matmul_swiglu_quant_weight_nz(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "                                      Tensor group_list, *, Tensor? bias=None,"
        "                                      Tensor? offset=None, float swiglu_limit=0.0) -> "
        "                                      (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant_weight_nz", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz);

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
        "                                                  Tensor? bias=None, Tensor? offset=None, float swiglu_limit=0.0) ->"
        "                                                  (Tensor output, Tensor output_scale, Tensor output_offset)"
    );
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz_tensor_list);

    ops.def(
        "grouped_matmul_swiglu_quant_v2(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,  Tensor group_list,  Tensor? smooth_scale=None,"
        "                                                   Tensor[]? weight_assist_matrix=None, Tensor? bias=None, int? dequant_mode=0, int? dequant_dtype=0, int? quant_mode=0,"
        "                                                 int? quant_dtype=0, bool transpose_weight=False, int group_list_type=0, int[2] tuning_config=[],float swiglu_limit=0.0) ->"
        "                                                  (Tensor output, Tensor output_scale)"
    );
    ops.impl("grouped_matmul_swiglu_quant_v2", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_v2);

    ops.def(
        "npu_lightning_indexer("
            "Tensor query, Tensor key, Tensor weights, "
            "*, "
            "Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_key=None, "
            "Tensor? block_table=None, "
            "str layout_query=\"BSND\", str layout_key=\"BSND\", "
            "int sparse_count=2048, int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, "
            "bool return_value=False"
        ") -> (Tensor sparse_indices, Tensor sparse_values)"
    );
    ops.impl("npu_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer);

    ops.def(
        "npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                           Tensor sparse_indices, float scale_value, *,"
        "                           Tensor? block_table=None, Tensor? actual_seq_lengths_query=None,"
        "                           Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None,"
        "                           Tensor? key_rope=None, int sparse_block_size=1,"
        "                           str layout_query='BSND', str layout_kv='BSND',"
        "                           int sparse_mode=3, int pre_tokens=9223372036854775807,"
        "                           int next_tokens=9223372036854775807, int attention_mode=2,"
        "                           bool return_softmax_lse=False) -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)"
    );
    ops.impl("npu_sparse_flash_attention", torch::kPrivateUse1, &vllm_ascend::npu_sparse_flash_attention);

    ops.def(
        "npu_kv_quant_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                                    Tensor sparse_indices, float scale_value, *,"
        "                                    int key_quant_mode=1, int value_quant_mode=1,"
        "                                    Tensor? key_dequant_scale=None,"
        "                                    Tensor? value_dequant_scale=None,"
        "                                    Tensor? block_table=None,"
        "                                    Tensor? actual_seq_lengths_query=None,"
        "                                    Tensor? actual_seq_lengths_kv=None,"
        "                                    int sparse_block_size=1,"
        "                                    str layout_query='BSND', str layout_kv='BSND',"
        "                                    int sparse_mode=3,"
        "                                    int pre_tokens=9223372036854775807,"
        "                                    int next_tokens=9223372036854775807,"
        "                                    int attention_mode=2,"
        "                                    int quant_scale_repo_mode=1,"
        "                                    int tile_size=128,"
        "                                    int rope_head_dim=64,"
        "                                    bool return_softmax_lse=False)"
        " -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)"
    );
    ops.impl("npu_kv_quant_sparse_flash_attention", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_flash_attention);

    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,"
        "                     Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, Tensor[] bias2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_ffn_combine", torch::kPrivateUse1, &vllm_ascend::dispatch_ffn_combine);

    ops.def("matmul_allreduce_add_rmsnorm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, \
        str groupTp, int tpRankSize, int tpRankId, float epsilon, bool isTransB, bool isGatherAddOut) -> (Tensor output, Tensor add_out)");
    ops.impl("matmul_allreduce_add_rmsnorm", torch::kPrivateUse1, &vllm_ascend::matmul_allreduce_add_rmsnorm);

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

    ops.def(
        "npu_hamming_dist_top_k(Tensor q, Tensor k_comp, Tensor k_comp_rope, Tensor k,"
        "                      Tensor seq_len, Tensor? chunk_size=None,"
        "                      int? max_seq_len=None, int? sink=None, int? recent=None, int? support_offload=None,"
        "                      Tensor? key_block_table=None, Tensor? mask=None, Tensor? indices=None) -> Tensor"
    );
    ops.impl("npu_hamming_dist_top_k", torch::kPrivateUse1, &vllm_ascend::npu_hamming_dist_top_k);

    ops.def(
        "npu_reshape_and_cache_bnsd(Tensor q, Tensor k_comp, Tensor slot_mapping, Tensor seq_len, Tensor k_out) -> Tensor"
    );
    ops.impl("npu_reshape_and_cache_bnsd", torch::kPrivateUse1, &vllm_ascend::npu_reshape_and_cache_bnsd);

    ops.def("npu_sign_bits_pack(Tensor input, int size) -> Tensor");
    ops.impl("npu_sign_bits_pack", torch::kPrivateUse1, &vllm_ascend::npu_sign_bits_pack);

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
        "npu_causal_conv1d_custom(Tensor output, Tensor x, "
        "                         Tensor weight, "
        "                         Tensor conv_state, "
        "                         Tensor? bias_opt, "
        "                         Tensor? query_start_loc_opt, "
        "                         Tensor? cache_indices_opt, "
        "                         Tensor? initial_state_mode_opt, "
        "                         Tensor? num_accepted_tokens_opt, "
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

    ops.def(
        "moe_gating_top_k_hash("
        "Tensor x, "
        "int k, "
        "Tensor? bias=None, "
        "Tensor? input_ids=None, "
        "Tensor? tid2eid=None, "
        "int k_group=1, "
        "int group_count=1, "
        "float routed_scaling_factor=1.0, "
        "float eps=1e-20, "
        "int group_select_mode=0, "
        "int renorm=0, "
        "int norm_type=0, "
        "bool out_flag=False"
        ") -> (Tensor y, Tensor expert_idx, Tensor out)"
        );
    ops.impl("moe_gating_top_k_hash", torch::kPrivateUse1,&vllm_ascend::moe_gating_top_k_hash);

    ops.def(
        "compressor("
            "Tensor x, Tensor wkv, Tensor wgate, "
            "Tensor(a!) state_cache, Tensor ape, Tensor norm_weight, "
            "Tensor rope_sin, Tensor rope_cos, "
            "Tensor? state_block_table, Tensor? cu_seqlens, "
            "Tensor? seqused, Tensor? start_pos, "
            "int rope_head_dim, int cmp_ratio, int coff, "
            "float norm_eps, int rotary_mode, int cache_mode"
        ") -> Tensor"
        );
    ops.impl("compressor", torch::kPrivateUse1, &vllm_ascend::compressor);

    ops.def(
        "compressor_metadata("
            "Tensor rope_cos, Tensor rope_sin, "
            "Tensor cu_seqlens, Tensor start_pos, Tensor kv_block_table, "
            "int kv_block_size, int slot_mapping_format, int compress_ratio, int num_compressed_tokens, "
            "int num_reqs_actual"
        ") -> (Tensor, Tensor, Tensor)"
        );
    ops.impl("compressor_metadata", torch::kPrivateUse1, &vllm_ascend::compressor_metadata);

    ops.def(
        "compressor_metadata_out("
            "Tensor rope_cos, Tensor rope_sin, "
            "Tensor cu_seqlens, Tensor start_pos, Tensor kv_block_table, "
            "int kv_block_size, int slot_mapping_format, int compress_ratio, int num_reqs_actual, "
            "Tensor(a!) compress_cos, Tensor(b!) compress_sin, Tensor(c!) slot_mapping"
        ") -> (Tensor(a!), Tensor(b!), Tensor(c!))"
        );
    ops.impl("compressor_metadata_out", torch::kPrivateUse1, &vllm_ascend::compressor_metadata_out);

    ops.def(
        "npu_vllm_quant_lightning_indexer("
            "Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, "
            "int query_quant_mode=0, int key_quant_mode=0, "
            "Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_key=None, "
            "Tensor? block_table=None, "
            "Tensor? metadata=None, "
            "str layout_query=\"BSND\", str layout_key=\"BSND\", "
            "int sparse_count=2048, int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, "
            "int cmp_ratio=1, bool return_value=False"
        ") -> (Tensor sparse_indices, Tensor sparse_values)"
        );
    ops.impl("npu_vllm_quant_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_vllm_quant_lightning_indexer_npu);

    ops.def(
        "npu_sparse_attn_sharedkv("
            "Tensor q, *, "
            "Tensor? ori_kv=None, "
            "Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, "
            "Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, "
            "Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "Tensor? sinks=None, "
            "Tensor? metadata=None, "
            "float softmax_scale=0, "
            "int cmp_ratio=0, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=128, "
            "int ori_win_right=0, "
            "str layout_q=\"BSND\", "
            "str layout_kv=\"PA_ND\", "
            "bool return_softmax_lse=False"
        ") -> (Tensor out, Tensor softmax_lse)"
        );
    ops.impl("npu_sparse_attn_sharedkv", torch::kPrivateUse1, &vllm_ascend::npu_sparse_attn_sharedkv_npu);

    ops.def(
        "npu_sparse_attn_sharedkv_metadata("
            "int num_heads_q, "
            "int num_heads_kv, "
            "int head_dim, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "int batch_size=0, "
            "int max_seqlen_q=0, "
            "int max_seqlen_kv=0, "
            "int ori_topk=0, "
            "int cmp_topk=0, "
            "int cmp_ratio=4, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=128, "
            "int ori_win_right=0, "
            "str layout_q=\"BSND\", "
            "str layout_kv=\"PA_ND\", "
            "bool has_ori_kv=True, "
            "bool has_cmp_kv=True, "
            "str device=\"npu\""
        ") -> (Tensor metadata)"
        );
    ops.impl("npu_sparse_attn_sharedkv_metadata", torch::kPrivateUse1, &vllm_ascend::npu_sparse_attn_sharedkv_metadata_npu);

    ops.def(
        "npu_vllm_quant_lightning_indexer_metadata("
            "int num_heads_q, "
            "int num_heads_k, "
            "int head_dim, "
            "int query_quant_mode, "
            "int key_quant_mode, "
            "Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_key=None, "
            "int batch_size=0, "
            "int max_seqlen_q=0, "
            "int max_seqlen_k=0, "
            "str layout_query=\"BSND\", "
            "str layout_key=\"BSND\", "
            "int sparse_count=2048, "
            "int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, "
            "int cmp_ratio=1, "
            "str device=\"npu\""
        ") -> (Tensor metadata)"
        );
    ops.impl("npu_vllm_quant_lightning_indexer_metadata", torch::kPrivateUse1, &vllm_ascend::npu_vllm_quant_lightning_indexer_metadata_npu);

    ops.def(
          "npu_hc_post("
            "Tensor x, "
            "Tensor residual, "
            "Tensor post, "
            "Tensor comb"
        ") -> (Tensor out)"
        );
    ops.impl("npu_hc_post", torch::kPrivateUse1, &vllm_ascend::npu_hc_post_npu);

    ops.def(
        "npu_hc_pre("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_npu);

    ops.def(
        "npu_hc_pre_v2("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre_v2", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v2_npu);

    ops.def(
        "npu_hc_pre_inv_rms("
            "Tensor x, float epsilon=1e-20"
        ") -> (Tensor out)"
        );
    ops.impl("npu_hc_pre_inv_rms", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_inv_rms_npu);

    ops.def(
        "npu_hc_pre_sinkhorn("
            "Tensor mixes, Tensor rsqrt, Tensor hc_scale, Tensor hc_base, Tensor x, "
            "int hc_mult, int hc_sinkhorn_iters, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre_sinkhorn", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_sinkhorn_npu);

    ops.def(
        "inplace_partial_rotary_mul("
            "Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice"
        ") -> ()"
    );
    ops.impl("inplace_partial_rotary_mul", torch::kPrivateUse1, &vllm_ascend::inplace_partial_rotary_mul_npu);

    ops.def(
        "npu_rms_norm_dynamic_quant("
            "Tensor x, "
            "Tensor gamma, "
            "Tensor? smooth_scale=None, "
            "Tensor? beta=None, "
            "float epsilon=1e-6"
        ") -> (Tensor y_out, Tensor scale_out)"
        );
    ops.impl("npu_rms_norm_dynamic_quant", torch::kPrivateUse1, &vllm_ascend::npu_rms_norm_dynamic_quant_npu);

    ops.def(
        "indexer_compress_epilog("
            "Tensor(a!) indexer_compress_cache, "
            "Tensor(b!) indexer_compress_cache_scale, "
            "Tensor x, "
            "Tensor slot_mapping, "
            "int quant_mode=1, "
            "bool round_scale=True"
        ") -> ()"
    );
    ops.impl("indexer_compress_epilog", torch::kPrivateUse1, &vllm_ascend::indexer_compress_epilog_npu);

    ops.def(
        "kv_compress_epilog("
            "Tensor(a!) kv_compress_cache, "
            "Tensor x, "
            "Tensor slot_mapping, "
            "int quant_group_size, "
            "int quant_mode, "
            "bool round_scale_flag, "
            "int layout"
        ") -> ()"
    );
    ops.impl("kv_compress_epilog", torch::kPrivateUse1, &vllm_ascend::kv_compress_epilog_npu);

    ops.def(
        "npu_kv_quant_sparse_attn_sharedkv("
            "Tensor q, "
            "int kv_quant_mode, "
            "Tensor? ori_kv=None, "
            "Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, "
            "Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, "
            "Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "Tensor? sinks=None, "
            "Tensor? metadata=None, "
            "int tile_size=0, "
            "int rope_head_dim=0, "
            "float softmax_scale=0.0, "
            "int cmp_ratio=0, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=127, "
            "int ori_win_right=0, "
            "str layout_q='BSND', "
            "str layout_kv='PA_ND', "
            "bool return_softmax_lse=False"
        ") -> (Tensor out, Tensor softmax_lse)"
    );
    ops.impl("npu_kv_quant_sparse_attn_sharedkv", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_attn_sharedkv_npu);

    ops.def(
        "npu_kv_quant_sparse_attn_sharedkv_metadata("
            "int num_heads_q, "
            "int num_heads_kv, "
            "int head_dim, "
            "int kv_quant_mode, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "int batch_size=0, "
            "int max_seqlen_q=0, "
            "int max_seqlen_kv=0, "
            "int ori_topk=0, "
            "int cmp_topk=0, "
            "int tile_size=0, "
            "int rope_head_dim=0, "
            "int cmp_ratio=-1, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=127, "
            "int ori_win_right=0, "
            "str layout_q='BSND', "
            "str layout_kv='PA_ND', "
            "bool has_ori_kv=True, "
            "bool has_cmp_kv=True, "
            "str device='npu'"
        ") -> Tensor"
    );
    ops.impl("npu_kv_quant_sparse_attn_sharedkv_metadata", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_attn_sharedkv_metadata_npu);

    ops.def(
        "npu_swiglu_group_quant(Tensor x, Tensor? topk_weight, Tensor? group_index, "
        "                       ScalarType dst_type=39, "
        "                       int quant_mode=1, int group_size=128, "
        "                       bool round_scale=False, bool ue8m0_scale=False, "
        "                       bool output_origin=False, int group_list_type=0, "
        "                       float clamp_value=0.0) "
        "-> (Tensor y, Tensor scale, Tensor y_origin)");
    ops.impl("npu_swiglu_group_quant", torch::kPrivateUse1, &vllm_ascend::npu_swiglu_group_quant_npu);

    ops.def(
        "npu_load_index_kv_cache("
            "Tensor kv_cache, Tensor slot_mapping"
        ") -> (Tensor out, Tensor out_scale)"
    );
    ops.impl("npu_load_index_kv_cache", torch::kPrivateUse1, &vllm_ascend::npu_load_index_kv_cache_npu);

    ops.def(
        "indexer_compress_epilog_v2("
            "Tensor(a!) indexer_compress_cache, "
            "Tensor x, "
            "Tensor slot_mapping, "
            "int layout=2"
        ") -> ()"
    );
    ops.impl("indexer_compress_epilog_v2", torch::kPrivateUse1,
             &vllm_ascend::indexer_compress_epilog_v2_npu);

    ops.def(
        "npu_dequant_swiglu_quant("
            "Tensor x, *, "
            "Tensor? weight_scale=None, "
            "Tensor? activation_scale=None, "
            "Tensor? bias=None, "
            "Tensor? quant_scale=None, "
            "Tensor? quant_offset=None, "
            "Tensor? group_index=None, "
            "bool activate_left=True, "
            "int quant_mode=0, "
            "int swiglu_mode=0, "
            "float clamp_limit=0.0, "
            "float glu_alpha=1.0, "
            "float glu_bias=0.0"
        ") -> (Tensor y, Tensor scale)"
    );
    ops.impl("npu_dequant_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::npu_dequant_swiglu_quant);

    ops.def(
        "npu_scatter_nd_update_v2("
                "Tensor(a!) var, Tensor indices, Tensor update"
            ") -> ()"
    );
    ops.impl("npu_scatter_nd_update_v2", torch::kPrivateUse1, &vllm_ascend::npu_scatter_nd_update_v2);

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
    // N-gram spec decode
    ops.def(
        "npu_ngram_spec_decode(Tensor(a!) token_ids, Tensor num_tokens_no_spec, "
        "Tensor sampled_token_ids, Tensor discard_request_mask, "
        "int vocab_size, int min_n, int max_n, int k) -> "
        "(Tensor token_ids, Tensor next_token_ids, Tensor draft_token_ids, Tensor num_valid_draft_tokens)"
    );
    ops.impl("npu_ngram_spec_decode", torch::kPrivateUse1,
             &vllm_ascend::npu_ngram_spec_decode);

    ops.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h);

    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o);

    //store_kv_block
     ops.def(
        "store_kv_block_metadata(Tensor slot_mapping_npu, Tensor group_len, Tensor group_key_idx, Tensor group_key_cache_idx, int block_size=0)"
         "-> ()"
     );
    ops.impl("store_kv_block_metadata", torch::kPrivateUse1, &vllm_ascend::store_kv_block_metadata);

    ops.def(
        "store_kv_block(Tensor key_in, Tensor key_cache_in, Tensor group_len, Tensor group_key_idx,Tensor group_key_cache_idx, int block_size=0) -> ()"
    );
    ops.impl("store_kv_block", torch::kPrivateUse1, &vllm_ascend::store_kv_block);
    
    // Fused GDN gating.
    ops.def(
        "npu_fused_gdn_gating(Tensor A_log, "
        "                     Tensor a, "
        "                     Tensor b, "
        "                     Tensor dt_bias, "
        "                     float beta=1.0, "
        "                     float threshold=20.0) -> (Tensor g, Tensor beta_output)");
    ops.impl("npu_fused_gdn_gating", torch::kPrivateUse1, &vllm_ascend::npu_fused_gdn_gating);
}
#endif
