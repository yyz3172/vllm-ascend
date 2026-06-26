/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <cstdlib>
#include <string>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_base_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnn_turboquant_decode_paged8bit.h"

// Test configuration constants - must match kernel expectations
constexpr int HEAD_SIZE = 128;
constexpr int DEFAULT_BLOCK_SIZE = 128;
constexpr int DEFAULT_NUM_KV_HEADS = 2;
constexpr int DEFAULT_NUM_BLOCKS = 6;
constexpr int BITS = 8;
constexpr int CODEBOOK_SIZE = 256;  // 2^8
constexpr int PACKED_BYTES = HEAD_SIZE + 2;  // 130 bytes per packed row

#ifndef VLLM_ASCEND_CUSTOM_OPP_PATH
#define VLLM_ASCEND_CUSTOM_OPP_PATH "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
#endif

void configure_custom_opp_path() {
    const char *custom_opp_path = VLLM_ASCEND_CUSTOM_OPP_PATH;
    if (custom_opp_path == nullptr || custom_opp_path[0] == '\0') {
        return;
    }

    const char *current = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (current == nullptr || current[0] == '\0') {
        setenv("ASCEND_CUSTOM_OPP_PATH", custom_opp_path, 1);
        return;
    }

    std::string current_paths(current);
    std::string custom_path(custom_opp_path);
    if (current_paths.find(custom_path) != std::string::npos) {
        return;
    }

    std::string updated_paths = custom_path + ":" + current_paths;
    setenv("ASCEND_CUSTOM_OPP_PATH", updated_paths.c_str(), 1);
}

int read_positive_int_env(const char *name, int default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }

    errno = 0;
    char *end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value <= 0 || value > INT_MAX) {
        std::cerr << "[TQ_DECODE_TEST] Invalid " << name << "=" << raw
                  << ", using default " << default_value << std::endl;
        return default_value;
    }

    return static_cast<int>(value);
}

// Helper: Generate random float16 data
void generate_random_float16(std::vector<uint16_t>& data, int size) {
    data.resize(size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < size; i++) {
        float f = dist(rng);
        data[i] = aclFloatToFloat16(f);
    }
}

// Helper: Generate codebook (uniform distribution)
void generate_codebook(std::vector<uint16_t>& codebook) {
    codebook.resize(CODEBOOK_SIZE);
    for (int i = 0; i < CODEBOOK_SIZE; i++) {
        // Generate uniform values in [-2, 2] range
        float val = -2.0f + (4.0f * i) / (CODEBOOK_SIZE - 1);
        codebook[i] = aclFloatToFloat16(val);
    }
}

// Helper: Generate rotation matrix (identity for simplicity)
void generate_rotation(std::vector<uint16_t>& rotation, int head_size) {
    rotation.resize(head_size * head_size);
    for (int i = 0; i < head_size; i++) {
        for (int j = 0; j < head_size; j++) {
            float val = (i == j) ? 1.0f : 0.0f;
            rotation[i * head_size + j] = aclFloatToFloat16(val);
        }
    }
}

// Helper: Quantize float16 to packed bytes (8-bit) with norm
void quantize_to_packed_bytes(
    const std::vector<uint16_t>& input,  // float16 input
    std::vector<uint8_t>& output,        // packed 8-bit output (130 bytes per row)
    const std::vector<uint16_t>& codebook,  // float16 codebook
    int head_size,
    int num_rows) {

    output.resize(num_rows * PACKED_BYTES);

    for (int row = 0; row < num_rows; row++) {
        // Find nearest codebook entry for each element
        for (int j = 0; j < head_size; j++) {
            uint16_t val = input[row * head_size + j];
            float f_val = aclFloat16ToFloat(val);

            // Find nearest centroid
            int idx = 0;
            float min_dist = FLT_MAX;

            for (int c = 0; c < CODEBOOK_SIZE; c++) {
                float cb_val = aclFloat16ToFloat(codebook[c]);
                float dist = std::abs(f_val - cb_val);

                if (dist < min_dist) {
                    min_dist = dist;
                    idx = c;
                }
            }

            // Store 8-bit index in packed output
            output[row * PACKED_BYTES + j] = static_cast<uint8_t>(idx);
        }

        // Store norm (1.0 for simplicity) in the last 2 bytes
        uint16_t norm_val = aclFloatToFloat16(1.0f);
        output[row * PACKED_BYTES + head_size] = static_cast<uint8_t>(norm_val & 0xFF);
        output[row * PACKED_BYTES + head_size + 1] = static_cast<uint8_t>((norm_val >> 8) & 0xFF);
    }
}

// Helper: Dequantize from packed bytes to float16 (CPU golden reference)
void dequantize_from_packed_bytes(
    const std::vector<uint8_t>& input,   // packed 8-bit input
    std::vector<uint16_t>& output,       // float16 output
    const std::vector<uint16_t>& codebook,  // float16 codebook
    int head_size,
    int num_rows) {

    output.resize(num_rows * head_size);

    for (int row = 0; row < num_rows; row++) {
        for (int j = 0; j < head_size; j++) {
            uint8_t idx = input[row * PACKED_BYTES + j];
            output[row * head_size + j] = codebook[idx];
        }
    }
}

int main() {
    std::cout << "[TQ_DECODE_TEST] Starting turboquant_decode_paged8bit test..." << std::endl;
    configure_custom_opp_path();

    const int block_size = read_positive_int_env("TQ_DECODE_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    const int num_kv_heads = read_positive_int_env("TQ_DECODE_NUM_KV_HEADS", DEFAULT_NUM_KV_HEADS);
    const int num_blocks = read_positive_int_env("TQ_DECODE_NUM_BLOCKS", DEFAULT_NUM_BLOCKS);
    const int gather_blocks = std::min(read_positive_int_env("TQ_DECODE_GATHER_BLOCKS", 1), num_blocks);
    const int decode_mode = std::getenv("TQ_DECODE_MODE") == nullptr ? 0 : std::atoi(std::getenv("TQ_DECODE_MODE"));

    std::cout << "[TQ_DECODE_TEST] Config: head_size=" << HEAD_SIZE
              << ", block_size=" << block_size
              << ", num_kv_heads=" << num_kv_heads
              << ", num_blocks=" << num_blocks
              << ", gather_blocks=" << gather_blocks
              << ", mode=" << decode_mode << std::endl;

    // Step 1: Initialize ACL
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[TQ_DECODE_TEST] Failed to initialize ACL" << std::endl;
        return -1;
    }

    // Get device count
    uint32_t device_count = 0;
    ret = aclrtGetDeviceCount(&device_count);
    if (ret != ACL_SUCCESS || device_count == 0) {
        std::cerr << "[TQ_DECODE_TEST] No ACL devices found" << std::endl;
        return -1;
    }

    // Set device
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[TQ_DECODE_TEST] Failed to set device" << std::endl;
        return -1;
    }

    // Create stream
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS || stream == nullptr) {
        std::cerr << "[TQ_DECODE_TEST] Failed to create stream" << std::endl;
        return -1;
    }

    std::cout << "[TQ_DECODE_TEST] ACL initialized, device=0, stream created" << std::endl;

    // Step 2: Generate test data
    int rows_per_block = block_size * num_kv_heads;
    int cache_rows = num_blocks * rows_per_block;
    int output_rows = gather_blocks * rows_per_block;

    std::vector<uint16_t> key_data, value_data;
    std::vector<uint16_t> codebook;
    std::vector<uint16_t> rotation;

    generate_random_float16(key_data, cache_rows * HEAD_SIZE);
    generate_random_float16(value_data, cache_rows * HEAD_SIZE);
    generate_codebook(codebook);
    generate_rotation(rotation, HEAD_SIZE);

    std::cout << "[TQ_DECODE_TEST] Generated test data: "
              << "key=" << key_data.size() << ", value=" << value_data.size()
              << ", codebook=" << codebook.size() << ", rotation=" << rotation.size() << std::endl;

    // Step 3: Quantize data to packed format
    std::vector<uint8_t> key_packed, value_packed;
    quantize_to_packed_bytes(key_data, key_packed, codebook, HEAD_SIZE, cache_rows);
    quantize_to_packed_bytes(value_data, value_packed, codebook, HEAD_SIZE, cache_rows);

    std::cout << "[TQ_DECODE_TEST] Packed data: key=" << key_packed.size()
              << ", value=" << value_packed.size() << std::endl;

    // Step 4: Set up block table and gather indices
    // gather_block_ids maps compact block index -> physical block index
    std::vector<int32_t> gather_block_ids(gather_blocks);
    for (int i = 0; i < gather_blocks; i++) {
        gather_block_ids[i] = i;  // Simple 1:1 mapping for testing
    }

    std::cout << "[TQ_DECODE_TEST] Gather block IDs: " << gather_block_ids.size() << " entries" << std::endl;

    // Step 5: Allocate device memory
    size_t key_packed_size = key_packed.size() * sizeof(uint8_t);
    size_t value_packed_size = value_packed.size() * sizeof(uint8_t);
    size_t gather_size = gather_block_ids.size() * sizeof(int32_t);
    size_t codebook_size = codebook.size() * sizeof(uint16_t);
    size_t rotation_size = rotation.size() * sizeof(uint16_t);

    // Output size: output_rows * HEAD_SIZE * sizeof(float16)
    size_t output_size = output_rows * HEAD_SIZE * sizeof(uint16_t);

    void *d_key_packed = nullptr, *d_value_packed = nullptr;
    void *d_gather = nullptr, *d_codebook = nullptr;
    void *d_rotation = nullptr, *d_key_out = nullptr, *d_value_out = nullptr;
    void *d_workspace = nullptr;

    aclTensor *key_cache_tensor = nullptr, *value_cache_tensor = nullptr;
    aclTensor *gather_tensor = nullptr, *codebook_tensor = nullptr, *rotation_tensor = nullptr;
    aclTensor *key_out_tensor = nullptr, *value_out_tensor = nullptr;

    ret = aclrtMalloc(&d_key_packed, key_packed_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&d_value_packed, value_packed_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&d_gather, gather_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&d_codebook, codebook_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&d_rotation, rotation_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&d_key_out, output_size, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&d_value_out, output_size, ACL_MEM_MALLOC_HUGE_FIRST);

    std::cout << "[TQ_DECODE_TEST] Allocated device memory successfully" << std::endl;

    // Step 6: Copy data to device
    ret = aclrtMemcpy(d_key_packed, key_packed_size, key_packed.data(), key_packed_size, ACL_MEMCPY_HOST_TO_DEVICE);
    ret = aclrtMemcpy(d_value_packed, value_packed_size, value_packed.data(), value_packed_size, ACL_MEMCPY_HOST_TO_DEVICE);
    ret = aclrtMemcpy(d_gather, gather_size, gather_block_ids.data(), gather_size, ACL_MEMCPY_HOST_TO_DEVICE);
    ret = aclrtMemcpy(d_codebook, codebook_size, codebook.data(), codebook_size, ACL_MEMCPY_HOST_TO_DEVICE);
    ret = aclrtMemcpy(d_rotation, rotation_size, rotation.data(), rotation_size, ACL_MEMCPY_HOST_TO_DEVICE);

    std::cout << "[TQ_DECODE_TEST] Copied data to device" << std::endl;

    // Step 7: Create aclTensor objects
    // Key/Value cache: [num_blocks, block_size, num_kv_heads, PACKED_BYTES] uint8
    int64_t cache_dims[4] = {num_blocks, block_size, num_kv_heads, PACKED_BYTES};
    int64_t cache_strides[4] = {
        block_size * num_kv_heads * PACKED_BYTES,
        num_kv_heads * PACKED_BYTES,
        PACKED_BYTES,
        1
    };
    key_cache_tensor = aclCreateTensor(
        cache_dims, 4, ACL_UINT8, cache_strides, 0, ACL_FORMAT_ND, cache_dims, 4, d_key_packed);
    value_cache_tensor = aclCreateTensor(
        cache_dims, 4, ACL_UINT8, cache_strides, 0, ACL_FORMAT_ND, cache_dims, 4, d_value_packed);

    // Gather block ids: [gather_blocks] int32
    int64_t gather_dims[1] = {gather_blocks};
    int64_t gather_strides[1] = {1};
    gather_tensor = aclCreateTensor(
        gather_dims, 1, ACL_INT32, gather_strides, 0, ACL_FORMAT_ND, gather_dims, 1, d_gather);

    // Codebook: [CODEBOOK_SIZE] float16
    int64_t codebook_dims[1] = {CODEBOOK_SIZE};
    int64_t codebook_strides[1] = {1};
    codebook_tensor = aclCreateTensor(
        codebook_dims, 1, ACL_FLOAT16, codebook_strides, 0, ACL_FORMAT_ND, codebook_dims, 1, d_codebook);

    // Rotation: [HEAD_SIZE, HEAD_SIZE] float16
    int64_t rotation_dims[2] = {HEAD_SIZE, HEAD_SIZE};
    int64_t rotation_strides[2] = {HEAD_SIZE, 1};
    rotation_tensor = aclCreateTensor(
        rotation_dims, 2, ACL_FLOAT16, rotation_strides, 0, ACL_FORMAT_ND, rotation_dims, 2, d_rotation);

    // Output tensors: [output_rows, HEAD_SIZE] float16
    int64_t output_dims[2] = {output_rows, HEAD_SIZE};
    int64_t output_strides[2] = {HEAD_SIZE, 1};
    key_out_tensor = aclCreateTensor(
        output_dims, 2, ACL_FLOAT16, output_strides, 0, ACL_FORMAT_ND, output_dims, 2, d_key_out);
    value_out_tensor = aclCreateTensor(
        output_dims, 2, ACL_FLOAT16, output_strides, 0, ACL_FORMAT_ND, output_dims, 2, d_value_out);

    std::cout << "[TQ_DECODE_TEST] Created aclTensor objects" << std::endl;

    // Step 8: Get workspace size and create executor
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto cleanup = [&]() {
        if (d_key_packed != nullptr) { aclrtFree(d_key_packed); }
        if (d_value_packed != nullptr) { aclrtFree(d_value_packed); }
        if (d_gather != nullptr) { aclrtFree(d_gather); }
        if (d_codebook != nullptr) { aclrtFree(d_codebook); }
        if (d_rotation != nullptr) { aclrtFree(d_rotation); }
        if (d_key_out != nullptr) { aclrtFree(d_key_out); }
        if (d_value_out != nullptr) { aclrtFree(d_value_out); }
        if (d_workspace != nullptr) { aclrtFree(d_workspace); }

        if (key_cache_tensor != nullptr) { aclDestroyTensor(key_cache_tensor); }
        if (value_cache_tensor != nullptr) { aclDestroyTensor(value_cache_tensor); }
        if (gather_tensor != nullptr) { aclDestroyTensor(gather_tensor); }
        if (codebook_tensor != nullptr) { aclDestroyTensor(codebook_tensor); }
        if (rotation_tensor != nullptr) { aclDestroyTensor(rotation_tensor); }
        if (key_out_tensor != nullptr) { aclDestroyTensor(key_out_tensor); }
        if (value_out_tensor != nullptr) { aclDestroyTensor(value_out_tensor); }

        if (stream != nullptr) { aclrtDestroyStream(stream); }
        aclrtResetDevice(0);
        aclFinalize();
    };

    aclnnStatus aclnn_ret = aclnnTurboquantDecodePaged8bitGetWorkspaceSize(
        key_cache_tensor,
        value_cache_tensor,
        gather_tensor,
        codebook_tensor,
        rotation_tensor,
        HEAD_SIZE,       // headSize
        block_size,      // blockSize
        num_kv_heads,    // numKvHeads
        gather_blocks,   // totalBlocks
        0,               // rowsPerCore (legacy, unused)
        0,               // outDtype (0=fp16)
        decode_mode,        // mode (0/1; pytest [0] uses 0)
        key_out_tensor,
        value_out_tensor,
        &workspace_size,
        &executor);

    if (aclnn_ret != ACL_SUCCESS) {
        std::cerr << "[TQ_DECODE_TEST] Failed to get workspace size, ret=" << aclnn_ret << std::endl;
        cleanup();
        return -1;
    }

    std::cout << "[TQ_DECODE_TEST] Got workspace size: " << workspace_size << " bytes" << std::endl;

    // Allocate workspace
    ret = aclrtMalloc(&d_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[TQ_DECODE_TEST] Failed to allocate workspace" << std::endl;
        cleanup();
        return -1;
    }

    // Step 9: Execute kernel
    std::cout << "[TQ_DECODE_TEST] Launching turboquant_decode_paged8bit kernel..." << std::endl;

    aclnn_ret = aclnnTurboquantDecodePaged8bit(
        d_workspace,
        workspace_size,
        executor,
        stream);

    if (aclnn_ret != ACL_SUCCESS) {
        std::cerr << "[TQ_DECODE_TEST] Failed to execute kernel, ret=" << aclnn_ret << std::endl;
        cleanup();
        return -1;
    }

    // Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[TQ_DECODE_TEST] Failed to synchronize stream, ret=" << ret << std::endl;
        cleanup();
        return -1;
    }

    std::cout << "[TQ_DECODE_TEST] Kernel execution completed" << std::endl;

    // Step 10: Copy results back to host
    std::vector<uint16_t> key_out_host(output_rows * HEAD_SIZE);
    std::vector<uint16_t> value_out_host(output_rows * HEAD_SIZE);

    ret = aclrtMemcpy(key_out_host.data(), output_size, d_key_out, output_size, ACL_MEMCPY_DEVICE_TO_HOST);
    ret = aclrtMemcpy(value_out_host.data(), output_size, d_value_out, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "[TQ_DECODE_TEST] Copied results back to host" << std::endl;

    // Step 11: Compute golden using CPU dequantization
    std::vector<uint16_t> key_golden, value_golden;
    dequantize_from_packed_bytes(key_packed, key_golden, codebook, HEAD_SIZE, output_rows);
    dequantize_from_packed_bytes(value_packed, value_golden, codebook, HEAD_SIZE, output_rows);

    std::cout << "[TQ_DECODE_TEST] Computed golden reference" << std::endl;

    // Step 12: Compare results
    bool key_match = true;
    bool value_match = true;
    int mismatches = 0;
    const float atol = 2e-2;

    for (int i = 0; i < key_golden.size() && i < key_out_host.size(); i++) {
        float golden = aclFloat16ToFloat(key_golden[i]);
        float actual = aclFloat16ToFloat(key_out_host[i]);

        if (std::abs(golden - actual) > atol) {
            key_match = false;
            mismatches++;
            if (mismatches < 5) {
                std::cout << "[TQ_DECODE_TEST] Key mismatch at " << i
                          << ": golden=" << golden << ", actual=" << actual << std::endl;
            }
        }
    }

    for (int i = 0; i < value_golden.size() && i < value_out_host.size(); i++) {
        float golden = aclFloat16ToFloat(value_golden[i]);
        float actual = aclFloat16ToFloat(value_out_host[i]);

        if (std::abs(golden - actual) > atol) {
            value_match = false;
            mismatches++;
        }
    }

    // Step 13: Report results
    if (key_match && value_match) {
        std::cout << "[TQ_DECODE_TEST] SUCCESS: All outputs match golden reference!" << std::endl;
    } else {
        std::cout << "[TQ_DECODE_TEST] FAILURE: " << mismatches << " mismatches found" << std::endl;
    }

    // Step 14: Cleanup
    cleanup();

    std::cout << "[TQ_DECODE_TEST] Cleanup completed" << std::endl;

    return (key_match && value_match) ? 0 : -1;
}
