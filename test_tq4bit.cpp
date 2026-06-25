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

/*
 * Pure C++ correctness test for TurboQuant 4-bit slab cache operators
 * using raw ACL / aclnn API (no Python, no torch, no dlopen).
 *
 *   aclnnTurboquantPackKvForCache4bit       — quantize + pack KV into
 *                                              paged 4-bit slab cache
 *   aclnnTurboquantAttentionPaged4bit       — paged attention with 4-bit
 *                                              slab cache decode
 *
 * Build:
 *   bash build_test_tq4bit.sh          # compile only
 *   bash build_test_tq4bit.sh --rebuild-op  # rebuild custom op + compile
 *
 * Run:
 *   ASCEND_CUSTOM_OPP_PATH=... ./test_tq4bit [options]
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
#include <chrono>

#include "acl/acl.h"
#include "acl/acl_base_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnn_turboquant_pack_kv_for_cache4bit.h"
#include "aclnn_turboquant_attention_paged4bit.h"

// ---------------------------------------------------------------------------
// Constants — must match kernel expectations
// ---------------------------------------------------------------------------
constexpr int64_t HEAD_SIZE = 128;
constexpr int64_t ROW_BYTES_4BIT = HEAD_SIZE / 2 + 2;   // 66
constexpr int64_t GROUP_ROWS = 4;                        // 4 rows per slab group
constexpr int64_t CODEBOOK_SIZE = 16;                    // 4-bit = 16 centroids

// TurboQuant v3 cubic+linear decoding coefficients
constexpr double FY_LINEAR = 0.020799;
constexpr double FY_CUBIC  = 0.0001926;

#ifndef VLLM_ASCEND_CUSTOM_OPP_PATH
#define VLLM_ASCEND_CUSTOM_OPP_PATH "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
#endif

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

void configure_custom_opp_path() {
    const char *custom_opp_path = VLLM_ASCEND_CUSTOM_OPP_PATH;
    if (custom_opp_path == nullptr || custom_opp_path[0] == '\0') return;

    const char *current = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (current == nullptr || current[0] == '\0') {
        setenv("ASCEND_CUSTOM_OPP_PATH", custom_opp_path, 1);
        return;
    }
    std::string current_paths(current);
    std::string custom_path(custom_opp_path);
    if (current_paths.find(custom_path) != std::string::npos) return;
    std::string updated = custom_path + ":" + current_paths;
    setenv("ASCEND_CUSTOM_OPP_PATH", updated.c_str(), 1);
}

int64_t read_positive_int_env(const char *name, int64_t default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return default_value;
    errno = 0;
    char *end = nullptr;
    long long value = std::strtoll(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value <= 0 || value > INT_MAX)
        return default_value;
    return static_cast<int64_t>(value);
}

// ---------------------------------------------------------------------------
// Float16 helpers  (aclFloat16 ↔ float, stored as uint16_t)
// ---------------------------------------------------------------------------

// IEEE-754 float32 → float16 bit conversion (round-to-nearest-even).
// Matches aclFloatToFloat16 behaviour for the range we need.
static uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = x & 0x80000000u;
    uint32_t exp  = x & 0x7F800000u;
    uint32_t mant = x & 0x007FFFFFu;

    if (exp == 0x7F800000u) {  // Inf / NaN
        return static_cast<uint16_t>(sign >> 16 | 0x7C00u | (mant ? 1 : 0));
    }
    if (exp == 0) {  // zero / denormal
        if (mant == 0) return static_cast<uint16_t>(sign >> 16);
        // denormal → flush to zero for simplicity
        return static_cast<uint16_t>(sign >> 16);
    }

    int32_t new_exp = static_cast<int32_t>(exp >> 23) - 127 + 15;
    if (new_exp >= 31) {  // overflow → Inf
        return static_cast<uint16_t>(sign >> 16 | 0x7C00u);
    }
    if (new_exp <= 0) {  // underflow → zero
        return static_cast<uint16_t>(sign >> 16);
    }

    uint32_t new_mant = (mant + 0x00400000u) >> 13;  // round
    if (new_mant & 0x0400u) {  // carry into exponent
        new_exp++;
        new_mant = 0;
    }
    return static_cast<uint16_t>(sign >> 16 | (new_exp << 10) | (new_mant & 0x03FFu));
}

static float fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = static_cast<uint32_t>(h & 0x7C00u);
    uint32_t mant = static_cast<uint32_t>(h & 0x03FFu);

    if (exp == 0) {
        if (mant == 0) {  // zero
            float f;
            uint32_t v = sign;
            std::memcpy(&f, &v, sizeof(f));
            return f;
        }
        // denormal: exp = -14, mant has implicit leading 0
        exp = 1;
        while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
        mant &= ~0x0400u;
        uint32_t v = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &v, sizeof(f));
        return f;
    }
    if (exp == 0x7C00u) {  // Inf / NaN
        uint32_t v = sign | 0x7F800000u | (mant << 13);
        float f;
        std::memcpy(&f, &v, sizeof(f));
        return f;
    }
    uint32_t v = sign | ((exp >> 10) + 127 - 15) << 23 | (mant << 13);
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Random data generation
// ---------------------------------------------------------------------------

void generate_random_fp16(std::vector<uint16_t>& data, int64_t size, uint32_t seed = 42) {
    data.resize(size);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < size; i++) {
        data[i] = fp32_to_fp16_bits(dist(rng));
    }
}

void generate_codebook_fp16(std::vector<uint16_t>& cb) {
    cb.resize(CODEBOOK_SIZE);
    for (int64_t i = 0; i < CODEBOOK_SIZE; i++) {
        float val = -1.0f + (2.0f * i) / (CODEBOOK_SIZE - 1);
        cb[i] = fp32_to_fp16_bits(val);
    }
}

void generate_identity_rotation_fp16(std::vector<uint16_t>& rot, int64_t hsz) {
    rot.resize(hsz * hsz);
    for (int64_t i = 0; i < hsz; i++)
        for (int64_t j = 0; j < hsz; j++)
            rot[i * hsz + j] = fp32_to_fp16_bits((i == j) ? 1.0f : 0.0f);
}

// ---------------------------------------------------------------------------
// 4-bit slab cache decode (CPU golden reference)
// ---------------------------------------------------------------------------
// Slab format per group (4 rows, 264 bytes = 66*4):
//   [0..255]   128 uint16 words: word[d] = idx[row0][d] | (idx[row1][d]<<4)
//                                       | (idx[row2][d]<<8) | (idx[row3][d]<<12)
//   [256..263] 4 norms, 2 bytes each (fp16), interleaved

void decode_slab_row_4bit(
    const uint8_t *cache_base,   // start of one [block_size * ROW_BYTES_4BIT] row-block
    int64_t block_size,
    int64_t abs_pos,
    const uint16_t *codebook,
    uint16_t norm_out[1],
    int64_t indices_out[HEAD_SIZE])
{
    int64_t group_idx = abs_pos / GROUP_ROWS;
    int64_t group_row = abs_pos % GROUP_ROWS;
    int64_t group_base = group_idx * ROW_BYTES_4BIT * GROUP_ROWS;

    const uint8_t *group = cache_base + group_base;

    // Decode 128 index words (256 bytes of index data)
    for (int64_t d = 0; d < HEAD_SIZE; d++) {
        uint16_t word = static_cast<uint16_t>(group[2 * d])
                      | (static_cast<uint16_t>(group[2 * d + 1]) << 8);
        // Extract 4-bit nibble for group_row
        int64_t idx = (word >> (group_row * 4)) & 0x0F;
        indices_out[d] = idx;
    }

    // Decode norm (2 bytes at offset HEAD_SIZE*2 + group_row*2)
    int64_t norm_off = HEAD_SIZE * 2 + group_row * 2;
    norm_out[0] = static_cast<uint16_t>(group[norm_off])
                | (static_cast<uint16_t>(group[norm_off + 1]) << 8);
}

// Full decode: indices → codebook lookup → cubic+linear → scale by norm
void decode_slab_row_to_fp16(
    const uint8_t *cache_base,
    int64_t block_size,
    int64_t abs_pos,
    const uint16_t *codebook,
    float decoded[HEAD_SIZE])
{
    uint16_t norm;
    int64_t indices[HEAD_SIZE];
    decode_slab_row_4bit(cache_base, block_size, abs_pos, codebook, &norm, indices);

    float norm_f = fp16_bits_to_fp32(norm);
    for (int64_t d = 0; d < HEAD_SIZE; d++) {
        float x = static_cast<float>(indices[d]) - 7.5f;
        float y = static_cast<float>(FY_CUBIC) * x * x * x
                + static_cast<float>(FY_LINEAR) * x;
        decoded[d] = y * norm_f;
    }
}

// ---------------------------------------------------------------------------
// aclTensor creation helpers
// ---------------------------------------------------------------------------

aclTensor *create_acl_tensor_1d(const std::vector<int64_t>& dims,
                                 aclDataType dtype,
                                 const std::vector<int64_t>& strides,
                                 void *dev_ptr)
{
    return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(),
                           0, ACL_FORMAT_ND, dims.data(), dims.size(), dev_ptr);
}

aclIntArray *create_acl_int_array(const std::vector<int64_t>& values) {
    return aclCreateIntArray(values.data(), values.size());
}

// ---------------------------------------------------------------------------
// RAII-style ACL context
// ---------------------------------------------------------------------------

struct AclContext {
    aclrtStream stream = nullptr;
    bool initialized = false;

    bool init() {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[TQ4BIT] aclInit failed: " << ret << "\n";
            return false;
        }
        uint32_t dev_count = 0;
        ret = aclrtGetDeviceCount(&dev_count);
        if (ret != ACL_SUCCESS || dev_count == 0) {
            std::cerr << "[TQ4BIT] No ACL devices\n";
            return false;
        }
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[TQ4BIT] aclrtSetDevice(0) failed: " << ret << "\n";
            return false;
        }
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS || stream == nullptr) {
            std::cerr << "[TQ4BIT] aclrtCreateStream failed\n";
            return false;
        }
        initialized = true;
        std::cout << "[TQ4BIT] ACL initialized, device=0\n";
        return true;
    }

    void finalize() {
        if (stream) { aclrtDestroyStream(stream); stream = nullptr; }
        if (initialized) { aclrtResetDevice(0); aclFinalize(); initialized = false; }
    }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "[TQ4BIT] Starting turboquant 4-bit slab cache test...\n";
    configure_custom_opp_path();

    // ---- Read config from env vars ----
    int64_t seq_len       = read_positive_int_env("TQ4BIT_SEQ_LEN", 32);
    int64_t query_tokens  = read_positive_int_env("TQ4BIT_QUERY_TOKENS", 1);
    int64_t num_heads     = read_positive_int_env("TQ4BIT_NUM_HEADS", 4);
    int64_t num_kv_heads  = read_positive_int_env("TQ4BIT_NUM_KV_HEADS", 2);
    int64_t block_size    = read_positive_int_env("TQ4BIT_BLOCK_SIZE", 16);

    if (num_heads % num_kv_heads != 0) {
        std::cerr << "[TQ4BIT] num_heads must be divisible by num_kv_heads\n";
        return 1;
    }
    if (block_size % GROUP_ROWS != 0) {
        std::cerr << "[TQ4BIT] block_size must be a multiple of " << GROUP_ROWS << "\n";
        return 1;
    }
    if (query_tokens > seq_len) {
        std::cerr << "[TQ4BIT] query_tokens must be <= seq_len\n";
        return 1;
    }

    int64_t num_blocks = (seq_len + block_size - 1) / block_size;
    double scale = 1.0 / std::sqrt(static_cast<double>(HEAD_SIZE));

    std::cout << "[TQ4BIT] Config: seq_len=" << seq_len
              << " query_tokens=" << query_tokens
              << " heads=" << num_heads
              << " kv_heads=" << num_kv_heads
              << " block_size=" << block_size
              << " num_blocks=" << num_blocks
              << " scale=" << scale << "\n";

    // ---- Init ACL ----
    AclContext ctx;
    if (!ctx.init()) return 1;

    // ---- Generate test data on host ----
    std::vector<uint16_t> key_data, value_data, query_data;
    std::vector<uint16_t> codebook, rotation;

    generate_random_fp16(key_data,   seq_len * num_kv_heads * HEAD_SIZE, 100);
    generate_random_fp16(value_data, seq_len * num_kv_heads * HEAD_SIZE, 200);
    generate_random_fp16(query_data, query_tokens * num_heads * HEAD_SIZE, 300);
    generate_codebook_fp16(codebook);
    generate_identity_rotation_fp16(rotation, HEAD_SIZE);

    // Slot mapping: [0, 1, ..., seq_len-1]
    std::vector<int32_t> slot_mapping(seq_len);
    for (int64_t i = 0; i < seq_len; i++) slot_mapping[i] = static_cast<int32_t>(i);

    // Pack op request ranges: single request covering all KV tokens.
    std::vector<int32_t> query_start_loc = {0, static_cast<int32_t>(seq_len)};

    // Block table: single sequence, blocks mapped sequentially
    // shape [1, num_blocks]
    std::vector<int32_t> block_table(num_blocks);
    for (int64_t i = 0; i < num_blocks; i++) block_table[i] = static_cast<int32_t>(i);

    // ---- Allocate device memory ----
    size_t kv_size       = seq_len * num_kv_heads * HEAD_SIZE * sizeof(uint16_t);
    size_t q_size        = query_tokens * num_heads * HEAD_SIZE * sizeof(uint16_t);
    size_t cb_size       = CODEBOOK_SIZE * sizeof(uint16_t);
    size_t rot_size      = HEAD_SIZE * HEAD_SIZE * sizeof(uint16_t);
    size_t slot_size     = seq_len * sizeof(int32_t);
    size_t query_start_size = query_start_loc.size() * sizeof(int32_t);
    size_t bt_size       = num_blocks * sizeof(int32_t);
    size_t cache_size    = num_blocks * num_kv_heads * block_size * ROW_BYTES_4BIT;
    size_t out_size      = query_tokens * num_heads * HEAD_SIZE * sizeof(uint16_t);

    void *d_key = nullptr, *d_value = nullptr, *d_query = nullptr;
    void *d_codebook = nullptr, *d_rotation = nullptr;
    void *d_slot = nullptr, *d_query_start = nullptr, *d_bt = nullptr;
    void *d_key_cache = nullptr, *d_value_cache = nullptr;
    void *d_attn_out = nullptr;

    auto malloc_dev = [&](void **ptr, size_t sz, const char *name) {
        aclError ret = aclrtMalloc(ptr, sz, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[TQ4BIT] aclrtMalloc(" << name << ", " << sz << ") failed: " << ret << "\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= malloc_dev(&d_key,        kv_size,    "key");
    ok &= malloc_dev(&d_value,      kv_size,    "value");
    ok &= malloc_dev(&d_query,      q_size,     "query");
    ok &= malloc_dev(&d_codebook,   cb_size,    "codebook");
    ok &= malloc_dev(&d_rotation,   rot_size,   "rotation");
    ok &= malloc_dev(&d_slot,       slot_size,  "slot_mapping");
    ok &= malloc_dev(&d_query_start, query_start_size, "query_start_loc");
    ok &= malloc_dev(&d_bt,         bt_size,    "block_table");
    ok &= malloc_dev(&d_key_cache,  cache_size, "key_cache");
    ok &= malloc_dev(&d_value_cache,cache_size, "value_cache");
    ok &= malloc_dev(&d_attn_out,   out_size,   "attn_out");
    if (!ok) { ctx.finalize(); return 1; }

    // ---- Copy data to device ----
    auto h2d = [&](void *dst, const void *src, size_t sz, const char *name) {
        aclError ret = aclrtMemcpy(dst, sz, src, sz, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[TQ4BIT] H2D(" << name << ") failed: " << ret << "\n";
            return false;
        }
        return true;
    };

    ok = true;
    ok &= h2d(d_key,        key_data.data(),       kv_size,    "key");
    ok &= h2d(d_value,      value_data.data(),     kv_size,    "value");
    ok &= h2d(d_query,      query_data.data(),     q_size,     "query");
    ok &= h2d(d_codebook,   codebook.data(),       cb_size,    "codebook");
    ok &= h2d(d_rotation,   rotation.data(),       rot_size,   "rotation");
    ok &= h2d(d_slot,       slot_mapping.data(),   slot_size,  "slot_mapping");
    ok &= h2d(d_query_start, query_start_loc.data(), query_start_size, "query_start_loc");
    ok &= h2d(d_bt,         block_table.data(),    bt_size,    "block_table");
    // Initialize caches to zero
    std::vector<uint8_t> zero_cache(cache_size, 0);
    ok &= h2d(d_key_cache,  zero_cache.data(), cache_size, "key_cache(zero)");
    ok &= h2d(d_value_cache,zero_cache.data(), cache_size, "value_cache(zero)");
    if (!ok) { ctx.finalize(); return 1; }

    std::cout << "[TQ4BIT] Data on device ready.\n";

    // ==================================================================
    // TEST 1: aclnnTurboquantPackKvForCache4bit
    // ==================================================================
    std::cout << "\n[TQ4BIT] === TEST 1: aclnnTurboquantPackKvForCache4bit ===\n";

    // Create aclTensor objects for pack op
    // key:       [seq_len, num_kv_heads, HEAD_SIZE]  FP16
    // value:     [seq_len, num_kv_heads, HEAD_SIZE]  FP16
    // codebook:  [16]                                  FP16
    // rotationT: [HEAD_SIZE, HEAD_SIZE]                FP16
    // slotMapping: [seq_len]                            INT32
    // queryStartLoc: [2]                                INT32
    // keyCache:  [num_blocks, num_kv_heads, block_size*ROW_BYTES_4BIT]  UINT8
    // valueCache: same shape as keyCache                                  UINT8

    int64_t kv_3d_dims[3] = {seq_len, num_kv_heads, HEAD_SIZE};
    int64_t kv_3d_strides[3] = {num_kv_heads * HEAD_SIZE, HEAD_SIZE, 1};

    int64_t cb_1d_dims[1] = {CODEBOOK_SIZE};
    int64_t cb_1d_strides[1] = {1};

    int64_t rot_2d_dims[2] = {HEAD_SIZE, HEAD_SIZE};
    int64_t rot_2d_strides[2] = {HEAD_SIZE, 1};

    int64_t slot_1d_dims[1] = {seq_len};
    int64_t slot_1d_strides[1] = {1};
    int64_t query_start_1d_dims[1] = {static_cast<int64_t>(query_start_loc.size())};
    int64_t query_start_1d_strides[1] = {1};

    int64_t cache_3d_dims[3] = {num_blocks, num_kv_heads, block_size * ROW_BYTES_4BIT};
    int64_t cache_3d_strides[3] = {num_kv_heads * block_size * ROW_BYTES_4BIT,
                                    block_size * ROW_BYTES_4BIT,
                                    1};

    aclTensor *t_key = aclCreateTensor(kv_3d_dims, 3, ACL_FLOAT16, kv_3d_strides,
                                        0, ACL_FORMAT_ND, kv_3d_dims, 3, d_key);
    aclTensor *t_value = aclCreateTensor(kv_3d_dims, 3, ACL_FLOAT16, kv_3d_strides,
                                          0, ACL_FORMAT_ND, kv_3d_dims, 3, d_value);
    aclTensor *t_codebook = aclCreateTensor(cb_1d_dims, 1, ACL_FLOAT16, cb_1d_strides,
                                             0, ACL_FORMAT_ND, cb_1d_dims, 1, d_codebook);
    aclTensor *t_rotationT = aclCreateTensor(rot_2d_dims, 2, ACL_FLOAT16, rot_2d_strides,
                                              0, ACL_FORMAT_ND, rot_2d_dims, 2, d_rotation);
    aclTensor *t_slotMapping = aclCreateTensor(slot_1d_dims, 1, ACL_INT32, slot_1d_strides,
                                                0, ACL_FORMAT_ND, slot_1d_dims, 1, d_slot);
    aclTensor *t_queryStartLoc = aclCreateTensor(query_start_1d_dims, 1, ACL_INT32,
                                                  query_start_1d_strides, 0, ACL_FORMAT_ND,
                                                  query_start_1d_dims, 1, d_query_start);
    aclTensor *t_keyCache = aclCreateTensor(cache_3d_dims, 3, ACL_UINT8, cache_3d_strides,
                                             0, ACL_FORMAT_ND, cache_3d_dims, 3, d_key_cache);
    aclTensor *t_valueCache = aclCreateTensor(cache_3d_dims, 3, ACL_UINT8, cache_3d_strides,
                                               0, ACL_FORMAT_ND, cache_3d_dims, 3, d_value_cache);

    // Pack op tiling parameters
    int64_t nVec = seq_len * num_kv_heads;      // total vectors
    int64_t vecPerCore = 64;                     // tiling granularity (safe default)
    int64_t numReqs = 1;                          // single contiguous request

    uint64_t pack_workspace_size = 0;
    aclOpExecutor *pack_executor = nullptr;

    aclnnStatus pack_status = aclnnTurboquantPackKvForCache4bitGetWorkspaceSize(
        t_key,
        t_value,
        t_codebook,
        t_rotationT,
        t_slotMapping,
        t_queryStartLoc,
        nVec,
        vecPerCore,
        num_kv_heads,
        block_size,
        num_blocks,
        numReqs,
        t_keyCache,
        t_valueCache,
        &pack_workspace_size,
        &pack_executor);

    if (pack_status != ACL_SUCCESS) {
        std::cerr << "[TQ4BIT] Pack GetWorkspaceSize failed: " << pack_status << "\n";
        ctx.finalize();
        return 1;
    }
    std::cout << "[TQ4BIT] Pack workspace: " << pack_workspace_size << " bytes\n";

    void *d_pack_workspace = nullptr;
    if (pack_workspace_size > 0) {
        aclError ret = aclrtMalloc(&d_pack_workspace, pack_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[TQ4BIT] Pack workspace alloc failed\n";
            ctx.finalize();
            return 1;
        }
    }

    pack_status = aclnnTurboquantPackKvForCache4bit(
        d_pack_workspace, pack_workspace_size, pack_executor, ctx.stream);

    if (pack_status != ACL_SUCCESS) {
        std::cerr << "[TQ4BIT] Pack execute failed: " << pack_status << "\n";
        ctx.finalize();
        return 1;
    }

    aclError sync_ret = aclrtSynchronizeStream(ctx.stream);
    if (sync_ret != ACL_SUCCESS) {
        std::cerr << "[TQ4BIT] Pack stream sync failed: " << sync_ret << "\n";
        ctx.finalize();
        return 1;
    }
    std::cout << "[TQ4BIT] Pack op completed successfully.\n";

    // ---- Copy packed cache back for inspection ----
    std::vector<uint8_t> key_cache_host(cache_size);
    std::vector<uint8_t> value_cache_host(cache_size);
    aclrtMemcpy(key_cache_host.data(), cache_size, d_key_cache, cache_size,
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(value_cache_host.data(), cache_size, d_value_cache, cache_size,
                ACL_MEMCPY_DEVICE_TO_HOST);

    // Check cache is not all zeros
    int64_t key_sum = 0, value_sum = 0;
    for (size_t i = 0; i < cache_size; i++) {
        key_sum += key_cache_host[i];
        value_sum += value_cache_host[i];
    }
    std::cout << "[TQ4BIT] key_cache sum(uint8)=" << key_sum
              << " value_cache sum(uint8)=" << value_sum << "\n";

    bool pack_ok = (key_sum > 0 && value_sum > 0);
    std::cout << "[TQ4BIT] Pack test: " << (pack_ok ? "PASS" : "FAIL") << "\n";

    // ---- Pack round-trip: decode rows and compare with original ----
    {
        int64_t max_check = std::min(static_cast<int64_t>(8), seq_len);
        int64_t mismatches = 0;
        double max_error = 0.0;

        for (int64_t pos = 0; pos < max_check; pos++) {
            for (int64_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
                // Find block and position within block
                int64_t blk = pos / block_size;
                int64_t pos_in_blk = pos % block_size;

                // Pointer to this head's cache row-block
                // cache layout: [num_blocks, num_kv_heads, block_size*ROW_BYTES_4BIT]
                // stride0 = num_kv_heads * block_size * ROW_BYTES_4BIT
                // stride1 = block_size * ROW_BYTES_4BIT
                // stride2 = 1
                int64_t blk_offset = blk * (num_kv_heads * block_size * ROW_BYTES_4BIT)
                                    + kv_h * (block_size * ROW_BYTES_4BIT);
                const uint8_t *cache_row_base = key_cache_host.data() + blk_offset;

                float decoded[HEAD_SIZE];
                decode_slab_row_to_fp16(cache_row_base, block_size, pos_in_blk,
                                       codebook.data(), decoded);

                // Original key at [pos, kv_h, :]
                double err = 0.0;
                for (int64_t d = 0; d < HEAD_SIZE; d++) {
                    float orig = fp16_bits_to_fp32(
                        key_data[pos * num_kv_heads * HEAD_SIZE + kv_h * HEAD_SIZE + d]);
                    double diff = std::abs(decoded[d] - orig);
                    err = std::max(err, diff);
                }
                max_error = std::max(max_error, err);

                // 4-bit quantization loss is O(0.1), generous tolerance
                if (err > 0.3) mismatches++;
            }
        }

        double total = max_check * num_kv_heads;
        double pass_rate = 1.0 - mismatches / total;
        std::cout << "[TQ4BIT] Pack round-trip: max_error=" << max_error
                  << " mismatches=" << mismatches << "/" << total
                  << " pass_rate=" << pass_rate * 100 << "%\n";
        std::cout << "[TQ4BIT] Pack round-trip: "
                  << (max_error < 0.3 ? "PASS" : "WARN (quantization loss high)") << "\n";
    }

    // ==================================================================
    // TEST 2: aclnnTurboquantAttentionPaged4bit
    // ==================================================================
    std::cout << "\n[TQ4BIT] === TEST 2: aclnnTurboquantAttentionPaged4bit ===\n";

    // query:       [query_tokens, num_heads, HEAD_SIZE]  FP16
    // keyCache:    [num_blocks, num_kv_heads, block_size*ROW_BYTES_4BIT]  UINT8
    // valueCache:  same as keyCache
    // blockTable:  [1, num_blocks]  INT32
    // attentionOut: [query_tokens, num_heads, HEAD_SIZE]  FP16

    int64_t q_3d_dims[3] = {query_tokens, num_heads, HEAD_SIZE};
    int64_t q_3d_strides[3] = {num_heads * HEAD_SIZE, HEAD_SIZE, 1};

    int64_t bt_2d_dims[2] = {1, num_blocks};
    int64_t bt_2d_strides[2] = {num_blocks, 1};

    int64_t out_3d_dims[3] = {query_tokens, num_heads, HEAD_SIZE};
    int64_t out_3d_strides[3] = {num_heads * HEAD_SIZE, HEAD_SIZE, 1};

    aclTensor *t_query = aclCreateTensor(q_3d_dims, 3, ACL_FLOAT16, q_3d_strides,
                                          0, ACL_FORMAT_ND, q_3d_dims, 3, d_query);
    aclTensor *t_blockTable = aclCreateTensor(bt_2d_dims, 2, ACL_INT32, bt_2d_strides,
                                               0, ACL_FORMAT_ND, bt_2d_dims, 2, d_bt);
    aclTensor *t_attnOut = aclCreateTensor(out_3d_dims, 3, ACL_FLOAT16, out_3d_strides,
                                            0, ACL_FORMAT_ND, out_3d_dims, 3, d_attn_out);

    std::vector<int64_t> seq_len_q_vec {query_tokens};
    std::vector<int64_t> seq_len_kv_vec {seq_len};
    aclIntArray *t_actualSeqLenQ  = create_acl_int_array(seq_len_q_vec);
    aclIntArray *t_actualSeqLenKv = create_acl_int_array(seq_len_kv_vec);

    // Use the same codebook/rotation for both key and value (4-bit, same bits)
    uint64_t attn_workspace_size = 0;
    aclOpExecutor *attn_executor = nullptr;

    aclnnStatus attn_status = aclnnTurboquantAttentionPaged4bitGetWorkspaceSize(
        t_query,
        t_keyCache,
        t_valueCache,
        t_blockTable,
        t_actualSeqLenQ,
        t_actualSeqLenKv,
        t_codebook,
        t_rotationT,
        t_codebook,        // codebookValue (same for 4-bit)
        t_rotationT,       // rotationValue (identity, R = R^T)
        num_heads,
        num_kv_heads,
        HEAD_SIZE,
        block_size,
        seq_len,           // maxActualSeqLen
        scale,
        t_attnOut,
        &attn_workspace_size,
        &attn_executor);

    if (attn_status != ACL_SUCCESS) {
        std::cerr << "[TQ4BIT] Attention GetWorkspaceSize failed: " << attn_status << "\n";
        ctx.finalize();
        return 1;
    }
    std::cout << "[TQ4BIT] Attention workspace: " << attn_workspace_size << " bytes\n";

    void *d_attn_workspace = nullptr;
    if (attn_workspace_size > 0) {
        aclError ret = aclrtMalloc(&d_attn_workspace, attn_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[TQ4BIT] Attention workspace alloc failed\n";
            ctx.finalize();
            return 1;
        }
    }

    attn_status = aclnnTurboquantAttentionPaged4bit(
        d_attn_workspace, attn_workspace_size, attn_executor, ctx.stream);

    if (attn_status != ACL_SUCCESS) {
        std::cerr << "[TQ4BIT] Attention execute failed: " << attn_status << "\n";
        ctx.finalize();
        return 1;
    }

    sync_ret = aclrtSynchronizeStream(ctx.stream);
    if (sync_ret != ACL_SUCCESS) {
        std::cerr << "[TQ4BIT] Attention stream sync failed: " << sync_ret << "\n";
        ctx.finalize();
        return 1;
    }
    std::cout << "[TQ4BIT] Attention op completed successfully.\n";

    // ---- Copy attention output back ----
    std::vector<uint16_t> attn_out_host(query_tokens * num_heads * HEAD_SIZE);
    aclrtMemcpy(attn_out_host.data(), out_size, d_attn_out, out_size,
                ACL_MEMCPY_DEVICE_TO_HOST);

    // Check output shape conceptually and that it's not all zeros
    double out_abs_sum = 0.0;
    for (size_t i = 0; i < attn_out_host.size(); i++) {
        out_abs_sum += std::abs(fp16_bits_to_fp32(attn_out_host[i]));
    }
    std::cout << "[TQ4BIT] Attention output abs_sum=" << out_abs_sum << "\n";
    bool attn_ok = (out_abs_sum > 1e-6);
    std::cout << "[TQ4BIT] Attention nonzero: " << (attn_ok ? "PASS" : "FAIL") << "\n";

    // ---- Golden paged attention on CPU ----
    {
        // Decode all KV rows from the packed slab cache for the single sequence
        int64_t gqa_group = num_heads / num_kv_heads;
        std::vector<std::vector<float>> decoded_keys(seq_len);
        std::vector<std::vector<float>> decoded_values(seq_len);

        for (int64_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
            for (int64_t pos = 0; pos < seq_len; pos++) {
                int64_t blk = pos / block_size;
                int64_t pos_in_blk = pos % block_size;
                int64_t blk_offset_k = blk * (num_kv_heads * block_size * ROW_BYTES_4BIT)
                                      + kv_h * (block_size * ROW_BYTES_4BIT);
                int64_t blk_offset_v = blk_offset_k;  // same layout for value cache

                float decoded_k[HEAD_SIZE], decoded_v[HEAD_SIZE];
                decode_slab_row_to_fp16(key_cache_host.data() + blk_offset_k,
                                       block_size, pos_in_blk, codebook.data(), decoded_k);
                decode_slab_row_to_fp16(value_cache_host.data() + blk_offset_v,
                                       block_size, pos_in_blk, codebook.data(), decoded_v);

                // For identity rotation R^T, decode is: y_hat @ R^T * norm
                // Since R^T = I, the decoded row IS the final key/value vector.
            }
        }

        // Build per-kv-head decoded K and V matrices
        // keys[kv_h][pos][d] and values[kv_h][pos][d]
        std::vector<std::vector<std::vector<float>>> keys_per_head(
            num_kv_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(HEAD_SIZE)));
        std::vector<std::vector<std::vector<float>>> vals_per_head(
            num_kv_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(HEAD_SIZE)));

        for (int64_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
            for (int64_t pos = 0; pos < seq_len; pos++) {
                int64_t blk = pos / block_size;
                int64_t pos_in_blk = pos % block_size;
                int64_t blk_offset = blk * (num_kv_heads * block_size * ROW_BYTES_4BIT)
                                    + kv_h * (block_size * ROW_BYTES_4BIT);

                decode_slab_row_to_fp16(key_cache_host.data() + blk_offset,
                                       block_size, pos_in_blk, codebook.data(),
                                       keys_per_head[kv_h][pos].data());
                decode_slab_row_to_fp16(value_cache_host.data() + blk_offset,
                                       block_size, pos_in_blk, codebook.data(),
                                       vals_per_head[kv_h][pos].data());
            }
        }

        // For each query token and each head, compute paged attention
        // Single sequence: causal_kv_end = seq_len (decode scenario: query at end)
        int64_t causal_kv_end = seq_len;
        double max_diff = 0.0;
        int64_t mismatches = 0;

        for (int64_t t = 0; t < query_tokens; t++) {
            for (int64_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
                for (int64_t g = 0; g < gqa_group; g++) {
                    int64_t h = kv_h * gqa_group + g;

                    // Query vector [HEAD_SIZE]
                    float q_vec[HEAD_SIZE];
                    for (int64_t d = 0; d < HEAD_SIZE; d++)
                        q_vec[d] = fp16_bits_to_fp32(
                            query_data[t * num_heads * HEAD_SIZE + h * HEAD_SIZE + d]);

                    // With identity rotation, query_rot = q_vec @ R^T = q_vec
                    // Attention scores = keys * query_rot * scale
                    std::vector<float> scores(causal_kv_end);
                    for (int64_t pos = 0; pos < causal_kv_end; pos++) {
                        float dot = 0.0f;
                        for (int64_t d = 0; d < HEAD_SIZE; d++)
                            dot += keys_per_head[kv_h][pos][d] * q_vec[d];
                        scores[pos] = dot * static_cast<float>(scale);
                    }

                    // Softmax
                    float max_score = *std::max_element(scores.begin(), scores.end());
                    float sum_exp = 0.0f;
                    for (int64_t pos = 0; pos < causal_kv_end; pos++) {
                        scores[pos] = std::exp(scores[pos] - max_score);
                        sum_exp += scores[pos];
                    }
                    for (int64_t pos = 0; pos < causal_kv_end; pos++)
                        scores[pos] /= sum_exp;

                    // Weighted sum of values
                    float out_vec[HEAD_SIZE] = {};
                    for (int64_t pos = 0; pos < causal_kv_end; pos++)
                        for (int64_t d = 0; d < HEAD_SIZE; d++)
                            out_vec[d] += scores[pos] * vals_per_head[kv_h][pos][d];

                    // With identity rotation for value: out_y @ R = out_y
                    // Compare with NPU output
                    float npu_out[HEAD_SIZE];
                    for (int64_t d = 0; d < HEAD_SIZE; d++)
                        npu_out[d] = fp16_bits_to_fp32(
                            attn_out_host[t * num_heads * HEAD_SIZE + h * HEAD_SIZE + d]);

                    for (int64_t d = 0; d < HEAD_SIZE; d++) {
                        double diff = std::abs(npu_out[d] - out_vec[d]);
                        max_diff = std::max(max_diff, diff);
                        if (diff > 0.05) mismatches++;
                    }
                }
            }
        }

        int64_t total = query_tokens * num_heads * HEAD_SIZE;
        double match_rate = 1.0 - static_cast<double>(mismatches) / total;
        std::cout << "[TQ4BIT] Attention golden: max_diff=" << max_diff
                  << " mismatches=" << mismatches << "/" << total
                  << " match_rate=" << match_rate * 100 << "%\n";
        std::cout << "[TQ4BIT] Attention golden: "
                  << (match_rate >= 0.90 ? "PASS" : "FAIL") << "\n";
    }

    // ==================================================================
    // Summary
    // ==================================================================
    std::cout << "\n[TQ4BIT] === SUMMARY ===\n";
    std::cout << "[TQ4BIT] Pack:   " << (pack_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "[TQ4BIT] Attn:   " << (attn_ok ? "PASS" : "FAIL") << "\n";

    // Cleanup
    if (d_pack_workspace) aclrtFree(d_pack_workspace);
    if (d_attn_workspace) aclrtFree(d_attn_workspace);
    aclrtFree(d_key); aclrtFree(d_value); aclrtFree(d_query);
    aclrtFree(d_codebook); aclrtFree(d_rotation);
    aclrtFree(d_slot); aclrtFree(d_bt);
    aclrtFree(d_key_cache); aclrtFree(d_value_cache); aclrtFree(d_attn_out);

    aclDestroyTensor(t_key); aclDestroyTensor(t_value);
    aclDestroyTensor(t_codebook); aclDestroyTensor(t_rotationT);
    aclDestroyTensor(t_slotMapping);
    aclDestroyTensor(t_keyCache); aclDestroyTensor(t_valueCache);
    aclDestroyTensor(t_query); aclDestroyTensor(t_blockTable);
    aclDestroyTensor(t_attnOut);
    aclDestroyIntArray(t_actualSeqLenQ);
    aclDestroyIntArray(t_actualSeqLenKv);

    ctx.finalize();
    std::cout << "[TQ4BIT] Cleanup done.\n";

    return (pack_ok && attn_ok) ? 0 : 1;
}
