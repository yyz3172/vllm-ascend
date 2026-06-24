/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All Rights Reserved.
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

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>
#include <glob.h>

#include "turboquant_attention_paged4bit/op_host/aclnn_turboquant_attention_paged4bit.h"
#include "turboquant_pack_kv_for_cache4bit/op_host/aclnn_turboquant_pack_kv_for_cache4bit.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHeadSize = 128;
constexpr int64_t kCodebookSize = 16;
constexpr int64_t kRowBytes4bit = kHeadSize / 2 + 2;

using AclCreateTensor = aclTensor* (*)(
    const int64_t* view_dims,
    uint64_t view_dims_num,
    aclDataType data_type,
    const int64_t* stride,
    int64_t offset,
    aclFormat format,
    const int64_t* storage_dims,
    uint64_t storage_dims_num,
    void* tensor_data);
using AclDestroyTensor = int (*)(const aclTensor* tensor);
using AclCreateIntArray = aclIntArray* (*)(const int64_t* value, uint64_t size);
using AclDestroyIntArray = aclnnStatus (*)(const aclIntArray* array);

struct Options {
    int device = 0;
    int64_t batch_size = 1;
    int64_t seq_len = 1024;
    int64_t query_tokens = 1;
    int64_t num_heads = 16;
    int64_t num_kv_heads = 8;
    int64_t block_size = 128;
    int64_t pack_tokens = 1;
    int64_t pack_mode = 0;
    int warmup = 10;
    int repeat = 100;
    double scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
    std::string slot_pattern = "contiguous";
    bool run_pack = true;
    bool run_attention = true;
    bool fill_cache = true;
};

[[noreturn]] void Fail(const std::string& message)
{
    throw std::runtime_error(message);
}

void CheckAcl(aclError ret, const std::string& what)
{
    if (ret != ACL_SUCCESS) {
        Fail(what + " failed, ret=" + std::to_string(ret));
    }
}

void CheckAclnn(aclnnStatus ret, const std::string& what)
{
    if (ret != ACL_SUCCESS) {
        const char* recent = aclGetRecentErrMsg();
        std::string msg = what + " failed, ret=" + std::to_string(ret);
        if (recent != nullptr && std::strlen(recent) != 0) {
            msg += ", recent=";
            msg += recent;
        }
        Fail(msg);
    }
}

int64_t ParseInt64(const char* text, const char* name)
{
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        Fail(std::string("invalid integer for ") + name + ": " + text);
    }
    return static_cast<int64_t>(value);
}

int64_t ParsePackMode(const char* text)
{
    const std::string value(text);
    if (value == "general" || value == "owned-groups") {
        return 0;
    }
    if (value == "direct" || value == "decode-direct") {
        return 1;
    }
    if (value == "logical-fast" || value == "logical-fast-fallback" ||
        value == "logical") {
        return 2;
    }
    return ParseInt64(text, "--pack-mode");
}

bool IsKnownSlotPattern(const std::string& pattern)
{
    return pattern == "contiguous" || pattern == "swap-pairs" ||
           pattern == "scatter-groups" || pattern == "reverse";
}

void PrintUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --device N           NPU device id, default 0\n"
        << "  --batch-size N       Number of sequences in the attention batch, default 1\n"
        << "  --seq-len N          KV sequence length, default 1024\n"
        << "  --query-tokens N     Total query tokens for attention, default 1\n"
        << "  --heads N            Query heads, default 16\n"
        << "  --kv-heads N         KV heads, default 8\n"
        << "  --block-size N       Paged cache block size, default 128\n"
        << "  --pack-tokens N      Tokens per pack call, default 1\n"
        << "  --pack-mode MODE     Pack branch: 0/general, 1/direct, 2/logical-fast, default 0\n"
        << "  --slot-pattern NAME  Slot mapping: contiguous, swap-pairs, scatter-groups, reverse, default contiguous\n"
        << "  --warmup N           Warmup iterations, default 10\n"
        << "  --repeat N           Timed iterations, default 100\n"
        << "  --pack-only          Run pack benchmark only\n"
        << "  --attention-only     Run attention benchmark only\n"
        << "  --skip-cache-fill    Do not prefill the attention cache with the pack op\n"
        << "  --help               Show this message\n";
}

Options ParseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                Fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--device") {
            opt.device = static_cast<int>(ParseInt64(need_value("--device"), "--device"));
        } else if (arg == "--batch-size") {
            opt.batch_size = ParseInt64(need_value("--batch-size"), "--batch-size");
        } else if (arg == "--seq-len") {
            opt.seq_len = ParseInt64(need_value("--seq-len"), "--seq-len");
        } else if (arg == "--query-tokens") {
            opt.query_tokens = ParseInt64(need_value("--query-tokens"), "--query-tokens");
        } else if (arg == "--heads") {
            opt.num_heads = ParseInt64(need_value("--heads"), "--heads");
        } else if (arg == "--kv-heads") {
            opt.num_kv_heads = ParseInt64(need_value("--kv-heads"), "--kv-heads");
        } else if (arg == "--block-size") {
            opt.block_size = ParseInt64(need_value("--block-size"), "--block-size");
        } else if (arg == "--pack-tokens") {
            opt.pack_tokens = ParseInt64(need_value("--pack-tokens"), "--pack-tokens");
        } else if (arg == "--pack-mode") {
            opt.pack_mode = ParsePackMode(need_value("--pack-mode"));
        } else if (arg == "--slot-pattern") {
            opt.slot_pattern = need_value("--slot-pattern");
        } else if (arg == "--warmup") {
            opt.warmup = static_cast<int>(ParseInt64(need_value("--warmup"), "--warmup"));
        } else if (arg == "--repeat") {
            opt.repeat = static_cast<int>(ParseInt64(need_value("--repeat"), "--repeat"));
        } else if (arg == "--pack-only") {
            opt.run_pack = true;
            opt.run_attention = false;
        } else if (arg == "--attention-only") {
            opt.run_pack = false;
            opt.run_attention = true;
        } else if (arg == "--skip-cache-fill") {
            opt.fill_cache = false;
        } else {
            Fail("unknown argument: " + arg);
        }
    }

    if (opt.device < 0 || opt.batch_size <= 0 || opt.seq_len <= 0 || opt.query_tokens <= 0 ||
        opt.num_heads <= 0 || opt.num_kv_heads <= 0 || opt.block_size <= 0 ||
        opt.pack_tokens <= 0 || opt.warmup < 0 || opt.repeat <= 0) {
        Fail("shape/count arguments must be positive; warmup must be >= 0");
    }
    if (opt.num_heads % opt.num_kv_heads != 0) {
        Fail("--heads must be divisible by --kv-heads");
    }
    if (opt.block_size % 4 != 0) {
        Fail("--block-size must be a multiple of 4 for the 4bit group layout");
    }
    if (opt.query_tokens % opt.batch_size != 0) {
        Fail("--query-tokens must be divisible by --batch-size");
    }
    if (opt.query_tokens / opt.batch_size > opt.seq_len) {
        Fail("--query-tokens / --batch-size must be <= --seq-len");
    }
    if (opt.pack_tokens > opt.batch_size * opt.seq_len) {
        Fail("--pack-tokens must be <= --batch-size * --seq-len");
    }
    if (opt.pack_mode < 0 || opt.pack_mode > 2) {
        Fail("--pack-mode must be 0/general, 1/direct, or 2/logical-fast");
    }
    if (!IsKnownSlotPattern(opt.slot_pattern)) {
        Fail("--slot-pattern must be one of: contiguous, swap-pairs, scatter-groups, reverse");
    }
    return opt;
}

std::vector<std::string> Glob(const std::string& pattern)
{
    glob_t g {};
    std::vector<std::string> out;
    if (glob(pattern.c_str(), 0, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; ++i) {
            out.emplace_back(g.gl_pathv[i]);
        }
    }
    globfree(&g);
    return out;
}

std::vector<std::string> SplitColonList(const char* value)
{
    std::vector<std::string> out;
    if (value == nullptr || std::strlen(value) == 0) {
        return out;
    }

    const std::string text(value);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t pos = text.find(':', start);
        const std::string item = text.substr(
            start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!item.empty()) {
            out.push_back(item);
        }
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return out;
}

bool TryDlopenGlobal(const std::string& path)
{
    return dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr;
}

void DlopenFirstMatch(const std::vector<std::string>& patterns, const char* lib_name)
{
    for (const auto& pattern : patterns) {
        for (const auto& path : Glob(pattern)) {
            if (TryDlopenGlobal(path)) {
                return;
            }
        }
    }
    if (TryDlopenGlobal(lib_name)) {
        return;
    }
    Fail(std::string("failed to dlopen ") + lib_name + ": " + dlerror());
}

void LoadCustomOpApi()
{
    const char* custom_opp = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    const char* local_custom_opp =
        "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend";
    if (custom_opp == nullptr || std::strlen(custom_opp) == 0) {
        setenv("ASCEND_CUSTOM_OPP_PATH", local_custom_opp, 0);
    } else if (std::string(custom_opp).find(local_custom_opp) == std::string::npos) {
        const std::string updated = std::string(custom_opp) + ":" + local_custom_opp;
        setenv("ASCEND_CUSTOM_OPP_PATH", updated.c_str(), 1);
    }

    std::vector<std::string> lib_patterns;
    for (const auto& opp_path : SplitColonList(std::getenv("ASCEND_CUSTOM_OPP_PATH"))) {
        lib_patterns.push_back(opp_path + "/op_api/lib/libcust_opapi.so");
    }
    lib_patterns.push_back(
        "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so");
    lib_patterns.push_back(
        "csrc/build/_CPack_Packages/Linux/External/CANN-custom_ops--linux.aarch64.run/packages/vendors/vllm-ascend/op_api/lib/libcust_opapi.so");

    DlopenFirstMatch(
        lib_patterns,
        "libcust_opapi.so");
}

void* ResolveOpApiSymbol(const char* name)
{
    static void* cust_handle = dlopen("libcust_opapi.so", RTLD_LAZY | RTLD_LOCAL);
    static void* op_handle = dlopen("libopapi.so", RTLD_LAZY | RTLD_LOCAL);
    if (cust_handle != nullptr) {
        void* func = dlsym(cust_handle, name);
        if (func != nullptr) {
            return func;
        }
    }
    if (op_handle != nullptr) {
        void* func = dlsym(op_handle, name);
        if (func != nullptr) {
            return func;
        }
    }
    return nullptr;
}

template <typename Fn>
Fn ResolveRequired(const char* name)
{
    void* func = ResolveOpApiSymbol(name);
    if (func == nullptr) {
        Fail(std::string("failed to resolve opapi symbol: ") + name);
    }
    return reinterpret_cast<Fn>(func);
}

uint16_t FloatToHalfBits(float value)
{
    union {
        float f;
        uint32_t u;
    } in {};
    in.f = value;

    const uint32_t sign = (in.u >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((in.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                 ((mant + 0x1000u) >> 13));
}

std::vector<uint16_t> RandomHalfVector(size_t count, float scale, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<uint16_t> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = FloatToHalfBits(dist(rng));
    }
    return out;
}

std::vector<uint16_t> Codebook()
{
    const float values[kCodebookSize] = {
        -0.2255f, -0.1651f, -0.1251f, -0.0941f,
        -0.0687f, -0.0468f, -0.0271f, -0.0085f,
         0.0093f,  0.0275f,  0.0469f,  0.0688f,
         0.0941f,  0.1242f,  0.1628f,  0.2206f,
    };
    std::vector<uint16_t> out(kCodebookSize);
    for (int64_t i = 0; i < kCodebookSize; ++i) {
        out[i] = FloatToHalfBits(values[i]);
    }
    return out;
}

std::vector<uint16_t> IdentityRotation()
{
    std::vector<uint16_t> out(kHeadSize * kHeadSize, FloatToHalfBits(0.0f));
    for (int64_t i = 0; i < kHeadSize; ++i) {
        out[i * kHeadSize + i] = FloatToHalfBits(1.0f);
    }
    return out;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t nbytes) { Reset(nbytes); }
    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void Reset(size_t nbytes)
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
        bytes = nbytes;
        if (bytes != 0) {
            CheckAcl(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        }
    }

    void CopyFromHost(const void* host, size_t nbytes)
    {
        if (nbytes > bytes) {
            Fail("host copy exceeds device buffer size");
        }
        CheckAcl(aclrtMemcpy(ptr, bytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy H2D");
    }

    void MemsetZero()
    {
        if (bytes != 0) {
            CheckAcl(aclrtMemset(ptr, bytes, 0, bytes), "aclrtMemset");
        }
    }
};

struct AclTensorGuard {
    aclTensor* tensor = nullptr;

    AclTensorGuard() = default;
    AclTensorGuard(
        void* data,
        aclDataType dtype,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& strides)
    {
        Reset(data, dtype, dims, strides);
    }
    ~AclTensorGuard()
    {
        if (tensor != nullptr) {
            auto destroy = reinterpret_cast<AclDestroyTensor>(
                ResolveOpApiSymbol("aclDestroyTensor"));
            if (destroy != nullptr) {
                destroy(tensor);
            }
        }
    }

    AclTensorGuard(const AclTensorGuard&) = delete;
    AclTensorGuard& operator=(const AclTensorGuard&) = delete;

    void Reset(
        void* data,
        aclDataType dtype,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& strides)
    {
        auto create = ResolveRequired<AclCreateTensor>("aclCreateTensor");
        int64_t storage_elems = 1;
        for (int64_t d : dims) {
            storage_elems *= d;
        }
        tensor = create(
            dims.data(),
            dims.size(),
            dtype,
            strides.data(),
            0,
            ACL_FORMAT_ND,
            &storage_elems,
            1,
            data);
        if (tensor == nullptr) {
            Fail("aclCreateTensor returned null");
        }
    }
};

struct AclIntArrayGuard {
    aclIntArray* array = nullptr;

    explicit AclIntArrayGuard(const std::vector<int64_t>& values)
    {
        auto create = ResolveRequired<AclCreateIntArray>("aclCreateIntArray");
        array = create(values.data(), values.size());
        if (array == nullptr) {
            Fail("aclCreateIntArray returned null");
        }
    }

    ~AclIntArrayGuard()
    {
        if (array != nullptr) {
            auto destroy = reinterpret_cast<AclDestroyIntArray>(
                ResolveOpApiSymbol("aclDestroyIntArray"));
            if (destroy != nullptr) {
                destroy(array);
            }
        }
    }

    AclIntArrayGuard(const AclIntArrayGuard&) = delete;
    AclIntArrayGuard& operator=(const AclIntArrayGuard&) = delete;
};

struct StreamGuard {
    aclrtStream stream = nullptr;

    StreamGuard()
    {
        CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    }

    ~StreamGuard()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
            aclrtDestroyStream(stream);
        }
    }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
};

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

void RunPack4bit(
    const AclTensorGuard& key,
    const AclTensorGuard& value,
    const AclTensorGuard& slot_mapping,
    const AclTensorGuard& codebook,
    const AclTensorGuard& rotation_t,
    const AclTensorGuard& key_cache,
    const AclTensorGuard& value_cache,
    int64_t pack_mode,
    int64_t n_vec,
    int64_t vec_per_core,
    int64_t num_heads,
    int64_t block_size,
    int64_t num_blocks,
    aclrtStream stream)
{
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CheckAclnn(
        aclnnTurboquantPackKvForCache4bitGetWorkspaceSize(
            key.tensor,
            value.tensor,
            codebook.tensor,
            rotation_t.tensor,
            slot_mapping.tensor,
            pack_mode,
            n_vec,
            vec_per_core,
            num_heads,
            block_size,
            num_blocks,
            key_cache.tensor,
            value_cache.tensor,
            &workspace_size,
            &executor),
        "aclnnTurboquantPackKvForCache4bitGetWorkspaceSize");

    DeviceBuffer workspace(workspace_size);
    CheckAclnn(
        aclnnTurboquantPackKvForCache4bit(workspace.ptr, workspace_size, executor, stream),
        "aclnnTurboquantPackKvForCache4bit");
}

int64_t SlotPositionForPattern(
    int64_t pos,
    int64_t seq_len,
    const std::string& slot_pattern)
{
    if (slot_pattern == "swap-pairs") {
        if ((pos % 2) == 0) {
            return (pos + 1 < seq_len) ? pos + 1 : pos;
        }
        return pos - 1;
    }
    if (slot_pattern == "scatter-groups") {
        return (pos * 5) % seq_len;
    }
    if (slot_pattern == "reverse") {
        return seq_len - 1 - pos;
    }
    return pos;
}

void RunAttention4bit(
    const AclTensorGuard& query,
    const AclTensorGuard& key_cache,
    const AclTensorGuard& value_cache,
    const AclTensorGuard& block_table,
    const AclIntArrayGuard& actual_seq_q,
    const AclIntArrayGuard& actual_seq_kv,
    const AclTensorGuard& codebook,
    const AclTensorGuard& rotation,
    const AclTensorGuard& output,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t max_actual_seq_len,
    double scale,
    aclrtStream stream)
{
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CheckAclnn(
        aclnnTurboquantAttentionPaged4bitGetWorkspaceSize(
            query.tensor,
            key_cache.tensor,
            value_cache.tensor,
            block_table.tensor,
            actual_seq_q.array,
            actual_seq_kv.array,
            codebook.tensor,
            rotation.tensor,
            codebook.tensor,
            rotation.tensor,
            num_heads,
            num_kv_heads,
            kHeadSize,
            block_size,
            max_actual_seq_len,
            scale,
            output.tensor,
            &workspace_size,
            &executor),
        "aclnnTurboquantAttentionPaged4bitGetWorkspaceSize");

    DeviceBuffer workspace(workspace_size);
    CheckAclnn(
        aclnnTurboquantAttentionPaged4bit(workspace.ptr, workspace_size, executor, stream),
        "aclnnTurboquantAttentionPaged4bit");
}

template <typename Fn>
double BenchUs(Fn&& fn, int warmup, int repeat, aclrtStream stream)
{
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream warmup");

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        fn();
    }
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream repeat");
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() /
           static_cast<double>(repeat);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string phase = "startup";
    try {
        phase = "parse arguments";
        const Options opt = ParseArgs(argc, argv);

        phase = "load custom op api";
        LoadCustomOpApi();

        phase = "acl init";
        CheckAcl(aclInit(nullptr), "aclInit");
        CheckAcl(aclrtSetDevice(opt.device), "aclrtSetDevice");
        StreamGuard stream;

        const int64_t blocks_per_seq = (opt.seq_len + opt.block_size - 1) / opt.block_size;
        const int64_t total_blocks = opt.batch_size * blocks_per_seq;
        const int64_t cache_fill_tokens = opt.batch_size * opt.seq_len;
        const int64_t query_tokens_per_seq = opt.query_tokens / opt.batch_size;
        const int64_t cache_row_span = opt.block_size * kRowBytes4bit;
        const int64_t full_n_vec = cache_fill_tokens * opt.num_kv_heads;
        const int64_t pack_n_vec = opt.pack_tokens * opt.num_kv_heads;
        const int64_t vec_per_core = pack_n_vec < 128
            ? std::max<int64_t>(16, ((pack_n_vec + 15) / 16) * 16)
            : 128;

        phase = "prepare host data";
        auto key_host = RandomHalfVector(full_n_vec * kHeadSize, 0.02f, 20260619u);
        auto value_host = RandomHalfVector(full_n_vec * kHeadSize, 0.02f, 20260620u);
        auto query_host = RandomHalfVector(
            opt.query_tokens * opt.num_heads * kHeadSize, 0.02f, 20260621u);
        auto codebook_host = Codebook();
        auto rotation_host = IdentityRotation();

        std::vector<int32_t> slot_mapping_host(static_cast<size_t>(cache_fill_tokens));
        for (int64_t seq = 0; seq < opt.batch_size; ++seq) {
            const int64_t slot_base = seq * blocks_per_seq * opt.block_size;
            for (int64_t pos = 0; pos < opt.seq_len; ++pos) {
                const int64_t slot_pos =
                    SlotPositionForPattern(pos, opt.seq_len, opt.slot_pattern);
                slot_mapping_host[static_cast<size_t>(seq * opt.seq_len + pos)] =
                    static_cast<int32_t>(slot_base + slot_pos);
            }
        }
        std::vector<int32_t> block_table_host(static_cast<size_t>(opt.batch_size * blocks_per_seq));
        for (int64_t seq = 0; seq < opt.batch_size; ++seq) {
            for (int64_t block = 0; block < blocks_per_seq; ++block) {
                block_table_host[static_cast<size_t>(seq * blocks_per_seq + block)] =
                    static_cast<int32_t>(seq * blocks_per_seq + block);
            }
        }

        phase = "allocate device buffers";
        DeviceBuffer key_dev(key_host.size() * sizeof(uint16_t));
        DeviceBuffer value_dev(value_host.size() * sizeof(uint16_t));
        DeviceBuffer query_dev(query_host.size() * sizeof(uint16_t));
        DeviceBuffer codebook_dev(codebook_host.size() * sizeof(uint16_t));
        DeviceBuffer rotation_dev(rotation_host.size() * sizeof(uint16_t));
        DeviceBuffer slot_mapping_dev(slot_mapping_host.size() * sizeof(int32_t));
        DeviceBuffer block_table_dev(block_table_host.size() * sizeof(int32_t));
        DeviceBuffer key_cache_dev(total_blocks * opt.num_kv_heads * cache_row_span);
        DeviceBuffer value_cache_dev(total_blocks * opt.num_kv_heads * cache_row_span);
        DeviceBuffer pack_key_cache_dev(total_blocks * opt.num_kv_heads * cache_row_span);
        DeviceBuffer pack_value_cache_dev(total_blocks * opt.num_kv_heads * cache_row_span);
        DeviceBuffer output_dev(opt.query_tokens * opt.num_heads * kHeadSize * sizeof(uint16_t));

        phase = "copy host data to device";
        key_dev.CopyFromHost(key_host.data(), key_dev.bytes);
        value_dev.CopyFromHost(value_host.data(), value_dev.bytes);
        query_dev.CopyFromHost(query_host.data(), query_dev.bytes);
        codebook_dev.CopyFromHost(codebook_host.data(), codebook_dev.bytes);
        rotation_dev.CopyFromHost(rotation_host.data(), rotation_dev.bytes);
        slot_mapping_dev.CopyFromHost(slot_mapping_host.data(), slot_mapping_dev.bytes);
        block_table_dev.CopyFromHost(block_table_host.data(), block_table_dev.bytes);
        key_cache_dev.MemsetZero();
        value_cache_dev.MemsetZero();
        pack_key_cache_dev.MemsetZero();
        pack_value_cache_dev.MemsetZero();
        output_dev.MemsetZero();

        phase = "create acl tensors";
        AclTensorGuard key_acl(
            key_dev.ptr,
            ACL_FLOAT16,
            {cache_fill_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard value_acl(
            value_dev.ptr,
            ACL_FLOAT16,
            {cache_fill_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard pack_key_acl(
            key_dev.ptr,
            ACL_FLOAT16,
            {opt.pack_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard pack_value_acl(
            value_dev.ptr,
            ACL_FLOAT16,
            {opt.pack_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard query_acl(
            query_dev.ptr,
            ACL_FLOAT16,
            {opt.query_tokens, opt.num_heads, kHeadSize},
            {opt.num_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard codebook_acl(
            codebook_dev.ptr, ACL_FLOAT16, {kCodebookSize}, {1});
        AclTensorGuard rotation_acl(
            rotation_dev.ptr, ACL_FLOAT16, {kHeadSize, kHeadSize}, {kHeadSize, 1});
        AclTensorGuard slot_mapping_acl(
            slot_mapping_dev.ptr, ACL_INT32, {cache_fill_tokens}, {1});
        AclTensorGuard pack_slot_mapping_acl(
            slot_mapping_dev.ptr, ACL_INT32, {opt.pack_tokens}, {1});
        AclTensorGuard block_table_acl(
            block_table_dev.ptr, ACL_INT32, {opt.batch_size, blocks_per_seq}, {blocks_per_seq, 1});
        AclTensorGuard key_cache_acl(
            key_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, cache_row_span},
            {opt.num_kv_heads * cache_row_span, cache_row_span, 1});
        AclTensorGuard value_cache_acl(
            value_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, cache_row_span},
            {opt.num_kv_heads * cache_row_span, cache_row_span, 1});
        AclTensorGuard pack_key_cache_acl(
            pack_key_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, cache_row_span},
            {opt.num_kv_heads * cache_row_span, cache_row_span, 1});
        AclTensorGuard pack_value_cache_acl(
            pack_value_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, cache_row_span},
            {opt.num_kv_heads * cache_row_span, cache_row_span, 1});
        AclTensorGuard output_acl(
            output_dev.ptr,
            ACL_FLOAT16,
            {opt.query_tokens, opt.num_heads, kHeadSize},
            {opt.num_heads * kHeadSize, kHeadSize, 1});
        std::vector<int64_t> actual_seq_q_host(static_cast<size_t>(opt.batch_size));
        std::vector<int64_t> actual_seq_kv_host(static_cast<size_t>(opt.batch_size), opt.seq_len);
        for (int64_t seq = 0; seq < opt.batch_size; ++seq) {
            actual_seq_q_host[static_cast<size_t>(seq)] = (seq + 1) * query_tokens_per_seq;
        }
        AclIntArrayGuard actual_seq_q(actual_seq_q_host);
        AclIntArrayGuard actual_seq_kv(actual_seq_kv_host);

        std::cout << "device=npu:" << opt.device
                  << " batch_size=" << opt.batch_size
                  << " seq_len=" << opt.seq_len
                  << " query_tokens=" << opt.query_tokens
                  << " query_tokens_per_seq=" << query_tokens_per_seq
                  << " heads=" << opt.num_heads
                  << " kv_heads=" << opt.num_kv_heads
                  << " block_size=" << opt.block_size
                  << " blocks_per_seq=" << blocks_per_seq
                  << " total_blocks=" << total_blocks
                  << " pack_tokens=" << opt.pack_tokens
                  << " pack_mode=" << opt.pack_mode
                  << " slot_pattern=" << opt.slot_pattern
                  << " fill_cache=" << static_cast<int>(opt.fill_cache)
                  << " warmup=" << opt.warmup
                  << " repeat=" << opt.repeat
                  << "\n";

        if (opt.fill_cache) {
            phase = "fill full cache";
            RunPack4bit(
                key_acl,
                value_acl,
                slot_mapping_acl,
                codebook_acl,
                rotation_acl,
                key_cache_acl,
                value_cache_acl,
                0,
                full_n_vec,
                128,
                opt.num_kv_heads,
                opt.block_size,
                total_blocks,
                stream.stream);
            CheckAcl(aclrtSynchronizeStream(stream.stream), "aclrtSynchronizeStream fill cache");
        }

        if (opt.run_pack) {
            phase = "benchmark pack";
            const double pack_us = BenchUs(
                [&]() {
                    RunPack4bit(
                        pack_key_acl,
                        pack_value_acl,
                        pack_slot_mapping_acl,
                        codebook_acl,
                        rotation_acl,
                        pack_key_cache_acl,
                        pack_value_cache_acl,
                        opt.pack_mode,
                        pack_n_vec,
                        vec_per_core,
                        opt.num_kv_heads,
                        opt.block_size,
                        total_blocks,
                        stream.stream);
                },
                opt.warmup,
                opt.repeat,
                stream.stream);
            std::cout << std::fixed << std::setprecision(2)
                      << "pack4bit_to_cache_avg_us=" << pack_us << "\n";
        }

        if (opt.run_attention) {
            phase = "benchmark attention";
            const double attn_us = BenchUs(
                [&]() {
                    RunAttention4bit(
                        query_acl,
                        key_cache_acl,
                        value_cache_acl,
                        block_table_acl,
                        actual_seq_q,
                        actual_seq_kv,
                        codebook_acl,
                        rotation_acl,
                        output_acl,
                        opt.num_heads,
                        opt.num_kv_heads,
                        opt.block_size,
                        opt.seq_len,
                        opt.scale,
                        stream.stream);
                },
                opt.warmup,
                opt.repeat,
                stream.stream);
            std::cout << std::fixed << std::setprecision(2)
                      << "attention_paged4bit_avg_us=" << attn_us << "\n";
        }

        phase = "cleanup";
        CheckAcl(aclrtResetDevice(opt.device), "aclrtResetDevice");
        CheckAcl(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR during " << phase << ": " << e.what() << "\n";
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
}
