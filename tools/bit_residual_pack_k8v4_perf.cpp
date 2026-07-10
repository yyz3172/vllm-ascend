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

#include "bit_residual_pack_k8v4/op_host/aclnn_bit_residual_pack_k8v4.h"
#include "bit_residual_attention_paged_k8v4/op_host/aclnn_bit_residual_attention_paged_k8v4.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHeadSize = 128;
constexpr int64_t kKeyGroupStride = 288;
constexpr int64_t kValueGroupStride = 288;

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
    std::vector<int64_t> q_lens {};
    std::vector<int64_t> kv_lens {};
    int64_t num_heads = 16;
    int64_t num_kv_heads = 8;
    int64_t block_size = 128;
    int64_t pack_tokens = -1;
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

bool IsKnownSlotPattern(const std::string& pattern)
{
    return pattern == "contiguous" || pattern == "swap-pairs" ||
           pattern == "scatter-groups" || pattern == "reverse";
}

std::vector<int64_t> ParseLensList(const char* text, const char* name)
{
    std::vector<int64_t> out;
    if (text == nullptr || std::strlen(text) == 0) {
        Fail(std::string("empty value for ") + name);
    }
    const std::string str(text);
    size_t start = 0;
    while (start <= str.size()) {
        const size_t pos = str.find(':', start);
        const std::string item = str.substr(
            start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!item.empty()) {
            out.push_back(ParseInt64(item.c_str(), name));
        } else {
            Fail(std::string("empty segment in ") + name);
        }
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    if (out.empty()) {
        Fail(std::string("no values parsed for ") + name);
    }
    return out;
}

void PrintUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --q-lens L[:L...] --kv-lens L[:L...] [options]\n\n"
        << "Variable-length mixed-batch benchmark for BitResidualPackK8v4 + Attention.\n\n"
        << "Required:\n"
        << "  --q-lens L[:L...]    Per-request query token counts, colon-separated\n"
        << "  --kv-lens L[:L...]   Per-request KV token counts, colon-separated\n\n"
        << "Options:\n"
        << "  --device N           NPU device id, default 0\n"
        << "  --heads N            Query heads, default 16\n"
        << "  --kv-heads N         KV heads, default 8\n"
        << "  --block-size N       Paged cache block size, default 128 (must be divisible by 4)\n"
        << "  --pack-tokens N      Tokens per pack call, default sum(q_lens)\n"
        << "  --slot-pattern NAME  Slot mapping: contiguous, swap-pairs, scatter-groups, reverse, default contiguous\n"
        << "  --warmup N           Warmup iterations, default 10\n"
        << "  --repeat N           Timed iterations, default 100\n"
        << "  --pack-only          Run pack benchmark only\n"
        << "  --attention-only     Run attention benchmark only\n"
        << "  --skip-cache-fill    Do not prefill the attention cache with the pack op\n"
        << "  --help               Show this message\n\n"
        << "Examples:\n"
        << "  # 1 prefill (128 tok) + 7 decode (1 tok each), kv=512/2048:\n"
        << "  " << argv0 << " --q-lens 128:1:1:1:1:1:1:1 --kv-lens 512:2048:2048:2048:2048:2048:2048:2048\n\n"
        << "  # All decode (8 requests, 1 tok each):\n"
        << "  " << argv0 << " --q-lens 1:1:1:1:1:1:1:1 --kv-lens 2048:2048:2048:2048:2048:2048:2048:2048\n";
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
        } else if (arg == "--q-lens") {
            opt.q_lens = ParseLensList(need_value("--q-lens"), "--q-lens");
        } else if (arg == "--kv-lens") {
            opt.kv_lens = ParseLensList(need_value("--kv-lens"), "--kv-lens");
        } else if (arg == "--heads") {
            opt.num_heads = ParseInt64(need_value("--heads"), "--heads");
        } else if (arg == "--kv-heads") {
            opt.num_kv_heads = ParseInt64(need_value("--kv-heads"), "--kv-heads");
        } else if (arg == "--block-size") {
            opt.block_size = ParseInt64(need_value("--block-size"), "--block-size");
        } else if (arg == "--pack-tokens") {
            opt.pack_tokens = ParseInt64(need_value("--pack-tokens"), "--pack-tokens");
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

    if (opt.q_lens.empty()) {
        Fail("--q-lens is required");
    }
    if (opt.kv_lens.empty()) {
        Fail("--kv-lens is required");
    }
    if (opt.q_lens.size() != opt.kv_lens.size()) {
        Fail("--q-lens and --kv-lens must have the same number of elements");
    }

    const int64_t batch_size = static_cast<int64_t>(opt.q_lens.size());
    if (opt.device < 0 || batch_size <= 0 || opt.num_heads <= 0 ||
        opt.num_kv_heads <= 0 || opt.block_size <= 0 || opt.pack_tokens == 0 ||
        opt.pack_tokens < -1 || opt.warmup < 0 || opt.repeat <= 0) {
        Fail("shape/count arguments must be positive; warmup must be >= 0");
    }
    if (opt.num_heads % opt.num_kv_heads != 0) {
        Fail("--heads must be divisible by --kv-heads");
    }
    if (opt.num_kv_heads < 8 || opt.num_kv_heads % 8 != 0) {
        Fail("--kv-heads must be >= 8 and divisible by 8");
    }
    if (opt.block_size % 4 != 0) {
        Fail("--block-size must be divisible by 4");
    }
    if (!IsKnownSlotPattern(opt.slot_pattern)) {
        Fail("--slot-pattern must be one of: contiguous, swap-pairs, scatter-groups, reverse");
    }

    for (int64_t i = 0; i < batch_size; ++i) {
        if (opt.q_lens[i] <= 0) {
            Fail("--q-lens[" + std::to_string(i) + "] must be positive");
        }
        if (opt.kv_lens[i] <= 0) {
            Fail("--kv-lens[" + std::to_string(i) + "] must be positive");
        }
        if (opt.q_lens[i] > opt.kv_lens[i]) {
            Fail("--q-lens[" + std::to_string(i) + "] must be <= --kv-lens[" +
                  std::to_string(i) + "]");
        }
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

    DlopenFirstMatch(lib_patterns, "libcust_opapi.so");
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

void RunPackK8v4(
    const AclTensorGuard& key,
    const AclTensorGuard& value,
    const AclTensorGuard& rotation_t,
    const AclTensorGuard& slot_mapping,
    const AclTensorGuard& query_start_loc,
    const AclTensorGuard& key_cache,
    const AclTensorGuard& value_cache,
    int64_t num_reqs,
    int64_t n_vec,
    int64_t vec_per_core,
    int64_t num_heads,
    int64_t block_size,
    int64_t num_blocks,
    aclrtStream stream)
{
    const int64_t key_stride_token = num_heads * kHeadSize;
    const int64_t key_stride_head = kHeadSize;
    const int64_t value_stride_token = num_heads * kHeadSize;
    const int64_t value_stride_head = kHeadSize;
    const int64_t key_storage_offset = 0;
    const int64_t value_storage_offset = 0;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CheckAclnn(
        aclnnBitResidualPackK8v4GetWorkspaceSize(
            key.tensor,
            value.tensor,
            rotation_t.tensor,
            slot_mapping.tensor,
            query_start_loc.tensor,
            n_vec,
            vec_per_core,
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
            key_cache.tensor,
            value_cache.tensor,
            &workspace_size,
            &executor),
        "aclnnBitResidualPackK8v4GetWorkspaceSize");

    DeviceBuffer workspace(workspace_size);
    CheckAclnn(
        aclnnBitResidualPackK8v4(workspace.ptr, workspace_size, executor, stream),
        "aclnnBitResidualPackK8v4");
}

void RunAttentionK8v4(
    const AclTensorGuard& query,
    const AclTensorGuard& key_cache,
    const AclTensorGuard& value_cache,
    const AclTensorGuard& block_table,
    const AclIntArrayGuard& actual_seq_q,
    const AclIntArrayGuard& actual_seq_kv,
    const AclTensorGuard& rotation_key,
    const AclTensorGuard& rotation_value,
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
        aclnnBitResidualAttentionPagedK8v4GetWorkspaceSize(
            query.tensor,
            key_cache.tensor,
            value_cache.tensor,
            block_table.tensor,
            actual_seq_q.array,
            actual_seq_kv.array,
            rotation_key.tensor,
            rotation_value.tensor,
            num_heads,
            num_kv_heads,
            kHeadSize,
            block_size,
            max_actual_seq_len,
            scale,
            output.tensor,
            &workspace_size,
            &executor),
        "aclnnBitResidualAttentionPagedK8v4GetWorkspaceSize");

    DeviceBuffer workspace(workspace_size);
    CheckAclnn(
        aclnnBitResidualAttentionPagedK8v4(workspace.ptr, workspace_size, executor, stream),
        "aclnnBitResidualAttentionPagedK8v4");
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

std::string JoinInts(const std::vector<int64_t>& vals, const std::string& sep)
{
    std::string out;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i > 0) {
            out += sep;
        }
        out += std::to_string(vals[i]);
    }
    return out;
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

        const int64_t batch_size = static_cast<int64_t>(opt.q_lens.size());

        std::vector<int64_t> blocks_per_seq(batch_size);
        int64_t max_kv_len = 0;
        int64_t total_kv_tokens = 0;
        int64_t total_query_tokens = 0;
        int64_t total_blocks = 0;
        for (int64_t i = 0; i < batch_size; ++i) {
            blocks_per_seq[i] = (opt.kv_lens[i] + opt.block_size - 1) / opt.block_size;
            max_kv_len = std::max(max_kv_len, opt.kv_lens[i]);
            total_kv_tokens += opt.kv_lens[i];
            total_query_tokens += opt.q_lens[i];
            total_blocks += blocks_per_seq[i];
        }

        const int64_t key_cache_row_span = (opt.block_size / 2) * kKeyGroupStride;
        const int64_t value_cache_row_span = (opt.block_size / 4) * kValueGroupStride;
        const int64_t full_n_vec = total_kv_tokens * opt.num_kv_heads;
        const bool explicit_pack_tokens = opt.pack_tokens > 0;
        const int64_t pack_tokens = explicit_pack_tokens ? opt.pack_tokens : total_query_tokens;
        if (pack_tokens > total_kv_tokens) {
            Fail("--pack-tokens must be <= sum(kv_lens)");
        }
        const int64_t pack_n_vec = pack_tokens * opt.num_kv_heads;
        const int64_t vec_per_core = pack_n_vec < 128
            ? std::max<int64_t>(16, ((pack_n_vec + 15) / 16) * 16)
            : 128;

        phase = "prepare host data";
        auto key_host = RandomHalfVector(full_n_vec * kHeadSize, 0.02f, 20260619u);
        auto value_host = RandomHalfVector(full_n_vec * kHeadSize, 0.02f, 20260620u);
        auto pack_key_host = RandomHalfVector(pack_n_vec * kHeadSize, 0.02f, 20260622u);
        auto pack_value_host = RandomHalfVector(pack_n_vec * kHeadSize, 0.02f, 20260623u);
        auto query_host = RandomHalfVector(
            total_query_tokens * opt.num_heads * kHeadSize, 0.02f, 20260621u);
        auto rotation_host = IdentityRotation();

        std::vector<int32_t> slot_mapping_host(static_cast<size_t>(total_kv_tokens));
        std::vector<int32_t> pack_slot_mapping_host(static_cast<size_t>(pack_tokens));
        {
            int64_t block_offset = 0;
            int64_t kv_token_offset = 0;
            int64_t default_pack_token_offset = 0;
            for (int64_t req = 0; req < batch_size; ++req) {
                const int64_t slot_base = block_offset * opt.block_size;
                for (int64_t pos = 0; pos < opt.kv_lens[req]; ++pos) {
                    const int64_t slot_pos =
                        SlotPositionForPattern(pos, opt.kv_lens[req], opt.slot_pattern);
                    slot_mapping_host[static_cast<size_t>(kv_token_offset + pos)] =
                        static_cast<int32_t>(slot_base + slot_pos);
                }
                if (!explicit_pack_tokens) {
                    const int64_t first_pack_pos = opt.kv_lens[req] - opt.q_lens[req];
                    for (int64_t pos = first_pack_pos; pos < opt.kv_lens[req]; ++pos) {
                        const int64_t slot_pos =
                            SlotPositionForPattern(pos, opt.kv_lens[req], opt.slot_pattern);
                        pack_slot_mapping_host[static_cast<size_t>(default_pack_token_offset)] =
                            static_cast<int32_t>(slot_base + slot_pos);
                        ++default_pack_token_offset;
                    }
                }
                kv_token_offset += opt.kv_lens[req];
                block_offset += blocks_per_seq[req];
            }
            if (explicit_pack_tokens) {
                int64_t remaining_pack_tokens = pack_tokens;
                int64_t pack_token_offset = 0;
                block_offset = 0;
                for (int64_t req = 0; req < batch_size && remaining_pack_tokens > 0; ++req) {
                    const int64_t slot_base = block_offset * opt.block_size;
                    const int64_t rows_for_req = std::min<int64_t>(
                        opt.kv_lens[req], remaining_pack_tokens);
                    for (int64_t pos = 0; pos < rows_for_req; ++pos) {
                        const int64_t slot_pos =
                            SlotPositionForPattern(pos, opt.kv_lens[req], opt.slot_pattern);
                        pack_slot_mapping_host[static_cast<size_t>(pack_token_offset)] =
                            static_cast<int32_t>(slot_base + slot_pos);
                        ++pack_token_offset;
                    }
                    remaining_pack_tokens -= rows_for_req;
                    block_offset += blocks_per_seq[req];
                }
            }
        }

        int64_t max_blocks_per_seq = 0;
        for (int64_t i = 0; i < batch_size; ++i) {
            max_blocks_per_seq = std::max(max_blocks_per_seq, blocks_per_seq[i]);
        }

        std::vector<int32_t> block_table_host(
            static_cast<size_t>(batch_size * max_blocks_per_seq), 0);
        {
            int64_t block_offset = 0;
            for (int64_t req = 0; req < batch_size; ++req) {
                for (int64_t block = 0; block < blocks_per_seq[req]; ++block) {
                    block_table_host[static_cast<size_t>(req * max_blocks_per_seq + block)] =
                        static_cast<int32_t>(block_offset + block);
                }
                block_offset += blocks_per_seq[req];
            }
        }

        std::vector<int64_t> actual_seq_q_host(static_cast<size_t>(batch_size));
        std::vector<int64_t> actual_seq_kv_host(static_cast<size_t>(batch_size));
        {
            int64_t cum_q = 0;
            for (int64_t req = 0; req < batch_size; ++req) {
                cum_q += opt.q_lens[req];
                actual_seq_q_host[req] = cum_q;
                actual_seq_kv_host[req] = opt.kv_lens[req];
            }
        }

        std::vector<int32_t> full_query_start_loc_host(static_cast<size_t>(batch_size + 1), 0);
        std::vector<int32_t> pack_query_start_loc_host;
        pack_query_start_loc_host.reserve(static_cast<size_t>(batch_size + 1));
        pack_query_start_loc_host.push_back(0);
        {
            int64_t cum_kv = 0;
            int64_t remaining_pack_tokens = pack_tokens;
            for (int64_t req = 0; req < batch_size; ++req) {
                cum_kv += opt.kv_lens[req];
                if (explicit_pack_tokens) {
                    if (remaining_pack_tokens > 0) {
                        const int64_t rows_for_req = std::min<int64_t>(
                            opt.kv_lens[req], remaining_pack_tokens);
                        pack_query_start_loc_host.push_back(
                            pack_query_start_loc_host.back() +
                            static_cast<int32_t>(rows_for_req));
                        remaining_pack_tokens -= rows_for_req;
                    }
                } else {
                    pack_query_start_loc_host.push_back(
                        pack_query_start_loc_host.back() +
                        static_cast<int32_t>(opt.q_lens[req]));
                }
                full_query_start_loc_host[static_cast<size_t>(req + 1)] =
                    static_cast<int32_t>(cum_kv);
            }
        }
        const int64_t pack_num_reqs =
            static_cast<int64_t>(pack_query_start_loc_host.size()) - 1;

        phase = "allocate device buffers";
        DeviceBuffer key_dev(key_host.size() * sizeof(uint16_t));
        DeviceBuffer value_dev(value_host.size() * sizeof(uint16_t));
        DeviceBuffer pack_key_dev(pack_key_host.size() * sizeof(uint16_t));
        DeviceBuffer pack_value_dev(pack_value_host.size() * sizeof(uint16_t));
        DeviceBuffer query_dev(query_host.size() * sizeof(uint16_t));
        DeviceBuffer rotation_dev(rotation_host.size() * sizeof(uint16_t));
        DeviceBuffer slot_mapping_dev(slot_mapping_host.size() * sizeof(int32_t));
        DeviceBuffer pack_slot_mapping_dev(pack_slot_mapping_host.size() * sizeof(int32_t));
        DeviceBuffer full_query_start_loc_dev(full_query_start_loc_host.size() * sizeof(int32_t));
        DeviceBuffer pack_query_start_loc_dev(pack_query_start_loc_host.size() * sizeof(int32_t));
        DeviceBuffer block_table_dev(block_table_host.size() * sizeof(int32_t));
        DeviceBuffer key_cache_dev(total_blocks * opt.num_kv_heads * key_cache_row_span);
        DeviceBuffer value_cache_dev(total_blocks * opt.num_kv_heads * value_cache_row_span);
        DeviceBuffer pack_key_cache_dev(total_blocks * opt.num_kv_heads * key_cache_row_span);
        DeviceBuffer pack_value_cache_dev(total_blocks * opt.num_kv_heads * value_cache_row_span);
        DeviceBuffer output_dev(total_query_tokens * opt.num_heads * kHeadSize * sizeof(uint16_t));

        phase = "copy host data to device";
        key_dev.CopyFromHost(key_host.data(), key_dev.bytes);
        value_dev.CopyFromHost(value_host.data(), value_dev.bytes);
        pack_key_dev.CopyFromHost(pack_key_host.data(), pack_key_dev.bytes);
        pack_value_dev.CopyFromHost(pack_value_host.data(), pack_value_dev.bytes);
        query_dev.CopyFromHost(query_host.data(), query_dev.bytes);
        rotation_dev.CopyFromHost(rotation_host.data(), rotation_dev.bytes);
        slot_mapping_dev.CopyFromHost(slot_mapping_host.data(), slot_mapping_dev.bytes);
        pack_slot_mapping_dev.CopyFromHost(
            pack_slot_mapping_host.data(), pack_slot_mapping_dev.bytes);
        full_query_start_loc_dev.CopyFromHost(
            full_query_start_loc_host.data(), full_query_start_loc_dev.bytes);
        pack_query_start_loc_dev.CopyFromHost(
            pack_query_start_loc_host.data(), pack_query_start_loc_dev.bytes);
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
            {total_kv_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard value_acl(
            value_dev.ptr,
            ACL_FLOAT16,
            {total_kv_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard pack_key_acl(
            pack_key_dev.ptr,
            ACL_FLOAT16,
            {pack_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard pack_value_acl(
            pack_value_dev.ptr,
            ACL_FLOAT16,
            {pack_tokens, opt.num_kv_heads, kHeadSize},
            {opt.num_kv_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard query_acl(
            query_dev.ptr,
            ACL_FLOAT16,
            {total_query_tokens, opt.num_heads, kHeadSize},
            {opt.num_heads * kHeadSize, kHeadSize, 1});
        AclTensorGuard rotation_acl(
            rotation_dev.ptr, ACL_FLOAT16, {kHeadSize, kHeadSize}, {kHeadSize, 1});
        AclTensorGuard slot_mapping_acl(
            slot_mapping_dev.ptr, ACL_INT32, {total_kv_tokens}, {1});
        AclTensorGuard pack_slot_mapping_acl(
            pack_slot_mapping_dev.ptr, ACL_INT32, {pack_tokens}, {1});
        AclTensorGuard full_query_start_loc_acl(
            full_query_start_loc_dev.ptr, ACL_INT32, {batch_size + 1}, {1});
        AclTensorGuard pack_query_start_loc_acl(
            pack_query_start_loc_dev.ptr, ACL_INT32, {pack_num_reqs + 1}, {1});
        AclTensorGuard block_table_acl(
            block_table_dev.ptr, ACL_INT32, {batch_size, max_blocks_per_seq}, {max_blocks_per_seq, 1});
        AclTensorGuard key_cache_acl(
            key_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, key_cache_row_span},
            {opt.num_kv_heads * key_cache_row_span, key_cache_row_span, 1});
        AclTensorGuard value_cache_acl(
            value_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, value_cache_row_span},
            {opt.num_kv_heads * value_cache_row_span, value_cache_row_span, 1});
        AclTensorGuard pack_key_cache_acl(
            pack_key_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, key_cache_row_span},
            {opt.num_kv_heads * key_cache_row_span, key_cache_row_span, 1});
        AclTensorGuard pack_value_cache_acl(
            pack_value_cache_dev.ptr,
            ACL_UINT8,
            {total_blocks, opt.num_kv_heads, value_cache_row_span},
            {opt.num_kv_heads * value_cache_row_span, value_cache_row_span, 1});
        AclTensorGuard output_acl(
            output_dev.ptr,
            ACL_FLOAT16,
            {total_query_tokens, opt.num_heads, kHeadSize},
            {opt.num_heads * kHeadSize, kHeadSize, 1});
        AclIntArrayGuard actual_seq_q(actual_seq_q_host);
        AclIntArrayGuard actual_seq_kv(actual_seq_kv_host);

        std::cout << "device=npu:" << opt.device
                  << " batch_size=" << batch_size
                  << " q_lens=[" << JoinInts(opt.q_lens, ",") << "]"
                  << " kv_lens=[" << JoinInts(opt.kv_lens, ",") << "]"
                  << " total_query_tokens=" << total_query_tokens
                  << " total_kv_tokens=" << total_kv_tokens
                  << " pack_tokens=" << pack_tokens
                  << " pack_num_reqs=" << pack_num_reqs
                  << " heads=" << opt.num_heads
                  << " kv_heads=" << opt.num_kv_heads
                  << " block_size=" << opt.block_size
                  << " blocks_per_seq=[" << JoinInts(blocks_per_seq, ",") << "]"
                  << " total_blocks=" << total_blocks
                  << " key_cache_row_span=" << key_cache_row_span
                  << " value_cache_row_span=" << value_cache_row_span
                  << " slot_pattern=" << opt.slot_pattern
                  << " fill_cache=" << static_cast<int>(opt.fill_cache)
                  << " run_pack=" << static_cast<int>(opt.run_pack)
                  << " run_attention=" << static_cast<int>(opt.run_attention)
                  << " warmup=" << opt.warmup
                  << " repeat=" << opt.repeat
                  << "\n";

        if (opt.fill_cache) {
            phase = "fill full cache";
            RunPackK8v4(
                key_acl,
                value_acl,
                rotation_acl,
                slot_mapping_acl,
                full_query_start_loc_acl,
                key_cache_acl,
                value_cache_acl,
                batch_size,
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
                    RunPackK8v4(
                        pack_key_acl,
                        pack_value_acl,
                        rotation_acl,
                        pack_slot_mapping_acl,
                        pack_query_start_loc_acl,
                        pack_key_cache_acl,
                        pack_value_cache_acl,
                        pack_num_reqs,
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
                      << "bit_residual_pack_k8v4_avg_us=" << pack_us << "\n";
        }

        if (opt.run_attention) {
            phase = "benchmark attention";
            const double attn_us = BenchUs(
                [&]() {
                    RunAttentionK8v4(
                        query_acl,
                        key_cache_acl,
                        value_cache_acl,
                        block_table_acl,
                        actual_seq_q,
                        actual_seq_kv,
                        rotation_acl,
                        rotation_acl,
                        output_acl,
                        opt.num_heads,
                        opt.num_kv_heads,
                        opt.block_size,
                        max_kv_len,
                        opt.scale,
                        stream.stream);
                },
                opt.warmup,
                opt.repeat,
                stream.stream);
            std::cout << std::fixed << std::setprecision(2)
                      << "bit_residual_attention_paged_k8v4_avg_us=" << attn_us << "\n";
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