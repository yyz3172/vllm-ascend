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

#include <ATen/core/dispatch/Dispatcher.h>
#include <Python.h>
#include <acl/acl_rt.h>
#include <c10/core/Device.h>
#include <dlfcn.h>
#include <glob.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHeadSize = 128;
constexpr int64_t kRowBytes4bit = kHeadSize / 2 + 2;

struct Options {
    int device_id = 0;
    int64_t seq_len = 1024;
    int64_t query_tokens = 1;
    int64_t num_heads = 16;
    int64_t num_kv_heads = 2;
    int64_t block_size = 128;
    int64_t pack_tokens = 1;
    int warmup = 10;
    int repeat = 100;
    double scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
    std::string extension_path;
};

void PrintUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --device N          NPU device id, default 0\n"
        << "  --seq-len N         KV sequence length, default 1024\n"
        << "  --query-tokens N    Query tokens for attention, default 1\n"
        << "  --heads N           Query heads, default 16\n"
        << "  --kv-heads N        KV heads, default 2\n"
        << "  --block-size N      Paged cache block size, default 128\n"
        << "  --pack-tokens N     Tokens per pack-to-cache call, default 1\n"
        << "  --warmup N          Warmup iterations, default 10\n"
        << "  --repeat N          Timed iterations, default 100\n"
        << "  --extension PATH    vllm_ascend_C extension path\n"
        << "  --help              Show this message\n";
}

int64_t ParseInt64(const char* text, const char* name)
{
    char* end = nullptr;
    long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + text);
    }
    return static_cast<int64_t>(value);
}

Options ParseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--device") {
            opt.device_id = static_cast<int>(ParseInt64(need_value("--device"), "--device"));
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
        } else if (arg == "--warmup") {
            opt.warmup = static_cast<int>(ParseInt64(need_value("--warmup"), "--warmup"));
        } else if (arg == "--repeat") {
            opt.repeat = static_cast<int>(ParseInt64(need_value("--repeat"), "--repeat"));
        } else if (arg == "--extension") {
            opt.extension_path = need_value("--extension");
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (opt.seq_len <= 0 || opt.query_tokens <= 0 || opt.block_size <= 0 ||
        opt.num_heads <= 0 || opt.num_kv_heads <= 0 || opt.pack_tokens <= 0) {
        throw std::invalid_argument("shape arguments must be positive");
    }
    if (opt.num_heads % opt.num_kv_heads != 0) {
        throw std::invalid_argument("--heads must be divisible by --kv-heads");
    }
    if (opt.query_tokens > opt.seq_len) {
        throw std::invalid_argument("--query-tokens must be <= --seq-len");
    }
    if (opt.pack_tokens > opt.seq_len) {
        throw std::invalid_argument("--pack-tokens must be <= --seq-len");
    }
    if (opt.warmup < 0 || opt.repeat <= 0) {
        throw std::invalid_argument("--warmup must be >= 0 and --repeat must be > 0");
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

std::string FindExtension()
{
    const std::vector<std::string> patterns = {
        "vllm_ascend/vllm_ascend_C*.so",
        "build/temp.*/vllm_ascend_C*.so",
        "build/lib.*/vllm_ascend/vllm_ascend_C*.so",
        "build/lib.*/*/vllm_ascend_C*.so",
    };
    for (const auto& pattern : patterns) {
        auto matches = Glob(pattern);
        if (!matches.empty()) {
            return matches.front();
        }
    }
    throw std::runtime_error("failed to find vllm_ascend_C*.so; pass --extension PATH");
}

void DlopenGlobal(const std::string& path)
{
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        throw std::runtime_error("dlopen failed for " + path + ": " + dlerror());
    }
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
    throw std::runtime_error(std::string("failed to dlopen ") + lib_name + ": " + dlerror());
}

void RunPython(const char* code)
{
    const int ret = PyRun_SimpleString(code);
    if (ret != 0) {
        throw std::runtime_error("embedded Python command failed");
    }
}

void SyncDevice()
{
    aclError ret = aclrtSynchronizeDevice();
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeDevice failed: " + std::to_string(ret));
    }
}

template <typename Fn>
double BenchUs(Fn&& fn, int warmup, int repeat)
{
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    SyncDevice();

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        fn();
    }
    SyncDevice();
    const auto end = std::chrono::steady_clock::now();
    const double total_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    return total_us / static_cast<double>(repeat);
}

at::Tensor CpuToNpu(const at::Tensor& cpu, const c10::Device& device)
{
    return cpu.to(device, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

}  // namespace

int main(int argc, char** argv)
{
    std::string phase = "startup";
    try {
        phase = "parse arguments";
        const Options opt = ParseArgs(argc, argv);

        setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 0);
        phase = "initialize Python runtime";
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }

        // Loading torch_npu explicitly makes PrivateUse1/NPU kernels available
        // even when this binary is launched outside Python.
        phase = "load torch global deps";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/python*/site-packages/torch/lib/libtorch_global_deps.so",
                "/usr/local/lib/python*/site-packages/torch/lib/libtorch_global_deps.so",
                "/usr/lib/python*/site-packages/torch/lib/libtorch_global_deps.so",
            },
            "libtorch_global_deps.so");
        phase = "load libpython";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/libpython3*.so*",
                "/usr/local/lib/libpython3*.so*",
                "/usr/lib*/libpython3*.so*",
            },
            "libpython3.11.so");
        phase = "load libc10";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/python*/site-packages/torch/lib/libc10.so",
                "/usr/local/lib/python*/site-packages/torch/lib/libc10.so",
                "/usr/lib/python*/site-packages/torch/lib/libc10.so",
            },
            "libc10.so");
        phase = "load libtorch";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/python*/site-packages/torch/lib/libtorch.so",
                "/usr/local/lib/python*/site-packages/torch/lib/libtorch.so",
                "/usr/lib/python*/site-packages/torch/lib/libtorch.so",
            },
            "libtorch.so");
        phase = "load libtorch_cpu";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/python*/site-packages/torch/lib/libtorch_cpu.so",
                "/usr/local/lib/python*/site-packages/torch/lib/libtorch_cpu.so",
                "/usr/lib/python*/site-packages/torch/lib/libtorch_cpu.so",
            },
            "libtorch_cpu.so");
        phase = "load libtorch_python";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/python*/site-packages/torch/lib/libtorch_python.so",
                "/usr/local/lib/python*/site-packages/torch/lib/libtorch_python.so",
                "/usr/lib/python*/site-packages/torch/lib/libtorch_python.so",
            },
            "libtorch_python.so");
        phase = "initialize torch_npu Python modules";
        RunPython(
            "import os\n"
            "os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')\n"
            "import torch\n"
            "import torch_npu\n");
        phase = "promote libtorch_npu symbols";
        DlopenFirstMatch(
            {
                "/usr/local/python*/lib/python*/site-packages/torch_npu/lib/libtorch_npu.so",
                "/usr/local/lib/python*/site-packages/torch_npu/lib/libtorch_npu.so",
                "/usr/lib/python*/site-packages/torch_npu/lib/libtorch_npu.so",
            },
            "libtorch_npu.so");

        phase = "find extension";
        const std::string extension =
            opt.extension_path.empty() ? FindExtension() : opt.extension_path;
        phase = "load vllm_ascend_C extension";
        DlopenGlobal(extension);

        phase = "prepare shapes";
        const c10::Device device(c10::DeviceType::PrivateUse1, opt.device_id);
        const int64_t num_blocks = (opt.seq_len + opt.block_size - 1) / opt.block_size;
        const int64_t cache_slots = num_blocks * opt.block_size;
        (void)cache_slots;

        using PackOp = void(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            at::Tensor&,
            at::Tensor&,
            int64_t);
        using AttentionOp = at::Tensor(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            at::IntArrayRef,
            at::IntArrayRef,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            int64_t,
            int64_t,
            int64_t,
            int64_t,
            int64_t,
            double);

        phase = "lookup torch op schemas";
        auto pack_op = c10::Dispatcher::singleton()
                           .findSchemaOrThrow(
                               "_C_ascend::turboquant_pack_kv_for_cache_4bit", "")
                           .typed<PackOp>();
        auto attention_op = c10::Dispatcher::singleton()
                                .findSchemaOrThrow(
                                    "_C_ascend::turboquant_attention_paged4bit", "")
                                .typed<AttentionOp>();

        const auto cpu_f32 = at::TensorOptions().device(c10::kCPU).dtype(at::kFloat);
        const auto npu_f16 = at::TensorOptions().device(device).dtype(at::kHalf);
        const auto npu_u8 = at::TensorOptions().device(device).dtype(at::kByte);

        phase = "generate random CPU tensors";
        at::manual_seed(20260618);
        auto key_cpu = at::randn({opt.seq_len, opt.num_kv_heads, kHeadSize}, cpu_f32);
        auto value_cpu = at::randn({opt.seq_len, opt.num_kv_heads, kHeadSize}, cpu_f32);
        auto query_cpu = at::randn({opt.query_tokens, opt.num_heads, kHeadSize}, cpu_f32);

        phase = "copy random tensors to NPU";
        auto key = CpuToNpu(key_cpu.to(at::kHalf), device);
        auto value = CpuToNpu(value_cpu.to(at::kHalf), device);
        auto query = CpuToNpu(query_cpu.to(at::kHalf), device);
        auto pack_key = key.narrow(0, 0, opt.pack_tokens).contiguous();
        auto pack_value = value.narrow(0, 0, opt.pack_tokens).contiguous();

        phase = "create codebook and metadata tensors";
        auto codebook = CpuToNpu(
            at::linspace(-1.0, 1.0, 16, cpu_f32).to(at::kHalf), device);
        auto rotation = CpuToNpu(
            at::eye(kHeadSize, cpu_f32).to(at::kHalf), device);
        auto rotation_t = rotation;

        auto slot_mapping = CpuToNpu(
            at::arange(opt.seq_len, at::TensorOptions().device(c10::kCPU).dtype(at::kInt)),
            device);
        auto pack_slot_mapping = slot_mapping.narrow(0, 0, opt.pack_tokens).contiguous();
        auto block_table = CpuToNpu(
            at::arange(num_blocks, at::TensorOptions().device(c10::kCPU).dtype(at::kInt))
                .view({1, num_blocks}),
            device);

        phase = "allocate cache tensors";
        auto key_cache = at::zeros(
            {num_blocks, opt.num_kv_heads, opt.block_size * kRowBytes4bit}, npu_u8);
        auto value_cache = at::zeros_like(key_cache);
        auto pack_key_cache = at::zeros(
            {num_blocks, opt.num_kv_heads, opt.block_size * kRowBytes4bit}, npu_u8);
        auto pack_value_cache = at::zeros_like(pack_key_cache);

        std::vector<int64_t> actual_q {opt.query_tokens};
        std::vector<int64_t> actual_kv {opt.seq_len};

        std::cout << "Loaded extension: " << extension << "\n"
                  << "device=npu:" << opt.device_id
                  << " seq_len=" << opt.seq_len
                  << " query_tokens=" << opt.query_tokens
                  << " heads=" << opt.num_heads
                  << " kv_heads=" << opt.num_kv_heads
                  << " block_size=" << opt.block_size
                  << " num_blocks=" << num_blocks
                  << " pack_tokens=" << opt.pack_tokens
                  << " warmup=" << opt.warmup
                  << " repeat=" << opt.repeat << "\n";

        // Fill full cache once so the attention benchmark reads quantized KV data.
        phase = "fill full 4bit cache";
        pack_op.call(
            key,
            value,
            slot_mapping,
            codebook,
            rotation_t,
            key_cache,
            value_cache,
            opt.block_size);
        SyncDevice();

        phase = "benchmark 4bit pack";
        const double pack_us = BenchUs(
            [&]() {
                pack_op.call(
                    pack_key,
                    pack_value,
                    pack_slot_mapping,
                    codebook,
                    rotation_t,
                    pack_key_cache,
                    pack_value_cache,
                    opt.block_size);
            },
            opt.warmup,
            opt.repeat);

        at::Tensor last_out;
        phase = "benchmark 4bit attention";
        const double attn_us = BenchUs(
            [&]() {
                last_out = attention_op.call(
                    query,
                    key_cache,
                    value_cache,
                    block_table,
                    at::IntArrayRef(actual_q),
                    at::IntArrayRef(actual_kv),
                    codebook,
                    rotation,
                    codebook,
                    rotation,
                    opt.num_heads,
                    opt.num_kv_heads,
                    kHeadSize,
                    opt.block_size,
                    opt.seq_len,
                    opt.scale);
            },
            opt.warmup,
            opt.repeat);

        std::cout << std::fixed << std::setprecision(2)
                  << "pack4bit_to_cache_avg_us=" << pack_us << "\n"
                  << "attention_paged4bit_avg_us=" << attn_us << "\n"
                  << "attention_out_shape=" << last_out.sizes() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR during " << phase << ": " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "ERROR during " << phase << ": unknown exception\n";
        return 1;
    }
}
