/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <Python.h>
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/core/Device.h>
#include <dlfcn.h>
#include <glob.h>
#include <torch/torch.h>

#include "aclnn_turboquant_rotate_matmul_probe.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kK = 128;
constexpr int64_t kN = 128;

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
using ProbeGetWorkspaceSize = aclnnStatus (*)(
    const aclTensor* a,
    const aclTensor* b,
    int64_t probe_mode,
    int64_t m,
    const aclTensor* c,
    uint64_t* workspace_size,
    aclOpExecutor** executor);
using ProbeRun = aclnnStatus (*)(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream);
using InitHugeMemThreadLocal = int (*)(void*, bool);
using UnInitHugeMemThreadLocal = void (*)(void*, bool);
using ReleaseHugeMem = void (*)(void*, bool);

struct Options {
    int device_id = 0;
    int warmup = 10;
    int repeat = 100;
};

void PrintUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --device N    NPU device id, default 0\n"
        << "  --warmup N    Warmup iterations, default 10\n"
        << "  --repeat N    Timed iterations, default 100\n"
        << "  --help        Show this message\n";
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
        } else if (arg == "--warmup") {
            opt.warmup = static_cast<int>(ParseInt64(need_value("--warmup"), "--warmup"));
        } else if (arg == "--repeat") {
            opt.repeat = static_cast<int>(ParseInt64(need_value("--repeat"), "--repeat"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (opt.device_id < 0 || opt.warmup < 0 || opt.repeat <= 0) {
        throw std::invalid_argument("--device must be >= 0, --warmup >= 0, --repeat > 0");
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

void* DlopenGlobal(const std::string& path)
{
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        throw std::runtime_error("dlopen failed for " + path + ": " + dlerror());
    }
    return handle;
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
        throw std::runtime_error(std::string("failed to resolve opapi symbol: ") + name);
    }
    return reinterpret_cast<Fn>(func);
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
    return std::chrono::duration<double, std::micro>(end - start).count() /
           static_cast<double>(repeat);
}

at::Tensor CpuToNpu(const at::Tensor& cpu, const c10::Device& device)
{
    return cpu.to(device, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

int64_t AlignUp16(int64_t value)
{
    return ((value + 15) / 16) * 16;
}

aclDataType ToAclType(at::ScalarType scalar_type)
{
    if (scalar_type == at::kHalf) {
        return ACL_FLOAT16;
    }
    if (scalar_type == at::kByte) {
        return ACL_UINT8;
    }
    throw std::runtime_error("unsupported tensor dtype for ACL tensor");
}

aclTensor* CreateAclTensor(const at::Tensor& tensor)
{
    if (!tensor.defined()) {
        throw std::runtime_error("undefined tensor");
    }
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("CreateAclTensor expects contiguous tensor");
    }
    auto create_tensor = ResolveRequired<AclCreateTensor>("aclCreateTensor");
    const int64_t storage_dim =
        static_cast<int64_t>(tensor.storage().nbytes() / tensor.itemsize());
    return create_tensor(
        tensor.sizes().data(),
        tensor.sizes().size(),
        ToAclType(tensor.scalar_type()),
        tensor.strides().data(),
        tensor.storage_offset(),
        ACL_FORMAT_ND,
        &storage_dim,
        1,
        const_cast<void*>(tensor.storage().data()));
}

void DestroyAclTensor(aclTensor* tensor)
{
    auto destroy_tensor = reinterpret_cast<AclDestroyTensor>(ResolveOpApiSymbol("aclDestroyTensor"));
    if (destroy_tensor != nullptr) {
        destroy_tensor(tensor);
    }
}

class AclStreamGuard {
public:
    AclStreamGuard()
    {
        aclError ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtCreateStream failed: " + std::to_string(ret));
        }
    }

    ~AclStreamGuard()
    {
        if (stream_ != nullptr) {
            aclrtSynchronizeStream(stream_);
            aclrtDestroyStream(stream_);
        }
    }

    AclStreamGuard(const AclStreamGuard&) = delete;
    AclStreamGuard& operator=(const AclStreamGuard&) = delete;

    aclrtStream get() const { return stream_; }

private:
    aclrtStream stream_ = nullptr;
};

class AclTensorGuard {
public:
    explicit AclTensorGuard(const at::Tensor& tensor) : tensor_(CreateAclTensor(tensor)) {}
    ~AclTensorGuard() { DestroyAclTensor(tensor_); }

    AclTensorGuard(const AclTensorGuard&) = delete;
    AclTensorGuard& operator=(const AclTensorGuard&) = delete;

    aclTensor* get() const { return tensor_; }

private:
    aclTensor* tensor_ = nullptr;
};

void RunProbeOp(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t probe_mode,
    int64_t m,
    const at::Tensor& c,
    aclrtStream stream)
{
    auto get_workspace =
        ResolveRequired<ProbeGetWorkspaceSize>("aclnnTurboquantRotateMatmulProbeGetWorkspaceSize");
    auto run = ResolveRequired<ProbeRun>("aclnnTurboquantRotateMatmulProbe");
    auto init_mem = reinterpret_cast<InitHugeMemThreadLocal>(
        ResolveOpApiSymbol("InitHugeMemThreadLocal"));
    auto uninit_mem = reinterpret_cast<UnInitHugeMemThreadLocal>(
        ResolveOpApiSymbol("UnInitHugeMemThreadLocal"));
    auto release_mem = reinterpret_cast<ReleaseHugeMem>(
        ResolveOpApiSymbol("ReleaseHugeMem"));

    if (init_mem != nullptr) {
        init_mem(nullptr, false);
    }

    AclTensorGuard a_acl(a);
    AclTensorGuard b_acl(b);
    AclTensorGuard c_acl(c);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus status = get_workspace(
        a_acl.get(), b_acl.get(), probe_mode, m, c_acl.get(), &workspace_size, &executor);
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string("GetWorkspaceSize failed: ") + aclGetRecentErrMsg());
    }

    at::Tensor workspace;
    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        workspace = at::empty(
            {static_cast<int64_t>(workspace_size)},
            at::TensorOptions().device(a.device()).dtype(at::kByte));
        workspace_addr = workspace.data_ptr();
    }

    status = run(workspace_addr, workspace_size, executor, stream);
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string("Probe run failed: ") + aclGetRecentErrMsg());
    }
    aclError sync_status = aclrtSynchronizeStream(stream);
    if (sync_status != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed: " + std::to_string(sync_status));
    }

    if (release_mem != nullptr) {
        release_mem(nullptr, false);
    }
    if (uninit_mem != nullptr) {
        uninit_mem(nullptr, false);
    }
}

struct ProbeCase {
    int64_t m;
    double avg_us;
    double max_abs_diff;
    double max_rel_diff;
    std::vector<double> row_max_abs_diff;
};

ProbeCase RunOneCase(
    int64_t m,
    const Options& opt,
    const c10::Device& device,
    aclrtStream stream)
{
    const int64_t m_pad = AlignUp16(m);
    const auto cpu_f32 = at::TensorOptions().device(c10::kCPU).dtype(at::kFloat);
    const auto cpu_f16 = at::TensorOptions().device(c10::kCPU).dtype(at::kHalf);
    const auto npu_f16 = at::TensorOptions().device(device).dtype(at::kHalf);

    auto a_cpu = at::randn({m, kK}, cpu_f32);
    auto b_cpu = at::randn({kK, kN}, cpu_f32);
    auto a_pad_cpu = at::zeros({m_pad, kK}, cpu_f16);
    a_pad_cpu.slice(0, 0, m).copy_(a_cpu.to(at::kHalf));

    auto a_npu = CpuToNpu(a_pad_cpu, device);
    auto b_npu = CpuToNpu(b_cpu.to(at::kHalf), device);
    auto c_npu = at::zeros({m_pad, kN}, npu_f16);

    RunProbeOp(a_npu, b_npu, /*probe_mode=*/2, m, c_npu, stream);
    SyncDevice();

    auto c_cpu = c_npu.slice(0, 0, m).to(c10::kCPU).to(at::kFloat);
    auto ref = at::matmul(a_cpu, b_cpu);
    auto abs_diff = (c_cpu - ref).abs();
    const double max_abs = abs_diff.max().item<double>();
    const double max_rel =
        (abs_diff / ref.abs().clamp_min(1e-5)).max().item<double>();
    std::vector<double> row_max_abs_diff;
    if (max_abs > 0.25) {
        auto row_max = std::get<0>(abs_diff.max(1)).contiguous();
        row_max_abs_diff.reserve(static_cast<size_t>(m));
        const float* row_max_ptr = row_max.data_ptr<float>();
        for (int64_t row = 0; row < m; ++row) {
            row_max_abs_diff.push_back(static_cast<double>(row_max_ptr[row]));
        }
    }

    const double avg_us = BenchUs(
        [&]() {
            RunProbeOp(a_npu, b_npu, /*probe_mode=*/2, m, c_npu, stream);
        },
        opt.warmup,
        opt.repeat);

    return ProbeCase{m, avg_us, max_abs, max_rel, std::move(row_max_abs_diff)};
}

}  // namespace

int main(int argc, char** argv)
{
    std::string phase = "startup";
    try {
        phase = "parse arguments";
        const Options opt = ParseArgs(argc, argv);

        setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
        phase = "initialize Python runtime";
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }

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
        phase = "load custom opapi";
        DlopenFirstMatch(
            {
                "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so",
            },
            "libcust_opapi.so");

        phase = "set device";
        const std::string set_device_code =
            "import torch\n"
            "torch.npu.set_device(" + std::to_string(opt.device_id) + ")\n";
        RunPython(set_device_code.c_str());
        const c10::Device device(c10::DeviceType::PrivateUse1, opt.device_id);
        AclStreamGuard stream;

        phase = "run probe cases";
        at::manual_seed(20260618);
        std::cout << "device=npu:" << opt.device_id
                  << " warmup=" << opt.warmup
                  << " repeat=" << opt.repeat << "\n";
        for (int64_t m : {1, 16, 32}) {
            const ProbeCase result = RunOneCase(m, opt, device, stream.get());
            std::cout << std::fixed << std::setprecision(4)
                      << "probe_mode2_m=" << result.m
                      << " avg_us=" << result.avg_us
                      << " max_abs_diff=" << result.max_abs_diff
                      << " max_rel_diff=" << result.max_rel_diff << "\n";
            if (!result.row_max_abs_diff.empty()) {
                std::cout << "  row_max_abs_diff=";
                for (size_t row = 0; row < result.row_max_abs_diff.size(); ++row) {
                    if (row != 0) {
                        std::cout << ",";
                    }
                    std::cout << result.row_max_abs_diff[row];
                }
                std::cout << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR during " << phase << ": " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "ERROR during " << phase << ": unknown exception\n";
        return 1;
    }
}
