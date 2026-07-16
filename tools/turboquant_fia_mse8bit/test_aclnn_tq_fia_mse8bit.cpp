/**
 * C++ correctness harness for vllm-ascend custom op TurboquantFiaMse8bit.
 *
 * Mirrors ops-transformer test_aclnn_..._tnd_pa_turboquant.cpp shapes/env hooks,
 * but calls aclnnTurboquantFiaMse8bit (FIA TQ P0 port).
 *
 * Env (same as verify_pi_ortho):
 *   Q_PATH, KV_KEY_PATH, KV_VALUE_PATH, GAMMA_PATH, PI_PATH
 *   KV_SEQ_LEN, GOLDEN_OUT_PATH, WS_DUMP_PATH
 *
 * Build:
 *   bash tools/turboquant_fia_mse8bit/build_test.sh [--rebuild-op]
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_turboquant_fia_mse8bit.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)          \
    do {                                 \
        printf(message, ##__VA_ARGS__);  \
        fflush(stdout);                  \
    } while (0)

constexpr int64_t kBatch = 16;
constexpr int64_t kTotalTokens = 16;
constexpr int64_t kNumHeads = 16;
constexpr int64_t kNumKvHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlockNum = 120;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kKvHidden = kNumKvHeads * kHeadDim;
constexpr int64_t kKvTokens = kBlockNum * kBlockSize;
constexpr int64_t kMaxBlockNumPerSeq = 16;
constexpr int64_t kKvSeqLenPerBatch = 1000;
constexpr int64_t kMaskSize = 2048;
constexpr uint16_t kFp16Zero = 0x0000;
constexpr uint16_t kFp16One = 0x3C00;

#ifndef VLLM_ASCEND_CUSTOM_OPP_PATH
#define VLLM_ASCEND_CUSTOM_OPP_PATH "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
#endif

void ConfigureCustomOppPath()
{
    const char *custom = VLLM_ASCEND_CUSTOM_OPP_PATH;
    if (custom == nullptr || custom[0] == '\0') {
        return;
    }
    const char *cur = getenv("ASCEND_CUSTOM_OPP_PATH");
    if (cur == nullptr || cur[0] == '\0') {
        setenv("ASCEND_CUSTOM_OPP_PATH", custom, 1);
        LOG_PRINT("[env] ASCEND_CUSTOM_OPP_PATH=%s\n", custom);
        return;
    }
    if (strstr(cur, custom) == nullptr) {
        string merged = string(custom) + ":" + cur;
        setenv("ASCEND_CUSTOM_OPP_PATH", merged.c_str(), 1);
        LOG_PRINT("[env] ASCEND_CUSTOM_OPP_PATH prepended %s\n", custom);
    }
}

int64_t ResolveKvSeqLenPerBatch()
{
    if (const char *env = getenv("KV_SEQ_LEN")) {
        char *end = nullptr;
        long long v = strtoll(env, &end, 10);
        if (end != env && v > 0) {
            LOG_PRINT("[KV_SEQ_LEN] override actualSeqLengthsKv = %lld (default %ld)\n", v,
                      static_cast<long>(kKvSeqLenPerBatch));
            return static_cast<int64_t>(v);
        }
    }
    return kKvSeqLenPerBatch;
}

int64_t GetShapeSize(const vector<int64_t> &shape)
{
    int64_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const vector<T> &hostData, const vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

vector<int64_t> BuildCumulativeActualSeqLengths(int64_t batch, int64_t tokensPerBatch)
{
    vector<int64_t> actual(batch, 0);
    for (int64_t i = 0; i < batch; ++i) {
        actual[static_cast<size_t>(i)] = (i + 1) * tokensPerBatch;
    }
    return actual;
}

vector<int32_t> BuildBlockTable(int64_t batch, int64_t maxBlockNumPerSeq)
{
    vector<int32_t> blockTable(static_cast<size_t>(batch * maxBlockNumPerSeq), 0);
    int32_t blockId = 0;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < maxBlockNumPerSeq; ++i) {
            blockTable[static_cast<size_t>(b * maxBlockNumPerSeq + i)] = blockId % static_cast<int32_t>(kBlockNum);
            ++blockId;
        }
    }
    return blockTable;
}

vector<uint16_t> BuildIdentityPi()
{
    vector<uint16_t> pi(static_cast<size_t>(kHeadDim * kHeadDim), kFp16Zero);
    for (int64_t i = 0; i < kHeadDim; ++i) {
        pi[static_cast<size_t>(i * kHeadDim + i)] = kFp16One;
    }
    return pi;
}

vector<uint16_t> LoadPiOrIdentity()
{
    const char *piPath = getenv("PI_PATH");
    if (piPath == nullptr || piPath[0] == '\0') {
        return BuildIdentityPi();
    }
    FILE *fp = fopen(piPath, "rb");
    if (fp == nullptr) {
        LOG_PRINT("[pi] WARN: PI_PATH=%s open failed, using identity\n", piPath);
        return BuildIdentityPi();
    }
    vector<uint16_t> pi(static_cast<size_t>(kHeadDim * kHeadDim), kFp16Zero);
    size_t need = pi.size() * sizeof(uint16_t);
    size_t got = fread(pi.data(), 1, need, fp);
    fclose(fp);
    if (got != need) {
        LOG_PRINT("[pi] WARN: PI_PATH=%s short read %zu/%zu, using identity\n", piPath, got, need);
        return BuildIdentityPi();
    }
    LOG_PRINT("[pi] loaded %zu fp16 from %s\n", pi.size(), piPath);
    return pi;
}

vector<uint16_t> LoadQueryOrOnes(const vector<int64_t> &shape)
{
    const char *path = getenv("Q_PATH");
    vector<uint16_t> data(static_cast<size_t>(GetShapeSize(shape)), kFp16One);
    if (path == nullptr || path[0] == '\0') {
        return data;
    }
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG_PRINT("[q] WARN: Q_PATH=%s open failed, using all-ones\n", path);
        return data;
    }
    size_t need = data.size() * sizeof(uint16_t);
    size_t got = fread(data.data(), 1, need, fp);
    fclose(fp);
    if (got != need) {
        LOG_PRINT("[q] WARN: Q_PATH=%s short read, using all-ones\n", path);
        fill(data.begin(), data.end(), kFp16One);
        return data;
    }
    LOG_PRINT("[q] loaded %zu fp16 from %s\n", data.size(), path);
    return data;
}

vector<int8_t> LoadKvOrOnes(const char *envName, const vector<int64_t> &shape)
{
    const char *path = getenv(envName);
    vector<int8_t> data(static_cast<size_t>(GetShapeSize(shape)), 1);
    if (path == nullptr || path[0] == '\0') {
        return data;
    }
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG_PRINT("[kv] WARN: %s=%s open failed, using all-ones\n", envName, path);
        return data;
    }
    size_t need = data.size() * sizeof(int8_t);
    size_t got = fread(data.data(), 1, need, fp);
    fclose(fp);
    if (got != need) {
        LOG_PRINT("[kv] WARN: %s short read, using all-ones\n", envName);
        fill(data.begin(), data.end(), 1);
        return data;
    }
    LOG_PRINT("[kv] loaded %zu int8 from %s\n", data.size(), path);
    return data;
}

vector<uint16_t> LoadGammaOrOnes(const char *envName, const vector<int64_t> &shape)
{
    const char *path = getenv(envName);
    vector<uint16_t> data(static_cast<size_t>(GetShapeSize(shape)), kFp16One);
    if (path == nullptr || path[0] == '\0') {
        return data;
    }
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG_PRINT("[gamma] WARN: %s=%s open failed, using all-ones\n", envName, path);
        return data;
    }
    size_t need = data.size() * sizeof(uint16_t);
    size_t got = fread(data.data(), 1, need, fp);
    fclose(fp);
    if (got != need) {
        LOG_PRINT("[gamma] WARN: %s short read, using all-ones\n", envName);
        fill(data.begin(), data.end(), kFp16One);
        return data;
    }
    LOG_PRINT("[gamma] loaded %zu fp16 from %s\n", data.size(), path);
    return data;
}

float HalfBitsToFloat(uint16_t bits)
{
    uint32_t sign = (bits >> 15U) & 0x1U;
    uint32_t exp = (bits >> 10U) & 0x1FU;
    uint32_t frac = bits & 0x3FFU;
    float val = 0.0f;
    if (exp == 0U) {
        val = ldexpf(static_cast<float>(frac), -24);
    } else if (exp == 31U) {
        val = (frac == 0U) ? INFINITY : NAN;
    } else {
        val = ldexpf(static_cast<float>(frac | 0x400U), static_cast<int>(exp) - 25);
    }
    return sign ? -val : val;
}

bool ValidateAttentionOutNonZero(const vector<uint16_t> &outHostData, int64_t sampleCount = 8)
{
    int64_t total = static_cast<int64_t>(outHostData.size());
    int64_t nonZeroCount = 0;
    float maxAbs = 0.0f;
    float sumAbs = 0.0f;
    for (int64_t i = 0; i < total; ++i) {
        if (outHostData[static_cast<size_t>(i)] == kFp16Zero) {
            continue;
        }
        ++nonZeroCount;
        float val = HalfBitsToFloat(outHostData[static_cast<size_t>(i)]);
        float absVal = fabsf(val);
        maxAbs = max(maxAbs, absVal);
        sumAbs += absVal;
    }
    LOG_PRINT("Output check: total=%ld nonZero=%ld maxAbs=%.6f meanAbs=%.6f\n", total, nonZeroCount, maxAbs,
              nonZeroCount > 0 ? (sumAbs / static_cast<float>(nonZeroCount)) : 0.0f);
    int64_t printCount = min(sampleCount, total);
    LOG_PRINT("Output sample[0:%ld]:", printCount);
    for (int64_t i = 0; i < printCount; ++i) {
        LOG_PRINT(" %.6f", HalfBitsToFloat(outHostData[static_cast<size_t>(i)]));
    }
    LOG_PRINT("\n");
    if (nonZeroCount == 0) {
        LOG_PRINT("ERROR: attentionOut is all zero, kernel likely did not write results.\n");
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    ConfigureCustomOppPath();
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 1);

    int32_t deviceId = 0;
    if (const char *devEnv = getenv("ASCEND_DEVICE_ID")) {
        deviceId = static_cast<int32_t>(strtol(devEnv, nullptr, 10));
        LOG_PRINT("[env] ASCEND_DEVICE_ID=%d\n", deviceId);
    }
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    vector<int64_t> queryShape = {kTotalTokens, kNumHeads, kHeadDim};
    vector<int64_t> outShape = {kTotalTokens, kNumHeads, kHeadDim};
    vector<int64_t> kvCacheShape = {kBlockNum, kBlockSize, kKvHidden};
    vector<int64_t> piShape = {kHeadDim, kHeadDim};
    vector<int64_t> gammaShape = {kKvTokens, kNumKvHeads};
    vector<int64_t> attenMaskShape = {kMaskSize, kMaskSize};
    vector<int64_t> blockTableShape = {kBatch, kMaxBlockNumPerSeq};

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *piDeviceAddr = nullptr;
    void *keyGammaDeviceAddr = nullptr;
    void *valueGammaDeviceAddr = nullptr;
    void *attenMaskDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *piTensor = nullptr;
    aclTensor *keyGammaTensor = nullptr;
    aclTensor *valueGammaTensor = nullptr;
    aclTensor *attenMaskTensor = nullptr;
    aclTensor *blockTableTensor = nullptr;
    aclTensor *outTensor = nullptr;
    aclIntArray *actualSeqQArr = nullptr;
    aclIntArray *actualSeqKvArr = nullptr;

    vector<uint16_t> queryHostData = LoadQueryOrOnes(queryShape);
    vector<int8_t> keyHostData = LoadKvOrOnes("KV_KEY_PATH", kvCacheShape);
    vector<int8_t> valueHostData = LoadKvOrOnes("KV_VALUE_PATH", kvCacheShape);
    vector<uint16_t> piHostData = LoadPiOrIdentity();
    vector<uint16_t> keyGammaHostData = LoadGammaOrOnes("GAMMA_PATH", gammaShape);
    vector<uint16_t> valueGammaHostData = keyGammaHostData;
    vector<int8_t> attenMaskHostData(static_cast<size_t>(GetShapeSize(attenMaskShape)), 0);
    vector<int32_t> blockTableHostData = BuildBlockTable(kBatch, kMaxBlockNumPerSeq);
    vector<uint16_t> outHostData(static_cast<size_t>(GetShapeSize(outShape)), kFp16Zero);

    vector<int64_t> actualSeqQ = BuildCumulativeActualSeqLengths(kBatch, 1);
    const int64_t kvSeqLen = ResolveKvSeqLenPerBatch();
    vector<int64_t> actualSeqKv(static_cast<size_t>(kBatch), kvSeqLen);

    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(keyHostData, kvCacheShape, &keyDeviceAddr, aclDataType::ACL_INT8, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(valueHostData, kvCacheShape, &valueDeviceAddr, aclDataType::ACL_INT8, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(piHostData, piShape, &piDeviceAddr, aclDataType::ACL_FLOAT16, &piTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(keyGammaHostData, gammaShape, &keyGammaDeviceAddr, aclDataType::ACL_FLOAT16, &keyGammaTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(valueGammaHostData, gammaShape, &valueGammaDeviceAddr, aclDataType::ACL_FLOAT16,
                          &valueGammaTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_INT8,
                          &attenMaskTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    actualSeqQArr = aclCreateIntArray(actualSeqQ.data(), actualSeqQ.size());
    actualSeqKvArr = aclCreateIntArray(actualSeqKv.data(), actualSeqKv.size());
    if (!CHECK_RET(actualSeqQArr != nullptr && actualSeqKvArr != nullptr)) {
        LOG_PRINT("aclCreateIntArray for actualSeqLen failed\n");
        return ACL_ERROR_RT_FAILURE;
    }

    double scaleValue = 1.0 / sqrt(static_cast<double>(kHeadDim));
    // Match FIA TQ example: sparseMode=3 (right-down causal) + 2048x2048 compress mask (0=keep).
    constexpr int64_t kPreTokens = 2147483647;
    constexpr int64_t kNextTokens = 2147483647;
    constexpr int64_t kSparseMode = 3;
    LOG_PRINT("TurboquantFiaMse8bit TND+PA: Pi=[%ld,%ld] gamma=[%ld,%ld] kvSeq=%ld sparseMode=%ld\n",
              kHeadDim, kHeadDim, kKvTokens, kNumKvHeads, kvSeqLen, kSparseMode);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnTurboquantFiaMse8bitGetWorkspaceSize(
        queryTensor, keyTensor, valueTensor, blockTableTensor, actualSeqQArr, actualSeqKvArr, attenMaskTensor,
        keyGammaTensor, valueGammaTensor, piTensor, kNumHeads, kNumKvHeads, kHeadDim, kBlockSize, scaleValue,
        kPreTokens, kNextTokens, kSparseMode, outTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnTurboquantFiaMse8bitGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    LOG_PRINT("workspaceSize=%lu bytes\n", static_cast<unsigned long>(workspaceSize));

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnTurboquantFiaMse8bit(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnTurboquantFiaMse8bit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    const size_t outBytes = static_cast<size_t>(GetShapeSize(outShape)) * sizeof(uint16_t);
    ret = aclrtMemcpy(outHostData.data(), outBytes, outDeviceAddr, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy attentionOut D2H failed. ERROR: %d\n", ret);
        return ret;
    }
    if (!ValidateAttentionOutNonZero(outHostData)) {
        return ACL_ERROR_RT_FAILURE;
    }

    if (const char *goldenOutPath = getenv("GOLDEN_OUT_PATH")) {
        FILE *fp = fopen(goldenOutPath, "wb");
        if (fp != nullptr) {
            size_t wrote = fwrite(outHostData.data(), sizeof(uint16_t), outHostData.size(), fp);
            fclose(fp);
            LOG_PRINT("[golden] wrote %zu fp16 to %s\n", wrote, goldenOutPath);
        } else {
            LOG_PRINT("[golden] WARN: failed to open %s\n", goldenOutPath);
        }
    }

    if (const char *wsDumpPath = getenv("WS_DUMP_PATH")) {
        if (workspaceSize > 0U && workspaceAddr != nullptr) {
            vector<uint8_t> wsHost(workspaceSize);
            ret = aclrtMemcpy(wsHost.data(), workspaceSize, workspaceAddr, workspaceSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == ACL_SUCCESS) {
                FILE *fp = fopen(wsDumpPath, "wb");
                if (fp != nullptr) {
                    size_t wrote = fwrite(wsHost.data(), 1, workspaceSize, fp);
                    fclose(fp);
                    LOG_PRINT("[wsdump] wrote %zu bytes to %s\n", wrote, wsDumpPath);
                }
            }
        }
    }

    LOG_PRINT("Run success.\n");

    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(piTensor);
    aclDestroyTensor(keyGammaTensor);
    aclDestroyTensor(valueGammaTensor);
    aclDestroyTensor(attenMaskTensor);
    aclDestroyTensor(blockTableTensor);
    aclDestroyTensor(outTensor);
    if (actualSeqQArr != nullptr) {
        aclDestroyIntArray(actualSeqQArr);
    }
    if (actualSeqKvArr != nullptr) {
        aclDestroyIntArray(actualSeqKvArr);
    }
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(piDeviceAddr);
    aclrtFree(keyGammaDeviceAddr);
    aclrtFree(valueGammaDeviceAddr);
    aclrtFree(attenMaskDeviceAddr);
    aclrtFree(blockTableDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
