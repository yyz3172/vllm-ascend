/**
 * C++ perf/correctness harness for BitResidualFiaPagedK8v4.
 *
 * Workloads (env BR_FIA_WORKLOAD):
 *   decode  (default): B=16, T=16 (1 Q / seq) — TQ FIA L6 scale
 *   prefill: B=16, T=B*Q_TOKENS_PER_BATCH (default 15 → T=240 ≈ serving Q=241)
 *
 * Shared: H=16, NKV=8, D=128, BS=128, blockNum=120, maxBlockPerSeq=16,
 *   kvSeq=1000 (override via KV_SEQ_LEN), sparseMode=3 + 2048x2048 compress mask.
 *
 * Cache layout: uint8 BR pack [num_blocks, nkv, packed_bytes].
 *
 * Env:
 *   BR_FIA_WORKLOAD=decode|prefill
 *   Q_TOKENS_PER_BATCH (prefill only, default 15)
 *   ASCEND_DEVICE_ID, KV_SEQ_LEN, ASCEND_CUSTOM_OPP_PATH
 *   Q_PATH, PI_PATH (optional fp16 bins; PI used for both rotation_key/value)
 *   GOLDEN_OUT_PATH, WS_DUMP_PATH
 *
 * Build:
 *   bash tools/bit_residual_fia_paged_k8v4/build_test.sh [--rebuild-op]
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
#include "aclnn_bit_residual_fia_paged_k8v4.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)          \
    do {                                 \
        printf(message, ##__VA_ARGS__);  \
        fflush(stdout);                  \
    } while (0)

// Geometry shared by decode / prefill (Qwen3-0.6B-like GQA).
constexpr int64_t kBatch = 16;
constexpr int64_t kNumHeads = 16;
constexpr int64_t kNumKvHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlockNum = 120;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kMaxBlockNumPerSeq = 16;
constexpr int64_t kKvSeqLenPerBatch = 1000;
constexpr int64_t kMaskSize = 2048;
constexpr int64_t kDefaultPrefillQPerBatch = 15;  // T=240 ≈ serving chunked Q=241
constexpr uint32_t kBrBlockRows = 16;
constexpr uint32_t kBrKeyTileBytes = 2112;  // 16 rows
constexpr uint32_t kBrValTileBytes = 1088;
constexpr int64_t kKeyPackedBytes =
    static_cast<int64_t>((kBlockSize / kBrBlockRows) * kBrKeyTileBytes);  // 16896
constexpr int64_t kValPackedBytes =
    static_cast<int64_t>((kBlockSize / kBrBlockRows) * kBrValTileBytes);  // 8704
constexpr uint16_t kFp16Zero = 0x0000;
constexpr uint16_t kFp16One = 0x3C00;

struct WorkloadConfig {
    const char *name;
    int64_t qTokensPerBatch;
    int64_t totalTokens;
};

WorkloadConfig ResolveWorkload()
{
    const char *mode = getenv("BR_FIA_WORKLOAD");
    if (mode == nullptr || mode[0] == '\0' || strcmp(mode, "decode") == 0) {
        return WorkloadConfig{"decode", 1, kBatch};
    }
    if (strcmp(mode, "prefill") != 0) {
        LOG_PRINT("[workload] WARN: unknown BR_FIA_WORKLOAD=%s, using decode\n", mode);
        return WorkloadConfig{"decode", 1, kBatch};
    }
    int64_t qPer = kDefaultPrefillQPerBatch;
    if (const char *env = getenv("Q_TOKENS_PER_BATCH")) {
        char *end = nullptr;
        long long v = strtoll(env, &end, 10);
        if (end != env && v > 0) {
            qPer = static_cast<int64_t>(v);
        }
    }
    return WorkloadConfig{"prefill", qPer, kBatch * qPer};
}

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

// Fill BR pack cache with non-zero codes + fp16-one meta so dequant is well-defined.
vector<uint8_t> BuildKeyCacheHost()
{
    const size_t n = static_cast<size_t>(kBlockNum * kNumKvHeads * kKeyPackedBytes);
    vector<uint8_t> data(n, 0);
    const uint32_t subBlocks = static_cast<uint32_t>(kBlockSize / kBrBlockRows);
    const uint32_t headStride = static_cast<uint32_t>(kKeyPackedBytes);
    for (int64_t blk = 0; blk < kBlockNum; ++blk) {
        for (int64_t n2 = 0; n2 < kNumKvHeads; ++n2) {
            uint8_t *head = data.data() + static_cast<size_t>((blk * kNumKvHeads + n2) * headStride);
            for (uint32_t sb = 0; sb < subBlocks; ++sb) {
                uint8_t *tile = head + sb * kBrKeyTileBytes;
                // codes: mid-range unsigned q in [0,255]
                memset(tile, 0x80, kBrBlockRows * 128U);
                // base / step: fp16 1.0 for each of 16 rows
                uint16_t *base = reinterpret_cast<uint16_t *>(tile + kBrBlockRows * 128U);
                uint16_t *step = reinterpret_cast<uint16_t *>(tile + kBrBlockRows * 130U);
                for (uint32_t r = 0; r < kBrBlockRows; ++r) {
                    base[r] = kFp16One;
                    step[r] = kFp16One;
                }
            }
        }
    }
    return data;
}

vector<uint8_t> BuildValueCacheHost()
{
    const size_t n = static_cast<size_t>(kBlockNum * kNumKvHeads * kValPackedBytes);
    vector<uint8_t> data(n, 0);
    const uint32_t subBlocks = static_cast<uint32_t>(kBlockSize / kBrBlockRows);
    const uint32_t headStride = static_cast<uint32_t>(kValPackedBytes);
    for (int64_t blk = 0; blk < kBlockNum; ++blk) {
        for (int64_t n2 = 0; n2 < kNumKvHeads; ++n2) {
            uint8_t *head = data.data() + static_cast<size_t>((blk * kNumKvHeads + n2) * headStride);
            for (uint32_t sb = 0; sb < subBlocks; ++sb) {
                uint8_t *tile = head + sb * kBrValTileBytes;
                // nibble codes: 0x44 -> idx4=4 for both low/high
                memset(tile, 0x44, kBrBlockRows * 64U);
                uint16_t *vmin = reinterpret_cast<uint16_t *>(tile + kBrBlockRows * 64U);
                uint16_t *vstep = reinterpret_cast<uint16_t *>(tile + kBrBlockRows * 66U);
                for (uint32_t r = 0; r < kBrBlockRows; ++r) {
                    vmin[r] = kFp16Zero;
                    vstep[r] = kFp16One;
                }
            }
        }
    }
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

    const WorkloadConfig wl = ResolveWorkload();
    const int64_t kvSeqLen = ResolveKvSeqLenPerBatch();

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

    vector<int64_t> queryShape = {wl.totalTokens, kNumHeads, kHeadDim};
    vector<int64_t> outShape = {wl.totalTokens, kNumHeads, kHeadDim};
    vector<int64_t> keyCacheShape = {kBlockNum, kNumKvHeads, kKeyPackedBytes};
    vector<int64_t> valueCacheShape = {kBlockNum, kNumKvHeads, kValPackedBytes};
    vector<int64_t> rotShape = {kHeadDim, kHeadDim};
    vector<int64_t> attenMaskShape = {kMaskSize, kMaskSize};
    vector<int64_t> blockTableShape = {kBatch, kMaxBlockNumPerSeq};

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *rotKeyDeviceAddr = nullptr;
    void *rotValDeviceAddr = nullptr;
    void *attenMaskDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *rotKeyTensor = nullptr;
    aclTensor *rotValTensor = nullptr;
    aclTensor *attenMaskTensor = nullptr;
    aclTensor *blockTableTensor = nullptr;
    aclTensor *outTensor = nullptr;
    aclIntArray *actualSeqQArr = nullptr;
    aclIntArray *actualSeqKvArr = nullptr;

    vector<uint16_t> queryHostData = LoadQueryOrOnes(queryShape);
    vector<uint8_t> keyHostData = BuildKeyCacheHost();
    vector<uint8_t> valueHostData = BuildValueCacheHost();
    vector<uint16_t> rotHostData = LoadPiOrIdentity();
    // Compress mask: zeros keep sparseMode=3 path active; for perf we care about
    // dequant + mm1/mm2 traffic more than exact causal masking.
    vector<int8_t> attenMaskHostData(static_cast<size_t>(GetShapeSize(attenMaskShape)), 0);
    vector<int32_t> blockTableHostData = BuildBlockTable(kBatch, kMaxBlockNumPerSeq);
    vector<uint16_t> outHostData(static_cast<size_t>(GetShapeSize(outShape)), kFp16Zero);

    vector<int64_t> actualSeqQ = BuildCumulativeActualSeqLengths(kBatch, wl.qTokensPerBatch);
    vector<int64_t> actualSeqKv(static_cast<size_t>(kBatch), kvSeqLen);

    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(keyHostData, keyCacheShape, &keyDeviceAddr, aclDataType::ACL_UINT8, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(valueHostData, valueCacheShape, &valueDeviceAddr, aclDataType::ACL_UINT8, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(rotHostData, rotShape, &rotKeyDeviceAddr, aclDataType::ACL_FLOAT16, &rotKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(rotHostData, rotShape, &rotValDeviceAddr, aclDataType::ACL_FLOAT16, &rotValTensor);
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
    constexpr int64_t kPreTokens = 2147483647;
    constexpr int64_t kNextTokens = 2147483647;
    constexpr int64_t kSparseMode = 3;
    LOG_PRINT(
        "BitResidualFiaPagedK8v4 TND+PA workload=%s: Q=%ld (q/seq=%ld) B=%ld "
        "rot=[%ld,%ld] keyPack=[%ld,%ld,%ld] valPack=[%ld,%ld,%ld] kvSeq=%ld sparseMode=%ld\n",
        wl.name, wl.totalTokens, wl.qTokensPerBatch, kBatch, kHeadDim, kHeadDim, kBlockNum, kNumKvHeads,
        kKeyPackedBytes, kBlockNum, kNumKvHeads, kValPackedBytes, kvSeqLen, kSparseMode);
    if (kvSeqLen > 512 && kBatch >= 8) {
        LOG_PRINT(
            "[FD] Workload B=%ld kv=%ld > s2Base=512: expect tiling key=1 when "
            "SplitCore records numOfFdHead>0 (aligned with TQ FIA L6).\n",
            static_cast<long>(kBatch), static_cast<long>(kvSeqLen));
    }
    if (strcmp(wl.name, "prefill") == 0) {
        const int64_t gS1 = wl.qTokensPerBatch * (kNumHeads / kNumKvHeads);
        LOG_PRINT(
            "[prefill] gSize*S1/seq=%ld mBase=512 → M-tiles/seq=%ld (serving Q=241 "
            "proxy for mm1/mm2 workspace MTE2)\n",
            static_cast<long>(gS1), static_cast<long>((gS1 + 511) / 512));
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBitResidualFiaPagedK8v4GetWorkspaceSize(
        queryTensor, keyTensor, valueTensor, blockTableTensor, actualSeqQArr, actualSeqKvArr, attenMaskTensor,
        rotKeyTensor, rotValTensor, kNumHeads, kNumKvHeads, kHeadDim, kBlockSize, scaleValue, kPreTokens,
        kNextTokens, kSparseMode, outTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnBitResidualFiaPagedK8v4GetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnBitResidualFiaPagedK8v4(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnBitResidualFiaPagedK8v4 failed. ERROR: %d\n", ret);
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
    aclDestroyTensor(rotKeyTensor);
    aclDestroyTensor(rotValTensor);
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
    aclrtFree(rotKeyDeviceAddr);
    aclrtFree(rotValDeviceAddr);
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
