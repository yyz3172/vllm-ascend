/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// TurboquantDecodePaged8bit OpDef registration.
//
// 算子语义（设计文档 §2.6 方案 X）:
//   把 block_table 寻址折进 kernel — 输入完整 packed KV cache + 紧凑物理块号列表,
//   输出按 seq 顺序紧凑排布的 fp16 K/V workspace。
//
// S1 阶段: kernel 仅做 gather_block_ids 寻址 + 把 packed 的 idx 字节搬到输出
// 第 0 维(uint8 → fp16 zero-extend),用于验证寻址链路。S2/S3 阶段补上 LUT + Cube。

#include "register/op_def_registry.h"

namespace ops {

class TurboquantDecodePaged8bit : public OpDef {
public:
    explicit TurboquantDecodePaged8bit(const char* name) : OpDef(name)
    {
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gather_block_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("codebook")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rotation")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("key_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("value_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("head_size").Int();        // 必须 == 128
        this->Attr("block_size").Int();       // BS,常用 128
        this->Attr("num_kv_heads").Int();     // H
        this->Attr("total_blocks").Int();     // gather_block_ids.numel()
        this->Attr("rows_per_core").Int();    // legacy(未用),tiling 内部按核数算 blocksPerCore
        this->Attr("out_dtype").Int();        // 0=fp16, 1=bf16(本期未启用)
        this->Attr("mode").Int();             // 0=KFC Cube, 1=AIV-only scalar 参考

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

            OpAICoreConfig aicoreConfigA2 = aicoreConfig;
            aicoreConfigA2.ExtendCfgInfo("jitCompile.flag", "static_false");
    
            this->AICore().AddConfig("ascend910_93", aicoreConfig);
            this->AICore().AddConfig("ascend910b", aicoreConfigA2);
    }
};

OP_ADD(TurboquantDecodePaged8bit);

}  // namespace ops
