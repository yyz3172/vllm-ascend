#include "register/op_def_registry.h"

namespace ops {
class TurboquantPackKvForCacheFused : public OpDef {
public:
    explicit TurboquantPackKvForCacheFused(const char* name) : OpDef(name)
    {
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codebook")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rotation_t")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("packed_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("packed_v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // 0 = MIX KFC batched pack (same as probe_mode=0 REGIST path)
        // 1 = AIV-only reference pack (manual rotate; debug / NO_KFC)
        this->Attr("pack_mode").Int();
        this->Attr("n_vec").Int();
        this->Attr("slot_w_k").Int();
        this->Attr("slot_w_v").Int();
        this->Attr("vec_per_core").Int();

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

OP_ADD(TurboquantPackKvForCacheFused);
}  // namespace ops
