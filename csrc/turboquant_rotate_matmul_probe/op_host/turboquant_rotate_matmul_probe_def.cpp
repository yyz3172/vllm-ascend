#include "register/op_def_registry.h"

namespace ops {
class TurboquantRotateMatmulProbe : public OpDef {
public:
    explicit TurboquantRotateMatmulProbe(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("probe_mode").Int();
        // Logical row count before host-side M padding (a may be [M_pad, 128] on device).
        this->Attr("m").Int();

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

OP_ADD(TurboquantRotateMatmulProbe);
}  // namespace ops
