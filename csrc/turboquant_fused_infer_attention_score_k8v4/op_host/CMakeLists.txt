add_ops_compile_options(
        OP_NAME TurboquantFusedInferAttentionScoreK8v4
        OPTIONS --cce-auto-sync=on
                -DASCENDC_DUMP=0
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnn PRIVATE
        turboquant_fused_infer_attention_score_k8v4_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_fused_infer_attention_score_k8v4_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        turboquant_fused_infer_attention_score_k8v4_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)
