add_ops_compile_options(
        OP_NAME TurboquantFusedInferAttentionScore8bit
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnn PRIVATE
        turboquant_fused_infer_attention_score8bit_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_fused_infer_attention_score_8bit_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        turboquant_fused_infer_attention_score_8bit_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

