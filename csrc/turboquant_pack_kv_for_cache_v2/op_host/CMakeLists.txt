add_ops_compile_options(
        OP_NAME TurboquantPackKvForCacheV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        turboquant_pack_kv_for_cache_v2_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_pack_kv_for_cache_v2_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _TQ_PACK_V2_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_turboquant_pack_kv_for_cache_v2.h")

install(FILES ${_TQ_PACK_V2_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
