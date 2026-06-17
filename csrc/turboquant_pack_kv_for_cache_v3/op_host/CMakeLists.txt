add_ops_compile_options(
        OP_NAME TurboquantPackKvForCacheV3
        OPTIONS --cce-auto-sync=on
                -DASCENDC_DUMP=0
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        turboquant_pack_kv_for_cache_v3_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_pack_kv_for_cache_v3_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _TQ_PACK_V3_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_turboquant_pack_kv_for_cache_v3.h")

install(FILES ${_TQ_PACK_V3_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
