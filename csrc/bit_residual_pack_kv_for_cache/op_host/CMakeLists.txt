add_ops_compile_options(
        OP_NAME BitResidualPackKvForCache
        OPTIONS --cce-auto-sync=on
                -DASCENDC_DUMP=0
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        bit_residual_pack_kv_for_cache_def.cpp
)

target_sources(optiling PRIVATE
        bit_residual_pack_kv_for_cache_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _BIT_RESIDUAL_PACK_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_bit_residual_pack_kv_for_cache.h")

install(FILES ${_BIT_RESIDUAL_PACK_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
