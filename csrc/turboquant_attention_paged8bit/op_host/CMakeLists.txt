set(_TQ_ATTN_COMPILE_OPTIONS
        --cce-auto-sync=on
        -DASCENDC_DUMP=1
        -Wno-deprecated-declarations
        -Werror
)

set(_TQ_ATTN_DEBUG_LINE "$ENV{KENEL_DEBUG_LINE}")
if("${_TQ_ATTN_DEBUG_LINE}" STREQUAL "")
    set(_TQ_ATTN_DEBUG_LINE "$ENV{KERNEL_DEBUG_LINE}")
endif()
string(TOUPPER "${_TQ_ATTN_DEBUG_LINE}" _TQ_ATTN_DEBUG_LINE)
if(_TQ_ATTN_DEBUG_LINE STREQUAL "ON")
    list(APPEND _TQ_ATTN_COMPILE_OPTIONS -g)
endif()

add_ops_compile_options(
        OP_NAME TurboquantAttentionPaged8bit
        OPTIONS ${_TQ_ATTN_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        turboquant_attention_paged8bit_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_attention_paged8bit_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _TQ_ATTN_PAGED_8BIT_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_turboquant_attention_paged8bit.h")

install(FILES ${_TQ_ATTN_PAGED_8BIT_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
