set(_TQ_ATTN_COMPILE_OPTIONS
        --cce-auto-sync=on
        -DASCENDC_DUMP=0
        -Wno-deprecated-declarations
        -Werror
)

set(TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE "OFF" CACHE STRING
        "Enable -g for TurboquantAttentionPaged4bit AscendC kernel compile.")

set(_TQ_ATTN_DEBUG_LINE "${TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE}")
if(DEFINED ENV{TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE})
    set(_TQ_ATTN_DEBUG_LINE "$ENV{TURBOQUANT_ATTENTION_PAGED4BIT_DEBUG_LINE}")
endif()
string(TOUPPER "${_TQ_ATTN_DEBUG_LINE}" _TQ_ATTN_DEBUG_LINE)
if(_TQ_ATTN_DEBUG_LINE STREQUAL "ON" OR _TQ_ATTN_DEBUG_LINE STREQUAL "1" OR
        _TQ_ATTN_DEBUG_LINE STREQUAL "TRUE")
    list(APPEND _TQ_ATTN_COMPILE_OPTIONS -g)
endif()

add_ops_compile_options(
        OP_NAME TurboquantAttentionPaged4bit
        OPTIONS ${_TQ_ATTN_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        turboquant_attention_paged4bit_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_attention_paged4bit_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _TQ_ATTN_PAGED_4BIT_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_turboquant_attention_paged4bit.h")

install(FILES ${_TQ_ATTN_PAGED_4BIT_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
