set(_TQ_PACK_COMPILE_OPTIONS
        --cce-auto-sync=on
        -DASCENDC_DUMP=0
        -Wno-deprecated-declarations
        -Werror
)

set(TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE "OFF" CACHE STRING
        "Enable -g for TurboquantPackKvForCache4bit AscendC kernel compile.")

set(_TQ_PACK_DEBUG_LINE "${TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE}")
if(DEFINED ENV{TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE})
    set(_TQ_PACK_DEBUG_LINE "$ENV{TURBOQUANT_PACK_KV_FOR_CACHE4BIT_DEBUG_LINE}")
endif()
string(TOUPPER "${_TQ_PACK_DEBUG_LINE}" _TQ_PACK_DEBUG_LINE)
if(_TQ_PACK_DEBUG_LINE STREQUAL "ON" OR _TQ_PACK_DEBUG_LINE STREQUAL "1" OR
        _TQ_PACK_DEBUG_LINE STREQUAL "TRUE")
    list(APPEND _TQ_PACK_COMPILE_OPTIONS -g)
endif()

add_ops_compile_options(
        OP_NAME TurboquantPackKvForCache4bit
        OPTIONS ${_TQ_PACK_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        turboquant_pack_kv_for_cache4bit_def.cpp
)

target_sources(optiling PRIVATE
        turboquant_pack_kv_for_cache4bit_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _TQ_PACK_TO_CACHE_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_turboquant_pack_kv_for_cache4bit.h")

install(FILES ${_TQ_PACK_TO_CACHE_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
