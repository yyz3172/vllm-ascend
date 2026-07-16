# Vendored/arch32 is on -I for FIA headers, but do NOT name any vendored file
# the same as AscendC headers (e.g. former kernel_common.h) — that shadows
# AscendC includes mid-kernel_operator.h and breaks matmul (FixpipeConfig/TPipe).
add_ops_compile_options(
        OP_NAME BitResidualFiaPagedK8v4
        OPTIONS --cce-auto-sync=on
                -DASCENDC_DUMP=0
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/vendored
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/vendored/arch32
                -I${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(op_host_aclnn PRIVATE
        bit_residual_fia_paged_k8v4_def.cpp
)

target_sources(optiling PRIVATE
        bit_residual_fia_paged_k8v4_tiling.cpp
        vendored/split_core.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        bit_residual_fia_paged_k8v4_tiling.cpp
        vendored/split_core.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/vendored
)

file(GLOB _BR_FIA_PAGED_K8V4_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_bit_residual_fia_paged_k8v4.h")

install(FILES ${_BR_FIA_PAGED_K8V4_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
