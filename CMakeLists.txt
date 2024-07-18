cmake_minimum_required(VERSION 3.18.0)

project(gemm_simple LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

LINK_DIRECTORIES(/usr/local/cuda/lib64)

add_executable(${PROJECT_NAME}
    main.cu
)

set(CUTE_INCLUDE_DIR /root/paddlejob/workspace/env_run/wufeisheng/TEST/test_cute/cutlass/include/ /root/paddlejob/workspace/env_run/wufeisheng/TEST/test_cute/cutlass/tools/util/include/)

set(ARCHS_ENABLED 80)
set(__CMAKE_CUDA_ARCHS)
foreach(ARCH ${ARCHS_ENABLED})
    set(CODES)
    list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-real)
    list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-virtual)
endforeach()
 

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${__CMAKE_CUDA_ARCHS})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${CUTE_INCLUDE_DIR}
)


set(CUBLASLT_LIBRARY cublasLt)
target_link_libraries(${PROJECT_NAME}
    ${CUBLASLT_LIBRARY}
)