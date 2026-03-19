cmake_minimum_required(VERSION 3.18)
project(GPU_Kernel LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
# Note: CUDA language is already enabled via project() command above
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/include)

# CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Function to add a kernel executable
function(add_kernel_executable kernel_name)
    set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/src/${kernel_name})
    set(KERNEL_SOURCE ${KERNEL_DIR}/${kernel_name}.cu)
    
    if(EXISTS ${KERNEL_SOURCE})
        add_executable(${kernel_name} ${KERNEL_SOURCE})
        target_link_libraries(${kernel_name} PRIVATE CUDA::cudart)
        set_target_properties(${kernel_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    else()
        message(WARNING "Kernel source ${KERNEL_SOURCE} not found, skipping ${kernel_name}")
    endif()
endfunction()

# Add all kernels
add_kernel_executable(sm_info)
add_kernel_executable(matrix_multiply)
add_kernel_executable(vector_add)
add_kernel_executable(matrix_transpose)
add_kernel_executable(flash_attn)
add_kernel_executable(parallel_reduction)
add_kernel_executable(gemm)
target_link_libraries(gemm PRIVATE CUDA::cublas)

# Reduction benchmark (NVIDIA-style 4M element performance table)
add_executable(reduction_benchmark src/parallel_reduction/reduction_benchmark.cu)
target_link_libraries(reduction_benchmark PRIVATE CUDA::cudart)
set_target_properties(reduction_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

