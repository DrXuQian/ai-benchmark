cmake_minimum_required(VERSION 3.16)
project(test_libspconv_accuracy CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
# 7.0 = V100, 7.5 = T4/RTX20xx, 8.0 = A100, 8.6 = RTX30xx, 8.9 = RTX40xx
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/spconv_standalone/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cumm/include  # If cumm headers are needed
    ${CUDA_INCLUDE_DIRS}
    /tmp  # For json.hpp
)

# Collect all source files from spconv_standalone/src
file(GLOB_RECURSE SPCONV_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/spconv_standalone/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spconv_standalone/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/spconv_standalone/src/*.cpp"
)

# Filter out test files and examples if any
list(FILTER SPCONV_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER SPCONV_SOURCES EXCLUDE REGEX ".*example.*")

# Create the libspconv library
add_library(spconv STATIC ${SPCONV_SOURCES})
target_compile_features(spconv PUBLIC cxx_std_14)
set_target_properties(spconv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set compile options
target_compile_options(spconv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_75,code=sm_75
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86
        -gencode arch=compute_89,code=sm_89
        -Xcompiler -fPIC
        -lineinfo
        --use_fast_math
    >
)

# Link libraries for spconv
target_link_libraries(spconv
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
    CUDA::cufft
    CUDA::curand
)

# Create test executable
add_executable(test_libspconv_accuracy test_libspconv_accuracy.cpp)

# Link test with spconv library
target_link_libraries(test_libspconv_accuracy
    spconv
    CUDA::cudart
    CUDA::cublas
    pthread
)

# Set properties for the test executable
set_target_properties(test_libspconv_accuracy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Add compile definitions
target_compile_definitions(spconv PRIVATE
    CUMM_CUDA_VERSION=${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
    CUMM_DISABLE_JIT=1
    SPCONV_DISABLE_JIT=1
)

message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Found ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "Number of source files: ${CMAKE_CURRENT_LIST_LINE}")