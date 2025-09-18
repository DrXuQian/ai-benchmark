cmake_minimum_required(VERSION 3.16)
project(spconv_cuda_kernels CUDA CXX)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/libspconv/spconv_standalone/include
    ${CUDA_INCLUDE_DIRS}
)

# Option to build tests
option(BUILD_TESTS "Build test programs" ON)

# Collect libspconv source files
file(GLOB_RECURSE SPCONV_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/libspconv/spconv_standalone/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/libspconv/spconv_standalone/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/libspconv/spconv_standalone/src/*.cpp"
)

# Filter out test files
list(FILTER SPCONV_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER SPCONV_SOURCES EXCLUDE REGEX ".*example.*")

# Create the libspconv static library
add_library(spconv STATIC ${SPCONV_SOURCES})
target_compile_features(spconv PUBLIC cxx_std_14)
set_target_properties(spconv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set compile options
target_compile_options(spconv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
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
)

# Add compile definitions
target_compile_definitions(spconv PRIVATE
    CUMM_CUDA_VERSION=${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
    CUMM_DISABLE_JIT=1
    SPCONV_DISABLE_JIT=1
)

# Build tests if requested
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Installation rules
install(TARGETS spconv
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY libspconv/spconv_standalone/include/
    DESTINATION include
)

# Print configuration
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build tests: ${BUILD_TESTS}")