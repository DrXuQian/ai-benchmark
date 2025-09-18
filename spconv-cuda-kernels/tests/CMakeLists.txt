cmake_minimum_required(VERSION 3.16)

# Download json.hpp if needed
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/json.hpp")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/json.hpp"
    )
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Simple compilation test
add_executable(test_simple test_simple_libspconv.cpp)
target_link_libraries(test_simple spconv CUDA::cudart)

# Accuracy test (requires test data from Python)
add_executable(test_accuracy test_libspconv_accuracy.cpp)
target_link_libraries(test_accuracy spconv CUDA::cudart)

# Add test target
enable_testing()
add_test(NAME SimpleTest COMMAND test_simple)
add_test(NAME AccuracyTest COMMAND test_accuracy)