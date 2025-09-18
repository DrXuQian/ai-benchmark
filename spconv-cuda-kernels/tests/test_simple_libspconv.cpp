/*
Simple test to verify libspconv compilation and basic functionality
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Test if we can include the main headers
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>

using namespace spconvlib;

int main() {
    std::cout << "Testing libspconv compilation..." << std::endl;

    // Check CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // Test creating a simple tensor
    std::vector<int64_t> shape = {10, 20};
    tv::Tensor tensor = tv::empty(shape, tv::float32, 0);  // GPU tensor
    std::cout << "Created tensor with shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
    std::cout << "Tensor size: " << tensor.size() << std::endl;

    // Test SpconvOps
    std::cout << "CUMM version: " << spconv::csrc::sparse::all::SpconvOps::cumm_version() << std::endl;
    std::cout << "PCCM version: " << spconv::csrc::sparse::all::SpconvOps::pccm_version() << std::endl;

    std::cout << "\n✅ libspconv basic test passed!" << std::endl;
    std::cout << "The library is correctly compiled and linked." << std::endl;

    return 0;
}