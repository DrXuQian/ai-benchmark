// Test program for extracted spconv kernels
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// Simple replacements for TensorView dependencies
namespace tv {
    template<typename T>
    struct KernelLoopX {
        int total;
        __device__ KernelLoopX(int size) : total(size) {}

        struct Iterator {
            int idx;
            int step;
            int total;

            __device__ Iterator(int start, int s, int t) : idx(start), step(s), total(t) {}
            __device__ Iterator& operator++() { idx += step; return *this; }
            __device__ int operator*() const { return idx; }
            __device__ bool operator!=(const Iterator& other) const { return idx < total; }
        };

        __device__ Iterator begin() const {
            return Iterator(blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x, total);
        }
        __device__ Iterator end() const { return Iterator(total, 0, total); }
    };

    namespace cuda {
        __device__ __forceinline__ int atomicAggInc(int* address) {
            return atomicAdd(address, 1);
        }
    }
}

// Include simplified kernel implementations
template<typename T>
__global__ void arange_kernel(T* data, int size) {
    for (int i : tv::KernelLoopX<int>(size)) {
        data[i] = T(i);
    }
}

template<typename T>
__global__ void fill_kernel(T* data, T val, int size) {
    for (int i : tv::KernelLoopX<int>(size)) {
        data[i] = T(val);
    }
}

template<typename T>
__global__ void maximum_value_kernel(T* data, T val, int size) {
    for (int i : tv::KernelLoopX<int>(size)) {
        data[i] = max(data[i], val);
    }
}

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

void test_arange_kernel() {
    std::cout << "Testing arange_kernel..." << std::endl;

    const int size = 1024;
    int* d_data;
    std::vector<int> h_data(size);

    CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    arange_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, size * sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < size; i++) {
        if (h_data[i] != i) {
            passed = false;
            std::cerr << "Mismatch at index " << i << ": expected " << i
                      << ", got " << h_data[i] << std::endl;
            break;
        }
    }

    if (passed) {
        std::cout << "✓ arange_kernel test passed!" << std::endl;
    } else {
        std::cout << "✗ arange_kernel test failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_data));
}

void test_fill_kernel() {
    std::cout << "Testing fill_kernel..." << std::endl;

    const int size = 1024;
    const float fill_value = 42.5f;
    float* d_data;
    std::vector<float> h_data(size);

    CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(float)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    fill_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, fill_value, size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < size; i++) {
        if (h_data[i] != fill_value) {
            passed = false;
            std::cerr << "Mismatch at index " << i << ": expected " << fill_value
                      << ", got " << h_data[i] << std::endl;
            break;
        }
    }

    if (passed) {
        std::cout << "✓ fill_kernel test passed!" << std::endl;
    } else {
        std::cout << "✗ fill_kernel test failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_data));
}

void test_maximum_value_kernel() {
    std::cout << "Testing maximum_value_kernel..." << std::endl;

    const int size = 1024;
    const float max_value = 50.0f;
    float* d_data;
    std::vector<float> h_data(size);

    // Initialize with some values
    for (int i = 0; i < size; i++) {
        h_data[i] = static_cast<float>(i % 100);
    }

    CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    maximum_value_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, max_value, size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < size; i++) {
        float expected = std::max(static_cast<float>(i % 100), max_value);
        if (h_data[i] != expected) {
            passed = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected
                      << ", got " << h_data[i] << std::endl;
            break;
        }
    }

    if (passed) {
        std::cout << "✓ maximum_value_kernel test passed!" << std::endl;
    } else {
        std::cout << "✗ maximum_value_kernel test failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    std::cout << "=== Testing Extracted SPConv Kernels ===" << std::endl;

    // Check CUDA device
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Run tests
    test_arange_kernel();
    std::cout << std::endl;

    test_fill_kernel();
    std::cout << std::endl;

    test_maximum_value_kernel();
    std::cout << std::endl;

    std::cout << "=== All tests completed ===" << std::endl;

    return 0;
}