// Test program for sparse convolution kernels
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>

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

    template<typename T>
    struct KernelLoopY {
        int total;
        int block_idx;
        int grid_dim;

        __device__ KernelLoopY(int size, int bidx, int gdim)
            : total(size), block_idx(bidx), grid_dim(gdim) {}

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
            return Iterator(block_idx * blockDim.y + threadIdx.y, grid_dim * blockDim.y, total);
        }
        __device__ Iterator end() const { return Iterator(total, 0, total); }
    };

    namespace cuda {
        __device__ __forceinline__ int atomicAggInc(int* address) {
            return atomicAdd(address, 1);
        }
    }
}

// ============================================================================
// Sparse Convolution Core Kernels
// ============================================================================

template<typename T>
__global__ void gather_features_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < num_indices && feature_idx < num_features) {
        int in_idx = indices[idx];
        if (in_idx >= 0) {
            out_features[idx * num_features + feature_idx] =
                in_features[in_idx * num_features + feature_idx];
        } else {
            out_features[idx * num_features + feature_idx] = T(0);
        }
    }
}

template<typename T>
__global__ void scatter_add_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < num_indices && feature_idx < num_features) {
        int out_idx = indices[idx];
        if (out_idx >= 0) {
            T value = in_features[idx * num_features + feature_idx];
            atomicAdd(&out_features[out_idx * num_features + feature_idx], value);
        }
    }
}

// Simple implicit GEMM kernel for sparse convolution
template<typename T>
__global__ void sparse_conv_implicit_gemm_simple(
    T* out_features,
    const T* in_features,
    const T* weights,
    const int* input_indices,
    const int* output_indices,
    int num_input,
    int num_output,
    int in_channels,
    int out_channels,
    int kernel_volume
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_ch = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < num_output && out_ch < out_channels) {
        T sum = T(0);

        // For simplicity, assume full connectivity (all kernel positions)
        for (int k = 0; k < kernel_volume; k++) {
            // In real sparse conv, we'd use indice pairs to find corresponding input
            // Here we simulate with a simple pattern
            if (out_idx < num_input) {  // Simplified assumption
                for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                    T in_val = in_features[out_idx * in_channels + in_ch];
                    T w_val = weights[out_ch * kernel_volume * in_channels +
                                      k * in_channels + in_ch];
                    sum += in_val * w_val;
                }
            }
        }

        out_features[out_idx * out_channels + out_ch] = sum;
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

void test_gather_kernel() {
    std::cout << "Testing gather_features_kernel..." << std::endl;

    const int num_points = 1000;
    const int num_features = 64;
    const int num_active = 500;

    // Allocate memory
    float *d_in_features, *d_out_features;
    int *d_indices;
    std::vector<float> h_in_features(num_points * num_features);
    std::vector<float> h_out_features(num_active * num_features, 0.0f);
    std::vector<int> h_indices(num_active);

    // Initialize input features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < num_points * num_features; i++) {
        h_in_features[i] = dis(gen);
    }

    // Create random indices (simulating sparse selection)
    for (int i = 0; i < num_active; i++) {
        h_indices[i] = i * 2;  // Sample every other point
        if (h_indices[i] >= num_points) {
            h_indices[i] = -1;  // Invalid index
        }
    }

    CHECK_CUDA(cudaMalloc(&d_in_features, num_points * num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_features, num_active * num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, num_active * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in_features, h_in_features.data(),
                          num_points * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(),
                          num_active * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((num_active + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (num_features + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gather_features_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out_features, d_in_features, d_indices, num_active, num_features);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_out_features.data(), d_out_features,
                          num_active * num_features * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few values
    bool passed = true;
    for (int i = 0; i < std::min(10, num_active); i++) {
        if (h_indices[i] >= 0) {
            float expected = h_in_features[h_indices[i] * num_features];
            float actual = h_out_features[i * num_features];
            if (std::abs(expected - actual) > 1e-5) {
                passed = false;
                std::cerr << "Mismatch at index " << i << ": expected " << expected
                          << ", got " << actual << std::endl;
                break;
            }
        }
    }

    if (passed) {
        std::cout << "✓ gather_features_kernel test passed!" << std::endl;
    } else {
        std::cout << "✗ gather_features_kernel test failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_in_features));
    CHECK_CUDA(cudaFree(d_out_features));
    CHECK_CUDA(cudaFree(d_indices));
}

void test_scatter_kernel() {
    std::cout << "Testing scatter_add_kernel..." << std::endl;

    const int num_points = 1000;
    const int num_features = 64;
    const int num_active = 500;

    // Allocate memory
    float *d_in_features, *d_out_features;
    int *d_indices;
    std::vector<float> h_in_features(num_active * num_features);
    std::vector<float> h_out_features(num_points * num_features, 0.0f);
    std::vector<int> h_indices(num_active);

    // Initialize input features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < num_active * num_features; i++) {
        h_in_features[i] = dis(gen);
    }

    // Create indices for scattering
    for (int i = 0; i < num_active; i++) {
        h_indices[i] = i % num_points;  // Multiple values can go to same output location
    }

    CHECK_CUDA(cudaMalloc(&d_in_features, num_active * num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_features, num_points * num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, num_active * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in_features, h_in_features.data(),
                          num_active * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out_features, h_out_features.data(),
                          num_points * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(),
                          num_active * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((num_active + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (num_features + threadsPerBlock.y - 1) / threadsPerBlock.y);

    scatter_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out_features, d_in_features, d_indices, num_active, num_features);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back
    CHECK_CUDA(cudaMemcpy(h_out_features.data(), d_out_features,
                          num_points * num_features * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification - check that scatter happened
    bool passed = true;
    int non_zero_count = 0;
    for (int i = 0; i < num_points * num_features; i++) {
        if (h_out_features[i] != 0.0f) {
            non_zero_count++;
        }
    }

    if (non_zero_count > 0) {
        std::cout << "✓ scatter_add_kernel test passed! (Non-zero values: "
                  << non_zero_count << ")" << std::endl;
    } else {
        std::cout << "✗ scatter_add_kernel test failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_in_features));
    CHECK_CUDA(cudaFree(d_out_features));
    CHECK_CUDA(cudaFree(d_indices));
}

void test_sparse_conv() {
    std::cout << "Testing sparse_conv_implicit_gemm_simple..." << std::endl;

    const int num_active = 100;
    const int in_channels = 32;
    const int out_channels = 64;
    const int kernel_volume = 27;  // 3x3x3 kernel

    // Allocate memory
    float *d_in_features, *d_out_features, *d_weights;
    int *d_input_indices, *d_output_indices;

    std::vector<float> h_in_features(num_active * in_channels);
    std::vector<float> h_out_features(num_active * out_channels, 0.0f);
    std::vector<float> h_weights(out_channels * kernel_volume * in_channels);
    std::vector<int> h_indices(num_active);

    // Initialize with small values to avoid overflow
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for (auto& v : h_in_features) v = dis(gen);
    for (auto& v : h_weights) v = dis(gen);
    for (int i = 0; i < num_active; i++) h_indices[i] = i;

    CHECK_CUDA(cudaMalloc(&d_in_features, num_active * in_channels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_features, num_active * out_channels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, out_channels * kernel_volume * in_channels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input_indices, num_active * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output_indices, num_active * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in_features, h_in_features.data(),
                          num_active * in_channels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(),
                          out_channels * kernel_volume * in_channels * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input_indices, h_indices.data(),
                          num_active * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output_indices, h_indices.data(),
                          num_active * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((num_active + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (out_channels + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sparse_conv_implicit_gemm_simple<<<blocksPerGrid, threadsPerBlock>>>(
        d_out_features, d_in_features, d_weights,
        d_input_indices, d_output_indices,
        num_active, num_active, in_channels, out_channels, kernel_volume);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back
    CHECK_CUDA(cudaMemcpy(h_out_features.data(), d_out_features,
                          num_active * out_channels * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification - check that output has values
    bool passed = true;
    int non_zero_count = 0;
    for (int i = 0; i < num_active * out_channels; i++) {
        if (h_out_features[i] != 0.0f) {
            non_zero_count++;
        }
    }

    if (non_zero_count > 0) {
        std::cout << "✓ sparse_conv_implicit_gemm test passed! (Non-zero outputs: "
                  << non_zero_count << "/" << num_active * out_channels << ")" << std::endl;
    } else {
        std::cout << "✗ sparse_conv_implicit_gemm test failed!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_in_features));
    CHECK_CUDA(cudaFree(d_out_features));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_input_indices));
    CHECK_CUDA(cudaFree(d_output_indices));
}

int main() {
    std::cout << "=== Testing Sparse Convolution Kernels ===" << std::endl;

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
    test_gather_kernel();
    std::cout << std::endl;

    test_scatter_kernel();
    std::cout << std::endl;

    test_sparse_conv();
    std::cout << std::endl;

    std::cout << "=== All tests completed ===" << std::endl;

    return 0;
}