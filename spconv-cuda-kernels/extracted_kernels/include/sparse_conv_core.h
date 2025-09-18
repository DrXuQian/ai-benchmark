// Core sparse convolution CUDA kernel declarations
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#pragma once

#include <cuda_runtime.h>
#include <tensorview/tensor.h>

// ============================================================================
// Gather Features Kernels
// ============================================================================

template<typename T>
__global__ void gather_features_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features
);

template<typename T, bool OneDim = false>
__global__ void gather_features_kernel_optimized(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
);

// ============================================================================
// Scatter Add Kernels
// ============================================================================

template<typename T>
__global__ void scatter_add_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features
);

template<typename T, bool OneDim = false>
__global__ void scatter_add_kernel_optimized(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
);

// ============================================================================
// Implicit GEMM Sparse Convolution Kernels
// ============================================================================

template<typename T, bool OneDim = false>
__global__ void sparse_conv_implicit_gemm_forward_kernel(
    T* out_features,
    const T* in_features,
    const T* filters,
    const int* indice_pairs,
    int num_out_features,
    int num_in_features,
    int num_filters,
    int kv_size,
    int RS,
    int num_indices,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T, bool OneDim = false>
__global__ void sparse_conv_implicit_gemm_backward_input_kernel(
    T* din_features,
    const T* dout_features,
    const T* filters,
    const int* indice_pairs,
    int num_out_features,
    int num_in_features,
    int RS,
    int num_indices,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T>
__global__ void sparse_conv_implicit_gemm_backward_filter_kernel(
    T* dfilters,
    const T* in_features,
    const T* dout_features,
    const int* indice_pairs,
    int num_out_features,
    int num_in_features,
    int RS,
    int num_indices
);

// ============================================================================
// Multi-kernel Implicit GEMM
// ============================================================================

template<typename T, bool OneDim = false>
__global__ void sparse_conv_multi_implicit_gemm_forward_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_features,
    int RS,
    int num_indices,
    T default_value,
    int num_blocks_x,
    int num_blocks_y
);

// ============================================================================
// Utility Kernels
// ============================================================================

template<typename T>
__global__ void sparse_conv_bias_add_kernel(
    T* features,
    const T* bias,
    int num_points,
    int num_channels
);

template<typename T>
__global__ void sparse_conv_activation_kernel(
    T* features,
    int num_elements,
    int activation_type,
    T alpha = T(0.01)
);

// ============================================================================
// Sparse Convolution Pooling Style Implicit GEMM Kernels
// ============================================================================

template<typename T, bool OneDim = false>
__global__ void sparse_conv_maxpool_implicit_gemm_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_features,
    int RS,
    int num_indices,
    T lowest,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T, bool OneDim = false>
__global__ void sparse_conv_avgpool_implicit_gemm_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int* count_out,
    int num_features,
    int RS,
    int num_indices,
    int num_blocks_x,
    int num_blocks_y
);

// ============================================================================
// Host API Functions
// ============================================================================

namespace sparse_conv {

// Launch gather features operation
template<typename T>
void launch_gather_features(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features,
    cudaStream_t stream = 0
);

// Launch scatter add operation
template<typename T>
void launch_scatter_add(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features,
    cudaStream_t stream = 0
);

// Launch sparse convolution forward pass
template<typename T>
void launch_sparse_conv_forward(
    T* out_features,
    const T* in_features,
    const T* filters,
    const int* indice_pairs,
    int num_out_features,
    int num_in_features,
    int RS,
    int num_indices,
    cudaStream_t stream = 0
);

// Launch sparse convolution backward pass (input gradients)
template<typename T>
void launch_sparse_conv_backward_input(
    T* din_features,
    const T* dout_features,
    const T* filters,
    const int* indice_pairs,
    int num_out_features,
    int num_in_features,
    int RS,
    int num_indices,
    cudaStream_t stream = 0
);

// Launch sparse convolution backward pass (filter gradients)
template<typename T>
void launch_sparse_conv_backward_filter(
    T* dfilters,
    const T* in_features,
    const T* dout_features,
    const int* indice_pairs,
    int num_out_features,
    int num_in_features,
    int RS,
    int num_indices,
    cudaStream_t stream = 0
);

// Launch maxpool style implicit GEMM
template<typename T>
void launch_sparse_conv_maxpool_implicit_gemm(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_features,
    int RS,
    int num_indices,
    T lowest,
    cudaStream_t stream = 0
);

// Launch avgpool style implicit GEMM
template<typename T>
void launch_sparse_conv_avgpool_implicit_gemm(
    T* out_features,
    const T* in_features,
    const int* indices,
    int* count_out,
    int num_features,
    int RS,
    int num_indices,
    cudaStream_t stream = 0
);

} // namespace sparse_conv

// ============================================================================
// Kernel Launch Configuration Helpers
// ============================================================================

struct SparseConvKernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;

    SparseConvKernelConfig(int num_indices, int num_features, int max_threads_per_block = 256);
};

// Calculate optimal launch configuration for 2D kernels
SparseConvKernelConfig get_2d_launch_config(int dim_x, int dim_y, int max_threads = 256);

// Calculate optimal launch configuration for 1D kernels
SparseConvKernelConfig get_1d_launch_config(int num_elements, int max_threads = 256);