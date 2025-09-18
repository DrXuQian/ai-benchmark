// Core sparse convolution CUDA kernels extracted from spconv
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <cub/cub.cuh>
#include <limits>

// ============================================================================
// Gather Features Kernel - Core sparse convolution operation
// ============================================================================

template<typename T>
__global__ void gather_features_kernel(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features
) {
    // Each thread handles one feature channel for one spatial location
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < num_indices && feature_idx < num_features) {
        int in_idx = indices[idx];
        if (in_idx >= 0) {  // Valid index
            out_features[idx * num_features + feature_idx] =
                in_features[in_idx * num_features + feature_idx];
        } else {
            // Invalid index, zero out
            out_features[idx * num_features + feature_idx] = T(0);
        }
    }
}

// Optimized version using memory coalescing
template<typename T, bool OneDim = false>
__global__ void gather_features_kernel_optimized(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        int in_idx = indices[i];
        auto out_ptr = out_features + i * num_features;

        if (in_idx >= 0) {
            auto in_ptr = in_features + in_idx * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
                out_ptr[j] = in_ptr[j];
            }
        } else {
            // Invalid index, zero out
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
                out_ptr[j] = T(0);
            }
        }
    }
}

// ============================================================================
// Scatter Add Kernel - Accumulate results back to output
// ============================================================================

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
        if (out_idx >= 0) {  // Valid index
            T value = in_features[idx * num_features + feature_idx];
            atomicAdd(&out_features[out_idx * num_features + feature_idx], value);
        }
    }
}

// Optimized version using memory coalescing
template<typename T, bool OneDim = false>
__global__ void scatter_add_kernel_optimized(
    T* out_features,
    const T* in_features,
    const int* indices,
    int num_indices,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        int out_idx = indices[i];
        auto in_ptr = in_features + i * num_features;

        if (out_idx >= 0) {
            auto out_ptr = out_features + out_idx * num_features;
            for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
                atomicAdd(&out_ptr[j], in_ptr[j]);
            }
        }
    }
}

// ============================================================================
// Implicit GEMM Sparse Convolution Forward Kernel
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
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        auto out_ptr = out_features + i * num_out_features;

        for (int oc : tv::KernelLoopX<int>(num_out_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
            T accumulator = T(0);

            // Iterate through kernel volume
            for (int k = 0; k < RS; ++k) {
                int in_idx = indice_pairs[k * num_indices + i];

                if (in_idx >= 0) {  // Valid input index
                    auto in_ptr = in_features + in_idx * num_in_features;
                    auto filter_ptr = filters + (k * num_out_features + oc) * num_in_features;

                    // Perform dot product for this output channel
                    for (int ic = 0; ic < num_in_features; ++ic) {
                        accumulator += in_ptr[ic] * filter_ptr[ic];
                    }
                }
            }

            out_ptr[oc] = accumulator;
        }
    }
}

// ============================================================================
// Implicit GEMM Sparse Convolution Backward Kernel (Input Gradients)
// ============================================================================

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
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        auto din_ptr = din_features + i * num_in_features;

        for (int ic : tv::KernelLoopX<int>(num_in_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
            T accumulator = T(0);

            // Iterate through kernel volume
            for (int k = 0; k < RS; ++k) {
                int out_idx = indice_pairs[k * num_indices + i];

                if (out_idx >= 0) {  // Valid output index
                    auto dout_ptr = dout_features + out_idx * num_out_features;
                    auto filter_ptr = filters + k * num_out_features * num_in_features + ic;

                    // Accumulate gradients for this input channel
                    for (int oc = 0; oc < num_out_features; ++oc) {
                        accumulator += dout_ptr[oc] * filter_ptr[oc * num_in_features];
                    }
                }
            }

            din_ptr[ic] = accumulator;
        }
    }
}

// ============================================================================
// Implicit GEMM Sparse Convolution Backward Kernel (Filter Gradients)
// ============================================================================

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
) {
    int k = blockIdx.x;  // Kernel position
    int oc = blockIdx.y; // Output channel
    int ic = threadIdx.x; // Input channel

    if (k < RS && oc < num_out_features && ic < num_in_features) {
        T accumulator = T(0);

        // Accumulate over all valid spatial locations
        for (int i = 0; i < num_indices; ++i) {
            int in_idx = indice_pairs[k * num_indices + i];

            if (in_idx >= 0) {  // Valid pair
                T in_val = in_features[in_idx * num_in_features + ic];
                T dout_val = dout_features[i * num_out_features + oc];
                accumulator += in_val * dout_val;
            }
        }

        int filter_idx = (k * num_out_features + oc) * num_in_features + ic;
        atomicAdd(&dfilters[filter_idx], accumulator);
    }
}

// ============================================================================
// Multi-kernel Implicit GEMM (for multiple kernel sizes)
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
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        auto out_ptr = out_features + i * num_features;

        for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
            auto indices_ptr = indices + i;
            T result = default_value;

            // Process first kernel position
            int in_idx = indices_ptr[0];
            bool valid = in_idx != -1;
            T in_val = valid ? in_features[in_idx * num_features + j] : default_value;
            result = valid ? in_val : result;
            indices_ptr += num_indices;

            // Process remaining kernel positions
            for (int k = 1; k < RS; ++k) {
                in_idx = indices_ptr[0];
                valid = in_idx != -1;
                in_val = valid ? in_features[in_idx * num_features + j] : default_value;

                // Apply operation (could be max, sum, etc.)
                result = valid ? (result + in_val) : result;  // Sum operation
                indices_ptr += num_indices;
            }

            out_ptr[j] = result;
        }
    }
}

// ============================================================================
// Utility kernels for sparse convolution
// ============================================================================

template<typename T>
__global__ void sparse_conv_bias_add_kernel(
    T* features,
    const T* bias,
    int num_points,
    int num_channels
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_idx < num_points && channel_idx < num_channels) {
        features[point_idx * num_channels + channel_idx] += bias[channel_idx];
    }
}

template<typename T>
__global__ void sparse_conv_activation_kernel(
    T* features,
    int num_elements,
    int activation_type,  // 0: ReLU, 1: LeakyReLU, etc.
    T alpha = T(0.01)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        T val = features[idx];

        switch (activation_type) {
            case 0:  // ReLU
                features[idx] = fmaxf(val, T(0));
                break;
            case 1:  // LeakyReLU
                features[idx] = val > T(0) ? val : alpha * val;
                break;
            default:
                // No activation
                break;
        }
    }
}

// ============================================================================
// Sparse Convolution Max Pool Style Implicit GEMM (from maxpool.py)
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
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        auto out_ptr = out_features + i * num_features;
        for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
            auto indices_ptr = indices + i;
            int in_idx = indices_ptr[0];
            T in, in_temp;
            in = lowest;
            bool valid = in_idx != -1;
            in_temp = valid ? in_features[in_idx * num_features + j] : lowest;
            in = (in < in_temp && valid) ? in_temp : in;
            indices_ptr += num_indices;

            for (int k = 1; k < RS; ++k) {
                in_idx = indices_ptr[0];
                valid = in_idx != -1;
                in_temp = valid ? in_features[in_idx * num_features + j] : lowest;
                in = (in < in_temp && valid) ? in_temp : in;
                indices_ptr += num_indices;
            }
            out_ptr[j] = in;
        }
    }
}

// ============================================================================
// Sparse Convolution Average Pool Style Implicit GEMM
// ============================================================================

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
) {
    int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
    int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;

    for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
        auto out_ptr = out_features + i * num_features;
        auto indices_ptr = indices + i;
        int in_idx = 0;
        int count = 0;

        // Count valid indices
        for (int k = 0; k < RS; ++k) {
            in_idx = indices_ptr[0];
            count += int(in_idx != -1);
            indices_ptr += num_indices;
        }

        if (count_out != nullptr) {
            count_out[i] = count;
        }

        for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
            indices_ptr = indices + i;
            int in_idx;
            T in, in_temp;
            in = T(0);

            for (int k = 0; k < RS; ++k) {
                in_idx = indices_ptr[0];
                bool valid = in_idx != -1;
                in_temp = valid ? in_features[in_idx * num_features + j] : T(0);
                in += in_temp;
                indices_ptr += num_indices;
            }

            out_ptr[j] = count > 0 ? in / T(count) : T(0);
        }
    }
}