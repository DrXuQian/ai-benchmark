// Header file for maxpool kernels
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#pragma once

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <limits>

// ============================================================================
// IndiceMaxPool kernel declarations
// ============================================================================

template<typename T, bool OneDim = false>
__global__ void forward_kernel(
    T* out_features,
    const T* in_features,
    const int* out_indices,
    const int* in_indices,
    int size,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T, bool OneDim = false>
__global__ void forward_implicit_gemm_kernel(
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
__global__ void backward_kernel(
    const T* out_features,
    const T* in_features,
    const T* dout_features,
    T* din_features,
    const int* out_indices,
    const int* in_indices,
    int size,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T, bool OneDim = false>
__global__ void backward_implicit_gemm_kernel(
    const T* out_features,
    const T* in_features,
    const T* dout_features,
    T* din_features,
    const int* indices_bwd,
    int num_features,
    int RS,
    int num_indices,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T, bool OneDim = false>
__global__ void forward_avgpool_implicit_gemm_kernel(
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

template<typename T, bool OneDim = false>
__global__ void backward_avgpool_implicit_gemm_kernel(
    const T* dout_features,
    T* din_features,
    const int* indices_bwd,
    const int* count_out,
    int num_features,
    int RS,
    int num_indices,
    int num_blocks_x,
    int num_blocks_y
);

__global__ void global_pool_rearrange_kernel(
    int* out_indices,
    const int* coords,
    int* counts,
    int num_indices,
    int indices_stride
);