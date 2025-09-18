// Header file for inference kernels
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#pragma once

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/gemm/core/constants.h>

// ============================================================================
// InferenceOpsKernel kernel declarations
// ============================================================================

template<typename T, bool OneDim = false>
__global__ void bias_add_inplace_kernel(
    T* out_features,
    const T* bias,
    int size,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T, bool OneDim = false>
__global__ void bias_add_act_inplace_kernel(
    T* out_features,
    const T* bias,
    tv::gemm::Activation act_type,
    T alpha,
    T beta,
    int size,
    int num_features,
    int num_blocks_x,
    int num_blocks_y
);

template<typename T>
__global__ void activation_inplace_kernel(
    T* out_features,
    tv::gemm::Activation act_type,
    T alpha,
    T beta,
    int size
);