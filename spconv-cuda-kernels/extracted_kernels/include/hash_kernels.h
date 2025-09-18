// Header file for hash kernels
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#pragma once

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/hash/ops.h>

// ============================================================================
// HashTableKernel kernel declarations
// ============================================================================

template<typename THashTableSplit>
__global__ void insert_exist_keys_kernel(
    THashTableSplit table,
    const typename THashTableSplit::key_type* __restrict__ key_ptr,
    const typename THashTableSplit::mapped_type* __restrict__ value_ptr,
    uint8_t* is_empty_ptr,
    size_t size
);