// Extracted CUDA kernels from spconv/csrc/hash/core.py
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/hash/ops.h>

// ============================================================================
// HashTableKernel kernels
// ============================================================================

template<typename THashTableSplit>
__global__ void insert_exist_keys_kernel(
    THashTableSplit table,
    const typename THashTableSplit::key_type* __restrict__ key_ptr,
    const typename THashTableSplit::mapped_type* __restrict__ value_ptr,
    uint8_t* is_empty_ptr,
    size_t size
) {
    auto value_data = table.value_ptr();
    for (size_t i : tv::KernelLoopX<size_t>(size)) {
        auto key = key_ptr[i];
        auto offset = table.lookup_offset(key);
        is_empty_ptr[i] = offset == -1;
        if (offset != -1) {
            value_data[offset] = value_ptr[i];
        }
    }
}