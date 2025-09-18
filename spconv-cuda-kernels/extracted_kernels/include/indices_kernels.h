// Header file for indices kernels
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#pragma once

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/hash/ops.h>
#include <limits>

// ============================================================================
// CudaCommonKernel kernel declarations
// ============================================================================

template<typename T>
__global__ void arange_kernel(T* data, int size);

template<typename T>
__global__ void fill_kernel(T* data, T val, int size);

template<typename T>
__global__ void maximum_value_kernel(T* data, T val, int size);

// ============================================================================
// SparseConvIndicesKernel kernel declarations
// ============================================================================

template<typename TIndiceUniq, typename TConvLocIter>
__global__ void calc_conv_indices_stage1(
    TConvLocIter loc_iter,
    const int* indices_in,
    int* indice_pairs,
    TIndiceUniq* indice_pairs_for_uniq,
    int* indice_num_per_loc,
    int num_indices_in,
    int indices_pair_size,
    int RS,
    bool transposed
);

template<typename TTable, typename TLayoutNPQ>
__global__ void build_conv_hash_table(
    TTable table,
    int* indices_out,
    const typename TTable::key_type* indice_pairs_for_uniq,
    TLayoutNPQ layout_npq,
    int num_indices
);

template<typename TTable, typename TLayoutNPQ>
__global__ void arange_hash_table_and_assign_out(
    TTable table,
    int* indices_out,
    int* count,
    int limit,
    TLayoutNPQ layout_npq
);

template<typename TTable>
__global__ void arange_hash_table(
    TTable table,
    typename TTable::key_type* out_indices_offset,
    int* count,
    int limit
);

template<typename T, typename TLayoutNPQ>
__global__ void assign_out_indices(
    int* indices_out,
    const T* out_indices_offset,
    TLayoutNPQ layout_npq,
    int size
);

template<typename TTable>
__global__ void calc_conv_indices_stage2(
    TTable table,
    const typename TTable::key_type* indice_pairs_uniq_before_sort,
    int* indice_pairs_out_part,
    int num_indices_in,
    int indices_pair_size
);

template<typename TTable>
__global__ void calc_conv_indices_stage2_bounded(
    TTable table,
    const typename TTable::key_type* indice_pairs_uniq_before_sort,
    const int* indice_pairs_in_part_temp,
    int* indice_pairs_in_part,
    int* indice_pairs_out_part,
    int* indice_num_per_loc,
    int num_indices_in,
    int indices_pair_size
);

template<typename TIndiceUniq, typename TConvLocIter>
__global__ void calc_conv_indices_stage1_mask(
    TConvLocIter loc_iter,
    const int* indices_in,
    int* indice_pairs_bwd,
    TIndiceUniq* indice_pairs_for_uniq,
    int* indice_num_per_loc,
    int num_indices_in,
    int RS,
    bool transposed
);

template<typename TIndiceUniq, typename TTable, typename TConvLocIter>
__global__ void calc_conv_indices_stage1_mask_direct_table(
    TTable table,
    TConvLocIter loc_iter,
    const int* indices_in,
    int* indice_pairs_bwd,
    TIndiceUniq* indice_pairs_for_uniq,
    int* indice_num_per_loc,
    int num_indices_in,
    int RS,
    bool transposed
);

template<typename TTable, bool CheckValueValid>
__global__ void calc_conv_indices_stage2_mask(
    TTable table,
    int* indice_pairs_fwd,
    int* indice_pairs_bwd,
    const typename TTable::key_type* indice_pairs_uniq_before_sort,
    uint32_t* mask_fwd,
    uint32_t* mask_bwd,
    int num_indices_in,
    int num_indices_out,
    int mask_int_count
);

__global__ void calc_conv_indices_stage2_mask_output(
    int* indice_pairs_bwd,
    uint32_t* mask_bwd,
    int num_indices_in,
    int kv,
    int mask_int_count
);

template<typename TTable, bool CheckValueValid>
__global__ void calc_conv_indices_stage2_inference_mask(
    TTable table,
    int* indice_pairs_fwd,
    int* indice_pairs_bwd,
    const typename TTable::key_type* indice_pairs_uniq_before_sort,
    uint32_t* mask_fwd,
    int num_indices_in,
    int num_indices_out,
    int mask_int_count
);

template<typename TTable, typename TLayoutNPQ>
__global__ void build_subm_conv_hash_table(
    TTable table,
    const int* indices_in,
    TLayoutNPQ layout_npq,
    int num_indices
);

template<typename T>
__global__ void clean_indices_uniq(T* indice_pairs_for_uniq, size_t size);

template<typename TTable, typename TConvLocIter>
__global__ void calc_subm_conv_indices(
    TConvLocIter loc_iter,
    TTable table,
    const int* indices_in,
    int* indice_pairs,
    int* indice_num_per_loc,
    int num_indices_in,
    int indices_pair_size,
    int RS
);