// Extracted CUDA kernels from spconv/csrc/sparse/indices.py
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/hash/ops.h>
#include <limits>

// ============================================================================
// CudaCommonKernel kernels
// ============================================================================

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

// ============================================================================
// SparseConvIndicesKernel kernels
// ============================================================================

template<typename TIndiceUniq, typename TConvLocIter>
__global__ void calc_conv_indices_stage1(
    TConvLocIter loc_iter,
    const int* indices_in,
    int* indice_pairs,  // Using int instead of template for simplicity
    TIndiceUniq* indice_pairs_for_uniq,
    int* indice_num_per_loc,
    int num_indices_in,
    int indices_pair_size,
    int RS,
    bool transposed
) {
    int filter_offset = blockIdx.y;
    loc_iter.set_filter_offset(filter_offset);
    int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;

    for (int i : tv::KernelLoopX<int>(num_indices_in)) {
        tv::array<int, 4> npq_offset;  // Assuming ndim+1=4 for 3D
        bool valid;
        if (transposed) {
            valid = loc_iter.query_nhw_out(indices_in + i * 4, npq_offset);
        } else {
            valid = loc_iter.query_npq(indices_in + i * 4, npq_offset);
        }
        if (valid) {
            int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
            int64_t offset = loc_iter.layout_npq(npq_offset);
            if (old_num < indices_pair_size) {
                indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = offset;
            }
        }
    }
}

template<typename TTable, typename TLayoutNPQ>
__global__ void build_conv_hash_table(
    TTable table,
    int* indices_out,
    const typename TTable::key_type* indice_pairs_for_uniq,
    TLayoutNPQ layout_npq,
    int num_indices
) {
    for (int output_index : tv::KernelLoopX<int>(num_indices)) {
        auto output_coord_offset = indice_pairs_for_uniq[output_index];
        layout_npq.inverse(output_coord_offset, indices_out + 4 * output_index);  // Assuming ndim+1=4
        table.insert(output_coord_offset, output_index);
    }
}

template<typename TTable, typename TLayoutNPQ>
__global__ void arange_hash_table_and_assign_out(
    TTable table,
    int* indices_out,
    int* count,
    int limit,
    TLayoutNPQ layout_npq
) {
    auto key_ptr = table.key_ptr();
    auto value_ptr = table.value_ptr();

    for (auto i : tv::KernelLoopX<int>(table.size())) {
        auto output_coord_offset = key_ptr[i];
        if (output_coord_offset != TTable::empty_key) {
            auto output_index = tv::cuda::atomicAggInc(count);
            if (output_index < limit) {
                value_ptr[i] = output_index;
                layout_npq.inverse(output_coord_offset, indices_out + 4 * output_index);
            } else {
                value_ptr[i] = -1;
            }
        }
    }
}

template<typename TTable>
__global__ void arange_hash_table(
    TTable table,
    typename TTable::key_type* out_indices_offset,
    int* count,
    int limit
) {
    auto key_ptr = table.key_ptr();
    auto value_ptr = table.value_ptr();

    for (auto i : tv::KernelLoopX<int>(table.size())) {
        auto output_coord_offset = key_ptr[i];
        if (output_coord_offset != TTable::empty_key) {
            auto output_index = tv::cuda::atomicAggInc(count);
            value_ptr[i] = output_index < limit ? output_index : -1;
            out_indices_offset[output_index] = output_coord_offset;
        }
    }
}

template<typename T, typename TLayoutNPQ>
__global__ void assign_out_indices(
    int* indices_out,
    const T* out_indices_offset,
    TLayoutNPQ layout_npq,
    int size
) {
    for (auto i : tv::KernelLoopX<int>(size)) {
        layout_npq.inverse(out_indices_offset[i], indices_out + 4 * i);
    }
}

template<typename TTable>
__global__ void calc_conv_indices_stage2(
    TTable table,
    const typename TTable::key_type* indice_pairs_uniq_before_sort,
    int* indice_pairs_out_part,
    int num_indices_in,
    int indices_pair_size
) {
    int filter_offset = blockIdx.y;
    auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
    auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * indices_pair_size;

    for (int i : tv::KernelLoopX<int>(num_indices_in)) {
        int output_coord_offset = indice_pairs_uniq_before_sort_filter[i];
        if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()) {
            auto table_offset = table.lookup_offset(output_coord_offset);
            if (table_offset != -1) {
                indice_pairs_out_part_filter[i] = table.value_ptr()[table_offset];
            }
        }
    }
}

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
) {
    int filter_offset = blockIdx.y;
    auto indice_pairs_in_part_filter = indice_pairs_in_part + filter_offset * indices_pair_size;
    auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
    auto indice_pairs_in_part_temp_filter = indice_pairs_in_part_temp + filter_offset * indices_pair_size;
    auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * indices_pair_size;

    for (int i : tv::KernelLoopX<int>(num_indices_in)) {
        int output_coord_offset = indice_pairs_uniq_before_sort_filter[i];
        if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()) {
            auto table_offset = table.lookup_offset(output_coord_offset);
            if (table_offset != -1) {
                int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                indice_pairs_in_part_filter[old_num] = indice_pairs_in_part_temp_filter[i];
                indice_pairs_out_part_filter[old_num] = table.value_ptr()[table_offset];
            }
        }
    }
}

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
) {
    int filter_offset = blockIdx.y;
    loc_iter.set_filter_offset(filter_offset);
    int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;

    for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
        tv::array<int, 4> npq_offset;
        bool valid;
        if (transposed) {
            valid = loc_iter.query_nhw_out(indices_in + input_index * 4, npq_offset);
        } else {
            valid = loc_iter.query_npq(indices_in + input_index * 4, npq_offset);
        }
        if (valid) {
            TIndiceUniq output_coord_offset = loc_iter.layout_npq(npq_offset);
            indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
        }
    }
}

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
) {
    int filter_offset = blockIdx.y;
    loc_iter.set_filter_offset(filter_offset);
    int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;

    for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
        tv::array<int, 4> npq_offset;
        bool valid;
        if (transposed) {
            valid = loc_iter.query_nhw_out(indices_in + input_index * 4, npq_offset);
        } else {
            valid = loc_iter.query_npq(indices_in + input_index * 4, npq_offset);
        }
        if (valid) {
            TIndiceUniq output_coord_offset = loc_iter.layout_npq(npq_offset);
            table.insert_key_only(output_coord_offset);
            indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
        }
    }
}

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
) {
    int filter_offset = blockIdx.y;
    int filter_pointer_offset = filter_offset / 32;
    uint32_t filter_mask_fwd = (1u << (filter_offset % 32));

    auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
    auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
    auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;

    for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
        auto output_coord_offset = indice_pairs_uniq_before_sort_filter[input_index];
        if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()) {
            auto table_offset = table.lookup_offset(output_coord_offset);
            if (table_offset != -1) {
                auto output_index = table.value_ptr()[table_offset];
                bool valid = CheckValueValid ? output_index >= 0 : true;
                if (valid) {
                    atomicOr(mask_fwd + output_index * mask_int_count + filter_pointer_offset, filter_mask_fwd);
                    indice_pairs_fwd_filter[output_index] = input_index;
                    if (indice_pairs_bwd != nullptr) {
                        indice_pairs_bwd_filter[input_index] = output_index;
                    }
                }
            }
        }
    }
}

__global__ void calc_conv_indices_stage2_mask_output(
    int* indice_pairs_bwd,
    uint32_t* mask_bwd,
    int num_indices_in,
    int kv,
    int mask_int_count
) {
    for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
        for (int mask_offset = 0; mask_offset < mask_int_count; ++mask_offset) {
            uint32_t mask = 0;
            for (int filter_offset = mask_offset * 32; filter_offset < mask_offset * 32 + 32 && filter_offset < kv; ++filter_offset) {
                auto val = indice_pairs_bwd[filter_offset * num_indices_in + input_index];
                mask |= (val != -1) << (filter_offset % 32);
            }
            mask_bwd[input_index * mask_int_count + mask_offset] = mask;
        }
    }
}

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
) {
    int filter_offset = blockIdx.y;
    int filter_pointer_offset = filter_offset / 32;
    uint32_t filter_mask_fwd = (1u << (filter_offset % 32));

    auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
    auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;

    for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
        auto output_coord_offset = indice_pairs_uniq_before_sort_filter[input_index];
        if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()) {
            auto table_offset = table.lookup_offset(output_coord_offset);
            if (table_offset != -1) {
                auto output_index = table.value_ptr()[table_offset];
                bool valid = CheckValueValid ? output_index >= 0 : true;
                if (valid) {
                    atomicOr(mask_fwd + output_index * mask_int_count + filter_pointer_offset, filter_mask_fwd);
                    indice_pairs_fwd_filter[output_index] = input_index;
                }
            }
        }
    }
}

template<typename TTable, typename TLayoutNPQ>
__global__ void build_subm_conv_hash_table(
    TTable table,
    const int* indices_in,
    TLayoutNPQ layout_npq,
    int num_indices
) {
    for (int i : tv::KernelLoopX<int>(num_indices)) {
        table.insert(layout_npq(indices_in + i * 4), i);
    }
}

template<typename T>
__global__ void clean_indices_uniq(T* indice_pairs_for_uniq, size_t size) {
    for (size_t i : tv::KernelLoopX<size_t>(size)) {
        indice_pairs_for_uniq[i] = std::numeric_limits<T>::max();
    }
}

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
) {
    int filter_offset = blockIdx.y;
    loc_iter.set_filter_offset(filter_offset);
    int indices_pair_size_mul_RS = indices_pair_size * RS;
    int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
    int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;

    if (filter_offset == (RS / 2)) {
        for (int i : tv::KernelLoopX<int>(num_indices_in)) {
            indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
            indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
        }
    } else {
        for (int i : tv::KernelLoopX<int>(num_indices_in)) {
            tv::array<int, 4> npq_offset;
            if (loc_iter.query_npq_no_stride(indices_in + i * 4, npq_offset)) {
                auto offset = loc_iter.layout_npq(npq_offset);
                auto table_offset = table.lookup_offset(offset);
                if (table_offset != -1) {
                    auto v = table.value_ptr()[table_offset];
                    int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                    indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                    indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = v;
                    indice_pairs[filter_offset_mul_indices_pair_size_1 + old_num] = v;
                    indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
                }
            }
        }
    }
}