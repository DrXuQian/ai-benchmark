// Extracted CUDA kernels from spconv/csrc/sparse/pointops.py
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/hash/ops.h>

// ============================================================================
// Point2VoxelKernel kernels
// ============================================================================

template<typename TTable>
__global__ void build_hash_table(
    TTable table,
    const float* points,  // Assuming float dtype
    int64_t* points_indice_data,
    int point_stride,
    tv::array<float, 3> vsize,  // Assuming 3D
    tv::array<float, 6> coors_range,
    tv::array<int, 3> grid_bound,
    tv::array<int64_t, 3> grid_stride,
    int num_points
) {
    for (int i : tv::KernelLoopX<int>(num_points)) {
        bool failed = false;
        int c;
        int64_t prod = 0;
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            c = floor((points[i * point_stride + (3 - 1) - j] - coors_range[j]) / vsize[j]);
            if ((c < 0 || c >= grid_bound[j])) {
                failed = true;
            }
            prod += grid_stride[j] * int64_t(c);
        }
        if (!failed) {
            points_indice_data[i] = prod;
            table.insert(prod, i);
        } else {
            points_indice_data[i] = -1;
        }
    }
}

template<typename TTable>
__global__ void assign_table(
    TTable table,
    int* indices,
    int* count,
    typename TTable::Layout layout,  // Using Layout from template
    int max_voxels
) {
    auto data = table.data();
    for (int i : tv::KernelLoopX<int>(table.size())) {
        auto& item = data[i];
        if (!item.empty()) {
            item.second = tv::cuda::atomicAggInc(count);
            if (item.second < max_voxels) {
                layout.inverse(item.first, indices + item.second * 3);  // Assuming 3D
            }
        }
    }
}

template<typename TTable>
__global__ void generate_voxel(
    TTable table,
    const float* points,  // Assuming float dtype
    const int64_t* points_indice_data,
    float* voxels,
    int* num_per_voxel,
    int64_t* points_voxel_id,
    int point_stride,
    int max_points_per_voxel,
    int max_voxels,
    tv::array<float, 3> vsize,
    tv::array<float, 6> coors_range,
    tv::array<int, 3> grid_bound,
    tv::array<int64_t, 3> grid_stride,
    int num_points
) {
    int voxel_stride0 = point_stride * max_points_per_voxel;
    for (int i : tv::KernelLoopX<int>(num_points)) {
        int64_t prod = points_indice_data[i];
        int voxel_id = -1;
        if (prod != -1) {
            auto voxel_index_pair = table.lookup(prod);
            if (!voxel_index_pair.empty() && voxel_index_pair.second < max_voxels) {
                voxel_id = voxel_index_pair.second;
                int old = atomicAdd(num_per_voxel + voxel_index_pair.second, 1);
                if (old < max_points_per_voxel) {
                    for (int j = 0; j < point_stride; ++j) {
                        voxels[voxel_index_pair.second * voxel_stride0 + old * point_stride + j] =
                            points[i * point_stride + j];
                    }
                }
            }
        }
        points_voxel_id[i] = voxel_id;
    }
}

__global__ void voxel_empty_fill_mean(
    float* voxels,  // Assuming float dtype
    int* num_per_voxel,
    int num_voxels,
    int num_points_per_voxel,
    int num_voxel_features
) {
    int voxel_stride = num_points_per_voxel * num_voxel_features;
    for (int i : tv::KernelLoopX<int>(num_voxels)) {
        int count = min(num_points_per_voxel, num_per_voxel[i]);
        num_per_voxel[i] = count;
        for (int j = 0; j < num_voxel_features; ++j) {
            auto voxel_ptr = voxels + i * voxel_stride + j;
            float sum_val = 0;  // Assuming float dtype
            for (int k = 0; k < count; ++k) {
                sum_val += voxel_ptr[0];
                voxel_ptr += num_voxel_features;
            }
            sum_val = count == 0 ? 0 : sum_val / count;
            for (int k = count; k < num_points_per_voxel; ++k) {
                voxel_ptr[0] = sum_val;
                voxel_ptr += num_voxel_features;
            }
        }
    }
}

__global__ void limit_num_per_voxel_value(
    int* num_per_voxel,
    int num_voxels,
    int num_points_per_voxel
) {
    for (int i : tv::KernelLoopX<int>(num_voxels)) {
        int count = min(num_points_per_voxel, num_per_voxel[i]);
        num_per_voxel[i] = count;
    }
}