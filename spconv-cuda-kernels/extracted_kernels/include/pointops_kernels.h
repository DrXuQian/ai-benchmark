// Header file for pointops kernels
// Copyright 2021 Yan Yan
// Licensed under the Apache License, Version 2.0

#pragma once

#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
#include <tensorview/hash/ops.h>

// ============================================================================
// Point2VoxelKernel kernel declarations
// ============================================================================

template<typename TTable>
__global__ void build_hash_table(
    TTable table,
    const float* points,
    int64_t* points_indice_data,
    int point_stride,
    tv::array<float, 3> vsize,
    tv::array<float, 6> coors_range,
    tv::array<int, 3> grid_bound,
    tv::array<int64_t, 3> grid_stride,
    int num_points
);

template<typename TTable>
__global__ void assign_table(
    TTable table,
    int* indices,
    int* count,
    typename TTable::Layout layout,
    int max_voxels
);

template<typename TTable>
__global__ void generate_voxel(
    TTable table,
    const float* points,
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
);

__global__ void voxel_empty_fill_mean(
    float* voxels,
    int* num_per_voxel,
    int num_voxels,
    int num_points_per_voxel,
    int num_voxel_features
);

__global__ void limit_num_per_voxel_value(
    int* num_per_voxel,
    int num_voxels,
    int num_points_per_voxel
);