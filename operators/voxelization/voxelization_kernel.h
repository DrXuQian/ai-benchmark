/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef VOXELIZATION_KERNEL_H
#define VOXELIZATION_KERNEL_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Constants
const unsigned int MAX_POINTS_NUM = 300000;

// Voxelization parameters
struct VoxelParams {
    float min_x_range = -54.0f;
    float max_x_range = 54.0f;
    float min_y_range = -54.0f;
    float max_y_range = 54.0f;
    float min_z_range = -5.0f;
    float max_z_range = 3.0f;

    float voxel_x_size = 0.075f;
    float voxel_y_size = 0.075f;
    float voxel_z_size = 0.2f;

    int max_points_per_voxel = 10;
    int max_voxels = 160000;
    int feature_num = 5;

    int grid_x_size = 1440;  // (max_x_range - min_x_range) / voxel_x_size
    int grid_y_size = 1440;  // (max_y_range - min_y_range) / voxel_y_size
    int grid_z_size = 40;    // (max_z_range - min_z_range) / voxel_z_size

    int getGridXSize() {
        return static_cast<int>(std::round((max_x_range - min_x_range) / voxel_x_size));
    }

    int getGridYSize() {
        return static_cast<int>(std::round((max_y_range - min_y_range) / voxel_y_size));
    }

    int getGridZSize() {
        return static_cast<int>(std::round((max_z_range - min_z_range) / voxel_z_size));
    }
};

// CUDA kernel launch functions
cudaError_t voxelizationLaunch(
    const float *points, size_t points_size,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
    int max_points_per_voxel,
    unsigned int *hash_table, unsigned int *num_points_per_voxel,
    float *voxel_features, unsigned int *voxel_indices,
    unsigned int *real_voxel_num, cudaStream_t stream = 0);

cudaError_t featureExtractionLaunch(
    float *voxels_temp, unsigned int *num_points_per_voxel,
    const unsigned int real_voxel_num, int max_points_per_voxel, int feature_num,
    half *voxel_features, cudaStream_t stream = 0);

// Binary data loading function
int loadData(const char *file, void **data, unsigned int *length);

#endif // VOXELIZATION_KERNEL_H