#pragma once

#include <iostream>

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
    int grid_x_size = 1440;
    int grid_y_size = 1440;
    int grid_z_size = 40;
    int feature_num = 5;
};

#ifdef __cplusplus
extern "C" {
#endif

void voxelizationLaunch(
    float* points, int points_size,
    VoxelParams& params,
    uint64_t* hash_table, int hash_table_size,
    float* voxels_temp, int* voxel_point_mask,
    __half* voxel_features, int* real_voxel_num,
    uint64_t* voxel_num_points, int* voxel_idxs
);

#ifdef __cplusplus
}
#endif