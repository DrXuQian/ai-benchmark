#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct BEVPoolParams {
    std::vector<int> camera_shape;
    unsigned int bev_width;
    unsigned int bev_height;
    unsigned int num_cameras;
    unsigned int channels;
    unsigned int depth_bins;
    unsigned int feature_height;
    unsigned int feature_width;
};

struct BEVPool {
    BEVPoolParams params;
    half* output_feature;
    int output_dims[4];
    unsigned int volumn_output;
};

struct Int3 {
    int x, y, z;
};

#ifdef __cplusplus
extern "C" {
#endif

BEVPool* bevpool_create(const BEVPoolParams* params);
void bevpool_destroy(BEVPool* pool);
half* bevpool_forward(
    BEVPool* pool,
    const half* camera_feature, 
    const half* depth_weights,
    const unsigned int* indices, 
    const int3* intervals, 
    unsigned int num_intervals,
    cudaStream_t stream = 0
);
void bevpool_get_shape(BEVPool* pool, int* shape);

#ifdef __cplusplus
}
#endif