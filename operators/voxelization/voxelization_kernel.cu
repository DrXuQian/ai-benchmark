#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "voxelization_kernel.h"

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define THREADS_FOR_VOXEL 256

__device__ uint64_t hash_func(uint64_t key) {
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return key;
}

__global__ void buildHashKernel(
    float* points, int points_size,
    float min_x_range, float max_x_range, 
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int grid_z_size, int feature_num,
    uint64_t* hash_table, int hash_table_size,
    unsigned int* real_voxel_num
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= points_size) return;

    float* point = points + point_idx * feature_num;
    float px = point[0];
    float py = point[1];
    float pz = point[2];

    if (px < min_x_range || px >= max_x_range || 
        py < min_y_range || py >= max_y_range || 
        pz < min_z_range || pz >= max_z_range) {
        return;
    }

    int voxel_x = min(max(0, (int)floor((px - min_x_range) / voxel_x_size)), grid_x_size - 1);
    int voxel_y = min(max(0, (int)floor((py - min_y_range) / voxel_y_size)), grid_y_size - 1);
    int voxel_z = min(max(0, (int)floor((pz - min_z_range) / voxel_z_size)), grid_z_size - 1);

    uint64_t voxel_offset = voxel_z * grid_y_size * grid_x_size + voxel_y * grid_x_size + voxel_x;
    uint64_t hash_key = hash_func(voxel_offset);

    for (uint64_t i = 0; i < hash_table_size; i++) {
        uint64_t hash_index = (hash_key + i) % hash_table_size;
        uint64_t old_voxel_id = atomicCAS((unsigned long long*)(hash_table + hash_index),
                                          (uint64_t)INT64_MAX, voxel_offset);
        
        if (old_voxel_id == (uint64_t)INT64_MAX) {
            atomicAdd(real_voxel_num, 1);
            break;
        } else if (old_voxel_id == voxel_offset) {
            break;
        }
    }
}

__global__ void voxelizationKernel(
    float* points, int points_size,
    float min_x_range, float max_x_range,
    float min_y_range, float max_y_range,
    float min_z_range, float max_z_range,
    float voxel_x_size, float voxel_y_size, float voxel_z_size,
    int grid_y_size, int grid_x_size, int grid_z_size, int feature_num,
    int max_points_per_voxel, int max_voxel_num,
    uint64_t* hash_table, int hash_table_size,
    float* voxels_temp, int* voxel_point_mask,
    int* real_voxel_num, uint64_t* voxel_num_points,
    int* voxel_idxs
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= points_size) return;

    float* point = points + point_idx * feature_num;
    float px = point[0];
    float py = point[1];
    float pz = point[2];

    if (px < min_x_range || px >= max_x_range || 
        py < min_y_range || py >= max_y_range || 
        pz < min_z_range || pz >= max_z_range) {
        return;
    }

    int voxel_x = min(max(0, (int)floor((px - min_x_range) / voxel_x_size)), grid_x_size - 1);
    int voxel_y = min(max(0, (int)floor((py - min_y_range) / voxel_y_size)), grid_y_size - 1);
    int voxel_z = min(max(0, (int)floor((pz - min_z_range) / voxel_z_size)), grid_z_size - 1);

    uint64_t voxel_offset = voxel_z * grid_y_size * grid_x_size + voxel_y * grid_x_size + voxel_x;
    uint64_t hash_key = hash_func(voxel_offset);

    for (uint64_t i = 0; i < hash_table_size; i++) {
        uint64_t hash_index = (hash_key + i) % hash_table_size;
        uint64_t table_voxel_id = hash_table[hash_index];
        
        if (table_voxel_id == voxel_offset) {
            uint64_t current_num = atomicAdd((unsigned long long*)voxel_num_points + hash_index, 1);
            if (current_num < max_points_per_voxel) {
                uint64_t point_index = hash_index * max_points_per_voxel * feature_num +
                                     current_num * feature_num;
                for (int f = 0; f < feature_num; f++) {
                    voxels_temp[point_index + f] = point[f];
                }
                voxel_point_mask[hash_index * max_points_per_voxel + current_num] = 1;
            }
            break;
        }
    }
}

__global__ void featureExtractionKernel(
    float* voxels_temp, int max_voxel_num, int max_points_per_voxel,
    int feature_num, uint64_t* voxel_num_points,
    __half* voxel_features, int* voxel_point_mask
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= max_voxel_num) return;

    uint64_t current_voxel_num_points = voxel_num_points[voxel_idx];
    if (current_voxel_num_points == 0) return;

    float features[5] = {0.0f};
    int valid_points = 0;

    for (int point_idx = 0; point_idx < max_points_per_voxel; point_idx++) {
        if (voxel_point_mask[voxel_idx * max_points_per_voxel + point_idx] == 1) {
            int base_idx = voxel_idx * max_points_per_voxel * feature_num + point_idx * feature_num;
            for (int f = 0; f < feature_num; f++) {
                features[f] += voxels_temp[base_idx + f];
            }
            valid_points++;
        }
    }

    if (valid_points > 0) {
        int output_idx = voxel_idx * feature_num;
        for (int f = 0; f < feature_num; f++) {
            voxel_features[output_idx + f] = __float2half(features[f] / valid_points);
        }
    }
}

extern "C" {

void voxelizationLaunch(
    float* points, int points_size,
    VoxelParams& params,
    uint64_t* hash_table, int hash_table_size,
    float* voxels_temp, int* voxel_point_mask,
    __half* voxel_features, int* real_voxel_num,
    uint64_t* voxel_num_points, int* voxel_idxs
) {
    int max_voxel_num = params.grid_x_size * params.grid_y_size * params.grid_z_size;

    unsigned int* d_real_voxel_num;
    checkCudaErrors(cudaMalloc(&d_real_voxel_num, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_real_voxel_num, 0, sizeof(unsigned int)));

    dim3 threads(THREADS_FOR_VOXEL);
    dim3 blocks(DIVUP(points_size, threads.x));

    buildHashKernel<<<blocks, threads>>>(
        points, points_size,
        params.min_x_range, params.max_x_range,
        params.min_y_range, params.max_y_range,
        params.min_z_range, params.max_z_range,
        params.voxel_x_size, params.voxel_y_size, params.voxel_z_size,
        params.grid_y_size, params.grid_x_size, params.grid_z_size, params.feature_num,
        hash_table, hash_table_size, d_real_voxel_num
    );

    checkCudaErrors(cudaDeviceSynchronize());

    voxelizationKernel<<<blocks, threads>>>(
        points, points_size,
        params.min_x_range, params.max_x_range,
        params.min_y_range, params.max_y_range,
        params.min_z_range, params.max_z_range,
        params.voxel_x_size, params.voxel_y_size, params.voxel_z_size,
        params.grid_y_size, params.grid_x_size, params.grid_z_size, params.feature_num,
        params.max_points_per_voxel, max_voxel_num,
        hash_table, hash_table_size, voxels_temp,
        voxel_point_mask, real_voxel_num, voxel_num_points, voxel_idxs
    );

    checkCudaErrors(cudaDeviceSynchronize());

    dim3 feature_threads(THREADS_FOR_VOXEL);
    dim3 feature_blocks(DIVUP(max_voxel_num, feature_threads.x));

    featureExtractionKernel<<<feature_blocks, feature_threads>>>(
        voxels_temp, max_voxel_num, params.max_points_per_voxel,
        params.feature_num, voxel_num_points, voxel_features, voxel_point_mask
    );

    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpy(real_voxel_num, d_real_voxel_num, 
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_real_voxel_num));
}

}