#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include "bevpool_kernel.h"

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define tile_size 10

typedef struct __align__(4){
    half val[tile_size];
} combined_half;

static __global__ void bevpool_half_pack10_kernel(
    const half* camera_feature, 
    const half* depth_weights, 
    unsigned int nchannel,
    const int3* intervals, 
    unsigned int n_intervals, 
    const unsigned int* indices,
    unsigned int out_h, 
    unsigned int out_w, 
    unsigned int ndepth, 
    unsigned int farea,
    half* output_bevfeat
) {
    int interval_index = blockIdx.y * blockDim.y + threadIdx.y;
    int feature_block = threadIdx.x * tile_size;

    if (interval_index >= n_intervals) return;
    
    int3 interval = intervals[interval_index];
    float accumulate[tile_size] = {0.0f};

    for (int i = interval.x; i < interval.y; i++) {
        int indice = indices[i];
        int camera_index = indice / (ndepth * farea);
        int fm_inner_index = indice % farea;
        float depth_weight = __half2float(depth_weights[indice]);
        unsigned int camera_feature_offset = (camera_index * farea + fm_inner_index) * nchannel + feature_block;
        
        // Check if we can safely access a full tile_size block
        if (feature_block + tile_size <= nchannel) {
            combined_half feature = *(combined_half*)(camera_feature + camera_feature_offset);
            #pragma unroll
            for (int j = 0; j < tile_size; j++) {
                accumulate[j] = fma(__half2float(feature.val[j]), depth_weight, accumulate[j]);
            }
        } else {
            // Handle remaining channels individually
            for (int j = 0; j < tile_size && (feature_block + j) < nchannel; j++) {
                half feature_val = camera_feature[camera_feature_offset + j];
                accumulate[j] = fma(__half2float(feature_val), depth_weight, accumulate[j]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < tile_size && (feature_block + j) < nchannel; j++) {
        unsigned int output_offset = interval.z + (feature_block + j) * out_h * out_w;
        output_bevfeat[output_offset] = __float2half(accumulate[j]);
    }
}

extern "C" {

BEVPool* bevpool_create(const BEVPoolParams* params) {
    BEVPool* pool = new BEVPool();
    pool->params = *params;
    
    unsigned int C = params->channels;
    pool->volumn_output = C * params->bev_width * params->bev_height;
    pool->output_dims[0] = 1;
    pool->output_dims[1] = (int)C;
    pool->output_dims[2] = (int)params->bev_height;
    pool->output_dims[3] = (int)params->bev_width;
    
    checkCudaErrors(cudaMalloc(&pool->output_feature, pool->volumn_output * sizeof(half)));
    
    return pool;
}

void bevpool_destroy(BEVPool* pool) {
    if (pool) {
        if (pool->output_feature) {
            checkCudaErrors(cudaFree(pool->output_feature));
        }
        delete pool;
    }
}

half* bevpool_forward(
    BEVPool* pool,
    const half* camera_feature, 
    const half* depth_weights,
    const unsigned int* indices, 
    const int3* intervals, 
    unsigned int num_intervals,
    cudaStream_t stream
) {
    unsigned int C = pool->params.channels;
    unsigned int D = pool->params.depth_bins;
    unsigned int H = pool->params.feature_height;
    unsigned int W = pool->params.feature_width;

    int thread_x = (C + tile_size - 1) / tile_size;  // Round up division
    int thread_y = (1024 / thread_x) < 32 ? (1024 / thread_x) : 32;  // Limit thread_y to avoid too large block size
    dim3 threads(thread_x, thread_y);
    dim3 blocks(1, (num_intervals + thread_y - 1) / thread_y);
    
    checkCudaErrors(cudaMemsetAsync(pool->output_feature, 0x00, pool->volumn_output * sizeof(half), stream));
    
    bevpool_half_pack10_kernel<<<blocks, threads, 0, stream>>>(
        camera_feature, depth_weights, C, intervals, num_intervals, indices, 
        pool->params.bev_height, pool->params.bev_width, D, W * H, pool->output_feature);
    
    checkCudaErrors(cudaDeviceSynchronize());
    return pool->output_feature;
}

void bevpool_get_shape(BEVPool* pool, int* shape) {
    for (int i = 0; i < 4; i++) {
        shape[i] = pool->output_dims[i];
    }
}

}