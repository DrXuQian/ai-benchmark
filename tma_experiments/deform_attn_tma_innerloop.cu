#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <stdexcept>
#include <map>

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

// Helper function to get cuTensorMapEncodeTiled function pointer
inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])

// TMA tile dimensions: 2x2x32 for bilinear interpolation
constexpr int TILE_H = 2;
constexpr int TILE_W = 2;
constexpr int TILE_C = 32;  // Load all 32 channels at once

template <typename scalar_t=__half, const int NUM_POINT= 8, const int NUM_LEVELS=4, const int CHANNELS = 32,
                                    const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
                                    const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3>
__global__ void ms_deformable_im2col_gpu_kernel_tma(
    const int n,
    const CUtensorMap* tma_descs_all,   // TMA descriptors [batch][level]
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_query,
    scalar_t *data_col) {

    // Shared memory for TMA loads - one tile per warp (16 warps in 512 threads)
    __shared__ alignas(128) scalar_t smem_tile[16][TILE_H][TILE_W][TILE_C];

    // Per-warp barriers (16 warps in 512 threads)
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier warp_bars[16];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        init(&warp_bars[warp_id], 32);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncwarp();

    CUDA_1D_KERNEL_LOOP(index, n) {
        int _temp = index << NUM_OUTPUT_SHIFT;
        const int c_col = _temp & (CHANNELS -1 );
        _temp = (_temp >> CHANNELS_SHIFT);
        const int sampling_index = _temp;
        const int b_col = (float)_temp/(float)num_query;
        const __half kZERO = __int2half_rz(0);
        const __half kONE = __int2half_rz(1);

        scalar_t *data_col_ptr = data_col + (index << NUM_OUTPUT_SHIFT);
        int data_weight_ptr = sampling_index << (LEVEL_SHIFT + POINT_SHIFT);
        int data_loc_w_ptr = data_weight_ptr << 1;

        scalar_t col[NUM_OUTPUT];
        #pragma unroll
        for (int idx = 0; idx < (NUM_OUTPUT >> 1); idx += 1) {
            reinterpret_cast<__half2*>(col)[idx] = __half2(0.0f, 0.0f);
        }

        scalar_t *data_half = const_cast<scalar_t *>(data_sampling_loc);
        scalar_t *data_attn_weight_half = const_cast<scalar_t *>(data_attn_weight);

        const half2 zp5 = __half2(0.5f, 0.5f);

        // Process each level
        for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const half2 spatial_hw = __half2(spatial_w, spatial_h);

            // Load sampling locations and attention weights
            half2 loc_hw_vec[NUM_POINT];
            half  weight_vec[NUM_POINT];

            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 4){
                LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
            }
            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8){
                LDST128BITS(weight_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
            }
            data_loc_w_ptr += (NUM_POINT << 1);
            data_weight_ptr += NUM_POINT;

            // Get TMA descriptor for this batch and level
            const int desc_idx = b_col * NUM_LEVELS + l_col;
            const CUtensorMap* tma_desc = &tma_descs_all[desc_idx];

            // Process each point - load TMA data on-demand
            #pragma unroll
            for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
                const half2 loc = loc_hw_vec[p_col];
                const scalar_t weight = weight_vec[p_col];
                half2 weighthalf2 = __half2(weight, weight);
                half2 hw_im = __hfma2(loc, spatial_hw, zp5);
                scalar_t h_im = __high2half(hw_im);
                scalar_t w_im = __low2half(hw_im);

                bool in_bounds = (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) &&
                                  h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1));

                int32_t h_coord = 0, w_coord = 0;
                if (in_bounds) {
                    int32_t const hLow = __half2int_rd(h_im);
                    int32_t const wLow = __half2int_rd(w_im);

                    // Clamp to valid range
                    h_coord = max(0, min(hLow, spatial_h - 2));
                    w_coord = max(0, min(wLow, spatial_w - 2));
                }

                // Issue TMA load for this specific 2x2x32 tile (only lane 0 of each warp)
                // Always issue load to keep warp synchronized, even if out of bounds
                if (lane_id == 0) {
                    // TMA coordinates - always load from channel 0
                    int32_t c_coord = 0;

                    // Issue TMA load using warp barrier - load into this warp's smem tile
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[warp_id][0][0][0]))),
                          "l"(reinterpret_cast<uint64_t>(tma_desc)),
                          "r"(c_coord), "r"(w_coord), "r"(h_coord),
                          "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id])))
                        : "memory"
                    );

                    // Set expected bytes for this TMA transfer
                    asm volatile(
                        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id]))),
                          "n"(TILE_H * TILE_W * TILE_C * sizeof(scalar_t))
                    );
                }

                // Warp waits for TMA to complete (all threads must participate)
                barrier::arrival_token token = warp_bars[warp_id].arrive();
                warp_bars[warp_id].wait(std::move(token));

                // Now compute bilinear interpolation using loaded TMA data (only if in bounds)
                if (in_bounds) {
                    int32_t const hLow = __half2int_rd(h_im);
                    int32_t const wLow = __half2int_rd(w_im);

                    const __half lh = __hsub(h_im, __int2half_rd(hLow));
                    const __half lw = __hsub(w_im, __int2half_rd(wLow));
                    const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);

                    // Compute bilinear weights
                    __half wdata[4];
                    wdata[0] = __hmul(hh, hw);
                    wdata[1] = __hmul(hh, lw);
                    wdata[2] = __hmul(lh, hw);
                    wdata[3] = __hmul(lh, lw);

                    // Bilinear interpolation using this warp's smem_tile
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j++) {
                        scalar_t val = 0;
                        val = __hfma(wdata[0], smem_tile[warp_id][0][0][c_col + j], val);
                        val = __hfma(wdata[1], smem_tile[warp_id][0][1][c_col + j], val);
                        val = __hfma(wdata[2], smem_tile[warp_id][1][0][c_col + j], val);
                        val = __hfma(wdata[3], smem_tile[warp_id][1][1][c_col + j], val);
                        col[j] = __hfma(weight, val, col[j]);
                    }
                }
            }
        }

        // Write output
        #pragma unroll
        for (int idx = 0; idx < NUM_OUTPUT; idx += 8){
            __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
            data_col_ptr += 8;
        }
    }
}

template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda_tma(cudaStream_t stream,
                               const CUtensorMap *tma_descs_all,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
    const int num_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
    const int num_actual_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
    const int num_threads = THREADS_IN_ONE_BLOCK;

    if (num_heads == 1 and num_point == 8 and num_levels == 4 and channels == 32){
        ms_deformable_im2col_gpu_kernel_tma<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, tma_descs_all, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query,  data_col);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in ms_deformable_im2col_cuda_tma: %s\n", cudaGetErrorString(err));
    }
}

// Helper function to read binary file
template <typename T>
std::vector<T> read_bin_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.seekg(0, std::ios::end);
    size_t size_bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(size_bytes / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), size_bytes);
    return data;
}

int main() {
    printf("=== Deformable Attention with TMA Loading (Inner Loop) ===\n\n");
    fflush(stdout);

    // Configuration
    const int batch_size = 1;  // Start with single batch for debugging
    const int num_query = 100;  // Small size for testing
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;
    const int spatial_size = 20454;

    // Spatial dimensions with padding
    struct LevelConfig {
        int H, W, H_padded, W_padded;
        int start_idx;
    } level_configs[4] = {
        {92, 160, 94, 162, 0},
        {46, 80, 48, 82, 15228},
        {23, 40, 25, 42, 19164},
        {12, 20, 14, 22, 20214}
    };

    // Prepare spatial shapes and level start indices
    std::vector<int64_t> spatial_shapes;
    std::vector<int64_t> level_start_indices;
    for (int l = 0; l < num_levels; l++) {
        spatial_shapes.push_back(level_configs[l].H);
        spatial_shapes.push_back(level_configs[l].W);
        level_start_indices.push_back(level_configs[l].start_idx);
    }

    // Allocate device memory for spatial info
    int64_t *d_spatial_shapes, *d_level_start_indices;
    cudaMalloc(&d_spatial_shapes, spatial_shapes.size() * sizeof(int64_t));
    cudaMalloc(&d_level_start_indices, level_start_indices.size() * sizeof(int64_t));
    cudaMemcpy(d_spatial_shapes, spatial_shapes.data(), spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_start_indices, level_start_indices.data(), level_start_indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    printf("Loading test data...\n");
    fflush(stdout);
    auto value_data = read_bin_file<__half>("working/value.bin");
    printf("Loaded value_data\n"); fflush(stdout);
    auto sampling_loc = read_bin_file<__half>("working/sampling_locations.bin");
    printf("Loaded sampling_loc\n"); fflush(stdout);
    auto attn_weight = read_bin_file<__half>("working/attention_weights.bin");
    printf("Loaded attn_weight\n"); fflush(stdout);

    printf("  Value data: %zu elements\n", value_data.size());
    printf("  Sampling locations: %zu elements\n", sampling_loc.size());
    printf("  Attention weights: %zu elements\n\n", attn_weight.size());
    fflush(stdout);

    // Create TMA descriptors for all batch×level combinations
    printf("Creating TMA descriptors for %d batches × %d levels...\n", batch_size, num_levels);
    fflush(stdout);

    // Get cuTensorMapEncodeTiled function pointer
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    if (!cuTensorMapEncodeTiled) {
        printf("Failed to get cuTensorMapEncodeTiled function pointer\n");
        return 1;
    }

    std::vector<CUtensorMap> h_tma_descs;
    std::vector<__half*> d_value_ptrs;

    // Allocate value data for all batches on device
    size_t value_size_per_batch = value_data.size() / 48;  // Original data is for 48 batches
    for (int b = 0; b < batch_size; b++) {
        __half* d_value_batch;
        cudaMalloc(&d_value_batch, value_size_per_batch * sizeof(__half));
        cudaMemcpy(d_value_batch, value_data.data(), value_size_per_batch * sizeof(__half), cudaMemcpyHostToDevice);
        d_value_ptrs.push_back(d_value_batch);

        // Create TMA descriptors for each level
        for (int l = 0; l < num_levels; l++) {
            CUtensorMap tma_desc;

            // TMA configuration for padded data
            uint64_t globalDim[3] = {
                channels,
                (uint64_t)level_configs[l].W_padded,
                (uint64_t)level_configs[l].H_padded
            };
            uint64_t globalStride[2] = {
                channels * sizeof(__half),
                level_configs[l].W_padded * channels * sizeof(__half)
            };
            uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
            uint32_t elementStride[3] = {1, 1, 1};

            // Calculate offset for this level
            size_t level_offset = level_configs[l].start_idx * channels * sizeof(__half);

            CUresult res = cuTensorMapEncodeTiled(
                &tma_desc,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                3,
                (void*)(d_value_batch + level_offset / sizeof(__half)),
                globalDim,
                globalStride,
                boxDim,
                elementStride,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );

            if (res != CUDA_SUCCESS) {
                printf("Failed to create TMA descriptor for batch %d, level %d: error %d\n", b, l, res);
                return 1;
            }
            h_tma_descs.push_back(tma_desc);
        }
        printf("  Created descriptors for batch %d\n", b);
        fflush(stdout);
    }

    // Copy TMA descriptors to device
    printf("  Copying %zu TMA descriptors to device...\n", h_tma_descs.size());
    fflush(stdout);
    CUtensorMap* d_tma_descs;
    cudaMalloc(&d_tma_descs, h_tma_descs.size() * sizeof(CUtensorMap));
    printf("  cudaMalloc succeeded\n");
    fflush(stdout);
    cudaMemcpy(d_tma_descs, h_tma_descs.data(), h_tma_descs.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    printf("  cudaMemcpy succeeded\n");
    fflush(stdout);
    printf("  Created %zu TMA descriptors\n\n", h_tma_descs.size());
    fflush(stdout);

    // Allocate device memory for input/output
    printf("Allocating device memory for input/output...\n");
    fflush(stdout);
    __half *d_sampling_loc, *d_attn_weight, *d_output;
    const size_t output_size = batch_size * num_query * num_heads * channels;

    // Use only portion of data needed for our batch/query size
    size_t sampling_loc_size = batch_size * num_query * num_heads * num_levels * num_points * 2;
    size_t attn_weight_size = batch_size * num_query * num_heads * num_levels * num_points;

    printf("  Allocating sampling_loc: %zu elements\n", sampling_loc_size);
    fflush(stdout);
    cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(__half));
    printf("  Allocating attn_weight: %zu elements\n", attn_weight_size);
    fflush(stdout);
    cudaMalloc(&d_attn_weight, attn_weight_size * sizeof(__half));
    printf("  Allocating output: %zu elements\n", output_size);
    fflush(stdout);
    cudaMalloc(&d_output, output_size * sizeof(__half));

    printf("  Copying data to device...\n");
    fflush(stdout);
    cudaMemcpy(d_sampling_loc, sampling_loc.data(), sampling_loc_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_weight, attn_weight.data(), attn_weight_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size * sizeof(__half));
    printf("  Device memory allocated and initialized\n\n");
    fflush(stdout);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after device memory allocation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaDeviceSynchronize succeeded\n");
    fflush(stdout);

    // Warmup
    printf("Running warmup...\n");
    fflush(stdout);
    for (int i = 0; i < 3; i++) {
        printf("  Warmup iteration %d...\n", i);
        fflush(stdout);
        ms_deformable_im2col_cuda_tma<__half, 512, 8, 3>(
            nullptr, d_tma_descs, d_spatial_shapes, d_level_start_indices,
            d_sampling_loc, d_attn_weight, batch_size, spatial_size,
            num_heads, channels, num_levels, num_query, num_points, d_output
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("  Kernel launched\n");
        fflush(stdout);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("  Kernel completed\n");
        fflush(stdout);
    }

    // Benchmark
    printf("Running benchmark (10 iterations)...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        ms_deformable_im2col_cuda_tma<__half, 512, 8, 3>(
            nullptr, d_tma_descs, d_spatial_shapes, d_level_start_indices,
            d_sampling_loc, d_attn_weight, batch_size, spatial_size,
            num_heads, channels, num_levels, num_query, num_points, d_output
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
        printf("  Iteration %2d: %.4f ms\n", i + 1, iter_time);
    }

    float avg_time = total_time / 10;
    printf("\n=== Performance Results ===\n");
    printf("Average time: %.4f ms\n", avg_time);
    printf("Throughput: %.2f queries/ms\n", (batch_size * num_query) / avg_time);

    // Cleanup
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaFree(d_tma_descs);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_indices);
    for (auto ptr : d_value_ptrs) {
        cudaFree(ptr);
    }

    printf("\n✅ Deformable Attention with TMA (Inner Loop) completed!\n");
    return 0;
}