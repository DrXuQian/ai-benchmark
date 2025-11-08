#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <stdexcept>
#include <map>

// TMA版本的deformable attention - 匹配原始kernel的线程分配和计算逻辑
//
// CRITICAL: TMA Dimension Mapping for [H][W][C] Memory Layout
// X = C (channels, innermost), Y = W (width), Z = H (height, outermost)

using barrier = cuda::barrier<cuda::thread_scope_block>;
using TmaDescriptor = CUtensorMap;

inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    if (err != cudaSuccess) {
        printf("Failed to get cuTensorMapEncodeTiled: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr);
}

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// TMA version kernel - matches original thread allocation
template <typename scalar_t=__half, const int NUM_POINT=8, const int NUM_LEVELS=4, const int CHANNELS=32,
          const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
          const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3>
__global__ void ms_deformable_im2col_tma_kernel(
    const int n,
    const scalar_t *data_value,
    const TmaDescriptor *tma_descriptors,  // [batch_size * NUM_LEVELS]
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_query,
    scalar_t *data_col) {

    // Shared memory for TMA loads (reused across levels and points)
    // Only load 2x2x32 at a time, reuse for all 4 corners
    __shared__ alignas(128) scalar_t smem_tile[2][2][CHANNELS];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;

    // Initialize barrier once
    if (tid == 0) {
        init(&bar, blockDim.x);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    CUDA_1D_KERNEL_LOOP(index, n) {
        int _temp = index << NUM_OUTPUT_SHIFT;
        const int c_col = _temp & (CHANNELS - 1);
        _temp = (_temp >> CHANNELS_SHIFT);
        const int sampling_index = _temp;
        const int b_col = (float)_temp / (float)num_query;
        const __half kZERO = __int2half_rz(0);
        const __half kONE = __int2half_rz(1);

        scalar_t *data_col_ptr = data_col + (index << NUM_OUTPUT_SHIFT);
        int data_weight_ptr = sampling_index << (LEVEL_SHIFT + POINT_SHIFT);
        int data_loc_w_ptr = data_weight_ptr << 1;

        scalar_t col[NUM_OUTPUT];
        #pragma unroll
        for (int idx = 0; idx < (NUM_OUTPUT >> 1); idx += 1) {
            reinterpret_cast<__half2*>(col)[idx] = half2(0.0f, 0.0f);
        }

        scalar_t *data_half = const_cast<scalar_t *>(data_sampling_loc);
        scalar_t *data_attn_weight_half = const_cast<scalar_t *>(data_attn_weight);
        const half2 zp5 = half2(0.5f, 0.5f);

        for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const half2 spatail_hw = half2(spatial_w, spatial_h);

            // Get TMA descriptor for this batch and level
            const TmaDescriptor* tma_desc = &tma_descriptors[b_col * NUM_LEVELS + l_col];

            // Load sampling locations and attention weights (same as original)
            half2 loc_hw_vec[NUM_POINT];
            half weight_vec[NUM_POINT];

            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 4) {
                LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
            }
            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8) {
                LDST128BITS(weight_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
            }

            data_loc_w_ptr += (NUM_POINT << 1);
            data_weight_ptr += NUM_POINT;

            // Process each sampling point
            #pragma unroll
            for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
                const half2 loc = loc_hw_vec[p_col];
                const scalar_t weight = weight_vec[p_col];
                half2 weighthalf2 = half2(weight, weight);
                half2 hw_im = __hfma2(loc, spatail_hw, zp5);
                scalar_t h_im = __high2half(hw_im);
                scalar_t w_im = __low2half(hw_im);

                if (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) &&
                    h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1)) {

                    int32_t const hLow = __half2int_rd(h_im);
                    int32_t const wLow = __half2int_rd(w_im);

                    const __half lh = __hsub(h_im, __int2half_rd(hLow));
                    const __half lw = __hsub(w_im, __int2half_rd(wLow));
                    const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);

                    __half pst_lh[4] = {hh, hh, lh, lh};
                    __half pst_rh[4] = {hw, lw, hw, lw};
                    __half wdata[4];
                    HALF2(wdata[0]) = __hmul2(HALF2(pst_lh[0]), HALF2(pst_rh[0]));
                    HALF2(wdata[2]) = __hmul2(HALF2(pst_lh[2]), HALF2(pst_rh[2]));
                    __half wdataexp[2];

                    // Load 4 corners using TMA
                    // We need to load a tile that covers both (hLow, wLow) and (hHigh, wHigh)
                    // TMA loads 2x2x32 starting at (hLow, wLow)

                    // Only one thread per warp issues TMA
                    if (lane_id == 0) {
                        int32_t tensor_coord_c = 0;      // X = C (always 0 for full channel load)
                        int32_t tensor_coord_w = wLow;   // Y = W
                        int32_t tensor_coord_h = hLow;   // Z = H

                        // Issue TMA load
                        asm volatile(
                            "{\n\t"
                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                            "}"
                            :
                            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[0][0][0]))),
                              "l"(reinterpret_cast<uint64_t>(tma_desc)),
                              "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                              "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                            : "memory"
                        );

                        // Update barrier
                        asm volatile(
                            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                            :
                            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                              "n"(2 * 2 * CHANNELS * sizeof(scalar_t))
                        );
                    }

                    // All threads in warp wait for TMA
                    barrier::arrival_token token = bar.arrive();
                    bar.wait(std::move(token));

                    // Reset barrier for next iteration
                    if (lane_id == 0) {
                        init(&bar, blockDim.x);
                        asm volatile("fence.proxy.async.shared::cta;");
                    }
                    __syncwarp();

                    // Now use loaded data from shared memory
                    // smem_tile[h_local][w_local][c] where h_local, w_local in [0,1]
                    scalar_t vdata2d[NUM_OUTPUT];

                    // Corner 0: (hLow, wLow) -> smem[0][0][c_col..]
                    HALF2(wdataexp[0]) = __hmul2(half2(wdata[0], wdata[0]), HALF2(weighthalf2));
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(vdata2d[j]) = LDST128BITS(smem_tile[0][0][c_col + j]);
                        #pragma unroll
                        for (int p = 0; p < 8; p += 2) {
                            HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
                        }
                    }

                    // Corner 1: (hLow, wHigh) -> smem[0][1][c_col..]
                    HALF2(wdataexp[0]) = __hmul2(half2(wdata[1], wdata[1]), HALF2(weighthalf2));
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(vdata2d[j]) = LDST128BITS(smem_tile[0][1][c_col + j]);
                        #pragma unroll
                        for (int p = 0; p < 8; p += 2) {
                            HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
                        }
                    }

                    // Corner 2: (hHigh, wLow) -> smem[1][0][c_col..]
                    HALF2(wdataexp[0]) = __hmul2(half2(wdata[2], wdata[2]), HALF2(weighthalf2));
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(vdata2d[j]) = LDST128BITS(smem_tile[1][0][c_col + j]);
                        #pragma unroll
                        for (int p = 0; p < 8; p += 2) {
                            HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
                        }
                    }

                    // Corner 3: (hHigh, wHigh) -> smem[1][1][c_col..]
                    HALF2(wdataexp[0]) = __hmul2(half2(wdata[3], wdata[3]), HALF2(weighthalf2));
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(vdata2d[j]) = LDST128BITS(smem_tile[1][1][c_col + j]);
                        #pragma unroll
                        for (int p = 0; p < 8; p += 2) {
                            HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
                        }
                    }
                }
            }
        }

        // Write output (same as original)
        #pragma unroll
        for (int idx = 0; idx < NUM_OUTPUT; idx += 8) {
            __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
            data_col_ptr += 8;
        }
    }
}

// Host function to create TMA descriptors
extern "C"
CUresult createTMADescriptorsForAllBatches(
    TmaDescriptor* h_descriptors,
    const void** d_value_ptrs,
    const int64_t* h_spatial_shapes,
    const int64_t* h_level_start_index,
    int batch_size,
    int num_levels,
    int channels
) {
    printf("Creating TMA descriptors: batch_size=%d, num_levels=%d, channels=%d\n",
           batch_size, num_levels, channels);

    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();
    if (!cuTensorMapEncodeTiled_func) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int level = 0; level < num_levels; ++level) {
            int descriptor_idx = batch_idx * num_levels + level;

            const int spatial_h = h_spatial_shapes[level * 2];
            const int spatial_w = h_spatial_shapes[level * 2 + 1];
            const int level_start = h_level_start_index[level];

            // TMA dimensions: X=C, Y=W, Z=H (innermost to outermost)
            cuuint64_t globalDim[3] = {
                (cuuint64_t)channels,    // X = C
                (cuuint64_t)spatial_w,   // Y = W
                (cuuint64_t)spatial_h    // Z = H
            };

            // Strides for [H][W][C] memory layout
            cuuint64_t globalStrides[2] = {
                channels * sizeof(__half),              // stride[0]: skip to next W
                spatial_w * channels * sizeof(__half)   // stride[1]: skip to next H
            };

            cuuint32_t boxDim[3] = {32, 2, 2};  // C=32, W=2, H=2
            cuuint32_t elementStrides[3] = {1, 1, 1};

            const void* level_ptr = (const __half*)d_value_ptrs[batch_idx] + level_start * channels;

            CUresult result = cuTensorMapEncodeTiled_func(
                &h_descriptors[descriptor_idx],
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                3,
                (void*)level_ptr,
                globalDim,
                globalStrides,
                boxDim,
                elementStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );

            if (result != CUDA_SUCCESS) {
                printf("ERROR: Descriptor[%d][%d] creation failed\n", batch_idx, level);
                return result;
            }
        }
    }

    printf("✓ All TMA descriptors created successfully\n");
    return CUDA_SUCCESS;
}

// Launch function
template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda_tma(
    cudaStream_t stream,
    const scalar_t *data_value,
    const TmaDescriptor *tma_descriptors,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *data_col) {

    const int num_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
    const int num_threads = THREADS_IN_ONE_BLOCK;

    if (num_heads == 1 && num_point == 8 && num_levels == 4 && channels == 32) {
        ms_deformable_im2col_tma_kernel<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT>
            <<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
                num_kernels,
                data_value,
                tma_descriptors,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_query,
                data_col
            );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in ms_deformable_im2col_cuda_tma: %s\n", cudaGetErrorString(err));
    }
}

// Explicit instantiation
template void ms_deformable_im2col_cuda_tma<__half, 512, 8, 3>(
    cudaStream_t stream,
    const __half *data_value,
    const TmaDescriptor *tma_descriptors,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const __half *data_sampling_loc,
    const __half *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    __half *data_col);
