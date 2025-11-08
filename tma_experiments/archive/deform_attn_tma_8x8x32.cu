#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>

// TMA deformable attention using 8x8x32 tiles (minimum size for SM 12.0)
// We load 8x8x32 but only use the 2x2 region needed for bilinear interpolation

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

__device__ __forceinline__ uint32_t __as_ptr_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <typename scalar_t, const int NUM_POINT=8, const int NUM_LEVELS=4,
          const int CHANNELS=32, const int NUM_OUTPUT=8>
__global__ void ms_deformable_im2col_tma_8x8x32(
    const int n,
    const scalar_t *data_value,
    const TmaDescriptor *tma_descriptors,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_query,
    scalar_t *data_col
) {
    // Shared memory for 8x8x32 tile (minimum TMA size)
    __shared__ __align__(128) scalar_t smem_tile[8][8][CHANNELS];

    // Barrier for TMA synchronization
    __shared__ __align__(8) uint64_t barrier;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(__as_ptr_smem(&barrier)), "r"(1));
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    int output_offset = index * NUM_OUTPUT;
    int c_col = output_offset & (CHANNELS - 1);
    int temp = output_offset >> 5;
    int sampling_index = temp;
    int b_col = temp / num_query;

    scalar_t *data_col_ptr = data_col + output_offset;

    scalar_t col[NUM_OUTPUT];
    #pragma unroll
    for (int i = 0; i < NUM_OUTPUT; i++) {
        col[i] = __float2half(0.0f);
    }

    int data_weight_ptr = sampling_index * NUM_LEVELS * NUM_POINT;
    int data_loc_ptr = data_weight_ptr * 2;

    const scalar_t kZERO = __float2half(0.0f);
    const scalar_t kONE = __float2half(1.0f);

    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
        const int level_start_id = data_level_start_index[l_col];
        const int spatial_h = data_spatial_shapes[l_col * 2];
        const int spatial_w = data_spatial_shapes[l_col * 2 + 1];

        const TmaDescriptor* tma_desc = &tma_descriptors[b_col * NUM_LEVELS + l_col];

        for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
            scalar_t loc_w = data_sampling_loc[data_loc_ptr + p_col * 2];
            scalar_t loc_h = data_sampling_loc[data_loc_ptr + p_col * 2 + 1];
            scalar_t weight = data_attn_weight[data_weight_ptr + p_col];

            scalar_t w_im = __hfma(loc_w, __int2half_rn(spatial_w), __float2half(0.5f));
            scalar_t h_im = __hfma(loc_h, __int2half_rn(spatial_h), __float2half(0.5f));

            if (h_im > kZERO && w_im > kZERO &&
                h_im < __int2half_rn(spatial_h + 1) &&
                w_im < __int2half_rn(spatial_w + 1)) {

                int hLow = __half2int_rd(h_im);
                int wLow = __half2int_rd(w_im);

                // Issue TMA to load 8x8x32 tile
                // We'll use the top-left 2x2 region for bilinear interpolation
                const int lane_id = threadIdx.x % 32;
                if (lane_id == 0) {
                    // Load tile starting at (hLow, wLow)
                    // Make sure we don't go out of bounds
                    int32_t coord_h = hLow;
                    int32_t coord_w = wLow;
                    int32_t coord_c = 0;

                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];"
                        :
                        : "r"(__as_ptr_smem(&smem_tile[0][0][0])),
                          "l"(tma_desc),
                          "r"(coord_h),
                          "r"(coord_w),
                          "r"(coord_c),
                          "r"(__as_ptr_smem(&barrier))
                        : "memory"
                    );

                    // Wait for TMA
                    asm volatile(
                        "{\n\t"
                        "  .reg .pred p;\n\t"
                        "  .reg .b32 r_bar;\n\t"
                        "  mov.b32 r_bar, %0;\n\t"
                        "$wait_loop_%=:\n\t"
                        "  mbarrier.try_wait.parity.shared.b64 p, [r_bar], 0;\n\t"
                        "  @!p bra $wait_loop_%=;\n\t"
                        "}\n\t"
                        :
                        : "r"(__as_ptr_smem(&barrier))
                    );
                }

                __syncwarp();

                // Bilinear interpolation using 2x2 region from the 8x8x32 tile
                scalar_t lh = __hsub(h_im, __int2half_rd(hLow));
                scalar_t lw = __hsub(w_im, __int2half_rd(wLow));
                scalar_t hh = __hsub(kONE, lh);
                scalar_t hw = __hsub(kONE, lw);

                scalar_t w00 = __hmul(hh, hw);
                scalar_t w01 = __hmul(hh, lw);
                scalar_t w10 = __hmul(lh, hw);
                scalar_t w11 = __hmul(lh, lw);

                #pragma unroll
                for (int c = 0; c < NUM_OUTPUT; c++) {
                    int ch_idx = (c_col + c) % CHANNELS;

                    // Use only the first 2x2 region from the loaded 8x8x32 tile
                    scalar_t val = __hfma(w00, smem_tile[0][0][ch_idx],
                                  __hfma(w01, smem_tile[0][1][ch_idx],
                                  __hfma(w10, smem_tile[1][0][ch_idx],
                                  __hmul(w11, smem_tile[1][1][ch_idx]))));

                    col[c] = __hfma(weight, val, col[c]);
                }

                // Reset barrier
                if (lane_id == 0) {
                    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                                 :: "r"(__as_ptr_smem(&barrier)), "r"(1));
                }
                __syncwarp();
            }
        }

        data_loc_ptr += NUM_POINT * 2;
        data_weight_ptr += NUM_POINT;
    }

    #pragma unroll
    for (int i = 0; i < NUM_OUTPUT; i++) {
        data_col_ptr[i] = col[i];
    }
}

// Create TMA descriptors for 8x8x32 tiles
extern "C"
CUresult createTMADescriptorsForAllBatches_8x8x32(
    TmaDescriptor* h_descriptors,
    const void** d_value_ptrs,
    const int64_t* h_spatial_shapes,
    const int64_t* h_level_start_index,
    int batch_size,
    int num_levels,
    int channels
) {
    printf("Creating TMA descriptors (8x8x32 tiles): batch_size=%d, num_levels=%d, channels=%d\n",
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

            cuuint64_t globalDim[3] = {
                (cuuint64_t)spatial_h,
                (cuuint64_t)spatial_w,
                (cuuint64_t)channels
            };

            cuuint64_t globalStrides[2] = {
                spatial_w * channels * sizeof(__half),
                channels * sizeof(__half)
            };

            // 8x8x32 tile (minimum for SM 12.0)
            cuuint32_t boxDim[3] = {8, 8, 32};
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

    printf("✓ All TMA descriptors (8x8x32) created successfully\n");
    return CUDA_SUCCESS;
}

// Launch function
template <typename scalar_t>
void ms_deformable_im2col_cuda_tma_8x8x32(
    cudaStream_t stream,
    const scalar_t* data_value,
    const TmaDescriptor* tma_descriptors,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col)
{
    const int num_kernels = batch_size * num_query * num_heads * channels / 8;
    const int num_threads = 256;

    if (num_kernels == 0) return;

    if (channels == 32 && num_levels == 4 && num_point == 8) {
        ms_deformable_im2col_tma_8x8x32<scalar_t, 8, 4, 32, 8>
            <<<(num_kernels + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
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
}

template void ms_deformable_im2col_cuda_tma_8x8x32<__half>(
    cudaStream_t stream,
    const __half* data_value,
    const TmaDescriptor* tma_descriptors,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const __half* data_sampling_loc,
    const __half* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    __half* data_col);
