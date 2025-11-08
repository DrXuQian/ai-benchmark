#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled
#include <cstdio>
#include <cstdlib>

// 真正的TMA实现 - 使用cp.async.bulk.tensor.3d PTX指令
// 使用动态entry point获取cuTensorMapEncodeTiled (required for SM 12.0)
//
// CRITICAL: TMA Dimension Mapping for [H][W][C] Memory Layout
// ============================================================
// Memory layout: [H][W][C] (row-major, C is innermost/contiguous)
// TMA dimensions: X, Y, Z map from innermost to outermost
// Therefore:
//   X = C (channels, innermost)
//   Y = W (width)
//   Z = H (height, outermost)
//
// Descriptor setup:
//   globalDim = [C, W, H]
//   stride[0] = C * sizeof(dtype)      // bytes to skip to next W
//   stride[1] = W * C * sizeof(dtype)  // bytes to skip to next H
//   boxDim = [32, 2, 2]                // load C=32, W=2, H=2
//
// PTX coordinates:
//   coord_x = c (channel index, always 0 for full channel load)
//   coord_y = w (width position)
//   coord_z = h (height position)

using TmaDescriptor = CUtensorMap;

// Get cuTensorMapEncodeTiled function pointer dynamically
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

// Helper function to convert pointer to shared memory address
__device__ __forceinline__ uint32_t __as_ptr_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <typename scalar_t, const int NUM_POINT=8, const int NUM_LEVELS=4,
          const int CHANNELS=32, const int NUM_OUTPUT=8>
__global__ void ms_deformable_im2col_tma_kernel(
    const int n,
    const scalar_t *data_value,
    const TmaDescriptor *tma_descriptors,  // One descriptor per batch*level
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_query,
    scalar_t *data_col
) {
    // Shared memory for 2x2x32 tile
    __shared__ __align__(128) scalar_t smem_tile[2][2][CHANNELS];

    // Barrier for TMA synchronization
    __shared__ __align__(8) uint64_t barrier;

    // Initialize barrier
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(__as_ptr_smem(&barrier)), "r"(1));
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    // Calculate output position
    int output_offset = index * NUM_OUTPUT;
    int c_col = output_offset & (CHANNELS - 1);
    int temp = output_offset >> 5;  // Divide by CHANNELS
    int sampling_index = temp;
    int b_col = temp / num_query;

    scalar_t *data_col_ptr = data_col + output_offset;

    // Initialize accumulator
    scalar_t col[NUM_OUTPUT];
    #pragma unroll
    for (int i = 0; i < NUM_OUTPUT; i++) {
        col[i] = __float2half(0.0f);
    }

    int data_weight_ptr = sampling_index * NUM_LEVELS * NUM_POINT;
    int data_loc_ptr = data_weight_ptr * 2;

    const scalar_t kZERO = __float2half(0.0f);
    const scalar_t kONE = __float2half(1.0f);

    // Process each level
    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
        const int spatial_h = data_spatial_shapes[l_col * 2];
        const int spatial_w = data_spatial_shapes[l_col * 2 + 1];

        // Get TMA descriptor for this batch and level
        const TmaDescriptor* tma_desc = &tma_descriptors[b_col * NUM_LEVELS + l_col];

        // Process each sampling point
        for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
            // Get sampling location
            scalar_t loc_w = data_sampling_loc[data_loc_ptr + p_col * 2];
            scalar_t loc_h = data_sampling_loc[data_loc_ptr + p_col * 2 + 1];
            scalar_t weight = data_attn_weight[data_weight_ptr + p_col];

            // Convert to image coordinates
            scalar_t w_im = __hfma(loc_w, __int2half_rn(spatial_w), __float2half(0.5f));
            scalar_t h_im = __hfma(loc_h, __int2half_rn(spatial_h), __float2half(0.5f));

            // Check bounds
            if (h_im > kZERO && w_im > kZERO &&
                h_im < __int2half_rn(spatial_h + 1) &&
                w_im < __int2half_rn(spatial_w + 1)) {

                int hLow = __half2int_rd(h_im);
                int wLow = __half2int_rd(w_im);

                // Only one thread issues TMA for the entire warp
                const int lane_id = threadIdx.x % 32;
                if (lane_id == 0) {
                    // TMA coordinates: X=C, Y=W, Z=H (innermost to outermost)
                    // Load 2x2x32 tile starting at (hLow, wLow, 0)
                    int32_t coord_c = 0;      // X coordinate (C=0)
                    int32_t coord_w = wLow;   // Y coordinate (W)
                    int32_t coord_h = hLow;   // Z coordinate (H)

                    // Issue TMA load using inline PTX
                    // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
                    //   [dstMem], [tensorMap, {X, Y, Z}], [smem_bar];
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];"
                        :
                        : "r"(__as_ptr_smem(&smem_tile[0][0][0])),
                          "l"(tma_desc),
                          "r"(coord_c),  // X = C
                          "r"(coord_w),  // Y = W
                          "r"(coord_h),  // Z = H
                          "r"(__as_ptr_smem(&barrier))
                        : "memory"
                    );

                    // Wait for TMA to complete
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

                // All threads wait for TMA completion
                __syncwarp();

                // Compute bilinear interpolation
                scalar_t lh = __hsub(h_im, __int2half_rd(hLow));
                scalar_t lw = __hsub(w_im, __int2half_rd(wLow));
                scalar_t hh = __hsub(kONE, lh);
                scalar_t hw = __hsub(kONE, lw);

                scalar_t w00 = __hmul(hh, hw);
                scalar_t w01 = __hmul(hh, lw);
                scalar_t w10 = __hmul(lh, hw);
                scalar_t w11 = __hmul(lh, lw);

                // Interpolate and accumulate
                #pragma unroll
                for (int c = 0; c < NUM_OUTPUT; c++) {
                    int ch_idx = (c_col + c) % CHANNELS;

                    scalar_t val = __hfma(w00, smem_tile[0][0][ch_idx],
                                  __hfma(w01, smem_tile[0][1][ch_idx],
                                  __hfma(w10, smem_tile[1][0][ch_idx],
                                  __hmul(w11, smem_tile[1][1][ch_idx]))));

                    col[c] = __hfma(weight, val, col[c]);
                }

                // Reset barrier for next iteration
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

    // Write output
    #pragma unroll
    for (int i = 0; i < NUM_OUTPUT; i++) {
        data_col_ptr[i] = col[i];
    }
}

// Host function to create TMA descriptors for all batches and levels
extern "C"
CUresult createTMADescriptorsForAllBatches(
    TmaDescriptor* h_descriptors,  // Output: array of descriptors [batch_size * num_levels]
    const void** d_value_ptrs,      // Input: device pointers for each batch
    const int64_t* h_spatial_shapes, // [num_levels * 2]: (H, W) for each level
    const int64_t* h_level_start_index, // [num_levels]: start index for each level
    int batch_size,
    int num_levels,
    int channels
) {
    printf("Creating TMA descriptors: batch_size=%d, num_levels=%d, channels=%d\n",
           batch_size, num_levels, channels);

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int level = 0; level < num_levels; ++level) {
            int descriptor_idx = batch_idx * num_levels + level;

            const int spatial_h = h_spatial_shapes[level * 2];
            const int spatial_w = h_spatial_shapes[level * 2 + 1];
            const int level_start = h_level_start_index[level];

            printf("  Descriptor[%d][%d]: H=%d, W=%d, C=%d\n",
                   batch_idx, level, spatial_h, spatial_w, channels);

            // Global tensor dimensions: X=C, Y=W, Z=H (innermost to outermost)
            // For [H][W][C] memory layout, TMA X,Y,Z maps to C,W,H
            cuuint64_t globalDim[3] = {
                (cuuint64_t)channels,    // X = C (innermost)
                (cuuint64_t)spatial_w,   // Y = W
                (cuuint64_t)spatial_h    // Z = H (outermost)
            };

            // Strides in bytes for [H][W][C] memory layout
            // stride[0] = bytes to skip from [h][w][c] to [h][w+1][c] (skip one column)
            // stride[1] = bytes to skip from [h][w][c] to [h+1][w][c] (skip one row)
            cuuint64_t globalStrides[2] = {
                channels * sizeof(__half),              // stride[0]: skip to next W
                spatial_w * channels * sizeof(__half)   // stride[1]: skip to next H
            };

            // Box/tile dimensions: X=32 (C), Y=2 (W), Z=2 (H)
            cuuint32_t boxDim[3] = {32, 2, 2};

            // Element strides: contiguous access pattern
            cuuint32_t elementStrides[3] = {1, 1, 1};

            // Offset pointer for this level
            const void* level_ptr = (const __half*)d_value_ptrs[batch_idx] + level_start * channels;

            // Get function pointer dynamically
            static auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();
            if (!cuTensorMapEncodeTiled_func) {
                printf("ERROR: Failed to get cuTensorMapEncodeTiled function pointer\n");
                return CUDA_ERROR_NOT_SUPPORTED;
            }

            CUresult result = cuTensorMapEncodeTiled_func(
                &h_descriptors[descriptor_idx],
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                3,  // rank
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
                const char* error_name;
                cuGetErrorName(result, &error_name);
                printf("  ERROR: Failed to create descriptor[%d][%d]: %s\n",
                       batch_idx, level, error_name);
                return result;
            }
        }
    }

    printf("✓ All TMA descriptors created successfully\n");
    return CUDA_SUCCESS;
}

// Host launch function
template <typename scalar_t>
void ms_deformable_im2col_cuda_tma(
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
        ms_deformable_im2col_tma_kernel<scalar_t, 8, 4, 32, 8>
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
    } else {
        printf("Unsupported configuration: channels=%d, levels=%d, points=%d\n",
               channels, num_levels, num_point);
    }
}

// Explicit instantiation
template void ms_deformable_im2col_cuda_tma<__half>(
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
