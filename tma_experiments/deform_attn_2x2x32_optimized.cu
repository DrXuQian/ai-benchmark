#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

// Deformable attention kernel with 2x2x32 tile loading optimization
// This kernel loads 2x2x32 tiles for efficient bilinear interpolation

template <typename scalar_t, const int NUM_POINT=8, const int NUM_LEVELS=4,
          const int CHANNELS=32, const int NUM_OUTPUT=8>
__global__ void ms_deformable_im2col_2x2x32(
    const int n,
    const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_query,
    scalar_t *data_col
) {
    // Shared memory for 2x2x32 tiles - reuse across levels and points
    __shared__ __align__(128) scalar_t smem_tile[2][2][CHANNELS];

    // Thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    // Calculate output position
    int output_offset = index * NUM_OUTPUT;
    int c_col = output_offset & (CHANNELS - 1);
    int temp = output_offset >> 5;  // Divide by CHANNELS
    int sampling_index = temp;
    int b_col = temp / num_query;

    // Output pointer
    scalar_t *data_col_ptr = data_col + output_offset;

    // Initialize accumulator
    scalar_t col[NUM_OUTPUT];
    #pragma unroll
    for (int i = 0; i < NUM_OUTPUT; i++) {
        col[i] = __float2half(0.0f);
    }

    // Get sampling location and weight pointers
    int data_weight_ptr = sampling_index * NUM_LEVELS * NUM_POINT;
    int data_loc_ptr = data_weight_ptr * 2;

    // Constants for bilinear interpolation
    const scalar_t kZERO = __float2half(0.0f);
    const scalar_t kONE = __float2half(1.0f);

    // Process each level
    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
        const int level_start_id = data_level_start_index[l_col];
        const int spatial_h = data_spatial_shapes[l_col * 2];
        const int spatial_w = data_spatial_shapes[l_col * 2 + 1];

        // Base pointer for this level
        const scalar_t *data_value_ptr = data_value +
            (b_col * spatial_size + level_start_id) * CHANNELS;

        // Process each sampling point
        for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
            // Get sampling location
            scalar_t loc_w = data_sampling_loc[data_loc_ptr + p_col * 2];
            scalar_t loc_h = data_sampling_loc[data_loc_ptr + p_col * 2 + 1];
            scalar_t weight = data_attn_weight[data_weight_ptr + p_col];

            // Convert normalized coordinates to image coordinates
            scalar_t w_im = __hfma(loc_w, __int2half_rn(spatial_w), __float2half(0.5f));
            scalar_t h_im = __hfma(loc_h, __int2half_rn(spatial_h), __float2half(0.5f));

            // Check if within bounds
            if (h_im > kZERO && w_im > kZERO &&
                h_im < __int2half_rn(spatial_h + 1) &&
                w_im < __int2half_rn(spatial_w + 1)) {

                // Get integer coordinates
                int hLow = __half2int_rd(h_im);
                int wLow = __half2int_rd(w_im);

                // Load 2x2x32 tile into shared memory cooperatively
                // Each warp loads one tile together
                const int warp_id = threadIdx.x / 32;
                const int lane_id = threadIdx.x % 32;

                if (lane_id < CHANNELS) {
                    int ch = lane_id;

                    // Load all 4 spatial positions for this channel
                    #pragma unroll
                    for (int dy = 0; dy < 2; dy++) {
                        #pragma unroll
                        for (int dx = 0; dx < 2; dx++) {
                            int y = hLow + dy;
                            int x = wLow + dx;

                            if (y >= 0 && y < spatial_h && x >= 0 && x < spatial_w) {
                                int src_idx = (y * spatial_w + x) * CHANNELS + ch;
                                smem_tile[dy][dx][ch] = data_value_ptr[src_idx];
                            } else {
                                smem_tile[dy][dx][ch] = kZERO;
                            }
                        }
                    }
                }

                __syncwarp();

                // Compute bilinear interpolation weights
                scalar_t lh = __hsub(h_im, __int2half_rd(hLow));
                scalar_t lw = __hsub(w_im, __int2half_rd(wLow));
                scalar_t hh = __hsub(kONE, lh);
                scalar_t hw = __hsub(kONE, lw);

                scalar_t w00 = __hmul(hh, hw);
                scalar_t w01 = __hmul(hh, lw);
                scalar_t w10 = __hmul(lh, hw);
                scalar_t w11 = __hmul(lh, lw);

                // Interpolate and accumulate for each output channel
                #pragma unroll
                for (int c = 0; c < NUM_OUTPUT; c++) {
                    int ch_idx = (c_col + c) % CHANNELS;

                    // Bilinear interpolation using the 2x2 tile
                    scalar_t val = __hfma(w00, smem_tile[0][0][ch_idx],
                                  __hfma(w01, smem_tile[0][1][ch_idx],
                                  __hfma(w10, smem_tile[1][0][ch_idx],
                                  __hmul(w11, smem_tile[1][1][ch_idx]))));

                    // Accumulate with attention weight
                    col[c] = __hfma(weight, val, col[c]);
                }
            }
        }

        // Update pointers for next level
        data_loc_ptr += NUM_POINT * 2;
        data_weight_ptr += NUM_POINT;
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < NUM_OUTPUT; i++) {
        data_col_ptr[i] = col[i];
    }
}

// Host function to launch the kernel
template <typename scalar_t>
void ms_deformable_im2col_cuda_2x2x32(
    cudaStream_t stream,
    const scalar_t* data_value,
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
    const int num_actual_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = 256;

    if (num_kernels == 0) return;

    if (channels == 32 && num_levels == 4 && num_point == 8) {
        const int NUM_POINT = 8;
        const int NUM_LEVELS = 4;
        const int CHANNELS = 32;
        const int NUM_OUTPUT = 8;

        ms_deformable_im2col_2x2x32<scalar_t, NUM_POINT, NUM_LEVELS, CHANNELS, NUM_OUTPUT>
            <<<(num_kernels + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                num_kernels,
                data_value,
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

// Explicit template instantiation
template void ms_deformable_im2col_cuda_2x2x32<__half>(
    cudaStream_t stream,
    const __half* data_value,
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
