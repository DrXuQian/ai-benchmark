#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>

// Deformable Attention - Data Loading Only (No Bilinear Interpolation)
// Based on deform_attn.cu, removed all bilinear computation
// Only loads 2×2×32 tiles from 4 corners for comparison with TMA version

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

template <typename scalar_t = __half, const int NUM_POINT = 8, const int NUM_LEVELS = 4,
          const int CHANNELS = 32, const int CHANNELS_SHIFT = 5, const int NUM_OUTPUT = 8>
__global__ void deform_attn_load_only_kernel(
    const int n, const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const int batch_size, const int spatial_size,
    const int num_query,
    scalar_t *data_col)  // Output: [batch][query][levels][points][2][2][32]
{
    CUDA_1D_KERNEL_LOOP(index, n) {
        // Parse thread index
        int _temp = index << 3;  // index * 8 (NUM_OUTPUT)
        const int c_col = _temp & (CHANNELS - 1);
        _temp = (_temp >> CHANNELS_SHIFT);
        const int sampling_index = _temp;
        const int b_col = (float)_temp / (float)num_query;

        const __half kZERO = __int2half_rz(0);
        const __half kONE = __int2half_rz(1);
        int32_t const wStride = CHANNELS;

        // Output pointer for this thread's 2×2×32 tiles
        // Layout: [levels][points][2][2][8 channels]
        scalar_t *output_base = data_col + (index << 3);

        int data_loc_w_ptr = (sampling_index << (2 + 3)) << 1;  // * NUM_LEVELS * NUM_POINT * 2
        const int data_value_ptr_init_offset = (b_col * spatial_size) << CHANNELS_SHIFT;

        const __half zp5_val = __float2half(0.5f);
        const half2 zp5 = half2(zp5_val, zp5_val);

        // Process all levels and points
        for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

            // hStride accounts for padding: (W+2) * C
            int32_t const hStride = (spatial_w + 2) << CHANNELS_SHIFT;

            const half2 spatial_hw = half2(spatial_w, spatial_h);

            const scalar_t *data_value_ptr =
                data_value + (data_value_ptr_init_offset + (level_start_id << CHANNELS_SHIFT));

            for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
                // Load sampling location
                const int loc_idx = data_loc_w_ptr + (p_col << 1);
                const half2 loc = *reinterpret_cast<const half2*>(&data_sampling_loc[loc_idx]);

                // Convert to image coordinates
                half2 hw_im = __hfma2(loc, spatial_hw, zp5);
                scalar_t h_im = __high2half(hw_im);
                scalar_t w_im = __low2half(hw_im);

                // Check bounds
                if (h_im > kZERO && w_im > kZERO &&
                    h_im < __int2half_rn(spatial_h + 1) && w_im < __int2half_rn(spatial_w + 1)) {

                    // Get 2×2 tile coordinates
                    int32_t const hLow = __half2int_rd(h_im);
                    int32_t const wLow = __half2int_rd(w_im);
                    int32_t const hHigh = hLow + 1;
                    int32_t const wHigh = wLow + 1;

                    // Calculate memory offsets
                    int32_t const hLowPtrOffset = hLow * hStride;
                    int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
                    int32_t const wLowPtrOffset = wLow << CHANNELS_SHIFT;
                    int32_t const wHighPtrOffset = wLowPtrOffset + wStride;

                    // Load 2×2×32 tile (only the 8 channels this thread handles)
                    __half tile_data[4][NUM_OUTPUT];  // [4 corners][8 channels]

                    // Corner 0: (hLow, wLow)
                    int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + c_col;
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(tile_data[0][j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr1 + j]);
                    }

                    // Corner 1: (hLow, wHigh)
                    int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + c_col;
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(tile_data[1][j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr2 + j]);
                    }

                    // Corner 2: (hHigh, wLow)
                    int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + c_col;
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(tile_data[2][j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr3 + j]);
                    }

                    // Corner 3: (hHigh, wHigh)
                    int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + c_col;
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(tile_data[3][j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr4 + j]);
                    }

                    // Write loaded data to output
                    // Output layout: [batch][query][levels][points][2][2][channels]
                    // This thread writes 8 channels (c_col to c_col+7)
                    const int base_idx = ((((sampling_index * NUM_LEVELS + l_col) * NUM_POINT + p_col) * 2 * 2) << CHANNELS_SHIFT) + c_col;

                    // Corner 0: h=0, w=0
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(data_col[base_idx + 0 * 2 * CHANNELS + 0 * CHANNELS + j]) = LDST128BITS(tile_data[0][j]);
                    }

                    // Corner 1: h=0, w=1
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(data_col[base_idx + 0 * 2 * CHANNELS + 1 * CHANNELS + j]) = LDST128BITS(tile_data[1][j]);
                    }

                    // Corner 2: h=1, w=0
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(data_col[base_idx + 1 * 2 * CHANNELS + 0 * CHANNELS + j]) = LDST128BITS(tile_data[2][j]);
                    }

                    // Corner 3: h=1, w=1
                    #pragma unroll
                    for (int j = 0; j < NUM_OUTPUT; j += 8) {
                        LDST128BITS(data_col[base_idx + 1 * 2 * CHANNELS + 1 * CHANNELS + j]) = LDST128BITS(tile_data[3][j]);
                    }
                }
            }
            data_loc_w_ptr += (NUM_POINT << 1);
        }
    }
}

// Helper to load binary file
template <typename T>
std::vector<T> load_binary(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        printf("Failed to open %s\n", filename);
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size_bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(size_bytes / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), size_bytes);
    return data;
}

int main() {
    printf("=== Deformable Attention Manual Loading Only ===\n\n");

    // Configuration
    const int batch = 1;
    const int num_query = 123376;  // Real number of queries from test data
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;

    printf("Configuration:\n");
    printf("  Batch: %d\n", batch);
    printf("  Queries: %d\n", num_query);
    printf("  Heads: %d\n", num_heads);
    printf("  Channels: %d\n", channels);
    printf("  Levels: %d\n", num_levels);
    printf("  Points: %d\n\n", num_points);

    // Load data
    printf("Loading test data...\n");
    auto h_value = load_binary<__half>("working/test_data_value.bin");
    auto h_spatial_shapes = load_binary<int64_t>("working/test_data_spatial_shapes.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");
    auto h_sampling_loc = load_binary<__half>("working/test_data_sampling_locations.bin");

    printf("  Value: %zu elements (%.2f MB)\n", h_value.size(), h_value.size() * sizeof(__half) / (1024.0 * 1024.0));
    printf("  Sampling locations: %zu elements\n", h_sampling_loc.size());
    printf("  Spatial shapes: %zu elements\n", h_spatial_shapes.size());
    printf("  Level start indices: %zu elements\n\n", h_level_start_index.size());

    // Print spatial shapes
    printf("Spatial shapes:\n");
    for (int i = 0; i < num_levels; i++) {
        printf("  Level %d: [%ld×%ld], start_idx=%ld\n",
               i, h_spatial_shapes[i*2], h_spatial_shapes[i*2+1], h_level_start_index[i]);
    }
    printf("\n");

    // Calculate spatial size
    int spatial_size = 0;
    for (int i = 0; i < num_levels; i++) {
        int h = h_spatial_shapes[i * 2];
        int w = h_spatial_shapes[i * 2 + 1];
        spatial_size += (h + 2) * (w + 2);  // Padded size
    }
    printf("Total spatial size (with padding): %d\n\n", spatial_size);

    // Allocate device memory
    __half *d_value, *d_sampling_loc, *d_output;
    int64_t *d_spatial_shapes, *d_level_start_index;

    cudaMalloc(&d_value, h_value.size() * sizeof(__half));
    cudaMalloc(&d_sampling_loc, h_sampling_loc.size() * sizeof(__half));
    cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t));
    cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t));

    // Output: [batch][query][levels][points][2][2][channels]
    size_t output_size = batch * num_query * num_heads * num_levels * num_points * 2 * 2 * channels;
    cudaMalloc(&d_output, output_size * sizeof(__half));

    cudaMemcpy(d_value, h_value.data(), h_value.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), h_sampling_loc.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    const int num_kernels = batch * num_query * num_heads * channels / 8;  // 8 outputs per thread
    const int threads = 512;
    const int blocks = (num_kernels + threads - 1) / threads;

    printf("Launching kernel...\n");
    printf("  Kernels: %d\n", num_kernels);
    printf("  Blocks: %d\n", blocks);
    printf("  Threads/block: %d\n\n", threads);

    // Warmup
    for (int i = 0; i < 3; i++) {
        deform_attn_load_only_kernel<<<blocks, threads>>>(
            num_kernels, d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, batch, spatial_size, num_query, d_output);
        cudaDeviceSynchronize();
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 10;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        deform_attn_load_only_kernel<<<blocks, threads>>>(
            num_kernels, d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, batch, spatial_size, num_query, d_output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
        printf("  Iteration %2d: %.4f ms\n", i + 1, iter_time);
    }

    float avg_time = total_time / iterations;

    printf("\n=== Performance Results ===\n");
    printf("  Average time: %.4f ms\n", avg_time);
    printf("  Throughput: %.2f queries/ms\n\n", num_query / avg_time);

    // Verification
    printf("=== Verification (first query, level 0, point 0) ===\n");
    std::vector<__half> h_output(128);
    cudaMemcpy(h_output.data(), d_output, 128 * sizeof(__half), cudaMemcpyDeviceToHost);

    printf("First tile (2×2×32):\n");
    printf("  Corner [0,0]:\n");
    for (int i = 0; i < 8; i++) {
        printf("    [%d]: %.4f\n", i, __half2float(h_output[i]));
    }

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_sampling_loc);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n✅ Manual loading test completed!\n");
    return 0;
}
