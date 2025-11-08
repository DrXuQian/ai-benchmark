#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// TMA Data Loading Kernel - matches deform_attn pattern
// threadIdx % 4 == 0 threads issue TMA loads
// Correct barrier initialization for subset of threads

using barrier = cuda::barrier<cuda::thread_scope_block>;
typedef __half dtype;

inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// Configuration for first level: H=92, W=160, C=32
#define SPATIAL_H 92
#define SPATIAL_W 160
#define CHANNELS 32
#define TILE_H 2
#define TILE_W 2
#define TILE_C 32
#define LOAD_SIZE (TILE_H * TILE_W * TILE_C * sizeof(dtype))  // 256 bytes

// TMA loading kernel
// Each threadIdx%4==0 thread loads 4 corner points for bilinear sampling
// No computation, just data loading verification
__global__ void tma_deform_attn_load_kernel(
    const __grid_constant__ CUtensorMap tma_desc,
    const dtype *sampling_loc,       // [batch][num_query][num_heads][num_levels][num_points][2]
    const int batch_size,
    const int num_query,
    const int num_levels,
    const int num_points,
    dtype *output  // Store loaded data for verification
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Shared memory for TMA loads - 8 slots for 8 TMA threads (num_points=8)
    __shared__ alignas(128) dtype smem_tile[8][2][2][32];

    // Barrier - CRITICAL: Initialize for threads that will arrive
    // All threads in the block will arrive, but only threadIdx%4==0 will issue TMA
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    // Number of threads that will issue TMA = blockDim.x / 4
    const int num_tma_threads = blockDim.x / 4;

    // Initialize barrier once - for ALL threads in block
    if (tid == 0) {
        init(&bar, blockDim.x);  // All threads participate in barrier
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    // Calculate which query this block is handling
    // Simplified: each block handles one sampling point
    if (bid >= batch_size * num_query) return;

    const int b_col = bid / num_query;
    const int q_col = bid % num_query;

    // Only threadIdx%4==0 threads will load data
    const bool is_loader = (tid % 4 == 0);
    const int loader_id = tid / 4;  // 0 to num_tma_threads-1

    // Process one level and one point per block (simplified for testing)
    const int l_col = 0;  // Test with first level
    const int p_col = loader_id % num_points;  // Distribute points across loaders

    if (is_loader && loader_id < num_points) {
        // Get sampling location
        // Index: [b][q][h][l][p][2]
        const int loc_idx = ((((b_col * num_query + q_col) * 1) * num_levels + l_col) * num_points + p_col) * 2;
        dtype loc_w_norm = sampling_loc[loc_idx];
        dtype loc_h_norm = sampling_loc[loc_idx + 1];

        // Convert to image coordinates
        dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
        dtype h_im = __hfma(loc_h_norm, __int2half_rn(SPATIAL_H), __float2half(0.5f));

        dtype kZERO = __float2half(0.0f);
        dtype kONE = __float2half(1.0f);

        // Check bounds
        if (h_im > kZERO && w_im > kZERO &&
            h_im < __int2half_rn(SPATIAL_H + 1) && w_im < __int2half_rn(SPATIAL_W + 1)) {

            int hLow = __half2int_rd(h_im);
            int wLow = __half2int_rd(w_im);

            // Clamp
            hLow = max(0, min(hLow, SPATIAL_H - 2));
            wLow = max(0, min(wLow, SPATIAL_W - 2));

            // TMA coordinates: X=C, Y=W, Z=H
            int32_t tensor_coord_c = 0;
            int32_t tensor_coord_w = wLow;
            int32_t tensor_coord_h = hLow;

            // Issue TMA load - each loader writes to its own slot
            asm volatile(
                "{\n\t"
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                "}"
                :
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[p_col][0][0][0]))),
                  "l"(reinterpret_cast<uint64_t>(&tma_desc)),
                  "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                  "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "memory"
            );

            // Update barrier - only TMA issuing threads call this
            asm volatile(
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                :
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                  "n"(2 * 2 * 32 * sizeof(dtype))
            );
        }
    }

    // ALL threads wait for TMA completion (not just loaders)
    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

    // Copy loaded data to output for verification
    // output[bid][point_id][128 elements from smem]
    // Each block outputs 8 points * 128 elements = 1024 elements
    for (int p = 0; p < num_points; p++) {
        for (int idx = tid; idx < 128; idx += blockDim.x) {
            int h = idx / 64;
            int w = (idx / 32) % 2;
            int c = idx % 32;
            output[bid * num_points * 128 + p * 128 + idx] = smem_tile[p][h][w][c];
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
    printf("Loaded %s: %zu elements\n", filename, data.size());
    return data;
}

int main() {
    printf("=== TMA Data Loading Test (threadIdx%%4==0) ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Configuration
    const int batch = 1;  // Test with 1 batch for simplicity
    const int num_query = 1000;  // Test with 1000 queries
    const int num_heads = 1;
    const int num_levels = 4;
    const int num_points = 8;

    printf("Configuration:\n");
    printf("  batch=%d, num_query=%d, num_levels=%d, num_points=%d\n",
           batch, num_query, num_levels, num_points);
    printf("  channels=%d\n\n", CHANNELS);

    // Load data
    printf("Loading test data...\n");
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_spatial_shapes = load_binary<int64_t>("working/test_data_spatial_shapes.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");
    printf("\n");

    // Print spatial shapes
    printf("Spatial shapes:\n");
    for (int i = 0; i < num_levels; i++) {
        printf("  Level %d: [%ld, %ld]\n", i, h_spatial_shapes[i*2], h_spatial_shapes[i*2+1]);
    }
    printf("\n");

    // Extract first level data for first batch
    const int level_start = h_level_start_index[0];  // 0
    const int level_size = SPATIAL_H * SPATIAL_W;     // 14720

    // Allocate device memory
    dtype *d_value, *d_sampling_loc, *d_output;

    // Copy first batch, first level value data
    size_t level_data_size = level_size * CHANNELS;
    CUDA_CHECK(cudaMalloc(&d_value, level_data_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data() + level_start * CHANNELS,
                          level_data_size * sizeof(dtype), cudaMemcpyHostToDevice));

    // Extract sampling locations for first batch
    // sampling_loc layout: [batch][num_query][num_heads][num_levels][num_points][2]
    size_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(),
                          sampling_loc_size * sizeof(dtype), cudaMemcpyHostToDevice));

    // Output: batch * num_query blocks, each outputs num_points * 128 elements
    const int num_blocks = batch * num_query;
    CUDA_CHECK(cudaMalloc(&d_output, num_blocks * num_points * 128 * sizeof(dtype)));

    // Create TMA descriptor
    printf("Creating TMA descriptor...\n");
    CUtensorMap tma_desc{};
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    // TMA dimensions: X=C, Y=W, Z=H
    uint64_t globalDim[3] = {CHANNELS, SPATIAL_W, SPATIAL_H};
    uint64_t globalStrides[2] = {
        CHANNELS * sizeof(dtype),              // stride[0]: skip to next W
        SPATIAL_W * CHANNELS * sizeof(dtype)   // stride[1]: skip to next H
    };
    uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
    uint32_t elementStrides[3] = {1, 1, 1};

    CUresult res = cuTensorMapEncodeTiled_func(
        &tma_desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        d_value,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("Failed to create TMA descriptor\n");
        return 1;
    }
    printf("✅ Created TMA descriptor\n\n");

    // Launch kernel
    printf("Launching TMA loading kernel...\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads per block: 256\n");
    printf("  TMA threads per block: %d (threadIdx%%4==0)\n\n", 256/4);

    tma_deform_attn_load_kernel<<<num_blocks, 256>>>(
        tma_desc,
        d_sampling_loc,
        batch,
        num_query,
        num_levels,
        num_points,
        d_output
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("✅ Kernel completed successfully\n\n");

    // Verify results - verify first few blocks, all 8 points per block
    const int num_verify_blocks = 100;  // Verify first 100 blocks
    std::vector<dtype> h_output(num_verify_blocks * num_points * 128);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(dtype), cudaMemcpyDeviceToHost));

    printf("Verifying TMA loaded data against CPU ground truth...\n");
    printf("Verifying %d blocks × %d points = %d total samples\n\n", num_verify_blocks, num_points, num_verify_blocks * num_points);

    int total_errors = 0;
    int total_elements = 0;
    float max_diff = 0.0f;

    for (int bid = 0; bid < num_verify_blocks; bid++) {
        const int b_col = bid / num_query;
        const int q_col = bid % num_query;

        // Verify all 8 points for this block
        for (int p = 0; p < num_points; p++) {
            // Get location for this point
            const int loc_idx = ((((b_col * num_query + q_col) * 1) * num_levels + 0) * num_points + p) * 2;
            dtype loc_w_norm = h_sampling_loc[loc_idx];
            dtype loc_h_norm = h_sampling_loc[loc_idx + 1];

            float w_im = __half2float(loc_w_norm) * SPATIAL_W + 0.5f;
            float h_im = __half2float(loc_h_norm) * SPATIAL_H + 0.5f;

            // Skip out of bounds
            if (h_im <= 0.0f || w_im <= 0.0f ||
                h_im >= (SPATIAL_H + 1) || w_im >= (SPATIAL_W + 1)) {
                continue;
            }

            int hLow = (int)floor(h_im);
            int wLow = (int)floor(w_im);
            hLow = std::max(0, std::min(hLow, SPATIAL_H - 2));
            wLow = std::max(0, std::min(wLow, SPATIAL_W - 2));

            // Compare each element in the loaded tile
            bool sample_has_error = false;
            for (int h = 0; h < TILE_H; h++) {
                for (int w = 0; w < TILE_W; w++) {
                    for (int c = 0; c < TILE_C; c++) {
                        // Calculate index in original value array
                        // value layout: [H][W][C]
                        int value_h = hLow + h;
                        int value_w = wLow + w;
                        int value_idx = (value_h * SPATIAL_W + value_w) * CHANNELS + c;

                        // Get ground truth from CPU
                        float gt_value = __half2float(h_value[level_start * CHANNELS + value_idx]);

                        // Get TMA loaded value - output[bid][p][tile_data]
                        int output_idx = bid * num_points * 128 + p * 128 + h * 64 + w * 32 + c;
                        float tma_value = __half2float(h_output[output_idx]);

                        // Compare
                        float diff = std::abs(gt_value - tma_value);
                        max_diff = std::max(max_diff, diff);

                        if (diff > 1e-6f) {
                            if (!sample_has_error) {
                                if (bid < 5 || total_errors < 10) {
                                    printf("  Block %d Point %d ERROR at [h=%d,w=%d,c=%d]: pos=[%d,%d] GT=%.6f, TMA=%.6f, diff=%.6f\n",
                                           bid, p, h, w, c, hLow+h, wLow+w, gt_value, tma_value, diff);
                                }
                            }
                            sample_has_error = true;
                            total_errors++;
                        }
                        total_elements++;
                    }
                }
            }

            if (!sample_has_error && bid < 3) {
                printf("  Block %d Point %d: ✅ PASS - Position [h=%d,w=%d], first value: %.6f\n",
                       bid, p, hLow, wLow, __half2float(h_output[bid * num_points * 128 + p * 128]));
            } else if (sample_has_error) {
                printf("  Block %d Point %d: ❌ FAIL - Position [h=%d,w=%d]\n",
                       bid, p, hLow, wLow);
            }
        }
    }

    printf("\n=== Verification Results ===\n");
    printf("Total elements checked: %d\n", total_elements);
    printf("Errors found: %d\n", total_errors);
    printf("Max difference: %.10f\n", max_diff);
    printf("Accuracy: %.4f%%\n", 100.0f * (total_elements - total_errors) / total_elements);

    if (total_errors == 0) {
        printf("\n🎉 ✅ ALL DATA VERIFIED CORRECT!\n");
        printf("TMA loaded data matches CPU ground truth perfectly.\n");
    } else {
        printf("\n❌ VERIFICATION FAILED\n");
        printf("TMA loaded data does NOT match ground truth.\n");
    }

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_sampling_loc);
    cudaFree(d_output);

    printf("\n✅ Test completed successfully!\n");
    return 0;
}
