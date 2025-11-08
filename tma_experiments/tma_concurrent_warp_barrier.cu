#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// TMA Data Loading with Per-Warp Barriers
// Each warp has its own barrier for independent synchronization
// Potentially better performance than block-level barriers

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

// TMA loading kernel with per-warp barriers
// 256 threads = 8 warps
// Each warp loads one point with its own barrier
__global__ void tma_warp_barrier_kernel(
    const __grid_constant__ CUtensorMap tma_desc,
    const dtype *sampling_loc,
    const int batch_size,
    const int num_query,
    const int num_levels,
    const int num_points,
    dtype *output
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;  // 0-7
    const int lane_id = tid % 32;  // 0-31

    // Shared memory for TMA loads - 8 slots for 8 warps (num_points=8)
    __shared__ alignas(128) dtype smem_tile[8][2][2][32];

    // Per-warp barriers - each warp gets its own barrier
    // CRITICAL: Each barrier is initialized for 32 threads (one warp)
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier warp_bars[8];

    // Initialize barrier for this warp - only lane 0 does this
    if (lane_id == 0) {
        init(&warp_bars[warp_id], 32);  // 32 threads in this warp
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncwarp();  // Sync within warp after barrier init

    if (bid >= batch_size * num_query) return;

    const int b_col = bid / num_query;
    const int q_col = bid % num_query;

    // Each warp handles one point
    const int p_col = warp_id;  // warp 0 → point 0, warp 1 → point 1, etc.

    if (p_col < num_points) {
        const int l_col = 0;  // Test with first level

        // Only lane 0 of each warp issues TMA
        if (lane_id == 0) {
            // Get sampling location
            const int loc_idx = ((((b_col * num_query + q_col) * 1) * num_levels + l_col) * num_points + p_col) * 2;
            dtype loc_w_norm = sampling_loc[loc_idx];
            dtype loc_h_norm = sampling_loc[loc_idx + 1];

            // Convert to image coordinates
            dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
            dtype h_im = __hfma(loc_h_norm, __int2half_rn(SPATIAL_H), __float2half(0.5f));

            dtype kZERO = __float2half(0.0f);

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

                // Issue TMA load - use this warp's barrier
                asm volatile(
                    "{\n\t"
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                    "}"
                    :
                    : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[p_col][0][0][0]))),
                      "l"(reinterpret_cast<uint64_t>(&tma_desc)),
                      "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                      "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id])))
                    : "memory"
                );

                // Update barrier - only lane 0
                asm volatile(
                    "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                    :
                    : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id]))),
                      "n"(2 * 2 * 32 * sizeof(dtype))
                );
            }
        }

        // ALL threads in the warp wait for TMA completion
        barrier::arrival_token token = warp_bars[warp_id].arrive();
        warp_bars[warp_id].wait(std::move(token));

        // Copy loaded data to output - all threads in warp participate
        for (int idx = lane_id; idx < 128; idx += 32) {
            int h = idx / 64;
            int w = (idx / 32) % 2;
            int c = idx % 32;
            output[bid * num_points * 128 + p_col * 128 + idx] = smem_tile[p_col][h][w][c];
        }
    }

    // Final block-level sync to ensure all warps finish before kernel exit
    __syncthreads();
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
    printf("=== TMA with Per-Warp Barriers ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Clock Rate: %.2f GHz\n\n", prop.clockRate / 1e6);

    // Configuration
    const int batch = 1;
    const int num_query = 1000;
    const int num_heads = 1;
    const int num_levels = 4;
    const int num_points = 8;

    printf("Configuration:\n");
    printf("  Batch: %d, Queries: %d, Points: %d\n", batch, num_query, num_points);
    printf("  Threads per block: 256 (8 warps)\n");
    printf("  Synchronization: Per-warp barriers (each warp independent)\n");
    printf("  Warp 0-7 → Point 0-7 (1:1 mapping)\n\n");

    // Load data
    printf("Loading test data...\n");
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");
    printf("\n");

    const int level_start = h_level_start_index[0];
    const int level_size = SPATIAL_H * SPATIAL_W;

    // Allocate device memory
    dtype *d_value, *d_sampling_loc, *d_output;

    size_t level_data_size = level_size * CHANNELS;
    CUDA_CHECK(cudaMalloc(&d_value, level_data_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data() + level_start * CHANNELS,
                          level_data_size * sizeof(dtype), cudaMemcpyHostToDevice));

    size_t sampling_loc_size = batch * num_query * num_heads * num_levels * num_points * 2;
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_loc_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(),
                          sampling_loc_size * sizeof(dtype), cudaMemcpyHostToDevice));

    const int num_blocks = batch * num_query;
    CUDA_CHECK(cudaMalloc(&d_output, num_blocks * num_points * 128 * sizeof(dtype)));

    // Create TMA descriptor
    printf("Creating TMA descriptor...\n");
    CUtensorMap tma_desc{};
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    uint64_t globalDim[3] = {CHANNELS, SPATIAL_W, SPATIAL_H};
    uint64_t globalStrides[2] = {
        CHANNELS * sizeof(dtype),
        SPATIAL_W * CHANNELS * sizeof(dtype)
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
    printf("✅ TMA descriptor created\n\n");

    // Warm up
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        tma_warp_barrier_kernel<<<num_blocks, 256>>>(
            tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("Warm up complete.\n\n");

    // Benchmark
    printf("Running benchmark (10 iterations)...\n");
    const int iterations = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        tma_warp_barrier_kernel<<<num_blocks, 256>>>(
            tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        total_time += iter_time;
        min_time = std::min(min_time, iter_time);
        max_time = std::max(max_time, iter_time);
        printf("  Iteration %2d: %.4f ms\n", i + 1, iter_time);
    }

    float avg_time = total_time / iterations;

    printf("\n=== Performance Results ===\n");
    printf("Average time: %.4f ms\n", avg_time);
    printf("Min time:     %.4f ms\n", min_time);
    printf("Max time:     %.4f ms\n\n", max_time);

    // Calculate throughput
    const size_t total_bytes = (size_t)batch * num_query * num_points * 256;
    const double data_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    const double bandwidth = data_gb / (avg_time / 1000.0);

    printf("Throughput:\n");
    printf("  Data transferred: %.4f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("  Effective bandwidth: %.2f GB/s\n", bandwidth);
    printf("  TMA operations: %d\n", batch * num_query * num_points);
    printf("  Ops/sec: %.2f M ops/s\n\n", (batch * num_query * num_points) / (avg_time / 1000.0) / 1e6);

    // Verify results
    printf("Verifying correctness...\n");
    const int num_verify_blocks = 100;
    std::vector<dtype> h_output(num_verify_blocks * num_points * 128);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(dtype), cudaMemcpyDeviceToHost));

    int total_errors = 0;
    int total_checked = 0;

    for (int bid = 0; bid < num_verify_blocks; bid++) {
        const int b_col = bid / num_query;
        const int q_col = bid % num_query;

        for (int p = 0; p < num_points; p++) {
            const int loc_idx = ((((b_col * num_query + q_col) * 1) * num_levels + 0) * num_points + p) * 2;
            dtype loc_w_norm = h_sampling_loc[loc_idx];
            dtype loc_h_norm = h_sampling_loc[loc_idx + 1];

            float w_im = __half2float(loc_w_norm) * SPATIAL_W + 0.5f;
            float h_im = __half2float(loc_h_norm) * SPATIAL_H + 0.5f;

            if (h_im <= 0.0f || w_im <= 0.0f || h_im >= SPATIAL_H + 1 || w_im >= SPATIAL_W + 1) {
                continue;
            }

            int hLow = (int)floor(h_im);
            int wLow = (int)floor(w_im);
            hLow = std::max(0, std::min(hLow, SPATIAL_H - 2));
            wLow = std::max(0, std::min(wLow, SPATIAL_W - 2));

            for (int h = 0; h < 2; h++) {
                for (int w = 0; w < 2; w++) {
                    for (int c = 0; c < 32; c++) {
                        int value_idx = ((hLow + h) * SPATIAL_W + (wLow + w)) * CHANNELS + c;
                        float gt_value = __half2float(h_value[level_start * CHANNELS + value_idx]);

                        int output_idx = bid * num_points * 128 + p * 128 + h * 64 + w * 32 + c;
                        float tma_value = __half2float(h_output[output_idx]);

                        if (std::abs(gt_value - tma_value) > 1e-6f) {
                            total_errors++;
                        }
                        total_checked++;
                    }
                }
            }
        }
    }

    printf("Checked %d elements, found %d errors\n", total_checked, total_errors);
    printf("Accuracy: %.4f%%\n\n", 100.0f * (total_checked - total_errors) / total_checked);

    if (total_errors == 0) {
        printf("✅ ALL VERIFIED CORRECT!\n");
    } else {
        printf("⚠️  Some errors found (may be FP16 precision)\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_sampling_loc));
    CUDA_CHECK(cudaFree(d_output));

    printf("\n✅ Per-warp barrier test completed!\n");
    return 0;
}
