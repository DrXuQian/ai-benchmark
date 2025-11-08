#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// TMA Loading Benchmark - measures throughput and latency

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

// TMA loading kernel with timing
__global__ void tma_benchmark_kernel(
    const __grid_constant__ CUtensorMap tma_desc,
    const dtype *sampling_loc,
    const int batch_size,
    const int num_query,
    const int num_levels,
    const int num_points,
    dtype *output,
    unsigned long long *timing_output  // Store per-block timing
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Start timing
    unsigned long long start_time = 0;
    if (tid == 0) {
        start_time = clock64();
    }

    // Shared memory for TMA loads
    __shared__ alignas(128) dtype smem_tile[8][2][2][32];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    const int num_tma_threads = blockDim.x / 4;

    if (tid == 0) {
        init(&bar, blockDim.x);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    if (bid >= batch_size * num_query) return;

    const int b_col = bid / num_query;
    const int q_col = bid % num_query;

    const bool is_loader = (tid % 4 == 0);
    const int loader_id = tid / 4;

    const int l_col = 0;  // First level
    const int p_col = loader_id % num_points;

    if (is_loader && loader_id < num_points) {
        const int loc_idx = ((((b_col * num_query + q_col) * 1) * num_levels + l_col) * num_points + p_col) * 2;
        dtype loc_w_norm = sampling_loc[loc_idx];
        dtype loc_h_norm = sampling_loc[loc_idx + 1];

        dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
        dtype h_im = __hfma(loc_h_norm, __int2half_rn(SPATIAL_H), __float2half(0.5f));

        dtype kZERO = __float2half(0.0f);

        if (h_im > kZERO && w_im > kZERO &&
            h_im < __int2half_rn(SPATIAL_H + 1) && w_im < __int2half_rn(SPATIAL_W + 1)) {

            int hLow = __half2int_rd(h_im);
            int wLow = __half2int_rd(w_im);

            hLow = max(0, min(hLow, SPATIAL_H - 2));
            wLow = max(0, min(wLow, SPATIAL_W - 2));

            int32_t tensor_coord_c = 0;
            int32_t tensor_coord_w = wLow;
            int32_t tensor_coord_h = hLow;

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

            asm volatile(
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                :
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                  "n"(2 * 2 * 32 * sizeof(dtype))
            );
        }
    }

    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

    // End timing after barrier
    if (tid == 0) {
        unsigned long long end_time = clock64();
        timing_output[bid] = end_time - start_time;
    }

    // Copy loaded data to output
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
    return data;
}

int main() {
    printf("=== TMA Loading Performance Benchmark ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Bandwidth: %.2f GB/s\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    // Configuration
    const int batch = 1;
    const int num_query = 1000;  // 1000 queries for meaningful benchmark
    const int num_heads = 1;
    const int num_levels = 4;
    const int num_points = 8;

    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch);
    printf("  Num queries: %d\n", num_query);
    printf("  Num points per query: %d\n", num_points);
    printf("  Total TMA loads per kernel: %d blocks × %d points = %d loads\n",
           batch * num_query, num_points, batch * num_query * num_points);
    printf("  Data per TMA load: %d bytes (2×2 tile, 32 channels, FP16)\n", LOAD_SIZE);
    printf("  Total data loaded: %.2f MB\n\n",
           (double)(batch * num_query * num_points * LOAD_SIZE) / (1024 * 1024));

    // Load data
    printf("Loading test data...\n");
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");
    printf("  Loaded %zu values, %zu locations\n\n", h_value.size(), h_sampling_loc.size());

    const int level_start = h_level_start_index[0];
    const int level_size = SPATIAL_H * SPATIAL_W;

    // Allocate device memory
    dtype *d_value, *d_sampling_loc, *d_output;
    unsigned long long *d_timing;

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
    CUDA_CHECK(cudaMalloc(&d_timing, num_blocks * sizeof(unsigned long long)));

    // Create TMA descriptor
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

    // Warm up
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        tma_benchmark_kernel<<<num_blocks, 256>>>(
            tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("Warm up complete.\n\n");

    // Benchmark
    printf("Running benchmark (10 iterations)...\n");
    const int num_iterations = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_time_ms = 0.0f;
    float min_time_ms = 1e9f;
    float max_time_ms = 0.0f;

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaEventRecord(start));

        tma_benchmark_kernel<<<num_blocks, 256>>>(
            tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output, d_timing);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float iter_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time_ms, start, stop));

        total_time_ms += iter_time_ms;
        min_time_ms = std::min(min_time_ms, iter_time_ms);
        max_time_ms = std::max(max_time_ms, iter_time_ms);

        printf("  Iteration %2d: %.4f ms\n", iter + 1, iter_time_ms);
    }

    float avg_time_ms = total_time_ms / num_iterations;

    printf("\n=== Performance Results ===\n");
    printf("Timing Statistics:\n");
    printf("  Average time: %.4f ms\n", avg_time_ms);
    printf("  Min time:     %.4f ms\n", min_time_ms);
    printf("  Max time:     %.4f ms\n", max_time_ms);
    printf("  Std dev:      %.4f ms\n", max_time_ms - min_time_ms);
    printf("\n");

    // Calculate throughput
    const size_t total_data_bytes = (size_t)batch * num_query * num_points * LOAD_SIZE;
    const double data_gb = total_data_bytes / (1024.0 * 1024.0 * 1024.0);
    const double throughput_gbps = data_gb / (avg_time_ms / 1000.0);
    const int total_tma_ops = batch * num_query * num_points;
    const double tma_ops_per_sec = total_tma_ops / (avg_time_ms / 1000.0);

    printf("Throughput Metrics:\n");
    printf("  Data transferred: %.4f MB\n", total_data_bytes / (1024.0 * 1024.0));
    printf("  Effective bandwidth: %.2f GB/s\n", throughput_gbps);
    printf("  TMA operations: %d\n", total_tma_ops);
    printf("  TMA ops/sec: %.2f M ops/s\n", tma_ops_per_sec / 1e6);
    printf("  Time per TMA load: %.2f ns\n", (avg_time_ms * 1e6) / total_tma_ops);
    printf("\n");

    // Analyze per-block timing
    std::vector<unsigned long long> h_timing(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_timing.data(), d_timing, num_blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long min_cycles = 1e18;
    unsigned long long max_cycles = 0;
    unsigned long long total_cycles = 0;

    for (int i = 0; i < num_blocks; i++) {
        min_cycles = std::min(min_cycles, h_timing[i]);
        max_cycles = std::max(max_cycles, h_timing[i]);
        total_cycles += h_timing[i];
    }

    double avg_cycles = (double)total_cycles / num_blocks;
    double cycles_to_us = 1.0 / (prop.clockRate / 1000.0);

    printf("Per-Block Timing (from clock64()):\n");
    printf("  Average cycles: %.0f (%.2f μs)\n", avg_cycles, avg_cycles * cycles_to_us);
    printf("  Min cycles:     %llu (%.2f μs)\n", min_cycles, min_cycles * cycles_to_us);
    printf("  Max cycles:     %llu (%.2f μs)\n", max_cycles, max_cycles * cycles_to_us);
    printf("  Cycle range:    %llu cycles\n", max_cycles - min_cycles);
    printf("\n");

    // Efficiency analysis
    const double theoretical_bw_gbps = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    const double efficiency = (throughput_gbps / theoretical_bw_gbps) * 100.0;

    printf("Efficiency Analysis:\n");
    printf("  Theoretical peak bandwidth: %.2f GB/s\n", theoretical_bw_gbps);
    printf("  Achieved bandwidth: %.2f GB/s\n", throughput_gbps);
    printf("  Memory efficiency: %.2f%%\n", efficiency);
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads per block: 256\n");
    printf("  TMA threads per block: 64 (threadIdx%%4==0)\n");
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_sampling_loc));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_timing));

    printf("✅ Benchmark completed successfully!\n");
    return 0;
}
