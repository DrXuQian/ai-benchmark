#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// Multi-Scale TMA with Stride-4 Thread Loading Pattern
// Key features:
// 1. Only tid%4==0 threads issue TMA loads (64 loader threads per block)
// 2. Each loader thread handles specific points across all levels
// 3. Supports multiple spatial scales (4 levels)
// 4. Block-wide barrier for synchronization
// 5. TMA descriptor prefetch optimization

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

// Configuration
#define NUM_LEVELS 4
#define NUM_POINTS 8
#define CHANNELS 32
#define TILE_H 2
#define TILE_W 2
#define TILE_C 32
#define THREADS_PER_BLOCK 256
#define LOADER_STRIDE 4  // Every 4th thread is a loader

// Spatial dimensions for each level
struct LevelConfig {
    int H, W;
    int64_t start_index;
};

__constant__ LevelConfig d_level_configs[NUM_LEVELS];

// Kernel with stride-4 loading pattern
__global__ void tma_stride4_loaders_kernel(
    const CUtensorMap* tma_descs_all,   // [batch][level] flattened array
    const dtype *sampling_loc,          // [batch][query][heads][levels][points][2]
    const int batch_size,
    const int num_query,
    dtype *output                       // [batch][query][levels][points][128]
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Shared memory: [queries][levels][points][tile] - for 64 queries per block
    // This would be too large! 64 * 4 * 8 * 2 * 2 * 32 * 2 bytes = 262KB per block
    // Instead, we process queries sequentially with smaller shared memory
    __shared__ alignas(128) dtype smem_tile[NUM_LEVELS][NUM_POINTS][TILE_H][TILE_W][TILE_C];

    // Single block-wide barrier for all threads
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (tid == 0) {
        init(&bar, THREADS_PER_BLOCK);  // All 256 threads participate
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    if (bid >= batch_size * num_query) return;

    const int b_col = bid / num_query;
    const int q_col = bid % num_query;

    // Thread mapping: only tid%4==0 threads are loaders
    const bool is_loader = (tid % LOADER_STRIDE == 0);
    const int loader_id = tid / LOADER_STRIDE;  // 0 to 63 for 256 threads

    // Prefetch TMA descriptors to L2 cache (only once per block)
    if (tid == 0) {
        #pragma unroll
        for (int l = 0; l < NUM_LEVELS; l++) {
            const int desc_idx = b_col * NUM_LEVELS + l;
            const CUtensorMap* desc_ptr = &tma_descs_all[desc_idx];
            asm volatile(
                "prefetch.tensormap [%0];\n\t"
                :: "l"(reinterpret_cast<uint64_t>(desc_ptr))
            );
        }
    }

    // Each loader thread (tid%4==0) processes ALL levels×points for this query
    // Every loader loads all 4 levels × 8 points = 32 tiles

    // Process all levels and points for this query
    for (int l_col = 0; l_col < NUM_LEVELS; l_col++) {
        for (int p_col = 0; p_col < NUM_POINTS; p_col++) {

            // Only loader threads (tid%4==0) issue TMA loads
            // Each loader loads ALL tiles
            if (is_loader) {

                // Get sampling location for this batch, query, level, point
                const int loc_idx = ((((b_col * num_query + q_col) * 1) * NUM_LEVELS + l_col) * NUM_POINTS + p_col) * 2;
                dtype loc_w_norm = sampling_loc[loc_idx];
                dtype loc_h_norm = sampling_loc[loc_idx + 1];

                // Get spatial dimensions for this level
                const int SPATIAL_H = d_level_configs[l_col].H;
                const int SPATIAL_W = d_level_configs[l_col].W;

                // Convert to image coordinates
                dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
                dtype h_im = __hfma(loc_h_norm, __int2half_rn(SPATIAL_H), __float2half(0.5f));

                dtype kZERO = __float2half(0.0f);

                // Check bounds
                if (h_im > kZERO && w_im > kZERO &&
                    h_im < __int2half_rn(SPATIAL_H + 1) && w_im < __int2half_rn(SPATIAL_W + 1)) {

                    int hLow = __half2int_rd(h_im);
                    int wLow = __half2int_rd(w_im);

                    // Clamp to valid range for non-padded data
                    hLow = max(0, min(hLow, SPATIAL_H - 2));
                    wLow = max(0, min(wLow, SPATIAL_W - 2));

                    // TMA coordinates
                    int32_t tensor_coord_c = 0;
                    int32_t tensor_coord_w = wLow;
                    int32_t tensor_coord_h = hLow;

                    // Get TMA descriptor for this batch and level
                    const int desc_idx = b_col * NUM_LEVELS + l_col;
                    const CUtensorMap* tma_desc = &tma_descs_all[desc_idx];

                    // Issue TMA load for this point
                    asm volatile(
                        "{\n\t"
                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                        "}"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[l_col][p_col][0][0][0]))),
                          "l"(reinterpret_cast<uint64_t>(tma_desc)),
                          "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                          "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                        : "memory"
                    );

                    // Expect data for this tile
                    asm volatile(
                        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                          "n"(TILE_H * TILE_W * TILE_C * sizeof(dtype))
                    );
                } else {
                    // Out of bounds - expect 0 bytes
                    asm volatile(
                        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                          "n"(0)
                    );
                }
            }
        }
    }

    // All threads wait for all loads to complete
    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

    // All threads participate in copying data to output
    // Each thread handles a portion of the data to maximize memory bandwidth
    for (int l_col = 0; l_col < NUM_LEVELS; l_col++) {
        for (int p_col = 0; p_col < NUM_POINTS; p_col++) {
            // Each thread copies elements with stride
            for (int idx = tid; idx < 128; idx += THREADS_PER_BLOCK) {
                int h = idx / 64;
                int w = (idx / 32) % 2;
                int c = idx % 32;

                // Calculate output index for this batch and query
                int out_idx = ((((b_col * num_query + q_col) * NUM_LEVELS + l_col) * NUM_POINTS + p_col) * 128) + idx;
                output[out_idx] = smem_tile[l_col][p_col][h][w][c];
            }
        }
    }

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
    printf("=== Multi-Scale TMA with Stride-4 Thread Loading Pattern ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Bandwidth: %.2f GB/s\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    // Configuration
    const int batch = 48;
    const int num_query = 1000;

    printf("Configuration:\n");
    printf("  Batches: %d\n", batch);
    printf("  Queries per batch: %d\n", num_query);
    printf("  Points per query: %d\n", NUM_POINTS);
    printf("  Levels: %d\n", NUM_LEVELS);
    printf("  Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("  Loading pattern: tid%%4==0 (stride-4)\n");
    printf("  Loader threads: %d per block\n", THREADS_PER_BLOCK / LOADER_STRIDE);
    printf("  Active loaders: %d (for %d points)\n", NUM_POINTS, NUM_POINTS);
    printf("  Total blocks: %d\n", batch * num_query);
    printf("  Total TMA operations: %d\n\n", batch * num_query * NUM_LEVELS * NUM_POINTS);

    // Load spatial shapes and level indices
    auto h_spatial_shapes = load_binary<int64_t>("working/test_data_spatial_shapes.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");

    // Setup level configurations
    LevelConfig h_level_configs[NUM_LEVELS];
    printf("Spatial Scales:\n");
    for (int i = 0; i < NUM_LEVELS; i++) {
        h_level_configs[i].H = h_spatial_shapes[i * 2];
        h_level_configs[i].W = h_spatial_shapes[i * 2 + 1];
        h_level_configs[i].start_index = h_level_start_index[i];
        printf("  Level %d: [%d×%d], start_idx=%ld\n",
               i, h_level_configs[i].H, h_level_configs[i].W,
               h_level_configs[i].start_index);
    }
    printf("\n");

    CUDA_CHECK(cudaMemcpyToSymbol(d_level_configs, h_level_configs, sizeof(LevelConfig) * NUM_LEVELS));

    // Load data
    printf("Loading test data...\n");
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");
    printf("  Value data: %zu elements (%.2f MB)\n", h_value.size(),
           h_value.size() * sizeof(dtype) / (1024.0 * 1024.0));
    printf("  Sampling locations: %zu elements (%.2f MB)\n\n", h_sampling_loc.size(),
           h_sampling_loc.size() * sizeof(dtype) / (1024.0 * 1024.0));

    // Allocate device memory for each batch×level combination
    dtype *d_value_levels[batch][NUM_LEVELS];
    printf("Creating TMA descriptors for all batch×level combinations...\n");
    printf("  Total descriptors: %d × %d = %d\n", batch, NUM_LEVELS, batch * NUM_LEVELS);

    // Calculate single batch size
    size_t single_batch_total_size = 0;
    for (int l = 0; l < NUM_LEVELS; l++) {
        single_batch_total_size += h_level_configs[l].H * h_level_configs[l].W * CHANNELS;
    }

    // Host-side TMA descriptors
    std::vector<CUtensorMap> h_tma_descs(batch * NUM_LEVELS);
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    size_t total_value_memory = 0;
    for (int b = 0; b < batch; b++) {
        for (int l = 0; l < NUM_LEVELS; l++) {
            size_t level_size = h_level_configs[l].H * h_level_configs[l].W * CHANNELS;
            total_value_memory += level_size;

            // Calculate offset in original data
            size_t batch_offset = b * single_batch_total_size;
            size_t level_offset = h_level_configs[l].start_index * CHANNELS;
            size_t start_offset = batch_offset + level_offset;

            // Allocate device memory for this batch×level
            CUDA_CHECK(cudaMalloc(&d_value_levels[b][l], level_size * sizeof(dtype)));

            // Copy this batch's level data
            CUDA_CHECK(cudaMemcpy(d_value_levels[b][l],
                                  h_value.data() + start_offset,
                                  level_size * sizeof(dtype),
                                  cudaMemcpyHostToDevice));

            // Create TMA descriptor
            uint64_t globalDim[3] = {CHANNELS, (uint64_t)h_level_configs[l].W, (uint64_t)h_level_configs[l].H};
            uint64_t globalStrides[2] = {
                CHANNELS * sizeof(dtype),
                h_level_configs[l].W * CHANNELS * sizeof(dtype)
            };
            uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
            uint32_t elementStrides[3] = {1, 1, 1};

            CUresult res = cuTensorMapEncodeTiled_func(
                &h_tma_descs[b * NUM_LEVELS + l],
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                3,
                d_value_levels[b][l],
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
                printf("Failed to create TMA descriptor for batch %d, level %d\n", b, l);
                return 1;
            }
        }

        if ((b + 1) % 10 == 0) {
            printf("  ✓ Created descriptors for batches 0-%d\n", b);
        }
    }
    printf("  Total value data: %.2f MB (%.2f MB per batch)\n\n",
           total_value_memory * sizeof(dtype) / (1024.0 * 1024.0),
           total_value_memory * sizeof(dtype) / (1024.0 * 1024.0) / batch);

    // Copy TMA descriptors to device
    CUtensorMap *d_tma_descs;
    CUDA_CHECK(cudaMalloc(&d_tma_descs, batch * NUM_LEVELS * sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tma_descs, h_tma_descs.data(),
                          batch * NUM_LEVELS * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    printf("  ✓ Copied all %d descriptors to device\n\n", batch * NUM_LEVELS);

    // Allocate sampling locations and output
    dtype *d_sampling_loc, *d_output;
    size_t sampling_size = batch * num_query * 1 * NUM_LEVELS * NUM_POINTS * 2;
    size_t output_size = batch * num_query * NUM_LEVELS * NUM_POINTS * 128;

    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_size * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemset(d_output, 0, output_size * sizeof(dtype)));

    // For simplicity, replicate first batch's sampling locations
    printf("Replicating sampling locations across %d batches...\n", batch);
    size_t batch_sampling_size = num_query * 1 * NUM_LEVELS * NUM_POINTS * 2;
    for (int b = 0; b < batch; b++) {
        CUDA_CHECK(cudaMemcpy(d_sampling_loc + b * batch_sampling_size,
                              h_sampling_loc.data(),
                              batch_sampling_size * sizeof(dtype),
                              cudaMemcpyHostToDevice));
    }
    printf("  ✓ Total sampling data: %.2f MB\n\n",
           sampling_size * sizeof(dtype) / (1024.0 * 1024.0));

    printf("Output buffer: %.2f MB\n\n",
           output_size * sizeof(dtype) / (1024.0 * 1024.0));

    // Launch kernel
    const int num_blocks = batch * num_query;
    const int threads_per_block = THREADS_PER_BLOCK;

    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        tma_stride4_loaders_kernel<<<num_blocks, threads_per_block>>>(
            d_tma_descs, d_sampling_loc, batch, num_query, d_output);
        cudaDeviceSynchronize();
    }
    printf("Warmup complete.\n\n");

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    printf("Running benchmark (%d iterations)...\n", iterations);
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        tma_stride4_loaders_kernel<<<num_blocks, threads_per_block>>>(
            d_tma_descs, d_sampling_loc, batch, num_query, d_output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        total_time += iter_time;
        min_time = fmin(min_time, iter_time);
        max_time = fmax(max_time, iter_time);
        printf("  Iteration %2d: %.4f ms\n", i + 1, iter_time);
    }

    float avg_time = total_time / iterations;

    printf("\n=== Performance Results ===\n");
    printf("Timing:\n");
    printf("  Average: %.4f ms\n", avg_time);
    printf("  Min: %.4f ms\n", min_time);
    printf("  Max: %.4f ms\n\n", max_time);

    printf("Throughput:\n");
    printf("  Queries/ms: %.2f\n", (batch * num_query) / avg_time);
    printf("  TMA ops/ms: %.2f K\n", (batch * num_query * NUM_LEVELS * NUM_POINTS) / avg_time / 1000);
    printf("  Effective bandwidth: %.2f GB/s\n\n",
           (batch * num_query * NUM_LEVELS * NUM_POINTS * 128 * sizeof(dtype)) / (avg_time * 1e6));

    printf("Efficiency:\n");
    printf("  Time per query: %.3f μs\n", avg_time * 1000 / (batch * num_query));
    printf("  Time per TMA operation: %.3f ns\n",
           avg_time * 1e6 / (batch * num_query * NUM_LEVELS * NUM_POINTS));

    printf("\nThread utilization:\n");
    printf("  Total threads: %d\n", THREADS_PER_BLOCK);
    printf("  Loader threads: %d (tid%%4==0)\n", THREADS_PER_BLOCK / LOADER_STRIDE);
    printf("  Active loaders: %d (handling %d points)\n", NUM_POINTS, NUM_POINTS);
    printf("  Idle loaders: %d\n", (THREADS_PER_BLOCK / LOADER_STRIDE) - NUM_POINTS);

    // Cleanup
    for (int b = 0; b < batch; b++) {
        for (int l = 0; l < NUM_LEVELS; l++) {
            CUDA_CHECK(cudaFree(d_value_levels[b][l]));
        }
    }
    CUDA_CHECK(cudaFree(d_tma_descs));
    CUDA_CHECK(cudaFree(d_sampling_loc));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n✅ Multi-scale stride-4 loader test completed!\n");
    return 0;
}