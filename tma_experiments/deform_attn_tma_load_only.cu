#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// Deformable Attention TMA Data Loading Only
// Based on deform_attn.cu - only TMA loading, no bilinear interpolation
//
// Key differences from previous attempt:
// 1. Use h_level_start_index to get each level's start offset in value buffer
// 2. Actual spatial shape is (H+2) × (W+2) due to padding for boundary handling
// 3. Only load data with TMA, output the loaded tiles (no compute)

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

// Spatial dimensions for each level
struct LevelConfig {
    int H, W;                // Original spatial dimensions (without padding)
    int H_padded, W_padded;  // Padded dimensions: (H+2) × (W+2)
    int64_t start_index;     // Start index in value buffer
};

__constant__ LevelConfig d_level_configs[NUM_LEVELS];

// TMA loading kernel - single stage blocking mode
__global__ void deform_attn_tma_load_kernel(
    const CUtensorMap* tma_desc_level0,
    const CUtensorMap* tma_desc_level1,
    const CUtensorMap* tma_desc_level2,
    const CUtensorMap* tma_desc_level3,
    const dtype *sampling_loc,      // [batch][query][1][levels][points][2]
    const int batch_size,
    const int num_query,
    dtype *output                   // [batch][query][levels][points][2][2][32]
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // DEBUG: Print first thread info
    if (bid == 0 && tid == 0) {
        printf("Kernel started: bid=%d, tid=%d, warp_id=%d, lane_id=%d\n",
               bid, tid, warp_id, lane_id);
    }

    // Shared memory for TMA tiles: [levels][points][2][2][32]
    __shared__ alignas(128) dtype smem_tile[NUM_LEVELS][NUM_POINTS][TILE_H][TILE_W][TILE_C];

    // Per-warp barriers
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier warp_bars[8];

    if (lane_id == 0) {
        init(&warp_bars[warp_id], 32);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncwarp();

    if (bid >= batch_size * num_query) return;

    const int b_col = bid / num_query;
    const int q_col = bid % num_query;
    const int p_col = warp_id;

    const dtype kZERO = __float2half(0.0f);

    if (p_col < NUM_POINTS) {
        // Process all 4 levels for this sampling point
        for (int l_col = 0; l_col < NUM_LEVELS; l_col++) {
            // === Issue TMA Load ===
            bool tile_valid = false;

            if (lane_id == 0) {
                // Get sampling location for this query, level, point
                // Data format: [batch=48][num_query][num_heads=1][num_levels][num_points][2]
                // We only use batch 0, so: (q_col * 1 * 4 * 8 + l_col * 8 + p_col) * 2
                const int loc_idx = (((q_col * 1) * NUM_LEVELS + l_col) * NUM_POINTS + p_col) * 2;
                dtype loc_w_norm = sampling_loc[loc_idx];      // normalized width
                dtype loc_h_norm = sampling_loc[loc_idx + 1];  // normalized height

                // Get spatial dimensions (original, without padding)
                const int SPATIAL_H = d_level_configs[l_col].H;
                const int SPATIAL_W = d_level_configs[l_col].W;

                // Convert normalized coordinates to image coordinates
                // Note: deform_attn.cu adds 0.5 offset
                dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
                dtype h_im = __hfma(loc_h_norm, __int2half_rn(SPATIAL_H), __float2half(0.5f));

                // Check bounds (same as original: h_im > 0 && h_im < SPATIAL_H + 1)
                if (h_im > kZERO && w_im > kZERO &&
                    h_im < __int2half_rn(SPATIAL_H + 1) && w_im < __int2half_rn(SPATIAL_W + 1)) {

                    tile_valid = true;

                    // Get integer coordinates for bilinear interpolation
                    int hLow = __half2int_rd(h_im);
                    int wLow = __half2int_rd(w_im);

                    // NO CLAMPING NEEDED!
                    // Tensor in memory is (H+2) × (W+2) × C with padding
                    // hLow/wLow are direct indices into this padded array
                    // Valid range: hLow ∈ [0, SPATIAL_H], wLow ∈ [0, SPATIAL_W]
                    // This allows loading 2×2 tiles that include padding when needed
                    // (Already guaranteed by bounds check above)

                    // TMA coordinates: X=C, Y=W, Z=H
                    // No need to add +1 offset because padding is built into tensor layout
                    int32_t tensor_coord_c = 0;
                    int32_t tensor_coord_w = wLow;
                    int32_t tensor_coord_h = hLow;

                    // Select TMA descriptor for this level
                    const CUtensorMap* tma_desc;
                    if (l_col == 0) tma_desc = tma_desc_level0;
                    else if (l_col == 1) tma_desc = tma_desc_level1;
                    else if (l_col == 2) tma_desc = tma_desc_level2;
                    else tma_desc = tma_desc_level3;

                    // Issue TMA load
                    asm volatile(
                        "{\n\t"
                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                        "}"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[l_col][p_col][0][0][0]))),
                          "l"(reinterpret_cast<uint64_t>(tma_desc)),
                          "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                          "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id])))
                        : "memory"
                    );

                    asm volatile(
                        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id]))),
                          "n"(TILE_H * TILE_W * TILE_C * sizeof(dtype))
                    );
                } else {
                    // Out of bounds - still need to signal barrier with 0 bytes
                    asm volatile(
                        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id]))),
                          "n"(0)  // 0 bytes expected for invalid tile
                    );
                }
            }

            // === Block Wait for TMA Completion ===
            barrier::arrival_token token = warp_bars[warp_id].arrive();
            warp_bars[warp_id].wait(std::move(token));

            // === Copy loaded tile to output ===
            // Share tile_valid flag across warp
            tile_valid = __shfl_sync(0xFFFFFFFF, tile_valid, 0);

            if (tile_valid) {
                // Output layout: [batch][query][levels][points][2][2][32]
                for (int idx = lane_id; idx < 128; idx += 32) {
                    int h = idx / 64;
                    int w = (idx / 32) % 2;
                    int c = idx % 32;
                    int out_idx = (((((bid * NUM_LEVELS + l_col) * NUM_POINTS + p_col) * 2 + h) * 2 + w) * CHANNELS) + c;
                    output[out_idx] = smem_tile[l_col][p_col][h][w][c];
                }
            } else {
                // Zero out output for invalid tiles
                for (int idx = lane_id; idx < 128; idx += 32) {
                    int h = idx / 64;
                    int w = (idx / 32) % 2;
                    int c = idx % 32;
                    int out_idx = (((((bid * NUM_LEVELS + l_col) * NUM_POINTS + p_col) * 2 + h) * 2 + w) * CHANNELS) + c;
                    output[out_idx] = __float2half(0.0f);
                }
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
    printf("=== Deformable Attention TMA Loading Only ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Configuration
    const int batch = 1;
    const int num_query = 123376;  // Real number of queries from test data

    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch);
    printf("  Queries: %d\n", num_query);
    printf("  Points per query: %d\n", NUM_POINTS);
    printf("  Levels: %d\n\n", NUM_LEVELS);

    // Load spatial shapes and level indices
    auto h_spatial_shapes = load_binary<int64_t>("working/test_data_spatial_shapes.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");

    // Setup level configurations with padding
    LevelConfig h_level_configs[NUM_LEVELS];
    printf("Spatial Scales (with padding):\n");
    for (int i = 0; i < NUM_LEVELS; i++) {
        h_level_configs[i].H = h_spatial_shapes[i * 2];
        h_level_configs[i].W = h_spatial_shapes[i * 2 + 1];
        h_level_configs[i].H_padded = h_level_configs[i].H + 2;  // Add padding
        h_level_configs[i].W_padded = h_level_configs[i].W + 2;  // Add padding
        h_level_configs[i].start_index = h_level_start_index[i];
        printf("  Level %d: [%d×%d] → [%d×%d] (padded), start_idx=%ld\n",
               i, h_level_configs[i].H, h_level_configs[i].W,
               h_level_configs[i].H_padded, h_level_configs[i].W_padded,
               h_level_configs[i].start_index);
    }
    printf("\n");

    CUDA_CHECK(cudaMemcpyToSymbol(d_level_configs, h_level_configs, sizeof(LevelConfig) * NUM_LEVELS));

    // Load data
    printf("Loading test data...\n");
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");
    printf("  Value data: %zu elements (%.2f MB)\n", h_value.size(), h_value.size() * sizeof(dtype) / (1024.0 * 1024.0));
    printf("  Sampling locations: %zu elements\n\n", h_sampling_loc.size());

    // Allocate device memory for BATCH 0 ONLY with PADDED dimensions
    // Keep the padded data format as-is
    dtype *d_value;
    size_t spatial_size = 20522;  // Total spatial positions per batch (with padding)
    size_t batch0_size = spatial_size * CHANNELS;
    CUDA_CHECK(cudaMalloc(&d_value, batch0_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), batch0_size * sizeof(dtype), cudaMemcpyHostToDevice));
    printf("Allocated batch 0 data: %zu elements (%.2f MB)\n",
           batch0_size, batch0_size * sizeof(dtype) / (1024.0 * 1024.0));

    // Create TMA descriptors for each level
    CUtensorMap tma_descs[NUM_LEVELS];
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    printf("\nCreating TMA descriptors with padded dimensions...\n");
    for (int l = 0; l < NUM_LEVELS; l++) {
        // Pointer to start of this level in padded data
        size_t offset = h_level_configs[l].start_index * CHANNELS;
        dtype* level_ptr = d_value + offset;

        printf("  Level %d: offset=%zu, H=%d (padded=%d), W=%d (padded=%d)\n",
               l, offset, h_level_configs[l].H, h_level_configs[l].H_padded,
               h_level_configs[l].W, h_level_configs[l].W_padded);

        // TMA descriptor with PADDED dimensions: (H+2) × (W+2) × C
        // The data in memory has this layout
        uint64_t globalDim[3] = {
            CHANNELS,
            (uint64_t)h_level_configs[l].W_padded,  // W+2
            (uint64_t)h_level_configs[l].H_padded   // H+2
        };
        uint64_t globalStrides[2] = {
            CHANNELS * sizeof(dtype),
            h_level_configs[l].W_padded * CHANNELS * sizeof(dtype)  // (W+2) * C
        };
        uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
        uint32_t elementStrides[3] = {1, 1, 1};

        CUresult res = cuTensorMapEncodeTiled_func(
            &tma_descs[l],
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3,
            level_ptr,
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
            printf("Failed to create TMA descriptor for level %d\n", l);
            return 1;
        }
        printf("  ✓ Created TMA descriptor for padded tensor [%d×%d×%d]\n",
               h_level_configs[l].H_padded, h_level_configs[l].W_padded, CHANNELS);
    }
    printf("\n");

    // Copy TMA descriptors to device
    CUtensorMap *d_tma_descs;
    CUDA_CHECK(cudaMalloc(&d_tma_descs, NUM_LEVELS * sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tma_descs, tma_descs, NUM_LEVELS * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    printf("✓ Copied TMA descriptors to device\n\n");

    // Allocate sampling locations and output
    dtype *d_sampling_loc, *d_output;
    size_t sampling_size = batch * num_query * 1 * NUM_LEVELS * NUM_POINTS * 2;
    size_t output_size = batch * num_query * NUM_LEVELS * NUM_POINTS * TILE_H * TILE_W * TILE_C;

    CUDA_CHECK(cudaMalloc(&d_sampling_loc, sampling_size * sizeof(dtype)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemset(d_output, 0, output_size * sizeof(dtype)));  // Initialize output

    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(),
                          sampling_size * sizeof(dtype), cudaMemcpyHostToDevice));

    // Launch kernel
    const int num_blocks = batch * num_query;
    const int threads_per_block = 256;

    printf("Launching kernel...\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads/block: %d\n\n", threads_per_block);

    // Warmup
    for (int i = 0; i < 3; i++) {
        deform_attn_tma_load_kernel<<<num_blocks, threads_per_block>>>(
            &d_tma_descs[0], &d_tma_descs[1], &d_tma_descs[2], &d_tma_descs[3],
            d_sampling_loc, batch, num_query, d_output);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel failed on warmup iteration %d: %s\n", i, cudaGetErrorString(err));
            return 1;
        }
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        deform_attn_tma_load_kernel<<<num_blocks, threads_per_block>>>(
            &d_tma_descs[0], &d_tma_descs[1], &d_tma_descs[2], &d_tma_descs[3],
            d_sampling_loc, batch, num_query, d_output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        total_time += iter_time;
        printf("  Iteration %2d: %.4f ms\n", i + 1, iter_time);
    }

    float avg_time = total_time / iterations;

    printf("\n=== Performance Results ===\n");
    printf("  Average time: %.4f ms\n", avg_time);
    printf("  Throughput: %.2f queries/ms\n", num_query / avg_time);
    printf("  TMA operations: %d (4 levels × 8 points × %d queries)\n",
           NUM_LEVELS * NUM_POINTS * num_query, num_query);

    // Verification (compare first few samples)
    printf("\n=== Verification (first query, level 0, point 0) ===\n");
    std::vector<dtype> h_output(128);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 128 * sizeof(dtype), cudaMemcpyDeviceToHost));

    printf("First 8 values:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d]: %.4f\n", i, __half2float(h_output[i]));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_sampling_loc));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_tma_descs));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n✅ TMA loading test completed!\n");
    return 0;
}
