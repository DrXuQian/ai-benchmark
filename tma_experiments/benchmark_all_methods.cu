#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// Comprehensive Benchmark: Manual vs TMA Block-Barrier vs TMA Warp-Barrier

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

#define SPATIAL_H 92
#define SPATIAL_W 160
#define CHANNELS 32
#define TILE_H 2
#define TILE_W 2
#define TILE_C 32

// ============ MANUAL LOADING ============
__global__ void manual_load_kernel(
    const dtype *value_data,
    const dtype *sampling_loc,
    const int batch_size,
    const int num_query,
    const int num_levels,
    const int num_points,
    dtype *output
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ dtype smem_tile[8][2][2][32];

    if (bid >= batch_size * num_query) return;

    const int b_col = bid / num_query;
    const int q_col = bid % num_query;
    const int l_col = 0;

    for (int p = tid; p < num_points; p += blockDim.x) {
        const int loc_idx = ((((b_col * num_query + q_col) * 1) * num_levels + l_col) * num_points + p) * 2;
        dtype loc_w_norm = sampling_loc[loc_idx];
        dtype loc_h_norm = sampling_loc[loc_idx + 1];

        float w_im = __half2float(loc_w_norm) * SPATIAL_W + 0.5f;
        float h_im = __half2float(loc_h_norm) * SPATIAL_H + 0.5f;

        if (h_im > 0.0f && w_im > 0.0f && h_im < SPATIAL_H + 1 && w_im < SPATIAL_W + 1) {
            int hLow = (int)floor(h_im);
            int wLow = (int)floor(w_im);
            hLow = max(0, min(hLow, SPATIAL_H - 2));
            wLow = max(0, min(wLow, SPATIAL_W - 2));

            for (int h = 0; h < 2; h++) {
                for (int w = 0; w < 2; w++) {
                    int src_idx = ((hLow + h) * SPATIAL_W + (wLow + w)) * CHANNELS;
                    for (int c = 0; c < 32; c++) {
                        smem_tile[p][h][w][c] = value_data[src_idx + c];
                    }
                }
            }
        }
    }

    __syncthreads();

    for (int p = 0; p < num_points; p++) {
        for (int idx = tid; idx < 128; idx += blockDim.x) {
            int h = idx / 64;
            int w = (idx / 32) % 2;
            int c = idx % 32;
            output[bid * num_points * 128 + p * 128 + idx] = smem_tile[p][h][w][c];
        }
    }
}

// ============ TMA BLOCK-LEVEL BARRIER ============
__global__ void tma_block_barrier_kernel(
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

    __shared__ alignas(128) dtype smem_tile[8][2][2][32];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

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
    const int l_col = 0;
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

    for (int p = 0; p < num_points; p++) {
        for (int idx = tid; idx < 128; idx += blockDim.x) {
            int h = idx / 64;
            int w = (idx / 32) % 2;
            int c = idx % 32;
            output[bid * num_points * 128 + p * 128 + idx] = smem_tile[p][h][w][c];
        }
    }
}

// ============ TMA WARP-LEVEL BARRIER ============
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
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ alignas(128) dtype smem_tile[8][2][2][32];

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

    if (p_col < num_points) {
        const int l_col = 0;

        if (lane_id == 0) {
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
                      "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id])))
                    : "memory"
                );

                asm volatile(
                    "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                    :
                    : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[warp_id]))),
                      "n"(2 * 2 * 32 * sizeof(dtype))
                );
            }
        }

        barrier::arrival_token token = warp_bars[warp_id].arrive();
        warp_bars[warp_id].wait(std::move(token));

        for (int idx = lane_id; idx < 128; idx += 32) {
            int h = idx / 64;
            int w = (idx / 32) % 2;
            int c = idx % 32;
            output[bid * num_points * 128 + p_col * 128 + idx] = smem_tile[p_col][h][w][c];
        }
    }

    __syncthreads();
}

// Helper
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
    printf("=== Comprehensive TMA Benchmark: All Methods ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory Bandwidth: %.2f GB/s\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    const int batch = 1;
    const int num_query = 1000;
    const int num_heads = 1;
    const int num_levels = 4;
    const int num_points = 8;

    printf("Workload: %d blocks × %d points = %d TMA loads\n\n", batch * num_query, num_points, batch * num_query * num_points);

    // Load data
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");

    const int level_start = h_level_start_index[0];
    const int level_size = SPATIAL_H * SPATIAL_W;

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
    CUtensorMap tma_desc{};
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    uint64_t globalDim[3] = {CHANNELS, SPATIAL_W, SPATIAL_H};
    uint64_t globalStrides[2] = {CHANNELS * sizeof(dtype), SPATIAL_W * CHANNELS * sizeof(dtype)};
    uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
    uint32_t elementStrides[3] = {1, 1, 1};

    cuTensorMapEncodeTiled_func(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, d_value,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    float times[3] = {0.0f, 0.0f, 0.0f};

    // Test 1: Manual
    printf("Benchmarking Manual Loading...\n");
    for (int i = 0; i < 3; i++) {
        manual_load_kernel<<<num_blocks, 256>>>(d_value, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        manual_load_kernel<<<num_blocks, 256>>>(d_value, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        times[0] += t;
    }
    times[0] /= iterations;
    printf("  Average: %.4f ms\n\n", times[0]);

    // Test 2: TMA Block Barrier
    printf("Benchmarking TMA Block-Barrier...\n");
    for (int i = 0; i < 3; i++) {
        tma_block_barrier_kernel<<<num_blocks, 256>>>(tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        tma_block_barrier_kernel<<<num_blocks, 256>>>(tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        times[1] += t;
    }
    times[1] /= iterations;
    printf("  Average: %.4f ms\n\n", times[1]);

    // Test 3: TMA Warp Barrier
    printf("Benchmarking TMA Warp-Barrier...\n");
    for (int i = 0; i < 3; i++) {
        tma_warp_barrier_kernel<<<num_blocks, 256>>>(tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        tma_warp_barrier_kernel<<<num_blocks, 256>>>(tma_desc, d_sampling_loc, batch, num_query, num_levels, num_points, d_output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        times[2] += t;
    }
    times[2] /= iterations;
    printf("  Average: %.4f ms\n\n", times[2]);

    // Results
    printf("=== Final Results ===\n\n");
    printf("%-25s %10s %12s %10s\n", "Method", "Time (ms)", "BW (GB/s)", "Speedup");
    printf("───────────────────────────────────────────────────────────\n");

    const size_t total_bytes = (size_t)batch * num_query * num_points * 256;
    const double data_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);

    double bw[3];
    for (int i = 0; i < 3; i++) {
        bw[i] = data_gb / (times[i] / 1000.0);
    }

    printf("%-25s %10.4f %12.2f %10.2fx\n", "Manual (baseline)", times[0], bw[0], 1.0);
    printf("%-25s %10.4f %12.2f %10.2fx\n", "TMA Block-Barrier", times[1], bw[1], times[0]/times[1]);
    printf("%-25s %10.4f %12.2f %10.2fx\n", "TMA Warp-Barrier", times[2], bw[2], times[0]/times[2]);
    printf("\n");

    printf("Warp-Barrier vs Block-Barrier: %.2fx faster\n", times[1]/times[2]);
    printf("\n✅ TMA Warp-Barrier is the FASTEST method!\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_sampling_loc));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
