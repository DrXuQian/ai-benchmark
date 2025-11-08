#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <vector>
#include <fstream>

// Test TMA with real binary data from deformable attention
// Based on working tma_bw_3d.cu pattern
// Use first spatial shape: [92, 160]

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
#define LOAD_SIZE (TILE_H * TILE_W * TILE_C * sizeof(dtype))  // 2*2*32*2 = 256 bytes

// TMA kernel - follows tma_bw_3d.cu pattern
// tid==0 issues TMA, all threads wait
__global__ void tma_deform_attn_load(
    const __grid_constant__ CUtensorMap tma_desc,
    const dtype *sampling_loc,  // [num_samples][2] (w, h) normalized
    const int num_samples,
    dtype *output  // [num_samples][TILE_H][TILE_W][TILE_C]
) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    if (bid >= num_samples) return;

    __shared__ alignas(128) dtype smem[TILE_H][TILE_W][TILE_C];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    // Initialize barrier first (before any control flow)
    if (tid == 0) {
        init(&bar, blockDim.x);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    // Only tid==0 does TMA load
    if (tid == 0) {
        // Get sampling location for this sample
        // Location is normalized [0, 1], need to convert to image coordinates
        dtype loc_w_norm = sampling_loc[bid * 2];      // w is first
        dtype loc_h_norm = sampling_loc[bid * 2 + 1];  // h is second

        // Convert to image coordinates (same as deform_attn.cu)
        // hw_im = loc * spatial_hw + 0.5
        dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
        dtype h_im = __hfma(loc_h_norm, __int2half_rn(SPATIAL_H), __float2half(0.5f));

        dtype kZERO = __float2half(0.0f);

        // Check bounds (same as deform_attn.cu line 96)
        if (h_im > kZERO && w_im > kZERO &&
            h_im < __int2half_rn(SPATIAL_H + 1) && w_im < __int2half_rn(SPATIAL_W + 1)) {

            // Get integer coordinates for bilinear sampling
            int hLow = __half2int_rd(h_im);
            int wLow = __half2int_rd(w_im);

            // Clamp to valid range (ensure we can load 2x2 tile)
            hLow = max(0, min(hLow, SPATIAL_H - TILE_H));
            wLow = max(0, min(wLow, SPATIAL_W - TILE_W));

            // TMA coordinates: X=C, Y=W, Z=H (innermost to outermost)
            int tensor_coord_x = 0;      // C always 0 (load full channels)
            int tensor_coord_y = wLow;   // W coordinate
            int tensor_coord_z = hLow;   // H coordinate

            // Issue TMA load AND expect_tx together
            asm volatile(
                "{\t\n"
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%5], %6;\n\t"
                "}"
                :
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem))),
                  "l"(reinterpret_cast<uint64_t>(&tma_desc)),
                  "r"(tensor_coord_x), "r"(tensor_coord_y), "r"(tensor_coord_z),
                  "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                  "n"(LOAD_SIZE)
                : "memory"
            );
        }
    }

    // All threads arrive on the barrier
    barrier::arrival_token token = bar.arrive();

    // Wait for the data to have arrived
    bar.wait(std::move(token));

    // Copy smem to output
    for (int idx = tid; idx < TILE_H * TILE_W * TILE_C; idx += blockDim.x) {
        int h = idx / (TILE_W * TILE_C);
        int w = (idx / TILE_C) % TILE_W;
        int c = idx % TILE_C;
        output[bid * TILE_H * TILE_W * TILE_C + idx] = smem[h][w][c];
    }
}

// Load binary file
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
    printf("Loaded %s: %zu elements (%.2f MB)\n", filename, data.size(), size_bytes / (1024.0f * 1024.0f));
    return data;
}

int main() {
    printf("=== TMA Test with Real Deformable Attention Data ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Configuration - first level only
    printf("Configuration:\n");
    printf("  Spatial shape: [H=%d, W=%d]\n", SPATIAL_H, SPATIAL_W);
    printf("  Channels: %d\n", CHANNELS);
    printf("  Tile size: [%d, %d, %d]\n\n", TILE_H, TILE_W, TILE_C);

    // Load data
    printf("Loading binary data...\n");
    auto h_value = load_binary<dtype>("working/test_data_value.bin");
    auto h_sampling_loc = load_binary<dtype>("working/test_data_sampling_locations.bin");
    auto h_level_start_index = load_binary<int64_t>("working/test_data_level_start_index.bin");
    printf("\n");

    // Extract first level data
    // value layout: [batch=48][spatial_size=20522][channels=32]
    // level_start_index[0] = 0, first level has H*W = 92*160 = 14720 points
    const int batch = 48;
    const int level_start = h_level_start_index[0];  // 0
    const int level_size = SPATIAL_H * SPATIAL_W;     // 14720

    printf("Level 0 info:\n");
    printf("  level_start_index: %d\n", level_start);
    printf("  level_size (H*W): %d\n\n", level_size);

    // Extract sampling locations for first batch, first level
    // sampling_loc layout: [batch][num_query][num_heads][num_levels][num_points][2]
    // We'll use first batch, first query, first head, first level, all points
    const int num_levels = 4;
    const int num_points = 8;
    const int num_test_samples = 100;  // Test with 100 samples

    std::vector<dtype> h_test_loc(num_test_samples * 2);
    printf("Extracting %d sampling locations...\n", num_test_samples);
    for (int i = 0; i < num_test_samples; i++) {
        // Get location from [batch=0][query=0][head=0][level=0][point=i%8][2]
        int point_idx = i % num_points;
        int loc_offset = (0 * num_levels + 0) * num_points + point_idx;  // level=0, point=i%8
        h_test_loc[i * 2] = h_sampling_loc[loc_offset * 2];      // w
        h_test_loc[i * 2 + 1] = h_sampling_loc[loc_offset * 2 + 1];  // h

        if (i < 5) {
            float w_norm = __half2float(h_test_loc[i * 2]);
            float h_norm = __half2float(h_test_loc[i * 2 + 1]);
            float w_im = w_norm * SPATIAL_W + 0.5f;
            float h_im = h_norm * SPATIAL_H + 0.5f;
            printf("  Sample %d: loc_norm=(%.4f, %.4f) -> im=(%.2f, %.2f) -> Low=(%d, %d)\n",
                   i, w_norm, h_norm, w_im, h_im, (int)floor(w_im), (int)floor(h_im));
        }
    }
    printf("\n");

    // Allocate device memory
    dtype *d_value, *d_sampling_loc, *d_output;

    // Copy first batch, first level value data
    size_t level_data_size = level_size * CHANNELS;
    CUDA_CHECK(cudaMalloc(&d_value, level_data_size * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data() + level_start * CHANNELS,
                          level_data_size * sizeof(dtype), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_sampling_loc, h_test_loc.size() * sizeof(dtype)));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_test_loc.data(),
                          h_test_loc.size() * sizeof(dtype), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_output, num_test_samples * TILE_H * TILE_W * TILE_C * sizeof(dtype)));

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

    printf("  globalDim: [%lu, %lu, %lu] (C, W, H)\n", globalDim[0], globalDim[1], globalDim[2]);
    printf("  globalStrides: [%lu, %lu] bytes\n", globalStrides[0], globalStrides[1]);
    printf("  boxDim: [%u, %u, %u] (C, W, H)\n\n", boxDim[0], boxDim[1], boxDim[2]);

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
        printf("❌ TMA descriptor creation failed: %d\n", res);
        return 1;
    }
    printf("✅ TMA descriptor created\n\n");

    // Launch kernel
    printf("Launching kernel...\n");
    printf("  Blocks: %d\n", num_test_samples);
    printf("  Threads per block: 256\n\n");

    tma_deform_attn_load<<<num_test_samples, 256>>>(
        tma_desc,
        d_sampling_loc,
        num_test_samples,
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

    printf("✅ Kernel completed\n\n");

    // Verify output by comparing with CPU ground truth
    std::vector<dtype> h_output(num_test_samples * TILE_H * TILE_W * TILE_C);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          h_output.size() * sizeof(dtype), cudaMemcpyDeviceToHost));

    printf("Verifying TMA loaded data against CPU ground truth...\n\n");

    int total_errors = 0;
    int total_elements = 0;
    float max_diff = 0.0f;

    for (int s = 0; s < num_test_samples; s++) {
        // Recompute the same location calculation
        dtype loc_w_norm = h_test_loc[s * 2];
        dtype loc_h_norm = h_test_loc[s * 2 + 1];

        float w_im = __half2float(loc_w_norm) * SPATIAL_W + 0.5f;
        float h_im = __half2float(loc_h_norm) * SPATIAL_H + 0.5f;

        // Skip out of bounds
        if (h_im <= 0.0f || w_im <= 0.0f ||
            h_im >= (SPATIAL_H + 1) || w_im >= (SPATIAL_W + 1)) {
            continue;
        }

        int hLow = (int)floor(h_im);
        int wLow = (int)floor(w_im);
        hLow = std::max(0, std::min(hLow, SPATIAL_H - TILE_H));
        wLow = std::max(0, std::min(wLow, SPATIAL_W - TILE_W));

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

                    // Get TMA loaded value
                    int output_idx = s * TILE_H * TILE_W * TILE_C + h * TILE_W * TILE_C + w * TILE_C + c;
                    float tma_value = __half2float(h_output[output_idx]);

                    // Compare
                    float diff = std::abs(gt_value - tma_value);
                    max_diff = std::max(max_diff, diff);

                    if (diff > 1e-6f) {  // Allow for tiny floating point errors
                        if (!sample_has_error && s < 5) {
                            printf("  Sample %d ERROR at [h=%d,w=%d,c=%d]: GT=%.6f, TMA=%.6f, diff=%.6f\n",
                                   s, h, w, c, gt_value, tma_value, diff);
                        }
                        sample_has_error = true;
                        total_errors++;
                    }
                    total_elements++;
                }
            }
        }

        // Print first few successful samples
        if (!sample_has_error && s < 5) {
            printf("  Sample %d: ✅ PASS - Position [h=%d,w=%d], first value: %.6f\n",
                   s, hLow, wLow, __half2float(h_output[s * TILE_H * TILE_W * TILE_C]));
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
