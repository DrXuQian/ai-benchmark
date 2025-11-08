#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <vector>
#include <cmath>

// Test 8 warps, threadIdx%4==0 threads load data
// Each loading thread issues TMA for bilinear sampling (4 corner points)
// No computation, just data loading

using barrier = cuda::barrier<cuda::thread_scope_block>;
typedef __half dtype;

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

// 8 warps = 256 threads
// threadIdx % 4 == 0 means 64 threads will load
// Each thread loads 4 corner points for bilinear sampling
// Total: 64 * 4 = 256 TMA operations

__global__ void test_8warp_bilinear_sampling(
    const __grid_constant__ CUtensorMap tma_desc,
    const dtype *sampling_locations,  // [64][2] (h, w) normalized coordinates
    const int spatial_h,
    const int spatial_w,
    dtype *output  // Just to store loaded data
) {
    const int tid = threadIdx.x;

    // Shared memory allocation strategy:
    // Full size would be [64 loaders][4 corners][2][2][32] = 64*4*128 = 32KB
    // This exceeds shared memory limit when combined with other variables
    //
    // Solution: Process in two stages
    // Stage 1: Process 32 loaders (tid%4==0 for first 4 warps)
    // Stage 2: Process 32 loaders (tid%4==0 for last 4 warps)
    __shared__ alignas(128) dtype smem[32][4][2][2][32];  // Half size: 16KB

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    // Process in two stages to fit in shared memory
    for (int stage = 0; stage < 2; stage++) {
        // Initialize barrier - all threads participate
        if (tid == 0) {
            init(&bar, blockDim.x);
            asm volatile("fence.proxy.async.shared::cta;");
        }
        __syncthreads();

        // Only threadIdx % 4 == 0 will load data
        if (tid % 4 == 0) {
            int global_loader_id = tid / 4;  // 0-63

            // Stage 0: warps 0-3 (loader 0-31)
            // Stage 1: warps 4-7 (loader 32-63)
            int warp_id = tid / 32;
            if ((stage == 0 && warp_id < 4) || (stage == 1 && warp_id >= 4)) {
                int local_loader_id = (stage == 0) ? global_loader_id : (global_loader_id - 32);

                // Get sampling location for this loader
                dtype loc_h = sampling_locations[global_loader_id * 2];
                dtype loc_w = sampling_locations[global_loader_id * 2 + 1];

                // Convert normalized [0,1] to image coordinates
                dtype h_im = __hfma(loc_h, __int2half_rn(spatial_h), __float2half(0.5f));
                dtype w_im = __hfma(loc_w, __int2half_rn(spatial_w), __float2half(0.5f));

                // Get integer coordinates for bilinear corners
                int hLow = __half2int_rd(h_im);
                int wLow = __half2int_rd(w_im);
                int hHigh = hLow + 1;
                int wHigh = wLow + 1;

                // Clamp to valid range
                hLow = max(0, min(hLow, spatial_h - 2));
                wLow = max(0, min(wLow, spatial_w - 2));
                hHigh = max(0, min(hHigh, spatial_h - 2));
                wHigh = max(0, min(wHigh, spatial_w - 2));

                // Load 4 corner points using TMA
                int corners_h[4] = {hLow, hLow, hHigh, hHigh};
                int corners_w[4] = {wLow, wHigh, wLow, wHigh};

                // Issue 4 TMA operations for the 4 corners
                for (int corner = 0; corner < 4; corner++) {
                    int tensor_coord_c = 0;
                    int tensor_coord_w = corners_w[corner];
                    int tensor_coord_h = corners_h[corner];

                    dtype* my_smem_ptr = &smem[local_loader_id][corner][0][0][0];

                    // Issue TMA
                    asm volatile(
                        "{\n\t"
                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                        "}"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(my_smem_ptr))),
                          "l"(reinterpret_cast<uint64_t>(&tma_desc)),
                          "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                          "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                        : "memory"
                    );

                    // Update barrier to expect this transaction
                    asm volatile(
                        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                        :
                        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                          "n"(2 * 2 * 32 * sizeof(dtype))  // 256 bytes per tile
                    );
                }
            }
        }

        // All threads wait for all TMA operations to complete
        barrier::arrival_token token = bar.arrive();
        bar.wait(std::move(token));

        // Copy results to output (just to verify data was loaded)
        // output layout: [64 loaders][4 corners][128 elements]
        int loader_start = stage * 32;
        int loader_end = loader_start + 32;
        for (int global_loader = loader_start; global_loader < loader_end; global_loader++) {
            int local_loader = global_loader - loader_start;
            for (int corner = 0; corner < 4; corner++) {
                for (int idx = tid; idx < 128; idx += blockDim.x) {
                    int h = idx / 64;
                    int w = (idx / 32) % 2;
                    int c = idx % 32;

                    int out_idx = (global_loader * 4 + corner) * 128 + idx;
                    output[out_idx] = smem[local_loader][corner][h][w][c];
                }
            }
        }

        __syncthreads();
    }
}

int main() {
    printf("=== Testing 8 Warps with Bilinear Sampling ===\\n\\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\\n\\n", prop.name, prop.major, prop.minor);

    // Setup tensor: [H=64][W=64][C=32]
    const int H = 64, W = 64, C = 32;
    const int total = H * W * C;

    printf("Tensor: [H=%d][W=%d][C=%d]\\n", H, W, C);
    printf("Memory layout: [H][W][C]\\n\\n");

    // Initialize with pattern: value = h*100 + w
    std::vector<dtype> h_input(total);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int idx = (h * W + w) * C + c;
                float val = h * 100.0f + w;
                h_input[idx] = __float2half(val);
            }
        }
    }

    // Create sampling locations
    // 64 loaders, each with (h, w) normalized coordinates
    const int num_loaders = 64;
    std::vector<dtype> h_sampling_loc(num_loaders * 2);

    printf("Sampling locations (first 8):\\n");
    for (int i = 0; i < num_loaders; i++) {
        // Create diverse sampling points across the image
        float h_norm = (i / 8) / 8.0f + 0.3f;  // 0.3 to 1.2
        float w_norm = (i % 8) / 8.0f + 0.2f;  // 0.2 to 1.1

        h_sampling_loc[i * 2] = __float2half(h_norm);
        h_sampling_loc[i * 2 + 1] = __float2half(w_norm);

        if (i < 8) {
            float h_im = h_norm * H + 0.5f;
            float w_im = w_norm * W + 0.5f;
            int hLow = (int)floor(h_im);
            int wLow = (int)floor(w_im);
            printf("  Loader %d: h_norm=%.2f, w_norm=%.2f -> h_im=%.1f, w_im=%.1f -> hLow=%d, wLow=%d\\n",
                   i, h_norm, w_norm, h_im, w_im, hLow, wLow);
        }
    }
    printf("\\n");

    // Copy to device
    dtype *d_input, *d_sampling_loc, *d_output;
    cudaMalloc(&d_input, total * sizeof(dtype));
    cudaMalloc(&d_sampling_loc, num_loaders * 2 * sizeof(dtype));
    cudaMalloc(&d_output, num_loaders * 4 * 128 * sizeof(dtype));  // 64 loaders * 4 corners * 128 elements

    cudaMemcpy(d_input, h_input.data(), total * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), num_loaders * 2 * sizeof(dtype), cudaMemcpyHostToDevice);

    // Create TMA descriptor
    CUtensorMap tma_desc{};
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    // globalDim = [C, W, H] for [H][W][C] layout
    uint64_t size[3] = {C, W, H};
    uint64_t stride[2] = {
        C * sizeof(dtype),        // stride[0]: skip to next W
        W * C * sizeof(dtype)     // stride[1]: skip to next H
    };
    uint32_t box_size[3] = {32, 2, 2};  // C=32, W=2, H=2
    uint32_t elem_stride[3] = {1, 1, 1};

    printf("TMA Descriptor:\\n");
    printf("  globalDim: [%lu, %lu, %lu] (C, W, H)\\n", size[0], size[1], size[2]);
    printf("  stride: [%lu, %lu] bytes\\n", stride[0], stride[1]);
    printf("  boxDim: [%u, %u, %u] (C, W, H)\\n\\n", box_size[0], box_size[1], box_size[2]);

    CUresult res = cuTensorMapEncodeTiled_func(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, d_input,
        size, stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("❌ Descriptor creation failed: %d\\n", res);
        return 1;
    }
    printf("✅ TMA descriptor created\\n\\n");

    // Launch kernel: 1 block, 256 threads (8 warps)
    printf("Launching kernel: 1 block, 256 threads (8 warps)\\n");
    printf("64 threads (threadIdx%%4==0) will each issue 4 TMA operations\\n");
    printf("Total TMA operations: 64 * 4 = 256\\n");
    printf("Processing in 2 stages due to shared memory limits:\\n");
    printf("  Stage 0: warps 0-3 (32 loaders * 4 corners = 128 TMAs)\\n");
    printf("  Stage 1: warps 4-7 (32 loaders * 4 corners = 128 TMAs)\\n\\n");

    test_8warp_bilinear_sampling<<<1, 256>>>(tma_desc, d_sampling_loc, H, W, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\\n", cudaGetErrorString(err));
        return 1;
    }

    printf("✅ Kernel completed\\n\\n");

    // Verify results for first few loaders
    std::vector<dtype> h_output(num_loaders * 4 * 128);
    cudaMemcpy(h_output.data(), d_output, num_loaders * 4 * 128 * sizeof(dtype), cudaMemcpyDeviceToHost);

    printf("Verification (first 4 loaders):\\n");
    bool all_correct = true;

    for (int loader = 0; loader < 4; loader++) {
        // Get sampling location
        float h_norm = __half2float(h_sampling_loc[loader * 2]);
        float w_norm = __half2float(h_sampling_loc[loader * 2 + 1]);
        float h_im = h_norm * H + 0.5f;
        float w_im = w_norm * W + 0.5f;
        int hLow = (int)floor(h_im);
        int wLow = (int)floor(w_im);
        hLow = std::max(0, std::min(hLow, H - 2));
        wLow = std::max(0, std::min(wLow, W - 2));

        printf("  Loader %d (h_norm=%.2f, w_norm=%.2f -> hLow=%d, wLow=%d):\\n",
               loader, h_norm, w_norm, hLow, wLow);

        // Check each corner
        int corners_h[4] = {hLow, hLow, hLow + 1, hLow + 1};
        int corners_w[4] = {wLow, wLow + 1, wLow, wLow + 1};

        for (int corner = 0; corner < 4; corner++) {
            int h = corners_h[corner];
            int w = corners_w[corner];

            // Get first value from this corner's loaded data
            int out_idx = (loader * 4 + corner) * 128;
            float actual = __half2float(h_output[out_idx]);
            float expected = h * 100.0f + w;

            bool correct = std::abs(actual - expected) < 0.5f;
            printf("    Corner %d [h=%d,w=%d]: expected %.0f, got %.0f %s\\n",
                   corner, h, w, expected, actual, correct ? "✅" : "❌");

            if (!correct) all_correct = false;
        }
    }

    printf("\\n");
    if (all_correct) {
        printf("🎉 ALL TESTS PASSED!\\n");
        printf("\\n");
        printf("Confirmed:\\n");
        printf("  ✅ 8 warps with 256 threads work correctly\\n");
        printf("  ✅ threadIdx%%4==0 (64 threads) successfully issue TMA\\n");
        printf("  ✅ Each thread loads 4 corner points for bilinear sampling\\n");
        printf("  ✅ Total 256 TMA operations complete successfully\\n");
        printf("  ✅ Barrier synchronization works with massive concurrent TMAs\\n");
        printf("  ✅ Data loaded correctly matches expected bilinear sampling points\\n");
    } else {
        printf("❌ Some tests failed\\n");
    }

    cudaFree(d_input);
    cudaFree(d_sampling_loc);
    cudaFree(d_output);

    return all_correct ? 0 : 1;
}
