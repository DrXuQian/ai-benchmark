#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <vector>

// Test multiple TMA copies in one warp
// 8 threads in a warp each issue TMA to different locations

using barrier = cuda::barrier<cuda::thread_scope_block>;
typedef __half dtype;

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

// Kernel: 1 warp (32 threads), 8 threads issue TMA
__global__ void test_multi_tma_warp(
    const __grid_constant__ CUtensorMap tma_desc,
    dtype *output
) {
    // Each of 8 threads loads to a different slot in shared memory
    // smem[8][2][2][32]: 8 slots, each holds a 2x2x32 tile
    __shared__ alignas(128) dtype smem[8][2][2][32];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Only one warp
    if (warp_id != 0) return;

    // Initialize barrier - expect 8 transactions (one per issuing thread)
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    // 8 threads (lane 0-7) each issue one TMA to different positions
    if (lane_id < 8) {
        // Each thread loads from different H position: h = lane_id * 2
        // This way we test loading from 8 different spatial locations

        int my_h = lane_id * 2;  // Load from h=0,2,4,6,8,10,12,14
        int my_w = 0;
        int my_c = 0;

        // Compute coordinates: X=C, Y=W, Z=H
        int tensor_coord_x = my_c;  // C coordinate
        int tensor_coord_y = my_w;  // W coordinate
        int tensor_coord_z = my_h;  // H coordinate

        // Destination: each thread writes to smem[lane_id][][][]
        dtype* my_smem_ptr = &smem[lane_id][0][0][0];

        // Issue TMA
        asm volatile(
            "{\t\n"
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
            "}"
            :
            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(my_smem_ptr))),
              "l"(reinterpret_cast<uint64_t>(&tma_desc)),
              "r"(tensor_coord_x), "r"(tensor_coord_y), "r"(tensor_coord_z),
              "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
            : "memory"
        );

        // Update barrier to expect this transaction
        // Each issuing thread increments expected transaction count
        asm volatile(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
            :
            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
              "n"(2 * 2 * 32 * sizeof(dtype))  // 128 elements * 2 bytes
        );
    }

    // All threads wait for all 8 TMA operations to complete
    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));

    // Copy results to output
    // output layout: [8 slots][128 elements]
    for (int slot = 0; slot < 8; slot++) {
        for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) {
            int h = idx / 64;
            int w = (idx / 32) % 2;
            int c = idx % 32;

            int out_idx = slot * 128 + idx;
            output[out_idx] = smem[slot][h][w][c];
        }
    }
}

int main() {
    printf("=== Testing Multiple TMA in One Warp ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Setup tensor: [H=32][W=4][C=32]
    const int H = 32, W = 4, C = 32;
    const int total = H * W * C;

    printf("Tensor: [H=%d][W=%d][C=%d]\n", H, W, C);
    printf("Memory layout: [H][W][C]\n\n");

    // Initialize with pattern: value = h*100 + w*10 + c
    std::vector<dtype> h_input(total);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int idx = (h * W + w) * C + c;
                float val = h * 100.0f + w * 10.0f + c;
                h_input[idx] = __float2half(val);
            }
        }
    }

    // Show what we expect to load
    printf("Expected values for each TMA operation:\n");
    for (int lane = 0; lane < 8; lane++) {
        int h_start = lane * 2;
        printf("  Lane %d (h=%d-%d, w=0-1, c=0-31): first value = %.0f\n",
               lane, h_start, h_start+1, __half2float(h_input[(h_start * W + 0) * C + 0]));
    }
    printf("\n");

    // Copy to device
    dtype *d_input, *d_output;
    cudaMalloc(&d_input, total * sizeof(dtype));
    cudaMalloc(&d_output, 8 * 128 * sizeof(dtype));  // 8 slots of 128 elements
    cudaMemcpy(d_input, h_input.data(), total * sizeof(dtype), cudaMemcpyHostToDevice);

    // Create TMA descriptor
    // globalDim = [C, W, H] for [H][W][C] layout
    CUtensorMap tma_desc{};
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

    uint64_t size[3] = {C, W, H};
    uint64_t stride[2] = {
        C * sizeof(dtype),        // stride[0]: skip to next W
        W * C * sizeof(dtype)     // stride[1]: skip to next H
    };
    uint32_t box_size[3] = {32, 2, 2};  // C=32, W=2, H=2
    uint32_t elem_stride[3] = {1, 1, 1};

    printf("TMA Descriptor:\n");
    printf("  globalDim: [%lu, %lu, %lu] (C, W, H)\n", size[0], size[1], size[2]);
    printf("  stride: [%lu, %lu] bytes\n", stride[0], stride[1]);
    printf("  boxDim: [%u, %u, %u] (C, W, H)\n\n", box_size[0], box_size[1], box_size[2]);

    CUresult res = cuTensorMapEncodeTiled_func(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, d_input,
        size, stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("❌ Descriptor creation failed: %d\n", res);
        return 1;
    }
    printf("✅ TMA descriptor created\n\n");

    // Launch kernel: 1 block, 32 threads (1 warp)
    printf("Launching kernel: 1 block, 32 threads (1 warp)\n");
    printf("8 threads will each issue 1 TMA operation\n\n");

    test_multi_tma_warp<<<1, 32>>>(tma_desc, d_output);

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

    // Verify results
    std::vector<dtype> h_output(8 * 128);
    cudaMemcpy(h_output.data(), d_output, 8 * 128 * sizeof(dtype), cudaMemcpyDeviceToHost);

    printf("Verification:\n");
    bool all_correct = true;

    for (int slot = 0; slot < 8; slot++) {
        int h_start = slot * 2;  // Each slot should have data from h = slot*2
        bool slot_correct = true;

        // Check first few values of this slot
        for (int local_idx = 0; local_idx < 4 && slot_correct; local_idx++) {
            int local_h = local_idx / 2;
            int local_w = local_idx % 2;
            int local_c = 0;

            int out_idx = slot * 128 + local_h * 64 + local_w * 32 + local_c;
            float actual = __half2float(h_output[out_idx]);

            int global_h = h_start + local_h;
            int global_w = local_w;
            float expected = global_h * 100.0f + global_w * 10.0f + local_c;

            if (std::abs(actual - expected) > 0.5f) {
                printf("  Slot %d: ❌ Mismatch at [%d][%d][%d]: expected %.0f, got %.0f\n",
                       slot, local_h, local_w, local_c, expected, actual);
                slot_correct = false;
                all_correct = false;
            }
        }

        if (slot_correct) {
            float first_val = __half2float(h_output[slot * 128]);
            printf("  Slot %d: ✅ Correct (first value: %.0f)\n", slot, first_val);
        }
    }

    printf("\n");
    if (all_correct) {
        printf("🎉 ALL TESTS PASSED!\n");
        printf("\n");
        printf("Confirmed:\n");
        printf("  ✅ Multiple TMA operations in one warp work correctly\n");
        printf("  ✅ Each thread can issue TMA to different shared memory locations\n");
        printf("  ✅ Barrier synchronization works with 8 concurrent TMAs\n");
        printf("  ✅ Single-stage (wait for completion) works correctly\n");
    } else {
        printf("❌ Some tests failed\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return all_correct ? 0 : 1;
}
