#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <vector>

// Verify TMA layout: Does X,Y,Z correspond to C,W,H or H,W,C?

using barrier = cuda::barrier<cuda::thread_scope_block>;
typedef __half dtype;

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

// Test kernel that loads and prints values
__global__ void test_tma_load(const __grid_constant__ CUtensorMap tma_desc, dtype *output)
{
    __shared__ alignas(128) dtype smem[2][2][32];  // Z=2, Y=2, X=32

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        asm volatile("fence.proxy.async.shared::cta;");

        // Load tile at coordinates (0, 0, 0)
        int tensor_coord_x = 0;  // X coordinate
        int tensor_coord_y = 0;  // Y coordinate
        int tensor_coord_z = 0;  // Z coordinate

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
              "n"(2 * 2 * 32 * sizeof(dtype))
            : "memory");

        barrier::arrival_token token = bar.arrive();
        bar.wait(std::move(token));
    }
    __syncthreads();

    // Copy to output: output[z][y][x]
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 32; x++) {
                int idx = z * 2 * 32 + y * 32 + x;
                if (threadIdx.x == idx) {
                    output[idx] = smem[z][y][x];
                }
            }
        }
    }
}

int main() {
    printf("=== TMA Layout Verification ===\n\n");

    // Setup: Create tensor with known pattern
    // We'll use layout: [H][W][C] = [4][4][32]
    const int H = 4, W = 4, C = 32;
    const int total = H * W * C;

    std::vector<dtype> h_input(total);

    // Fill with pattern: value = h*1000 + w*100 + c
    printf("Filling tensor with pattern: value = h*1000 + w*100 + c\n");
    printf("Tensor shape: [H=%d][W=%d][C=%d]\n", H, W, C);
    printf("Memory layout: row-major (C innermost)\n\n");

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int idx = (h * W + w) * C + c;
                float val = h * 1000.0f + w * 100.0f + c;
                h_input[idx] = __float2half(val);
            }
        }
    }

    // Show some example values
    printf("Example values in memory:\n");
    printf("  [h=0][w=0][c=0] = %.0f (index 0)\n", __half2float(h_input[0]));
    printf("  [h=0][w=0][c=1] = %.0f (index 1)\n", __half2float(h_input[1]));
    printf("  [h=0][w=0][c=31] = %.0f (index 31)\n", __half2float(h_input[31]));
    printf("  [h=0][w=1][c=0] = %.0f (index 32)\n", __half2float(h_input[32]));
    printf("  [h=1][w=0][c=0] = %.0f (index 128)\n", __half2float(h_input[128]));
    printf("\n");

    // Copy to device
    dtype *d_input, *d_output;
    cudaMalloc(&d_input, total * sizeof(dtype));
    cudaMalloc(&d_output, 2 * 2 * 32 * sizeof(dtype));
    cudaMemcpy(d_input, h_input.data(), total * sizeof(dtype), cudaMemcpyHostToDevice);

    // Test 1: X=C, Y=W, Z=H (expected correct layout for HWC memory)
    printf("=== Test 1: globalDim = [C=32, W=4, H=4], boxDim = [32, 2, 2] ===\n");
    printf("Hypothesis: X=Channels, Y=Width, Z=Height\n");
    {
        CUtensorMap tma_desc{};
        auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

        uint64_t size[3] = {C, W, H};  // X=C, Y=W, Z=H
        uint64_t stride[2] = {
            C * sizeof(dtype),          // stride[0]: bytes to skip for Y (skip one column)
            W * C * sizeof(dtype)       // stride[1]: bytes to skip for Z (skip one row)
        };
        uint32_t box_size[3] = {32, 2, 2};  // Load 32x2x2 tile
        uint32_t elem_stride[3] = {1, 1, 1};

        printf("stride[0] = %lu bytes (skip to next W)\n", stride[0]);
        printf("stride[1] = %lu bytes (skip to next H)\n\n", stride[1]);

        CUresult res = cuTensorMapEncodeTiled_func(
            &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, d_input,
            size, stride, box_size, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        if (res != CUDA_SUCCESS) {
            printf("❌ Descriptor creation FAILED (code %d)\n\n", res);
        } else {
            printf("✅ Descriptor created\n");

            test_tma_load<<<1, 128>>>(tma_desc, d_output);
            cudaDeviceSynchronize();

            std::vector<dtype> h_output(128);
            cudaMemcpy(h_output.data(), d_output, 128 * sizeof(dtype), cudaMemcpyDeviceToHost);

            printf("Loaded values (first 8 of each position):\n");
            printf("  smem[0][0][0-7]: ");
            for (int i = 0; i < 8; i++) printf("%.0f ", __half2float(h_output[i]));
            printf("\n");

            printf("  smem[0][1][0-7]: ");
            for (int i = 0; i < 8; i++) printf("%.0f ", __half2float(h_output[32 + i]));
            printf("\n");

            printf("  smem[1][0][0-7]: ");
            for (int i = 0; i < 8; i++) printf("%.0f ", __half2float(h_output[64 + i]));
            printf("\n");

            printf("  smem[1][1][0-7]: ");
            for (int i = 0; i < 8; i++) printf("%.0f ", __half2float(h_output[96 + i]));
            printf("\n\n");

            // Verify
            bool correct = true;
            float expected[4][8];
            // Expected: smem[z][y][x] should contain value from [h=z][w=y][c=x]
            for (int z = 0; z < 2; z++) {
                for (int y = 0; y < 2; y++) {
                    for (int x = 0; x < 8; x++) {
                        int output_idx = z * 64 + y * 32 + x;
                        float actual = __half2float(h_output[output_idx]);
                        float expected_val = z * 1000.0f + y * 100.0f + x;
                        expected[z*2+y][x] = expected_val;

                        if (std::abs(actual - expected_val) > 0.5f) {
                            printf("Mismatch at [%d][%d][%d]: expected %.0f, got %.0f\n",
                                   z, y, x, expected_val, actual);
                            correct = false;
                        }
                    }
                }
            }

            if (correct) {
                printf("✅ Test 1 PASSED! X=C, Y=W, Z=H is CORRECT\n");
            } else {
                printf("❌ Test 1 FAILED\n");
            }
        }
    }
    printf("\n");

    // Test 2: X=H, Y=W, Z=C (alternative hypothesis)
    printf("=== Test 2: globalDim = [H=4, W=4, C=32], boxDim = [2, 2, 32] ===\n");
    printf("Hypothesis: X=Height, Y=Width, Z=Channels\n");
    {
        CUtensorMap tma_desc{};
        auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();

        uint64_t size[3] = {H, W, C};  // X=H, Y=W, Z=C
        uint64_t stride[2] = {
            H * sizeof(dtype),          // stride[0]
            W * H * sizeof(dtype)       // stride[1]
        };
        uint32_t box_size[3] = {2, 2, 32};
        uint32_t elem_stride[3] = {1, 1, 1};

        printf("stride[0] = %lu bytes\n", stride[0]);
        printf("stride[1] = %lu bytes\n\n", stride[1]);

        CUresult res = cuTensorMapEncodeTiled_func(
            &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, d_input,
            size, stride, box_size, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        if (res != CUDA_SUCCESS) {
            printf("❌ Descriptor creation FAILED (code %d)\n", res);
        } else {
            printf("✅ Descriptor created\n");
            printf("(Will not match expected pattern if hypothesis is wrong)\n");
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
