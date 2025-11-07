#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled
#include <cstdio>
#include <vector>

using TmaDescriptor = CUtensorMap;

// Get cuTensorMapEncodeTiled function pointer dynamically
PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    if (err != cudaSuccess) {
        printf("Failed to get cuTensorMapEncodeTiled entry point: %s\n",
               cudaGetErrorString(err));
        return nullptr;
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr);
}

// Simple test kernel using TMA
__device__ __forceinline__ uint32_t __as_ptr_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void test_tma_kernel(
    __half* output,
    const TmaDescriptor* tma_desc
) {
    __shared__ __align__(128) __half smem[2][2][32];
    __shared__ __align__(8) uint64_t barrier;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                     :: "r"(__as_ptr_smem(&barrier)), "r"(1));
    }
    __syncthreads();

    // Only thread 0 issues TMA
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t coord_h = 0, coord_w = 0, coord_c = 0;

        // TMA load
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4}], [%5];"
            :
            : "r"(__as_ptr_smem(&smem[0][0][0])),
              "l"(tma_desc),
              "r"(coord_h), "r"(coord_w), "r"(coord_c),
              "r"(__as_ptr_smem(&barrier))
            : "memory"
        );

        // Wait
        asm volatile(
            "{\n\t"
            "  .reg .pred p;\n\t"
            "  .reg .b32 r_bar;\n\t"
            "  mov.b32 r_bar, %0;\n\t"
            "$wait_loop_%=:\n\t"
            "  mbarrier.try_wait.parity.shared.b64 p, [r_bar], 0;\n\t"
            "  @!p bra $wait_loop_%=;\n\t"
            "}\n\t"
            :
            : "r"(__as_ptr_smem(&barrier))
        );
    }
    __syncthreads();

    // Copy to output
    if (threadIdx.x < 32) {
        output[threadIdx.x] = smem[0][0][threadIdx.x];
    }
}

int main() {
    cuInit(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        printf("ERROR: TMA requires SM 9.0 or higher (Hopper)\n");
        printf("Your device is SM %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    printf("=== Testing TMA Descriptor Creation ===\n");

    // Allocate input tensor: 100x100x32
    const int H = 100, W = 100, C = 32;
    size_t total = H * W * C;
    __half *d_input, *d_output;
    cudaMalloc(&d_input, total * sizeof(__half));
    cudaMalloc(&d_output, 32 * sizeof(__half));

    // Initialize input
    std::vector<__half> h_input(total);
    for (size_t i = 0; i < total; i++) {
        h_input[i] = __float2half((i % 100) / 100.0f);
    }
    cudaMemcpy(d_input, h_input.data(), total * sizeof(__half), cudaMemcpyHostToDevice);

    // Create TMA descriptor
    printf("\nCreating TMA descriptor for 2x2x32 tile...\n");
    printf("Tensor dimensions: %dx%dx%d\n", H, W, C);
    printf("Tile dimensions: 2x2x32\n\n");

    TmaDescriptor h_tma_desc;

    cuuint64_t globalDim[3] = {(cuuint64_t)H, (cuuint64_t)W, (cuuint64_t)C};

    // IMPORTANT: stride array has rank-1 elements (not rank!)
    // stride[0] = bytes between consecutive elements in dim[0]
    // stride[1] = bytes between consecutive elements in dim[1]
    // stride for last dimension is implicit (sizeof element)
    cuuint64_t globalStrides[2] = {
        W * C * sizeof(__half),  // stride for dimension 0 (H): skip one row
        C * sizeof(__half)        // stride for dimension 1 (W): skip one column
    };

    cuuint32_t boxDim[3] = {2, 2, 32};
    cuuint32_t elementStrides[3] = {1, 1, 1};

    // Get function pointer dynamically (required for SM 12.0!)
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();
    if (!cuTensorMapEncodeTiled_func) {
        printf("❌ Failed to get cuTensorMapEncodeTiled function pointer\n");
        return 1;
    }

    CUresult result = cuTensorMapEncodeTiled_func(
        &h_tma_desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        (void*)d_input,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        const char *err_name, *err_str;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_str);
        printf("❌ TMA descriptor creation FAILED\n");
        printf("Error: %s - %s\n", err_name, err_str);
        printf("\nThis is expected on SM 12.0 (Blackwell)\n");
        printf("Please run this test on Hopper (SM 9.0)\n");
        return 1;
    }

    printf("✅ TMA descriptor created successfully!\n\n");

    // Copy descriptor to device
    TmaDescriptor *d_tma_desc;
    cudaMalloc(&d_tma_desc, sizeof(TmaDescriptor));
    cudaMemcpy(d_tma_desc, &h_tma_desc, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    // Launch kernel
    printf("Launching TMA kernel...\n");
    test_tma_kernel<<<1, 32>>>(d_output, d_tma_desc);

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

    printf("✅ Kernel executed successfully!\n\n");

    // Verify results
    std::vector<__half> h_output(32);
    cudaMemcpy(h_output.data(), d_output, 32 * sizeof(__half), cudaMemcpyDeviceToHost);

    printf("First 8 output values: ");
    for (int i = 0; i < 8; i++) {
        printf("%.3f ", __half2float(h_output[i]));
    }
    printf("\n");

    printf("Expected values:       ");
    for (int i = 0; i < 8; i++) {
        printf("%.3f ", __half2float(h_input[i]));
    }
    printf("\n\n");

    bool correct = true;
    for (int i = 0; i < 32; i++) {
        float expected = __half2float(h_input[i]);
        float actual = __half2float(h_output[i]);
        if (std::abs(expected - actual) > 1e-3f) {
            printf("Mismatch at %d: expected %.3f, got %.3f\n", i, expected, actual);
            correct = false;
        }
    }

    if (correct) {
        printf("✅ TMA test PASSED!\n");
        printf("\nTMA is working correctly on this GPU!\n");
    } else {
        printf("❌ TMA test FAILED\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_tma_desc);

    return correct ? 0 : 1;
}
