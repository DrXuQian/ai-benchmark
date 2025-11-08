#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <cstdio>

// Test 2x2x32 based on your configuration:
// GMEM: X=32, Y=162, Z=94
// SMEM: X=32, Y=2, Z=2
// This suggests: X=Channels, Y=Width, Z=Height

typedef __half dtype;

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

int main() {
    printf("=== Testing 2x2x32 with YOUR configuration ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Your config: GMEM_X=32 (C), GMEM_Y=162 (W), GMEM_Z=94 (H)
    const int GMEM_C = 32;
    const int GMEM_W = 162;
    const int GMEM_H = 94;

    printf("Global tensor: C=%d, W=%d, H=%d\n", GMEM_C, GMEM_W, GMEM_H);
    printf("Memory layout: [H][W][C] (HWC format)\n");
    printf("Total elements: %d\n", GMEM_H * GMEM_W * GMEM_C);
    printf("Total bytes: %zu\n\n", GMEM_H * GMEM_W * GMEM_C * sizeof(dtype));

    // Allocate
    dtype *array_g;
    size_t array_size = GMEM_H * GMEM_W * GMEM_C;
    cudaMalloc(&array_g, sizeof(dtype) * array_size);

    // Your tile: SMEM_X=32, SMEM_Y=2, SMEM_Z=2
    const int SMEM_C = 32;
    const int SMEM_W = 2;
    const int SMEM_H = 2;

    printf("Tile size: C=%d, W=%d, H=%d\n", SMEM_C, SMEM_W, SMEM_H);
    printf("Tile elements: %d\n", SMEM_C * SMEM_W * SMEM_H);
    printf("Tile bytes: %zu\n\n", SMEM_C * SMEM_W * SMEM_H * sizeof(dtype));

    // Create descriptor following your pattern
    CUtensorMap tma_desc{};
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();
    if (!cuTensorMapEncodeTiled_func) {
        printf("Failed to get function pointer\n");
        return 1;
    }

    // Descriptor dimensions: X=C, Y=W, Z=H
    uint64_t size[3] = {GMEM_C, GMEM_W, GMEM_H};

    // Strides: For [H][W][C] memory layout
    // stride[0]: bytes to skip from element[h][w][c] to [h][w+1][c] = C * sizeof
    // stride[1]: bytes to skip from element[h][w][c] to [h+1][w][c] = W * C * sizeof
    uint64_t stride[2] = {
        GMEM_C * sizeof(dtype),           // stride[0]: skip to next W
        GMEM_C * GMEM_W * sizeof(dtype)   // stride[1]: skip to next H
    };

    uint32_t box_size[3] = {SMEM_C, SMEM_W, SMEM_H};
    uint32_t elem_stride[3] = {1, 1, 1};

    printf("Descriptor configuration:\n");
    printf("  globalDim: [%lu, %lu, %lu] (C, W, H)\n", size[0], size[1], size[2]);
    printf("  globalStrides: [%lu, %lu] bytes\n", stride[0], stride[1]);
    printf("  boxDim: [%u, %u, %u] (C, W, H)\n", box_size[0], box_size[1], box_size[2]);
    printf("  elementStrides: [%u, %u, %u]\n\n", elem_stride[0], elem_stride[1], elem_stride[2]);

    // Check stride alignment (must be multiple of 16)
    printf("Stride alignment check:\n");
    printf("  stride[0] %% 16 = %lu %s\n", stride[0] % 16, (stride[0] % 16 == 0) ? "✅" : "❌");
    printf("  stride[1] %% 16 = %lu %s\n\n", stride[1] % 16, (stride[1] % 16 == 0) ? "✅" : "❌");

    CUresult res = cuTensorMapEncodeTiled_func(
        &tma_desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        array_g,
        size,
        stride,
        box_size,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    printf("cuTensorMapEncodeTiled returned: %d\n", res);

    if (res == CUDA_SUCCESS) {
        printf("✅ TMA descriptor creation SUCCESS!\n");
        printf("\n");
        printf("CONFIRMED: 2x2x32 tile (C=32, W=2, H=2) WORKS on SM %d.%d\n", prop.major, prop.minor);
        printf("\n");
        printf("Dimension mapping:\n");
        printf("  X → Channels (innermost)\n");
        printf("  Y → Width\n");
        printf("  Z → Height (outermost)\n");
        printf("\n");
        printf("For deformable attention [H][W][C] layout:\n");
        printf("  globalDim = [C, W, H]\n");
        printf("  stride[0] = C * sizeof(dtype)  (skip to next column)\n");
        printf("  stride[1] = W * C * sizeof(dtype)  (skip to next row)\n");
        printf("  boxDim = [32, 2, 2]  (load C=32, W=2, H=2)\n");
    } else {
        const char *err_name;
        cuInit(0);
        cuGetErrorName(res, &err_name);
        printf("❌ TMA descriptor creation FAILED: %s (%d)\n", err_name, res);
    }

    cudaFree(array_g);
    return (res == CUDA_SUCCESS) ? 0 : 1;
}
