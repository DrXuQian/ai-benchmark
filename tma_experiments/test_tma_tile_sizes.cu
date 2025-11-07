#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <cstdio>

// Test different tile sizes to find which ones work

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

void test_tile_size(const char* name, int tile_x, int tile_y, int tile_z) {
    typedef __half dtype;
    const int GMEM_X = 128;
    const int GMEM_Y = 128;
    const int GMEM_Z = 128;

    dtype *array_g;
    cudaMalloc(&array_g, GMEM_X * GMEM_Y * GMEM_Z * sizeof(dtype));

    CUtensorMap tma_desc{};
    uint64_t size[3] = {GMEM_X, GMEM_Y, GMEM_Z};
    uint64_t stride[2] = {GMEM_X * sizeof(dtype), GMEM_X * sizeof(dtype) * GMEM_Y};
    uint32_t box_size[3] = {(uint32_t)tile_x, (uint32_t)tile_y, (uint32_t)tile_z};
    uint32_t elem_stride[3] = {1, 1, 1};

    auto func = get_cuTensorMapEncodeTiled();
    CUresult res = func(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, array_g,
        size, stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    printf("%-20s [%2d,%2d,%2d]: ", name, tile_x, tile_y, tile_z);
    if (res == CUDA_SUCCESS) {
        printf("✅ SUCCESS\n");
    } else {
        printf("❌ FAIL (code %d)\n", res);
    }

    cudaFree(array_g);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    printf("Testing float16 tile sizes:\n");
    printf("%-20s %s\n", "Name", "Tile [X,Y,Z]");
    printf("----------------------------------------\n");

    // Our desired size
    test_tile_size("Desired 2x2x32", 2, 2, 32);

    // Powers of 2
    test_tile_size("4x4x4", 4, 4, 4);
    test_tile_size("8x8x8", 8, 8, 8);
    test_tile_size("16x16x16", 16, 16, 16);
    test_tile_size("32x32x32", 32, 32, 32);

    // Variations
    test_tile_size("4x4x32", 4, 4, 32);
    test_tile_size("8x8x32", 8, 8, 32);
    test_tile_size("16x16x32", 16, 16, 32);

    test_tile_size("2x2x16", 2, 2, 16);
    test_tile_size("4x4x16", 4, 4, 16);

    test_tile_size("8x8x16", 8, 8, 16);
    test_tile_size("16x16x8", 16, 16, 8);

    // Minimum sizes
    test_tile_size("1x1x16", 1, 1, 16);
    test_tile_size("2x2x8", 2, 2, 8);
    test_tile_size("2x2x4", 2, 2, 4);

    printf("\n");
    return 0;
}
