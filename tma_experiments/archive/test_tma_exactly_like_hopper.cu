#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <cstdio>

// Exactly mimicking NVIDIA Hopper Benchmark approach

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",
                                               &cuTensorMapEncodeTiled_ptr,
                                               cudaEnableDefault,
                                               &driver_status);
    if (err != cudaSuccess) {
        printf("cudaGetDriverEntryPoint failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Test 1: float (like their original)
    printf("=== Test 1: float (like Hopper benchmark) ===\n");
    {
        typedef float dtype;
        const int GMEM_X = 64;
        const int GMEM_Y = 64;
        const int GMEM_Z = 64;
        const int SMEM_X = 16;
        const int SMEM_Y = 16;
        const int SMEM_Z = 16;

        dtype *array_g;
        size_t array_size = GMEM_X * GMEM_Y * GMEM_Z;
        cudaMalloc(&array_g, sizeof(dtype) * array_size);

        CUtensorMap tma_desc{};

        auto rank = 3;
        uint64_t size[3] = {GMEM_X, GMEM_Y, GMEM_Z};
        uint64_t stride[2] = {GMEM_X * sizeof(dtype), GMEM_X * sizeof(dtype) * GMEM_Y};
        uint32_t box_size[3] = {SMEM_X, SMEM_Y, SMEM_Z};
        uint32_t elem_stride[3] = {1, 1, 1};

        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
        if (!cuTensorMapEncodeTiled) {
            printf("Failed to get function pointer\n");
            return 1;
        }

        CUresult res = cuTensorMapEncodeTiled(
            &tma_desc,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            rank,
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

        printf("cuTensorMapEncodeTiled returned CUresult: %d\n", res);
        if (res == CUDA_SUCCESS) {
            printf("✅ float test PASSED\n\n");
        } else {
            printf("❌ float test FAILED\n\n");
        }

        cudaFree(array_g);
    }

    // Test 2: float16 with same tile size as float32
    printf("=== Test 2: float16 with 16x16x16 tile ===\n");
    {
        typedef __half dtype;
        const int GMEM_X = 64;
        const int GMEM_Y = 64;
        const int GMEM_Z = 64;
        const int SMEM_X = 16;
        const int SMEM_Y = 16;
        const int SMEM_Z = 16;

        dtype *array_g;
        size_t array_size = GMEM_X * GMEM_Y * GMEM_Z;
        cudaMalloc(&array_g, sizeof(dtype) * array_size);

        CUtensorMap tma_desc{};

        auto rank = 3;
        uint64_t size[3] = {GMEM_X, GMEM_Y, GMEM_Z};
        uint64_t stride[2] = {GMEM_X * sizeof(dtype), GMEM_X * sizeof(dtype) * GMEM_Y};
        uint32_t box_size[3] = {SMEM_X, SMEM_Y, SMEM_Z};
        uint32_t elem_stride[3] = {1, 1, 1};

        printf("Tensor: %dx%dx%d (float16)\n", GMEM_X, GMEM_Y, GMEM_Z);
        printf("Tile: %dx%dx%d\n", SMEM_X, SMEM_Y, SMEM_Z);
        printf("Stride[0]: %lu bytes\n", stride[0]);
        printf("Stride[1]: %lu bytes\n", stride[1]);

        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

        CUresult res = cuTensorMapEncodeTiled(
            &tma_desc,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            rank,
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

        printf("cuTensorMapEncodeTiled returned CUresult: %d\n", res);
        if (res == CUDA_SUCCESS) {
            printf("✅ float16 test PASSED\n\n");
        } else {
            const char *err_name;
            cuInit(0);
            cuGetErrorName(res, &err_name);
            printf("❌ float16 test FAILED: %s\n\n", err_name);
        }

        cudaFree(array_g);
    }

    return 0;
}
