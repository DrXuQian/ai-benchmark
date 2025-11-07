#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <cstdio>

// Simple TMA test with float16, mimicking Hopper benchmark structure

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    if (err != cudaSuccess) {
        printf("Failed to get entry point: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr);
}

int main() {
    printf("Device: ");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Small tensor for testing: 16x16x32 (float16)
    const int H = 16, W = 16, C = 32;
    size_t total = H * W * C;

    printf("Testing TMA with float16\n");
    printf("Tensor: %dx%dx%d = %zu elements\n", H, W, C, total);
    printf("Memory: %zu bytes\n\n", total * sizeof(__half));

    // Allocate
    __half *d_data;
    cudaMalloc(&d_data, total * sizeof(__half));

    // Create descriptor
    CUtensorMap tma_desc{};

    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();
    if (!cuTensorMapEncodeTiled_func) {
        printf("❌ Failed to get function pointer\n");
        return 1;
    }

    // Descriptor parameters
    uint64_t globalDim[3] = {H, W, C};
    uint64_t globalStrides[2] = {
        W * C * sizeof(__half),   // stride[0]
        C * sizeof(__half)         // stride[1]
    };
    uint32_t boxDim[3] = {2, 2, 32};  // 2x2x32 tile
    uint32_t elemStride[3] = {1, 1, 1};

    printf("Global dimensions: [%lu, %lu, %lu]\n", globalDim[0], globalDim[1], globalDim[2]);
    printf("Global strides: [%lu, %lu] bytes\n", globalStrides[0], globalStrides[1]);
    printf("Box dimensions: [%u, %u, %u]\n", boxDim[0], boxDim[1], boxDim[2]);
    printf("Element strides: [%u, %u, %u]\n\n", elemStride[0], elemStride[1], elemStride[2]);

    CUresult result = cuTensorMapEncodeTiled_func(
        &tma_desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        (void*)d_data,
        globalDim,
        globalStrides,
        boxDim,
        elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    printf("cuTensorMapEncodeTiled returned: %d\n", result);

    if (result == CUDA_SUCCESS) {
        printf("✅ TMA descriptor created successfully!\n");
        printf("\nSUCCESS: float16 TMA works on SM %d.%d\n", prop.major, prop.minor);
    } else {
        const char *err_name;
        cuInit(0);
        cuGetErrorName(result, &err_name);
        printf("❌ TMA descriptor creation failed: %s (%d)\n", err_name, result);
    }

    cudaFree(d_data);
    return (result == CUDA_SUCCESS) ? 0 : 1;
}
