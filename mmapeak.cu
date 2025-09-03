#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

__global__ void tensor_core_mma_kernel(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b, 
    uint64_t idescE,
    uint32_t scaleC,
    uint32_t tsfa_addr,
    uint32_t tsfb_addr
) {
    // 直接使用你提供的PTX代码
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(tsfa_addr), "r"(tsfb_addr)
    );
}

int main() {
    // 检查CUDA版本和设备
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "设备: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    
    // 分配设备内存
    void* d_scale_a;
    void* d_scale_b;
    cudaMalloc(&d_scale_a, 64);
    cudaMalloc(&d_scale_b, 64);
    
    // 准备参数数据
    uint32_t tmem_c = 0x10000000;  // tensor memory地址
    uint64_t desc_a = 0x0000000000000000ULL;  // 描述符A
    uint64_t desc_b = 0x0000000000000000ULL;  // 描述符B
    uint64_t idescE = 0x0000000100000000ULL;  // 扩展描述符
    uint32_t scaleC = 0x3F800000;  // scale值 (1.0f的hex)
    uint32_t tsfa_addr = (uint32_t)(uintptr_t)d_scale_a;  // scale_a地址
    uint32_t tsfb_addr = (uint32_t)(uintptr_t)d_scale_b;  // scale_b地址
    
    // 初始化scale数据
    float scale_data = 1.0f;
    cudaMemcpy(d_scale_a, &scale_data, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_b, &scale_data, sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动kernel
    dim3 block(32);
    dim3 grid(1);
    
    std::cout << "启动PTX kernel..." << std::endl;
    
    tensor_core_mma_kernel<<<grid, block>>>(
        tmem_c,
        desc_a,
        desc_b,
        idescE,
        scaleC,
        tsfa_addr,
        tsfb_addr
    );
    
    // 检查执行结果
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel错误: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "执行错误: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "PTX指令执行成功!" << std::endl;
    
    // 清理
    cudaFree(d_scale_a);
    cudaFree(d_scale_b);
    
    return 0;
}

// 编译命令:
// nvcc -arch=sm_90 -o ptx_test ptx_test.cu