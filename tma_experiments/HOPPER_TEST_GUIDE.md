# Hopper Testing Guide

这个指南说明如何在Hopper GPU上测试TMA实现。

## 前提条件

- Hopper GPU (H100 或 H200, SM 9.0)
- CUDA 12.0 或更高版本
- NVIDIA驱动支持Hopper架构

## 快速测试步骤

### 1. 克隆仓库
```bash
git clone https://github.com/DrXuQian/ai-benchmark.git
cd ai-benchmark/tma_experiments
```

### 2. 验证TMA Descriptor API
这是最重要的第一步！

```bash
make test_tma_descriptor
./test_tma_descriptor
```

**预期输出 (成功):**
```
Device: NVIDIA H100 (SM 9.0)

=== Testing TMA Descriptor Creation ===

Creating TMA descriptor for 2x2x32 tile...
Tensor dimensions: 100x100x32
Tile dimensions: 2x2x32

✅ TMA descriptor created successfully!

Launching TMA kernel...
✅ Kernel executed successfully!

First 8 output values: 0.000 0.010 0.020 0.030 0.040 0.050 0.060 0.070
Expected values:       0.000 0.010 0.020 0.030 0.040 0.050 0.060 0.070

✅ TMA test PASSED!

TMA is working correctly on this GPU!
```

**如果失败:** TMA descriptor API不能工作，deform_attn_tma.cu也不会工作。

### 3. 运行Microbenchmark
```bash
make tma_microbench
./tma_microbench
```

**预期输出:**
```
Device: NVIDIA H100 (SM 9.0)

=== TMA-style 2x2x32 Tile Copy Benchmark ===
Tensor: 100x100x32
Tile: 2x2x32
Grid: 2500 blocks (50x50 tiles)
Block: 128 threads
✓ Correctness test passed!

Benchmarking...
Average time: 0.XXX ms
Bandwidth: XXX.XX GB/s
Throughput: XXX.XX M tiles/sec
```

### 4. 编译Deformable Attention内核

```bash
# 编译所有版本
make all
```

这会生成：
- `deform_attn_baseline.o` - 原始实现
- `deform_attn_2x2x32_optimized.o` - 优化的tile loading版本
- `deform_attn_tma.o` - 真正的TMA实现

## 集成到PyTorch

### 使用TMA版本

1. 在PyTorch扩展中链接 `deform_attn_tma.o`
2. 调用前创建TMA descriptors:

```cpp
#include "deform_attn_tma.cu"

// 在初始化时创建descriptors
TmaDescriptor* h_descriptors = new TmaDescriptor[batch_size * num_levels];
CUresult result = createTMADescriptorsForAllBatches(
    h_descriptors,
    d_value_ptrs,        // 每个batch的device指针数组
    h_spatial_shapes,    // [num_levels * 2]
    h_level_start_index, // [num_levels]
    batch_size,
    num_levels,
    channels
);

// 拷贝到device
TmaDescriptor* d_descriptors;
cudaMalloc(&d_descriptors, batch_size * num_levels * sizeof(TmaDescriptor));
cudaMemcpy(d_descriptors, h_descriptors, ...);

// 调用kernel
ms_deformable_im2col_cuda_tma(
    stream,
    data_value,
    d_descriptors,  // 传入TMA descriptors
    ...
);
```

## 性能测试

创建一个完整的测试程序比较三个版本：

```bash
nvcc -arch=sm_90 -O3 \
    deform_attn_baseline.o \
    deform_attn_2x2x32_optimized.o \
    deform_attn_tma.o \
    test_program.cu \
    -o test_deform_attn \
    -lcuda
```

## 预期性能提升

基于2x2x32 tile loading模式：

1. **Baseline → Tile Loading**: ~10-20% 提升
   - 减少重复的内存访问
   - 更好的共享内存利用

2. **Tile Loading → TMA**: ~5-15% 提升
   - 硬件加速的tensor加载
   - 更低的延迟
   - 更高的带宽利用率

总体预期: **15-35%** 性能提升

## 调试提示

### TMA Descriptor创建失败
```
❌ TMA descriptor creation FAILED
Error: CUDA_ERROR_INVALID_VALUE
```

可能原因：
1. 不在Hopper GPU上运行
2. Tensor维度不满足对齐要求
3. Stride不是16字节的倍数

检查：
```bash
nvidia-smi  # 确认是H100/H200
nvcc --version  # 确认CUDA 12.0+
```

### Kernel执行错误
```
Kernel execution error: an illegal instruction was encountered
```

可能原因：
1. TMA descriptor没有正确初始化
2. 编译时arch不对 (必须是sm_90)
3. Shared memory地址没有128字节对齐

### 性能不如预期

检查：
1. 是否使用了`-O3`编译优化
2. GPU是否在运行其他任务
3. Tensor大小是否足够大以隐藏延迟

## 下一步

TMA测试通过后：

1. 集成到完整的deformable attention PyTorch扩展
2. 端到端性能测试
3. 与CUTLASS等优化库对比
4. 尝试不同的tile size配置

## 问题报告

如果遇到问题，请提供：
1. GPU型号和SM版本
2. CUDA版本和驱动版本
3. 完整的错误信息
4. `test_tma_descriptor`的输出

Repository: https://github.com/DrXuQian/ai-benchmark
