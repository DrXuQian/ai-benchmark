# 共享内存分析：STAGES=2 超限问题

## 问题描述

当 `STAGES=2` 时，编译器报错：
```
ptxas error: Entry function uses too much shared data (0x10100 bytes, 0xc000 max)
```

## 根本原因

### 当前配置
- `smem_tile[STAGES][16][8][2][2][32]` - 主要的数据缓冲
- `warp_bars[STAGES][16]` - 每个warp每个stage的barrier

### 内存占用计算

1. **smem_tile 占用**:
   ```
   STAGES=2, WARPS=16, QUERIES_PER_WARP=8, TILE_H=2, TILE_W=2, CHANNELS=32
   = 2 × 16 × 8 × 2 × 2 × 32 × 2 bytes (FP16)
   = 65,536 bytes = 64 KB
   ```

2. **warp_bars 占用**:
   ```
   = 2 × 16 × 8 bytes (barrier大小)
   = 256 bytes
   ```

3. **总共享内存**:
   ```
   = 65,536 + 256 = 65,792 bytes = 64.25 KB = 0x10100
   ```

4. **SM 9.0a 限制**:
   ```
   默认限制: 49,152 bytes = 48 KB = 0xc000
   超出: 16,640 bytes = 16.25 KB (超出 33.9%)
   ```

## 为什么 STAGES=1 可以工作？

```
STAGES=1:
  smem_tile = 1 × 16 × 8 × 2 × 2 × 32 × 2 = 32,768 bytes
  warp_bars = 1 × 16 × 8 = 128 bytes
  Total = 32,896 bytes = 32.12 KB ✓ (在48 KB限制内)
```

## 优化方案

### 方案1: 减少 WARPS (推荐)
```
STAGES=2, WARPS=8:  32,896 bytes = 32.12 KB ✓
```
- **优点**: 正好在限制内，简单直接
- **缺点**: 占用率降低 (512 threads → 256 threads)
- **适用**: 如果kernel不是compute-bound

### 方案2: 减少 QUERIES_PER_WARP (推荐)
```
STAGES=2, WARPS=16, QUERIES=4:  33,024 bytes = 32.25 KB ✓
```
- **优点**: 保持16个warp，更好的occupancy
- **缺点**: 每个warp处理的query减半
- **适用**: 大部分场景

### 方案3: 使用动态共享内存
```c++
extern __shared__ char shared_memory[];
scalar_t* smem_tile = reinterpret_cast<scalar_t*>(shared_memory);
barrier* warp_bars = reinterpret_cast<barrier*>(smem_tile + tile_offset);
```
- **优点**: 灵活性最高，可以在运行时调整
- **缺点**: 代码复杂度增加，需要手动管理偏移

### 方案4: 请求更大的共享内存 (H100)
```c++
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
```
- **优点**: 不需要改代码结构
- **缺点**: 降低occupancy，可能影响性能
- **适用**: H100 GPU，且kernel是memory-bound

## 不同配置对比

| STAGES | WARPS | QUERIES | 共享内存 | 状态 | Threads |
|--------|-------|---------|----------|------|---------|
| 1 | 16 | 8 | 32.12 KB | ✓ | 512 |
| 2 | 16 | 8 | 64.25 KB | ✗ | 512 |
| 2 | 8  | 8 | 32.12 KB | ✓ | 256 |
| 2 | 16 | 4 | 32.25 KB | ✓ | 512 |
| 2 | 8  | 4 | 16.12 KB | ✓ | 256 |

## 推荐配置

### 短期方案 (立即可用)
```c++
// 修改共享内存声明，减少QUERIES_PER_WARP
__shared__ alignas(128) scalar_t smem_tile[STAGES][16][4][2][2][CHANNELS];  // 8 → 4
```

### 长期方案 (性能优化)
1. Profile测试不同配置的性能
2. 根据实际workload选择最优参数
3. 考虑使用H100的大共享内存特性

## H100 特殊优化

H100支持最大228 KB共享内存，可以支持更多STAGES:

| STAGES | 共享内存 | H100状态 |
|--------|----------|----------|
| 1 | 32.12 KB | ✓ |
| 2 | 64.25 KB | ✓ |
| 3 | 96.38 KB | ✓ |
| 4 | 128.50 KB | ✓ |

通过 `cudaFuncSetAttribute` 可以请求更大的共享内存，代价是降低occupancy。
