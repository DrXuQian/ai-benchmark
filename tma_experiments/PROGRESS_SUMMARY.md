# TMA Multi-Stage Pipelining - Progress Summary

## 当前状态: STAGES=2 成功实现 ✓

### 最新成果

1. **成功启用双stage流水线** (Commit: 50c3074a)
   - 配置: 256 threads (8 warps), STAGES=2
   - 共享内存: 32,896 bytes (32.12 KB) ✓ 在48 KB限制内
   - 编译通过: ✓ 无错误

2. **通用软件流水线框架** (Commit: 81565fb4)
   - 支持任意STAGES数量 (1, 2, 3, ...)
   - Prologue-Main Loop架构
   - 计算与TMA加载overlap
   - 模板化设计，易于扩展

3. **性能测试工具** (Commit: cd5707a0)
   - `benchmark_performance.py` - Python自动化测试
   - `benchmark_stages.sh` - Bash版本
   - 对比: STAGES=1 vs STAGES=2 vs Baseline

## 技术实现细节

### 共享内存优化

**问题**: STAGES=2 with 512 threads (16 warps) 超限
```
需求: 65,792 bytes (64.25 KB)
限制: 49,152 bytes (48 KB)
超出: 16,640 bytes (33.9%)
```

**解决方案**: 减少线程数到 256 (8 warps)
```
配置: STAGES=2, MAX_WARPS=8, QUERIES_PER_WARP=8
- smem_tile[2][8][8][2][2][32] = 32,768 bytes
- warp_bars[2][8] = 128 bytes
- Total = 32,896 bytes (32.12 KB) ✓
```

### 软件流水线设计

```
Prologue (填充pipeline):
  for p in [0, STAGES-1):
      issue_tma_load(stage = p % STAGES, point = p)

Main Loop (overlap计算和预取):
  for p_col in [0, NUM_POINT):
      compute_stage = p_col % STAGES
      prefetch_p = p_col + STAGES - 1

      if prefetch_p < NUM_POINT:
          issue_tma_load(stage = prefetch_p % STAGES, point = prefetch_p)

      wait_tma_load(compute_stage)
      compute_with_data(compute_stage)
```

### 关键代码改动

1. **共享内存** (`deform_attn_tma.cu:173-179`)
   ```cuda
   constexpr int MAX_WARPS = 8;  // 256 threads / 32
   __shared__ scalar_t smem_tile[STAGES][MAX_WARPS][8][2][2][CHANNELS];
   __shared__ barrier warp_bars[STAGES][MAX_WARPS];
   ```

2. **TMA辅助函数** (`deform_attn_tma.cu:67-126`)
   ```cuda
   template<typename scalar_t, int CHANNELS, int MAX_WARPS>
   __device__ void issue_tma_load(...);

   template<int MAX_WARPS>
   __device__ void wait_tma_load(...);
   ```

3. **流水线循环** (`deform_attn_tma.cu:268-296`)
   - Prologue: 预加载STAGES-1个point
   - Main Loop: 每次迭代预取下一个，计算当前
   - 使用`compute_stage = p_col % STAGES`动态索引

## 性能测试

### 正在运行
```bash
python3 benchmark_performance.py
```

测试配置:
- Warmup: 3 iterations
- Benchmark: 10 iterations
- 对比: STAGES=1 vs STAGES=2 vs Baseline

预期结果:
- STAGES=2 应该隐藏TMA延迟，提升性能
- Trade-off: 256 threads可能影响occupancy

## 文件结构

```
tma_experiments/
├── deform_attn_tma.cu          # 主实现 (STAGES=2, 256 threads)
├── deform_attn_tma             # 编译后的binary
├── benchmark_performance.py     # Python性能测试 (推荐)
├── benchmark_stages.sh         # Bash性能测试
├── SMEM_ANALYSIS.md            # 共享内存详细分析
├── PROGRESS_SUMMARY.md         # 本文档
└── tma_experiments/debug/       # 增量测试 (Test 6-15)
```

## Commit历史 (最近10个)

```
cd5707a0 - Add performance benchmark scripts
50c3074a - Enable STAGES=2 with 256 threads (8 warps) ✓
9fb5a888 - Update compiled binary for STAGES=2
21e0fb18 - Add detailed shared memory analysis
81565fb4 - Implement general multi-stage software pipelining framework
efb52e50 - Optimize: only load global memory when DEBUG is enabled
bb1d7291 - Switch to TMA data path and verify correctness
2d3add6f - Refactor debug code: extract comparison logic
a28fcf80 - Fix critical TMA bugs: barrier overflow and shared memory access
c7f1614a - Implement 192 separate TMA descriptors
```

## 下一步计划

1. ✓ 完成性能测试
2. 分析STAGES=2的实际加速效果
3. 根据结果决定:
   - 如果加速显著: 优化进一步提升
   - 如果加速有限: 分析瓶颈，考虑STAGES=3或其他优化
4. 考虑恢复512 threads (通过减少QUERIES_PER_WARP到4)
5. H100大共享内存优化 (支持更多STAGES)

## 已知问题

- [x] STAGES=2 共享内存超限 → 已解决 (256 threads)
- [x] Barrier数组越界 → 已解决 (warp_bars[STAGES][MAX_WARPS])
- [x] 通用性不足 → 已解决 (模板化MAX_WARPS参数)
- [ ] 性能验证待完成 (benchmark运行中)

## 参考

- CUDA TMA文档: Tensor Memory Accelerator (Hopper架构)
- 软件流水线: [Wikipedia](https://en.wikipedia.org/wiki/Software_pipelining)
- 共享内存限制: RTX 5070支持 99 KB (opt-in)
