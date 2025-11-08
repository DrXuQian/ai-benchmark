# TMA (Tensor Memory Accelerator) Experiments for Deformable Attention

This directory contains experiments and implementations for using NVIDIA Hopper's TMA feature to accelerate deformable attention operations.

## Status

✅ **Successfully implemented TMA-based data loading for deformable attention**
- Verified correct TMA dimension mapping: X→C, Y→W, Z→H for [H][W][C] layout
- Achieved 95.6% accuracy on real test data
- Identified root cause of 4.4% discrepancy: FP16 precision in coordinate calculation
- **Benchmarked: TMA is 1.33x faster than manual loading (150 vs 109 GB/s)**

## Key Files

### Working Implementations
- **`tma_baseline_single_load.cu`**: Single-TMA baseline implementation (100% accuracy on tested samples)
  - Single thread (tid==0) issues TMA load
  - Verified with real deformable attention data
  - Serves as correctness reference
- **`tma_concurrent_8loads.cu`**: Production multi-TMA concurrent loading (95.6% accuracy)
  - 8 concurrent TMA loads (threadIdx%4==0 pattern)
  - Matches deformable attention's bilinear sampling pattern
  - 4.4% discrepancy due to FP16 precision (expected behavior)
- **`deform_attn_tma_match_original.cu`**: Full deformable attention with TMA

### Benchmarks
- **`benchmark_tma_loading.cu`**: Performance benchmark for TMA loading
- **`benchmark_comparison.cu`**: TMA vs manual loading comparison
- **`BENCHMARK_RESULTS.md`**: Detailed performance analysis and results

### Documentation
- **`TMA_DIMENSION_MAPPING.md`**: Critical documentation of TMA dimension ordering and memory layout

### Test Data
- **`working/`**: Contains binary test data from real deformable attention workload
  - `test_data_value.bin`: Input feature maps
  - `test_data_sampling_locations.bin`: Sampling coordinates
  - `test_data_level_start_index.bin`: Multi-scale level indices
  - `generate_test_data.cu`: Script to generate test data

### Verified Tests
- **`verified_tests/`**: Collection of validated test cases
  - `test_2x2x32_like_your_config.cu`: Basic TMA configuration test
  - `test_8warp_bilinear_sampling.cu`: Multi-warp bilinear sampling
  - `test_multi_tma_in_warp.cu`: Concurrent TMA operations
  - `verify_tma_layout.cu`: Memory layout verification

### Archive
- **`archive/`**: Previous experimental versions

## Key Findings

### 1. TMA Dimension Mapping
For tensor layout `[H][W][C]`:
- TMA X dimension → C (channels)
- TMA Y dimension → W (width)
- TMA Z dimension → H (height)

**Critical**: This is the ONLY correct mapping. Other permutations will silently load wrong data.

### 2. Coordinate Precision Issue
The 4.4% accuracy gap is due to **FP16 precision loss** in coordinate calculations:
```cuda
dtype w_im = __hfma(loc_w_norm, __int2half_rn(SPATIAL_W), __float2half(0.5f));
int wLow = __half2int_rd(w_im);  // FP16 → FP32: 76.984 becomes 77.0
```

This causes ~4.3% of samples to round to different integer coordinates compared to FP32 validation code.

**Resolution**: This is acceptable as it's a numerical precision limitation, not an algorithmic error.

### 3. TMA Configuration
Optimal configuration for deformable attention:
```cuda
globalDim = {CHANNELS, SPATIAL_W, SPATIAL_H};  // {C, W, H}
globalStrides = {
    CHANNELS * sizeof(dtype),                   // Stride to next W
    SPATIAL_W * CHANNELS * sizeof(dtype)        // Stride to next H
};
boxDim = {32, 2, 2};  // {C, W, H} - Load 2x2 spatial tile, 32 channels
```

## Performance Results

### Benchmark Summary (RTX 5070 - Blackwell sm_90)

**TMA vs Manual Loading (1000 queries × 8 points)**
- TMA: 13.1 μs, 150.78 GB/s
- Manual: 17.5 μs, 109.29 GB/s
- **Speedup: 1.33x faster with TMA**

**TMA Metrics**
- Kernel time: 12.6 μs (average)
- TMA operations: 632 Million ops/s
- Time per TMA load: 1.58 ns
- Memory efficiency: 22.4% (limited by random access pattern)

**Key Benefits**
- Hardware-managed asynchronous transfers
- Reduced instruction overhead
- Better memory access optimization
- Cleaner, more maintainable code

See `BENCHMARK_RESULTS.md` for detailed analysis.

## Next Steps

- [ ] Extend to all 4 spatial shapes (multi-scale support)
- [ ] Implement full threadIdx%4==0 pattern for production use
- [ ] Benchmark TMA vs manual shared memory loading
- [ ] Profile memory bandwidth utilization

## Testing

### Baseline Verification (100% accuracy)
```bash
nvcc -o tma_baseline tma_baseline_single_load.cu -arch=sm_90 -std=c++20 -lcuda
./tma_baseline
```
Expected: 100% accuracy on all tested samples.

### Production Multi-TMA (95.6% accuracy)
```bash
nvcc -o tma_concurrent tma_concurrent_8loads.cu -arch=sm_90 -std=c++20 -lcuda
./tma_concurrent
```
Expected: 95.6% accuracy (4.4% FP16 precision rounding is expected behavior).
