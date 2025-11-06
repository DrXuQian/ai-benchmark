# TMA 2x2x32 Tile Loading Experiments

This repository contains experiments for implementing deformable attention using TMA (Tensor Memory Accelerator) with 2x2x32 tile loading patterns.

## Files

### Microbenchmarks

- **tma_2x2x32_final.cu** - 2x2x32 tile copy microbenchmark
  - Tests basic 2x2x32 tile loading pattern
  - Achieves ~311 GB/s bandwidth on RTX 5070 (SM 12.0)
  - Uses cooperative loading with proper shared memory alignment

### Deformable Attention Kernels

- **deform_attn.cu** - Original deformable attention implementation
  - Baseline reference implementation
  - Uses 4 separate loads for 2x2 neighbors in bilinear interpolation

- **deform_attn_2x2x32_optimized.cu** - Optimized with 2x2x32 tile loading
  - Loads entire 2x2x32 tiles into shared memory cooperatively
  - Reuses shared memory across levels (sequential processing)
  - Parameters: 8 points, 4 levels, 32 channels, 8 output channels per thread

## Build Instructions

### Microbenchmark
```bash
nvcc -arch=sm_90 -O3 tma_2x2x32_final.cu -o tma_2x2x32_final
./tma_2x2x32_final
```

### Deformable Attention
```bash
nvcc -arch=sm_90 -O3 -c deform_attn_2x2x32_optimized.cu -o deform_attn_2x2x32_optimized.o
```

## Current Status

### Working on SM 12.0 (RTX 5070)
- ✅ 2x2x32 tile loading microbenchmark compiles and runs
- ✅ Achieves 311 GB/s bandwidth
- ✅ Correctness verified

### Issue on SM 12.0
- ❌ `cuTensorMapEncodeTiled` API returns `CUDA_ERROR_INVALID_VALUE`
- All TMA descriptor creation attempts fail, even for simple 1D cases
- Tested with CUDA 12.8, Driver 576.88

### Need to Test on Hopper (SM 9.0)
The TMA descriptor API should work properly on Hopper architecture. The experiments need to be validated on:
- H100 (SM 9.0)
- H200 (SM 9.0)

## TMA Implementation Strategy

### Current Approach (Without TMA Descriptor)
Uses cooperative loading with:
- 128-byte aligned shared memory
- Warp-level synchronization
- LDG instructions for cache optimization

### Target Approach (With TMA)
Should use:
```cuda
cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
  [smem], [tensorMap, {y, x, c}], [barrier];
```

## Configuration

### Tensor Layout
- Input: [Batch][Height][Width][Channels]
- Memory order: HWC (Height-major, Width-next, Channel-innermost)
- Example: 100x100x32 tensor

### Tile Configuration
- Tile size: 2x2x32
- Total elements per tile: 128
- Each thread processes 8 output channels
- 32 threads cooperatively load one tile

### Deformable Attention Parameters
- Batch size: 48
- Queries: 15,422
- Heads: 8
- Levels: 4 (multi-scale)
- Points per level: 8 (sampling points)
- Channels: 32
- Output channels per thread: 8

## Key Optimizations

1. **Shared Memory Reuse**: Only allocate 2x2x32 shared memory, reuse across levels
2. **Cooperative Loading**: 32 threads in a warp load one tile together
3. **Aligned Access**: 128-byte alignment for shared memory
4. **Vectorized Loads**: Use LDG for read-only cache utilization
5. **Bilinear Interpolation**: Compute weights once, apply to all channels

## Next Steps

1. Test TMA descriptor creation on Hopper GPU
2. If descriptor works, implement full TMA version with `cp.async.bulk.tensor`
3. Compare performance: baseline vs tile-loading vs TMA
4. Measure speedup for full deformable attention pipeline

## Notes

- SM 12.0 (Blackwell) may have different TMA descriptor requirements
- CUDA 12.8 driver may not fully support SM 12.0 TMA features yet
- Fallback to cooperative loading provides good performance baseline
