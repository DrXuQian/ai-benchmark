# Warp Specialization Implementation Guide

This directory contains multiple iterations of warp specialization optimizations for the TMA-based deformable attention kernel.

## Quick Start

### Working Implementation
The baseline **Phase 1** implementation is fully working and verified:
```bash
nvcc -O3 -std=c++20 -arch=sm_90a -lcuda -o deform_attn_warp_spec deform_attn_tma_warp_spec.cu
./deform_attn_warp_spec batch=48 spatial_size=20522 num_query=20522 \
    num_heads=1 channels=32 num_levels=4 num_points=8 im2col_step=1 dir=working
```

## File Organization

### Implementation Files

| File | Status | Description |
|------|--------|-------------|
| `deform_attn_tma_warp_spec.cu` | ✅ Working | Phase 1: Basic warp specialization with 2:6 producer:consumer ratio |
| `deform_attn_tma_warp_spec_phase2.cu` | ⚠️ Needs Debug | Phase 2: Adds shared memory metadata to eliminate redundant computation |
| `deform_attn_tma_warp_spec_phase3.cu` | ❌ Abandoned | Phase 3: Flag-based sync (causes deadlock) |
| `deform_attn_tma_warp_spec_phase5_1prod.cu` | ⚠️ Needs Debug | 1:7 producer:consumer ratio |
| `deform_attn_tma_warp_spec_phase5_2prod.cu` | ⚠️ Needs Debug | 2:6 ratio (same as Phase 2) |
| `deform_attn_tma_warp_spec_phase5_3prod.cu` | ⚠️ Needs Debug | 3:5 ratio |
| `deform_attn_tma_warp_spec_phase5_4prod.cu` | ⚠️ Needs Debug | 4:4 ratio (equal split) |

### Documentation Files

| File | Purpose |
|------|---------|
| `WARP_SPECIALIZATION_DESIGN.md` | Original design document with architecture overview |
| `OPTIMIZATION_ROADMAP.md` | Detailed plan for 10 optimization phases |
| `OPTIMIZATION_SUMMARY.md` | Progress summary and results |
| `WARP_SPEC_README.md` | This file - navigation guide |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `test_producer_ratios.sh` | Automated testing of different producer/consumer ratios |

## Optimization Phases

### ✅ Phase 1: Basic Warp Specialization
- **File**: `deform_attn_tma_warp_spec.cu`
- **Changes**: Split warps into producers (TMA loads) and consumers (compute)
- **Status**: Fully working, verified correct
- **Key Innovation**: 2 producer warps, 6 consumer warps with block-level barrier

### ⚠️ Phase 2: Shared Memory Metadata
- **File**: `deform_attn_tma_warp_spec_phase2.cu`
- **Changes**: Producers write metadata to shared memory, consumers read it
- **Status**: Compiles, but hangs on execution with large batch sizes
- **Benefit**: Eliminates ~75% of redundant coordinate computation
- **Issue**: Barrier synchronization problem in persistent kernel loop

### ❌ Phase 3: Flag-Based Synchronization
- **File**: `deform_attn_tma_warp_spec_phase3.cu`
- **Changes**: Replace barrier with per-point flags and spin-wait
- **Status**: Abandoned due to deadlock
- **Lesson**: Block-level barriers are actually efficient; complex sync adds risk

### ⚠️ Phase 5: Producer/Consumer Ratio Tuning
- **Files**: `deform_attn_tma_warp_spec_phase5_*prod.cu`
- **Changes**: Test 1:7, 2:6, 3:5, 4:4 ratios
- **Status**: All compile successfully, same runtime issue as Phase 2
- **Next**: Need to fix barrier issue before benchmarking

## Key Concepts

### Warp Specialization Pattern
```
Block of 256 threads = 8 warps

Producer Warps (0-1):
  - Compute sampling coordinates
  - Issue TMA loads
  - Write metadata to shared memory

Consumer Warps (2-7):
  - Read metadata from shared memory
  - Wait for TMA completion
  - Perform bilinear interpolation
  - Accumulate results
```

### Shared Memory Layout (Phase 2+)
```cpp
struct SharedPointMeta {
    half2 weighthalf2;      // Replicated attention weight
    scalar_t h_im, w_im;    // Image coordinates
    int32_t hLow, wLow;     // Integer coordinates
    int within_range;       // Boundary check
};
__shared__ SharedPointMeta<scalar_t> shared_meta[NUM_POINT];  // 8 points
```

Memory overhead: ~256 bytes

### Synchronization Pattern
```cpp
// Initialize barrier for all threads
if (tid == 0) {
    init(&prod_cons_barrier, blockDim.x);
}
__syncthreads();

// Producer writes metadata
if (is_producer_warp && warp_id == 0 && lane_id == 0) {
    shared_meta[p] = computed_metadata;
}

// Synchronize before consumers read
prod_cons_barrier.arrive_and_wait();

// All warps read and use metadata
const auto& smeta = shared_meta[p];
```

## Known Issues

### Runtime Hang with Large Batch Sizes
**Symptom**: Kernels hang when using barrier synchronization with persistent kernel loop

**Affected Files**:
- Phase 2 and all Phase 5 variants

**Possible Causes**:
1. Not all threads participating in barrier correctly
2. Persistent loop causing early thread exit
3. Query count mismatch with grid size

**Debug Steps**:
1. Test with smaller batch sizes
2. Add printf debugging for barrier arrivals
3. Verify persistent loop bounds
4. Check thread divergence patterns

## Future Optimizations

See `OPTIMIZATION_ROADMAP.md` for detailed plans. Summary:

| Phase | Optimization | Expected Gain | Difficulty |
|-------|--------------|---------------|------------|
| 4 | Deeper pipeline (STAGES=3-4) | 5-15% | Medium |
| 6 | Warp shuffle primitives | 3-8% | Low |
| 7 | Prefetch next query | 5-10% | Medium |
| 8 | Optimize bilinear interpolation | 2-5% | Low |
| 10 | Multi-level TMA prefetching | 10-15% | High |

**Total potential improvement**: 25-53% over Phase 1

## Testing Commands

### Correctness Test
```bash
./deform_attn_warp_spec batch=48 spatial_size=20522 num_query=20522 \
    num_heads=1 channels=32 num_levels=4 num_points=8 im2col_step=1 dir=working
```

### Quick Test (small batch)
```bash
./deform_attn_warp_spec batch=1 spatial_size=100 num_query=100 \
    num_heads=1 channels=32 num_levels=4 num_points=8 im2col_step=1 dir=working
```

### Batch Ratio Testing
```bash
./test_producer_ratios.sh
```

## Code Locations

### Key Functions
- **Kernel**: `ms_deformable_im2col_cuda` (line ~130)
- **TMA Load Helper**: `issue_tma_load` (line ~89)
- **TMA Wait Helper**: `wait_tma_load` (line ~114)

### Key Modifications from Baseline
1. **Warp role identification** (line ~193-199)
2. **Shared memory metadata** (line ~208-211 in Phase 2+)
3. **Producer/consumer split logic** (line ~278-320)
4. **Barrier synchronization** (line ~213-227 in Phase 1/2)

## Performance Expectations

Based on implementation:
- **Phase 1 baseline**: 2 producer + 6 consumer warps
- **Phase 2 (if working)**: +5-10% from eliminating redundant work
- **Optimal ratio (Phase 5)**: +2-5% from better balance
- **Combined future opts**: +20-40% additional improvement possible

## Development History

1. **Nov 11, 22:09**: Phase 1 completed and verified
2. **Nov 11, 22:31**: Phase 2 implemented (shared memory optimization)
3. **Nov 11, 22:45**: Phase 3 attempted and abandoned (deadlock issues)
4. **Nov 12, 06:50**: Phase 5 variants created (ratio tuning)
5. **Nov 12, 06:58**: Documentation finalized

## References

- **Design Doc**: `WARP_SPECIALIZATION_DESIGN.md`
- **Roadmap**: `OPTIMIZATION_ROADMAP.md`
- **Progress**: `OPTIMIZATION_SUMMARY.md`
- **Base Implementation**: `deform_attn_tma_persistent.cu`

## Contributors Notes

If continuing this work:
1. **Priority 1**: Debug the barrier hang issue in Phase 2
2. **Priority 2**: Benchmark Phase 5 variants once fixed
3. **Priority 3**: Implement Phase 6 (warp shuffle) - independent of barrier issue
4. **Priority 4**: Continue through optimization roadmap

For questions or issues, refer to the detailed analysis in `OPTIMIZATION_SUMMARY.md`.
