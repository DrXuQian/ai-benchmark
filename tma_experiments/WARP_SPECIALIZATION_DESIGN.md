# Warp Specialization Design for TMA Deformable Attention

## Architecture Overview

### Current Setup
- Block size: 256 threads = 8 warps
- Current: All 8 warps do the same work (TMA load + compute)

### Proposed Warp Specialization
- **Producer Warps**: 2 warps (warp 0-1) = 64 threads
  - Responsible for: TMA loading, coordinate computation
  - Each producer warp handles 4 queries (32 threads / 4 threads per query = 8 queries per warp, but we limit to 4 for balance)

- **Consumer Warps**: 6 warps (warp 2-7) = 192 threads
  - Responsible for: Bilinear interpolation, accumulation
  - Each consumer warp handles compute for multiple queries

### Work Distribution

**Block processes**: 8 queries total
- Producer warps 0-1: Load data for queries 0-7
- Consumer warps 2-7: Compute for queries 0-7

## Data Flow

### 1. Shared Memory Layout

```cpp
// Existing TMA tile storage
__shared__ scalar_t smem_tile[STAGES][num_warps][queries_per_warp][2*2][CHANNELS];
__shared__ barrier warp_bars[STAGES][num_warps];

// NEW: Metadata exchange between producer and consumer
struct SharedPointMeta {
    half2 weighthalf2;  // Attention weight replicated
    __half h_im, w_im;  // Image coordinates
    int hLow, wLow;     // Integer coordinates
    bool within_range;  // Boundary check result
};

__shared__ SharedPointMeta point_metadata[NUM_POINT][8];  // 8 queries per block
__shared__ cuda::barrier<cuda::thread_scope_block> meta_barrier;  // Synchronize producer/consumer
```

### 2. Producer Warp Workflow

**Warp 0-1 (queries 0-7)**:
```
for each level:
    // Load sampling locations & weights (redundant with consumer, but necessary)
    load loc_hw_vec[NUM_POINT]
    load weight_vec[NUM_POINT]

    for each point (with pipelining):
        // Compute coordinates
        h_im, w_im = transform(loc_hw_vec[p], spatial_h, spatial_w)
        hLow, wLow = floor(h_im, w_im)
        within_range = check_bounds(h_im, w_im, spatial_h, spatial_w)

        // Store to shared memory for consumers
        point_metadata[p][query_id].h_im = h_im
        point_metadata[p][query_id].w_im = w_im
        point_metadata[p][query_id].hLow = hLow
        point_metadata[p][query_id].wLow = wLow
        point_metadata[p][query_id].within_range = within_range
        point_metadata[p][query_id].weighthalf2 = half2(weight_vec[p], weight_vec[p])

        // Issue TMA load
        if (is_loader_thread):
            issue_tma_load(stage_id, hLow, wLow, ...)

    // Signal consumers that metadata is ready
    meta_barrier.arrive_and_wait()
```

### 3. Consumer Warp Workflow

**Warp 2-7 (compute for queries 0-7)**:
```
for each level:
    // Wait for producer to finish metadata computation
    meta_barrier.arrive_and_wait()

    // Each consumer warp processes specific queries
    // Warp 2-3: queries 0-3
    // Warp 4-5: queries 4-7
    // Warp 6-7: additional parallelism on queries 0-7

    for each point:
        // Wait for TMA to finish loading this stage
        wait_tma_load(stage_id, ...)

        // Read metadata from shared memory (written by producer)
        auto& meta = point_metadata[p][query_id]
        h_im = meta.h_im
        w_im = meta.w_im
        hLow = meta.hLow
        wLow = meta.wLow
        weighthalf2 = meta.weighthalf2

        if (meta.within_range):
            // Compute bilinear interpolation weights
            lh = h_im - hLow
            lw = w_im - wLow
            hh = 1 - lh
            hw = 1 - lw

            // Load from TMA shared memory and accumulate
            for spatial_idx in [0,1,2,3]:
                vdata = smem_tile[stage_id][warp_id][query_id][spatial_idx][c_col:c_col+8]
                weight = compute_weight(lh, lw, hh, hw, spatial_idx)
                col += weight * weighthalf2 * vdata
```

## Handling Redundant Computations

### Problem: Redundant Work
1. **Sampling location/weight loading**: Both producer and consumer need this
2. **Coordinate computation**: Only producer needs (h_im, w_im, hLow, wLow)
3. **Bilinear weight computation**: Only consumer needs (lh, lw, hh, hw)

### Solution: Data Partitioning

**Producer Warps Compute & Store in SMEM**:
- ✅ h_im, w_im (image coordinates)
- ✅ hLow, wLow (integer coordinates)
- ✅ within_range (boundary check)
- ✅ weighthalf2 (attention weight, replicated for half2 operations)

**Consumer Warps Compute Locally**:
- ❌ Bilinear interpolation weights (lh, lw, hh, hw) - computed from h_im, w_im
- ❌ Per-spatial-point weights - computed on the fly

**Avoided Redundancy**:
- Sampling locations/weights: Loaded by producer, not needed by consumer (only h_im/w_im needed)
- Coordinate transforms: Done once by producer, shared via SMEM

## Performance Considerations

### Benefits
1. **Better overlap**: Producer can prefetch next point while consumer computes current point
2. **Reduced TMA contention**: Only 2 warps issue TMA instead of 8
3. **Cache efficiency**: Producer warps keep TMA descriptors hot in L1

### Costs
1. **Barrier synchronization**: meta_barrier adds latency
2. **SMEM usage**: Additional ~2KB for metadata (8 queries × 8 points × 32 bytes)
3. **Load imbalance**: If producer is slower than consumer (unlikely with TMA)

### Optimization Opportunities
1. **Double buffering metadata**: Use 2 metadata buffers to hide synchronization
2. **Dynamic work distribution**: Consumers can steal work if producer falls behind
3. **Warp-level barriers**: Use warp-level synchronization instead of block-level

## Implementation Steps

1. ✅ Define SharedPointMeta structure
2. ✅ Add metadata shared memory allocation
3. ✅ Implement producer warp logic (warp 0-1)
4. ✅ Implement consumer warp logic (warp 2-7)
5. ✅ Add metadata barriers
6. ✅ Test correctness
7. ✅ Profile and optimize

## Expected Performance Impact

- **Baseline**: Current persistent kernel
- **Target**: 10-15% improvement from:
  - Reduced TMA contention (8 warps → 2 warps issuing TMA)
  - Better instruction-level parallelism (producer/consumer overlap)
  - More efficient use of compute resources
