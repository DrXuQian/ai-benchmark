# TMA Descriptor Prefetch Optimization

## Problem

When using multiple TMA descriptors (e.g., 192 descriptors for 48 batches × 4 levels), accessing descriptors from global memory introduces latency. Each TMA operation needs to:

1. Load the descriptor from memory (if not in cache)
2. Decode the descriptor
3. Issue the TMA load

Descriptor access can become a bottleneck, especially when each block accesses different descriptors.

## Solution: Prefetch TMA Descriptors

NVIDIA Hopper provides the `prefetch.tensormap` PTX instruction to bring descriptors into L2 cache before use.

## Performance Impact

| Version | Bandwidth (GB/s) | Latency (ms) | TMA ops/sec | Memory Efficiency |
|---------|-----------------|--------------|-------------|-------------------|
| **Without prefetch** | 314 | 1.167 | 1.32 B | 46.7% |
| **With prefetch** | **350-407** | **0.94-1.10** | **1.40-1.57 B** | **52-61%** |
| **Improvement** | **+11-30%** | **-6-20%** | **+6-19%** | **+5-14%** |

Performance varies due to GPU frequency scaling and thermal conditions.

## Implementation Approaches

### Approach 1: `prefetch.tensormap` Instruction (Recommended ✅)

**Best for:** Multi-batch scenarios where each block accesses a small set of descriptors

```cuda
__global__ void tma_kernel(const CUtensorMap* tma_descs_all, ...) {
    const int b_col = blockIdx.x / num_query;  // Batch ID

    // Prefetch all descriptors this block will use
    if (threadIdx.x == 0) {  // Only one thread needs to prefetch
        #pragma unroll
        for (int l = 0; l < NUM_LEVELS; l++) {
            const int desc_idx = b_col * NUM_LEVELS + l;
            const CUtensorMap* desc_ptr = &tma_descs_all[desc_idx];

            asm volatile(
                "prefetch.tensormap [%0];\n\t"
                :: "l"(reinterpret_cast<uint64_t>(desc_ptr))
            );
        }
    }

    // Now use descriptors - they're in L2 cache
    for (int l = 0; l < NUM_LEVELS; l++) {
        const int desc_idx = b_col * NUM_LEVELS + l;
        const CUtensorMap* tma_desc = &tma_descs_all[desc_idx];
        // Issue TMA with cached descriptor
    }
}
```

**Advantages:**
- Simple to implement
- Low overhead (single instruction per descriptor)
- Hardware-optimized for tensormap data
- Non-blocking (prefetch happens asynchronously)

**When to use:**
- Each block uses a small number of descriptors (< 10)
- Descriptors vary per block (like multi-batch)
- Want maximum performance with minimal code change

---

### Approach 2: `__grid_constant__` Parameters

**Best for:** Single descriptor or very few descriptors shared by all blocks

```cuda
__global__ void tma_kernel(
    const __grid_constant__ CUtensorMap tma_desc,  // Automatically cached
    ...
) {
    // Descriptor is guaranteed in constant cache
    // No prefetch needed
}
```

**Advantages:**
- Zero prefetch overhead
- Guaranteed constant cache residency
- Compiler-optimized

**Limitations:**
- ❌ All blocks must use the **same** descriptor(s)
- ❌ Limited to ~4KB of constant data per kernel launch
- ❌ Not suitable for multi-batch with 192 descriptors

**When to use:**
- Single-batch, single-scale scenarios
- All blocks access identical descriptors
- Maximum performance for simple cases

---

### Approach 3: Load to Shared Memory

**Best for:** Very large descriptor sets or when multiple warps need same descriptors

```cuda
__global__ void tma_kernel(const CUtensorMap* tma_descs_all, ...) {
    __shared__ CUtensorMap smem_descs[NUM_LEVELS];

    const int b_col = blockIdx.x / num_query;

    // Cooperative load by multiple threads
    if (threadIdx.x < NUM_LEVELS) {
        const int desc_idx = b_col * NUM_LEVELS + threadIdx.x;
        smem_descs[threadIdx.x] = tma_descs_all[desc_idx];
    }
    __syncthreads();

    // Use descriptors from shared memory
    for (int l = 0; l < NUM_LEVELS; l++) {
        const CUtensorMap* tma_desc = &smem_descs[l];
        // Issue TMA
    }
}
```

**Advantages:**
- Fastest access after loading (on-chip)
- Good for repeated descriptor access
- Cooperative loading can be fast

**Disadvantages:**
- Uses precious shared memory (128 bytes per descriptor)
- Requires `__syncthreads()` barrier
- Synchronization overhead
- TMA can't directly use shared memory descriptors (need address conversion)

**When to use:**
- Descriptors accessed many times per block
- Have spare shared memory
- Can amortize sync overhead

---

### Approach 4: Manual L2 Cache Prefetch

**Best for:** Generic prefetching when `prefetch.tensormap` not available

```cuda
__global__ void tma_kernel(const CUtensorMap* tma_descs_all, ...) {
    const int b_col = blockIdx.x / num_query;

    // Generic L2 prefetch (not tensormap-specific)
    if (threadIdx.x == 0) {
        for (int l = 0; l < NUM_LEVELS; l++) {
            const int desc_idx = b_col * NUM_LEVELS + l;
            const void* desc_ptr = &tma_descs_all[desc_idx];

            asm volatile(
                "prefetch.global.L2 [%0];\n\t"
                :: "l"(desc_ptr)
            );
        }
    }
}
```

**Advantages:**
- Works on older architectures
- Generic prefetch mechanism

**Disadvantages:**
- Less optimized than `prefetch.tensormap`
- May not prefetch entire descriptor structure
- No hardware specialization for TMA metadata

---

## Recommendation

For **multi-batch multi-scale TMA** (current implementation):

✅ **Use Approach 1: `prefetch.tensormap`**

Reasons:
1. Simple 4-line addition to kernel
2. 11-30% performance improvement
3. Hardware-optimized for TMA descriptors
4. Zero shared memory overhead
5. Works with 192 descriptors seamlessly

## Implementation in `tma_multiscale_multibatch.cu`

Current implementation uses `prefetch.tensormap` at line 87-97:

```cuda
// Prefetch TMA descriptors for this batch's 4 levels into L2 cache
if (lane_id == 0 && p_col == 0) {
    #pragma unroll
    for (int l = 0; l < NUM_LEVELS; l++) {
        const int desc_idx = b_col * NUM_LEVELS + l;
        const CUtensorMap* desc_ptr = &tma_descs_all[desc_idx];
        asm volatile(
            "prefetch.tensormap [%0];\n\t"
            :: "l"(reinterpret_cast<uint64_t>(desc_ptr))
        );
    }
}
```

Only one thread (warp 0, lane 0) prefetches all 4 descriptors for the current batch. This happens once per block before any TMA operations.

## Best Practices

1. **Prefetch early**: Before any computation that might hide latency
2. **Prefetch once**: One thread per block is sufficient
3. **Use `#pragma unroll`**: Let compiler optimize the prefetch loop
4. **Prefetch all needed descriptors**: Don't prefetch just-in-time (latency not hidden)

## Alternative: Just-in-Time Prefetch

For scenarios with unpredictable descriptor access:

```cuda
// Prefetch right before use (less effective)
for (int l = 0; l < NUM_LEVELS; l++) {
    const int desc_idx = b_col * NUM_LEVELS + l;

    // Prefetch this level's descriptor
    if (lane_id == 0) {
        asm volatile(
            "prefetch.tensormap [%0];\n\t"
            :: "l"(reinterpret_cast<uint64_t>(&tma_descs_all[desc_idx]))
        );
    }

    // Use it (may still have some latency)
    // ...
}
```

Less effective because prefetch latency not fully hidden by computation.

## Hardware Details

On NVIDIA Hopper/Blackwell:
- `prefetch.tensormap` is a hint to the memory system
- Brings 128-byte cache line containing descriptor to L2
- Non-blocking operation
- Multiple prefetches can be in-flight
- L2 cache size: 24-96 MB (depends on GPU model)
- RTX 5070: 48 MB L2 (can hold ~393,216 descriptors theoretically)

## Verification

Performance verified on RTX 5070 (Blackwell):
- 192 TMA descriptors (48 batches × 4 levels)
- With prefetch: 350-407 GB/s (avg ~380 GB/s)
- Without prefetch: 314 GB/s
- **Average improvement: ~21%**

## References

- NVIDIA Hopper TMA Programming Guide
- PTX ISA: `prefetch.tensormap` instruction
- Implementation: `tma_multiscale_multibatch.cu:84-98`
