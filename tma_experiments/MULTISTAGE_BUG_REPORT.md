# Multi-Stage Pipelining Bug Report

## Status: CRITICAL BUG - TMA Loading Garbage Data

### Problem
When implementing STAGES=2 multi-stage software pipelining, TMA is loading incorrect/garbage data instead of the expected values.

**Observed behavior:**
- Expected: `[0.0000, 0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.0600, 0.0700]`
- Actual (TMA): `[0.0001, 0.0000, 37184.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]`

### Root Cause Analysis

The issue is **barrier reuse in multi-stage pipelining**.

#### Background: CUDA mbarrier Semantics
When using TMA with `mbarrier`:
1. Barrier is initialized with expected arrival count (e.g., 32 for a warp)
2. `mbarrier.expect_tx` sets expected transaction bytes
3. TMA automatically signals when transaction completes
4. All 32 threads manually `arrive()`
5. Barrier releases when BOTH conditions met: all arrivals + TMA transaction done
6. **Barrier automatically advances to next phase** (phase-based, alternating)

#### The Bug
In STAGES=2 with 8 points, barrier usage pattern:
- Prologue: Load point 0 → stage 0 barrier
- Iteration 0: Load point 1 → stage 1 barrier, wait stage 0
- Iteration 1: Load point 2 → stage 0 barrier (REUSE!), wait stage 1  ← **BUG HERE**
- Iteration 2: Load point 3 → stage 1 barrier (REUSE!), wait stage 0  ← **BUG HERE**
- ...

**The problem:** After first use, the barrier phases advance, but we're issuing new TMA loads without proper phase synchronization!

#### What Went Wrong

1. **Initial Hypothesis (WRONG)**: Barriers need manual reset with `arrive_and_drop()` or `init()`
   - Tried: `arrive_and_drop()` → CUDA launch failure
   - Tried: `init()` after wait → Deadlock/hang

2. **Actual Issue**: Phase mismatch in barrier reuse
   - First load to stage 0: phase 0
   - Second load to stage 0: still expects phase 0, but barrier is now in phase 1!
   - TMA writes to wrong phase, data lost or corrupted

### Last Known Working Version

**Commit:** `bb1d7291` - "Switch to TMA data path and verify correctness"
- Configuration: STAGES=1 (no pipelining)
- Barrier structure: `warp_bars[16]` (1D array, one barrier per warp)
- Works perfectly: TMA loads correct data

### Changes That Broke It

**Commit:** `81565fb4` - "Implement general multi-stage software pipelining framework"

Key changes:
1. Changed barrier from `warp_bars[MAX_WARPS]` to `warp_bars[STAGES][MAX_WARPS]`
2. Added prologue loop to prefill pipeline
3. Added main loop with prefetch logic
4. Barrier now indexed by both stage and warp

### Attempted Fixes (All Failed)

1. **Barrier reset with `arrive_and_drop()`**
   ```cuda
   if ((threadIdx.x % 32) == 0) {
       warp_bars[stage_idx][warp_id].arrive_and_drop();
   }
   ```
   Result: CUDA Error "unspecified launch failure"

2. **Barrier re-initialization**
   ```cuda
   if ((threadIdx.x % 32) == 0) {
       init(&warp_bars[stage_idx][warp_id], 32);
       asm volatile("fence.proxy.async.shared::cta;");
   }
   __syncwarp();
   ```
   Result: Deadlock/timeout (kernel hangs)

### Proper Solution (TODO)

Several approaches to investigate:

#### Option 1: Phase-Aware Barriers
Use `mbarrier` with explicit phase tracking:
```cuda
__shared__ uint64_t barrier_phases[STAGES][MAX_WARPS];

// Before TMA load
uint64_t expected_phase = barrier_phases[stage_idx][warp_id];
// Issue TMA with current phase
// After wait, increment phase
barrier_phases[stage_idx][warp_id]++;
```

#### Option 2: Double-Buffered Barriers
Use 2 barriers per stage, alternating:
```cuda
__shared__ barrier warp_bars[STAGES][MAX_WARPS][2];  // Extra dimension for ping-pong
int barrier_idx = (iteration_count % 2);
```

#### Option 3: Explicit Reset Pattern
Follow NVIDIA's async copy pattern:
```cuda
// After wait completes
__syncwarp();
if (lane_id == 0) {
    // Reset barrier to initial state for reuse
    warp_bars[stage_idx][warp_id].reset();  // If such API exists
}
__syncwarp();
```

#### Option 4: Single-Phase Per Use
Don't reuse barriers at all - use enough barriers for all iterations:
```cuda
__shared__ barrier warp_bars[NUM_POINT][MAX_WARPS];  // One barrier per point!
```
**Problem:** Excessive shared memory (8 points × 8 warps × barrier size)

### Recommended Next Steps

1. **Research NVIDIA examples** for multi-stage TMA pipelining
   - Check CUTLASS implementations
   - Review Hopper TMA programming guide section on multi-buffering

2. **Simplify test case**
   - Create minimal reproducer with just 2 points, STAGES=2
   - Add extensive barrier state logging

3. **Consider alternative approach**
   - Instead of stage-indexed barriers, use point-indexed barriers
   - Or use software-managed phase tracking

4. **Consult CUDA docs**
   - `cuda::barrier` phase semantics
   - `mbarrier.expect_tx` with multiple phases
   - TMA descriptor reuse patterns

### Files Affected

- `deform_attn_tma.cu` - Main implementation (currently BROKEN with STAGES=2)
- `SMEM_ANALYSIS.md` - Shared memory analysis
- `PROGRESS_SUMMARY.md` - Previous progress (now outdated)

### Performance Impact

**Cannot measure performance** until correctness is restored.

The multi-stage pipelining implementation is fundamentally broken and needs complete barrier usage redesign.

---

**Date:** 2025-11-09
**Severity:** CRITICAL - Blocks all STAGES≥2 testing
**Priority:** P0 - Must fix before any performance evaluation
