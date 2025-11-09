#!/bin/bash

# Performance benchmark script for comparing STAGES=1, STAGES=2, and baseline deformable attention

set -e

echo "========================================"
echo "TMA Multi-Stage Performance Benchmark"
echo "========================================"
echo ""

# Test configuration
BATCH=1
SPATIAL_SIZE=20522
NUM_QUERY=20522
NUM_HEADS=1
CHANNELS=32
NUM_LEVELS=4
NUM_POINTS=8
IM2COL_STEP=1
DATA_DIR="working_simple"

# Number of warmup and benchmark iterations
WARMUP_ITERS=5
BENCH_ITERS=20

echo "Configuration:"
echo "  batch=$BATCH, spatial_size=$SPATIAL_SIZE, num_query=$NUM_QUERY"
echo "  num_heads=$NUM_HEADS, channels=$CHANNELS, num_levels=$NUM_LEVELS, num_points=$NUM_POINTS"
echo "  Warmup iterations: $WARMUP_ITERS"
echo "  Benchmark iterations: $BENCH_ITERS"
echo ""

# Function to compile with specific STAGES
compile_stages() {
    local stages=$1
    local threads=$2
    local max_warps=$3
    local output_binary=$4

    echo "Compiling STAGES=$stages (threads=$threads, warps=$max_warps)..."

    # Modify source file temporarily
    sed -i "s/const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3, const int STAGES=[0-9]*/const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3, const int STAGES=$stages/g" deform_attn_tma.cu
    sed -i "s/const int THREADS_IN_ONE_BLOCK=[0-9]*/const int THREADS_IN_ONE_BLOCK=$threads/g" deform_attn_tma.cu
    sed -i "s/constexpr int MAX_WARPS = [0-9]*/constexpr int MAX_WARPS = $max_warps/g" deform_attn_tma.cu

    # Compile
    nvcc -O3 -std=c++20 -arch=sm_90a -lcuda -o "$output_binary" deform_attn_tma.cu 2>&1 | grep -E "error|warning" || true

    if [ ! -f "$output_binary" ]; then
        echo "ERROR: Compilation failed for STAGES=$stages"
        exit 1
    fi

    echo "  ✓ Compiled successfully"

    # Restore original
    git checkout deform_attn_tma.cu 2>/dev/null || true
}

# Function to run benchmark
run_benchmark() {
    local binary=$1
    local name=$2

    echo ""
    echo "--- Benchmarking: $name ---"

    # Warmup
    echo "Warming up ($WARMUP_ITERS iterations)..."
    for i in $(seq 1 $WARMUP_ITERS); do
        ./"$binary" batch=$BATCH spatial_size=$SPATIAL_SIZE num_query=$NUM_QUERY \
            num_heads=$NUM_HEADS channels=$CHANNELS num_levels=$NUM_LEVELS \
            num_points=$NUM_POINTS im2col_step=$IM2COL_STEP dir=$DATA_DIR \
            > /dev/null 2>&1
    done

    # Benchmark
    echo "Running benchmark ($BENCH_ITERS iterations)..."

    total_time=0
    for i in $(seq 1 $BENCH_ITERS); do
        output=$(./"$binary" batch=$BATCH spatial_size=$SPATIAL_SIZE num_query=$NUM_QUERY \
            num_heads=$NUM_HEADS channels=$CHANNELS num_levels=$NUM_LEVELS \
            num_points=$NUM_POINTS im2col_step=$IM2COL_STEP dir=$DATA_DIR 2>&1)

        # Extract kernel time (assumes output contains "Kernel time: X.XX ms")
        time=$(echo "$output" | grep -oP "Kernel time: \K[0-9.]+")

        if [ -z "$time" ]; then
            echo "Warning: Could not extract time from iteration $i"
            continue
        fi

        total_time=$(echo "$total_time + $time" | bc)

        # Show progress
        if [ $((i % 5)) -eq 0 ]; then
            echo "  Progress: $i/$BENCH_ITERS iterations"
        fi
    done

    # Calculate statistics
    avg_time=$(echo "scale=4; $total_time / $BENCH_ITERS" | bc)

    echo ""
    echo "Results for $name:"
    echo "  Average kernel time: $avg_time ms"
    echo "  Total time ($BENCH_ITERS iters): $total_time ms"

    # Store result
    echo "$avg_time" > /tmp/benchmark_${name// /_}.txt
}

# Compile all versions
echo "========================================"
echo "Step 1: Compiling all versions"
echo "========================================"
echo ""

compile_stages 1 256 8 "deform_attn_stage1"
compile_stages 2 256 8 "deform_attn_stage2"

# Also compile baseline (non-TMA version) if it exists
if [ -f "deform_attn.cu" ]; then
    echo "Compiling baseline (non-TMA)..."
    nvcc -O3 -std=c++20 -arch=sm_90a -o deform_attn_baseline deform_attn.cu 2>&1 | grep -E "error|warning" || true
    BASELINE_EXISTS=true
else
    echo "Note: deform_attn.cu not found, skipping baseline"
    BASELINE_EXISTS=false
fi

echo ""
echo "========================================"
echo "Step 2: Running benchmarks"
echo "========================================"

# Run benchmarks
run_benchmark "deform_attn_stage1" "STAGES=1 (256 threads, 8 warps)"
run_benchmark "deform_attn_stage2" "STAGES=2 (256 threads, 8 warps)"

if [ "$BASELINE_EXISTS" = true ]; then
    run_benchmark "deform_attn_baseline" "Baseline (non-TMA)"
fi

# Compare results
echo ""
echo "========================================"
echo "Performance Comparison"
echo "========================================"
echo ""

stage1_time=$(cat /tmp/benchmark_STAGES=1_\(256_threads,_8_warps\).txt 2>/dev/null || echo "N/A")
stage2_time=$(cat /tmp/benchmark_STAGES=2_\(256_threads,_8_warps\).txt 2>/dev/null || echo "N/A")

echo "| Version | Avg Time (ms) | Speedup vs STAGES=1 |"
echo "|---------|---------------|---------------------|"

if [ "$stage1_time" != "N/A" ]; then
    echo "| STAGES=1 | $stage1_time | 1.00x (baseline) |"
fi

if [ "$stage2_time" != "N/A" ] && [ "$stage1_time" != "N/A" ]; then
    speedup=$(echo "scale=2; $stage1_time / $stage2_time" | bc)
    echo "| STAGES=2 | $stage2_time | ${speedup}x |"
fi

if [ "$BASELINE_EXISTS" = true ]; then
    baseline_time=$(cat /tmp/benchmark_Baseline_\(non-TMA\).txt 2>/dev/null || echo "N/A")
    if [ "$baseline_time" != "N/A" ] && [ "$stage1_time" != "N/A" ]; then
        speedup_vs_baseline=$(echo "scale=2; $baseline_time / $stage1_time" | bc)
        echo "| Baseline | $baseline_time | ${speedup_vs_baseline}x (vs STAGES=1) |"
    fi
fi

echo ""
echo "Benchmark complete!"
echo "Temporary binaries: deform_attn_stage1, deform_attn_stage2"
