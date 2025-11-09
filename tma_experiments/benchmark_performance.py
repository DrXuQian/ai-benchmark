#!/usr/bin/env python3
"""
Performance benchmark for comparing STAGES=1 vs STAGES=2 vs baseline
"""

import subprocess
import re
import time
import statistics
from pathlib import Path

# Test configuration
CONFIG = {
    'batch': 1,
    'spatial_size': 20522,
    'num_query': 20522,
    'num_heads': 1,
    'channels': 32,
    'num_levels': 4,
    'num_points': 8,
    'im2col_step': 1,
    'dir': 'working_simple'
}

WARMUP_ITERS = 3
BENCH_ITERS = 10


def compile_version(stages, threads, max_warps, output_name):
    """Compile a specific version"""
    print(f"Compiling STAGES={stages} (threads={threads}, warps={max_warps})...")

    # Backup original
    subprocess.run(['cp', 'deform_attn_tma.cu', 'deform_attn_tma.cu.bak'], check=False)

    # Modify source
    with open('deform_attn_tma.cu', 'r') as f:
        content = f.read()

    # Replace STAGES
    content = re.sub(
        r'const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3, const int STAGES=\d+',
        f'const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3, const int STAGES={stages}',
        content
    )

    # Replace THREADS_IN_ONE_BLOCK
    content = re.sub(
        r'const int THREADS_IN_ONE_BLOCK=\d+',
        f'const int THREADS_IN_ONE_BLOCK={threads}',
        content
    )

    # Replace MAX_WARPS
    content = re.sub(
        r'constexpr int MAX_WARPS = \d+',
        f'constexpr int MAX_WARPS = {max_warps}',
        content
    )

    with open('deform_attn_tma.cu', 'w') as f:
        f.write(content)

    # Compile
    result = subprocess.run(
        ['nvcc', '-O3', '-std=c++20', '-arch=sm_90a', '-lcuda', '-o', output_name, 'deform_attn_tma.cu'],
        capture_output=True,
        text=True
    )

    # Restore original
    subprocess.run(['mv', 'deform_attn_tma.cu.bak', 'deform_attn_tma.cu'], check=False)

    if result.returncode != 0:
        print(f"ERROR: Compilation failed!")
        print(result.stderr)
        return False

    print(f"  ✓ Compiled successfully")
    return True


def run_benchmark(binary_path, name):
    """Run benchmark for a specific binary"""
    print(f"\n--- Benchmarking: {name} ---")

    # Build command
    cmd = [f'./{binary_path}']
    for key, value in CONFIG.items():
        cmd.append(f'{key}={value}')

    # Warmup
    print(f"Warming up ({WARMUP_ITERS} iterations)...")
    for i in range(WARMUP_ITERS):
        subprocess.run(cmd, capture_output=True, check=False)

    # Benchmark
    print(f"Running benchmark ({BENCH_ITERS} iterations)...")
    times = []

    for i in range(BENCH_ITERS):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Extract kernel time
        match = re.search(r'Kernel time:\s*([0-9.]+)\s*ms', result.stdout)
        if match:
            kernel_time = float(match.group(1))
            times.append(kernel_time)
        else:
            print(f"Warning: Could not extract time from iteration {i+1}")

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{BENCH_ITERS} iterations")

    if not times:
        print(f"ERROR: No valid timing data collected for {name}")
        return None

    # Calculate statistics
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults for {name}:")
    print(f"  Average: {avg_time:.4f} ms")
    print(f"  Std dev: {std_time:.4f} ms")
    print(f"  Min:     {min_time:.4f} ms")
    print(f"  Max:     {max_time:.4f} ms")

    return {
        'name': name,
        'avg': avg_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'times': times
    }


def main():
    print("=" * 60)
    print("TMA Multi-Stage Performance Benchmark")
    print("=" * 60)
    print(f"\nConfiguration: {CONFIG}")
    print(f"Warmup iterations: {WARMUP_ITERS}")
    print(f"Benchmark iterations: {BENCH_ITERS}\n")

    results = {}

    # Compile and benchmark STAGES=1
    if compile_version(1, 256, 8, 'deform_attn_stage1'):
        results['stage1'] = run_benchmark('deform_attn_stage1', 'STAGES=1 (256 threads, 8 warps)')

    # Compile and benchmark STAGES=2
    if compile_version(2, 256, 8, 'deform_attn_stage2'):
        results['stage2'] = run_benchmark('deform_attn_stage2', 'STAGES=2 (256 threads, 8 warps)')

    # Benchmark baseline if exists
    if Path('deform_attn').exists():
        print("\nFound baseline binary, benchmarking...")
        results['baseline'] = run_benchmark('deform_attn', 'Baseline (original)')

    # Print comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    print()

    if 'stage1' in results and results['stage1']:
        baseline_time = results['stage1']['avg']
        print(f"{'Version':<30} {'Avg Time (ms)':<15} {'Speedup':<10}")
        print("-" * 60)

        for key in ['stage1', 'stage2', 'baseline']:
            if key in results and results[key]:
                r = results[key]
                speedup = baseline_time / r['avg']
                print(f"{r['name']:<30} {r['avg']:>10.4f} ms   {speedup:>6.2f}x")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    # Save results
    with open('benchmark_results.txt', 'w') as f:
        f.write("TMA Multi-Stage Performance Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        for key, r in results.items():
            if r:
                f.write(f"{r['name']}:\n")
                f.write(f"  Average: {r['avg']:.4f} ms\n")
                f.write(f"  Std dev: {r['std']:.4f} ms\n")
                f.write(f"  Min:     {r['min']:.4f} ms\n")
                f.write(f"  Max:     {r['max']:.4f} ms\n")
                f.write(f"  All times: {[f'{t:.4f}' for t in r['times']]}\n\n")

    print("\nDetailed results saved to: benchmark_results.txt")


if __name__ == '__main__':
    main()
