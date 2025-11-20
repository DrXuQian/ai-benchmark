#!/usr/bin/env python3
import numpy as np
import struct

def read_half_binary(filename):
    """Read binary file containing half-precision floats"""
    with open(filename, 'rb') as f:
        data = f.read()

    # Convert bytes to uint16 array
    uint16_data = np.frombuffer(data, dtype=np.uint16)

    # Convert uint16 to float16
    float16_data = uint16_data.view(np.float16)

    return float16_data, uint16_data

def compare_outputs():
    # Read original output
    try:
        orig_float, orig_uint = read_half_binary('working_test/output_cuda.bin')
        print(f"Original output found: {len(orig_float)} values")
        print(f"  First 10 as float16: {orig_float[:10]}")
        print(f"  First 10 as uint16: {orig_uint[:10]}")
        print(f"  Min/Max: {np.min(orig_float):.4f} / {np.max(orig_float):.4f}")
        print(f"  Mean/Std: {np.mean(orig_float):.4f} / {np.std(orig_float):.4f}")
        print()
    except FileNotFoundError:
        print("Original output not found. Running original implementation first...")
        orig_float, orig_uint = None, None

    # Read optimized output
    try:
        opt_float, opt_uint = read_half_binary('working_test/output_cuda_unrolled_fixed.bin')
        print(f"Optimized output found: {len(opt_float)} values")
        print(f"  First 10 as float16: {opt_float[:10]}")
        print(f"  First 10 as uint16: {opt_uint[:10]}")
        print(f"  Min/Max: {np.min(opt_float):.4f} / {np.max(opt_float):.4f}")
        print(f"  Mean/Std: {np.mean(opt_float):.4f} / {np.std(opt_float):.4f}")
        print()
    except FileNotFoundError:
        print("Optimized output not found.")
        return

    if orig_float is not None:
        # Compare the outputs
        print("Comparison:")
        print(f"  Output sizes match: {len(orig_float) == len(opt_float)}")

        if len(orig_float) == len(opt_float):
            # Calculate differences
            diff = np.abs(orig_float - opt_float)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")
            print(f"  Number of exact matches: {np.sum(diff == 0)} / {len(diff)}")
            print(f"  Number of differences > 0.01: {np.sum(diff > 0.01)}")
            print(f"  Number of differences > 0.1: {np.sum(diff > 0.1)}")
            print(f"  Number of differences > 1.0: {np.sum(diff > 1.0)}")

            # Show where the first differences occur
            if max_diff > 0:
                first_diff_idx = np.where(diff > 0)[0][0]
                print(f"\n  First difference at index {first_diff_idx}:")
                print(f"    Original: {orig_float[first_diff_idx]:.6f} (uint16: {orig_uint[first_diff_idx]})")
                print(f"    Optimized: {opt_float[first_diff_idx]:.6f} (uint16: {opt_uint[first_diff_idx]})")
                print(f"    Difference: {diff[first_diff_idx]:.6f}")

                # Show a few more examples
                print(f"\n  Sample differences (first 5 non-zero):")
                diff_indices = np.where(diff > 0)[0][:5]
                for idx in diff_indices:
                    print(f"    Index {idx}: orig={orig_float[idx]:.4f}, opt={opt_float[idx]:.4f}, diff={diff[idx]:.4f}")

if __name__ == "__main__":
    compare_outputs()