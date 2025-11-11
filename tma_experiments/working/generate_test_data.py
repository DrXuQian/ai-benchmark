#!/usr/bin/env python3
"""
Generate proper test data for deformable attention with valid sampling locations.
Ensures that sampling locations are within bounds to produce non-zero output.
"""
import numpy as np
import os

# Parameters matching the test configuration
batch = 48
num_query = 20522
num_heads = 1
channels = 32
num_levels = 4
num_points = 8

# Spatial shapes for each level (from working_pattern)
spatial_shapes = [(92, 160), (46, 80), (23, 40), (12, 20)]

# Calculate level start indices (with padding +2 for each dimension)
level_start_index = [0]
for i, (h, w) in enumerate(spatial_shapes[:-1]):
    level_start_index.append(level_start_index[-1] + (h+2) * (w+2))
level_start_index = np.array(level_start_index, dtype=np.int64)

spatial_size = level_start_index[-1] + (spatial_shapes[-1][0]+2) * (spatial_shapes[-1][1]+2)

print(f"Configuration:")
print(f"  batch={batch}, num_query={num_query}, num_heads={num_heads}")
print(f"  channels={channels}, num_levels={num_levels}, num_points={num_points}")
print(f"  spatial_size={spatial_size}")
print(f"  level_start_index: {level_start_index}")
print(f"  spatial_shapes: {spatial_shapes}")

# Generate value data: simple incrementing pattern
print("\n[1/4] Generating value data...")
value = np.zeros((batch, spatial_size, channels), dtype=np.float16)

for b in range(batch):
    offset = 0
    for l, (h, w) in enumerate(spatial_shapes):
        h_padded, w_padded = h + 2, w + 2
        for hi in range(h_padded):
            for wi in range(w_padded):
                for c in range(channels):
                    # Simple pattern with smaller values to avoid overflow
                    # Encoding: b*10 + l + h*0.1 + w*0.01 + c*0.0001
                    val = b * 10 + l + hi * 0.1 + wi * 0.01 + c * 0.0001
                    value[b, offset + hi * w_padded + wi, c] = val
        offset += h_padded * w_padded

value_flat = value.reshape(-1)

# Generate sampling locations: CRITICAL - must be within valid range
print("[2/4] Generating sampling locations (within bounds)...")
sampling_loc = np.zeros((batch, num_query, num_heads, num_levels, num_points, 2), dtype=np.float16)

np.random.seed(42)  # Reproducible

for b in range(batch):
    for q in range(num_query):
        for head in range(num_heads):
            for l in range(num_levels):
                h, w = spatial_shapes[l]
                for p in range(num_points):
                    # Sampling locations should be in normalized coordinates [0, 1]
                    # These will be multiplied by spatial_h and spatial_w in the kernel
                    # We want them to land within the valid region (1 to h, 1 to w)
                    # So normalized coords should be roughly [1/h, (h-1)/h] and [1/w, (w-1)/w]

                    # Use center region to ensure validity
                    h_norm = 0.3 + 0.4 * np.random.rand()  # Range [0.3, 0.7]
                    w_norm = 0.3 + 0.4 * np.random.rand()  # Range [0.3, 0.7]

                    sampling_loc[b, q, head, l, p, 0] = h_norm  # height
                    sampling_loc[b, q, head, l, p, 1] = w_norm  # width

# Generate attention weights: uniform positive values
print("[3/4] Generating attention weights...")
attn_weight = np.ones((batch, num_query, num_heads, num_levels, num_points), dtype=np.float16)

# Normalize weights per level to sum to 1 (for each query)
attn_weight = attn_weight / num_points

# Save all files
print("[4/4] Saving binary files...")
os.makedirs('../working', exist_ok=True)

value_flat.astype(np.float16).tofile('../working/value.bin')
level_start_index.tofile('../working/level_start_index.bin')
np.array(spatial_shapes, dtype=np.int64).reshape(-1).tofile('../working/spatial_shapes.bin')
sampling_loc.reshape(-1).astype(np.float16).tofile('../working/sampling_locations.bin')
attn_weight.reshape(-1).astype(np.float16).tofile('../working/attention_weights.bin')

print("\n✓ Test data generated successfully!")
print(f"\nFile sizes:")
print(f"  value.bin: {os.path.getsize('../working/value.bin'):,} bytes")
print(f"  sampling_locations.bin: {os.path.getsize('../working/sampling_locations.bin'):,} bytes")
print(f"  attention_weights.bin: {os.path.getsize('../working/attention_weights.bin'):,} bytes")
print(f"  level_start_index.bin: {os.path.getsize('../working/level_start_index.bin'):,} bytes")
print(f"  spatial_shapes.bin: {os.path.getsize('../working/spatial_shapes.bin'):,} bytes")

# Verify sampling locations are within bounds
print("\n✓ Sampling location validation:")
for l in range(num_levels):
    h, w = spatial_shapes[l]
    locs = sampling_loc[:, :, :, l, :, :]
    h_coords = locs[:, :, :, :, 0] * h
    w_coords = locs[:, :, :, :, 1] * w

    h_valid = np.sum((h_coords > 0) & (h_coords < h+1))
    w_valid = np.sum((w_coords > 0) & (w_coords < w+1))
    total = locs.shape[0] * locs.shape[1] * locs.shape[2] * locs.shape[3]

    print(f"  Level {l} (h={h}, w={w}): {h_valid}/{total} h valid, {w_valid}/{total} w valid")

print("\n✓ Expected output: Non-zero values from bilinear interpolation")
print(f"  If all sampling locations are valid, output should contain interpolated values")
