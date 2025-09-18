#!/usr/bin/env python3
"""Check the test data to understand the issue"""

import numpy as np
import json

# Load indices
# Load metadata to get correct shape
with open('test_data/gather_indices.meta.json', 'r') as f:
    meta = json.load(f)
    indices_shape = meta['shape']

indices = np.fromfile('test_data/gather_indices.bin', dtype=np.float32).astype(np.int64).reshape(indices_shape)
print(f"Indices shape: {indices.shape}")
print(f"Indices min: {indices.min()}, max: {indices.max()}")
print(f"First 10 indices: {indices[:10]}")

# Load input features
input_features = np.fromfile('test_data/gather_input_features.bin', dtype=np.float32)
input_features = input_features.reshape(500, 16)
print(f"\nInput features shape: {input_features.shape}")
print(f"First row: {input_features[indices[0], :5]}")

# Load expected output
output_features = np.fromfile('test_data/gather_output_features.bin', dtype=np.float32)
output_features = output_features.reshape(200, 16)
print(f"\nExpected output shape: {output_features.shape}")
print(f"First row: {output_features[0, :5]}")

# Verify gather operation
print(f"\nVerifying gather operation:")
for i in range(5):
    idx = indices[i]
    print(f"Index {i}: gather from position {idx}")
    print(f"  Input[{idx}][:3] = {input_features[idx, :3]}")
    print(f"  Output[{i}][:3] = {output_features[i, :3]}")
    print(f"  Match: {np.allclose(input_features[idx], output_features[i])}")