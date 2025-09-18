#!/usr/bin/env python3
"""
Generate test data using spconv Python API for comparison with extracted CUDA kernels
"""

import numpy as np
import torch
import spconv.pytorch as spconv
import pickle
import json

def save_tensor_to_file(tensor, filename):
    """Save tensor in a format readable by C++"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Save as binary for C++ to read
    tensor.astype(np.float32).tofile(filename + '.bin')

    # Also save metadata
    meta = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'size': tensor.size
    }
    with open(filename + '.meta.json', 'w') as f:
        json.dump(meta, f)

    print(f"Saved {filename}: shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor

def test_sparse_conv_basic():
    """Test basic sparse convolution operations"""
    print("=== Testing Basic Sparse Convolution ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sparse data (simulating 3D voxel grid)
    batch_size = 1
    num_points = 1000
    num_features = 32
    spatial_shape = [128, 128, 32]  # z, y, x for 3D

    # Generate random sparse indices (N, 4) - [batch, z, y, x]
    indices = []
    for b in range(batch_size):
        # Generate random 3D coordinates
        coords = np.random.randint(0, spatial_shape[0], size=(num_points, 3))
        batch_coords = np.concatenate([
            np.full((num_points, 1), b),
            coords
        ], axis=1)
        indices.append(batch_coords)

    indices = np.concatenate(indices, axis=0)
    # Remove duplicates
    indices = np.unique(indices, axis=0)
    num_actual_points = len(indices)
    print(f"Number of unique points: {num_actual_points}")

    # Generate random features
    features = np.random.randn(num_actual_points, num_features).astype(np.float32)

    # Convert to torch tensors
    indices_th = torch.from_numpy(indices).int().cuda()
    features_th = torch.from_numpy(features).float().cuda()

    # Create SparseConvTensor
    sparse_tensor = spconv.SparseConvTensor(
        features_th,
        indices_th,
        spatial_shape,
        batch_size
    )

    # Define sparse convolution layer
    conv = spconv.SubMConv3d(
        in_channels=num_features,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        indice_key="subm0"
    ).cuda()

    # Initialize weights with known values for testing
    with torch.no_grad():
        conv.weight.data = torch.randn_like(conv.weight) * 0.1
        conv.bias.data = torch.randn_like(conv.bias) * 0.01

    # Perform forward pass
    output = conv(sparse_tensor)

    # Save input data
    save_tensor_to_file(indices, 'test_data/input_indices')
    save_tensor_to_file(features, 'test_data/input_features')
    save_tensor_to_file(conv.weight.data, 'test_data/conv_weight')
    save_tensor_to_file(conv.bias.data, 'test_data/conv_bias')

    # Save output data
    save_tensor_to_file(output.indices, 'test_data/output_indices')
    save_tensor_to_file(output.features, 'test_data/output_features')

    # Save sparse convolution metadata
    meta = {
        'batch_size': batch_size,
        'spatial_shape': spatial_shape,
        'num_input_points': num_actual_points,
        'num_output_points': output.features.shape[0],
        'in_channels': num_features,
        'out_channels': 64,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    }
    with open('test_data/conv_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Output shape: {output.features.shape}")
    print(f"Output indices shape: {output.indices.shape}")

    return sparse_tensor, conv, output

def test_gather_scatter_operations():
    """Test gather and scatter operations specifically"""
    print("\n=== Testing Gather/Scatter Operations ===")

    torch.manual_seed(42)

    # Create simple test case for gather
    num_total = 500
    num_active = 200
    num_features = 16

    # Create dense features
    dense_features = torch.randn(num_total, num_features).cuda()

    # Create indices for gathering (randomly select active points)
    active_indices = torch.randperm(num_total)[:num_active].cuda()

    # Perform gather operation (this is what gather_features_kernel does)
    gathered_features = dense_features[active_indices]

    # Save data for C++ testing
    save_tensor_to_file(dense_features, 'test_data/gather_input_features')
    save_tensor_to_file(active_indices, 'test_data/gather_indices')
    save_tensor_to_file(gathered_features, 'test_data/gather_output_features')

    print(f"Gather test - input: {dense_features.shape}, indices: {active_indices.shape}, output: {gathered_features.shape}")

    # Test scatter operation
    scatter_input = torch.randn(num_active, num_features).cuda()
    scatter_output = torch.zeros(num_total, num_features).cuda()

    # Perform scatter add (this is what scatter_add_kernel does)
    scatter_output.index_add_(0, active_indices, scatter_input)

    # Save scatter test data
    save_tensor_to_file(scatter_input, 'test_data/scatter_input_features')
    save_tensor_to_file(active_indices, 'test_data/scatter_indices')
    save_tensor_to_file(scatter_output, 'test_data/scatter_output_features')

    print(f"Scatter test - input: {scatter_input.shape}, output: {scatter_output.shape}")

def test_regular_sparse_conv():
    """Test regular (non-submanifold) sparse convolution"""
    print("\n=== Testing Regular Sparse Convolution ===")

    torch.manual_seed(42)

    # Create sparse data
    batch_size = 1
    num_points = 800
    num_features = 32
    spatial_shape = [64, 64, 16]

    # Generate indices
    indices = []
    for b in range(batch_size):
        coords = np.random.randint(0, spatial_shape[0], size=(num_points, 3))
        batch_coords = np.concatenate([
            np.full((num_points, 1), b),
            coords
        ], axis=1)
        indices.append(batch_coords)

    indices = np.unique(np.concatenate(indices, axis=0), axis=0)
    features = np.random.randn(len(indices), num_features).astype(np.float32)

    indices_th = torch.from_numpy(indices).int().cuda()
    features_th = torch.from_numpy(features).float().cuda()

    sparse_tensor = spconv.SparseConvTensor(
        features_th,
        indices_th,
        spatial_shape,
        batch_size
    )

    # Regular sparse conv (not submanifold)
    conv = spconv.SparseConv3d(
        in_channels=num_features,
        out_channels=48,
        kernel_size=3,
        stride=2,  # Stride 2 reduces spatial dimensions
        padding=1,
        bias=True,
        indice_key="spconv0"
    ).cuda()

    # Initialize weights
    with torch.no_grad():
        conv.weight.data = torch.randn_like(conv.weight) * 0.1
        conv.bias.data = torch.randn_like(conv.bias) * 0.01

    output = conv(sparse_tensor)

    # Save data
    save_tensor_to_file(indices, 'test_data/regular_conv_input_indices')
    save_tensor_to_file(features, 'test_data/regular_conv_input_features')
    save_tensor_to_file(conv.weight.data, 'test_data/regular_conv_weight')
    save_tensor_to_file(conv.bias.data, 'test_data/regular_conv_bias')
    save_tensor_to_file(output.indices, 'test_data/regular_conv_output_indices')
    save_tensor_to_file(output.features, 'test_data/regular_conv_output_features')

    print(f"Regular conv - Input points: {len(indices)}, Output points: {len(output.indices)}")
    print(f"Output shape: {output.features.shape}")

def main():
    import os
    os.makedirs('test_data', exist_ok=True)

    print("Generating test data using spconv Python API...")
    print("This data will be used to verify the extracted CUDA kernels")
    print("=" * 60)

    # Run tests
    test_gather_scatter_operations()
    sparse_tensor, conv, output = test_sparse_conv_basic()
    test_regular_sparse_conv()

    print("\n" + "=" * 60)
    print("Test data generated successfully in test_data/")
    print("Files created:")
    for f in sorted(os.listdir('test_data')):
        print(f"  - {f}")

if __name__ == '__main__':
    main()