#!/usr/bin/env python3
"""
Generate comprehensive test data using PyTorch spconv for accuracy comparison
"""

import numpy as np
import torch
import spconv.pytorch as spconv
import json
import time

def save_tensor_to_file(tensor, filename):
    """Save tensor in a format readable by C++"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Save as binary
    tensor.astype(np.float32).tofile(filename + '.bin')

    # Save metadata
    meta = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'size': tensor.size
    }
    with open(filename + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return tensor

def test_complete_sparse_conv():
    """Test complete sparse convolution pipeline"""
    print("=== Testing Complete Sparse Convolution Pipeline ===")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    batch_size = 2
    num_points = 5000
    in_channels = 64
    out_channels = 128
    spatial_shape = [41, 1600, 1408]  # Typical for 3D detection

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Points per batch: {num_points}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Spatial shape: {spatial_shape}")

    # Generate sparse indices (simulating voxelized point cloud)
    indices_list = []
    for b in range(batch_size):
        # Generate random 3D coordinates within spatial bounds
        coords = np.stack([
            np.random.randint(0, spatial_shape[0], num_points),
            np.random.randint(0, spatial_shape[1], num_points),
            np.random.randint(0, spatial_shape[2], num_points)
        ], axis=1)

        batch_coords = np.concatenate([
            np.full((num_points, 1), b),
            coords
        ], axis=1)
        indices_list.append(batch_coords)

    indices = np.concatenate(indices_list, axis=0)
    # Remove duplicates to get actual sparse structure
    indices = np.unique(indices, axis=0)
    actual_points = len(indices)

    print(f"  Actual sparse points: {actual_points}")

    # Generate features
    features = torch.randn(actual_points, in_channels, dtype=torch.float32).cuda()
    indices = torch.from_numpy(indices).int().cuda()

    # Create SparseConvTensor
    x = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )

    # Test 1: SubManifold Convolution
    print("\n1. Testing SubMConv3d (preserves sparsity)...")
    subm_conv = spconv.SubMConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        indice_key="subm0"
    ).cuda()

    # Initialize with controlled weights
    torch.nn.init.xavier_uniform_(subm_conv.weight)
    torch.nn.init.zeros_(subm_conv.bias)

    start = time.time()
    subm_output = subm_conv(x)
    torch.cuda.synchronize()
    subm_time = time.time() - start

    print(f"  Input points: {x.features.shape[0]}")
    print(f"  Output points: {subm_output.features.shape[0]}")
    print(f"  Time: {subm_time*1000:.2f} ms")

    # Save SubMConv test data
    save_tensor_to_file(x.indices, 'test_data/subm_input_indices')
    save_tensor_to_file(x.features, 'test_data/subm_input_features')
    save_tensor_to_file(subm_conv.weight, 'test_data/subm_weight')
    save_tensor_to_file(subm_conv.bias, 'test_data/subm_bias')
    save_tensor_to_file(subm_output.indices, 'test_data/subm_output_indices')
    save_tensor_to_file(subm_output.features, 'test_data/subm_output_features')

    # Test 2: Regular Sparse Convolution (changes sparsity)
    print("\n2. Testing SparseConv3d (reduces sparsity)...")
    sparse_conv = spconv.SparseConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=2,  # Stride 2 reduces spatial resolution
        padding=1,
        bias=True,
        indice_key="spconv0"
    ).cuda()

    torch.nn.init.xavier_uniform_(sparse_conv.weight)
    torch.nn.init.zeros_(sparse_conv.bias)

    start = time.time()
    sparse_output = sparse_conv(x)
    torch.cuda.synchronize()
    sparse_time = time.time() - start

    print(f"  Input points: {x.features.shape[0]}")
    print(f"  Output points: {sparse_output.features.shape[0]}")
    print(f"  Time: {sparse_time*1000:.2f} ms")

    # Save SparseConv test data
    save_tensor_to_file(sparse_conv.weight, 'test_data/sparse_weight')
    save_tensor_to_file(sparse_conv.bias, 'test_data/sparse_bias')
    save_tensor_to_file(sparse_output.indices, 'test_data/sparse_output_indices')
    save_tensor_to_file(sparse_output.features, 'test_data/sparse_output_features')

    # Test 3: Sequential layers (typical network pattern)
    print("\n3. Testing Sequential Convolutions...")
    net = spconv.SparseSequential(
        spconv.SubMConv3d(in_channels, 64, 3, padding=1, bias=True, indice_key="subm1"),
        # Note: BatchNorm not available in spconv, using convs with bias instead
        spconv.SubMConv3d(64, 128, 3, padding=1, bias=True, indice_key="subm2"),
        spconv.SparseConv3d(128, out_channels, 3, stride=2, padding=1, bias=True, indice_key="spconv1")
    ).cuda()

    start = time.time()
    seq_output = net(x)
    torch.cuda.synchronize()
    seq_time = time.time() - start

    print(f"  Input points: {x.features.shape[0]}")
    print(f"  Output points: {seq_output.features.shape[0]}")
    print(f"  Time: {seq_time*1000:.2f} ms")

    # Save sequential output
    save_tensor_to_file(seq_output.indices, 'test_data/seq_output_indices')
    save_tensor_to_file(seq_output.features, 'test_data/seq_output_features')

    # Test 4: Inverse convolution (transpose)
    print("\n4. Testing Inverse/Transpose Convolution...")
    inverse_conv = spconv.SparseInverseConv3d(
        in_channels=out_channels,
        out_channels=in_channels,
        kernel_size=3,
        indice_key="spconv0"  # Use same key as sparse_conv
    ).cuda()

    torch.nn.init.xavier_uniform_(inverse_conv.weight)
    torch.nn.init.zeros_(inverse_conv.bias)

    start = time.time()
    inverse_output = inverse_conv(sparse_output)
    torch.cuda.synchronize()
    inverse_time = time.time() - start

    print(f"  Input points: {sparse_output.features.shape[0]}")
    print(f"  Output points: {inverse_output.features.shape[0]}")
    print(f"  Restored to original resolution: {inverse_output.features.shape[0] == x.features.shape[0]}")
    print(f"  Time: {inverse_time*1000:.2f} ms")

    save_tensor_to_file(inverse_conv.weight, 'test_data/inverse_weight')
    save_tensor_to_file(inverse_conv.bias, 'test_data/inverse_bias')
    save_tensor_to_file(inverse_output.indices, 'test_data/inverse_output_indices')
    save_tensor_to_file(inverse_output.features, 'test_data/inverse_output_features')

    # Save metadata
    metadata = {
        'batch_size': batch_size,
        'spatial_shape': spatial_shape,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'num_input_points': actual_points,
        'subm_output_points': subm_output.features.shape[0],
        'sparse_output_points': sparse_output.features.shape[0],
        'seq_output_points': seq_output.features.shape[0],
        'inverse_output_points': inverse_output.features.shape[0],
        'timing': {
            'subm_conv_ms': subm_time * 1000,
            'sparse_conv_ms': sparse_time * 1000,
            'sequential_ms': seq_time * 1000,
            'inverse_conv_ms': inverse_time * 1000
        }
    }

    with open('test_data/spconv_test_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata

def test_dense_to_sparse():
    """Test conversion between dense and sparse representations"""
    print("\n=== Testing Dense/Sparse Conversion ===")

    # Create a small dense tensor
    batch_size = 1
    spatial_shape = [10, 10, 10]
    channels = 16

    # Create dense tensor with known pattern
    dense = torch.zeros(batch_size, channels, *spatial_shape)

    # Add some non-zero values in specific locations
    dense[0, :, 2:4, 3:6, 1:8] = torch.randn(channels, 2, 3, 7)
    dense[0, :, 5:7, 2:5, 3:6] = torch.randn(channels, 2, 3, 3)

    dense = dense.cuda()

    # Convert to sparse
    sparse = spconv.SparseConvTensor.from_dense(dense)

    print(f"Dense shape: {list(dense.shape)}")
    print(f"Sparse points: {sparse.features.shape[0]}")
    print(f"Sparsity: {1 - sparse.features.shape[0] / (np.prod(spatial_shape)):.2%}")

    # Convert back to dense
    dense_reconstructed = sparse.dense()

    # Check reconstruction accuracy
    # Note: dense_reconstructed might have different shape due to spconv internals
    if dense.shape == dense_reconstructed.shape:
        diff = torch.abs(dense - dense_reconstructed).max()
        print(f"Max reconstruction error: {diff.item():.6e}")
    else:
        print(f"Shape mismatch - Original: {list(dense.shape)}, Reconstructed: {list(dense_reconstructed.shape)}")
        # Try to extract the relevant part
        if len(dense_reconstructed.shape) == 4:  # Missing batch dimension
            dense_reconstructed = dense_reconstructed.unsqueeze(0)
        if dense_reconstructed.shape[2:] != dense.shape[2:]:
            print(f"Spatial dimensions differ, skipping comparison")

    save_tensor_to_file(sparse.indices, 'test_data/dense_to_sparse_indices')
    save_tensor_to_file(sparse.features, 'test_data/dense_to_sparse_features')
    save_tensor_to_file(dense, 'test_data/original_dense')
    save_tensor_to_file(dense_reconstructed, 'test_data/reconstructed_dense')

def main():
    import os
    os.makedirs('test_data', exist_ok=True)

    print("=" * 80)
    print("Generating comprehensive test data for spconv accuracy testing")
    print("=" * 80)

    # Run all tests
    metadata = test_complete_sparse_conv()
    test_dense_to_sparse()

    print("\n" + "=" * 80)
    print("Test data generation complete!")
    print(f"Generated files in test_data/")
    print("\nSummary:")
    print(f"  Total input points: {metadata['num_input_points']}")
    print(f"  SubM output points: {metadata['subm_output_points']}")
    print(f"  Sparse output points: {metadata['sparse_output_points']}")
    print(f"  Sequential output points: {metadata['seq_output_points']}")
    print("\nTiming:")
    for key, value in metadata['timing'].items():
        print(f"  {key}: {value:.2f} ms")

if __name__ == '__main__':
    main()