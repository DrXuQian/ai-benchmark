#!/usr/bin/env python3
"""
Test script to verify PyTorch CUDA operations
"""

import torch
import numpy as np
import time
import sys
import os

# Add operators to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'operators'))

def test_cuda_availability():
    """Test if CUDA is available"""
    print("=" * 60)
    print("CUDA Availability Test")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} (Compute {props.major}.{props.minor})")
        
        return True
    else:
        print("❌ CUDA is not available")
        return False

def test_voxelization():
    """Test voxelization operation"""
    print("\n" + "=" * 60)
    print("Voxelization Test")
    print("=" * 60)
    
    try:
        from voxelization import voxelize_points
        
        device = torch.device('cuda')
        
        # Generate sample point cloud
        num_points = 50000
        points = torch.rand(num_points, 5, device=device)
        
        # Scale to reasonable ranges
        points[:, 0] = points[:, 0] * 108.0 - 54.0  # x: [-54, 54]
        points[:, 1] = points[:, 1] * 108.0 - 54.0  # y: [-54, 54] 
        points[:, 2] = points[:, 2] * 8.0 - 5.0     # z: [-5, 3]
        points[:, 3] = points[:, 3]                 # intensity: [0, 1]
        points[:, 4] = points[:, 4] * 0.1           # time: [0, 0.1]
        
        print(f"Input points shape: {points.shape}")
        print(f"Point cloud bounds:")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        # Time the operation
        torch.cuda.synchronize()
        start_time = time.time()
        
        voxel_features, voxel_coords, num_points_per_voxel = voxelize_points(points)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Find valid voxels
        valid_mask = num_points_per_voxel > 0
        valid_voxels = valid_mask.sum().item()
        
        print(f"✅ Voxelization successful!")
        print(f"Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Output shapes:")
        print(f"  Voxel features: {voxel_features.shape}")
        print(f"  Voxel coords: {voxel_coords.shape}")
        print(f"  Num points per voxel: {num_points_per_voxel.shape}")
        print(f"Valid voxels: {valid_voxels}")
        
        if valid_voxels > 0:
            avg_points = num_points_per_voxel[valid_mask].float().mean()
            print(f"Average points per valid voxel: {avg_points:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voxelization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nms():
    """Test NMS operation"""
    print("\n" + "=" * 60)
    print("NMS Test")
    print("=" * 60)
    
    try:
        from nms import nms_3d
        
        device = torch.device('cuda')
        
        # Generate sample bounding boxes
        num_boxes = 2000
        boxes = torch.zeros(num_boxes, 11, device=device)
        
        # Random positions
        boxes[:, 0] = torch.rand(num_boxes, device=device) * 100.0 - 50.0  # x
        boxes[:, 1] = torch.rand(num_boxes, device=device) * 100.0 - 50.0  # y
        boxes[:, 2] = torch.rand(num_boxes, device=device) * 6.0 - 3.0     # z
        
        # Random sizes  
        boxes[:, 3] = torch.rand(num_boxes, device=device) * 3.0 + 1.0     # w
        boxes[:, 4] = torch.rand(num_boxes, device=device) * 3.0 + 1.0     # l
        boxes[:, 5] = torch.rand(num_boxes, device=device) * 2.0 + 1.0     # h
        
        # Random rotation
        boxes[:, 6] = torch.rand(num_boxes, device=device) * 2.0 * np.pi   # rotation
        
        # Random velocities
        boxes[:, 7] = torch.rand(num_boxes, device=device) * 20.0 - 10.0   # vx
        boxes[:, 8] = torch.rand(num_boxes, device=device) * 20.0 - 10.0   # vy
        
        # Random scores
        boxes[:, 9] = torch.rand(num_boxes, device=device)                 # score
        
        # Random labels
        boxes[:, 10] = torch.randint(0, 10, (num_boxes,), device=device, dtype=torch.float32)
        
        iou_threshold = 0.5
        
        print(f"Input boxes shape: {boxes.shape}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"Score range: [{boxes[:, 9].min():.3f}, {boxes[:, 9].max():.3f}]")
        
        # Time the operation
        torch.cuda.synchronize()
        start_time = time.time()
        
        keep_indices = nms_3d(boxes, iou_threshold=iou_threshold)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"✅ NMS successful!")
        print(f"Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Input boxes: {num_boxes}")
        print(f"Kept boxes: {len(keep_indices)}")
        suppression_ratio = (num_boxes - len(keep_indices)) / num_boxes * 100
        print(f"Suppression ratio: {suppression_ratio:.1f}%")
        
        if len(keep_indices) > 0:
            kept_boxes = boxes[keep_indices]
            print(f"Kept boxes score range: [{kept_boxes[:, 9].min():.3f}, {kept_boxes[:, 9].max():.3f}]")
            print(f"Average kept score: {kept_boxes[:, 9].mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ NMS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geometric_kernel_attn():
    """Test Geometric Kernel Attention operation"""
    print("\n" + "=" * 60)
    print("Geometric Kernel Attention Test")
    print("=" * 60)
    
    try:
        from geometric_kernel_attn import geometric_kernel_attn_forward
        
        device = torch.device('cuda')
        
        # Geometric Kernel Attention configuration
        batch_size = 1
        spatial_size = 1568  # e.g., 28*28 + 14*14 + 7*7 + 4*4
        num_heads = 8
        channels = 256
        num_queries = 100
        num_levels = 4
        num_points = 4
        
        print(f"Geometric Kernel Attention Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Spatial size: {spatial_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Channels: {channels}")
        print(f"  Num queries: {num_queries}")
        print(f"  Num levels: {num_levels}")
        print(f"  Num points: {num_points}")
        
        # Generate sample data
        value = torch.rand(batch_size, spatial_size, num_heads, channels, 
                          device=device, dtype=torch.float32)
        
        spatial_shapes = torch.tensor([[28, 28], [14, 14], [7, 7], [4, 4]], 
                                    device=device, dtype=torch.int64)
        
        level_start_index = torch.tensor([0, 784, 980, 1029], 
                                       device=device, dtype=torch.int64)
        
        sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2,
                                device=device, dtype=torch.float32) * 2 - 1  # normalize to [-1, 1]
        
        attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points,
                               device=device, dtype=torch.float32)
        
        # Normalize attention weights
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        print(f"Input shapes:")
        print(f"  Value: {value.shape}")
        print(f"  Spatial shapes: {spatial_shapes.shape}")
        print(f"  Level start index: {level_start_index.shape}")
        print(f"  Sampling loc: {sampling_loc.shape}")
        print(f"  Attention weight: {attn_weight.shape}")
        
        # Time the operation
        torch.cuda.synchronize()
        start_time = time.time()
        
        output = geometric_kernel_attn_forward(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"✅ Geometric Kernel Attention successful!")
        print(f"Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Output mean: {output.mean():.6f}")
        print(f"Output std: {output.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Geometric Kernel Attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bevpool():
    """Test BEVPool operation"""
    print("\n" + "=" * 60)
    print("BEVPool Test")
    print("=" * 60)
    
    try:
        from bevpool import bevpool_forward
        
        device = torch.device('cuda')
        
        # BEVPool configuration
        bev_width = 100
        bev_height = 100
        num_cameras = 3
        channels = 64
        depth_bins = 59
        feature_height = 16
        feature_width = 44
        
        print(f"BEVPool Configuration:")
        print(f"  Cameras: {num_cameras}")
        print(f"  Channels: {channels}")
        print(f"  Depth bins: {depth_bins}")
        print(f"  Feature size: {feature_height}x{feature_width}")
        print(f"  BEV size: {bev_height}x{bev_width}")
        
        # Generate sample data
        camera_features = torch.rand(
            num_cameras, channels, depth_bins, feature_height, feature_width,
            device=device, dtype=torch.float16
        ) * 2.0 - 1.0
        
        depth_weights = torch.rand(
            num_cameras, depth_bins, feature_height, feature_width,
            device=device, dtype=torch.float16
        )
        
        # Generate indices and intervals for BEV mapping
        num_intervals = bev_height * bev_width
        indices_per_pixel = 3  # Average 3 camera pixels per BEV pixel
        total_indices = num_intervals * indices_per_pixel
        
        max_index = num_cameras * depth_bins * feature_height * feature_width - 1
        indices = torch.randint(0, max_index + 1, (total_indices,), device=device, dtype=torch.int32)
        
        # Create intervals
        intervals = torch.zeros(num_intervals, 3, device=device, dtype=torch.int32)
        for i in range(num_intervals):
            start_idx = i * indices_per_pixel
            end_idx = (i + 1) * indices_per_pixel
            intervals[i] = torch.tensor([start_idx, end_idx, i], dtype=torch.int32)
        
        print(f"Input shapes:")
        print(f"  Camera features: {camera_features.shape}")
        print(f"  Depth weights: {depth_weights.shape}")
        print(f"  Indices: {indices.shape}")
        print(f"  Intervals: {intervals.shape}")
        
        # Time the operation
        torch.cuda.synchronize()
        start_time = time.time()
        
        bev_features = bevpool_forward(
            camera_features, depth_weights, indices, intervals,
            bev_width, bev_height, num_cameras, channels,
            depth_bins, feature_height, feature_width
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"✅ BEVPool successful!")
        print(f"Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Output BEV features shape: {bev_features.shape}")
        
        # Check output statistics
        non_zero_count = (bev_features != 0).sum().item()
        total_elements = bev_features.numel()
        
        print(f"Non-zero values: {non_zero_count}/{total_elements} ({non_zero_count/total_elements*100:.1f}%)")
        print(f"Value range: [{bev_features.min():.6f}, {bev_features.max():.6f}]")
        print(f"Mean absolute value: {bev_features.abs().mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ BEVPool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("AI Benchmark CUDA Operations Test Suite")
    print("This test will verify that all PyTorch CUDA operations work correctly.")
    print()
    
    # Test CUDA availability first
    if not test_cuda_availability():
        print("\n❌ Cannot run tests without CUDA. Exiting.")
        sys.exit(1)
    
    results = []
    
    # Run individual tests
    results.append(("Voxelization", test_voxelization()))
    results.append(("NMS", test_nms()))
    results.append(("BEVPool", test_bevpool()))
    results.append(("Geometric Kernel Attention", test_geometric_kernel_attn()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! PyTorch CUDA operations are working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()