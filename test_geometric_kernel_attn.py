#!/usr/bin/env python3
"""
Standalone test for Geometric Kernel Attention operator
"""

import torch
import numpy as np
import time
import sys
import os

def test_geometric_kernel_attn():
    """Test the geometric kernel attention operation"""
    print("=" * 60)
    print("Geometric Kernel Attention Standalone Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return False
        
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Add operators to Python path
    operators_path = os.path.join(os.path.dirname(__file__), 'operators')
    if operators_path not in sys.path:
        sys.path.insert(0, operators_path)
    
    try:
        from geometric_kernel_attn import geometric_kernel_attn_forward
        
        # Test configuration - realistic sizes for vision transformer attention
        batch_size = 2
        spatial_size = 1568  # 28*28 + 14*14 + 7*7 + 4*4 = 784 + 196 + 49 + 16
        num_heads = 8
        channels = 256
        num_queries = 100
        num_levels = 4
        num_points = 4
        
        print(f"Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Spatial size: {spatial_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Channels: {channels}")
        print(f"  Num queries: {num_queries}")
        print(f"  Num levels: {num_levels}")
        print(f"  Num points per query: {num_points}")
        
        # Create realistic test data
        value = torch.randn(batch_size, spatial_size, num_heads, channels, 
                          device=device, dtype=torch.float32)
        
        # Multi-scale spatial shapes (pyramid levels)
        spatial_shapes = torch.tensor([[28, 28], [14, 14], [7, 7], [4, 4]], 
                                    device=device, dtype=torch.int64)
        
        # Level start indices (cumulative spatial sizes)
        level_start_index = torch.tensor([0, 784, 980, 1029], 
                                       device=device, dtype=torch.int64)
        
        # Sampling locations in normalized coordinates [-1, 1]
        sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2,
                                device=device, dtype=torch.float32) * 2 - 1
        
        # Attention weights (normalized)
        attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points,
                               device=device, dtype=torch.float32)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        print(f"\nInput tensor shapes:")
        print(f"  Value: {value.shape}")
        print(f"  Spatial shapes: {spatial_shapes.shape}")
        print(f"  Level start index: {level_start_index.shape}")
        print(f"  Sampling locations: {sampling_loc.shape}")
        print(f"  Attention weights: {attn_weight.shape}")
        
        # Test forward pass
        print(f"\nRunning forward pass...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        output = geometric_kernel_attn_forward(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight, im2col_step=64
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Verify output
        expected_shape = (batch_size, num_queries, num_heads, channels)
        
        print(f"✅ Geometric Kernel Attention successful!")
        print(f"Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Output shape: {output.shape} (expected: {expected_shape})")
        print(f"Shape correct: {output.shape == expected_shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Output mean: {output.mean():.6f}")
        print(f"Output std: {output.std():.6f}")
        print(f"Has NaN: {torch.isnan(output).any()}")
        print(f"Has Inf: {torch.isinf(output).any()}")
        
        # Test different batch sizes
        print(f"\nTesting different configurations...")
        
        # Small test
        small_output = geometric_kernel_attn_forward(
            value[:1], spatial_shapes, level_start_index, 
            sampling_loc[:1], attn_weight[:1], im2col_step=32
        )
        print(f"Small batch test: {small_output.shape}")
        
        # Different im2col_step
        output2 = geometric_kernel_attn_forward(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight, im2col_step=32
        )
        print(f"Different im2col_step: outputs close = {torch.allclose(output, output2, atol=1e-5)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_geometric_kernel_attn()
    
    if success:
        print("\n🎉 Geometric Kernel Attention operator is working correctly!")
        print("\nUsage:")
        print("from operators.geometric_kernel_attn import geometric_kernel_attn_forward")
        print("output = geometric_kernel_attn_forward(value, spatial_shapes, level_start_index,")
        print("                                       sampling_loc, attn_weight, im2col_step)")
    else:
        print("\n❌ Geometric Kernel Attention operator test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)