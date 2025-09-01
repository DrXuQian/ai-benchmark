import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import numpy as np

class BEVPoolOp(nn.Module):
    def __init__(self, bev_width, bev_height, num_cameras, channels, depth_bins, feature_height, feature_width):
        super(BEVPoolOp, self).__init__()
        
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.num_cameras = num_cameras
        self.channels = channels
        self.depth_bins = depth_bins
        self.feature_height = feature_height
        self.feature_width = feature_width
        
        # Get the directory of this file (kernels and PyTorch files are in same dir)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # JIT compile the CUDA extension
        self.bevpool_cuda = load(
            name="bevpool_cuda",
            sources=[
                os.path.join(current_dir, "bevpool_kernel.cu"),
                os.path.join(current_dir, "bevpool_binding.cpp")
            ],
            extra_cuda_cflags=['-O3'],
            verbose=True
        )
        
        # Initialize the BEVPool instance
        params = {
            'bev_width': bev_width,
            'bev_height': bev_height,
            'num_cameras': num_cameras,
            'channels': channels,
            'depth_bins': depth_bins,
            'feature_height': feature_height,
            'feature_width': feature_width
        }
        self.bevpool_instance = self.bevpool_cuda.create(params)
    
    def __del__(self):
        if hasattr(self, 'bevpool_instance') and self.bevpool_instance is not None:
            self.bevpool_cuda.destroy(self.bevpool_instance)
    
    def forward(self, camera_features, depth_weights, indices, intervals):
        """
        Args:
            camera_features: (N*C*D*H*W,) tensor with camera features
            depth_weights: (N*D*H*W,) tensor with depth weights
            indices: (M,) tensor with indices mapping
            intervals: (K, 3) tensor with [start_idx, end_idx, bev_idx] intervals
        
        Returns:
            bev_features: (1, C, bev_height, bev_width) tensor
        """
        return self.bevpool_cuda.forward(
            self.bevpool_instance, 
            camera_features, 
            depth_weights, 
            indices, 
            intervals
        )


def create_bevpool_op(bev_width, bev_height, num_cameras, channels, depth_bins, feature_height, feature_width):
    """Factory function to create BEVPool operation"""
    return BEVPoolOp(bev_width, bev_height, num_cameras, channels, depth_bins, feature_height, feature_width)


def bevpool_forward(camera_features, depth_weights, indices, intervals, 
                   bev_width=200, bev_height=200, num_cameras=6, channels=80, 
                   depth_bins=118, feature_height=32, feature_width=88):
    """
    BEV pooling operation
    
    Args:
        camera_features: torch.Tensor of shape (N, C, D, H, W)
        depth_weights: torch.Tensor of shape (N, D, H, W) 
        indices: torch.Tensor of shape (M,) mapping camera pixels to BEV
        intervals: torch.Tensor of shape (K, 3) with [start, end, bev_pos] for each BEV pixel
        bev_width: int, width of output BEV feature map
        bev_height: int, height of output BEV feature map
        num_cameras: int, number of cameras
        channels: int, number of feature channels
        depth_bins: int, number of depth bins
        feature_height: int, height of camera features
        feature_width: int, width of camera features
    
    Returns:
        bev_features: torch.Tensor of shape (1, C, bev_height, bev_width)
    """
    op = create_bevpool_op(bev_width, bev_height, num_cameras, channels, 
                          depth_bins, feature_height, feature_width)
    return op(camera_features.flatten(), depth_weights.flatten(), indices, intervals)


if __name__ == "__main__":
    # Test the BEVPool operation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # BEVPool configuration
    bev_width = 200
    bev_height = 200
    num_cameras = 6
    channels = 80
    depth_bins = 118  
    feature_height = 32
    feature_width = 88
    
    print(f"BEVPool Configuration:")
    print(f"  Cameras: {num_cameras}")
    print(f"  Channels: {channels}")
    print(f"  Depth bins: {depth_bins}")
    print(f"  Feature size: {feature_height}x{feature_width}")
    print(f"  BEV size: {bev_height}x{bev_width}")
    
    # Generate sample data
    total_features = num_cameras * channels * depth_bins * feature_height * feature_width
    total_weights = num_cameras * depth_bins * feature_height * feature_width
    
    camera_features = torch.rand(total_features, device=device, dtype=torch.float16) * 2.0 - 1.0
    depth_weights = torch.rand(total_weights, device=device, dtype=torch.float16)
    
    # Generate indices and intervals for BEV mapping
    num_intervals = bev_height * bev_width
    indices_per_pixel = 4  # Average 4 camera pixels per BEV pixel
    total_indices = num_intervals * indices_per_pixel
    
    indices = torch.randint(0, total_weights, (total_indices,), device=device, dtype=torch.int32)
    
    # Create intervals
    intervals = torch.zeros(num_intervals, 3, device=device, dtype=torch.int32)
    for i in range(num_intervals):
        start_idx = i * indices_per_pixel
        end_idx = (i + 1) * indices_per_pixel
        intervals[i] = torch.tensor([start_idx, end_idx, i], dtype=torch.int32)
    
    print(f"Input camera features: {camera_features.shape}")
    print(f"Input depth weights: {depth_weights.shape}")
    print(f"Indices: {indices.shape}")
    print(f"Intervals: {intervals.shape}")
    
    try:
        # Reshape camera features for the function call
        camera_features_reshaped = camera_features.reshape(num_cameras, channels, depth_bins, feature_height, feature_width)
        depth_weights_reshaped = depth_weights.reshape(num_cameras, depth_bins, feature_height, feature_width)
        
        bev_features = bevpool_forward(
            camera_features_reshaped,
            depth_weights_reshaped,
            indices,
            intervals,
            bev_width, bev_height, num_cameras, channels,
            depth_bins, feature_height, feature_width
        )
        
        print(f"BEVPool successful!")
        print(f"Output BEV features shape: {bev_features.shape}")
        
        # Check for non-zero values to verify computation
        non_zero_count = (bev_features != 0).sum().item()
        total_elements = bev_features.numel()
        
        print(f"Non-zero output values: {non_zero_count}/{total_elements}")
        print(f"Value range: [{bev_features.min():.6f}, {bev_features.max():.6f}]")
        print(f"Mean absolute value: {bev_features.abs().mean():.6f}")
        
    except Exception as e:
        print(f"Error during BEVPool: {e}")
        import traceback
        traceback.print_exc()