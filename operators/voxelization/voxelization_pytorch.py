import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import numpy as np

class VoxelizationOp(nn.Module):
    _cuda_module = None  # Cache the compiled module
    
    def __init__(self):
        super(VoxelizationOp, self).__init__()
        
    def _get_cuda_module(self):
        if VoxelizationOp._cuda_module is None:
            # Get the directory of this file (kernels and PyTorch files are in same dir)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # JIT compile the CUDA extension only once
            VoxelizationOp._cuda_module = load(
                name="voxelization_cuda",
                sources=[
                    os.path.join(current_dir, "voxelization_kernel.cu"),
                    os.path.join(current_dir, "voxelization_binding.cpp")
                ],
                extra_cuda_cflags=['-O3'],
                verbose=False  # Set to False for cleaner output
            )
        return VoxelizationOp._cuda_module
    
    def forward(self, points, voxel_params):
        """
        Args:
            points: (N, 5) tensor with [x, y, z, intensity, time]
            voxel_params: dict with voxelization parameters
        
        Returns:
            voxel_features: (max_voxels, feature_num) tensor
            voxel_coords: (max_voxels, 3) tensor 
            num_points_per_voxel: (max_voxels,) tensor
        """
        return self._get_cuda_module().forward(points, voxel_params)


def create_voxelization_op():
    """Factory function to create voxelization operation"""
    return VoxelizationOp()


def voxelize_points(points, voxel_params=None):
    """
    Voxelize point cloud data
    
    Args:
        points: torch.Tensor of shape (N, 5) with [x, y, z, intensity, time]
        voxel_params: dict with voxelization parameters
    
    Returns:
        voxel_features: torch.Tensor of shape (max_voxels, feature_num)
        voxel_coords: torch.Tensor of shape (max_voxels, 3)
        num_points_per_voxel: torch.Tensor of shape (max_voxels,)
    """
    if voxel_params is None:
        voxel_params = {
            'min_x_range': -54.0,
            'max_x_range': 54.0,
            'min_y_range': -54.0, 
            'max_y_range': 54.0,
            'min_z_range': -5.0,
            'max_z_range': 3.0,
            'voxel_x_size': 0.075,
            'voxel_y_size': 0.075,
            'voxel_z_size': 0.2,
            'max_points_per_voxel': 10,
            'grid_x_size': 100,  # Much smaller for testing
            'grid_y_size': 100,  # Much smaller for testing
            'grid_z_size': 10,
            'feature_num': 5
        }
    
    op = create_voxelization_op()
    return op(points, voxel_params)


if __name__ == "__main__":
    # Test the voxelization operation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate sample point cloud
    num_points = 10000
    points = torch.rand(num_points, 5, device=device) 
    
    # Scale to reasonable ranges
    points[:, 0] = points[:, 0] * 108.0 - 54.0  # x: [-54, 54]
    points[:, 1] = points[:, 1] * 108.0 - 54.0  # y: [-54, 54] 
    points[:, 2] = points[:, 2] * 8.0 - 5.0     # z: [-5, 3]
    points[:, 3] = points[:, 3]                 # intensity: [0, 1]
    points[:, 4] = points[:, 4] * 0.1           # time: [0, 0.1]
    
    print(f"Input points shape: {points.shape}")
    print(f"Point range - X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Point range - Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Point range - Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    try:
        voxel_features, voxel_coords, num_points_per_voxel = voxelize_points(points)
        
        # Find valid voxels (non-zero features)
        valid_mask = num_points_per_voxel > 0
        valid_voxels = valid_mask.sum().item()
        
        print(f"Voxelization successful!")
        print(f"Output voxel features shape: {voxel_features.shape}")
        print(f"Output voxel coords shape: {voxel_coords.shape}")
        print(f"Valid voxels: {valid_voxels}")
        print(f"Average points per valid voxel: {num_points_per_voxel[valid_mask].float().mean():.2f}")
        
    except Exception as e:
        print(f"Error during voxelization: {e}")
        import traceback
        traceback.print_exc()