import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import numpy as np

class GeometricKernelAttnFunction(torch.autograd.Function):
    """
    Geometric Kernel Attention Function with forward and backward passes
    """
    _cuda_module = None  # Cache the compiled module
    
    @staticmethod
    def _get_cuda_module():
        if GeometricKernelAttnFunction._cuda_module is None:
            # JIT compile the CUDA extension only once
            current_dir = os.path.dirname(os.path.abspath(__file__))
            GeometricKernelAttnFunction._cuda_module = load(
                name="geometric_kernel_attn_cuda",
                sources=[
                    os.path.join(current_dir, "geometric_kernel_attn_kernel.cu"),
                    os.path.join(current_dir, "geometric_kernel_attn_binding.cpp")
                ],
                extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr'],
                verbose=False  # Set to False for cleaner output
            )
        return GeometricKernelAttnFunction._cuda_module
    
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step=64):
        # Save for backward pass
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight)
        ctx.im2col_step = im2col_step
        
        # Get cached CUDA module
        cuda_module = GeometricKernelAttnFunction._get_cuda_module()
        
        # Call forward CUDA function
        return cuda_module.forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight = ctx.saved_tensors
        im2col_step = ctx.im2col_step
        
        # Get cached CUDA module
        cuda_module = GeometricKernelAttnFunction._get_cuda_module()
        
        # Call backward CUDA function
        grad_value, grad_sampling_loc, grad_attn_weight = cuda_module.backward(
            value, spatial_shapes, level_start_index, sampling_loc, 
            attn_weight, grad_output, im2col_step
        )
        
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class GeometricKernelAttnOp(nn.Module):
    """
    Geometric Kernel Attention Operation Module
    """
    def __init__(self, im2col_step=64):
        super(GeometricKernelAttnOp, self).__init__()
        self.im2col_step = im2col_step
    
    def forward(self, value, spatial_shapes, level_start_index, sampling_loc, attn_weight):
        """
        Args:
            value: (N, S, H, C) - Input feature values
                N: batch size, S: spatial size, H: num_heads, C: channels
            spatial_shapes: (L, 2) - Spatial shape of each level [H, W]
                L: number of levels  
            level_start_index: (L,) - Starting index for each level
            sampling_loc: (N, Q, H, L, P, 2) - Sampling locations
                Q: num_queries, L: num_levels, P: num_points
            attn_weight: (N, Q, H, L, P) - Attention weights
        
        Returns:
            output: (N, Q, H, C) - Output features after geometric kernel attention
        """
        return GeometricKernelAttnFunction.apply(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight, self.im2col_step
        )


def geometric_kernel_attn_forward(value, spatial_shapes, level_start_index, 
                                 sampling_loc, attn_weight, im2col_step=64):
    """
    Geometric Kernel Attention Forward Function
    
    Performs multi-scale geometric kernel attention on input features using 
    learnable sampling locations and attention weights.
    
    Args:
        value: Input feature tensor (N, S, H, C)
        spatial_shapes: Spatial dimensions for each pyramid level (L, 2) 
        level_start_index: Starting indices for each level (L,)
        sampling_loc: Sampling locations (N, Q, H, L, P, 2)
        attn_weight: Attention weights (N, Q, H, L, P)
        im2col_step: Step size for im2col operation (default: 64)
    
    Returns:
        output: Attention output features (N, Q, H, C)
    """
    return GeometricKernelAttnFunction.apply(
        value, spatial_shapes, level_start_index, 
        sampling_loc, attn_weight, im2col_step
    )


def create_geometric_kernel_attn_op(im2col_step=64):
    """Factory function to create geometric kernel attention operation"""
    return GeometricKernelAttnOp(im2col_step=im2col_step)


if __name__ == "__main__":
    # Test the geometric kernel attention operation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        exit(1)
        
    try:
        # Test configuration
        batch_size = 2
        spatial_size = 3136  # e.g., 56*56 
        num_heads = 8
        channels = 256
        num_queries = 300
        num_levels = 4
        num_points = 4
        
        # Create test tensors
        value = torch.rand(batch_size, spatial_size, num_heads, channels, 
                          device=device, dtype=torch.float32)
        
        spatial_shapes = torch.tensor([[56, 56], [28, 28], [14, 14], [7, 7]], 
                                    device=device, dtype=torch.int64)
        
        level_start_index = torch.tensor([0, 3136, 4720, 4916], 
                                       device=device, dtype=torch.int64)
        
        sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2,
                                device=device, dtype=torch.float32) * 2 - 1  # normalize to [-1, 1]
        
        attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points,
                               device=device, dtype=torch.float32)
        
        print(f"Input shapes:")
        print(f"  Value: {value.shape}")
        print(f"  Spatial shapes: {spatial_shapes.shape}")
        print(f"  Level start index: {level_start_index.shape}")
        print(f"  Sampling loc: {sampling_loc.shape}")
        print(f"  Attention weight: {attn_weight.shape}")
        
        # Test forward pass
        output = geometric_kernel_attn_forward(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight
        )
        
        print(f"✅ Geometric kernel attention successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Output mean: {output.mean():.6f}")
        
        # Test with module
        gka_op = create_geometric_kernel_attn_op()
        output2 = gka_op(value, spatial_shapes, level_start_index, 
                        sampling_loc, attn_weight)
        
        print(f"✅ Module interface successful!")
        print(f"Outputs match: {torch.allclose(output, output2, atol=1e-6)}")
        
    except Exception as e:
        print(f"❌ Error during geometric kernel attention: {e}")
        import traceback
        traceback.print_exc()