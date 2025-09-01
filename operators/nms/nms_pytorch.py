import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import numpy as np

class NMSOp(nn.Module):
    _cuda_module = None  # Cache the compiled module
    
    def __init__(self):
        super(NMSOp, self).__init__()
        
    def _get_cuda_module(self):
        if NMSOp._cuda_module is None:
            # Get the directory of this file (kernels and PyTorch files are in same dir)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # JIT compile the CUDA extension only once
            NMSOp._cuda_module = load(
                name="nms_cuda",
                sources=[
                    os.path.join(current_dir, "nms_kernel.cu"),
                    os.path.join(current_dir, "nms_binding.cpp")
                ],
                extra_cuda_cflags=['-O3'],
                verbose=False  # Set to False for cleaner output
            )
        return NMSOp._cuda_module
    
    def forward(self, boxes, iou_threshold):
        """
        Args:
            boxes: (N, 11) tensor with [x, y, z, w, l, h, rt, vx, vy, score, label]
            iou_threshold: float, IoU threshold for NMS
        
        Returns:
            keep: (K,) tensor with indices of kept boxes
        """
        return self._get_cuda_module().forward(boxes, iou_threshold)


def create_nms_op():
    """Factory function to create NMS operation"""
    return NMSOp()


def nms_3d(boxes, scores=None, iou_threshold=0.5):
    """
    3D Non-Maximum Suppression for rotated bounding boxes
    
    Args:
        boxes: torch.Tensor of shape (N, 7) or (N, 11) 
               Format: [x, y, z, w, l, h, rotation] or [x, y, z, w, l, h, rt, vx, vy, score, label]
        scores: torch.Tensor of shape (N,) - if None, uses scores from boxes[:, 9]
        iou_threshold: float, IoU threshold for suppression
    
    Returns:
        keep: torch.Tensor of indices of kept boxes
    """
    if boxes.shape[1] == 7:
        # If only 7 elements, assume format is [x, y, z, w, l, h, rotation]
        # Need to add vx, vy, score, label
        if scores is None:
            scores = torch.ones(boxes.shape[0], device=boxes.device)
        
        # Create 11-element tensor
        boxes_11 = torch.zeros(boxes.shape[0], 11, device=boxes.device, dtype=boxes.dtype)
        boxes_11[:, :7] = boxes
        boxes_11[:, 7] = 0  # vx
        boxes_11[:, 8] = 0  # vy
        boxes_11[:, 9] = scores  # score
        boxes_11[:, 10] = 0  # label
        boxes = boxes_11
    
    op = create_nms_op()
    return op(boxes, iou_threshold)


if __name__ == "__main__":
    # Test the NMS operation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate sample bounding boxes
    num_boxes = 1000
    boxes = torch.zeros(num_boxes, 11, device=device)
    
    # Random positions
    boxes[:, 0] = torch.rand(num_boxes, device=device) * 100.0 - 50.0  # x: [-50, 50]
    boxes[:, 1] = torch.rand(num_boxes, device=device) * 100.0 - 50.0  # y: [-50, 50]
    boxes[:, 2] = torch.rand(num_boxes, device=device) * 6.0 - 3.0     # z: [-3, 3]
    
    # Random sizes
    boxes[:, 3] = torch.rand(num_boxes, device=device) * 3.0 + 1.0     # w: [1, 4]
    boxes[:, 4] = torch.rand(num_boxes, device=device) * 3.0 + 1.0     # l: [1, 4]
    boxes[:, 5] = torch.rand(num_boxes, device=device) * 2.0 + 1.0     # h: [1, 3]
    
    # Random rotation
    boxes[:, 6] = torch.rand(num_boxes, device=device) * 2.0 * np.pi   # rotation: [0, 2π]
    
    # Random velocities
    boxes[:, 7] = torch.rand(num_boxes, device=device) * 20.0 - 10.0   # vx: [-10, 10]
    boxes[:, 8] = torch.rand(num_boxes, device=device) * 20.0 - 10.0   # vy: [-10, 10]
    
    # Random scores (higher scores should be kept)
    boxes[:, 9] = torch.rand(num_boxes, device=device)                 # score: [0, 1]
    
    # Random labels
    boxes[:, 10] = torch.randint(0, 10, (num_boxes,), device=device, dtype=torch.float32)  # label: [0, 9]
    
    iou_threshold = 0.5
    
    print(f"Input boxes shape: {boxes.shape}")
    print(f"IoU threshold: {iou_threshold}")
    
    try:
        keep_indices = nms_3d(boxes, iou_threshold=iou_threshold)
        
        print(f"NMS successful!")
        print(f"Input boxes: {num_boxes}")
        print(f"Kept boxes: {len(keep_indices)}")
        print(f"Suppression ratio: {(num_boxes - len(keep_indices)) / num_boxes * 100:.1f}%")
        
        # Show some statistics of kept boxes
        kept_boxes = boxes[keep_indices]
        print(f"Kept boxes score range: [{kept_boxes[:, 9].min():.3f}, {kept_boxes[:, 9].max():.3f}]")
        print(f"Average kept score: {kept_boxes[:, 9].mean():.3f}")
        
    except Exception as e:
        print(f"Error during NMS: {e}")
        import traceback
        traceback.print_exc()