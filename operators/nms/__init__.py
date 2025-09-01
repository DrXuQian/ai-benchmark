"""
NMS (Non-Maximum Suppression) Operation

3D Non-Maximum Suppression for rotated bounding boxes with PyTorch integration.
"""

from .nms_pytorch import nms_3d, create_nms_op, NMSOp

__all__ = [
    'nms_3d',
    'create_nms_op',
    'NMSOp'
]