# AI Benchmark CUDA Operations

This project provides PyTorch wrappers for optimized CUDA kernels including voxelization, NMS, and BEVPool operations commonly used in AI applications, particularly for autonomous driving and 3D perception tasks.

## Features

1. **Voxelization Kernel** - Converts 3D point cloud data into voxel grid representation with PyTorch integration
2. **NMS (Non-Maximum Suppression) Kernel** - Filters overlapping 3D bounding boxes with rotated IoU support
3. **BEVPool Kernel** - Transforms multi-camera features into Bird's Eye View representation for multi-camera fusion
4. **Geometric Kernel Attention** - Multi-scale geometric kernel attention for spatial feature reasoning (from MapTR)

## Key Improvements

✅ **Modular Architecture**: Kernels organized in separate directories with clean interfaces  
✅ **PyTorch Integration**: Full PyTorch tensor support with JIT compilation  
✅ **Memory Optimization**: Efficient GPU memory management and error handling  
✅ **Build System**: Automated compilation with setup.py and pip installation  
✅ **Testing Suite**: Comprehensive tests for both PyTorch wrappers and direct kernel execution

## Project Structure

```
ai-benchmark/
├── operators/                        # Self-contained operator modules  
│   ├── __init__.py                   # Main operators package
│   ├── voxelization/                 # Voxelization operator
│   │   ├── __init__.py               # Module interface
│   │   ├── voxelization_kernel.cu    # CUDA kernel implementation
│   │   ├── voxelization_kernel.h     # Kernel header
│   │   ├── voxelization_pytorch.py  # PyTorch wrapper
│   │   └── voxelization_binding.cpp # C++ binding
│   ├── nms/                          # NMS operator  
│   │   ├── __init__.py               # Module interface
│   │   ├── nms_kernel.cu             # CUDA kernel implementation
│   │   ├── nms_kernel.h              # Kernel header
│   │   ├── nms_pytorch.py            # PyTorch wrapper
│   │   └── nms_binding.cpp           # C++ binding
│   ├── bevpool/                      # BEVPool operator
│   │   ├── __init__.py               # Module interface  
│   │   ├── bevpool_kernel.cu         # CUDA kernel implementation
│   │   ├── bevpool_kernel.h          # Kernel header
│   │   ├── bevpool_pytorch.py        # PyTorch wrapper
│   │   └── bevpool_binding.cpp       # C++ binding
│   └── geometric_kernel_attn/        # Geometric Kernel Attention operator
│       ├── __init__.py               # Module interface
│       ├── geometric_kernel_attn_kernel.cu    # CUDA kernel implementation
│       ├── geometric_kernel_attn.h   # Kernel header
│       ├── geometric_kernel_attn_pytorch.py  # PyTorch wrapper
│       └── geometric_kernel_attn_binding.cpp # C++ binding
├── test_pytorch_ops.py               # PyTorch operation tests
├── test_new_structure.py             # Structure verification test
├── test_geometric_kernel_attn.py     # Geometric kernel attention test
└── README.md                         # This file
```

## Installation

### Prerequisites
- CUDA Toolkit (version 11.0 or later, tested with 12.x)
- PyTorch with CUDA support
- NVIDIA GPU with compute capability 7.0 or higher
- C++ compiler with C++14 support
- Python 3.7+

### Self-Contained Setup (No Build Required!)

1. **Install PyTorch with CUDA**:
   ```bash
   pip install torch numpy
   ```

2. **That's it!** Each operator is completely self-contained and will automatically compile when first used.

### JIT Compilation (Automatic)
All operators use just-in-time compilation with `torch.utils.cpp_extension.load()`. The CUDA kernels are automatically compiled when you first import and use an operator - no manual build steps, no setup.py, no configuration needed!

## Usage

### Voxelization
```python
from operators.voxelization import voxelize_points

# Generate point cloud (N, 5) - [x, y, z, intensity, time]
points = torch.rand(10000, 5, device='cuda')

# Voxelize with custom parameters
voxel_features, voxel_coords, num_points = voxelize_points(points)
```

### NMS
```python
from operators.nms import nms_3d

# Bounding boxes (N, 11) - [x, y, z, w, l, h, rotation, vx, vy, score, label]
boxes = torch.rand(1000, 11, device='cuda')

# Apply NMS
keep_indices = nms_3d(boxes, iou_threshold=0.5)
filtered_boxes = boxes[keep_indices]
```

### BEVPool
```python
from operators.bevpool import bevpool_forward

# Camera features and depth weights
camera_features = torch.rand(6, 80, 118, 32, 88, device='cuda', dtype=torch.float16)
depth_weights = torch.rand(6, 118, 32, 88, device='cuda', dtype=torch.float16)

# Generate BEV features (indices and intervals are application-specific)
bev_features = bevpool_forward(
    camera_features, depth_weights, indices, intervals,
    bev_width=200, bev_height=200, num_cameras=6, channels=80
)
```

### Geometric Kernel Attention
```python
from operators.geometric_kernel_attn import geometric_kernel_attn_forward

# Multi-scale feature pyramid inputs
batch_size, num_queries, num_heads, channels = 2, 100, 8, 256
spatial_size = 1568  # Combined spatial size across levels

value = torch.rand(batch_size, spatial_size, num_heads, channels, device='cuda')
spatial_shapes = torch.tensor([[28, 28], [14, 14], [7, 7], [4, 4]], device='cuda')
level_start_index = torch.tensor([0, 784, 980, 1029], device='cuda')
sampling_loc = torch.rand(batch_size, num_queries, num_heads, 4, 4, 2, device='cuda') * 2 - 1
attn_weight = torch.rand(batch_size, num_queries, num_heads, 4, 4, device='cuda')

# Apply geometric kernel attention
output = geometric_kernel_attn_forward(
    value, spatial_shapes, level_start_index, 
    sampling_loc, attn_weight, im2col_step=64
)  # Output: (batch_size, num_queries, num_heads, channels)
```

### Unified Import (Alternative)
```python
# Import all operators from main package
from operators import voxelize_points, nms_3d, bevpool_forward, geometric_kernel_attn_forward

# Or import individual operators  
import operators.voxelization as vox
import operators.nms as nms
import operators.bevpool as bevpool
import operators.geometric_kernel_attn as gka
```

## Testing

### Comprehensive Test Suite
```bash
python test_pytorch_ops.py
```
This test verifies:
- ✅ CUDA availability and device properties
- ✅ Kernel compilation and execution
- ✅ Correctness of outputs  
- ✅ Performance measurements
- ✅ Memory management

## Kernel Details

### Voxelization Kernel

**Purpose**: Converts unstructured 3D point cloud data into a structured voxel grid representation suitable for 3D CNN processing.

**Key Features**:
- Hash-based voxel indexing for efficient lookup
- Configurable voxel grid parameters
- Point filtering based on spatial ranges  
- Feature extraction and averaging per voxel
- Half-precision output for memory efficiency

**Parameters**:
- Input: Point cloud (N×5: x,y,z,intensity,time)
- Output: Voxel features, voxel indices, point counts
- Grid size: 1440×1440×40 voxels (configurable)
- Voxel size: 0.075m × 0.075m × 0.2m (configurable)

### NMS Kernel

**Purpose**: Removes overlapping 3D bounding boxes to eliminate duplicate detections in object detection.

**Key Features**:
- GPU-accelerated rotated bounding box IoU computation
- Polygon intersection calculation for arbitrary orientations
- Efficient bit-mask based suppression
- Supports thousands of boxes in parallel

**Parameters**:
- Input: Bounding boxes (N×11: x,y,z,w,l,h,rotation,vx,vy,score,label)
- Output: Indices of boxes to keep after NMS
- IoU threshold: 0.5 (configurable)
- Supports rotated 3D boxes

### BEVPool Kernel

**Purpose**: Transforms multi-camera feature representations into unified Bird's Eye View (BEV) features for 3D perception tasks.

**Key Features**:
- Half-precision floating point operations for memory efficiency
- Vectorized memory access using aligned data structures
- Depth-weighted feature aggregation from multiple camera views
- Optimized for high-throughput BEV feature generation

**Parameters**:
- Input: Multi-camera features (N×C×D×H×W), depth weights, spatial indices, intervals
- Output: BEV features (1×C×H_bev×W_bev)
- Default config: 6 cameras, 80 channels, 118 depth bins
- Camera features: 32×88 resolution per camera
- BEV output: 200×200 grid (configurable)

## Performance

Typical performance on RTX 3080:
- **Voxelization**: ~5-15ms for 100K points → 50K voxels
- **NMS**: ~2-8ms for 5K boxes → 500 kept boxes
- **BEVPool**: ~3-10ms for 6 cameras → 200×200 BEV features

Performance scales with:
- Point cloud density (voxelization)
- Number of input boxes (NMS)
- IoU threshold (NMS)
- Number of cameras and BEV resolution (BEVPool)
- Feature channel count (BEVPool)

## Integration

These kernels can be integrated into larger pipelines:

```cpp
// Voxelization example
VoxelParams params; // Configure grid parameters
voxelizationLaunch(d_points, points_size, params, ...);

// NMS example  
std::vector<int> keep = nms_cpu(boxes, iou_threshold);

// BEVPool example
BEVPoolParams bevParams; // Configure BEV parameters  
BEVPool bevpool(bevParams);
half* bev_features = bevpool.forward(d_camera_features, d_depth_weights, 
                                    d_indices, d_intervals, num_intervals);
```

## Customization

All kernels support various customizations:

**Voxelization**:
- Modify `VoxelParams` for different grid sizes/voxel dimensions
- Adjust feature dimensions and processing
- Change spatial ranges for different environments

**NMS**:
- Modify IoU threshold for different suppression levels
- Adjust `NMS_THREADS_PER_BLOCK` for different GPU architectures
- Support different box formats by modifying `BoundingBox` struct

**BEVPool**:
- Modify `BEVPoolParams` for different camera configurations
- Adjust BEV output resolution and feature channels
- Change `tile_size` for different GPU memory bandwidth
- Support different camera feature formats and depth representations

## Memory Requirements

**Voxelization**:
- Point cloud: `N_points × 5 × 4` bytes
- Voxel features: `N_voxels × 5 × 2` bytes (half precision)
- Hash table: `hash_size × 8` bytes

**NMS**:
- Boxes: `N_boxes × 11 × 4` bytes  
- Mask: `N_boxes × ceil(N_boxes/64) × 8` bytes

**BEVPool**:
- Camera features: `N_cameras × C × D × H × W × 2` bytes (half precision)
- Depth weights: `N_cameras × D × H × W × 2` bytes (half precision)
- BEV output: `C × H_bev × W_bev × 2` bytes (half precision)
- Indices and intervals: Variable based on camera-to-BEV mapping density

## License

These implementations are based on the original NVIDIA Lidar AI Solution repository and follow the same licensing terms.