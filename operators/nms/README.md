# NMS (Non-Maximum Suppression) Benchmarks

Benchmarks for NMS performance in 2D and 3D object detection scenarios.

## Files

- `benchmark_nms.py` - General NMS performance benchmark
- `benchmark_pointtransformer_nms.py` - PointTransformer++ specific benchmark with CenterPoint config

## Usage

```bash
# General benchmark
python benchmark_nms.py

# PointTransformer++ benchmark
python benchmark_pointtransformer_nms.py
```

## Results (NVIDIA GeForce RTX 5070)

### General Object Detection
| Boxes | Latency (ms) | Performance |
|-------|--------------|-------------|
| 150   | ~0.5         | Excellent   |
| 500   | ~0.6         | Excellent   |
| 1000  | ~0.7         | Excellent   |
| 3000  | ~1.2         | Very Good   |
| 5000  | ~2.0         | Good        |

### PointTransformer++ (3D Detection)
Based on CenterPoint configuration for Waymo:
- **Input**: 4096 boxes (NMS_PRE_MAXSIZE)
- **Output**: 500 boxes max (NMS_POST_MAXSIZE)
- **IoU Threshold**: 0.7
- **Latency**: ~1.6ms for 4096 boxes

| Configuration | Boxes | IoU | Latency (ms) | Kept |
|---------------|-------|-----|--------------|------|
| Pre-NMS (typical) | 4096 | 0.7 | 1.68 | ~4028 |
| Dense Scene | 2000 | 0.7 | 1.15 | ~1986 |
| Normal Scene | 1000 | 0.7 | 0.81 | ~997 |

## Key Findings

- TorchVision NMS is highly optimized with CUDA kernels
- Sub-millisecond latency for typical detection scenarios
- NMS is not a bottleneck in object detection pipelines

## Why No Custom Implementation?

TorchVision's NMS already uses optimized CUDA kernels that are:
- Production-tested
- Well-maintained
- Near-optimal performance
- No compilation hassles

Custom implementations would require significant effort for marginal gains (~20-30% at best).