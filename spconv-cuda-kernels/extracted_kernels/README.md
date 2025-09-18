# SPConv CUDA Kernels Extraction

This directory contains extracted CUDA kernel implementations from the SPConv repository. The kernels were originally defined in Python files using pccm decorators and have been converted to pure CUDA C++ code.

## Core Sparse Convolution Operations

The main sparse convolution operations are implemented through three fundamental kernels:

1. **Gather Features** (`gather_features_kernel`): Collects input features based on sparse indices
2. **Implicit GEMM** (`sparse_conv_implicit_gemm_*`): Performs the actual convolution computation using matrix multiplication patterns
3. **Scatter Add** (`scatter_add_kernel`): Accumulates results back to the output feature map

These kernels work together to implement the sparse convolution operation:
```
Input Features → Gather → Implicit GEMM → Scatter Add → Output Features
```

## Directory Structure

```
spconv_extracted/
├── kernels/           # CUDA kernel implementations (.cu files)
├── include/           # Header files (.h files)
└── README.md         # This file
```

## Extracted Kernels

### 1. indices_kernels.cu / indices_kernels.h
**Source**: `/spconv/csrc/sparse/indices.py`

**Kernels extracted**:
- **CudaCommonKernel**:
  - `arange_kernel` - Fill array with sequential values
  - `fill_kernel` - Fill array with constant value
  - `maximum_value_kernel` - Apply maximum operation with constant

- **SparseConvIndicesKernel**:
  - `calc_conv_indices_stage1` - First stage of convolution indices calculation
  - `build_conv_hash_table` - Build hash table for convolution
  - `arange_hash_table_and_assign_out` - Assign output indices from hash table
  - `arange_hash_table` - Process hash table entries
  - `assign_out_indices` - Assign output indices from offset array
  - `calc_conv_indices_stage2` - Second stage of convolution indices calculation
  - `calc_conv_indices_stage2_bounded` - Bounded version of stage 2
  - `calc_conv_indices_stage1_mask` - Masked version of stage 1
  - `calc_conv_indices_stage1_mask_direct_table` - Direct table version
  - `calc_conv_indices_stage2_mask` - Masked version of stage 2
  - `calc_conv_indices_stage2_mask_output` - Output mask calculation
  - `calc_conv_indices_stage2_inference_mask` - Inference mask version
  - `build_subm_conv_hash_table` - Build submanifold convolution hash table
  - `clean_indices_uniq` - Clean unique indices array
  - `calc_subm_conv_indices` - Calculate submanifold convolution indices

### 2. inference_kernels.cu / inference_kernels.h
**Source**: `/spconv/csrc/sparse/inference.py`

**Kernels extracted**:
- **InferenceOpsKernel**:
  - `bias_add_inplace_kernel` - Add bias to features in-place
  - `bias_add_act_inplace_kernel` - Add bias and apply activation in-place
  - `activation_inplace_kernel` - Apply activation function in-place

### 3. maxpool_kernels.cu / maxpool_kernels.h
**Source**: `/spconv/csrc/sparse/maxpool.py`

**Kernels extracted**:
- **IndiceMaxPool**:
  - `forward_kernel` - Forward pass for max pooling
  - `forward_implicit_gemm_kernel` - Implicit GEMM forward pass
  - `backward_kernel` - Backward pass for max pooling
  - `backward_implicit_gemm_kernel` - Implicit GEMM backward pass
  - `forward_avgpool_implicit_gemm_kernel` - Average pooling forward pass
  - `backward_avgpool_implicit_gemm_kernel` - Average pooling backward pass
  - `global_pool_rearrange_kernel` - Global pooling rearrangement

### 4. pointops_kernels.cu / pointops_kernels.h
**Source**: `/spconv/csrc/sparse/pointops.py`

**Kernels extracted**:
- **Point2VoxelKernel**:
  - `build_hash_table` - Build hash table for point-to-voxel conversion
  - `assign_table` - Assign hash table entries
  - `generate_voxel` - Generate voxel data from points
  - `voxel_empty_fill_mean` - Fill empty voxel positions with mean values
  - `limit_num_per_voxel_value` - Limit number of points per voxel

### 5. hash_kernels.cu / hash_kernels.h
**Source**: `/spconv/csrc/hash/core.py`

**Kernels extracted**:
- **HashTableKernel**:
  - `insert_exist_keys_kernel` - Insert values for existing keys in hash table

### 6. sparse_conv_core.cu / sparse_conv_core.h
**Source**: Synthesized from multiple spconv sources

**Kernels extracted**:
- **Core Sparse Convolution Operations**:
  - `gather_features_kernel` - Basic gather operation for sparse features
  - `gather_features_kernel_optimized` - Memory-optimized gather with coalescing
  - `scatter_add_kernel` - Basic scatter-add operation for accumulation
  - `scatter_add_kernel_optimized` - Memory-optimized scatter-add with atomics

- **Implicit GEMM Sparse Convolution**:
  - `sparse_conv_implicit_gemm_forward_kernel` - Forward pass implicit GEMM
  - `sparse_conv_implicit_gemm_backward_input_kernel` - Input gradient computation
  - `sparse_conv_implicit_gemm_backward_filter_kernel` - Filter gradient computation
  - `sparse_conv_multi_implicit_gemm_forward_kernel` - Multi-kernel GEMM operations

- **Pooling-Style Implicit GEMM** (extracted from maxpool.py patterns):
  - `sparse_conv_maxpool_implicit_gemm_kernel` - Max pooling style operations
  - `sparse_conv_avgpool_implicit_gemm_kernel` - Average pooling style operations

- **Utility Kernels**:
  - `sparse_conv_bias_add_kernel` - Add bias to sparse features
  - `sparse_conv_activation_kernel` - Apply activation functions (ReLU, LeakyReLU)

## Key Features Preserved

1. **Template Support**: All kernels maintain their template parameters for type flexibility
2. **CUDA Kernel Structure**: Proper `__global__` function declarations
3. **TensorView Integration**: Maintains compatibility with TensorView library
4. **Original Logic**: Kernel logic is preserved exactly as in the original implementation
5. **Memory Layout**: Original memory access patterns are maintained

## Usage Notes

- These kernels require the TensorView library for compilation
- Template parameters need to be properly instantiated when using the kernels
- The kernels assume specific memory layouts and data types as used in SPConv
- Some kernels have hardcoded assumptions (e.g., 3D coordinates) that may need adjustment for different use cases

## Compilation Requirements

- CUDA Toolkit
- TensorView library
- Compatible C++ compiler (C++14 or later)

## Original License

These kernels are extracted from SPConv, which is licensed under the Apache License, Version 2.0.