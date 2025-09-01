#include "geometric_kernel_attn.h"
#include "geometric_kernel_attn_cuda.cu"

// Interface functions that dispatch to CUDA implementations
at::Tensor geometric_kernel_attn_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step) {
    
    if (value.type().is_cuda()) {
        return geometric_kernel_attn_cuda_forward(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight, im2col_step);
    }
    AT_ERROR("Geometric kernel attention not implemented on CPU");
}

std::vector<at::Tensor> geometric_kernel_attn_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step) {
    
    if (value.type().is_cuda()) {
        return geometric_kernel_attn_cuda_backward(
            value, spatial_shapes, level_start_index,
            sampling_loc, attn_weight, grad_output, im2col_step);
    }
    AT_ERROR("Geometric kernel attention backward not implemented on CPU");
}