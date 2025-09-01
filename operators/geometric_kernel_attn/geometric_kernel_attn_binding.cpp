#include <torch/extension.h>
#include <vector>
#include "geometric_kernel_attn.h"

// Forward function binding
at::Tensor geometric_kernel_attn_forward_binding(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step) {
    
    // Input validation
    TORCH_CHECK(value.is_cuda(), "Input value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "Input spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "Input level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "Input sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "Input attn_weight must be a CUDA tensor");
    
    TORCH_CHECK(value.dtype() == torch::kFloat32, "Input value must be float32");
    TORCH_CHECK(sampling_loc.dtype() == torch::kFloat32, "Input sampling_loc must be float32");
    TORCH_CHECK(attn_weight.dtype() == torch::kFloat32, "Input attn_weight must be float32");
    
    TORCH_CHECK(value.is_contiguous(), "Input value must be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "Input spatial_shapes must be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "Input level_start_index must be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "Input sampling_loc must be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "Input attn_weight must be contiguous");
    
    return geometric_kernel_attn_forward(
        value, spatial_shapes, level_start_index, 
        sampling_loc, attn_weight, im2col_step);
}

// Backward function binding
std::vector<at::Tensor> geometric_kernel_attn_backward_binding(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step) {
    
    // Input validation
    TORCH_CHECK(value.is_cuda(), "Input value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "Input spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "Input level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "Input sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "Input attn_weight must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "Input grad_output must be a CUDA tensor");
    
    TORCH_CHECK(value.dtype() == torch::kFloat32, "Input value must be float32");
    TORCH_CHECK(sampling_loc.dtype() == torch::kFloat32, "Input sampling_loc must be float32");
    TORCH_CHECK(attn_weight.dtype() == torch::kFloat32, "Input attn_weight must be float32");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "Input grad_output must be float32");
    
    TORCH_CHECK(value.is_contiguous(), "Input value must be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "Input spatial_shapes must be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "Input level_start_index must be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "Input sampling_loc must be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "Input attn_weight must be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "Input grad_output must be contiguous");
    
    return geometric_kernel_attn_backward(
        value, spatial_shapes, level_start_index,
        sampling_loc, attn_weight, grad_output, im2col_step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &geometric_kernel_attn_forward_binding, "Geometric Kernel Attention forward");
    m.def("backward", &geometric_kernel_attn_backward_binding, "Geometric Kernel Attention backward");
}