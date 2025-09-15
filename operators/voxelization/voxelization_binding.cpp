#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <climits>
#include <cstdint>
#include "voxelization_kernel.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> voxelization_forward(
    torch::Tensor points,
    std::map<std::string, double> voxel_params_dict
) {
    // Check inputs
    TORCH_CHECK(points.is_cuda(), "Points must be a CUDA tensor");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "Points must be float32");
    TORCH_CHECK(points.dim() == 2, "Points must be 2D tensor");
    TORCH_CHECK(points.size(1) >= 5, "Points must have at least 5 features");

    int points_size = points.size(0);
    int feature_num = points.size(1);

    // Setup voxel parameters
    VoxelParams params;
    params.min_x_range = (float)voxel_params_dict["min_x_range"];
    params.max_x_range = (float)voxel_params_dict["max_x_range"];
    params.min_y_range = (float)voxel_params_dict["min_y_range"];
    params.max_y_range = (float)voxel_params_dict["max_y_range"];
    params.min_z_range = (float)voxel_params_dict["min_z_range"];
    params.max_z_range = (float)voxel_params_dict["max_z_range"];
    params.voxel_x_size = (float)voxel_params_dict["voxel_x_size"];
    params.voxel_y_size = (float)voxel_params_dict["voxel_y_size"];
    params.voxel_z_size = (float)voxel_params_dict["voxel_z_size"];
    params.max_points_per_voxel = (int)voxel_params_dict["max_points_per_voxel"];
    params.grid_x_size = params.getGridXSize();
    params.grid_y_size = params.getGridYSize();
    params.grid_z_size = params.getGridZSize();
    params.feature_num = feature_num;
    params.max_voxels = (int)voxel_params_dict.count("max_voxels") ?
                        (int)voxel_params_dict["max_voxels"] : params.max_voxels;

    // Calculate hash table size (similar to NVIDIA implementation)
    int hash_table_size = points_size * 2 * 2;

    // Allocate GPU memory
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    auto hash_table = torch::empty({hash_table_size}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    auto voxels_temp = torch::zeros({params.max_voxels, params.max_points_per_voxel, feature_num}, options);
    auto voxel_features = torch::zeros({params.max_voxels, feature_num}, torch::TensorOptions().dtype(torch::kFloat16).device(points.device()));
    auto num_points_per_voxel = torch::zeros({params.max_voxels}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    auto voxel_indices = torch::zeros({params.max_voxels, 4}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    auto real_voxel_num = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));

    // Initialize hash table with UINT32_MAX (empty key)
    hash_table.fill_(UINT32_MAX);

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Call CUDA kernel using NVIDIA implementation
    cudaError_t err = voxelizationLaunch(
        points.data_ptr<float>(),
        points_size,
        params.min_x_range, params.max_x_range,
        params.min_y_range, params.max_y_range,
        params.min_z_range, params.max_z_range,
        params.voxel_x_size, params.voxel_y_size, params.voxel_z_size,
        params.grid_y_size, params.grid_x_size, feature_num, params.max_voxels,
        params.max_points_per_voxel,
        (unsigned int*)hash_table.data_ptr<int>(),
        (unsigned int*)num_points_per_voxel.data_ptr<int>(),
        voxels_temp.data_ptr<float>(),
        (unsigned int*)voxel_indices.data_ptr<int>(),
        (unsigned int*)real_voxel_num.data_ptr<int>(),
        stream
    );

    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error in voxelizationLaunch: ", cudaGetErrorString(err));
    }

    // Feature extraction
    cudaMemcpyAsync(&params.max_voxels, real_voxel_num.data_ptr<int>(),
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    err = featureExtractionLaunch(
        voxels_temp.data_ptr<float>(),
        (unsigned int*)num_points_per_voxel.data_ptr<int>(),
        params.max_voxels,
        params.max_points_per_voxel,
        feature_num,
        (half*)voxel_features.data_ptr<at::Half>(),
        stream
    );

    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error in featureExtractionLaunch: ", cudaGetErrorString(err));
    }

    cudaStreamSynchronize(stream);

    // Extract voxel coordinates from voxel_indices (batch, z, y, x format)
    auto voxel_coords = voxel_indices.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});

    return std::make_tuple(
        voxel_features.to(torch::kFloat32),  // Convert back to float32 for compatibility
        voxel_coords,
        num_points_per_voxel
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &voxelization_forward, "Voxelization forward (NVIDIA implementation)");
}