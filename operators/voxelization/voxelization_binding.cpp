#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

// Forward declarations
struct VoxelParams {
    float min_x_range = -54.0f;
    float max_x_range = 54.0f;
    float min_y_range = -54.0f;
    float max_y_range = 54.0f;
    float min_z_range = -5.0f;
    float max_z_range = 3.0f;
    float voxel_x_size = 0.075f;
    float voxel_y_size = 0.075f;
    float voxel_z_size = 0.2f;
    int max_points_per_voxel = 10;
    int grid_x_size = 1440;
    int grid_y_size = 1440;
    int grid_z_size = 40;
    int feature_num = 5;
};

extern "C" void voxelizationLaunch(
    float* points, int points_size,
    VoxelParams& params,
    uint64_t* hash_table, int hash_table_size,
    float* voxels_temp, int* voxel_point_mask,
    __half* voxel_features, int* real_voxel_num,
    uint64_t* voxel_num_points, int* voxel_idxs
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> voxelization_forward(
    torch::Tensor points,
    std::map<std::string, float> voxel_params_dict
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
    params.min_x_range = voxel_params_dict["min_x_range"];
    params.max_x_range = voxel_params_dict["max_x_range"];
    params.min_y_range = voxel_params_dict["min_y_range"];
    params.max_y_range = voxel_params_dict["max_y_range"];
    params.min_z_range = voxel_params_dict["min_z_range"];
    params.max_z_range = voxel_params_dict["max_z_range"];
    params.voxel_x_size = voxel_params_dict["voxel_x_size"];
    params.voxel_y_size = voxel_params_dict["voxel_y_size"];
    params.voxel_z_size = voxel_params_dict["voxel_z_size"];
    params.max_points_per_voxel = (int)voxel_params_dict["max_points_per_voxel"];
    params.grid_x_size = (int)voxel_params_dict["grid_x_size"];
    params.grid_y_size = (int)voxel_params_dict["grid_y_size"];
    params.grid_z_size = (int)voxel_params_dict["grid_z_size"];
    params.feature_num = feature_num;
    
    int max_voxel_num = params.grid_x_size * params.grid_y_size * params.grid_z_size;
    int hash_table_size = max_voxel_num * 2; // 2x for better hash performance
    
    // Allocate GPU memory
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    auto hash_table = torch::empty({hash_table_size}, torch::TensorOptions().dtype(torch::kInt64).device(points.device()));
    auto voxels_temp = torch::zeros({max_voxel_num, params.max_points_per_voxel, feature_num}, options);
    auto voxel_point_mask = torch::zeros({max_voxel_num, params.max_points_per_voxel}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    auto voxel_features = torch::zeros({max_voxel_num, feature_num}, torch::TensorOptions().dtype(torch::kFloat16).device(points.device()));
    auto voxel_num_points = torch::zeros({hash_table_size}, torch::TensorOptions().dtype(torch::kInt64).device(points.device()));
    auto voxel_idxs = torch::zeros({max_voxel_num}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    auto real_voxel_num_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    
    // Initialize hash table with max values
    hash_table.fill_(UINT64_MAX);
    
    // Call CUDA kernel
    voxelizationLaunch(
        points.data_ptr<float>(),
        points_size,
        params,
        (uint64_t*)hash_table.data_ptr<int64_t>(),
        hash_table_size,
        voxels_temp.data_ptr<float>(),
        voxel_point_mask.data_ptr<int>(),
        (__half*)voxel_features.data_ptr<at::Half>(),
        real_voxel_num_tensor.data_ptr<int>(),
        (uint64_t*)voxel_num_points.data_ptr<int64_t>(),
        voxel_idxs.data_ptr<int>()
    );
    
    // Generate voxel coordinates
    auto voxel_coords = torch::zeros({max_voxel_num, 3}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    
    // Simple coordinate generation (this could be optimized with CUDA kernel)
    auto voxel_coords_cpu = voxel_coords.cpu();
    for (int i = 0; i < max_voxel_num; i++) {
        int z = i / (params.grid_y_size * params.grid_x_size);
        int y = (i % (params.grid_y_size * params.grid_x_size)) / params.grid_x_size;
        int x = i % params.grid_x_size;
        voxel_coords_cpu[i][0] = x;
        voxel_coords_cpu[i][1] = y;
        voxel_coords_cpu[i][2] = z;
    }
    voxel_coords.copy_(voxel_coords_cpu);
    
    // Convert voxel_num_points to appropriate format
    auto num_points_per_voxel = torch::zeros({max_voxel_num}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));
    
    return std::make_tuple(
        voxel_features.to(torch::kFloat32),  // Convert back to float32 for compatibility
        voxel_coords,
        num_points_per_voxel
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &voxelization_forward, "Voxelization forward");
}