#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <map>
#include <iostream>

struct BEVPoolParams {
    std::vector<int> camera_shape;
    unsigned int bev_width;
    unsigned int bev_height;
    unsigned int num_cameras;
    unsigned int channels;
    unsigned int depth_bins;
    unsigned int feature_height;
    unsigned int feature_width;
};

struct BEVPool {
    BEVPoolParams params;
    half* output_feature;
    int output_dims[4];
    unsigned int volumn_output;
};

extern "C" {
    BEVPool* bevpool_create(const BEVPoolParams* params);
    void bevpool_destroy(BEVPool* pool);
    half* bevpool_forward(
        BEVPool* pool,
        const half* camera_feature, 
        const half* depth_weights,
        const unsigned int* indices, 
        const int3* intervals, 
        unsigned int num_intervals,
        cudaStream_t stream
    );
    void bevpool_get_shape(BEVPool* pool, int* shape);
}

// Wrapper class to manage BEVPool instance
class BEVPoolWrapper {
public:
    BEVPool* pool_ptr;
    
    BEVPoolWrapper(std::map<std::string, int> params_dict) {
        BEVPoolParams params;
        params.bev_width = params_dict["bev_width"];
        params.bev_height = params_dict["bev_height"];
        params.num_cameras = params_dict["num_cameras"];
        params.channels = params_dict["channels"];
        params.depth_bins = params_dict["depth_bins"];
        params.feature_height = params_dict["feature_height"];
        params.feature_width = params_dict["feature_width"];
        params.camera_shape = {
            params.num_cameras,
            (int)params.channels,
            (int)params.depth_bins,
            (int)params.feature_height,
            (int)params.feature_width
        };
        
        pool_ptr = bevpool_create(&params);
    }
    
    ~BEVPoolWrapper() {
        if (pool_ptr) {
            bevpool_destroy(pool_ptr);
        }
    }
    
    torch::Tensor forward(
        torch::Tensor camera_features,
        torch::Tensor depth_weights,
        torch::Tensor indices,
        torch::Tensor intervals
    ) {
        // Check inputs
        TORCH_CHECK(camera_features.is_cuda(), "Camera features must be a CUDA tensor");
        TORCH_CHECK(depth_weights.is_cuda(), "Depth weights must be a CUDA tensor");
        TORCH_CHECK(indices.is_cuda(), "Indices must be a CUDA tensor");
        TORCH_CHECK(intervals.is_cuda(), "Intervals must be a CUDA tensor");
        
        TORCH_CHECK(camera_features.dtype() == torch::kFloat16, "Camera features must be float16");
        TORCH_CHECK(depth_weights.dtype() == torch::kFloat16, "Depth weights must be float16");
        TORCH_CHECK(indices.dtype() == torch::kInt32, "Indices must be int32");
        TORCH_CHECK(intervals.dtype() == torch::kInt32, "Intervals must be int32");
        
        TORCH_CHECK(intervals.dim() == 2 && intervals.size(1) == 3, "Intervals must be (N, 3) tensor");
        
        unsigned int num_intervals = intervals.size(0);
        
        // Call CUDA kernel
        half* result = bevpool_forward(
            pool_ptr,
            (const half*)camera_features.data_ptr<at::Half>(),
            (const half*)depth_weights.data_ptr<at::Half>(),
            (const unsigned int*)indices.data_ptr<int>(),
            (const int3*)intervals.data_ptr<int>(),  // Assuming int3 is compatible with 3 int32s
            num_intervals,
            0  // default stream
        );
        
        // Get output shape
        int shape[4];
        bevpool_get_shape(pool_ptr, shape);
        
        // Create output tensor
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(camera_features.device());
        
        auto output = torch::from_blob(
            result, 
            {shape[0], shape[1], shape[2], shape[3]}, 
            options
        );
        
        // Convert to float32 for compatibility
        return output.to(torch::kFloat32);
    }
};

// Factory function
std::shared_ptr<BEVPoolWrapper> bevpool_create_wrapper(std::map<std::string, int> params) {
    return std::make_shared<BEVPoolWrapper>(params);
}

torch::Tensor bevpool_forward_wrapper(
    std::shared_ptr<BEVPoolWrapper> pool,
    torch::Tensor camera_features,
    torch::Tensor depth_weights,
    torch::Tensor indices,
    torch::Tensor intervals
) {
    return pool->forward(camera_features, depth_weights, indices, intervals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BEVPoolWrapper>(m, "BEVPoolWrapper");
    
    m.def("create", &bevpool_create_wrapper, "Create BEVPool instance");
    m.def("forward", &bevpool_forward_wrapper, "BEVPool forward");
}