/*
Test libspconv accuracy against PyTorch spconv using the EXACT same CUDA kernels
This uses the libspconv example as a base
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include the generated libspconv headers
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>

// JSON for loading metadata
#include "/tmp/json.hpp"
using json = nlohmann::json;

using namespace spconvlib;
using StaticAllocator = spconv::csrc::sparse::alloc::StaticAllocator;
using SpconvOps = spconv::csrc::sparse::all::SpconvOps;
using ConvGemmOps = spconv::csrc::sparse::convops::spops::ConvGemmOps;
using SimpleExternalSpconvMatmul = spconv::csrc::sparse::convops::SimpleExternalSpconvMatmul;
using ConvMain = cumm::conv::main::ConvMainUnitTest;
using GemmMain = cumm::gemm::main::GemmMainUnitTest;
using GemmTunerSimple = spconv::csrc::sparse::convops::spops::GemmTuner;
using ConvTunerSimple = spconv::csrc::sparse::convops::spops::ConvTuner;

// Helper function to load binary data
template<typename T>
std::vector<T> load_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = file.tellg();
    size_t numElements = fileSize / sizeof(T);
    std::vector<T> data(numElements);

    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    return data;
}

// Load metadata
json load_metadata(const std::string& filename) {
    std::ifstream file(filename);
    json metadata;
    file >> metadata;
    return metadata;
}

// Helper to create tv::Tensor from loaded data
tv::Tensor create_tensor_from_vector(const std::vector<float>& data, std::vector<int64_t> shape, int device = 0) {
    tv::Tensor tensor = tv::empty(shape, tv::float32, device);
    if (device == 0) {
        // GPU tensor
        cudaMemcpy(tensor.raw_data(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        // CPU tensor
        memcpy(tensor.raw_data(), data.data(), data.size() * sizeof(float));
    }
    return tensor;
}

tv::Tensor create_tensor_from_vector(const std::vector<int32_t>& data, std::vector<int64_t> shape, int device = 0) {
    tv::Tensor tensor = tv::empty(shape, tv::int32, device);
    if (device == 0) {
        cudaMemcpy(tensor.raw_data(), data.data(), data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
        memcpy(tensor.raw_data(), data.data(), data.size() * sizeof(int32_t));
    }
    return tensor;
}

// Compare two tensors for accuracy
bool compare_tensors(const tv::Tensor& expected, const tv::Tensor& actual,
                     float tolerance = 1e-5f, const std::string& name = "") {
    if (expected.shape() != actual.shape()) {
        std::cerr << "Shape mismatch for " << name << std::endl;
        std::cerr << "Expected: ";
        for (auto s : expected.shape()) std::cout << s << " ";
        std::cerr << "\nActual: ";
        for (auto s : actual.shape()) std::cout << s << " ";
        std::cerr << std::endl;
        return false;
    }

    auto expected_cpu = expected.cpu();
    auto actual_cpu = actual.cpu();

    const float* exp_ptr = expected_cpu.data_ptr<float>();
    const float* act_ptr = actual_cpu.data_ptr<float>();

    size_t num_elements = expected.size();
    size_t errors = 0;
    float max_diff = 0.0f;
    float total_diff = 0.0f;

    for (size_t i = 0; i < num_elements; i++) {
        float diff = std::abs(exp_ptr[i] - act_ptr[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);

        if (diff > tolerance) {
            if (errors < 10) {  // Only print first 10 errors
                std::cerr << "Mismatch at " << i << ": expected=" << exp_ptr[i]
                          << ", actual=" << act_ptr[i] << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    float avg_diff = total_diff / num_elements;

    std::cout << name << " comparison:" << std::endl;
    std::cout << "  Max diff: " << max_diff << std::endl;
    std::cout << "  Avg diff: " << avg_diff << std::endl;
    std::cout << "  Errors: " << errors << "/" << num_elements;

    if (errors > 0) {
        std::cout << " (" << (100.0f * errors / num_elements) << "%)" << std::endl;
    } else {
        std::cout << " - PASSED ✓" << std::endl;
    }

    return errors == 0;
}

void test_subm_convolution() {
    std::cout << "\n=== Testing SubMConv3d (Submanifold Convolution) ===" << std::endl;

    // Load test data generated by PyTorch spconv
    auto input_indices_float = load_binary_file<float>("test_data/subm_input_indices.bin");
    auto input_features = load_binary_file<float>("test_data/subm_input_features.bin");
    auto weight = load_binary_file<float>("test_data/subm_weight.bin");
    auto bias = load_binary_file<float>("test_data/subm_bias.bin");
    auto expected_output = load_binary_file<float>("test_data/subm_output_features.bin");

    // Load metadata
    auto meta = load_metadata("test_data/spconv_test_metadata.json");
    int batch_size = meta["batch_size"];
    auto spatial_shape = meta["spatial_shape"].get<std::vector<int>>();
    int in_channels = meta["in_channels"];
    int out_channels = meta["out_channels"];
    int num_input_points = meta["num_input_points"];

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Spatial shape: [" << spatial_shape[0] << ", " << spatial_shape[1]
              << ", " << spatial_shape[2] << "]" << std::endl;
    std::cout << "  Input points: " << num_input_points << std::endl;
    std::cout << "  Channels: " << in_channels << " -> " << out_channels << std::endl;

    // Convert indices from float to int32
    std::vector<int32_t> input_indices_int(input_indices_float.size());
    for (size_t i = 0; i < input_indices_float.size(); i++) {
        input_indices_int[i] = static_cast<int32_t>(input_indices_float[i]);
    }

    // Create tensors
    tv::Tensor indices = create_tensor_from_vector(input_indices_int, {num_input_points, 4}, 0);
    tv::Tensor features = create_tensor_from_vector(input_features, {num_input_points, in_channels}, 0);
    tv::Tensor filters = create_tensor_from_vector(weight, {out_channels, 3, 3, 3, in_channels}, 0);
    tv::Tensor bias_tensor = create_tensor_from_vector(bias, {out_channels}, 0);

    // Convolution parameters
    std::vector<int> ksize = {3, 3, 3};
    std::vector<int> stride = {1, 1, 1};
    std::vector<int> padding = {1, 1, 1};
    std::vector<int> dilation = {1, 1, 1};

    int KV = ksize[0] * ksize[1] * ksize[2];
    bool is_subm = true;

    // Get architecture for kernel selection (same as PyTorch version)
    auto arch = ConvGemmOps::get_compute_capability();
    std::cout << "GPU architecture: " << arch[0] << "." << arch[1] << std::endl;

    // Allocate output
    tv::Tensor out_features = tv::empty({num_input_points, out_channels}, tv::float32, 0);
    cudaMemset(out_features.raw_data(), 0, out_features.size() * sizeof(float));

    // Create workspace for index generation
    int workspace_size = SpconvOps::get_indice_gen_workspace_size(
        KV, num_input_points, num_input_points, 0, is_subm, false, false);
    tv::Tensor workspace = tv::empty({workspace_size}, tv::uint8, 0);

    // Get workspace tensors
    auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
        workspace.raw_data(), KV, num_input_points, num_input_points,
        0, is_subm, false, false);

    // Allocate index pairs and other required tensors
    tv::Tensor pair = tv::empty({2, KV, num_input_points}, tv::int32, 0);
    tv::Tensor indices_kernel_num = tv::zeros({KV}, tv::int32, 0);
    tv::Tensor out_inds = tv::empty({num_input_points, 4}, tv::int32, 0);

    ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair});
    ws_tensors.insert({SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});
    ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});

    StaticAllocator alloc(ws_tensors);

    // Generate index pairs (same algorithm as PyTorch)
    int num_act_out = SpconvOps::get_indice_pairs(
        alloc, indices, batch_size, spatial_shape,
        static_cast<int>(tv::gemm::SparseConvAlgo::kNative),
        ksize, stride, padding, dilation, {0, 0, 0},
        is_subm, false, 0, num_input_points, num_input_points);

    std::cout << "Number of output points: " << num_act_out << std::endl;

    // Perform convolution using SAME kernels as PyTorch
    std::unordered_map<std::string, tv::Tensor> conv_tensors{
        {SPCONV_ALLOC_FEATURES, features},
        {SPCONV_ALLOC_FILTERS, filters},
        {SPCONV_ALLOC_OUT_FEATURES, out_features}
    };

    StaticAllocator conv_alloc(conv_tensors);
    SimpleExternalSpconvMatmul ext_mm(conv_alloc);
    GemmTunerSimple gemm_tuner(GemmMain::get_all_algo_desp());

    // This calls the EXACT same CUDA kernels as PyTorch spconv
    ConvGemmOps::indice_conv(
        conv_alloc, ext_mm, gemm_tuner,
        true,   // has_bias
        false,  // has_relu
        features, filters, pair, indices_kernel_num,
        arch,
        num_act_out,
        false,  // inverse
        is_subm,
        static_cast<int>(tv::gemm::SparseConvAlgo::kNative),
        0,      // stream
        bias_tensor,
        1.0f,   // alpha (for leaky relu)
        0.0f,   // beta
        tv::gemm::Activation::kNone);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Compare with expected output
    tv::Tensor expected = create_tensor_from_vector(expected_output,
                                                    {num_input_points, out_channels}, 0);

    bool passed = compare_tensors(expected, out_features, 1e-4f, "SubMConv3d output");

    if (passed) {
        std::cout << "\n✅ SubMConv3d test PASSED - using SAME kernels as PyTorch!" << std::endl;
    } else {
        std::cout << "\n❌ SubMConv3d test FAILED" << std::endl;
    }
}

void test_sparse_convolution() {
    std::cout << "\n=== Testing SparseConv3d (Regular Sparse Convolution) ===" << std::endl;

    // Load test data
    auto input_indices_float = load_binary_file<float>("test_data/subm_input_indices.bin");
    auto input_features = load_binary_file<float>("test_data/subm_input_features.bin");
    auto weight = load_binary_file<float>("test_data/sparse_weight.bin");
    auto bias = load_binary_file<float>("test_data/sparse_bias.bin");
    auto expected_indices_float = load_binary_file<float>("test_data/sparse_output_indices.bin");
    auto expected_output = load_binary_file<float>("test_data/sparse_output_features.bin");

    auto meta = load_metadata("test_data/spconv_test_metadata.json");
    int batch_size = meta["batch_size"];
    auto spatial_shape = meta["spatial_shape"].get<std::vector<int>>();
    int in_channels = meta["in_channels"];
    int out_channels = meta["out_channels"];
    int num_input_points = meta["num_input_points"];
    int num_output_points = meta["sparse_output_points"];

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input points: " << num_input_points << std::endl;
    std::cout << "  Expected output points: " << num_output_points << std::endl;

    // Convert indices
    std::vector<int32_t> input_indices_int(input_indices_float.size());
    for (size_t i = 0; i < input_indices_float.size(); i++) {
        input_indices_int[i] = static_cast<int32_t>(input_indices_float[i]);
    }

    // Create tensors
    tv::Tensor indices = create_tensor_from_vector(input_indices_int, {num_input_points, 4}, 0);
    tv::Tensor features = create_tensor_from_vector(input_features, {num_input_points, in_channels}, 0);
    tv::Tensor filters = create_tensor_from_vector(weight, {out_channels, 3, 3, 3, in_channels}, 0);
    tv::Tensor bias_tensor = create_tensor_from_vector(bias, {out_channels}, 0);

    // Convolution parameters (stride=2 for sparse conv)
    std::vector<int> ksize = {3, 3, 3};
    std::vector<int> stride = {2, 2, 2};  // Stride 2!
    std::vector<int> padding = {1, 1, 1};
    std::vector<int> dilation = {1, 1, 1};

    int KV = ksize[0] * ksize[1] * ksize[2];
    bool is_subm = false;  // NOT submanifold

    // Calculate output dimensions
    std::vector<int> out_dims;
    for (int i = 0; i < 3; i++) {
        int out_dim = (spatial_shape[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1) / stride[i] + 1;
        out_dims.push_back(out_dim);
    }

    int out_inds_num_limit = 100000;  // Upper bound for output indices

    // Get architecture
    auto arch = ConvGemmOps::get_compute_capability();

    // Create workspace
    int workspace_size = SpconvOps::get_indice_gen_workspace_size(
        KV, num_input_points, out_inds_num_limit, 0, is_subm, false, false);
    tv::Tensor workspace = tv::empty({workspace_size}, tv::uint8, 0);

    auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
        workspace.raw_data(), KV, num_input_points, out_inds_num_limit,
        0, is_subm, false, false);

    // Allocate tensors
    tv::Tensor pair = tv::empty({2, KV, out_inds_num_limit}, tv::int32, 0);
    tv::Tensor indices_kernel_num = tv::zeros({KV}, tv::int32, 0);
    tv::Tensor out_inds = tv::empty({out_inds_num_limit, 4}, tv::int32, 0);

    ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair});
    ws_tensors.insert({SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});
    ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});

    StaticAllocator alloc(ws_tensors);

    // Generate index pairs - using SAME algorithm as PyTorch
    int num_act_out = SpconvOps::get_indice_pairs(
        alloc, indices, batch_size, out_dims,
        static_cast<int>(tv::gemm::SparseConvAlgo::kNative),
        ksize, stride, padding, dilation, {0, 0, 0},
        is_subm, false, 0, out_inds_num_limit, num_input_points);

    std::cout << "Actual output points: " << num_act_out << std::endl;

    // Allocate output features
    tv::Tensor out_features = tv::empty({out_inds_num_limit, out_channels}, tv::float32, 0);
    cudaMemset(out_features.raw_data(), 0, out_features.size() * sizeof(float));

    // Perform convolution
    std::unordered_map<std::string, tv::Tensor> conv_tensors{
        {SPCONV_ALLOC_FEATURES, features},
        {SPCONV_ALLOC_FILTERS, filters},
        {SPCONV_ALLOC_OUT_FEATURES, out_features}
    };

    StaticAllocator conv_alloc(conv_tensors);
    SimpleExternalSpconvMatmul ext_mm(conv_alloc);
    GemmTunerSimple gemm_tuner(GemmMain::get_all_algo_desp());

    // EXACT same kernel call as PyTorch
    ConvGemmOps::indice_conv(
        conv_alloc, ext_mm, gemm_tuner,
        true, false,
        features, filters, pair, indices_kernel_num,
        arch, out_features.dim(0),
        false, is_subm,
        static_cast<int>(tv::gemm::SparseConvAlgo::kNative),
        0, bias_tensor, 1.0f, 0.0f,
        tv::gemm::Activation::kNone);

    cudaDeviceSynchronize();

    // Compare outputs
    tv::Tensor expected = create_tensor_from_vector(expected_output,
                                                    {num_output_points, out_channels}, 0);
    tv::Tensor actual_slice = out_features.slice_first_axis(0, num_act_out);

    bool passed = compare_tensors(expected, actual_slice, 1e-4f, "SparseConv3d output");

    if (passed) {
        std::cout << "\n✅ SparseConv3d test PASSED - using SAME kernels as PyTorch!" << std::endl;
    } else {
        std::cout << "\n❌ SparseConv3d test FAILED" << std::endl;
    }
}

int main() {
    std::cout << "===================================================================" << std::endl;
    std::cout << "Testing libspconv accuracy against PyTorch spconv" << std::endl;
    std::cout << "Using EXACT SAME CUDA kernels from spconv library" << std::endl;
    std::cout << "===================================================================" << std::endl;

    // Check CUDA device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // Run tests
    test_subm_convolution();
    test_sparse_convolution();

    std::cout << "\n===================================================================" << std::endl;
    std::cout << "All tests completed using IDENTICAL kernels as PyTorch spconv!" << std::endl;
    std::cout << "===================================================================" << std::endl;

    return 0;
}