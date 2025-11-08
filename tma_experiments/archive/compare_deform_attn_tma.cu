#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <map>
#include <stdexcept>
#include <numeric>
#include <algorithm>

// Copy necessary utilities from deform_attn.cu (without main)
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])

// Original kernel declaration (copy from deform_attn.cu)
template <typename scalar_t=__half, const int NUM_POINT= 8, const int NUM_LEVELS=4, const int CHANNELS = 32,
                                    const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
                                    const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3>
__global__ void ms_deformable_im2col_gpu_kernel_template(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_query,
    scalar_t *data_col);

template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col);

// Include utility functions
template <typename T>
std::vector<T> read_bin_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.seekg(0, std::ios::end);
    long long file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(file_size / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

template<typename T>
T get_param(const std::map<std::string, std::string>& args, const std::string& key, T default_value) {
    if (args.count(key)) {
        try {
            if constexpr (std::is_same_v<T, std::string>) {
                return args.at(key);
            } else if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(std::stoll(args.at(key)));
            } else if constexpr (std::is_floating_point_v<T>) {
                return static_cast<T>(std::stod(args.at(key)));
            } else {
                static_assert(sizeof(T) == 0, "Unsupported type for get_param");
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse '" << key << "'. Using default value." << std::endl;
            return default_value;
        }
    }
    return default_value;
}

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args_map;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        size_t pos = arg.find('=');
        if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            args_map[key] = value;
        }
    }
    return args_map;
}

// TMA kernel declarations
template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda_tma(
    cudaStream_t stream,
    const scalar_t *data_value,
    const CUtensorMap *tma_descriptors,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *data_col);

extern "C"
CUresult createTMADescriptorsForAllBatches(
    CUtensorMap* h_descriptors,
    const void** d_value_ptrs,
    const int64_t* h_spatial_shapes,
    const int64_t* h_level_start_index,
    int batch_size,
    int num_levels,
    int channels);

float compare_outputs(const std::vector<__half>& a, const std::vector<__half>& b) {
    if (a.size() != b.size()) {
        printf("Size mismatch: %zu vs %zu\n", a.size(), b.size());
        return -1.0f;
    }

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_mismatches = 0;
    const float threshold = 0.01f;  // 1% relative error threshold

    for (size_t i = 0; i < a.size(); i++) {
        float val_a = __half2float(a[i]);
        float val_b = __half2float(b[i]);
        float diff = std::abs(val_a - val_b);
        float relative_diff = diff / (std::abs(val_a) + 1e-6f);

        max_diff = std::max(max_diff, diff);
        avg_diff += diff;

        if (relative_diff > threshold) {
            if (num_mismatches < 10) {  // Print first 10 mismatches
                printf("  Mismatch at %zu: original=%.6f, tma=%.6f, diff=%.6f (%.2f%%)\n",
                       i, val_a, val_b, diff, relative_diff * 100.0f);
            }
            num_mismatches++;
        }
    }

    avg_diff /= a.size();
    printf("Comparison results:\n");
    printf("  Max diff: %.6f\n", max_diff);
    printf("  Avg diff: %.6f\n", avg_diff);
    printf("  Mismatches (>%.0f%%): %d / %zu (%.2f%%)\n",
           threshold * 100.0f, num_mismatches, a.size(),
           100.0f * num_mismatches / a.size());

    return max_diff;
}

int main(int argc, char* argv[]) {
    printf("=== Comparing Original vs TMA Deformable Attention ===\n\n");

    // Parse arguments (same as original)
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " [key=value] ..." << std::endl;
        std::cout << "Example: " << argv[0] << " batch=48 spatial_size=20522 dir=data/binary_400x800/cross_attention_cut" << std::endl;
        return 1;
    }

    auto args = parse_args(argc, argv);

    const int batch               = get_param<int>(args, "batch", 48);
    const int spatial_size        = get_param<int>(args, "spatial_size", 20522);
    const int num_query           = get_param<int>(args, "num_query", 123376);
    const int num_heads           = get_param<int>(args, "num_heads", 1);
    const int channels            = get_param<int>(args, "channels", 32);
    const int num_levels          = get_param<int>(args, "num_levels", 4);
    const int num_points          = get_param<int>(args, "num_points", 8);
    const int im2col_step         = get_param<int>(args, "im2col_step", 64);
    const std::string data_dir    = get_param<std::string>(args, "dir", "data/binary_400x800/cross_attention");

    printf("Configuration:\n");
    printf("  batch=%d, spatial_size=%d, num_query=%d\n", batch, spatial_size, num_query);
    printf("  num_heads=%d, channels=%d, num_levels=%d, num_points=%d\n",
           num_heads, channels, num_levels, num_points);
    printf("  data_dir=%s\n\n", data_dir.c_str());

    // Load data
    long long value_elements = batch * spatial_size * channels;
    printf("Loading data from .bin files...\n");
    auto h_value = read_bin_file<__half>(data_dir + "/value.bin");
    auto h_spatial_shapes = read_bin_file<int64_t>(data_dir + "/spatial_shapes.bin");
    auto h_level_start_index = read_bin_file<int64_t>(data_dir + "/level_start_index.bin");
    auto h_sampling_loc = read_bin_file<__half>(data_dir + "/sampling_locations.bin");
    auto h_attn_weight = read_bin_file<__half>(data_dir + "/attention_weights.bin");

    printf("  value: %zu elements\n", h_value.size());
    printf("  spatial_shapes: %zu elements\n", h_spatial_shapes.size());
    printf("  level_start_index: %zu elements\n", h_level_start_index.size());
    printf("  sampling_loc: %zu elements\n", h_sampling_loc.size());
    printf("  attn_weight: %zu elements\n\n", h_attn_weight.size());

    // Allocate device memory
    __half *d_value, *d_sampling_loc, *d_attn_weight, *d_output_orig, *d_output_tma;
    int64_t *d_spatial_shapes, *d_level_start_index;

    CUDA_CHECK(cudaMalloc(&d_value, h_value.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, h_sampling_loc.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, h_attn_weight.size() * sizeof(__half)));

    long long output_elements = batch * num_query * num_heads * channels;
    CUDA_CHECK(cudaMalloc(&d_output_orig, output_elements * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output_tma, output_elements * sizeof(__half)));

    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), h_value.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), h_sampling_loc.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), h_attn_weight.size() * sizeof(__half), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ==================== Run Original Kernel ====================
    printf("\n=== Running Original Kernel ===\n");

    // Warmup
    ms_deformable_im2col_cuda(
        stream, d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels, num_levels, num_query,
        num_points, d_output_orig);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iters = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
        ms_deformable_im2col_cuda(
            stream, d_value, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels, num_levels, num_query,
            num_points, d_output_orig);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float orig_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&orig_time_ms, start, stop));
    orig_time_ms /= num_iters;

    printf("Original kernel time: %.3f ms\n", orig_time_ms);

    // Copy result
    std::vector<__half> h_output_orig(output_elements);
    CUDA_CHECK(cudaMemcpy(h_output_orig.data(), d_output_orig, output_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    // ==================== Run TMA Kernel ====================
    printf("\n=== Running TMA Kernel ===\n");

    // Create TMA descriptors
    std::vector<CUtensorMap> h_tma_descriptors(batch * num_levels);
    std::vector<const void*> d_value_ptrs(batch);

    for (int b = 0; b < batch; b++) {
        d_value_ptrs[b] = d_value + b * spatial_size * channels;
    }

    CUresult res = createTMADescriptorsForAllBatches(
        h_tma_descriptors.data(),
        d_value_ptrs.data(),
        h_spatial_shapes.data(),
        h_level_start_index.data(),
        batch,
        num_levels,
        channels
    );

    if (res != CUDA_SUCCESS) {
        printf("Failed to create TMA descriptors: %d\n", res);
        return 1;
    }

    // Copy descriptors to device
    CUtensorMap *d_tma_descriptors;
    CUDA_CHECK(cudaMalloc(&d_tma_descriptors, h_tma_descriptors.size() * sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tma_descriptors, h_tma_descriptors.data(),
                          h_tma_descriptors.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // Warmup
    ms_deformable_im2col_cuda_tma(
        stream, d_value, d_tma_descriptors, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels, num_levels, num_query,
        num_points, d_output_tma);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
        ms_deformable_im2col_cuda_tma(
            stream, d_value, d_tma_descriptors, d_spatial_shapes, d_level_start_index,
            d_sampling_loc, d_attn_weight,
            batch, spatial_size, num_heads, channels, num_levels, num_query,
            num_points, d_output_tma);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tma_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tma_time_ms, start, stop));
    tma_time_ms /= num_iters;

    printf("TMA kernel time: %.3f ms\n", tma_time_ms);

    // Copy result
    std::vector<__half> h_output_tma(output_elements);
    CUDA_CHECK(cudaMemcpy(h_output_tma.data(), d_output_tma, output_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    // ==================== Compare Results ====================
    printf("\n=== Accuracy Comparison ===\n");
    float max_diff = compare_outputs(h_output_orig, h_output_tma);

    // ==================== Performance Summary ====================
    printf("\n=== Performance Summary ===\n");
    printf("Original kernel: %.3f ms\n", orig_time_ms);
    printf("TMA kernel:      %.3f ms\n", tma_time_ms);
    printf("Speedup:         %.2fx\n", orig_time_ms / tma_time_ms);

    if (max_diff < 0.01f) {
        printf("\n✅ Accuracy test PASSED (max diff < 0.01)\n");
    } else {
        printf("\n❌ Accuracy test FAILED (max diff = %.6f)\n", max_diff);
    }

    // Cleanup
    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output_orig);
    cudaFree(d_output_tma);
    cudaFree(d_tma_descriptors);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
