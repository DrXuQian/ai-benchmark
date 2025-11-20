#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <stdint.h>

#define HALF2(value) *(reinterpret_cast<half2*>(&(value)))
#define LDST128BITS(value) *(reinterpret_cast<float4*>(&(value)))

#define GET_BLOCKS(num_kernels, num_threads) ((num_kernels + num_threads - 1) / num_threads)

template <typename scalar_t, const int NUM_POINT = 8, const int NUM_LEVEL = 4, const int NUM_CHANNELS = 32,
          const int CHANNELS_SHIFT = 5, const int SPATIAL_W_SHIFT = 7, const int SPATIAL_SHIFT = 14,
          const int NUM_OUTPUT = 8, const int OUTPUT_SHIFT = 3, const int UNROLL_FACTOR = 4>
__global__ void ms_deformable_im2col_gpu_kernel_template(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_query, scalar_t *data_col) {
  const int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outputIdx >= n) {
    return;
  }

  // Use static assertions to ensure compatibility
  static_assert(NUM_POINT % UNROLL_FACTOR == 0, "NUM_POINT must be divisible by UNROLL_FACTOR");

  int _temp = outputIdx;
  const int c_col = _temp - (_temp >> OUTPUT_SHIFT << OUTPUT_SHIFT); _temp >>= OUTPUT_SHIFT;
  const int h_col = _temp - (_temp >> OUTPUT_SHIFT << OUTPUT_SHIFT); _temp >>= OUTPUT_SHIFT;
  const int w_col = _temp - (_temp >> NUM_CHANNELS << NUM_CHANNELS); _temp >>= NUM_CHANNELS;
  const int b_col = _temp;

  const scalar_t kONE = 1.0f;
  const half2 zp5 = half2(0.5f, 0.5f);
  const int hStride = NUM_CHANNELS << SPATIAL_W_SHIFT;
  const int wStride = NUM_CHANNELS;

  scalar_t *data_col_ptr = data_col + outputIdx;
  int data_weight_ptr = ((b_col * num_query + w_col) * 1) * NUM_LEVEL * NUM_POINT + h_col * NUM_POINT;
  int data_loc_w_ptr = ((b_col * num_query + w_col) * 1) * NUM_LEVEL * NUM_POINT * 2 + h_col * NUM_POINT * 2;
  const int data_value_ptr_init_offset = b_col * spatial_size * 1 * NUM_CHANNELS;

  __half col[NUM_OUTPUT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  __half *data_attn_weight_half = const_cast<__half*>(data_attn_weight);
  __half *data_loc_half = const_cast<__half*>(data_sampling_loc);

  const int qid_stride = w_col * 1 * NUM_CHANNELS + c_col;

  for (int l_col = 0; l_col < NUM_LEVEL; ++l_col) {
    const int level_start_id = data_level_start_index[l_col];
    const int spatial_h = data_spatial_shapes[l_col << 1];
    const int spatial_w = data_spatial_shapes[l_col << 1 | 1];
    half2 spatail_hw = half2(__int2half_rd(spatial_w) , __int2half_rd(spatial_h));

    const __half *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * 1 * NUM_CHANNELS);

    half2 loc_hw_vec[NUM_POINT];
    scalar_t weight_vec[NUM_POINT];
    #pragma unroll
    for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8){
      LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_loc_half[data_loc_w_ptr + (pack_id << 1)]));
    }
    #pragma unroll
    for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8){
      LDST128BITS(weight_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
    }
    data_loc_w_ptr += (NUM_POINT << 1);
    data_weight_ptr += NUM_POINT;

    if constexpr (UNROLL_FACTOR == 4) {
      // Process in batches of 4
      for (int batch = 0; batch < NUM_POINT; batch += 4) {
        __half vdata2d[4][NUM_OUTPUT];
        half2 weighthalf2[4];
        __half wdata[4][4];
        int32_t ptrs[4][4];

        // Compute all metadata for 4 points and determine validity
        bool any_valid = false;
        bool valid[4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int p = batch + i;
          const half2 loc = loc_hw_vec[p];
          const scalar_t weight = weight_vec[p];
          weighthalf2[i] = half2(weight, weight);
          half2 hw_im = __hfma2(loc, spatail_hw, zp5);
          scalar_t h_im = __high2half(hw_im);
          scalar_t w_im = __low2half(hw_im);

          valid[i] = (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) &&
                      h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1));
          any_valid |= valid[i];

          if (valid[i]) {
            int32_t const hLow = __half2int_rd(h_im);
            int32_t const wLow = __half2int_rd(w_im);
            const __half lh = __hsub(h_im, __int2half_rd(hLow));
            const __half lw = __hsub(w_im, __int2half_rd(wLow));
            const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);

            int32_t const hLowPtrOffset = hLow * hStride;
            int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
            int32_t const wLowPtrOffset = wLow << CHANNELS_SHIFT;
            int32_t const wHighPtrOffset = wLowPtrOffset + wStride;

            ptrs[i][0] = (hLowPtrOffset + wLowPtrOffset + c_col) / NUM_OUTPUT;
            ptrs[i][1] = (hLowPtrOffset + wHighPtrOffset + c_col) / NUM_OUTPUT;
            ptrs[i][2] = (hHighPtrOffset + wLowPtrOffset + c_col) / NUM_OUTPUT;
            ptrs[i][3] = (hHighPtrOffset + wHighPtrOffset + c_col) / NUM_OUTPUT;

            wdata[i][0] = __hmul(hh, hw);
            wdata[i][1] = __hmul(hh, lw);
            wdata[i][2] = __hmul(lh, hw);
            wdata[i][3] = __hmul(lh, lw);
          }
        }

        // Skip entire batch if no valid points
        if (!any_valid) continue;

        // Process each corner - now we only check valid once per point per corner
        #pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
          // Interleave load and compute for better instruction scheduling
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (valid[i]) {
              // Load data
              LDST128BITS(vdata2d[i][0]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptrs[i][corner]]);

              // Compute immediately after load for this point
              half2 wdataexp = __hmul2(half2(wdata[i][corner], wdata[i][corner]), weighthalf2[i]);
              #pragma unroll
              for (int j = 0; j < NUM_OUTPUT; j += 2) {
                HALF2(col[j]) = __hfma2(wdataexp, HALF2(vdata2d[i][j]), HALF2(col[j]));
              }
            }
          }
        }
      }

    } else if (UNROLL_FACTOR == 2) {
      // Process in batches of 2
      for (int batch = 0; batch < NUM_POINT; batch += 2) {
        __half vdata2d[2][NUM_OUTPUT];
        half2 weighthalf2[2];
        __half wdata[2][4];
        int32_t ptrs[2][4];

        bool any_valid = false;
        bool valid[2];

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int p = batch + i;
          const half2 loc = loc_hw_vec[p];
          const scalar_t weight = weight_vec[p];
          weighthalf2[i] = half2(weight, weight);
          half2 hw_im = __hfma2(loc, spatail_hw, zp5);
          scalar_t h_im = __high2half(hw_im);
          scalar_t w_im = __low2half(hw_im);

          valid[i] = (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) &&
                      h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1));
          any_valid |= valid[i];

          if (valid[i]) {
            int32_t const hLow = __half2int_rd(h_im);
            int32_t const wLow = __half2int_rd(w_im);
            const __half lh = __hsub(h_im, __int2half_rd(hLow));
            const __half lw = __hsub(w_im, __int2half_rd(wLow));
            const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);

            int32_t const hLowPtrOffset = hLow * hStride;
            int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
            int32_t const wLowPtrOffset = wLow << CHANNELS_SHIFT;
            int32_t const wHighPtrOffset = wLowPtrOffset + wStride;

            ptrs[i][0] = (hLowPtrOffset + wLowPtrOffset + c_col) / NUM_OUTPUT;
            ptrs[i][1] = (hLowPtrOffset + wHighPtrOffset + c_col) / NUM_OUTPUT;
            ptrs[i][2] = (hHighPtrOffset + wLowPtrOffset + c_col) / NUM_OUTPUT;
            ptrs[i][3] = (hHighPtrOffset + wHighPtrOffset + c_col) / NUM_OUTPUT;

            wdata[i][0] = __hmul(hh, hw);
            wdata[i][1] = __hmul(hh, lw);
            wdata[i][2] = __hmul(lh, hw);
            wdata[i][3] = __hmul(lh, lw);
          }
        }

        if (!any_valid) continue;

        #pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
          #pragma unroll
          for (int i = 0; i < 2; ++i) {
            if (valid[i]) {
              LDST128BITS(vdata2d[i][0]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptrs[i][corner]]);
              half2 wdataexp = __hmul2(half2(wdata[i][corner], wdata[i][corner]), weighthalf2[i]);
              #pragma unroll
              for (int j = 0; j < NUM_OUTPUT; j += 2) {
                HALF2(col[j]) = __hfma2(wdataexp, HALF2(vdata2d[i][j]), HALF2(col[j]));
              }
            }
          }
        }
      }

    } else {
      // Fallback: Process points one by one (original approach)
      #pragma unroll
      for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
        const half2 loc = loc_hw_vec[p_col];
        const scalar_t weight = weight_vec[p_col];
        half2 weighthalf2 = half2(weight, weight);
        half2 hw_im = __hfma2(loc, spatail_hw, zp5);
        scalar_t h_im = __high2half(hw_im);
        scalar_t w_im = __low2half(hw_im);

        if (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) && h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1)) {
          int32_t const hLow = __half2int_rd(h_im);
          int32_t const wLow = __half2int_rd(w_im);
          int32_t const hHigh = hLow + 1;
          int32_t const wHigh = wLow + 1;
          const __half lh = __hsub(h_im, __int2half_rd(hLow));
          const __half lw = __hsub(w_im, __int2half_rd(wLow));
          const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);
          int32_t const hLowPtrOffset = hLow * hStride;
          int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
          int32_t const wLowPtrOffset = wLow << CHANNELS_SHIFT;
          int32_t const wHighPtrOffset = wLowPtrOffset + wStride;

          __half pst_lh[4] = {hh, hh, lh, lh};
          __half pst_rh[4] = {hw, lw, hw, lw};
          __half wdata[4];
          HALF2(wdata[0]) = __hmul2(HALF2(pst_lh[0]), HALF2(pst_rh[0]));
          HALF2(wdata[2]) = __hmul2(HALF2(pst_lh[2]), HALF2(pst_rh[2]));

          __half wdataexp[2];
          __half vdata2d[NUM_OUTPUT];

          // Corner 1
          int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[0], wdata[0]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr1 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }

          // Corner 2
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[1], wdata[1]), HALF2(weighthalf2));
          int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + c_col;
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr2 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }

          // Corner 3
          int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[2], wdata[2]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr3 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }

          // Corner 4
          int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[3], wdata[3]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr4 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }
        }
      }
    }
  }

  #pragma unroll
  for (int idx = 0; idx < NUM_OUTPUT; idx += 8){
    __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
    data_col_ptr += 8;
  }
}

template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col,
                               const int unroll_factor) {
  const int num_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  const int num_threads = THREADS_IN_ONE_BLOCK;

  if (num_heads == 1 && num_point == 8 && num_levels == 4 && channels == 32) {
    if (unroll_factor == 4) {
      ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 5, 7, 14, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_query, data_col);
    } else if (unroll_factor == 2) {
      ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 5, 7, 14, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_query, data_col);
    } else {
      ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 5, 7, 14, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_query, data_col);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

// Helper functions remain the same
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
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        size_t pos = arg.find('=');
        if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            args_map[key] = value;
        } else {
            args_map[arg] = "";
        }
    }
    return args_map;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    int batch_size = get_param(args, "batch_size", 1);
    int num_query = get_param(args, "num_query", 40000);
    int num_heads = get_param(args, "num_heads", 1);
    int num_levels = get_param(args, "num_levels", 4);
    int num_points = get_param(args, "num_points", 8);
    int num_channels = get_param(args, "num_channels", 32);
    int num_iterations = get_param(args, "num_iterations", 1000);
    int unroll_factor = get_param(args, "unroll", 4);

    std::string value_path = get_param<std::string>(args, "value_path",
        "/home/qianxu/ai-benchmark/cuda_deformable_attention_data/value.bin");
    std::string shape_path = get_param<std::string>(args, "shape_path",
        "/home/qianxu/ai-benchmark/cuda_deformable_attention_data/spatial_shapes.bin");
    std::string level_path = get_param<std::string>(args, "level_path",
        "/home/qianxu/ai-benchmark/cuda_deformable_attention_data/level_start_index.bin");
    std::string loc_path = get_param<std::string>(args, "loc_path",
        "/home/qianxu/ai-benchmark/cuda_deformable_attention_data/sampling_locations.bin");
    std::string weight_path = get_param<std::string>(args, "weight_path",
        "/home/qianxu/ai-benchmark/cuda_deformable_attention_data/attention_weights.bin");

    // Read data
    auto value = read_bin_file<__half>(value_path);
    auto spatial_shapes = read_bin_file<int64_t>(shape_path);
    auto level_start_index = read_bin_file<int64_t>(level_path);
    auto sampling_locations = read_bin_file<__half>(loc_path);
    auto attention_weights = read_bin_file<__half>(weight_path);

    int64_t spatial_size = 0;
    for(int i = 0; i < spatial_shapes.size(); i += 2) {
        spatial_size += spatial_shapes[i] * spatial_shapes[i + 1];
    }

    size_t output_size = batch_size * num_query * num_heads * num_channels;
    std::vector<__half> output(output_size);

    // Allocate GPU memory
    __half *d_value, *d_sampling_locations, *d_attention_weights, *d_output;
    int64_t *d_spatial_shapes, *d_level_start_index;

    CUDA_CHECK(cudaMalloc(&d_value, value.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, level_start_index.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sampling_locations, sampling_locations.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attention_weights, attention_weights.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(__half)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_value, value.data(), value.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, spatial_shapes.data(), spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, level_start_index.data(), level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_locations, sampling_locations.data(), sampling_locations.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attention_weights, attention_weights.data(), attention_weights.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 100; ++i) {
        ms_deformable_im2col_cuda<__half>(0, d_value, d_spatial_shapes, d_level_start_index,
                                          d_sampling_locations, d_attention_weights,
                                          batch_size, spatial_size, num_heads, num_channels,
                                          num_levels, num_query, num_points, d_output,
                                          unroll_factor);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < num_iterations; ++i) {
        ms_deformable_im2col_cuda<__half>(0, d_value, d_spatial_shapes, d_level_start_index,
                                          d_sampling_locations, d_attention_weights,
                                          batch_size, spatial_size, num_heads, num_channels,
                                          num_levels, num_query, num_points, d_output,
                                          unroll_factor);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_time_ms = elapsed_ms / num_iterations;

    std::cout << "Optimized Valid Checks - Unroll Factor " << unroll_factor << std::endl;
    std::cout << "Average kernel execution time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (output_size * sizeof(__half)) / (avg_time_ms * 1e6) << " GB/s" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_spatial_shapes));
    CUDA_CHECK(cudaFree(d_level_start_index));
    CUDA_CHECK(cudaFree(d_sampling_locations));
    CUDA_CHECK(cudaFree(d_attention_weights));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}