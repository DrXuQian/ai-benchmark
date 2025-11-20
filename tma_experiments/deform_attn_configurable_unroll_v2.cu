#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <stdexcept>
#include <map>

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])

// Template with configurable unroll factor
// UNROLL_FACTOR: how many points to fully unroll (1, 2, 4, or 8)
template <typename scalar_t=__half, const int NUM_POINT= 8, const int NUM_LEVELS=4, const int CHANNELS = 32,
                                    const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
                                    const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3, const int UNROLL_FACTOR=1>
__global__ void ms_deformable_im2col_gpu_kernel_template(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_query,
    scalar_t *data_col) {
    CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index << NUM_OUTPUT_SHIFT;
    const int c_col = _temp & (CHANNELS -1 );
    _temp = (_temp >> CHANNELS_SHIFT);
    const int sampling_index = _temp;
    const int b_col = (float)_temp/(float)num_query;
    const __half kZERO = __int2half_rz(0);
    const __half kONE = __int2half_rz(1);
    int32_t const wStride = CHANNELS;

    scalar_t *data_col_ptr = data_col + (index << NUM_OUTPUT_SHIFT);
    int data_weight_ptr = sampling_index << (LEVEL_SHIFT + POINT_SHIFT);
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int data_value_ptr_init_offset = (b_col * spatial_size) << CHANNELS_SHIFT;

    scalar_t col[NUM_OUTPUT];
    #pragma unroll
    for (int idx = 0; idx < (NUM_OUTPUT >> 1); idx += 1) {
        reinterpret_cast<__half2*>(col)[idx] = half2(0.0f, 0.0f);
    }
    scalar_t *data_half = const_cast<scalar_t *>(data_sampling_loc);
    scalar_t *data_attn_weight_half = const_cast<scalar_t *>(data_attn_weight);

    const half2 zp5 = half2(0.5f, 0.5f);
    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      int32_t const hStride = (spatial_w + 2) << CHANNELS_SHIFT;
      const half2 spatail_hw = half2(spatial_w, spatial_h);

      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + (level_start_id << (CHANNELS_SHIFT)));

      // Load all data for NUM_POINT
      half2 loc_hw_vec[NUM_POINT];
      half  weight_vec[NUM_POINT];

      #pragma unroll
      for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 4){
        LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
      }
      #pragma unroll
      for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8){
        LDST128BITS(weight_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
      }
      data_loc_w_ptr += (NUM_POINT << 1);
      data_weight_ptr += NUM_POINT;

      // Process points with different unroll factors
      if (UNROLL_FACTOR == 8) {
        // Fully unroll all 8 points
        #pragma unroll
        for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
          ProcessPoint(p_col, loc_hw_vec, weight_vec, spatail_hw, zp5, spatial_h, spatial_w,
                      hStride, wStride, c_col, kONE, kZERO, data_value_ptr, col);
        }
      } else if (UNROLL_FACTOR == 4) {
        // Unroll by groups of 4
        #pragma unroll
        for (int p_col = 0; p_col < 4; ++p_col) {
          ProcessPoint(p_col, loc_hw_vec, weight_vec, spatail_hw, zp5, spatial_h, spatial_w,
                      hStride, wStride, c_col, kONE, kZERO, data_value_ptr, col);
        }
        #pragma unroll
        for (int p_col = 4; p_col < 8; ++p_col) {
          ProcessPoint(p_col, loc_hw_vec, weight_vec, spatail_hw, zp5, spatial_h, spatial_w,
                      hStride, wStride, c_col, kONE, kZERO, data_value_ptr, col);
        }
      } else if (UNROLL_FACTOR == 2) {
        // Unroll by groups of 2
        #pragma unroll
        for (int group = 0; group < NUM_POINT; group += 2) {
          #pragma unroll
          for (int p = 0; p < 2; ++p) {
            int p_col = group + p;
            ProcessPoint(p_col, loc_hw_vec, weight_vec, spatail_hw, zp5, spatial_h, spatial_w,
                        hStride, wStride, c_col, kONE, kZERO, data_value_ptr, col);
          }
        }
      } else {
        // No unrolling
        for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
          ProcessPoint(p_col, loc_hw_vec, weight_vec, spatail_hw, zp5, spatial_h, spatial_w,
                      hStride, wStride, c_col, kONE, kZERO, data_value_ptr, col);
        }
      }
    }

    #pragma unroll
    for (int idx = 0; idx < NUM_OUTPUT; idx += 8){
      __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
      data_col_ptr += 8;
    }
  }

  // Helper function to process a single point (inlined)
  __device__ __forceinline__ static void ProcessPoint(
      int p_col, const half2* loc_hw_vec, const half* weight_vec,
      half2 spatail_hw, half2 zp5, int spatial_h, int spatial_w,
      int32_t hStride, int32_t wStride, int c_col, __half kONE, __half kZERO,
      const scalar_t* data_value_ptr, scalar_t* col) {

    const half2 loc = loc_hw_vec[p_col];
    const scalar_t weight = weight_vec[p_col];
    half2 weighthalf2 = half2(weight, weight);
    half2 hw_im = __hfma2(loc, spatail_hw, zp5);
    scalar_t h_im = __high2half(hw_im);
    scalar_t w_im = __low2half(hw_im);

    if (h_im > (scalar_t)(0) && w_im > (scalar_t)(0) &&
        h_im < (scalar_t)(spatial_h + 1) && w_im < (scalar_t)(spatial_w + 1)) {
      int32_t const hLow = __half2int_rd(h_im);
      int32_t const wLow = __half2int_rd(w_im);
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

      // Corner 1: top-left
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

      // Corner 2: top-right
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

      // Corner 3: bottom-left
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

      // Corner 4: bottom-right
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
};

template <typename scalar_t=__half, const int THREADS_IN_ONE_BLOCK=512, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, const int unroll_factor,
                               scalar_t *data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  const int num_threads = THREADS_IN_ONE_BLOCK;

  if (num_heads == 1 && num_point == 8 && num_levels == 4 && channels == 32) {
    // Launch kernel with specified unroll factor
    switch(unroll_factor) {
      case 8:
        ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 8>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query, data_col);
        break;
      case 4:
        ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 4>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query, data_col);
        break;
      case 2:
        ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 2>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query, data_col);
        break;
      default:
        ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT, 1>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query, data_col);
        break;
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

// Helper functions
template <typename T>
std::vector<T> read_bin_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<T> data(size / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

template <typename T>
void write_bin_file(const std::string& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq_pos = arg.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = arg.substr(0, eq_pos);
            std::string value = arg.substr(eq_pos + 1);
            args[key] = value;
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    // Default values
    int batch = args.count("batch") ? std::stoi(args["batch"]) : 48;
    int spatial_size = args.count("spatial_size") ? std::stoi(args["spatial_size"]) : 20522;
    int num_query = args.count("num_query") ? std::stoi(args["num_query"]) : 20522;
    int num_heads = args.count("num_heads") ? std::stoi(args["num_heads"]) : 1;
    int channels = args.count("channels") ? std::stoi(args["channels"]) : 32;
    int num_levels = args.count("num_levels") ? std::stoi(args["num_levels"]) : 4;
    int num_point = args.count("num_points") ? std::stoi(args["num_points"]) : 8;
    int im2col_step = args.count("im2col_step") ? std::stoi(args["im2col_step"]) : 1;
    int unroll_factor = args.count("unroll") ? std::stoi(args["unroll"]) : 1;
    std::string dir = args.count("dir") ? args["dir"] : ".";

    // Print configuration
    printf("==== Configurable Unroll Deformable Attention ====\n");
    printf("Configuration:\n");
    printf("  batch=%d, spatial_size=%d, num_query=%d\n", batch, spatial_size, num_query);
    printf("  num_heads=%d, channels=%d, num_levels=%d, num_point=%d\n",
           num_heads, channels, num_levels, num_point);
    printf("  im2col_step=%d, UNROLL_FACTOR=%d, dir=%s\n", im2col_step, unroll_factor, dir.c_str());
    printf("\n");

    // Validate unroll factor
    if (unroll_factor != 1 && unroll_factor != 2 && unroll_factor != 4 && unroll_factor != 8) {
        printf("Error: unroll_factor must be 1, 2, 4, or 8 (got %d)\n", unroll_factor);
        return 1;
    }

    try {
        // Read input data
        auto value_data = read_bin_file<__half>(dir + "/value_data.bin");
        auto spatial_shapes_data = read_bin_file<int64_t>(dir + "/spatial_shapes_data.bin");
        auto level_start_index_data = read_bin_file<int64_t>(dir + "/level_start_index_data.bin");
        auto sampling_loc_data = read_bin_file<__half>(dir + "/sampling_loc_data.bin");
        auto attn_weight_data = read_bin_file<__half>(dir + "/attn_weight_data.bin");

        size_t output_size = im2col_step * batch * num_query * num_heads * channels;
        std::vector<__half> output_data(output_size, __half(0.0f));

        // Allocate device memory
        __half *d_value, *d_sampling_loc, *d_attn_weight, *d_output;
        int64_t *d_spatial_shapes, *d_level_start_index;

        cudaMalloc(&d_value, value_data.size() * sizeof(__half));
        cudaMalloc(&d_spatial_shapes, spatial_shapes_data.size() * sizeof(int64_t));
        cudaMalloc(&d_level_start_index, level_start_index_data.size() * sizeof(int64_t));
        cudaMalloc(&d_sampling_loc, sampling_loc_data.size() * sizeof(__half));
        cudaMalloc(&d_attn_weight, attn_weight_data.size() * sizeof(__half));
        cudaMalloc(&d_output, output_size * sizeof(__half));

        // Copy to device
        cudaMemcpy(d_value, value_data.data(), value_data.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_spatial_shapes, spatial_shapes_data.data(),
                   spatial_shapes_data.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_level_start_index, level_start_index_data.data(),
                   level_start_index_data.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sampling_loc, sampling_loc_data.data(),
                   sampling_loc_data.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_attn_weight, attn_weight_data.data(),
                   attn_weight_data.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, output_size * sizeof(__half));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Warm up
        for (int i = 0; i < 3; ++i) {
            ms_deformable_im2col_cuda<__half>(
                stream, d_value, d_spatial_shapes, d_level_start_index,
                d_sampling_loc, d_attn_weight, batch / im2col_step, spatial_size,
                num_heads, channels, num_levels, num_query, num_point, unroll_factor, d_output
            );
        }
        cudaStreamSynchronize(stream);

        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int num_runs = 100;
        cudaEventRecord(start, stream);
        for (int i = 0; i < num_runs; ++i) {
            ms_deformable_im2col_cuda<__half>(
                stream, d_value, d_spatial_shapes, d_level_start_index,
                d_sampling_loc, d_attn_weight, batch / im2col_step, spatial_size,
                num_heads, channels, num_levels, num_query, num_point, unroll_factor, d_output
            );
        }
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Average kernel execution time (unroll=%d): %.4f ms\n", unroll_factor, milliseconds / num_runs);

        // Copy output back
        cudaMemcpy(output_data.data(), d_output, output_size * sizeof(__half), cudaMemcpyDeviceToHost);

        // Write output
        std::string output_filename = dir + "/output_cuda_unroll_" + std::to_string(unroll_factor) + ".bin";
        write_bin_file(output_filename, output_data);
        printf("Output written to %s\n", output_filename.c_str());

        // Print first few values
        printf("First 10 output values (as float):\n  [ ");
        for (int i = 0; i < std::min(10, (int)output_size); ++i) {
            printf("%.2f", (float)output_data[i]);
            if (i < std::min(10, (int)output_size) - 1) printf(", ");
        }
        printf(" ]\n");

        // Cleanup
        cudaFree(d_value);
        cudaFree(d_spatial_shapes);
        cudaFree(d_level_start_index);
        cudaFree(d_sampling_loc);
        cudaFree(d_attn_weight);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }

    return 0;
}