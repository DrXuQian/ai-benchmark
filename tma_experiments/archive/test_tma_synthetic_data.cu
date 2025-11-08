#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

// Copy necessary macros and utilities
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// ==================== Original Kernel ====================
template <typename scalar_t=__half, const int NUM_POINT=8, const int NUM_LEVELS=4, const int CHANNELS=32,
          const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
          const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3>
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

          int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[0],  wdata[0]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr1 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }

          HALF2(wdataexp[0]) = __hmul2(half2(wdata[1],  wdata[1]), HALF2(weighthalf2));
          int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + c_col;
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr2 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }

          int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[2],  wdata[2]), HALF2(weighthalf2));
          #pragma unroll
          for (int j = 0; j < NUM_OUTPUT; j += 8){
            LDST128BITS(vdata2d[j]) = LDST128BITS(const_cast<__half*>(data_value_ptr)[ptr3 + j]);
            #pragma unroll
            for (int p = 0; p < 8; p += 2){
              HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d[p]), HALF2(col[p]));
            }
          }

          int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + c_col;
          HALF2(wdataexp[0]) = __hmul2(half2(wdata[3],  wdata[3]), HALF2(weighthalf2));
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
    #pragma unroll
    for (int idx = 0; idx < NUM_OUTPUT; idx += 8){
      __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
      data_col_ptr += 8;
    }
  }
}

template <typename scalar_t=__half>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels / 8;
  const int num_threads = 512;
  if (num_heads == 1 && num_point == 8 && num_levels == 4 && channels == 32){
        ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, 8, 3>
          <<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query, data_col);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

// ==================== TMA Kernel (external) ====================
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

// ==================== Utility Functions ====================
void generate_pseudo_data(
    std::vector<__half>& value,
    std::vector<int64_t>& spatial_shapes,
    std::vector<int64_t>& level_start_index,
    std::vector<__half>& sampling_loc,
    std::vector<__half>& attn_weight,
    int batch, int spatial_size, int num_query, int num_heads, int channels,
    int num_levels, int num_points) {

    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Value: [batch][spatial_size][channels]
    value.resize(batch * spatial_size * channels);
    for (size_t i = 0; i < value.size(); i++) {
        value[i] = __float2half(dis(gen) * 2.0f - 1.0f);  // [-1, 1]
    }

    // Spatial shapes: [[92,160], [46,80], [23,40], [12,20]]
    spatial_shapes = {92, 160, 46, 80, 23, 40, 12, 20};

    // Level start index: [0, 15228, 19164, 20214]
    // Verify: 92*160=14720, 46*80=3680, 23*40=920, 12*20=240
    // But given values are [0, 15228, 19164, 20214]
    // Let's use H*W+pad layout: 92*160 + pad = 15228
    level_start_index = {0, 15228, 19164, 20214};

    // Sampling locations: [batch][num_query][num_heads][num_levels][num_points][2]
    // Shape: [48][123376][1][4][8][2]
    sampling_loc.resize(batch * num_query * num_heads * num_levels * num_points * 2);
    for (size_t i = 0; i < sampling_loc.size(); i++) {
        sampling_loc[i] = __float2half(dis(gen));  // [0, 1] normalized coordinates
    }

    // Attention weights: [batch][num_query][num_heads][num_levels][num_points]
    // Shape: [48][123376][1][4][8]
    attn_weight.resize(batch * num_query * num_heads * num_levels * num_points);
    for (size_t i = 0; i < attn_weight.size(); i++) {
        attn_weight[i] = __float2half(dis(gen) * 0.5f);  // [0, 0.5] for weights
    }

    printf("Generated pseudo data:\n");
    printf("  value: %zu elements\n", value.size());
    printf("  spatial_shapes: %zu elements\n", spatial_shapes.size());
    printf("  level_start_index: %zu elements\n", level_start_index.size());
    printf("  sampling_loc: %zu elements\n", sampling_loc.size());
    printf("  attn_weight: %zu elements\n\n", attn_weight.size());
}

float compare_outputs(const std::vector<__half>& a, const std::vector<__half>& b, int max_print = 10) {
    if (a.size() != b.size()) {
        printf("Size mismatch: %zu vs %zu\n", a.size(), b.size());
        return -1.0f;
    }

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_mismatches = 0;
    const float threshold = 0.01f;  // 1% relative error

    for (size_t i = 0; i < a.size(); i++) {
        float val_a = __half2float(a[i]);
        float val_b = __half2float(b[i]);
        float diff = std::abs(val_a - val_b);
        float relative_diff = diff / (std::abs(val_a) + 1e-6f);

        max_diff = std::max(max_diff, diff);
        avg_diff += diff;

        if (relative_diff > threshold) {
            if (num_mismatches < max_print) {
                printf("  Mismatch at %zu: original=%.6f, tma=%.6f, diff=%.6f (%.2f%%)\n",
                       i, val_a, val_b, diff, relative_diff * 100.0f);
            }
            num_mismatches++;
        }
    }

    avg_diff /= a.size();
    printf("\nComparison results:\n");
    printf("  Max diff: %.6f\n", max_diff);
    printf("  Avg diff: %.6f\n", avg_diff);
    printf("  Mismatches (>%.0f%%): %d / %zu (%.2f%%)\n",
           threshold * 100.0f, num_mismatches, a.size(),
           100.0f * num_mismatches / a.size());

    return max_diff;
}

// ==================== Main ====================
int main() {
    printf("=== Testing Original vs TMA Deformable Attention (Synthetic Data) ===\n\n");

    // Configuration matching your specification
    const int batch = 48;
    const int spatial_size = 20522;
    const int num_query = 123376;
    const int num_heads = 1;
    const int channels = 32;
    const int num_levels = 4;
    const int num_points = 8;

    printf("Configuration:\n");
    printf("  batch=%d, spatial_size=%d, num_query=%d\n", batch, spatial_size, num_query);
    printf("  num_heads=%d, channels=%d, num_levels=%d, num_points=%d\n\n",
           num_heads, channels, num_levels, num_points);

    // Generate pseudo data
    std::vector<__half> h_value, h_sampling_loc, h_attn_weight;
    std::vector<int64_t> h_spatial_shapes, h_level_start_index;

    printf("Generating pseudo data...\n");
    generate_pseudo_data(h_value, h_spatial_shapes, h_level_start_index,
                        h_sampling_loc, h_attn_weight,
                        batch, spatial_size, num_query, num_heads, channels,
                        num_levels, num_points);

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

    ms_deformable_im2col_cuda(
        stream, d_value, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels, num_levels, num_query,
        num_points, d_output_orig);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iters = 100;
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

    printf("Original kernel avg time: %.3f ms\n", orig_time_ms);

    // Copy result
    std::vector<__half> h_output_orig(output_elements);
    CUDA_CHECK(cudaMemcpy(h_output_orig.data(), d_output_orig, output_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    printf("First 10 output values: ");
    for (int i = 0; i < 10 && i < output_elements; i++) {
        printf("%.4f ", __half2float(h_output_orig[i]));
    }
    printf("\n");

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
    ms_deformable_im2col_cuda_tma<__half, 512, 8, 3>(
        stream, d_value, d_tma_descriptors, d_spatial_shapes, d_level_start_index,
        d_sampling_loc, d_attn_weight,
        batch, spatial_size, num_heads, channels, num_levels, num_query,
        num_points, d_output_tma);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
        ms_deformable_im2col_cuda_tma<__half, 512, 8, 3>(
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

    printf("TMA kernel avg time: %.3f ms\n", tma_time_ms);

    // Copy result
    std::vector<__half> h_output_tma(output_elements);
    CUDA_CHECK(cudaMemcpy(h_output_tma.data(), d_output_tma, output_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    printf("First 10 output values: ");
    for (int i = 0; i < 10 && i < output_elements; i++) {
        printf("%.4f ", __half2float(h_output_tma[i]));
    }
    printf("\n");

    // ==================== Compare Results ====================
    printf("\n=== Accuracy Comparison ===\n");
    float max_diff = compare_outputs(h_output_orig, h_output_tma, 20);

    // ==================== Performance Summary ====================
    printf("\n=== Performance Summary ===\n");
    printf("Original kernel: %.3f ms\n", orig_time_ms);
    printf("TMA kernel:      %.3f ms\n", tma_time_ms);
    printf("Speedup:         %.2fx\n", orig_time_ms / tma_time_ms);
    printf("Throughput (original): %.2f GB/s\n",
           (h_value.size() + h_sampling_loc.size() + output_elements) * sizeof(__half) / (orig_time_ms / 1000.0f) / 1e9);
    printf("Throughput (TMA):      %.2f GB/s\n",
           (h_value.size() + h_sampling_loc.size() + output_elements) * sizeof(__half) / (tma_time_ms / 1000.0f) / 1e9);

    if (max_diff < 0.1f) {
        printf("\n✅ Accuracy test PASSED (max diff < 0.1)\n");
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
