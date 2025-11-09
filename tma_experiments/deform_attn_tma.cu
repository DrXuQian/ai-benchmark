#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <stdexcept>
#include <map>

#define TILE_H 2
#define TILE_W 2
#define TILE_C 32
#define THREADS_IN_ONE_BLOCK 512
#define DEBUG false
using barrier = cuda::barrier<cuda::thread_scope_block>;

inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* func_ptr = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &func_ptr,
                                               cudaEnableDefault, &driver_status);
    return (err == cudaSuccess) ? reinterpret_cast<PFN_cuTensorMapEncodeTiled>(func_ptr) : nullptr;
}

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])

// Debug helper functions
__device__ inline void debug_print_tma_smem_comparison(
    int tid, int b_col, int l_col, int p_col, int blockIdx_x,
    const __half* vdata_global, const __half* vdata_tma,
    int ptr_idx, int h_pos, int w_pos, int c_col, int num_output = 8)
{
    if (tid == 0 && b_col == 0 && l_col == 0 && p_col == 0 && blockIdx_x == 0) {
        printf("\n[ptr%d: h=%d,w=%d] Global vs TMA:\n", ptr_idx, h_pos, w_pos);
        printf("  Global[%d-%d]: ", c_col, c_col + num_output - 1);
        for (int i = 0; i < num_output; i++)
            printf("%.2f ", __half2float(vdata_global[i]));
        printf("\n  TMA   [%d-%d]: ", c_col, c_col + num_output - 1);
        for (int i = 0; i < num_output; i++)
            printf("%.2f ", __half2float(vdata_tma[i]));

        bool match = true;
        for (int i = 0; i < num_output; i++)
            if (vdata_global[i] != vdata_tma[i]) match = false;
        printf("  %s\n", match ? "✓" : "✗ MISMATCH");
    }
}

// Pre-compute all point coordinates and metadata before pipelining
template <typename scalar_t>
struct PointMeta {
    half2 loc;
    scalar_t weight;
    half2 weighthalf2;
    scalar_t h_im, w_im;
    int within_range;
    int32_t hLow, wLow, hHigh, wHigh;
};

// TMA load helper: issue async load to specific stage buffer
template<typename scalar_t, int STAGES, int CHANNELS, int NUMBERS_OF_WARPS, int QUERIES_PER_WARP>
__device__ __forceinline__ void issue_tma_load(
    int stage_idx,
    int warp_id,
    int query_id_in_warp,
    int is_loader_thread,
    int within_range,
    int hLow,
    int wLow,
    const CUtensorMap* tma_desc,
    scalar_t smem_tile[STAGES][NUMBERS_OF_WARPS][QUERIES_PER_WARP][2][2][CHANNELS],
    cuda::barrier<cuda::thread_scope_block> warp_bars[STAGES][NUMBERS_OF_WARPS])
{
    if (is_loader_thread and within_range) {
        // TMA coordinates: X=C, Y=W, Z=H
        int32_t tensor_coord_c = 0;
        int32_t tensor_coord_w = wLow;
        int32_t tensor_coord_h = hLow;

        // Issue TMA load to the specified stage
        asm volatile(
            "{\n\t"
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
            "}"
            :
            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[stage_idx][warp_id][query_id_in_warp][0][0][0]))),
              "l"(reinterpret_cast<uint64_t>(tma_desc)),
              "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
              "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[stage_idx][warp_id])))
            : "memory"
        );

        asm volatile(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
            :
            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[stage_idx][warp_id]))),
              "n"(2 * 2 * CHANNELS * sizeof(scalar_t))
        );
    } else {
        // Out of bounds - expect 0 bytes
        asm volatile(
            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
            :
            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&warp_bars[stage_idx][warp_id]))),
              "n"(0)
        );
    }
}

// Wait for TMA load to complete for specific stage
template<int STAGES, int NUMBERS_OF_WARPS>
__device__ __forceinline__ void wait_tma_load(
    int stage_idx,
    int warp_id,
    cuda::barrier<cuda::thread_scope_block> warp_bars[STAGES][NUMBERS_OF_WARPS])
{
    __syncwarp();
    cuda::barrier<cuda::thread_scope_block>::arrival_token token = warp_bars[stage_idx][warp_id].arrive();
    warp_bars[stage_idx][warp_id].wait(std::move(token));
}

template <typename scalar_t=__half, const int NUM_POINT= 8, const int NUM_LEVELS=4, const int CHANNELS = 32, 
                                    const int POINT_SHIFT=3, const int LEVEL_SHIFT=2, const int CHANNELS_SHIFT=5,
                                    const int NUM_OUTPUT=8, const int NUM_OUTPUT_SHIFT=3, const int STAGES=1>
__global__ void ms_deformable_im2col_gpu_kernel_template(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_query,
    const CUtensorMap* d_tma_descs, scalar_t *data_col) {
    CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index << NUM_OUTPUT_SHIFT;
    const int c_col = _temp & (CHANNELS -1 ); //_temp % CHANNELS;
    _temp = (_temp >> CHANNELS_SHIFT);
    const int sampling_index = _temp;
    const int b_col = (float)_temp/(float)num_query;
    const __half kZERO = __int2half_rz(0);
    const __half kONE = __int2half_rz(1);

    scalar_t *data_col_ptr = data_col + (index << NUM_OUTPUT_SHIFT);
    int data_weight_ptr = sampling_index << (LEVEL_SHIFT + POINT_SHIFT); // * NUM_LEVELS * NUM_POINT;
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

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int query_id_in_warp = lane_id >> 2;
    const int is_loader_thread = lane_id % 4 == 0;

    constexpr int number_of_warps = THREADS_IN_ONE_BLOCK / 32;
    constexpr int queries_per_warp = 32 / (CHANNELS / NUM_OUTPUT);
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ alignas(128) scalar_t smem_tile[STAGES][number_of_warps][queries_per_warp][2][2][CHANNELS];  // max 512 threads = 16 warps, 8 queries per warp
    __shared__ barrier warp_bars[STAGES][number_of_warps];
    if (lane_id == 0) {
        for (int s = 0; s < STAGES; s ++){
            init(&warp_bars[s][warp_id], 32);
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }

    __syncwarp();
    // Prefetch TMA descriptors for this batch's 4 levels into L2 cache
    // This reduces descriptor access latency during TMA operations
    // Only one thread (warp 0, lane 0) needs to prefetch all 4 descriptors
    if (lane_id == 0 && warp_id == 0) {
        #pragma unroll
        for (int l = 0; l < NUM_LEVELS; l++) {
            const int desc_idx = b_col * NUM_LEVELS + l;
            const CUtensorMap* desc_ptr = &d_tma_descs[desc_idx];
            // prefetch.tensormap brings descriptor to L2 cache
            asm volatile(
                "prefetch.tensormap [%0];\n\t"
                :: "l"(reinterpret_cast<uint64_t>(desc_ptr))
            );
        }
    }


    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      // h -> hight , w -> low 
      const half2 spatail_hw = half2(spatial_w, spatial_h);

      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + (level_start_id << (CHANNELS_SHIFT)));
      // load data_sampling_loc and  data_attn_weight for NUM_POINT
      // NUM_POINT 4;
      half2 loc_hw_vec[NUM_POINT]; // 8 FP16 = 128 bit  
      half  weight_vec[NUM_POINT]; // 4 FP16 = 64 bit 

      #pragma unroll
      for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 4){
        LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
      }
      #pragma unroll
      for (int pack_id = 0; pack_id < NUM_POINT; pack_id += 8){
        // FLOAT2(weight_vec[pack_id])      = FLOAT2(data_attn_weight_half[data_weight_ptr + pack_id]) ;
        LDST128BITS(weight_vec[pack_id])      = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
      }
      data_loc_w_ptr += (NUM_POINT << 1);
      data_weight_ptr += NUM_POINT;

      PointMeta<scalar_t> point_meta[STAGES];

      // initialization of point_meta for first STAGES-1 points
      #pragma unroll
      for (int p = 0; p < STAGES - 1; ++p) {
          auto& cur_point_meta = point_meta[p];
          cur_point_meta.loc = loc_hw_vec[p];
          cur_point_meta.weight = weight_vec[p];
          cur_point_meta.weighthalf2 = half2(weight_vec[p], weight_vec[p]);
          half2 hw_im = __hfma2(cur_point_meta.loc, spatail_hw, zp5);
          cur_point_meta.h_im = __high2half(hw_im);
          cur_point_meta.w_im = __low2half(hw_im);
          cur_point_meta.within_range = (cur_point_meta.h_im > (scalar_t)(0) &&
                                        cur_point_meta.w_im > (scalar_t)(0) &&
                                        cur_point_meta.h_im < (scalar_t)(spatial_h + 1) &&
                                        cur_point_meta.w_im < (scalar_t)(spatial_w + 1));
          cur_point_meta.hLow = __half2int_rd(cur_point_meta.h_im);
          cur_point_meta.wLow = __half2int_rd(cur_point_meta.w_im);
          issue_tma_load<scalar_t, STAGES, CHANNELS, number_of_warps, queries_per_warp>(
              p, warp_id, query_id_in_warp,
              is_loader_thread, cur_point_meta.within_range,
              cur_point_meta.hLow, cur_point_meta.wLow,
              &d_tma_descs[b_col * NUM_LEVELS + l_col],
              smem_tile, warp_bars);
      }

      #pragma unroll
      for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
        int cur_stage_id = p_col % STAGES;
        auto& cur_point_meta = point_meta[cur_stage_id];
        if (p_col + STAGES - 1 < NUM_POINT) {
            int next_p = p_col + STAGES - 1;
            int next_stage_id = next_p % STAGES;
            auto& next_point_meta = point_meta[next_stage_id];
            next_point_meta.loc = loc_hw_vec[next_p];
            next_point_meta.weight = weight_vec[next_p];
            next_point_meta.weighthalf2 = half2(weight_vec[next_p], weight_vec[next_p]);
            half2 hw_im = __hfma2(next_point_meta.loc, spatail_hw, zp5);
            next_point_meta.h_im = __high2half(hw_im);
            next_point_meta.w_im = __low2half(hw_im);
            next_point_meta.within_range = (next_point_meta.h_im > (scalar_t)(0) &&
                                            next_point_meta.w_im > (scalar_t)(0) &&
                                            next_point_meta.h_im < (scalar_t)(spatial_h + 1) &&
                                            next_point_meta.w_im < (scalar_t)(spatial_w + 1));
            next_point_meta.hLow = __half2int_rd(next_point_meta.h_im);
            next_point_meta.wLow = __half2int_rd(next_point_meta.w_im);
            issue_tma_load<scalar_t, STAGES, CHANNELS, number_of_warps, queries_per_warp>(
                next_stage_id, warp_id, query_id_in_warp,
                is_loader_thread, next_point_meta.within_range,
                next_point_meta.hLow, next_point_meta.wLow,
                &d_tma_descs[b_col * NUM_LEVELS + l_col],
                smem_tile, warp_bars);
        }
        wait_tma_load<STAGES, number_of_warps>(cur_stage_id, warp_id, warp_bars);

        if (cur_point_meta.within_range) {
            int hLow = cur_point_meta.hLow;
            int wLow = cur_point_meta.wLow;
            const scalar_t h_im = cur_point_meta.h_im;
            const scalar_t w_im = cur_point_meta.w_im;
            half2 weighthalf2 = cur_point_meta.weighthalf2;
            const __half lh = __hsub(h_im, __int2half_rd(hLow));
            const __half lw = __hsub(w_im, __int2half_rd(wLow));
            const __half hh = __hsub(kONE, lh), hw = __hsub(kONE, lw);
            __half pst_lh[4] = {hh, hh, lh, lh};
            __half pst_rh[4] = {hw, lw, hw, lw};
            __half wdata[4] ;
            HALF2(wdata[0]) = __hmul2(HALF2(pst_lh[0]), HALF2(pst_rh[0]));
            HALF2(wdata[2]) = __hmul2(HALF2(pst_lh[2]), HALF2(pst_rh[2]));
            // // expand wdata from [w0, w1, w2, w3] to  [w0, w0, w1, w1, w2, w2, ..., w3, w3]
            __half wdataexp[2];
            __half vdata2d_tma[NUM_OUTPUT];  // For TMA loaded data comparison
            HALF2(wdataexp[0]) = __hmul2(half2(wdata[0],  wdata[0]), HALF2(weighthalf2));
            #pragma unroll
            for (int j = 0; j < NUM_OUTPUT; j += 8){
                // Load from TMA shared memory for comparison
                LDST128BITS(vdata2d_tma[j]) = LDST128BITS(smem_tile[cur_stage_id][warp_id][query_id_in_warp][0][0][c_col + j]);
                #pragma unroll
                for (int p = 0; p < 8; p += 2){
                    HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d_tma[p]), HALF2(col[p]));
                }
            }
            HALF2(wdataexp[0]) = __hmul2(half2(wdata[1],  wdata[1]), HALF2(weighthalf2));
            #pragma unroll
            for (int j = 0; j < NUM_OUTPUT; j += 8){
                LDST128BITS(vdata2d_tma[j]) = LDST128BITS(smem_tile[cur_stage_id][warp_id][query_id_in_warp][0][1][c_col + j]);
                #pragma unroll
                for (int p = 0; p < 8; p += 2){
                    HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d_tma[p]), HALF2(col[p]));
                }
            }
            HALF2(wdataexp[0]) = __hmul2(half2(wdata[2],  wdata[2]), HALF2(weighthalf2));
            #pragma unroll
            for (int j = 0; j < NUM_OUTPUT; j += 8){
                LDST128BITS(vdata2d_tma[j]) = LDST128BITS(smem_tile[cur_stage_id][warp_id][query_id_in_warp][1][0][c_col + j]);
                #pragma unroll
                for (int p = 0; p < 8; p += 2){
                    HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d_tma[p]), HALF2(col[p]));
                }
            }
            HALF2(wdataexp[0]) = __hmul2(half2(wdata[3],  wdata[3]), HALF2(weighthalf2));
            #pragma unroll
            for (int j = 0; j < NUM_OUTPUT; j += 8){
                LDST128BITS(vdata2d_tma[j]) = LDST128BITS(smem_tile[cur_stage_id][warp_id][query_id_in_warp][1][1][c_col + j]);
                #pragma unroll
                for (int p = 0; p < 8; p += 2){
                    HALF2(col[p]) = __hfma2(HALF2(wdataexp[0]), HALF2(vdata2d_tma[p]), HALF2(col[p]));
                }
            }
        }
      }
    }
    #pragma unroll
    for (int idx = 0; idx < NUM_OUTPUT; idx += 8){
      // LDST128BITS(*data_col_ptr) = LDST128BITS(col[idx]);
      __stcg(reinterpret_cast<float4*>(data_col_ptr), *reinterpret_cast<float4*>(&col[idx]));
      data_col_ptr += 8;
    }
  }
}
template <typename scalar_t=__half, const int OUTPUTS_IN_THREAD=8, const int OUTPUTS_SHIFT=3>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, const CUtensorMap* d_tma_descs, scalar_t *data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
  // 8 warp, optimal threads for MIG 
  const int num_threads = THREADS_IN_ONE_BLOCK;
  if (num_heads == 1 and num_point == 8 and num_levels == 4 and channels == 32){
        ms_deformable_im2col_gpu_kernel_template<scalar_t, 8, 4, 32, 3, 2, 5, OUTPUTS_IN_THREAD, OUTPUTS_SHIFT>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
              num_kernels, data_value, data_spatial_shapes, data_level_start_index,
              data_sampling_loc, data_attn_weight, batch_size, spatial_size,
              num_query, d_tma_descs, data_col);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

// 辅助函数：从 .bin 文件读取数据到 std::vector
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
            // 1. 如果 T 是 std::string，直接返回值
            if constexpr (std::is_same_v<T, std::string>) {
                return args.at(key);
            } // 2. 如果 T 是整数类型
            else if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(std::stoll(args.at(key)));
            }
            // 3. 如果 T 是浮点数类型
            else if constexpr (std::is_floating_point_v<T>) {
                return static_cast<T>(std::stod(args.at(key))); }
            // 4. (可选) 如果有其他类型，可以在这里添加
            else {
            // 如果遇到不支持的类型，在编译时就报错
                static_assert(sizeof(T) == 0, "Unsupported type for get_param");
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse '" << key << "'. Using default value." << std::endl;
            // 注意：不打印 e.what() 和 default_value，因为类型可能不支持 << 操作符
            return default_value;
        }
    }
    return default_value;
}

// 函数检查 CUDA API 调用的返回值
#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// 解析命令行参数的辅助函数
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

int main(int argc, char* argv[]) {
    // === 1. 解析命令行参数 ===
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " [key=value] ..." << std::endl;
        std::cout << "Example: " << argv[0] << " batch=48 spatial_size=20522 dir=data/binary_400x800/cross_attention_cut" << std::endl;
        std::cout << "\nRequired parameters:" << std::endl;
        std::cout << "  batch, spatial_size, num_query, num_heads, channels, num_levels, num_points, im2col_step" << std::endl;
        std::cout << "Optional parameters:" << std::endl;
        std::cout << "  dir (default: data/binary_400x800/cross_attention)" << std::endl;
        return 1;
    }
    auto args = parse_args(argc, argv);
    // === 2. 从解析的参数中获取元数据和维度 ===
    const int batch               = get_param<int>(args, "batch", 48);
    const int spatial_size        = get_param<int>(args, "spatial_size", 20522); 
    const int num_query           = get_param<int>(args, "num_query", 123376);
    const int num_heads           = get_param<int>(args, "num_heads", 1);
    const int channels            = get_param<int>(args, "channels", 32);
    const int num_levels          = get_param<int>(args, "num_levels", 4);
    const int num_points          = get_param<int>(args, "num_points", 8);
    const int im2col_step         = get_param<int>(args, "im2col_step", 64);
    const std::string data_dir    = get_param<std::string>(args, "dir", "data/binary_400x800/cross_attention");
 
    // 根据维度计算元素数量
    long long value_elements = batch * num_query * num_heads * channels;
    std::cout << "\nLoading data from .bin files in '" << data_dir << "'..." << std::endl;
    auto h_value = read_bin_file<__half>(data_dir + "/value.bin");
    auto h_spatial_shapes = read_bin_file<int64_t>(data_dir + "/spatial_shapes.bin");
    auto h_level_start_index = read_bin_file<int64_t>(data_dir + "/level_start_index.bin");
    auto h_sampling_loc = read_bin_file<__half>(data_dir + "/sampling_locations.bin");
    auto h_attn_weight = read_bin_file<__half>(data_dir + "/attention_weights.bin");
    __half* d_value, *d_sampling_loc, *d_attn_weight, *d_output;
    int64_t* d_spatial_shapes;
    int64_t* d_level_start_index;


    CUDA_CHECK(cudaMalloc(&d_value, h_value.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spatial_shapes, h_spatial_shapes.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_level_start_index, h_level_start_index.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sampling_loc, h_sampling_loc.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_weight, h_attn_weight.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, value_elements * sizeof(__half)));
    std::cout << "Copying data from Host to Device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), h_value.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spatial_shapes, h_spatial_shapes.data(), h_spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_level_start_index, h_level_start_index.data(), h_level_start_index.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sampling_loc, h_sampling_loc.data(), h_sampling_loc.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_weight, h_attn_weight.data(), h_attn_weight.size() * sizeof(__half), cudaMemcpyHostToDevice));
    
    
    // Allocate device memory for each BATCH and LEVEL
    // Total: 48 batches × 4 levels = 192 separate buffers
    CUtensorMap h_tma_descs[batch][num_levels];  // Host-side array
    auto cuTensorMapEncodeTiled_func = get_cuTensorMapEncodeTiled();
    if (!cuTensorMapEncodeTiled_func) {
        printf("Failed to get cuTensorMapEncodeTiled function pointer\n");
        return 1;
    }
    printf("Creating TMA descriptors...\n");
    printf("level_start_index: [");
    for (int l = 0; l < num_levels; l++) {
        printf("%lld%s", h_level_start_index[l], (l < num_levels-1) ? ", " : "");
    }
    printf("]\n");
    printf("spatial_shapes: [");
    for (int l = 0; l < num_levels; l++) {
        printf("(%lld,%lld)%s", h_spatial_shapes[l*2], h_spatial_shapes[l*2+1],
               (l < num_levels-1) ? ", " : "");
    }
    printf("]\n");
    printf("single_batch_total_size=%zu, spatial_size=%d, channels=%d\n",
           spatial_size * (size_t)channels, spatial_size, channels);
    fflush(stdout);
    size_t single_batch_total_size = spatial_size * channels;
    for (int b = 0; b < batch; b++) {
        for (int l = 0; l < num_levels; l++) {
            size_t batch_offset = b * single_batch_total_size;
            auto value_of_batch = d_value + batch_offset;
            int h = h_spatial_shapes[l * 2];
            int w = h_spatial_shapes[l * 2 + 1];

            // Create TMA descriptor for this batch×level combination
            uint64_t globalDim[3] = {(uint64_t)channels, (uint64_t)(w + 2), (uint64_t)(h + 2)};
            uint64_t globalStrides[2] = {
                channels * sizeof(half),
                (w + 2) * channels * sizeof(half)
            };
            uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
            uint32_t elementStrides[3] = {1, 1, 1};
            auto _value = value_of_batch + h_level_start_index[l] * channels;

            if (b == 0) {
                size_t offset_bytes = ((char*)_value - (char*)d_value);
                printf("  Level %d: offset=%zu bytes, alignment=%zu\n",
                       l, offset_bytes, offset_bytes % 128);
            }

            CUresult res = cuTensorMapEncodeTiled_func(
                &h_tma_descs[b][l],
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                3,
                _value,
                globalDim,
                globalStrides,
                boxDim,
                elementStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );

            if (res != CUDA_SUCCESS) {
                printf("Failed to create TMA descriptor for batch %d, level %d: error %d\n", b, l, res);
                return 1;
            }
        }
        printf("  ✓ Created descriptors for batch %d\n", b);
        fflush(stdout);
    }

    // Copy TMA descriptors to device
    printf("Copying TMA descriptors to device...\n");
    fflush(stdout);
    CUtensorMap *d_tma_descs;
    CUDA_CHECK(cudaMalloc(&d_tma_descs, batch * num_levels * sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tma_descs, h_tma_descs,
                          batch * num_levels * sizeof(CUtensorMap),
                          cudaMemcpyHostToDevice));
    printf("  ✓ Copied all %d descriptors to device\n\n", batch * num_levels);
    
    cudaStream_t stream;
    cudaError_t status = cudaStreamCreate(&stream);
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(status));
        // 在这里处理错误，例如退出程序
    }
    ms_deformable_im2col_cuda(
              stream,
              d_value,
              d_spatial_shapes,
              d_level_start_index,
              d_sampling_loc,
              d_attn_weight,
              batch, spatial_size, num_heads, channels, num_levels, num_query,
              num_points, d_tma_descs, d_output);
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待 kernel 执行完成
    std::vector<__half> h_output(value_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    size_t count_output = std::min((size_t)100, h_output.size());
    std::cout << "  [ ";
    for (size_t i = 0; i < count_output; ++i) {
        std::cout << static_cast<float>(h_output[i]) << (i == count_output - 1 ? "" : ", ");
    }
    std::cout << " ]\n";

    // === 7. 释放所有 GPU 内存 ===
    std::cout << "Freeing GPU memory..." << std::endl;
    cudaFree(d_value);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_index);
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);

    return 0;
}