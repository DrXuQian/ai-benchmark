#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <chrono>

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;
using dtype = half;

// Constants matching deform_attn.cu
constexpr int NUM_POINTS = 8;
constexpr int NUM_LEVELS = 4;
constexpr int CHANNELS = 32;
constexpr int NUM_OUTPUT = 8;  // Output elements per thread
constexpr int OUTPUTS_IN_THREAD = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int LOADER_STRIDE = 4;  // Only tid%4==0 threads are loaders

// TMA tile dimensions: 2x2x32
constexpr int TILE_H = 2;
constexpr int TILE_W = 2;
constexpr int TILE_C = 32;

// Macros from deform_attn.cu
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])

// Level configuration
struct LevelConfig {
    int H, W;
    int H_padded, W_padded;
    int start_idx;
};

__constant__ LevelConfig d_level_configs[NUM_LEVELS];

// Kernel matching deform_attn.cu structure with TMA and stride-4 pattern
__global__ void tma_stride4_deform_kernel(
    const CUtensorMap* tma_descs_all,   // [batch][level] flattened array
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const dtype *data_sampling_loc,     // [batch][query][heads][levels][points][2]
    const dtype *data_attn_weight,      // [batch][query][heads][levels][points]
    const int batch_size,
    const int spatial_size,
    const int num_query,
    const int num_heads,
    const int channels,
    const int n,                        // Total number of output elements / NUM_OUTPUT
    dtype *data_col                     // Output
) {
    const int tid = threadIdx.x;

    // Shared memory for TMA loads - sized for one block's work
    __shared__ alignas(128) dtype smem_tile[NUM_LEVELS][NUM_POINTS][TILE_H][TILE_W][TILE_C];

    // Block-wide barrier for synchronization
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (tid == 0) {
        init(&bar, THREADS_PER_BLOCK);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    // Only tid%4==0 threads are loaders
    const bool is_loader = (tid % LOADER_STRIDE == 0);

    CUDA_1D_KERNEL_LOOP(index, n) {
        // Decode index following deform_attn.cu logic
        int _temp = index << 3;  // index * NUM_OUTPUT
        const int c_col = _temp & (channels - 1);  // Channel index
        _temp = (_temp >> 5);  // Divide by 32 (CHANNELS)
        const int sampling_index = _temp;
        const int b_col = _temp / num_query;  // Batch index
        const int q_col = _temp % num_query;  // Query index

        // Prefetch TMA descriptors (once per block)
        if (tid == 0 && blockIdx.x < batch_size) {
            #pragma unroll
            for (int l = 0; l < NUM_LEVELS; l++) {
                const int desc_idx = b_col * NUM_LEVELS + l;
                const CUtensorMap* desc_ptr = &tma_descs_all[desc_idx];
                asm volatile(
                    "prefetch.tensormap [%0];\n\t"
                    :: "l"(reinterpret_cast<uint64_t>(desc_ptr))
                );
            }
        }

        // Output pointer for this thread
        dtype *data_col_ptr = data_col + (index << 3);

        // Initialize output to zero
        dtype col[NUM_OUTPUT];
        #pragma unroll
        for (int idx = 0; idx < (NUM_OUTPUT >> 1); idx++) {
            reinterpret_cast<half2*>(col)[idx] = __half2(0.0f, 0.0f);
        }

        // Pointers for sampling locations and attention weights
        int data_weight_ptr = sampling_index << 5;  // * NUM_LEVELS * NUM_POINTS
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int data_value_ptr_init_offset = b_col * spatial_size * channels;

        dtype *data_half = const_cast<dtype*>(data_sampling_loc);
        dtype *data_attn_weight_half = const_cast<dtype*>(data_attn_weight);

        const half2 zp5 = __half2(0.5f, 0.5f);

        // Process each level
        for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
            const int level_start_id = data_level_start_index[l_col];
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = data_spatial_shapes[spatial_h_ptr];
            const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
            const half2 spatial_hw = __half2(spatial_w, spatial_h);

            // Load sampling locations and attention weights (matching deform_attn.cu)
            half2 loc_hw_vec[NUM_POINTS];
            half weight_vec[NUM_POINTS];

            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINTS; pack_id += 4) {
                LDST128BITS(loc_hw_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_half[data_loc_w_ptr + (pack_id << 1)]));
            }
            #pragma unroll
            for (int pack_id = 0; pack_id < NUM_POINTS; pack_id += 8) {
                LDST128BITS(weight_vec[pack_id]) = __ldcg(reinterpret_cast<float4*>(&data_attn_weight_half[data_weight_ptr + pack_id]));
            }
            data_loc_w_ptr += (NUM_POINTS << 1);
            data_weight_ptr += NUM_POINTS;

            // Issue TMA loads for all points (only loader threads)
            #pragma unroll
            for (int p_col = 0; p_col < NUM_POINTS; ++p_col) {
                if (is_loader) {
                    const half2 loc = loc_hw_vec[p_col];
                    half2 hw_im = __hfma2(loc, spatial_hw, zp5);
                    dtype h_im = __high2half(hw_im);
                    dtype w_im = __low2half(hw_im);

                    if (h_im > __float2half(0.0f) && w_im > __float2half(0.0f) &&
                        h_im < __int2half_rn(spatial_h + 1) && w_im < __int2half_rn(spatial_w + 1)) {

                        int32_t const hLow = __half2int_rd(h_im);
                        int32_t const wLow = __half2int_rd(w_im);

                        // TMA coordinates (clamped to valid range)
                        int32_t tensor_coord_h = max(0, min(hLow, spatial_h - 2));
                        int32_t tensor_coord_w = max(0, min(wLow, spatial_w - 2));
                        int32_t tensor_coord_c = 0;

                        // Get TMA descriptor for this batch and level
                        const int desc_idx = b_col * NUM_LEVELS + l_col;
                        const CUtensorMap* tma_desc = &tma_descs_all[desc_idx];

                        // Issue TMA load
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3, %4}], [%5];\n\t"
                            :
                            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&smem_tile[l_col][p_col][0][0][0]))),
                              "l"(reinterpret_cast<uint64_t>(tma_desc)),
                              "r"(tensor_coord_c), "r"(tensor_coord_w), "r"(tensor_coord_h),
                              "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                            : "memory"
                        );

                        // Expect data for this tile
                        asm volatile(
                            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                            :
                            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                              "n"(TILE_H * TILE_W * TILE_C * sizeof(dtype))
                        );
                    } else {
                        // Out of bounds - expect 0 bytes
                        asm volatile(
                            "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n\t"
                            :
                            : "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))),
                              "n"(0)
                        );
                    }
                }
            }
        }

        // Wait for all TMA loads to complete
        barrier::arrival_token token = bar.arrive();
        bar.wait(std::move(token));

        // Now compute bilinear interpolation using loaded data
        // This part would use smem_tile data and compute the output
        // For now, just copy some data to verify loading works
        #pragma unroll
        for (int i = 0; i < NUM_OUTPUT; i++) {
            col[i] = smem_tile[0][0][0][0][c_col];
        }

        // Write output
        #pragma unroll
        for (int i = 0; i < NUM_OUTPUT; i++) {
            data_col_ptr[i] = col[i];
        }
    }
}

// Helper to load binary file
template <typename T>
std::vector<T> load_binary(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + std::string(filename));
    }
    file.seekg(0, std::ios::end);
    size_t size_bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(size_bytes / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), size_bytes);
    return data;
}

int main() {
    printf("=== TMA Stride-4 Deformable Attention (deform_attn.cu style) ===\n\n");
    fflush(stdout);

    // Configuration
    const int batch = 48;
    const int num_query = 1000;
    const int num_heads = 1;
    const int channels = 32;

    // Calculate kernel launch configuration (matching deform_attn.cu)
    const int num_kernels = batch * num_query * num_heads * channels / OUTPUTS_IN_THREAD;
    const int num_blocks = (num_kernels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Configuration:\n");
    printf("  Batches: %d\n", batch);
    printf("  Queries per batch: %d\n", num_query);
    printf("  Heads: %d\n", num_heads);
    printf("  Channels: %d\n", channels);
    printf("  Output elements per thread: %d\n", OUTPUTS_IN_THREAD);
    printf("  Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("  Loader pattern: tid%%4==0 (stride-4)\n");
    printf("  Total kernels: %d\n", num_kernels);
    printf("  Total blocks: %d\n\n", num_blocks);

    // Level configurations
    LevelConfig h_level_configs[NUM_LEVELS] = {
        {92, 160, 94, 162, 0},
        {46, 80, 48, 82, 15228},
        {23, 40, 25, 42, 19164},
        {12, 20, 14, 22, 20214}
    };

    // Copy to device constant memory
    cudaMemcpyToSymbol(d_level_configs, h_level_configs, sizeof(h_level_configs));

    // Prepare spatial shapes and level start indices
    std::vector<int64_t> spatial_shapes;
    std::vector<int64_t> level_start_indices;

    for (int l = 0; l < NUM_LEVELS; l++) {
        spatial_shapes.push_back(h_level_configs[l].H);
        spatial_shapes.push_back(h_level_configs[l].W);
        level_start_indices.push_back(h_level_configs[l].start_idx);
    }

    int64_t *d_spatial_shapes, *d_level_start_indices;
    cudaMalloc(&d_spatial_shapes, spatial_shapes.size() * sizeof(int64_t));
    cudaMalloc(&d_level_start_indices, level_start_indices.size() * sizeof(int64_t));
    cudaMemcpy(d_spatial_shapes, spatial_shapes.data(), spatial_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_start_indices, level_start_indices.data(), level_start_indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Load test data
    printf("Loading test data...\n");
    auto value_data = load_binary<dtype>("working/value.bin");
    auto sampling_loc = load_binary<dtype>("working/sampling_locations.bin");
    auto attn_weight = load_binary<dtype>("working/attention_weights.bin");

    printf("  Value data: %zu elements\n", value_data.size());
    printf("  Sampling locations: %zu elements\n", sampling_loc.size());
    printf("  Attention weights: %zu elements\n\n", attn_weight.size());

    // Allocate device memory
    dtype *d_sampling_loc, *d_attn_weight, *d_output;
    const size_t output_size = batch * num_query * num_heads * channels;

    cudaMalloc(&d_sampling_loc, sampling_loc.size() * sizeof(dtype));
    cudaMalloc(&d_attn_weight, attn_weight.size() * sizeof(dtype));
    cudaMalloc(&d_output, output_size * sizeof(dtype));

    cudaMemcpy(d_sampling_loc, sampling_loc.data(), sampling_loc.size() * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_weight, attn_weight.data(), attn_weight.size() * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size * sizeof(dtype));

    // Create TMA descriptors for all batch×level combinations
    printf("Creating TMA descriptors...\n");
    std::vector<CUtensorMap> h_tma_descs;
    std::vector<dtype*> d_value_ptrs;

    for (int b = 0; b < batch; b++) {
        size_t batch_offset = 0;
        for (int l = 0; l < NUM_LEVELS; l++) {
            batch_offset = h_level_configs[l].start_idx * channels;
        }

        // Allocate and copy value data for this batch
        dtype* d_value_batch;
        cudaMalloc(&d_value_batch, value_data.size() * sizeof(dtype) / batch);
        cudaMemcpy(d_value_batch, value_data.data(), value_data.size() * sizeof(dtype) / batch, cudaMemcpyHostToDevice);
        d_value_ptrs.push_back(d_value_batch);

        for (int l = 0; l < NUM_LEVELS; l++) {
            CUtensorMap tma_desc;
            uint64_t globalDim[3] = {
                CHANNELS,
                (uint64_t)h_level_configs[l].W_padded,
                (uint64_t)h_level_configs[l].H_padded
            };
            uint64_t globalStride[3] = {
                sizeof(dtype),
                CHANNELS * sizeof(dtype),
                h_level_configs[l].W_padded * CHANNELS * sizeof(dtype)
            };
            uint32_t boxDim[3] = {TILE_C, TILE_W, TILE_H};
            uint32_t elementStride[3] = {1, 1, 1};

            size_t level_offset = h_level_configs[l].start_idx * CHANNELS * sizeof(dtype);

            cuTensorMapEncodeTiled(
                &tma_desc,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                3,
                (void*)(d_value_batch + level_offset / sizeof(dtype)),
                globalDim,
                globalStride,
                boxDim,
                elementStride,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
            h_tma_descs.push_back(tma_desc);
        }
    }

    // Copy TMA descriptors to device
    CUtensorMap* d_tma_descs;
    cudaMalloc(&d_tma_descs, h_tma_descs.size() * sizeof(CUtensorMap));
    cudaMemcpy(d_tma_descs, h_tma_descs.data(), h_tma_descs.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    printf("  Created %zu TMA descriptors\n\n", h_tma_descs.size());

    // Warmup
    printf("Running warmup...\n");
    for (int i = 0; i < 3; i++) {
        tma_stride4_deform_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_tma_descs, d_spatial_shapes, d_level_start_indices,
            d_sampling_loc, d_attn_weight, batch, 20454, num_query,
            num_heads, channels, num_kernels, d_output
        );
    }
    cudaDeviceSynchronize();

    // Benchmark
    printf("Running benchmark (10 iterations)...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        tma_stride4_deform_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_tma_descs, d_spatial_shapes, d_level_start_indices,
            d_sampling_loc, d_attn_weight, batch, 20454, num_query,
            num_heads, channels, num_kernels, d_output
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
        printf("  Iteration %2d: %.4f ms\n", i + 1, iter_time);
    }

    float avg_time = total_time / 10;
    printf("\n=== Performance Results ===\n");
    printf("Average time: %.4f ms\n", avg_time);
    printf("Throughput: %.2f queries/ms\n", (batch * num_query) / avg_time);

    // Cleanup
    cudaFree(d_sampling_loc);
    cudaFree(d_attn_weight);
    cudaFree(d_output);
    cudaFree(d_tma_descs);
    cudaFree(d_spatial_shapes);
    cudaFree(d_level_start_indices);
    for (auto ptr : d_value_ptrs) {
        cudaFree(ptr);
    }

    printf("\n✅ Test completed!\n");
    return 0;
}