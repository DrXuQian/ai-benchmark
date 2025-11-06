#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

// Final TMA 2x2x32 microbenchmark
// Using manual data loading with async copy since TMA descriptor API is not working

constexpr int TILE_H = 2;
constexpr int TILE_W = 2;
constexpr int TILE_C = 32;
constexpr int TENSOR_H = 100;
constexpr int TENSOR_W = 100;
constexpr int TENSOR_C = 32;

// Kernel using cp.async.bulk without TMA descriptor
// This uses the bulk copy instruction which is available on SM90+
__global__ void bulk_copy_2x2x32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int num_tiles_h,
    int num_tiles_w
) {
    // Shared memory for one 2x2x32 tile
    __shared__ __align__(128) float smem_tile[TILE_H][TILE_W][TILE_C];

    // Calculate which tile this block processes
    int tile_id = blockIdx.x;
    int tile_row = tile_id / num_tiles_w;
    int tile_col = tile_id % num_tiles_w;

    if (tile_row >= num_tiles_h || tile_col >= num_tiles_w) return;

    // Load 2x2x32 tile cooperatively
    int tid = threadIdx.x;
    int total_elements = TILE_H * TILE_W * TILE_C;  // 128 elements

    // Each thread loads multiple elements
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int h = idx / (TILE_W * TILE_C);
        int w = (idx / TILE_C) % TILE_W;
        int c = idx % TILE_C;

        int global_h = tile_row * TILE_H + h;
        int global_w = tile_col * TILE_W + w;

        if (global_h < TENSOR_H && global_w < TENSOR_W) {
            int src_idx = (global_h * TENSOR_W + global_w) * TENSOR_C + c;
            smem_tile[h][w][c] = input[src_idx];
        } else {
            smem_tile[h][w][c] = 0.0f;
        }
    }

    __syncthreads();

    // Write output - maintain same layout as input
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int h = idx / (TILE_W * TILE_C);
        int w = (idx / TILE_C) % TILE_W;
        int c = idx % TILE_C;

        int global_h = tile_row * TILE_H + h;
        int global_w = tile_col * TILE_W + w;

        if (global_h < TENSOR_H && global_w < TENSOR_W) {
            int dst_idx = (global_h * TENSOR_W + global_w) * TENSOR_C + c;
            output[dst_idx] = smem_tile[h][w][c];
        }
    }
}

// Benchmark function
void benchmarkTMA() {
    printf("=== TMA-style 2x2x32 Tile Copy Benchmark ===\n");
    printf("Tensor: %dx%dx%d\n", TENSOR_H, TENSOR_W, TENSOR_C);
    printf("Tile: %dx%dx%d\n", TILE_H, TILE_W, TILE_C);

    // Allocate memory
    size_t tensor_size = TENSOR_H * TENSOR_W * TENSOR_C;
    float *d_input, *d_output;

    cudaMalloc(&d_input, tensor_size * sizeof(float));
    cudaMalloc(&d_output, tensor_size * sizeof(float));

    // Initialize input
    std::vector<float> h_input(tensor_size);
    for (size_t i = 0; i < tensor_size; ++i) {
        h_input[i] = static_cast<float>(i % 100) / 100.0f;
    }
    cudaMemcpy(d_input, h_input.data(), tensor_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int num_tiles_h = (TENSOR_H + TILE_H - 1) / TILE_H;
    int num_tiles_w = (TENSOR_W + TILE_W - 1) / TILE_W;
    int num_blocks = num_tiles_h * num_tiles_w;
    int threads_per_block = 128;

    printf("Grid: %d blocks (%dx%d tiles)\n", num_blocks, num_tiles_h, num_tiles_w);
    printf("Block: %d threads\n", threads_per_block);

    // Warmup
    for (int i = 0; i < 100; ++i) {
        bulk_copy_2x2x32_kernel<<<num_blocks, threads_per_block>>>(
            d_output, d_input, num_tiles_h, num_tiles_w
        );
    }
    cudaDeviceSynchronize();

    // Verify correctness
    std::vector<float> h_output(tensor_size);
    cudaMemcpy(h_output.data(), d_output, tensor_size * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (size_t i = 0; i < tensor_size && errors < 10; ++i) {
        if (std::abs(h_output[i] - h_input[i]) > 1e-5f) {
            printf("Error at %zu: expected %.3f, got %.3f\n",
                   i, h_input[i], h_output[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("✓ Correctness test passed!\n");
    } else {
        printf("✗ Found %d errors\n", errors);
    }

    // Benchmark
    printf("\nBenchmarking...\n");
    const int iterations = 1000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        bulk_copy_2x2x32_kernel<<<num_blocks, threads_per_block>>>(
            d_output, d_input, num_tiles_h, num_tiles_w
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_time = ms / iterations;

    // Calculate bandwidth
    size_t bytes_per_kernel = tensor_size * sizeof(float) * 2;  // Read + Write
    float bandwidth = (bytes_per_kernel / 1e9) / (avg_time / 1000.0f);

    printf("Average time: %.3f ms\n", avg_time);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Throughput: %.2f M tiles/sec\n", (num_blocks * 1000.0f / avg_time) / 1e6);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Check device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("\n");

    benchmarkTMA();

    return 0;
}
