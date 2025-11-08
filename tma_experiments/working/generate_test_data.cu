#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <random>

// Generate pseudo test data and save to binary files
// Configuration: batch=48, spatial_size=20522, num_query=123376,
//                num_heads=1, channels=32, num_levels=4, num_points=8
// spatial_shapes: [[92,160],[46,80],[23,40],[12,20]]
// level_start_index: [0, 15228, 19164, 20214]

void save_binary(const char* filename, const void* data, size_t size_bytes) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        printf("Failed to open %s for writing\n", filename);
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(data), size_bytes);
    printf("Saved %s (%zu bytes)\n", filename, size_bytes);
}

int main() {
    printf("=== Generating Test Data for Deformable Attention ===\n\n");

    // Configuration
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

    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 1. Value: [batch][spatial_size][channels]
    printf("Generating value data...\n");
    std::vector<__half> value(batch * spatial_size * channels);
    for (size_t i = 0; i < value.size(); i++) {
        value[i] = __float2half(dis(gen) * 2.0f - 1.0f);  // [-1, 1]
    }
    save_binary("test_data_value.bin", value.data(), value.size() * sizeof(__half));

    // 2. Spatial shapes: [[92,160], [46,80], [23,40], [12,20]]
    printf("Generating spatial_shapes data...\n");
    std::vector<int64_t> spatial_shapes = {92, 160, 46, 80, 23, 40, 12, 20};
    save_binary("test_data_spatial_shapes.bin", spatial_shapes.data(), spatial_shapes.size() * sizeof(int64_t));

    // 3. Level start index: [0, 15228, 19164, 20214]
    printf("Generating level_start_index data...\n");
    std::vector<int64_t> level_start_index = {0, 15228, 19164, 20214};
    save_binary("test_data_level_start_index.bin", level_start_index.data(), level_start_index.size() * sizeof(int64_t));

    // 4. Sampling locations: [batch][num_query][num_heads][num_levels][num_points][2]
    printf("Generating sampling_locations data...\n");
    std::vector<__half> sampling_loc(batch * num_query * num_heads * num_levels * num_points * 2);
    for (size_t i = 0; i < sampling_loc.size(); i++) {
        sampling_loc[i] = __float2half(dis(gen));  // [0, 1] normalized coordinates
    }
    save_binary("test_data_sampling_locations.bin", sampling_loc.data(), sampling_loc.size() * sizeof(__half));

    // 5. Attention weights: [batch][num_query][num_heads][num_levels][num_points]
    printf("Generating attention_weights data...\n");
    std::vector<__half> attn_weight(batch * num_query * num_heads * num_levels * num_points);
    for (size_t i = 0; i < attn_weight.size(); i++) {
        attn_weight[i] = __float2half(dis(gen) * 0.5f);  // [0, 0.5]
    }
    save_binary("test_data_attention_weights.bin", attn_weight.data(), attn_weight.size() * sizeof(__half));

    printf("\n=== Data Generation Complete ===\n");
    printf("Total data size: %.2f MB\n",
           (value.size() * sizeof(__half) +
            sampling_loc.size() * sizeof(__half) +
            attn_weight.size() * sizeof(__half) +
            spatial_shapes.size() * sizeof(int64_t) +
            level_start_index.size() * sizeof(int64_t)) / (1024.0f * 1024.0f));

    return 0;
}
