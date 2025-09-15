/*
 * Test program for NVIDIA voxelization with binary point cloud data
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "operators/voxelization/voxelization_kernel.h"

// Load binary point cloud data
int loadBinaryData(const char* filename, float** points, int* num_points) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return -1;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Each point has 5 floats: x, y, z, intensity, ring
    size_t pointSize = 5 * sizeof(float);
    *num_points = fileSize / pointSize;

    if (fileSize % pointSize != 0) {
        std::cerr << "Warning: File size is not a multiple of point size" << std::endl;
    }

    // Allocate host memory
    float* host_points = new float[*num_points * 5];

    // Read binary data
    file.read(reinterpret_cast<char*>(host_points), fileSize);
    file.close();

    // Allocate and copy to GPU
    cudaMalloc((void**)points, *num_points * 5 * sizeof(float));
    cudaMemcpy(*points, host_points, *num_points * 5 * sizeof(float), cudaMemcpyHostToDevice);

    delete[] host_points;

    std::cout << "Loaded " << *num_points << " points from " << filename << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file>" << std::endl;
        return 1;
    }

    // Load point cloud data
    float* d_points;
    int num_points;
    if (loadBinaryData(argv[1], &d_points, &num_points) != 0) {
        return 1;
    }

    // Setup voxelization parameters
    VoxelParams params;
    params.min_x_range = -54.0f;
    params.max_x_range = 54.0f;
    params.min_y_range = -54.0f;
    params.max_y_range = 54.0f;
    params.min_z_range = -5.0f;
    params.max_z_range = 3.0f;
    params.voxel_x_size = 0.075f;
    params.voxel_y_size = 0.075f;
    params.voxel_z_size = 0.2f;
    params.max_points_per_voxel = 10;
    params.max_voxels = 160000;
    params.feature_num = 5;
    params.grid_x_size = params.getGridXSize();
    params.grid_y_size = params.getGridYSize();
    params.grid_z_size = params.getGridZSize();

    std::cout << "Voxelization parameters:" << std::endl;
    std::cout << "  Grid size: " << params.grid_x_size << " x "
              << params.grid_y_size << " x " << params.grid_z_size << std::endl;
    std::cout << "  Max voxels: " << params.max_voxels << std::endl;
    std::cout << "  Max points per voxel: " << params.max_points_per_voxel << std::endl;

    // Allocate memory for voxelization
    int hash_table_size = num_points * 2 * 2;
    unsigned int* d_hash_table;
    float* d_voxels_temp;
    half* d_voxel_features;
    unsigned int* d_num_points_per_voxel;
    unsigned int* d_voxel_indices;
    unsigned int* d_real_voxel_num;
    unsigned int h_real_voxel_num = 0;

    cudaMalloc((void**)&d_hash_table, hash_table_size * sizeof(unsigned int));
    cudaMalloc((void**)&d_voxels_temp, params.max_voxels * params.max_points_per_voxel * params.feature_num * sizeof(float));
    cudaMalloc((void**)&d_voxel_features, params.max_voxels * params.feature_num * sizeof(half));
    cudaMalloc((void**)&d_num_points_per_voxel, params.max_voxels * sizeof(unsigned int));
    cudaMalloc((void**)&d_voxel_indices, params.max_voxels * 4 * sizeof(unsigned int));
    cudaMalloc((void**)&d_real_voxel_num, sizeof(unsigned int));

    // Initialize
    cudaMemset(d_hash_table, 0xff, hash_table_size * sizeof(unsigned int));
    cudaMemset(d_voxels_temp, 0, params.max_voxels * params.max_points_per_voxel * params.feature_num * sizeof(float));
    cudaMemset(d_num_points_per_voxel, 0, params.max_voxels * sizeof(unsigned int));
    cudaMemset(d_real_voxel_num, 0, sizeof(unsigned int));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run voxelization
    std::cout << "Running voxelization..." << std::endl;
    cudaError_t err = voxelizationLaunch(
        d_points, num_points,
        params.min_x_range, params.max_x_range,
        params.min_y_range, params.max_y_range,
        params.min_z_range, params.max_z_range,
        params.voxel_x_size, params.voxel_y_size, params.voxel_z_size,
        params.grid_y_size, params.grid_x_size, params.feature_num,
        params.max_voxels, params.max_points_per_voxel,
        d_hash_table, d_num_points_per_voxel,
        d_voxels_temp, d_voxel_indices, d_real_voxel_num, stream
    );

    if (err != cudaSuccess) {
        std::cerr << "Voxelization failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy real voxel count
    cudaMemcpy(&h_real_voxel_num, d_real_voxel_num, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::cout << "Number of occupied voxels: " << h_real_voxel_num << std::endl;

    // Run feature extraction
    std::cout << "Running feature extraction..." << std::endl;
    err = featureExtractionLaunch(
        d_voxels_temp, d_num_points_per_voxel,
        h_real_voxel_num, params.max_points_per_voxel,
        params.feature_num, d_voxel_features, stream
    );

    if (err != cudaSuccess) {
        std::cerr << "Feature extraction failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaStreamSynchronize(stream);

    // Get statistics
    unsigned int* h_num_points_per_voxel = new unsigned int[params.max_voxels];
    cudaMemcpy(h_num_points_per_voxel, d_num_points_per_voxel,
               params.max_voxels * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int total_points_in_voxels = 0;
    int non_empty_voxels = 0;
    for (int i = 0; i < params.max_voxels; i++) {
        if (h_num_points_per_voxel[i] > 0) {
            non_empty_voxels++;
            total_points_in_voxels += h_num_points_per_voxel[i];
        }
    }

    std::cout << "\nVoxelization Results:" << std::endl;
    std::cout << "  Total points: " << num_points << std::endl;
    std::cout << "  Points in voxels: " << total_points_in_voxels << std::endl;
    std::cout << "  Non-empty voxels: " << non_empty_voxels << std::endl;
    if (non_empty_voxels > 0) {
        std::cout << "  Average points per voxel: "
                  << (float)total_points_in_voxels / non_empty_voxels << std::endl;
    }

    // Cleanup
    delete[] h_num_points_per_voxel;
    cudaFree(d_points);
    cudaFree(d_hash_table);
    cudaFree(d_voxels_temp);
    cudaFree(d_voxel_features);
    cudaFree(d_num_points_per_voxel);
    cudaFree(d_voxel_indices);
    cudaFree(d_real_voxel_num);
    cudaStreamDestroy(stream);

    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}