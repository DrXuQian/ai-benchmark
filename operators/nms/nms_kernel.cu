#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include "nms_kernel.h"

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define DIVUP(x, y) (((x) + (y) - 1) / (y))
const int NMS_THREADS_PER_BLOCK = 64;
const int DET_CHANNEL = 11;

__device__ inline float min_d(float a, float b) {
    return a > b ? b : a;
}

__device__ inline float max_d(float a, float b) {
    return a > b ? a : b;
}

__device__ inline float cross_d(const float2 &A, const float2 &B, const float2 &C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

__device__ inline void get_rotated_vertices(float x, float y, float w, float l, float theta, float2* corners) {
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    
    float dx1 = -w / 2.0f, dy1 = -l / 2.0f;
    float dx2 = w / 2.0f, dy2 = -l / 2.0f;
    float dx3 = w / 2.0f, dy3 = l / 2.0f;
    float dx4 = -w / 2.0f, dy4 = l / 2.0f;
    
    corners[0].x = x + dx1 * cos_theta - dy1 * sin_theta;
    corners[0].y = y + dx1 * sin_theta + dy1 * cos_theta;
    corners[1].x = x + dx2 * cos_theta - dy2 * sin_theta;
    corners[1].y = y + dx2 * sin_theta + dy2 * cos_theta;
    corners[2].x = x + dx3 * cos_theta - dy3 * sin_theta;
    corners[2].y = y + dx3 * sin_theta + dy3 * cos_theta;
    corners[3].x = x + dx4 * cos_theta - dy4 * sin_theta;
    corners[3].y = y + dx4 * sin_theta + dy4 * cos_theta;
}

__device__ inline bool line_segment_intersect(const float2 &A, const float2 &B, 
                                             const float2 &C, const float2 &D, 
                                             float2 &intersect) {
    float s1 = cross_d(C, D, A);
    float s2 = cross_d(C, D, B);
    float s3 = cross_d(A, B, C);
    float s4 = cross_d(A, B, D);

    if (!(s1 * s2 < 0 && s3 * s4 < 0)) return false;

    float s5 = cross_d(A, D, B);
    if (fabsf(s5 - s1) > 1e-8) {
        intersect.x = (s5 * C.x - s1 * D.x) / (s5 - s1);
        intersect.y = (s5 * C.y - s1 * D.y) / (s5 - s1);
    } else {
        intersect.x = (A.x + B.x) / 2.0f;
        intersect.y = (A.y + B.y) / 2.0f;
    }
    return true;
}

__device__ inline float polygon_area(float2* int_pts, int num_of_inter) {
    float area = 0.0f;
    for (int i = 0; i < num_of_inter - 1; i++) {
        area += int_pts[i].x * int_pts[i + 1].y - int_pts[i + 1].x * int_pts[i].y;
    }
    return fabsf(area) / 2.0f;
}

__device__ inline float rotated_boxes_intersection(float x1, float y1, float w1, float l1, float r1,
                                                  float x2, float y2, float w2, float l2, float r2) {
    float2 corners1[4], corners2[4];
    float2 intersections[24];
    
    get_rotated_vertices(x1, y1, w1, l1, r1, corners1);
    get_rotated_vertices(x2, y2, w2, l2, r2, corners2);

    int num_intersections = 0;

    for (int i = 0; i < 4; i++) {
        bool inside = true;
        for (int j = 0; j < 4; j++) {
            float2 edge_start = corners2[j];
            float2 edge_end = corners2[(j + 1) % 4];
            if (cross_d(edge_start, edge_end, corners1[i]) < 0) {
                inside = false;
                break;
            }
        }
        if (inside) {
            intersections[num_intersections++] = corners1[i];
        }
    }

    for (int i = 0; i < 4; i++) {
        bool inside = true;
        for (int j = 0; j < 4; j++) {
            float2 edge_start = corners1[j];
            float2 edge_end = corners1[(j + 1) % 4];
            if (cross_d(edge_start, edge_end, corners2[i]) < 0) {
                inside = false;
                break;
            }
        }
        if (inside) {
            intersections[num_intersections++] = corners2[i];
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float2 intersect;
            if (line_segment_intersect(corners1[i], corners1[(i + 1) % 4],
                                     corners2[j], corners2[(j + 1) % 4], intersect)) {
                intersections[num_intersections++] = intersect;
            }
        }
    }

    if (num_intersections < 3) return 0.0f;

    float2 center = {0.0f, 0.0f};
    for (int i = 0; i < num_intersections; i++) {
        center.x += intersections[i].x;
        center.y += intersections[i].y;
    }
    center.x /= num_intersections;
    center.y /= num_intersections;

    for (int i = 0; i < num_intersections - 1; i++) {
        for (int j = 0; j < num_intersections - 1 - i; j++) {
            float angle1 = atan2f(intersections[j].y - center.y, intersections[j].x - center.x);
            float angle2 = atan2f(intersections[j + 1].y - center.y, intersections[j + 1].x - center.x);
            if (angle1 > angle2) {
                float2 temp = intersections[j];
                intersections[j] = intersections[j + 1];
                intersections[j + 1] = temp;
            }
        }
    }

    return polygon_area(intersections, num_intersections);
}

__device__ inline float devIoU(float* box1, float* box2) {
    float x1 = box1[0], y1 = box1[1], w1 = box1[3], l1 = box1[4], r1 = box1[6];
    float x2 = box2[0], y2 = box2[1], w2 = box2[3], l2 = box2[4], r2 = box2[6];
    
    float area1 = w1 * l1;
    float area2 = w2 * l2;
    float intersection = rotated_boxes_intersection(x1, y1, w1, l1, r1, x2, y2, w2, l2, r2);
    
    return intersection / (area1 + area2 - intersection + 1e-8f);
}

__global__ void nms_cuda(int n_boxes, float iou_threshold, float* dev_boxes, uint64_t* dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    
    if (row_start > col_start) return;
    
    const int row_size = min(n_boxes - row_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
    const int col_size = min(n_boxes - col_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
    
    __shared__ float block_boxes[NMS_THREADS_PER_BLOCK * DET_CHANNEL];
    
    if (threadIdx.x < col_size) {
        for (int i = 0; i < DET_CHANNEL; i++) {
            block_boxes[threadIdx.x * DET_CHANNEL + i] = 
                dev_boxes[(col_start * NMS_THREADS_PER_BLOCK + threadIdx.x) * DET_CHANNEL + i];
        }
    }
    __syncthreads();
    
    if (threadIdx.x < row_size) {
        const int cur_box_idx = row_start * NMS_THREADS_PER_BLOCK + threadIdx.x;
        float* cur_box = dev_boxes + cur_box_idx * DET_CHANNEL;
        uint64_t t = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0;
        
        for (int i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * DET_CHANNEL) > iou_threshold) {
                t |= 1ULL << i;
            }
        }
        dev_mask[cur_box_idx * gridDim.y + col_start] = t;
    }
}

extern "C" {

void nms_launch(float* boxes, int n_boxes, float iou_threshold, uint64_t* mask) {
    dim3 blocks(DIVUP(n_boxes, NMS_THREADS_PER_BLOCK), DIVUP(n_boxes, NMS_THREADS_PER_BLOCK));
    dim3 threads(NMS_THREADS_PER_BLOCK);
    
    nms_cuda<<<blocks, threads>>>(n_boxes, iou_threshold, boxes, mask);
    checkCudaErrors(cudaDeviceSynchronize());
}

std::vector<int> nms_cpu(std::vector<BoundingBox>& boxes, float iou_threshold) {
    int n_boxes = boxes.size();
    if (n_boxes == 0) return std::vector<int>();
    
    std::vector<int> indices(n_boxes);
    for (int i = 0; i < n_boxes; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&boxes](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });
    
    std::vector<float> box_data(n_boxes * DET_CHANNEL);
    for (int i = 0; i < n_boxes; i++) {
        int idx = indices[i];
        box_data[i * DET_CHANNEL + 0] = boxes[idx].x;
        box_data[i * DET_CHANNEL + 1] = boxes[idx].y;
        box_data[i * DET_CHANNEL + 2] = boxes[idx].z;
        box_data[i * DET_CHANNEL + 3] = boxes[idx].w;
        box_data[i * DET_CHANNEL + 4] = boxes[idx].l;
        box_data[i * DET_CHANNEL + 5] = boxes[idx].h;
        box_data[i * DET_CHANNEL + 6] = boxes[idx].rt;
        box_data[i * DET_CHANNEL + 7] = boxes[idx].vx;
        box_data[i * DET_CHANNEL + 8] = boxes[idx].vy;
        box_data[i * DET_CHANNEL + 9] = boxes[idx].score;
        box_data[i * DET_CHANNEL + 10] = boxes[idx].label;
    }
    
    float* d_boxes;
    uint64_t* d_mask;
    int mask_size = n_boxes * DIVUP(n_boxes, NMS_THREADS_PER_BLOCK);
    
    checkCudaErrors(cudaMalloc(&d_boxes, n_boxes * DET_CHANNEL * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mask, mask_size * sizeof(uint64_t)));
    
    checkCudaErrors(cudaMemcpy(d_boxes, box_data.data(), 
                              n_boxes * DET_CHANNEL * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_mask, 0, mask_size * sizeof(uint64_t)));
    
    nms_launch(d_boxes, n_boxes, iou_threshold, d_mask);
    
    std::vector<uint64_t> mask(mask_size);
    checkCudaErrors(cudaMemcpy(mask.data(), d_mask, mask_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    std::vector<uint64_t> remv(DIVUP(n_boxes, 64));
    std::vector<int> keep;
    
    for (int i = 0; i < n_boxes; i++) {
        int nblock = i / NMS_THREADS_PER_BLOCK;
        int inblock = i % NMS_THREADS_PER_BLOCK;
        
        if (!(remv[nblock] & (1ULL << inblock))) {
            keep.push_back(indices[i]);
            uint64_t* p = &mask[i * DIVUP(n_boxes, NMS_THREADS_PER_BLOCK)];
            for (int j = nblock; j < DIVUP(n_boxes, NMS_THREADS_PER_BLOCK); j++) {
                remv[j] |= p[j];
            }
        }
    }
    
    checkCudaErrors(cudaFree(d_boxes));
    checkCudaErrors(cudaFree(d_mask));
    
    return keep;
}

}