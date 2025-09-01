#pragma once

#include <iostream>
#include <vector>
#include <memory>

struct BoundingBox {
    float x, y, z, w, l, h, rt, vx, vy, score, label;
};

#ifdef __cplusplus
extern "C" {
#endif

void nms_launch(float* boxes, int n_boxes, float iou_threshold, uint64_t* mask);

std::vector<int> nms_cpu(std::vector<BoundingBox>& boxes, float iou_threshold);

#ifdef __cplusplus
}
#endif