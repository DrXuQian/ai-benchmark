#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>

struct BoundingBox {
    float x, y, z, w, l, h, rt, vx, vy, score, label;
};

extern "C" std::vector<int> nms_cpu(std::vector<BoundingBox>& boxes, float iou_threshold);

torch::Tensor nms_forward(
    torch::Tensor boxes,
    float iou_threshold
) {
    // Check inputs
    TORCH_CHECK(boxes.is_cuda(), "Boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.dtype() == torch::kFloat32, "Boxes must be float32");
    TORCH_CHECK(boxes.dim() == 2, "Boxes must be 2D tensor");
    TORCH_CHECK(boxes.size(1) == 11, "Boxes must have 11 features [x,y,z,w,l,h,rt,vx,vy,score,label]");
    TORCH_CHECK(iou_threshold >= 0.0 && iou_threshold <= 1.0, "IoU threshold must be in [0, 1]");
    
    int n_boxes = boxes.size(0);
    if (n_boxes == 0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(boxes.device()));
    }
    
    // Move boxes to CPU for processing
    auto boxes_cpu = boxes.cpu();
    auto boxes_accessor = boxes_cpu.accessor<float, 2>();
    
    // Convert to BoundingBox format
    std::vector<BoundingBox> bbox_list(n_boxes);
    for (int i = 0; i < n_boxes; i++) {
        bbox_list[i].x = boxes_accessor[i][0];
        bbox_list[i].y = boxes_accessor[i][1];
        bbox_list[i].z = boxes_accessor[i][2];
        bbox_list[i].w = boxes_accessor[i][3];
        bbox_list[i].l = boxes_accessor[i][4];
        bbox_list[i].h = boxes_accessor[i][5];
        bbox_list[i].rt = boxes_accessor[i][6];
        bbox_list[i].vx = boxes_accessor[i][7];
        bbox_list[i].vy = boxes_accessor[i][8];
        bbox_list[i].score = boxes_accessor[i][9];
        bbox_list[i].label = boxes_accessor[i][10];
    }
    
    // Call CUDA NMS function
    std::vector<int> keep = nms_cpu(bbox_list, iou_threshold);
    
    // Convert result to tensor
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(boxes.device());
    auto result = torch::zeros({(int)keep.size()}, options);
    
    if (!keep.empty()) {
        auto result_cpu = torch::from_blob(keep.data(), {(int)keep.size()}, torch::TensorOptions().dtype(torch::kInt32));
        result.copy_(result_cpu.to(torch::kInt64));
    }
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nms_forward, "NMS forward");
}