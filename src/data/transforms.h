// src/data/transforms.h
#pragma once

#include <torch/torch.h>

#include <vector>

namespace svdkd::data
{
    // 归一化参数。ImageNet。
    struct NormStats
    {
        std::vector<float> mean;   // 每通道一个（CHW 里的 C）
        std::vector<float> std;
    };
    // Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    NormStats imagenet_norm();

    // 训练
    // RandomResizedCrop(out_size) → RandomHorizontalFlip() → ToTensor → Normalize
    // 入：uint8   (N, 3, 32, 32) 出：float32 (N, 3, out_size, out_size)
    torch::Tensor transform_train(
        const torch::Tensor& images,
        const NormStats& norm,
        int64_t out_size = 224,
        torch::Device device);

    // 验证
    // Resize(resize_to) → CenterCrop(out_size) → ToTensor → Normalize
    // 入：uint8   (N, 3, 32, 32)      出：float32 (N, 3, out_size, out_size)
    torch::Tensor transform_val(
        const torch::Tensor& images,
        const NormStats& norm,
        int64_t resize_to = 256,
        int64_t out_size = 224,
        torch::Device device);

}  // namespace svdkd::data