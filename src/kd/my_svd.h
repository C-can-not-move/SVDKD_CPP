#pragma once

#include <torch/torch.h>

namespace svdkd::kd
{
    // 师生各自的特征图 -> 两条展平的 sigma 向量，形状都是 (B, C * min(H,W))。
    struct MySvdOut
    {
        torch::Tensor s_t;   // 教师
        torch::Tensor s_s;   // 学生
    };

    MySvdOut my_svd(const torch::Tensor& a, const torch::Tensor& b);

}  // namespace svdkd::kd