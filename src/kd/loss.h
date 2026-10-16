#pragma once

#include <torch/torch.h>

namespace svdkd::kd
{
    // 一次前向里的四项损失 + 总和。
    // 四项都是【标量张量】(0 维)，不是 double —— 因为 total 要进 backward()。
    struct LossParts
    {
        torch::Tensor ce;        // 交叉熵:学生 logits vs 标签 (交叉熵CrossEntropy)
        torch::Tensor teacher;   // 蒸馏项:学生 logits vs 教师 logits (KL散度)
        torch::Tensor prop;      // 特征蒸馏项:学生 h vs 教师 h (KL散度)
        torch::Tensor delta;     // 正则项Δ:奇异值特征λ差的 MSE*超参(4*(1-lambda_stu)*MSE(features))
        torch::Tensor total;     // 四项之和 —— 只有它需要 backward
    };

    // KL 散度，reduction显式指定为batchmean。默认reduction是kmean
    // 传入(原始logits)即没使用softmax运算过的值，函数内部有log_softmax / softmax。
    torch::Tensor kl_batchmean(
        const torch::Tensor& logits_stu,
        const torch::Tensor& logits_tea,
        double temperature);

    // SVD-KD 的总损失：L = CE + T^2*KL(logits) + T^2*KL(features) + 4*(1-lambda_stu)*MSE(features)
    // 4是§4.2 的 beta = 4
    LossParts svdkd_loss(
        const torch::Tensor& logits_stu,
        const torch::Tensor& logits_tea,
        const torch::Tensor& feat_stu,/* 学生中间层特征 */
        const torch::Tensor& feat_tea,/* 教师中间层特征 */
        const torch::Tensor& labels,
        double temperature,
        double lambda_stu);

}  // namespace svdkd::kd