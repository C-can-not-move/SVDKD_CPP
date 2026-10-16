#include "kd/loss.h"

namespace svdkd::kd
{
    torch::Tensor kl_batchmean(
        const torch::Tensor& logits_stu,
        const torch::Tensor& logits_tea,
        double temperature)
    {
        TORCH_CHECK(logits_stu.dim() >= 2 && logits_stu.sizes().equals(logits_tea.sizes()),"kl_batchmean: expect same shape with dim >= 2");

        const auto x = torch::log_softmax(logits_stu / temperature, /*dim=*/1);
        const auto y = torch::softmax(logits_tea / temperature, /*dim=*/1);

        return torch::nn::functional::kl_div(
            x, 
            y,
            torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)
        );
    }

    LossParts svdkd_loss(
        const torch::Tensor& logits_stu,
        const torch::Tensor& logits_tea,
        const torch::Tensor& feat_stu,
        const torch::Tensor& feat_tea,
        const torch::Tensor& labels,
        double temperature,
        double lambda_stu)
    {
        TORCH_CHECK(feat_stu.sizes().equals(feat_tea.sizes()),"svdkd_loss: student/teacher feature shape mismatch");

        const auto ce = torch::nn::functional::cross_entropy(logits_stu, labels);

        const auto scale = temperature * temperature;

        const auto teacher = kl_batchmean(logits_stu, logits_tea, temperature) * scale;
        const auto prop = kl_batchmean(feat_stu, feat_tea, temperature) * scale;

        const auto delta =torch::nn::functional::mse_loss(
            feat_stu, 
            feat_tea,
            torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean)
        )* (4.0 * (1.0 - lambda_stu));

        LossParts out;
        out.ce = ce;
        out.teacher = teacher;
        out.prop = prop;
        out.delta = delta;
        out.total = ce + teacher + prop + delta;
        return out;
    }

}  // namespace svdkd::kd