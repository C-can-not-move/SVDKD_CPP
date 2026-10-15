#include "kd/my_svd.h"

#include <tuple>

namespace svdkd::kd
{

    namespace
    {
        // my_svd.py 里的 compute_svd：SVD处理最后两维，再把 sigma 展平。
		// (B,C,H,W)-> (B,C,min(H,W)) -> (B,C*min(H,W))
        // .flatten(1)就是最后的操作
        torch::Tensor sigma_flat(const torch::Tensor& x)
        {
            return std::get<1>(torch::linalg_svd(x, /*full_matrices=*/false)).flatten(1);
        }

        // 把通道太多的教师砍到和学生对上
		// 输入(B, C_T, H, W)为教师特征图,Keep为学生通道数
        torch::Tensor select_channels(const torch::Tensor& x, int64_t keep)
        {
			const auto s = std::get<1>(torch::linalg_svd(x, false));// (B, C_T, min(H,W))
            const auto scores = s.sum(2);// (B,C_T)核范数
            const auto q = torch::flip(torch::argsort(scores, static_cast < int64_t>(1)), { 1 });// q[b, j] = 「第 b 个样本里，第 j 重要的通道编号」

            const int64_t B = x.size(0);
            const int64_t C_T = x.size(1);
            const int64_t H = x.size(2);
            const int64_t W = x.size(3);
                
            const auto idx = q.view({ B, C_T, 1, 1 }).expand({ B, C_T, H, W });
            //gather(B, C_T, H, W)      通道按 q 重排
            //narrow(B, keep, H, W)     只留前 keep 个
            return x.gather(1, idx).narrow(1, 0, keep);
        }
    }  // namespace

    MySvdOut my_svd(const torch::Tensor& a, const torch::Tensor& b)
    {
        TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "expect (B, C, H, W)");
        TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,"expect float32");

        const int64_t t_c = a.size(1);
        const int64_t t_h = a.size(2);
        const int64_t s_c = b.size(1);
        const int64_t s_h = b.size(2);

        MySvdOut out;

        if (t_c == s_c && t_h == s_h)                  // 分支 1：两边一样，什么都不做
        {
            out.s_t = sigma_flat(a);
            out.s_s = sigma_flat(b);
        }
        else if (t_c == s_c && t_h != s_h)             // 分支 2：只有宽高不同
        {
            out.s_t = std::get<1>(torch::linalg_svd(a, false)).slice(2, 0, s_h).flatten(1);
            out.s_s = sigma_flat(b);
        }
        else if (t_c != s_c && t_h == s_h)             // 分支 3：只有通道不同
        {
            out.s_t = sigma_flat(select_channels(a, s_c));
            out.s_s = sigma_flat(b);
        }
        else                                           // 分支 4：都不同
        {
            const auto tea = select_channels(a, s_c);
            out.s_t = std::get<1>(torch::linalg_svd(tea, false)).slice(2, 0, s_h).flatten(1);
            out.s_s = sigma_flat(b);
        }
        return out;
    }

}  // namespace svdkd::kd