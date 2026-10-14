// SVDKD_C++ —— 冒烟测试
// 目的：一次构建验证整条链（找到库 → 链接 → DLL → CUDA 运行期），
//       并确认 SVD-KD 依赖的两个算子行为正确。

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>   // CUDA build 才有这个头

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

    int g_fail = 0;

    void section(const char* t) {
        std::cout << "\n=== " << t << " ===\n";
    }

    void check(bool ok, const std::string& what, const std::string& detail = "") {
        std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << what;
        if (!detail.empty()) std::cout << "    " << detail;
        std::cout << '\n';
        if (!ok) ++g_fail;
    }

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::cout << std::setprecision(10);

    // ---------------------------------------------------------------- 1
    section("1. 版本 / 构建信息");
    std::cout << "  TORCH_VERSION : " << TORCH_VERSION << '\n';

    // ---------------------------------------------------------------- 2
    section("2. CUDA 运行期");
    const bool cuda = torch::cuda::is_available();
    std::cout << "  is_available  : " << (cuda ? "yes" : "no") << '\n';
    std::cout << "  device_count  : "
        << static_cast<int>(torch::cuda::device_count()) << '\n';
    if (cuda) {
        auto* p = at::cuda::getDeviceProperties(0);
        std::cout << "  device 0      : " << p->name
            << "   sm_" << p->major << p->minor << '\n';
    }
    check(cuda, "CUDA 可用（本机 1660 Ti / 服务器 4090 都应 yes）");

    // ---------------------------------------------------------------- 3
    section("3. 解析对照：shear  A = [[1,1],[0,1]]");
    // sigma = [phi, 1/phi], 乘积 = |det A| = 1
    auto A = torch::tensor({ {1.0, 1.0}, {0.0, 1.0} }, torch::kFloat64);
    auto s2 = at::linalg_svdvals(A);
    std::cout << "  sigma         : " << s2[0].item<double>()
        << " , " << s2[1].item<double>() << '\n'
        << "  product       : " << (s2[0] * s2[1]).item<double>() << '\n';
    check(std::abs(s2[0].item<double>() - 1.6180339887) < 1e-9,
        "sigma[0] == phi", "expect 1.6180339887");
    check(std::abs((s2[0] * s2[1]).item<double>() - 1.0) < 1e-12,
        "sigma[0]*sigma[1] == |det| == 1");

    // ---------------------------------------------------------------- 4
    section("4. ★ 批处理语义：4D 张量上的 SVD");
    // 论文代码假设：SVD 作用在【最后两维】，批过 (B,C)
    auto X = torch::randn({ 2, 3, 4, 6 });
    auto svd = at::linalg_svd(X, /*full_matrices=*/false);
    auto& U4 = std::get<0>(svd);
    auto& S4 = std::get<1>(svd);
    auto& Vh4 = std::get<2>(svd);
    std::cout << "  input (2,3,4,6)\n"
        << "    U  " << U4.sizes() << '\n'
        << "    S  " << S4.sizes() << "     <- 期望 [2,3,4] = (B,C,min(H,W))\n"
        << "    Vh " << Vh4.sizes() << '\n';
    check(S4.dim() == 3 && S4.size(0) == 2 && S4.size(1) == 3 && S4.size(2) == 4,
        "S 的形状是 (B,C,min(H,W))");
    auto rec = U4.matmul(S4.diag_embed()).matmul(Vh4);
    double rerr = (rec - X).abs().max().item<double>();
    check(rerr < 1e-5, "U·diag(S)·Vh == X", "max|err| = " + std::to_string(rerr));

    // ---------------------------------------------------------------- 5
    section("5. ★★ 反向：d(sum sigma)/dA == U @ Vh ?");
    // 这是唯一还没在你的工具链里实测过的东西
    auto M = torch::randn({ 7, 5 }, torch::kFloat64).requires_grad_(true);
    auto svd2 = at::linalg_svd(M, false);
    auto& U2 = std::get<0>(svd2);
    auto& S2 = std::get<1>(svd2);
    auto& Vh2 = std::get<2>(svd2);
    S2.sum().backward();
    auto expected = U2.detach().matmul(Vh2.detach());
    double gerr = (M.grad() - expected).abs().max().item<double>();
    std::cout << "  max|err|      : " << gerr << '\n';
    check(gerr < 1e-10, "svdvals 反向可用，且 == U @ Vh",
        "Python 那边测出来是 0.000e+00");
    
    // ---- 加在第 5 节里（U2 / S2 / Vh2 就在上面）----
    {
        const int64_t k = U2.size(-1);                     // = 5，跟着 U2 走
        const auto    Id = torch::eye(k, U2.options());     // ★ 跟 U2 同 dtype 同 device
        const auto    G = U2.transpose(-2, -1).matmul(U2); // (k,k)

        std::cout << "  U2 " << U2.sizes()
            << "   UᵀU " << G.sizes()
            << "   Id " << Id.sizes() << '\n';        // ★ 把 shape 打出来

        const double e = (G - Id).abs().max().item<double>();
        check(e < 1e-12, "UᵀU == I_k  (thin SVD 的 U 是列正交)",
            "max|err| = " + std::to_string(e));
    }
    

    // ---------------------------------------------------------------- 6
    section("6. 在 GPU 上真的执行（本机 sm_75 / 服务器 sm_89）");
    if (cuda) {
        auto xg = torch::randn({ 64, 256, 2, 2 },
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto sg = at::linalg_svdvals(xg);
        check(sg.size(0) == 64 && sg.size(1) == 256 && sg.size(2) == 2,
            "cuda svdvals 形状 (64,256,2)");
        check(std::isfinite(sg.sum().item<float>()), "cuda svdvals 结果有限");

        auto xr = xg.clone().detach().requires_grad_(true);
        at::linalg_svdvals(xr).sum().backward();
        check(std::isfinite(xr.grad().abs().sum().item<float>()),
            "cuda 反向结果有限（无 NaN）");

        // 每步 .item() 强制同步 -> 测的是真实单次延迟
        const int N = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            auto r = at::linalg_svdvals(xg);
            (void)r.sum().item<float>();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        std::cout << "  GPU (64,256,2,2) svdvals : " << ms << " ms/call"
            << "   (Python/cu132 那边是 14.672)\n";
    }

    // ----------------------------------------------------------------
    std::cout << "\n";
    if (g_fail == 0) {
        std::cout << "ALL CHECKS PASSED\n";
    }
    else {
        std::cout << "*** " << g_fail << " CHECK(S) FAILED ***\n";
    }
    return g_fail ? 1 : 0;
}