// window 控制台显示utf-8字符
#ifdef _WIN32
#include <windows.h>
#endif

#include "SVDKD_C++.h"

int  main()
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);   // 65001，输出
	SetConsoleCP(CP_UTF8);         // 输入（读中文时要）
#endif

    using namespace svdkd::model;

    // ================= 层 0：结构（键数） =================
    auto m = h_resnet10(/*num_classes=*/10);

    const int64_t np = m->named_parameters().size();
    const int64_t nb = m->named_buffers().size();
    std::cout << "params  = " << np << "   (期望 38)\n";
    std::cout << "buffers = " << nb << "   (期望 36)\n";
    std::cout << "合计    = " << np + nb << "   (期望 74)\n\n";

    // ================= 层 1：参数量 =================
    int64_t n10 = 0;
    for (const auto& p : m->parameters()) { n10 += p.numel(); }
    std::cout << "nc=10  params = " << n10 << "   (期望 4,910,922)\n";

    auto m100 = h_resnet10(/*num_classes=*/100);
    int64_t n100 = 0;
    for (const auto& p : m100->parameters()) { n100 += p.numel(); }
    std::cout << "nc=100 params = " << n100 << "   (期望 4,957,092  ← 论文 L2290)\n\n";

    // ================= 层 2：前向形状 =================
    m->eval();                                  // ← 必须
    auto x = torch::randn({ 2, 3, 224, 224 });

    torch::NoGradGuard no_grad;
    auto out = m->forward(x);
    std::cout << "logits " << out.logits.sizes() << "   (期望 [2, 10])\n";
    std::cout << "h1     " << out.h1.sizes() << "   (期望 [2, 64, 56, 56])\n";
    std::cout << "h2     " << out.h2.sizes() << "   (期望 [2, 128, 28, 28])\n";
    std::cout << "h3     " << out.h3.sizes() << "   (期望 [2, 256, 14, 14])\n";

    return 0;

}