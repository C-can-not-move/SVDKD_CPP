// window 控制台显示utf-8字符
#ifdef _WIN32
#include <windows.h>
#endif

#include "SVDKD_C++.h"

namespace
{
    namespace fs = std::filesystem;

    int g_fail = 0;

    struct Ref
    {
        std::string   name;
        torch::Tensor tensor;
        bool          is_weight = false;
    };

    std::vector<int64_t> parse_shape(const std::string& s)
    {
        std::vector<int64_t> out;
        std::stringstream ss(s);
        std::string part;
        while (std::getline(ss, part, 'x'))
        {
            if (!part.empty())
            {
                out.push_back(std::stoll(part));
            }
        }
        return out;
    }

    // 读 all.txt + all.bin。
    // 返回的张量都是对同一个 buf 的【视图】—— buf 必须活得比它们久。
    std::vector<Ref> load_refs(const fs::path& dir, std::vector<char>& buf)
    {
        std::ifstream idx((dir / "all.txt").string());
        std::ifstream blob((dir / "all.bin").string(), std::ios::binary);
        if (!idx || !blob)
        {
            throw std::runtime_error("cannot open all.txt / all.bin under " + dir.string());
        }

        blob.seekg(0, std::ios::end);
        const auto n = static_cast<size_t>(blob.tellg());
        blob.seekg(0, std::ios::beg);
        buf.resize(n);
        blob.read(buf.data(), static_cast<std::streamsize>(n));

        std::vector<Ref> refs;
        std::string line;
        while (std::getline(idx, line))
        {
            if (line.empty())
            {
                continue;
            }
            std::stringstream ls(line);

            std::string name;
            std::string dtype;
            std::string shape;
            int64_t offset = 0;
            int64_t numel = 0;
            ls >> name >> dtype >> shape >> offset >> numel;

            const bool    is_i64 = (dtype == "torch.int64");
            const auto    dt = is_i64 ? torch::kInt64 : torch::kFloat32;
            const int64_t bytes = numel * (is_i64 ? 8 : 4);

            if (offset < 0 || offset + bytes > static_cast<int64_t>(buf.size()))
            {
                throw std::runtime_error("all.bin too short for " + name);
            }

            Ref r;
            r.name = name;
            r.is_weight = (name.rfind("__", 0) != 0);
            r.tensor = torch::from_blob(buf.data() + offset, parse_shape(shape), dt);
            refs.push_back(std::move(r));
        }
        return refs;
    }

    void report(const std::string& name, const torch::Tensor& got, const torch::Tensor& want)
    {
        std::cout << name << "  ";

        if (!got.sizes().equals(want.sizes()))
        {
            ++g_fail;
            std::cout << "[FAIL] SHAPE   got " << got.sizes()
                << "   want " << want.sizes() << "\n";
            return;
        }

        const auto   g = got.detach().to(torch::kFloat64);
        const auto   w = want.detach().to(torch::kFloat64);
        const double maxerr = (g - w).abs().max().item<double>();
        const double scale = std::max(1e-30, w.abs().max().item<double>());
        const double rel = maxerr / scale;

        const bool ok = (rel < 1e-6);
        if (!ok)
        {
            ++g_fail;
        }
        std::cout << (ok ? "[ ok ]" : "[FAIL]")
            << "   max|err| = " << maxerr
            << "   rel = " << rel
            << "   scale = " << scale << "\n";
    }
}  // namespace

int  main(int argc, char** argv)
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);   // 65001，输出
	SetConsoleCP(CP_UTF8);         // 输入（读中文时要）
#endif

    namespace fs = std::filesystem;
    using namespace svdkd::model;

    const fs::path refs = (argc > 1)
        ? fs::path(argv[1])
        : fs::path(R"(D:\DATA\CODE\C++\SVDKD_C++\test_tools\refs)");

    try
    {
        torch::NoGradGuard no_grad;

        std::cout << "refs = " << refs.string() << "\n";

        std::vector<char> buf;                 // ★ 生命周期必须长过所有 tensor
        const auto refs_t = load_refs(refs, buf);
        std::cout << "loaded " << refs_t.size() << " reference tensors\n";

        std::unordered_map<std::string, torch::Tensor> by_name;
        for (const auto& r : refs_t)
        {
            by_name.emplace(r.name, r.tensor);
        }

        // ================= 层 0 / 层 1：结构 =================
        auto m = h_resnet10(/*num_classes=*/10);
        m->eval();                             // ★ BN 要用 running stats

        const int64_t np = static_cast<int64_t>(m->named_parameters().size());
        const int64_t nb = static_cast<int64_t>(m->named_buffers().size());
        std::cout << "\n[层0] params = " << np << "   buffers = " << nb
            << "   合计 = " << np + nb << "   (期望 38 + 36 = 74)\n";
        if (np + nb != 74)
        {
            ++g_fail;
        }

        int64_t numel = 0;
        for (const auto& p : m->parameters())
        {
            numel += p.numel();
        }
        std::cout << "[层1] params numel = " << numel << "   (期望 4,910,922)\n";
        if (numel != 4910922)
        {
            ++g_fail;
        }

        // ================= 层 3a：装权重（逐项，能报错）=================
        std::unordered_map<std::string, torch::Tensor> mine;
        for (auto& p : m->named_parameters())
        {
            mine.emplace(p.key(), p.value());
        }
        for (auto& b : m->named_buffers())
        {
            mine.emplace(b.key(), b.value());
        }

        int64_t loaded = 0;
        for (auto& [name, target] : mine)
        {
            const auto it = by_name.find(name);
            if (it == by_name.end())
            {
                continue;
            }
            if (target.scalar_type() != it->second.scalar_type())
            {
                throw std::runtime_error("dtype mismatch for " + name);
            }
            target.copy_(it->second.reshape(target.sizes()));     // ★ 按【目标形状】对齐
            ++loaded;
        }

        std::cout << "\n[层3a] weights loaded = " << loaded << " / " << mine.size()
            << "   (期望 74)\n";
        if (loaded != static_cast<int64_t>(mine.size()))
        {
            ++g_fail;
        }

        for (const auto& [name, t] : mine)
        {
            if (by_name.find(name) == by_name.end())
            {
                std::cout << "    MISSING (ref 里没有): " << name << "\n";
            }
        }
        for (const auto& r : refs_t)
        {
            if (r.is_weight && mine.find(r.name) == mine.end())
            {
                std::cout << "    EXTRA   (模型里没有): " << r.name << "\n";
            }
        }

        // ================= 层 2 / 层 3c：前向并比对 =================

        const auto x = by_name.at("__x");
        std::cout << "\nx " << x.sizes() << "   " << x.scalar_type() << "\n";

        const auto out = m->forward(x);

        std::cout << "\n[层3c]\n";
        report("logits", out.logits, by_name.at("__logits"));
        report("h1    ", out.h1, by_name.at("__h1"));
        report("h2    ", out.h2, by_name.at("__h2"));
        report("h3    ", out.h3, by_name.at("__h3"));
    }
    catch (const std::exception& e)
    {
        std::cout << "\nEXCEPTION: " << e.what() << "\n";
        return 2;
    }

    std::cout << "\n" << (g_fail == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_fail))
        << "\n";
    return (g_fail == 0) ? 0 : 1;

}