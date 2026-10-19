#include "model/checkpoint.h"

#include <torch/serialize.h>

#include <iostream>

namespace svdkd::model
{
    void save_checkpoint(const ResNet& model, const std::string& path)
    {
        torch::serialize::OutputArchive archive;

        size_t n = 0;
        for (auto& p : model->named_parameters())
        {
            archive.write(p.key(), p.value().to(torch::kCPU));   // 统一搬到 CPU 再写 ⇒ 文件与设备无关
            ++n;
        }
        for (auto& b : model->named_buffers())
        {
            archive.write(b.key(), b.value().to(torch::kCPU));
            ++n;
        }

        archive.save_to(path);
        std::cout << "[ckpt] saved " << n << " tensors -> " << path << '\n';
    }

    size_t load_checkpoint(ResNet& model, const std::string& path)
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);

        // copy_ 进【叶子张量】必须关掉 autograd，否则报
        //   "a leaf Variable that requires grad is being used in an in-place operation"
        torch::NoGradGuard no_grad;

        size_t n = 0;
        for (auto& p : model->named_parameters())
        {
            torch::Tensor src;
            archive.read(p.key(), src);                              // 缺键会抛
            p.value().copy_(src.reshape(p.value().sizes()));         // 按目标形状对齐（0 维那个坑）
            ++n;
        }
        for (auto& b : model->named_buffers())
        {
            torch::Tensor src;
            archive.read(b.key(), src);
            b.value().copy_(src.reshape(b.value().sizes()));
            ++n;
        }

        std::cout << "[ckpt] loaded " << n << " tensors <- " << path << '\n';
        return n;
    }

}  // namespace svdkd::model