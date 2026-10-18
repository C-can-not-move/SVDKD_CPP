#include "train/trainer.h"

#include "data/transforms.h"

#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace svdkd::train
{
    TrainResult train_classifier(
        svdkd::model::ResNet model,
        const torch::Tensor& train_images,
        const torch::Tensor& train_labels,
        const torch::Tensor& val_images,
        const torch::Tensor& val_labels,
        torch::optim::Optimizer& optimizer,
        const TrainConfig& cfg)
    {
        using namespace svdkd::data;

        // 设备跟着模型走
        const auto device = model->parameters().front().device();

        TORCH_CHECK(train_images.dim() == 4, "train: expect (N, C, H, W)");
        TORCH_CHECK(train_images.size(0) == train_labels.size(0),"train: images / labels count mismatch");
        TORCH_CHECK(train_images.scalar_type() == torch::kUInt8,"train: expect uint8 images, got ", train_images.scalar_type());
        TORCH_CHECK(train_labels.scalar_type() == torch::kInt64,"train: expect int64 labels, got ", train_labels.scalar_type());
        TORCH_CHECK(cfg.batch_size > 0, "train: batch_size must be > 0");
        // 优化器必须抓的是这个模型的参数
        // 写不出 Tensor<float>，只能写 Tensor + scalar_type() 在运行时问它。类型没有消失，它是被搬到了运行期。
        {
            std::unordered_set<const void*> owned;
            for (const auto& g : optimizer.param_groups())
            {
                for (const auto& t : g.params())
                {
                    owned.insert(t.data_ptr());
                }
            }

            for (const auto& p : model->parameters())
            {
                TORCH_CHECK(owned.count(p.data_ptr()) != 0,
                    "train_classifier: 优化器抓的不是这个模型的参数 —— "
                    "要在模型构造完成之后再建优化器，参数取 model->parameters()");
            }
        }
        const auto norm = imagenet_norm();

        TrainResult res;
        const int64_t N = train_images.size(0);

        for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch)
        {
            // ── 训练 ──  BN 要用批统计 ⇒ train()
            model->train();

            const auto perm = torch::randperm(N, torch::kInt64);   // 洗牌就是这一行

            double  running = 0.0;
            int64_t nb = 0;

			for (int64_t start = 0; start < N; start += cfg.batch_size)// 一个 epoch 里，按 batch_size 切 N 张图
            {
				const int64_t len = std::min(cfg.batch_size, N - start);// 最后一批可能不够 batch_size
                if (cfg.drop_last && len < cfg.batch_size)
                {
                    break;
                }

                const auto idx = perm.narrow(0, start, len);// idx 装的是「编号」—— 不是「位置」

                const auto xb = train_images.index_select(0, idx);   // // 仍是 CPU uint8 (len,3,32,32)
                const auto yb = train_labels.index_select(0, idx).to(device);

                const auto xf = transform_train(xb, norm, cfg.out_size).to(device);;

				// 纯净训练循环：前向 ⇒ loss ⇒ 反向 ⇒ 更新
                optimizer.zero_grad();
                const auto loss = torch::nn::functional::cross_entropy(model->forward(xf).logits, yb);
                loss.backward();
                optimizer.step();
				// 纯净训练循环结束

                running += loss.item<double>();
                ++nb;

                if (cfg.log_every > 0 && (nb % cfg.log_every) == 0)
                {
                    std::cout << "  epoch " << epoch << "  batch " << nb
                        << "  avg loss " << (running / static_cast<double>(nb)) << "\n";
                }
            }

            /* 一轮epoch结束 */
            res.last_train_loss = (nb > 0) ? running / static_cast<double>(nb) : 0.0;

            // ── 验证 ──  BN 要用 running stats ⇒ eval()；且不要建计算图 ⇒ NoGradGuard
            model->eval();
            {
                torch::NoGradGuard no_grad;

                int64_t correct = 0;
                int64_t total = 0;
                const int64_t VN = val_images.size(0);

                for (int64_t start = 0; start < VN; start += cfg.batch_size)
                {
                    const int64_t len = std::min(cfg.batch_size, VN - start);

                    const auto xb = val_images.narrow(0, start, len);
                    const auto yb = val_labels.narrow(0, start, len).to(device);

                    const auto xf = transform_val(xb, norm, cfg.resize_to, cfg.out_size).to(device);
                    const auto pred = model->forward(xf).logits.argmax(1);

                    correct += pred.eq(yb).sum().item<int64_t>();
                    total += len;
                }

                const double acc = (total > 0)? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
                if (acc > res.best_val_acc)
                {
                    res.best_val_acc = acc;
                }

                std::cout << "[epoch " << epoch << "/" << cfg.epochs << "]"
                    << "   train_loss " << res.last_train_loss
                    << "   val_acc " << acc
                    << "   (" << correct << "/" << total << ")\n";
            }
        }

        return res;
    }

}  // namespace svdkd::train