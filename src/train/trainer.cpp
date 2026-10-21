#include "train/trainer.h"

#include "data/transforms.h"
#include "model/checkpoint.h"
#include "kd/my_svd.h"
#include "kd/loss.h"

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
		TORCH_CHECK(train_images.size(0) == train_labels.size(0), "train: images / labels count mismatch");
		TORCH_CHECK(train_images.scalar_type() == torch::kUInt8, "train: expect uint8 images, got ", train_images.scalar_type());
		TORCH_CHECK(train_labels.scalar_type() == torch::kInt64, "train: expect int64 labels, got ", train_labels.scalar_type());
		TORCH_CHECK(cfg.batch_size > 0, "train: batch_size must be > 0");
		TORCH_CHECK(train_images.device() == device,"train_classifier: train_images 在 ", train_images.device()," 而模型在 ", device, " —— index_select 要求 index 与被索引张量同设备");
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

				const auto idx = perm.narrow(0, start, len).to(device);// idx 装的是「编号」—— 不是「位置」

				const auto xb = train_images.index_select(0, idx);     // 已在 device 上（main 里整批驻留）
				const auto yb = train_labels.index_select(0, idx).to(device);

				const auto xf = transform_train(xb, norm, cfg.out_size, device);

				// 纯净训练循环：前向 ⇒ loss ⇒ 反向 ⇒ 更新
				optimizer.zero_grad();
				const auto loss = torch::nn::functional::cross_entropy(model->forward(xf).logits, yb);// ⭐这里默认是Kmean均值
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
			const EvalResult eval_result = evaluate(model, val_images, val_labels, cfg);
			const double acc = eval_result.acc();

			res.final_val_acc = acc;
			if (acc > res.best_val_acc)
			{
				res.best_val_acc = acc;
				res.best_epoch = epoch;

				if (cfg.save_path.has_value())
				{
					svdkd::model::save_checkpoint(model, *cfg.save_path);
				}
			}

			std::cout << "[epoch " << epoch << "/" << cfg.epochs << "]"
				<< "   train_loss " << res.last_train_loss
				<< "   val_acc " << acc
				<< "   (" << eval_result.correct << "/" << eval_result.total << ")\n";

		}

		return res;
	}

	TrainResult train_distill(svdkd::model::ResNet student,
		const torch::Tensor& train_images,
		const torch::Tensor& train_labels,
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer,
		const TrainConfig& cfg,
		DistillOpts Dopts)
	{
		using namespace svdkd::data;

		const auto device = student->parameters().front().device();

		// ── 前置检查 ──
		TORCH_CHECK(!Dopts.teacher.is_empty(), "train_distill: teacher 为空");
		TORCH_CHECK(!Dopts.teacher->is_training(),"train_distill: teacher 必须处于 eval() 模式（BN 用 running stats）—— 由调用者设置");
		TORCH_CHECK(Dopts.teacher->parameters().front().device() == device,"train_distill: teacher 在 ", Dopts.teacher->parameters().front().device()," 而 student 在 ", device);

		TORCH_CHECK(train_images.dim() == 4, "train_distill: expect (N, C, H, W)");
		TORCH_CHECK(train_images.size(0) == train_labels.size(0), "train_distill: images / labels count mismatch");
		TORCH_CHECK(train_images.device() == device, "train_distill: train_images 在 ", train_images.device(), " 而模型在 ", device, " —— index_select 要求同设备");
		TORCH_CHECK(train_images.scalar_type() == torch::kUInt8, "train_distill: expect uint8 images");
		TORCH_CHECK(train_labels.scalar_type() == torch::kInt64, "train_distill: expect int64 labels");
		TORCH_CHECK(cfg.batch_size > 0, "train_distill: batch_size must be > 0");
		

		// 优化器必须抓的是学生的参数，不是教师的
		{
			std::unordered_set<const void*> owned;
			for (const auto& g : optimizer.param_groups())
				for (const auto& t : g.params())
					owned.insert(t.data_ptr());

			for (const auto& p : student->parameters())
				TORCH_CHECK(owned.count(p.data_ptr()) != 0,"train_distill: 优化器抓的不是这个学生的参数");
		}

		const auto norm = imagenet_norm();

		TrainResult res;
		const int64_t N = train_images.size(0);

		for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch)
		{
			student->train();          // 教师不受影响（两个独立的 impl）

			const auto perm = torch::randperm(N, torch::kInt64);

			double  running = 0.0;
			int64_t nb = 0;

			for (int64_t start = 0; start < N; start += cfg.batch_size)
			{
				const int64_t len = std::min(cfg.batch_size, N - start);
				if (cfg.drop_last && len < cfg.batch_size) 
					break;

				const auto idx = perm.narrow(0, start, len).to(device);
				const auto xb = train_images.index_select(0, idx);
				const auto yb = train_labels.index_select(0, idx).to(device);

				// 师生必须吃同一批转换后的图 —— 所以 xf 只算一次
				const auto xf = transform_train(xb, norm, cfg.out_size, device);


				// ── 训练开始 ──
				optimizer.zero_grad();

				const auto s_out = student->forward(xf);

				// ⭐⭐⭐⭐⭐教师前向：必须单独包一层 NoGradGuard⭐⭐⭐⭐⭐
				/*
				* libtorch 的 autograd 是每个张量记住"我是谁造的"
				* 在跟学生比较的时候记录了，所以教师的前向必须在 no_grad 里
				* 合并点就是"师生张量第一次进同一个二元运算"那一行 
				* -kl_batchmean 里 log_softmax(stu/T) 和 softmax(tea/T) 相减（在 kl_div 内部）、以及 mse_loss(s_hid, t_hid) 相减。
				* 从那一刻起，total.grad_fn 的连通图里同时含学生和教师的节点
				* 不过其实对训练没有什么影响，因为教师的参数不需要梯度，反向时不会更新教师的参数。
				* 但是每步都白算一遍教师模型的反向，浪费时间。
				* 这段就是python里面的 with torch.no_grad(): teacher(x) 语义
				*/
				torch::Tensor t_logits, t_h3;
				{
					torch::NoGradGuard ng;// 构造 → grad 从这一刻关掉                   
					const auto t_out = Dopts.teacher->forward(xf);// 这一段不建图
					t_logits = t_out.logits;
					t_h3 = t_out.h3;                      
				}// 出作用域 → ng 析构 → grad 恢复

				const auto sv = svdkd::kd::my_svd(t_h3, s_out.h3);      // 教师在前

				const auto parts = svdkd::kd::svdkd_loss(
					s_out.logits, 
					t_logits,
					sv.s_s, 
					sv.s_t,
					yb, 
					Dopts.T, 
					Dopts.lambda_stu);

				parts.total.backward();// ⭐进backward()的东西都是标量
				optimizer.step();
				// ── 训练结束 ──

				running += parts.total.item<double>();
				++nb;

				if (nb <= 3)      // 添加个观测点
				{
					std::cout << "  [kd] ce=" << parts.ce.item<double>()
						<< "  teacher=" << parts.teacher.item<double>()
						<< "  prop=" << parts.prop.item<double>()
						<< "  delta=" << parts.delta.item<double>()
						<< "  total=" << parts.total.item<double>() << '\n';
				}

				if (cfg.log_every > 0 && (nb % cfg.log_every) == 0)
				{
					std::cout << "  epoch " << epoch << "  batch " << nb
						<< "  avg loss " << (running / static_cast<double>(nb)) << '\n';
				}
			}

			res.last_train_loss = (nb > 0) ? running / static_cast<double>(nb) : 0.0;

			const EvalResult eval_result = evaluate(student, val_images, val_labels, cfg);
			const double acc = eval_result.acc();

			res.final_val_acc = acc;
			if (acc > res.best_val_acc)
			{
				res.best_val_acc = acc;
				res.best_epoch = epoch;
				if (cfg.save_path.has_value())
					svdkd::model::save_checkpoint(student, *cfg.save_path);
			}

			std::cout << "[epoch " << epoch << "/" << cfg.epochs << "]"
				<< "   train_loss " << res.last_train_loss
				<< "   val_acc " << acc
				<< "   (" << eval_result.correct << "/" << eval_result.total << ")" << std::endl;
		}

		return res;
	}

	EvalResult evaluate(svdkd::model::ResNet model,
		const torch::Tensor& images,
		const torch::Tensor& labels,
		const TrainConfig& cfg)
	{
		TORCH_CHECK(cfg.batch_size > 0, "evaluate: batch_size must be > 0");
		TORCH_CHECK(images.size(0) == labels.size(0), "evaluate: images / labels count mismatch");
		
		const auto device = model->parameters().front().device();
		const auto norm = svdkd::data::imagenet_norm();
		torch::NoGradGuard no_grad;
		const bool was_training = model->is_training();   // is_training() 是 const
		model->eval();                       // 注意：会改变共享 impl 的模式

		int64_t correct = 0, total = 0;
		for (int64_t s = 0; s < images.size(0); s += cfg.batch_size)
		{
			const int64_t len = std::min(cfg.batch_size, images.size(0) - s);
			const auto xb = images.narrow(0, s, len);
			const auto yb = labels.narrow(0, s, len).to(device);
			const auto xf = svdkd::data::transform_val(xb, norm, cfg.resize_to, cfg.out_size, device);

			const auto pred = model->forward(xf).logits.argmax(1);
			correct += pred.eq(yb).sum().item<int64_t>();
			total += len;
		}
		if (was_training) 
			model->train();                 // 恢复调用者进来时的状态
		EvalResult res;
		res.correct = correct;
		res.total = total;
		return res;
	}
}  // namespace svdkd::train