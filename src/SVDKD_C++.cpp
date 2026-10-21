// window 控制台显示 utf-8
#ifdef _WIN32
#include <windows.h>
#endif

#include "SVDKD_C++.h"

int main(int argc, char** argv)
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif

	try
	{
		namespace fs = std::filesystem;

		auto rc = svdkd::cfg::defaults(); // 需要控制台传入GPU设备号，改成控制台传入的方式
		if (argc > 1)
			rc.device_index = std::stoi(argv[1]);  // 只认第一个参数：用哪张卡

		const fs::path data_dir = rc.data_dir;
		fs::create_directories(rc.out_dir);
		torch::manual_seed(rc.seed);

		const auto device = (rc.device_index >= 0 && torch::cuda::is_available())
			? torch::Device(torch::kCUDA, rc.device_index)
			: torch::Device(torch::kCPU);

		std::cout << "[cfg] data_dir = " << rc.data_dir << "\n"
			<< "[cfg] out_dir  = " << rc.out_dir << "\n"
			<< "[cfg] device   = " << device
			<< "  (requested " << rc.device_index << ")\n"
			<< "[cfg] seed     = " << rc.seed << std::endl;

		std::cout << "[train] loading" << std::endl;

		auto train = svdkd::data::load_cifar10_bin(data_dir.string(), /*train=*/true);
		auto val = svdkd::data::load_cifar10_bin(data_dir.string(), /*train=*/false);

		std::cout << "[train] train " << train.images.sizes() << " " << train.images.scalar_type()
			<< "   val " << val.images.sizes() << std::endl;
		const auto train_x = train.images.to(device);   // 50000×3×32×32 uint8 = 153.6 MB
		const auto train_y = train.labels.to(device);   // 400 KB
		const auto val_x = val.images.to(device);     // 10000 张 = 30.7 MB
		const auto val_y = val.labels.to(device);     // 80 KB

		// 顺序：先模型，再优化器。
		// ── 教师：读权重 → 上设备 → eval（只做一次，train_distill 只检查不修改）──
		svdkd::train::DistillOpts diss;
		diss.teacher = svdkd::model::h_resnet34(/*num_classes=*/10);
		svdkd::model::load_checkpoint(diss.teacher, rc.out_dir + "/resnet34-cifar10.pt");
		diss.teacher->to(device);
		diss.teacher->eval();

		auto net = svdkd::model::h_resnet10(/*num_classes=*/10);
		// auto net = svdkd::model::h_resnet34(/*num_classes=*/10);
		net->to(device);// 先搬模型，再建优化器 —— 优化器的构建依赖于模型参数。

		torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));

		svdkd::train::TrainConfig cfg;
		cfg.epochs = 200;
		cfg.batch_size = 128;
		cfg.log_every = 100;
		cfg.save_path = rc.out_dir + "/resnet34-cifar10-distill.pt";

		cfg.print();
		// lr 不属于 TrainConfig：它由调用者决定，train_classifier 收到的是优化器本身。
		// 从优化器读回来打，日志里的值就不可能和实际生效的值不一致。
		std::cout << "[opt] lr = " << optimizer.param_groups().front().options().get_lr() << '\n';

		/*
		const auto r = svdkd::train::train_classifier(
			net,
			train_x,
			train_y,
			val_x,
			val_y,
			optimizer,
			cfg);
		*/

		const auto r = svdkd::train::train_distill(
			net, 
			train_x, 
			train_y, 
			val_x, 
			val_y, 
			optimizer, 
			cfg, 
			diss);

		// 内存里现在是【最后一轮】，不是最好的。
		// 要真的用上 best，得把它读回来 —— 否则「存了最好的」等于没存。
		if (cfg.save_path.has_value() && r.best_epoch > 0)
		{
			svdkd::model::load_checkpoint(net, *cfg.save_path);
			std::cout << "[ckpt] restored best (epoch " << r.best_epoch << ")\n";
		}

		// 恢复后核一次：同一份权重 + 同一份验证集 ⇒ 应该精确等于 best_val_acc。
		{
			torch::NoGradGuard no_grad;
			net->eval();

			const auto norm_v = svdkd::data::imagenet_norm();
			int64_t correct = 0, total = 0;

			for (int64_t s = 0; s < val_x.size(0); s += cfg.batch_size)
			{
				const int64_t len = std::min(cfg.batch_size, val_x.size(0) - s);

				const auto xb = val_x.narrow(0, s, len);
				const auto yb = val_y.narrow(0, s, len);
				const auto xf = svdkd::data::transform_val(xb, norm_v, cfg.resize_to, cfg.out_size, device);

				const auto pred = net->forward(xf).logits.argmax(1);
				correct += pred.eq(yb).sum().item<int64_t>();
				total += len;
			}

			std::cout << "[ckpt] verify after restore: " << correct << "/" << total
				<< " = " << static_cast<double>(correct) / total
				<< "   (应 == " << r.best_val_acc << ")\n";
		}

		r.print();

		return 0;
	}
	catch (const c10::Error& e)
	{
		std::cout << "\n[c10::Error]\n" << e.what() << std::endl;
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cout << "\n[std::exception]\n" << e.what() << std::endl;
		return 1;
	}

	return 0;
}