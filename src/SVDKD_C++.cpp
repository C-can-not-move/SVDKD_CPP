// window 控制台显示 utf-8
#ifdef _WIN32
#include <windows.h>
#endif

#include "SVDKD_C++.h"

int main()
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif

	try
	{
		namespace fs = std::filesystem;
		const auto rc = svdkd::cfg::defaults();

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

		// 只切一小撮
		const int64_t kTrain = 640;
		const int64_t kVal = 320;
		train.images = train.images.narrow(0, 0, kTrain);
		train.labels = train.labels.narrow(0, 0, kTrain);
		val.images = val.images.narrow(0, 0, kVal);
		val.labels = val.labels.narrow(0, 0, kVal);

		std::cout << "[train] train " << train.images.sizes() << " " << train.images.scalar_type()
			<< "   val " << val.images.sizes() << std::endl;

		// 顺序：先模型，再优化器。
		auto net = svdkd::model::h_resnet10(/*num_classes=*/10);
		net->to(device);// 先搬模型，再建优化器 —— 优化器的构建依赖于模型参数。
		torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4));

		svdkd::train::TrainConfig cfg;
		cfg.epochs = 100;
		cfg.batch_size = 32;
		cfg.log_every = 5;
		cfg.save_path = rc.out_dir + "/resnet10-cifar10.pt";

		const auto r = svdkd::train::train_classifier(
			net,
			train.images,
			train.labels,
			val.images,
			val.labels,
			optimizer,
			cfg);
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