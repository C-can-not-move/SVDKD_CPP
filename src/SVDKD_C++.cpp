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
		const auto device = torch::cuda::is_available()? torch::Device(torch::kCUDA, 0): torch::Device(torch::kCPU);
		std::cout << "[device] " << device << std::endl;

		namespace fs = std::filesystem;
		torch::manual_seed(2026);

		const fs::path data_dir = R"(D:\DATA\CODE\C++\SVDKD_C++\dataset\cifar-10-batches-bin)";

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
		cfg.epochs = 2;
		cfg.batch_size = 32;
		cfg.log_every = 5;

		const auto r = svdkd::train::train_classifier(
			net,
			train.images,
			train.labels,
			val.images,
			val.labels,
			optimizer,
			cfg);
		std::cout << "\nbest_val_acc = " << r.best_val_acc << "    last_train_loss = " << r.last_train_loss << std::endl;
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