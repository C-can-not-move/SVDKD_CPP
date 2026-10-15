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

	// 读取数据测试
	const std::string data_dir = "D:/DATA/CODE/C++/SVDKD_C++/dataset/cifar-10-batches-bin";
	auto raw = svdkd::data::load_cifar10_bin(data_dir, /*train=*/true);

	// ---------------- 形状 ----------------
	std::cout << "images : " << raw.images.sizes() << "\n";   // [50000, 3, 32, 32]
	std::cout << "labels : " << raw.labels.sizes() << "\n";   // [50000]

	// ScalarType 是 enum class，转 int 一定安全：0=Byte 4=Long 6=Float 7=Double
	std::cout << "dtype  : images=" << static_cast<int>(raw.images.scalar_type())
		<< " (应为 0)   labels=" << static_cast<int>(raw.labels.scalar_type())
		<< " (应为 4)\n\n";

	// ---------------- 数据前几项 ----------------
	std::cout << "labels[0..7] =\n" << raw.labels.narrow(0, 0, 8) << "\n\n";

	// 第 0 张图的 R 面，左上角 2 行 x 8 列
	std::cout << "img[0].R[0..1, 0..7] =\n"
		<< raw.images[0][0].narrow(0, 0, 2).narrow(1, 0, 8) << "\n\n";

	// 第 0 张图在 (0,0) 那个像素的 R/G/B 三个值
	std::cout << "img[0].pixel(0,0) RGB = "
		<< raw.images[0].select(1, 0).select(1, 0) << "\n\n";

	// ---------------- 逐字节对拍 ----------------
	// 文件行布局：[0]=label  [1..1024]=R 面  [1025..2048]=G 面  [2049..3072]=B 面
	// 第 1 张图从第 3073 字节开始
	std::cout << "--- 和 data_batch_1.bin 的前几个字节对拍 ---\n";
	std::cout << "byte[0]     (label0)  == labels[0]           -> "
		<< raw.labels[0].item<int64_t>() << "\n";
	std::cout << "byte[1..15] (R 前 15) == img[0].R 前 15 个     -> "
		<< raw.images[0][0].flatten().narrow(0, 0, 15) << "\n";
	std::cout << "byte[1025]  (G 第 1)  == img[0].G[0][0]       -> "
		<< raw.images[0][1][0][0].item<int64_t>() << "\n";
	std::cout << "byte[2049]  (B 第 1)  == img[0].B[0][0]       -> "
		<< raw.images[0][2][0][0].item<int64_t>() << "\n";
	std::cout << "byte[3073]  (label1)  == labels[1]           -> "
		<< raw.labels[1].item<int64_t>() << "\n";

	return 0;
}