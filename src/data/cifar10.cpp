# include "data/cifar10.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svdkd::data
{
    /*
    * 每一个batch文件包含10000行数据。
	* 一行3073字节。标签1字节，像素3072（3*32*32）字节。
    */
    constexpr int64_t kImagesPerFile = 10000;
    constexpr int64_t kRowBytes = 3073;     // 1 + 3072
    constexpr int64_t kPixelBytes = 3072;   // 3 * 32 * 32

    std::pair<torch::Tensor, torch::Tensor> load_one_batch(const std::filesystem::path& file)
    {
        const int64_t expected = kImagesPerFile * kRowBytes;

        std::ifstream in(file, std::ios::binary | std::ios::ate);
        if (!in)
        {
            throw std::runtime_error("load_cifar10_bin: cannot open " + file.string());
        }
        in.seekg(0, std::ios::beg);

        std::vector<uint8_t> buf(static_cast<size_t>(expected));
        if (!in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expected)))
        {
            throw std::runtime_error("load_cifar10_bin: short read on " + file.string());
        }

        // torch::from_blob不接管内存。clone()之后张量才持有数据，
        // buf在函数返回时析构就安全了。
        torch::Tensor rows = torch::from_blob(
            buf.data(),
            { kImagesPerFile, kRowBytes },
            torch::kUInt8
        ).clone();

        // 第0列是标签。select(1,0)直接降掉那一维。
        torch::Tensor labels = rows.select(1, 0).to(torch::kInt64);// (10000,)

        // 后3072列是像素。
        // 注意：narrow 出来的行步长仍是 3073 —— 【不连续】，
        //       所以这里必须 reshape（会拷贝一份），写成 view 会直接抛错。
        torch::Tensor images = rows.narrow(1, 1, kPixelBytes)
            .reshape({ kImagesPerFile, 3, 32, 32 });

        return { images, labels };
    }

	Cifar10Raw load_cifar10_bin(const std::string& dir, bool train)
	{
        const std::filesystem::path root(dir);

        std::vector<std::string> names{ "test_batch.bin" };
        if (train)
        {
            names = { "data_batch_1.bin", 
                      "data_batch_2.bin", 
                      "data_batch_3.bin",
                      "data_batch_4.bin", 
                      "data_batch_5.bin" };
        }

        std::vector<torch::Tensor> images;
        std::vector<torch::Tensor> labels;
        images.reserve(names.size());
        labels.reserve(names.size());

        for (const std::string& name : names)
        {
            auto [img, lab] = load_one_batch(root / name);
            images.push_back(img);
            labels.push_back(lab);
        }

        Cifar10Raw out;
        out.images = torch::cat(images,/*dim=*/0);   // (50000 or 10000, 3, 32, 32) uint8
        out.labels = torch::cat(labels,/*dim=*/0);   // (N,) int64
        return out;
	}
}
