#include "data/transforms.h"

namespace svdkd::data
{
	/*
	torchvision 的 ToTensor 做两件事：
	① HWC -> CHW      ② uint8 -> float 并除以 255
	这里的C++数据集就是CHW 的 uint8 张量。
	这里只剩 ②。
	*/
	torch::Tensor to_float01(const torch::Tensor& x)
	{
		return x.to(torch::kFloat32).div(255.0);
	}

	// 把 (C,) 那组参数摊成 (1, C, 1, 1)，然后按照通道广播。
	torch::Tensor channel_tensor(const std::vector<float>& v)
	{
		auto t = torch::empty({ 1, static_cast<int64_t>(v.size()), 1, 1 },torch::kFloat32);
		for (int64_t c = 0; c < t.size(1); ++c)
		{
			t.select(1, c).fill_(v[static_cast<size_t>(c)]);
		}
		return t;
	}

	// Normalize：out[c] = (in[c] - mean[c]) / std[c]
	// torchvision 最终落到 _functional_tensor.normalize，是这个式子。
	torch::Tensor normalize(const torch::Tensor& x, const NormStats& norm)
	{
		TORCH_CHECK(x.is_floating_point(), "normalize: expect a float tensor");
		TORCH_CHECK(x.dim() >= 3, "normalize: expect at least (C, H, W)");
		TORCH_CHECK(norm.mean.size() == norm.std.size(),"normalize: mean/std length mismatch");
		TORCH_CHECK(static_cast<int64_t>(norm.mean.size()) == x.size(-3),"normalize: channel count mismatch");

		const auto mean = channel_tensor(norm.mean);
		const auto std = channel_tensor(norm.std);
		return x.sub(mean).div(std);
	}

	// dataload_cf100.py L11：
	// Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	NormStats imagenet_norm()
	{
		return NormStats{
			{0.485f, 0.456f, 0.406f},
			{0.229f, 0.224f, 0.225f},
		};
	}

}  // namespace svdkd::data