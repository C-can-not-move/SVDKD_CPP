#include "data/transforms.h"

namespace svdkd::data
{
	namespace
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
		torch::Tensor channel_tensor(const std::vector<float>& v, const torch::TensorOptions& opt)
		{
			auto t = torch::empty({ 1, static_cast<int64_t>(v.size()), 1, 1 }, opt);
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
			TORCH_CHECK(norm.mean.size() == norm.std.size(), "normalize: mean/std length mismatch");
			TORCH_CHECK(static_cast<int64_t>(norm.mean.size()) == x.size(-3), "normalize: channel count mismatch");

			const auto mean = channel_tensor(norm.mean, x.options());
			const auto std = channel_tensor(norm.std, x.options());
			return x.sub(mean).div(std);
		}

		// torchvision F.resize
		// 短边 = target，长边 = int(target * long / short)，这里使用的是截断——三种取整方式之一。
		std::pair<int64_t, int64_t> resized_hw(int64_t h, int64_t w, int64_t target)
		{
			// 先判断短边是 h 还是 w
			const bool w_le_h = (w <= h);
			const int64_t short_side = w_le_h ? w : h;
			const int64_t long_side = w_le_h ? h : w;

			const int64_t new_short = target;
			const int64_t new_long = static_cast<int64_t>(
				static_cast<double>(target) * static_cast<double>(long_side) / static_cast<double>(short_side));

			const int64_t new_w = w_le_h ? new_short : new_long;
			const int64_t new_h = w_le_h ? new_long : new_short;
			return { new_h, new_w };
		}

		// Resize(target)：短边对齐到 target。
		torch::Tensor resize_shorter_side(const torch::Tensor& x, int64_t target)
		{
			const auto hw = resized_hw(x.size(-2), x.size(-1), target);

			return torch::nn::functional::interpolate(
				x,
				torch::nn::functional::InterpolateFuncOptions()
				.size(std::vector<int64_t>({ hw.first, hw.second }))
				.mode(torch::kBilinear)
				.align_corners(false));
		}
		// torchvision 的默认值
		constexpr double kScaleMin = 0.08;
		constexpr double kScaleMax = 1.0;
		constexpr double kRatioMin = 3.0 / 4.0;
		constexpr double kRatioMax = 4.0 / 3.0;
		constexpr float  kFlipP = 0.5f;

		struct CropBox { int64_t i, j, h, w; };   // i = top, j = left

		// ai还是太好用了:)
		// torchvision RandomResizedCrop.get_params（transforms.py L928-970）的抄本。
		// 最多重试 10 次 —— 抽到的 area / aspect 放不进原图就重抽。
		// w/h 的取整是 round，而 Python 的 round 是【银行家舍入】⇒ 使用std::nearbyint
		// 10 次都不成 → 兜底中央裁；那两处用的是 //2(向下取整)
		// 返回的是裁剪框的左上角坐标 + 高宽。即索引
		CropBox random_crop_box(int64_t height, int64_t width)
		{
			const double area = static_cast<double>(height) * static_cast<double>(width);
			const double log_lo = std::log(kRatioMin);
			const double log_hi = std::log(kRatioMax);

			for (int attempt = 0; attempt < 10; ++attempt)
			{
				const double target_area = area * torch::empty({ 1 }).uniform_(kScaleMin, kScaleMax).item<float>();
				const double aspect = std::exp(torch::empty({ 1 }).uniform_(log_lo, log_hi).item<float>());

				const int64_t w = static_cast<int64_t>(std::nearbyint(std::sqrt(target_area * aspect)));
				const int64_t h = static_cast<int64_t>(std::nearbyint(std::sqrt(target_area / aspect)));
					
				if (w > 0 && w <= width && h > 0 && h <= height)
				{
					const int64_t i = torch::randint(0, height - h + 1, { 1 }).item<int64_t>();
					const int64_t j = torch::randint(0, width - w + 1, { 1 }).item<int64_t>();
					return { i, j, h, w };
				}
			}

			// 兜底：中央裁
			const double in_ratio = static_cast<double>(width) / static_cast<double>(height);
			int64_t h = 0;
			int64_t w = 0;
			if (in_ratio < kRatioMin)
			{
				w = width;
				h = static_cast<int64_t>(std::nearbyint(static_cast<double>(w) / kRatioMin));
			}
			else if (in_ratio > kRatioMax)
			{
				h = height;
				w = static_cast<int64_t>(std::nearbyint(static_cast<double>(h) * kRatioMax));
			}
			else
			{
				w = width;
				h = height;
			}
			return { (height - h) / 2, (width - w) / 2, h, w };
		}

		// 单张图的 crop + resize
		torch::Tensor crop_resize(const torch::Tensor& one, const CropBox& box, int64_t out_size)
		{
			const auto cropped = one.narrow(-2, box.i, box.h).narrow(-1, box.j, box.w);

			return torch::nn::functional::interpolate(
				cropped,
				torch::nn::functional::InterpolateFuncOptions()
				.size(std::vector<int64_t>({ out_size, out_size }))
				.mode(torch::kBilinear)
				.align_corners(false));
		}

		// torchvision F.center_crop（L556-594);
		// crop_top  = int(round((H - crop_h) / 2.0));
		// crop_left = int(round((W - crop_w) / 2.0));
		// round 是 Python 的 round = 银行家舍入（half-to-even);
		// C++ 的 std::round 是 half-away-from-zero 所以使用 std::nearbyint;
		torch::Tensor center_crop(const torch::Tensor& x, int64_t crop_h, int64_t crop_w)
		{
			const int64_t h = x.size(-2);
			const int64_t w = x.size(-1);

			TORCH_CHECK(crop_h <= h && crop_w <= w, "center_crop: crop larger than image (needs padding, not implemented)");

			const int64_t top = static_cast<int64_t>(std::nearbyint((h - crop_h) / 2.0));
			const int64_t left = static_cast<int64_t>(std::nearbyint((w - crop_w) / 2.0));

			return x.narrow(-2, top, crop_h).narrow(-1, left, crop_w);
		}
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
	// 训练
	// RandomResizedCrop(out_size) -> RandomHorizontalFlip -> ToTensor -> Normalize
	torch::Tensor transform_train(
		const torch::Tensor& images,
		const NormStats& norm,
		int64_t out_size,
		torch::Device device)
	{
		TORCH_CHECK(images.dim() == 4, "transform_train: expect (N, C, H, W)");
		TORCH_CHECK(images.scalar_type() == torch::kUInt8,"transform_train: expect uint8 input, got ", images.scalar_type());
		TORCH_CHECK(images.size(1) == static_cast<int64_t>(norm.mean.size()),"transform_train: channel count mismatch");

		const int64_t N = images.size(0);
		const int64_t C = images.size(1);
		const int64_t H = images.size(2);
		const int64_t W = images.size(3);

		// 先整批转 float。crop 是纯索引 ⇒ 先转后转【结果完全相同】
		// 唯一的原因是 resize 不接受整数张量。
		const auto x = to_float01(images.to(device));

		auto out = torch::empty({ N, C, out_size, out_size } , x.options());

		// 逐样本循环，而不是整批共用一组随机数。
		// torchvision 那边是「一张图走完整条 Compose 再走下一张」，所以
		// 「crop 抽两下 + 翻转抽一下」这个【消耗顺序】也必须是逐样本的。
		// 把翻转放进循环里，也正是为了这个顺序。
		for (int64_t n = 0; n < N; ++n)
		{
			const auto box = random_crop_box(H, W);

			// 旧的错误用法：auto one = crop_resize(x.select(0, n), box, out_size);
			// ★ narrow 保留维度 ⇒ (1, C, H, W)。
			//   select 会降成 (C,H,W)，而 bilinear 的 interpolate 只收 4 维。
		    auto one = crop_resize(x.narrow(0, n, 1), box, out_size);
			if (torch::rand({ 1 }).item<float>() < kFlipP)
			{
				// 旧的错误用法：one = torch::flip(one, { 2 });      // (C, S, S) 的最后一位是宽度
				one = torch::flip(one, { -1 });     // 最后一维 = 宽度（与秩无关）
			}

			// 这里可以放心用 copy_：out 是刚 empty 出来的普通张量，没有requires_grad，不会碰「leaf 张量不能原地改」那条检查。
			// 旧的错误用法：out.select(0, n).copy_(one);
			out.narrow(0, n, 1).copy_(one);
		}

		return normalize(out, norm);
	}

	// 验证
	// Resize(resize_to) -> CenterCrop(out_size) -> ToTensor -> Normalize
	torch::Tensor transform_val(
		const torch::Tensor& images,
		const NormStats& norm,
		int64_t resize_to,
		int64_t out_size,
		torch::Device device)
	{
		TORCH_CHECK(images.dim() == 4, "transform_val: expect (N, C, H, W)");
		TORCH_CHECK(images.scalar_type() == torch::kUInt8, "transform_val: expect uint8 input, got ", images.scalar_type());
		TORCH_CHECK(images.size(1) == static_cast<int64_t>(norm.mean.size()), "transform_val: channel count mismatch");

		// 顺序上于torchvision的处理不同,libtorch的interpolate不接受整数张量
		auto x = to_float01(images.to(device));       // ToTensor 的那一半
		x = resize_shorter_side(x, resize_to);        // Resize(resize_to)
		x = center_crop(x, out_size, out_size);       // CenterCrop(out_size)
		return normalize(x, norm);                    // Normalize
	}
}  // namespace svdkd::data