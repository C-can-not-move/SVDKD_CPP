#include "model/resnet.h"

#include <stdexcept>
#include <utility>

namespace svdkd::model
{

	// ------------------------------------------------------------------------
	// make_layer — dis_model.py L102-115
	// ------------------------------------------------------------------------
	torch::nn::Sequential make_layer(
		int64_t& in_channel, 
		int64_t channel,
		int64_t block_num, 
		int64_t stride)
	{
		const int64_t out_channel = channel * BasicBlockImpl::kExpansion;

		torch::nn::Sequential downsample{ nullptr };
		if (stride != 1 || in_channel != out_channel)
		{
			downsample = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel,out_channel,/*kernel_size=*/1).stride(stride).bias(false)),
				torch::nn::BatchNorm2d(out_channel));
		}

		torch::nn::Sequential layers;
		// push_back 自动命名 "0","1","2"。
		// 所以键会是 layer1.0.conv1.weight 这种形式。
		layers->push_back(BasicBlock(in_channel, channel, stride, downsample));

		in_channel = out_channel;   // 副作用,修改外层的 in_channel，供下一层使用

		for (int64_t i = 1; i < block_num; ++i)
		{
			layers->push_back(BasicBlock(in_channel, channel));
		}
		return layers;
	}

	// ------------------------------------------------------------------------
	// BasicBlockImpl
	// ------------------------------------------------------------------------
	BasicBlockImpl::BasicBlockImpl(
		int64_t in_channel, 
		int64_t out_channel,
		int64_t stride,
		torch::nn::Sequential downsample)
	{
		this->conv1 = register_module("conv1", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(in_channel, out_channel, /*kernel_size=*/3).stride(stride).padding(1).bias(false)));
		this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channel));

		this->conv2 = register_module("conv2", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(out_channel, out_channel, /*kernel_size=*/3).stride(1).padding(1).bias(false)));
		this->bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channel));

		this->relu = register_module("relu", torch::nn::ReLU());// 默认就是 inplace = false 

		if (!downsample.is_empty())
		{
			this->downsample = register_module("downsample", downsample);
		}
	}

	torch::Tensor BasicBlockImpl::forward(torch::Tensor x)
	{
		torch::Tensor identity = x;
		if (!downsample.is_empty())
		{
			identity = downsample->forward(x);
		}
		// torch的结构树里面块是无顺序的，具体的网络结构在forward里面体现。relu层是无参数模块，所以可以复用同一个实例。bn层是有参数的，所以每个block都要有自己的bn层。
		// 这里就是一个标准的残差块
		auto out = relu->forward(bn1->forward(conv1->forward(x)));
		out = bn2->forward(conv2->forward(out));
		out = out + identity;
		out = relu->forward(out);
		return out;
	}

	// ------------------------------------------------------------------------
	// ResNetImpl
	// ------------------------------------------------------------------------
	ResNetImpl::ResNetImpl(std::vector<int64_t> blocks_num, int64_t num_classes)
	{
		TORCH_CHECK(blocks_num.size() == 4, "blocks_num must have exactly 4 entries");

		this->conv1 = register_module("conv1", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(/*in=*/3, /*out=*/64, /*kernel_size=*/7).stride(2).padding(3).bias(false)));
		this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
		this->relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
		this->maxpool = register_module("maxpool", torch::nn::MaxPool2d(
			torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1)));

		// 通道数沿这条链往下传：64 → 64 → 128 → 256 → 512
		int64_t in_channel = 64;
		this->layer1 = register_module("layer1", make_layer(in_channel, 64, blocks_num[0], /*stride=*/1));
		this->layer2 = register_module("layer2", make_layer(in_channel, 128, blocks_num[1], /*stride=*/2));
		this->layer3 = register_module("layer3", make_layer(in_channel, 256, blocks_num[2], /*stride=*/2));
		this->layer4 = register_module("layer4", make_layer(in_channel, 512, blocks_num[3], /*stride=*/2));

		this->avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
			torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));
		this->fc = register_module("fc", torch::nn::Linear(512, num_classes));

		// named_parameters() 递归进已注册的子模块(register_module)
		// 凡是 4 维的参数，用 kaiming_normal 原地填一遍，在这个构造函数里面就是conv
		// '_'这个后缀，一般代表原地修改
		for (auto& item : named_parameters())
		{
			if (item.value().dim() == 4)
			{
				torch::nn::init::kaiming_normal_(item.value(), /*a=*/0,torch::kFanOut, torch::kReLU);
			}
		}
	}

	ResNetOut ResNetImpl::forward(torch::Tensor x)
	{
		x = maxpool->forward(relu->forward(bn1->forward(conv1->forward(x))));

		ResNetOut out;
		out.h1 = layer1->forward(x);
		out.h2 = layer2->forward(out.h1);
		out.h3 = layer3->forward(out.h2);

		auto h4 = layer4->forward(out.h3);
		out.logits = fc->forward(torch::flatten(avgpool->forward(h4), 1));
		return out;
	}

	// ------------------------------------------------------------------------
	// 工厂 —— 各一行
	// ------------------------------------------------------------------------
	ResNet h_resnet10(int64_t num_classes)
	{
		return ResNet(std::vector<int64_t>{ 1, 1, 1, 1 }, num_classes);
	}

	ResNet h_resnet18(int64_t num_classes)
	{
		return ResNet(std::vector<int64_t>{ 2, 2, 2, 2 }, num_classes);
	}

	ResNet h_resnet34(int64_t num_classes)
	{
		return ResNet(std::vector<int64_t>{ 3, 4, 6, 3 }, num_classes);
	}

}  // namespace svdkd::model

