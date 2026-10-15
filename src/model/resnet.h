#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace svdkd::model
{

    // ------------------------------------------------------------------------
    // BasicBlock — dis_model.py L5-35，expansion == 1
	// Impl是 Module 的实现类，TORCH_MODULE(BasicBlock) 会生成一个持有 Impl 的类。特殊宏运算需要在 Impl 里写。
    // CIFAR-10 的 resnet10/18/34 全部由它组成。
    // Bottleneck（resnet50/101）暂时不需要。
    // ------------------------------------------------------------------------
    class BasicBlockImpl : public torch::nn::Module
    {
    public:
        static constexpr int64_t kExpansion = 1;

		// 构造函数
        BasicBlockImpl(
            int64_t in_channel, 
            int64_t out_channel,
            int64_t stride = 1,
            torch::nn::Sequential downsample = nullptr);

        // 残差相加后再过 ReLU ⇒ 取出来的 h1/h2/h3 是【post-ReLU】
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d      conv1{ nullptr };   // 3x3, stride=stride, pad=1, no bias
        torch::nn::BatchNorm2d bn1{ nullptr };
        torch::nn::Conv2d      conv2{ nullptr };   // 3x3, stride=1,      pad=1, no bias
        torch::nn::BatchNorm2d bn2{ nullptr };
        torch::nn::ReLU        relu{ nullptr };    // 官方此处【没有】inplace
        torch::nn::Sequential  downsample{ nullptr };
    };
    TORCH_MODULE(BasicBlock);
	// 为resnet50和101留的Bottleneck暂时不需要
     
    
    // ------------------------------------------------------------------------
    // ResNet — dis_model.py L79-139
    // ------------------------------------------------------------------------
    // 用来返回各层的输出，方便 SVD-KD 取 h3。
    struct ResNetOut
    {
        torch::Tensor logits;   // (N, num_classes)
        torch::Tensor h1;       // layer1 出参  (N,  64, 56, 56)  @ 224x224
        torch::Tensor h2;       // layer2 出参  (N, 128, 28, 28)
        torch::Tensor h3;       // layer3 出参  (N, 256, 14, 14)  ← SVD-KD 用这个
    };

    class ResNetImpl : public torch::nn::Module
    {
    public:
        /* blocks_num 是运行时参数，不是模板参数
        * {1,1,1,1} = ResNet-10    {2,2,2,2} = ResNet-18    {3,4,6,3} = ResNet-34
        * 单参数构造函数一律走explicit
        */
        explicit ResNetImpl(std::vector<int64_t> blocks_num, int64_t num_classes = 10);

        ResNetOut forward(torch::Tensor x);

    private:
        // ImageNet 式 stem：7x7/s2 + maxpool/s2。有它，输入就必须是 224。
        torch::nn::Conv2d            conv1{ nullptr };
        torch::nn::BatchNorm2d       bn1{ nullptr };
        torch::nn::ReLU              relu{ nullptr };      // stem 这个【是】inplace
        torch::nn::MaxPool2d         maxpool{ nullptr };
        torch::nn::Sequential        layer1{ nullptr };
        torch::nn::Sequential        layer2{ nullptr };
        torch::nn::Sequential        layer3{ nullptr };
        torch::nn::Sequential        layer4{ nullptr };
        torch::nn::AdaptiveAvgPool2d avgpool{ nullptr };
        torch::nn::Linear            fc{ nullptr };
    };
    TORCH_MODULE(ResNet);

    ResNet h_resnet10(int64_t num_classes = 10);
    ResNet h_resnet18(int64_t num_classes = 10);
    ResNet h_resnet34(int64_t num_classes = 10);

}  // namespace svdkd::model