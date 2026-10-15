// src/data/cifar10.h
#pragma once
#include <torch/torch.h>
#include <string>

namespace svdkd::data 
{

    struct Cifar10Raw 
    {
        torch::Tensor images;   // (N, 3, 32, 32) uint8
        torch::Tensor labels;   // (N)            int64
    };

    Cifar10Raw load_cifar10_bin(const std::string& dir, bool train);

}  // namespace svdkd::data