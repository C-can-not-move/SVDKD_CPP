// src/train/trainer.h
#pragma once

#include <torch/torch.h>
#include "model/resnet.h"

#include <cstdint>

namespace svdkd::train
{
	struct TrainConfig
	{
		int64_t epochs = 3;            // 代码中是1
		int64_t batch_size = 32;       // 按代码（论文写 128）
		double  lr = 1e-4;             // 按代码（论文写 1e-3）
		int64_t out_size = 224;        // 照抄
		int64_t resize_to = 256;       // 照抄（验证链）
		bool    drop_last = false;     // 照抄 DataLoader 的默认（最后一批留着）
		int64_t log_every = 50;        // 每多少批打一行
	};

	struct TrainResult
	{
		double best_val_acc = 0.0;
		double last_train_loss = 0.0;
	};

	// 从头训一个分类器。
	// model按值传是有意的：ResNet 是 ModuleHolder，内部是 shared_ptr ⇒拷贝共享同一份实现 ⇒ 函数里的训练会改到调用者的模型。
	// 数据是整批张量，函数内部自己洗牌切片（torch::randperm + index_select），不用 DataLoader。
	TrainResult train_classifier(svdkd::model::ResNet model,
		const torch::Tensor& train_images,   // uint8 (N,3,32,32)
		const torch::Tensor& train_labels,   // int64 (N)
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		const TrainConfig& cfg);
}  // namespace svdkd::train