// src/train/trainer.h
#pragma once

#include <torch/torch.h>
#include <torch/optim.h>

#include "model/resnet.h"

#include <cstdint>
#include <optional>
#include <iostream>
#include <string>

namespace svdkd::train
{
	struct TrainConfig
	{
		int64_t epochs = 20;           // 原论文代码中是1
		int64_t batch_size = 32;       // 按代码（论文写 128）
		int64_t out_size = 224;        // 照抄
		int64_t resize_to = 256;       // 照抄（验证链）
		bool    drop_last = false;     // 照抄 DataLoader 的默认（最后一批留着）
		int64_t log_every = 50;        // 每多少批打一行
		std::optional<std::string> save_path;  // 有值就保存模型

		void print() const             // 新增一个打印函数
		{
			std::cout << "[train-cfg] epochs     = " << epochs << '\n'
				<< "[train-cfg] batch_size = " << batch_size << '\n'
				<< "[train-cfg] out_size   = " << out_size << '\n'
				<< "[train-cfg] resize_to  = " << resize_to << '\n'
				<< "[train-cfg] drop_last  = " << (drop_last ? "true" : "false") << '\n'
				<< "[train-cfg] log_every  = " << log_every << '\n'
				<< "[train-cfg] save_path  = " << (save_path ? save_path->c_str() : "(none)") << '\n';
		}             
	};

	struct EvalResult
	{
		int64_t correct = 0;
		int64_t total = 0;

		double acc() const
		{
			return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
		}
	};

	struct TrainResult
	{
		double  best_val_acc = 0.0;   // 最好那一次的验证准确率
		int64_t best_epoch = 0;       // 它出现在第几轮 —— 收敛判据：== epochs 说明还没收敛
		double  final_val_acc = 0.0;  // 最后一轮 —— 和 best 差得远 = 在震荡
		double  last_train_loss = 0.0;

		void print() const
		{
			std::cout << "best_val_acc = " << best_val_acc
				<< "  (epoch " << best_epoch << ")"
				<< "   final_val_acc = " << final_val_acc
				<< "   last_train_loss = " << last_train_loss << '\n';
		}

	};

	// 蒸馏怎么配置。用于整理训练函数。
	struct DistillOpts
	{
		svdkd::model::ResNet teacher;
		double T = 5.0;
		double lambda_stu = 0.0;
	};

	// 从头训一个分类器。
	// model按值传是有意的：ResNet 是 ModuleHolder，内部是 shared_ptr ⇒拷贝共享同一份实现 ⇒ 函数里的训练会改到调用者的模型。
	// 数据是整批张量，函数内部自己洗牌切片（torch::randperm + index_select），不用 DataLoader。
	TrainResult train_classifier(svdkd::model::ResNet model,
		const torch::Tensor& train_images,   // uint8 (N,3,32,32)
		const torch::Tensor& train_labels,   // int64 (N)
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer, 
		const TrainConfig& cfg);

	// 蒸馏训练。学生模型按值传，教师模型按引用传。
	TrainResult train_distill(svdkd::model::ResNet student,
		const torch::Tensor& train_images,
		const torch::Tensor& train_labels,
		const torch::Tensor& val_images,
		const torch::Tensor& val_labels,
		torch::optim::Optimizer& optimizer,
		const TrainConfig& cfg,
		DistillOpts Dopts);

	// 评估一个分类器在验证集上的准确率。没有训练循环。
	EvalResult evaluate(svdkd::model::ResNet model,          // 按值：holder 共享 impl
		const torch::Tensor& images,
		const torch::Tensor& labels,
		const TrainConfig& cfg);
}  // namespace svdkd::train