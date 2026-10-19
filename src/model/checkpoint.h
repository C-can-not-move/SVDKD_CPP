#pragma once

#include <string>

#include "model/resnet.h"

namespace svdkd::model
{
	// 把模型的参数+buffer存成扁平的archive 键 -> 张量。
	//  不用 Module::save —— 那个是嵌套的。
	//  named_buffers() 必须一起存：BN 的 running_mean / running_var /num_batches_tracked 都在那里。
	//  只存参数的话，存出来的模型在 eval() 下是废的 —— 而且加载时不会报错。
	void save_checkpoint(const ResNet& model, const std::string& path);

	// 模型的每个参数和buffer都必须在archive里,缺键会抛。
	// 返回读进去的键数。
	size_t load_checkpoint(ResNet& model, const std::string& path);

}  // namespace svdkd::model

	// TODO(resume): 这里现在只存【成品】—— 参数 + buffer，用于两件事：
	//   「训练完读回最好那次」和「蒸馏时读教师权重」。
	//
	//   续训（断点继续）【还没做】。真要做得另存一个文件，内容是：
	//     - 模型 params + buffers      （现在就有）
	//     - optimizer.state_dict()     （Optimizer 有虚函数 save/load）
	//     - epoch / best_acc
	//     - ★ RNG 状态               ← 最难的一项。没有它，续训后的数据顺序
	//                                 和「一次跑完」不是同一条曲线。
	//
	//   ★ 而且【必须分成两个文件】：best.pt（成品，只读）与 last.pt（进度）。
	//     同一个文件不能同时是「最好那一次」和「当前进度」—— 它们指向不同的轮次。