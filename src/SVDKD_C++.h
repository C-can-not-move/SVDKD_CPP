#pragma once

#include "data/cifar10.h"
#include "data/transforms.h"
#include "model/resnet.h"
#include "kd/my_svd.h"
#include "kd/loss.h"
#include "train/trainer.h"

#include <torch/torch.h>
#include <torch/optim.h> 

#include <filesystem>
#include <iostream>
#include <string>

