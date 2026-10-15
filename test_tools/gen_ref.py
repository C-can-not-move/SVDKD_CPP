"""生成 resnet 对拍的参考夹具（层 3 测试的 Python 侧）。

由 SVD-KD 官方的 dis_model.py 产出，C++ 侧读同一批文件比对。

    用法： D:\\Software\\Anaconda\\envs\\DLenvs\\python.exe gen_ref.py
    可选： 第一个参数指向官方代码目录（含 dis_model.py）

产出（都在 ./refs/）：
    resnet10_sd.pt        state_dict —— 给 C++ 用 Module::load 装进模型
    resnet10_x.pt         输入张量 (2, 3, 224, 224)
    resnet10_logits.pt    fc 裸输出（softmax 之前）
    resnet10_h1.pt        layer1 输出
    resnet10_h2.pt        layer2 输出
    resnet10_h3.pt        layer3 输出     ← SVD-KD 用这个
    resnet10_keys.txt     74 个键名，每行一个 —— 给 C++ 侧做文本 diff
    resnet10_checksums.txt  各张量的 sum/max/mean —— 编 C++ 之前就能先对一眼
"""
import os
import sys

import torch

OFFICIAL = (sys.argv[1] if len(sys.argv) > 1
            else r"D:\DATA\CODE\DSH\Workspace\papers\蒸馏\SVD蒸馏\01_SVD-KD\official_code")
sys.path.insert(0, OFFICIAL)

from dis_model import h_resnet10  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
os.makedirs(OUT, exist_ok=True)

# 权重和输入用【两个独立】的种子：改权重种子时不会连带改掉输入。
torch.manual_seed(0)

m = h_resnet10(num_classes=10).eval()
sd = m.state_dict()

# ---- state_dict ----
torch.save(sd, os.path.join(OUT, "resnet10_sd.pt"))

# ---- 键名清单（给 C++ 侧 diff）----
keys = list(sd.keys())
with open(os.path.join(OUT, "resnet10_keys.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(keys) + "\n")

# ---- 输入 + 四个输出 ----
g = torch.Generator().manual_seed(12345)
x = torch.randn(2, 3, 224, 224, generator=g)

with torch.no_grad():
    y = m(x)

tensors = {"x": x, "logits": y[0], "h1": y[1], "h2": y[2], "h3": y[3]}
for name, t in tensors.items():
    torch.save(t.contiguous(), os.path.join(OUT, f"resnet10_{name}.pt"))

# ---- 校验和（cross-language 的第一道肉眼检查）----
lines = []
for name, t in tensors.items():
    tf = t.to(torch.float64)
    lines.append(f"{name:8s} shape={str(tuple(t.shape)):22s} dtype={str(t.dtype):14s} "
                 f"sum={tf.sum().item():+.6f}  max={tf.max().item():+.6f}  "
                 f"mean={tf.mean().item():+.6f}  std={tf.std().item():.6f}")
with open(os.path.join(OUT, "resnet10_checksums.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- 结构自检 ----
n_params = sum(p.numel() for p in m.parameters())
print()
print(f"state_dict keys = {len(sd)}   （期望 74）")
print(f"parameters      = {sum(1 for _ in m.parameters())}   （期望 38）")
print(f"buffers         = {sum(1 for _ in m.buffers())}   （期望 36）")
print(f"params numel    = {n_params:,}   （期望 4,910,922，nc=10）")
print(f"                 = {sum(p.numel() for p in h_resnet10(num_classes=100).parameters()):,}"
      f"   （期望 4,957,092，nc=100 ← 论文 L2290）")
print()
print(f"WROTE {OUT}")
