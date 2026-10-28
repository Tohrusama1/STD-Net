# STD-Net

面向软件定义网络异常检测的时空双分支学习框架。

[English README](README.md) · [数据说明](data/README.md) · [整理与修复记录](docs/migration.md)

本仓库从原研究工程整理而来，保留 **Transformer 时序分支、GCN + GAT 图分支、门控融合、不确定性评分**，将预处理、图构建、模型、损失和训练评估拆分为独立模块。

**本版修复了分组中的标签泄漏、测试集重新做特征筛选等问题，因此需要重新训练，不能直接声称复现了论文表格中的数值。** 完整差异见 [迁移记录](docs/migration.md)。

## 安装与运行

在仓库根目录运行，使用 Python 3.10 或以上版本：

```bash
python -m pip install -e .
python -m stdnet train --data data/insdn/train.csv --config configs/default.json --output runs/insdn-01
python -m stdnet evaluate --data data/insdn/test.csv --run runs/insdn-01 --device cpu
python -m unittest discover -s tests -v
```

建议先创建虚拟环境，并按机器情况安装 PyTorch。数据集需自行准备；原始数据、模型权重和运行产物不会进入 Git。

## 使用时需要了解

- 配置统一放在 `configs/default.json`，所有保留参数都会实际生效。一次优化处理一个用户图。
- 训练集拟合用户划分策略、IV 特征列表和标准化器；评估阶段复用它们。用户 ID 不包含真实标签。
- 默认 `label_rule: "any"` 保留原代码的“同一时间点只要有异常流即为异常”。可设为 `"ratio"`，按训练集异常比例作为参考聚合标签，具体差异见迁移记录。
- 图节点表示用户的时间点，图边是时间相邻和跳跃连接，并非真实网络交换机拓扑。
- 模型按最低训练损失保存；本次没有增加验证集选择，也没有用测试标签调阈值。
- `latent_std` 是潜在 logit 的标准差，不代表经过校准的预测置信区间。
- 每次训练与评估均使用新的输出目录，避免覆盖已有实验。

## 整理范围

发布版移除了 IDE 配置、缓存、大数据文件、旧权重、日志、临时图表，以及未经核验的基线原型和重复实验脚本。原工程保留完整；[源文件清单](docs/source_inventory.json) 记录了全部 109 个文件的处理方式。

默认安装仅保留实际需要的 PyTorch、PyG、NumPy、pandas 和 scikit-learn，删除 TensorFlow、torchvision、torchaudio 及未使用的可选依赖。

仓库含英文首页、中文说明、结构图、数据规范、复现实验边界、回归测试和 GitHub Actions 配置。尚未建立远程仓库或执行上传。原工程未附许可证，本次未擅自指定授权协议。
