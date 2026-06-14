# Project 1 学习与运行指南

这份指南帮你从“完全新手”的角度阅读整个项目。建议不要一上来就看训练脚本，先从最小的层和公式开始。

## 1. 项目结构

```text
codes/
├── mynn/
│   ├── op.py
│   ├── models.py
│   ├── optimizer.py
│   ├── lr_scheduler.py
│   ├── metric.py
│   ├── runner.py
│   └── answer.md
├── project1_utils.py
├── run_project1_experiments.py
├── make_report_from_results.py
├── report.md
├── guide.md
├── test_train.py
├── test_model.py
├── weight_visualization.py
├── dataset/
├── saved_models/
├── best_models/
├── figs/
└── results/
```

## 2. 每个文件是干什么的

### `mynn/op.py`

神经网络最底层的“积木”。

你应该重点读：

1. `Layer`
2. `Linear`
3. `ReLU`
4. `Dropout`
5. `MultiCrossEntropyLoss`
6. `conv2D`
7. `L2Regularization`
8. `softmax`

对应项目要求：

- Part A：`Linear`、`MultiCrossEntropyLoss`
- Part B：`conv2D`
- Part C Direction 2：`Dropout`、`L2Regularization`

### `mynn/models.py`

把 `op.py` 里的层组装成完整模型。

你应该重点读：

1. `Model_MLP`
2. `Model_CNN`
3. `forward`
4. `backward`
5. `save_model`
6. `load_model`

对应项目要求：

- Part A：MLP baseline
- Part B：CNN model

### `mynn/optimizer.py`

负责更新参数。

你应该重点读：

1. `SGD`
2. `MomentGD`

对应项目要求：

- Part C Direction 1：Optimization

### `mynn/lr_scheduler.py`

负责调整学习率。

你应该重点读：

1. `StepLR`
2. `MultiStepLR`
3. `ExponentialLR`

对应项目要求：

- Part C Direction 1：Optimization

### `mynn/metric.py`

目前只有 `accuracy`，用来计算分类准确率。

### `mynn/runner.py`

原始训练器，适合理解基本训练流程。  
完整项目实验主要使用 `run_project1_experiments.py`，因为它包含更多 Part C 实验功能。

### `mynn/answer.md`

语法和反向传播问题的中文解释文件。  
适合你遇到 `super()`、`__call__`、`assert`、`conv2D.backward` 等不懂时回来查。

### `project1_utils.py`

完整实验的工具箱。

它负责：

- 读取 MNIST
- 数据集切分
- MLP/CNN 输入形状转换
- 训练模型
- 评估模型
- 数据增强
- 鲁棒性扰动
- 混淆矩阵
- 可视化图片保存

### `run_project1_experiments.py`

完整项目的主入口。

它会运行：

1. Part A：MLP baseline
2. Part B：CNN baseline
3. Part C Direction 1：Momentum
4. Part C Direction 1：MultiStepLR
5. Part C Direction 2：L2 regularization
6. Part C Direction 2：Dropout + early stopping
7. Part C Direction 3：Data augmentation
8. Part C Direction 4：Robustness analysis
9. Part C Direction 5：Error analysis and visualization

注意：虽然 Part C 只要求做两个方向，但本项目出于学习目的把五个方向都实现了。

### `make_report_from_results.py`

读取实验结果 JSON，自动生成 `report_generated.md`。

### `report.md`

手写报告模板。  
你可以先看它理解报告结构，真正运行完实验后再用 `make_report_from_results.py` 生成带结果表的版本。

## 3. 推荐阅读顺序

### 第一步：先读语法解释

读：

```text
mynn/answer.md
```

目标：

- 理解 `class`
- 理解 `self`
- 理解 `super().__init__()`
- 理解 `__call__`
- 理解 `assert`
- 理解基础反向传播直觉

### 第二步：读 `op.py` 的基础层

建议顺序：

1. `Layer`
2. `Linear.forward`
3. `Linear.backward`
4. `ReLU.forward`
5. `ReLU.backward`
6. `softmax`
7. `MultiCrossEntropyLoss`

目标：

- 看懂 MLP 的 forward/backward 是怎么连起来的。

### 第三步：读 `models.py` 的 MLP

读：

```text
Model_MLP.__init__
Model_MLP.forward
Model_MLP.backward
```

目标：

- 理解一堆层怎么组成一个完整模型。
- 理解 forward 是从前往后。
- 理解 backward 是从后往前。

### 第四步：读训练流程

读：

```text
project1_utils.py -> train_model
```

目标：

- 理解一个 epoch 做什么。
- 理解 batch 是什么。
- 理解 loss.backward 和 optimizer.step 的顺序。

### 第五步：读 CNN

建议顺序：

1. `op.py -> conv2D.forward`
2. `op.py -> conv2D.backward`
3. `models.py -> Model_CNN`

目标：

- 理解卷积层如何从 `[N, C, H, W]` 得到输出特征图。
- 理解 CNN 为什么需要 Flatten 后才能接 Linear。

### 第六步：读 Part C

建议顺序：

1. `optimizer.py -> MomentGD`
2. `lr_scheduler.py`
3. `op.py -> Dropout`
4. `project1_utils.py -> augment_batch`
5. `project1_utils.py -> robustness_suite`
6. `project1_utils.py -> confusion_matrix` 和画图函数

### 第七步：读完整实验入口

读：

```text
run_project1_experiments.py
```

目标：

- 理解每个实验怎么配置。
- 理解结果保存在哪里。

## 4. 如何运行

### 先跑 quick 模式检查代码

在 `codes/` 目录下运行：

```powershell
python run_project1_experiments.py --mode quick
```

quick 模式只用很少的数据，目的是确认所有代码都能跑通。
当前 quick 模式默认使用 1024 张训练图、256 张验证图、256 张测试图、3 个 epoch；
如果 CNN 在 quick 模式下仍然停留在 10% 左右，说明代码或超参数大概率有问题。

输出位置：

```text
results/project1_results_quick.json
figs/project1/quick/
saved_models/project1/quick/
```

### 再跑 full 模式用于报告

在 `codes/` 目录下运行：

```powershell
python run_project1_experiments.py --mode full --epochs 5
```

默认 batch size 是 128。你也可以显式写出来：

```powershell
python run_project1_experiments.py --mode full --epochs 5 --batch-size 128
```

注意：CNN 的 `conv2D` 已经改成 NumPy 向量化实现，full 模式仍会比 MLP 慢，但不应该再慢到每个 CNN 实验数小时。  
如果时间紧，可以先用：

```powershell
python run_project1_experiments.py --mode full --epochs 2
```

### 生成报告草稿

运行：

```powershell
python make_report_from_results.py --results .\results\project1_results_full.json --output report_generated.md
```

然后把 `report_generated.md` 整理成最终 PDF。

正式实验的图和模型会保存在：

```text
figs/project1/full/
saved_models/project1/full/
```

## 5. 你应该重点观察什么

### Part A

- MLP loss 是否下降？
- MLP train accuracy 和 valid accuracy 是否同步上升？
- 是否出现过拟合？

### Part B

- CNN 是否比 MLP 好？
- CNN 的提升是否明显？
- CNN 是否训练更慢？

### Direction 1

- Momentum 是否更快收敛？
- Learning rate schedule 是否让后期更稳定？

### Direction 2

- L2 或 Dropout 是否降低过拟合？
- Early stopping 是否在验证集不再提升时提前停止？

### Direction 3

- 数据增强是否提升测试集或扰动集表现？

### Direction 4

- 模型在旋转、平移、噪声下准确率下降多少？
- CNN 是否比 MLP 更稳定？

### Direction 5

- 混淆矩阵中哪些数字最容易互相混淆？
- 错误样本是不是本身就难认？
- CNN 第一层卷积核是否学到了边缘或笔画模式？

## 6. 最终提交前检查清单

- `op.py`、`models.py`、`optimizer.py`、`lr_scheduler.py` 能正常运行。
- `run_project1_experiments.py --mode quick` 能跑通。
- `run_project1_experiments.py --mode full` 至少跑出你报告中使用的结果。
- `figs/project1/` 里有 learning curves 和可视化图片。
- `saved_models/project1/` 里有模型权重。
- `report_generated.md` 已整理成 PDF。
- PDF 里写了姓名、学号、GitHub 链接和模型权重链接。
- 上传 GitHub 时不要上传数据集和大模型文件。
