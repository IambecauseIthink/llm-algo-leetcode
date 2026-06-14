# 第零部分：前置知识与环境准备

## 本部分概览

第零部分用于补齐进入后续学习所需的基础能力。内容覆盖 Python、NumPy、PyTorch、训练循环和调试工具，目标是让读者顺利进入第一部分和第二部分。

当前已补齐 00/01/04/05/07/08/09/12/13 的练习资产，02/03/06/10/11 的理论文档也已就绪。

## 学习组划分

本章分为 4 个学习组。

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **0A: Python 基础** | 00-01 | Python 语法与 NumPy | Easy |
| **0B: PyTorch 基础** | 02-05 | Tensor、Autograd、模块定义 | Easy-Medium |
| **0C: 深度学习基础** | 06-09 | 训练循环、激活函数、归一化 | Medium |
| **0D: 工具与调试** | 10-13 | Profiling、显存优化、调试技巧 | Medium |

### 组级入口

| 组页 | 学习组 | 作用 |
|:---|:---|:---|
| [0A](./0A.md) | 0A: Python 基础 | 先把 Python / NumPy 基础打牢 |
| [0B](./0B.md) | 0B: PyTorch 基础 | 过渡到 Tensor、Autograd 和 Module |
| [0C](./0C.md) | 0C: 深度学习基础 | 衔接训练循环、激活函数和归一化 |
| [0D](./0D.md) | 0D: 工具与调试 | 面向 profiling、显存和调试 |

## 学习顺序

建议按以下顺序学习：

1. 0A：Python 基础
2. 0B：PyTorch 基础
3. 0C：深度学习基础
4. 0D：工具与调试

如果你已经有 Python 或 PyTorch 基础，可以直接从 0B 开始。

## 题目与内容

### 0A: Python 基础（00-01）

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 00 | [Python Essentials for LLM](./00_Python_Essentials_for_LLM.md) | Easy | 列表推导、字典、函数、装饰器、类 |
| 01 | [NumPy and Einsum](./01_NumPy_and_Einsum.md) | Easy | 数组操作、广播、einsum 符号 |

### 0B: PyTorch 基础（02-05）

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 02 | [PyTorch Tensor Fundamentals](./02_PyTorch_Tensor_Fundamentals.md) | Easy | Tensor 创建、操作、设备转移、数据类型 |
| 03 | [PyTorch Autograd and Backward](./03_PyTorch_Autograd_and_Backward.md) | Medium | 自动求导、梯度计算、反向传播 |
| 04 | [PyTorch nn.Module Basics](./04_PyTorch_nn_Module_Basics.md) | Medium | 模块定义、前向传播、参数管理 |
| 05 | [PyTorch Optimizers and Loss](./05_PyTorch_Optimizers_and_Loss.md) | Medium | 损失函数、优化器、学习率 |

### 0C: 深度学习基础（06-09）

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 06 | [Simple Neural Network Training](./06_Simple_Neural_Network_Training.md) | Medium | 训练循环、验证、保存模型 |
| 07 | [Activation Functions](./07_Activation_Functions.md) | Easy | ReLU、GELU、SiLU 的实现与对比 |
| 08 | [Normalization Techniques](./08_Normalization_Techniques.md) | Medium | BatchNorm、LayerNorm 的原理与实现 |
| 09 | [Attention Mechanism Intro](./09_Attention_Mechanism_Intro.md) | Medium | Scaled Dot-Product Attention 基础 |

### 0D: 工具与调试（10-13）

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 10 | [PyTorch Profiling Basics](./10_PyTorch_Profiling_Basics.md) | Medium | PyTorch profiler、时间线分析 |
| 11 | [Memory Profiling and Optimization](./11_Memory_Profiling_and_Optimization.md) | Medium | 显存占用、内存峰值、基础优化 |
| 12 | [Debugging Techniques](./12_Debugging_Techniques.md) | Medium | 断点、日志、异常排查 |
| 13 | [Jupyter and Git Basics](./13_Jupyter_and_Git_Basics.md) | Medium | Notebook 执行顺序、测试与复盘 |

## Notebook 配合

第零部分的 notebook 以基础操作和验证为主，适合先看导学，再做练习，再跑测试。

- 已有 notebook 的页面：00、01、04、05、07、08、09、12、13
- 先只保留正文的页面：02、03、06、10、11

公开 notebook 需要可跑、可测、结构清楚；正文页先按内容维护，不强行挂入口。
