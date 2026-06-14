# 第一部分：硬件、数学与系统

## 概览

本部分包含 10 个讨论题，覆盖大模型的硬件基础、数学推导和系统架构。它负责把 Chapter 0 的基础能力，连接到 Chapter 2 / 3 的工程实现。

## 学习组划分

当前 10 节内容先映射到 5 个主线组；后续新增内容再沿这条链扩展。

> 导航说明：侧边栏和组级入口默认收起，先看总览，再点开具体组页。

| 学习组 | 核心问题 | 当前内容映射 | 主题 |
|:---|:---|:---|:---|
| **1A: 数值基础与算力估算** | 先要算什么？ | 01-02 | 数据格式、参数量、FLOPs |
| **1B: 单卡硬件与访存优化** | 单卡怎么跑得快？ | 03-04 | GPU 架构、内存层次、Attention 访存 |
| **1C: 分布式通信与显存共享** | 一张卡不够怎么办？ | 05-06 | 通信拓扑、ZeRO、显存切分 |
| **1D: 异构调度与算子编程** | 怎么精细控制计算和数据流？ | 07-08 | CPU/GPU 协同、CUDA/Triton |
| **1E: 编译优化与硬件生态** | 怎么自动优化和做迁移？ | 09-10 | AI 编译器、芯片现状、TCO |

### 组级入口

| 组页 | 学习组 | 作用 |
|:---|:---|:---|
| [1A](./1A.md) | 1A: 数值基础与算力估算 | 先把显存、精度、参数量和 FLOPs 算清楚 |
| [1B](./1B.md) | 1B: 单卡硬件与访存优化 | 理解 GPU 架构、Attention 和访存瓶颈 |
| [1C](./1C.md) | 1C: 多卡通信与显存共享 | 处理通信拓扑、ZeRO 和并行扩展 |
| [1D](./1D.md) | 1D: 异构调度与算子编程 | 连接 CPU/GPU 协同、CUDA/Triton 和运行时调度 |
| [1E](./1E.md) | 1E: 编译优化与算力生态 | 面向编译器、芯片迁移和成本决策 |

## 学习建议

- 快速入门：1A → 1B
- 系统学习：1A → 1B → 1C → 1D → 1E
- 桥接 Chapter 2 / 3：1A → 1B → 1C → 1D → 1E
- 专项突破：按组维护

## 🔮 扩展候选池（21-33）

以下页面先作为扩展占位，不进入当前主学习路径。当前 `P0` 项已经进入正文草稿阶段，后续内容成熟后再补 Notebook。

### 建议落地顺序

- **P0**：与当前主线衔接最紧密，优先补正文
- **P1**：与现有内容关联强，但可以稍后补
- **P2**：更偏专题化，先占位即可

| 序号 | 逻辑标签 | 暂定主题 | 归属 | 优先级 | 状态 |
|:---|:---|:---|:---|:---|:---|
| 21 | 1A-03 | 量化理论与 INT4/INT8 | 1A | P0 | 草稿 |
| 22 | 1A-04 | MoE 模型参数量计算 | 1A | P0 | 草稿 |
| 23 | 1B-03 | Tensor Core 深度剖析 | 1B | P0 | 草稿 |
| 24 | 1B-04 | SRAM 优化技术 | 1B | P2 | 占位 |
| 25 | 1B-05 | 稀疏计算与稀疏注意力 | 1B | P1 | 占位 |
| 26 | 1C-03 | 并行策略决策框架 | 1C | P0 | 草稿 |
| 27 | 1C-04 | 通信调度优化 | 1C | P1 | 占位 |
| 28 | 1C-05 | 容错与 Checkpoint | 1C | P2 | 占位 |
| 29 | 1D-03 | CUDA Stream 高级调度 | 1D | P0 | 草稿 |
| 30 | 1D-04 | 动态 Shape 处理 | 1D | P1 | 占位 |
| 31 | 1D-05 | GPU 虚拟化与 MIG | 1D | P2 | 占位 |
| 32 | 1E-03 | TVM / MLIR 深度实践 | 1E | P1 | 占位 |
| 33 | 1E-04 | 算力评估与 TCO 模型 | 1E | P0 | 草稿 |

---

## 预留桥接页（11-20）

`11-14` 偏 Chapter 2 前置，`15-20` 偏 Chapter 3 前置。后续直接补正文，不改入口。

| 题号 | 暂定标题 | 归属 | 桥接方向 | 状态 |
|:---|:---|:---|:---|:---|
| 11 | KV Cache and Memory Growth | 1B 单卡硬件与访存优化 | Chapter 2 前置 | 占位 |
| 12 | Tensor Core and Mixed Precision | 1B 单卡硬件与访存优化 | Chapter 2 前置 | 占位 |
| 13 | Profiling and Bottleneck Analysis | 1B 单卡硬件与访存优化 | Chapter 2 / 3 前置 | 占位 |
| 14 | FlashAttention Memory Model | 1B 单卡硬件与访存优化 | Chapter 2 前置 | 占位 |
| 15 | CUDA Execution Model | 1C 系统与编译 | Chapter 3 前置 | 占位 |
| 16 | Warp, Block, and Shared Memory Basics | 1C 系统与编译 | Chapter 3 前置 | 占位 |
| 17 | CUDA Stream and Asynchrony | 1C 系统与编译 | Chapter 3 前置 | 占位 |
| 18 | Triton Block Model | 1C 系统与编译 | Chapter 3 前置 | 占位 |
| 19 | Operator Fusion Introduction | 1C 系统与编译 | Chapter 3 前置 | 占位 |
| 20 | NCCL and AllReduce Basics | 1C 系统与编译 | Chapter 2 / 3 前置 | 占位 |

---

## 📗 1A: 数值基础与算力估算（01-02）

### 🎯 学习目标

- ✅ 理解不同数据格式（FP32、FP16、BF16、INT8）的区别
- ✅ 掌握 Transformer 参数量的计算方法
- ✅ 能够推导训练和推理的 FLOPs
- ✅ 理解混合精度训练的原理

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 01 | [Data Types and Precision](./01_Data_Types_and_Precision.md) | Easy | FP32/FP16/BF16/INT8、混合精度 |
| 02 | [LLM Params and FLOPs](./02_LLM_Params_and_FLOPs.md) | Medium | 参数量计算、FLOPs 推导 |

### 📖 详细题目指南

#### 01: Data Types and Precision

**学习重点：**
- **数据格式占用**：FP32（4 Bytes）、FP16（2 Bytes）、INT8（1 Byte）
- **位分布差异**：FP16（5 位指数 + 10 位尾数）vs BF16（8 位指数 + 7 位尾数）
- **混合精度训练**：为什么需要 FP32 主权重？如何避免梯度下溢？
- **FP8 新趋势**：H100 的 FP8 Tensor Core

**面试高频问题：**
- Q: 7B 模型用 FP16 加载需要多少显存？
- A: 7B × 2 Bytes = 14 GB（纯权重，不含 KV Cache 和激活值）

**实用价值：**
- 快速估算模型显存占用
- 理解量化的原理和收益
- 选择合适的数据格式（训练用 BF16，推理用 INT8）

---

#### 02: LLM Params and FLOPs

**学习重点：**
- **参数量分解**：Embedding + Attention + FFN
- **FLOPs 推导**：前向推理 vs 完整训练（前向 + 反向 + 优化器）
- **Chinchilla 定律**：最优的模型大小和数据量比例

**核心公式：**
- **参数量**：`2Vd + L(12d² + 13d)` （V=词表，d=隐藏维度，L=层数）
- **训练 FLOPs**：`6 × Params × Tokens`（近似公式）
- **推理 FLOPs**：`2 × Params × Tokens`

**面试高频问题：**
- Q: LLaMA-7B 的参数量是如何分布的？
- A: Embedding（32K × 4096 × 2）+ 32 层 Transformer（每层约 200M）

**实用价值：**
- 估算训练时间和成本
- 理解模型架构的参数分布
- 优化模型设计（如使用 GQA 减少 KV 参数）

---

## 📗 1B: 单卡硬件与访存优化（03-06）

### 🎯 学习目标

- ✅ 理解 GPU 的内存层次（HBM、L2、SRAM、寄存器）
- ✅ 掌握 Attention 的显存瓶颈和优化方法
- ✅ 理解分布式通信拓扑（NVLink、PCIe、InfiniBand）
- ✅ 能够计算 ZeRO 的显存节省

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 03 | [GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md) | Medium | GPU 架构、内存层次、带宽 |
| 04 | [Attention Memory Optimization](./04_Attention_Memory_Optimization.md) | Hard | Attention 显存、FlashAttention 原理 |
| 05 | [Communication Topologies](./05_Communication_Topologies.md) | Medium | NVLink、PCIe、InfiniBand、通信带宽 |
| 06 | [VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.md) | Hard | 显存计算、ZeRO-1/2/3、梯度累积 |

### 核心概念解析

#### 章节阅读顺序建议

建议先按 **1A -> 1B -> 1C** 走一遍，再按专题回看：
- 想算清楚模型和成本，回看 1A
- 想理解显存和 Attention，回看 1B
- 想进入实现和优化，回看 1C

这也是 Chapter 1 作为桥梁章的主要职责：把“知道”变成“能算、能判、能解释”。

#### GPU 内存层次（03）

**从快到慢：**
1. **寄存器（Register）**：最快，但容量极小（每个线程几 KB）
2. **共享内存（Shared Memory / SRAM）**：快，容量小（A100: 164KB/SM）
3. **L2 缓存**：中等速度（A100: 40MB）
4. **全局内存（HBM）**：慢，但容量大（A100: 80GB）

**关键指标：**
- **HBM 带宽**：A100（1.5 TB/s）、H100（3.35 TB/s）
- **计算吞吐**：A100（312 TFLOPS FP16）、H100（989 TFLOPS FP16）
- **Memory Bound**：当带宽成为瓶颈时，计算单元空闲

---

#### FlashAttention 原理（04）

**标准 Attention 的问题：**
- 显存占用：O(N²)，存储完整的 Attention 矩阵
- 128K 序列：需要 128K × 128K × 2 Bytes = 32 GB（单个矩阵！）

**FlashAttention 的解决方案：**
1. **Tiling（分块）**：将 Q、K、V 分块处理
2. **Online Softmax**：增量更新 Softmax 的 max 和 sum
3. **SRAM 优化**：在 Shared Memory 中缓存数据，减少 HBM 访问

**性能提升：**
- 显存：O(N²) → O(N)
- 速度：2-4x 加速
- 支持序列长度：4K → 128K+

---

#### ZeRO 显存优化（06）

**传统 DDP 的显存占用：**
- 模型参数：2Φ（FP16）
- 梯度：2Φ
- 优化器状态：12Φ（FP32 参数 + 动量 + 方差）
- **总计：16Φ**

**ZeRO 的三个级别：**

| 方案 | 切分内容 | 显存占用 | 通信开销 |
|------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | `4Φ + 12Φ/N` | 与 DDP 相同 |
| **ZeRO-2** | 优化器状态 + 梯度 | `2Φ + 14Φ/N` | 与 DDP 相同 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | `16Φ/N` | 增加 50% |

**实用价值：**
- ZeRO-1：8 卡训练，优化器显存节省 87.5%
- ZeRO-3：支持超大模型（175B+）训练

---

## 📗 1C: 系统与编译（07-10）

### 🎯 学习目标

- ✅ 理解 CPU-GPU 异构调度
- ✅ 掌握 CUDA 和 Triton 的编程模型
- ✅ 了解 AI 编译器的优化技术
- ✅ 认识国产 AI 芯片的生态

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 07 | [CPU GPU Heterogeneous Scheduling](./07_CPU_GPU_Heterogeneous_Scheduling.md) | Medium | 异构计算、数据传输、Stream |
| 08 | [Programming Models CUDA Triton](./08_Programming_Models_CUDA_Triton.md) | Hard | CUDA、Triton、编程范式 |
| 09 | [AI Compilers and Graph Optimization](./09_AI_Compilers_and_Graph_Optimization.md) | Hard | 计算图优化、算子融合、XLA |
| 10 | [Domestic AI Chips Overview](./10_Domestic_AI_Chips_Overview.md) | Medium | 国产芯片、生态、适配 |

### 核心概念解析

#### CUDA vs Triton（08）

**CUDA C++：**
- **优势**：性能上限高，控制力强
- **劣势**：学习曲线陡峭，代码复杂
- **适用场景**：需要深度优化（如 FlashAttention V3）

**Triton：**
- **优势**：Python 语法，自动优化，易于调试
- **劣势**：性能约为 CUDA 的 80-95%
- **适用场景**：常规融合算子（RMSNorm、SwiGLU）

**技术选型原则：**
1. 先用 PyTorch 验证正确性
2. 用 Profiler 定位瓶颈
3. 瓶颈占比 < 10%：不优化
4. 瓶颈占比 10-20%：用 Triton
5. 瓶颈占比 > 20%：考虑 CUDA

---

#### AI 编译器优化（09）

**核心优化技术：**
1. **算子融合（Operator Fusion）**：减少内存往返
2. **内存优化（Memory Planning）**：复用缓冲区
3. **并行化（Parallelization）**：数据并行、模型并行
4. **自动调优（Auto-tuning）**：搜索较优配置

**主流编译器：**
- **XLA**：TensorFlow 的编译器
- **TorchScript / TorchInductor**：PyTorch 的编译器
- **TVM**：通用深度学习编译器
- **MLIR**：多层次中间表示

---

## 💡 学习建议

### 学习方法

1. **理论与实践结合**：第一章学理论，第二章写代码验证
2. **动手计算**：每个公式都自己推导一遍
3. **对比验证**：用实际模型验证你的计算（如 LLaMA-7B）
4. **建立直觉**：记住关键数字（A100 带宽、FLOPs、显存）

### 关键数字速查表

**GPU 性能（A100 80GB）：**
- HBM 带宽：1.5 TB/s
- FP16 算力：312 TFLOPS
- FP32 算力：19.5 TFLOPS
- Shared Memory：164 KB/SM

**数据格式：**
- FP32：4 Bytes
- FP16/BF16：2 Bytes
- INT8：1 Byte
- INT4：0.5 Byte

**显存占用（混合精度训练）：**
- 模型参数：2Φ
- 梯度：2Φ
- 优化器状态：12Φ
- 总计：16Φ

### 常见问题

**Q: 第一章没有代码，如何学习？**
- A: 第一章是理论基础，建议配合第二章的代码实践。例如，学完 01 题后，可以做第二章的 20 题（量化）

**Q: 数学推导太复杂，可以跳过吗？**
- A: 不建议跳过。参数量和 FLOPs 计算是面试必考题，也是理解模型架构的基础

**Q: 如何验证自己的计算是否正确？**
- A: 用实际模型验证。例如，计算 LLaMA-7B 的参数量，然后用 `model.parameters()` 验证

**Q: 第一章的知识在实际工作中有用吗？**
- A: 非常有用！选择 GPU、估算训练成本、优化显存占用、设计分布式方案都需要这些知识

---

### 代码练习（Code Practice Notebooks）

目前仓库已经提供了 4 个 Jupyter Notebook 代码练习文件；随着 Chapter 1 扩展到 23 节，Notebook 规划也同步扩展为“已发布资产 + 后续练习池”两层结构。

| 文件 | 对应题目 | 内容 | 难度 |
|:---|:---|:---|:---|
| `[01_Data_Types_and_Precision_Practice.md](./01_Data_Types_and_Precision_Practice.md)` | 01题 | 显存计算、混合精度、量化 | Easy |
| `[02_LLM_Params_and_FLOPs_Practice.md](./02_LLM_Params_and_FLOPs_Practice.md)` | 02题 | Transformer 参数计算、FLOPs 推导、Chinchilla 缩放律 | Medium |
| `[03_GPU_Architecture_and_Memory_Practice.md](./03_GPU_Architecture_and_Memory_Practice.md)` | 03题 | 内存层级分析、Attention 显存、FlashAttention 节省 | Hard |
| `[06_VRAM_Calculation_and_ZeRO_Practice.md](./06_VRAM_Calculation_and_ZeRO_Practice.md)` | 06题 | 混合精度显存、DDP vs ZeRO、梯度累积、显存优化 | Hard |

**每个 Notebook 包含：**
- 📝 详细的理论说明
- 🔧 4-5 个 TODO 代码填空题
- 🧪 自动化测试用例
- 📚 参考答案与深度解析

**使用方式：**
1. 在 Colab 中打开 Notebook

---

### Chapter 1 资产分布说明

扩展后的 Chapter 1 不是“每一节都配一个 Notebook”的结构，而是按内容职责分成四类：

**1. 有独立 Notebook 的基础题**
- `01` Data Types and Precision
- `02` LLM Params and FLOPs
- `03` GPU Architecture and Memory
- `06` VRAM Calculation and ZeRO

**2. 以理论页为主、通过后续章节配合学习的专题页**
- `04` Attention Memory Optimization
- `05` Communication Topologies
- `07` CPU GPU Heterogeneous Scheduling
- `08` Programming Models CUDA Triton
- `09` AI Compilers and Graph Optimization
- `10` Domestic AI Chips Overview

**3. 扩展后可做 Notebook 的新增页**
- `1A-03` Quantization Theory & INT4/INT8
- `1A-04` MoE Parameter and Compute
- `1B-04` SRAM Optimization Techniques
- `1B-05` Sparse Computation and Sparse Attention
- `1C-03` Parallel Strategy Decision Framework
- `1C-04` Communication Scheduling Optimization
- `1C-05` Fault Tolerance and Checkpointing
- `1D-03` CUDA Stream Advanced Scheduling
- `1D-04` Dynamic Shape Handling
- `1D-05` GPU Virtualization and MIG
- `1E-03` TVM / MLIR Deep Practice
- `1E-04` TCO and Cost Model

**4. 仍然不优先做 Notebook 的页**
- `1B-03` Tensor Core Deep Dive
- `1C-01` 通信拓扑与互连技术
- `1C-02` ZeRO 优化器深度
- `1E-02` 芯片现状与替代方案

**5. 预留桥接页**
- `11-20`

这意味着：
- 1A / 1B 里的基础计算题更适合配 Notebook
- 1C 里很多页面更适合先把理论和工程直觉讲清楚，再接 Chapter 2 / 3 的实战页
- Chapter 1 的目标是“打基础 + 建桥梁”，不是把所有内容都写成同一种练习格式
- `1B-03` 更偏性能直觉和原理，不是当前优先练习对象
- `1C-01` / `1C-02` 更依赖多卡环境，后续再统一补
- `1E-02` 更偏扩展阅读，不适合单独做 notebook

#### 扩展后 Notebook 覆盖总览

| 分组 | 总节数 | 可做 Notebook | 不适合 | 覆盖率 |
|:---|:---:|:---:|:---:|:---:|
| 1A | 4 | 4 | 0 | 100% |
| 1B | 5 | 4 | 1 | 80% |
| 1C | 5 | 3 | 2 | 60% |
| 1D | 5 | 5 | 0 | 100% |
| 1E | 4 | 3 | 1 | 75% |
| **总计** | **23** | **19** | **4** | **83%** |

#### Notebook 环境约束

- `1A`、`1B` 的基础计算题优先使用单卡环境
- `1C-01` / `1C-02` 需要多卡环境，建议至少 2 GPU，最好 4 GPU
- `1D` 里部分调度和 Graph 练习建议有 CUDA 设备
- `1E-03` 可能需要额外安装 TVM，编译时间更长
- Notebook 仍然遵循“先理论、后练习、再答案”的结构

---

### 样板章节与 Notebook 规范

如果要先从 Chapter 1 里挑 3 页做补强，我建议按这个顺序：

1. **03 GPU Architecture and Memory**
   - 这是最适合作为样板章节的页面
   - 它本身就有“理论 + 公式 + Notebook”的完整闭环
   - 适合统一成 Chapter 1 Notebook 的标准模板

2. **05 Communication Topologies**
   - 这一页适合作为“工程判断”样板
   - 重点是把带宽、拓扑和并行策略讲成可判断的问题

3. **08 Programming Models CUDA Triton**
   - 这一页适合作为“从原理到实现”的样板
   - 重点是把编程模型、执行模型和算子开发串起来

#### Notebook 应该怎么写

Chapter 1 的 Notebook 建议统一为下面的结构：

1. **标题页**
   - 题号、名称、难度、标签、目标人群
   - 一句说明：这份 notebook 对应哪篇理论文档

2. **学习目标**
   - 3 到 5 条即可
   - 写清楚这份 notebook 要解决什么问题

3. **Part 分段**
   - 每个 Part 只解决一个核心问题
   - 先给直觉说明，再给 TODO 代码
   - 一段内容不要同时塞太多概念

4. **TODO 代码单元**
   - 只放必要的函数或计算逻辑
   - 输入输出要明确
   - 尽量保持无状态、可重复执行

5. **测试单元**
   - 每个函数都配一个 `test_...()` 函数
   - 用 `assert` 做确定性验证
   - 测试结果要能直接在 notebook 里看到

6. **STOP HERE**
   - 在参考答案前明确给出停止线
   - 防止学习者直接跳过练习

7. **参考代码与解析**
   - 给出完整答案
   - 解释为什么这样写
   - 补充常见错误和工程含义

#### Notebook 应该怎么测试

建议按这个顺序验：

1. **本地顺序执行**
   - `python test_chapter0_1_notebooks.py`
   - 适合 Chapter 0 / 1 这种“单元自包含”的练习 notebook

2. **逐个答案验证**
   - `python test_notebook_answers.py --mode answer <notebook>`
   - 适合带“题目区 + 参考答案区”的 notebook

3. **防透题检查**
   - `python test_notebook_answers.py --mode question <notebook>`
   - 确认题目区不会把答案漏出来

4. **文档链接检查**
   - `python check_chapter_links.py --scope source`
   - `python check_chapter_links.py --scope docs`

5. **站点构建验证**
   - `npm run docs:build`
   - 确认文档页和导览页能正常构建

#### Notebook 写作约束

- 尽量保持确定性，不依赖随机结果
- 尽量不要依赖外网或手工交互
- 测试逻辑要写成可复用的 `test_` 函数
- 结果尽量用 `assert` 校验，不只靠打印
- 练习和答案的边界要清楚，不能把答案提前放到题目区

## 📝 学习检查清单

完成本章学习后，你应该能够：

**1A: 数值基础与算力估算**
- [ ] 快速计算模型的显存占用（给定参数量和数据格式）
- [ ] 推导 Transformer 的参数量公式
- [ ] 计算训练和推理的 FLOPs
- [ ] 解释混合精度训练的原理

**1B: 单卡硬件与访存优化**
- [ ] 画出 GPU 的内存层次图
- [ ] 解释为什么 Attention 是 Memory Bound
- [ ] 说明 FlashAttention 如何优化显存和速度
- [ ] 计算 ZeRO-1/2/3 的显存节省

**1C: 系统与编译**
- [ ] 理解 CPU-GPU 数据传输的开销
- [ ] 在 PyTorch、Triton、CUDA 之间做技术选型
- [ ] 说明 AI 编译器的核心优化技术
- [ ] 了解国产 AI 芯片的生态

---

## 🔗 与其他章节的联系

**第一章 → 第二章：**
- 01 题（数据格式）→ 20 题（量化）
- 02 题（参数量）→ 05-08 题（模型架构）
- 04 题（Attention 优化）→ 04 题（Attention 实现）
- 06 题（ZeRO）→ 23 题（ZeRO 模拟）

**第一章 → 第三章：**
- 03 题（GPU 架构）→ 01-05 题（Triton 基础）
- 04 题（FlashAttention）→ 08 题（Triton Flash Attention）
- 08 题（编程模型）→ 18-19 题（CUDA 编程）

---

## 🎓 结语

第一章是整个仓库的理论基石，虽然没有代码，但这些知识是理解后续章节的关键。

**学习建议：**
- **不要死记硬背**：理解原理，推导公式
- **动手计算**：每个公式都自己算一遍
- **结合实践**：学完理论立即做第二章的相关题目
- **建立直觉**：记住关键数字，快速估算

**记住：**
- 理论是实践的基础，实践是理论的验证
- 第一章的知识会在面试和工作中反复用到
- 理解硬件和系统，才能写出高性能的代码

祝学习愉快！🚀
