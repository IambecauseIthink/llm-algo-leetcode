# 40. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化
**难度：** Hard | **环境：** CPU-first | **标签：** `量化压缩`, `权重量化`, `GPTQ/AWQ` | **目标人群：** 量化压缩学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

第 25 节和第 26 节已经把量化的两条主线铺开：W8A16 说明了 weight-only 量化如何减少权重读取压力，QLoRA 说明了 4-bit 权重如何服务于低成本微调。但部署阶段还会遇到一个更细的问题：同样是把权重压到低比特，哪些权重更敏感，哪些误差可以接受，校准数据又应该如何参与量化决策？

本节用一个极简 `WeightQuantizerSim` 模拟 GPTQ / AWQ 的核心直觉：GPTQ 更关注校准后的重构误差，AWQ 更强调激活感知和敏感通道保护。学完后，你应该能看清“校准 -> 分组 -> 量化 -> 保护 -> 反量化 -> 误差检查”这条权重量化链路。

本节还会把三类对象放到同一条链路中比较：GPTQ 关注校准后的误差补偿，AWQ 关注激活感知和敏感通道保护，GGUF 负责量化权重的文件格式与部署封装。真实 artifact、backend 和 kernel 的验证继续连接到 [67. 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)。

**关键词：** `GPTQ`, `AWQ`, `weight quantization`

![GPTQ 与 AWQ 的校准路径](../public/02_PyTorch_Algorithms/40_gptq_awq_map_cn.svg)

---

## 前置阅读

**导语：** 进入本节前，先能区分权重、激活和 KV Cache 的量化对象，再观察校准数据如何影响低比特权重的误差。
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](./26_QLoRA_and_4bit_Quantization.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)

---

### Step 1: 为什么 4-bit 量化需要校准

W8A16 已经说明低比特可以减少权重存储，但继续压到 4-bit 后，所有权重使用同一套规则可能放大敏感通道的误差。本节先建立一个判断框架：校准数据提供激活统计，分组 scale 控制局部动态范围，GPTQ / AWQ 再用不同方式处理误差或保护敏感通道。

本节只模拟机制变量，不生成真实 GPTQ / AWQ artifact。学习重点是看清输入、校准信息、量化决策和重构误差之间的关系。

### Step 2: 校准数据与分组 scale

校准样本不是训练数据，而是用来观察激活分布的代表性输入。模拟器先按输入通道汇总激活强度，再把权重按 `group_size` 划分，每组使用独立 scale。需要观察两个变量：校准统计是否能区分敏感通道，以及分组粒度变化后误差和元数据如何变化。

| 变量 | 改变什么 | 观察结果 |
|---|---|---|
| `calibration_samples` | 激活统计的样本量 | 重要性估计是否稳定 |
| `group_size` | 每组共享 scale 的范围 | 重构误差与 scale 数量 |

### Step 3: GPTQ 与 AWQ 的策略差异

两种方法都属于部署前的权重量化，但关注点不同：

- GPTQ：以层输出重构误差为主要观察对象；
- AWQ：利用激活统计识别敏感通道，再对这些通道采取保护策略；
- 共同点：都需要校准输入，且都不能仅凭权重绝对值判断最终质量。

本节的模拟结果只回答“分组、校准和保护策略如何影响局部误差”，真实模型质量和 backend 速度留给 67 节。

### Step 4: 实现、测试与结果解读

下面的题目区实现 `WeightQuantizerSim`，测试区检查量化权重 dtype、scale 形状、敏感通道标记、恢复形状和重构误差。完成后再阅读参考实现和解析，重点对照每个 TODO 如何改变量化状态。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

```


```python
class WeightQuantizerSim(nn.Module):
    """极简版 GPTQ / AWQ 权重量化模拟器。"""

    def __init__(self, bits: int = 4, group_size: int = 32, method: str = "gptq", protect_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.bits = bits
        self.group_size = group_size
        self.method = method.lower()
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("scales", torch.empty(0), persistent=False)
        self.register_buffer("protected_weight", torch.empty(0), persistent=False)
        self.register_buffer("protected_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("importance", torch.empty(0), persistent=False)
        self.weight_shape = None

    def _collect_importance(self, activations: torch.Tensor, in_features: int) -> torch.Tensor:
        act = activations.detach().float()
        if act.ndim == 1:
            importance = act.abs()
        else:
            reduce_dims = tuple(range(act.ndim - 1))
            # ==========================================
            # TODO 1: 根据校准激活统计输入通道重要性
            # 提示: 对除最后一维外的维度求 RMS，最后一维对应 in_features
            # ==========================================
            # importance = ???
        if importance.numel() != in_features:
            raise ValueError(f"Calibration importance dim mismatch: expected {in_features}, got {importance.numel()}")
        return importance

    def fit(self, weight: torch.Tensor, activations: torch.Tensor | None = None) -> "WeightQuantizerSim":
        w = weight.detach().float()
        if w.ndim != 2:
            raise ValueError("WeightQuantizerSim only supports 2D linear weights.")

        out_features, in_features = w.shape
        self.weight_shape = (out_features, in_features)
        importance = torch.ones(in_features, device=w.device, dtype=w.dtype) if activations is None else self._collect_importance(activations, in_features)
        self.importance = importance

        # ==========================================
        # TODO 2: 计算输入维度需要被切成多少个 group
        # 提示: 使用向上取整，最后一组可以不足 group_size
        # ==========================================
        # n_groups = ???
        qweight = torch.zeros_like(w, dtype=torch.int8)
        scales = torch.zeros((out_features, n_groups), dtype=w.dtype, device=w.device)
        protected_weight = torch.zeros_like(w)
        protected_mask = torch.zeros_like(w, dtype=torch.bool)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                wg = w[row, start:end]
                ig = importance[start:end]
                if wg.numel() == 0:
                    continue

                mask = torch.zeros_like(ig, dtype=torch.bool)
                if self.method == "awq":
                    k = max(1, int(round(wg.numel() * self.protect_ratio)))
                    k = min(k, wg.numel())
                    topk = torch.topk(ig, k=k, largest=True).indices
                    # ==========================================
                    # TODO 3: 标记本组中需要保护的敏感通道
                    # 提示: topk 是通道下标，把这些位置在 mask 中置为 True
                    # ==========================================
                    # mask[topk] = ???
                    protected_mask[row, start:end] = mask
                    protected_weight[row, start:end] = wg * mask.to(wg.dtype)

                base = wg[~mask]
                if base.numel() == 0:
                    base = wg
                # ==========================================
                # TODO 4: 为未保护的普通通道计算分组 scale
                # 提示: 对称量化 scale = absmax / qmax，并用 eps 避免除零
                # ==========================================
                # scale = ???

                q_group = torch.zeros_like(wg, dtype=torch.int8)
                q_group[~mask] = torch.clamp(torch.round(wg[~mask] / scale), -self.qmax, self.qmax).to(torch.int8)
                qweight[row, start:end] = q_group
                scales[row, g] = scale

        self.qweight = qweight
        self.scales = scales
        self.protected_weight = protected_weight
        self.protected_mask = protected_mask
        return self

    def dequantize(self) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before dequantize().")

        out_features, in_features = self.weight_shape
        n_groups = self.scales.size(1)
        weight = torch.zeros((out_features, in_features), dtype=self.scales.dtype, device=self.scales.device)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                scale = self.scales[row, g]
                q_group = self.qweight[row, start:end].to(self.scales.dtype)
                # ==========================================
                # TODO 5: 将整数权重反量化回浮点近似值
                # 提示: 量化时除以 scale，恢复时乘回 scale
                # ==========================================
                # dequant = ???
                protected = self.protected_mask[row, start:end]
                if protected.any():
                    dequant = dequant.clone()
                    dequant[protected] = self.protected_weight[row, start:end][protected]
                weight[row, start:end] = dequant

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before forward().")
        weight = self.dequantize().to(x.dtype)
        return F.linear(x, weight)

    def mse(self, weight: torch.Tensor) -> torch.Tensor:
        recon = self.dequantize().to(weight.dtype)
        # ==========================================
        # TODO 6: 计算原始权重和恢复权重之间的均方误差
        # 提示: 先相减、平方，再求平均
        # ==========================================
        # error = ???
        return error

```


```python
# 测试你的实现
def test_weight_quantizer():
    try:
        torch.manual_seed(0)
        weight = torch.randn(4, 8)
        acts = torch.randn(16, 8)
        sim = WeightQuantizerSim(bits=4, group_size=4, method="awq", protect_ratio=0.25).fit(weight, acts)
        restored = sim.dequantize()
        y = sim.forward(torch.randn(2, 8))

        assert sim.qweight.dtype == torch.int8
        assert sim.scales.shape == (4, 2)
        assert sim.importance.shape == (8,)
        assert sim.protected_mask.any()
        assert restored.shape == weight.shape
        assert y.shape == (2, 4)
        assert float(sim.mse(weight)) >= 0.0

        gptq = WeightQuantizerSim(bits=4, group_size=4, method="gptq").fit(weight, acts)
        assert not gptq.protected_mask.any()
        assert gptq.dequantize().shape == weight.shape

        print("✅ WeightQuantizerSim 测试通过")
    except NotImplementedError as e:
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except (AttributeError, NameError, TypeError, ValueError, RuntimeError, AssertionError) as e:
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_weight_quantizer()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python

class WeightQuantizerSim(nn.Module):
    """极简版 GPTQ / AWQ 权重量化模拟器。"""

    def __init__(self, bits: int = 4, group_size: int = 32, method: str = "gptq", protect_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.bits = bits
        self.group_size = group_size
        self.method = method.lower()
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("scales", torch.empty(0), persistent=False)
        self.register_buffer("protected_weight", torch.empty(0), persistent=False)
        self.register_buffer("protected_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("importance", torch.empty(0), persistent=False)
        self.weight_shape = None

    def _collect_importance(self, activations: torch.Tensor, in_features: int) -> torch.Tensor:
        act = activations.detach().float()
        if act.ndim == 1:
            importance = act.abs()
        else:
            reduce_dims = tuple(range(act.ndim - 1))
            # ==========================================
            # TODO 1: 根据校准激活统计输入通道重要性
            # 提示: 对除最后一维外的维度求 RMS，最后一维对应 in_features
            # ==========================================
            importance = act.pow(2).mean(dim=reduce_dims).sqrt()
        if importance.numel() != in_features:
            raise ValueError(f"Calibration importance dim mismatch: expected {in_features}, got {importance.numel()}")
        return importance

    def fit(self, weight: torch.Tensor, activations: torch.Tensor | None = None) -> "WeightQuantizerSim":
        w = weight.detach().float()
        if w.ndim != 2:
            raise ValueError("WeightQuantizerSim only supports 2D linear weights.")

        out_features, in_features = w.shape
        self.weight_shape = (out_features, in_features)
        importance = torch.ones(in_features, device=w.device, dtype=w.dtype) if activations is None else self._collect_importance(activations, in_features)
        self.importance = importance

        # ==========================================
        # TODO 2: 计算输入维度需要被切成多少个 group
        # 提示: 使用向上取整，最后一组可以不足 group_size
        # ==========================================
        n_groups = (in_features + self.group_size - 1) // self.group_size
        qweight = torch.zeros_like(w, dtype=torch.int8)
        scales = torch.zeros((out_features, n_groups), dtype=w.dtype, device=w.device)
        protected_weight = torch.zeros_like(w)
        protected_mask = torch.zeros_like(w, dtype=torch.bool)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                wg = w[row, start:end]
                ig = importance[start:end]
                if wg.numel() == 0:
                    continue

                mask = torch.zeros_like(ig, dtype=torch.bool)
                if self.method == "awq":
                    k = max(1, int(round(wg.numel() * self.protect_ratio)))
                    k = min(k, wg.numel())
                    topk = torch.topk(ig, k=k, largest=True).indices
                    # ==========================================
                    # TODO 3: 标记本组中需要保护的敏感通道
                    # 提示: topk 是通道下标，把这些位置在 mask 中置为 True
                    # ==========================================
                    mask[topk] = True
                    protected_mask[row, start:end] = mask
                    protected_weight[row, start:end] = wg * mask.to(wg.dtype)

                base = wg[~mask]
                if base.numel() == 0:
                    base = wg
                # ==========================================
                # TODO 4: 为未保护的普通通道计算分组 scale
                # 提示: 对称量化 scale = absmax / qmax，并用 eps 避免除零
                # ==========================================
                scale = (base.abs().max() / self.qmax).clamp_min(self.eps)

                q_group = torch.zeros_like(wg, dtype=torch.int8)
                q_group[~mask] = torch.clamp(torch.round(wg[~mask] / scale), -self.qmax, self.qmax).to(torch.int8)
                qweight[row, start:end] = q_group
                scales[row, g] = scale

        self.qweight = qweight
        self.scales = scales
        self.protected_weight = protected_weight
        self.protected_mask = protected_mask
        return self

    def dequantize(self) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before dequantize().")

        out_features, in_features = self.weight_shape
        n_groups = self.scales.size(1)
        weight = torch.zeros((out_features, in_features), dtype=self.scales.dtype, device=self.scales.device)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                scale = self.scales[row, g]
                q_group = self.qweight[row, start:end].to(self.scales.dtype)
                # ==========================================
                # TODO 5: 将整数权重反量化回浮点近似值
                # 提示: 量化时除以 scale，恢复时乘回 scale
                # ==========================================
                dequant = q_group * scale
                protected = self.protected_mask[row, start:end]
                if protected.any():
                    dequant = dequant.clone()
                    dequant[protected] = self.protected_weight[row, start:end][protected]
                weight[row, start:end] = dequant

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before forward().")
        weight = self.dequantize().to(x.dtype)
        return F.linear(x, weight)

    def mse(self, weight: torch.Tensor) -> torch.Tensor:
        recon = self.dequantize().to(weight.dtype)
        # ==========================================
        # TODO 6: 计算原始权重和恢复权重之间的均方误差
        # 提示: 先相减、平方，再求平均
        # ==========================================
        error = torch.mean((weight.float() - recon.float()) ** 2)
        return error

```

### 解析

**1. TODO 1: 统计通道重要性**
- **实现方式**：`importance = act.pow(2).mean(dim=reduce_dims).sqrt()`
- **关键点**：最后一维对应输入通道，其他维度是 batch 或序列维度，需要被聚合掉
- **技术细节**：这里用 RMS 近似衡量通道激活强度；激活越大的通道，权重误差越容易影响输出

**2. TODO 2: 计算分组数量**
- **实现方式**：`n_groups = (in_features + self.group_size - 1) // self.group_size`
- **关键点**：分组数要向上取整，因为最后一组可能不足 `group_size`
- **技术细节**：分组量化让每组拥有独立 scale，比整层共享一个 scale 更能适应局部数值范围

**3. TODO 3: 标记 AWQ 敏感通道**
- **实现方式**：`mask[topk] = True`
- **关键点**：`topk` 来自本组内 importance 最大的通道，这些位置会被 `protected_mask` 记录
- **技术细节**：本节用“保留原始浮点权重”模拟 AWQ 的敏感通道保护，真实实现通常会采用更细的 scale 搜索和重缩放策略

**4. TODO 4: 计算分组 scale**
- **实现方式**：`scale = (base.abs().max() / self.qmax).clamp_min(self.eps)`
- **关键点**：对称量化用本组绝对最大值确定动态范围，并用 `eps` 避免全零分组除零
- **技术细节**：`qmax = 2 ** (bits - 1) - 1`，4-bit 对称量化时有效正向上限是 7

**5. TODO 5: 反量化恢复权重**
- **实现方式**：`dequant = q_group * scale`
- **关键点**：量化时是 `round(w / scale)`，恢复时就乘回同一个 scale
- **技术细节**：如果当前位置被 `protected_mask` 标记，反量化结果会被原始 `protected_weight` 覆盖

**6. TODO 6: 计算重构误差**
- **实现方式**：`error = torch.mean((weight.float() - recon.float()) ** 2)`
- **关键点**：MSE 用来衡量量化恢复权重和原始权重之间的平均平方偏差
- **技术细节**：这个误差只检查权重重构，不等价于最终模型精度；真实评估还要看校准集或下游任务指标

**GPTQ / AWQ 核心机制**
- **GPTQ 直觉**：利用校准数据估计量化对层输出的影响，让低比特权重尽量维持原始层行为
- **AWQ 直觉**：激活越强的通道越敏感，少量通道需要更保守地量化或直接保护
- **分组量化**：按 group 计算 scale，可以减少极端值对整层量化范围的支配

**工程优化要点**
- **存储收益**：4-bit 权重量化能显著降低模型权重显存和加载带宽
- **元数据成本**：分组越细，scale 越多，精度通常更好，但元数据开销也更大
- **部署实践**：真实 GPTQ / AWQ 还涉及校准集选择、kernel 支持、group size、zero point、packing 格式和端到端精度评估

### Step 5：可选 GPU 实验——测量 GPTQ / AWQ 模拟器

![GPTQ AWQ GPU 机制实验流程](../public/02_PyTorch_Algorithms/40_gptq_awq_gpu_mechanism_flow.svg)

实验从真实模型的 `q_proj` forward hook 取得校准激活，再在 GPU 上比较 GPTQ / AWQ 教学模拟器的校准耗时、分组和重构误差。它验证的是“真实模型状态上的机制模拟”，不生成真实 GPTQ / AWQ artifact，也不启动 vLLM / SGLang；证据等级记为 `gpu_simulation_on_real_model_state`。

先运行 `dry_run` 检查环境，再切换到 `real_gpu`。`CALIBRATION_SAMPLES` 会控制重复校准文本的数量；真实 artifact、kernel、吞吐和任务质量转到 67 节。


```python
import json
import platform
import time
from pathlib import Path

RUN_MODE = 'dry_run'  # cpu / dry_run / real_gpu；dry_run 只做环境检查
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # real_gpu 使用真实权重和真实层输入
CALIBRATION_PROMPTS = ['Explain quantization.', 'Why does KV Cache grow?', 'Compare GPTQ and AWQ.']
SEED = 42
OUT_FEATURES = 1024
IN_FEATURES = 1024
CALIBRATION_SAMPLES = 32
GROUP_SIZE = 32
BITS = 4
PROTECT_RATIO = 0.05
WARMUP = 2
ITERS = 10
OUTPUT_PATH = Path('benchmarks/results/40_gptq_awq_gpu.json')

torch.manual_seed(SEED)
cuda_available = torch.cuda.is_available()
if RUN_MODE == 'real_gpu' and not cuda_available:
    raise RuntimeError('RUN_MODE=real_gpu 但 CUDA 不可用，请先完成 GPU 环境预检。')
device = torch.device('cuda' if RUN_MODE == 'real_gpu' else 'cpu')
runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'cuda_available': cuda_available, 'device': torch.cuda.get_device_name(0) if cuda_available else 'cpu'}

def _sync():
    """确保 CUDA 异步操作完成后再读取计时或显存。"""
    if device.type == 'cuda': torch.cuda.synchronize()

def _measure(fn):
    """测量一次校准模拟的平均耗时。"""
    for _ in range(WARMUP): fn()
    _sync(); start = time.perf_counter()
    for _ in range(ITERS): fn()
    _sync()
    return round((time.perf_counter() - start) * 1000 / ITERS, 4)

evidence_level = 'environment_preflight' if RUN_MODE == 'dry_run' else 'gpu_simulation_on_real_model_state'
result = {'stage': evidence_level, 'run_mode': RUN_MODE, 'runtime': runtime, 'config': {
    'out_features': OUT_FEATURES, 'in_features': IN_FEATURES, 'calibration_samples': CALIBRATION_SAMPLES,
    'bits': BITS, 'group_size': GROUP_SIZE, 'protect_ratio': PROTECT_RATIO,
    'warmup': WARMUP, 'iters': ITERS, 'seed': SEED, 'model_id': MODEL_ID,
}, 'evidence_level': evidence_level}
if RUN_MODE == 'dry_run':
    result['decision'] = {'decision': 'ready_to_measure', 'reason': '仅完成环境与配置检查，尚未运行 GPU 校准测量。'}
else:
    # real_gpu 通过 forward hook 读取真实 q_proj 输入；cpu 模式保留小型确定性张量。
    if RUN_MODE == 'real_gpu':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        calibration_texts = [CALIBRATION_PROMPTS[i % len(CALIBRATION_PROMPTS)] for i in range(CALIBRATION_SAMPLES)]
        batch = tokenizer(calibration_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        source = model.model.layers[0].self_attn.q_proj
        captured = {}
        handle = source.register_forward_hook(lambda _m, inputs, _out: captured.setdefault('activations', inputs[0].detach()))
        with torch.no_grad(): model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'), use_cache=False)
        handle.remove()
        weight = source.weight.detach().float()
        activations = captured['activations'].reshape(-1, weight.shape[-1]).float()
        OUT_FEATURES, IN_FEATURES = weight.shape
        del model, source, batch, captured
        if device.type == 'cuda': torch.cuda.empty_cache()
    else:
        weight = torch.randn(OUT_FEATURES, IN_FEATURES, device=device)
        activations = torch.randn(CALIBRATION_SAMPLES, IN_FEATURES, device=device)
    runs = {}
    for method in ('gptq', 'awq'):
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        sim = WeightQuantizerSim(bits=BITS, group_size=GROUP_SIZE, method=method, protect_ratio=PROTECT_RATIO).to(device)
        elapsed = _measure(lambda: sim.fit(weight, activations))
        restored = sim.dequantize()
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None
        runs[method] = {'latency_ms': elapsed, 'peak_memory_mb': None if peak is None else round(peak, 2),
                       'reconstruction_mse': round(float(sim.mse(weight)), 8),
                       'protected_channels': int(sim.protected_mask.any(dim=0).sum())}
    result['config'].update({'out_features': OUT_FEATURES, 'in_features': IN_FEATURES,
                            'actual_activation_shape': list(activations.shape), 'actual_calibration_samples': int(batch['input_ids'].shape[0]) if RUN_MODE == 'real_gpu' else CALIBRATION_SAMPLES,
                            'state_source': 'real_model_q_proj_hook' if RUN_MODE == 'real_gpu' else 'synthetic_cpu'})
    result.update({'runs': runs, 'decision': {'decision': 'measure',
        'reason': '比较真实模型状态上的 GPTQ/AWQ 模拟误差；不代表真实 artifact 或 backend 收益。'}})
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### GPU 实验结果记录

| 方法 | bits | group_size | calibration samples | protect_ratio | reconstruction MSE | latency (ms) | peak memory (MB) | evidence level |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GPTQ simulation |  |  |  | 0 |  |  |  | gpu_simulation_on_real_model_state |
| AWQ simulation |  |  |  |  |  |  |  | gpu_simulation_on_real_model_state |

模拟器结果只说明校准统计和重构误差关系；真实 GPTQ / AWQ artifact、kernel 和任务质量需要转到 67。
## 相关阅读

完成校准、分组、敏感通道保护和误差检查后，可以继续阅读 GPTQ / AWQ 原论文与真实部署项目。

- [GPTQ 原论文：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ 原论文：Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [AutoGPTQ 官方仓库](https://github.com/AutoGPTQ/AutoGPTQ)
- [41. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化](./41_FP8_and_KV_Cache_Quantization.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
