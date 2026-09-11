# 41. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化
**难度：** Hard | **环境：** CPU-first | **标签：** `量化压缩`, `FP8`, `KV Cache` | **目标人群：** 量化压缩学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

第 40 节关注的是权重量化：把模型参数压得更小，降低加载和访存成本。但推理阶段的压力不只来自权重。长上下文生成时，KV Cache 会随着序列长度和并发请求持续增长；同时，部分激活或中间张量也会带来带宽压力。只压权重，不能完全解决长上下文推理的显存和带宽瓶颈。

本节用一个极简 `FP8KVCacheSim` 模拟两类推理量化：用对称低精度量化近似 FP8 张量，用分组 scale 量化 KV Cache。学完后，你应该能看清“量化值、scale、反量化、误差检查”这条闭环，以及为什么 KV Cache 通常需要按最后一维分组处理。

本节关注运行时张量和 KV Cache：CPU 模拟解释 scale、分组和误差，`40` 继续处理 GPTQ/AWQ 权重量化，`67` 负责 GGUF、真实 FP8 kernel 和 backend 验证。

**关键词：** `FP8`, `KV cache quantization`, `deployment`

![FP8 与 KV Cache 量化路径](../public/02_PyTorch_Algorithms/41_fp8_kv_quant_flow_cn.svg)

---

## 前置阅读

**导语：** 进入本节前，先能区分权重量化与运行时张量量化，再观察 FP8 和 KV Cache 量化如何改变存储、带宽与误差。
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)
- [40. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化](./40_GPTQ_and_AWQ_Weight_Quantization.md)

---

### Step 1: 为什么运行时还要量化 KV Cache

生成式推理会持续追加 KV Cache；上下文越长、并发越高，缓存越容易成为显存容量和带宽压力。本节关注两个运行时对象：普通推理张量的低精度表示，以及 KV Cache 的分组量化。

先记住一条主线：量化值、scale、原始形状和恢复误差必须一起保存，才能形成可检查的运行时状态。

### Step 2: FP8 近似量化的状态闭环

本节用 INT8 容器模拟低精度张量的保存路径：先根据 `absmax` 得到 scale，再把浮点值映射为整数，恢复时用同一个 scale 还原近似值。这里验证的是量化闭环，不是硬件 FP8 的 E4M3 / E5M2 编码或 Tensor Core kernel。

### Step 3: KV Cache 的分组量化

KV Cache 沿最后一维分组，每组保存一套 scale。`kv_group_size` 越小，scale 对局部数值范围的适应性通常越强，但元数据和量化处理次数也会增加。需要同时观察缓存字节数、恢复误差和量化/恢复耗时。

| 变量 | 影响对象 | 主要观察 |
|---|---|---|
| `seq_len` | 缓存总量 | 长上下文压力 |
| `kv_group_size` | scale 数量与局部误差 | 压缩比、MSE、耗时 |
| `dtype` / 容器位宽 | 单元素存储 | 原始字节与量化字节 |

### Step 4: 实现提示

- 先完成 `_sym_quantize`：`absmax → scale → q`。
- `quantize_fp8` 记录低精度值、scale 和原始 shape。
- `quantize_kv_cache` 沿最后一维分组，并为每组保存 scale。
- 最后用恢复结果计算 MSE，并检查 shape 是否保持一致。


```python
import torch
import torch.nn as nn

```


```python
class FP8KVCacheSim(nn.Module):
    """极简版 FP8 与 KV Cache 量化模拟器。"""

    def __init__(self, fp8_qmax: int = 127, kv_group_size: int = 64, eps: float = 1e-8):
        super().__init__()
        if kv_group_size <= 0:
            raise ValueError("kv_group_size must be positive")
        self.fp8_qmax = fp8_qmax
        self.kv_group_size = kv_group_size
        self.eps = eps

        self.register_buffer("fp8_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("fp8_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("kv_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("kv_scale", torch.empty(0), persistent=False)
        self.fp8_shape = None
        self.kv_shape = None

    def _sym_quantize(self, x: torch.Tensor, qmax: int):
        x = x.detach().float()
        # ==========================================
        # TODO 1: 补完对称量化闭环
        # 提示: 先算 absmax，再算 scale = qmax / absmax.clamp_min(self.eps)，
        # 最后做 round + clamp + int8 转换得到 q。
        # ==========================================
        # absmax = ???
        # scale = ???
        # q = ???
        return q, scale

    def _sym_dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        return q.to(scale.dtype) / scale.clamp_min(self.eps)

    def quantize_fp8(self, x: torch.Tensor):
        q, scale = self._sym_quantize(x, self.fp8_qmax)
        self.fp8_q = q
        self.fp8_scale = scale
        # ==========================================
        # TODO 2a: 记录 FP8 近似张量的原始 shape
        # 提示: 这里先记录 self.fp8_shape = tuple(x.shape)。
        # 后面 quantize_kv_cache 里再补 n_groups，用它初始化 scales。
        # ==========================================
        # self.fp8_shape = ???
        return q, scale

    def dequantize_fp8(self):
        if self.fp8_shape is None:
            raise RuntimeError("Call quantize_fp8() before dequantize_fp8().")
        return self._sym_dequantize(self.fp8_q, self.fp8_scale)

    def quantize_kv_cache(self, kv_cache: torch.Tensor):
        kv = kv_cache.detach().float()
        if kv.ndim < 2:
            raise ValueError("KV cache should have at least 2 dimensions.")

        last_dim = kv.size(-1)
        # ==========================================
        # TODO 2b: 记录 KV Cache 的分组状态
        # 提示: n_groups 用向上取整计算，最后一组可以不足 kv_group_size。
        # ==========================================
        # n_groups = ???
        qkv = torch.zeros_like(kv, dtype=torch.int8)
        scales = torch.zeros(kv.shape[:-1] + (n_groups,), dtype=kv.dtype, device=kv.device)

        flat = kv.reshape(-1, last_dim)
        flat_q = qkv.reshape(-1, last_dim)
        flat_scale = scales.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                chunk = flat[row, start:end]
                if chunk.numel() == 0:
                    continue
                q, scale = self._sym_quantize(chunk, self.fp8_qmax)
                flat_q[row, start:end] = q
                flat_scale[row, g] = scale

        self.kv_q = qkv
        self.kv_scale = scales
        self.kv_shape = tuple(kv.shape)
        return qkv, scales

    def dequantize_kv_cache(self):
        if self.kv_shape is None:
            raise RuntimeError("Call quantize_kv_cache() before dequantize_kv_cache().")

        kv = self.kv_q.to(self.kv_scale.dtype)
        last_dim = kv.size(-1)
        n_groups = self.kv_scale.size(-1)
        flat = kv.reshape(-1, last_dim)
        flat_out = torch.zeros_like(flat, dtype=self.kv_scale.dtype)
        flat_scale = self.kv_scale.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                scale = flat_scale[row, g]
                # ==========================================
                # TODO 3a: 恢复当前 KV Cache 分组
                # 提示: 先用当前 group 的 scale 恢复 flat[row, start:end]，
                # 再把 restored_chunk 写回 flat_out 的同一区间。
                # ==========================================
                # restored_chunk = ???
                flat_out[row, start:end] = restored_chunk

        return flat_out.reshape(self.kv_shape)

    def fit(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        self.quantize_fp8(hidden_states)
        if kv_cache is not None:
            self.quantize_kv_cache(kv_cache)
        return self

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        fp8_q, fp8_scale = self._sym_quantize(hidden_states, self.fp8_qmax)
        fp8_restored = self._sym_dequantize(fp8_q, fp8_scale)

        if kv_cache is None:
            return fp8_restored

        self.quantize_kv_cache(kv_cache)
        kv_restored = self.dequantize_kv_cache()
        return fp8_restored, kv_restored

    def mse(self, original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3b: 计算恢复误差
        # 提示: 把 original / restored 转成 float 后，相减平方再求平均。
        # ==========================================
        # error = ???
        return error

```


```python
# 测试你的实现
def test_fp8_kv_cache_quantization():
    try:
        torch.manual_seed(0)
        sim = FP8KVCacheSim(fp8_qmax=127, kv_group_size=4)
        hidden = torch.randn(2, 8)
        kv = torch.randn(2, 3, 8)

        sim.fit(hidden, kv)
        hidden_restore = sim.dequantize_fp8()
        kv_restore = sim.dequantize_kv_cache()
        out_hidden, out_kv = sim.forward(hidden, kv)

        assert sim.fp8_q.dtype == torch.int8
        assert sim.kv_q.dtype == torch.int8
        assert sim.fp8_shape == tuple(hidden.shape)
        assert sim.kv_shape == tuple(kv.shape)
        assert sim.kv_scale.shape == (2, 3, 2)
        assert hidden_restore.shape == hidden.shape
        assert kv_restore.shape == kv.shape
        assert out_hidden.shape == hidden.shape
        assert out_kv.shape == kv.shape
        assert float(sim.mse(hidden, hidden_restore)) >= 0.0

        zero = torch.zeros(2, 7)
        zero_kv = torch.zeros(1, 2, 7)
        sim.fit(zero, zero_kv)
        assert torch.isfinite(sim.dequantize_fp8()).all()
        assert torch.isfinite(sim.dequantize_kv_cache()).all()
        assert sim.kv_scale.shape[-1] == 2, '非整除的最后一维应向上取整分组'

        print('✅ FP8KVCacheSim 测试通过')
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except (AttributeError, NameError, TypeError, ValueError, RuntimeError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_fp8_kv_cache_quantization()

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
# TODO：下面是题目区的参考实现。

class FP8KVCacheSim(nn.Module):
    """极简版 FP8 与 KV Cache 量化模拟器。"""

    def __init__(self, fp8_qmax: int = 127, kv_group_size: int = 64, eps: float = 1e-8):
        super().__init__()
        if kv_group_size <= 0:
            raise ValueError("kv_group_size must be positive")
        self.fp8_qmax = fp8_qmax
        self.kv_group_size = kv_group_size
        self.eps = eps

        self.register_buffer("fp8_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("fp8_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("kv_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("kv_scale", torch.empty(0), persistent=False)
        self.fp8_shape = None
        self.kv_shape = None

    def _sym_quantize(self, x: torch.Tensor, qmax: int):
        x = x.detach().float()
        # ==========================================
        # TODO 1: 补完对称量化闭环
        # 提示: 先算 absmax，再算 scale = qmax / absmax.clamp_min(self.eps)，
        # 最后做 round + clamp + int8 转换得到 q。
        # ==========================================
        # absmax = ???
        absmax = torch.max(torch.abs(x))
        # scale = ???
        scale = qmax / absmax.clamp_min(self.eps)
        # q = ???
        q = torch.clamp(torch.round(x * scale), -qmax, qmax).to(torch.int8)
        return q, scale

    def _sym_dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        return q.to(scale.dtype) / scale.clamp_min(self.eps)

    def quantize_fp8(self, x: torch.Tensor):
        q, scale = self._sym_quantize(x, self.fp8_qmax)
        self.fp8_q = q
        self.fp8_scale = scale
        # ==========================================
        # TODO 2a: 记录 FP8 近似张量的原始 shape
        # 提示: 这里先记录 self.fp8_shape = tuple(x.shape)。
        # 后面 quantize_kv_cache 里再补 n_groups，用它初始化 scales。
        # ==========================================
        # self.fp8_shape = ???
        self.fp8_shape = tuple(x.shape)
        return q, scale

    def dequantize_fp8(self):
        if self.fp8_shape is None:
            raise RuntimeError("Call quantize_fp8() before dequantize_fp8().")
        return self._sym_dequantize(self.fp8_q, self.fp8_scale)

    def quantize_kv_cache(self, kv_cache: torch.Tensor):
        kv = kv_cache.detach().float()
        if kv.ndim < 2:
            raise ValueError("KV cache should have at least 2 dimensions.")

        last_dim = kv.size(-1)
        # ==========================================
        # TODO 2b: 记录 KV Cache 的分组状态
        # 提示: n_groups 用向上取整计算，最后一组可以不足 kv_group_size。
        # ==========================================
        # n_groups = ???
        n_groups = (last_dim + self.kv_group_size - 1) // self.kv_group_size
        qkv = torch.zeros_like(kv, dtype=torch.int8)
        scales = torch.zeros(kv.shape[:-1] + (n_groups,), dtype=kv.dtype, device=kv.device)

        flat = kv.reshape(-1, last_dim)
        flat_q = qkv.reshape(-1, last_dim)
        flat_scale = scales.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                chunk = flat[row, start:end]
                if chunk.numel() == 0:
                    continue
                q, scale = self._sym_quantize(chunk, self.fp8_qmax)
                flat_q[row, start:end] = q
                flat_scale[row, g] = scale

        self.kv_q = qkv
        self.kv_scale = scales
        self.kv_shape = tuple(kv.shape)
        return qkv, scales

    def dequantize_kv_cache(self):
        if self.kv_shape is None:
            raise RuntimeError("Call quantize_kv_cache() before dequantize_kv_cache().")

        kv = self.kv_q.to(self.kv_scale.dtype)
        last_dim = kv.size(-1)
        n_groups = self.kv_scale.size(-1)
        flat = kv.reshape(-1, last_dim)
        flat_out = torch.zeros_like(flat, dtype=self.kv_scale.dtype)
        flat_scale = self.kv_scale.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                scale = flat_scale[row, g]
                # ==========================================
                # TODO 3a: 恢复当前 KV Cache 分组
                # 提示: 先用当前 group 的 scale 恢复 flat[row, start:end]，
                # 再把 restored_chunk 写回 flat_out 的同一区间。
                # ==========================================
                # restored_chunk = ???
                restored_chunk = self._sym_dequantize(flat[row, start:end], scale)
                flat_out[row, start:end] = restored_chunk

        return flat_out.reshape(self.kv_shape)

    def fit(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        self.quantize_fp8(hidden_states)
        if kv_cache is not None:
            self.quantize_kv_cache(kv_cache)
        return self

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        fp8_q, fp8_scale = self._sym_quantize(hidden_states, self.fp8_qmax)
        fp8_restored = self._sym_dequantize(fp8_q, fp8_scale)

        if kv_cache is None:
            return fp8_restored

        self.quantize_kv_cache(kv_cache)
        kv_restored = self.dequantize_kv_cache()
        return fp8_restored, kv_restored

    def mse(self, original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3b: 计算恢复误差
        # 提示: 把 original / restored 转成 float 后，相减平方再求平均。
        # ==========================================
        # error = ???
        error = torch.mean((original.float() - restored.float()) ** 2)
        return error

```

### 解析

TODO 1：`_sym_quantize` 负责补完最小对称量化闭环。先用 `absmax = torch.max(torch.abs(x))` 找到动态范围，再用 `scale = qmax / absmax.clamp_min(self.eps)` 计算缩放系数，最后做 `round + clamp + int8` 得到低精度张量 `q`。

TODO 2a/2b：`quantize_fp8` 和 `quantize_kv_cache` 负责补完状态记录。前者保存 `self.fp8_shape = tuple(x.shape)`；后者用 `n_groups = (last_dim + self.kv_group_size - 1) // self.kv_group_size` 决定 KV Cache 沿最后一维的分组数，并初始化分组 scale。

TODO 3a/3b：`dequantize_kv_cache` 和 `mse` 负责补完恢复与误差检查。前者用每个 group 自己的 scale 恢复 `restored_chunk`，再写回 `flat_out`；后者用 `torch.mean((original.float() - restored.float()) ** 2)` 计算重构误差，完成“量化 -> 恢复 -> 评估”的最小闭环。

**FP8 与 KV Cache 量化核心机制**
- **FP8 近似**：用低精度值和 scale 保存张量，降低带宽和存储压力
- **KV Cache 分组**：对最后一维分组保存 scale，使长上下文缓存可以更细粒度地压缩和恢复
- **量化闭环**：任何推理量化都要同时记录低精度值、scale、shape 和恢复误差

**工程优化要点**
- **硬件格式**：真实 FP8 通常涉及 E4M3 / E5M2、Tensor Core 支持和 kernel 路径，本节只模拟核心思想
- **缓存收益**：KV Cache 量化对长上下文和高并发更有价值，因为缓存大小会随序列长度线性增长
- **精度边界**：KV Cache 参与后续 attention，过度压缩可能影响生成质量，需要结合 perplexity、任务指标和在线效果验证

### Step 5：可选 GPU 实验——测量 FP8 / KV Cache 模拟路径

![FP8 KV Cache GPU 机制实验流程](../public/02_PyTorch_Algorithms/41_fp8_kv_gpu_mechanism_flow.svg)

实验从真实模型的 `past_key_values` 取得一段实际 KV 状态，再比较分组量化的原始字节数、量化字节数、恢复误差、耗时和峰值显存。当前使用 INT8 容器模拟量化闭环，并把 K/V 拼成统一教学张量；它不代表真实 FP8 Tensor Core、backend 内部 KV 布局或 serving 收益，证据等级记为 `gpu_simulation_on_real_kv_state`。

先运行 `dry_run` 检查环境，再切换到 `real_gpu`。结果必须同时记录配置的最大长度和实际 `prompt_tokens`；如果要研究长上下文增长，应改变输入长度并分别保存报告。


```python
import json
import platform
import time
from pathlib import Path

RUN_MODE = 'dry_run'  # cpu / dry_run / real_gpu；dry_run 只做环境检查
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # real_gpu 使用真实模型生成 KV Cache
PROMPT = 'Explain how KV Cache grows during generation.'
SEED = 42
BATCH_SIZE = 1
NUM_HEADS = 16
SEQ_LEN = 512
HEAD_DIM = 64
KV_GROUP_SIZE = 32
WARMUP = 5
ITERS = 20
OUTPUT_PATH = Path('benchmarks/results/41_fp8_kv_gpu.json')

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
    """测量量化和恢复过程的平均耗时。"""
    for _ in range(WARMUP): fn()
    _sync(); start = time.perf_counter()
    for _ in range(ITERS): fn()
    _sync()
    return round((time.perf_counter() - start) * 1000 / ITERS, 4)

shape = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
evidence_level = 'environment_preflight' if RUN_MODE == 'dry_run' else 'gpu_simulation_on_real_kv_state'
result = {'stage': evidence_level, 'run_mode': RUN_MODE, 'runtime': runtime, 'config': {
    'shape': list(shape), 'kv_group_size': KV_GROUP_SIZE, 'warmup': WARMUP, 'iters': ITERS, 'seed': SEED, 'model_id': MODEL_ID,
}, 'evidence_level': evidence_level}
if RUN_MODE == 'dry_run':
    result['decision'] = {'decision': 'ready_to_measure', 'reason': '仅完成环境与配置检查，尚未运行 GPU KV Cache 测量。'}
else:
    # real_gpu 从真实模型的 past_key_values 读取 KV；cpu 模式保留小型确定性张量。
    if RUN_MODE == 'real_gpu':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
        inputs = tokenizer(PROMPT, return_tensors='pt', truncation=True, max_length=SEQ_LEN).to(device)
        actual_prompt_tokens = int(inputs['input_ids'].shape[1])
        with torch.no_grad(): outputs = model(**inputs, use_cache=True, return_dict=True)
        past = outputs.past_key_values
        if hasattr(past, 'to_legacy_cache'): past = past.to_legacy_cache()
        key, value = past[0][0], past[0][1]
        kv = torch.cat([key, value], dim=1).float()
        shape = tuple(kv.shape)
        del model, outputs, past, inputs
        if device.type == 'cuda': torch.cuda.empty_cache()
    else:
        kv = torch.randn(*shape, device=device)
    result['config'].update({'shape': list(shape), 'prompt_tokens': actual_prompt_tokens if RUN_MODE == 'real_gpu' else None,
                             'state_source': 'real_model_past_key_values' if RUN_MODE == 'real_gpu' else 'synthetic_cpu',
                             'layout_note': 'K/V concatenated on head axis for teaching'})
    sim = FP8KVCacheSim(kv_group_size=KV_GROUP_SIZE).to(device)
    def quantize_and_restore():
        sim.quantize_kv_cache(kv)
        return sim.dequantize_kv_cache()
    latency = _measure(quantize_and_restore)
    restored = sim.dequantize_kv_cache()
    raw_bytes = int(kv.numel() * kv.element_size())
    quant_bytes = int(sim.kv_q.numel() * sim.kv_q.element_size() + sim.kv_scale.numel() * sim.kv_scale.element_size())
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None
    result.update({'metrics': {'raw_bytes': raw_bytes, 'quantized_bytes_with_scale': quant_bytes,
        'compression_ratio': round(raw_bytes / quant_bytes, 4), 'mse': round(float(sim.mse(kv, restored)), 8),
        'quantize_restore_latency_ms': latency, 'peak_memory_mb': None if peak is None else round(peak, 2),
        'restored_shape': list(restored.shape)},
        'decision': {'decision': 'measure', 'reason': '仅观察 KV Cache 分组量化的容量、误差和恢复代价；不代表真实 serving 收益。'}})
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### GPU 实验结果记录

| 实验组 | shape | kv_group_size | 原始字节 | 量化字节（含 scale） | 压缩比 | MSE | 量化/恢复耗时 (ms) | peak memory (MB) | evidence level |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP16 KV Cache |  |  |  |  |  |  |  |  | gpu mechanism |
| 分组量化 KV Cache |  |  |  |  |  |  |  |  | gpu mechanism |

量化字节数包含 scale 元数据；真实 FP8 kernel、KV Cache serving 和端到端质量需要转到对应 backend 项目验证。
## 相关阅读

完成 FP8 scale、KV Cache 分组和误差检查后，可以继续阅读 FP8 格式、缓存调度和真实部署 backend。

- [FP8 原论文：FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [NVIDIA Transformer Engine 官方仓库](https://github.com/NVIDIA/TransformerEngine)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
