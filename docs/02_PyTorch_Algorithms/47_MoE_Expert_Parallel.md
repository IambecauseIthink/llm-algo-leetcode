# 47. MoE Expert Parallel | MoE 专家并行

**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `MoE`, `Expert Parallel`, `All-to-All` | **目标人群：** 并行通信学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/47_MoE_Expert_Parallel.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

MoE 不是简单地“把参数变多”，它真正难的地方在于：token 被路由到不同专家后，专家往往分布在不同设备上，训练和推理都会引入额外通信、负载不均和 capacity 溢出。也就是说，MoE 一旦走到多设备阶段，它面对的就不再只是模型结构问题，而是一个典型的并行与通信问题。

这是一节**机制原理节**：它和 `06`、`07` 是前后承接关系。`06` 主讲 router 如何选专家，`07` 主讲负载均衡损失如何约束路由；而 `47` 开始回答另一个问题：当专家真的被分到不同设备上后，路由结果会怎样变成 dispatch、热点和 all-to-all 通信成本。

这一节用纯 Python / PyTorch 风格的最小模拟，把 expert load、capacity、overflow 和通信账本串起来。你会据此判断路由偏斜、overflow 和 all-to-all 开销是否已经抵消稀疏激活带来的收益。

**关键词：** `expert parallel`, `dispatch`, `all-to-all`

---

## 前置阅读

**导语：** 进入本节前，先能说明 Router 如何选择专家、负载均衡如何约束选择，再观察专家跨设备后如何产生 dispatch 和 all-to-all 通信。

- [06. MoE Router | MoE 路由器](./06_MoE_Router.md)
- [07. MoE Load Balancing Loss | MoE 负载均衡损失](./07_MoE_Load_Balancing_Loss.md)
- [46. Communication Profiling with NCCL | NCCL 通信 Profiling](./46_Communication_Profiling_with_NCCL.md)
---
### Step 1: 核心思想与痛点

MoE 的核心是“稀疏激活”：每个 token 只激活少量专家，所以理论上可以在较低计算代价下扩展参数规模。

但一旦专家分布在不同设备上，就会出现三个典型问题：
- token 必须被 dispatch 到对应设备上的专家；
- 不同专家收到的 token 数量可能差很多；
- 所有 token 计算完后，还要把结果 gather 回来。

所以 MoE 专家并行的核心不是“让更多专家跑起来”，而是“让专家分布、token 路由和通信代价同时可控”。

![MoE 专家并行总览](../public/02_PyTorch_Algorithms/47_expert_parallel_overview.svg)

### Step 2: 专家并行的实现框架

这一节先把最小结构拆清楚：

- `routes`：每个 token 命中的专家列表。
- `expert_loads`：每个 expert 实际收到多少路由。
- `capacity`：每个 expert 在给定 `capacity_factor` 下允许处理多少路由。
- `overflow`：超过 capacity 的路由数，代表溢出或丢弃风险。
- `dispatch_bytes` / `all_to_all_bytes`：近似估算通信代价。

这不是完整的并行系统，但足够用来判断 MoE expert parallel 的瓶颈在负载还是通信。

![MoE Dispatch 与 Combine 的数据流](../public/02_PyTorch_Algorithms/47_dispatch_combine_flow.svg)

#### 图解：token 如何进入专家并行

```text
token batch
   -> router 选 top-k experts
   -> dispatch 到各 device 上的 experts
   -> expert 计算
   -> gather 回原始 batch
```

如果某个 expert 拿到的 token 明显更多，就会出现热点；如果 dispatch 很频繁，就会被 all-to-all 通信拖慢。
### Step 3: 代价模型与边界

设 `A` 表示总路由次数，也就是 `sum(len(route_i))`。

- 平均负载：`A / num_experts`
- capacity：`ceil((A / num_experts) * capacity_factor)`
- overflow：`sum(max(load_e - capacity, 0))`
- dispatch 近似开销：`A * hidden_size * bytes_per_elem`
- all-to-all 近似开销：`2 * dispatch_bytes`

其中 `capacity_factor` 越大，越不容易溢出，但也会让每个 expert 预留更多空间；`hidden_size` 和 dtype 字节数越大，通信代价越重。
### Step 4: 动手实战

完成下面三个函数：

1. `summarize_expert_parallel`：汇总 expert 负载、capacity、overflow 和通信量。
2. `recommend_moe_parallel_plan`：根据负载与 overflow 给出 `accept / tune / reject`。
3. `compare_dense_vs_moe_cost`：给一个最小 dense vs MoE 的成本对比。

注意：这里要做的是**并行与通信的最小评估**，不是训练项目。
### 提示

- `routes` 里的每个元素表示一个 token 被路由到哪些专家；先把它展开统计成每个 expert 的负载。
- `TODO 1` 建议按这个顺序做：`total_tokens / total_routes -> expert_loads -> capacity -> overflow -> dispatch_bytes / all_to_all_bytes -> balance_ratio`。
- `capacity = ceil((total_routes / num_experts) * capacity_factor)`，`overflow` 只统计超出 capacity 的那部分路由数。
- `TODO 2` 不需要设计复杂策略，只要把 `accept / tune / reject` 写成最小规则判断即可。
- `TODO 3` 不是做真实性能模型，而是返回一个最小的 dense vs MoE 对比字典。

```python
from collections import Counter
from dataclasses import dataclass
import math

```


```python
@dataclass
class MoEParallelSummary:
    total_tokens: int
    total_routes: int
    expert_loads: list
    capacity: int
    overflow: int
    dispatch_bytes: int
    all_to_all_bytes: int
    balance_ratio: float


def summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0):
    """
    TODO 1:
    汇总 expert parallel 的最小账本，返回 MoEParallelSummary。
    需要包含 expert_loads、capacity、overflow、dispatch_bytes、all_to_all_bytes、balance_ratio。
    """
    # 提示：先统计 total_tokens / total_routes，再得到 expert_loads。
    # 提示：接着补出 capacity、overflow、dispatch_bytes、all_to_all_bytes、balance_ratio。
    # total_tokens = ???
    # total_routes = ???
    # expert_loads = ???
    # capacity = ???
    # overflow = ???
    # dispatch_bytes = ???
    # all_to_all_bytes = ???
    # balance_ratio = ???
    raise NotImplementedError


def recommend_moe_parallel_plan(summary: MoEParallelSummary) -> str:
    """
    TODO 2:
    根据 summary 的 overflow 和 balance_ratio，返回 'accept' / 'tune' / 'reject'。
    """
    # 提示：先处理空 summary，再判断 accept / tune，剩下的统一 reject。
    # if ???:
    #     return "accept"
    # if ???:
    #     return "tune"
    raise NotImplementedError


def compare_dense_vs_moe_cost(token_count, top_k, hidden_size, num_experts, bytes_per_elem=2):
    """
    TODO 3:
    返回一个最小 dense vs MoE 成本对比字典。
    """
    # 提示：先算 dense_compute_units 和 moe_routes，再补通信量与 sparsity_ratio。
    # dense_compute_units = ???
    # moe_routes = ???
    # moe_dispatch_bytes = ???
    # moe_all_to_all_bytes = ???
    # sparsity_ratio = ???
    raise NotImplementedError

```

### 测试

运行下面的测试，检查你的并行负载统计、容量判断和代价估算是否正确。

```python
def test_moe_expert_parallel():
    try:
        balanced = summarize_expert_parallel(
            routes=[[0, 1], [1, 2], [0, 2]],
            num_experts=3,
            hidden_size=4,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert balanced.total_tokens == 3
        assert balanced.total_routes == 6
        assert balanced.expert_loads == [2, 2, 2]
        assert balanced.capacity == 2
        assert balanced.overflow == 0
        assert balanced.dispatch_bytes == 48
        assert balanced.all_to_all_bytes == 96
        assert balanced.balance_ratio == 1.0
        assert recommend_moe_parallel_plan(balanced) == "accept"

        tuned = summarize_expert_parallel(
            routes=[[0], [0], [0], [1]],
            num_experts=2,
            hidden_size=8,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert tuned.total_routes == 4
        assert tuned.expert_loads == [3, 1]
        assert tuned.capacity == 2
        assert tuned.overflow == 1
        assert recommend_moe_parallel_plan(tuned) == "tune"

        rejected = summarize_expert_parallel(
            routes=[[0]] * 10 + [[1]],
            num_experts=2,
            hidden_size=8,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert rejected.total_routes == 11
        assert rejected.expert_loads == [10, 1]
        assert rejected.capacity == 6
        assert rejected.overflow == 4
        assert rejected.balance_ratio > 1.8
        assert recommend_moe_parallel_plan(rejected) == "reject"

        empty = summarize_expert_parallel(
            routes=[],
            num_experts=2,
            hidden_size=8,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert empty.total_tokens == 0
        assert empty.total_routes == 0
        assert empty.expert_loads == [0, 0]
        assert empty.capacity == 0
        assert empty.overflow == 0
        assert recommend_moe_parallel_plan(empty) == "reject"

        cost = compare_dense_vs_moe_cost(token_count=4, top_k=2, hidden_size=8, num_experts=4)
        assert cost["dense_compute_units"] == 32
        assert cost["moe_routes"] == 8
        assert cost["moe_dispatch_bytes"] == 128
        assert cost["moe_all_to_all_bytes"] == 256
        assert cost["sparsity_ratio"] == 0.5
        print("moe expert parallel passed")
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e


test_moe_expert_parallel()
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
from collections import Counter
from dataclasses import dataclass
import math

@dataclass
class MoEParallelSummary:
    total_tokens: int
    total_routes: int
    expert_loads: list
    capacity: int
    overflow: int
    dispatch_bytes: int
    all_to_all_bytes: int
    balance_ratio: float


def summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0):
    """
    TODO 1:
    汇总 expert parallel 的最小账本，返回 MoEParallelSummary。
    需要包含 expert_loads、capacity、overflow、dispatch_bytes、all_to_all_bytes、balance_ratio。
    """
    # 提示 1: 先校验 num_experts，再遍历 routes 统计每个 expert 的路由数。
    # 提示 2: capacity = ceil((total_routes / num_experts) * capacity_factor)。
    # 提示 3: dispatch_bytes = total_routes * hidden_size * bytes_per_elem，all_to_all_bytes = dispatch_bytes * 2。
    # 提示 4: balance_ratio = max(expert_loads) / avg_load；如果 avg_load 为 0，则返回 0.0。
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    counts = Counter()
    total_tokens = len(routes)
    total_routes = 0
    for token_routes in routes:
        for expert_id in token_routes:
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError("expert id out of range")
            counts[expert_id] += 1
            total_routes += 1

    expert_loads = [counts.get(i, 0) for i in range(num_experts)]
    avg_load = total_routes / num_experts if num_experts else 0.0
    capacity = math.ceil(avg_load * capacity_factor) if total_routes else 0
    overflow = sum(max(load - capacity, 0) for load in expert_loads)
    dispatch_bytes = total_routes * hidden_size * bytes_per_elem
    all_to_all_bytes = dispatch_bytes * 2
    balance_ratio = (max(expert_loads) / avg_load) if avg_load > 0 else 0.0

    return MoEParallelSummary(
        total_tokens=total_tokens,
        total_routes=total_routes,
        expert_loads=expert_loads,
        capacity=capacity,
        overflow=overflow,
        dispatch_bytes=dispatch_bytes,
        all_to_all_bytes=all_to_all_bytes,
        balance_ratio=round(balance_ratio, 3),
    )


def recommend_moe_parallel_plan(summary: MoEParallelSummary) -> str:
    """
    TODO 2:
    根据 summary 的 overflow 和 balance_ratio，返回 'accept' / 'tune' / 'reject'。
    """
    # 提示 1: 空 summary 直接 reject。
    # 提示 2: 无 overflow 且负载足够均匀时 accept。
    # 提示 3: 少量 overflow 且负载尚可时 tune，其余 reject。
    if summary.total_routes == 0:
        return "reject"
    if summary.overflow == 0 and summary.balance_ratio <= 1.2:
        return "accept"
    if summary.overflow <= 1 and summary.balance_ratio <= 1.8:
        return "tune"
    return "reject"


def compare_dense_vs_moe_cost(token_count, top_k, hidden_size, num_experts, bytes_per_elem=2):
    """
    TODO 3:
    返回一个最小 dense vs MoE 成本对比字典。
    """
    # 提示 1: dense_compute_units = token_count * hidden_size。
    # 提示 2: moe_routes = token_count * top_k。
    # 提示 3: sparsity_ratio 可以写成 top_k / num_experts；num_experts 为 0 时返回 0.0。
    dense_compute_units = token_count * hidden_size
    moe_routes = token_count * top_k
    moe_dispatch_bytes = moe_routes * hidden_size * bytes_per_elem
    moe_all_to_all_bytes = moe_dispatch_bytes * 2
    return {
        "dense_compute_units": dense_compute_units,
        "moe_routes": moe_routes,
        "moe_dispatch_bytes": moe_dispatch_bytes,
        "moe_all_to_all_bytes": moe_all_to_all_bytes,
        "sparsity_ratio": round(top_k / num_experts, 3) if num_experts else 0.0,
    }

```

### 解析

**1. TODO 1：汇总 expert parallel 的最小账本**
- 先校验 `num_experts`，再遍历 `routes` 统计每个 expert 的路由数。
- 然后按 `capacity_factor` 计算 `capacity`，再汇总 `overflow`、`dispatch_bytes`、`all_to_all_bytes` 和 `balance_ratio`。
- 这一部分的核心不是模拟真实分布式 kernel，而是建立最小并行通信账本，帮助你看清“负载是否偏、容量是否溢出、通信是否变贵”。

**2. TODO 2：给出 `accept / tune / reject` 决策**
- `accept`：没有 overflow，且负载足够均匀，说明当前 expert parallel 计划基本成立。
- `tune`：有少量 overflow 或轻度失衡，说明方案还能救，但需要继续调 capacity、路由策略或 expert 布局。
- `reject`：空 summary、overflow 过多或负载明显偏热，说明这套并行方案不适合直接采用。

**3. TODO 3：比较 dense vs MoE 的最小成本**
- `dense_compute_units` 用最小方式表示 dense 路线的计算规模；`moe_routes` 和通信量则表示稀疏激活带来的并行代价。
- `sparsity_ratio = top_k / num_experts` 反映了每个 token 实际激活专家数占总专家数的比例。
- 这不是精确性能模型，而是一个最小比较框架，帮助你快速判断 MoE 的稀疏收益是否可能抵得过额外通信。

**4. 这一页的边界**
- 它讲的是 MoE 专家并行和通信代价。
- 它不负责完整训练闭环，也不负责项目级 benchmark。
## 相关阅读

完成 expert load、capacity、overflow 和 all-to-all 账本后，可以继续阅读 MoE 论文、分布式实现与真实 benchmark。

- [Switch Transformers 原论文](https://arxiv.org/abs/2101.03961)
- [Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)
- [79. Distributed Parallel Benchmark | 分布式并行 Benchmark](./79_Distributed_Parallel_Benchmark.md)
- [80. MoE Expert Parallel Benchmark | MoE 专家并行 Benchmark](./80_MoE_Expert_Parallel_Benchmark.md)
- [81. Distributed Inference Project | 分布式推理项目](./81_Distributed_Inference_Project.md)
