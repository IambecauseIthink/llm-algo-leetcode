# 49. Parallelism Strategy Selection | 并行策略选型
**难度：** Medium | **环境：** CPU-first | **标签：** `并行通信`, `并行策略`, `策略选择` | **目标人群：** 并行通信学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

`49` 聚焦并行路线里最容易停留在“方案很多，但不知道先试哪个”的问题：当前主要瓶颈到底是什么、哪些并行组合值得优先验证、什么时候这条问题链已经值得拆成独立选型页。它把并行原理进一步收敛成可执行的选型顺序。

**关键词：** `dp`, `tp`, `pp`, `ep`, `bottleneck shift`

---

## 前置阅读

**导语：** 进入本节前，先能区分 DP、TP、PP 和 EP 的切分对象与通信代价，再根据瓶颈选择优先验证的组合。
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](./27_ZeRO_Optimizer_Sim.md)
- [28. Pipeline Parallelism MicroBatch | Pipeline 并行与 MicroBatch](./28_Pipeline_Parallelism_MicroBatch.md)
- [29. Tensor Parallelism Sim | Tensor Parallelism 模拟](./29_Tensor_Parallelism_Sim.md)

---

### Step 1: 先识别主要瓶颈在哪一层

- 先区分参数显存、激活显存、通信量和流水线气泡。
- 如果瓶颈判断错了，并行切分方案通常会把问题从一处挪到另一处。

![并行策略选择总览](../public/02_PyTorch_Algorithms/49_parallel_strategy_overview.svg)

### Step 2: 把并行组合写成显式候选

- 候选可能是 `DP+ZeRO`、`TP+PP`、`EP+DP` 等组合。
- 每个组合都要绑定自己的主要收益和副作用。
- 先写清候选，后面才能谈 benchmark 和项目验证。

![并行候选的收益与代价](../public/02_PyTorch_Algorithms/49_strategy_tradeoff.svg)

### Step 3: 判断是否值得单独展开

- 如果并行组合已经超出当前 2.9 主线的解释范围，就值得继续拆页。

### Step 4: 动手实战

1. 补全 `summarize_parallel_bottlenecks`，识别主要瓶颈。
2. 补全 `rank_parallel_candidates`，给并行候选排序。
3. 补全 `recommend_parallel_followup`，输出是否继续展开。

### 提示

- 这页不是让你做完整并行自动搜索，而是先固定三步判断：当前主瓶颈是谁、候选方案怎么排优先级、这条选型链路是否已经值得单独扩页。
- `TODO 1` 只需要从指标里找出数值最大的瓶颈，并记录它的严重程度。
- `TODO 2` 只要按 `score` 从高到低给候选方案排序，不需要在这里展开多目标搜索。
- `TODO 3` 先判断瓶颈是否足够明显，再结合候选数量，给出是否值得继续展开的结论。


```python
from typing import Dict, List

```


```python
def summarize_parallel_bottlenecks(metrics: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 1: 找出当前最主要的并行瓶颈。
    """
    # 提示：可以直接从 metrics.items() 里找 value 最大的一项，返回瓶颈名字和严重程度。
    # primary, severity = ???
    raise NotImplementedError


def rank_parallel_candidates(candidates: List[Dict[str, float]]) -> List[str]:
    """
    TODO 2: 按优先级给并行候选排序。
    """
    # 提示：先按 score 从高到低排序，再只返回每个候选的 name。
    # ranked = ???
    raise NotImplementedError


def recommend_parallel_followup(summary: Dict[str, object], ranked: List[str]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续扩成独立并行选型页。
    """
    # 提示：先判断 severity 是否超过阈值，再结合 ranked 的长度给出 needs_dedicated_page 和 reason。
    # needs_dedicated_page = ???
    # reason = ???
    raise NotImplementedError

```


```python
def test_parallel_strategy_selection():
    try:
        summary = summarize_parallel_bottlenecks({'memory_pressure': 0.85, 'comm_overhead': 0.40, 'pipeline_bubble': 0.10})
        assert summary['primary_bottleneck'] == 'memory_pressure'
        assert summary['severity'] == 0.85
        ranked = rank_parallel_candidates([
            {'name': 'dp_zero', 'score': 0.72},
            {'name': 'tp_pp', 'score': 0.64},
            {'name': 'ep_dp', 'score': 0.68},
        ])
        assert ranked == ['dp_zero', 'ep_dp', 'tp_pp']
        decision = recommend_parallel_followup(summary, ranked)
        assert decision['needs_dedicated_page'] is True
        print('测试通过：并行策略选型页面模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_parallel_strategy_selection()

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
def summarize_parallel_bottlenecks(metrics: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 1: 找出当前最主要的并行瓶颈。
    """
    # 提示：可以直接从 metrics.items() 里找 value 最大的一项，返回瓶颈名字和严重程度。
    # primary, severity = ???
    primary, severity = max(metrics.items(), key=lambda item: item[1])
    return {'primary_bottleneck': primary, 'severity': severity}


def rank_parallel_candidates(candidates: List[Dict[str, float]]) -> List[str]:
    """
    TODO 2: 按优先级给并行候选排序。
    """
    # 提示：先按 score 从高到低排序，再只返回每个候选的 name。
    # ranked = ???
    return [item.get('name', '') for item in sorted(candidates, key=lambda item: item.get('score', 0.0), reverse=True)]


def recommend_parallel_followup(summary: Dict[str, object], ranked: List[str]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续扩成独立并行选型页。
    """
    # 提示：先判断 severity 是否超过阈值，再结合 ranked 的长度给出 needs_dedicated_page 和 reason。
    # needs_dedicated_page = ???
    # reason = ???
    needs_page = summary.get('severity', 0.0) > 0.3 and len(ranked) >= 2
    reason = '并行瓶颈和候选组合已经形成独立选型问题' if needs_page else '当前并行边界仍可由 2.9 主线覆盖'
    return {'needs_dedicated_page': needs_page, 'reason': reason}

```

### 解析

TODO 1：`summarize_parallel_bottlenecks` 先回答“当前最明显的并行瓶颈是什么”。只有瓶颈定位准确，后面的并行组合排序才不会变成拍脑袋试方案。

TODO 2：`rank_parallel_candidates` 负责把候选方案转成一个明确优先级列表。这里先不做复杂搜索，只要求把已有候选按 `score` 排序，形成最小可执行的实验顺序。

TODO 3：`recommend_parallel_followup` 用来判断这条并行选型链路是否已经复杂到值得独立扩页。如果瓶颈足够明显、候选组合也不止一个，就说明它已经是稳定的选型问题，而不只是分散在多页里的补充说明。

## 相关阅读

完成 DP、TP、PP 和 EP 的切分对象与通信代价比较后，可以继续阅读并行框架实现和分布式基准，验证候选方案的排序是否可靠。

- [Megatron-LM 原论文](https://arxiv.org/abs/2104.04473)
- [DeepSpeed 官方仓库](https://github.com/microsoft/DeepSpeed)
- [79. Distributed Parallel Benchmark | 分布式并行基准](./79_Distributed_Parallel_Benchmark.md)
- [80. MoE Expert Parallel Benchmark | MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
