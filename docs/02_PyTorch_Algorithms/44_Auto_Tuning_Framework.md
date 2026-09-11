# 44. Auto Tuning Framework | 自动调优框架
**难度：** Medium | **环境：** CPU-first | **标签：** `显存优化`, `自动调优`, `基准测试` | **目标人群：** 显存优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/44_Auto_Tuning_Framework.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

自动调优框架的核心不是“把参数全都扫一遍”，而是先明确目标、约束和搜索空间，再用统一评价函数筛掉不值得继续跑的配置。前面的 profiling 和显存分析页更多是在回答“瓶颈在哪里”，而 `44` 进一步回答“既然已经知道瓶颈了，接下来该怎样把配置搜索变成一个可复用的决策流程”。

这一节用一个最小教学框架完成三步：筛掉违反约束的配置，用统一分数比较收益与代价，再输出下一轮最值得验证的候选。你会把 profiling 和显存账本中的观察量组织成“目标/约束 → feasible set → score → recommendation”的调优闭环，并把结果交给下一节的裁剪规划。

**关键词：** `search space`, `constraint`, `score`, `early stop`

---

## 前置阅读

**导语：** 进入本节前，先能读出 profiling 和显存分析中的约束，再观察哪些配置值得继续试、哪些应当尽早淘汰。
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [36. Decode Scheduling | Decode 调度](./36_Decode_Scheduling.md)
- [43. Unified Memory Management | 统一内存管理](./43_Unified_Memory_Management.md)

---

### Step 1: 先写清目标和约束

- 目标可能是更快、更省显存，或固定精度下的综合最优。
- 约束则可能是最大显存、最大延迟或最小精度阈值。
- 只有目标和约束都明确，搜索空间才有意义。

![自动调优总览](../public/02_PyTorch_Algorithms/44_auto_tuning_overview.svg)

### Step 2: 把搜索空间和评价函数写清楚

![自动调优的候选评估](../public/02_PyTorch_Algorithms/44_search_score.svg)

- 搜索空间至少包括配置名、关键超参和预估成本。
- 评价函数要同时考虑收益和约束违反情况。
- 先做最小版本，再决定是否扩展到更复杂策略。

### Step 3: 用早停和排序减少无效尝试

- 若某个配置明显违反约束，应尽早停止。
- 剩余配置再按统一分数排序，选出下一轮候选。

### Step 4: 动手实战

1. 补全 `filter_feasible_configs`，筛出满足约束的配置。
2. 补全 `score_tuning_config`，给单个配置打分。
3. 补全 `recommend_tuning_config`，输出推荐配置。

### 提示

- `TODO 1` 先按显存和延迟约束过滤，再返回满足条件的配置列表。
- `TODO 2` 先写一个最小打分公式，把吞吐、质量、显存和延迟合成一个统一分数。
- `TODO 3` 先筛 feasible，再在其中选最高分配置，最后返回推荐名和可行配置数。


```python
from typing import Dict, List

```


```python
def filter_feasible_configs(configs: List[Dict[str, float]], max_memory_gb: float, max_latency_ms: float) -> List[Dict[str, float]]:
    """
    TODO 1: 筛出满足约束的配置。
    """
    # 提示：只保留 memory_gb 和 latency_ms 都不超过约束的配置。
    # feasible = ???
    raise NotImplementedError


def score_tuning_config(config: Dict[str, float]) -> Dict[str, float]:
    """
    TODO 2: 给单个配置打分。
    """
    # 提示：先算 score，再返回 {'name': ..., 'score': ...}。
    # score = ???
    raise NotImplementedError


def recommend_tuning_config(configs: List[Dict[str, float]], max_memory_gb: float, max_latency_ms: float) -> Dict[str, object]:
    """
    TODO 3: 输出推荐配置。
    """
    # 提示：先拿到 feasible；如果为空直接返回 None 和 0；否则选分数最高的配置。
    # feasible = ???
    # best = ???
    raise NotImplementedError

```


```python
def test_auto_tuning_template():
    try:
        configs = [
            {'name': 'cfg_a', 'throughput': 110, 'memory_gb': 14.0, 'latency_ms': 85, 'quality': 0.94},
            {'name': 'cfg_b', 'throughput': 120, 'memory_gb': 16.0, 'latency_ms': 82, 'quality': 0.93},
            {'name': 'cfg_c', 'throughput': 108, 'memory_gb': 13.0, 'latency_ms': 95, 'quality': 0.96},
        ]
        feasible = filter_feasible_configs(configs, max_memory_gb=15.0, max_latency_ms=90.0)
        assert [cfg['name'] for cfg in feasible] == ['cfg_a']
        assert 'score' in score_tuning_config(configs[0])
        recommendation = recommend_tuning_config(configs, max_memory_gb=15.0, max_latency_ms=90.0)
        assert recommendation['recommended_name'] == 'cfg_a'
        assert recommendation['feasible_count'] == 1
        print('测试通过：自动调优框架模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_auto_tuning_template()

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
def filter_feasible_configs(configs: List[Dict[str, float]], max_memory_gb: float, max_latency_ms: float) -> List[Dict[str, float]]:
    """
    TODO 1: 筛出满足约束的配置。
    """
    # 提示：只保留 memory_gb 和 latency_ms 都不超过约束的配置。
    # feasible = ???
    return [
        config
        for config in configs
        if config.get('memory_gb', 10**9) <= max_memory_gb and config.get('latency_ms', 10**9) <= max_latency_ms
    ]


def score_tuning_config(config: Dict[str, float]) -> Dict[str, float]:
    """
    TODO 2: 给单个配置打分。
    """
    # 提示：先算 score，再返回 {'name': ..., 'score': ...}。
    # score = ???
    score = config.get('throughput', 0.0) + config.get('quality', 0.0) * 100 - config.get('memory_gb', 0.0) * 2 - config.get('latency_ms', 0.0) * 0.5
    return {'name': config.get('name', 'config'), 'score': score}


def recommend_tuning_config(configs: List[Dict[str, float]], max_memory_gb: float, max_latency_ms: float) -> Dict[str, object]:
    """
    TODO 3: 输出推荐配置。
    """
    # 提示：先拿到 feasible；如果为空直接返回 None 和 0；否则选分数最高的配置。
    # feasible = ???
    # best = ???
    feasible = filter_feasible_configs(configs, max_memory_gb=max_memory_gb, max_latency_ms=max_latency_ms)
    if not feasible:
        return {'recommended_name': None, 'feasible_count': 0}
    best = max(feasible, key=lambda config: score_tuning_config(config)['score'])
    return {'recommended_name': best.get('name', 'config'), 'feasible_count': len(feasible)}

```

### 解析

**1. TODO 1：筛出满足约束的配置**
- 先按 `memory_gb` 和 `latency_ms` 过滤，再保留满足约束的配置列表。
- 这一步的意义是先把明显不可能上线的配置挡掉，避免后续评分浪费在无效候选上。

**2. TODO 2：给单个配置打分**
- 打分函数把吞吐、质量、显存和延迟合成一个统一分数，用最小方式表达“收益和成本”的综合权衡。
- 这里的公式不是唯一答案，重点是先建立一个可比较、可排序的评价口径。

**3. TODO 3：输出推荐配置**
- 先拿到 feasible 配置；如果为空直接返回空推荐；否则从中选出分数最高的配置。
- 自动调优的核心不是“全扫一遍”，而是先过滤、再评分、最后推荐。

**4. 这页的定位**
- 自动调优框架最重要的是先过滤，再评分，最后推荐。
- 没有约束过滤的搜索，通常只是在扩大无效实验成本。

## 相关阅读

完成约束筛选和候选评分后，可以继续把搜索结果收束成具体资源决策，并进入端到端优化验证。

- [Optuna 官方文档](https://optuna.readthedocs.io/en/stable/)
- [Optuna 官方仓库](https://github.com/optuna/optuna)
- [45. Memory Cut Planning | 显存裁剪规划](./45_Memory_Cut_Planning.md)
- [49. Parallelism Strategy Selection | 并行策略选型](./49_Parallelism_Strategy_Selection.md)
- [74. Profiling Driven End-to-End Optimization | profiling 驱动优化项目](./74_Profiling_Driven_End_to_End_Optimization.md)
