# 45. Memory Cut Planning | 显存裁剪规划
**难度：** Medium | **环境：** CPU-first | **标签：** `显存优化`, `预算规划`, `裁剪策略` | **目标人群：** 显存优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/45_Memory_Cut_Planning.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

显存优化里最容易被忽略的一步，不是再找一个新技巧，而是先决定预算不够时到底该先裁什么。很多训练任务不是因为完全没有优化手段而跑不通，而是因为峰值来源没分清、预算余量没留够、裁剪顺序写得不够明确，最后在 checkpoint、offload、batch size 和模型规模之间来回试错。

本节把前面的显存账本、统一预算和候选评分收束成一份可执行的 `cut order`。学完后，你应该能判断峰值来自哪里、预算要留多少 `headroom`，以及资源不足时应按什么顺序调整 activation、临时 buffer、batch size 或模型规模。

**关键词：** `peak memory`, `buffer`, `headroom`, `cut order`

---

## 前置阅读

**导语：** 进入本节前，先能从显存账本识别峰值来源，并能读出运行时预算，再决定哪一类对象应当优先调整。
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [19. Activation Checkpointing and Activation Offload | 激活检查点与激活卸载](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [43. Unified Memory Management | 统一内存管理](./43_Unified_Memory_Management.md)

---

### Step 1: 先识别峰值来自哪里

- 区分参数、激活、KV cache 和临时缓冲的峰值贡献。
- 不要只看总显存，要看峰值是否集中在少数阶段。
- 只有峰值来源清楚，裁剪顺序才可能合理。

![显存裁剪规划总览](../public/02_PyTorch_Algorithms/45_memory_cut_overview.svg)

### Step 2: 明确预算留边和裁剪顺序

- 显存方案不能把预算用到 100%，要留出运行时 headroom。
- 当预算不够时，先裁哪类对象，应该提前写清楚。
- 这一步直接决定后续实验是“可运行”还是“频繁 OOM”。

### Step 3: 输出是否值得继续单独展开

- 如果峰值来源和裁剪顺序已经复杂到影响多个页面，就值得升级成专门的显存专题补页。

![显存裁剪动作的选择](../public/02_PyTorch_Algorithms/45_cut_decision_matrix.svg)

### Step 4: 动手实战

1. 补全 `summarize_peak_sources`，识别峰值来源。
2. 补全 `plan_memory_cuts`，给出裁剪顺序。
3. 补全 `recommend_memory_followup`，输出是否继续展开。

### 提示

- 这页不是让你做完整显存建模，而是先把“最大峰值来源”“可裁剪顺序”“是否值得单独扩页”这三步判断固定下来。
- `TODO 1` 只需要找出 `peak_gb` 最大的组件，并返回它的名字和峰值。
- `TODO 2` 先筛出 `cuttable=True` 的组件，再按 `peak_gb` 从大到小挑出真正需要执行的裁剪步骤，顺手计算裁剪后的估计峰值。
- `TODO 3` 先判断是否真的存在裁剪动作，再结合 `estimated_peak_gb` 给出是否值得继续展开的结论。


```python
from typing import Dict, List

```


```python
def summarize_peak_sources(components: List[Dict[str, float]]) -> Dict[str, object]:
    """
    TODO 1: 找出峰值最大的显存来源。
    """
    # 提示：可以先用 max 找到 peak_gb 最大的组件，再返回它的 name 和 peak_gb。
    # largest = ???
    raise NotImplementedError


def plan_memory_cuts(components: List[Dict[str, float]], target_peak_gb: float) -> Dict[str, object]:
    """
    TODO 2: 给出显存裁剪顺序并估计裁剪后的峰值。
    """
    # 提示：先筛出 cuttable=True 的组件并按 peak_gb 降序排序，
    # 再从 total_peak 开始依次扣减，把真正执行过的组件写进 cut_order，直到不高于 target_peak_gb。
    # cut_candidates = ???
    # total_peak = ???
    # cut_order = ???
    # reduced_peak = ???
    raise NotImplementedError


def recommend_memory_followup(summary: Dict[str, object], plan: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续扩成独立显存页。
    """
    # 提示：先判断 cut_order 是否非空，再结合 estimated_peak_gb 给出 needs_dedicated_page 和 reason。
    # needs_dedicated_page = ???
    # reason = ???
    raise NotImplementedError

```


```python
def test_memory_cut_planning():
    try:
        components = [
            {'name': 'weights', 'peak_gb': 10.0, 'cuttable': False},
            {'name': 'activations', 'peak_gb': 8.0, 'cuttable': True},
            {'name': 'temporary_buffers', 'peak_gb': 3.0, 'cuttable': True},
        ]
        summary = summarize_peak_sources(components)
        assert summary['largest_source'] == 'weights'
        assert summary['peak_gb'] == 10.0
        plan = plan_memory_cuts(components, target_peak_gb=15.0)
        assert plan['cut_order'] == ['activations']
        assert plan['estimated_peak_gb'] == 13.0
        decision = recommend_memory_followup(summary, plan)
        assert decision['needs_dedicated_page'] is True
        print('测试通过：显存裁剪规划页面模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_memory_cut_planning()

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
def summarize_peak_sources(components: List[Dict[str, float]]) -> Dict[str, object]:
    """
    TODO 1: 找出峰值最大的显存来源。
    """
    # 提示：可以先用 max 找到 peak_gb 最大的组件，再返回它的 name 和 peak_gb。
    # largest = ???
    largest = max(components, key=lambda item: item.get('peak_gb', 0.0))
    return {'largest_source': largest.get('name', ''), 'peak_gb': largest.get('peak_gb', 0.0)}


def plan_memory_cuts(components: List[Dict[str, float]], target_peak_gb: float) -> Dict[str, object]:
    """
    TODO 2: 给出显存裁剪顺序并估计裁剪后的峰值。
    """
    # 提示：先筛出 cuttable=True 的组件并按 peak_gb 降序排序，
    # 再从 total_peak 开始依次扣减，把真正执行过的组件写进 cut_order，直到不高于 target_peak_gb。
    # cut_candidates = ???
    # total_peak = ???
    # cut_order = ???
    # reduced_peak = ???
    cut_candidates = sorted(
        [component for component in components if component.get('cuttable', False)],
        key=lambda item: item.get('peak_gb', 0.0),
        reverse=True,
    )
    total_peak = sum(component.get('peak_gb', 0.0) for component in components)
    cut_order = []
    reduced_peak = total_peak
    for component in cut_candidates:
        if reduced_peak <= target_peak_gb:
            break
        reduced_peak -= component.get('peak_gb', 0.0)
        cut_order.append(component.get('name', ''))
    return {'cut_order': cut_order, 'estimated_peak_gb': max(reduced_peak, 0.0)}


def recommend_memory_followup(summary: Dict[str, object], plan: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续扩成独立显存页。
    """
    # 提示：先判断 cut_order 是否非空，再结合 estimated_peak_gb 给出 needs_dedicated_page 和 reason。
    # needs_dedicated_page = ???
    # reason = ???
    needs_page = bool(plan.get('cut_order')) and plan.get('estimated_peak_gb', 0.0) <= 15.0
    reason = '峰值来源和裁剪顺序已经形成独立判断链路' if needs_page else '当前显存边界仍可由已有页面覆盖'
    return {'needs_dedicated_page': needs_page, 'reason': reason}

```

### 解析

TODO 1：`summarize_peak_sources` 先把“谁在制造最大峰值”说清楚。只有峰值来源被显式识别出来，后面才知道应该优先优化激活、临时缓冲，还是别的对象。

TODO 2：`plan_memory_cuts` 把“显存不够”从一句模糊结论变成可执行顺序。这里先筛出可裁剪组件，再按峰值贡献从大到小尝试裁剪，只把真正执行过的步骤写进 `cut_order`，并估计裁剪后的剩余峰值。

TODO 3：`recommend_memory_followup` 负责判断这条显存链路是否已经复杂到值得独立扩页。如果已经形成稳定的峰值来源判断和裁剪顺序，就说明它不再只是零散备注，而是一页完整的显存规划主题。

## 相关阅读

完成 `cut order` 后，可以把裁剪顺序带回真实瓶颈分析和预算压缩项目，检查它是否转化为稳定收益。

- [PyTorch CUDA 内存管理文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [PagedAttention 原论文：Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [74. Profiling Driven End-to-End Optimization | profiling 驱动优化项目](./74_Profiling_Driven_End_to_End_Optimization.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
