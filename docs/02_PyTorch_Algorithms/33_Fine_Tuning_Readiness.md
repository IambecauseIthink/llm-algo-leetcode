# 33. Fine Tuning Readiness | 微调 Readiness
**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `微调准备`, `可行性检查` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

`33` 关注微调实验开始前最容易被忽略的 readiness 判断：训练计划是否写清楚了，数据规模和预算是否匹配，评测目标是否真的对应当前训练范围。`32` 先回答数据是不是已经整理到可训练状态，而 `33` 再进一步回答：即使数据可用，这次微调是否真的已经具备启动条件，还是还停留在“想法上可做、工程上还没准备好”的阶段。

这一节把微调启动前的判断整理成三个可执行问题：训练计划是否明确，时间和显存预算是否可承受，评测目标是否值得进入完整项目。你会把前面的 LoRA 方案、数据入口和训练闭环汇总成“计划 → 预算 → 是否立项 → 项目验证”的 readiness 链路。

**关键词：** `training plan`, `budget`, `evaluation`, `readiness`

---

## 前置阅读

**导语：** 进入本节前，先能说明训练如何运行、数据如何进入，以及评测指标要回答什么问题，再据此检查一项微调任务是否具备启动条件。
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [32. Data Engineering for SFT | SFT 数据工程](./32_Data_Engineering_for_SFT.md)

---

### Step 1: 先把训练计划写清楚

- 固定模型、数据规模、训练步数、batch size 和评测窗口。
- 明确目标是提升 loss、格式稳定性，还是特定任务指标。
- 如果计划本身写不清楚，后面的实验结论大概率不可复用。

### Step 2: 把资源预算和训练范围对齐

- 训练闭环至少要对齐数据规模、训练时长和显存预算。
- 如果目标比预算大很多，应优先缩范围，而不是直接开始训练。
- 先看“是否能完成”，再看“能否做得更好”。

### Step 3: 用 readiness 判断是否进入项目页

- 预留页的价值在于提前判断是否值得继续扩成项目。
- 只有计划、预算和评测都达标，才应该推进到完整项目验证。

### Step 4: 动手实战

1. 补全 `summarize_training_plan`，汇总训练计划。
2. 补全 `check_training_budget`，判断预算是否够用。
3. 补全 `recommend_finetuning_scope`，输出是否进入项目页。

### 提示

- 这页不是让你设计完整训练系统，而是先做一次最小 readiness 检查：计划是否清楚，预算是否够，是否值得继续升级成项目页。
- `TODO 1` 先把训练计划里的核心字段收出来，不要在这里扩展额外策略。
- `TODO 2` 只比较时间预算和显存预算是否满足，最后再合成 `budget_ok`。
- `TODO 3` 先看目标是否存在、数据规模是否大于 0、预算是否通过，再决定是否 `promote_to_project`。


```python
from typing import Dict, List

```


```python
def summarize_training_plan(config: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 1: 汇总训练计划的最小信息。
    """
    # 提示：只需要返回 goal、dataset_size、total_steps、eval_every 这 4 个字段。
    # goal = ???
    # dataset_size = ???
    # total_steps = ???
    # eval_every = ???
    raise NotImplementedError


def check_training_budget(config: Dict[str, object], available_hours: float, available_memory_gb: float) -> Dict[str, object]:
    """
    TODO 2: 判断训练预算是否足够。
    """
    # 提示：先分别判断 estimated_hours 和 peak_memory_gb 是否在预算内，
    # 再合成 budget_ok = time_ok and memory_ok。
    # time_ok = ???
    # memory_ok = ???
    # budget_ok = ???
    raise NotImplementedError


def recommend_finetuning_scope(summary: Dict[str, object], budget: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否进入项目页。
    """
    # 提示：先判断 goal 是否存在、dataset_size 是否大于 0、budget_ok 是否为 True，
    # 再给出 promote_to_project 和 reason。
    # promote = ???
    # reason = ???
    raise NotImplementedError

```


```python
def test_finetuning_readiness_template():
    try:
        config = {'goal': 'improve_instruction_following', 'dataset_size': 20000, 'total_steps': 1200, 'eval_every': 100, 'estimated_hours': 6.0, 'peak_memory_gb': 14.0}
        summary = summarize_training_plan(config)
        assert summary['goal'] == 'improve_instruction_following'
        assert summary['dataset_size'] == 20000
        budget = check_training_budget(config, available_hours=8.0, available_memory_gb=16.0)
        assert budget['budget_ok'] is True
        decision = recommend_finetuning_scope(summary, budget)
        assert decision['promote_to_project'] is True
        print('测试通过：微调 readiness 页面模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_finetuning_readiness_template()

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
def summarize_training_plan(config: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 1: 汇总训练计划的最小信息。
    """
    # 提示：只需要返回 goal、dataset_size、total_steps、eval_every 这 4 个字段。
    # goal = ???
    # dataset_size = ???
    # total_steps = ???
    # eval_every = ???
    return {
        'goal': config.get('goal', ''),
        'dataset_size': int(config.get('dataset_size', 0)),
        'total_steps': int(config.get('total_steps', 0)),
        'eval_every': int(config.get('eval_every', 0)),
    }


def check_training_budget(config: Dict[str, object], available_hours: float, available_memory_gb: float) -> Dict[str, object]:
    """
    TODO 2: 判断训练预算是否足够。
    """
    # 提示：先分别判断 estimated_hours 和 peak_memory_gb 是否在预算内，
    # 再合成 budget_ok = time_ok and memory_ok。
    # time_ok = ???
    # memory_ok = ???
    # budget_ok = ???
    time_ok = float(config.get('estimated_hours', 0.0)) <= available_hours
    memory_ok = float(config.get('peak_memory_gb', 0.0)) <= available_memory_gb
    return {'time_ok': time_ok, 'memory_ok': memory_ok, 'budget_ok': time_ok and memory_ok}


def recommend_finetuning_scope(summary: Dict[str, object], budget: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否进入项目页。
    """
    # 提示：先判断 goal 是否存在、dataset_size 是否大于 0、budget_ok 是否为 True，
    # 再给出 promote_to_project 和 reason。
    # promote = ???
    # reason = ???
    promote = bool(summary.get('goal')) and summary.get('dataset_size', 0) > 0 and budget.get('budget_ok', False)
    reason = '训练计划和预算已对齐' if promote else '训练计划或预算尚未就绪'
    return {'promote_to_project': promote, 'reason': reason}

```

### 解析

TODO 1：`summarize_training_plan` 先把训练计划里的最小字段固定下来，只保留 `goal`、`dataset_size`、`total_steps` 和 `eval_every`，让后面的预算判断和项目判断都建立在同一份计划摘要上。

TODO 2：`check_training_budget` 负责把时间预算和显存预算拆开判断。先分别得到 `time_ok` 和 `memory_ok`，再合成 `budget_ok`，避免把“预算不足”混成一个模糊结论。

TODO 3：`recommend_finetuning_scope` 负责输出是否值得升级成项目页。只有当目标存在、数据规模有效、预算通过时，才建议 `promote_to_project = True`，把这页真正变成一张 readiness 决策页，而不是空泛占位页。

## 相关阅读

完成训练计划、预算和评测目标检查后，可以继续阅读训练框架文档与项目页，把 readiness 结果落实到真实微调实验。

- [Hugging Face Transformers Trainer 文档](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)
- [Hugging Face Accelerate 官方仓库](https://github.com/huggingface/accelerate)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](./60_LoRA_Fine_Tuning_Project.md)
- [62. Instruction Fine-Tuning Project | 指令微调项目](./62_Instruction_Fine_Tuning_Project.md)
- [2.3](./2_3.md)
