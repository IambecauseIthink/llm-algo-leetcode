# 52. Alignment Conflicts and Thresholds | 对齐冲突与阈值
**难度：** Medium | **环境：** CPU-first | **标签：** `后训练对齐`, `评测`, `阈值决策` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/52_Alignment_Conflicts_and_Thresholds.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

对齐里最危险的误判之一，是看到单一指标上涨就默认模型“变好了”。偏好胜率、格式稳定性、安全性和任务完成度并不总是同向变化；如果这些指标开始互相冲突，而团队又没有事先定义最低阈值和优先级，在线更新或偏好优化就很容易把模型推向一个看似更强、实际上更不稳的方向。

这是一节**机制判断节**：`50` 先把偏好数据和评测口径立住，`51` 再把离线优化推进到在线反馈闭环，而 `52` 继续回答另一个更接近上线的问题: 当多个指标不一致时，到底以什么边界做最终判断。学完后，你应该能识别常见的对齐冲突、给关键指标设定最低阈值，并把“训练有收益”与“可以上线”这两个结论明确区分开。

**关键词：** `reward conflict`, `evaluation drift`, `safety threshold`

---

## 前置阅读

- [15. DPO Loss Tutorial | DPO 损失教程](./15_DPO_Loss_Tutorial.md)
- [50. Preference Data and Evaluation | 偏好数据与评测](./50_Preference_Data_and_Evaluation.md)
- [51. Online DPO | 在线 DPO](./51_Online_DPO.md)

## 相关阅读

**导语：** 学完对齐冲突与阈值后，下一步重点不是继续背指标名称，而是看这些冲突判断怎样进入偏好优化项目和在线 benchmark，确认“指标上涨”是否真的还能满足上线边界。
- [84. DPO Preference Project | DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [85. GRPO Groupwise Alignment Project | GRPO 组内对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)
- [86. DPO Online Benchmark | DPO 在线基准](./86_DPO_Online_Benchmark.md)

---

### Step 1: 先识别评测和奖励是否一致

- 区分偏好胜率、格式稳定性、安全性和人工主观质量。
- 如果奖励信号和最终评测目标不一致，对齐训练会很容易偏航。

### Step 2: 明确冲突和阈值

- 可能的冲突包括：胜率上升但安全性下降，或格式更稳但任务完成度下降。
- 上线前必须定义最低阈值，避免单一指标掩盖坏结果。

### Step 3: 判断是否值得继续单独展开

- 如果冲突类型和阈值判断已经超出当前对齐主线，就值得升级成独立补页。

### Step 4: 动手实战

1. 补全 `summarize_alignment_metrics`，汇总多种对齐指标。
2. 补全 `detect_alignment_conflict`，识别冲突。
3. 补全 `recommend_alignment_followup`，输出是否继续展开。

### 提示

- 这页不是让你做完整对齐评测框架，而是先固定三步判断：关键指标是什么、是否已经出现冲突、这类冲突是否已经值得单独扩页。
- `TODO 1` 只需要把输入里的关键指标收出来，不要在这里引入额外聚合逻辑。
- `TODO 2` 先判断安全分是否低于阈值，再判断偏好胜率是否过低，按这个顺序输出冲突类型。
- `TODO 3` 先看是否存在冲突，再给出是否值得继续展开的结论和原因。


```python
from typing import Dict

```


```python
def summarize_alignment_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    TODO 1: 汇总最关键的对齐指标。
    """
    # 提示：只需要返回 win_rate、safety_score、format_score 这 3 个字段。
    # win_rate = ???
    # safety_score = ???
    # format_score = ???
    raise NotImplementedError


def detect_alignment_conflict(summary: Dict[str, float], safety_threshold: float) -> Dict[str, object]:
    """
    TODO 2: 判断是否出现对齐冲突。
    """
    # 提示：先判断 safety_score 是否低于 safety_threshold，
    # 再判断 win_rate 是否低于 0.5，最后返回无冲突结果。
    raise NotImplementedError


def recommend_alignment_followup(conflict: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续扩成独立对齐页。
    """
    # 提示：先看 has_conflict，再给出 needs_dedicated_page 和 reason。
    # needs_dedicated_page = ???
    # reason = ???
    raise NotImplementedError

```


```python
def test_alignment_conflicts_and_thresholds():
    try:
        summary = summarize_alignment_metrics({'win_rate': 0.63, 'safety_score': 0.58, 'format_score': 0.91})
        assert summary['win_rate'] == 0.63
        assert summary['safety_score'] == 0.58
        conflict = detect_alignment_conflict(summary, safety_threshold=0.6)
        assert conflict['has_conflict'] is True
        assert conflict['conflict_type'] == 'safety_below_threshold'
        decision = recommend_alignment_followup(conflict)
        assert decision['needs_dedicated_page'] is True
        print('测试通过：对齐冲突与阈值页面模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_alignment_conflicts_and_thresholds()

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
def summarize_alignment_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    TODO 1: 汇总最关键的对齐指标。
    """
    # 提示：只需要返回 win_rate、safety_score、format_score 这 3 个字段。
    # win_rate = ???
    # safety_score = ???
    # format_score = ???
    return {
        'win_rate': metrics.get('win_rate', 0.0),
        'safety_score': metrics.get('safety_score', 0.0),
        'format_score': metrics.get('format_score', 0.0),
    }


def detect_alignment_conflict(summary: Dict[str, float], safety_threshold: float) -> Dict[str, object]:
    """
    TODO 2: 判断是否出现对齐冲突。
    """
    # 提示：先判断 safety_score 是否低于 safety_threshold，
    # 再判断 win_rate 是否低于 0.5，最后返回无冲突结果。
    if summary.get('safety_score', 0.0) < safety_threshold:
        return {'has_conflict': True, 'conflict_type': 'safety_below_threshold'}
    if summary.get('win_rate', 0.0) < 0.5:
        return {'has_conflict': True, 'conflict_type': 'preference_win_rate_too_low'}
    return {'has_conflict': False, 'conflict_type': 'none'}


def recommend_alignment_followup(conflict: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续扩成独立对齐页。
    """
    # 提示：先看 has_conflict，再给出 needs_dedicated_page 和 reason。
    # needs_dedicated_page = ???
    # reason = ???
    needs_page = conflict.get('has_conflict', False)
    reason = '对齐指标之间已经出现独立冲突，需要单独展开' if needs_page else '当前对齐边界仍可由现有主线覆盖'
    return {'needs_dedicated_page': needs_page, 'reason': reason}

```

### 解析

TODO 1：`summarize_alignment_metrics` 先把最关键的对齐指标固定下来。这里不追求复杂聚合，而是确保后续所有冲突判断都基于同一份最小指标摘要。

TODO 2：`detect_alignment_conflict` 负责把“指标变好了但结论可能更差”具体化。这里先检查安全阈值，再检查偏好胜率，把冲突类型从模糊印象变成明确判断。

TODO 3：`recommend_alignment_followup` 用来判断这条对齐冲突链路是否已经复杂到值得独立扩页。如果冲突已经真实出现，就说明这不再只是评测备注，而是一条独立的上线边界与对齐分析主题。
