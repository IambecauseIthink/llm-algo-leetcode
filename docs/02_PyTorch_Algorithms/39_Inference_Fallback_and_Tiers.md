# 39. Inference Fallback and Tiers | 推理分层与回退策略
**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `推理服务`, `回退策略` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/39_Inference_Fallback_and_Tiers.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

`39` 聚焦 serving 决策里最容易被经验化处理的问题：请求应该怎么分层，资源紧张时先退化什么，以及什么情况下这套规则已经复杂到值得单独展开。`38` 先回答系统要不要把 prefill 和 decode 拆池，而 `39` 则进一步回答：即使系统结构已经确定，面对不同请求层次和资源压力时，具体应该先牺牲什么、保留什么。

这一节不实现完整 serving control plane，而是先把 fallback 规则收敛成三个最小判断：请求流量如何分层、资源紧张时最小退化动作是什么、这套规则是否已经复杂到值得继续扩成正式实现页或 benchmark。学完后，你应该能看清“流量分层 -> 退化策略 -> 是否继续扩页”这条判断链，而不是把线上退化行为留给临场经验处理。

**关键词：** `request mix`, `latency`, `throughput`, `fallback`

---

## 前置阅读

- [36. Decode Scheduling | Decode 调度](./36_Decode_Scheduling.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)
- [38. Prefill-Decode Disaggregation | PD 分离](./38_Prefill_Decode_Disaggregation.md)

## 相关阅读

- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [68. Speculative Decoding Benchmark | 推测解码基准](./68_Speculative_Decoding_Benchmark.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)

---

### Step 1: 先把请求类型分层

- 区分长 prompt、短 prompt、高生成长度和高并发请求。
- 不同请求层次的瓶颈不一样，不能共用同一套最优策略。
- 先把流量分层，后面的调度和回退策略才有意义。

### Step 2: 明确退化策略和优先级

- 当资源紧张时，系统要知道先牺牲什么。
- 退化策略可能是降 batch、关缓存、切回 baseline 或延后某类请求。
- 这些规则如果不提前写清楚，线上行为会很不稳定。

### Step 3: 输出是否需要新专题页

- 预留页的目标是判断这个问题是否已经复杂到值得独立展开。
- 如果指标边界和退化规则都清楚，就可以进一步升级为正式实现页或 benchmark。

### Step 4: 动手实战

1. 补全 `summarize_inference_tiers`，统计请求层次。
2. 补全 `select_fallback_policy`，根据资源压力选择退化策略。
3. 补全 `recommend_inference_followup`，输出是否值得继续展开。

### 提示

- `TODO 1` 先把请求按 `prompt_heavy / decode_heavy / balanced` 分层，再统计数量。
- `TODO 2` 只需要根据 `memory_pressure` 和 `queue_delay_ms` 选择一个最小退化策略。
- `TODO 3` 先判断当前请求分层和 fallback 是否已经复杂到值得扩成独立页，再写出原因。


```python
from typing import Dict, List

```


```python
def summarize_inference_tiers(requests: List[Dict[str, int]], long_prompt_threshold: int, long_decode_threshold: int) -> Dict[str, int]:
    """
    TODO 1: 统计请求层次。
    """
    # 提示：先创建结果字典，再根据 prompt/decode 长度把请求分到三类里。
    # result = ???
    # if ???:
    #     result['prompt_heavy'] += 1
    # elif ???:
    #     result['decode_heavy'] += 1
    # else:
    #     result['balanced'] += 1
    raise NotImplementedError


def select_fallback_policy(system_state: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 2: 根据资源压力选择退化策略。
    """
    # 提示：先判断 memory_pressure，再判断 queue_delay_ms，最后返回默认策略。
    # if ???:
    #     return {'policy': ???, 'priority': ???}
    raise NotImplementedError


def recommend_inference_followup(summary: Dict[str, int], fallback: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续展开。
    """
    # 提示：先判断 needs_dedicated_page，再给出对应 reason。
    # needs_dedicated_page = ???
    # reason = ???
    raise NotImplementedError

```


```python
def test_inference_reserved_template():
    try:
        requests = [
            {'prompt_tokens': 3000, 'decode_tokens': 64},
            {'prompt_tokens': 128, 'decode_tokens': 512},
            {'prompt_tokens': 800, 'decode_tokens': 128},
        ]
        summary = summarize_inference_tiers(requests, long_prompt_threshold=2048, long_decode_threshold=256)
        assert summary == {'prompt_heavy': 1, 'decode_heavy': 1, 'balanced': 1}
        fallback = select_fallback_policy({'memory_pressure': 0.92, 'queue_delay_ms': 140})
        assert fallback['policy'] == 'disable_optional_cache'
        decision = recommend_inference_followup(summary, fallback)
        assert decision['needs_dedicated_page'] is True
        print('测试通过：推理预留页模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_inference_reserved_template()

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
def summarize_inference_tiers(requests: List[Dict[str, int]], long_prompt_threshold: int, long_decode_threshold: int) -> Dict[str, int]:
    """
    TODO 1: 统计请求层次。
    """
    # 提示：先创建结果字典，再根据 prompt/decode 长度把请求分到三类里。
    # result = ???
    # if ???:
    #     result['prompt_heavy'] += 1
    # elif ???:
    #     result['decode_heavy'] += 1
    # else:
    #     result['balanced'] += 1
    result = {'prompt_heavy': 0, 'decode_heavy': 0, 'balanced': 0}
    for request in requests:
        prompt_tokens = request.get('prompt_tokens', 0)
        decode_tokens = request.get('decode_tokens', 0)
        if prompt_tokens > long_prompt_threshold and decode_tokens <= long_decode_threshold:
            result['prompt_heavy'] += 1
        elif decode_tokens > long_decode_threshold and prompt_tokens <= long_prompt_threshold:
            result['decode_heavy'] += 1
        else:
            result['balanced'] += 1
    return result


def select_fallback_policy(system_state: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 2: 根据资源压力选择退化策略。
    """
    # 提示：先判断 memory_pressure，再判断 queue_delay_ms，最后返回默认策略。
    # if ???:
    #     return {'policy': ???, 'priority': ???}
    if system_state.get('memory_pressure', 0.0) > 0.9:
        return {'policy': 'disable_optional_cache', 'priority': 'protect_residency'}
    if system_state.get('queue_delay_ms', 0.0) > 100:
        return {'policy': 'reduce_batch_size', 'priority': 'protect_latency'}
    return {'policy': 'keep_current_plan', 'priority': 'stable'}


def recommend_inference_followup(summary: Dict[str, int], fallback: Dict[str, object]) -> Dict[str, object]:
    """
    TODO 3: 输出是否值得继续展开。
    """
    # 提示：先判断 needs_dedicated_page，再给出对应 reason。
    # needs_dedicated_page = ???
    # reason = ???
    needs_page = (summary.get('prompt_heavy', 0) + summary.get('decode_heavy', 0)) > 0 and fallback.get('policy') != 'keep_current_plan'
    reason = '请求分层和退化策略已经足够复杂，值得单独展开' if needs_page else '当前边界仍可由现有推理页覆盖'
    return {'needs_dedicated_page': needs_page, 'reason': reason}

```

### 解析

**1. TODO 1：统计请求层次**
- 先根据 `prompt_tokens` 和 `decode_tokens` 把请求分成 `prompt_heavy / decode_heavy / balanced` 三类。
- 这一步的目标是先看流量有没有明显分层，因为不同层次的请求通常不能共用同一套最优 serving 策略。

**2. TODO 2：根据资源压力选择退化策略**
- 先看 `memory_pressure`，再看 `queue_delay_ms`，最后再回到默认策略。
- 这一步回答的是“资源紧张时系统先牺牲什么”，把退化规则从隐含经验变成显式策略。

**3. TODO 3：输出是否值得继续展开**
- 先判断当前请求分层和 fallback 是否已经复杂到需要独立专题页，再给出对应原因。
- 预留页的核心不是完成大系统，而是判断这个问题是否已经值得升级成正式实现页或 benchmark。

**4. 这页的定位**
- 推理预留页重点不是再造一个技巧，而是把系统边界和退化策略写清楚。
- 只有请求层次和回退规则都成型，才值得继续扩成独立实现页。
