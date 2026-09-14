# 16. GRPO Loss Tutorial | 群体相对策略优化损失教程

**难度：** Medium-Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `GRPO`, `奖励优化` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在偏好优化里，模型常常会针对同一个 prompt 生成多个候选答案。此时我们不只关心某个答案的绝对奖励，更关心它在同一组候选里相对好还是相对差。

GRPO 的重点就是把这种“组内比较”变成训练信号：先把同组奖励归一化成相对优势，再用类似 PPO 的裁剪目标限制策略更新幅度，从而减少对显式 Critic 的依赖。本节实现一个简化版 GRPO Loss。完成后，你应该能看懂 `group_ids`、相对优势和策略比率裁剪各自解决什么问题，并把 `RLHF -> DPO -> GRPO` 这条对齐链路串起来。

**关键词：** `GRPO`, `group relative`, `reward`

---
## 前置阅读

**导语：** 先理解训练闭环、偏好优化和显存账本，再看 GRPO 如何用组内相对优势稳定策略更新。

- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [P0: 20. Profiling and Memory Ledger | 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)
- [15. DPO Loss Tutorial | 直接偏好优化损失教程](../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.md)


## 相关阅读

**导语：** 完成对齐损失后，可以继续从性能分析和通信基础理解大规模对齐训练的工程瓶颈。

- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)

---
### Step 1: 核心思想

> **为什么需要 GRPO？**
> GRPO 关注的是同一组样本内部的相对优劣，而不是把每个样本都单独拉到一个绝对奖励空间里。
> 这种做法的好处是：
> - 训练目标更稳，减少极端奖励对更新方向的冲击。
> - 可以和策略比率裁剪一起使用，限制一次更新的幅度。
> - 在某些场景下可以减少对显式 Critic 的依赖。

#### 组内比较为什么更稳

GRPO 的关键不是“再造一个 PPO 变体”，而是把一个 prompt 下的多个候选答案放在同一个组里比较，从而降低奖励尺度波动带来的训练不稳定。

| 维度 | 单样本 reward | 组内相对 reward |
|:---|:---|:---|
| 评价方式 | 直接看绝对奖励 | 看同组里谁更好 |
| 稳定性 | 容易受奖励尺度影响 | 先中心化再标准化，方差更小 |
| 数据组织 | 一条样本就能训练 | 需要同一 prompt 下的多个候选 |
| 训练直觉 | 绝对分数高就更新 | 组内相对更优的样本获得更强信号 |

#### 一个最小例子

假设同一个 prompt 生成了 4 个候选：

- `group 0`: reward = `[1.0, 2.0]`
- `group 1`: reward = `[0.5, 1.5]`

GRPO 不直接拿这 4 个分数去做全局比较，而是分别在各自组内做均值和标准差归一化。这样做的结果是：

- 组内平均值被拉回 0，避免组与组之间的绝对奖励尺度干扰更新。
- 高于组均值的候选得到正优势，低于组均值的候选得到负优势。
- 策略更新更关注“同一个 prompt 下谁更好”，而不是“不同 prompt 的 reward 绝对值有多大”。

这也是 GRPO 适合做组内排序、候选比较和生成优化的原因。
### Step 2: 数学形式

给定同一组中的奖励 $r_i$，先计算组内均值和标准差：

$$
\bar r = \frac{1}{N} \sum_{i=1}^{N} r_i, \quad\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i - \bar r)^2 + \epsilon}
$$

然后把相对优势定义为：

$$
A_i = \frac{r_i - \bar r}{\sigma}
$$

最后再代入类似 PPO 的 clipped objective：

$$
L = -\mathbb{E}[\min(ratio \cdot A, clip(ratio) \cdot A)]
$$
这一节的实现链路就是先做组内归一化，再构造 clipped surrogate，最后汇总成 GRPO loss。

### Step 3: 代码实现框架与任务拆解

这一节的实现顺序很简单：先把同一组候选的奖励做组内归一化，再构造策略比率和 clipped surrogate，最后汇总成 GRPO loss。

#### 实现顺序

1. `advantages`：按 `group_ids` 分组，把 reward 做去均值和标准差归一化。
2. `ratio / surr1 / surr2`：再算策略比率，并构造两个 surrogate 目标。
3. `loss`：最后取更保守的一侧，得到最终的 GRPO loss。

#### 实现节奏

- 如果 `advantages` 的组内中心化错了，后面的 loss 没有意义。
- 如果 `ratio` 或 `clamp` 口径错了，GRPO 就会退化成错误的策略更新。
- 如果 `loss` 没有取 `min(surr1, surr2)`，就失去了 clipped objective 的稳定性。

```python
import torch

```


```python
def compute_grpo_loss(log_probs_new, log_probs_old, rewards, group_ids, clip_range=0.2, eps=1e-6):
    """
    简化版 GRPO Loss。
    rewards/group_ids 允许把同一 prompt 下的多个候选答案分到一组。
    """
    # ==========================================
    # TODO 1: 计算组内相对优势
    # ==========================================
    # advantages = ???
    
    # ==========================================
    # TODO 2: 计算策略比率与两个 surrogate 目标
    # ==========================================
    # ratio = ???
    # surr1 = ???
    # surr2 = ???
    
    # ==========================================
    # TODO 3: 计算最终 loss 并返回
    # ==========================================
    # loss = ???
    return loss, advantages

```


```python
# 运行此单元格以测试你的实现
def test_grpo_loss():
    try:
        log_new = torch.tensor([-1.0, -0.5, -1.5, -0.2], requires_grad=True)
        log_old = torch.tensor([-1.1, -0.4, -1.6, -0.3])
        rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
        group_ids = torch.tensor([0, 0, 1, 1])
        loss, adv = compute_grpo_loss(log_new, log_old, rewards, group_ids)
        assert loss.ndim == 0, "Loss 应该是标量"
        assert torch.isfinite(loss), "Loss 不能是 NaN/Inf"
        assert torch.allclose(adv[group_ids == 0].mean(), torch.tensor(0.0), atol=1e-6), "组内优势均值应接近 0"
        loss.backward()
        assert log_new.grad is not None, "梯度没有回传到策略分数"
        print("✅ 测试通过！GRPO 简化版 Loss 可运行。")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_grpo_loss()

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
import torch

def compute_grpo_loss(log_probs_new, log_probs_old, rewards, group_ids, clip_range=0.2, eps=1e-6):
    # ==========================================
    # TODO 1: 计算组内相对优势
    # ==========================================
    # advantages = ???
    advantages = torch.zeros_like(rewards)
    for gid in group_ids.unique(sorted=True):
        mask = group_ids == gid
        group_rewards = rewards[mask]
        centered = group_rewards - group_rewards.mean()
        denom = group_rewards.std(unbiased=False).clamp_min(eps)
        advantages[mask] = centered / denom

    # ==========================================
    # TODO 2: 计算策略比率与两个 surrogate 目标
    # ==========================================
    # ratio = ???
    ratio = torch.exp(log_probs_new - log_probs_old)
    # surr1 = ???
    surr1 = ratio * advantages
    # surr2 = ???
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages

    # ==========================================
    # TODO 3: 计算最终 loss 并返回
    # ==========================================
    # loss = ???
    loss = -torch.min(surr1, surr2).mean()
    return loss, advantages

```

### 解析

**1. TODO 1: 计算组内相对优势**

- **实现方式**：按 `group_ids` 把同组奖励聚合，再做去均值和标准差归一化。
- **代码核心**：`advantages[mask] = centered / denom`
- **数学含义**：这里的优势是“组内相对值”，不是单样本的绝对奖励。
- **工程意义**：把同一 prompt 下多个候选答案放在一起比较，可以减少不同样本尺度差异。

**2. TODO 2: 计算策略比率与两个 surrogate 目标**

- **实现方式**：先算 `ratio = exp(log_probs_new - log_probs_old)`，再构造 `surr1` 和 `surr2`。
- **代码核心**：`surr1 = ratio * advantages`，`surr2 = clamp(ratio) * advantages`
- **数学含义**：这一步沿用了 PPO 的 clipped objective 思路，用两个 surrogate 限制单步更新幅度。
- **工程意义**：既允许模型朝更好的组内排序移动，又避免策略比率变化过大。

**3. TODO 3: 计算最终 loss 并返回**

- **实现方式**：`loss = -torch.min(surr1, surr2).mean()`
- **代码核心**：`torch.min` 取更保守的 surrogate 估计，再对 batch 求均值。
- **数学含义**：这是一个偏悲观的优化目标，避免模型过度相信单次更新带来的收益。
- **工程意义**：把组内相对优势和策略裁剪结合起来，形成一个稳定的简化版 GRPO loss。

**进阶思考**

- 为什么 GRPO 通常不需要显式 Critic？
- 如果把组内归一化换成全局归一化，会发生什么？
- 这个实现和 PPO 的 clipped surrogate 有哪些本质相同与不同？
