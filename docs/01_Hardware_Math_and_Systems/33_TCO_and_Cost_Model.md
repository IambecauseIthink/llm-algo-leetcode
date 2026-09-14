# 33. TCO and Cost Model | 算力评估与 TCO 模型

**难度：** Medium | **环境：** CPU-first | **标签：** `系统工程`, `成本评估`, `TCO` | **目标人群：** 硬件约束学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

算力采购和部署决策，真正难的地方从来不是找到最低报价，而是把采购、能耗、运维、迁移、停机和替换风险放进同一张长期账本里看。只比较单卡价格，很容易把后续运行成本和系统切换代价完全漏掉。

这一页在整个教程的纵向主线里属于 `Part 01` 的系统决策收口页，不是某条主学习路线的核心机制页，但会给后续硬件选型、部署和长期维护补一层总账判断。学完这里，后面再看 `67` 以及 `Part 04` 的系统优化页时，你会更容易把成本判断和硬件选型、编译适配、长期维护放回同一条决策链；如果这里没学明白，后面很容易只按采购价格或单次 benchmark 做决策，而忽略能耗、运维、迁移和停机风险这些会持续累积的长期成本。按专题归类，这一页更接近全教程的系统背景收口，也和 `编译与图优化专题`、`推理优化路线` 共享一部分部署判断视角。

**关键词：** `TCO`, `cost`, `power`

---
## 前置阅读

**导语：** 这一页先把编译、选型和成本判断放到同一条判断链里，再看 TCO 为什么不能只看单卡报价。

- [09. AI Compilers and Graph Optimization | AI 编译器与计算图优化](./09_AI_Compilers_and_Graph_Optimization.md)
- [10. AI Chips Overview and Alternatives | 算力现状与替代方案](./10_Domestic_AI_Chips_Overview.md)
- [32. TVM / MLIR Deep Practice | TVM / MLIR 深度实践](./32_TVM_MLIR_Deep_Practice.md)

## 相关阅读

**导语：** 如果想继续把成本、动态 shape 和虚拟化带来的长期影响串起来，可以接着看这些页。

- [29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度](./29_CUDA_Stream_Advanced_Scheduling.md)
- [30. Dynamic Shape Handling | 动态 Shape 处理](./30_Dynamic_Shape_Handling.md)
- [31. GPU Virtualization and MIG | GPU 虚拟化与 MIG](./31_GPU_Virtualization_and_MIG.md)
---
## Q1：TCO 为什么比单卡报价更重要？

<details><summary>点击展开查看解析</summary>

TCO 看的是总拥有成本，不只是买卡的钱。

它会把采购、能耗、运维、迁移、停机风险和替换成本都放进来。对于长期运行的系统来说，单卡便宜不一定代表整体便宜，因为后续的电费、维护和迁移可能更贵。

所以真正要比较的是一段时间内谁更能稳定地把业务跑起来。
</details>
### Q1小验证：为什么长期成本更关键

看总账，而不是只看首付。

```python
def tco(purchase, power_per_month, ops_per_month, migration, months=12):
    return purchase + (power_per_month + ops_per_month) * months + migration

gpu_a = tco(120, 3, 2, 10, months=24)
gpu_b = tco(90, 5, 4, 22, months=24)
print('A:', gpu_a)
print('B:', gpu_b)
print('cheaper:', 'A' if gpu_a < gpu_b else 'B')
```

## Q2：为什么迁移成本和风险要算进模型里？

<details><summary>点击展开查看解析</summary>

硬件切换不只是换设备，还会带来软件适配、调试、重构和性能回归风险。

如果迁移需要重新适配编译器、kernel、通信库或部署流程，那这些工作本身就是成本的一部分。

所以 TCO 不是财务问题和技术问题的二选一，而是把两者放进同一个判断框架。
</details>
### Q2小验证：迁移为什么会贵

适配成本本身就是成本。

```python
def migration_cost(adapt, test, rollback):
    return adapt + test + rollback

print(migration_cost(5, 3, 2))
```

## Q3：怎样用成本模型做更合理的选型？

<details><summary>点击展开查看解析</summary>

做选型时，最好把成本、风险和收益放到同一个表里。

常见比较维度包括：
- 初始采购成本；
- 运行时电力和运维成本；
- 软件栈成熟度；
- 迁移和维护风险；
- 未来扩展的稳定性。

这样得到的不是“最低价”，而是“最适合当前约束的方案”。
</details>
### Q3小验证：选型判断先看什么

先看长期约束，再看短期价格。

```python
def score_option(total_cost, risk, growth):
    # 分数越低越好：成本、风险和增长不确定性一起看。
    return total_cost + risk * 20 + growth * 10

options = {
    'stable_gpu': score_option(60, 0.2, 0.3),
    'cheap_alt': score_option(45, 0.6, 0.7),
}
for name, score in options.items():
    print(name, score)
print('choose:', min(options, key=options.get))
```

## Q4：为什么扩容节奏和折旧周期也会改变 TCO 结论？

<details><summary>点击展开查看解析</summary>

TCO 不只是“买得起还是用得起”，还取决于你要在多长时间内把这套系统跑满。

如果扩容太慢，前期买来的算力可能闲置，单位产出成本会被拉高；如果折旧周期太短，硬件还没充分发挥就要更新，账面成本也会迅速上升。

因此真正的成本模型，往往要同时考虑：
- 初始采购成本；
- 运行成本；
- 迁移和切换成本；
- 扩容与折旧节奏。

把这些变量放在一起看，才更接近真实的技术选型决策。
</details>
### Q4小验证：扩容和折旧为什么要一起看

```python
def effective_cost(purchase, ops, migration, idle, depreciation):
    return purchase + ops + migration + idle + depreciation

scenarios = {
    'fast_expand': effective_cost(100, 20, 8, 20, 10),
    'slow_expand': effective_cost(100, 20, 8, 5, 28),
}
for name, cost in scenarios.items():
    print(name, cost)
print('more cost effective:', min(scenarios, key=scenarios.get))

```

## ⚠️ 常见误区

- 只看采购价会漏掉长期成本。
- 迁移和运维不是附带项，它们会真实影响 TCO。
- 选型不是找最便宜，而是找最适合约束的方案。
- 成本模型应同时覆盖技术和业务风险。