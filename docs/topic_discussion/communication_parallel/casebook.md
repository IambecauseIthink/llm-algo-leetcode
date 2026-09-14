# 通信与并行正文

这页只做并行问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先判断单卡边界和切分对象，再区分同步、状态驻留、层/算子切分与动态路由，最后用 profiling 和 benchmark 检查通信代价。不要从并行方法名反推系统一定会加速。

## 判断表

先分清问题在通信原语、状态分摊、层切分、算子切分还是 benchmark 验证，再判断收益是不是被同步等待和调度代价吞掉了。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 卡数加了，但速度没有明显变好 | `communication bound` | [01](./01_why_parallel_and_communication.md), [02](./02_data_parallel_and_synchronization.md) | 看 AllReduce、同步等待、拓扑 |
| 显存回来了，但训练节奏更差 | `state sharding trade-off` | [03](./03_state_sharding_and_zero.md) | 比较 DDP、FSDP、ZeRO 的状态代价 |
| Pipeline 跑起来了，但气泡很大 | `pipeline scheduling mismatch` | [04](./04_pipeline_and_tensor_parallel.md) | 调 micro-batch、阶段划分和时序 |
| Tensor Parallel 能跑，但通信代价太高 | `tensor split overhead` | [04](./04_pipeline_and_tensor_parallel.md) | 看切分粒度和通信频率 |
| benchmark 好看，但真实收益不稳 | `validation gap` | [06](./06_benchmark_and_parallel_decision.md) | 回到 workload 和热点验证 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| AllReduce / 拓扑 | 通信是不是主要瓶颈 | 多卡天然就线性加速 |
| 状态分摊 | 显存是不是靠切状态换回来的 | 显存下降就等于整体更优 |
| pipeline 气泡 | 时间是不是浪费在流水线空转上 | 只看吞吐，不看阶段利用率 |
| tensor split | 切分是不是把通信频率抬太高 | 会切分就等于值得切分 |
| benchmark | 并行收益是否真的成立 | 只看单次数字，不看 workload 一致性 |

## 本节要点

这页的职责不是列并行方法名，而是把并行选型里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小决策模板

记录 `单卡瓶颈 -> 切分对象 -> 通信模式 -> 等待/负载不均 -> 单卡与多卡对照 -> 决策`。至少同时保留显存、吞吐/延迟、通信占比和扩展效率。
