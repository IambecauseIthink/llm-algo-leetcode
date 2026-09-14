# 通信与并行深入阅读

假设你把训练从单卡扩到多卡：显存确实回来了，但速度没有按预期提升，甚至有时还更慢。接下来你会开始怀疑是同步太重、切分不对，还是 benchmark 本身就没说明白。

这条线最重要的是按暴露顺序判断：系统先为什么走向并行，切分以后代价从哪里回来，最后哪些策略真的值得保留。

对应专题正文：[01 为什么需要并行与通信](./01_why_parallel_and_communication.md)。先定义单卡边界和预期收益，再决定是否值得跨卡。

## 第一段：并行的起点不是“想上多卡”，而是单卡先碰到了边界

故事通常从单卡显存或吞吐先顶到边界开始。系统先从最朴素的 DDP 出发，但一旦跨卡，通信就开始回来索取代价。

这一步对应 [01 为什么需要并行与通信](./01_why_parallel_and_communication.md) 和 [02 数据并行与梯度同步](./02_data_parallel_and_synchronization.md)。

## 第二段：先分清是同步问题还是状态问题

如果卡数加了、速度却没明显上去，第一步不是继续切更多层，而是先分清：问题在 AllReduce 和同步等待，还是在参数、梯度、优化器状态的驻留方式。也就是先区分 DDP，还是已经需要 FSDP / ZeRO。

这一步对应 [02 数据并行与梯度同步](./02_data_parallel_and_synchronization.md) 和 [03 状态切分与 ZeRO](./03_state_sharding_and_zero.md)。

## 第三段：状态分摊之后，才轮到层切分和算子切分

如果状态分摊已经不能解决问题，才会继续进入 Pipeline 或 Tensor Parallel。这里真正要看的不是“有没有切”，而是切了以后气泡、同步和通信频率有没有把收益重新吞掉。

这一步对应 [04 Pipeline 与 Tensor Parallel](./04_pipeline_and_tensor_parallel.md)；如果模型含有动态专家路由，还要进入 [05 Expert Parallel 与通信热点](./05_expert_parallel_and_communication_hotspots.md)。

## 第四段：最后必须回到热点和 benchmark

真正的收口不在“用了哪种并行方法”，而在通信热点、等待时间和 benchmark 是否证明它值得保留。把这条故事走完以后，一个更像真实结论的说法通常不是“我们做了并行”，而是：瓶颈先在同步，再到状态分摊，再到层切分，最终被接受的是那组在通信代价上真正站得住的切分组合。

这一步对应 [06 Benchmark 与并行决策](./06_benchmark_and_parallel_decision.md)，最终报告必须能解释扩展收益和通信代价。
