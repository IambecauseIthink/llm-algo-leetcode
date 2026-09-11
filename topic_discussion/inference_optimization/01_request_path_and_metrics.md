# 01. Request Path and Metrics | 请求链路与指标口径

## 页面目标

从一次最小推理请求开始，沿着 `Prefill`、`KV Cache` 和 `Decode` 走一遍，认识 `TTFT`、`TPOT`、throughput、P99 和 peak memory。完成请求阶段与指标的映射后，你应该能够说明请求经过了哪些阶段，把阶段连接到可观察指标，并根据现象判断下一步该看哪类机制。需要补充 Attention 基础时，可先阅读 [04 Attention（MHA / GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)。

## 核心机制

请求可以先拆成 `Prefill`、`Decode` 和输出三个主要处理阶段；`KV Cache` 不是独立的串行阶段，而是由 Prefill 写入、由 Decode 持续读取和追加的状态对象。`TTFT` 是首 token 延迟，`TPOT` 是后续 token 的平均生成间隔；吞吐、P99 和峰值显存分别帮助我们观察系统产出、尾部延迟和资源占用。下图把主要阶段、状态对象与指标放在同一条链路中。

不同阶段关注的指标不同：

![请求链路与指标定位](../../docs/public/topic_discussion/inference_optimization/request_lifecycle_zh.svg)

| 指标 | 它回答的问题 | 主要关联阶段 |
|:---|:---|:---|
| `TTFT` | 首个 token 多久返回？ | prefill、Attention、prompt length |
| `TPOT` | 后续 token 多久生成一个？ | decode、KV Cache、调度 |
| `throughput` | 单位时间生成多少 token？ | batching、调度、生成策略 |
| `peak memory` | 运行期间最高占用多少显存？ | 权重、KV Cache、batch、量化 |
| `P50 / P95 / P99` | 典型请求和慢请求的延迟是多少？ | 排队、batch、调度、系统抖动 |

参考入口：指标定义可看 [vLLM Metrics](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md)；Serving 实现可看 [vLLM](https://github.com/vllm-project/vllm)。

## 判断框架

比较前先固定 `workload`，也就是一组可复现的输入和运行条件：模型、backend（实际执行推理的框架或服务）、Prompt（输入文本）、generated tokens（输出长度）、batch（一次并行处理的请求数）、并发、dtype（计算数据类型）和 cache policy（缓存管理方式）。然后根据现象选择下一步：

| 观察到的现象 | 优先怀疑的瓶颈 | 下一步 |
|:---|:---|:---|
| 长 prompt 使 `TTFT` 明显升高 | `prefill-bound`（输入处理受限）或 Attention 访存 | 进入 `02` |
| `TPOT` 高、生成阶段占比大 | `decode-bound`（逐 token 生成受限）、KV Cache 或调度 | 进入 `03` / `04` |
| peak memory 接近预算、batch 上不去 | `memory-bound`（显存或访存受限）或 Cache 容量 | 进入 `04` / `05` |
| 多请求时 P99 明显升高 | 排队、batch 组织或调度 | 进入 `04` / `06` |
| 没有明显单点瓶颈 | 需要端到端比较 | 进入 `06` |
