# 大模型结构和原理正文

这页只做结构问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先确认 token 和 decoder 主干，再定位 norm、attention、RoPE、MLP 与 MoE 在 block 中的位置，最后用代表模型和 61 项目验证结构差异。不要从模型名称直接跳到结论。

## 判断表

先分清问题出在 token / embedding、norm、attention、RoPE、MLP、MoE 还是真实模型实现差异，再判断它会影响训练、推理还是显存判断。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 知道模块名字，但说不清在 block 哪一段 | `block placement mismatch` | [01](./01_transformer_decoder.md), [06](./06_block_residual_path.md) | 先画清 hidden state 在 block 里的路径 |
| attention 会算，但结构关系不清楚 | `attention structure mismatch` | [04](./04_attention_evolution.md), [05](./05_rope_position_encoding.md) | 看 Q/K/V 关系、RoPE 位置、head 组织 |
| MLP / SwiGLU 只会背名词 | `ffn mismatch` | [07](./07_mlp_ffn_evolution.md) | 看 gate / up / down 的职责 |
| MoE 看起来只是“大一点的 MLP” | `moe mismatch` | [09](./09_moe_sparsity_evolution.md) | 看 router、Top-K、负载均衡和 expert 路由 |
| 真实模型源码和教程图对不上 | `implementation gap` | [08](./08_representative_models.md) | 看 norm 位置、命名、局部 trick 和变体 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| block 路径 | hidden state 到底怎么穿过结构主干 | 只背组件名，不看顺序 |
| norm / residual | 稳定性和主干信号怎么维持 | 把 norm 当成独立小技巧 |
| attention / RoPE | 上下文交互和位置信息怎么进入 | RoPE 只是额外 patch |
| MLP / MoE | dense 和 sparse 路径分别怎么扩展容量 | MoE 只是多几个 FFN |
| 代表模型 | 不同模型到底改了哪里 | 模型名不同就等于结构完全不同 |

## 本节要点

这页的职责不是重复组件定义，而是把结构理解里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小结构审计模板

记录 `输入表示 -> block 路径 -> 组件替换 -> dense / sparse 分支 -> 真实模型差异 -> 可观测指标`。至少说明结构变化影响的是训练稳定性、推理成本、显存，还是模型质量。
