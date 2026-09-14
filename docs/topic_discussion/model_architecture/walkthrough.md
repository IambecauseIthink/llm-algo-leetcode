# 大模型结构和原理深入阅读

假设你现在不再满足于“知道几个结构名词”，而是想真正看懂一个 token 的 hidden state 怎样穿过现代 LLM block。接下来你会发现，norm、attention、RoPE、MLP、MoE 和工程 trick 并不是几张分散的卡片，而是一条连续结构主干上的不同位置选择。

这条线最重要的是按结构顺序判断：token 先怎么进入模型，hidden state 在 block 里怎么走，dense 路径和 sparse 路径在哪里分开，最后真实模型又在什么地方做了局部改写。

对应专题正文：[01 Transformer Decoder](./01_transformer_decoder.md)。先建立 decoder-only 总览，再沿 hidden state 路径拆开组件。

## 第一段：先从 token 进入 block

故事通常从 token 进入模型开始。第一步不是背 embedding 定义，而是先确认 hidden state 进入 block 以前已经具备什么信息、维度怎样保持一致、主干信号之后会沿什么路径传播。

这一步对应 [02 Tokenization / BPE / Embedding](./02_tokenization_embedding.md)。

## 第二段：attention 和 RoPE 先决定上下文关系

一旦 token 进入 attention，问题就不只是“算子怎么算”，而是“上下文关系怎样组织、位置信息怎样进入”。这时要把 Q/K/V 的关系、head 组织和 RoPE 的位置放回同一张图里看。

这一步对应 [03 Norm Evolution](./03_norm_evolution.md)、[04 Attention Evolution](./04_attention_evolution.md) 和 [05 RoPE / Position Encoding](./05_rope_position_encoding.md)。

## 第三段：MLP / SwiGLU 再决定表示如何被扩展

attention 之后，hidden state 会进入 MLP 路径。这里真正要看的不是“还有一层前馈网络”，而是 gate / up / down 怎样一起改变表示容量，为什么现代模型更喜欢门控结构而不是朴素 FFN。

这一步对应 [06 Block / Residual Path](./06_block_residual_path.md) 和 [07 MLP / FFN Evolution](./07_mlp_ffn_evolution.md)。

## 第四段：MoE 把 dense 路径换成路由问题

如果结构升级到 MoE，问题就不再只是 MLP 变大，而是 token 要不要被 router 分给不同 expert，负载为什么会失衡，为什么 sparse routing 会单独成为一条演化线。

这一步对应 [09 MoE / Sparsity Evolution](./09_moe_sparsity_evolution.md)。

## 第五段：最后回到真实模型实现

真正的闭环不在“会画一个教科书 block”，而在你看到真实模型源码时，不会被 norm 位置、命名变化、局部 trick 或代表模型差异带偏。把这条故事走完以后，一个更像真实结论的说法通常不是“我知道 Attention 和 RoPE”，而是：我能把 token 的路径、组件位置、dense / sparse 分叉和真实实现差异放回同一张结构图里。

这一步对应 [08 Representative Models / Cross Module Comparison](./08_representative_models.md)，最终回到 `61 Model Architecture Exploration` 做结构验证。
