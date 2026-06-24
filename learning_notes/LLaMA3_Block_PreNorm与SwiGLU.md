# 第五章 LLaMA3 Block：Pre-Norm、Attention、MLP 与 SwiGLU

> 对应 `02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb`。
> 围绕我的提问整理：MLP 在 LLaMA 3 的哪里、SwiGLU 和 Attention 是什么关系、为什么用 `*` 不是 `@`、Pre-Norm 到底在 norm 谁、`post_attention_layernorm` 为什么不是 MLP 的 Post-Norm、计算图是什么。

---

## 0. 这一章先抓住一条主线

LLaMA 3 的一个 Decoder Layer 不是一团乱的网络，而是两段连续的更新：

```text
第一段：Attention 子层更新 hidden state
第二段：MLP / SwiGLU 子层继续更新 hidden state
```

最重要的结构是：

```python
h1 = h0 + Attention(RMSNorm(h0))
h2 = h1 + MLP(RMSNorm(h1))
```

所以这一章可以用一句话记住：

> 每一层先用 Attention 做一次 token 间通信，再用 MLP/SwiGLU 对每个 token 的 hidden 向量做一次非线性变换；两段都是 Pre-Norm，并且两段都有自己的 residual。

---

## 1. LLaMA 3 Decoder Layer 的整体位置

一个大模型不是只有一层 Transformer，而是很多个 Decoder Layer 堆起来：

```text
tokens
  ↓
Embedding
  ↓
Decoder Layer 0
  ↓
Decoder Layer 1
  ↓
...
  ↓
Decoder Layer N-1
  ↓
Final RMSNorm
  ↓
LM Head
  ↓
logits
```

而这一章讲的是其中一个 Decoder Layer 内部的结构：

```text
输入 hidden_states
  ↓
RMSNorm
  ↓
Self-Attention
  ↓
Residual Add
  ↓
RMSNorm
  ↓
MLP / SwiGLU
  ↓
Residual Add
  ↓
输出 hidden_states，传给下一层
```

对应代码骨架：

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size)

        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        # Attention 子层
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP 子层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

---

## 2. Attention 和 MLP 的相对位置关系

我一开始容易混淆的一点是：SwiGLU/MLP 是不是 Attention 的一部分？

答案：**不是。**

在一个 LLaMA Decoder Layer 里，Attention 和 MLP 是两个并列的子层，只是顺序上 Attention 在前，MLP 在后：

```text
hidden_states
  ↓
Attention 子层
  ↓
MLP 子层
  ↓
下一层
```

它们的分工不同：

| 模块 | 主要作用 | 是否混合 token |
| --- | --- | --- |
| Attention | 让不同 token 之间交换信息 | 是 |
| MLP / SwiGLU | 对每个 token 自己的 hidden 向量做非线性变换 | 通常不混合 token |

假设：

```python
hidden_states.shape = [batch, seq_len, hidden_size]
```

Attention 关心的是 token 和 token 的关系，会在 `seq_len` 维度上做交互：

```text
第 5 个 token 可以看第 1、2、3、4、5 个 token 的信息
```

MLP 则更像是对每个 token 单独套同一个小网络：

```text
hidden_states[b, 0, :] → MLP → 新的 hidden_states[b, 0, :]
hidden_states[b, 1, :] → MLP → 新的 hidden_states[b, 1, :]
hidden_states[b, 2, :] → MLP → 新的 hidden_states[b, 2, :]
```

MLP 不负责让 token 之间通信；token 之间通信主要由 Attention 完成。

---

## 3. LLaMA 的 MLP 为什么叫 SwiGLU

普通 Transformer MLP 常见写法是：

```python
out = down_proj(activation(up_proj(x)))
```

它通常只有两个线性层：

```text
hidden_size → intermediate_size → hidden_size
```

LLaMA 风格的 SwiGLU MLP 有三个线性层：

```python
gate_proj: hidden_size → intermediate_size
up_proj:   hidden_size → intermediate_size
down_proj: intermediate_size → hidden_size
```

代码是：

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )
```

公式写成：

```text
MLP(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
```

其中 `⊙` 是逐元素乘法，也就是 PyTorch 里的 `*`。

---

## 4. `*` 和 `@` 在 SwiGLU 里的区别

这行代码：

```python
F.silu(self.gate_proj(x)) * self.up_proj(x)
```

用的是 `*`，不是 `@`。

原因是这里要做的是**门控**，不是矩阵乘法。

假设：

```python
x.shape = [B, S, H]
```

经过两个投影：

```python
gate = F.silu(self.gate_proj(x))  # [B, S, I]
up = self.up_proj(x)              # [B, S, I]
```

两者 shape 一样，所以可以逐元素相乘：

```python
hidden = gate * up                # [B, S, I]
```

这表示：

```text
gate[b, s, i] 控制 up[b, s, i] 这一格信息保留多少
```

也就是每个 token、每个 hidden channel 都有自己的门控值。

如果写成 `@`：

```python
gate @ up
```

那就变成矩阵乘法，语义完全不对，而且维度通常也对不上。矩阵乘法已经发生在这些线性层内部了：

```python
self.gate_proj(x)  # 本质是 x @ W_gate.T
self.up_proj(x)    # 本质是 x @ W_up.T
self.down_proj(...)# 本质是 hidden @ W_down.T
```

所以 SwiGLU 的中间融合步骤必须是：

```text
线性投影 → 激活 → 逐元素门控 → 线性投影回 hidden_size
```

---

## 5. Shape 流水线：从 `[B,S,H]` 到 `[B,S,H]`

假设：

```python
B = batch_size
S = seq_len
H = hidden_size
I = intermediate_size
```

SwiGLU MLP 的 shape 变化是：

```text
x:                         [B, S, H]
gate_proj(x):              [B, S, I]
F.silu(gate_proj(x)):      [B, S, I]
up_proj(x):                [B, S, I]
逐元素相乘:                 [B, S, I]
down_proj(...):            [B, S, H]
```

最后必须回到 `[B,S,H]`，因为 MLP 的输出要和 residual 相加：

```python
hidden_states = residual + hidden_states
```

如果 MLP 输出不是 `[B,S,H]`，残差连接就加不了。

---

## 6. Pre-Norm 到底是什么意思

Pre-Norm 的意思是：

> 在 Attention 或 MLP 子层计算之前，先做 normalization。

LLaMA 的形式是：

```python
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

不是：

```python
h = RMSNorm(h + Attention(h))
h = RMSNorm(h + MLP(h))
```

后者才是 Post-Norm。

用一段代码看最清楚：

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states)
hidden_states = residual + hidden_states
```

这段是 Attention 的 Pre-Norm，因为 norm 在 Attention 前面。

第二段：

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

这段也是 Pre-Norm，因为 norm 在 MLP 前面。

所以一个 LLaMA Decoder Layer 里有两个 Pre-Norm：

```text
input_layernorm：Attention 的 Pre-Norm
post_attention_layernorm：MLP 的 Pre-Norm
```

---

## 7. `post_attention_layernorm` 为什么不是 MLP 的 Post-Norm

这个名字最容易误导。

`post_attention_layernorm` 的意思是：

```text
它在 Attention 之后
```

不是：

```text
它在 MLP 之后
```

它的位置是：

```text
Attention 之后，MLP 之前
```

所以从 Attention 的角度看，它叫：

```text
post_attention_layernorm
```

但从 MLP 的角度看，它其实就是：

```text
pre_mlp_layernorm
```

如果换一个更直观的命名，可以写成：

```python
self.attn_norm = RMSNorm(hidden_size)
self.mlp_norm = RMSNorm(hidden_size)

h = h + self_attn(attn_norm(h))
h = h + mlp(mlp_norm(h))
```

这就很清楚：两个子层都是 Pre-Norm。

真正的 MLP Post-Norm 会长这样：

```python
residual = hidden_states
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
```

但 LLaMA 不是这样。

---

## 8. Normalization 具体做了什么

LLaMA 使用的是 RMSNorm。它做的不是 softmax，也不是把值压到 0 到 1，而是对每个 token 的 hidden 向量做数值尺度校准。

假设某个 token 的 hidden 向量是：

```python
h = [h1, h2, h3, ..., hH]
```

RMSNorm 会先算均方根：

```text
rms = sqrt(mean(h ** 2) + eps)
```

然后把整个向量除以这个尺度：

```text
h_norm = h / rms
```

最后乘上可学习参数：

```text
out = h_norm * weight
```

伪代码：

```python
def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

关键是 `dim=-1`：

```python
hidden_states.shape = [B, S, H]
```

RMSNorm 是对每个 token 的 hidden 维度单独做：

```text
hidden_states[b, s, :]
```

它不会混合 batch，也不会混合 token。

直观例子：

```text
原始向量: [30, 40]
rms ≈ 35.36
归一化后: [0.85, 1.13]
```

如果原始向量整体放大：

```text
原始向量: [300, 400]
rms ≈ 353.6
归一化后: [0.85, 1.13]
```

所以 normalization 主要消除“整体尺度忽大忽小”的影响，保留向量内部的相对模式。

它不是：

```text
不是 softmax
不是 attention score
不是把所有值变成概率
不是让 token 之间交换信息
不是把值限制在 0 到 1
```

一句话：

> Normalization 是给 hidden state 做数值尺度校准，让进入 Attention 和 MLP 的输入更稳定。

---

## 9. 为什么 Pre-Norm 对深层模型重要

Transformer 会堆很多层。每一层都有 Attention 和 MLP，如果输入数值尺度在层与层之间不断漂移，训练会不稳定。

Pre-Norm 的结构是：

```text
原始 hidden_states ───────────────┐
                                  +
RMSNorm(hidden_states) → 子层计算 ┘
```

也就是：

1. residual 主路保留原始 hidden state，直接往后传。
2. 子层输入先被 RMSNorm 校准数值尺度。
3. Attention 或 MLP 只负责算一个 update。
4. update 加回 residual。

这样做的好处：

- 每个子层看到的输入尺度更稳定。
- residual 主路提供更直接的梯度通路。
- 深层模型更容易训练。

这也是为什么 LLaMA 这类现代大模型普遍采用 Pre-Norm。

---

## 10. 计算图是什么

计算图就是 PyTorch 记录的一张依赖关系图：

> 输出是由哪些输入、参数和操作一步步算出来的。

简单例子：

```python
y = x * w
z = y + b
loss = z.sum()
```

计算图可以画成：

```text
x ─┐
   ├─ (*) ─ y ─┐
w ─┘           ├─ (+) ─ z ─ sum ─ loss
b ─────────────┘
```

反向传播时，PyTorch 沿着图倒着走：

```text
loss → z → y → x/w
         → b
```

所以只要某个参数参与了从输入到 loss 的计算，它就能拿到梯度。

---

## 11. SwiGLU 的计算图

这段代码：

```python
return self.down_proj(
    F.silu(self.gate_proj(x)) * self.up_proj(x)
)
```

计算图是：

```text
x ── gate_proj ── silu ─┐
                        ├─ (*) ── down_proj ── output
x ── up_proj ───────────┘
```

它有两条分支：

```text
gate 分支：x → gate_proj → silu
up 分支：  x → up_proj
```

然后两条分支用逐元素乘法合并，再进入 `down_proj`。

如果后面有：

```python
loss = output.sum()
loss.backward()
```

梯度会沿着计算图反向传播到：

```python
gate_proj.weight
up_proj.weight
down_proj.weight
x
```

如果代码里忘了调用 `self.mlp(hidden_states)`，那么 MLP 的参数就不在计算图里，反向传播时它们的 `.grad` 可能是 `None`。

---

## 12. 整个 Decoder Layer 的计算图

一个 LLaMA Decoder Layer 的计算图可以画成：

```text
hidden0
   │
   ├──────────── residual0 ───────────┐
   │                                  │
   ↓                                  │
input_layernorm                       │
   ↓                                  │
self_attn                             │
   ↓                                  │
   └──────────── add ─────────────────┘
                  │
                  ↓
               hidden1
                  │
                  ├──────── residual1 ───────────┐
                  │                              │
                  ↓                              │
       post_attention_layernorm                  │
                  ↓                              │
                 mlp                             │
                  ↓                              │
                  └──────── add ─────────────────┘
                               │
                               ↓
                            hidden2
```

注意两个 residual 不是同一个：

```text
residual0 = Attention 子层开始前的 hidden0
residual1 = MLP 子层开始前的 hidden1
```

这点很容易写错。第二段 MLP 不能再加回最初的 `hidden0`，而是要加回 Attention 更新后的 `hidden1`。

---

## 13. 最容易混淆的几个点

### 13.1 MLP 不在 Attention 里面

错误理解：

```text
Attention 里面包含 MLP/SwiGLU
```

正确理解：

```text
Attention 和 MLP 是同一个 Decoder Layer 里的两个连续子层
```

顺序是：

```text
Attention → MLP
```

### 13.2 `post_attention_layernorm` 不是 MLP 的 Post-Norm

错误理解：

```text
post_attention_layernorm 这个名字里有 post，所以 MLP 是 Post-Norm
```

正确理解：

```text
post_attention_layernorm 表示它在 Attention 后面，但它在 MLP 前面，所以它是 MLP 的 Pre-Norm
```

### 13.3 SwiGLU 中间用 `*`，不是 `@`

错误理解：

```python
F.silu(gate_proj(x)) @ up_proj(x)
```

正确写法：

```python
F.silu(gate_proj(x)) * up_proj(x)
```

因为这里做的是逐元素门控，不是矩阵乘法。

### 13.4 第二段 residual 要重新保存

错误写法：

```python
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states)
hidden_states = residual + hidden_states

hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states  # 错：这里还是最初的 residual
```

正确写法：

```python
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states                 # 重新保存 Attention 后的 hidden state
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states
```

---

## 14. 一张表总结本章模块

| 名字 | 位于哪里 | 做什么 | Shape |
| --- | --- | --- | --- |
| `input_layernorm` | Attention 前 | 校准 Attention 输入尺度 | `[B,S,H] → [B,S,H]` |
| `self_attn` | 第一子层 | token 之间交换信息 | `[B,S,H] → [B,S,H]` |
| 第一次 residual add | Attention 后 | 把 Attention update 加回主路 | `[B,S,H] + [B,S,H]` |
| `post_attention_layernorm` | Attention 后，MLP 前 | 校准 MLP 输入尺度 | `[B,S,H] → [B,S,H]` |
| `mlp` / `SwiGLU` | 第二子层 | 每个 token 内部做非线性特征变换 | `[B,S,H] → [B,S,H]` |
| 第二次 residual add | MLP 后 | 把 MLP update 加回主路 | `[B,S,H] + [B,S,H]` |

---

## 15. 最终心智模型

把 `hidden_states` 想成一条主干。每个子层都不是直接替换主干，而是先从主干上取一个规范化后的版本，算出一份 update，再加回主干：

```python
def prenorm_step(h, norm, sublayer):
    return h + sublayer(norm(h))

h = prenorm_step(h, input_layernorm, self_attn)
h = prenorm_step(h, post_attention_layernorm, mlp)
```

这就是 LLaMA Decoder Layer 的核心。

---

## 16. 三句话总结

1. LLaMA 3 每个 Decoder Layer 先做 Attention，再做 MLP/SwiGLU；Attention 负责 token 间通信，MLP 负责每个 token 内部的非线性变换。
2. LLaMA 是 Pre-Norm：`h = h + sublayer(norm(h))`；`post_attention_layernorm` 只是“在 Attention 后”，但它仍然是 MLP 前的 norm。
3. SwiGLU 的核心是 `down_proj(SiLU(gate_proj(x)) * up_proj(x))`，中间的 `*` 是逐元素门控，不是矩阵乘法。

