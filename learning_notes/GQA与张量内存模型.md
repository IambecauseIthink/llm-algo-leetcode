# 第四章 Attention：GQA、KV Cache 与张量内存模型

> 对应 `02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`。
> 围绕我的提问整理：`repeat_kv` 里的 `expand` 看不懂、维度 2→4 在内存里到底怎么搬、expand 怎么知道复制哪部分、为什么 GQA 一定要复制凑大矩阵、stride 怎么算。
> 这条主线把 PyTorch 张量的底层内存模型彻底打通，对后面看 FlashAttention、各种 `contiguous` 报错都是地基。

---

## 0. 这一串问题的脉络

```
KV Cache 拼接 → repeat_kv 复制KV头 → expand 怎么复制 → stride 机制 → stride 怎么算
            └──────────── 为什么一定要凑成大矩阵？────────────┘
```

---

## 1. KV Cache 拼接 与 repeat_kv 的代码

```python
# ① KV Cache：把新 K 拼到历史末尾（dim=2 是 seq 维）
cache = torch.randn(2, 2, 5, 8)   # [B, H_kv, 旧长度5, D]
new_k = torch.randn(2, 2, 1, 8)   # 新 token 的 K，长度1
updated = torch.cat([cache, new_k], dim=2)  # [2, 2, 6, 8]

# ② repeat_kv：每个 KV 头复制 n_rep 份，对齐 Query 头数
n_rep = 2                          # 4 个 Q 头 / 2 个 KV 头
x = updated[:, :, None, :, :].expand(2, 2, n_rep, 6, 8)
x = x.reshape(2, 2 * n_rep, 6, 8)  # [2, 4, 6, 8] 对齐 Q
```

GQA：Query 头多、KV 头少（4 个 Q 头、2 个 KV 头）。算注意力时每个 Q 头都要有对应 K/V，所以要把 2 个 KV 头“扩”成 4 个 —— 每个 KV 头复制 `n_rep=2` 份：

```
KV 头:  [kv0, kv1]
复制后: [kv0, kv0, kv1, kv1]   ← 对齐 4 个 Q 头
```

---

## 2. `updated[:, :, None, :, :].expand(...)` 语法拆解

`updated` 形状 `[2, 2, 6, 8]` = `[B, H_kv, seq, D]`，分两步：

### 第一步：`[:, :, None, :, :]` 插入新轴

`None`（等价 `np.newaxis`）在该位置插入一个**长度为 1** 的新轴。`:` 表示该维原样保留。

```
updated:               [2, 2,    6, 8]
updated[:,:,None,:,:]: [2, 2, 1, 6, 8]   ← 在 H_kv 后插入长度1的轴
```

### 第二步：`.expand(2, 2, n_rep, 6, 8)` 撑大

`expand` 把**长度为 1** 的维扩展到指定大小：

```
[2, 2, 1, 6, 8]  →  [2, 2, 2, 6, 8]   ← 那个 1 撑成 n_rep=2
```

### 为什么必须“先插轴再 expand”

`expand` 只能扩展**已存在且长度为 1** 的维度。原来的 `H_kv=2` 不是 1，没法直接在它上面变出 n_rep 份。所以套路固定三步：

```python
x = updated[:, :, None, :, :]          # ① 插长度1新轴  [B,H_kv,1,seq,D]
x = x.expand(B, H_kv, n_rep, seq, D)   # ② 撑成 n_rep    [B,H_kv,n_rep,seq,D]
x = x.reshape(B, H_kv*n_rep, seq, D)   # ③ 并进 head     [B,H_kv*n_rep,seq,D]
```

新轴插在 `H_kv` **右边**，reshape 合并时顺序才是 `[kv0,kv0,kv1,kv1]`（每个头连续复制 n_rep 份），正好匹配 Q 头分组。插左边就会变成 `[kv0,kv1,kv0,kv1]`，和 Q 对不上。

---

## 3. 维度 2→4 在内存里到底怎么搬（数据层面）

把数据具体化，只看 `H_kv` 和 `D` 两维：

```
kv0 = [a, b, ...]      内存里就是 [kv0, kv1] 挨着放
kv1 = [x, y, ...]
```

### expand 后：内存没动

`1→2` 是**假的**。`expand` 靠 **stride=0**：让新轴的索引 0、1 指向同一块内存。

```
逻辑上看到:          内存里实际只有一份:
 新轴[0] → kv0         kv0 = [a,b,...]
 新轴[1] → kv0  ┐      kv1 = [x,y,...]
               ├ 都指向同一个     (没有第二份)
 新轴[0] → kv1
 新轴[1] → kv1  ┘
```

### reshape 才真正搬数据（“2→4”的真相）

reshape 要一块**真正连续**的内存，但内存里只有 2 份真实数据。于是 reshape **触发一次真实拷贝（物化）**，按逻辑顺序把数据逐元素写进新连续内存：

```
拷贝前(2份):          拷贝后(4份, 全新连续内存):
 kv0 = [a,b,...]        头0 = [a,b,...]  ← 复制自 kv0
 kv1 = [x,y,...]        头1 = [a,b,...]  ← 又一份 kv0
                        头2 = [x,y,...]  ← 复制自 kv1
                        头3 = [x,y,...]  ← 又一份 kv1
```

**“2 变 4”：reshape 读 expand 给的逻辑视图（0、1 都读 kv0，2、3 都读 kv1），逐字节写进新内存，kv0/kv1 各被物理写两遍。这一刻内存里才真有 4 份。**

真实数字验证：

```python
kv0 = torch.tensor([1, 2]); kv1 = torch.tensor([7, 8])
updated = torch.stack([kv0, kv1])          # [[1,2],[7,8]]
x = updated[:, None, :].expand(2, 2, 2).reshape(4, 2)
# tensor([[1,2],[1,2],[7,8],[7,8]])  ← 各出现两次且连续
```

---

## 4. expand 怎么“知道”复制哪部分 —— 它根本没复制

### 张量 = 一块平铺内存 + 一份“读取说明书(stride)”

**stride[i] = 第 i 维索引 +1 时，内存地址跳几格。** 读 `x[i,j]` 的地址 = `i*stride[0] + j*stride[1]`。

### expand 的秘密：把被撑开那根轴的 stride 设为 0

```
expand 后:
shape  = [2, 2, 2]
stride = [2, 0, 1]    ← 新轴 stride = 0
内存仍是  [1, 2, 7, 8]  ← 只有4个数
```

读 `x[i, r, j]` 地址 = `i*2 + r*0 + j*1`。**`r*0` 恒为 0**，所以新轴索引 r 取 0 还是 1，地址完全一样，读到同一个数。

```
x[0,0,0] → 地址0 → 1      x[0,1,0] → 地址0 → 1   (r变了,地址没变!)
x[0,0,1] → 地址1 → 2      x[0,1,1] → 地址1 → 2
```

**所以 expand 不“知道”也不“决定”复制谁 —— 它没有复制逻辑，只是把新轴 stride 设 0，让那根轴在地址计算里失效(乘0)。“复制哪部分”由新轴插在哪决定：新轴和谁相邻，谁就是被重复读取的单元。**

真正的复制只在 reshape/contiguous 需要连续内存时才发生。

---

## 5. stride 怎么计算

### 连续张量的公式

$$\text{stride}[i] = \prod_{j>i}\text{shape}[j] \quad(\text{第 i 维右边所有维度大小的乘积})$$

- 最后一维右边为空，空乘积 = 1 → **最后一维 stride 恒为 1**。
- 根源：PyTorch 默认**行优先(row-major)**存储，最后一维在内存里变化最快。
- 口算：**从右往左累乘 shape**。

### 例子

```
[2, 3, 4]    → stride (12, 4, 1)   # 1, ×4=4, ×3=12 倒着写
[2, 2]       → stride (2, 1)
[2, 2, 6, 8] → stride (96, 48, 8, 1)
```

验证 `[2,3,4]` 的 `x[1,0,0]` 地址 = `1*12 = 12`（跳过第0个 3×4=12 元素的块）。✓

### 很多操作只改 stride，不搬数据（零拷贝）

```python
# expand: 把某维 stride 设 0
torch.randn(2,1,4).expand(2,3,4).stride()   # (4, 0, 1)

# transpose/permute: 交换对应维的 shape 和 stride
x = torch.randn(2,3,4)          # stride (12,4,1)
x.transpose(0,1).stride()       # (4,12,1) ← 不再符合公式 → 非连续
```

### 为什么有时必须 `.contiguous()`

`reshape`/`view` 要求内存按行优先连续（stride 符合公式）。transpose/expand 后变“非连续”，直接 view 会报错：

```
RuntimeError: view size is not compatible with input tensor's size and stride
```

`.contiguous()` 会**真的拷贝一份**，生成符合标准公式 stride 的新连续内存。这正是“reshape 触发物化拷贝”的本质：把乱掉的 stride 重新规整成“右边乘积”标准形式。

### 判断连续

张量连续 ⟺ stride 等于“右边维度乘积”标准值（且无 stride=0）。`x.is_contiguous()` 就是这个检查。

---

## 6. 为什么 GQA 一定要复制、凑成大矩阵？（数学不需要，工程为了快）

**直觉是对的：数值上完全不需要复制。** 可以让多个 Q 头共享同一份 KV、各算各的，结果一模一样：

```python
for g in range(n_kv_heads):      # 不复制
    k = kv_cache[:, g]
    for q in group_of(g):
        attn[q] = Q[q] @ k.T     # 数值与“复制成4份再算”完全相等
```

那为什么还要复制？**GPU 最怕“多次小运算”，最爱“一次大运算”：**

1. **一次大矩阵乘 ≫ 多次小矩阵乘**：
   - 每次 kernel 调用有固定**启动开销**，小运算里占比极大（算1微秒，启动等5微秒）。
   - 单头运算量小，**填不满上万核心**，大量闲置。
   - 凑成一个 `[B,H,S,D]` 大矩阵 → 只发起一次 kernel，开销摊薄、核心吃饱。
2. **现成高效算子只吃规整形状**：cuBLAS / FlashAttention 被优化到极致，但要求 Q、K 头数对齐。复制对齐后直接套用，省心又快。
3. **expand 让复制几乎不花钱**：stride=0 零拷贝；FlashAttention 等更进一步，**根本不物化**，kernel 内部用共享视图让多个 Q 头读同一份 KV —— 正是你设想的“不复制、共享”！

### 三种方案权衡

| 方案 | 速度 | 显存 | 复杂度 |
|------|------|------|--------|
| 循环各算 | 慢(kernel开销大、核心闲) | 省 | 中 |
| 复制成大矩阵(repeat_kv) | 快 | 多一点(或expand零拷贝) | **极简**，复用标准算子 |
| 融合kernel(FlashAttn) | 最快 | 最省 | 高(手写CUDA) |

`repeat_kv` 是“几乎最快 + 代码极简 + 复用成熟算子”的甜点方案。

**注意区分两件事**：GQA 省的是**KV Cache 的存储量**（只存 2 份 KV 而非 4 份），这个收益始终在；复制只发生在“算的那一刻”为了对齐算子，且常常是零拷贝视图。两者不冲突。

---

## 7. 一句话总结

- `repeat_kv` = 在 KV 头右边插长度1新轴 → expand 零拷贝撑成 n_rep → reshape 物化并进 head，得到 `[kv0,kv0,kv1,kv1]` 对齐 Q 头。
- **expand 不复制**，只把新轴 stride 设 0，让索引在该轴“空转”读同一份；**reshape 才真正搬数据**。
- **连续张量 stride = 该维右边所有维度大小之积**（末维恒1），根源是行优先存储；transpose/expand 只改 stride，view/contiguous 才搬数据。
- GQA 复制**数学上非必需**，纯为迎合 GPU“一次大运算远快于多次小运算”并复用规整算子；最优实现(FlashAttention)其实就是“共享不复制”。
