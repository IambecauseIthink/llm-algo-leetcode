module.exports = {
  "01": [
    {
      id: "rmsnorm-scale",
      title: "RMS：用向量自己的大小做刻度尺",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "hidden_size 是最后一维特征数。",
        "mean(dim=-1, keepdim=True) 会保留可广播维度。",
        "eps 用来避免除以 0。"
      ],
      intuition: "RMSNorm 不关心向量平均值是不是 0，只把每个 token 的 hidden 向量缩放到稳定尺度。",
      exampleHtml: `<div class="scale-demo"><span>原向量: [3,4]</span><span>RMS≈3.54</span><span>缩放后: [0.85,1.13]</span></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：对最后一维求均值并保留维度</h4><pre><code>x = torch.randn(2, 3, 4)
last_mean = x.mean(dim=-1, keepdim=True)  # [2,3,1]
y = x / (last_mean.abs() + 1e-6)</code></pre></div>`,
      checkpoint: {
        question: "x shape 为 [2,3,4]，x.pow(2).mean(dim=-1, keepdim=True) 的 shape 是？",
        options: ["[2,3,1]", "[2,4]", "[1,3,4]"],
        answer: 0,
        explain: "沿最后一维 4 求均值，keepdim=True 会把最后一维保留为 1，方便和原张量广播。"
      },
      homework: [
        "定义 shape 为 [hidden_size] 的可学习 weight。",
        "用平方、均值、rsqrt 写出归一化。",
        "检查输出 shape 与输入一致。"
      ]
    },
    {
      id: "rmsnorm-precision",
      title: "先用 FP32 计算，再回到输入 dtype",
      todo: "TODO 2 / TODO 3",
      prerequisite: [
        "float16 的可表示范围有限。",
        "平方会放大数值。",
        ".to(x.dtype) 可以恢复原始精度。"
      ],
      intuition: "归一化里的平方像放大镜，半精度数值太大时容易溢出，所以先换到更稳的 FP32。",
      exampleHtml: `<div class="dtype-demo"><span>FP16: 300² → overflow</span><span>FP32: 300² → 90000</span><span>输出再转回 FP16</span></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：临时升精度做统计量</h4><pre><code>scores = torch.tensor([1000, 2000], dtype=torch.float16)
mean32 = scores.float().mean()
centered = (scores.float() - mean32).to(scores.dtype)</code></pre></div>`,
      checkpoint: {
        question: "为什么 RMSNorm 中常先把 x 转成 float32 再平方？",
        options: ["避免 FP16 平方溢出", "让 shape 自动变小", "让参数不参与训练"],
        answer: 0,
        explain: "平方会迅速放大数值；FP32 计算统计量更稳，最后再转回输入 dtype。"
      },
      homework: [
        "在 pow(2).mean 前使用 float32。",
        "最终输出转回输入 dtype。",
        "和参考实现比较 dtype、shape、数值误差。"
      ]
    }
  ],
  "02": [
    {
      id: "swiglu-gate",
      title: "门控：一路决定开关，一路携带内容",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "Linear 会把最后一维映射到新维度。",
        "SiLU 是平滑激活函数。",
        "两个同 shape 张量可以逐元素相乘。"
      ],
      intuition: "SwiGLU 有两条并行支路：gate 支路决定哪些信息通过，up 支路提供要被筛选的内容。",
      exampleHtml: `<div class="gate-demo"><span>gate → SiLU → 开关强度</span><span>up → 内容特征</span><strong>逐元素相乘 → down</strong></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：用门控信号筛选数值</h4><pre><code>knob = torch.tensor([-1.0, 0.0, 2.0])
value = torch.tensor([10.0, 10.0, 10.0])
filtered = torch.sigmoid(knob) * value</code></pre></div>`,
      checkpoint: {
        question: "SwiGLU 为什么需要 gate 和 up 两条升维支路？",
        options: ["一条产生门控，一条产生内容", "为了把 batch size 翻倍", "为了删除 down_proj"],
        answer: 0,
        explain: "gate 经激活后像开关，up 提供内容，两者相乘后再降维。"
      },
      homework: [
        "定义 gate/up/down 三个投影或融合投影。",
        "确认 gate 与 up 的 shape 相同。",
        "检查最终输出回到 hidden_size。"
      ]
    },
    {
      id: "swiglu-size-align",
      title: "8/3 hidden：让门控 MLP 参数量接近传统 MLP",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "矩阵参数量约等于输入维度 × 输出维度。",
        "SwiGLU 升维阶段有两组矩阵。",
        "向上取整到 multiple_of 方便硬件和并行切分。"
      ],
      intuition: "传统 MLP 是两块大矩阵，SwiGLU 是三块矩阵；为了总量接近，要把中间维度从 4d 调小到 8/3d。",
      exampleHtml: `<div class="param-demo"><span>传统: 2 × d × 4d = 8d²</span><span>SwiGLU: 3 × d × h</span><strong>h ≈ 8d/3</strong></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：向上补齐到倍数</h4><pre><code>people = 26
row = 8
aligned = ((people + row - 1) // row) * row  # 32</code></pre></div>`,
      checkpoint: {
        question: "hidden_size=12 时，理论 SwiGLU intermediate_size 约是多少？",
        options: ["32", "48", "12"],
        answer: 0,
        explain: "8/3 × 12 = 32；如果还要求 multiple_of，再继续向上对齐。"
      },
      homework: [
        "写出 8/3 hidden_size 的计算。",
        "把结果向 multiple_of 对齐。",
        "用参数量检查维度是否合理。"
      ]
    }
  ],
  "03": [
    {
      id: "rope-complex-pairs",
      title: "两个相邻数配成一个复数，再旋转",
      todo: "TODO 2 / TODO 3",
      prerequisite: [
        "head_dim 需要能两两配对。",
        "复数乘法可以表示二维旋转。",
        "旋转会改变方向但保持长度。"
      ],
      intuition: "RoPE 把 hidden 向量最后一维两两成对，看成很多小箭头；不同位置让箭头转不同角度。",
      exampleHtml: `<div class="rotate-demo"><span>[1,0]</span><span>旋转 90°</span><span>[0,1]</span><em>长度仍为 1</em></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：把二维坐标解释成复数</h4><pre><code>points = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
z = torch.view_as_complex(points)
back = torch.view_as_real(z)</code></pre></div>`,
      checkpoint: {
        question: "一个向量旋转后，哪件事通常保持不变？",
        options: ["向量长度", "batch size 自动变 1", "所有元素都变成正数"],
        answer: 0,
        explain: "旋转只改变方向，不改变模长，这也是 RoPE 测试中会检查的性质。"
      },
      homework: [
        "把最后一维 reshape 成 [..., head_dim/2, 2]。",
        "用 view_as_complex 转成复数。",
        "乘频率后用 view_as_real 和 flatten 还原。"
      ]
    },
    {
      id: "rope-frequency-table",
      title: "位置 × 频率：先做一张旋转角度表",
      todo: "TODO 1",
      prerequisite: [
        "位置索引是 0 到 seq_len-1。",
        "不同维度使用不同旋转频率。",
        "torch.outer 可以生成二维表。"
      ],
      intuition: "每一行对应一个位置，每一列对应一组维度频率；表格里的值就是该位置该维度要转的角度。",
      exampleHtml: `<table class="freq-table"><tr><th>pos</th><th>freq0</th><th>freq1</th></tr><tr><td>0</td><td>0θ</td><td>0φ</td></tr><tr><td>1</td><td>1θ</td><td>1φ</td></tr></table>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：用 outer 生成二维表</h4><pre><code>days = torch.arange(3).float()
rates = torch.tensor([1.0, 10.0])
table = torch.outer(days, rates)  # [3,2]</code></pre></div>`,
      checkpoint: {
        question: "seq_len=8、head_dim=64 时，freqs_cis 的 shape 应该是？",
        options: ["[8,32]", "[8,64]", "[64,8]"],
        answer: 0,
        explain: "最后一维两两配成复数，所以频率列数是 head_dim/2。"
      },
      homework: [
        "用 arange(0, dim, 2) 生成频率维度。",
        "用 outer 得到位置角度表。",
        "用 polar 生成 cos+i sin 的复数表。"
      ]
    }
  ],
  "04": [
    {
      id: "attention-shape-map",
      title: "多头注意力先看 shape，再看公式",
      todo: "TODO 1 / TODO 3 / TODO 4",
      prerequisite: [
        "Q/K/V 都来自输入的线性投影。",
        "多头格式常用 [B,H,S,D]。",
        "Q @ K^T 会得到 token 对 token 的分数表。"
      ],
      intuition: "每个 query token 都拿自己的向量去和所有 key token 打分，softmax 后再按分数混合 value。",
      exampleHtml: `<div class="attn-grid"><span>rows: query tokens</span><span>cols: key tokens</span><strong>score[i,j] = qi 看 kj 的程度</strong></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：最后两维做矩阵乘法</h4><pre><code>a = torch.randn(2, 3, 5, 4)
b = torch.randn(2, 3, 7, 4)
scores = a @ b.transpose(-2, -1)  # [2,3,5,7]</code></pre></div>`,
      checkpoint: {
        question: "q shape=[2,4,5,8]，k shape=[2,4,7,8]，q @ k.transpose(-2,-1) 的 shape 是？",
        options: ["[2,4,5,7]", "[2,5,4,7]", "[2,4,8,8]"],
        answer: 0,
        explain: "batch 和 head 维保持，query 长度 5 对 key 长度 7 打分。"
      },
      homework: [
        "把投影结果 view 成 [B,S,H,D] 再 transpose。",
        "计算缩放点积注意力。",
        "把 [B,H,S,D] 合并回 [B,S,H*D]。"
      ]
    },
    {
      id: "gqa-kv-cache",
      title: "GQA：少存 KV 头，用时再扩成 Query 头",
      todo: "TODO 2 / repeat_kv",
      prerequisite: [
        "自回归生成一次只新增少量 token。",
        "KV Cache 存历史 key/value。",
        "GQA 让多个 Query 头共享一组 KV 头。"
      ],
      intuition: "不要给每个 query 头都存一份 KV；先存少量共享 KV，真正算 attention 前临时复制。",
      exampleHtml: `<div class="gqa-demo"><span>Q heads: 0 1 2 3</span><span>KV heads: A B</span><span>0/1→A, 2/3→B</span></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：沿时间轴拼接新记录</h4><pre><code>history = torch.randn(2, 5, 3)
new = torch.randn(2, 1, 3)
updated = torch.cat([history, new], dim=1)</code></pre></div>`,
      checkpoint: {
        question: "num_heads=8、num_kv_heads=2 时，每个 KV 头要服务几个 Query 头？",
        options: ["4", "2", "8"],
        answer: 0,
        explain: "num_queries_per_kv = 8 / 2 = 4。"
      },
      homework: [
        "先在 seq 维拼接旧 cache 和新 KV。",
        "再 repeat_kv 扩到 Query 头数。",
        "检查 new_kv_cache 保存的是未扩展的 KV。"
      ]
    }
  ],
  "05": [
    {
      id: "llama-mlp-swiglu",
      title: "LLaMA MLP：gate、up、down 三步走",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "nn.Module 里子层要在 __init__ 定义。",
        "bias=False 是 LLaMA MLP 的常见设置。",
        "gate 和 up 输出必须同 shape 才能相乘。"
      ],
      intuition: "gate 先产生筛选信号，up 产生内容，二者相乘后由 down 投影回 hidden_size。",
      exampleHtml: `<div class="mlp-demo"><span>x → gate_proj → SiLU</span><span>x → up_proj</span><strong>相乘 → down_proj → hidden</strong></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：写一个带子层的简单模块</h4><pre><code>class ScaleValue(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    def forward(self, x):
        return torch.sigmoid(self.scale(x)) * self.value(x)</code></pre></div>`,
      checkpoint: {
        question: "LLaMA 风格 SwiGLU 中，哪两个投影的输出需要逐元素相乘？",
        options: ["gate_proj 和 up_proj", "down_proj 和输入 x", "RMSNorm 和 Attention"],
        answer: 0,
        explain: "gate 经 SiLU 后作为门控信号，与 up 的内容特征逐元素相乘。"
      },
      homework: [
        "定义 gate_proj、up_proj、down_proj。",
        "用 F.silu(gate_proj(x)) * up_proj(x)。",
        "确认输出 shape 回到 hidden_size。"
      ]
    },
    {
      id: "llama-prenorm-residual",
      title: "Pre-Norm 残差：先归一化，再走子层，最后加回来",
      todo: "TODO 3",
      prerequisite: [
        "残差连接是 output = input + update。",
        "Pre-Norm 表示子层前先 norm。",
        "一个 Decoder Layer 有 Attention 和 MLP 两个子层。"
      ],
      intuition: "残差像主路，Attention/MLP 像旁路加工；旁路输出只是在主路表示上加一份更新。",
      exampleHtml: `<div class="residual-demo"><span>x ─────────────┐</span><span>norm → attention ─┤ +</span><span>再 norm → mlp ────┤ +</span></div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：通用 pre-norm 残差模板</h4><pre><code>residual = x
update = sublayer(norm(x))
x = residual + update</code></pre></div>`,
      checkpoint: {
        question: "Pre-Norm Transformer block 中，归一化发生在子层的什么位置？",
        options: ["子层之前", "子层之后", "只在最终输出后"],
        answer: 0,
        explain: "Pre-Norm 的顺序是 norm → sublayer → residual add。"
      },
      homework: [
        "写出 Attention block 的 residual + norm + self_attn + add。",
        "写出 MLP block 的 residual + norm + mlp + add。",
        "运行 backward 检查所有参数都有梯度。"
      ]
    }
  ],
  ...require("./lesson_overrides_extra")
};
