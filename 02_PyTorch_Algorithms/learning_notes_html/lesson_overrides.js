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
      title: "门控：一路当开关，一路当内容（顺便数清楚有几块矩阵）",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "Linear(d_in, d_out) 就是一块权重矩阵，把最后一维从 d_in 映射到 d_out。",
        "SiLU(x) = x · sigmoid(x)，是一条平滑的激活曲线，输出可正可负、像“软开关”。",
        "两个 shape 完全相同的张量可以逐元素相乘（Hadamard ⊙）。"
      ],
      intuition: "传统 MLP 升维后只过一次激活，像一条单行道；SwiGLU 把升维这步分成并行两条：gate 支路算“开关强度”，up 支路算“内容”，两者相乘后再降维。记住它一共用了 3 块矩阵——下一关的参数推导全靠这个数字。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>普通 MLP：一条单行道（2 块矩阵）</strong>
              <p>x 先升维，过一次固定激活（如 ReLU 把负数一刀切成 0），再降维。开关是写死的，不会随数据变。</p>
              <div class="mini-flow"><span>x</span><span>W_up → 激活</span><strong>W_down</strong></div>
            </div>
            <div class="story-arrow">SwiGLU：把“升维 + 激活”这一步拆成并行两条支路</div>
            <div class="story-panel">
              <strong>门控 MLP：内容 × 开关（3 块矩阵）</strong>
              <p>up 支路（W_up）产出“内容”；gate 支路（W_gate）产出一个经 SiLU 的“开关强度”。两条 shape 一样，逐元素相乘，再用 W_down 降回原维度。</p>
              <div class="gate-demo">
                <span>gate = SiLU(x · W_gate) → 开关</span>
                <span>up = x · W_up → 内容</span>
                <strong>W_down( gate ⊙ up )</strong>
              </div>
              <p>开关由数据自己学：该放行的位置接近 1，该压制的接近 0，比固定的 ReLU 灵活得多——这就是 SwiGLU 表达力更强的来源。</p>
              <p>数一数矩阵：<strong>W_gate、W_up、W_down 共 3 块</strong>。传统 MLP 只有 2 块。多出来的那块 gate 就是门控的“代价”，下一关要把它算进账。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：用“开关 × 内容”筛选数值</h4><pre><code>knob  = torch.tensor([-1.0, 0.0, 2.0])   # 开关原始打分
value = torch.tensor([10.0, 10.0, 10.0]) # 内容

# sigmoid 把开关压到 0~1：负数→趋近 0(关)，正数→趋近 1(开)
gate = torch.sigmoid(knob)               # [0.27, 0.50, 0.88]
filtered = gate * value                  # [2.7, 5.0, 8.8]
# 真正的 SwiGLU 用 F.silu 代替 sigmoid，思路完全一样</code></pre></div>`,
      checkpoint: {
        question: "SwiGLU 为什么需要 gate 和 up 两条升维支路？",
        options: ["一条产生门控开关，一条产生内容，相乘实现“数据自己学的过滤”", "为了把 batch size 翻倍", "为了删除 down_proj"],
        answer: 0,
        explain: "gate 经 SiLU 后像逐元素开关，up 提供内容，两者相乘后再降维。代价是升维从 1 块矩阵变成 2 块（gate+up），全模块共 3 块。"
      },
      homework: [
        "定义 gate/up/down 三个投影（工业写法是把 gate 和 up 融合成一块 gate_up_proj）。",
        "前向里确认 gate 与 up 的 shape 完全相同，才能逐元素相乘。",
        "检查最终输出 shape 回到 hidden_size。"
      ]
    },
    {
      id: "swiglu-size-align",
      title: "为什么是 8/3 d？——给门控“多出来的那块矩阵”买单",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "一块权重矩阵的参数量 ≈ 输入维度 × 输出维度。",
        "传统 MLP 用 2 块矩阵；SwiGLU 因为多了 gate 支路，用 3 块（见上一关）。",
        "“计算开销 / 参数量相同”是人为加的约束，目的是做公平对比，不是天然规律。"
      ],
      intuition: "8/3 不是魔法数字，而是一道初中解方程题的答案。先想清楚我们到底在跟谁比、为什么要比平，再把两边的矩阵数清楚，令它们相等，h 自然就解出来了。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先想清楚：我们到底在比什么、为什么要比平？</strong>
              <p>SwiGLU 听起来更强，但它比传统 MLP 多了一条 gate 支路——等于凭空多塞了一整块大矩阵。如果直接加上去，模型自然变大、变慢。</p>
              <p>那它效果好，究竟是<em>“门控结构聪明”</em>，还是单纯<em>“参数更多、堆料堆出来的”</em>？为了把这两者分开，研究者<strong>人为规定：让 SwiGLU 的参数量 = 传统 MLP 的参数量</strong>。这样一来，在“同样大小、同样算力预算”下 SwiGLU 还更好，功劳才能确凿地算到门控结构头上。这就是“让计算开销完全相同”的真正动机——它是一条公平竞赛规则，不是物理定律。</p>
            </div>
            <div class="story-arrow">既然要比平，那就把两边的参数各数一遍</div>
            <div class="story-panel">
              <strong>1. 传统 MLP 有多少参数？→ 8d²</strong>
              <p>设输入维度 = d，按惯例中间层放大到 4d。两块矩阵：升维一块、降维一块。一块的参数量 ≈ 行 × 列。</p>
              <div class="param-demo">
                <span>W_up：d × 4d = 4d²</span>
                <span>W_down：4d × d = 4d²</span>
                <strong>合计 = 8d²</strong>
              </div>
            </div>
            <div class="story-arrow">SwiGLU 升维分叉成两块，于是是 3 块矩阵</div>
            <div class="story-panel">
              <strong>2. SwiGLU 有多少参数？→ 3 · d · h</strong>
              <p>设 SwiGLU 的中间维度是未知数 h。升维分成 gate、up 两块（各 d × h），降维一块（h × d）。</p>
              <div class="param-demo">
                <span>W_gate：d × h</span>
                <span>W_up：d × h</span>
                <span>W_down：h × d</span>
                <strong>合计 = 3 · d · h</strong>
              </div>
            </div>
            <div class="story-arrow">套用公平规则：令两边相等，解出 h</div>
            <div class="story-panel">
              <strong>3. 解方程 → h = 8/3 d</strong>
              <p>把竞赛规则写成等式，两边都有一个 d，约掉即可：</p>
              <div class="formula"><span>3 · d · h = 8d²</span><strong>h = 8d / 3 ≈ 2.67 d</strong></div>
              <p>看懂这个结果：SwiGLU 的中间维度不是 4d，而是缩到约 2.67d。<strong>缩小的这部分，正好补偿掉多养一块 gate 矩阵的开销</strong>，于是总参数追平。这就是 LLaMA 源码里 <code>int(8 * hidden_size / 3)</code> 的全部来历。</p>
            </div>
            <div class="story-arrow">最后一步工程修正：对齐到整齐的倍数</div>
            <div class="story-panel">
              <strong>4. 向上对齐 multiple_of（如 256）</strong>
              <p>8/3 算出来常是零碎数（d=4096 时 ≈ 10922）。GPU 喜欢整齐维度：Tensor Core 要求对齐、张量并行时还得能被 GPU 张数整除。所以向上取整到 256 的倍数：10922 → 11008。</p>
              <p>对齐会让参数量比理论值<em>略大一点点</em>，这是为了硬件效率付的小钱，不影响“追平传统 MLP”的初衷。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：把一个数向上补齐到某个倍数</h4><p>TODO 2 要把 8/3 d 向上对齐到 multiple_of。先用排座位的小例子练这个“向上取整”技巧：</p><pre><code>people = 26      # 要坐的人数（类比理论 intermediate_size）
row    = 8       # 每排 8 个座位（类比 multiple_of）

# 直接整除会向下丢人：26 // 8 = 3 排，只坐 24 人 ❌
# 先 +(row-1) 再整除，就能“向上取整”到能装下所有人的排数
aligned = ((people + row - 1) // row) * row   # = 32 ✅</code></pre><p class="syntax-tip">读法：<code>(n + m - 1) // m * m</code> 是“把 n 向上对齐到 m 的倍数”的标准写法，记下来，工程里到处用。</p></div>`,
      checkpoint: {
        question: "LLaMA 为什么把 SwiGLU 的中间维度从 4d 缩到约 8/3 d？",
        options: [
          "因为多了 gate 支路（共 3 块矩阵），缩小中间维度让总参数重新等于传统 MLP，从而公平对比",
          "因为 8/3 是 GPU 硬件唯一支持的维度比例",
          "为了让参数变多、模型更大，效果自然更好"
        ],
        answer: 0,
        explain: "SwiGLU 升维多一条 gate 支路，3·d·h 比传统的 8d² 更费参数。令 3·d·h = 8d² 解得 h = 8/3 d，性能提升才能归功于门控结构本身，而不是堆参数。"
      },
      homework: [
        "TODO 1：写出 int(8/3 · hidden_size) 的计算（注意整数除法）。",
        "TODO 2：用 (n + m - 1) // m * m 把结果向上对齐到 multiple_of。",
        "跑测试：4096 应得到 11008，并用 3·d·h 验证参数量与传统 MLP 接近。"
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
