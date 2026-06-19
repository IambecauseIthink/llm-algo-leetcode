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
      id: "rope-frequency-table",
      title: "位置变转角：为什么‘旋转’能编码相对距离",
      todo: "TODO 1",
      prerequisite: [
        "一个 token 在序列里的位置就是它的序号 m（0,1,2,…）。",
        "二维平面上的一个向量可以画成从原点出发的“小箭头”，旋转角度 θ 只改方向、不改长度。",
        "两个向量的点积 = |a||b|·cos(夹角)；夹角只取决于两者各自转了多少。"
      ],
      intuition: "RoPE 的核心魔法：把“位置 m”变成“把箭头转 mθ 角度”。当 query 在位置 m、key 在位置 n 时，它们的点积只跟差值 (m−n) 有关——模型于是天生感知“相对距离”，而不用记死“绝对位置”。这一关先建立这个直觉，再做出那张‘位置 × 频率 = 角度’的表。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 痛点：绝对位置编码“记死了”，换个长度就懵</strong>
              <p>老式做法是给每个位置发一个固定向量（位置 0 一个、位置 1 一个…）再加到 token 上。问题是：训练时只见过 0~4095，推理来个第 8000 位，模型从没见过，直接懵。我们真正想要的，其实不是“你在第几位”，而是“你俩离多远”——也就是<strong>相对距离</strong>。</p>
            </div>
            <div class="story-arrow">换个思路：不把位置“加”进去，而是按位置把向量“转”一个角度</div>
            <div class="story-panel">
              <strong>1. 位置 = 转角：序号越大，箭头转得越多</strong>
              <p>把 token 向量看成平面上的小箭头。位置 m 就让它转 mθ：位置 0 不转，位置 1 转 θ，位置 2 转 2θ…… 像钟表指针，走得越远转得越多。</p>
              <svg viewBox="0 0 360 130" width="100%" style="max-width:520px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <defs><marker id="ar1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#2563eb"/></marker></defs>
                <g stroke="#2563eb" stroke-width="2.5" marker-end="url(#ar1)" fill="none">
                  <line x1="40" y1="105" x2="40" y2="35"/>
                  <line x1="160" y1="105" x2="220" y2="50"/>
                  <line x1="280" y1="105" x2="330" y2="78"/>
                </g>
                <g fill="#657184" font-size="12" text-anchor="middle">
                  <text x="40" y="122">位置 0：转 0</text>
                  <text x="175" y="122">位置 1：转 θ</text>
                  <text x="300" y="122">位置 2：转 2θ</text>
                </g>
              </svg>
            </div>
            <div class="story-arrow">关键收益：点积只剩下“差值”</div>
            <div class="story-panel">
              <strong>2. 魔法发生处：q 转 mθ、k 转 nθ，点积只依赖 (m−n)</strong>
              <p>注意力要算 query·key。query 在位置 m 转了 mθ，key 在位置 n 转了 nθ。两个箭头的夹角正好是 (m−n)θ，于是点积 = |q||k|·cos((m−n)θ)——<strong>绝对位置 m、n 各自消失了，只剩相对距离 (m−n)</strong>。这就是 RoPE 能自然泛化到更长序列的根本原因。</p>
              <svg viewBox="0 0 360 140" width="100%" style="max-width:520px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <defs><marker id="ar2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/></marker></defs>
                <line x1="60" y1="115" x2="60" y2="115" />
                <g stroke-width="2.5" fill="none">
                  <line x1="120" y1="115" x2="180" y2="35" stroke="#2563eb" marker-end="url(#ar2)" color="#2563eb"/>
                  <line x1="120" y1="115" x2="215" y2="80" stroke="#12805c" marker-end="url(#ar2)" color="#12805c"/>
                </g>
                <path d="M165 95 A 40 40 0 0 1 180 60" stroke="#bd6516" fill="none" stroke-width="1.5"/>
                <g font-size="13" font-weight="700">
                  <text x="185" y="32" fill="#2563eb">q（位置 m，转 mθ）</text>
                  <text x="222" y="84" fill="#12805c">k（位置 n，转 nθ）</text>
                  <text x="150" y="58" fill="#bd6516" font-size="12">夹角 = (m−n)θ</text>
                </g>
              </svg>
              <div class="formula"><span>q·k = |q||k| · cos((m−n)θ)</span><strong>只剩相对距离 (m−n)</strong></div>
            </div>
            <div class="story-arrow">为什么要“多组”频率：让模型既能分清近处，也能感知远处</div>
            <div class="story-panel">
              <strong>3. 不同维度配不同频率：像秒针/分针/时针</strong>
              <p>如果所有维度都用同一个 θ，转太快的远处会“绕圈重合”，转太慢的近处又分不开。RoPE 让 head_dim 里不同的“维度对”用不同频率：低索引维度转得快（管细微差别），高索引维度转得慢（管大跨度距离）。频率公式：θ_i = 10000<sup>−2i/d</sup>，i 越大频率越低。</p>
              <div class="rotate-demo"><span>维度对 0：高频（秒针）</span><span>维度对 1：中频（分针）</span><span>维度对 i：低频（时针）</span></div>
            </div>
            <div class="story-arrow">把上面这些角度，按 位置 × 频率 摆成一张表（这就是 TODO 1）</div>
            <div class="story-panel">
              <strong>4. 做表：每行一个位置，每列一组频率，格子里是“要转的角度”</strong>
              <p>用 <code>torch.outer(位置, 频率)</code> 一次性算出所有 (位置, 维度) 的角度，再用 <code>torch.polar(1, 角度)</code> 把角度变成复数 cos+i·sin（模长固定为 1，所以只转不缩）。最终表的 shape 是 [seq_len, head_dim/2]——列数是 head_dim/2，因为后面要两两配对成复数。</p>
              <table class="freq-table"><tr><th>位置 m</th><th>freq0 (快)</th><th>freq1</th><th>… freqi (慢)</th></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>1</td><td>1·θ0</td><td>1·θ1</td><td>1·θi</td></tr><tr><td>2</td><td>2·θ0</td><td>2·θ1</td><td>2·θi</td></tr></table>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：用 outer 把“位置 × 频率”摊成一张角度表</h4><pre><code>pos   = torch.arange(3).float()        # 位置 0,1,2
freqs = torch.tensor([1.0, 0.1])       # 两组频率：一快一慢

# outer：第 i 行第 j 列 = pos[i] * freqs[j]
angle = torch.outer(pos, freqs)        # shape [3, 2]
# tensor([[0.0, 0.0],     位置0：都不转
#         [1.0, 0.1],     位置1：快的转1.0，慢的转0.1
#         [2.0, 0.2]])    位置2：转得更多

# polar(模长, 角度) → 复数 cos+i·sin，模长全取 1 表示“只转不缩”
cis = torch.polar(torch.ones_like(angle), angle)  # complex [3,2]</code></pre><p class="syntax-tip">读法：<code>outer(a,b)[i,j] = a[i]*b[j]</code>，正好对应“位置 × 频率 = 该转的角度”。<code>polar</code> 把角度打包成可直接相乘的复数旋转因子 e<sup>iθ</sup>。</p></div>`,
      checkpoint: {
        question: "RoPE 为什么用“按位置旋转”而不是“给每个位置加一个固定向量”？",
        options: [
          "旋转后 q·k 只依赖相对距离 (m−n)，能自然泛化到更长序列",
          "因为旋转计算比加法更省显存",
          "因为加法会改变向量长度，旋转不会"
        ],
        answer: 0,
        explain: "q 转 mθ、k 转 nθ，点积里 m、n 合并成 (m−n)，模型直接获得相对位置感，不依赖训练时见过的绝对位置。"
      },
      homework: [
        "TODO 1：用 arange(0, dim, 2) 配 θ_i = 10000^(−2i/d) 算出每组频率。",
        "TODO 1：用 torch.outer(位置, 频率) 得到 [seq_len, dim/2] 的角度表。",
        "TODO 1：用 torch.polar(ones, 角度) 生成复数旋转表 freqs_cis，确认 shape 是 [seq_len, head_dim/2]。"
      ]
    },
    {
      id: "rope-complex-pairs",
      title: "把数字两两配成小箭头，用复数乘法转一下",
      todo: "TODO 2 / TODO 3",
      prerequisite: [
        "head_dim 是偶数，可以把相邻两个数配成一对 (x, y) 当作平面坐标。",
        "复数乘法 = 旋转：(a+bi)·(cosθ+i·sinθ) 就是把箭头 (a,b) 转 θ 度。",
        "旋转只改方向不改长度，所以旋转前后向量模长不变（测试会查这一点）。"
      ],
      intuition: "拿到上一关的复数角度表后，把 query/key 的最后一维也两两配对、看成复数，直接乘以角度表——一次复数乘法就完成了所有维度对的旋转。最后再把复数拆回实数、展平回原 shape。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>1. 两两配对：head_dim 个数 → head_dim/2 个小箭头</strong>
              <p>把最后一维 reshape 成 [..., head_dim/2, 2]，相邻两个数 (x0,x1) 组成一个 2D 坐标，再用 <code>view_as_complex</code> 看成一个复数 x0 + i·x1。head_dim=64 就得到 32 个复数（32 个待旋转的小箭头）。</p>
              <svg viewBox="0 0 360 120" width="100%" style="max-width:520px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <g font-size="13" text-anchor="middle">
                  <rect x="20" y="20" width="40" height="30" fill="#e8f0ff" stroke="#a9c2f6"/><text x="40" y="40">x0</text>
                  <rect x="60" y="20" width="40" height="30" fill="#e8f0ff" stroke="#a9c2f6"/><text x="80" y="40">x1</text>
                  <rect x="110" y="20" width="40" height="30" fill="#fff1dd" stroke="#e8b66f"/><text x="130" y="40">x2</text>
                  <rect x="150" y="20" width="40" height="30" fill="#fff1dd" stroke="#e8b66f"/><text x="170" y="40">x3</text>
                  <text x="230" y="40" fill="#657184">… 共 head_dim 个</text>
                </g>
                <g stroke="#657184" stroke-width="1" fill="none"><path d="M60 55 L80 80"/><path d="M150 55 L170 80"/></g>
                <g font-size="13" text-anchor="middle" font-weight="700">
                  <text x="80" y="100" fill="#1d4ed8">复数 x0 + i·x1</text>
                  <text x="170" y="100" fill="#73450d">复数 x2 + i·x3</text>
                </g>
              </svg>
            </div>
            <div class="story-arrow">复数乘法 = 旋转：乘以 e^{iθ} 就把箭头转 θ 度</div>
            <div class="story-panel">
              <strong>2. 一次乘法转全部：x_复数 × freqs_cis</strong>
              <p>把配好的复数张量乘以上一关的 freqs_cis（每个位置、每个维度对该转的角度）。复数乘法 (a+bi)(cosθ+i·sinθ) 自动展开成二维旋转矩阵的效果——一行代码替你转了所有维度对。</p>
              <svg viewBox="0 0 360 140" width="100%" style="max-width:520px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <defs><marker id="ar3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/></marker></defs>
                <circle cx="120" cy="95" r="62" fill="none" stroke="#dce2e8" stroke-dasharray="4 4"/>
                <line x1="120" y1="95" x2="178" y2="73" stroke="#94a3b8" stroke-width="2" marker-end="url(#ar3)" color="#94a3b8"/>
                <line x1="120" y1="95" x2="150" y2="40" stroke="#2563eb" stroke-width="2.5" marker-end="url(#ar3)" color="#2563eb"/>
                <path d="M178 73 A 62 62 0 0 0 152 42" stroke="#bd6516" fill="none" stroke-width="1.5" marker-end="url(#ar3)" color="#bd6516"/>
                <g font-size="13" font-weight="700">
                  <text x="182" y="72" fill="#94a3b8">旋转前</text>
                  <text x="120" y="34" fill="#2563eb">旋转后</text>
                  <text x="195" y="110" fill="#bd6516" font-size="12">转 θ，长度不变</text>
                </g>
              </svg>
              <div class="formula"><span>(a + bi)(cosθ + i·sinθ)</span><strong>= 把箭头 (a,b) 旋转 θ</strong></div>
            </div>
            <div class="story-arrow">转完拆回实数，并补上 FP32 精度这道坑</div>
            <div class="story-panel">
              <strong>3. 还原 shape + 精度陷阱</strong>
              <p>用 <code>view_as_real</code> 把复数拆回 (实部, 虚部) 两个数，再 <code>flatten</code> 合并回 head_dim，shape 与输入完全一致。<strong>关键坑</strong>：复数乘法在 FP16/BF16 下极易出 NaN，所以进复数前先 <code>.float()</code> 升到 FP32 计算，算完再 <code>type_as(xq)</code> 转回原精度——这是 LLaMA 源码强制要求的。旋转不改长度，所以测试会校验旋转前后模长一致。</p>
              <div class="mini-flow"><span>reshape [..,d/2,2]</span><span>view_as_complex</span><span>× freqs_cis</span><strong>view_as_real → flatten</strong></div>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：把二维坐标当复数旋转，再拆回来</h4><pre><code>import math
# 一个小箭头 (1, 0)，想转 90°
v = torch.tensor([[1.0, 0.0]])          # shape [1, 2]
z = torch.view_as_complex(v)            # 1 + 0i

theta = math.pi / 2                     # 90°
rot = torch.polar(torch.ones(1), torch.tensor([theta]))  # e^{iθ}

z_rot = z * rot                         # 复数乘法 = 旋转
out = torch.view_as_real(z_rot)         # 拆回 [[~0, 1]] → (0,1)
print(out, out.norm())                  # 长度仍是 1（旋转不变性）</code></pre><p class="syntax-tip">读法：<code>view_as_complex</code> 把最后一维 2 个数看成一个复数；乘 e<sup>iθ</sup> 旋转；<code>view_as_real</code> 再拆回两个实数。真实代码里进这步前要 <code>.float()</code> 防 NaN。</p></div>`,
      checkpoint: {
        question: "RoPE 里为什么进复数乘法前要先把 q/k 转成 float32？",
        options: [
          "复数乘法在 FP16/BF16 下容易出 NaN，先升 FP32 算完再转回原精度更稳",
          "因为 view_as_complex 只支持 float32 这一种类型",
          "为了让向量长度在旋转后变大"
        ],
        answer: 0,
        explain: "旋转涉及 cos/sin 连乘，半精度数值范围有限易发散；LLaMA 等源码统一先 .float() 再旋转，最后 type_as 转回。"
      },
      homework: [
        "TODO 2：把 xq/xk 先 .float()，reshape 成 [.., head_dim/2, 2]，再 view_as_complex。",
        "TODO 3：乘以广播后的 freqs_cis 完成旋转，view_as_real 后 flatten 回 head_dim。",
        "TODO 3：用 type_as(xq) 转回原精度；跑测试确认 shape 一致、旋转前后模长不变。"
      ]
    }
  ],
  "04": [
    {
      id: "attention-shape-map",
      title: "注意力到底在算什么：打分表 → 概率 → 混合 value",
      todo: "TODO 1 / TODO 3 / TODO 4",
      prerequisite: [
        "Q/K/V 都是输入 x 经过三个不同 Linear 投影得到的——可以理解成同一句话的三种“视角”。",
        "多头：把宽向量切成 H 份，每份独立算注意力，让不同头关注不同模式。计算时用 [B,H,S,D] 布局。",
        "softmax 把一行打分变成“加起来等于 1 的概率”，再用这组概率去加权求和。"
      ],
      intuition: "一句话版的注意力：每个 token 拿自己的 query 去问所有 token 的 key“我跟你多相关？”，得到一张打分表；softmax 把每一行变成概率；最后按这组概率把所有 token 的 value 混合起来，就是这个 token 的新表示。这一关把 4 个公式步骤配上 shape 流转图，一步步走通。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>1. 切多头：一根宽向量拆成 H 个小专家</strong>
              <p>投影后 Q 是 [B, S, H·D]，先 <code>view</code> 成 [B, S, H, D]，再 <code>transpose(1,2)</code> 成 [B, H, S, D]。为什么转置？因为后面要让每个头（H）独立地在它自己的 [S, D] 上做矩阵乘法，把 H 提到前面当“批量维”最方便。</p>
              <div class="mini-flow"><span>[B, S, H·D]</span><span>view → [B, S, H, D]</span><strong>transpose(1,2) → [B, H, S, D]</strong></div>
              <p style="color:#657184">直觉：H 个头像 H 个分工不同的读者，有的盯语法、有的盯指代，各读各的。</p>
            </div>
            <div class="story-arrow">每个头里：query 逐一去问每个 key（TODO 3 第一步）</div>
            <div class="story-panel">
              <strong>2. Q @ Kᵀ：打一张 token×token 的相关性分数表</strong>
              <p>Q 是 [.., S, D]，K 转置成 [.., D, S]，相乘得到 [.., S, S]。第 i 行第 j 列 = 第 i 个 query 和第 j 个 key 的点积 = “token i 该多关注 token j”。</p>
              <svg viewBox="0 0 360 150" width="100%" style="max-width:480px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <g font-size="12" text-anchor="middle">
                  <text x="30" y="20" fill="#2563eb" font-weight="700">query↓ / key→</text>
                  <text x="120" y="20">k0</text><text x="170" y="20">k1</text><text x="220" y="20">k2</text>
                  <text x="80" y="50" fill="#2563eb">q0</text><text x="80" y="85" fill="#2563eb">q1</text><text x="80" y="120" fill="#2563eb">q2</text>
                </g>
                <g font-size="12" text-anchor="middle">
                  <rect x="100" y="38" width="44" height="24" fill="#fde68a" stroke="#d9a15a"/><text x="122" y="55">9.1</text>
                  <rect x="150" y="38" width="44" height="24" fill="#fef3c7" stroke="#d9a15a"/><text x="172" y="55">2.0</text>
                  <rect x="200" y="38" width="44" height="24" fill="#fef9e7" stroke="#d9a15a"/><text x="222" y="55">0.3</text>
                  <rect x="100" y="73" width="44" height="24" fill="#fef3c7" stroke="#d9a15a"/><text x="122" y="90">1.5</text>
                  <rect x="150" y="73" width="44" height="24" fill="#fde68a" stroke="#d9a15a"/><text x="172" y="90">8.4</text>
                  <rect x="200" y="73" width="44" height="24" fill="#fef9e7" stroke="#d9a15a"/><text x="222" y="90">0.6</text>
                  <rect x="100" y="108" width="44" height="24" fill="#fef9e7" stroke="#d9a15a"/><text x="122" y="125">0.4</text>
                  <rect x="150" y="108" width="44" height="24" fill="#fef3c7" stroke="#d9a15a"/><text x="172" y="125">1.1</text>
                  <rect x="200" y="108" width="44" height="24" fill="#fde68a" stroke="#d9a15a"/><text x="222" y="125">7.9</text>
                </g>
                <text x="300" y="88" font-size="12" fill="#657184" text-anchor="middle">颜色越深=</text>
                <text x="300" y="104" font-size="12" fill="#657184" text-anchor="middle">越相关</text>
              </svg>
            </div>
            <div class="story-arrow">除以 √D，再 softmax（TODO 3 第二、三步）</div>
            <div class="story-panel">
              <strong>3. 为什么要除以 √D？防止 softmax “一家独大”</strong>
              <p>点积是 D 个数相加，D 越大，分数的方差越大、数值越极端。直接 softmax 会几乎把全部概率压到一个 token 上（接近 one-hot），其它位置梯度趋近 0，训练学不动。除以 √D 把方差拉回稳定区间，softmax 才平滑。</p>
              <div class="formula"><span>scores = Q·Kᵀ / √D</span><span>probs = softmax(scores)</span><strong>每行加起来 = 1</strong></div>
              <p style="color:#657184">Causal Mask：生成任务里给上三角填 −∞，让 token 只能看自己和左边——softmax 后这些位置概率自然变 0。</p>
            </div>
            <div class="story-arrow">用概率加权 value，再把多头拼回去（TODO 3 末 + TODO 4）</div>
            <div class="story-panel">
              <strong>4. probs @ V → 合并多头 → 输出投影</strong>
              <p>probs [.., S, S] 乘 V [.., S, D] 得 [.., S, D]：每个 token 的新向量 = 按注意力概率混合所有 token 的 value。最后 <code>transpose(1,2)</code> 把 H 放回去、<code>contiguous().view</code> 合并成 [B, S, H·D]，再过 <code>o_proj</code> 回到 hidden_dim。</p>
              <div class="mini-flow"><span>probs @ V → [B,H,S,D]</span><span>transpose+view → [B,S,H·D]</span><strong>o_proj → [B,S,hidden]</strong></div>
              <p style="color:#bd6516"><strong>易错点</strong>：<code>view</code> 前必须 <code>.contiguous()</code>，因为 transpose 只是改了步长、内存没真正重排，直接 view 会报错。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：批量矩阵乘法只动最后两维</h4><pre><code>import torch.nn.functional as F
# [B, H, S, D] 想象成 B*H 个独立的 [S, D] 小矩阵
q = torch.randn(2, 4, 5, 8)   # 5 个 query，每个 8 维
k = torch.randn(2, 4, 7, 8)   # 7 个 key

# K 最后两维转置 [.., D, S]，再和 Q 相乘 → 打分表 [.., S, S']
scores = q @ k.transpose(-2, -1)      # [2, 4, 5, 7]
probs  = F.softmax(scores / 8**0.5, dim=-1)  # 沿 key 维归一化
# 前面的 batch、head 维（2,4）原样保留，只有最后两维参与乘法</code></pre><p class="syntax-tip">读法：<code>@</code> 对 4D 张量只在最后两维做矩阵乘法，前面的维度当“批量”广播。<code>softmax(dim=-1)</code> 表示“对每一行的 key 打分做归一化”。</p></div>`,
      checkpoint: {
        question: "Scaled Dot-Product Attention 里为什么要除以 √head_dim？",
        options: [
          "点积随维度增大方差变大，不缩放 softmax 会接近 one-hot 导致梯度消失",
          "为了让 Q 和 K 的 shape 对齐才能相乘",
          "为了把注意力分数变成整数"
        ],
        answer: 0,
        explain: "D 个数相加方差 ∝ D，缩放到 1/√D 把分数拉回稳定区间，softmax 才平滑、梯度才不消失。"
      },
      homework: [
        "TODO 1：投影结果 view 成 [B,S,H,D] 再 transpose(1,2) → [B,H,S,D]。",
        "TODO 3：算 Q@Kᵀ/√D，加 mask，softmax，再 @ V。",
        "TODO 4：transpose 回来 + contiguous().view 合并多头 → o_proj。"
      ]
    },
    {
      id: "gqa-kv-cache",
      title: "KV Cache 与 GQA：推理为什么慢，又怎么省显存",
      todo: "TODO 2 / repeat_kv",
      prerequisite: [
        "自回归生成：模型一次只吐 1 个 token，然后把它接回输入再算下一个。",
        "算第 N 个 token 时要和前面所有 token 的 Key/Value 做注意力——但前面那些 K/V 每步都一样，没必要重算。",
        "GQA：让多个 Query 头共享同一组 K/V 头，介于 MHA（各用各的）和 MQA（全共享一个）之间。"
      ],
      intuition: "两个独立但相关的优化：① KV Cache——把算过的 K/V 存起来，每步只算新 token 的，避免 O(N²) 重算；② GQA——只存少量共享的 KV 头省显存，真正算注意力前再临时复制成 Query 头数。注意顺序：先拼接缓存、存下未扩展的 KV，最后才 repeat_kv 扩充。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>1. 痛点：不缓存就要每步重算整段历史（O(N²)）</strong>
              <p>生成第 100 个 token 时，注意力要用到前 99 个 token 的 K/V。可这 99 个 K/V 在前几步早就算过了，而且不会变。每步都从头算 → 总开销 ∝ N²，序列越长越爆炸。</p>
            </div>
            <div class="story-arrow">解法：把历史 K/V 存进 Cache，每步只算新增的那一个（TODO 2）</div>
            <div class="story-panel">
              <strong>2. KV Cache：沿 seq 维 cat 一下就行</strong>
              <p>缓存里存着历史 K/V，形状 [B, H_kv, 旧长度, D]。新 token 算出自己的 xk/xv（长度 1），用 <code>torch.cat(..., dim=2)</code> 拼到时间轴末尾，长度 +1。下一步再接着拼。</p>
              <svg viewBox="0 0 360 96" width="100%" style="max-width:500px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <g font-size="12" text-anchor="middle">
                  <rect x="20" y="35" width="40" height="30" fill="#e8f0ff" stroke="#a9c2f6"/><text x="40" y="54">k0</text>
                  <rect x="62" y="35" width="40" height="30" fill="#e8f0ff" stroke="#a9c2f6"/><text x="82" y="54">k1</text>
                  <rect x="104" y="35" width="40" height="30" fill="#e8f0ff" stroke="#a9c2f6"/><text x="124" y="54">k2</text>
                  <text x="82" y="22" fill="#657184">cache（已存）</text>
                  <text x="170" y="54" font-size="20" fill="#12805c">＋</text>
                  <rect x="195" y="35" width="40" height="30" fill="#e7f6ee" stroke="#99d6bf"/><text x="215" y="54">k3</text>
                  <text x="215" y="22" fill="#12805c">新 token</text>
                  <text x="270" y="54" font-size="16" fill="#657184">→</text>
                  <rect x="295" y="38" width="58" height="24" fill="#fff1dd" stroke="#e8b66f"/><text x="324" y="54">长度+1</text>
                </g>
              </svg>
            </div>
            <div class="story-arrow">另一条线：怎么让 Cache 本身更小？→ GQA</div>
            <div class="story-panel">
              <strong>3. MHA → MQA → GQA：在效果和显存间取平衡</strong>
              <p>MHA 每个 Query 头配一个独立 KV 头，Cache 最大、效果最好；MQA 所有 Query 头共用 1 个 KV 头，Cache 最小但效果掉；GQA 折中——分组共享。下图 4 个 Q 头、2 个 KV 头，每 2 个 Q 共享 1 个 KV，Cache 直接砍半。</p>
              <svg viewBox="0 0 360 120" width="100%" style="max-width:500px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:6px">
                <g font-size="12" text-anchor="middle">
                  <rect x="20" y="15" width="34" height="24" fill="#e8f0ff" stroke="#a9c2f6"/><text x="37" y="31">Q0</text>
                  <rect x="58" y="15" width="34" height="24" fill="#e8f0ff" stroke="#a9c2f6"/><text x="75" y="31">Q1</text>
                  <rect x="120" y="15" width="34" height="24" fill="#fff1dd" stroke="#e8b66f"/><text x="137" y="31">Q2</text>
                  <rect x="158" y="15" width="34" height="24" fill="#fff1dd" stroke="#e8b66f"/><text x="175" y="31">Q3</text>
                  <rect x="38" y="80" width="36" height="24" fill="#c8d9ff" stroke="#1e3f88"/><text x="56" y="96">KV A</text>
                  <rect x="138" y="80" width="36" height="24" fill="#f8d8aa" stroke="#73450d"/><text x="156" y="96">KV B</text>
                </g>
                <g stroke="#94a3b8" stroke-width="1.5" fill="none">
                  <path d="M37 39 L56 80"/><path d="M75 39 L56 80"/>
                  <path d="M137 39 L156 80"/><path d="M175 39 L156 80"/>
                </g>
                <text x="250" y="60" font-size="12" fill="#657184" text-anchor="middle">2 组共享</text>
                <text x="250" y="78" font-size="12" fill="#657184" text-anchor="middle">Cache 砍半</text>
              </svg>
            </div>
            <div class="story-arrow">关键顺序：先存“瘦” KV，再临时扩充（repeat_kv）</div>
            <div class="story-panel">
              <strong>4. 延迟扩充：缓存窄的，算注意力前才复制宽的</strong>
              <p>矩阵乘法要求 Q 头数和 K/V 头数一致。GQA 的解法不是把 KV 存成宽的（那就退化回 MHA、白省了），而是<strong>只缓存窄的 num_kv_heads 个头</strong>，每次前向用 <code>repeat_kv</code> 临时把每个 KV 头复制 num_queries_per_kv 份，对齐到 Query 头数。</p>
              <div class="mini-flow"><span>cat 更新 cache（窄）</span><span>new_kv_cache = 窄 KV</span><strong>repeat_kv → 宽，喂给注意力</strong></div>
              <p style="color:#bd6516"><strong>易错点</strong>：<code>repeat_kv</code> 必须在“存进 new_kv_cache 之后”才做。先扩充再缓存 = 缓存了宽 KV = 显存退回 MHA，GQA 优势归零。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：沿时间维拼接 + 复制头</h4><pre><code># ① KV Cache：把新 K 拼到历史末尾（dim=2 是 seq 维）
cache = torch.randn(2, 2, 5, 8)   # [B, H_kv, 旧长度5, D]
new_k = torch.randn(2, 2, 1, 8)   # 新 token 的 K，长度1
updated = torch.cat([cache, new_k], dim=2)  # [2, 2, 6, 8]

# ② repeat_kv：每个 KV 头复制 n_rep 份，对齐 Query 头数
#   先插一个新轴 expand，再 reshape 把它并进 head 维
n_rep = 2                          # 4 个 Q 头 / 2 个 KV 头
x = updated[:, :, None, :, :].expand(2, 2, n_rep, 6, 8)
x = x.reshape(2, 2 * n_rep, 6, 8)  # [2, 4, 6, 8] 对齐 Q</code></pre><p class="syntax-tip">读法：<code>cat(dim=2)</code> 在 seq 维接长；<code>expand</code> 不复制内存只“假装”有 n_rep 份，<code>reshape</code> 时才真正铺开，把 KV 头数从 2 撑到 4。</p></div>`,
      checkpoint: {
        question: "GQA 实现里，为什么 repeat_kv 要放在“更新 new_kv_cache 之后”？",
        options: [
          "只缓存窄的 num_kv_heads 个头才省显存；先扩充再缓存会把 Cache 撑回 MHA 大小",
          "因为 repeat_kv 会改变 batch size",
          "因为 cat 操作必须在 softmax 之后"
        ],
        answer: 0,
        explain: "GQA 省显存的本质是 Cache 只存窄 KV。扩充只是为了满足矩阵乘法的头数对齐，是临时的、每步重做的，绝不能写进缓存。"
      },
      homework: [
        "TODO 2：用 torch.cat([k_cache, xk], dim=2) 在 seq 维拼接历史与新 KV。",
        "确认 new_kv_cache 存的是拼接后、repeat_kv 之前的“窄”KV。",
        "理解 repeat_kv：num_queries_per_kv = num_heads // num_kv_heads，扩充后头数对齐 Query。"
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
