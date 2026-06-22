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
      predict: {
        hook: "普通 MLP 升维只用 1 块矩阵就够了。SwiGLU 偏要拆成 gate + up 两块，凭空多花一倍参数——它图的到底是什么？",
        question: "先押一个假设：SwiGLU 多花一块升维矩阵，最核心的目的是？",
        options: [
          "让模型一次能处理两倍长的序列",
          "让“哪些信息该通过”由数据自己学出来，而不是像 ReLU 那样写死",
          "为了之后可以省掉 down_proj 降维矩阵"
        ],
        answer: 1,
        revealNote: "押好了。带着“这块矩阵是为可学习的开关买单”这个念头，往下看它怎么用 gate ⊙ up 实现。"
      },
      checkpoint: {
        question: "巩固：gate 和 up 两条支路的输出，必须满足什么条件才能执行 gate ⊙ up？",
        options: ["两者 shape 完全相同（逐元素相乘的前提）", "两者必须都是正数", "up 的维度要比 gate 大一倍"],
        answer: 0,
        explain: "逐元素相乘（Hadamard）要求两个张量形状逐维一致，所以 gate 和 up 的 Linear 输出维度都设成 intermediate_size。"
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
      predict: {
        hook: "传统 MLP 中间层惯例放大到 4d。换成 SwiGLU 后，LLaMA 却把它缩到了奇怪的 8/3 d ≈ 2.67d。这个分数到底从哪冒出来的？",
        question: "先押一个假设：8/3 这个比例最可能是怎么定下来的？",
        options: [
          "GPU 硬件只支持 8/3 这一种维度比例",
          "工程师试出来的经验值，没有公式",
          "解一个方程定出来的——让 SwiGLU 的总参数量正好等于传统 MLP"
        ],
        answer: 2,
        revealNote: "押好了。如果真是“解方程”，那方程长什么样？往下数清两边的矩阵，亲手解一遍。"
      },
      checkpoint: {
        question: "巩固（动手算）：hidden_size = 12 时，按 8/3 算出的理论 intermediate_size 约是多少？",
        options: ["32", "48", "12"],
        answer: 0,
        explain: "8/3 × 12 = 32。若还要求对齐 multiple_of，再在此基础上向上取整。"
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
      predict: {
        hook: "老办法是给每个位置发一个固定向量、加到 token 上。RoPE 偏不加，而是按位置把向量“转”一个角度。多此一举？还是另有玄机？",
        question: "先押一个假设：用“旋转”代替“相加”，最关键的好处是？",
        options: [
          "算 q·k 时结果只跟两个 token 的位置差 (m−n) 有关，模型天生懂“相对距离”",
          "旋转比加法计算更快、更省显存",
          "旋转能让向量变长，从而携带更多信息"
        ],
        answer: 0,
        revealNote: "押好了。下面就来验证：q 转 mθ、k 转 nθ，点积里 m、n 会发生什么？"
      },
      checkpoint: {
        question: "巩固（动手算）：seq_len=8、head_dim=64 时，freqs_cis 这张角度表的 shape 应该是？",
        options: ["[8, 32]", "[8, 64]", "[64, 8]"],
        answer: 0,
        explain: "行数 = 位置数 = 8；列数 = head_dim/2 = 32，因为后面要把相邻两维配成一个复数。"
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
      predict: {
        hook: "head_dim=64，意味着要把 32 对维度各转一个角度。如果写循环一对一对转，又慢又丑。有没有一行代码全转完的办法？",
        question: "先押一个假设：RoPE 用什么技巧一次性旋转所有维度对？",
        options: [
          "用一个 for 循环逐对调用旋转矩阵",
          "把每对维度看成一个复数，整体乘以复数 e^{iθ}（复数乘法 = 旋转）",
          "先 softmax 归一化，再按概率旋转"
        ],
        answer: 1,
        revealNote: "押好了。关键就是“两个实数 → 一个复数 → 乘 e^{iθ}”。往下看它怎么配对、怎么乘、怎么拆回来。"
      },
      checkpoint: {
        question: "巩固：RoPE 进复数乘法前为什么要先把 q/k 转成 float32？",
        options: [
          "复数乘法在 FP16/BF16 下容易出 NaN，先升 FP32 算完再 type_as 转回更稳",
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
        "先记一个生活类比：注意力就像“查资料”——你心里的问题是 Query，每份资料的标题是 Key，资料的正文是 Value。标题越对得上你的问题，就越多地去读它的正文。",
        "Q/K/V 都是同一个输入 x 各乘一个不同的 Linear 矩阵得到的。同一句话，从“我想问什么(Q)/我能回答什么(K)/我有什么内容(V)”三个角度各投影一次。",
        "“多头”= 把每个 token 的长向量切成 H 段，每段单独做一遍上面的查资料，让不同的“头”各盯一种规律（有的盯语法、有的盯指代）。计算时摆成 [B, H, S, D]：批次、头、序列长度、每头维度。",
        "softmax：把一排任意大小的分数压成“加起来等于 1 的概率”，分数越高拿到的概率越大。"
      ],
      intuition: "一句话版：每个 token 拿自己的 Query 去和所有 token 的 Key 比一比“我跟你多相关”，得到一张打分表；softmax 把每一行变成一组概率；再按这组概率去混合所有 token 的 Value，混出来的就是这个 token 的新表示。下面把这 4 步拆开、每步配图，并在关键处停下来让你先想一想再看解释。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先建立画面：一句话里，每个词都在“偷看”别的词</strong>
              <p>读“它躺在桌上”时，你要搞清楚“它”指什么，就得回头看前面的词。注意力做的就是这件事：让每个 token 都能按需去看序列里的其他 token，再把看到的信息揉进自己的新表示里。“按需”——就是下面那张打分表决定的。</p>
              <details class="think">
                <summary>想一想：为什么 Q、K、V 要用三个不同的矩阵，而不是直接拿 x 本身去两两点积？</summary>
                <div class="think-body">
                  <p>如果直接用 x 和 x 点积，那“我作为提问方”和“我作为被查方”用的是同一个向量，模型没法分别学习“我想找什么”和“我能提供什么匹配特征”这两件不同的事。</p>
                  <p>拆成 Q、K 两个独立投影后，同一个词可以“以一种姿态发问、以另一种姿态被检索”，匹配关系才学得灵活。V 再单独一个矩阵，是因为“被选中后该交出什么内容”又是第三件事。三个角色、三个矩阵，各司其职。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">第一步：把每个 token 的长向量切成 H 个头（TODO 1）</div>
            <div class="story-panel">
              <strong>1. 切多头：一根宽向量拆成 H 个小专家</strong>
              <p>投影后 Q 是 [B, S, H·D]，先 <code>view</code> 成 [B, S, H, D]（把最后一根长向量切成 H 段），再 <code>transpose(1,2)</code> 成 [B, H, S, D]。为什么要转置？因为下一步要让每个头（H）<strong>各自独立</strong>地在它自己的 [S, D] 小表上做矩阵乘法，把 H 提到前面、和 batch 并排当“批量维”最顺手。</p>
              <div class="mini-flow"><span>[B, S, H·D]</span><span>view → [B, S, H, D]</span><strong>transpose(1,2) → [B, H, S, D]</strong></div>
              <p style="color:#657184">直觉：H 个头像 H 个分工不同的读者，有的盯语法、有的盯指代，各读各的，最后再汇总。</p>
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
              <p>点积 = 把 D 个数乘起来再相加。D 越大，这个和的“波动范围”（方差）就越大，分数越容易出现某个特别大的极端值。直接喂给 softmax，它会几乎把全部概率压到那一个 token 上（接近非 0 即 1 的 one-hot），别的位置概率趋近 0、梯度也趋近 0，模型就学不动了。除以 √D 正好把方差拉回到大约 1 的稳定区间，softmax 才平滑、可训练。</p>
              <div class="formula"><span>scores = Q·Kᵀ / √D</span><span>probs = softmax(scores)</span><strong>每行加起来 = 1</strong></div>
              <details class="think">
                <summary>想一想：为什么偏偏是除以“√D”，而不是除以 D 或别的数？</summary>
                <div class="think-body">
                  <p>假设 Q、K 的每个分量都是均值 0、方差 1 的独立随机数。点积是 D 个“乘积项”相加，每项方差约为 1，独立相加后总方差 ≈ D，于是标准差 ≈ √D。</p>
                  <p>我们想把分数的“典型尺度”拉回到 1 左右，所以除以标准差 √D 最自然——除以 D 会矫枉过正、把分数压得太小，反而让 softmax 太平、谁都差不多。这就是原论文选 √D 的来历。</p>
                </div>
              </details>
              <p style="color:#657184">Causal Mask：生成任务里给打分表的“上三角”填 −∞（代表“还没出现的未来 token”），softmax 后这些位置概率自然变 0，保证每个 token 只能看自己和左边。</p>
            </div>
            <div class="story-arrow">用概率加权 value，再把多头拼回去（TODO 3 末 + TODO 4）</div>
            <div class="story-panel">
              <strong>4. probs @ V → 合并多头 → 输出投影</strong>
              <p>probs [.., S, S] 乘 V [.., S, D] 得 [.., S, D]：每个 token 的新向量 = 按注意力概率混合所有 token 的 value。最后 <code>transpose(1,2)</code> 把 H 放回去、<code>contiguous().view</code> 合并成 [B, S, H·D]，再过 <code>o_proj</code> 回到 hidden_dim。</p>
              <div class="mini-flow"><span>probs @ V → [B,H,S,D]</span><span>transpose+view → [B,S,H·D]</span><strong>o_proj → [B,S,hidden]</strong></div>
              <p style="color:#bd6516"><strong>易错点</strong>：<code>view</code> 前必须 <code>.contiguous()</code>，因为 transpose 只是改了“怎么读这块内存”的步长、并没真正搬动数据，直接 view 会因内存不连续报错。</p>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：这张 S×S 打分表，正是大模型的成本中心</div>
              <p>注意力之所以打败了 RNN，关键在于它<strong>一步就能让任意两个 token 直接相连</strong>（RNN 要一格一格传，远距离信息会衰减），而且整张打分表能在 GPU 上并行算完。这是 Transformer 能堆大、能记长上下文的根本。</p>
              <p>但代价也在这张表：它的大小是 S×S，序列翻倍、显存和计算就翻 4 倍。这就是为什么“长上下文”这么贵，也催生了 <strong>FlashAttention</strong>（不把整张表落地到显存，分块边算边累加）这类工程优化——本仓库后面的 FlashAttention 关就是专门拆这个的。今天你手写的这版是“教科书版”，理解它才能看懂工业版在优化什么。</p>
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
      predict: {
        hook: "注意力公式里，Q·Kᵀ 算完不直接 softmax，非要先除以一个 √head_dim。这个除法看着多余，去掉会怎样？",
        question: "先押一个假设：如果去掉 √head_dim 这个缩放，最可能出什么问题？",
        options: [
          "Q 和 K 的 shape 对不上，根本没法相乘",
          "注意力分数会变成小数，不再是整数",
          "维度越大点积方差越大，softmax 会几乎全压到一个 token 上，梯度消失、学不动"
        ],
        answer: 2,
        revealNote: "押好了。带着“缩放是为了驯服方差”这个想法，往下看 Q·Kᵀ → /√D → softmax 这条链。"
      },
      checkpoint: {
        question: "巩固（动手算）：q=[2,4,5,8]、k=[2,4,7,8]，scores = q @ k.transpose(-2,-1) 的 shape 是？",
        options: ["[2, 4, 5, 7]", "[2, 5, 4, 7]", "[2, 4, 8, 8]"],
        answer: 0,
        explain: "batch=2、head=4 原样保留，只动最后两维：5 个 query 对 7 个 key 打分 → [2,4,5,7]。"
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
        "先理解“自回归生成”：模型像打字一样一次只吐 1 个字，然后把这个字接到句子末尾，再用更长的句子算下一个字，循环往复。",
        "算新字时，注意力要让它去看前面所有字的 Key/Value。但前面那些字的 K/V 上一步就算过了、而且不会再变——重算就是纯浪费。",
        "三种省法的名字：MHA（每个 Query 头配一个专属 KV 头，最费）、MQA（所有 Query 头挤用 1 个 KV 头，最省但效果掉）、GQA（分组共享，折中，LLaMA-2/3 采用）。"
      ],
      intuition: "这一关是两个独立但配合使用的优化：① KV Cache——把算过的 K/V 存起来，每步只算新字那一个，把“每步重算整段历史”的浪费去掉；② GQA——让缓存里只存少量共享的 KV 头来省显存，等真正算注意力前再临时把它们复制成和 Query 一样多。最关键的一个细节：复制（repeat_kv）必须放在“存进缓存之后”，先存窄的、后扩宽的——下面会讲为什么反过来就前功尽弃。",
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
              <details class="think">
                <summary>想一想：KV Cache 把时间省下来了，那它代价是什么？</summary>
                <div class="think-body">
                  <p>天下没有免费的午餐：省了重算的时间，代价是<strong>显存</strong>。每生成一个 token，缓存就长大一截，要一直占着显存直到这句话生成完。</p>
                  <p>粗略感受一下规模：层数 L、KV 头数 H_kv、每头维度 D、序列长 S、再算上 K 和 V 两份，缓存大小 ≈ 2 · L · H_kv · D · S。序列越长占用线性增长——长上下文推理时，KV Cache 往往比模型权重还吃显存。这正是下面要用 GQA 来砍它的原因。</p>
                </div>
              </details>
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
              <details class="think">
                <summary>想一想：MQA 把 KV 砍到只剩 1 个头、最省显存，为什么大家不直接全用它？</summary>
                <div class="think-body">
                  <p>KV 头可以理解成“资料库的不同检索视角”。MHA 有 H 个独立视角，能从很多角度去匹配；MQA 把它压成 1 个，所有 Query 头只能共用同一套 Key/Value，表达力明显变窄，效果通常会掉。</p>
                  <p>GQA 是中庸之道：保留几组（比如 8 组）视角，让每组被若干 Query 头共享。显存接近 MQA，效果接近 MHA——这就是 LLaMA-2/3 选它的原因。工程里很多决策都是这种“在两个极端之间找平衡点”。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">关键顺序：先存“瘦” KV，再临时扩充（repeat_kv）</div>
            <div class="story-panel">
              <strong>4. 延迟扩充：缓存窄的，算注意力前才复制宽的</strong>
              <p>矩阵乘法要求 Q 头数和 K/V 头数一致。GQA 的解法不是把 KV 存成宽的（那就退化回 MHA、白省了），而是<strong>只缓存窄的 num_kv_heads 个头</strong>，每次前向用 <code>repeat_kv</code> 临时把每个 KV 头复制 num_queries_per_kv 份，对齐到 Query 头数。</p>
              <div class="mini-flow"><span>cat 更新 cache（窄）</span><span>new_kv_cache = 窄 KV</span><strong>repeat_kv → 宽，喂给注意力</strong></div>
              <p style="color:#bd6516"><strong>易错点</strong>：<code>repeat_kv</code> 必须在“存进 new_kv_cache 之后”才做。先扩充再缓存 = 缓存里存了宽 KV = 显存退回 MHA，GQA 等于白做。</p>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：GQA 为什么是 70B 模型能跑起来的关键之一</div>
              <p>LLaMA-2 70B 有 64 个 Query 头，但只用 8 个 KV 头（每 8 个 Q 共享 1 个 KV），KV Cache 直接降到 MHA 的 <strong>1/8</strong>——在长上下文推理时能省下几十 GB 显存，这往往决定了一张卡到底装不装得下、能开多长上下文、能同时服务多少用户。</p>
              <p>而“先存窄、用时再扩”的延迟扩充，正是 vLLM、TensorRT-LLM 这些推理框架的标准做法。它们之所以敢这么干，是因为注意力是 <strong>Memory-bound</strong>（瓶颈在搬数据而非算数）——临时复制几下的计算量可以忽略，省下的显存带宽才是大头。本仓库后面的 vLLM PagedAttention 关会接着讲怎么把这些 KV 块管得更省。</p>
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
      predict: {
        hook: "GQA 省显存，靠的是只存少量“窄”KV 头。但矩阵乘法又要求 KV 头数和 Query 头数一致——于是代码里有个 repeat_kv 把 KV 复制宽。问题来了：这步该放在缓存之前还是之后？",
        question: "先押一个假设：repeat_kv（把 KV 扩充到 Query 头数）应该放在哪里？",
        options: [
          "放在更新缓存之前——先扩充好再存，省得每步重复扩充",
          "放在更新缓存之后——缓存里只存窄 KV，每步前向时临时扩充",
          "放在 softmax 之后——等算完注意力再扩充"
        ],
        answer: 1,
        revealNote: "押好了。这一步顺序是 GQA 省显存成败的关键，往下看“先存窄、后扩宽”为什么不能反过来。"
      },
      checkpoint: {
        question: "巩固（动手算）：num_heads=8、num_kv_heads=2 时，每个 KV 头要被复制几份去服务 Query 头？",
        options: ["4", "2", "8"],
        answer: 0,
        explain: "num_queries_per_kv = num_heads // num_kv_heads = 8 // 2 = 4，所以 repeat_kv 把每个 KV 头复制 4 份。"
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
      title: "LLaMA MLP：为什么是 gate、up、down 三条路",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "标准 Transformer 的 MLP 通常是两层：先把 hidden_size 放大到 4 倍左右，再用激活函数，最后投影回 hidden_size。",
        "SwiGLU 多了一条 gate 分支：一条分支判断“哪些特征该打开”，另一条分支提供“要传递的内容”，两者逐元素相乘。",
        "LLaMA 里线性层常用 bias=False。对初学者来说先按接口写对；直觉上它少一组偏置参数，归一化和残差已经承担了很多平移校正。",
        "写 PyTorch 模块时先分清两件事：__init__ 负责注册零件，forward 负责把张量按顺序流过这些零件。"
      ],
      intuition: "普通 MLP 像“把信息放大后统一加工”；SwiGLU 更像“内容通道 + 阀门通道”：up_proj 负责把可用内容拿出来，gate_proj 经 SiLU 后决定每个中间特征开多大，最后 down_proj 把中间表示收回到原来的 hidden_size，方便残差相加。写作业时不要先背答案，先追每一步的 shape，shape 对了，代码通常就顺了。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先把 TODO 1/2 拆成两个小任务</strong>
              <p>TODO 1 在 <code>__init__</code>：只是把三块 Linear 注册成模块，不做计算。TODO 2 在 <code>forward</code>：拿输入 x 依次过 gate/up/down。新手最容易把“定义层”和“调用层”混在一起；把它们分开，作业难度会立刻下降一半。</p>
              <table class="freq-table">
                <tr><th style="width:110px">任务</th><th>你要完成什么</th><th>检查自己有没有写偏</th></tr>
                <tr>
                  <td><strong>TODO 1</strong></td>
                  <td style="text-align:left">在 <code>__init__</code> 里注册 <code>gate_proj</code>、<code>up_proj</code>、<code>down_proj</code> 三个线性层。</td>
                  <td style="text-align:left">这里只搭零件，不调用 <code>F.silu</code>，也不处理输入张量。</td>
                </tr>
                <tr>
                  <td><strong>TODO 2</strong></td>
                  <td style="text-align:left">在 <code>forward</code> 里让 x 走完整条数据流：gate 分支开门，up 分支给内容，down 分支收回维度。</td>
                  <td style="text-align:left">先分别算两条分支，再逐元素相乘；不要把 <code>*</code> 写成矩阵乘法。</td>
                </tr>
              </table>
            </div>
            <div class="story-arrow">先看普通 MLP，再看 LLaMA 为什么拆成三条线</div>
            <div class="story-panel">
              <strong>1. 先看普通 MLP：两次投影，中间放大</strong>
              <p>输入 x 的 shape 是 [B, S, d]。普通 MLP 常见写法是 <code>d → 4d → d</code>：先扩宽，让模型有更多中间特征可组合；再投影回 d，才能和残差主路相加。</p>
              <div class="mini-flow"><span>x<br>[B,S,d]</span><span>up<br>[B,S,4d]</span><span>activation</span><strong>down<br>[B,S,d]</strong></div>
            </div>
            <div class="story-arrow">SwiGLU 多一条 gate 分支：不是只激活，而是“先决定开关，再传内容”</div>
            <div class="story-panel">
              <strong>2. gate 和 up 必须同 shape，因为它们要逐元素相乘</strong>
              <p><code>gate_proj(x)</code> 和 <code>up_proj(x)</code> 都输出 [B, S, intermediate_size]。SiLU 让 gate 变成平滑门控信号：有些位置放大、有些位置压低；再和 up 分支的内容逐元素相乘。</p>
              <svg viewBox="0 0 520 180" width="100%" style="max-width:680px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <defs>
                  <marker id="arrow05a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L7,3 z" fill="#657184"></path>
                  </marker>
                </defs>
                <g font-size="13" text-anchor="middle">
                  <rect x="18" y="70" width="72" height="36" rx="8" fill="#eaf2ff" stroke="#a9c2f6"></rect>
                  <text x="54" y="93">x [B,S,d]</text>
                  <line x1="90" y1="88" x2="145" y2="50" stroke="#657184" stroke-width="2" marker-end="url(#arrow05a)"></line>
                  <line x1="90" y1="88" x2="145" y2="126" stroke="#657184" stroke-width="2" marker-end="url(#arrow05a)"></line>
                  <rect x="145" y="26" width="110" height="38" rx="8" fill="#fff6e8" stroke="#e8b66f"></rect>
                  <text x="200" y="50">gate_proj → SiLU</text>
                  <rect x="145" y="112" width="110" height="38" rx="8" fill="#e7f6ee" stroke="#99d6bf"></rect>
                  <text x="200" y="136">up_proj</text>
                  <text x="315" y="92" font-size="24" fill="#8a4d0a">⊙</text>
                  <line x1="255" y1="45" x2="302" y2="82" stroke="#657184" stroke-width="2" marker-end="url(#arrow05a)"></line>
                  <line x1="255" y1="131" x2="302" y2="98" stroke="#657184" stroke-width="2" marker-end="url(#arrow05a)"></line>
                  <line x1="328" y1="90" x2="376" y2="90" stroke="#657184" stroke-width="2" marker-end="url(#arrow05a)"></line>
                  <rect x="376" y="70" width="92" height="38" rx="8" fill="#f4f1ff" stroke="#c9bdf0"></rect>
                  <text x="422" y="94">down_proj</text>
                  <text x="422" y="125" fill="#657184">回到 [B,S,d]</text>
                </g>
              </svg>
              <p>这就是 notebook 里 forward 的核心公式：<code>down(SiLU(gate(x)) * up(x))</code>。你写代码时只要保证 gate/up 输出维度一致，down 的输入就是 intermediate_size。</p>
              <table class="freq-table">
                <tr><th style="width:140px">层</th><th>维度方向</th><th>它负责什么</th></tr>
                <tr><td><code>gate_proj</code></td><td><code>hidden_size → intermediate_size</code></td><td style="text-align:left">产生门控信号，后面先过 SiLU。</td></tr>
                <tr><td><code>up_proj</code></td><td><code>hidden_size → intermediate_size</code></td><td style="text-align:left">产生被门控的内容，shape 要和 gate 对齐。</td></tr>
                <tr><td><code>down_proj</code></td><td><code>intermediate_size → hidden_size</code></td><td style="text-align:left">把中间表示收回 hidden_size，方便残差相加。</td></tr>
              </table>
              <details class="think">
                <summary>想一想：为什么 gate 和 up 都从同一个 x 出发？</summary>
                <div class="think-body">
                  <p>因为它们要对同一份 token 表示做两种解释：一条问“哪些中间特征该放行”，另一条问“具体内容是什么”。如果两条分支看的不是同一个 x，门控就很难和内容一一对齐。</p>
                  <p>这也解释了为什么二者输出 shape 必须一样：逐元素相乘时，第 i 个门只控制第 i 个内容特征。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">为什么 intermediate_size 常接近 8/3 × hidden_size？</div>
            <div class="story-panel">
              <strong>3. 8/3 不是玄学，是为了让 SwiGLU 的开销接近原始 4d MLP</strong>
              <p>普通 MLP 两个大矩阵：<code>d → 4d</code> 和 <code>4d → d</code>，参数量大约是 <code>8d²</code>。</p>
              <p>SwiGLU 有三个矩阵：gate、up、down。若中间维度是 m，参数量大约是 <code>d·m + d·m + m·d = 3dm</code>。为了和普通 MLP 的 <code>8d²</code> 接近，令 <code>3dm ≈ 8d²</code>，得到 <code>m ≈ 8/3·d</code>。</p>
              <div class="matrix-flow"><span>普通 MLP<br>2 个矩阵<br>≈ 8d²</span><span>SwiGLU<br>3 个矩阵<br>≈ 3dm</span><strong>令 3dm≈8d²<br>m≈8/3d</strong></div>
              <details class="think">
                <summary>为什么要让计算开销接近原始模型，而不是随便变大？</summary>
                <div class="think-body">
                  <p>因为这样才能公平比较“结构改进”本身。如果参数量和 FLOPs 暴涨，效果变好可能只是因为模型更大，不一定是 SwiGLU 更好。</p>
                  <p>工程上也要控制训练和推理预算。保持相近开销，意味着可以把普通 MLP 换成 SwiGLU，同时显存、速度和部署成本不会突然失控。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">回到本 notebook：测试给你的是一个已经对齐过的 intermediate_size</div>
            <div class="story-panel">
              <strong>4. 读懂测试里的 1376：它不是 hidden_size，也不是 4d</strong>
              <p>测试中 <code>hidden_size=512</code>、<code>intermediate_size=1376</code>。按 8/3 粗算是 1365.3，工程里会向硬件友好的倍数对齐，所以变成 1376。你在 TODO 里不用重新计算它，只要按传进来的 <code>intermediate_size</code> 搭三块 Linear。</p>
              <div class="ratio-board"><span>hidden_size = 512</span><span>理论 8/3d ≈ 1365</span><strong>notebook 传入 1376</strong></div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：MLP 往往是 decoder block 的算力大户</div>
              <p>一个 LLaMA block 里 Attention 很显眼，但 MLP 也非常重：每层都要跑 gate、up、down 三个大矩阵。工业实现常把 gate/up 融合成一次更大的矩阵乘法来减少启动开销，但数学结构仍然是你现在写的这条链。</p>
              <p>所以这关不是“写三行 Linear”这么简单；它是在训练你把论文结构、张量形状和工程实现连成一张图。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：写一个“门控内容”的小模块</h4><p>下面用不同命名练结构，不直接复制 notebook 的类名和字段名。</p><pre><code>class TinyGatedBlock(nn.Module):
    def __init__(self, in_dim, mid_dim):
        super().__init__()
        self.switch = nn.Linear(in_dim, mid_dim, bias=False)
        self.value = nn.Linear(in_dim, mid_dim, bias=False)
        self.out = nn.Linear(mid_dim, in_dim, bias=False)

    def forward(self, x):
        opened = F.silu(self.switch(x))      # [B, S, mid_dim]
        content = self.value(x)             # [B, S, mid_dim]
        mixed = opened * content            # 逐元素乘法，shape 不变
        return self.out(mixed)              # [B, S, in_dim]</code></pre><p class="syntax-tip">读法：子层在 <code>__init__</code> 里注册，计算在 <code>forward</code> 里串起来；<code>*</code> 是逐元素乘法，不是矩阵乘法，所以两边 shape 必须完全一致或可广播。</p></div>
        <div class="syntax-card">
          <h4>举一反三：把玩具写法搬回 notebook</h4>
          <p>你不需要背整行答案，只要把小例子的名字和维度换成 notebook 的变量。</p>
          <table class="freq-table">
            <tr><th style="width:150px">玩具例子</th><th>Notebook 里换成</th><th>为什么这样换</th></tr>
            <tr><td><code>switch</code></td><td><code>gate_proj</code></td><td style="text-align:left">都是“开关/门控”分支。</td></tr>
            <tr><td><code>value</code></td><td><code>up_proj</code></td><td style="text-align:left">都是内容分支，等待被 gate 逐元素筛选。</td></tr>
            <tr><td><code>out</code></td><td><code>down_proj</code></td><td style="text-align:left">都是把中间维度投回输入维度。</td></tr>
            <tr><td><code>in_dim</code></td><td><code>hidden_size</code></td><td style="text-align:left">输入 token 表示的最后一维。</td></tr>
            <tr><td><code>mid_dim</code></td><td><code>intermediate_size</code></td><td style="text-align:left">gate/up 两条分支共同使用的中间维度。</td></tr>
          </table>
          <p class="syntax-tip">迁移口诀：两个升维层都写 <code>nn.Linear(hidden_size, intermediate_size, bias=False)</code>；降维层反过来写 <code>nn.Linear(intermediate_size, hidden_size, bias=False)</code>。forward 里先分别调用层，再把结果组合起来。</p>
        </div>`,
      predict: {
        hook: "SwiGLU 比普通 MLP 多了一条 gate 分支。如果中间维度还写成 4d，参数和计算会明显变大。",
        question: "先判断：为什么 LLaMA 常把 SwiGLU 的 intermediate_size 设到接近 8/3 × hidden_size？",
        options: [
          "为了让三个投影的总开销接近普通 4d MLP，方便公平替换和控制预算",
          "因为 8/3 可以让张量自动变成整数，不需要取整",
          "因为 SiLU 只能处理 8/3 倍宽度的向量"
        ],
        answer: 0,
        revealNote: "关键是参数量近似相等：普通 MLP ≈ 8d²，SwiGLU ≈ 3dm，所以 m≈8/3d。实际模型还会按硬件友好的倍数取整。"
      },
      checkpoint: {
        question: "x 的 shape 是 [2, 16, 512]，intermediate_size=1376。经过 gate_proj 和 up_proj 后，哪一步能合法相乘？",
        options: ["两个输出都是 [2,16,1376]，先对 gate 做 SiLU，再与 up 输出逐元素相乘", "gate 输出 [2,16,512]，up 输出 [2,16,1376]，直接相乘", "down_proj 输出 [2,16,512] 后再和 up_proj 输出相乘"],
        answer: 0,
        explain: "gate/up 都从 hidden_size 投影到同一个 intermediate_size，shape 对齐后才能逐元素相乘；down_proj 是乘完后再把维度收回 hidden_size。"
      },
      homework: [
        "TODO 1：在 __init__ 定义 gate_proj、up_proj、down_proj，三者都用 bias=False；gate/up 是 hidden_size→intermediate_size，down 是 intermediate_size→hidden_size。",
        "TODO 2：forward 里先分别算 gate 分支和 up 分支，再用逐元素乘法 *，最后交给 down_proj；这里不要写成矩阵乘法 @。",
        "测试时确认 LlamaMLP 的输出 shape 回到 [batch_size, seq_len, hidden_size]，并且 gate/up/down 三组参数都连在计算图上。"
      ]
    },
    {
      id: "llama-prenorm-residual",
      title: "Pre-Norm 残差：每个子层只写一份 update",
      todo: "TODO 3",
      prerequisite: [
        "残差连接是 output = input + update：主路 input 保留，子层只负责提供一份增量。",
        "Pre-Norm 表示先归一化再进子层：norm → attention/mlp → add residual。",
        "LLaMA Decoder Layer 有两个连续子层：Attention 子层先更新一次，MLP 子层再更新一次。两次都要各自保存 residual。",
        "本 notebook 的 DummyAttention 直接返回一个张量；真实模型可能还会返回 cache 或 attention weights，初学时先把主数据流写顺。"
      ],
      intuition: "把 hidden_states 想成一条主干信息流。Attention 和 MLP 不直接替换主干，而是在“归一化后的输入”上算一份 update，再加回主干。这样深层网络反传时始终有一条清晰的梯度通路。写作业时只记一个口令：每个子层开始先拍一张 residual 快照，子层结束把 update 加回这张快照。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. TODO 3 是“同一个模板连续套两次”</strong>
              <p>不要把整个 Decoder Layer 当成一团复杂网络。它只是先做一次 Attention 版的 <code>prenorm_step</code>，再做一次 MLP 版的 <code>prenorm_step</code>。两次输入输出 shape 都是 [B, S, hidden]，所以它们可以像积木一样接起来。</p>
              <table class="freq-table">
                <tr><th style="width:120px">阶段</th><th>输入</th><th>要调用的模块</th><th>输出</th></tr>
                <tr><td><strong>第 1 段</strong></td><td><code>hidden0</code></td><td><code>input_layernorm → self_attn → residual add</code></td><td><code>hidden1</code></td></tr>
                <tr><td><strong>第 2 段</strong></td><td><code>hidden1</code></td><td><code>post_attention_layernorm → mlp → residual add</code></td><td><code>hidden2</code></td></tr>
              </table>
            </div>
            <div class="story-arrow">先分清两种 Transformer 写法：LLaMA 用 Pre-Norm</div>
            <div class="story-panel">
              <strong>1. Post-Norm 和 Pre-Norm 的区别先别混</strong>
              <div class="mini-flow"><span>Post-Norm<br>sublayer → add → norm</span><strong>Pre-Norm<br>norm → sublayer → add</strong></div>
              <p>LLaMA 用的是 Pre-Norm。写 TODO 时不要把 layernorm 放到残差相加之后。</p>
              <details class="think">
                <summary>为什么 Pre-Norm 对深层网络更友好？</summary>
                <div class="think-body">
                  <p>残差主路几乎是一条直通路：梯度可以沿着加法直接传回前面层，不必每次都先穿过归一化和复杂子层。</p>
                  <p>这对几十层甚至上百层的 decoder 很关键。初学者写代码时不用证明它，只要把顺序放对：先 norm，再子层，最后 add。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">第一段：Attention update</div>
            <div class="story-panel">
              <strong>2. 保存旧 hidden_states，再让归一化后的版本去跑 attention</strong>
              <svg viewBox="0 0 540 150" width="100%" style="max-width:700px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <defs>
                  <marker id="arrow05b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L7,3 z" fill="#657184"></path>
                  </marker>
                </defs>
                <g font-size="13" text-anchor="middle">
                  <rect x="20" y="56" width="86" height="34" rx="8" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="63" y="78">hidden</text>
                  <path d="M106 73 C150 20, 275 20, 345 64" fill="none" stroke="#2f7fc4" stroke-width="3" marker-end="url(#arrow05b)"></path>
                  <text x="210" y="24" fill="#2f7fc4">residual 主路</text>
                  <line x1="106" y1="73" x2="160" y2="73" stroke="#657184" stroke-width="2" marker-end="url(#arrow05b)"></line>
                  <rect x="160" y="56" width="92" height="34" rx="8" fill="#fff6e8" stroke="#e8b66f"></rect><text x="206" y="78">input_norm</text>
                  <line x1="252" y1="73" x2="305" y2="73" stroke="#657184" stroke-width="2" marker-end="url(#arrow05b)"></line>
                  <rect x="305" y="56" width="82" height="34" rx="8" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="346" y="78">self_attn</text>
                  <line x1="387" y1="73" x2="430" y2="73" stroke="#657184" stroke-width="2" marker-end="url(#arrow05b)"></line>
                  <circle cx="452" cy="73" r="18" fill="#f4f1ff" stroke="#c9bdf0"></circle><text x="452" y="78" font-size="20">+</text>
                  <line x1="470" y1="73" x2="510" y2="73" stroke="#657184" stroke-width="2" marker-end="url(#arrow05b)"></line>
                  <text x="511" y="101" fill="#657184">hidden'</text>
                </g>
              </svg>
              <p>第一段的代码顺序是：保存 residual → input_layernorm → self_attn → residual + update。</p>
            </div>
            <div class="story-arrow">第二段：MLP update，结构完全一样，只是换子层</div>
            <div class="story-panel">
              <strong>3. 注意 residual 要重新保存为 Attention 后的新 hidden_states</strong>
              <p>MLP 子层不是加回最开始的 x，而是加回 Attention 更新之后的 hidden_states。所以第二段开始要再写一次 <code>residual = hidden_states</code>。</p>
              <div class="residual-demo"><span>hidden' ─ residual ─────────────┐</span><span>post_attention_norm → mlp ─────┤ +</span><strong>hidden'' 继续传给下一层</strong></div>
              <table class="freq-table">
                <tr><th style="width:120px">常见错法</th><th>为什么不对</th></tr>
                <tr><td>第二段还加最初的 x</td><td style="text-align:left">MLP 应该接着 Attention 后的新状态更新，否则会跳过第一段更新。</td></tr>
                <tr><td>把 norm 后的张量当 residual</td><td style="text-align:left">残差主路应该保存子层入口的原始状态，不是归一化后的临时张量。</td></tr>
                <tr><td>忘记调用 mlp</td><td style="text-align:left">MLP 参数没有参与输出，测试里的 grad 检查会失败。</td></tr>
              </table>
              <details class="think">
                <summary>为什么测试会检查“所有参数都有梯度”？</summary>
                <div class="think-body">
                  <p>如果你忘了调用 self_attn 或 mlp，或者把某段 update 丢掉，对应参数就不参与输出计算，反向传播时 grad 会是 None。</p>
                  <p>所以这个测试不是只看 shape，而是在检查两段子层都真的接进了计算图。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">用测试倒推你该检查什么</div>
            <div class="story-panel">
              <strong>4. 测试失败时，按这三步定位</strong>
              <p>shape 错，先看 down_proj 是否回到 hidden_size；grad 缺失，先看对应模块有没有真的被调用；数值能跑但结构怪，先看 layernorm 是不是放到了 residual add 之前。</p>
              <table class="freq-table">
                <tr><th style="width:170px">测试现象</th><th>优先检查</th></tr>
                <tr><td><code>out.shape</code> 报错</td><td style="text-align:left">每一段输出是否仍是 [B, S, H]，尤其是 <code>down_proj</code> 是否收回 hidden_size。</td></tr>
                <tr><td>某参数 <code>grad is None</code></td><td style="text-align:left">对应子层是否真的被调用，并且结果是否参与了最终输出。</td></tr>
                <tr><td>顺序越写越乱</td><td style="text-align:left">回到固定骨架：保存 residual → norm → block → add。</td></tr>
              </table>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：Decoder Layer 是大模型的最小可复用单元</div>
              <p>真实 LLaMA 会把这个 block 堆很多层：8B 模型有几十层，70B 模型更多。每一层都遵循同一套“Pre-Norm + Attention + Residual + MLP + Residual”的节奏。</p>
              <p>一旦你能手写这个玩具版，就已经掌握了读大模型源码时最重要的骨架；后面换成 RoPE、GQA、KV Cache 或张量并行，本质都是往这个骨架里替换更强的组件。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>语法热身：通用 pre-norm 残差模板</h4><p>先用玩具子层练顺序。重点是 residual 在 norm 前保存，add 在子层后发生。</p><pre><code>def prenorm_step(x, norm, block):
    residual = x
    x = norm(x)
    update = block(x)
    x = residual + update
    return x

# 一个 Decoder Layer 通常连续做两次：
x = prenorm_step(x, norm_a, attention_like_block)
x = prenorm_step(x, norm_b, mlp_like_block)</code></pre><p class="syntax-tip">读法：不要写成 <code>x = norm(residual + block(x))</code>，那是 Post-Norm 的味道；LLaMA 这里要先 norm，再进子层，最后加 residual。</p></div>
        <div class="syntax-card">
          <h4>举一反三：把模板内联到 TODO 3</h4>
          <p>notebook 没有要求你真的定义 <code>prenorm_step</code> 函数。你可以把模板拆开写两遍：第一遍把 <code>norm</code> 换成 <code>input_layernorm</code>、<code>block</code> 换成 <code>self_attn</code>；第二遍换成 <code>post_attention_layernorm</code> 和 <code>mlp</code>。</p>
          <table class="freq-table">
            <tr><th style="width:120px">模板位置</th><th>第 1 遍换成</th><th>第 2 遍换成</th></tr>
            <tr><td><code>norm</code></td><td><code>input_layernorm</code></td><td><code>post_attention_layernorm</code></td></tr>
            <tr><td><code>block</code></td><td><code>self_attn</code></td><td><code>mlp</code></td></tr>
            <tr><td>开始动作</td><td colspan="2" style="text-align:left">每一遍都先保存当前 <code>hidden_states</code>，再进入 norm。</td></tr>
          </table>
          <p class="syntax-tip">迁移口诀：每个子层四行骨架都是“保存 residual、归一化、调用子层、加回 residual”。第二遍开始时的 residual 一定是 Attention 后的新 hidden_states。</p>
        </div>`,
      predict: {
        hook: "TODO 3 最容易写错的点不是加号，而是 residual 保存的时机。尤其第二段 MLP，residual 应该是谁？",
        question: "先判断：MLP block 开始前，residual 应该保存哪个张量？",
        options: [
          "最原始输入 x，因为整层都应该加回同一个起点",
          "Attention block 完成后的 hidden_states，因为 MLP 是在这个新状态上继续更新",
          "post_attention_layernorm 的输出，因为它已经归一化"
        ],
        answer: 1,
        revealNote: "第二段 residual 要重新保存 Attention 后的 hidden_states。每个子层都有自己的 residual 主路。"
      },
      checkpoint: {
        question: "LLaMA Decoder Layer 的正确顺序是哪一个？",
        options: ["residual=x → norm → attention → add；再 residual=当前hidden → norm → mlp → add", "norm → attention → norm → mlp → 最后只加一次最初的 residual", "attention → residual add → mlp → layernorm 放在最后"],
        answer: 0,
        explain: "Pre-Norm 残差是每个子层一段：先保存当前 hidden_states，再 norm、子层、加回 residual。Attention 和 MLP 各做一次。"
      },
      homework: [
        "TODO 3 第一段：写 Attention block，顺序是保存当前 hidden_states → input_layernorm → self_attn → 加回刚才保存的 residual。",
        "TODO 3 第二段：不要复用最初的 residual；先把 Attention 后的 hidden_states 重新保存，再写 post_attention_layernorm → mlp → residual add。",
        "跑 notebook 测试：输出 shape 必须等于输入 shape；若某个参数 grad 为 None，就回头检查对应子层有没有参与最终输出。"
      ]
    }
  ],
  ...require("./lesson_overrides_extra")
};
