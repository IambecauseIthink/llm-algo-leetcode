const fs = require("fs");
const path = require("path");
const lessonOverrides = require("./lesson_overrides");
const curriculumV2 = require("./curriculum_v2");
const enhanceFoundationLessons = require("./foundation_enhancements");

const root = __dirname;
const notesDir = path.join(root, "notes");
fs.mkdirSync(notesDir, { recursive: true });

const levels = [
  {
    id: "00",
    title: "PyTorch Warmup",
    file: "00_PyTorch_Warmup.ipynb",
    category: "foundation",
    difficulty: "Easy",
    tags: ["Tensor", "Embedding", "Autograd"],
    summary: "把 Tensor 变形、Embedding 查表、Linear+ReLU 手写反传这三块基础打稳。",
    concepts: [
      "Tensor shape 是 PyTorch 算法题的第一语言：先确认每个维度的语义，再决定 permute、reshape 或 rearrange。",
      "Embedding 本质是查表：input_ids 只是行索引，权重矩阵的行就是 token 向量。",
      "手写 backward 时，先穿过激活函数，再穿过线性层；ReLU 需要保存 z > 0 的 mask。"
    ],
    quiz: {
      question: "把图像特征 [B, C, H, W] 转成 Transformer 序列 [B, H*W, C]，哪种 einops 写法正确？",
      options: [
        "rearrange(x, 'b c h w -> b (h w) c')",
        "rearrange(x, 'b c h w -> b c (h w)')",
        "rearrange(x, 'b c h w -> c b (h w)')"
      ],
      answer: 0,
      explain: "空间维度 H 和 W 被合并为序列长度，通道 C 留作每个 token 的特征维。"
    },
    handsOn: [
      "在 notebook 里补全 tensor_warmup，并用 torch.allclose 验证原生写法和 einops 写法一致。",
      "把 Embedding 的官方输出和 emb_layer.weight[input_ids] 做一次数值比较。",
      "手写 LinearReLUFunction.backward，确认 x、weight、bias 的梯度都和 PyTorch 官方实现一致。"
    ],
    formula: "先学会看懂 shape、查表、梯度流，再回 notebook 写 PyTorch 代码",
    lessons: [
      {
        id: "tensor",
        title: "Tensor 变形：先学会读 shape，再写 reshape",
        todo: "TODO 1.1 / 1.2",
        prerequisite: [
          "Tensor 的 shape 不是一串数字，而是每个轴的语义。",
          "图像特征常见布局是 [B, C, H, W]：B 是样本数，C 是通道，H/W 是空间网格。",
          "Transformer 更喜欢序列布局：[B, S, C]，其中 S 可以是 H * W。"
        ],
        intuition: "把一张 H×W 的小棋盘摊平成一条路线：每个格子变成一个 token，格子里带着 C 个通道特征。",
        exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>1. 原始小图：[B=1, C=2, H=2, W=3]</strong>
              <p>同一张图有两个通道，像两张叠在一起的 2×3 透明胶片。</p>
              <div class="channel-stack">
                <div class="channel-plane warm">
                  <span class="plane-title">Channel 0</span>
                  <span>a</span><span>b</span><span>c</span>
                  <span>d</span><span>e</span><span>f</span>
                </div>
                <div class="channel-plane cool">
                  <span class="plane-title">Channel 1</span>
                  <span>A</span><span>B</span><span>C</span>
                  <span>D</span><span>E</span><span>F</span>
                </div>
              </div>
            </div>
            <div class="story-arrow">换轴：把 C 放到每个格子里</div>
            <div class="story-panel">
              <strong>2. 通道放后：[B=1, H=2, W=3, C=2]</strong>
              <p>现在每个空间位置都带着一个 2 维小向量。</p>
              <div class="pixel-grid">
                <span>[a,A]</span><span>[b,B]</span><span>[c,C]</span>
                <span>[d,D]</span><span>[e,E]</span><span>[f,F]</span>
              </div>
            </div>
            <div class="story-arrow">摊平 H×W：按行走成序列</div>
            <div class="story-panel">
              <strong>3. Token 序列：[B=1, S=6, C=2]</strong>
              <p>Transformer 看到的是 6 个 token，每个 token 有 2 个特征。</p>
              <div class="token-rail">
                <span><b>t0</b>[a,A]</span>
                <span><b>t1</b>[b,B]</span>
                <span><b>t2</b>[c,C]</span>
                <span><b>t3</b>[d,D]</span>
                <span><b>t4</b>[e,E]</span>
                <span><b>t5</b>[f,F]</span>
              </div>
            </div>
          </div>
          <p>你不需要先背 <code>permute(0, 2, 3, 1)</code>。先问自己：我要把 C 从第 2 个轴挪到最后，再把 H 和 W 合成序列长度。</p>`,
        syntaxHtml: `
          <div class="syntax-card">
            <h4>语法热身：换轴和合并维度怎么写</h4>
            <p>下面用一个“时间序列”小例子练语法，不使用 notebook 里的图像 shape。</p>
            <pre><code># x: [batch, time, feature]
x = torch.randn(2, 4, 3)

# 把 feature 放到 time 前面: [batch, feature, time]
y = x.permute(0, 2, 1)

# 把 batch 和 time 合并: [batch * time, feature]
z = x.reshape(2 * 4, 3)

# einops 写法：名字就是维度含义
z2 = rearrange(x, "b t f -> (b t) f")</code></pre>
            <p class="syntax-tip">读法：<code>permute</code> 是“重新排列轴”，<code>reshape</code> 是“重新分组元素”，<code>einops</code> 是“把变形规则写成句子”。</p>
          </div>`,
        checkpoint: {
          question: "如果 x 的 shape 是 [2, 3, 4, 5]，转成图像 token 序列后 shape 应该是多少？",
          options: ["[2, 20, 3]", "[2, 3, 20]", "[20, 2, 3]"],
          answer: 0,
          explain: "B 保持 2，H*W = 4*5 = 20，C 保持 3，所以是 [2, 20, 3]。"
        },
        homework: [
          "回到 notebook，实现原生 PyTorch 写法：先 permute，再 reshape。",
          "再用 einops 写出同一个变换，让字符串表达出 b c h w -> b (h w) c。",
          "跑测试时重点看 shape 和 torch.allclose，两者同时通过才算真的理解。"
        ]
      },
      {
        id: "embedding",
        title: "Embedding 查表：token id 是地址，不是连续数值",
        todo: "TODO 2.1 / 2.2",
        prerequisite: [
          "文本会先被 tokenizer 变成 token id，例如 [4, 1, 3]。",
          "神经网络不能直接理解 id 的大小关系；id 只是查表地址。",
          "Embedding 权重表的 shape 是 [vocab_size, hidden_dim]，第 i 行就是 token i 的向量。"
        ],
        intuition: "像在词典里按页码找词条：id 是页码，Embedding 表是词典，查出来的一整行向量才是模型真正使用的表示。",
        exampleHtml: `
          <div class="lookup-board">
            <div class="ids"><strong>input_ids</strong><span class="id-token">2</span><span class="id-token">0</span><span class="id-token">3</span></div>
            <div class="table">
              <span class="hit">ID 0</span><span class="hit">[0.1, 0.4]</span>
              <span>ID 1</span><span>[0.8, 0.2]</span>
              <span class="hit">ID 2</span><span class="hit">[0.5, 0.9]</span>
              <span class="hit">ID 3</span><span class="hit">[0.3, 0.7]</span>
            </div>
            <div class="ids"><strong>输出向量序列</strong><span>[0.5, 0.9]</span><span>[0.1, 0.4]</span><span>[0.3, 0.7]</span></div>
          </div>
          <p><code>nn.Embedding(input_ids)</code> 和 <code>weight[input_ids]</code> 做的是同一个核心动作：按 id 取行。</p>`,
        syntaxHtml: `
          <div class="syntax-card">
            <h4>语法热身：Embedding 和高级索引</h4>
            <p>这里用“颜色编号查颜色向量”的例子，不使用 notebook 的 vocab/hidden 设置。</p>
            <pre><code># 4 种颜色，每种颜色用 3 个数字表示
palette = torch.tensor([
    [1.0, 0.0, 0.0],  # red
    [0.0, 1.0, 0.0],  # green
    [0.0, 0.0, 1.0],  # blue
    [1.0, 1.0, 0.0],  # yellow
])

color_ids = torch.tensor([[2, 0], [3, 1]])

# 高级索引：每个 id 替换成对应行
colors = palette[color_ids]  # shape: [2, 2, 3]

# nn.Embedding 的调用方式也一样：传入 id，返回向量
emb = nn.Embedding(num_embeddings=4, embedding_dim=3)
vectors = emb(color_ids)</code></pre>
            <p class="syntax-tip">读法：<code>weight[input_ids]</code> 会把 input_ids 的每个整数当成“行号”，输出 shape 会在 input_ids 后面追加 embedding_dim。</p>
          </div>`,
        checkpoint: {
          question: "给定 weight 有 10 行、hidden_dim=4，input_ids 的 shape 是 [2, 3]，Embedding 输出 shape 是什么？",
          options: ["[2, 3, 4]", "[10, 2, 3]", "[2, 4]"],
          answer: 0,
          explain: "每个 id 会变成一个 hidden_dim=4 的向量，所以在 [2,3] 后面追加 4。"
        },
        homework: [
          "回到 notebook，先实例化 nn.Embedding 并得到官方输出。",
          "再用 emb_layer.weight[input_ids] 手动查表。",
          "用 torch.allclose 验证官方查表和手动查表完全一致。"
        ]
      },
      {
        id: "backward",
        title: "Linear + ReLU 反传：梯度像闯关一样倒着走",
        todo: "TODO 3.1 / 3.2 / 3.3",
        prerequisite: [
          "forward 不是只算输出，还要保存 backward 会用到的中间信息。",
          "ReLU 的导数很简单：z > 0 时梯度通过，z <= 0 时梯度变 0。",
          "Linear 的三个梯度要和三个输入形状对齐：x、weight、bias。"
        ],
        intuition: "把前向看成三扇门：Linear 门、ReLU 门、Loss 门。反向时梯度从 Loss 倒着回来，ReLU 门先决定哪些位置能通行。",
        exampleHtml: `
          <div class="grad-lesson">
            <span>z<br><strong>[-1, 0, 2]</strong></span>
            <span class="shape-op">生成 mask</span>
            <span>z &gt; 0<br><strong>[0, 0, 1]</strong></span>
            <span class="shape-op">乘上传回梯度</span>
            <span>grad_z<br><strong>[0, 0, 5]</strong></span>
          </div>
          <div class="grad-lesson compact">
            <span>grad_z</span>
            <span class="shape-op">→</span>
            <span>grad_x = grad_z @ weight</span>
            <span class="shape-op">→</span>
            <span>grad_weight = grad_z.T @ x</span>
          </div>
          <p>如果前向里某个 <code>z</code> 是负数，ReLU 把它压成 0；反向时这条路也会被 mask 关掉。</p>`,
        syntaxHtml: `
          <div class="syntax-card">
            <h4>语法热身：自定义 autograd 函数长什么样</h4>
            <p>这里用“平方函数”的 backward 练结构，不直接写 notebook 的 Linear+ReLU。</p>
            <pre><code>class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * 2 * x
        return grad_x

# 调用自定义 Function 要用 .apply
y = SquareFunction.apply(x)</code></pre>
            <p class="syntax-tip">读法：<code>ctx.save_for_backward</code> 像是在前向路上放书签；<code>backward</code> 里先取回书签，再按链式法则乘上传回来的 <code>grad_output</code>。</p>
          </div>`,
        checkpoint: {
          question: "已知 grad_output = [5, 5, 5]，z = [-1, 0, 2]，经过 ReLU 反向后 grad_z 是什么？",
          options: ["[0, 0, 5]", "[5, 5, 5]", "[-5, 0, 5]"],
          answer: 0,
          explain: "ReLU 只有 z > 0 的位置导数为 1；-1 和 0 都不传梯度。"
        },
        homework: [
          "回到 notebook，在 forward 里计算 z、y，并保存 x、weight、mask。",
          "在 backward 里先写 grad_z = grad_output * mask。",
          "再补 Linear 的三个梯度：grad_x、grad_weight、grad_bias，并和官方 autograd 对齐。"
        ]
      }
    ]
  },
  {
    id: "01",
    title: "RMSNorm Tutorial",
    file: "01_RMSNorm_Tutorial.ipynb",
    category: "architecture",
    difficulty: "Easy",
    tags: ["Norm", "LLM", "Scale"],
    summary: "理解 RMSNorm 为什么省掉均值中心化，也能稳定 Transformer hidden states。",
    concepts: [
      "LayerNorm 会减均值再除标准差；RMSNorm 只用 root mean square 做缩放，计算更轻。",
      "RMSNorm 常用于 LLaMA 系模型，目标是稳定激活尺度，而不是强制每个 token 的 hidden state 零均值。",
      "可学习参数 gamma 负责把归一化后的向量重新缩放到模型需要的尺度。"
    ],
    quiz: {
      question: "RMSNorm 相比 LayerNorm 最核心的简化是什么？",
      options: ["不再减去均值，只按均方根缩放", "完全没有可学习参数", "只适用于 batch size 为 1"],
      answer: 0,
      explain: "RMSNorm 保留按尺度归一化的效果，但省掉 mean-centering。"
    },
    handsOn: [
      "实现 rms = sqrt(mean(x^2) + eps)，并检查输出 shape 是否与输入一致。",
      "把你的 RMSNorm 输出和 notebook 中参考实现对齐。",
      "调大输入幅度，观察归一化前后向量范数的变化。"
    ],
    formula: "RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma"
  },
  {
    id: "02",
    title: "SwiGLU Activation",
    file: "02_SwiGLU_Activation.ipynb",
    category: "architecture",
    difficulty: "Easy",
    tags: ["FFN", "Activation", "Gate"],
    summary: "把 FFN 中的门控激活拆开看：一支给内容，一支决定通过多少。",
    concepts: [
      "SwiGLU 把前馈层拆成 value 分支和 gate 分支，gate 经过 SiLU 后逐元素调制 value。",
      "相比普通 ReLU/GELU FFN，SwiGLU 通常能用更强的门控机制提升表达力。",
      "实现时要特别关注 hidden_dim、intermediate_dim 以及元素乘法的 shape 对齐。"
    ],
    quiz: {
      question: "SwiGLU 中 gate 分支通常会经过哪个激活函数？",
      options: ["SiLU / Swish", "Softmax", "Sigmoid 后再归一化到和为 1"],
      answer: 0,
      explain: "SwiGLU 的名字来自 Swish/SiLU gate 与 GLU 结构的组合。"
    },
    handsOn: [
      "实现 value = Wv x 和 gate = SiLU(Wg x)，再计算 value * gate。",
      "打印每一步 shape，确认逐元素乘法前两个张量形状一致。",
      "和普通 GELU FFN 做参数量或输出分布对比。"
    ],
    formula: "SwiGLU(x) = (x Wv) * SiLU(x Wg)"
  },
  {
    id: "03",
    title: "RoPE Tutorial",
    file: "03_RoPE_Tutorial.ipynb",
    category: "architecture",
    difficulty: "Medium",
    tags: ["Position", "Attention", "Rotation"],
    summary: "用二维旋转理解 RoPE 如何把位置信息注入 Q/K，并天然表达相对位置。",
    concepts: [
      "RoPE 不是把 position embedding 加到 hidden state，而是旋转 Q 和 K 的成对维度。",
      "旋转角度随 token 位置变化，因此 Q/K 点积会携带相对位置信息。",
      "实现时通常把 hidden dim 两两配对，使用 cos/sin cache 避免重复计算。"
    ],
    quiz: {
      question: "RoPE 主要作用在哪两个张量上？",
      options: ["Query 和 Key", "Value 和 Output", "Embedding table 和 LM head"],
      answer: 0,
      explain: "RoPE 通过旋转 Q/K 改变注意力分数中的位置关系。"
    },
    handsOn: [
      "实现 rotate_half，并验证两两维度旋转后的 shape 不变。",
      "构造两个位置的 cos/sin，观察同一个向量在不同位置的旋转结果。",
      "把 RoPE 应用到 Q/K 后，比较 attention score 的变化。"
    ],
    formula: "q_rot = q * cos(pos) + rotate_half(q) * sin(pos)"
  },
  {
    id: "04",
    title: "Attention MHA GQA",
    file: "04_Attention_MHA_GQA.ipynb",
    category: "architecture",
    difficulty: "Medium",
    tags: ["Attention", "MHA", "GQA"],
    summary: "从 shape 和 KV cache 成本理解 MHA、MQA、GQA 的差异。",
    concepts: [
      "MHA 中 Q/K/V 都有多个 head；GQA 让多个 query heads 共享较少的 key/value heads。",
      "GQA 的重要收益是减少 KV cache 体积，推理时尤其关键。",
      "实现 GQA 时需要把 K/V heads repeat 或 broadcast 到 query heads 对应的分组。"
    ],
    quiz: {
      question: "GQA 相比标准 MHA 最直接减少的是哪部分推理存储？",
      options: ["KV cache", "token embedding table", "optimizer states"],
      answer: 0,
      explain: "GQA 使用更少的 K/V heads，因此每层每 token 缓存的 K/V 更少。"
    },
    handsOn: [
      "打印 Q、K、V 的 shape，区分 num_heads 和 num_kv_heads。",
      "实现 repeat_kv，把 K/V 对齐到 query heads 数量。",
      "比较 MHA 与 GQA 的 KV cache 元素数量。"
    ],
    formula: "Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V"
  },
  {
    id: "05",
    title: "LLaMA3 Block Tutorial",
    file: "05_LLaMA3_Block_Tutorial.ipynb",
    category: "architecture",
    difficulty: "Medium",
    tags: ["Decoder Block", "Residual", "RMSNorm"],
    summary: "把 RMSNorm、Attention、SwiGLU FFN 和残差连接组装成现代 decoder block。",
    concepts: [
      "LLaMA 风格 block 通常使用 pre-norm：先归一化，再进入 attention 或 FFN。",
      "残差连接让每个子层学习增量，帮助深层网络稳定训练。",
      "一个 block 的主要路径是 norm -> attention -> residual -> norm -> mlp -> residual。"
    ],
    quiz: {
      question: "Pre-norm Transformer block 中，归一化通常发生在子层的什么位置？",
      options: ["子层之前", "子层之后且没有残差", "只在最后一层"],
      answer: 0,
      explain: "Pre-norm 先对输入归一化，再送入 attention 或 MLP。"
    },
    handsOn: [
      "画出 block 的数据流，标注两条 residual add 的位置。",
      "在 notebook 中打印每个子层输入输出 shape。",
      "替换一个组件，例如把 FFN 输出置零，观察残差路径如何保留输入。"
    ],
    formula: "x = x + Attention(RMSNorm(x)); x = x + MLP(RMSNorm(x))"
  },
  {
    id: "06",
    title: "MoE Router",
    file: "06_MoE_Router.ipynb",
    category: "architecture",
    difficulty: "Medium",
    tags: ["MoE", "Router", "Top-k"],
    summary: "理解专家模型如何为每个 token 选择少数专家，而不是激活整个 FFN。",
    concepts: [
      "MoE 的核心是稀疏激活：每个 token 只走 top-k 个专家。",
      "Router 输出每个 token 对专家的打分，softmax 后得到路由概率。",
      "实现时要处理 top-k 选择、专家输出加权合并以及容量限制。"
    ],
    quiz: {
      question: "MoE 中 top-k router 的 k 表示什么？",
      options: ["每个 token 选择的专家数量", "模型层数", "每个专家的 hidden dim"],
      answer: 0,
      explain: "top-k 决定一个 token 会被派发给几个专家。"
    },
    handsOn: [
      "给一批 token 构造 router logits，并用 torch.topk 取专家 id。",
      "统计每个专家收到多少 token。",
      "把专家输出按 router weight 加权求和。"
    ],
    formula: "expert_ids, weights = topk(softmax(router(x)), k)"
  },
  {
    id: "07",
    title: "MoE Load Balancing Loss",
    file: "07_MoE_Load_Balancing_Loss.ipynb",
    category: "training",
    difficulty: "Hard",
    tags: ["MoE", "Load Balancing", "Auxiliary Loss"],
    summary: "对照路由器“想分给谁”的 P_i 与实际“分给谁”的 f_i，用辅助损失阻止路由崩塌。",
    concepts: [
      "没有约束时，router 可能总把 token 发给少数热门专家，造成容量浪费和训练不稳。",
      "负载均衡损失通常同时考虑专家被选择的频率和 router 分配概率。",
      "这类辅助 loss 不直接优化语言建模目标，但能让 MoE 系统更可训练。"
    ],
    quiz: {
      question: "MoE load balancing loss 主要想避免什么问题？",
      options: ["所有 token 挤到少数专家", "学习率永远为 0", "embedding 维度变小"],
      answer: 0,
      explain: "负载不均会导致一些专家过载，另一些专家几乎不学习。"
    },
    handsOn: [
      "统计 top-k 结果中每个专家的 token 占比。",
      "计算 router probability 在专家维度上的平均值。",
      "观察负载极不均时辅助 loss 的数值变化。"
    ],
    formula: "balance_loss ∝ num_experts * sum(tokens_per_expert * prob_per_expert)"
  },
  {
    id: "08",
    title: "Architecture Tricks",
    file: "08_Architecture_Tricks.ipynb",
    category: "architecture",
    difficulty: "Medium",
    tags: ["LLM", "Tricks", "Stability"],
    summary: "梳理现代 LLM 结构中的常用技巧：norm、激活、位置编码、残差与初始化。",
    concepts: [
      "架构技巧通常服务于三个目标：训练稳定、推理高效、表达力更强。",
      "RMSNorm、RoPE、SwiGLU、GQA 是许多现代 decoder-only LLM 的常见组合。",
      "理解 trick 时要问：它改变了数学表达、计算量、显存，还是训练动态？"
    ],
    quiz: {
      question: "评估一个架构技巧时，最应该同时关注哪三类影响？",
      options: ["表达力、计算量、稳定性", "文件名、注释数量、变量长度", "只看参数量"],
      answer: 0,
      explain: "架构设计通常是在效果、成本和稳定性之间取平衡。"
    },
    handsOn: [
      "为每个 trick 写一句话：解决什么问题，代价是什么。",
      "把这些 trick 标注到 LLaMA block 的数据流图上。",
      "选一个 trick 做 ablation 思考：去掉它会怎样？"
    ],
    formula: "Good architecture = quality + stability + efficiency"
  },
  {
    id: "09",
    title: "SFT Training Loop",
    file: "09_SFT_Training_Loop.ipynb",
    category: "training",
    difficulty: "Medium",
    tags: ["SFT", "Loss", "Optimizer"],
    summary: "把数据 batch、forward、loss、backward、optimizer.step 串成监督微调循环。",
    concepts: [
      "SFT 的训练目标通常是 next-token prediction，只在需要学习的位置计算 loss。",
      "训练循环的基本顺序是 zero_grad -> forward -> loss -> backward -> step。",
      "labels 中的 ignore_index 可用于屏蔽 prompt 或 padding 部分。"
    ],
    quiz: {
      question: "为什么训练循环里通常要先 optimizer.zero_grad()？",
      options: ["PyTorch 默认会累积梯度", "它会加载 tokenizer", "它会自动保存 checkpoint"],
      answer: 0,
      explain: "如果不清零，当前 batch 的梯度会和历史梯度累加。"
    },
    handsOn: [
      "手写一个最小训练循环，确保四步顺序正确。",
      "打印 loss.item()，确认 loss 随训练迭代有变化。",
      "用 ignore_index 屏蔽不该训练的位置。"
    ],
    formula: "loss = CrossEntropy(logits[:, :-1], labels[:, 1:])"
  },
  {
    id: "10",
    title: "LoRA Tutorial",
    file: "10_LoRA_Tutorial.ipynb",
    category: "training",
    difficulty: "Medium",
    tags: ["LoRA", "PEFT", "Low-rank"],
    summary: "用低秩矩阵增量训练大模型，让微调参数量从全量变成小而可控。",
    concepts: [
      "LoRA 冻结原始权重 W，只训练低秩增量 BA。",
      "rank r 越小，训练参数越少，但表达能力也更受限。",
      "常见做法是在 attention projection 或 FFN projection 上注入 LoRA。"
    ],
    quiz: {
      question: "LoRA 微调时通常被冻结的是哪部分？",
      options: ["原始预训练权重 W", "LoRA A/B 矩阵", "训练数据"],
      answer: 0,
      explain: "LoRA 只训练低秩适配器，原模型权重保持不变。"
    },
    handsOn: [
      "实现 y = xW + scale * xAB 的前向。",
      "计算 LoRA 参数量，并和全量 Linear 参数量比较。",
      "尝试不同 rank，观察参数量如何变化。"
    ],
    formula: "W' = W + (alpha / r) * B A"
  },
  {
    id: "11",
    title: "LR Schedulers WSD Cosine",
    file: "11_LR_Schedulers_WSD_Cosine.ipynb",
    category: "training",
    difficulty: "Medium",
    tags: ["Learning Rate", "Scheduler", "Warmup"],
    summary: "理解 warmup、stable、decay 与 cosine 曲线如何影响训练节奏。",
    concepts: [
      "Warmup 用较小学习率启动训练，减少早期不稳定。",
      "Cosine decay 平滑降低学习率，常用于训练后期收敛。",
      "WSD 把学习率阶段拆为 warmup、stable、decay，便于控制长训练。"
    ],
    quiz: {
      question: "Warmup 阶段的主要目的是什么？",
      options: ["让训练早期更稳定", "增加模型参数量", "减少 tokenizer 词表"],
      answer: 0,
      explain: "训练刚开始梯度和激活分布尚不稳定，直接用大学习率更容易震荡。"
    },
    handsOn: [
      "画出 warmup + cosine 的学习率曲线。",
      "实现 step 到 lr 的函数，并打印关键 step 的 lr。",
      "比较无 warmup 和有 warmup 的前几个 step 学习率。"
    ],
    formula: "lr(t) = lr_min + 0.5 * (lr_max-lr_min) * (1 + cos(pi * t / T))"
  },
  {
    id: "12",
    title: "RLHF PPO Memory",
    file: "12_RLHF_PPO_Memory.ipynb",
    category: "training",
    difficulty: "Hard",
    tags: ["RLHF", "PPO", "Memory"],
    summary: "拆解 PPO 式 RLHF 中 policy、reference、reward、value 带来的显存压力。",
    concepts: [
      "RLHF PPO 往往同时涉及 policy、reference、reward model 和 value head。",
      "显存不只来自参数，还来自 activation、optimizer states、KV cache 和 rollout batch。",
      "估算显存时要区分训练态和推理态，尤其是是否需要保存激活用于反传。"
    ],
    quiz: {
      question: "PPO 训练显存通常比普通 SFT 更复杂，主要因为它可能同时持有哪些模型或状态？",
      options: ["policy/reference/reward/value 等组件", "只多了一个 tokenizer", "只多了更多注释"],
      answer: 0,
      explain: "RLHF PPO 的组件更多，且 rollout 与训练阶段的状态都要考虑。"
    },
    handsOn: [
      "列出 PPO 流程中每个模型是否需要梯度。",
      "粗估参数、优化器状态和激活各占多少显存。",
      "思考哪些组件可以 offload 或用更低精度。"
    ],
    formula: "actor_loss = -mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))"
  },
  {
    id: "13",
    title: "DPO Loss Tutorial",
    file: "13_DPO_Loss_Tutorial.ipynb",
    category: "training",
    difficulty: "Medium",
    tags: ["DPO", "Preference", "Alignment"],
    summary: "用 chosen/rejected 偏好对直接优化 policy，不显式训练 reward model。",
    concepts: [
      "DPO 输入是偏好对：chosen response 和 rejected response。",
      "核心信号来自 policy 相对 reference 在两条回答上的 logprob 差异。",
      "beta 控制偏好优化强度，也影响 policy 偏离 reference 的程度。"
    ],
    quiz: {
      question: "DPO 训练数据的核心形式是什么？",
      options: ["chosen/rejected 偏好对", "只有无标签文本", "图像分类标签"],
      answer: 0,
      explain: "DPO 直接利用偏好对优化回答排序。"
    },
    handsOn: [
      "计算 chosen 和 rejected 的 sequence logprob。",
      "写出 policy log-ratio 与 reference log-ratio 的差。",
      "改变 beta，观察 loss 对偏好差距的敏感度。"
    ],
    formula: "loss = -log sigmoid(beta * ((logp_c-logp_r) - (logpref_c-logpref_r)))"
  },
  {
    id: "14",
    title: "Attention Backward Math",
    file: "14_Attention_Backward_Math.ipynb",
    category: "architecture",
    difficulty: "Hard",
    tags: ["Attention", "Backward", "Matrix Grad"],
    summary: "沿着 O = softmax(QK^T)V 反向拆梯度，理解注意力 backward 的主路径。",
    concepts: [
      "Attention backward 可分为穿过 V、softmax、scale dot-product 三段。",
      "Softmax backward 不是简单逐元素相乘，它和整行概率分布相关。",
      "理解矩阵维度是推导 dQ、dK、dV 的关键。"
    ],
    quiz: {
      question: "Attention backward 中，softmax 的梯度为什么需要特别处理？",
      options: ["softmax 输出同一行元素相互耦合", "softmax 没有导数", "softmax 只支持整数"],
      answer: 0,
      explain: "一个 logit 改变会影响同一行所有 softmax 概率。"
    },
    handsOn: [
      "从 O = P V 开始推导 dV 和 dP。",
      "实现一行 softmax backward，并和 autograd 对比。",
      "继续从 scores = QK^T / sqrt(d) 推导 dQ 和 dK。"
    ],
    formula: "P = softmax(S); O = P V"
  },
  {
    id: "15",
    title: "FlashAttention Sim",
    file: "15_FlashAttention_Sim.ipynb",
    category: "inference",
    difficulty: "Hard",
    tags: ["FlashAttention", "Tiling", "Memory"],
    summary: "用分块和在线 softmax 理解 FlashAttention 如何避免 materialize NxN 注意力矩阵。",
    concepts: [
      "普通 attention 会显式形成 S 和 P，显存随序列长度平方增长。",
      "FlashAttention 按块读取 Q/K/V，在 SRAM 中完成局部计算并维护在线 softmax 统计量。",
      "核心收益来自 IO-aware：减少 HBM 读写，而不是改变 attention 数学结果。"
    ],
    quiz: {
      question: "FlashAttention 的核心优化目标是什么？",
      options: ["减少 HBM 读写和 NxN 中间矩阵", "改变 softmax 定义", "删除 V 矩阵"],
      answer: 0,
      explain: "FlashAttention 保持精确 attention，但重排计算以降低内存 IO。"
    },
    handsOn: [
      "实现普通 attention，记录 scores 的 shape。",
      "用小块模拟分块 attention 的循环。",
      "比较显式 scores 矩阵和分块统计量需要的存储。"
    ],
    formula: "online_softmax keeps row max m and normalizer l per block"
  },
  {
    id: "16",
    title: "Decoding Strategies",
    file: "16_Decoding_Strategies.ipynb",
    category: "inference",
    difficulty: "Medium",
    tags: ["Decoding", "Sampling", "Beam"],
    summary: "比较 greedy、temperature、top-k、top-p、beam search 的生成行为。",
    concepts: [
      "Greedy 每步选最大概率 token，稳定但可能缺少多样性。",
      "Temperature 调整分布尖锐程度，top-k/top-p 限制候选集合。",
      "Beam search 保留多条高分路径，适合某些确定性任务，但开放生成可能变得保守。"
    ],
    quiz: {
      question: "提高 temperature 通常会让采样结果怎样变化？",
      options: ["更随机、更有多样性", "完全变成 greedy", "词表大小变成 1"],
      answer: 0,
      explain: "较高 temperature 会压平概率分布，使低概率 token 更有机会被采到。"
    },
    handsOn: [
      "给一组 logits 分别做 temperature scaling。",
      "实现 top-k 或 top-p 过滤，再从过滤后的分布采样。",
      "比较同一 prompt 下 greedy 和 sampling 的输出差异。"
    ],
    formula: "p = softmax(logits / temperature)"
  },
  {
    id: "17",
    title: "vLLM PagedAttention",
    file: "17_vLLM_PagedAttention.ipynb",
    category: "inference",
    difficulty: "Hard",
    tags: ["vLLM", "KV Cache", "Serving"],
    summary: "把 KV cache 看成分页内存，理解连续批处理下如何减少碎片并提升吞吐。",
    concepts: [
      "LLM serving 的瓶颈之一是每个请求长度不同，KV cache 管理容易碎片化。",
      "PagedAttention 把 KV cache 切成 block，像虚拟内存分页一样管理。",
      "这种设计支持更灵活的 batch 调度和更高的显存利用率。"
    ],
    quiz: {
      question: "PagedAttention 借鉴了哪类系统思想？",
      options: ["虚拟内存分页", "图像卷积池化", "数据库 SQL join"],
      answer: 0,
      explain: "它把 KV cache 分块管理，减少连续大块分配带来的碎片。"
    },
    handsOn: [
      "画出两个不同长度请求的 KV block 分配图。",
      "模拟 block table：逻辑 token block 到物理 cache block 的映射。",
      "思考请求结束后哪些 block 可以回收。"
    ],
    formula: "logical_block_id -> physical_kv_block_id"
  },
  {
    id: "18",
    title: "Speculative Decoding",
    file: "18_Speculative_Decoding.ipynb",
    category: "inference",
    difficulty: "Hard",
    tags: ["Draft Model", "Verify", "Decoding"],
    summary: "用小 draft model 一次提出多个 token，再由大模型并行验证以加速生成。",
    concepts: [
      "Speculative decoding 不改变目标模型分布，目标是减少大模型逐 token 调用次数。",
      "Draft model 先快速生成候选 token，target model 对候选进行验证和接受。",
      "加速效果取决于 draft 质量、接受率和 target 并行验证效率。"
    ],
    quiz: {
      question: "Speculative decoding 中 draft model 的作用是什么？",
      options: ["快速提出候选 token", "替代 target model 做最终分布", "只负责分词"],
      answer: 0,
      explain: "Draft model 生成候选，target model 仍负责校验以保持分布正确性。"
    },
    handsOn: [
      "模拟 draft 一次生成 k 个 token。",
      "写出 target 验证候选 token 的接受/拒绝流程。",
      "改变接受率，估算理论加速比变化。"
    ],
    formula: "speedup depends on accepted draft tokens per target call"
  },
  {
    id: "19",
    title: "SGLang RadixAttention",
    file: "19_SGLang_RadixAttention.ipynb",
    category: "inference",
    difficulty: "Hard",
    tags: ["SGLang", "Prefix Cache", "Radix Tree"],
    summary: "利用前缀复用，把共享 prompt 的 KV cache 放进 radix tree 中管理。",
    concepts: [
      "许多推理请求共享系统提示词或上下文前缀，重复计算这些 KV cache 很浪费。",
      "Radix tree 适合存储和匹配共享前缀。",
      "Prefix cache 命中率越高，prefill 阶段节省越明显。"
    ],
    quiz: {
      question: "RadixAttention 主要优化哪类重复？",
      options: ["共享前缀的 KV cache 重复计算", "矩阵乘法的乘号数量", "optimizer state 存储"],
      answer: 0,
      explain: "它让相同前缀的请求复用已计算的 KV cache。"
    },
    handsOn: [
      "列出三条 prompt，找出它们的公共前缀。",
      "画一个 radix tree，标注每个节点对应的 token span。",
      "估算命中公共前缀后可少算多少 prefill token。"
    ],
    formula: "shared prefix -> reusable KV cache nodes"
  },
  {
    id: "20",
    title: "Quantization W8A16",
    file: "20_Quantization_W8A16.ipynb",
    category: "inference",
    difficulty: "Medium",
    tags: ["Quantization", "W8A16", "Inference"],
    summary: "理解权重 int8、激活 fp16 的推理量化路径和 scale/zero-point 的作用。",
    concepts: [
      "W8A16 表示权重用 8-bit，激活仍用 16-bit。",
      "量化通常需要 scale，把浮点范围映射到整数范围。",
      "主要收益是减少权重显存和带宽，但要控制量化误差。"
    ],
    quiz: {
      question: "W8A16 中的 W8 指什么？",
      options: ["权重 8-bit", "窗口大小为 8", "每层 8 个 head"],
      answer: 0,
      explain: "W 表示 weight，8 表示 8-bit 表示。"
    },
    handsOn: [
      "对一个小权重矩阵计算 absmax scale。",
      "把 fp16/fp32 权重量化到 int8，再反量化回来。",
      "比较量化前后矩阵乘法输出误差。"
    ],
    formula: "q = round(x / scale); x_hat = q * scale"
  },
  {
    id: "21",
    title: "Gradient Checkpointing",
    file: "21_Gradient_Checkpointing.ipynb",
    category: "training",
    difficulty: "Medium",
    tags: ["Memory", "Recompute", "Training"],
    summary: "用反向时重算部分激活来换取训练显存下降。",
    concepts: [
      "普通训练会保存大量中间激活用于 backward。",
      "Gradient checkpointing 只保存检查点，反向时重算检查点之间的前向。",
      "它降低显存，但增加计算量，是典型的 compute-memory tradeoff。"
    ],
    quiz: {
      question: "Gradient checkpointing 用什么换显存？",
      options: ["更多重算计算量", "更大的词表", "更长的文件名"],
      answer: 0,
      explain: "少存激活，反向传播时重新计算部分前向。"
    },
    handsOn: [
      "标注一个多层网络中哪些 activation 会被保存。",
      "使用 torch.utils.checkpoint 包一段模块。",
      "比较开启前后的显存占用和耗时。"
    ],
    formula: "less saved activations + recompute during backward"
  },
  {
    id: "22",
    title: "QLoRA and 4bit Quantization",
    file: "22_QLoRA_and_4bit_Quantization.ipynb",
    category: "training",
    difficulty: "Hard",
    tags: ["QLoRA", "4-bit", "PEFT"],
    summary: "把冻结的 4-bit 基座模型和可训练 LoRA 适配器结合起来做低显存微调。",
    concepts: [
      "QLoRA 通常冻结 4-bit 量化后的 base model，只训练 LoRA 参数。",
      "4-bit quantization 显著降低权重存储，但训练中仍要处理计算 dtype。",
      "NF4、double quantization、paged optimizer 是 QLoRA 相关的重要工程点。"
    ],
    quiz: {
      question: "QLoRA 中通常真正训练的是哪部分？",
      options: ["LoRA adapter 参数", "4-bit base model 全部权重", "tokenizer vocabulary"],
      answer: 0,
      explain: "Base model 量化并冻结，LoRA adapter 承担可训练增量。"
    },
    handsOn: [
      "计算全量微调和 LoRA 微调的可训练参数比例。",
      "解释为什么 4-bit base 仍可能用 bf16 做计算。",
      "画出 base weight、dequant、LoRA branch 的前向路径。"
    ],
    formula: "output = dequant(W4bit) x + LoRA(x)"
  },
  {
    id: "23",
    title: "ZeRO Optimizer Sim",
    file: "23_ZeRO_Optimizer_Sim.ipynb",
    category: "training",
    difficulty: "Hard",
    tags: ["ZeRO", "Distributed", "Optimizer"],
    summary: "用切分参数、梯度和优化器状态理解 ZeRO 三个阶段的显存节省。",
    concepts: [
      "数据并行会在每张卡复制参数、梯度和优化器状态。",
      "ZeRO-1 切 optimizer states，ZeRO-2 进一步切 gradients，ZeRO-3 连 parameters 也切。",
      "节省显存的代价是需要更多通信来按需聚合。"
    ],
    quiz: {
      question: "ZeRO-3 相比 ZeRO-1/2 额外切分了什么？",
      options: ["模型参数", "训练数据文本", "Python 注释"],
      answer: 0,
      explain: "ZeRO-3 将 parameters 也 shard 到不同数据并行 rank。"
    },
    handsOn: [
      "列出 DP 下每张卡复制的三类状态。",
      "对比 ZeRO-1/2/3 各自切分哪些状态。",
      "用 world_size=4 粗估每阶段单卡状态量。"
    ],
    formula: "ZeRO memory per rank decreases as states are sharded across ranks"
  },
  {
    id: "24",
    title: "Tensor Parallelism Sim",
    file: "24_Tensor_Parallelism_Sim.ipynb",
    category: "training",
    difficulty: "Hard",
    tags: ["Tensor Parallel", "MatMul", "Distributed"],
    summary: "把大矩阵乘法沿列或行切开，理解 tensor parallel 的通信位置。",
    concepts: [
      "Column parallel 通常切输出维，多个 rank 各算一部分输出。",
      "Row parallel 通常切输入维，需要对各 rank 局部结果做 all-reduce。",
      "Megatron 风格 TP 的关键是把线性层切分和通信配对设计好。"
    ],
    quiz: {
      question: "Row parallel linear 通常需要哪种通信来合并局部输出？",
      options: ["All-reduce", "Top-k sampling", "Tokenizer decode"],
      answer: 0,
      explain: "各 rank 计算部分输入贡献，需要求和得到完整输出。"
    },
    handsOn: [
      "把一个 Linear 权重按列切成两块，分别计算输出后 concat。",
      "把权重按行切分，分别计算局部输出后求和。",
      "标注每种切法需要 concat 还是 all-reduce。"
    ],
    formula: "Y = X [W1 W2] -> concat(XW1, XW2)"
  },
  {
    id: "25",
    title: "Pipeline Parallelism MicroBatch",
    file: "25_Pipeline_Parallelism_MicroBatch.ipynb",
    category: "training",
    difficulty: "Hard",
    tags: ["Pipeline Parallel", "Microbatch", "Bubble"],
    summary: "把模型层切成多个 stage，再用 microbatch 填满流水线，减少空泡。",
    concepts: [
      "Pipeline parallel 按层切模型，不同 stage 放在不同设备上。",
      "Microbatch 让多个小批次在不同 stage 上交错执行，提高设备利用率。",
      "流水线空泡来自 warmup 和 cooldown，microbatch 越多相对空泡越小。"
    ],
    quiz: {
      question: "Pipeline parallel 中 microbatch 的主要作用是什么？",
      options: ["填充流水线、减少空泡", "改变 tokenizer", "减少模型层数到 1"],
      answer: 0,
      explain: "多个 microbatch 可以在不同 stage 上交错执行，避免设备长时间空闲。"
    },
    handsOn: [
      "画出 2 个 stage、4 个 microbatch 的执行时间表。",
      "标出 warmup、steady 和 cooldown 阶段。",
      "改变 microbatch 数量，观察空泡比例如何变化。"
    ],
    formula: "pipeline bubble ratio roughly decreases as microbatches increase"
  }
];

// Upstream restructured everything after 11. Preserve the customized 00-11
// metadata and replace the legacy tail with the canonical 12-32 curriculum.
levels.splice(12, levels.length - 12, ...curriculumV2);

const categoryLabel = {
  foundation: "基础",
  architecture: "结构",
  training: "训练",
  inference: "推理"
};

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(value) {
  return esc(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inList = false;
  let inQuote = false;

  function closeList() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  }

  function closeQuote() {
    if (inQuote) {
      html.push("</blockquote>");
      inQuote = false;
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      closeQuote();
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeList();
      closeQuote();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    if (line.startsWith(">")) {
      closeList();
      if (!inQuote) {
        html.push("<blockquote>");
        inQuote = true;
      }
      html.push(`<p>${inlineMarkdown(line.replace(/^>\s?/, ""))}</p>`);
      continue;
    }

    const bullet = line.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (bullet) {
      closeQuote();
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }

    closeList();
    closeQuote();
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }

  closeList();
  closeQuote();
  return html.join("\n");
}

function notebookGuideHtml(level) {
  const notebookPath = path.join(root, "..", level.file);
  if (!fs.existsSync(notebookPath)) return "";

  const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
  const markdown = notebook.cells
    .filter((cell) => cell.cell_type === "markdown")
    .map((cell) => Array.isArray(cell.source) ? cell.source.join("") : String(cell.source || ""))
    .filter((text) => {
      const compact = text.replace(/\s/g, "");
      return compact && !compact.includes("STOPHERE");
    })
    .join("\n\n");

  if (!markdown.trim()) return "";

  return `
    <section class="card notebook-guide section">
      <div class="quest-title">
        <h2>Notebook 原文导学</h2>
        <span class="reward">来自 ${esc(level.file)}</span>
      </div>
      <div class="notebook-content">
        ${markdownToHtml(markdown)}
      </div>
    </section>`;
}

function slug(level) {
  return `${level.id}_${level.file.replace(/\.ipynb$/, "").replace(/^\d+_/, "").toLowerCase()}.html`;
}

function genericExampleHtml(level) {
  return `
          <div class="shape-story">
            <div class="story-panel">
              <strong>1. 先认出本关的核心对象</strong>
              <p>${esc(level.summary)}</p>
              <div class="concept-badges">
                ${level.tags.map((tag) => `<span>${esc(tag)}</span>`).join("")}
              </div>
            </div>
            <div class="story-arrow">把名词翻译成张量、函数或训练步骤</div>
            <div class="story-panel">
              <strong>2. 再看输入输出关系</strong>
              <p>每个 notebook 的 TODO 都在训练一种固定动作：先确认输入是什么，再确认输出应该长什么样，最后用测试检查数值或 shape。</p>
              <div class="mini-pipeline">
                <span>输入</span><span>核心操作</span><span>输出/测试</span>
              </div>
            </div>
          </div>`;
}

function genericSyntaxHtml(level) {
  return `
          <div class="syntax-card">
            <h4>语法热身：读 PyTorch 练习时先抓这三件事</h4>
            <p>下面是通用读法，不直接等同于本关 notebook 答案。</p>
            <pre><code># 1. 先看 shape，别急着写公式
print(x.shape)

# 2. 写模块时，参数通常放在 __init__
class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)

# 3. 测试通常会检查 shape 或数值接近
assert out.shape[0] == x.shape[0]
torch.allclose(out_a, out_b, atol=1e-5)</code></pre>
            <p class="syntax-tip">读法：先找 TODO 附近的输入变量和期望输出，再回到课程区确认这个操作在数学上想做什么。</p>
          </div>`;
}

function genericLessons(level) {
  return [
    {
      id: "core",
      title: `${level.title}：先建立本关直觉`,
      todo: "本关 TODO",
      prerequisite: level.concepts,
      intuition: "先不要背实现。把本关拆成输入、核心操作、输出三段，写代码时就不容易迷路。",
      exampleHtml: genericExampleHtml(level),
      syntaxHtml: genericSyntaxHtml(level),
      checkpoint: {
        question: level.quiz.question,
        options: level.quiz.options,
        answer: level.quiz.answer,
        explain: level.quiz.explain
      },
      homework: level.handsOn
    }
  ];
}

function stableHash(text) {
  let hash = 2166136261;
  for (const char of String(text)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function answerIndexFor(checkpoint, levelId, lessonId) {
  const rawAnswer = checkpoint.answer;
  const answer = typeof rawAnswer === "number"
    ? rawAnswer
    : checkpoint.options.indexOf(rawAnswer);
  if (answer < 0 || answer >= checkpoint.options.length) {
    throw new Error(`Invalid checkpoint answer for ${levelId}/${lessonId}`);
  }
  return answer;
}

function shuffledCheckpoint(level, lesson, lessonIndex) {
  const originalAnswer = answerIndexFor(lesson.checkpoint, level.id, lesson.id);
  const options = lesson.checkpoint.options;
  const correctOption = options[originalAnswer];
  const targetAnswer = (Number(level.id) + lessonIndex) % options.length;
  const slots = options.map((_, index) => index).filter((index) => index !== targetAnswer);
  const distractors = options
    .map((option, index) => ({ option, index }))
    .filter((entry) => entry.index !== originalAnswer)
    .sort((a, b) => stableHash(`${level.id}:${lesson.id}:${a.option}`) - stableHash(`${level.id}:${lesson.id}:${b.option}`));
  const shuffled = Array(options.length);
  shuffled[targetAnswer] = correctOption;
  slots.forEach((slot, index) => {
    shuffled[slot] = distractors[index].option;
  });
  return {
    ...lesson.checkpoint,
    options: shuffled,
    answer: targetAnswer
  };
}

function lessonLevelPage(level, prev, next) {
  const lessonStyles = level.lessons.map((lesson) => lesson.styles || "").filter(Boolean).join("\n");
  const lessons = level.lessons.map((lesson, index) => {
    const checkpoint = shuffledCheckpoint(level, lesson, index);
    const prerequisite = lesson.prerequisite.map((item) => `<li>${esc(item)}</li>`).join("");
    const homework = lesson.homework.map((item, taskIndex) => `
              <label class="homework-item">
                <input type="checkbox" data-homework="${lesson.id}-${taskIndex}">
                <span>${esc(item)}</span>
              </label>`).join("");
    const options = checkpoint.options.map((option, optionIndex) => `
              <label class="checkpoint-option">
                <input type="radio" name="checkpoint-${lesson.id}" value="${optionIndex}">
                <span>${esc(option)}</span>
              </label>`).join("");

    if (lesson.predict) {
      const predictOptions = lesson.predict.options.map((option, optionIndex) => `
              <label class="predict-option">
                <input type="radio" name="predict-${lesson.id}" value="${optionIndex}">
                <span>${esc(option)}</span>
              </label>`).join("");

      return `
        <article class="lesson-card inquiry" data-lesson="${lesson.id}">
          <div class="lesson-top">
            <span class="level-pill">Mission ${index + 1}</span>
            <span class="todo-pill">${esc(lesson.todo)}</span>
          </div>
          <h2>${esc(lesson.title)}</h2>

          <div class="lesson-section predict">
            <h3>🎯 先猜一猜（下注解锁）</h3>
            <p class="predict-hook">${esc(lesson.predict.hook)}</p>
            <p class="predict-q">${esc(lesson.predict.question)}</p>
            <div class="predict-options">${predictOptions}</div>
            <div class="predict-feedback" data-predict-feedback="${lesson.id}">先押一个假设，下面的讲解就会解锁。猜错完全没关系——带着疑问读，记得最牢。</div>
          </div>

          <div class="gated" data-gated="${lesson.id}">
            <div class="lesson-section">
              <h3>先补的知识</h3>
              <ul>${prerequisite}</ul>
            </div>

            <div class="lesson-section intuition">
              <h3>图解原理</h3>
              <p>${esc(lesson.intuition)}</p>
              ${lesson.exampleHtml}
              ${lesson.syntaxHtml || ""}
            </div>

            <div class="lesson-section checkpoint">
              <h3>巩固一下</h3>
              <p>${esc(checkpoint.question)}</p>
              <div class="checkpoint-options">${options}</div>
              <div class="feedback" data-feedback="${lesson.id}"></div>
            </div>
          </div>

          <div class="lesson-section homework">
            <h3>回到 notebook 的作业</h3>
            <p>这里不直接写答案。你已经拿到足够输入，最后用 notebook 的 TODO 做举一反三。</p>
            ${homework}
          </div>
        </article>`;
    }

    return `
        <article class="lesson-card" data-lesson="${lesson.id}">
          <div class="lesson-top">
            <span class="level-pill">Mission ${index + 1}</span>
            <span class="todo-pill">${esc(lesson.todo)}</span>
          </div>
          <h2>${esc(lesson.title)}</h2>

          <div class="lesson-section">
            <h3>先补的知识</h3>
            <ul>${prerequisite}</ul>
          </div>

          <div class="lesson-section intuition">
            <h3>图解原理</h3>
            <p>${esc(lesson.intuition)}</p>
            ${lesson.exampleHtml}
            ${lesson.syntaxHtml || ""}
          </div>

          <div class="lesson-section checkpoint">
            <h3>闯关题</h3>
            <p>${esc(checkpoint.question)}</p>
            <div class="checkpoint-options">${options}</div>
            <div class="feedback" data-feedback="${lesson.id}"></div>
          </div>

          <div class="lesson-section homework">
            <h3>回到 notebook 的作业</h3>
            <p>这里不直接写答案。你已经拿到足够输入，最后用 notebook 的 TODO 做举一反三。</p>
            ${homework}
          </div>
        </article>`;
  }).join("");

  const checkpointData = Object.fromEntries(level.lessons.map((lesson, index) => {
    const checkpoint = shuffledCheckpoint(level, lesson, index);
    return [
      lesson.id,
      {
        answer: checkpoint.answer,
        explain: checkpoint.explain
      }
    ];
  }));
  const predictData = Object.fromEntries(level.lessons
    .filter((lesson) => lesson.predict)
    .map((lesson) => [
      lesson.id,
      {
        answer: answerIndexFor(lesson.predict, level.id, lesson.id),
        revealNote: lesson.predict.revealNote
      }
    ]));
  const predictTotal = level.lessons.filter((lesson) => lesson.predict).length;
  const homeworkCount = level.lessons.reduce((total, lesson) => total + lesson.homework.length, 0);
  const checkpointLabel = `${level.lessons.length} 道`;
  const prevLink = prev ? `<a class="ghost" href="${slug(prev)}">上一关：L${prev.id}</a>` : `<a class="ghost" href="../index.html">返回地图</a>`;
  const nextLink = next ? `<a class="primary" href="${slug(next)}">下一关：L${next.id}</a>` : `<a class="primary" href="../index.html">回到地图</a>`;

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Level ${level.id} | ${esc(level.title)}</title>
  <style>
    :root {
      --bg: #f7f6f1;
      --paper: #ffffff;
      --ink: #182033;
      --muted: #657184;
      --line: #dce2e8;
      --blue: #2563eb;
      --green: #12805c;
      --amber: #bd6516;
      --rose: #be3f5b;
      --soft-blue: #e8f0ff;
      --soft-green: #e7f6ee;
      --soft-amber: #fff1dd;
      --soft-rose: #fff0f4;
      --shadow: 0 16px 40px rgba(24, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(37, 99, 235, 0.045) 1px, transparent 1px),
        linear-gradient(rgba(18, 128, 92, 0.045) 1px, transparent 1px),
        var(--bg);
      background-size: 30px 30px;
    }
    a { color: inherit; text-decoration: none; }
    .shell { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 24px 0 44px; }
    .top { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 18px; }
    .hud { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .ghost, .primary {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
    }
    .primary { border-color: var(--blue); color: var(--blue); background: var(--soft-blue); font-weight: 800; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }
    .card, .lesson-card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 22px;
    }
    .hero-main {
      background:
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(232,240,255,0.78)),
        var(--paper);
      display: grid;
      align-content: center;
    }
    .hero-map {
      background:
        linear-gradient(135deg, rgba(232,240,255,0.95), rgba(231,246,238,0.95)),
        var(--paper);
      display: grid;
      gap: 12px;
      align-content: center;
    }
    .eyebrow { margin: 0 0 10px; color: var(--green); font-weight: 900; }
    h1 { margin: 0; font-size: clamp(34px, 5vw, 56px); line-height: 1.02; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 24px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }
    p, li { color: var(--muted); line-height: 1.65; font-size: 15px; }
    .lead { font-size: 18px; max-width: 760px; }
    .pill, .level-pill, .todo-pill {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      border: 1px solid var(--line);
      background: #fbfcfd;
      color: var(--muted);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
      font-weight: 800;
    }
    .level-pill { border-color: #b9cdfb; background: var(--soft-blue); color: #1d4ed8; }
    .todo-pill { border-color: #f0c483; background: var(--soft-amber); color: #7b430c; }
    .meta, .lesson-top { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .xp-wrap {
      width: min(100%, 540px);
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      padding: 4px;
    }
    .xp-bar {
      width: 0%;
      height: 14px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--green), var(--blue));
      transition: width 220ms ease;
    }
    .map-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .map-step {
      min-height: 86px;
      border: 1px solid rgba(37,99,235,0.22);
      background: rgba(255,255,255,0.78);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      place-items: center;
      text-align: center;
      font-weight: 900;
      color: #1e3f88;
    }
    .formula {
      border: 1px solid rgba(24,32,51,0.13);
      border-radius: 8px;
      background: rgba(255,255,255,0.82);
      padding: 13px;
      line-height: 1.55;
      color: var(--muted);
    }
    .lessons { display: grid; gap: 16px; }
    .lesson-card { display: grid; gap: 16px; }
    .lesson-section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 16px;
    }
    .intuition { background: #fffdf8; border-color: #efd1a0; }
    .checkpoint { background: #f8fbff; border-color: #bdd0ff; }
    .homework { background: #f7fff9; border-color: #a9dbc7; }
    .predict { background: #f4f1ff; border-color: #c9bdf0; }
    .predict h3 { color: #6d4fc4; }
    .predict-hook {
      font-style: italic;
      color: #4a3a7a;
      border-left: 4px solid #b8a6ea;
      background: rgba(255,255,255,0.6);
      border-radius: 6px;
      padding: 10px 12px;
      margin: 6px 0 12px;
      line-height: 1.6;
    }
    .predict-q { font-weight: 800; margin: 0 0 6px; }
    .predict-options { display: grid; gap: 10px; margin-top: 4px; }
    .predict-option {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      border: 1px solid #cabff0;
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      cursor: pointer;
      transition: border-color .15s, background .15s;
    }
    .predict-option:hover { border-color: #8b6fe0; }
    .predict-option.picked { border-color: #6d4fc4; background: #efeaff; box-shadow: inset 0 0 0 1px #b8a6ea; }
    .predict-feedback {
      min-height: 24px;
      margin-top: 12px;
      font-weight: 700;
      color: #6d4fc4;
      line-height: 1.55;
    }
    .predict-feedback.hit { color: var(--green); }
    .predict-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-top: 10px;
      padding: 6px 12px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 13px;
    }
    .predict-badge.hit { background: var(--soft-green); color: var(--green); border: 1px solid #99d6bf; }
    .predict-badge.miss { background: var(--soft-blue); color: #1d4ed8; border: 1px solid #b9cdfb; }
    .gated {
      position: relative;
      display: grid;
      gap: 16px;
      margin-top: 16px;
      border-radius: 12px;
    }
    .gated.locked { user-select: none; }
    .gated.locked > * {
      filter: blur(7px) grayscale(0.35);
      opacity: 0.55;
      pointer-events: none;
      transition: filter .45s ease, opacity .45s ease;
    }
    .gated.locked::after {
      content: "🔒 先在上方押一个假设，解锁讲解";
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      font-weight: 900;
      font-size: 16px;
      color: #4a3a7a;
      background: rgba(244,241,255,0.4);
      border: 2px dashed #b8a6ea;
      border-radius: 12px;
      pointer-events: none;
      z-index: 2;
    }
    .think {
      border: 1px solid #d9a15a;
      border-left: 4px solid #d9a15a;
      border-radius: 8px;
      background: #fffaf2;
      padding: 0;
      margin: 12px 0;
      overflow: hidden;
    }
    .think > summary {
      list-style: none;
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 800;
      color: #8a4d0a;
      display: flex;
      align-items: center;
      gap: 8px;
      user-select: none;
    }
    .think > summary::-webkit-details-marker { display: none; }
    .think > summary::before {
      content: "🤔";
      font-size: 16px;
    }
    .think > summary::after {
      content: "点击看一种思路 ▸";
      margin-left: auto;
      font-size: 12px;
      font-weight: 700;
      color: #b07a2e;
      background: #fff1dd;
      border: 1px solid #e8b66f;
      border-radius: 999px;
      padding: 3px 10px;
      white-space: nowrap;
    }
    .think[open] > summary::after { content: "收起 ▾"; }
    .think[open] > summary { border-bottom: 1px dashed #e8c79a; }
    .think .think-body {
      padding: 12px 14px 14px;
      line-height: 1.7;
      color: #4a3a2a;
    }
    .think .think-body p { margin: 0 0 8px; }
    .think .think-body p:last-child { margin-bottom: 0; }
    .field-note {
      border: 1px solid #99c0e0;
      border-left: 4px solid #2f7fc4;
      border-radius: 8px;
      background: #f1f8fe;
      padding: 12px 14px;
      margin: 12px 0;
      line-height: 1.7;
    }
    .field-note .fn-title {
      font-weight: 800;
      color: #1f5f96;
      display: flex;
      align-items: center;
      gap: 7px;
      margin-bottom: 6px;
    }
    .field-note .fn-title::before { content: "🏭"; }
    .field-note p { margin: 0 0 8px; }
    .field-note p:last-child { margin-bottom: 0; }
    .shape-story {
      display: grid;
      gap: 10px;
      margin: 12px 0;
    }
    .story-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .story-panel p {
      margin: 0;
    }
    .story-arrow {
      min-height: 40px;
      border: 1px dashed #d9a15a;
      border-radius: 8px;
      background: var(--soft-amber);
      color: #7b430c;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 8px;
      font-weight: 900;
    }
    .channel-stack {
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 1fr));
      gap: 12px;
    }
    .channel-plane,
    .pixel-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(54px, 1fr));
      gap: 7px;
    }
    .channel-plane {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 34px 10px 10px;
      position: relative;
    }
    .channel-plane span:not(.plane-title),
    .pixel-grid span,
    .token-rail span {
      min-height: 44px;
      border-radius: 7px;
      display: grid;
      place-items: center;
      text-align: center;
      font-weight: 900;
    }
    .channel-plane.warm {
      background: #fff6e8;
      border-color: #e8b66f;
    }
    .channel-plane.cool {
      background: #eaf2ff;
      border-color: #a9c2f6;
    }
    .channel-plane.warm span:not(.plane-title) {
      background: #f8d8aa;
      color: #73450d;
    }
    .channel-plane.cool span:not(.plane-title) {
      background: #c8d9ff;
      color: #1e3f88;
    }
    .plane-title {
      position: absolute;
      top: 9px;
      left: 10px;
      font-size: 13px;
      font-weight: 900;
      color: var(--muted);
    }
    .pixel-grid span {
      background: linear-gradient(135deg, #f8d8aa 0 48%, #c8d9ff 52% 100%);
      color: #182033;
      border: 1px solid #d2b98f;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .token-rail {
      display: grid;
      grid-template-columns: repeat(6, minmax(90px, 1fr));
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .token-rail span {
      background: var(--soft-green);
      border: 1px solid #a9dbc7;
      color: #123f31;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .token-rail b {
      display: block;
      color: var(--green);
      margin-bottom: 3px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .syntax-card {
      border: 1px solid #b8c8e8;
      border-radius: 8px;
      background: #f8fbff;
      padding: 14px;
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .syntax-card h4 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
      color: #1e3f88;
    }
    .syntax-card p {
      margin: 0;
    }
    .syntax-card pre {
      margin: 0;
      border: 1px solid #1f2937;
      border-radius: 8px;
      background: #0b1220;
      color: #e8f0ff;
      padding: 14px;
      overflow-x: auto;
      font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .syntax-card pre code {
      background: transparent;
      color: inherit;
      padding: 0;
      border-radius: 0;
    }
    .syntax-tip {
      border-left: 4px solid var(--blue);
      background: #eef4ff;
      border-radius: 6px;
      padding: 10px;
    }
    .concept-badges,
    .mini-pipeline,
    .scale-demo,
    .dtype-demo,
    .gate-demo,
    .param-demo,
    .rotate-demo,
    .attn-grid,
    .gqa-demo,
    .mlp-demo,
    .residual-demo,
    .mini-flow,
    .mini-table,
    .timeline,
    .curve-legend,
    .ratio-board,
    .clip-box,
    .pref-pair,
    .reward-gap,
    .matrix-flow,
    .softmax-row,
    .tile-grid,
    .online-book,
    .lookup,
    .branches,
    .gpu-grid,
    .flow,
    .rail,
    .split,
    .sum-flow,
    .lesson-section .formula {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .concept-badges span,
    .mini-pipeline span,
    .scale-demo span,
    .scale-demo b,
    .scale-demo strong,
    .dtype-demo span,
    .gate-demo span,
    .gate-demo strong,
    .param-demo span,
    .param-demo strong,
    .rotate-demo span,
    .rotate-demo em,
    .attn-grid span,
    .attn-grid strong,
    .gqa-demo span,
    .mlp-demo span,
    .mlp-demo strong,
    .residual-demo span,
    .mini-flow div,
    .mini-flow span,
    .mini-flow strong,
    .mini-table div,
    .timeline span,
    .curve-legend span,
    .ratio-board span,
    .ratio-board strong,
    .clip-box span,
    .pref-pair span,
    .pref-pair strong,
    .reward-gap span,
    .reward-gap strong,
    .matrix-flow span,
    .matrix-flow strong,
    .softmax-row span,
    .softmax-row strong,
    .tile-grid span,
    .tile-grid strong,
    .online-book span,
    .online-book strong,
    .lookup span,
    .lookup b,
    .lookup em,
    .branches span,
    .branches b,
    .branches strong,
    .gpu-grid section,
    .gpu-grid footer,
    .flow span,
    .flow b,
    .flow strong,
    .rail span,
    .rail em,
    .split span,
    .split b,
    .split strong,
    .sum-flow b,
    .sum-flow strong,
    .sum-flow em,
    .lesson-section .formula b,
    .lesson-section .formula span {
      min-height: 48px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      place-items: center;
      text-align: center;
      line-height: 1.45;
    }
    .mini-flow b,
    .mini-flow strong,
    .gate-demo strong,
    .param-demo strong,
    .mlp-demo strong,
    .pref-pair strong,
    .reward-gap strong,
    .tile-grid strong,
    .online-book strong,
    .branches strong,
    .flow strong,
    .split strong,
    .sum-flow strong {
      border-color: #99d6bf;
      background: var(--soft-green);
      color: #123f31;
    }
    .timeline span,
    .curve-legend span,
    .rail span {
      border-color: #b9cdfb;
      background: var(--soft-blue);
      color: #1d4ed8;
      font-weight: 800;
    }
    .freq-table {
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .freq-table th,
    .freq-table td {
      border: 1px solid var(--line);
      padding: 10px;
      text-align: center;
    }
    .freq-table th {
      background: var(--soft-blue);
      color: #1d4ed8;
    }
    .shape-lesson,
    .grad-lesson {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 8px;
      align-items: stretch;
      margin: 12px 0;
    }
    .shape-card,
    .grad-lesson span {
      min-height: 88px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      place-items: center;
      text-align: center;
      gap: 4px;
    }
    .grad-lesson.compact span {
      min-height: 58px;
      font-size: 13px;
    }
    .shape-card strong,
    .grad-lesson strong { font-size: 18px; color: var(--blue); }
    .shape-card small { color: var(--muted); line-height: 1.35; }
    .shape-op {
      display: grid;
      place-items: center;
      color: var(--amber);
      font-weight: 900;
      white-space: nowrap;
    }
    .lookup-board {
      display: grid;
      grid-template-columns: minmax(150px, 0.8fr) minmax(220px, 1.2fr) minmax(180px, 1fr);
      gap: 10px;
      margin: 12px 0;
      align-items: stretch;
    }
    .ids,
    .table {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    .ids { display: grid; place-items: center; text-align: center; gap: 8px; }
    .id-token {
      min-width: 46px;
      min-height: 38px;
      border: 1px solid #b9cdfb;
      border-radius: 8px;
      background: var(--soft-blue);
      color: #1d4ed8;
      display: grid;
      place-items: center;
      font-weight: 900;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .table {
      display: grid;
      grid-template-columns: 70px minmax(0, 1fr);
      gap: 6px;
      align-content: center;
    }
    .table span {
      border-radius: 7px;
      background: #f5f7fa;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .table .hit {
      background: var(--soft-green);
      color: #123f31;
      font-weight: 900;
      box-shadow: inset 0 0 0 1px #a9dbc7;
    }
    .checkpoint-option,
    .homework-item {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin-top: 10px;
      cursor: pointer;
    }
    .checkpoint-option.correct,
    .homework-item.done { border-color: #99d6bf; background: var(--soft-green); }
    .checkpoint-option.wrong { border-color: #e5a6b6; background: var(--soft-rose); }
    .feedback { min-height: 26px; margin-top: 12px; font-weight: 900; }
    .ok { color: var(--green); }
    .warn { color: var(--rose); }
    .complete {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      border: 1px solid #9bd6bf;
      background: var(--soft-green);
      border-radius: 8px;
      padding: 14px;
      margin-top: 16px;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #eef2f7;
      border-radius: 5px;
      padding: 2px 5px;
    }
    @media (max-width: 920px) {
      .hero,
      .channel-stack,
      .shape-lesson,
      .grad-lesson,
      .lookup-board { grid-template-columns: 1fr; }
      .shape-op { min-height: 34px; }
    }
    @media (max-width: 640px) {
      .shell { width: min(100% - 22px, 1120px); }
      .top { display: grid; }
      .map-steps { grid-template-columns: 1fr; }
    }
  </style>${lessonStyles ? `
  <style>
${lessonStyles}
  </style>` : ""}
</head>
<body>
  <main class="shell">
    <div class="top">
      <a class="ghost" href="../index.html">返回关卡地图</a>
      <div class="hud">
        ${predictTotal ? `<span class="pill" id="predict-count">预判 0 / ${predictTotal}</span>` : ""}
        <span class="pill" id="status">闯关题：0 / ${level.lessons.length}</span>
        <span class="pill" id="xp">XP 0 / ${level.lessons.length * 100}</span>
      </div>
    </div>

    <section class="hero">
      <div class="card hero-main">
        <p class="eyebrow">Level ${level.id} | 零基础导学关卡</p>
        <h1>${esc(level.title)}</h1>
        <p class="lead">${predictTotal
          ? `每个 Mission 先<strong>猜一猜</strong>（押个假设解锁讲解）→ 看<strong>图解原理</strong> → 做<strong>巩固题</strong> → 回 notebook 写代码。带着问题学，比直接读记得牢。`
          : `先用小例子把必要知识学会，再用 ${checkpointLabel}闯关题检查理解，最后回到 notebook 写真正的 PyTorch 作业。`}</p>
        <div class="meta">
          <span class="pill">${esc(level.file)}</span>
          ${level.tags.map((tag) => `<span class="pill">${esc(tag)}</span>`).join("")}
        </div>
        <div class="xp-wrap" aria-label="本关经验条"><div class="xp-bar" id="xp-bar"></div></div>
      </div>
      <div class="card hero-map">
        <div class="map-steps">
          ${predictTotal
            ? `<div class="map-step">先猜一猜</div>
          <div class="map-step">图解原理</div>
          <div class="map-step">巩固 + 作业</div>`
            : `<div class="map-step">概念输入</div>
          <div class="map-step">闯关检查</div>
          <div class="map-step">Notebook 作业</div>`}
        </div>
        <div class="formula">${esc(level.formula)}</div>
      </div>
    </section>

${level.notebookGuide || ""}

    <section class="lessons">
      ${lessons}
    </section>

    <section class="complete">
      <div>
        <strong id="complete-title">完成 ${checkpointLabel}闯关题后，本关即算完成；作业 checklist 用来辅助你回 notebook 练习。</strong>
        <div class="pill">进度只保存在当前浏览器 localStorage，分享 HTML 不会带走你的记录。</div>
      </div>
      <div>
        ${prevLink}
        ${nextLink}
      </div>
    </section>
  </main>

  <script>
    const levelId = "${level.id}";
    const checkpointData = ${JSON.stringify(checkpointData)};
    const predictData = ${JSON.stringify(predictData)};
    const predictTotal = ${predictTotal};
    const checkpointTotal = ${level.lessons.length};
    const homeworkTotal = ${homeworkCount};
    const statePrefix = Number(levelId) <= 11 ? "pytorch-level-" : "pytorch-v2-level-";
    const checkpointKey = statePrefix + levelId + "-checkpoints";
    const homeworkKey = statePrefix + levelId + "-homework";
    const predictKey = statePrefix + levelId + "-predicts";
    const completeKey = "pytorch-levels-complete-v2";
    if (localStorage.getItem(completeKey) === null) {
      const legacyCompleted = JSON.parse(localStorage.getItem("pytorch-levels-complete") || "[]");
      localStorage.setItem(completeKey, JSON.stringify(legacyCompleted.filter((id) => Number(id) <= 11)));
    }
    const checkpointsDone = new Set(JSON.parse(localStorage.getItem(checkpointKey) || "[]"));
    const homeworkDone = new Set(JSON.parse(localStorage.getItem(homeworkKey) || "[]"));
    const predictsMade = new Map(JSON.parse(localStorage.getItem(predictKey) || "[]"));

    function save() {
      localStorage.setItem(checkpointKey, JSON.stringify([...checkpointsDone]));
      localStorage.setItem(homeworkKey, JSON.stringify([...homeworkDone]));
      localStorage.setItem(predictKey, JSON.stringify([...predictsMade]));
    }

    function renderPredict(lessonId) {
      const picked = predictsMade.get(lessonId);
      const gate = document.querySelector('[data-gated="' + lessonId + '"]');
      const feedback = document.querySelector('[data-predict-feedback="' + lessonId + '"]');
      if (picked === undefined) {
        if (gate) gate.classList.add("locked");
        return;
      }
      document.querySelectorAll('input[name="predict-' + lessonId + '"]').forEach((input) => {
        input.closest(".predict-option").classList.toggle("picked", Number(input.value) === picked);
        if (Number(input.value) === picked) input.checked = true;
      });
      if (gate) gate.classList.remove("locked");
      if (feedback) {
        const hit = picked === predictData[lessonId].answer;
        const badge = hit
          ? '<span class="predict-badge hit">🎯 预判命中</span>'
          : '<span class="predict-badge miss">🔓 已解锁 · 猜错正是学习的开始</span>';
        const rawNote = predictData[lessonId].revealNote;
        const note = hit
          ? rawNote
          : '正确思路：' + rawNote.replace(/^对[。！!，,:：]?\\s*/, '');
        feedback.innerHTML = badge + '<div style="margin-top:8px;font-weight:600;color:#4a3a7a">' + note + '</div>';
        feedback.className = "predict-feedback" + (hit ? " hit" : "");
      }
    }

    function refresh() {
      document.querySelectorAll("[data-homework]").forEach((input) => {
        const done = homeworkDone.has(input.dataset.homework);
        input.checked = done;
        input.closest(".homework-item").classList.toggle("done", done);
      });

      Object.keys(predictData).forEach(renderPredict);
      if (predictTotal) {
        const pc = document.querySelector("#predict-count");
        if (pc) pc.textContent = "预判 " + predictsMade.size + " / " + predictTotal;
      }

      for (const lessonId of checkpointsDone) {
        const answer = checkpointData[lessonId].answer;
        const input = document.querySelector('input[name="checkpoint-' + lessonId + '"][value="' + answer + '"]');
        const feedback = document.querySelector('[data-feedback="' + lessonId + '"]');
        if (input) {
          input.checked = true;
          input.closest(".checkpoint-option").classList.add("correct");
        }
        if (feedback) {
          feedback.textContent = "已通过：" + checkpointData[lessonId].explain;
          feedback.className = "feedback ok";
        }
      }

      const done = checkpointsDone.size;
      document.querySelector("#status").textContent = "闯关题：" + done + " / " + checkpointTotal;
      document.querySelector("#xp").textContent = "XP " + (done * 100) + " / " + (checkpointTotal * 100);
      document.querySelector("#xp-bar").style.width = (done / checkpointTotal * 100) + "%";

      if (done === checkpointTotal) {
        const completed = new Set(JSON.parse(localStorage.getItem(completeKey) || "[]"));
        completed.add(levelId);
        localStorage.setItem(completeKey, JSON.stringify([...completed]));
        document.querySelector("#complete-title").textContent = "闯关题已完成。现在回 notebook 写作业，做最后的举一反三。";
      }
    }

    document.querySelector(".lessons").addEventListener("change", (event) => {
      if (event.target.matches('input[name^="predict-"]')) {
        const lessonId = event.target.name.replace("predict-", "");
        predictsMade.set(lessonId, Number(event.target.value));
        save();
        renderPredict(lessonId);
        if (predictTotal) {
          const pc = document.querySelector("#predict-count");
          if (pc) pc.textContent = "预判 " + predictsMade.size + " / " + predictTotal;
        }
        return;
      }

      if (event.target.matches('input[name^="checkpoint-"]')) {
        const lessonId = event.target.name.replace("checkpoint-", "");
        const selected = Number(event.target.value);
        const data = checkpointData[lessonId];
        const feedback = document.querySelector('[data-feedback="' + lessonId + '"]');
        document.querySelectorAll('input[name="' + event.target.name + '"]').forEach((input) => {
          input.closest(".checkpoint-option").classList.remove("correct", "wrong");
        });
        if (selected === data.answer) {
          event.target.closest(".checkpoint-option").classList.add("correct");
          feedback.textContent = "答对了：" + data.explain;
          feedback.className = "feedback ok";
          checkpointsDone.add(lessonId);
        } else {
          event.target.closest(".checkpoint-option").classList.add("wrong");
          feedback.textContent = "再想一下。先看本节的图解原理，再回来看这道题。";
          feedback.className = "feedback warn";
        }
        save();
        refresh();
      }

      if (event.target.matches("[data-homework]")) {
        if (event.target.checked) {
          homeworkDone.add(event.target.dataset.homework);
        } else {
          homeworkDone.delete(event.target.dataset.homework);
        }
        save();
        refresh();
      }
    });

    refresh();
  </script>
</body>
</html>
`;
}

function levelPage(level, prev, next) {
  const customLessons = level.lessons || lessonOverrides[level.id];
  const lessons = enhanceFoundationLessons(level.id, customLessons || genericLessons(level));
  const notebookGuide = customLessons ? "" : notebookGuideHtml(level);
  return lessonLevelPage({ ...level, lessons, notebookGuide }, prev, next);
}

function indexPage() {
  const cards = levels.map((level) => `
        <article class="level" data-category="${level.category}" data-id="${level.id}">
          <div class="level-head">
            <span class="badge">L${level.id}</span>
            <span class="status" data-status="${level.id}">待挑战</span>
          </div>
          <h2>${esc(level.title)}</h2>
          <p>${esc(level.summary)}</p>
          <div class="chips">${level.tags.map((tag) => `<span class="chip">${esc(tag)}</span>`).join("")}</div>
          <a class="start" href="notes/${slug(level)}">进入关卡</a>
        </article>`).join("");

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PyTorch Algorithms 闯关地图</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --paper: #ffffff;
      --ink: #172033;
      --muted: #657184;
      --line: #dde3ea;
      --blue: #2563eb;
      --green: #16835f;
      --soft-blue: #e9f0ff;
      --soft-green: #e8f6ef;
      --shadow: 0 16px 40px rgba(23, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(37, 99, 235, 0.05) 1px, transparent 1px),
        linear-gradient(rgba(22, 131, 95, 0.05) 1px, transparent 1px),
        var(--bg);
      background-size: 28px 28px;
    }
    a { color: inherit; text-decoration: none; }
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 44px; }
    .hero {
      min-height: 270px;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr);
      gap: 24px;
      align-items: center;
      padding: 24px 0 18px;
    }
    h1 { margin: 0 0 12px; font-size: clamp(32px, 5vw, 58px); line-height: 1.02; letter-spacing: 0; }
    .lead { max-width: 760px; color: var(--muted); font-size: 18px; line-height: 1.65; margin: 0; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 22px; max-width: 590px; }
    .stat { border: 1px solid var(--line); background: rgba(255,255,255,0.84); border-radius: 8px; padding: 12px; }
    .stat strong { display: block; font-size: 24px; line-height: 1.1; }
    .stat span { color: var(--muted); font-size: 13px; }
    .map-art {
      min-height: 270px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
      padding: 20px;
      display: grid;
      gap: 12px;
      align-content: center;
    }
    .flow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .flow div {
      min-height: 84px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 10px;
      font-weight: 900;
    }
    .flow div:nth-child(1) { background: var(--soft-blue); color: var(--blue); }
    .flow div:nth-child(2) { background: var(--soft-green); color: var(--green); }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin: 16px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 14px 0;
    }
    .filters { display: flex; gap: 8px; flex-wrap: wrap; }
    button {
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 8px;
      padding: 9px 12px;
      font: inherit;
      cursor: pointer;
    }
    button:hover, button.active { border-color: var(--blue); color: var(--blue); background: var(--soft-blue); }
    .progress { color: var(--muted); font-size: 14px; }
    .levels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .level {
      min-height: 250px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      display: grid;
      grid-template-rows: auto auto 1fr auto auto;
      gap: 10px;
    }
    .level.complete { border-color: #9bd6bf; }
    .level-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 42px;
      height: 30px;
      padding: 0 9px;
      border-radius: 999px;
      background: var(--soft-blue);
      color: var(--blue);
      font-weight: 900;
      font-size: 13px;
    }
    .status { color: var(--muted); font-size: 13px; }
    .level h2 { margin: 0; font-size: 18px; line-height: 1.3; letter-spacing: 0; }
    .level p { margin: 0; color: var(--muted); line-height: 1.55; font-size: 14px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; align-content: start; }
    .chip {
      border: 1px solid var(--line);
      background: #fbfcfd;
      border-radius: 999px;
      padding: 5px 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .start {
      display: inline-flex;
      justify-content: center;
      align-items: center;
      border: 1px solid var(--blue);
      color: var(--blue);
      background: var(--soft-blue);
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 900;
    }
    @media (max-width: 980px) { .hero, .levels { grid-template-columns: 1fr; } }
    @media (max-width: 620px) {
      .shell { width: min(100% - 22px, 1180px); }
      .stats, .flow { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <h1>PyTorch Algorithms 闯关地图</h1>
        <p class="lead">每个 notebook 都有一个可分享的 HTML 导学关卡。每关固定包含三件事：课程输入、闯关检查、Notebook 作业。HTML 负责把概念和关键语法讲清楚，notebook 负责最后的代码刷题检验。</p>
        <div class="stats" aria-label="学习进度">
          <div class="stat"><strong id="done-count">0</strong><span>已通关</span></div>
          <div class="stat"><strong>${levels.length}</strong><span>Notebook 关卡</span></div>
          <div class="stat"><strong>3</strong><span>每关学习模块</span></div>
        </div>
      </div>
      <div class="map-art" aria-label="学习流程">
        <div class="flow">
          <div>知识点<br>图解直觉</div>
          <div>闯关答题<br>即时反馈</div>
          <div>Notebook 作业<br>刷题检验</div>
        </div>
        <p class="lead">建议顺序：先看 HTML 建立直觉，完成少量闯关题，再回到 notebook 写代码作业。</p>
      </div>
    </section>

    <section class="toolbar">
      <div class="filters" aria-label="关卡过滤器">
        <button class="active" data-filter="all">全部</button>
        <button data-filter="foundation">基础</button>
        <button data-filter="architecture">结构</button>
        <button data-filter="training">训练</button>
        <button data-filter="inference">推理</button>
      </div>
      <div class="progress" id="progress-text">读取本地通关记录中</div>
    </section>

    <section class="levels" id="levels" aria-label="Notebook 关卡列表">
${cards}
    </section>
  </main>

  <script>
    const completeKey = "pytorch-levels-complete-v2";
    if (localStorage.getItem(completeKey) === null) {
      const legacyCompleted = JSON.parse(localStorage.getItem("pytorch-levels-complete") || "[]");
      localStorage.setItem(completeKey, JSON.stringify(legacyCompleted.filter((id) => Number(id) <= 11)));
    }
    const completed = new Set(JSON.parse(localStorage.getItem(completeKey) || "[]"));
    const cards = [...document.querySelectorAll(".level")];
    const doneCount = document.querySelector("#done-count");
    const progressText = document.querySelector("#progress-text");

    function refresh() {
      cards.forEach((card) => {
        const done = completed.has(card.dataset.id);
        card.classList.toggle("complete", done);
        card.querySelector("[data-status]").textContent = done ? "已通关" : "待挑战";
      });
      doneCount.textContent = completed.size;
      progressText.textContent = "本地记录：" + completed.size + " / " + cards.length + " 关已通关";
    }

    document.querySelectorAll("[data-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        const filter = button.dataset.filter;
        cards.forEach((card) => {
          card.style.display = filter === "all" || card.dataset.category === filter ? "" : "none";
        });
      });
    });

    refresh();
  </script>
</body>
</html>
`;
}

levels.forEach((level, index) => {
  const filePath = path.join(notesDir, slug(level));
  fs.writeFileSync(filePath, levelPage(level, levels[index - 1], levels[index + 1]));
});

const expectedPages = new Set(levels.map(slug));
for (const file of fs.readdirSync(notesDir)) {
  if (file.endsWith(".html") && !expectedPages.has(file)) {
    fs.unlinkSync(path.join(notesDir, file));
  }
}

fs.writeFileSync(path.join(root, "index.html"), indexPage());

console.log(`Generated ${levels.length} level pages and index.html`);
