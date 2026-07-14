// Add the inquiry-based teaching loop used by Level 05 without replacing the
// existing Level 00/01 explanations. Every block below is appended to the
// original lesson so earlier diagrams, examples and homework remain intact.

const enhancements = {
  "00": {
    tensor: {
      predict: {
        hook: "把 [B,C,H,W] 变成 [B,H×W,C] 时，最危险的不是 API 名字，而是元素顺序。reshape 只会重新分组，不会自动把 C 挪到最后。",
        question: "先判断：为什么通常要先 permute，再 reshape？",
        options: [
          "先把轴排成 [B,H,W,C]，再合并 H/W，才能保证每个 token 的 C 个特征仍在一起",
          "因为 reshape 只能处理二维张量",
          "因为 permute 会自动复制数据"
        ],
        answer: 0,
        revealNote: "对。先确定轴语义，再合并连续的空间轴；只看元素总数相等，可能得到 shape 对、语义错的结果。"
      },
      exampleHtml: `
        <div class="shape-story">
          <div class="story-panel">
            <strong>4. 从玩具 shape 迁移到 Notebook TODO</strong>
            <table class="freq-table">
              <tr><th style="width:150px">玩具例子</th><th>Notebook 变量</th><th>检查点</th></tr>
              <tr><td><code>[b,t,f]</code></td><td><code>x: [B,C,H,W]</code></td><td>先说清每个轴代表什么</td></tr>
              <tr><td><code>permute</code></td><td><code>[B,H,W,C]</code></td><td>C 必须来到最后</td></tr>
              <tr><td><code>reshape</code></td><td><code>[B,H*W,C]</code></td><td>只合并 H 和 W</td></tr>
            </table>
          </div>
          <div class="story-arrow">测试失败时，不要盲换 API</div>
          <div class="story-panel">
            <strong>5. 三类常见错法</strong>
            <table class="freq-table">
              <tr><th style="width:180px">现象</th><th>优先检查</th></tr>
              <tr><td>shape 变成 [B,C,H×W]</td><td style="text-align:left">C 没有移动到最后，permute 顺序不对。</td></tr>
              <tr><td>shape 正确但 allclose 失败</td><td style="text-align:left">元素顺序被打乱；重新画出第一个 token 应来自哪个像素。</td></tr>
              <tr><td>reshape 报不连续</td><td style="text-align:left">理解 reshape/contiguous 的关系，或用 einops 明确表达轴语义。</td></tr>
            </table>
          </div>
          <div class="field-note">
            <div class="fn-title">行业视角：shape 语义是读模型代码的第一道门</div>
            <p>ViT patch、图像 token、Attention head 拆分和 KV Cache 都在做同一类事情：重新组织轴，但不能破坏“哪个数属于哪个 token”的语义。</p>
          </div>
        </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>举一反三：每次变形前写一行 shape 合同</h4><pre><code># 输入合同: x [B,C,H,W]
# 目标合同: tokens [B,H*W,C]
# 数量守恒: C*H*W == (H*W)*C

tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
assert tokens.shape == (B, H * W, C)</code></pre><p class="syntax-tip">shape 合同只能保证外形；最后仍要用一个小张量或 allclose 检查元素顺序。</p></div>`,
      homework: ["用 B=1、C=2、H=2、W=2 的连续整数张量手算第一个 token，确认不是只碰巧通过 shape 测试。"]
    },
    embedding: {
      predict: {
        hook: "token id 看起来是整数，但 9 并不比 2 更‘大’或更‘重要’。它只是词表中的行号。",
        question: "先判断：如果把 token id 直接当连续数值送进 Linear，最根本的问题是什么？",
        options: [
          "模型会误以为 id 之间存在大小和距离关系，而 id 本来只是离散地址",
          "Linear 不支持整数以外的数据",
          "Embedding 会自动删除重复 token"
        ],
        answer: 0,
        revealNote: "对。Embedding 把离散地址变成可学习向量，让相似性由训练决定，而不是由 id 数字大小决定。"
      },
      exampleHtml: `
        <div class="shape-story">
          <div class="story-panel">
            <strong>4. 把查表动作映射回 Notebook</strong>
            <table class="freq-table">
              <tr><th style="width:160px">对象</th><th>含义</th><th>shape</th></tr>
              <tr><td><code>input_ids</code></td><td style="text-align:left">每个位置的词表行号</td><td>[B,S]</td></tr>
              <tr><td><code>emb_layer.weight</code></td><td style="text-align:left">整本可学习词典</td><td>[V,H]</td></tr>
              <tr><td><code>output</code></td><td style="text-align:left">每个 id 查到的一行向量</td><td>[B,S,H]</td></tr>
            </table>
          </div>
          <div class="story-arrow">错误通常发生在索引，不在矩阵乘法</div>
          <div class="story-panel">
            <strong>5. 测试排查表</strong>
            <table class="freq-table">
              <tr><th style="width:170px">报错/现象</th><th>优先检查</th></tr>
              <tr><td>indices must be Long</td><td style="text-align:left"><code>input_ids.dtype</code> 应为整型索引。</td></tr>
              <tr><td>index out of range</td><td style="text-align:left">所有 id 必须满足 <code>0 ≤ id &lt; vocab_size</code>。</td></tr>
              <tr><td>shape 少了 hidden 维</td><td style="text-align:left">输出应在 input_ids 后追加 embedding_dim。</td></tr>
            </table>
          </div>
          <div class="field-note">
            <div class="fn-title">行业视角：Embedding 往往是大模型最大的单块参数之一</div>
            <p>词表 V 很大时，[V,H] 权重会占据显著显存；词表扩容、权重共享和并行切分都从这张查找表出发。</p>
          </div>
        </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>举一反三：用高级索引验证 nn.Embedding</h4><pre><code>official = emb_layer(input_ids)
manual = emb_layer.weight[input_ids]

assert official.shape == (*input_ids.shape, emb_layer.embedding_dim)
assert torch.allclose(official, manual)</code></pre><p class="syntax-tip">这组对照把“Embedding 是查表”从比喻变成了可运行证据。</p></div>`,
      homework: ["额外检查 input_ids 的最小值、最大值和 dtype，解释三者为什么共同决定查表是否合法。"]
    },
    backward: {
      predict: {
        hook: "Linear+ReLU 的 backward 必须逆着 forward 走。forward 最后经过 ReLU，所以传回来的梯度第一站不是 Linear，而是 ReLU mask。",
        question: "先判断：grad_output 回来后，应该先计算什么？",
        options: [
          "先乘 (z>0) 得到 grad_z，再由 grad_z 推出 x、weight、bias 的梯度",
          "先更新 weight，再计算 mask",
          "直接把 grad_output 原样返回给所有输入"
        ],
        answer: 0,
        revealNote: "对。链式法则就是按前向的逆序逐门通过；ReLU 先决定哪些位置的梯度还能继续流动。"
      },
      exampleHtml: `
        <div class="shape-story">
          <div class="story-panel">
            <strong>4. 反向 shape 账本</strong>
            <table class="freq-table">
              <tr><th>梯度</th><th>公式</th><th>必须匹配</th></tr>
              <tr><td><code>grad_x</code></td><td><code>grad_z @ weight</code></td><td><code>x.shape</code></td></tr>
              <tr><td><code>grad_weight</code></td><td><code>grad_z.T @ x</code></td><td><code>weight.shape</code></td></tr>
              <tr><td><code>grad_bias</code></td><td><code>grad_z.sum(dim=0)</code></td><td><code>bias.shape</code></td></tr>
            </table>
          </div>
          <div class="story-arrow">数值对，不代表计算图一定对</div>
          <div class="story-panel">
            <strong>5. 测试失败时按顺序定位</strong>
            <table class="freq-table">
              <tr><th style="width:180px">现象</th><th>优先检查</th></tr>
              <tr><td>负 z 位置仍有梯度</td><td style="text-align:left">是否先乘了 ReLU mask。</td></tr>
              <tr><td>weight 梯度转置</td><td style="text-align:left">矩阵乘法顺序是否让结果 shape 等于 weight。</td></tr>
              <tr><td>bias 梯度多一维</td><td style="text-align:left">是否沿 batch/token 维求和，只保留输出特征维。</td></tr>
            </table>
          </div>
          <div class="field-note">
            <div class="fn-title">行业视角：手写 backward 是理解 fused kernel 的入口</div>
            <p>FlashAttention、融合激活和自定义 Triton 算子都需要明确“前向保存什么、反向重算什么、每个梯度是什么 shape”。这道小题就是最小版本。</p>
          </div>
        </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>举一反三：用 autograd 当裁判</h4><pre><code># custom 与 reference 使用同一份输入和参数
loss_custom.backward()
loss_reference.backward()

assert torch.allclose(custom_grad_x, reference_grad_x, atol=1e-5)
assert torch.allclose(custom_grad_w, reference_grad_w, atol=1e-5)</code></pre><p class="syntax-tip">不要只比 forward 输出；自定义 Function 的核心验收对象是每一个输入梯度。</p></div>`,
      homework: ["把 forward 顺序和 backward 顺序画成两条相反箭头，并逐项标出 ctx 需要保存的张量。"]
    }
  },
  "01": {
    "rmsnorm-scale": {
      predict: {
        hook: "RMSNorm 的名字已经给出线索：它使用 root mean square，而不是 LayerNorm 的 mean + variance。少做的那一步是什么？",
        question: "先判断：RMSNorm 与 LayerNorm 最关键的区别是什么？",
        options: [
          "RMSNorm 不减去均值，只按平方均值的平方根缩放",
          "RMSNorm 不包含任何可学习参数",
          "RMSNorm 只能处理二维输入"
        ],
        answer: 0,
        revealNote: "对。它保留方向，只把最后一维 hidden vector 的整体尺度拉回稳定范围，再乘可学习 weight。"
      },
      exampleHtml: `
        <div class="shape-story">
          <div class="story-panel">
            <strong>3. 三个 TODO 对应三层职责</strong>
            <table class="freq-table">
              <tr><th style="width:110px">任务</th><th>要完成什么</th><th>检查点</th></tr>
              <tr><td>TODO 1</td><td style="text-align:left">创建 <code>[hidden_size]</code> 的可学习缩放参数</td><td><code>nn.Parameter</code></td></tr>
              <tr><td>TODO 2</td><td style="text-align:left">沿最后一维计算 mean square 与 rsqrt</td><td><code>keepdim=True</code></td></tr>
              <tr><td>TODO 3</td><td style="text-align:left">归一化结果乘 weight，再恢复输入 dtype</td><td>shape/dtype 不变</td></tr>
            </table>
          </div>
          <div class="story-arrow">公式短，广播最容易出错</div>
          <div class="story-panel">
            <strong>4. 常见错法</strong>
            <table class="freq-table">
              <tr><th style="width:170px">写法</th><th>为什么不对</th></tr>
              <tr><td><code>mean()</code> 不写 dim</td><td style="text-align:left">会把 batch、sequence 一起平均，每个 token 不再独立归一化。</td></tr>
              <tr><td><code>keepdim=False</code></td><td style="text-align:left">统计量 shape 难以和原输入按最后一维广播。</td></tr>
              <tr><td>忘记 eps</td><td style="text-align:left">全零或极小向量可能产生 inf/NaN。</td></tr>
            </table>
          </div>
          <div class="field-note">
            <div class="fn-title">行业视角：RMSNorm 已成为现代 Decoder 的常见默认件</div>
            <p>LLaMA 等模型选择 RMSNorm，是因为它保留核心尺度稳定作用，同时比完整 LayerNorm 更简洁。工程实现仍必须重视精度、广播和 dtype。</p>
          </div>
        </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>举一反三：先写 shape 注释，再写公式</h4><pre><code># x: [B,S,H]
variance = x.float().pow(2).mean(dim=-1, keepdim=True)  # [B,S,1]
normalized = x.float() * torch.rsqrt(variance + eps)    # [B,S,H]
output = normalized * weight                            # weight: [H]</code></pre><p class="syntax-tip">广播链路是 [B,S,1] → [B,S,H]，再让 [H] 对齐最后一维。</p></div>`,
      homework: ["用全零输入和放大 1000 倍的输入各测一次，观察 eps 与归一化尺度分别解决什么问题。"]
    },
    "rmsnorm-precision": {
      predict: {
        hook: "半精度输入本身可能没溢出，但平方会迅速放大数值。归一化统计量正好包含 x²。",
        question: "先判断：为什么应在平方之前转成 FP32，而不是平方之后再转？",
        options: [
          "如果 FP16 的平方已经溢出成 inf，之后转 FP32 也无法恢复原值",
          "因为 FP32 会自动减少 hidden_size",
          "因为 weight 只能和 FP32 相乘"
        ],
        answer: 0,
        revealNote: "对。升精度必须发生在危险运算之前；先溢出再转换，只是把 FP16 的 inf 搬进 FP32。"
      },
      exampleHtml: `
        <div class="shape-story">
          <div class="story-panel">
            <strong>3. dtype 数据流：只在危险区间升精度</strong>
            <div class="mini-pipeline"><span>输入 FP16/BF16</span><span>统计量 FP32</span><span>缩放 FP32</span><span>输出回原 dtype</span></div>
            <p>目标不是把整层永久改成 FP32，而是让平方、均值和 rsqrt 这段数值敏感计算更稳。</p>
          </div>
          <div class="story-arrow">测试要同时看数值、shape 和 dtype</div>
          <div class="story-panel">
            <strong>4. 三类测试信号</strong>
            <table class="freq-table">
              <tr><th style="width:150px">测试</th><th>它在防什么</th></tr>
              <tr><td><code>isfinite</code></td><td style="text-align:left">平方溢出或除零产生 inf/NaN。</td></tr>
              <tr><td><code>allclose</code></td><td style="text-align:left">公式、eps 或广播方向写错。</td></tr>
              <tr><td><code>out.dtype==x.dtype</code></td><td style="text-align:left">忘记把结果还给后续混合精度计算图。</td></tr>
            </table>
          </div>
          <div class="field-note">
            <div class="fn-title">行业视角：混合精度不是“全都用半精度”</div>
            <p>工业训练会把矩阵乘法放在 BF16/FP16，把归一化统计、loss reduction 等敏感环节保留更高精度。关键是知道哪一步会放大误差。</p>
          </div>
        </div>`,
      syntaxHtml: `<div class="syntax-card"><h4>举一反三：把 dtype 恢复写成显式合同</h4><pre><code>input_dtype = x.dtype
x32 = x.float()
variance = x32.pow(2).mean(dim=-1, keepdim=True)
normalized = x32 * torch.rsqrt(variance + eps)
output = (normalized * weight.float()).to(input_dtype)

assert output.dtype == input_dtype</code></pre></div>`,
      homework: ["比较 x.float().pow(2) 与 x.pow(2).float() 在大数 FP16 输入上的差异，并解释为什么后者来不及补救。"]
    }
  }
};

module.exports = function enhanceFoundationLessons(levelId, lessons) {
  const levelEnhancements = enhancements[levelId];
  if (!levelEnhancements) return lessons;

  return lessons.map((lesson) => {
    const extra = levelEnhancements[lesson.id];
    if (!extra) return lesson;
    return {
      ...lesson,
      predict: extra.predict || lesson.predict,
      exampleHtml: `${lesson.exampleHtml || ""}${extra.exampleHtml || ""}`.trim(),
      syntaxHtml: `${lesson.syntaxHtml || ""}${extra.syntaxHtml || ""}`.trim(),
      homework: [...(lesson.homework || []), ...(extra.homework || [])]
    };
  });
};
