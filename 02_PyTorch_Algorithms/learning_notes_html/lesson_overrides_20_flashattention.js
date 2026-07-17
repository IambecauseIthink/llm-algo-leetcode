const esc = (value) => String(value)
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;");

const code = (title, lines) => `<div class="syntax-card"><h4>语法热身：${esc(title)}</h4><pre><code>${lines.map(esc).join("\n")}</code></pre></div>`;

const checkpoint = (question, options, answer, explain) => ({ question, options, answer, explain });

const lesson = ({ id, title, todo, prerequisite, intuition, exampleHtml, syntaxHtml, styles, predict, checkpoint: lessonCheckpoint, homework }) => ({
  id,
  title,
  todo,
  prerequisite,
  intuition,
  exampleHtml,
  syntaxHtml,
  styles,
  predict,
  checkpoint: lessonCheckpoint,
  homework
});

const flashStyles = `
    .flash-course {
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }
    .flash-course h4,
    .flash-practice h4 {
      margin: 0;
      font-size: 17px;
      color: #182033;
    }
    .flash-course p,
    .flash-practice p { margin: 0; }
    .flash-roadmap {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      border: 1px solid #cdd9ef;
      border-radius: 8px;
      overflow: hidden;
      background: #f7faff;
    }
    .flash-roadmap span {
      min-height: 82px;
      padding: 12px;
      display: grid;
      align-content: center;
      gap: 4px;
      border-right: 1px solid #cdd9ef;
      color: #42516b;
      line-height: 1.45;
    }
    .flash-roadmap span:last-child { border-right: 0; }
    .flash-roadmap b {
      color: #1d4ed8;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .flash-split,
    .flash-practice {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }
    .flash-panel,
    .flash-example,
    .flash-map {
      padding: 14px;
      border: 1px solid #dce3eb;
      border-radius: 8px;
      background: #fff;
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .flash-panel.standard { border-top: 4px solid #657184; }
    .flash-panel.blocked { border-top: 4px solid #16835f; background: #f5fbf8; }
    .flash-panel.warn { border-top: 4px solid #d97706; background: #fff9ef; }
    .flash-line {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 8px;
    }
    .flash-line span,
    .flash-line strong {
      min-height: 58px;
      padding: 9px;
      border: 1px solid #dce3eb;
      border-radius: 6px;
      background: #fff;
      display: grid;
      place-items: center;
      text-align: center;
      line-height: 1.4;
    }
    .flash-line strong {
      border-color: #99d6bf;
      background: #e9f7f0;
      color: #125b45;
    }
    .flash-contract {
      display: grid;
      grid-template-columns: minmax(125px, 0.34fr) minmax(0, 1fr);
      border: 1px solid #dce3eb;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .flash-contract span,
    .flash-contract strong {
      min-height: 48px;
      padding: 10px 12px;
      display: grid;
      align-items: center;
      border-bottom: 1px solid #dce3eb;
      line-height: 1.45;
    }
    .flash-contract span {
      border-right: 1px solid #dce3eb;
      background: #f7f9fc;
      color: #526077;
      font-weight: 800;
    }
    .flash-contract strong {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }
    .flash-contract span:nth-last-child(-n+2),
    .flash-contract strong:nth-last-child(-n+2) { border-bottom: 0; }
    .flash-callout {
      padding: 12px 14px;
      border-left: 4px solid #d97706;
      border-radius: 6px;
      background: #fff6e7;
      color: #74400d;
      line-height: 1.6;
      font-weight: 750;
    }
    .flash-shapes {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      background: #fff;
      border: 1px solid #dce3eb;
    }
    .flash-shapes th,
    .flash-shapes td {
      padding: 10px;
      border: 1px solid #dce3eb;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    .flash-shapes th {
      background: #eaf1ff;
      color: #1d4ed8;
    }
    .flash-shapes code { white-space: normal; }
    .flash-formula {
      display: grid;
      gap: 8px;
    }
    .flash-formula div {
      display: grid;
      grid-template-columns: 36px minmax(180px, 0.75fr) minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid #dce3eb;
      border-radius: 7px;
      background: #fff;
    }
    .flash-formula b {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: #2563eb;
      color: #fff;
    }
    .flash-formula code {
      color: #172033;
      font-weight: 800;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .flash-formula span { color: #5f6d82; }
    .flash-example { background: #f5fbf8; border-color: #a9d9c6; }
    .flash-map { background: #f7f9fc; }
    .flash-map ul { margin: 0; padding-left: 20px; }
    .flash-map li { margin: 6px 0; line-height: 1.5; }
    .flash-practice .syntax-card { margin-top: 0; }
    .flash-test-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .flash-test-grid span {
      min-height: 72px;
      padding: 10px;
      border: 1px solid #dce3eb;
      border-radius: 7px;
      background: #fff;
      display: grid;
      place-items: center;
      text-align: center;
      line-height: 1.4;
    }
    @media (max-width: 760px) {
      .flash-roadmap,
      .flash-split,
      .flash-practice,
      .flash-test-grid { grid-template-columns: 1fr; }
      .flash-roadmap span {
        border-right: 0;
        border-bottom: 1px solid #cdd9ef;
      }
      .flash-roadmap span:last-child { border-bottom: 0; }
      .flash-contract { grid-template-columns: 1fr; }
      .flash-contract span { border-right: 0; }
      .flash-contract span:nth-last-child(-n+2) { border-bottom: 1px solid #dce3eb; }
      .flash-formula div { grid-template-columns: 36px minmax(0, 1fr); }
      .flash-formula div span { grid-column: 2; }
      .flash-shapes { font-size: 13px; }
      .flash-shapes th,
      .flash-shapes td { padding: 7px; }
    }`;

module.exports = [
  lesson({
    id: "flash-state-and-tiles",
    title: "先读懂函数：分块不改答案，只改计算顺序",
    todo: "TODO 1 + 两层分块循环",
    prerequisite: [
      "本 Notebook 暂时没有 Batch 和 Head 维度，q、k、v 都是 [seq_len, dim]。先把二维版本学明白，再迁移到真实多头实现。",
      "Python 切片 start:start+block_size 即使越过末尾也不会报错，它会自然返回较短的最后一块。",
      "FlashAttention 仍然计算精确 Attention；这里的核心是逐块读取 K/V，并且不把完整 [seq_len, seq_len] 分数矩阵长期保存下来。",
      "对每一行 query，算法只需要持续记住三个状态：当前输出 out、见过的最大值 m、指数和 l。"
    ],
    intuition: "先把代码当成两层翻页：外层固定一小块 Q，内层把 K/V 一块块翻完。每翻一页，就把这一页的信息并入 m、l 和 out；翻完全部 K/V 后，这块 Q 的答案就完成了。",
    styles: flashStyles,
    exampleHtml: `<div class="flash-course">
        <div class="flash-roadmap">
          <span><b>1. 固定 Q 块</b>一次只处理几行 query</span>
          <span><b>2. 扫描 K/V 块</b>内层循环逐块读取</span>
          <span><b>3. 保存滚动状态</b>每行只保留 out、m、l</span>
        </div>

        <div class="flash-split">
          <section class="flash-panel standard">
            <h4>普通写法：先造出完整分数表</h4>
            <div class="flash-line">
              <span>Q [S,D]</span><span>K^T [D,S]</span><strong>Scores [S,S]</strong>
            </div>
            <p>序列变长时，S×S 的中间矩阵按平方增长。这里的模拟作业，就是为了理解如何避免长期保存它。</p>
          </section>
          <section class="flash-panel blocked">
            <h4>分块写法：固定 Q 块，逐块读 K/V</h4>
            <div class="flash-line">
              <span>Q_i [Bq,D]</span><span>K_j^T [D,Bk]</span><strong>S_ij [Bq,Bk]</strong>
            </div>
            <p>每次只出现一个小分数块。数学结果没有近似，变化的是计算顺序和中间数据的生存时间。</p>
          </section>
        </div>

        <div class="flash-contract">
          <span><code>out</code></span><strong>[seq_len, dim]，每个 query 当前已经归一化的输出</strong>
          <span><code>m</code></span><strong>[seq_len, 1]，每行目前见过的最大 score；初始值是 -inf</strong>
          <span><code>l</code></span><strong>[seq_len, 1]，以当前 m 为基准的指数和；初始值是 0</strong>
          <span>为什么是列向量</span><strong>[seq_len, 1] 可以沿最后一维广播到 score 块的每一列</strong>
        </div>

        <div class="flash-callout">初始化的含义比写法更重要：还没看任何 K 块，所以输出贡献和指数和都是 0；最大值设为 -inf，保证第一块中任何正常分数都能成为新最大值。</div>
      </div>`,
    syntaxHtml: `<div class="flash-practice">
        ${code("用另一个张量练初始化、切片和最后一块", [
          "rows, width = 5, 3",
          "x = torch.randn(rows, width)",
          "",
          "buffer = torch.zeros((rows, width), device=x.device, dtype=x.dtype)",
          "running_max = torch.full((rows, 1), -float('inf'), device=x.device, dtype=x.dtype)",
          "running_sum = torch.zeros((rows, 1), device=x.device, dtype=x.dtype)",
          "",
          "for start in range(0, rows, 3):",
          "    chunk = x[start:start + 3]",
          "    print(chunk.shape)  # [3, 3]，然后是 [2, 3]"
        ])}
        <div class="flash-map">
          <h4>从例子迁移到 TODO 1</h4>
          <ul>
            <li><code>buffer</code> 对应需要返回的输出状态，shape 要同时包含序列行和特征列。</li>
            <li><code>running_max</code> 与 <code>running_sum</code> 都是每个 query 一份统计量，所以第二维保留为 1。</li>
            <li><code>device</code> 和 <code>dtype</code> 跟随输入，可以避免 CPU/GPU 或精度不一致。</li>
            <li>不要手工补齐最后一块；Notebook 的第二个测试专门覆盖 seq_len 不能整除 block_size 的情况。</li>
          </ul>
        </div>
      </div>`,
    predict: {
      hook: "代码一开始还没有读过任何 score，却要先给每行的最大值 m 一个初值。",
      question: "m 为什么应该从负无穷开始，而不是从 0 开始？",
      options: [
        "这样第一块里的负分数也能正确成为最大值",
        "因为负无穷会让所有输出永远为 0",
        "因为 block_size 只能是负数"
      ],
      answer: 0,
      revealNote: "如果所有 score 都是负数，m=0 会引入一个从未出现过的假最大值；-inf 才代表“还没看过任何数据”。"
    },
    checkpoint: checkpoint(
      "seq_len=5、block_size=3 时，第二个切片 q[3:6] 的 shape 是什么？",
      ["[2, dim]", "[3, dim] 并自动补零", "会越界报错"],
      0,
      "Python 切片允许终点超过长度，最后一块会自然缩短为剩余的 2 行。"
    ),
    homework: [
      "在 Notebook 函数顶部先给 q、k、v 和三个状态标注 shape，再写 TODO 1；不要只凭变量名猜维度。",
      "检查 out、m、l 的初值分别表达“无输出、未见最大值、无指数贡献”，并让新张量跟随 q 的 device 与 dtype。",
      "沿着外层 i、内层 j 手工走一遍 seq_len=5、block_size=3，确认最后的 Q 块和 K/V 块都允许少于 3 行。"
    ]
  }),

  lesson({
    id: "flash-score-max-normalizer",
    title: "处理一个 K/V 块：先守住 shape，再更新 m 与 l",
    todo: "TODO 2 / TODO 3 / TODO 4 / TODO 5",
    prerequisite: [
      "矩阵乘法 [Bq,D] @ [D,Bk] 会得到 [Bq,Bk]：每一行属于一个 query，每一列属于当前 K 块中的一个 key。",
      "q_block 已经在外层乘过 1/sqrt(dim)，所以计算 S_ij 时不要再次缩放。",
      "max 和 sum 都要沿最后一维做，因为 softmax 是对每个 query 的所有 key 分数归一化。",
      "keepdim=True 会把结果保留为 [Bq,1]，后续才能稳定地与 [Bq,Bk] 广播。"
    ],
    intuition: "这一阶段先不碰 V，只处理 softmax 的账本。当前块给出一组新分数；m_new 统一新旧标尺，P_ij 把当前分数换到这把标尺下，l_new 再把旧块和新块的指数贡献合起来。",
    exampleHtml: `<div class="flash-course">
        <table class="flash-shapes">
          <thead><tr><th>张量</th><th>shape</th><th>每一维在说什么</th></tr></thead>
          <tbody>
            <tr><td><code>q_block</code></td><td><code>[Bq, D]</code></td><td>Bq 个 query，每个 query 有 D 个特征</td></tr>
            <tr><td><code>k_block.T</code></td><td><code>[D, Bk]</code></td><td>转置后 D 对齐做点积，Bk 成为分数列</td></tr>
            <tr><td><code>S_ij</code></td><td><code>[Bq, Bk]</code></td><td>每个 query 对当前 Bk 个 key 的分数</td></tr>
            <tr><td><code>m_block / m_new</code></td><td><code>[Bq, 1]</code></td><td>每个 query 只保留一个最大值</td></tr>
            <tr><td><code>P_ij</code></td><td><code>[Bq, Bk]</code></td><td>当前块在新最大值标尺下的指数值</td></tr>
            <tr><td><code>l_block / l_new</code></td><td><code>[Bq, 1]</code></td><td>每个 query 只保留一个指数和</td></tr>
          </tbody>
        </table>

        <div class="flash-formula">
          <div><b>1</b><code>m_block = row_max(S_ij)</code><span>只在当前 K 块里找每行最大值</span></div>
          <div><b>2</b><code>m_new = maximum(m_i, m_block)</code><span>把旧块和当前块放到同一个新标尺</span></div>
          <div><b>3</b><code>P_ij = exp(S_ij - m_new)</code><span>减去行最大值，避免 exp 溢出</span></div>
          <div><b>4</b><code>old_scale = exp(m_i - m_new)</code><span>把过去保存的指数和换算到新标尺</span></div>
          <div><b>5</b><code>l_new = l_i * old_scale + sum(P_ij)</code><span>旧贡献修正后，再加当前块贡献</span></div>
        </div>

        <div class="flash-split">
          <section class="flash-example">
            <h4>只看一行的小例子</h4>
            <p>旧块最大值是 2，新块分数是 [4, 0]，所以新最大值变成 4。</p>
            <div class="flash-contract">
              <span>旧贡献缩放</span><strong>exp(2 - 4) = exp(-2)，旧值整体变小</strong>
              <span>新块指数</span><strong>exp([4, 0] - 4) = [1, exp(-4)]</strong>
              <span>共同标尺</span><strong>旧块和新块现在都以 m_new=4 为基准</strong>
            </div>
          </section>
          <section class="flash-panel warn">
            <h4>两个最容易漏掉的细节</h4>
            <p><code>q_block</code> 已经乘过 scale，S_ij 这里只做矩阵乘法。</p>
            <p><code>m_i</code> 和 <code>m_block</code> 是两个同 shape 张量，要逐元素比较，使用的是 <code>torch.maximum</code>。</p>
          </section>
        </div>
      </div>`,
    syntaxHtml: `<div class="flash-practice">
        ${code("用独立矩阵练转置、逐行归约和广播", [
          "left = torch.randn(2, 3)   # [Bq=2, D=3]",
          "right = torch.randn(4, 3)  # [Bk=4, D=3]",
          "scores = left @ right.transpose(-2, -1)  # [2, 4]",
          "",
          "block_max = scores.max(dim=-1, keepdim=True).values  # [2, 1]",
          "old_max = torch.full((2, 1), -float('inf'))",
          "new_max = torch.maximum(old_max, block_max)",
          "weights = torch.exp(scores - new_max)               # 广播到 [2, 4]",
          "block_sum = weights.sum(dim=-1, keepdim=True)        # [2, 1]"
        ])}
        <div class="flash-map">
          <h4>从例子迁移到 TODO 2-5</h4>
          <ul>
            <li><code>left/right/scores</code> 分别迁移成当前 Q 块、当前 K 块和 S_ij。</li>
            <li><code>block_max/new_max</code> 对应局部最大值和新全局最大值。</li>
            <li><code>weights</code> 对应当前块的未归一化指数 P_ij；它还不是最终 softmax 概率。</li>
            <li><code>block_sum</code> 只是新块贡献，必须和经过 old_scale 修正的旧 l_i 合并。</li>
          </ul>
        </div>
      </div>`,
    predict: {
      hook: "固定一个有 Bq 行的 Q 块，再拿一个有 Bk 行的 K 块做点积。",
      question: "S_ij = q_block @ k_block.transpose(-2, -1) 的 shape 是什么？",
      options: ["[Bq, Bk]", "[D, D]", "[Bk, Bq, D]"],
      answer: 0,
      revealNote: "每个 query 都要和当前块的每个 key 得到一个分数，因此行数来自 Q，列数来自 K。"
    },
    checkpoint: checkpoint(
      "m_block 和 l_block 为什么都要使用 keepdim=True？",
      ["保留 [Bq,1]，方便和分数块按行广播", "让 block_size 自动翻倍", "把浮点数变成整数"],
      0,
      "去掉最后一维后容易得到 [Bq]，它与 [Bq,Bk] 的广播方向不符合逐行 softmax 的需要。"
    ),
    homework: [
      "TODO 2 先写出 S_ij 的 shape 推导，再完成矩阵乘法；确认 q_block 的 scale 没有乘第二次。",
      "TODO 3 用逐行 max 得到 [Bq,1]，再逐元素合并旧 m_i 与当前 m_block。",
      "TODO 4/5 按顺序写 P_ij、l_block、old_scale 和 l_new；每写一行都检查 shape，而不是一次抄完整公式。"
    ]
  }),

  lesson({
    id: "flash-merge-output",
    title: "合并输出并写回：把旧答案换到新标尺，再加入当前块",
    todo: "TODO 6 + 状态更新 + 全局写回",
    prerequisite: [
      "out_i 保存的是已经除过 l_i 的归一化输出，不是未归一化分子。",
      "当 m 从 m_i 变成 m_new 时，旧贡献要乘 old_scale = exp(m_i - m_new)。",
      "P_ij @ v_block 得到当前 K/V 块对每个 query 的新输出分子，shape 是 [Bq, dim]。",
      "一个 K/V 块处理完后要更新 m_i、l_i；内层循环结束后，再把 out_i、m_i、l_i 写回全局切片。"
    ],
    intuition: "把 out_i 想成旧数据的加权平均。要和新数据合并，不能直接把两个平均数相加；先用旧总权重 l_i 找回旧贡献，再按新最大值缩放，加入新块的加权和，最后除以新的总权重 l_new。",
    exampleHtml: `<div class="flash-course">
        <div class="flash-formula">
          <div><b>1</b><code>old_numerator = out_i * l_i * old_scale</code><span>从旧平均值恢复旧分子，并换到 m_new 标尺</span></div>
          <div><b>2</b><code>new_numerator = P_ij @ v_block</code><span>当前块的指数权重乘 Value</span></div>
          <div><b>3</b><code>out_new = (old_numerator + new_numerator) / l_new</code><span>合并分子，再除以新指数和</span></div>
        </div>

        <div class="flash-split">
          <section class="flash-panel blocked">
            <h4>与 Notebook 参考公式完全等价</h4>
            <div class="flash-contract">
              <span>旧部分</span><strong>out_i * (l_i * old_scale / l_new)</strong>
              <span>新部分</span><strong>(P_ij @ v_block) / l_new</strong>
              <span>合并</span><strong>out_new = old_part + new_part</strong>
            </div>
          </section>
          <section class="flash-example">
            <h4>用第一块做自检</h4>
            <p>第一块开始时 out_i=0、l_i=0、m_i=-inf。旧部分自然为 0，只剩当前块贡献除以 l_new，这正是普通 softmax 在第一个块上的结果。</p>
            <p>如果第一块就出现 NaN，优先检查是否先算了 l_new、分母是否为 0，以及 m_new 的 shape 是否正确。</p>
          </section>
        </div>

        <div class="flash-roadmap">
          <span><b>每个 j 块后</b>out_i 更新为合并结果</span>
          <span><b>滚动状态</b>m_i=m_new，l_i=l_new</span>
          <span><b>所有 j 完成</b>写回 out/m/l 的 i 切片</span>
        </div>

        <div class="flash-test-grid">
          <span><strong>8 / 4 / block 2</strong><br>检查常规分块</span>
          <span><strong>5 / 3 / block 3</strong><br>检查不完整尾块</span>
          <span><strong>3 / 2 / block 1</strong><br>检查逐 token 更新</span>
        </div>

        <div class="flash-callout">测试比较的是分块结果与标准 Attention 的最大绝对误差。shape 正确但误差大，通常不是循环边界问题，而是旧贡献没有按 m_new 重缩放、scale 乘了两次，或状态更新/写回漏了一步。</div>
      </div>`,
    syntaxHtml: `<div class="flash-practice">
        ${code("用滚动加权平均理解输出合并", [
          "# old_avg 已经是归一化平均值，old_total 是旧权重和",
          "old_scale = torch.exp(old_max - new_max)",
          "new_total = old_total * old_scale + block_total",
          "",
          "old_share = old_avg * (old_total * old_scale / new_total)",
          "new_share = block_weighted_sum / new_total",
          "merged_avg = old_share + new_share"
        ])}
        <div class="flash-map">
          <h4>从例子迁移到 TODO 6 和写回</h4>
          <ul>
            <li><code>old_avg</code> 对应 out_i，<code>block_weighted_sum</code> 对应 P_ij @ v_block。</li>
            <li>先得到新的 out_i，再把 m_i、l_i 更新为 m_new、l_new，下一轮 j 才能接着累计。</li>
            <li>内层循环结束只代表当前 Q 块完成；还要把三个局部状态写回全局的 i 切片。</li>
            <li>最后依次跑 Notebook 的三个测试，用误差和边界 case 判断是否真正等价。</li>
          </ul>
        </div>
      </div>`,
    predict: {
      hook: "旧 out_i 和当前块的 P_ij @ v_block 都有 [Bq, dim] 的 shape，看起来可以直接相加。",
      question: "为什么不能直接写 out_i + P_ij @ v_block？",
      options: [
        "两部分使用的最大值标尺和归一化权重不同，必须先修正并除以 l_new",
        "因为两个张量的 shape 永远不同",
        "因为 FlashAttention 不能使用加法"
      ],
      answer: 0,
      revealNote: "shape 一样只说明能执行相加，不说明数学含义一致。在线 softmax 的难点正是统一标尺和归一化因子。"
    },
    checkpoint: checkpoint(
      "合并新块前，为什么旧 out_i 需要带上 l_i * old_scale？",
      ["out_i 是旧的归一化平均值，要恢复并重标定旧分子", "为了把 dim 改成 seq_len", "为了删除当前 V 块"],
      0,
      "旧输出已经除过旧 l_i；合并前需要恢复它代表的加权贡献，并按新的最大值标尺缩放。"
    ),
    homework: [
      "TODO 6 先分别写出旧部分和新部分，核对两者都是 [Bq, dim]，再相加得到新的 out_i。",
      "每处理完一个 K/V 块，更新 m_i 与 l_i；每处理完一个 Q 块，把 out_i、m_i、l_i 写回对应全局切片。",
      "运行全部三个测试；若误差超标，按“scale 是否重复、old_scale 是否使用、状态是否逐块更新、尾块是否自然切片”四项逐一排查。"
    ]
  })
];
