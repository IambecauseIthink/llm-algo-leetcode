const { advancedStyles, checkpoint, code, lesson } = require("./advanced_lesson_helpers");

module.exports = {
  "29": [
    lesson({
      id: "tp-column-slice-compute-gather",
      title: "Column Parallel：切输出列，每张卡只产出一段特征",
      todo: "TODO 1-3：Column Parallel 的权重分片、局部矩阵乘法与 All-Gather",
      prerequisite: [
        "先给维度起名字：X 是 [batch, in_features]，A 是 [in_features, out_features]，完整结果 Y 是 [batch, out_features]。",
        "Column Parallel 中的 column 指 A 的列，也就是输出特征。输入 X 不切，它会交给每张卡。",
        "二维切片 A[:, start:end] 保留全部输入行，只取一段输出列；矩阵乘法的中间维必须同为 in_features。",
        "torch.cat 是拼接，不做数值相加。沿 dim=1 拼接后，输出特征数才会恢复为 out_features。"
      ],
      intuition: "把线性层想成同时计算很多个输出栏目。每张卡负责其中一组栏目：它们都读取同一份 X，但只保存 A 的一部分列。局部结果本来就是最终 Y 的不同列，所以最后按列排回去，不需要把数值相加。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>先读契约</b>X [B,I]，A [I,O]</span>
          <span><b>按列切 A</b>每片 [I,O/G]</span>
          <span><b>各卡局部算</b>X @ A_g 得 [B,O/G]</span>
          <span><b>All-Gather</b>沿输出维拼成 [B,O]</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>变量先有语义，再看下标</h4>
            <div class="adv-contract">
              <span><code>chunk_size</code></span><strong>每张卡负责的输出特征数 O/G</strong>
              <span><code>start_idx:end_idx</code></span><strong>第 i 张卡在 A 的列区间</strong>
              <span><code>a_chunk</code></span><strong>局部权重 [I,O/G]</strong>
              <span><code>y_local</code></span><strong>局部输出 [B,O/G]</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>两张卡的真实数据流</h4>
            <div class="adv-flow">
              <span>X [B,I]<br>两卡共享</span>
              <span>A0 [I,O/2]<br>A1 [I,O/2]</span>
              <span>Y0 [B,O/2]<br>Y1 [B,O/2]</span>
              <strong>cat dim=1<br>Y [B,O]</strong>
            </div>
            <p>Y0 与 Y1 覆盖不同的输出列，拼接顺序必须与权重切片顺序一致。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>动作</th><th>输入 shape</th><th>输出 shape</th><th>不变量</th></tr></thead>
          <tbody>
            <tr><td>TODO 1 切权重列</td><td><code>A [I,O]</code></td><td><code>a_chunk [I,O/G]</code></td><td>输入维 I 不变</td></tr>
            <tr><td>TODO 2 局部乘法</td><td><code>X [B,I] @ a_chunk [I,O/G]</code></td><td><code>y_local [B,O/G]</code></td><td>中间维 I 对齐</td></tr>
            <tr><td>TODO 3 拼接</td><td>G 个 <code>[B,O/G]</code></td><td><code>Y_tp [B,O]</code></td><td>batch B 不变</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">Notebook 已在函数入口断言 O 能被 G 整除。不要把 Column Parallel 误解成切 X：这一页的 TODO 1 只切 A 的列，TODO 2 的每张卡仍使用完整 X。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用商品特征练习按输出列分片", [
          "samples = torch.randn(3, 6)       # [B=3, I=6]",
          "projection = torch.randn(6, 12)  # [I=6, O=12]",
          "workers = 3",
          "width = projection.shape[1] // workers",
          "",
          "local_outputs = []",
          "for worker in range(workers):",
          "    left = worker * width",
          "    right = left + width",
          "    local_weight = projection[:, left:right]  # [6, 4]",
          "    local_outputs.append(samples @ local_weight)",
          "",
          "joined = torch.cat(local_outputs, dim=1)       # [3, 12]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>projection</code> -&gt; <code>A</code>；<code>workers</code> -&gt; <code>num_gpus</code>。</li>
            <li><code>left/right</code> -&gt; <code>start_idx/end_idx</code>；列切片语法支撑 TODO 1。</li>
            <li><code>samples @ local_weight</code> -&gt; TODO 2 的 X 与本地权重相乘。</li>
            <li><code>torch.cat(..., dim=1)</code> -&gt; TODO 3 的输出特征拼接；不要改成 sum。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "设 X 是 [4,16]，A 是 [16,32]，两张卡平均切 A 的输出列。",
        question: "第 1 张卡完成局部矩阵乘法后，y_local 的 shape 应该是什么？",
        options: ["[4,16]", "[2,32]", "[4,32]"],
        answer: 0,
        revealNote: "每张卡保留全部 4 个样本，只负责 32/2=16 个输出特征，因此是 [4,16]。"
      },
      checkpoint: checkpoint(
        "Column Parallel 的多个 y_local 为什么要沿 dim=1 拼接？",
        ["它们是同一批样本的不同输出特征列", "它们是不同 batch，必须沿 dim=0 拼接", "它们是同一输出的部分和，必须先求和"],
        0,
        "局部权重按 A 的列切开，局部输出自然对应 Y 的不同列；dim=1 正是输出特征维。"
      ),
      homework: [
        "完成 TODO 1-3：先写出每行代码前后的 shape，再依次实现列切片、局部矩阵乘法和按特征维拼接。",
        "shape/dtype/device：a_chunk 应为 [in_features,out_features/num_gpus]，y_local 为 [batch,out_features/num_gpus]；切片和矩阵乘法会保持 X/A 原有 dtype 与 device，不要另造 CPU 张量。",
        "测试：运行 test_tensor_parallel，重点核对 Y_col.shape == Y_ref.shape 且最大绝对误差小于 1e-5。",
        "常见错误排查：若 shape 少一半，检查是否漏了 cat；若乘法报错，检查是否切了 A 的 dim=0；若列顺序错，检查分片 append 顺序。"
      ]
    }),

    lesson({
      id: "tp-row-slice-compute-reduce",
      title: "Row Parallel：同步切输入维，每张卡算完整输出的部分和",
      todo: "TODO 4-6：Row Parallel 的输入/权重同步分片、局部乘法与 All-Reduce",
      prerequisite: [
        "Row Parallel 中的 row 指 A 的行，也就是输入特征。A 按 dim=0 切时，X 必须按自己的 dim=1 切同一段特征。",
        "局部 X 是 [batch, in_features/G]，局部 A 是 [in_features/G, out_features]，两者仍可做矩阵乘法。",
        "每张卡的 y_local 都是 [batch, out_features]；它只包含一部分输入特征带来的贡献。",
        "torch.stack 会新增 GPU 维得到 [G,batch,out_features]，再沿 dim=0 求和才模拟 All-Reduce Sum。"
      ],
      intuition: "完整点积本来就是把每个输入特征的乘积加起来。Row Parallel 只是把这串加法分给多张卡：每张卡用相同的一段输入特征和对应的权重行算一个部分和，最后逐元素相加。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>不能只切权重</h4>
            <div class="adv-flow">
              <span>X [B,I]</span>
              <span>A_g [I/G,O]</span>
              <strong>中间维不相等<br>无法相乘</strong>
            </div>
            <p>A 的行缩短后，X 的特征列也必须取完全相同的 start:end 区间。</p>
          </section>
          <section class="adv-panel good">
            <h4>同步切片后的局部乘法</h4>
            <div class="adv-flow">
              <span>X_g [B,I/G]</span>
              <span>A_g [I/G,O]</span>
              <strong>Y_g [B,O]</strong>
            </div>
            <p>每张卡都产出完整输出 shape，但数值上只是部分贡献。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>x_chunk = X[:, start:end]</code><span>切 X 的特征列，保留所有样本</span></div>
          <div><b>2</b><code>a_chunk = A[start:end, :]</code><span>切 A 的输入行，保留所有输出列</span></div>
          <div><b>3</b><code>y_local = x_chunk @ a_chunk</code><span>得到 [B,O] 的部分和</span></div>
          <div><b>4</b><code>stack then sum dim=0</code><span>新增卡维后逐元素归约</span></div>
        </div>

        <div class="adv-contract">
          <span>代数拆分</span><strong>X @ A = X0 @ A0 + X1 @ A1 + ...</strong>
          <span>切片配对</span><strong>X 的第 g 段特征必须对应 A 的第 g 段输入行</strong>
          <span>通信语义</span><strong>All-Reduce Sum 合并数值贡献，不是拼接 shape</strong>
        </div>

        <div class="adv-callout">看到多个局部结果 shape 都是 [B,O] 时，要问“它们是不同区域，还是同一区域的部分值？”Row Parallel 属于后者，所以 TODO 6 是求和。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用传感器分组练习同步切输入和权重", [
          "readings = torch.randn(5, 8)    # [B=5, I=8]",
          "decoder = torch.randn(8, 3)    # [I=8, O=3]",
          "workers = 4",
          "span = readings.shape[1] // workers",
          "",
          "partial_sums = []",
          "for worker in range(workers):",
          "    lo = worker * span",
          "    hi = lo + span",
          "    local_x = readings[:, lo:hi]",
          "    local_w = decoder[lo:hi, :]",
          "    partial_sums.append(local_x @ local_w)",
          "",
          "merged = torch.stack(partial_sums, dim=0).sum(dim=0)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>readings/decoder</code> -&gt; <code>X/A</code>；两处使用同一个 <code>lo:hi</code> 支撑 TODO 4。</li>
            <li><code>local_x @ local_w</code> -&gt; TODO 5 的本地部分和。</li>
            <li><code>partial_sums</code> -&gt; Notebook 的 <code>y_outputs</code>；列表名不同不改变语义。</li>
            <li><code>stack(...).sum(dim=0)</code> -&gt; TODO 6 的 All-Reduce Sum 模拟。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "X 是 [4,16]，A 是 [16,32]，两张卡沿输入维切分。",
        question: "每张卡的 x_chunk、a_chunk、y_local shape 依次是什么？",
        options: ["[4,8]、[8,32]、[4,32]", "[2,16]、[16,16]、[2,16]", "[4,16]、[8,32]、[4,32]"],
        answer: 0,
        revealNote: "输入特征 16 被平分为 8；局部乘法的输出特征仍是完整的 32。"
      },
      checkpoint: checkpoint(
        "Row Parallel 中，为什么不能用 torch.cat(y_outputs, dim=1) 恢复结果？",
        ["每个 y_local 已覆盖相同的 [batch,out_features] 区域，应该逐元素求和", "cat 只能处理整数", "因为 y_outputs 中只能有一个张量"],
        0,
        "这些输出是同一结果的部分和，不是互不重叠的输出列；拼接会错误地把 out_features 扩大 G 倍。"
      ),
      homework: [
        "完成 TODO 4-6：用同一 start_idx:end_idx 同步切 X 的 dim=1 与 A 的 dim=0，再完成局部乘法和按卡维求和。",
        "shape/dtype/device：x_chunk 为 [batch,in_features/num_gpus]，a_chunk 为 [in_features/num_gpus,out_features]，每个 y_local 为 [batch,out_features]；切片保持原 dtype/device，stack 要求各结果一致。",
        "测试：运行 test_tensor_parallel，核对 Y_row.shape 与 Y_ref 相同，且 diff_row 小于 1e-5。",
        "常见错误排查：矩阵维不匹配时检查两处切片区间；输出宽度变大时检查是否误用 cat；数值偏差大时检查 sum 是否沿新加的 dim=0。"
      ]
    }),

    lesson({
      id: "tp-contract-tests-and-composition",
      title: "从两种通信回到测试：shape 对了，还要证明数值等价",
      todo: "TODO 1-6 收口：对照 Y_ref、理解整除约束与 Column -&gt; Row 组合",
      prerequisite: [
        "Y_ref = X @ A 是单卡基准；并行模拟的目标不是得到相近 shape，而是得到同一个矩阵。",
        "torch.max(torch.abs(Y_ref - Y_tp)) 给出最大绝对误差，测试要求 Column 和 Row 都小于 1e-5。",
        "Column 的局部输出分占不同列，通信是拼接；Row 的局部输出重叠在同一 shape，通信是求和。",
        "两种函数都要求被切维度能被 num_gpus 整除：Column 检查 out_features，Row 检查 in_features。"
      ],
      intuition: "判断通信方式只需要追踪局部结果在最终 Y 中的位置。位置互不重叠就拼接；位置相同但各自只有部分数值就求和。测试用 shape 保护位置，用误差保护数值。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>Column：区域不同</h4>
            <div class="adv-contract">
              <span>切分轴</span><strong>A dim=1，输出特征轴</strong>
              <span>局部 shape</span><strong>[B,O/G]</strong>
              <span>合并动作</span><strong>cat dim=1</strong>
              <span>通信类比</span><strong>All-Gather</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>Row：区域相同</h4>
            <div class="adv-contract">
              <span>切分轴</span><strong>A dim=0 与 X dim=1</strong>
              <span>局部 shape</span><strong>[B,O]</strong>
              <span>合并动作</span><strong>stack 后 sum dim=0</strong>
              <span>通信类比</span><strong>All-Reduce Sum</strong>
            </div>
          </section>
        </div>

        <div class="adv-checks">
          <span><strong>shape 检查</strong><br>Y_tp 与 Y_ref 完全一致</span>
          <span><strong>数值检查</strong><br>max abs diff &lt; 1e-5</span>
          <span><strong>Column 约束</strong><br>O % G == 0</span>
          <span><strong>Row 约束</strong><br>I % G == 0</span>
        </div>

        <div class="adv-roadmap">
          <span><b>W1 按列切</b>各卡得到一段隐藏特征</span>
          <span><b>中间不急着聚合</b>局部隐藏特征直接进入下一层</span>
          <span><b>W2 按行切</b>各卡计算输出部分和</span>
          <span><b>末尾归约</b>一次 All-Reduce 得完整输出</span>
        </div>

        <div class="adv-callout">Notebook 的两段“维度约束检查”都用 try/except 捕获 AssertionError，也会捕获块内手工抛出的同类异常。它们不能单独证明函数一定拒绝非法维度；真正的约束来自两个函数入口已经写好的 assert。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用独立分片函数做 shape 与数值双检查", [
          "features = torch.randn(2, 10)",
          "weight = torch.randn(10, 6)",
          "reference = features @ weight",
          "",
          "left = features @ weight[:, :3]",
          "right = features @ weight[:, 3:]",
          "candidate = torch.cat([left, right], dim=1)",
          "",
          "assert candidate.shape == reference.shape",
          "error = (candidate - reference).abs().max()",
          "assert error < 1e-5"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>reference</code> -&gt; 测试里的 <code>Y_ref</code>。</li>
            <li><code>left/right</code> -&gt; Column TODO 2 产生的各个 <code>y_local</code>。</li>
            <li><code>candidate</code> -&gt; TODO 3 或 TODO 6 返回的 <code>Y_tp</code>。</li>
            <li><code>shape</code> 与 <code>error</code> 两道检查 -&gt; Notebook 对 Y_col/Y_row 的断言顺序。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某个 Row Parallel 实现返回 [4,64]，而 Y_ref 是 [4,32]。",
        question: "最可能发生了什么？",
        options: ["把两个 [4,32] 部分和沿 dim=1 拼接了", "A 的输出列被正确拼接", "最大绝对误差恰好为 0"],
        answer: 0,
        revealNote: "Row 的每个局部输出本来就是 [4,32]；cat 会把宽度错误地扩大到 64，正确动作是逐元素求和。"
      },
      checkpoint: checkpoint(
        "以下哪组“切分维度 -&gt; 合并动作”与 Notebook 完全一致？",
        ["Column: A dim=1 -&gt; cat dim=1；Row: X dim=1 与 A dim=0 -&gt; sum", "Column: X dim=0 -&gt; sum；Row: A dim=1 -&gt; cat", "Column 与 Row 都沿 batch 切分并取平均"],
        0,
        "Column 拆输出列，所以拼接；Row 拆点积的输入项，所以同步切 X/A 后求和。"
      ),
      homework: [
        "收口 TODO 1-6：在每个 append 和最终 return 前打印一次 shape，确认 Column 是分列后拼接，Row 是同 shape 部分和后归约。",
        "shape/dtype/device：最终 Y_col/Y_row 都应与 Y_ref 的 [batch,out_features]、dtype、device 一致；本 CPU-first Notebook 不应主动迁移设备或改精度。",
        "测试：完整运行 test_tensor_parallel，记录 diff_col 与 diff_row；同时手工调用一次非法 Column 输出维度和非法 Row 输入维度，确认入口 assert 真正触发。",
        "常见错误排查：shape 错先查切分/合并轴，shape 对但误差大再查分片顺序、X/A 区间配对和求和维度。"
      ]
    })
  ],

  "30": [
    lesson({
      id: "lora-parameter-meaning",
      title: "先认清 LoRA 的两块小矩阵，再数可训练参数",
      todo: "当前 Notebook 无 TODO 占位：独立复写 lora_trainable_params 与 full_linear_params",
      prerequisite: [
        "普通线性权重把 in_dim 个输入特征映射到 out_dim 个输出特征，因此权重表有 in_dim * out_dim 个元素。",
        "LoRA 不直接训练这张大表，而是训练两个低秩因子：一个把 in_dim 压到 rank，另一个把 rank 展开到 out_dim。",
        "参数量统计只数矩阵元素，不乘 batch size，也不乘训练样本数。",
        "本 Notebook 的三个函数输入是 Python 整数，返回 int 或 float；这里还没有创建真实 torch.Tensor。"
      ],
      intuition: "先把 rank 理解成一条窄通道。原权重直接从 in_dim 连到 out_dim；LoRA 先经过 rank 个中间方向。两块小矩阵分别有 rank * in_dim 和 out_dim * rank 个参数，相加就是 LoRA 可训练参数量。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>全参数线性层</h4>
            <div class="adv-flow">
              <span>输入<br>in_dim</span>
              <strong>W<br>[out_dim,in_dim]</strong>
              <span>输出<br>out_dim</span>
            </div>
            <p>若只统计权重，不含 bias，参数量是 in_dim * out_dim。</p>
          </section>
          <section class="adv-panel good">
            <h4>LoRA 的低秩路径</h4>
            <div class="adv-flow">
              <span>输入<br>in_dim</span>
              <span>A<br>[rank,in_dim]</span>
              <span>B<br>[out_dim,rank]</span>
              <strong>增量输出<br>out_dim</strong>
            </div>
            <p>两个因子的元素总数是 rank * in_dim + out_dim * rank。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>in_dim</code><span>一条输入样本含多少特征</span></div>
          <div><b>2</b><code>out_dim</code><span>线性层要产生多少输出特征</span></div>
          <div><b>3</b><code>rank</code><span>LoRA 中间窄通道的宽度</span></div>
          <div><b>4</b><code>rank * (in_dim + out_dim)</code><span>把两块低秩因子的参数数目相加</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>对象</th><th>概念 shape</th><th>参数量</th><th>是否在模板中创建张量</th></tr></thead>
          <tbody>
            <tr><td>完整权重</td><td><code>[out_dim,in_dim]</code></td><td><code>out_dim * in_dim</code></td><td>否，只做整数估算</td></tr>
            <tr><td>LoRA 因子 A</td><td><code>[rank,in_dim]</code></td><td><code>rank * in_dim</code></td><td>否</td></tr>
            <tr><td>LoRA 因子 B</td><td><code>[out_dim,rank]</code></td><td><code>out_dim * rank</code></td><td>否</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">当前 Notebook 是“项目参数估算模板”，没有要求真正注入 nn.Linear，也没有编号 TODO。先准确复写三个标量函数并通过断言，再把公式迁移到真实模型。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用图像投影层练习参数计数", [
          "input_width = 12",
          "output_width = 20",
          "bottleneck = 3",
          "",
          "factor_down = bottleneck * input_width",
          "factor_up = output_width * bottleneck",
          "adapter_count = factor_down + factor_up",
          "dense_count = input_width * output_width",
          "",
          "print(adapter_count, dense_count)  # 96, 240"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；学习目标是能独立复写项目模板函数。</li>
            <li><code>input_width/output_width/bottleneck</code> -&gt; <code>in_dim/out_dim/rank</code>。</li>
            <li><code>adapter_count</code> -&gt; <code>lora_trainable_params</code> 的返回语义。</li>
            <li><code>dense_count</code> -&gt; <code>full_linear_params</code> 的返回语义；不要额外加入 batch 或样本数。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "一个线性层 in_dim=12、out_dim=20，LoRA rank=3，只统计两块适配器矩阵。",
        question: "LoRA 可训练参数量是多少？",
        options: ["96", "240", "720"],
        answer: 0,
        revealNote: "3 * 12 + 20 * 3 = 36 + 60 = 96；240 是完整线性权重参数量。"
      },
      checkpoint: checkpoint(
        "为什么 lora_trainable_params 的公式中 in_dim 和 out_dim 都只乘一次 rank？",
        ["LoRA 有两个因子，分别连接 in_dim - rank 和 rank - out_dim", "因为 batch size 永远等于 1", "因为完整权重已经被冻结后自动变成 rank 个元素"],
        0,
        "两个低秩矩阵分别含 rank*in_dim 与 out_dim*rank 个元素，相加即可。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：遮住参考区，独立写出 lora_trainable_params 和 full_linear_params，并用另一组尺寸手算核对。",
        "shape/dtype/device：模板只接收 Python int 并返回 int，没有 tensor shape/dtype/device；迁移到真实 PyTorch 时，LoRA A/B 概念 shape 分别是 [rank,in_dim]、[out_dim,rank]，并应跟随 base weight 的 dtype/device。",
        "测试：用 Notebook 的 in_dim=8、out_dim=8、rank=2，确认 trainable 精确等于 32、full 精确等于 64。",
        "常见错误排查：结果偏小检查是否漏掉一个因子；偏大检查是否误乘 in_dim*out_dim*rank；类型异常检查是否把维度写成字符串。"
      ]
    }),

    lesson({
      id: "lora-ratio-and-project-boundary",
      title: "把参数量变成比例：知道 LoRA 省了什么，也知道测试没证明什么",
      todo: "当前 Notebook 无 TODO 占位：独立复写 lora_param_ratio，并读懂项目指标与基础断言边界",
      prerequisite: [
        "比例的分子是 LoRA 可训练参数量，分母是完整线性权重参数量；两者必须描述同一个 in_dim/out_dim 层。",
        "Python 的 / 返回浮点数；例如 32/64 得 0.5，格式化为百分比才显示 50%。",
        "rank 增大时，LoRA 参数量线性增大；完整权重参数量不随 rank 改变。",
        "参数更少通常会减少参数、梯度和优化器状态开销，但当前模板没有实际测 peak memory、step time 或 loss。"
      ],
      intuition: "绝对参数量回答“要训练多少个数”，比例回答“相对完整权重只训练多少”。比例越小，参数侧越轻；但训练效果、激活显存和速度必须另外测，不能从一个比例函数直接推出。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-contract">
          <span>分子</span><strong>LoRA trainable = rank * (in_dim + out_dim)</strong>
          <span>分母</span><strong>full linear = in_dim * out_dim</strong>
          <span>比例</span><strong>trainable / full，返回 float</strong>
          <span>rank 趋势</span><strong>rank 翻倍时分子与比例都翻倍</strong>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>基础测试真正证明的事</h4>
            <div class="adv-checks">
              <span>trainable 公式<br>8,8,2 -&gt; 32</span>
              <span>full 公式<br>8,8 -&gt; 64</span>
              <span>ratio 数值<br>接近 0.5</span>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>基础测试尚未证明的事</h4>
            <div class="adv-checks">
              <span>base model 是否冻结</span>
              <span>loss 是否下降</span>
              <span>peak memory 是否减少</span>
              <span>step time 是否更快</span>
            </div>
          </section>
        </div>

        <div class="adv-roadmap">
          <span><b>参数侧</b>rank、插入层数、trainable ratio</span>
          <span><b>资源侧</b>peak memory、激活与优化器状态</span>
          <span><b>时间侧</b>step time、吞吐与同步</span>
          <span><b>效果侧</b>loss 曲线与稳定性</span>
        </div>

        <div class="adv-callout">项目结论要分层写：参数公式是确定的；显存、速度和效果是实验结果。不要把“可训练参数少”直接写成“训练一定更快、总显存一定按同比例下降”。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用多个适配层汇总参数比例", [
          "layers = [",
          "    {'in': 24, 'out': 32, 'rank': 4},",
          "    {'in': 32, 'out': 16, 'rank': 2},",
          "]",
          "",
          "adapter_total = sum(x['rank'] * (x['in'] + x['out']) for x in layers)",
          "dense_total = sum(x['in'] * x['out'] for x in layers)",
          "trainable_share = adapter_total / dense_total",
          "print(f'{trainable_share:.2%}')"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；<code>trainable_share</code> -&gt; <code>lora_param_ratio</code> 的单层返回值。</li>
            <li><code>adapter_total</code> 的单层项 -&gt; <code>lora_trainable_params</code>。</li>
            <li><code>dense_total</code> 的单层项 -&gt; <code>full_linear_params</code>。</li>
            <li>多层汇总是独立扩展场景；Notebook 基础断言只校验单层 8x8、rank=2。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "保持 in_dim 与 out_dim 不变，把 LoRA rank 从 8 提高到 16。",
        question: "只看 Notebook 的参数公式，trainable 参数量与 ratio 会怎样变化？",
        options: ["两者都变为原来的 2 倍", "trainable 不变，ratio 减半", "两者都变为原来的 4 倍"],
        answer: 0,
        revealNote: "LoRA 参数量对 rank 是一次函数；完整权重参数量不变，因此比例也随 rank 线性翻倍。"
      },
      checkpoint: checkpoint(
        "Notebook 的 ratio=0.5 基础断言能够直接证明哪件事？",
        ["在 8x8、rank=2 的参数公式下，LoRA 参数数是完整权重的一半", "真实训练峰值显存一定减少一半", "LoRA 的 loss 一定优于全参数微调"],
        0,
        "该断言只检查参数算术；训练显存、速度和效果都需要额外实验。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：独立实现 lora_param_ratio，必须复用同一组 in_dim/out_dim/rank 的分子与分母语义。",
        "shape/dtype/device：模板返回 Python float，无 tensor shape/device；真实 LoRA 指标记录时另行核对 A/B shape、参数 dtype/device 与冻结状态，不能从 ratio 推断。",
        "测试：运行 test_lora_project_template，确认 abs(ratio - 0.5) &lt; 1e-12；再用 rank 翻倍的自测验证比例线性变化。",
        "常见错误排查：比例大于预期时检查分子分母是否颠倒、是否把百分数再乘 100；项目结论过度时检查是否把未测的显存/速度/loss 当成断言结果。"
      ]
    }),

    lesson({
      id: "lora-project-measurement-contract",
      title: "从公式走向可交付实验：四类指标必须在同一配置下记录",
      todo: "当前 Notebook 无 TODO 占位：把参数模板迁移为 LoRA 项目的可比实验记录",
      prerequisite: [
        "Notebook Markdown 要求比较全参数微调与 LoRA，但代码单元目前只提供三个参数估算函数。",
        "公平对照意味着模型骨架、样本、训练步数、batch、学习率与计时口径尽量一致，只改变微调方式。",
        "trainable params 是参数侧指标；peak memory、step time 与 loss curve 分属资源、时间和效果侧。",
        "LoRA 主要直接减少可训练参数、梯度和优化器状态；激活仍由前向图产生，未必按参数比例下降。"
      ],
      intuition: "一个可交付结论不是“LoRA 很省”，而是一行有上下文的数据：在相同模型和样本下，rank 是多少、插了几层、可训练比例多少、峰值显存和单步时间多少、loss 是否稳定下降。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-map">
          <h4>同一实验行应回答的四类问题</h4>
          <div class="adv-grid two">
            <section class="adv-panel blue"><h4>参数侧</h4><p>trainable params、ratio、rank、adapter 插入层数。</p></section>
            <section class="adv-panel neutral"><h4>资源侧</h4><p>peak memory、激活、是否使用 checkpointing/offload。</p></section>
            <section class="adv-panel good"><h4>时间侧</h4><p>step time、吞吐、warmup 与测量步数。</p></section>
            <section class="adv-panel warn"><h4>效果侧</h4><p>loss 曲线、最终 loss、是否出现不稳定。</p></section>
          </div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>对照项</th><th>必须保持一致</th><th>允许改变</th><th>结论范围</th></tr></thead>
          <tbody>
            <tr><td>全参数 baseline</td><td>模型、数据、batch、步数、计时口径</td><td>全部参数 requires_grad</td><td>完整训练成本</td></tr>
            <tr><td>LoRA</td><td>同上</td><td>冻结 base、rank、插入层</td><td>适配器训练成本</td></tr>
          </tbody>
        </table>

        <div class="adv-flow">
          <span>冻结 base</span>
          <span>注入少数 adapter</span>
          <span>同批样本训练</span>
          <span>记录四类指标</span>
          <strong>写“收益 + 代价”</strong>
        </div>

        <div class="adv-callout">基础测试通过只代表参数估算函数正确。完整项目还需要真实训练、显存和计时数据；这部分在当前 Notebook 中是实验延伸，不是隐藏的代码断言。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用统一字典记录另一组微调实验", [
          "run = {",
          "    'method': 'adapter',",
          "    'rank': 4,",
          "    'inserted_layers': 3,",
          "    'trainable_params': 720,",
          "    'peak_mem_mb': 384.0,",
          "    'step_time_ms': 21.6,",
          "    'losses': [1.8, 1.5, 1.3],",
          "}",
          "assert run['losses'][-1] <= run['losses'][0]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；<code>trainable_params</code> -&gt; 参数估算函数输出及“参数侧”记录。</li>
            <li><code>rank/inserted_layers</code> -&gt; Markdown 要求写清的 LoRA 配置。</li>
            <li><code>peak_mem_mb/step_time_ms/losses</code> -&gt; 资源、时间、效果三条实验线。</li>
            <li>这个字典是独立记录支架，不替代 Notebook 的三个基础函数和断言。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某次 LoRA 实验的 trainable ratio 很低，但没有记录 batch、step time 或 loss。",
        question: "下面哪条结论最严谨？",
        options: ["只能确认参数侧较轻，速度与效果仍需在公平配置下测量", "已经证明训练一定更快且 loss 更低", "已经证明激活显存也按同比例下降"],
        answer: 0,
        revealNote: "参数比例只覆盖参数侧；资源、时间和效果需要各自的观测值与一致实验条件。"
      },
      checkpoint: checkpoint(
        "比较全参数微调与 LoRA 时，哪一种做法能让结论最可信？",
        ["固定模型、样本、batch、训练步数和测量口径，只改变微调方式", "LoRA 用更短数据并只报告最好一次", "两边使用不同模型，再直接比较总耗时"],
        0,
        "单变量对照能把观察到的差异尽量归因于微调方式，而不是数据或配置变化。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：先跑通三个参数函数，再按 Markdown 的四类指标为 full/LoRA 各设计一条同字段记录。",
        "shape/dtype/device：模板标量无 tensor shape/device；真实实验需记录输入 shape/batch、LoRA A/B dtype/device，并确认 base 参数被冻结、adapter 参数可训练。",
        "测试：保留 test_lora_project_template 的 32、64、0.5 三个断言；新增实验记录时不要声称这些断言覆盖 peak memory、step time 或 loss。",
        "常见错误排查：对照不可比时检查数据与配置是否一致；显存不降时区分参数状态与激活；loss 不降时检查 requires_grad、adapter 插入层和学习率。"
      ]
    })
  ],

  "31": [
    lesson({
      id: "inference-benchmark-loop",
      title: "先把计时器写可信：预热不计入，正式迭代取平均",
      todo: "当前 Notebook 无 TODO 占位：独立复写 benchmark_fn 的调用次数与平均耗时契约",
      prerequisite: [
        "fn 是一个不需要参数的可调用对象；benchmark_fn 负责重复调用它，不关心 fn 内部具体做哪种推理。",
        "warmup 次调用用于让缓存、运行时和模型进入稳定状态，不应计入 start 到 total 的正式时间。",
        "time.perf_counter() 适合测经过时间；两次读数相减得到秒。",
        "total / iters 是单次平均秒数。Notebook 的测试用 warmup=0、iters=3，要求 fn 恰好被调用 3 次。"
      ],
      intuition: "计时像测跑步：先让系统热身，再在同一条起终线之间跑固定轮数，最后除以轮数。预热若放进计时区，启动成本会混入平均值；少调用或多调用一次，测试计数器都会暴露。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>warmup loop</b>调用 fn，但不计时</span>
          <span><b>start</b>读取 perf_counter</span>
          <span><b>timed loop</b>恰好调用 iters 次</span>
          <span><b>end - start</b>得到 total 秒</span>
          <span><b>total / iters</b>返回平均秒数</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>Notebook 基础调用契约</h4>
            <div class="adv-contract">
              <span><code>warmup=0</code></span><strong>预热阶段调用 0 次</strong>
              <span><code>iters=3</code></span><strong>正式阶段调用 3 次</strong>
              <span><code>counter['n']</code></span><strong>测试后必须精确等于 3</strong>
              <span><code>avg</code></span><strong>经过时间不能小于 0</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>计时值不是吞吐</h4>
            <p><code>benchmark_fn</code> 返回单次平均秒数，并不知道每次处理了几个 token。</p>
            <p>若要得到 token/s，还需要用一次调用处理的 token 数除以平均秒数。</p>
          </section>
        </div>

        <div class="adv-checks">
          <span>相同模型与权重</span>
          <span>相同输入与生成长度</span>
          <span>相同 batch/context</span>
          <span>相同 warmup/iters</span>
        </div>

        <div class="adv-callout">当前函数是 CPU-first 的最小模板。若 fn 在 CUDA 上异步执行，严谨计时通常还需在边界同步；Notebook 当前代码与基础测试没有加入这一步，不要悄悄改变其调用次数契约。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用文本清洗函数练习预热与正式计时", [
          "import time",
          "",
          "calls = {'count': 0}",
          "def normalize_text():",
          "    calls['count'] += 1",
          "    return '  hello  '.strip().upper()",
          "",
          "for _ in range(2):",
          "    normalize_text()",
          "",
          "begin = time.perf_counter()",
          "for _ in range(5):",
          "    normalize_text()",
          "average_seconds = (time.perf_counter() - begin) / 5",
          "assert calls['count'] == 7"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；目标是独立复写 <code>benchmark_fn</code>。</li>
            <li><code>normalize_text</code> -&gt; 参数 <code>fn</code>；<code>2/5</code> -&gt; <code>warmup/iters</code>。</li>
            <li><code>begin</code> -&gt; Notebook 的 <code>start</code>；<code>average_seconds</code> -&gt; 返回值。</li>
            <li><code>calls</code> -&gt; 测试里的 <code>counter</code>，用于验证没有漏调或多调。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "调用 benchmark_fn(fn, warmup=3, iters=10)。",
        question: "fn 总共会执行多少次，其中多少次进入正式计时？",
        options: ["13 次，总计时 10 次", "10 次，总计时 10 次", "13 次，总计时 13 次"],
        answer: 0,
        revealNote: "先执行 3 次预热，再执行 10 次正式迭代；计时区只包住后 10 次。"
      },
      checkpoint: checkpoint(
        "benchmark_fn 为什么返回 total / iters，而不是直接返回 total？",
        ["为了得到一次 fn 调用的平均耗时，便于用统一口径比较", "为了把秒自动转换成 MB", "为了让 warmup 也进入计时"],
        0,
        "总时间会随迭代次数增长；除以 iters 才得到可比较的单次平均时间。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：遮住参考代码，独立复写 benchmark_fn，保持预热循环在 start 之前、正式循环恰好 iters 次。",
        "shape/dtype/device：benchmark_fn 本身只接 callable 与 Python int，没有张量 shape/dtype/device；比较真实推理时必须让各策略使用相同输入 shape、dtype 与 device。",
        "测试：运行 warmup=0、iters=3 的 counter 断言，确认 counter['n']==3 且 avg>=0.0；再自测 warmup=2 时总调用数为 iters+2。",
        "常见错误排查：counter 过大检查是否在计时后又调用 fn；平均值异常检查 start 位置和除数；GPU 数字虚低时检查异步执行与同步边界。"
      ]
    }),

    lesson({
      id: "inference-phase-summary",
      title: "拆开 prefill 与 decode：先求总时间，再解释 decode 占比",
      todo: "当前 Notebook 无 TODO 占位：独立复写 summarize_inference_result 的五字段返回契约",
      prerequisite: [
        "prefill_ms 是处理输入 prompt 的阶段耗时，decode_ms 是逐步生成输出 token 的阶段耗时；两者单位都必须是毫秒。",
        "total_ms 是两阶段相加；decode_share 是 decode_ms / total，不是百分数字符串。",
        "当 total 为 0 时不能做除法，模板约定 decode_share 返回 0.0。",
        "返回字典有五个固定键；耗时与显存保留 2 位小数，占比保留 3 位小数。"
      ],
      intuition: "总耗时只告诉你“整次有多慢”，阶段占比告诉你“时间主要花在哪里”。先保证两个阶段使用同一单位，再用 decode 除以总时间；这样短 prompt 与长生成任务可以得到不同的瓶颈画像。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>prefill_ms<br>读完整 prompt</span>
          <span>decode_ms<br>逐 token 生成</span>
          <strong>total_ms<br>两者相加</strong>
          <strong>decode_share<br>decode / total</strong>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>返回字典的字段合同</h4>
            <div class="adv-contract">
              <span><code>prefill_ms</code></span><strong>round(..., 2)</strong>
              <span><code>decode_ms</code></span><strong>round(..., 2)</strong>
              <span><code>total_ms</code></span><strong>round(prefill + decode, 2)</strong>
              <span><code>decode_share</code></span><strong>round(decode / total, 3)</strong>
              <span><code>peak_mem_mb</code></span><strong>round(..., 2)</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>零耗时保护</h4>
            <p>如果 prefill_ms 和 decode_ms 都是 0，total 也是 0。</p>
            <p>此时模板返回 decode_share=0.0，避免 ZeroDivisionError。这个分支表达“没有可分配的总时间”。</p>
          </section>
        </div>

        <div class="adv-checks">
          <span><strong>10 + 5</strong><br>total_ms = 15.0</span>
          <span><strong>5 / 15</strong><br>decode_share = 0.333</span>
          <span><strong>256.0</strong><br>peak_mem_mb 原值保留</span>
        </div>

        <div class="adv-callout">先相加未舍入的输入，再对结果 round。不要先把阶段占比乘 100；Notebook 测试期待 0.333，而不是 33.3。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用音频流水线练习阶段汇总", [
          "load_ms = 7.25",
          "process_ms = 2.75",
          "memory_mb = 96.126",
          "",
          "combined_ms = load_ms + process_ms",
          "process_share = process_ms / combined_ms if combined_ms else 0.0",
          "report = {",
          "    'load_ms': round(load_ms, 2),",
          "    'process_ms': round(process_ms, 2),",
          "    'combined_ms': round(combined_ms, 2),",
          "    'process_share': round(process_share, 3),",
          "    'memory_mb': round(memory_mb, 2),",
          "}"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；目标是独立复写 <code>summarize_inference_result</code>。</li>
            <li><code>load_ms/process_ms</code> -&gt; <code>prefill_ms/decode_ms</code>。</li>
            <li><code>combined_ms/process_share</code> -&gt; <code>total_ms/decode_share</code>。</li>
            <li><code>memory_mb</code> -&gt; <code>peak_mem_mb</code>；返回键必须使用 Notebook 规定的名字。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prefill_ms=10.0，decode_ms=5.0。",
        question: "按 Notebook 的三位小数约定，decode_share 是多少？",
        options: ["0.333", "0.5", "33.333"],
        answer: 0,
        revealNote: "总时间是 15.0，decode 占 5/15=0.333...，保留三位后为 0.333。"
      },
      checkpoint: checkpoint(
        "当 prefill_ms=0 且 decode_ms=0 时，模板应返回什么 decode_share？",
        ["0.0", "1.0", "抛出除零异常"],
        0,
        "模板用条件表达式保护 total=0 的情况，约定占比为 0.0。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：独立复写 summarize_inference_result，逐一写齐五个固定键、零总时间分支和 round 位数。",
        "shape/dtype/device：三个输入与返回值都是 Python 数值，没有 tensor shape/device；实际采集时要保证 prefill/decode 使用同一输入 shape、dtype、device 与毫秒单位。",
        "测试：运行 10.0、5.0、256.0 用例，确认 total_ms==15.0、decode_share==0.333、peak_mem_mb==256.0；另测全零输入。",
        "常见错误排查：占比为 33.3 时检查是否误乘 100；KeyError 时检查字典键拼写；除零时检查条件分支；总时间偏差时检查单位是否混用秒与毫秒。"
      ]
    }),

    lesson({
      id: "inference-fair-comparison",
      title: "把数字放进公平对照：一次只改变一个推理策略",
      todo: "当前 Notebook 无 TODO 占位：用 benchmark 与 summary 支撑策略 -&gt; 代价 -&gt; 收益对照",
      prerequisite: [
        "Notebook Markdown 要求固定同一模型与同一批输入，再比较 FlashAttention、PagedAttention、量化或 batch 策略。",
        "prefill 与 decode 的工作量不同，必须分阶段报告，不能只看 total_ms。",
        "延迟、吞吐、峰值显存是不同指标；较低延迟不自动意味着较高批量吞吐。",
        "短、中、长 prompt 会改变瓶颈，实验记录必须包含 batch、context length、生成 token 数与 cache 策略。"
      ],
      intuition: "性能表的每一行都像一次受控实验：其余条件冻结，只换一个策略。这样看到 total、decode_share 或 memory 变化时，才知道变化来自哪里，而不是输入长度或 batch 偷偷变了。",
      exampleHtml: `<div class="adv-course">
        <table class="adv-shapes">
          <thead><tr><th>必须记录</th><th>为什么</th><th>比较时如何固定</th></tr></thead>
          <tbody>
            <tr><td>batch / context / output tokens</td><td>决定实际工作量</td><td>同一组输入与生成长度</td></tr>
            <tr><td>prefill_ms / decode_ms</td><td>定位阶段瓶颈</td><td>使用相同阶段边界</td></tr>
            <tr><td>token/s</td><td>表示吞吐</td><td>相同 token 计数口径</td></tr>
            <tr><td>peak_mem_mb</td><td>表示资源上限</td><td>相同 device 与重置方式</td></tr>
          </tbody>
        </table>

        <div class="adv-grid three">
          <section class="adv-panel neutral"><h4>短输入</h4><p>启动与基础 prefill 成本更显眼。</p></section>
          <section class="adv-panel blue"><h4>中输入</h4><p>cache 策略开始影响复用收益。</p></section>
          <section class="adv-panel warn"><h4>长输入</h4><p>KV cache、分页和调度更可能成为瓶颈。</p></section>
        </div>

        <div class="adv-roadmap">
          <span><b>baseline</b>最简单 greedy/teacher-forcing</span>
          <span><b>替换一项</b>attention、cache、量化或 batch</span>
          <span><b>同口径测量</b>warmup、iters、阶段边界一致</span>
          <span><b>写结论</b>收益、代价、适用输入档位</span>
        </div>

        <div class="adv-callout">当前基础测试只验证两个 helper 的算术和调用次数，并未真正运行模型或比较四种策略。最终对照表需要你额外采集真实数据。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用独立结果表比较两种缓存设置", [
          "rows = [",
          "    {'cache': 'none', 'batch': 2, 'context': 128, 'total_ms': 30.0, 'tokens': 40},",
          "    {'cache': 'reuse', 'batch': 2, 'context': 128, 'total_ms': 24.0, 'tokens': 40},",
          "]",
          "",
          "for row in rows:",
          "    row['tokens_per_s'] = row['tokens'] / (row['total_ms'] / 1000)",
          "",
          "assert rows[0]['batch'] == rows[1]['batch']",
          "assert rows[0]['context'] == rows[1]['context']"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；<code>total_ms</code> -&gt; summary helper 的阶段总时间。</li>
            <li><code>tokens_per_s</code> -&gt; Markdown 要求记录的吞吐指标，需由 token 数和秒换算。</li>
            <li><code>batch/context</code> -&gt; 策略对照中必须固定的工作量变量。</li>
            <li><code>cache</code> -&gt; 逐项替换的推理策略；一次只改变这一项。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "策略 A 用 batch=1、context=64，策略 B 用 batch=8、context=1024，B 的 token/s 更高。",
        question: "可以直接断言 B 的 cache 策略更好吗？",
        options: ["不可以，工作量与 batch/context 同时变化，无法只归因于 cache", "可以，只要 token/s 更高就足够", "可以，因为 context 越长必然越公平"],
        answer: 0,
        revealNote: "多个变量同时改变时，吞吐差异可能来自 batch 或上下文长度，而不只是 cache。"
      },
      checkpoint: checkpoint(
        "为了比较量化前后推理性能，哪组条件最应该保持一致？",
        ["模型、输入、batch、生成长度、warmup/iters 与测量边界", "只保持字典键名字一致", "量化版使用更短 prompt 与更少输出 token"],
        0,
        "固定工作量和测量口径，才能把观察到的变化主要归因于量化策略。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：先跑通 benchmark_fn 与 summarize_inference_result，再为一种 baseline 和一种替换策略设计同字段结果行。",
        "shape/dtype/device：helper 本身处理标量；真实对照必须记录并固定输入 tensor shape、模型/输入 dtype 与 device，量化实验若 dtype 改变要把它列为策略差异。",
        "测试：保留模板对 total、decode_share、peak_mem、counter 和 avg 的全部断言；额外检查两行实验的 batch/context/output tokens 相同。",
        "常见错误排查：结果不可比先查工作量；token/s 异常查毫秒到秒换算；GPU 延迟虚低查同步；内存波动查峰值统计是否在每个策略前重置。"
      ]
    })
  ],

  "32": [
    lesson({
      id: "training-step-measurement",
      title: "测一个训练 step：先预热，再重置峰值，最后统一成毫秒",
      todo: "当前 Notebook 无 TODO 占位：独立复写 measure_train_step 的计时与峰值显存契约",
      prerequisite: [
        "train_step_fn 是一个无需参数的完整训练步函数；每调用一次，应该完成相同口径的工作。",
        "warmup 在正式测量前执行，用于稳定运行状态；Notebook 默认 warmup=2、iters=8。",
        "CUDA 可用时，正式计时前重置 peak memory，测量结束后读取 max_memory_allocated；CPU 路径约定 peak_mem_mb=0.0。",
        "perf_counter 的差值单位是秒；除以 iters 得单步秒数，再乘 1000 得 step_time_ms。"
      ],
      intuition: "训练计时要画清边界：预热先跑完；若在 GPU 上，清空旧的峰值记录；然后只包住固定次数的训练 step。最终返回一项时间和一项资源，让不同配置能用同一口径比较。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>1. warmup</b>调用 train_step_fn，不计时</span>
          <span><b>2. reset peak</b>仅 CUDA 可用时执行</span>
          <span><b>3. timed loop</b>正式执行 iters 次</span>
          <span><b>4. average</b>秒 / iters * 1000</span>
          <span><b>5. memory</b>字节 / 1024² 得 MB</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>CPU-first 分支</h4>
            <div class="adv-contract">
              <span>计时</span><strong>仍然正常执行并返回非负毫秒</strong>
              <span>峰值显存</span><strong>保持初始化值 0.0</strong>
              <span>基础测试</span><strong>只要求两个键存在且数值非负</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>CUDA 分支</h4>
            <div class="adv-contract">
              <span>正式测量前</span><strong>reset_peak_memory_stats()</strong>
              <span>正式测量后</span><strong>max_memory_allocated()</strong>
              <span>单位换算</span><strong>bytes / (1024 ** 2) -&gt; MB</strong>
            </div>
          </section>
        </div>

        <div class="adv-checks">
          <span><strong>counter == 2</strong><br>warmup=0,iters=2</span>
          <span><strong>step_time_ms</strong><br>键存在且 &gt;= 0</span>
          <span><strong>peak_mem_mb</strong><br>键存在且 &gt;= 0</span>
        </div>

        <div class="adv-callout">当前模板没有显式 CUDA synchronize。若 train_step_fn 在 GPU 上含异步工作，工程级精确计时要考虑同步；但复写 Notebook 基础函数时先保持它现有的调用与返回契约。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用数组更新任务练习平均 step 计时", [
          "import time",
          "",
          "state = {'steps': 0}",
          "def update_once():",
          "    state['steps'] += 1",
          "",
          "for _ in range(1):",
          "    update_once()",
          "",
          "begin = time.perf_counter()",
          "for _ in range(4):",
          "    update_once()",
          "avg_ms = (time.perf_counter() - begin) / 4 * 1000",
          "result = {'step_time_ms': round(avg_ms, 2), 'peak_mem_mb': 0.0}"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；目标是独立复写 <code>measure_train_step</code>。</li>
            <li><code>update_once</code> -&gt; <code>train_step_fn</code>；<code>1/4</code> -&gt; <code>warmup/iters</code>。</li>
            <li><code>avg_ms</code> -&gt; 返回字典的 <code>step_time_ms</code>。</li>
            <li>例子固定 CPU 峰值为 0.0；Notebook 还需按 CUDA 可用性执行 reset 与读取峰值。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "调用 measure_train_step(train_step, warmup=0, iters=2)。",
        question: "若 train_step 每次只让 counter 加 1，测试结束时 counter 应是多少？",
        options: ["2", "8", "10"],
        answer: 0,
        revealNote: "没有预热调用，正式循环正好执行 2 次，因此计数器为 2。"
      },
      checkpoint: checkpoint(
        "为什么 CUDA 峰值统计要在 warmup 之后重置？",
        ["避免把预热阶段的分配峰值混入正式测量", "把 GPU 张量自动转为 CPU", "让 iters 自动变成 0"],
        0,
        "重置后读取到的峰值更接近正式 timed loop 的资源上限。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：独立复写 measure_train_step，保持 warmup、条件重置、正式循环、平均毫秒和条件读取峰值的顺序。",
        "shape/dtype/device：helper 返回标量字典；真实 train_step 的输入 shape、模型/输入 dtype 与 device 必须在各配置间一致，CUDA 峰值只反映当前设备上的张量分配。",
        "测试：运行 warmup=0、iters=2 用例，确认 counter==2、两个键都存在且数值非负；CPU 上 peak_mem_mb=0.0 是模板预期。",
        "常见错误排查：counter 不对查循环边界；时间大 1000 倍查单位；CPU 调 CUDA API 报错查 is_available 分支；GPU 时间虚低查异步同步边界。"
      ]
    }),

    lesson({
      id: "training-delta-semantics",
      title: "读懂改前减改后：正差值才表示更快或更省",
      todo: "当前 Notebook 无 TODO 占位：独立复写 summarize_training_result 的差值方向与布尔字段",
      prerequisite: [
        "base_metrics 是优化前基线，tuned_metrics 是优化后配置；两者都应含 step_time_ms 与 peak_mem_mb。",
        "模板把 delta 定义为 base - tuned，而不是 tuned - base。",
        "若 tuned 更小，base - tuned 为正，因此 time_improved 或 memory_improved 为 True。",
        "差值保留 2 位小数；布尔判断使用未舍入的 delta > 0，恰好相等不算 improved。"
      ],
      intuition: "把 delta 理解成“省下了多少”：原来 120 ms，现在 98 ms，省下 22 ms，所以是正数。显存同理。负数不是程序坏了，而是优化后反而更慢或更占内存。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>正差值：改后更好</h4>
            <div class="adv-contract">
              <span>时间</span><strong>120 - 98 = +22 ms</strong>
              <span>显存</span><strong>8192 - 6144 = +2048 MB</strong>
              <span>布尔值</span><strong>time_improved=True，memory_improved=True</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>负差值：存在代价</h4>
            <div class="adv-contract">
              <span>时间</span><strong>100 - 130 = -30 ms</strong>
              <span>含义</span><strong>优化后每步慢 30 ms</strong>
              <span>布尔值</span><strong>time_improved=False</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>time_delta = base time - tuned time</code><span>正数表示每步省时</span></div>
          <div><b>2</b><code>mem_delta = base memory - tuned memory</code><span>正数表示峰值显存下降</span></div>
          <div><b>3</b><code>delta &gt; 0</code><span>严格大于 0 才标记 improved</span></div>
          <div><b>4</b><code>round(delta, 2)</code><span>统一报告精度</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>返回键</th><th>类型</th><th>正值/True 的含义</th></tr></thead>
          <tbody>
            <tr><td><code>step_time_delta_ms</code></td><td>float</td><td>tuned 每步更快</td></tr>
            <tr><td><code>peak_mem_delta_mb</code></td><td>float</td><td>tuned 峰值显存更低</td></tr>
            <tr><td><code>time_improved</code></td><td>bool</td><td>time_delta &gt; 0</td></tr>
            <tr><td><code>memory_improved</code></td><td>bool</td><td>mem_delta &gt; 0</td></tr>
          </tbody>
        </table>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用数据加载优化练习差值方向", [
          "before = {'load_ms': 18.4, 'ram_mb': 720.0}",
          "after = {'load_ms': 14.1, 'ram_mb': 760.0}",
          "",
          "load_saved = before['load_ms'] - after['load_ms']",
          "ram_saved = before['ram_mb'] - after['ram_mb']",
          "report = {",
          "    'load_delta_ms': round(load_saved, 2),",
          "    'ram_delta_mb': round(ram_saved, 2),",
          "    'load_improved': load_saved > 0,",
          "    'ram_improved': ram_saved > 0,",
          "}",
          "# 时间改善，但 ram_saved 为负，说明内存是代价"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；目标是独立复写 <code>summarize_training_result</code>。</li>
            <li><code>before/after</code> -&gt; <code>base_metrics/tuned_metrics</code>。</li>
            <li><code>load_saved</code> -&gt; <code>time_delta</code>；<code>ram_saved</code> -&gt; <code>mem_delta</code>。</li>
            <li>例子的字段名不同；返回 Notebook 时必须恢复四个规定键。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "baseline 每步 100 ms，tuned 每步 125 ms，模板计算 base - tuned。",
        question: "step_time_delta_ms 与 time_improved 应该是什么？",
        options: ["-25.0 与 False", "+25.0 与 True", "0.0 与 True"],
        answer: 0,
        revealNote: "100-125=-25，说明 tuned 反而慢了 25 ms，因此 time_improved 为 False。"
      },
      checkpoint: checkpoint(
        "若 base 与 tuned 的 peak_mem_mb 完全相等，memory_improved 是什么？",
        ["False", "True", "无法返回"],
        0,
        "mem_delta 等于 0，而模板使用严格条件 mem_delta > 0，所以结果是 False。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：独立复写 summarize_training_result，先算 base-tuned 的两个 delta，再返回两个 round 数值和两个严格大于 0 的布尔值。",
        "shape/dtype/device：输入是含 Python 数值的字典，输出是 float/bool，无 tensor shape/device；这些 metrics 必须来自相同输入 shape、dtype 与 device 的训练对照。",
        "测试：Notebook 当前基础测试未断言 summary；请保留示例 120/8192 对 98/6144，自查应得到 22.0、2048.0、True、True，并补测负差值与零差值。",
        "常见错误排查：符号反了检查减法顺序；布尔值不符检查是否用了 >=；KeyError 检查四个输入/输出键；结论冲突时确认时间和显存单位一致。"
      ]
    }),

    lesson({
      id: "training-controlled-analysis",
      title: "从两个 helper 到瓶颈结论：固定训练任务，一次只改一个变量",
      todo: "当前 Notebook 无 TODO 占位：组织 baseline/tuned 对照并区分数据、计算与显存瓶颈",
      prerequisite: [
        "Notebook 要求固定样本、optimizer、lr、batch size 与 accumulation 策略，除非其中某一项正是本轮唯一对照变量。",
        "gradient accumulation、activation checkpointing、offload 和更小 batch 会同时影响时间与显存，收益常伴随代价。",
        "measure_train_step 提供 step_time 与 peak memory，summary 只比较前后差值；它们不会自动判断瓶颈来源。",
        "要定位数据、计算或显存回收问题，还需结合 profiler、阶段计时和多次重复测量。"
      ],
      intuition: "先得到稳定 baseline，再只拨动一个开关。若显存下降但 step 变慢，这可能是 checkpointing 用重算换显存；若 GPU 等数据，问题可能在数据准备。性能分析不是只追求两个 True，而是解释每个变化的来源与代价。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>固定任务</b>同样本、模型、optimizer、lr</span>
          <span><b>建立 baseline</b>多次测 step time 与 peak memory</span>
          <span><b>只改一项</b>accumulation、checkpoint 或 offload</span>
          <span><b>计算 delta</b>时间与显存分别判断</span>
          <span><b>解释瓶颈</b>数据、计算、通信或回收</span>
        </div>

        <div class="adv-grid three">
          <section class="adv-panel neutral"><h4>数据瓶颈</h4><p>训练 step 等待 batch；先拆出数据准备时间。</p></section>
          <section class="adv-panel blue"><h4>计算瓶颈</h4><p>前向/反向占主导；看算子与利用率。</p></section>
          <section class="adv-panel warn"><h4>显存瓶颈</h4><p>OOM、频繁 offload 或回收；看峰值与传输。</p></section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>只改变的变量</th><th>常见收益</th><th>可能代价</th><th>应观察</th></tr></thead>
          <tbody>
            <tr><td>activation checkpointing</td><td>保存更少激活</td><td>反向时重算，step 变慢</td><td>peak memory 与 step time</td></tr>
            <tr><td>offload</td><td>设备显存下降</td><td>主机传输与等待</td><td>显存、传输、空闲间隙</td></tr>
            <tr><td>gradient accumulation</td><td>较小 micro-batch 也能模拟大 batch</td><td>每次 optimizer update 需要更多 micro-step</td><td>计时口径与吞吐</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">“step_time 改善”和“memory 改善”是两个独立判断。真实优化常是一正一负；报告应写清交换关系，而不是只挑有利指标。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用独立配置表保证一次只改 checkpoint", [
          "baseline_cfg = {",
          "    'batch': 4, 'lr': 1e-3, 'accum': 2, 'checkpoint': False",
          "}",
          "tuned_cfg = {",
          "    'batch': 4, 'lr': 1e-3, 'accum': 2, 'checkpoint': True",
          "}",
          "",
          "for key in ['batch', 'lr', 'accum']:",
          "    assert baseline_cfg[key] == tuned_cfg[key]",
          "",
          "base_metrics = {'step_time_ms': 44.0, 'peak_mem_mb': 900.0}",
          "tuned_metrics = {'step_time_ms': 53.0, 'peak_mem_mb': 610.0}"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li>本页无编号 TODO；<code>base_metrics/tuned_metrics</code> -&gt; summary helper 的两个输入。</li>
            <li><code>checkpoint</code> -&gt; Markdown 提议逐个切换的 activation checkpointing 变量。</li>
            <li><code>batch/lr/accum</code> 的相等断言 -&gt; 固定训练配置的公平性检查。</li>
            <li>例子展示显存改善、时间退化；需要同时报告两种 delta。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "开启 activation checkpointing 后，peak memory 降低，但 step time 上升。",
        question: "最合理的初步解释是什么？",
        options: ["用反向重算换取更少的激活保存，形成时间与显存交换", "两个指标必须同时改善，否则代码一定错", "说明 batch size 自动变成了 0"],
        answer: 0,
        revealNote: "Checkpointing 的典型机制就是少存激活、需要时重算，因此省显存同时增加计算是合理现象。"
      },
      checkpoint: checkpoint(
        "为了判断 offload 是否导致 step 变慢，最可靠的第一步是什么？",
        ["固定其余训练配置，只切换 offload，并用相同测量口径重复比较", "同时改 batch、lr 和模型大小", "只看一次运行的最终 loss"],
        0,
        "单变量、同口径、重复测量能最大程度隔离 offload 的影响。"
      ),
      homework: [
        "当前 Notebook 无 TODO 占位：用 measure_train_step 生成 baseline/tuned 指标，再用 summarize_training_result 写一行“收益 + 代价 + 可能瓶颈”结论。",
        "shape/dtype/device：两次实验必须固定训练输入 shape、模型与输入 dtype/device；改变 checkpoint/offload 时记录哪些张量位置变化，避免把配置漂移当优化收益。",
        "测试：先通过 counter、返回键和非负值的基础断言，再为 summary 增加正、负、零 delta 自检，并验证对照配置除目标变量外完全相同。",
        "常见错误排查：波动大时增加 warmup/iters 并重复；显存为 0 先确认是否在 CPU；GPU 时间虚低查同步；loss 行为变化查是否意外改变 batch、accum 或 optimizer step 口径。"
      ]
    })
  ]
};
