const { advancedStyles, checkpoint, code, lesson } = require("./advanced_lesson_helpers");

module.exports = {
  "16": [
    lesson({
      id: "grpo-group-relative-advantages",
      title: "先在组内比较：把奖励变成相对优势",
      todo: "TODO 1：计算组内相对优势",
      prerequisite: [
        "rewards 是每个候选答案得到的浮点奖励；group_ids 是同长度的整数标签，相同标签表示这些候选来自同一个 prompt。",
        "布尔 mask 既能从一维张量里选出某一组，也能把结果写回原来的样本位置。",
        "标准化要先减本组均值，再除本组标准差。这里使用总体标准差，也就是 std(unbiased=False)。",
        "eps 不是额外奖励，而是分母下限；当一组只有一个样本或所有奖励相同时，它负责避免除以 0。"
      ],
      intuition: "GRPO 不先问一个奖励在全世界算不算高，而是问它在同一 prompt 的候选答案里排得怎样。组内减均值把比较中心移到 0，除以标准差再把不同组的奖励尺度拉到可比范围。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>1. 按组取样本</b>group_ids 决定谁和谁比较</span>
          <span><b>2. 本组去中心</b>reward 减去本组 mean</span>
          <span><b>3. 本组缩放</b>除以 std 的安全下限</span>
          <span><b>4. 写回原位置</b>advantages 与 rewards 对齐</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>变量先对号入座</h4>
            <div class="adv-contract">
              <span>rewards</span><strong><code>[num_samples]</code> 浮点奖励</strong>
              <span>group_ids</span><strong><code>[num_samples]</code> 整数组号</strong>
              <span>mask</span><strong><code>[num_samples]</code> 布尔选择器</strong>
              <span>advantages</span><strong>同 shape、同位置的相对分数</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>两组不能混在一起算</h4>
            <p>组 A 的奖励是 10 和 12，组 B 的奖励是 1 和 3。虽然绝对数值不同，两组里更好的候选都应得到正优势，更差的候选都应得到负优势。</p>
            <div class="adv-flow">
              <span>A: 10, 12</span><strong>中心 11</strong><span>负, 正</span>
              <span>B: 1, 3</span><strong>中心 2</strong><span>负, 正</span>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>步骤</th><th>输入 shape</th><th>输出 shape</th><th>含义</th></tr></thead>
          <tbody>
            <tr><td>布尔选择</td><td><code>rewards [S]</code></td><td><code>group_rewards [G]</code></td><td>只看当前组的 G 个候选</td></tr>
            <tr><td>均值与标准差</td><td><code>group_rewards [G]</code></td><td>两个标量</td><td>每一组都有自己的中心和尺度</td></tr>
            <tr><td>写回</td><td><code>normalized [G]</code></td><td><code>advantages [S]</code></td><td>恢复为原样本顺序</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要对全部 rewards 一次性标准化。那样比较的是不同 prompt 之间的绝对奖励，破坏了“group relative”这四个字最核心的语义。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用班级内成绩练布尔分组与总体标准差", [
          "scores = torch.tensor([62., 82., 110., 130.])",
          "class_ids = torch.tensor([3, 3, 8, 8])",
          "z_scores = torch.zeros_like(scores)",
          "",
          "for class_id in class_ids.unique(sorted=True):",
          "    chosen = class_ids == class_id",
          "    local = scores[chosen]",
          "    centered = local - local.mean()",
          "    scale = local.std(unbiased=False).clamp_min(1e-5)",
          "    z_scores[chosen] = centered / scale"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>scores</code> -&gt; <code>rewards</code>：都保存待比较的浮点数。</li>
            <li><code>class_ids</code> -&gt; <code>group_ids</code>：都用相同整数标出同一组。</li>
            <li><code>chosen</code> -&gt; Notebook 的组内布尔 <code>mask</code>。</li>
            <li><code>z_scores</code> -&gt; <code>advantages</code>：先按输入创建，再分组写回。</li>
            <li><code>1e-5</code> -&gt; 函数参数 <code>eps</code>，不要在 TODO 中另写一个固定常量。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某一组只有两个奖励 2 和 4。减去均值 3 后得到 -1 和 1，总体标准差正好是 1。",
        question: "这一组写回 advantages 的两个值是什么？",
        options: ["[-1, 1]", "[2, 4]", "[-0.5, 0.5]"],
        answer: 0,
        revealNote: "组内标准化只使用本组统计量；中心化后再除以总体标准差 1，所以数值不再变化。"
      },
      checkpoint: checkpoint(
        "如果某组只有一个奖励 7，按 Notebook 的 eps 保护后，它的优势应是什么？",
        ["0，因为 centered 为 0", "7，因为只有一个样本", "NaN，因为标准差为 0"],
        0,
        "先减本组均值得到 0；标准差虽然为 0，但 clamp_min(eps) 让分母有效，因此结果是 0。"
      ),
      homework: [
        "完成 TODO 1：让 advantages 与 rewards 都是 [num_samples]，并保持 rewards 的 dtype 与 device；group_ids 应为同 device 的整数张量。",
        "先用 group_ids.unique(...) 遍历组，再通过 mask 选取、计算 mean 与 std(unbiased=False)，最后写回原位置；不要做全局标准化。",
        "运行 Notebook 测试并关注组 0 优势均值 atol=1e-6 的断言；若出现 NaN，检查 eps 是否用于分母下限，若均值不为 0，检查是否漏了按组计算或写回。"
      ]
    }),

    lesson({
      id: "grpo-ratio-and-clipped-surrogates",
      title: "再限制更新幅度：从 log-prob 到两个 surrogate",
      todo: "TODO 2：计算策略比率、surr1 与 surr2",
      prerequisite: [
        "log_probs_new 与 log_probs_old 都是每个候选答案的对数概率，shape 与 advantages 相同。",
        "对数相减再取 exp 等价于概率相除：exp(log p_new - log p_old) = p_new / p_old。",
        "ratio=1 表示新旧策略给出的概率相同；大于 1 表示新策略提高了该候选的概率。",
        "clip_range=0.2 时，裁剪区间是 [0.8, 1.2]，但最终是否真的受裁剪还要结合优势正负和 minimum 判断。"
      ],
      intuition: "优势告诉我们希望某个候选的概率往哪边走，ratio 告诉我们新策略已经走了多远。GRPO 同时算一条不设限的路线和一条把 ratio 限在安全区间的路线，下一课再从两条路线里选更保守的收益。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>ratio 的语义</h4>
            <div class="adv-contract">
              <span><code>ratio = 1.0</code></span><strong>新旧概率不变</strong>
              <span><code>ratio = 1.25</code></span><strong>新概率是旧概率的 1.25 倍</strong>
              <span><code>ratio = 0.70</code></span><strong>新概率降到旧概率的 70%</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>为什么在 log 空间做差</h4>
            <p>模型通常直接输出 log-prob。先相减再 exp，既贴合输入语义，也避免手工把两个很小的概率相除。</p>
            <div class="adv-flow">
              <span>log new</span><span>减 log old</span><strong>exp 得 ratio</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>ratio</code><span>度量新策略相对旧策略的概率变化</span></div>
          <div><b>2</b><code>surr1</code><span>原始 ratio 乘组内优势，保留实际更新幅度</span></div>
          <div><b>3</b><code>surr2</code><span>先把 ratio 限在安全区间，再乘同一个优势</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape</th><th>dtype/device</th><th>容易写错</th></tr></thead>
          <tbody>
            <tr><td><code>ratio</code></td><td><code>[S]</code></td><td>跟随 log_probs_new</td><td>忘记 exp，只留下 log-ratio</td></tr>
            <tr><td><code>surr1</code></td><td><code>[S]</code></td><td>浮点、可连回 new</td><td>误用 old log-prob 直接乘优势</td></tr>
            <tr><td><code>surr2</code></td><td><code>[S]</code></td><td>浮点、可求梯度</td><td>裁剪 advantages 而不是 ratio</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">clamp 只改变数值，不应该 detach。梯度必须仍能从后续 loss 回到 log_probs_new，这正是 Notebook 最后调用 backward() 要检查的路径。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用广告策略的新旧 log-score 练 ratio 与边界", [
          "after = torch.tensor([-0.7, -1.8], requires_grad=True)",
          "before = torch.tensor([-0.9, -1.5])",
          "preference = torch.tensor([0.6, -0.4])",
          "limit = 0.15",
          "",
          "multiplier = torch.exp(after - before)",
          "free_value = multiplier * preference",
          "safe_multiplier = torch.clamp(multiplier, 1 - limit, 1 + limit)",
          "safe_value = safe_multiplier * preference"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>after</code> / <code>before</code> -&gt; <code>log_probs_new</code> / <code>log_probs_old</code>。</li>
            <li><code>multiplier</code> -&gt; <code>ratio</code>，由 log-prob 差取指数得到。</li>
            <li><code>preference</code> -&gt; TODO 1 得到的 <code>advantages</code>。</li>
            <li><code>free_value</code> / <code>safe_value</code> -&gt; <code>surr1</code> / <code>surr2</code>。</li>
            <li><code>limit</code> -&gt; 函数参数 <code>clip_range</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "一个好候选的优势 A=2，当前 ratio=1.5，而 clip_range=0.2。",
        question: "两个 surrogate 分别是多少？",
        options: ["surr1=3.0，surr2=2.4", "surr1=2.4，surr2=3.0", "两者都是 2.0"],
        answer: 0,
        revealNote: "原始路线使用 1.5×2，裁剪路线先把 1.5 限到 1.2，再乘 2。"
      },
      checkpoint: checkpoint(
        "当 A=-1、ratio=0.5、clip_range=0.2 时，torch.min(surr1, surr2) 选中哪个值？",
        ["-0.8", "-0.5", "0.8"],
        0,
        "surr1=-0.5，裁剪 ratio 为 0.8 后 surr2=-0.8；minimum 选择更小、更保守的 -0.8。"
      ),
      homework: [
        "完成 TODO 2，并逐行确认 ratio、surr1、surr2 都是 [num_samples] 浮点张量，device 与 log_probs_new 一致，计算图没有被 detach。",
        "ratio 必须来自 new-old 的 log-prob 差再取 exp；clamp 的上下界使用 1-clip_range 与 1+clip_range，裁剪对象是 ratio。",
        "运行 Notebook 测试；若 backward 后 log_new.grad 为空，沿 loss -> minimum -> surrogate -> ratio -> log_probs_new 反查；常见错误是转成 Python 数值、使用 no_grad 或误把 new 替换成 old。"
      ]
    }),

    lesson({
      id: "grpo-conservative-loss-and-tests",
      title: "最后收成一个 loss：逐样本保守选择再求均值",
      todo: "TODO 3：计算最终 loss 并返回 loss、advantages",
      prerequisite: [
        "surr1 和 surr2 都是 [num_samples]，每个位置对应同一个候选答案的两种收益估计。",
        "torch.min(a, b) 在两个同 shape 张量之间逐元素取较小值；它不同于只对一个张量做全局 min。",
        "优化器执行梯度下降，所以要在希望最大化的 surrogate 前加负号，把问题改写为最小化 loss。",
        "mean 会把每个候选的值聚合成零维标量，同时对应批量平均的训练目标。"
      ],
      intuition: "对每个候选先在原始路线与裁剪路线之间拿较悲观的收益，再把所有候选平均。前面的负号只是在适配“优化器最小化 loss”这条约定，不会改变我们希望提高好候选、压低差候选的方向。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>逐位置比较</b>surr1[i] 对 surr2[i]</span>
          <span><b>保守收益</b>每个位置取较小值</span>
          <span><b>批量聚合</b>对所有候选求 mean</span>
          <span><b>改成最小化</b>最外层加负号</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>shape 收缩过程</h4>
            <div class="adv-flow">
              <span>surr1 [S]</span><span>surr2 [S]</span><span>minimum [S]</span><strong>loss []</strong>
            </div>
            <p><code>[]</code> 表示零维 Tensor。它仍然能保存 grad_fn，并不是 Python float。</p>
          </section>
          <section class="adv-panel warn">
            <h4>返回值有两个用途</h4>
            <div class="adv-contract">
              <span>loss</span><strong>用于 backward，必须是有限标量</strong>
              <span>advantages</span><strong>用于检查组内均值和理解样本方向</strong>
            </div>
          </section>
        </div>

        <div class="adv-checks">
          <span><strong>loss.ndim == 0</strong><br>聚合成标量</span>
          <span><strong>isfinite(loss)</strong><br>没有 NaN 或 Inf</span>
          <span><strong>组 0 mean 约为 0</strong><br>分组标准化正确</span>
          <span><strong>log_new.grad 存在</strong><br>计算图连通</span>
        </div>

        <div class="adv-callout">只通过现有断言还不等于所有数学细节都正确。测试只显式检查组 0 的均值，没有逐元素核对优势；你仍要确认每个组都使用自己的总体标准差。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用两个候选估计练逐元素保守聚合", [
          "estimate_a = torch.tensor([0.7, -0.3], requires_grad=True)",
          "estimate_b = torch.tensor([0.5, -0.6], requires_grad=True)",
          "",
          "pessimistic = torch.minimum(estimate_a, estimate_b)",
          "objective = -pessimistic.mean()",
          "objective.backward()",
          "print(objective.ndim)  # 0"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>estimate_a</code> / <code>estimate_b</code> -&gt; <code>surr1</code> / <code>surr2</code>。</li>
            <li><code>torch.minimum</code> -&gt; Notebook 可使用的逐元素 <code>torch.min</code> 语义。</li>
            <li><code>pessimistic.mean()</code> -&gt; 对全部候选聚合，而不是只取一个全局最小样本。</li>
            <li><code>objective</code> -&gt; 返回的 <code>loss</code>，应保留梯度连接。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "逐元素较小值是 [0.5, -0.6]。",
        question: "加负号并求均值后的 loss 是多少？",
        options: ["0.05", "-0.05", "0.55"],
        answer: 0,
        revealNote: "[0.5, -0.6] 的均值是 -0.05，最外层负号把它变成 0.05。"
      },
      checkpoint: checkpoint(
        "哪种写法能同时满足 Notebook 的标量断言并保留到 log_probs_new 的梯度？",
        ["对逐元素 minimum 求 mean，并保留 Tensor", "调用 .item() 后返回 Python 数字", "只返回 surr1[0]"],
        0,
        ".item() 会离开计算图；只取一个元素也丢掉了其余候选。逐元素选择后求均值才符合目标。"
      ),
      homework: [
        "完成 TODO 3：loss 必须是 shape [] 的浮点 Tensor，dtype/device 跟随 surrogate，并与 log_probs_new 保持可微连接；按函数签名同时返回 advantages。",
        "依次跑完 Notebook 测试的四项检查：标量、有限值、组 0 均值接近 0、backward 后 log_new.grad 非空。",
        "若 loss shape 是 [S]，检查是否漏了 mean；若 backward 报错，检查是否调用 item/detach；若 loss 非有限，回到 TODO 1 检查零标准差保护和 rewards dtype。"
      ]
    })
  ],

  "17": [
    lesson({
      id: "attention-backward-output-branch",
      title: "从 dout 分叉：先求 dV，再求 dP",
      todo: "TODO 1 + TODO 2：由 out = P @ V 求 dV 与 dP",
      prerequisite: [
        "前向保存了 q、k、v、p；backward 收到的 dout 是损失对 out 的上游梯度。",
        "Notebook 使用批量矩阵乘法，q、k、v、dout 都是 [B,N,d]，p 是 [B,N,N]。",
        "transpose(-2, -1) 只交换最后两个维度，保留最前面的 batch 维。",
        "矩阵乘法反向会沿两条支路传播：out 对 V 的支路产生 dV，out 对 P 的支路产生 dP。"
      ],
      intuition: "先别急着穿过 Softmax。把最后一步 out=P@V 单独看成一台混合器：P 决定每行取多少 V，dout 告诉我们输出要怎样改变。沿 V 插口倒推得到 dV，沿 P 插口倒推得到 dP。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>P [B,N,N]</span><span>V [B,N,d]</span><strong>out [B,N,d]</strong><span>dout [B,N,d]</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>回到 V 支路</h4>
            <div class="adv-flow">
              <span>P 转置 [B,N,N]</span><span>dout [B,N,d]</span><strong>dV [B,N,d]</strong>
            </div>
            <p>P 的最后两维交换，是因为每个 value 收集所有输出位置传回来的贡献。</p>
          </section>
          <section class="adv-panel blue">
            <h4>回到 P 支路</h4>
            <div class="adv-flow">
              <span>dout [B,N,d]</span><span>V 转置 [B,d,N]</span><strong>dP [B,N,N]</strong>
            </div>
            <p>每个注意力概率都需要知道：改变它会怎样改变 d 个输出特征。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>目标</th><th>左因子</th><th>右因子</th><th>结果</th></tr></thead>
          <tbody>
            <tr><td><code>dV</code></td><td><code>P^T [B,N,N]</code></td><td><code>dout [B,N,d]</code></td><td><code>[B,N,d]</code></td></tr>
            <tr><td><code>dP</code></td><td><code>dout [B,N,d]</code></td><td><code>V^T [B,d,N]</code></td><td><code>[B,N,N]</code></td></tr>
          </tbody>
        </table>

        <div class="adv-callout">这里的转置不是把整个三维张量倒序。使用 transpose(-2, -1) 才能固定 B，只交换参与矩阵乘法的两个轴。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用批量加权汇总练两条反向支路", [
          "weights = torch.randn(3, 4, 6)   # [batch, query, item]",
          "items = torch.randn(3, 6, 5)     # [batch, item, feature]",
          "upstream = torch.randn(3, 4, 5)  # 与输出同 shape",
          "",
          "grad_items = torch.matmul(weights.transpose(-2, -1), upstream)",
          "grad_weights = torch.matmul(upstream, items.transpose(-2, -1))",
          "print(grad_items.shape)    # [3, 6, 5]",
          "print(grad_weights.shape)  # [3, 4, 6]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>weights</code> -&gt; <code>p</code>，都是归一化后的混合权重。</li>
            <li><code>items</code> -&gt; <code>v</code>，都是被加权汇总的特征。</li>
            <li><code>upstream</code> -&gt; <code>dout</code>。</li>
            <li><code>grad_items</code> -&gt; TODO 1 的 <code>dv</code>；<code>grad_weights</code> -&gt; TODO 2 的 <code>dp</code>。</li>
            <li>例子 query 数与 item 数不同，能更清楚看出转置究竟发生在哪个轴。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "dout 是 [B,N,d]，v.transpose(-2,-1) 是 [B,d,N]。",
        question: "两者做批量矩阵乘法后，dp 的 shape 是什么？",
        options: ["[B,N,N]", "[B,d,d]", "[N,B,N]"],
        answer: 0,
        revealNote: "中间的 d 被消去，两个 N 分别来自输出位置和 value 位置。"
      },
      checkpoint: checkpoint(
        "为了得到与 v 同 shape 的 dv，应该转置哪个张量的最后两维？",
        ["p", "dout", "v"],
        0,
        "P^T @ dout 把输出位置的梯度收集回每个 value；dout 或 v 的转置无法得到所需乘法契约。"
      ),
      homework: [
        "完成 TODO 1/2：dv 为 [B,N,d]，dp 为 [B,N,N]；dtype/device 必须与 float64 的 q、k、v、dout 一致。",
        "每个 matmul 前写出最后两维，确认只用 transpose(-2,-1) 交换矩阵轴，没有打乱 batch 维。",
        "Notebook 先用 allclose 检查固定的 forward，随后 gradcheck 才检查你的 backward；若 shape 报错，先分别打印 p、v、dout、dv、dp 的 shape。"
      ]
    }),

    lesson({
      id: "attention-backward-softmax",
      title: "穿过 Softmax：用逐行修正代替巨大雅可比",
      todo: "TODO 3：由 dP 求 dp_mul_p、row_sum 与 dS",
      prerequisite: [
        "p 的每一行都是 scores 对最后一维做 Softmax 得到的概率，行内元素和为 1。",
        "dp 是损失对这些概率的梯度，shape 与 p 都是 [B,N,N]。",
        "逐元素乘法 dp*p 不会收缩维度；sum(dim=-1) 才会聚合每一行。",
        "keepdim=True 把行统计量保留成 [B,N,1]，这样它能沿最后一维广播回 [B,N,N]。"
      ],
      intuition: "Softmax 的一行概率彼此牵连：抬高一个 score 会挤压同一行的其他概率。所以不能只做 dp*p，还要减掉这整行共同承担的加权平均影响。row_sum 就是这条“行内联动”的修正量。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>dp_mul_p</code><span>每个概率先乘自己收到的上游信号</span></div>
          <div><b>2</b><code>row_sum</code><span>最后一维求和，得到每一行的共同修正</span></div>
          <div><b>3</b><code>ds</code><span>每个位置减去行修正，再乘回自己的概率</span></div>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>广播契约</h4>
            <div class="adv-contract">
              <span><code>dp, p</code></span><strong><code>[B,N,N]</code></strong>
              <span><code>row_sum</code></span><strong><code>[B,N,1]</code></strong>
              <span><code>dp-row_sum</code></span><strong><code>[B,N,N]</code></strong>
              <span><code>ds</code></span><strong><code>[B,N,N]</code></strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>一个重要自检</h4>
            <p>Softmax 对整行同时加同一个常数不会变化，因此正确的 dS 每一行求和应接近 0。</p>
            <div class="adv-flow">
              <span>dS 一行</span><span>沿最后维求和</span><strong>约等于 0</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">最常见错误是 row_sum 得到 [B,N]。它没有保留最后的单例维，与 [B,N,N] 对齐时可能广播失败，或者在某些恰好相等的维度上悄悄沿错方向。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用一组分类权重练 Softmax 的向量-雅可比积", [
          "scores = torch.tensor([[0.4, -0.1, 1.2]], dtype=torch.float64)",
          "weights = torch.softmax(scores, dim=-1)",
          "grad_weights = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)",
          "",
          "weighted = grad_weights * weights",
          "baseline = weighted.sum(dim=-1, keepdim=True)",
          "grad_scores = weights * (grad_weights - baseline)",
          "print(grad_scores.sum(dim=-1))  # 接近 0"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>weights</code> -&gt; 前向保存的 <code>p</code>。</li>
            <li><code>grad_weights</code> -&gt; TODO 2 得到的 <code>dp</code>。</li>
            <li><code>weighted</code> -&gt; <code>dp_mul_p</code>。</li>
            <li><code>baseline</code> -&gt; <code>row_sum</code>，必须沿 dim=-1 且 keepdim=True。</li>
            <li><code>grad_scores</code> -&gt; <code>ds</code>，shape 不变。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "如果某一行 dp 的三个值完全相同，例如 [5,5,5]，改变任一 score 只是在概率间重新分配，但所有位置收到同样的收益信号。",
        question: "这时该行 ds 应是什么？",
        options: ["全 0", "等于 p", "全 5"],
        answer: 0,
        revealNote: "row_sum 等于 5×sum(p)=5，所以 dp-row_sum 全为 0；共同平移式信号不会改变 Softmax 前的 score。"
      },
      checkpoint: checkpoint(
        "p 和 dp 是 [2,8,8] 时，正确的 row_sum shape 是什么？",
        ["[2,8,1]", "[2,8]", "[2,1,8]"],
        0,
        "沿最后一维归约并保留维度，才能把每个 query 行的修正量广播到该行所有 key。"
      ),
      homework: [
        "完成 TODO 3：dp_mul_p、ds 是 [B,N,N]，row_sum 是 [B,N,1]；全部保持 float64 和原 device，不要新建 CPU 常量张量。",
        "严格沿 dim=-1 聚合并使用 keepdim=True；可临时检查 ds.sum(dim=-1) 是否在浮点误差内接近 0。",
        "若 gradcheck 报 Softmax 相关梯度不一致，排查是否漏乘 p、row_sum 是否来自 dp*p，以及减法括号是否是 p*(dp-row_sum)。"
      ]
    }),

    lesson({
      id: "attention-backward-qk-and-gradcheck",
      title: "回到 Q 和 K：补上 scale，交给 gradcheck 验收",
      todo: "TODO 4：由 dS 求 dQ、dK，并按顺序返回梯度",
      prerequisite: [
        "前向 scores=(q @ k^T)*scale，其中 scale=1/sqrt(d)，所以这条常数因子也必须出现在反向。",
        "ds 是 [B,N,N]，q 和 k 都是 [B,N,d]。",
        "对 q 支路使用 ds @ k；对 k 支路要先交换 ds 的最后两维，再乘 q。",
        "自定义 Function 的 backward 返回值必须与 forward 输入顺序一一对应：q、k、v。"
      ],
      intuition: "现在 dS 已经表示每个 query-key 分数应该怎样变化。固定 K 看，dS 把变化汇回每个 query 得到 dQ；固定 Q 看，需要把 query-key 方向反过来，才能汇回每个 key 得到 dK。最后别把前向的缩放因子落在路上。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>Q 支路</h4>
            <div class="adv-flow">
              <span>dS [B,N,N]</span><span>K [B,N,d]</span><strong>dQ [B,N,d]</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>K 支路</h4>
            <div class="adv-flow">
              <span>dS^T [B,N,N]</span><span>Q [B,N,d]</span><strong>dK [B,N,d]</strong>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span>前向常数</span><strong><code>scale = 1 / sqrt(d)</code></strong>
          <span>影响范围</span><strong>dQ 与 dK 都乘 scale；dV 不经过 scores，不乘 scale</strong>
          <span>返回顺序</span><strong><code>dq, dk, dv</code> 对齐 <code>q, k, v</code></strong>
        </div>

        <div class="adv-checks">
          <span><strong>forward allclose</strong><br>固定前向等于原生实现</span>
          <span><strong>float64</strong><br>数值差分更精确</span>
          <span><strong>eps=1e-6</strong><br>用微小扰动估计导数</span>
          <span><strong>atol=1e-4</strong><br>比较解析与数值梯度</span>
        </div>

        <div class="adv-callout">gradcheck 不只看“有没有梯度”，而是逐个输入比较你的解析反向与有限差分。漏 scale、转置错轴或返回顺序错，通常都会被它抓住。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用非方形双线性打分练左右输入梯度", [
          "left = torch.randn(2, 3, 5, dtype=torch.float64)",
          "right = torch.randn(2, 7, 5, dtype=torch.float64)",
          "grad_score = torch.randn(2, 3, 7, dtype=torch.float64)",
          "factor = 0.25",
          "",
          "grad_left = torch.matmul(grad_score, right) * factor",
          "grad_right = torch.matmul(grad_score.transpose(-2, -1), left) * factor",
          "print(grad_left.shape, grad_right.shape)  # [2,3,5], [2,7,5]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>left</code> / <code>right</code> -&gt; <code>q</code> / <code>k</code>。</li>
            <li><code>grad_score</code> -&gt; TODO 3 得到的 <code>ds</code>。</li>
            <li><code>grad_left</code> / <code>grad_right</code> -&gt; <code>dq</code> / <code>dk</code>。</li>
            <li><code>factor</code> -&gt; ctx 中保存的 <code>scale</code>；Notebook 的具体数值由 d 决定。</li>
            <li>非方形例子显示：K 支路转置的是 grad_score，不是把 q 或 k 的 batch 轴移动。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "前向 score 除以 sqrt(d)，但 backward 里的 dq、dk 都忘了乘 scale。",
        question: "forward allclose 与 gradcheck 最可能分别怎样？",
        options: ["forward 通过，gradcheck 失败", "两者都通过", "forward 失败，gradcheck 通过"],
        answer: 0,
        revealNote: "forward 代码本来就包含 scale，不受 TODO 4 影响；gradcheck 会发现解析梯度比真实梯度大 sqrt(d) 倍。"
      },
      checkpoint: checkpoint(
        "CustomAttention.forward(ctx, q, k, v) 对应的 backward 正确返回顺序是什么？",
        ["dq, dk, dv", "dv, dk, dq", "ds, dp, dout"],
        0,
        "Autograd 按 forward 的输入位置接收梯度，返回顺序错会把数值正确的梯度交给错误变量。"
      ),
      homework: [
        "完成 TODO 4：dq、dk 都是 [B,N,d]、float64、与输入同 device，并各自乘一次 ctx.scale；dv 不乘该 scale。",
        "确认 backward 最后按 q、k、v 的输入顺序返回 dq、dk、dv，然后运行 forward allclose 与完整 gradcheck。",
        "gradcheck 失败时按顺序查：TODO 1/2 的转置、TODO 3 的 keepdim 与括号、TODO 4 的 scale 和返回顺序；不要通过放宽 atol 掩盖公式错误。"
      ]
    })
  ],

  "18": [
    lesson({
      id: "relu-backward-gate",
      title: "ReLU 反向是一扇门：正数放行，其余归零",
      todo: "TODO 1：构造 ReLU 反向门控 mask",
      prerequisite: [
        "grad_out 是后续计算传回来的上游梯度，x 是 ReLU 前的输入；二者在 Notebook 中 shape 都是 [5]。",
        "比较 x>0 会得到布尔 Tensor：正输入位置是 True，0 和负输入位置是 False。",
        "PyTorch 在 x=0 处采用导数 0，因此判断条件必须是严格大于 0。",
        "乘法前把布尔 mask 转成 grad_out.dtype，可让返回梯度保持期望的浮点类型和 device。"
      ],
      intuition: "前向 ReLU 把非正数挡住。反向时不需要再算一次复杂函数，只要回看原输入 x：当时通过门的正数位置保留 grad_out，没通过的位置把梯度变成 0。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>看原输入 x</b>不是看 grad_out 的正负</span>
          <span><b>生成布尔门</b>x 正数位置为 True</span>
          <span><b>对齐 dtype</b>布尔门转成浮点门</span>
          <span><b>逐元素相乘</b>输出 shape 不变</span>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>x</th><th>ReLU 前向</th><th>局部导数</th><th>grad_out=3 时 grad_in</th></tr></thead>
          <tbody>
            <tr><td><code>-2</code></td><td><code>0</code></td><td><code>0</code></td><td><code>0</code></td></tr>
            <tr><td><code>0</code></td><td><code>0</code></td><td><code>0</code></td><td><code>0</code></td></tr>
            <tr><td><code>4</code></td><td><code>4</code></td><td><code>1</code></td><td><code>3</code></td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>正确的数据流</h4>
            <div class="adv-flow">
              <span>x [5]</span><span>mask [5]</span><span>grad_out [5]</span><strong>grad_in [5]</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>两个常见误会</h4>
            <p>门由 x 决定，不由 grad_out 决定；上游梯度可以是负数，正输入位置仍应原样放行它。</p>
            <p>使用 x&gt;=0 会让 x=0 的梯度变成 1，与 Notebook 中 F.relu 的自动求导不一致。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用传感器阈值门练逐元素反向", [
          "preactivation = torch.tensor([-1.5, 0.0, 2.5])",
          "flow = torch.tensor([4.0, -3.0, 0.5])",
          "",
          "gate = (preactivation > 0).to(flow.dtype)",
          "grad_input = flow * gate",
          "print(grad_input)  # [0.0, -0.0, 0.5]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>preactivation</code> -&gt; <code>x</code>，门的开关只由它决定。</li>
            <li><code>flow</code> -&gt; <code>grad_out</code>，它是从后面传来的梯度。</li>
            <li><code>gate</code> -&gt; TODO 1 的 <code>mask</code>。</li>
            <li><code>grad_input</code> -&gt; 函数当前 return 表达式的结果。</li>
            <li>例子故意在正输入处放了 0.5，说明门只决定放行或阻断，不把梯度强行改成 1。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "x=[2,-1]，grad_out=[-4,7]。第一个上游梯度虽然是负数，但对应 x 是正数。",
        question: "relu_backward 应返回什么？",
        options: ["[-4,0]", "[0,7]", "[4,0]"],
        answer: 0,
        revealNote: "门看 x：第一个位置放行原值 -4，第二个位置阻断为 0。"
      },
      checkpoint: checkpoint(
        "为什么 mask 使用 x>0 而不是 x>=0？",
        ["要与 PyTorch 在 x=0 处选择导数 0 的约定一致", "因为 >= 不能用于 Tensor", "为了把 mask 变成 float64"],
        0,
        "两种比较都能用于 Tensor，但边界导数约定不同；测试中的 x 包含 0，会检查出这个差别。"
      ),
      homework: [
        "完成 TODO 1：mask 与 x/grad_out 都是 [5]，转成 grad_out.dtype，并保持原 device；返回梯度也应是 [5] 浮点 Tensor。",
        "运行测试中包含 -2、-0.5、0、1、3 的边界样本，确认手写结果与 F.relu 自动求导 allclose。",
        "若 x=0 位置不一致，检查是否误写 >=；若 dtype 异常，检查是否忘了 .to(grad_out.dtype)；若负上游梯度被抹掉，检查是否错误地用 grad_out 构造 mask。"
      ]
    }),

    lesson({
      id: "softmax-cross-entropy-backward",
      title: "分类损失反向：概率减 one-hot，再除 batch",
      todo: "TODO 2：计算 probs、one_hot、loss 与 grad",
      prerequisite: [
        "logits 是 [B,C] 浮点分数，B 是样本数、C 是类别数；labels 是 [B] 的整数类别下标。",
        "Softmax 必须沿类别维 dim=-1，使每个样本一行的概率和为 1。",
        "scatter_ 需要 labels.unsqueeze(1) 把索引从 [B] 变成 [B,1]，再沿 dim=1 写入 one-hot。",
        "Notebook 的 loss 对 batch 使用 mean，所以 logits 梯度不仅是 probs-one_hot，还要除以 B。"
      ],
      intuition: "每个样本先得到一行类别概率。one-hot 在正确类别放 1，其他位置放 0；probs-one_hot 就是在说“预测比目标多出来多少”。因为最终 loss 是 B 个样本的平均，每个样本对总 loss 只贡献 1/B。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>probs</code><span>每行 logits 沿类别维变成概率</span></div>
          <div><b>2</b><code>one_hot</code><span>把 labels 转成与 probs 同 shape 的目标表</span></div>
          <div><b>3</b><code>loss</code><span>取正确类别负对数，先按类求和再按 batch 求均值</span></div>
          <div><b>4</b><code>grad</code><span>probs 减 one_hot，并除以 batch size</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>Notebook shape</th><th>dtype</th><th>语义</th></tr></thead>
          <tbody>
            <tr><td><code>logits</code></td><td><code>[2,3]</code></td><td>float</td><td>两个样本、三个类别的原始分数</td></tr>
            <tr><td><code>labels</code></td><td><code>[2]</code></td><td>long</td><td>每个样本的正确类别下标</td></tr>
            <tr><td><code>probs, one_hot, grad</code></td><td><code>[2,3]</code></td><td>跟随 logits</td><td>逐类别概率、目标和梯度</td></tr>
            <tr><td><code>loss</code></td><td><code>[]</code></td><td>跟随 logits</td><td>批量平均标量</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>每行梯度和应接近 0</h4>
            <p>probs 一行和为 1，one_hot 一行和也为 1，所以两者相减后行和为 0；再除 B 仍为 0。</p>
          </section>
          <section class="adv-panel warn">
            <h4>测试覆盖与盲点</h4>
            <p>Notebook 直接把 manual_grad 与 F.cross_entropy 的自动梯度比较，但只打印手写 loss，没有断言它等于 ce。loss 仍要按给定公式正确实现。</p>
          </section>
        </div>

        <div class="adv-callout">log 里的小常数只用于避免概率极小时出现 log(0)。它加在 probs 上，不加在 logits 上；grad 仍使用化简后的 (probs-one_hot)/B。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用三条天气样本练四分类交叉熵", [
          "scores = torch.tensor([[1.2, 0.1, -0.4, 0.7],",
          "                       [0.0, 0.5, 1.4, -0.2],",
          "                       [-0.5, 1.1, 0.3, 0.2]])",
          "targets = torch.tensor([3, 2, 1])",
          "",
          "dist = torch.softmax(scores, dim=-1)",
          "indicator = torch.zeros_like(dist)",
          "indicator.scatter_(1, targets.unsqueeze(1), 1.0)",
          "per_item = -(indicator * torch.log(dist + 1e-12)).sum(dim=1)",
          "objective = per_item.mean()",
          "dscore = (dist - indicator) / scores.size(0)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>scores</code> / <code>targets</code> -&gt; <code>logits</code> / <code>labels</code>。</li>
            <li><code>dist</code> -&gt; <code>probs</code>，Softmax 沿类别维。</li>
            <li><code>indicator</code> -&gt; <code>one_hot</code>，先 zeros_like 再 scatter_。</li>
            <li><code>objective</code> -&gt; <code>loss</code>；<code>dscore</code> -&gt; <code>grad</code>。</li>
            <li>例子 batch=3、类别数=4；Notebook 改成 batch=2、类别数=3，写法不依赖固定数字。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某个样本的预测概率是 [0.7,0.2,0.1]，正确类别是第 0 类，batch size=2。",
        question: "这个样本对 logits 的批量平均梯度行是什么？",
        options: ["[-0.15,0.10,0.05]", "[-0.30,0.20,0.10]", "[0.35,0.10,0.05]"],
        answer: 0,
        revealNote: "先做 probs-one_hot 得 [-0.3,0.2,0.1]，再除以 B=2。"
      },
      checkpoint: checkpoint(
        "labels 是 [B] 时，为什么 scatter_ 前要 labels.unsqueeze(1)？",
        ["把索引变成 [B,1]，让每一行沿类别维写入一个 1", "把标签转成浮点概率", "把 batch size 翻倍"],
        0,
        "scatter_ 沿 dim=1 写类别位置，需要每行提供一个二维列索引；unsqueeze 不改变标签值或 dtype。"
      ),
      homework: [
        "完成 TODO 2：probs、one_hot、grad 都是 [B,C]，loss 是 shape []；labels 保持 long，其他张量跟随 logits 的浮点 dtype 与 device。",
        "按顺序实现 Softmax、zeros_like+scatter_、逐样本负对数后 mean、以及除以 logits.size(0) 的梯度；不要把 batch 大小写死成 2。",
        "运行 Notebook 测试，让 manual_grad 与 F.cross_entropy 自动梯度做 atol=1e-6 比较；若差固定为 B 倍，检查是否漏除 batch，若类别错位，检查 dim 和 unsqueeze，若 loss 为 Inf，检查 log 的小常数。"
      ]
    })
  ],

  "19": [
    lesson({
      id: "checkpoint-recompute-state-flow",
      title: "先看显存账本：少保存激活，反向时再重算",
      todo: "核心机制：理解 run_with_checkpointing 要改变什么",
      prerequisite: [
        "训练前向不仅产生最终输出，还会为 backward 保存中间激活；层数、batch、序列长度和隐藏维度都会放大这部分显存。",
        "checkpoint 不删除模型参数，也不关闭梯度；它主要减少被长期保存的中间激活。",
        "反向走到被 checkpoint 包裹的区域时，会从保存的输入重新执行该区域前向，恢复求梯度需要的值。",
        "Activation Offload 是把激活搬到 CPU 等存储层，checkpointing 是丢掉后重算；本 Notebook 的唯一 TODO 实现的是后者。"
      ],
      intuition: "普通训练像把每一步草稿都留在 GPU 上，反向直接查草稿。Checkpoint 只留下某段的入口 x，段内草稿前向后可以不长期保留；反向需要时，用入口再算一遍。省下的是空间，付出的是重复计算时间。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>普通执行的状态流</h4>
            <div class="adv-flow">
              <span>block 输入</span><span>Norm 激活</span><span>FFN 扩张激活</span><strong>block 输出</strong>
            </div>
            <p>为了反向，段内多个中间值会在前向结束后继续占据显存。</p>
          </section>
          <section class="adv-panel good">
            <h4>Checkpoint 的状态流</h4>
            <div class="adv-flow">
              <strong>保留 block 输入</strong><span>段内值不长期保留</span><span>反向时重算</span><strong>计算参数梯度</strong>
            </div>
            <p>函数输出数值和梯度目标不变，变化的是保存与重算策略。</p>
          </section>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>Checkpointing</h4>
            <div class="adv-contract">
              <span>动作</span><strong>丢弃部分中间激活</strong>
              <span>取回方式</span><strong>反向时重新计算</strong>
              <span>主要代价</span><strong>额外计算时间</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>Activation Offload</h4>
            <div class="adv-contract">
              <span>动作</span><strong>把激活搬离 GPU</strong>
              <span>取回方式</span><strong>需要时再传回 GPU</strong>
              <span>主要代价</span><strong>设备间传输时间</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>Notebook 对象</th><th>shape</th><th>设备</th><th>checkpoint 后是否改变</th></tr></thead>
          <tbody>
            <tr><td><code>x_input</code></td><td><code>[2,2048,2048]</code></td><td>CUDA</td><td>输出接口不改变</td></tr>
            <tr><td>FFN 中间激活</td><td><code>[2,2048,8192]</code></td><td>CUDA</td><td>不再全部长期保存</td></tr>
            <tr><td>block 输出</td><td><code>[2,2048,2048]</code></td><td>CUDA</td><td>继续传给下一 block</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">本题以 Transformer Block 为 checkpoint 单位。不要把它误解成每个 Linear、ReLU 都要单独包裹，也不要在 TODO 中实现 CPU offload。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用手工重算理解为何只需保留段入口", [
          "def expensive_segment(seed):",
          "    hidden = torch.sin(seed) * 3",
          "    return hidden.square() + seed",
          "",
          "checkpoint_input = torch.tensor([0.2, 0.8], requires_grad=True)",
          "first_output = expensive_segment(checkpoint_input)",
          "# 概念上丢掉段内 hidden；反向需要时从 checkpoint_input 重算",
          "restored_output = expensive_segment(checkpoint_input)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>expensive_segment</code> -&gt; 一个 <code>block</code> 的前向函数。</li>
            <li><code>checkpoint_input</code> -&gt; 当前循环中的 <code>x</code>。</li>
            <li><code>hidden</code> -&gt; block 内 Norm、ReLU、FFN 产生的中间激活。</li>
            <li>手工第二次调用 -&gt; checkpoint 在 backward 期间自动完成的 recomputation。</li>
            <li>Notebook TODO 不手工写第二次前向，而是交给 PyTorch 的 checkpoint API 管理。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "同一个 block 的参数、输入和随机状态都不变，只把普通前向换成 checkpoint 管理。",
        question: "最终 block 输出的 shape 应怎样变化？",
        options: ["不变，仍与该 block 原输出一致", "隐藏维度自动减半", "输出被搬到 CPU"],
        answer: 0,
        revealNote: "checkpoint 改的是 Autograd 保存中间值的策略，不改 block 的数学函数、shape 或输出设备。"
      },
      checkpoint: checkpoint(
        "哪一句最准确地区分 checkpointing 与 activation offload？",
        ["前者用重计算换显存，后者用设备间搬运换 GPU 显存", "两者都会停止参数求梯度", "两者都要求把模型权重移到 CPU"],
        0,
        "本 Notebook 实现的是重计算；offload 只作为工程对比出现，没有对应 TODO。"
      ),
      homework: [
        "先在 TODO 旁标注：循环进入和离开每个 block 的 x 都是 [B,S,D]，dtype 不变、device 仍是 CUDA，requires_grad 路径必须保留。",
        "用一句话解释为什么 checkpoint 后输出数值不应改变，以及为什么 backward 会增加计算；确认没有把 offload 或 detach 混入实现。",
        "Notebook 的显存测试仅在 CUDA 可用时执行；若被跳过，先做 CPU 小模型的输出/梯度自检。若参数梯度为空，检查是否用了 no_grad/detach；若输出 device 变成 CPU，说明误做了 offload。"
      ]
    }),

    lesson({
      id: "checkpoint-api-loop-and-gpu-test",
      title: "把每个 block 交给 checkpoint，并读懂 GPU 测试",
      todo: "TODO：在循环内调用 checkpoint，关注 use_reentrant=False",
      prerequisite: [
        "checkpoint 的第一个位置参数是要执行的函数或 nn.Module，后面的位置参数是传给它的输入。",
        "一个 SimpleTransformerBlock 接收 x 并返回新的 x，所以包装后的返回值必须重新赋给循环状态。",
        "Notebook 已从 torch.utils.checkpoint 导入 checkpoint，不需要改 import，也不需要自己写 autograd.Function。",
        "use_reentrant=False 是当前 Notebook 明确要求的调用模式；应作为关键字参数传入，不能误传给 block.forward。"
      ],
      intuition: "循环仍然是一条 block 接 block 的流水线。唯一变化是每次不直接调用 block(x)，而是让 checkpoint 代为调用，然后把结果继续交给下一个 block。赋值不能漏，否则后面的层拿到的仍是旧状态。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>取当前 block</b>来自 ModuleList 循环</span>
          <span><b>包装调用</b>函数是 block，输入是 x</span>
          <span><b>选择模式</b>use_reentrant=False</span>
          <span><b>更新流水状态</b>返回值重新赋给 x</span>
        </div>

        <div class="adv-contract">
          <span>函数位置</span><strong>当前 <code>block</code>，可直接调用的 nn.Module</strong>
          <span>输入位置</span><strong>当前 <code>x [B,S,D]</code></strong>
          <span>关键字</span><strong><code>use_reentrant=False</code></strong>
          <span>返回值</span><strong>下一轮循环使用的新 <code>x [B,S,D]</code></strong>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>GPU 测试真正测了什么</h4>
            <div class="adv-checks">
              <span>普通路径 forward + backward</span>
              <span>记录 mem_normal 峰值</span>
              <span>checkpoint 路径 forward + backward</span>
              <span>比较 mem_ckpt 与 mem_normal</span>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>测试有两个盲点</h4>
            <p>没有 CUDA 时函数直接 return，“忽略测试”不代表 TODO 正确。</p>
            <p>显存比较使用 if 打印而非 assert；空循环甚至可能因少做计算而显示更省显存，所以还必须核对输出和参数梯度。</p>
          </section>
        </div>

        <div class="adv-flow">
          <span>x0</span><strong>checkpoint(block0, x0)</strong><span>x1</span><strong>checkpoint(block1, x1)</strong><span>x2</span>
        </div>

        <div class="adv-callout">完成 TODO 后要删掉 pass。保留 pass 本身不会执行 block，返回的只是原始输入；这不是 checkpointing，而是跳过了整个模型。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用图像处理 stage 练 checkpoint 调用签名", [
          "stage = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 6))",
          "stage_input = torch.randn(4, 6, requires_grad=True)",
          "",
          "wrapped_output = checkpoint(stage, stage_input, use_reentrant=False)",
          "loss = wrapped_output.square().mean()",
          "loss.backward()",
          "print(wrapped_output.shape)  # [4, 6]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>stage</code> -&gt; for 循环当前的 <code>block</code>。</li>
            <li><code>stage_input</code> -&gt; 当前的 <code>x</code>。</li>
            <li><code>wrapped_output</code> -&gt; 更新后的 <code>x</code>，要继续传给下一层。</li>
            <li><code>use_reentrant=False</code> -&gt; Notebook 提示要求关注的关键字参数。</li>
            <li>例子只演示一个 stage；TODO 要把同一调用模式放在已有 blocks 循环内部。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "循环里调用 checkpoint 得到了新输出，却没有把它重新赋给 x。",
        question: "下一轮 block 会收到什么？",
        options: ["旧 x，网络状态没有向前推进", "自动收到 checkpoint 的返回值", "一个 CPU Tensor"],
        answer: 0,
        revealNote: "Python 不会自动替换变量。checkpoint 返回普通 Tensor，必须显式更新循环状态。"
      },
      checkpoint: checkpoint(
        "Notebook 的 TODO 中，use_reentrant=False 应传给谁？",
        ["checkpoint 函数，作为关键字参数", "SimpleTransformerBlock.forward，作为位置参数", "torch.cuda.max_memory_allocated"],
        0,
        "它控制 checkpoint 的实现模式，不是 block 业务输入，也与显存统计函数无关。"
      ),
      homework: [
        "完成唯一 TODO：在已有 for block in blocks 循环中包装当前 block 与 x，使用 use_reentrant=False，并把返回的 [B,S,D] CUDA 浮点 Tensor 重新赋给 x；删除 pass。",
        "有 NVIDIA GPU 时运行 Notebook：两条路径都要完成 forward、sum().backward()，并人工确认 mem_ckpt <= mem_normal 的成功分支；无 GPU 的跳过信息不算验证通过。",
        "额外自检同一小模型的普通输出与 checkpoint 输出 allclose，并确认各 block 参数 grad 非空；若显存看似极低但输出等于原始输入，检查是否仍保留 pass 或忘了调用 block。"
      ]
    })
  ]
};
