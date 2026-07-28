const {
  advancedStyles,
  checkpoint,
  code,
  esc,
  lesson
} = require("../advanced_lesson_helpers");

const mapping = (title, rows) => `<div class="adv-map">
  <h4>${esc(title)}</h4>
  <ul>${rows.map(([from, to]) => `<li><code>${esc(from)}</code>：${esc(to)}</li>`).join("")}</ul>
</div>`;

const practice = (title, lines, mapTitle, rows) => `<div class="adv-practice">
  ${code(title, lines)}
  ${mapping(mapTitle, rows)}
</div>`;

const contract = (rows) => `<div class="adv-contract">
  ${rows.map(([label, detail]) => `<span>${esc(label)}</span><strong>${esc(detail)}</strong>`).join("")}
</div>`;

const flow = (steps, resultIndex = steps.length - 1) => `<div class="adv-flow">
  ${steps.map((step, index) => index === resultIndex
    ? `<strong>${esc(step)}</strong>`
    : `<span>${esc(step)}</span>`).join("")}
</div>`;

module.exports = {
  "19": [
    lesson({
      id: "checkpoint-call-contract",
      title: "先读调用契约：把每个 Block 的前向交给 checkpoint",
      todo: "唯一核心 TODO：在循环中更新 x = checkpoint(block, x, use_reentrant=False)",
      prerequisite: [
        "blocks 是 nn.ModuleList；循环中的 block 是一个可调用的 nn.Module，调用 block(x) 会返回下一层输入。",
        "x 是需要梯度的浮点张量。Checkpoint 不改变它的逻辑 shape；本题的测试输入是 [2, 2048, 2048]，device 是 CUDA。",
        "checkpoint 的第一个位置参数是要执行的函数，后面的位置参数是传给该函数的输入；use_reentrant=False 是本 Notebook 要求的关键字参数。",
        "循环必须把返回值重新赋给 x，否则后面的 Block 仍会收到旧状态，网络就没有按层向前传播。"
      ],
      intuition: "普通循环写的是 x = block(x)。Checkpoint 版本没有换模型，也没有跳过计算，只是在 block 外面加了一个“反向时可以重算”的包装器。因此最稳的解题方式是先写出普通调用，再把 block(x) 改写成 checkpoint(block, x, ...)。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>输入检查点</b>保存这一段开始时的 x</span>
          <span><b>前向执行</b>仍然调用当前 block</span>
          <span><b>少存激活</b>中间结果不全部长期保留</span>
          <span><b>反向重算</b>需要梯度时再次执行该段前向</span>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>普通版本</h4>
            ${flow(["x", "block(x)", "新的 x"])}
            <p>Autograd 为反向传播保留 Block 内部需要的中间激活。</p>
          </section>
          <section class="adv-panel good">
            <h4>Checkpoint 版本</h4>
            ${flow(["x", "checkpoint(block, x, ...)", "同形状的新 x"])}
            <p>数学前向不变，变化的是 Autograd 保存与重算中间量的策略。</p>
          </section>
        </div>
        ${contract([
          ["函数", "run_with_checkpointing(blocks: nn.ModuleList, x: torch.Tensor)"],
          ["循环输入", "当前 block 和上一层返回的 x"],
          ["循环输出", "新的 x，继续传给下一个 block"],
          ["关键参数", "use_reentrant=False"],
          ["最终返回", "最后一个 Block 的输出张量"]
        ])}
        <div class="adv-callout">不要把 checkpoint 写成 checkpoint(block(x), ...)。第一个参数必须是“稍后可以重新调用的函数”，而不是已经算完的张量。</div>
      </div>`,
      syntaxHtml: practice(
        "用一个不同的小函数练“函数和参数分开传入”",
        [
          "import torch",
          "from torch.utils.checkpoint import checkpoint",
          "",
          "def add_bias(values, bias):",
          "    return values + bias",
          "",
          "values = torch.randn(4, requires_grad=True)",
          "bias = torch.ones(4, requires_grad=True)",
          "result = checkpoint(add_bias, values, bias, use_reentrant=False)",
          "result.sum().backward()"
        ],
        "从语法例子迁移到 Notebook",
        [
          ["add_bias", "对应循环中的 block；二者都是可以稍后再次调用的 callable"],
          ["values", "对应当前 x；它是传给 callable 的输入张量"],
          ["result", "对应 checkpoint 返回的新 x，必须接住并继续向后传"],
          ["use_reentrant=False", "原样映射到 TODO，不能误当成 block 的参数"]
        ]
      ),
      predict: {
        hook: "循环已经来到第 3 个 Transformer Block，当前变量 x 是前两个 Block 更新后的状态。",
        question: "哪种写法既执行当前 Block，又把结果继续传给下一层？",
        options: [
          "x = checkpoint(block, x, use_reentrant=False)",
          "checkpoint(x, block, use_reentrant=False)",
          "checkpoint(block(x), use_reentrant=False)"
        ],
        answer: 0,
        revealNote: "checkpoint 的第一个参数是函数，后面才是函数输入；返回值必须重新保存到 x。"
      },
      checkpoint: checkpoint(
        "为什么 TODO 中不能只写 checkpoint(block, x, use_reentrant=False) 而不赋值？",
        ["后续 Block 会继续使用旧 x，层间数据流被截断", "因为 checkpoint 只能返回 Python 列表", "因为 x 会自动变成 CPU 张量"],
        0,
        "checkpoint 返回当前 Block 的输出。循环状态要靠 x = ... 显式向前更新。"
      ),
      homework: [
        "先把 TODO 改写成普通等价式 x = block(x)，确认自己能指出函数、输入和返回值，再替换为 checkpoint 调用。",
        "完成唯一核心 TODO，并逐项核对参数顺序：block 在前、x 在后、use_reentrant=False 是关键字参数。",
        "若报“Tensor object is not callable”，检查是否误把 block(x) 的结果放在第一个参数；若输出链路不对，检查是否忘记给 x 重新赋值。"
      ]
    }),

    lesson({
      id: "checkpoint-memory-test",
      title: "再读测试：结果要能 backward，收益看 CUDA 峰值",
      todo: "测试契约：正常前向与 checkpoint 前向都执行 backward，并比较 max_memory_allocated",
      prerequisite: [
        "Checkpoint 是训练期优化，只有执行 backward 才会触发重算；只看一次 forward 不能验证完整机制。",
        "测试先运行普通版本并记录 mem_normal，再清理输出、梯度和 CUDA 缓存，随后记录 mem_ckpt。",
        "输入 x_input 设置 requires_grad=True，最终对 out.sum() 调用 backward；这保证梯度链路必须保持完整。",
        "本节的显存测试明确要求 NVIDIA GPU；没有 CUDA 时 Notebook 会跳过，而不是说明实现已经通过。"
      ],
      intuition: "这道题不是测试“输出数值变小”，而是测试“同一条可求导计算，在少保存激活的情况下仍能完成反向”。评价顺序应是：先保证梯度链路正确，再比较峰值显存，最后讨论时间换空间是否值得。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>run_without_checkpointing</code><span>完成 forward + backward，记录 mem_normal</span></div>
          <div><b>2</b><code>清理状态</code><span>删除旧输出、清空 x_input.grad、重置峰值统计</span></div>
          <div><b>3</b><code>run_with_checkpointing</code><span>再次完成 forward + backward，记录 mem_ckpt</span></div>
          <div><b>4</b><code>比较峰值</code><span>可见断言接受 mem_ckpt 不大于 mem_normal</span></div>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>空间从哪里省</h4>
            <p>前向不长期保存每个 Block 内的所有中间激活，只保留重算所需的边界输入。</p>
          </section>
          <section class="adv-panel warn">
            <h4>时间花到哪里</h4>
            <p>反向传播需要恢复激活时，会重新执行对应 Block 的前向计算。</p>
          </section>
        </div>
        <div class="adv-checks">
          <span><b>shape</b><br>前后都保持 [B,S,D]</span>
          <span><b>device</b><br>测试张量和 Blocks 都在 CUDA</span>
          <span><b>梯度</b><br>x_input.requires_grad=True</span>
          <span><b>指标</b><br>峰值 allocated memory</span>
        </div>
      </div>`,
      syntaxHtml: practice(
        "用小网络练峰值统计的调用顺序",
        [
          "import torch",
          "import torch.nn as nn",
          "",
          "tiny_stage = nn.Linear(32, 32).cuda()",
          "torch.cuda.empty_cache()",
          "torch.cuda.reset_peak_memory_stats()",
          "",
          "sample = torch.randn(8, 32, device='cuda', requires_grad=True)",
          "result = tiny_stage(sample)",
          "result.sum().backward()",
          "",
          "peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)",
          "print(peak_mb)"
        ],
        "从语法例子迁移到 Notebook 测试",
        [
          ["tiny_stage(sample)", "对应 run_without_checkpointing 或 run_with_checkpointing 的一次完整前向"],
          ["result.sum().backward()", "对应测试中触发 Autograd 和 checkpoint 重算的动作"],
          ["reset_peak_memory_stats", "必须在每种策略开始前调用，避免沿用上一轮峰值"],
          ["peak_mb", "对应 mem_normal 或 mem_ckpt，单位从 byte 换算为 MB"]
        ]
      ),
      predict: {
        hook: "你只运行了 run_with_checkpointing(blocks, x)，没有调用 backward。",
        question: "此时为什么还不能完整观察 checkpoint 的机制？",
        options: [
          "因为反向阶段的重计算尚未发生",
          "因为 forward 会自动删除模型参数",
          "因为 checkpoint 只支持整数张量"
        ],
        answer: 0,
        revealNote: "Checkpoint 的时间开销和激活恢复主要发生在 backward；只跑 forward 看不到完整训练闭环。"
      },
      checkpoint: checkpoint(
        "本地没有 CUDA，测试打印“忽略测试”时，最准确的结论是什么？",
        ["当前环境没有验证真实显存峰值，需要在 GPU 环境再跑", "实现一定正确", "实现一定错误"],
        0,
        "跳过只说明环境条件不满足。代码语法可以检查，但显存收益仍需 CUDA 测试。"
      ),
      homework: [
        "在 GPU 环境运行可见测试，确认普通版本和 checkpoint 版本都能完成 backward，而不只是函数能返回张量。",
        "若 mem_ckpt 反而更高，先确认两轮之间是否删除旧输出、清空 input.grad 并重置峰值统计，再检查 TODO 是否真的包裹了每个 block。",
        "用一句话写出本节边界：checkpoint 优化的是训练激活显存，不等同于 activation offload，也不会减少参数或优化器状态本身。"
      ]
    })
  ],

  "20": [
    lesson({
      id: "flash-tiles-and-state",
      title: "先建状态账本：分块改变计算顺序，不改变 Attention 答案",
      todo: "TODO 1：初始化 out、m、l；理解两层分块循环与最后一块",
      prerequisite: [
        "本题刻意省略 Batch 和 Head 维度，q、k、v 都是 [seq_len, dim]；返回 out 也必须是 [seq_len, dim]。",
        "外层固定一块 Q，内层遍历所有 K/V 块。Python 切片允许终点超过长度，所以最后一块可以比 block_size 短。",
        "每个 query 行都维护自己的最大值 m 和指数和 l，因此它们是 [seq_len, 1] 列向量，便于向 score 的列方向广播。",
        "q_block 已在外层乘 scale = 1/sqrt(dim)，内层计算分数时不能再次缩放。"
      ],
      intuition: "把内层循环看成逐页读取 K/V：每读一页，就把这页对当前 Q 块的贡献合并进三个滚动状态。out 记当前归一化输出，m 记目前见过的最大 score，l 记以 m 为基准的指数和。翻完所有页后，这块 Q 的精确 Attention 才完成。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>外层 i</b>固定 q[i:i+block_size]</span>
          <span><b>内层 j</b>逐块读取 k 和 v</span>
          <span><b>滚动合并</b>更新 out_i、m_i、l_i</span>
          <span><b>写回</b>保存当前 Q 块的最终状态</span>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>标准 Attention 的大中间量</h4>
            ${flow(["Q [S,D]", "Kᵀ [D,S]", "Scores [S,S]"])}
            <p>完整分数矩阵会随序列长度按平方增长。</p>
          </section>
          <section class="adv-panel good">
            <h4>当前模拟只保留一个小分数块</h4>
            ${flow(["Q_i [Bq,D]", "K_jᵀ [D,Bk]", "S_ij [Bq,Bk]"])}
            <p>结果仍是精确 Attention，改变的是中间量的生存时间。</p>
          </section>
        </div>
        ${contract([
          ["out", "[seq_len, dim]；初始为 0，逐块形成最终输出"],
          ["m", "[seq_len, 1]；初始为 -inf，表示尚未看过任何 score"],
          ["l", "[seq_len, 1]；初始为 0，表示尚无指数贡献"],
          ["scale", "Python float，值为 1 / sqrt(dim)"],
          ["device", "新张量至少跟随 q.device；可见测试使用 CPU float32"]
        ])}
        <div class="adv-callout">m 不能从 0 开始。若第一块 score 全是负数，0 会变成一个数据中从未出现的假最大值；-inf 才表示“空状态”。</div>
      </div>`,
      syntaxHtml: practice(
        "用分批处理成绩表练初始化和尾块切片",
        [
          "import torch",
          "",
          "rows, width = 5, 3",
          "scores = torch.randn(rows, width)",
          "",
          "summary = torch.zeros((rows, width), device=scores.device)",
          "running_max = torch.full((rows, 1), -float('inf'), device=scores.device)",
          "running_sum = torch.zeros((rows, 1), device=scores.device)",
          "",
          "for start in range(0, rows, 3):",
          "    batch = scores[start:start + 3]",
          "    print(batch.shape)  # [3,3]，然后 [2,3]"
        ],
        "从语法例子迁移到 TODO 1",
        [
          ["summary", "对应 out，行数与序列长度一致，列数与 value 特征维一致"],
          ["running_max", "对应 m；每一行只保存一个最大值，所以末维是 1"],
          ["running_sum", "对应 l；每一行只保存一个指数和，所以末维是 1"],
          ["scores.device", "对应 q.device，避免创建在错误设备上的状态张量"]
        ]
      ),
      predict: {
        hook: "seq_len=5、block_size=3，外层循环第二次执行 i=3。",
        question: "q[3:6] 的实际 shape 是什么？",
        options: ["[2, dim]", "[3, dim] 并自动补零", "切片会越界报错"],
        answer: 0,
        revealNote: "切片终点可以超过序列长度，Python 会自然返回剩余两行。可见测试专门覆盖了不能整除的尾块。"
      },
      checkpoint: checkpoint(
        "为什么 m 和 l 使用 [seq_len, 1]，而不是 [seq_len]？",
        ["保留列维后可广播到每行的 score 列", "为了让它们变成整数", "为了删除 dim 维"],
        0,
        "Online Softmax 对每个 query 行维护一个统计量；[Bq,1] 能自然与 [Bq,Bk] 运算。"
      ),
      homework: [
        "完成 TODO 1 前先在纸上写出 out、m、l 的 shape 和初值含义，再根据 q.device 创建它们。",
        "手工走一遍 seq_len=5、block_size=3 的 i、j 取值，确认 Q 和 K/V 的尾块都允许只有 2 行。",
        "若第一轮出现未定义变量，检查三个状态是否都在 scale 和循环之前初始化；若设备报错，检查新状态是否跟随 q.device。"
      ]
    }),

    lesson({
      id: "flash-online-softmax",
      title: "再合并一个分数块：最大值换基准，旧统计量必须重标定",
      todo: "TODO 2–5：S_ij、m_block/m_new、P_ij、l_block/l_new",
      prerequisite: [
        "矩阵乘法 [Bq,D] @ [D,Bk] 得到 [Bq,Bk]；每行属于一个 query，每列属于当前 K 块中的一个 key。",
        "Softmax 沿 key 维计算，所以 max 和 sum 都使用 dim=-1，并保留 keepdim=True。",
        "m_new = maximum(m_i, m_block) 是逐元素比较两个 [Bq,1] 张量；它与 torch.max(S_ij, dim=...) 的归约作用不同。",
        "P_ij = exp(S_ij - m_new) 已经把当前块放到新的共同基准下，因此 l_block 直接对 P_ij 求和。"
      ],
      intuition: "难点不是求新块的指数和，而是“最大值基准可能变了”。旧 l_i 是按旧 m_i 计算的；当 m_new 更大时，旧指数整体要乘 exp(m_i-m_new) 才能和新块相加。这就像两份用不同汇率记录的账，先换成同一种基准才能合并。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>S_ij = q_block @ k_block.T</code><span>[Bq,D] @ [D,Bk] → [Bq,Bk]</span></div>
          <div><b>2</b><code>m_block = max(S_ij, dim=-1)</code><span>当前 K 块每个 query 行的局部最大值</span></div>
          <div><b>3</b><code>m_new = maximum(m_i, m_block)</code><span>选择旧块与新块共同的稳定基准</span></div>
          <div><b>4</b><code>P_ij = exp(S_ij - m_new)</code><span>广播减法后再取指数</span></div>
          <div><b>5</b><code>l_new = l_i * exp(m_i-m_new) + sum(P_ij)</code><span>修正旧账，再加入新块</span></div>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape</th><th>检查重点</th></tr></thead>
          <tbody>
            <tr><td><code>S_ij</code></td><td>[Bq,Bk]</td><td>q_block 已缩放，不重复乘 scale</td></tr>
            <tr><td><code>m_block</code></td><td>[Bq,1]</td><td>dim=-1 且 keepdim=True</td></tr>
            <tr><td><code>m_new</code></td><td>[Bq,1]</td><td>torch.maximum 做逐元素比较</td></tr>
            <tr><td><code>P_ij</code></td><td>[Bq,Bk]</td><td>减 m_new 时按列广播</td></tr>
            <tr><td><code>l_new</code></td><td>[Bq,1]</td><td>旧 l 先按新最大值修正</td></tr>
          </tbody>
        </table>
        <div class="adv-callout">如果直接写 l_i + l_block，只在最大值从未变化时才碰巧正确。一旦新块出现更大 score，旧的指数和必须先乘修正因子。</div>
      </div>`,
      syntaxHtml: practice(
        "用两批传感器读数练在线 log-sum-exp",
        [
          "import torch",
          "",
          "old_max = torch.tensor([[2.0]])",
          "old_sum = torch.tensor([[1.5]])",
          "new_values = torch.tensor([[1.0, 4.0, 3.0]])",
          "",
          "batch_max = new_values.max(dim=-1, keepdim=True).values",
          "joint_max = torch.maximum(old_max, batch_max)",
          "batch_exp = torch.exp(new_values - joint_max)",
          "batch_sum = batch_exp.sum(dim=-1, keepdim=True)",
          "joint_sum = old_sum * torch.exp(old_max - joint_max) + batch_sum"
        ],
        "从语法例子迁移到 Notebook TODO",
        [
          ["new_values", "对应 S_ij；每行是一位 query 对当前 K 块的分数"],
          ["batch_max", "对应 m_block，沿最后一维归约并保留列维"],
          ["joint_max", "对应 m_new，合并旧 m_i 与当前 m_block"],
          ["batch_exp", "对应 P_ij，所有分数都使用 m_new 这个共同基准"],
          ["joint_sum", "对应 l_new，先修正旧 l_i 再加 l_block"]
        ]
      ),
      predict: {
        hook: "旧最大值 m_i=2，新块最大值 m_block=5，因此 m_new=5。",
        question: "旧指数和 l_i 应乘哪个修正因子后再与新块相加？",
        options: ["exp(2-5)", "exp(5-2)", "完全不修正"],
        answer: 0,
        revealNote: "旧项原本以 2 为基准，改成更大的 5 后，每个旧指数都要缩小 exp(2-5) 倍。"
      },
      checkpoint: checkpoint(
        "TODO 3 中 torch.max 与 torch.maximum 分别负责什么？",
        ["前者沿 score 列归约，后者逐元素比较旧、新最大值", "二者都删除 batch 维", "前者只支持 CPU，后者只支持 GPU"],
        0,
        "先从 S_ij 每行取 m_block，再把 m_block 与同形状的 m_i 逐行比较得到 m_new。"
      ),
      homework: [
        "按 shape 链完成 TODO 2–5：每写一行就标注输出 shape，确保所有行级统计量都保持 [Bq,1]。",
        "用极端值手算一行：旧最大值 2、新块最大值 5，验证旧 l 的修正因子小于 1，而不是大于 1。",
        "若结果出现 inf/NaN，检查是否先减 m_new 再 exp；若广播报错，检查 max/sum 是否漏了 keepdim=True。"
      ]
    }),

    lesson({
      id: "flash-output-and-tests",
      title: "最后更新输出并写回：旧贡献和新贡献都要除以同一个 l_new",
      todo: "TODO 6 + 状态更新 + 外层写回；通过三组标准 Attention 对照测试",
      prerequisite: [
        "out_i 是已经归一化的旧输出 [Bq,D]，不能直接与 P_ij @ v_block 相加；两者当前所用的归一化尺度不同。",
        "P_ij @ v_block 的 shape 是 [Bq,Bk] @ [Bk,D] = [Bq,D]，与 out_i 同形状。",
        "每处理完一个 K/V 块，都要令 m_i=m_new、l_i=l_new，下一轮才会从最新状态继续。",
        "内层循环结束后，out_i、m_i、l_i 只覆盖当前 Q 切片；必须写回全局对应的 i:i+block_size 区域。"
      ],
      intuition: "输出更新可以拆成两部分：先把旧的归一化输出缩放到新分母下，再把当前块的加权 V 贡献除以同一个 l_new。这个“同分母再相加”是保持数学等价的关键。循环状态没更新或最终没写回，都会让后续块看见过期数据。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>旧输出的权重</h4>
            <p><code>l_i * exp(m_i - m_new) / l_new</code></p>
            <p>把以前已经归一化的 out_i 调整到新的最大值与总指数和下。</p>
          </section>
          <section class="adv-panel good">
            <h4>当前块的贡献</h4>
            <p><code>(P_ij @ v_block) / l_new</code></p>
            <p>先按当前块指数权重聚合 V，再除以合并后的共同分母。</p>
          </section>
        </div>
        <div class="adv-roadmap">
          <span><b>更新 out_i</b>合并旧输出与当前 V 块</span>
          <span><b>更新 m_i/l_i</b>供下一个 K/V 块使用</span>
          <span><b>结束内层</b>当前 Q 块已看完全部 keys</span>
          <span><b>写回全局</b>保存 out、m、l 对应切片</span>
        </div>
        <div class="adv-checks">
          <span><b>case 1</b><br>S=8,D=4,block=2</span>
          <span><b>case 2</b><br>S=5,D=3,block=3</span>
          <span><b>case 3</b><br>S=3,D=2,block=1</span>
          <span><b>误差</b><br>max abs diff &lt; 1e-5</span>
        </div>
      </div>`,
      syntaxHtml: practice(
        "用“旧平均 + 新一批数据”练同分母合并",
        [
          "import torch",
          "",
          "old_mean = torch.tensor([[10.0, 20.0]])",
          "old_weight = torch.tensor([[2.0]])",
          "new_weighted_sum = torch.tensor([[9.0, 12.0]])",
          "new_weight = torch.tensor([[3.0]])",
          "",
          "total_weight = old_weight + new_weight",
          "merged = old_mean * (old_weight / total_weight)",
          "merged = merged + new_weighted_sum / total_weight",
          "print(merged.shape)  # [1,2]"
        ],
        "从语法例子迁移到 TODO 6",
        [
          ["old_mean", "对应 out_i，它已经是旧块归一化后的输出"],
          ["old_weight", "对应修正后的 l_i * exp(m_i-m_new)"],
          ["new_weighted_sum", "对应 P_ij @ v_block"],
          ["total_weight", "对应 l_new，旧、新贡献共享的分母"],
          ["merged", "对应更新后的 out_i，shape 仍为 [Bq,D]"]
        ]
      ),
      predict: {
        hook: "当前 Q 块已经遍历完所有 K/V 块，但代码没有执行 out[i:i+block_size] = out_i。",
        question: "函数最终返回的 out 会出现什么问题？",
        options: ["对应切片仍保留初始化值，计算结果没有进入返回张量", "PyTorch 会自动猜出 out_i 应写到哪里", "out 会自动增加一个维度"],
        answer: 0,
        revealNote: "out_i 在循环中被重新绑定为新张量；必须显式写回全局 out 的当前切片。"
      },
      checkpoint: checkpoint(
        "为什么可见测试包含 seq_len=5、block_size=3？",
        ["验证最后一块较短时切片、广播和写回仍正确", "验证 block_size 必须整除 seq_len", "验证只能使用 GPU"],
        0,
        "第二组用例专门覆盖尾块大小不一致；硬编码固定 Bq/Bk 会在这里暴露。"
      ),
      homework: [
        "完成 TODO 6、m_i/l_i 更新和三个全局写回；逐项确认 out_i 与 P_ij @ v_block 都是 [Bq,D]。",
        "运行三组测试并记录最大绝对误差。若只有非整除用例失败，优先排查是否用固定 block_size 创建了临时张量或写回边界。",
        "若所有结果都有系统性偏差，检查 scale 是否只应用一次；若后几个 K 块贡献异常，检查每轮是否更新 m_i 和 l_i。"
      ]
    })
  ],

  "21": [
    lesson({
      id: "decoding-temperature",
      title: "Temperature：先保护除数，再保持 logits 的形状与索引",
      todo: "TODO 1：temp = max(temperature, 1e-6)，返回 logits / temp",
      prerequisite: [
        "logits 是 [batch, vocab_size] 的浮点张量，每一列仍代表固定的 token_id；Temperature 不排序也不删除列。",
        "Softmax 前除以 T：T 小于 1 会放大差距，T 大于 1 会缩小差距，T=1 不改变 logits。",
        "temperature 是 Python float，作为除数必须设置极小正数下限，避免 0 导致无定义结果。",
        "逐元素除法不改变 logits 的 shape、dtype 或 device。"
      ],
      intuition: "Temperature 像调节分数表的对比度：低温把分差拉大，高温把分差压小。它只缩放数值，不负责把分数变成概率；Softmax 在完整解码流水线最后统一执行。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid three">
          <section class="adv-panel good"><h4>T &lt; 1</h4><p>除以小数，分差放大，采样更集中。</p></section>
          <section class="adv-panel neutral"><h4>T = 1</h4><p>数值不变，保留原始相对差距。</p></section>
          <section class="adv-panel warn"><h4>T &gt; 1</h4><p>除以大数，分差缩小，采样更分散。</p></section>
        </div>
        ${contract([
          ["输入", "logits: [B,V] 浮点张量；temperature: Python float"],
          ["安全下限", "max(temperature, 1e-6)"],
          ["计算", "logits / temp，逐元素缩放"],
          ["输出", "[B,V]，token 列顺序、dtype、device 不变"],
          ["可见断言", "T=0.5 时 index 5 与 6 的分差变为原来的 2 倍"]
        ])}
      </div>`,
      syntaxHtml: practice(
        "用考试分数练安全除法和差值变化",
        [
          "import torch",
          "",
          "ratings = torch.tensor([[8.0, 5.0, 2.0]])",
          "temperature = 0.5",
          "safe_temperature = max(temperature, 1e-6)",
          "adjusted = ratings / safe_temperature",
          "",
          "print(adjusted.shape)             # [1,3]",
          "print(adjusted[0,0] - adjusted[0,1])  # 6.0"
        ],
        "从语法例子迁移到 TODO 1",
        [
          ["ratings", "对应 logits，每一列的位置含义不能改变"],
          ["safe_temperature", "对应 temp，使用 Python max 设置正数下限"],
          ["adjusted", "对应返回值 logits / temp，不在这里调用 Softmax"],
          ["差值 6.0", "原差值 3.0 除以 0.5 后翻倍，对应可见测试的检查方式"]
        ]
      ),
      predict: {
        hook: "两个 logits 是 4.0 和 3.1，temperature=0.5。",
        question: "缩放后的分差是多少？",
        options: ["1.8", "0.45", "shape 会变成 [V,B]"],
        answer: 0,
        revealNote: "原分差 0.9 除以 0.5，得到 1.8；逐元素除法不会转置张量。"
      },
      checkpoint: checkpoint(
        "TODO 1 为什么不直接 return F.softmax(logits / temp, dim=-1)？",
        ["函数契约只负责调温，完整流水线稍后统一归一化", "Softmax 只能用于整数", "Temperature 会删除 vocab 维"],
        0,
        "三个过滤函数都在 logits 空间工作，decode_next_token 最后才执行 Softmax 和 multinomial。"
      ),
      homework: [
        "完成 temp 的正数下限和 logits / temp；确认函数没有排序、采样或提前 Softmax。",
        "运行 Temperature 断言。若分差变成一半，检查是否误写成 logits * temperature；若出现 inf，检查是否漏了下限。",
        "分别手算 T=0.5、1.0、2.0 时 [4.0,3.0] 的分差，解释哪一种更确定。"
      ]
    }),

    lesson({
      id: "decoding-top-k",
      title: "Top-K：用第 K 大值做门槛，过滤但不删除词表列",
      todo: "TODO 2：filter_value、kth_values 与 torch.where 条件替换",
      prerequisite: [
        "函数顶部已处理 top_k<=0 或 top_k>=vocab_size 的边界；这些情况直接返回原 logits。",
        "torch.topk(logits, top_k, dim=-1) 会为每个 batch 行分别返回 [B,K] 的最大值和索引。",
        "取 values 的最后一个元素可得到第 K 大门槛；使用 [..., -1:] 保留 [B,1]，便于广播。",
        "被过滤位置设为 -inf，而不是删除列；后续 Softmax 会给它们概率 0，原 token_id 索引仍然有效。"
      ],
      intuition: "Top-K 是“每一行单独划线”：先找到该行前 K 名中最低的分数，再把低于这条线的位置盖成 -inf。张量宽度不变，所以后面采样到的列号仍能直接当 token_id。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>topk</b>每行找前 K 大值</span>
          <span><b>cutoff</b>取第 K 大，保留末维</span>
          <span><b>compare</b>logits &lt; cutoff</span>
          <span><b>where</b>低分位置替换为 -inf</span>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>对象</th><th>shape</th><th>作用</th></tr></thead>
          <tbody>
            <tr><td><code>logits</code></td><td>[B,V]</td><td>原词表分数与 token 位置</td></tr>
            <tr><td>Top-K values</td><td>[B,K]</td><td>每行最大的 K 个值</td></tr>
            <tr><td><code>kth_values</code></td><td>[B,1]</td><td>每行广播门槛</td></tr>
            <tr><td>过滤结果</td><td>[B,V]</td><td>低于门槛处为 -inf</td></tr>
          </tbody>
        </table>
        <div class="adv-callout">可见测试没有门槛并列，因此恰好保留 3 个位置。一般情况下若多个值等于第 K 大门槛，“小于门槛才过滤”可能保留多于 K 个并列项。</div>
      </div>`,
      syntaxHtml: practice(
        "每位评委只保留得分最高的两个方案",
        [
          "import torch",
          "",
          "ratings = torch.tensor([[2.0, 7.0, 5.0, 1.0],",
          "                        [8.0, 3.0, 6.0, 4.0]])",
          "best_values, _ = torch.topk(ratings, 2, dim=-1)",
          "cutoff = best_values[..., -1:]",
          "blocked = torch.tensor(float('-inf'), device=ratings.device)",
          "screened = torch.where(ratings < cutoff, blocked, ratings)",
          "",
          "print(cutoff.shape)   # [2,1]",
          "print(screened.shape) # [2,4]"
        ],
        "从语法例子迁移到 TODO 2",
        [
          ["ratings", "对应 logits，每个 batch 行独立筛选"],
          ["best_values", "对应 torch.topk 返回的前 K 大值"],
          ["cutoff", "对应 kth_values，末尾切片保留长度 1 的维度"],
          ["blocked", "对应 filter_value=-inf，并放在 logits.device"],
          ["screened", "对应过滤后的 logits，shape 与原索引不变"]
        ]
      ),
      predict: {
        hook: "一行 logits 是 [4.0,1.0,3.0,2.0]，K=2。",
        question: "第 K 大门槛和最终保留值分别是什么？",
        options: ["门槛 3.0，保留 4.0 与 3.0", "门槛 4.0，只保留 4.0", "门槛 2.0，保留三个值"],
        answer: 0,
        revealNote: "前两大是 4.0、3.0，最后一个 Top-K 值 3.0 就是门槛；只过滤严格小于门槛的位置。"
      },
      checkpoint: checkpoint(
        "为什么 kth_values 使用 [..., -1:] 而不是 [..., -1]？",
        ["保留 [B,1] 以便沿词表维广播", "把值转换成索引", "让 top_k 自动加一"],
        0,
        "长度为 1 的末维能与 [B,V] 逐行比较；[B] 在多 batch 情况下不一定按预期广播。"
      ),
      homework: [
        "保留函数已有边界分支，只补 filter_value、Top-K values、[B,1] 门槛与 torch.where。",
        "运行可见 Top-K 测试，非 -inf 数量应为 3。若不对，检查是否取成第一大值、比较方向写反或误删等于门槛的位置。",
        "额外用两行 logits 验证每行门槛不同，确认不是对整个 batch 只求一条全局门槛。"
      ]
    }),

    lesson({
      id: "decoding-top-p",
      title: "Top-p：在排名坐标累计概率，再 scatter 回 token 坐标",
      todo: "TODO 3：移除掩码、右移、-inf 过滤与 scatter_ 恢复",
      prerequisite: [
        "sorted_logits 与 sorted_indices 已由 Notebook 按最后一维降序得到；二者 shape 都是 [B,V]。",
        "cumulative_probs 已经对排序后的 logits 做 Softmax 再 cumsum，累计发生在概率而不是原始分数上。",
        "第一个使累计概率超过 top_p 的 token 仍需保留，因此 remove 掩码必须向右平移一格，并把第一项设为 False。",
        "排序位置不是 token_id；过滤后必须根据 sorted_indices 沿 dim=-1 散射回原词表顺序。"
      ],
      intuition: "Top-p 同时使用两个坐标系：先在“从高概率到低概率”的排名坐标里决定候选集合，再回到“原始 token_id”坐标里采样。右移掩码是为了把首次跨过阈值的那一项也收入最小候选集合。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>sort</b>分数降序并记录原索引</span>
          <span><b>softmax+cumsum</b>得到累计概率</span>
          <span><b>shift mask</b>保留首次越界项</span>
          <span><b>scatter</b>恢复原 token_id 顺序</span>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>阈值 0.8 的排名坐标</h4>
            ${flow(["0.54", "0.76", "0.86 首次越界", "0.94 后续"], 2)}
            <p>右移前第三项会被标记；右移后第三项保留，从第四项开始删除。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>恢复 token 坐标</h4>
            ${contract([
              ["排序位置 0", "可能来自原 token 5"],
              ["排序位置 1", "可能来自原 token 6"],
              ["排序位置 2", "可能来自原 token 1"],
              ["scatter 后", "保留值回到列 5、6、1"]
            ])}
          </section>
        </div>
        <div class="adv-callout">右移时要读取 clone()：左右切片有重叠，直接原地从同一张量复制可能在写入时污染后面还要读取的旧掩码。</div>
      </div>`,
      syntaxHtml: practice(
        "按累计权重筛商品，再恢复商品原编号",
        [
          "import torch",
          "",
          "scores = torch.tensor([[0.2, 2.0, 1.0, 3.0]])",
          "ranked, original_pos = torch.sort(scores, dim=-1, descending=True)",
          "running = torch.cumsum(torch.softmax(ranked, dim=-1), dim=-1)",
          "drop = running > 0.75",
          "drop[..., 1:] = drop[..., :-1].clone()",
          "drop[..., 0] = False",
          "ranked[drop] = float('-inf')",
          "restored = torch.zeros_like(scores).scatter_(-1, original_pos, ranked)"
        ],
        "从语法例子迁移到 TODO 3",
        [
          ["ranked", "对应 sorted_logits，已经按分数降序"],
          ["original_pos", "对应 sorted_indices，记录每个排序值来自哪个 token 列"],
          ["running", "对应 cumulative_probs，shape 仍为 [B,V]"],
          ["drop", "对应 sorted_indices_to_remove，需要右移并保护首项"],
          ["restored", "对应 restored_logits，沿 dim=-1 恢复原词表顺序"]
        ]
      ),
      predict: {
        hook: "排序后的累计概率是 [0.54,0.76,0.86,0.94]，top_p=0.8。",
        question: "右移“累计概率大于阈值”的掩码后，应保留几个候选？",
        options: ["3 个", "2 个", "4 个"],
        answer: 0,
        revealNote: "第三个候选让累计概率首次超过 0.8，它仍是达到阈值所必需的成员。"
      },
      checkpoint: checkpoint(
        "最后 scatter_ 的直接目的是什么？",
        ["让过滤后的分数重新对应正确 token_id", "把概率和强制设为 0", "把 batch 维拼到词表维"],
        0,
        "后续 multinomial 返回的是列索引；恢复原顺序后，这个列索引才仍是正确 token_id。"
      ),
      homework: [
        "完成布尔掩码、右移 clone、首项保护、-inf 赋值与 scatter_；所有中间量 shape 都应保持 [B,V]。",
        "运行 Top-p 测试与完整管线测试：应保留 3 个有限值，next_token shape 为 [1,1]。",
        "若保留 2 个，检查是否忘了右移；若 token_id 对不上，检查 scatter_ 的 dim、index、src；若掩码异常，检查右移时是否漏了 clone。"
      ]
    })
  ],

  "22": [
    lesson({
      id: "paged-prefill",
      title: "Prefill：先向上取整，再做一次完整容量检查",
      todo: "当前练习 TODO 1–2：needed_blocks、OOM 检查与 block_table 分配",
      prerequisite: [
        "physical_kv_cache 已在当前练习的 __init__ 中给出，shape 是 [num_blocks, block_size, head_dim]；不要重复实现参考解析里的初始化 TODO。",
        "req.seq_len 是 prompt token 数，req.block_table 是 List[int]，按逻辑顺序记录物理 block_id。",
        "free_blocks 是可用物理块编号列表；pop(0) 会返回队首编号并从列表移除。",
        "最后一块即使只使用一个 token，也必须占用完整物理块，因此需要整数向上取整。"
      ],
      intuition: "把物理块当成固定容量的箱子。Prompt 要一次性装完：先算需要几个箱子，再确认库存足够，最后才逐个拿走。先检查完整容量能避免分到一半才 OOM，留下 block_table 和 free_blocks 不一致的半更新状态。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>需求</b>req.seq_len 个 token</span>
          <span><b>容量</b>每块 block_size 个位置</span>
          <span><b>取整</b>(seq_len + size - 1) // size</span>
          <span><b>分配</b>free_blocks → req.block_table</span>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>长度 6、block_size 4</h4>
            ${flow(["token 0..3：第一块", "token 4..5：第二块"], 1)}
            <p>6 // 4 只得到 1，会漏掉尾部两个 token；向上取整得到 2。</p>
          </section>
          <section class="adv-panel good">
            <h4>块表只记录地址</h4>
            ${flow(["逻辑块 0 → 物理 ID 7", "逻辑块 1 → 物理 ID 2"], 1)}
            <p>物理 ID 可以不连续，列表顺序负责表达逻辑顺序。</p>
          </section>
        </div>
        ${contract([
          ["needed_blocks", "Python int；必须覆盖全部 req.seq_len"],
          ["free_blocks", "List[int]；每次 pop 后长度减一"],
          ["req.block_table", "List[int]；按逻辑块顺序 append"],
          ["不足分支", "抛 RuntimeError(\"OOM\")"],
          ["可见用例", "长度 6、块大小 4 → 2 块，10 块池剩 8 块"]
        ])}
      </div>`,
      syntaxHtml: practice(
        "给订单分配固定容量的货箱",
        [
          "item_count = 11",
          "box_capacity = 5",
          "boxes_needed = (item_count + box_capacity - 1) // box_capacity",
          "available = [20, 7, 31, 9]",
          "assigned = []",
          "",
          "if len(available) < boxes_needed:",
          "    raise RuntimeError('FULL')",
          "for _ in range(boxes_needed):",
          "    assigned.append(available.pop(0))"
        ],
        "从语法例子迁移到当前练习",
        [
          ["item_count", "对应 req.seq_len"],
          ["box_capacity", "对应 self.block_size"],
          ["boxes_needed", "对应 TODO 1 的 needed_blocks"],
          ["available", "对应 self.free_blocks"],
          ["assigned", "对应 req.block_table；异常文本要按 Notebook 写成 OOM"]
        ]
      ),
      predict: {
        hook: "prompt_len=5、block_size=4，空闲池只有 1 个块。",
        question: "allocate_for_prefill 应怎样处理？",
        options: ["需要 2 块，分配前直接抛 RuntimeError(\"OOM\")", "只分 1 块并忽略最后一个 token", "先拿走 1 块再返回成功"],
        answer: 0,
        revealNote: "向上取整得到 2，而库存只有 1。测试要求异常文本包含 OOM。"
      },
      checkpoint: checkpoint(
        "为什么应在 for 循环 pop 之前检查 len(free_blocks)？",
        ["避免失败时已经拿走部分块，造成半更新状态", "为了让 block_id 变成浮点数", "为了把 seq_len 清零"],
        0,
        "容量检查先于状态修改，OOM 时两个列表都能保持原样。"
      ),
      homework: [
        "完成当前练习 TODO 1–2：整数向上取整、完整容量检查、循环 pop(0) 并 append。",
        "运行 Prefill 与 OOM 用例：长度 6 应分 2 块；只有 1 块却要容纳 5 个 token 时必须抛含 OOM 的 RuntimeError。",
        "若块数少一，检查是否用了向下取整；若 OOM 后 free_blocks 已减少，检查是否把容量检查写进了分配循环。"
      ]
    }),

    lesson({
      id: "paged-decode",
      title: "Decode：先看已经加一的新长度，只在新块第一个位置补块",
      todo: "当前练习 TODO 3：is_new_block_needed 与按需追加一个物理块",
      prerequisite: [
        "函数开头已经执行 req.seq_len += 1；TODO 看到的是新 token 加入后的长度。",
        "block_size=4 时，长度 1–4 使用第一块，5–8 使用第二块，9–12 使用第三块。",
        "当前 Notebook 的判断契约是 (req.seq_len % self.block_size) == 1，表示刚进入新块的第一个槽位。",
        "Decode 每次只新增一个 token，因此跨界时最多追加一个物理块；空闲列表为空要抛 RuntimeError(\"OOM\")。"
      ],
      intuition: "不要孤立背取模公式，先画位置：长度从 4 加到 5 时，新 token 落在第二块第一个位置，所以补块；从 5 加到 6 时仍在第二块，不补；从 8 加到 9 时再补第三块。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>req.seq_len += 1</code><span>Notebook 已经把本轮新 token 计入长度</span></div>
          <div><b>2</b><code>seq_len % block_size</code><span>检查新 token 在当前逻辑块中的位置</span></div>
          <div><b>3</b><code>余数 == 1</code><span>表示刚进入新块的第一个位置</span></div>
          <div><b>4</b><code>pop + append</code><span>只有跨界时才申请一个新物理块</span></div>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>加一后的长度</th><th>余数（block_size=4）</th><th>动作</th></tr></thead>
          <tbody>
            <tr><td>7</td><td>3</td><td>继续使用现有块</td></tr>
            <tr><td>8</td><td>0</td><td>现有块刚好填满，不提前分配</td></tr>
            <tr><td>9</td><td>1</td><td>进入新块，追加一个 block_id</td></tr>
          </tbody>
        </table>
        <div class="adv-callout">常见偏一位错误是判断余数等于 0。这里长度已经先加一；余数 0 表示新 token 填满旧块，余数 1 才表示它进入下一块。</div>
      </div>`,
      syntaxHtml: practice(
        "给连续座位按排扩容",
        [
          "seats_per_row = 4",
          "current_people = 8",
          "current_people += 1",
          "",
          "needs_row = (current_people % seats_per_row) == 1",
          "free_rows = [12, 13]",
          "assigned_rows = [4, 9]",
          "if needs_row:",
          "    if not free_rows:",
          "        raise RuntimeError('FULL')",
          "    assigned_rows.append(free_rows.pop(0))"
        ],
        "从语法例子迁移到 TODO 3",
        [
          ["current_people += 1", "对应已经写好的 req.seq_len += 1"],
          ["seats_per_row", "对应 self.block_size"],
          ["needs_row", "对应 is_new_block_needed"],
          ["free_rows", "对应 self.free_blocks；为空时异常文本改为 OOM"],
          ["assigned_rows", "对应 req.block_table，只在跨界时 append 一个 ID"]
        ]
      ),
      predict: {
        hook: "block_size=4，请求当前长度 8；allocate_for_decode 开头先把长度加到 9。",
        question: "这一步是否需要新 Block？",
        options: ["需要，因为 9 % 4 == 1", "不需要，因为 8 % 4 == 0", "需要一次分配 4 个 Block"],
        answer: 0,
        revealNote: "判断使用的是加一后的长度 9，它刚进入第三个逻辑块的第一个位置。"
      },
      checkpoint: checkpoint(
        "长度 6 连续调用三次 decode 后，为什么第 9 个 token 才让块表从 2 块变成 3 块？",
        ["长度 7、8 仍在第二块，长度 9 才进入第三块", "因为每三次调用固定扩容", "因为 block_id 必须是 9"],
        0,
        "扩容由块边界决定，不由调用次数固定决定。"
      ),
      homework: [
        "完成 is_new_block_needed 和跨界分配分支；不要重复 req.seq_len += 1，也不要每次 decode 都分块。",
        "按可见测试从长度 6 手算到 9：第一次后仍 2 块，第三次后变 3 块，空闲池从 8 变 7。",
        "若长度 8 时提前扩容，检查是否判断余数 0；若长度 9 不扩容，检查是否在加一前计算了条件。"
      ]
    }),

    lesson({
      id: "paged-cache-rebuild",
      title: "按块表恢复逻辑序列：先索引每块，再沿 token 维拼接并截尾",
      todo: "当前练习 TODO 5：blocks、torch.cat(dim=0) 与 [:req.seq_len]",
      prerequisite: [
        "physical_kv_cache 的 shape 是 [num_blocks, block_size, head_dim]；用一个 block_id 索引后得到 [block_size, head_dim]。",
        "req.block_table 按逻辑顺序保存物理 ID，因此列表推导式也必须按这个顺序取块。",
        "token 位于每块的第 0 维，所以多个块沿 dim=0 拼接，结果是 [allocated_blocks*block_size, head_dim]。",
        "最后一个块可能没有填满，返回前要切到 req.seq_len，得到精确 [seq_len, head_dim]。"
      ],
      intuition: "物理地址可以跳来跳去，但 block_table 的列表顺序就是逻辑顺序。先按表把每个二维块取出来，再沿 token 方向接成长序列；最后裁掉尾块未使用的槽位，才能得到请求真正拥有的 KV 长度。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>块表 [7,2,9]</b>定义逻辑块顺序</span>
          <span><b>逐块索引</b>每块 [block_size, head_dim]</span>
          <span><b>cat dim=0</b>拼成逻辑 token 序列</span>
          <span><b>[:seq_len]</b>裁掉尾块空槽</span>
        </div>
        ${contract([
          ["物理池", "[num_blocks, block_size, head_dim]"],
          ["单个 blocks 元素", "[block_size, head_dim]"],
          ["cat_blocks", "[块表长度 × block_size, head_dim]"],
          ["返回 cache", "[req.seq_len, head_dim]"],
          ["可见边界", "长度 5、块大小 4 → 两块拼成 8 行，再截成 5 行"]
        ])}
        <div class="adv-callout">不能直接返回 physical_kv_cache[req.block_table]：它保留三维 [num_used_blocks, block_size, head_dim]，而测试要求逻辑连续的二维 [seq_len, head_dim]。</div>
      </div>`,
      syntaxHtml: practice(
        "按目录顺序拼接分散的数据页",
        [
          "import torch",
          "",
          "pages = torch.arange(5 * 3 * 2).reshape(5, 3, 2)",
          "page_table = [3, 0]",
          "ordered_pages = [pages[page_id] for page_id in page_table]",
          "continuous = torch.cat(ordered_pages, dim=0)",
          "used_rows = continuous[:5]",
          "",
          "print(continuous.shape)  # [6,2]",
          "print(used_rows.shape)   # [5,2]"
        ],
        "从语法例子迁移到 TODO 5",
        [
          ["pages", "对应 self.physical_kv_cache"],
          ["page_table", "对应 req.block_table，顺序不能排序或去重"],
          ["ordered_pages", "对应 blocks 列表，每项是一个物理缓存块"],
          ["continuous", "对应 cat_blocks，沿 dim=0 拼接 token"],
          ["used_rows", "对应返回值 cat_blocks[:req.seq_len]"]
        ]
      ),
      predict: {
        hook: "block_size=4、head_dim=8，块表有 2 个 ID，请求 seq_len=5。",
        question: "cat 前、cat 后、截断后的 shape 依次是什么？",
        options: [
          "每块 [4,8]；拼接 [8,8]；返回 [5,8]",
          "每块 [4,8]；拼接 [2,4,8]；返回 [5,8]",
          "每块 [8]；拼接 [16]；返回 [5]"
        ],
        answer: 0,
        revealNote: "单块索引删除物理块维，沿 token 维拼成 8 行，再按真实长度保留前 5 行。"
      },
      checkpoint: checkpoint(
        "为什么 blocks 的读取顺序必须服从 req.block_table？",
        ["列表位置表达逻辑块顺序，物理 ID 本身可能不连续", "因为 torch.cat 会自动排序", "因为 head_dim 是 block_id"],
        0,
        "块表就是逻辑到物理的映射；按物理 ID 排序会打乱请求的 token 顺序。"
      ),
      homework: [
        "完成列表推导式、torch.cat(dim=0) 和最终切片；确认没有修改 req.block_table 或 free_blocks。",
        "运行两组拼装测试：长度 9 应返回 [9,64] 且三段值为 1/2/3；长度 5 应返回 [5,8] 且第二块只取 1 行。",
        "若返回三维张量，检查是否漏了 cat；若尾部 shape 太长，检查是否漏了 [:req.seq_len]；若内容顺序错，检查是否排序了 block_id。"
      ]
    })
  ],

  "23": [
    lesson({
      id: "speculative-direct-accept",
      title: "先对齐当前位置：p 不小于 q 时，当前草稿必接受",
      todo: "TODO 1：读取对应 token 的 p、q；当 p >= q 时 append token_id",
      prerequisite: [
        "draft_probs 和 target_probs 都是 [K, vocab_size]；第 i 行对应第 i 个草稿位置，列 token_id 对应该 token 的概率。",
        "draft_tokens 在可见测试中是 Python 列表 [10,20,30,40]；token_id 可直接作为张量列索引。",
        "p 和 q 已通过 [i, token_id].item() 取成 Python float，后续比较不是张量广播。",
        "接受概率是 min(1,p/q)。当 p>=q 时结果为 1，因此不需要调用随机数。"
      ],
      intuition: "验证时必须同时对齐“位置 i”和“该位置草稿选中的 token_id”。我们只比较这个候选 token 在两个模型下的概率，不比较整行最大值。若目标模型给它的概率至少和草稿模型一样高，草稿没有过度推荐它，可以直接接受。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>位置 i</b>取 draft_tokens[i]</span>
          <span><b>同一列</b>读取 target_probs[i, token_id]</span>
          <span><b>同一列</b>读取 draft_probs[i, token_id]</span>
          <span><b>比较</b>p >= q 时直接 append</span>
        </div>
        ${contract([
          ["draft_probs", "[K,V] 浮点张量；草稿模型分布"],
          ["target_probs", "[K,V] 浮点张量；目标模型分布"],
          ["draft_tokens", "长度 K 的 token_id 序列；可见测试使用 Python list"],
          ["p/q", "当前 i、当前 token_id 对应的 Python float"],
          ["accepted_tokens", "按原顺序追加的 Python list"]
        ])}
        <div class="adv-grid two">
          <section class="adv-panel good"><h4>p=0.8，q=0.5</h4><p>p>=q，接受概率为 1，直接追加 token。</p></section>
          <section class="adv-panel warn"><h4>不要比较错误位置</h4><p><code>target_probs[i].max()</code> 回答的是“目标模型最喜欢谁”，不是“草稿 token 能否接受”。</p></section>
        </div>
      </div>`,
      syntaxHtml: practice(
        "逐项核验候选商品的两方评分",
        [
          "import torch",
          "",
          "reviewer_a = torch.tensor([[0.2, 0.8], [0.7, 0.3]])",
          "reviewer_b = torch.tensor([[0.1, 0.9], [0.6, 0.4]])",
          "chosen_items = [1, 0]",
          "approved = []",
          "",
          "for i, item_id in enumerate(chosen_items):",
          "    p = reviewer_a[i, item_id].item()",
          "    q = reviewer_b[i, item_id].item()",
          "    if p >= q:",
          "        approved.append(item_id)"
        ],
        "从语法例子迁移到 TODO 1",
        [
          ["i", "对应草稿位置，选择概率矩阵的行"],
          ["item_id", "对应 token_id，选择该行中的词表列"],
          ["reviewer_a", "对应 target_probs，取目标概率 p"],
          ["reviewer_b", "对应 draft_probs，取草稿概率 q"],
          ["approved.append", "对应 accepted_tokens.append(token_id)"]
        ]
      ),
      predict: {
        hook: "位置 0 的草稿 token 是 10，目标概率 p=0.8，草稿概率 q=0.5。",
        question: "TODO 1 应执行什么？",
        options: ["直接把 10 追加到 accepted_tokens", "先调用 torch.rand 再决定", "立刻 break"],
        answer: 0,
        revealNote: "p>=q 时 min(1,p/q)=1，没有随机拒绝的可能。"
      },
      checkpoint: checkpoint(
        "为什么读取概率要写 [i, token_id]？",
        ["既对齐草稿位置，又对齐该位置实际采样的 token", "为了把 [K,V] 转置", "为了一次取出整张词表"],
        0,
        "接受规则只针对草稿模型在第 i 步实际选择的 token，而不是整行所有 token。"
      ),
      homework: [
        "完成 TODO 1 的 if 分支，只在 p>=q 时 append；不要在该分支消耗随机数。",
        "手工核对可见测试第 0 项：行 0、列 10，p=0.8、q=0.5，应无条件接受 token 10。",
        "若 accepted_tokens 中出现错误 token，检查 append 的是不是 token_id；若第一项触发随机数，检查比较方向是否写反。"
      ]
    }),

    lesson({
      id: "speculative-random-stop",
      title: "p 小于 q 时掷一次硬币；一旦拒绝，后续草稿全部停止",
      todo: "TODO 2：r = torch.rand(1).item()；r < p/q 时接受，否则 break",
      prerequisite: [
        "只有 p<q 才进入随机分支，此时接受概率 p/q 位于 0 到 1 之间。",
        "torch.rand(1) 返回 shape [1] 的张量；.item() 把单元素张量转为 Python float，便于与 p/q 比较。",
        "接受时继续 for 循环验证下一个草稿 token；拒绝时必须 break，而不是 continue。",
        "后续草稿建立在前面 token 已接受的前缀上；当前 token 被拒绝后，后续条件上下文已经不成立。"
      ],
      intuition: "当小模型比大模型更自信时，需要按 p/q 打折：随机数落在接受区间内就保留，否则拒绝。拒绝不仅影响当前 token，还切断整条草稿链，因为后面的 token 是基于这个未被接受的前缀生成的。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>第 1 项：p/q=0.8，r=0.5</h4>
            ${flow(["0", "r=0.5", "0.8"], 1)}
            <p>r 落在 [0,0.8) 接受区间内，追加 token 并继续。</p>
          </section>
          <section class="adv-panel warn">
            <h4>第 2 项：p/q≈0.11，r=0.9</h4>
            ${flow(["0", "0.11", "r=0.9"], 2)}
            <p>r 超出接受区间，拒绝并 break；第 3 项不再验证。</p>
          </section>
        </div>
        <div class="adv-steps">
          <div><b>1</b><code>r = torch.rand(1).item()</code><span>只在 p&lt;q 的分支生成随机数</span></div>
          <div><b>2</b><code>if r &lt; p / q</code><span>成功则 append 当前 token</span></div>
          <div><b>3</b><code>else: break</code><span>失败立即终止整个 for 循环</span></div>
        </div>
        <div class="adv-callout">可见测试把 torch.rand 临时替换成固定返回 0.5、0.9 的函数。因此随机调用次数和顺序也是测试契约：p>=q 的第 0 项不能提前消耗随机数。</div>
      </div>`,
      syntaxHtml: practice(
        "按通过率逐关检查，失败立即停止",
        [
          "pass_rates = [0.8, 0.2, 0.9]",
          "draws = [0.5, 0.7, 0.1]",
          "passed = []",
          "",
          "for level, (rate, draw) in enumerate(zip(pass_rates, draws)):",
          "    if draw < rate:",
          "        passed.append(level)",
          "    else:",
          "        break",
          "",
          "print(passed)  # [0]"
        ],
        "从语法例子迁移到 TODO 2",
        [
          ["rate", "对应 p/q，本轮草稿 token 的接受概率"],
          ["draw", "对应 r，由 torch.rand(1).item() 生成"],
          ["passed.append", "对应接受后 append token_id"],
          ["break", "对应拒绝当前 token 后停止验证后续草稿"],
          ["passed", "对应最终 accepted_tokens，保持前缀顺序"]
        ]
      ),
      predict: {
        hook: "前两项已接受；第三项 p=0.1、q=0.9、r=0.9。",
        question: "函数最终应做什么？",
        options: ["拒绝第三项并停止，返回前两个 token", "跳过第三项后继续验证第四项", "仍接受第三项，因为 r 大"],
        answer: 0,
        revealNote: "p/q≈0.11，r=0.9 不小于接受概率；当前拒绝后后续草稿前缀失效，必须 break。"
      },
      checkpoint: checkpoint(
        "为什么拒绝分支是 break 而不是 continue？",
        ["后续草稿依赖当前 token 已接受的前缀，拒绝后条件上下文改变", "因为 continue 会删除 target_probs", "因为 Python 列表不能继续循环"],
        0,
        "投机验证接受的是连续前缀；中间出现拒绝后，不能跳洞继续接受后续 token。"
      ),
      homework: [
        "完成 TODO 2：只在 p<q 时生成一个随机数，r<p/q 则 append，否则 break。",
        "按可见测试走表：第 0 项必接受，第 1 项用 r=0.5 接受，第 2 项用 r=0.9 拒绝，最终严格等于 [10,20]。",
        "若结果包含 40，检查是否误用 continue；若第 1 项被拒绝，检查随机数调用是否被第 0 项提前消耗或比较符号是否写反。"
      ]
    })
  ],

  "24": [
    lesson({
      id: "radix-lcp",
      title: "最长公共前缀：双重边界内逐位比较，第一次不同就停止",
      todo: "TODO 1：实现 _lcp_len(cached_tokens, prompt_tokens)",
      prerequisite: [
        "cached_tokens 和 prompt_tokens 都是 Python token_id 序列，本题使用列表，不涉及张量 shape、dtype 或 device。",
        "公共前缀必须从索引 0 连续开始；中间一旦不同，后面即使再次相等也不能计入。",
        "循环条件要同时保护两个序列边界，避免一个短序列已经结束却继续索引。",
        "match_len 同时表示当前比较索引和已经连续匹配的 token 数。"
      ],
      intuition: "LCP 不是统计两列里共有多少 token，而是数“开头连续相同了多久”。因此算法只需要一个从 0 开始的指针：两边都没结束且当前位置相等就加一，遇到第一次不同立刻停。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>[1,2,3] 与 [1,2,4]</h4>
            ${flow(["索引0：1=1", "索引1：2=2", "索引2：3≠4"], 1)}
            <p>连续命中长度为 2，第三项不同后立即停止。</p>
          </section>
          <section class="adv-panel blue">
            <h4>[7,8] 与 [7,8,9,10]</h4>
            ${flow(["索引0命中", "索引1命中", "缓存序列结束"], 1)}
            <p>短序列整体是长序列前缀，返回 2，不会越界。</p>
          </section>
        </div>
        ${contract([
          ["输入类型", "两个 Python token 序列"],
          ["match_len 初值", "0"],
          ["继续条件", "match_len 同时小于两个序列长度"],
          ["相等", "match_len += 1"],
          ["不等", "break 并返回当前 match_len"]
        ])}
        <div class="adv-callout">不要用集合交集。集合会丢掉顺序和连续性，例如 [1,2,3] 与 [1,9,3] 有两个共同元素，但最长公共前缀只有 1。</div>
      </div>`,
      syntaxHtml: practice(
        "比较两个路径开头连续相同的站点",
        [
          "saved_route = ['A', 'B', 'C']",
          "new_route = ['A', 'B', 'D', 'E']",
          "same = 0",
          "",
          "while same < len(saved_route) and same < len(new_route):",
          "    if saved_route[same] == new_route[same]:",
          "        same += 1",
          "    else:",
          "        break",
          "",
          "print(same)  # 2"
        ],
        "从语法例子迁移到 TODO 1",
        [
          ["saved_route", "对应 cached_tokens"],
          ["new_route", "对应 prompt_tokens"],
          ["same", "对应 match_len，既是索引也是命中长度"],
          ["两个 len 条件", "分别防止缓存路径或新 prompt 越界"],
          ["break", "对应第一次 token 不同后立即停止"]
        ]
      ),
      predict: {
        hook: "缓存是 [1,2,3]，新 prompt 是 [1,9,3]。",
        question: "最长公共前缀长度是多少？",
        options: ["1", "2，因为 1 和 3 都出现", "3"],
        answer: 0,
        revealNote: "索引 1 已经出现 2≠9，公共前缀在这里终止，后面的 3 不再计入。"
      },
      checkpoint: checkpoint(
        "while 条件为什么要同时检查两个序列长度？",
        ["任意一个序列结束都不能再访问该索引", "为了把列表变成张量", "为了自动排序 token"],
        0,
        "完整前缀命中时，一个序列可能更短；双边界能安全返回短序列长度。"
      ),
      homework: [
        "完成 _lcp_len 的 while、相等递增和不等 break；不要使用 set、sort 或 zip 后统计全部相等项。",
        "运行两个基础断言：[1,2,3] 与 [1,2,4] 返回 2；[7,8] 与更长的 [7,8,9,10] 也返回 2。",
        "额外验证空列表与完全不匹配列表都返回 0；若越界，检查 while 是否缺少任一长度条件。"
      ]
    }),

    lesson({
      id: "radix-best-candidate",
      title: "遍历缓存路径：每个 child 算一次 LCP，只保留最大命中",
      todo: "TODO 2：遍历 root.children 并更新 best_match_len",
      prerequisite: [
        "insert(tokens) 会创建 TreeNode(tokens) 并追加到 root.children；当前教学实现只有单层候选路径。",
        "child.key_tokens 是该候选缓存路径的 token 列表。",
        "_lcp_len 已负责比较一个候选与新 prompt；match_prefix 只负责在多个候选结果中取最大值。",
        "best_match_len 从 0 开始，因此没有 child 或完全无命中时会自然返回 0。"
      ],
      intuition: "这是一个标准的“遍历 + 累计最优值”模式：每看一个候选就计算局部分数 match_len，再与当前最好成绩比较。不要在第一次命中时提前返回，因为后面可能有更长的缓存路径。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>候选 A</b>[0,1,2,3] → 命中 4</span>
          <span><b>候选 B</b>[0,1,2,3,4] → 命中 5</span>
          <span><b>候选 C</b>[9,9,9] → 命中 0</span>
          <span><b>最终最大值</b>best_match_len = 5</span>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>错误：第一次命中就 return</h4>
            <p>候选 A 命中 4 后立即返回，会错过候选 B 的更长命中 5。</p>
          </section>
          <section class="adv-panel good">
            <h4>正确：循环结束后 return</h4>
            <p>每轮只更新 best_match_len，所有 child 都比较完成后再返回。</p>
          </section>
        </div>
        <div class="adv-checks">
          <span><b>状态</b><br>best_match_len 是 Python int</span>
          <span><b>局部值</b><br>match_len 来自 _lcp_len</span>
          <span><b>更新</b><br>只在更大时替换</span>
          <span><b>无命中</b><br>保持 0</span>
        </div>
      </div>`,
      syntaxHtml: practice(
        "从多条历史路线中找最长重合开头",
        [
          "def common_prefix_len(left, right):",
          "    count = 0",
          "    while count < len(left) and count < len(right):",
          "        if left[count] != right[count]:",
          "            break",
          "        count += 1",
          "    return count",
          "",
          "history = [['A', 'B'], ['A', 'B', 'C'], ['X']]",
          "current = ['A', 'B', 'C', 'D']",
          "best = 0",
          "",
          "for route in history:",
          "    local = common_prefix_len(route, current)",
          "    if local > best:",
          "        best = local",
          "",
          "print(best)  # 3"
        ],
        "从语法例子迁移到 TODO 2",
        [
          ["history", "对应 self.root.children"],
          ["route", "对应 child；实际 token 序列在 child.key_tokens"],
          ["common_prefix_len", "对应 self._lcp_len"],
          ["local", "对应当前候选的 match_len"],
          ["best", "对应 best_match_len，循环完成后再返回"]
        ]
      ),
      predict: {
        hook: "第一个 child 与 prompt 命中 4 个 token，第二个 child 命中 5 个。",
        question: "match_prefix 应返回什么？",
        options: ["5", "4，因为先遇到", "9，把两次命中相加"],
        answer: 0,
        revealNote: "目标是所有缓存路径中的最长公共前缀，取最大值而不是第一项或总和。"
      },
      checkpoint: checkpoint(
        "为什么完全不匹配 [7,6,5] 时可以返回 0？",
        ["best_match_len 初始为 0，所有局部命中都不会让它变大", "因为 root.children 会被删除", "因为 token 会自动补零"],
        0,
        "0 正好表示没有可复用前缀，也让后续切片自然得到空前缀和完整后缀。"
      ),
      homework: [
        "完成 root.children 遍历、_lcp_len 调用和最大值更新；把 return 放在 for 循环结束之后。",
        "运行多候选测试，新 prompt [0,1,2,3,4,5] 必须选择 5，而不是先遇到的 4；无命中返回 0。",
        "若返回 4，检查是否提前 return；若返回最后一个候选的 0，检查是否每轮无条件覆盖 best_match_len。"
      ]
    }),

    lesson({
      id: "radix-split-prompt",
      title: "把命中长度变成工程动作：前 H 个复用，剩余后缀重算",
      todo: "TODO 3：hit_len、hit_prefix、miss_suffix 与三元组返回",
      prerequisite: [
        "match_prefix(prompt_tokens) 返回 Python int H，表示可直接复用的最长前缀 token 数。",
        "Python 切片 prompt_tokens[:H] 取前 H 项，prompt_tokens[H:] 取从 H 开始的剩余项。",
        "H=0 时前缀切片自然是 []，后缀自然是完整 prompt；H 等于 prompt 长度时后缀自然为空。",
        "函数返回顺序必须严格是 hit_prefix、miss_suffix、hit_len，测试会按这个顺序解包。"
      ],
      intuition: "前两步只算出了一个数字 H，这一步才把 H 变成推理系统的工作划分：前 H 个 token 的 KV Cache 可以复用，H 之后的 token 必须送给模型重新计算。切片正好表达这条边界。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>命中 H=5</h4>
            ${flow(["prompt [0,1,2,3,4,5]", "hit [0,1,2,3,4]", "miss [5]"], 1)}
            <p>前五个 token 复用缓存，只计算最后一个 token。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>无命中 H=0</h4>
            ${flow(["prompt [7,6,5]", "hit []", "miss [7,6,5]"], 2)}
            <p>没有可复用缓存，完整 prompt 都进入重算后缀。</p>
          </section>
        </div>
        ${contract([
          ["hit_len", "match_prefix(prompt_tokens) 的返回值"],
          ["hit_prefix", "prompt_tokens[:hit_len]"],
          ["miss_suffix", "prompt_tokens[hit_len:]"],
          ["返回顺序", "(hit_prefix, miss_suffix, hit_len)"],
          ["系统含义", "前缀复用 KV，后缀重新执行模型计算"]
        ])}
        <div class="adv-callout">不要从某个 child.key_tokens 直接返回前缀。测试契约要求拆分当前 prompt_tokens；这样返回内容一定保持新请求本身的 token 序列。</div>
      </div>`,
      syntaxHtml: practice(
        "按已下载长度拆分文件任务",
        [
          "all_parts = ['p0', 'p1', 'p2', 'p3']",
          "cached_count = 3",
          "reused = all_parts[:cached_count]",
          "remaining = all_parts[cached_count:]",
          "",
          "result = (reused, remaining, cached_count)",
          "print(result)  # (['p0','p1','p2'], ['p3'], 3)"
        ],
        "从语法例子迁移到 TODO 3",
        [
          ["all_parts", "对应 prompt_tokens"],
          ["cached_count", "对应 hit_len，由 match_prefix 计算"],
          ["reused", "对应 hit_prefix，切片终点不包含 hit_len"],
          ["remaining", "对应 miss_suffix，从 hit_len 开始"],
          ["result", "对应函数的三元组返回顺序"]
        ]
      ),
      predict: {
        hook: "prompt=[0,1,2,3,4,5]，match_prefix 返回 hit_len=5。",
        question: "split_prompt 应返回什么？",
        options: [
          "([0,1,2,3,4], [5], 5)",
          "([0,1,2,3,4,5], [], 5)",
          "([5], [0,1,2,3,4], 5)"
        ],
        answer: 0,
        revealNote: "[:5] 取索引 0 到 4，[5:] 从索引 5 开始，返回顺序是前缀、后缀、长度。"
      },
      checkpoint: checkpoint(
        "hit_len=0 时为什么不需要额外 if 分支？",
        ["Python 的 [:0] 自动得到空列表，[0:] 自动得到完整列表", "因为 match_prefix 会抛异常", "因为 0 会自动变成 None"],
        0,
        "切片天然覆盖无命中边界，使实现既短又与测试契约一致。"
      ),
      homework: [
        "完成 hit_len 调用、两个切片和三元组返回，严格保持返回顺序。",
        "运行命中与无命中测试：H=5 时返回五项前缀和单项后缀；H=0 时返回空前缀和完整后缀。",
        "若前缀多一项，检查切片是否误写成 :hit_len+1；若解包内容错位，检查 return 三项顺序。"
      ]
    })
  ]
};
