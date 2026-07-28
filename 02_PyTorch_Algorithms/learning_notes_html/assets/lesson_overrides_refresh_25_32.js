const {
  advancedStyles,
  checkpoint,
  code,
  lesson
} = require("../advanced_lesson_helpers");
const lessons25To28 = require("../lesson_overrides_25_28");
const lessons29To32 = require("../lesson_overrides_29_32");

module.exports = {
  "25": lessons25To28["25"],
  "26": lessons25To28["26"],
  "27": lessons25To28["27"],
  "28": lessons25To28["28"],
  "29": lessons29To32["29"],

  "30": [
    lesson({
      id: "lora-parameter-ledger",
      title: "先把矩阵数清楚：LoRA 参数量不是凭感觉省",
      todo: "TODO 1-3：lora_trainable_params、full_linear_params 与 lora_param_ratio",
      prerequisite: [
        "一个 Linear 的 weight shape 是 [out_dim, in_dim]，元素个数等于 out_dim * in_dim；本 Notebook 明确不统计 bias。",
        "LoRA 不直接训练完整 weight，而是训练 A 和 B：A 的 shape 是 [rank, in_dim]，B 的 shape 是 [out_dim, rank]。",
        "张量的参数量就是各维度相乘；两个独立矩阵的参数量要相加，不能相乘。",
        "参数占比 ratio 的分母是完整 Linear 的参数量，分子是 LoRA A、B 的可训练参数总量。"
      ],
      intuition: "把完整 Linear 想成一张 out_dim 行、in_dim 列的大表。LoRA 不重写整张表，而是保存两张窄表 A 和 B。先分别数两张窄表有多少格，再相加；最后除以大表格数，才知道“省了多少”是否有数据依据。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>认 shape</b>A=[rank,in]，B=[out,rank]</span>
          <span><b>数元素</b>每个矩阵的各维相乘</span>
          <span><b>合账本</b>rank*in + out*rank</span>
          <span><b>算占比</b>LoRA 参数 / 完整 weight 参数</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>LoRA 旁路的参数账</h4>
            <div class="adv-contract">
              <span><code>A</code></span><strong><code>[rank, in_dim]</code>，参数量 rank * in_dim</strong>
              <span><code>B</code></span><strong><code>[out_dim, rank]</code>，参数量 out_dim * rank</strong>
              <span>总可训练量</span><strong><code>rank * (in_dim + out_dim)</code></strong>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>全参数 Linear 的基线账</h4>
            <div class="adv-contract">
              <span><code>weight</code></span><strong><code>[out_dim, in_dim]</code></strong>
              <span>总参数量</span><strong><code>in_dim * out_dim</code></strong>
              <span>本课边界</span><strong>只统计 weight，不额外加 bias</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>测试输入</th><th>A 参数</th><th>B 参数</th><th>LoRA 总量</th><th>全参总量</th><th>ratio</th></tr></thead>
          <tbody>
            <tr><td><code>in=8,out=8,rank=2</code></td><td>16</td><td>16</td><td><strong>32</strong></td><td><strong>64</strong></td><td><strong>0.5</strong></td></tr>
          </tbody>
        </table>

        <div class="adv-callout">初学者最常见的错误是写成 rank * in_dim * out_dim。LoRA 有两张矩阵，所以是两个参数量相加；B @ A 只是在前向时组合，不会把参数个数相乘。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用两段窄木板练习“各自计数再相加”", [
          "def adapter_cells(input_width, output_width, bottleneck):",
          "    down_cells = bottleneck * input_width",
          "    up_cells = output_width * bottleneck",
          "    return down_cells + up_cells",
          "",
          "def full_grid_cells(input_width, output_width):",
          "    return input_width * output_width",
          "",
          "small = adapter_cells(6, 10, 2)   # 12 + 20 = 32",
          "full = full_grid_cells(6, 10)     # 60",
          "share = small / full"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>input_width/output_width/bottleneck</code> -&gt; <code>in_dim/out_dim/rank</code>。</li>
            <li><code>down_cells + up_cells</code> -&gt; TODO 1 的 <code>rank * (in_dim + out_dim)</code>。</li>
            <li><code>full_grid_cells</code> -&gt; TODO 2，只计算 <code>in_dim * out_dim</code>。</li>
            <li><code>share</code> -&gt; TODO 3 的 <code>trainable / total</code>；先调用前两个 helper，不要重复写两套公式。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "一个 Linear 的 in_dim=8、out_dim=8，LoRA rank=2。",
        question: "A 和 B 一共有多少个可训练参数？",
        options: ["32", "128", "16"],
        answer: 0,
        revealNote: "A 有 2*8=16 个参数，B 有 8*2=16 个参数，合计 32。"
      },
      checkpoint: checkpoint(
        "为什么 lora_param_ratio 应调用前两个计数函数再相除？",
        ["复用同一套参数定义，避免三个函数的口径漂移", "这样会让 ratio 自动变成整数", "因为 Python 不允许在函数里做乘法"],
        0,
        "把计数逻辑集中在两个 helper 中，后续修改统计口径时 ratio 也会自动保持一致。"
      ),
      homework: [
        "完成 TODO 1-3，并先手算测试用例：trainable=32、total=64、ratio=0.5，再运行 Notebook 断言。",
        "再自测 in_dim=6、out_dim=10、rank=2：期待 LoRA 参数 32、全参 60、ratio 约 0.5333；说明 rank 并非越小就一定远小于全参。",
        "若结果大很多，检查是否把 A、B 参数量相乘；若 ratio > 1，检查 rank 是否过大以及分子分母是否写反；若差一个 out_dim，检查是否误加了 bias。"
      ]
    }),

    lesson({
      id: "lora-project-summary",
      title: "再统一差值方向：资源看节省，loss 看代价",
      todo: "TODO 4：summarize_lora_project 的四个项目指标",
      prerequisite: [
        "baseline_metrics 与 lora_metrics 都是 Python 字典，必须用方括号和键名读取数值，例如 metrics['peak_mem_mb']。",
        "param_reduction 是比例：1 - LoRA 可训练参数 / baseline 可训练参数；0.5 表示减少 50%。",
        "资源类 delta 使用 baseline - LoRA：结果为正表示 LoRA 更省显存或更快。",
        "loss_delta 使用 LoRA - baseline：结果为正表示 LoRA 的最终 loss 更高，是效果代价；四个字段最后按 Notebook 要求 round。"
      ],
      intuition: "报告里的正负号必须先约定语义。显存和时间问的是“省了多少”，所以旧值减新值；loss 问的是“比 baseline 多坏了多少”，所以 LoRA 减 baseline。公式不是随便统一成一个减法方向，而是让每个数字回答清楚的问题。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>资源类：正数是收益</h4>
            <div class="adv-contract">
              <span>参数减少率</span><strong><code>1 - lora_params / baseline_params</code></strong>
              <span>显存差</span><strong><code>baseline_mem - lora_mem</code></strong>
              <span>时间差</span><strong><code>baseline_time - lora_time</code></strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>效果类：正数是代价</h4>
            <div class="adv-contract">
              <span>loss 差</span><strong><code>lora_loss - baseline_loss</code></strong>
              <span>正值</span><strong>LoRA loss 更高，效果略差</strong>
              <span>负值</span><strong>LoRA loss 更低，效果更好</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>字段</th><th>Notebook 测试数据</th><th>期待值</th><th>如何解释</th></tr></thead>
          <tbody>
            <tr><td><code>param_reduction</code></td><td>64 -&gt; 32</td><td>0.5</td><td>可训练参数减少 50%</td></tr>
            <tr><td><code>peak_mem_delta_mb</code></td><td>1024 -&gt; 768</td><td>256.0</td><td>节省 256 MB</td></tr>
            <tr><td><code>step_time_delta_ms</code></td><td>20 -&gt; 22</td><td>-2.0</td><td>LoRA 反而慢 2 ms</td></tr>
            <tr><td><code>loss_delta</code></td><td>0.50 -&gt; 0.52</td><td>0.02</td><td>LoRA loss 高 0.02</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要看到负数就认定代码错误。Notebook 的时间测试故意让 LoRA 更慢，因此 step_time_delta_ms 必须是 -2.0；这是项目需要记录的取舍。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用压缩文件方案练习收益与质量代价的不同方向", [
          "original = {'bytes': 1000, 'load_ms': 8.0, 'quality_loss': 0.00}",
          "packed = {'bytes': 650, 'load_ms': 9.5, 'quality_loss': 0.03}",
          "",
          "size_saved = original['bytes'] - packed['bytes']",
          "time_saved = original['load_ms'] - packed['load_ms']",
          "quality_delta = packed['quality_loss'] - original['quality_loss']",
          "report = {",
          "    'size_reduction': round(1 - packed['bytes'] / original['bytes'], 4),",
          "    'time_delta_ms': round(time_saved, 2),",
          "    'quality_delta': round(quality_delta, 4),",
          "}"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>original/packed</code> -&gt; <code>baseline_metrics/lora_metrics</code>。</li>
            <li><code>size_reduction</code> -&gt; <code>param_reduction</code>，都使用 <code>1 - candidate / baseline</code>。</li>
            <li><code>time_saved</code> -&gt; <code>time_delta</code>；正数才表示候选更快。</li>
            <li><code>quality_delta</code> -&gt; <code>loss_delta</code>；按候选减 baseline 记录质量代价。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "baseline 每步 20 ms，LoRA 每步 22 ms；时间差定义为 baseline - LoRA。",
        question: "step_time_delta_ms 应该是多少，表示什么？",
        options: ["-2.0，LoRA 慢 2 ms", "2.0，LoRA 快 2 ms", "0.0，两者相同"],
        answer: 0,
        revealNote: "20-22=-2。负数保留了真实性：这组示例中，参数与显存收益伴随着时间代价。"
      },
      checkpoint: checkpoint(
        "baseline loss=0.50、LoRA loss=0.52 时，为什么 loss_delta 是 +0.02？",
        ["它按 LoRA - baseline 记录效果代价", "它按 baseline - LoRA 记录节省", "round 会自动改变符号"],
        0,
        "loss 越低通常越好，因此候选减 baseline 的正值能直接表示增加了多少 loss。"
      ),
      homework: [
        "完成 TODO 4，严格返回 param_reduction、peak_mem_delta_mb、step_time_delta_ms、loss_delta 四个键，并使用 Notebook 指定的小数位。",
        "用当前测试数据逐个验算 0.5、256.0、-2.0、0.02；再增加“LoRA 更快且 loss 更低”的反向案例，观察两个 delta 的符号。",
        "若 KeyError，逐字检查输入字典键；若 param_reduction 得到 32，说明写成了参数数量差而非比例；若 loss_delta 为 -0.02，检查唯一采用 candidate-baseline 的字段。"
      ]
    }),

    lesson({
      id: "lora-controlled-project",
      title: "最后把账本变成结论：固定实验，只改变 LoRA 方案",
      todo: "项目交付：用四个 helper 支撑 baseline vs LoRA 的可复现实验报告",
      prerequisite: [
        "四个 helper 只完成参数和指标汇总，不会替你训练模型；完整项目还需固定数据、batch size、seq len、optimizer、学习率和 step 数。",
        "LoRA 减少的是可训练参数、梯度和相应优化器状态；冻结底座仍要存储并参与前向，不能把 param_reduction 当作总显存减少率。",
        "同口径对比至少要同时看 trainable params、step time、peak memory 与 final loss，单一指标不能决定方案。",
        "rank、插层位置、学习率和训练步数是结论可复现的必要配置。"
      ],
      intuition: "Notebook 的函数像实验室里的四个量具：会数参数、算比例、比较资源与效果。但项目结论还需要你确保两次实验使用同一把尺、同一批样本。只有一次改一个变量，差值才可以归因给 LoRA。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>固定 baseline</b>模型、数据、batch、seq、steps</span>
          <span><b>记录 LoRA</b>rank、插层、学习率</span>
          <span><b>同口径测量</b>参数、时间、显存、loss</span>
          <span><b>调用 summary</b>保留收益与代价</span>
          <span><b>写出判断</b>是否值得采用及下一轮</span>
        </div>

        <div class="adv-grid three">
          <section class="adv-panel blue"><h4>参数问题</h4><p>少训练了多少？用 parameter ledger 和 ratio 回答。</p></section>
          <section class="adv-panel good"><h4>资源问题</h4><p>显存和 step time 变化多少？保留正负号。</p></section>
          <section class="adv-panel warn"><h4>效果问题</h4><p>loss 是否仍可接受？不能只因显存下降就通过。</p></section>
        </div>

        <div class="adv-contract">
          <span>公平对照</span><strong>除 LoRA 插入方案外，其余训练条件保持一致</strong>
          <span>本课代码产物</span><strong>参数账本 + baseline/LoRA summary 字典</strong>
          <span>完整项目产物</span><strong>对比表、loss 曲线、profiling 记录、配置与结论</strong>
          <span>结论句式</span><strong>参数/显存收益 + 时间/效果代价 + 是否接受 + 下一步</strong>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用配置字典检查两次实验只改变 adapter", [
          "baseline_cfg = {'batch': 4, 'seq': 128, 'lr': 1e-3, 'steps': 200}",
          "lora_cfg = {'batch': 4, 'seq': 128, 'lr': 1e-3, 'steps': 200}",
          "",
          "for key in ['batch', 'seq', 'lr', 'steps']:",
          "    assert baseline_cfg[key] == lora_cfg[key]",
          "",
          "baseline_metrics = {",
          "    'trainable_params': 64, 'step_time_ms': 20.0,",
          "    'peak_mem_mb': 1024.0, 'final_loss': 0.50",
          "}",
          "lora_metrics = {",
          "    'trainable_params': 32, 'step_time_ms': 22.0,",
          "    'peak_mem_mb': 768.0, 'final_loss': 0.52",
          "}"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook 项目</h4>
          <ul>
            <li>配置相等断言 -&gt; Step 1-3 的“固定 baseline、同口径对比”。</li>
            <li>两个 metrics 字典 -&gt; TODO 4 的真实输入契约。</li>
            <li>Notebook helper 的 summary -&gt; 对比表中的差值列，而不是完整训练过程本身。</li>
            <li>最终还需记录 rank 与插层位置，解释为什么选择或放弃当前 LoRA 方案。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "LoRA 实验把 batch 从 4 改成 2，同时显存从 1024 MB 降到 700 MB。",
        question: "能否把全部显存下降归因于 LoRA？",
        options: ["不能，batch 也变化了，实验存在混杂变量", "能，只要用了 LoRA 就可以", "能，因为显存单位相同"],
        answer: 0,
        revealNote: "batch 变小本身就会减少激活显存。必须固定 batch 才能解释 LoRA 的独立影响。"
      },
      checkpoint: checkpoint(
        "哪一句最符合本课的完整项目结论？",
        ["LoRA 参数与显存下降，但每步慢 2 ms、loss 高 0.02；在当前效果阈值可接受时保留，并继续调 rank", "LoRA 参数少，所以一定更快、更准", "测试通过，因此不需要记录训练配置"],
        0,
        "项目判断要同时陈述收益、代价、约束与下一步，而不是把单个指标当成全部答案。"
      ),
      homework: [
        "先通过四个 helper 的基础测试，再为一个自选 hidden_size/rank 生成参数账本，写出具体参数占比。",
        "补一张 baseline vs LoRA 表，至少记录 trainable_params、step_time_ms、peak_mem_mb、final_loss、rank、插层位置、lr 和 steps。",
        "写一条可审计结论：收益、代价、是否达到 loss 阈值、下一轮只调整哪个变量；若结果不可解释，先检查两份配置是否真正同口径。"
      ]
    })
  ],

  "31": [
    lesson({
      id: "inference-benchmark-contract",
      title: "先学会公平计时：预热不进账，正式迭代求平均",
      todo: "TODO 1：benchmark_fn 的 warmup、perf_counter 与平均耗时",
      prerequisite: [
        "Python 函数可以作为参数传入另一个函数；在 benchmark_fn 中写 fn() 才是真正执行一次待测任务。",
        "range(warmup) 和 range(iters) 分别控制预热次数与正式测量次数，两个循环不能混成一个。",
        "time.perf_counter() 返回高精度单调时钟读数；结束时间减开始时间才是正式阶段总秒数。",
        "Notebook 的返回值是每次调用的平均秒数 total / iters，不是毫秒，也不包括 warmup。"
      ],
      intuition: "第一次运行常包含初始化和缓存成本，所以先让任务“热身”，但热身成绩不计入比赛。正式开始后只在循环外读一次起点、一次终点，再用总时间除以次数，所有候选方案才使用同一计时口径。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>for _ in range(warmup): fn()</code><span>执行预热，不计时</span></div>
          <div><b>2</b><code>start = time.perf_counter()</code><span>正式循环前记录起点</span></div>
          <div><b>3</b><code>for _ in range(iters): fn()</code><span>只测正式迭代</span></div>
          <div><b>4</b><code>end - start</code><span>得到总秒数，再除以 iters</span></div>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>测试直接约束调用次数</h4>
            <div class="adv-contract">
              <span>输入</span><strong><code>warmup=0, iters=3</code></strong>
              <span>fn 的行为</span><strong>每次让 <code>counter['n'] += 1</code></strong>
              <span>断言</span><strong><code>counter['n'] == 3</code>，平均值非负</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>不要把计时器放进循环</h4>
            <p>每轮反复重置 start，最后只会保留最后一次间隔；本课要测整个正式阶段，再求平均。</p>
          </section>
        </div>

        <div class="adv-callout">这个教学 helper 没有 CUDA synchronize。用于当前 CPU-first 测试时保持 Notebook 原契约；真实 GPU benchmark 还要在计时边界处理异步执行。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用文本清洗函数练习预热与平均计时", [
          "import time",
          "",
          "def clean_text():",
          "    '  hello  '.strip().upper()",
          "",
          "for _ in range(2):",
          "    clean_text()",
          "",
          "begin = time.perf_counter()",
          "for _ in range(5):",
          "    clean_text()",
          "finish = time.perf_counter()",
          "average_seconds = (finish - begin) / 5"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>clean_text</code> -&gt; <code>fn</code>，函数对象传入后在循环中用括号调用。</li>
            <li><code>2</code> -&gt; <code>warmup</code>，该循环发生在 start 之前。</li>
            <li><code>5</code> -&gt; <code>iters</code>，总耗时只覆盖正式循环。</li>
            <li><code>finish - begin</code> -&gt; <code>total</code>，TODO 返回 <code>total / iters</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "benchmark_fn(fn, warmup=2, iters=3)，fn 每次让计数器加 1。",
        question: "整个函数返回前，fn 一共被调用几次？",
        options: ["5 次", "3 次", "2 次"],
        answer: 0,
        revealNote: "预热调用 2 次，正式调用 3 次；平均耗时只统计后 3 次，但函数总调用数是 5。"
      },
      checkpoint: checkpoint(
        "start 应放在哪里，才能排除预热开销？",
        ["warmup 循环之后、正式 iters 循环之前", "两个循环之前", "正式循环内部最后一行"],
        0,
        "计时窗口必须只包围正式测量循环。"
      ),
      homework: [
        "完成 TODO 1，运行 warmup=0、iters=3 测试，确认 counter 恰好为 3 且返回值非负。",
        "再用 warmup=2、iters=4 自测计数器应为 6；打印 total 与 total/iters，确认函数返回平均秒数。",
        "若调用次数多了，检查是否在计时前后额外调用 fn；若结果大约差 iters 倍，检查是否忘记除法；若 GPU 数据虚低，记录异步同步是工程扩展，不要擅自改变当前测试契约。"
      ]
    }),

    lesson({
      id: "inference-summary-math",
      title: "把阶段时间翻译成指标：总延迟、占比与吞吐",
      todo: "TODO 2：summarize_inference_result 的计算、除零保护与返回字段",
      prerequisite: [
        "prefill_ms 与 decode_ms 都以毫秒为单位，总延迟是两者相加。",
        "decode_share 是比例，用 decode_ms / total_ms；总时间为 0 时返回 0.0，避免除零。",
        "throughput 是每秒 token 数，所以要先把 total_ms 除以 1000 转成秒，再用 generated_tokens 除以秒数。",
        "返回字典必须保留六个精确键，并按 Notebook 要求分别 round 到 2 位或 3 位。"
      ],
      intuition: "同一组阶段时间可以回答三种不同问题：用户一共等多久、等待主要花在哪一阶段、每秒产出多少 token。单位和分母不同，所以必须先写清公式，再填返回字典。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>prefill<br><code>10 ms</code></span>
          <span>decode<br><code>5 ms</code></span>
          <strong>total<br><code>15 ms</code></strong>
          <span>decode share<br><code>5/15=0.333</code></span>
          <span>throughput<br><code>100/0.015</code></span>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>返回键</th><th>公式或来源</th><th>测试期待</th><th>round</th></tr></thead>
          <tbody>
            <tr><td><code>prefill_ms</code></td><td>输入原值</td><td>10.0</td><td>2 位</td></tr>
            <tr><td><code>decode_ms</code></td><td>输入原值</td><td>5.0</td><td>2 位</td></tr>
            <tr><td><code>total_ms</code></td><td>prefill + decode</td><td>15.0</td><td>2 位</td></tr>
            <tr><td><code>decode_share</code></td><td>decode / total</td><td>0.333</td><td>3 位</td></tr>
            <tr><td><code>throughput_tok_s</code></td><td>tokens / (total_ms / 1000)</td><td>6666.67</td><td>2 位</td></tr>
            <tr><td><code>peak_mem_mb</code></td><td>输入原值</td><td>256.0</td><td>2 位</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">throughput 的常见千倍错误来自直接用 token / 毫秒。函数名写着 tok_s，分母必须是秒。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用下载过程练习阶段占比和每秒产出", [
          "connect_ms = 20.0",
          "transfer_ms = 80.0",
          "bytes_received = 5000",
          "",
          "total_ms = connect_ms + transfer_ms",
          "transfer_share = transfer_ms / total_ms if total_ms else 0.0",
          "bytes_per_s = (",
          "    bytes_received / (total_ms / 1000.0)",
          "    if total_ms and bytes_received else 0.0",
          ")"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>connect_ms/transfer_ms</code> -&gt; <code>prefill_ms/decode_ms</code>。</li>
            <li><code>transfer_share</code> -&gt; <code>decode_share</code>，分母为两个阶段总和。</li>
            <li><code>bytes_received</code> -&gt; <code>generated_tokens</code>，都除以总秒数得到每秒产出。</li>
            <li>例子中的条件表达式 -&gt; Notebook 对 total_ms=0 或 token=0 的除零保护。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prefill=10 ms、decode=5 ms、生成 100 tokens，总时间是 15 ms。",
        question: "正确的 throughput 约是多少？",
        options: ["6666.67 tok/s", "6.67 tok/s", "1500 tok/s"],
        answer: 0,
        revealNote: "15 ms=0.015 s，100/0.015=6666.67 tok/s。"
      },
      checkpoint: checkpoint(
        "total_ms 为 0 时，decode_share 和 throughput 应如何处理？",
        ["返回 0.0，避免除零", "直接相除得到可用结果", "删除这两个返回键"],
        0,
        "Notebook 参考实现用条件表达式为零时间建立明确边界。"
      ),
      homework: [
        "完成 TODO 2，并用 10、5、256、100 的测试输入核对 total=15、share=0.333、throughput=6666.67。",
        "补测 prefill=0、decode=0、generated_tokens=100，确认两个除法指标都是 0.0 且六个键仍齐全。",
        "若吞吐差 1000 倍，检查毫秒转秒；若 share 变成百分数 33.3，记住测试要求比例 0.333；若断言缺键，逐字核对返回字典。"
      ]
    }),

    lesson({
      id: "inference-report-and-experiment",
      title: "把数字排成可比证据：表头、行字段与单变量实验",
      todo: "TODO 3：format_comparison_report 与 baseline/candidate 项目报告",
      prerequisite: [
        "Markdown 表格至少需要表头、分隔行和数据行；每行的竖线列数必须一致。",
        "rows 中每个元素是二元 tuple，可以在 for name, summary in rows 中直接拆成配置名与指标字典。",
        "summary 的主要字段用方括号读取；note 是可选字段，使用 summary.get('note', '') 可避免 KeyError。",
        "多行字符串用 '\\n'.join(lines) 连接；Notebook 测试要求结果包含 Baseline、Candidate 与 throughput(tok/s)。"
      ],
      intuition: "benchmark 和 summary 产生数字，report 才让两组数字站在同一列里。固定列顺序不仅为了好看，还能阻止比较时偷偷换口径。真正的项目再用这张表回答：只改一个变量后，哪项改善、哪项退化。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>header</code><span>声明六列名称，包含 throughput(tok/s)</span></div>
          <div><b>2</b><code>sep</code><span>为每一列提供 Markdown 分隔符</span></div>
          <div><b>3</b><code>lines = [header, sep]</code><span>先建立表格骨架</span></div>
          <div><b>4</b><code>for name, summary in rows</code><span>按相同字段拼 baseline 与 candidate</span></div>
          <div><b>5</b><code>"\\n".join(lines)</code><span>输出完整多行文本</span></div>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>固定的字段顺序</h4>
            <p>配置、prefill、decode、throughput、peak memory、备注。每个数据行都必须按这个顺序取值。</p>
          </section>
          <section class="adv-panel good">
            <h4>公平实验的边界</h4>
            <p>模型、输入集、batch、seq len、硬件与测量方式固定，一轮只改变后端、精度、cache 或其他一个变量。</p>
          </section>
        </div>

        <div class="adv-contract">
          <span>基础测试</span><strong>字符串含 Baseline、Candidate、throughput(tok/s)</strong>
          <span>报告意义</span><strong>同时比较 latency、throughput、peak memory，不靠单指标选型</strong>
          <span>完整产物</span><strong>对比表 + profiling 证据 + 瓶颈判断 + 下一轮计划</strong>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用两家快递数据练习逐行拼 Markdown 表格", [
          "rows = [",
          "    ('Route-A', {'time': 12.0, 'cost': 8.0, 'note': 'baseline'}),",
          "    ('Route-B', {'time': 9.5, 'cost': 10.0}),",
          "]",
          "lines = [",
          "    '| 路线 | 时间 | 成本 | 备注 |',",
          "    '| --- | --- | --- | --- |',",
          "]",
          "for name, result in rows:",
          "    row = f\"| {name} | {result['time']} | {result['cost']} | {result.get('note', '')} |\"",
          "    lines.append(row)",
          "report = '\\n'.join(lines)"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>Route-A/Route-B</code> -&gt; <code>baseline_name/candidate_name</code>。</li>
            <li><code>result</code> -&gt; 每一份 <code>summary</code> 字典。</li>
            <li>四列表头 -&gt; Notebook 的六列表头，必须包含精确文本 <code>throughput(tok/s)</code>。</li>
            <li><code>result.get(...)</code> -&gt; 可选 note；其余五个数值字段按规定键读取。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "candidate 的 summary 没有 note 键。",
        question: "使用 summary.get('note', '') 的结果是什么？",
        options: ["返回空字符串，表格仍可生成", "立即抛出 KeyError", "删除整行 candidate"],
        answer: 0,
        revealNote: "dict.get 允许提供默认值，这里用空字符串填可选备注列。"
      },
      checkpoint: checkpoint(
        "为什么 baseline 与 candidate 必须使用同一字段顺序？",
        ["让同列代表同一指标，比较才不会错位", "让 throughput 自动变大", "这样不需要运行 benchmark"],
        0,
        "固定 schema 是公平比较的基础，视觉对齐背后是语义对齐。"
      ),
      homework: [
        "完成 TODO 3，确认报告含两组名称、throughput(tok/s) 表头，且每行竖线列数一致。",
        "分别调用 TODO 1-2 生成 baseline 与 candidate summary，再交给 TODO 3；写出一条“收益 + 代价 + 约束下选择”的结论。",
        "若 lines 未定义，检查是否先初始化 [header, sep]；若出现 KeyError，核对 summary 六个键并只对 note 使用 get；若表格错列，逐项比较 header 与 report_row 顺序。"
      ]
    })
  ],

  "32": [
    lesson({
      id: "training-step-measurement",
      title: "量一轮训练：预热、平均 step time 与条件显存统计",
      todo: "TODO 1：measure_train_step 的计时、平均值和 CUDA 峰值显存",
      prerequisite: [
        "train_step_fn 是可调用函数；warmup 与 iters 都会真正执行它，但只有 iters 循环进入正式计时。",
        "elapsed 必须是 (end - start) / iters，单位为秒/step；返回时再乘 1000 变成 step_time_ms。",
        "CUDA 峰值统计只在 torch.cuda.is_available() 为 True 时调用，CPU 环境保持 peak_mem_mb=0.0。",
        "torch.cuda.max_memory_allocated() 返回 bytes，除以 1024 ** 2 才是 MB；重置峰值要放在 warmup 后、正式计时前。"
      ],
      intuition: "训练测量有两把尺：时间尺算每一步平均多久，显存尺记录正式阶段最高水位。先预热，再清空旧峰值，随后测固定轮数；CPU 没有 CUDA 水位计，就明确返回 0.0，让同一个 helper 仍可运行。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>预热</b>执行 warmup 次，不进正式平均</span>
          <span><b>重置峰值</b>仅 CUDA，排除预热分配</span>
          <span><b>正式计时</b>start -&gt; iters 次 -&gt; end</span>
          <span><b>算平均</b>(end-start)/iters</span>
          <span><b>读峰值</b>bytes / 1024²，CPU 保持 0</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>返回字典契约</h4>
            <div class="adv-contract">
              <span><code>step_time_ms</code></span><strong>平均秒数 * 1000，round 到 2 位</strong>
              <span><code>peak_mem_mb</code></span><strong>CUDA 峰值 MB；CPU 为 0.0，round 到 2 位</strong>
            </div>
          </section>
          <section class="adv-panel blue">
            <h4>基础测试契约</h4>
            <div class="adv-contract">
              <span>调用</span><strong><code>warmup=0, iters=2</code></strong>
              <span>counter</span><strong>必须恰好等于 2</strong>
              <span>两个指标</span><strong>键存在且数值都非负</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">当前 Notebook 参考实现没有显式 torch.cuda.synchronize()。先完成页面规定的函数契约；真实 GPU 精准计时要再考虑异步 kernel 的同步边界。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用批量更新任务练习预热、平均耗时与默认资源值", [
          "import time",
          "",
          "state = {'steps': 0}",
          "def update_once():",
          "    state['steps'] += 1",
          "",
          "for _ in range(1):",
          "    update_once()",
          "",
          "start = time.perf_counter()",
          "for _ in range(4):",
          "    update_once()",
          "end = time.perf_counter()",
          "elapsed = (end - start) / 4",
          "result = {'step_ms': round(elapsed * 1000, 2), 'resource_mb': 0.0}"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>update_once</code> -&gt; <code>train_step_fn</code>，每次循环都要真正调用。</li>
            <li><code>1/4</code> -&gt; <code>warmup/iters</code>，elapsed 只除正式迭代数。</li>
            <li><code>step_ms</code> -&gt; <code>step_time_ms</code>，秒乘 1000。</li>
            <li><code>resource_mb=0.0</code> -&gt; CPU 分支；CUDA 分支要重置并读取峰值，再换算 MB。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "measure_train_step(train_step, warmup=0, iters=2)，train_step 每次把 counter 加 1。",
        question: "测试结束时 counter 应是多少？",
        options: ["2", "8", "10"],
        answer: 0,
        revealNote: "预热为 0，正式循环执行两次，所以计数器恰好为 2。"
      },
      checkpoint: checkpoint(
        "为什么 reset_peak_memory_stats 要放在 warmup 之后？",
        ["避免把预热阶段的内存峰值算进正式测量", "把 GPU 张量自动搬到 CPU", "让 iters 自动归零"],
        0,
        "重置后记录的峰值只覆盖本轮正式迭代，更符合对比口径。"
      ),
      homework: [
        "完成 TODO 1，先在 CPU 跑基础测试：counter=2、两个键存在、step_time_ms>=0、peak_mem_mb=0.0。",
        "逐项核对单位：elapsed 是秒/step，返回时间乘 1000；CUDA memory 是 bytes，返回显存除以 1024**2。",
        "若 counter 不对，检查两个循环边界；若时间差 iters 倍，检查是否求平均；若 CPU 调 CUDA API 报错，检查所有 CUDA 语句是否都在 is_available 分支内。"
      ]
    }),

    lesson({
      id: "training-delta-summary",
      title: "比较改前改后：baseline - tuned 的正数才表示改善",
      todo: "TODO 2：summarize_training_result 的差值方向与布尔字段",
      prerequisite: [
        "base_metrics 与 tuned_metrics 都必须含 step_time_ms 和 peak_mem_mb 两个键。",
        "本课统一定义 delta = baseline - tuned；tuned 数值更小时，差值为正。",
        "time_improved 与 memory_improved 使用未舍入差值严格判断 delta > 0；相等时为 False。",
        "返回的两个 delta round 到 2 位，两个 improved 字段必须是 Python bool。"
      ],
      intuition: "把 delta 读成“优化后省下多少”。原来 120 ms，现在 98 ms，省 22 ms；原来 8192 MB，现在 6144 MB，省 2048 MB。若结果为负，不是一定写错，而是 tuned 付出了更多成本。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>Notebook 的正向测试</h4>
            <div class="adv-contract">
              <span>时间</span><strong><code>120 - 98 = 22.0 ms</code></strong>
              <span>显存</span><strong><code>8192 - 6144 = 2048.0 MB</code></strong>
              <span>布尔</span><strong><code>True / True</code></strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>别忽略负数与零</h4>
            <div class="adv-contract">
              <span>base 100，tuned 125</span><strong>delta=-25，improved=False</strong>
              <span>base 100，tuned 100</span><strong>delta=0，improved=False</strong>
              <span>判断条件</span><strong>严格 <code>&gt; 0</code>，不是 <code>&gt;= 0</code></strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>返回键</th><th>类型</th><th>公式</th><th>正值/True 含义</th></tr></thead>
          <tbody>
            <tr><td><code>step_time_delta_ms</code></td><td>float</td><td>base time - tuned time</td><td>tuned 更快</td></tr>
            <tr><td><code>peak_mem_delta_mb</code></td><td>float</td><td>base memory - tuned memory</td><td>tuned 更省</td></tr>
            <tr><td><code>time_improved</code></td><td>bool</td><td>time_delta &gt; 0</td><td>时间改善</td></tr>
            <tr><td><code>memory_improved</code></td><td>bool</td><td>mem_delta &gt; 0</td><td>显存改善</td></tr>
          </tbody>
        </table>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用数据加载优化练习 baseline - tuned", [
          "before = {'load_ms': 18.4, 'ram_mb': 720.0}",
          "after = {'load_ms': 14.1, 'ram_mb': 760.0}",
          "",
          "load_delta = before['load_ms'] - after['load_ms']",
          "ram_delta = before['ram_mb'] - after['ram_mb']",
          "report = {",
          "    'load_delta_ms': round(load_delta, 2),",
          "    'ram_delta_mb': round(ram_delta, 2),",
          "    'load_improved': load_delta > 0,",
          "    'ram_improved': ram_delta > 0,",
          "}"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>before/after</code> -&gt; <code>base_metrics/tuned_metrics</code>。</li>
            <li><code>load_delta</code> -&gt; <code>time_delta</code>，都按旧值减新值。</li>
            <li><code>ram_delta</code> -&gt; <code>mem_delta</code>；例子故意为负，表示内存代价。</li>
            <li>四个 report 字段 -&gt; Notebook 返回字典的两个 float 与两个 bool。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "baseline 每步 100 ms，tuned 每步 125 ms。",
        question: "step_time_delta_ms 与 time_improved 应是什么？",
        options: ["-25.0 与 False", "25.0 与 True", "0.0 与 True"],
        answer: 0,
        revealNote: "100-125=-25，说明 tuned 反而慢了 25 ms。"
      },
      checkpoint: checkpoint(
        "base 与 tuned 的 peak_mem_mb 完全相等时，memory_improved 是什么？",
        ["False", "True", "无法返回"],
        0,
        "差值为 0，而条件是严格大于 0，所以结果为 False。"
      ),
      homework: [
        "完成 TODO 2，用 Notebook 数据核对 22.0、2048.0、True、True 四个断言。",
        "增加一组 tuned 更慢、更占显存的反例和一组完全相同的零差值案例，确认符号与严格布尔条件。",
        "若两个 delta 都是负号，检查是否写成 tuned-base；若相等时返回 True，检查是否误用 >=；若 KeyError，逐字检查两个输入键。"
      ]
    }),

    lesson({
      id: "training-controlled-analysis",
      title: "从两个 helper 走向瓶颈结论：一次只改一个变量",
      todo: "项目交付：measure -&gt; compare -&gt; profiling 归因与取舍报告",
      prerequisite: [
        "measure_train_step 只给出平均 step time 与整体峰值显存，不会自动区分数据、forward、backward 或 optimizer 瓶颈。",
        "baseline 与 tuned 必须固定模型、数据、batch、seq len、optimizer、精度和训练 step；一轮只改变一个目标变量。",
        "step time 改善、显存改善与 loss/收敛约束是不同维度，真实优化允许出现交换关系。",
        "完整交付还要结合 profiler 或阶段计时，说明瓶颈来源与下一轮优先级。"
      ],
      intuition: "两个 helper 负责“量”和“比”，但因果解释仍需要实验设计。开启 checkpointing 后显存下降、时间上升，可能是用重计算换存储；数据加载变快但 GPU 仍空闲，则要继续看拷贝和同步。性能分析不是追求两个 True，而是解释每个变化。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>固定任务</b>同模型、数据、batch、seq、optimizer</span>
          <span><b>建立 baseline</b>warmup 后重复测量</span>
          <span><b>只改一项</b>精度、checkpoint、loader 等</span>
          <span><b>比较 delta</b>时间与显存分别判断</span>
          <span><b>解释瓶颈</b>结合 loss 与 profiler</span>
        </div>

        <div class="adv-grid three">
          <section class="adv-panel neutral"><h4>数据瓶颈</h4><p>DataLoader、CPU 预处理或 H2D 拷贝让计算等待。</p></section>
          <section class="adv-panel blue"><h4>计算瓶颈</h4><p>forward/backward 热点算子或 optimizer step 占主导。</p></section>
          <section class="adv-panel warn"><h4>显存瓶颈</h4><p>激活、梯度、优化器状态或临时 buffer 接近上限。</p></section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>单一改动</th><th>可能收益</th><th>可能代价</th><th>必须一起观察</th></tr></thead>
          <tbody>
            <tr><td>activation checkpointing</td><td>峰值显存下降</td><td>反向重算，step 变慢</td><td>time、memory、loss</td></tr>
            <tr><td>混合精度</td><td>计算/显存改善</td><td>数值稳定性风险</td><td>吞吐、loss、溢出</td></tr>
            <tr><td>DataLoader 优化</td><td>减少等待</td><td>CPU/RAM 使用增加</td><td>数据阶段与 GPU 利用率</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">measure 返回的 peak_mem_mb 在 CPU 是 0.0，这不表示训练没有内存占用，只表示该 helper 只统计 CUDA allocated memory。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用配置断言保证只切换 checkpoint", [
          "baseline_cfg = {",
          "    'batch': 4, 'seq': 128, 'precision': 'fp32',",
          "    'checkpoint': False",
          "}",
          "tuned_cfg = {",
          "    'batch': 4, 'seq': 128, 'precision': 'fp32',",
          "    'checkpoint': True",
          "}",
          "",
          "for key in ['batch', 'seq', 'precision']:",
          "    assert baseline_cfg[key] == tuned_cfg[key]",
          "",
          "base_metrics = {'step_time_ms': 44.0, 'peak_mem_mb': 900.0}",
          "tuned_metrics = {'step_time_ms': 53.0, 'peak_mem_mb': 610.0}"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook 项目</h4>
          <ul>
            <li>配置相等断言 -&gt; Step 1 与 Step 3 的公平 baseline 要求。</li>
            <li><code>base_metrics/tuned_metrics</code> -&gt; TODO 2 的两个输入字典。</li>
            <li>checkpoint 是唯一变化 -&gt; 可以把显存下降与时间上升初步归因于重计算交换。</li>
            <li>下一步仍要结合 profiling 和 loss，确认热点与训练行为没有异常。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "开启 activation checkpointing 后，peak memory 降低，但 step time 上升。",
        question: "最合理的初步解释是什么？",
        options: ["少保存激活、反向时重算，形成时间与显存交换", "两个指标必须同时改善，否则代码一定错", "checkpoint 自动把 batch 设为 0"],
        answer: 0,
        revealNote: "Checkpointing 的典型机制就是用额外计算换更低激活存储。"
      },
      checkpoint: checkpoint(
        "为了判断混合精度是否真正改善训练，最可靠的第一步是什么？",
        ["固定其余配置，只切换精度，并重复测量 time、memory 与 loss", "同时修改 batch、optimizer 和数据集", "只看一次运行的最终耗时"],
        0,
        "单变量、同口径、重复测量才能建立较可信的因果关系。"
      ),
      homework: [
        "用 TODO 1 生成 baseline/tuned 指标，再用 TODO 2 汇总，写一条包含收益、代价、loss 约束与可能瓶颈的结论。",
        "保存两份配置并断言除目标变量外完全一致；至少重复测量多轮，避免把一次抖动当作优化。",
        "若时间波动大，增加 warmup/iters 并检查数据加载与同步；若 CUDA 显存为 0，确认运行设备；若指标改善但 loss 异常，优先回退并排查精度或训练配置变化。"
      ]
    })
  ]
};
