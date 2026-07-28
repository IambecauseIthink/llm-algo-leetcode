const {
  advancedStyles,
  checkpoint,
  code,
  lesson
} = require("../advanced_lesson_helpers");

module.exports = {
  "33": [
    lesson({
      id: "profiling-benchmark-baseline",
      title: "先固定测量尺：warmup 不计时，正式轮次才求平均",
      todo: "TODO 1：完成 benchmark_fn 的计时区间、秒到毫秒换算与平均值",
      prerequisite: [
        "函数也是一个值：参数 fn 保存的是“稍后要调用的函数”，写 fn() 才会真正执行它。",
        "range(n) 会产生 0 到 n-1 共 n 个整数，所以 for _ in range(iters) 会调用 fn 恰好 iters 次。",
        "time.perf_counter() 返回高精度时间点；结束时间减开始时间才是经过的秒数。",
        "warmup 只负责预热，不应落在 start 和 total 之间；正式计时结果要除以 iters，再乘 1000 转成毫秒。"
      ],
      intuition: "性能优化先要有一把稳定的尺。预热像运动前热身，它让初始化和缓存抖动先发生；真正比赛开始时才按下秒表。若把 warmup 也计入，或只跑一轮，baseline 和 tuned 的比较就可能是在比较偶然噪声。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>预热</b>调用 warmup 次，不纳入统计</span>
          <span><b>记开始点</b>start = perf_counter()</span>
          <span><b>正式执行</b>调用 iters 次</span>
          <span><b>算平均毫秒</b>(end - start) / iters * 1000</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>变量契约</h4>
            <div class="adv-contract">
              <span><code>fn</code></span><strong>一个无参数、可重复调用的函数</strong>
              <span><code>warmup</code></span><strong>计时前的预热次数</strong>
              <span><code>iters</code></span><strong>正式测量次数</strong>
              <span>返回值</span><strong>单次平均耗时，单位 ms，且应非负</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>计时边界</h4>
            <div class="adv-flow">
              <span>warmup loop<br>不计时</span>
              <span>start</span>
              <span>iters loop<br>正式计时</span>
              <strong>total / iters<br>再转 ms</strong>
            </div>
            <p>Notebook 测试把 warmup 设为 0、iters 设为 2，因此 counter 必须正好增加 2。</p>
          </section>
        </div>

        <div class="adv-callout">TODO 1 是 CPU-first 模板。真实 GPU kernel 常常异步执行，工程中还要在计时边界同步 CUDA；不要把这项额外要求擅自加进当前 Notebook。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用文件解析任务练习“预热后计时”", [
          "import time",
          "",
          "def measure(parser, warmup=1, repeats=4):",
          "    for _ in range(warmup):",
          "        parser()",
          "",
          "    begin = time.perf_counter()",
          "    for _ in range(repeats):",
          "        parser()",
          "    elapsed_seconds = time.perf_counter() - begin",
          "    return elapsed_seconds / repeats * 1000"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>parser</code> 对应 <code>fn</code>，都要用括号调用。</li>
            <li><code>begin</code> 对应 TODO 的 <code>start</code>；<code>elapsed_seconds</code> 对应 <code>total</code>。</li>
            <li><code>repeats</code> 对应 <code>iters</code>；先除次数，再乘 1000，得到 <code>avg_time_ms</code>。</li>
            <li>把三行迁回 TODO 时保留 Notebook 已有的两个循环，不要再额外调用一次 <code>fn()</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "warmup=3、iters=10，正式 10 次总共耗时 0.05 秒。",
        question: "benchmark_fn 应返回多少毫秒？",
        options: ["5 ms", "50 ms", "500 ms"],
        answer: 0,
        revealNote: "0.05 / 10 = 0.005 秒；乘 1000 后是 5 ms。warmup 不进入这段总耗时。"
      },
      checkpoint: checkpoint(
        "start 最合理的位置在哪里？",
        ["warmup 循环之后、正式 iters 循环之前", "warmup 循环之前", "正式 iters 循环之后"],
        0,
        "这样 total 只覆盖需要比较的正式轮次，预热开销不会污染平均耗时。"
      ),
      homework: [
        "完成 TODO 1，并口头解释 start、total、avg_time_ms 三个变量各自的单位。",
        "测试契约：warmup=0、iters=2 时 fn 必须调用 2 次，返回值必须大于或等于 0。",
        "边界诊断：若 counter 变成 3，检查是否在循环外又调用了 fn；若结果放大 1000 倍，检查秒与毫秒的换算顺序。",
        "举一反三：把独立例子的 repeats 改成 8，验证总耗时增长时平均值为什么不一定成倍增长。"
      ]
    }),

    lesson({
      id: "profiling-metric-direction",
      title: "再统一正负号：低指标用 baseline - tuned，高指标反过来",
      todo: "TODO 2：汇总 step time、peak memory、throughput 的差值与改善布尔值",
      prerequisite: [
        "Python 字典用方括号按键取值，例如 metrics['step_time_ms']。",
        "step time 和 peak memory 越低越好，所以 baseline 减 tuned 为正时表示改善。",
        "throughput 越高越好，所以必须用 tuned 减 baseline，正数才仍表示改善。",
        "round(value, 2) 只负责把结果保留两位小数；判断改善应基于原始差值变量。"
      ],
      intuition: "报告最容易埋下的错误不是算错数字，而是同一个正号表达了相反含义。解决办法是先统一约定：summary 里所有性能差值都让“正数 = tuned 更好”。然后根据指标是越低越好还是越高越好，决定减法方向。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid three">
          <section class="adv-panel blue">
            <h4>step time</h4>
            <div class="adv-contract">
              <span>偏好</span><strong>越低越好</strong>
              <span>公式</span><strong>base - tuned</strong>
              <span>Notebook</span><strong>120 - 96 = 24 ms</strong>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>peak memory</h4>
            <div class="adv-contract">
              <span>偏好</span><strong>越低越好</strong>
              <span>公式</span><strong>base - tuned</strong>
              <span>Notebook</span><strong>8192 - 7168 = 1024 MB</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>throughput</h4>
            <div class="adv-contract">
              <span>偏好</span><strong>越高越好</strong>
              <span>公式</span><strong>tuned - base</strong>
              <span>Notebook</span><strong>100 - 80 = 20</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>输出键</th><th>数值来源</th><th>布尔判断</th><th>测试期望</th></tr></thead>
          <tbody>
            <tr><td><code>step_time_delta_ms</code></td><td><code>time_delta</code></td><td><code>time_delta &gt; 0</code></td><td>24.0 / true</td></tr>
            <tr><td><code>peak_mem_delta_mb</code></td><td><code>memory_delta</code></td><td><code>memory_delta &gt; 0</code></td><td>1024.0 / true</td></tr>
            <tr><td><code>throughput_delta</code></td><td><code>throughput_delta</code></td><td><code>throughput_delta &gt; 0</code></td><td>20.0 / true</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">先在纸上给每个指标写“越低/越高”，再写减法。不要从三个测试答案倒推硬编码数字；函数必须适用于任意 baseline 和 tuned 字典。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用配送方案练习统一“正数代表改善”", [
          "old = {'minutes': 42.0, 'orders_per_hour': 18.0}",
          "new = {'minutes': 35.0, 'orders_per_hour': 24.0}",
          "",
          "time_gain = old['minutes'] - new['minutes']",
          "capacity_gain = new['orders_per_hour'] - old['orders_per_hour']",
          "",
          "result = {",
          "    'time_gain': round(time_gain, 2),",
          "    'capacity_gain': round(capacity_gain, 2),",
          "    'time_improved': time_gain > 0,",
          "    'capacity_improved': capacity_gain > 0,",
          "}"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>old/new</code> 对应 <code>base_metrics/tuned_metrics</code>。</li>
            <li><code>minutes</code> 对应 step time 或 memory，沿用 <code>base - tuned</code>。</li>
            <li><code>orders_per_hour</code> 对应 throughput，沿用 <code>tuned - base</code>。</li>
            <li>最后把三个差值写入 Notebook 已给出的 summary 键，不要更名，否则测试按键读取会失败。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "baseline throughput=100，tuned throughput=92。",
        question: "按 Notebook 的统一约定，throughput_delta 和 improved 应是什么？",
        options: ["-8，False", "8，True", "-8，True"],
        answer: 0,
        revealNote: "高指标用 tuned - baseline：92 - 100 = -8；负数表示吞吐退步。"
      },
      checkpoint: checkpoint(
        "为什么 memory_delta 使用 baseline - tuned？",
        ["显存越低越好，这样 tuned 更省显存时差值为正", "因为所有指标都必须固定用 baseline - tuned", "因为 round 只接受正数"],
        0,
        "减法方向由指标偏好决定；throughput 恰好需要相反方向。"
      ),
      homework: [
        "完成 TODO 2，并为三个差值各写一句“为什么这样减”。",
        "测试契约：24.0、1024.0、20.0 三个差值及三个改善布尔值都必须与测试一致。",
        "反例测试：自行构造 tuned 更慢、更占显存、吞吐更低的字典，确认三个差值均为负、三个布尔值均为 False。",
        "错误诊断：数值绝对值正确但布尔值相反，优先检查减法方向；KeyError 则检查字典键的拼写。"
      ]
    }),

    lesson({
      id: "profiling-report-loop",
      title: "最后让数字变成行动：列表生成表格，结论回扣瓶颈",
      todo: "TODO 3：生成指标行，并把 bottleneck 与 next_action 写入优化报告",
      prerequisite: [
        "f-string 用大括号把变量值嵌进字符串，例如 f'耗时：{value}'。",
        "条件表达式写成 A if condition else B，可在表格中选择“改善”或“未改善”。",
        "rows 是字符串列表；[header, sep] + rows + [conclusion] 会拼成一个更长的列表。",
        "'\\n'.join(lines) 会在每两行之间插入换行符，适合生成 Markdown 表格。"
      ],
      intuition: "profiling 的闭环不是停在“快了 24 ms”。一份可执行报告必须同时保留证据、瓶颈判断和下一步动作。表格回答发生了什么，结论回答为什么以及下一轮做什么。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>header + sep</b>建立 Markdown 表头</span>
          <span><b>rows</b>把三项差值逐行格式化</span>
          <span><b>conclusion</b>串联瓶颈与下一步</span>
          <span><b>join</b>输出一个多行字符串</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>报告必须出现的内容</h4>
            <div class="adv-checks">
              <span>step time<br>变化与判断</span>
              <span>peak memory<br>变化与判断</span>
              <span>throughput<br>变化与判断</span>
              <span>bottleneck<br>next_action</span>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>闭环不等于全都改善</h4>
            <p>某项布尔值为 False 时，表格应如实显示“未改善”。优化可以带来取舍，报告不能为了好看隐藏代价。</p>
            <div class="adv-contract">
              <span>瓶颈</span><strong>backward kernel 占比过高</strong>
              <span>下一步</span><strong>保留混合精度并检查 optimizer</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">测试只检查关键子串，但你的实现仍应完整生成三行指标。不要把 bottleneck 或 next_action 写死为测试样例，它们是函数参数。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用学习复盘练习生成 Markdown 报告", [
          "score = 8",
          "passed = score >= 6",
          "rows = [",
          "    f\"| 得分 | {score} | {'通过' if passed else '未通过'} |\",",
          "]",
          "next_step = '复习错题并再测一次'",
          "conclusion = f\"下一步：{next_step}。\"",
          "",
          "report = \"\\n\".join([",
          "    '| 项目 | 数值 | 判断 |',",
          "    '| --- | --- | --- |',",
          "] + rows + [conclusion])"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>score/passed</code> 对应 summary 中的一组差值和 improved 布尔值。</li>
            <li>单行 <code>rows</code> 扩成 step time、peak memory、throughput 三行。</li>
            <li><code>next_step</code> 对应 <code>next_action</code>；结论中还要加入 <code>bottleneck</code>。</li>
            <li>保留 Notebook 已给出的 <code>header</code> 与 <code>sep</code>，最终仍用换行符 join。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "time_improved=False，但 step_time_delta_ms=-3.5。",
        question: "step time 表格行的判断文字应该是什么？",
        options: ["未改善", "改善", "无法生成"],
        answer: 0,
        revealNote: "显示文字由布尔键决定；负差值也应该原样保留，帮助读者看到退步幅度。"
      },
      checkpoint: checkpoint(
        "为什么报告末尾还要写 bottleneck 和 next_action？",
        ["让指标变化能连接到原因判断和下一轮可执行动作", "为了让字符串更长", "因为 Markdown 表格不能单独显示数字"],
        0,
        "优化报告的价值在于支持决策，而不只是保存一组测量数字。"
      ),
      homework: [
        "完成 TODO 3：生成三行 rows 和一个 conclusion，并通过 join 返回完整报告。",
        "测试契约：报告中必须包含表头、传入的 bottleneck 文本和 next_action 文本。",
        "自行构造一项未改善的 summary，确认报告能同时展示改善项与未改善项。",
        "错误诊断：若表格挤在一行，检查是否使用 '\\n'.join；若出现 NameError，检查 rows 和 conclusion 是否在 return 前赋值。"
      ]
    })
  ],

  "34": [
    lesson({
      id: "parallel-benchmark-fairness",
      title: "公平比较从计时开始：同一 workload、同一单位、同一轮次",
      todo: "TODO 1：完成分布式并行 benchmark 的 warmup 与平均 latency 计时",
      prerequisite: [
        "并行策略比较必须固定模型、输入、global batch、硬件数量和后端，否则差异无法归因。",
        "warmup 调用不计时，正式 iters 调用才构成 total。",
        "time.perf_counter() 的差值单位是秒，报告中的 latency 使用毫秒。",
        "本 Notebook 是 CPU-first 最小模板，不会真的启动 ZeRO、Pipeline 或 Tensor Parallel 多卡进程。"
      ],
      intuition: "选择并行策略像比较三种运输方案：货物、路程和车辆数量都相同，时间才有可比性。代码里的 benchmark_fn 只是共同秒表；真正的公平性还来自 Step 1 固定 workload 的实验纪律。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>代码层计时契约</h4>
            <div class="adv-flow">
              <span>warmup 次<br>不统计</span>
              <span>perf_counter<br>开始</span>
              <span>iters 次<br>正式运行</span>
              <strong>平均 latency<br>ms</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>实验层公平契约</h4>
            <div class="adv-checks">
              <span>模型与输入一致</span>
              <span>global batch 一致</span>
              <span>GPU 数与拓扑一致</span>
              <span>运行后端一致</span>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>fn</code></span><strong>待测的一次完整工作负载</strong>
          <span><code>warmup=2</code></span><strong>默认先运行两次，不进入平均值</strong>
          <span><code>iters=5</code></span><strong>默认正式运行五次</strong>
          <span><code>avg_time_ms</code></span><strong>每次平均耗时，非负浮点数</strong>
        </div>

        <div class="adv-callout">真实 GPU 多卡 benchmark 还需要同步、通信 trace 和稳定性检查。当前 TODO 只要求实现 CPU 计时骨架，先通过它建立可复用的评测入口。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用排序算法练习统一计时入口", [
          "import time",
          "",
          "def average_ms(task, warmup=2, trials=5):",
          "    for _ in range(warmup):",
          "        task()",
          "",
          "    start = time.perf_counter()",
          "    for _ in range(trials):",
          "        task()",
          "    total_seconds = time.perf_counter() - start",
          "    return total_seconds / trials * 1000"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>task/trials</code> 对应 <code>fn/iters</code>。</li>
            <li><code>total_seconds</code> 对应 TODO 的 <code>total</code>，其单位仍是秒。</li>
            <li>返回表达式对应 <code>avg_time_ms</code>：除以正式轮次，再乘 1000。</li>
            <li>测试用 counter 验证调用次数，因此不能把 warmup 循环移动到正式计时循环内部。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "warmup=0、iters=2，fn 每调用一次就让 counter 加 1。",
        question: "函数返回前 counter 应该是多少？",
        options: ["2", "0", "4"],
        answer: 0,
        revealNote: "没有预热调用，正式循环执行两次，测试正是用它排查额外调用。"
      },
      checkpoint: checkpoint(
        "只把 TP 的 batch size 调大后，能否把它和原 baseline 直接比较来证明 TP 更快？",
        ["不能，workload 同时变化，收益无法只归因于 TP", "能，只要 throughput 更高", "能，因为 benchmark_fn 会自动修正 batch 差异"],
        0,
        "统一计时函数不会替你固定实验条件，公平性仍需由 benchmark 设计保证。"
      ),
      homework: [
        "完成 TODO 1，并确认结果单位是单次平均毫秒。",
        "测试契约：warmup=0、iters=2 时 counter 为 2，avg 大于或等于 0。",
        "写下比较 ZeRO、Pipeline、TP 时必须固定的四个 workload 条件。",
        "错误诊断：调用次数错误先查循环边界；数值异常偏大先查是否忘记除以 iters。"
      ]
    }),

    lesson({
      id: "parallel-four-metric-ledger",
      title: "四本账一起看：显存、吞吐、延迟、通信都统一为正收益",
      todo: "TODO 2：计算四项 baseline / parallel 差值及改善判断",
      prerequisite: [
        "peak_mem_mb、latency_ms、communication_ms 越低越好，使用 baseline - parallel。",
        "throughput 越高越好，使用 parallel - baseline。",
        "通信时间下降为正收益；若并行方案引入更多通信，communication_delta 会是负数。",
        "summary 的键名是后续报告和测试的接口，必须逐字保留。"
      ],
      intuition: "并行方案经常用一项资源换另一项资源：更省显存，却增加通信；吞吐变高，却让单请求延迟变差。四项指标放在同一张账本里，才不会把局部收益误判成整体胜利。",
      exampleHtml: `<div class="adv-course">
        <table class="adv-shapes">
          <thead><tr><th>指标</th><th>偏好</th><th>差值公式</th><th>测试数据</th></tr></thead>
          <tbody>
            <tr><td>peak memory</td><td>低</td><td><code>base - parallel</code></td><td>12000 - 9000 = 3000 MB</td></tr>
            <tr><td>throughput</td><td>高</td><td><code>parallel - base</code></td><td>100 - 80 = 20</td></tr>
            <tr><td>latency</td><td>低</td><td><code>base - parallel</code></td><td>120 - 96 = 24 ms</td></tr>
            <tr><td>communication</td><td>低</td><td><code>base - parallel</code></td><td>30 - 24 = 6 ms</td></tr>
          </tbody>
        </table>

        <div class="adv-grid three">
          <section class="adv-panel blue">
            <h4>ZeRO 常看</h4>
            <p>训练状态切分后的单卡显存，以及 Reduce-Scatter / All-Gather 代价。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>Pipeline 常看</h4>
            <p>按层切分后的显存、吞吐、stage 等待和 bubble ratio。</p>
          </section>
          <section class="adv-panel good">
            <h4>TP 常看</h4>
            <p>单层矩阵分摊后的显存与计算，以及 All-Reduce / All-Gather 开销。</p>
          </section>
        </div>

        <div class="adv-callout">四个 improved 都由对应 delta 是否大于 0 得出。不要让“communication overhead 本来是代价”扰乱公式：这里比较的是代价减少了多少。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用数据库方案练习多指标取舍", [
          "base = {'memory': 64, 'queries': 120, 'latency': 18}",
          "candidate = {'memory': 48, 'queries': 150, 'latency': 21}",
          "",
          "memory_gain = base['memory'] - candidate['memory']",
          "query_gain = candidate['queries'] - base['queries']",
          "latency_gain = base['latency'] - candidate['latency']",
          "",
          "tradeoff = {",
          "    'memory_improved': memory_gain > 0,",
          "    'queries_improved': query_gain > 0,",
          "    'latency_improved': latency_gain > 0,",
          "}"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>memory</code> 对应 <code>peak_mem_mb</code>，低者更好。</li>
            <li><code>queries</code> 对应 <code>throughput</code>，高者更好。</li>
            <li><code>latency</code> 对应 <code>latency_ms</code>；再用同样模式加入 <code>communication_ms</code>。</li>
            <li>将四个原始差值写入已给出的 summary，数值统一 <code>round(..., 2)</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "baseline communication=20 ms，parallel communication=32 ms。",
        question: "communication_delta 与 communication_improved 是什么？",
        options: ["-12，False", "12，True", "-12，True"],
        answer: 0,
        revealNote: "低指标用 baseline - parallel：20 - 32 = -12，说明并行方案增加了通信等待。"
      },
      checkpoint: checkpoint(
        "某方案显存下降 3000 MB，但 communication_delta=-15 ms。正确解读是什么？",
        ["它省显存，但通信时间增加了 15 ms，需要权衡", "四项指标都改善", "通信指标应取绝对值后判定改善"],
        0,
        "负差值是有价值的代价证据，不能取绝对值把退步伪装成收益。"
      ),
      homework: [
        "完成 TODO 2，确保四个差值的正号都代表 parallel 更好。",
        "测试契约：依次得到 3000.0、20.0、24.0、6.0，四个 improved 均为 True。",
        "构造“显存改善但通信恶化”的样例，确认 summary 能同时出现 True 和 False。",
        "错误诊断：若只有 throughput 符号错，检查它是否误用了 base - parallel；若 KeyError，核对 communication_ms 拼写。"
      ]
    }),

    lesson({
      id: "parallel-selection-report",
      title: "报告不是排行榜：策略名、四项取舍与推荐条件缺一不可",
      todo: "TODO 3：生成并行策略 Markdown 报告与 recommendation 结论",
      prerequisite: [
        "format_parallel_report 接收 strategy_name、summary、recommendation，不能依赖函数外的固定变量。",
        "rows 要覆盖 memory、throughput、latency、communication 四项。",
        "每一行既展示 delta，也根据对应布尔键展示改善或未改善。",
        "返回字符串的第一行先写策略名，再接 Markdown 表格和推荐结论。"
      ],
      intuition: "并行选型没有脱离 workload 的冠军。报告要把策略是谁、改善了什么、牺牲了什么、为什么推荐写在一起。这样同事才能知道这条建议只适用于当前模型规模和通信条件。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>策略名</b>ZeRO / Pipeline / TP</span>
          <span><b>四项 rows</b>收益和代价同屏</span>
          <span><b>recommendation</b>说明适用条件</span>
          <span><b>下一轮</b>指出需继续观察的开销</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>Notebook 测试要看到</h4>
            <div class="adv-checks">
              <span>Tensor Parallelism</span>
              <span>Markdown 表头</span>
              <span>四项指标变化</span>
              <span>All-Reduce 建议</span>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>推荐要有边界</h4>
            <p>“优先保留 TP”不是永恒结论；样例理由是当前单层矩阵较大，同时还要继续观察 All-Reduce。换 workload 后应重新测量。</p>
          </section>
        </div>

        <div class="adv-callout">测试会搜索字符串，但不要只拼测试关键字。完整四行让报告在指标不全都改善时仍可用于真实决策。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用存储方案练习参数化报告", [
          "plan_name = '分片存储'",
          "saved_gb = 12",
          "saved = saved_gb > 0",
          "rows = [",
          "    f\"| 容量 | {saved_gb} GB | {'改善' if saved else '未改善'} |\",",
          "]",
          "advice = '适合容量瓶颈，继续观察网络读取'",
          "",
          "report = \"\\n\".join(",
          "    [f\"方案：{plan_name}\", '| 指标 | 变化 | 判断 |', '| --- | --- | --- |']",
          "    + rows",
          "    + [f\"推荐结论：{advice}。\"]",
          ")"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>plan_name</code> 对应 <code>strategy_name</code>，必须来自函数参数。</li>
            <li>容量单行扩展成 memory、throughput、latency、communication 四行。</li>
            <li><code>advice</code> 对应 <code>recommendation</code>。</li>
            <li>第一行前缀改成 Notebook 规定的“策略：”，结论前缀改成“推荐结论：”。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "summary 中 memory_improved=True，latency_improved=False。",
        question: "报告应如何呈现？",
        options: ["memory 写改善，latency 写未改善", "只保留改善的 memory 行", "两行都写改善"],
        answer: 0,
        revealNote: "报告要保留完整取舍，不能隐藏不利指标。"
      },
      checkpoint: checkpoint(
        "为什么 strategy_name 必须出现在报告里？",
        ["否则同一份指标无法知道对应 ZeRO、Pipeline 还是 TP", "因为 join 只能处理带策略名的列表", "因为测试不允许中文结论"],
        0,
        "指标离开策略和 workload 上下文就无法支持选型。"
      ),
      homework: [
        "完成 TODO 3，依次生成四个 rows、conclusion，并返回带策略名的多行字符串。",
        "测试契约：结果包含 Tensor Parallelism、指定表头和 recommendation 中的 All-Reduce。",
        "分别传入 ZeRO、Pipeline、TP 三个名字，确认函数不会把策略写死。",
        "错误诊断：缺少一行先查 rows 列表；字符串没有换行先查 join；推荐没变化则检查是否使用了 recommendation 参数。"
      ]
    })
  ],

  "35": [
    lesson({
      id: "quantized-benchmark-latency",
      title: "量化先测端到端延迟：低 bit 不自动等于更快",
      todo: "TODO 1：完成量化推理 benchmark 的 warmup、计时与平均 latency",
      prerequisite: [
        "baseline 与 quantized 必须使用相同输入、batch、seq len、解码策略、硬件和后端。",
        "warmup 在秒表外，iters 循环在秒表内。",
        "perf_counter 的差值是总秒数；除以 iters 后乘 1000 得单次平均毫秒。",
        "量化只改变数值表示或存储，并不保证 kernel 生效，也不保证瓶颈一定在权重读取。"
      ],
      intuition: "模型变小只是部署候选的起点。反量化、kernel 支持和数据搬运都可能抵消收益，所以必须让 baseline 与量化方案跑过同一把秒表，再讨论 latency 是否真的下降。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>函数契约</h4>
            <div class="adv-contract">
              <span>输入</span><strong>fn、warmup=2、iters=5</strong>
              <span>预热</span><strong>执行但不计入 latency</strong>
              <span>正式测量</span><strong>只包住 iters 次 fn()</strong>
              <span>返回</span><strong>avg_latency_ms，非负</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>同口径比较</h4>
            <div class="adv-checks">
              <span>相同 prompt</span>
              <span>相同 batch / seq len</span>
              <span>相同解码策略</span>
              <span>相同后端与硬件</span>
            </div>
          </section>
        </div>

        <div class="adv-flow">
          <span>baseline<br>同一 benchmark</span>
          <span>quantized<br>同一 benchmark</span>
          <strong>比较 latency<br>而非只看 bit 数</strong>
        </div>

        <div class="adv-callout">当前 Notebook 不执行真实模型量化，也不要求 CUDA 同步。TODO 1 只实现可复用的 CPU-first 计时模板。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用图片压缩器练习平均延迟", [
          "import time",
          "",
          "def latency_ms(compress_one, warmup=2, runs=5):",
          "    for _ in range(warmup):",
          "        compress_one()",
          "",
          "    start = time.perf_counter()",
          "    for _ in range(runs):",
          "        compress_one()",
          "    total = time.perf_counter() - start",
          "    return total / runs * 1000"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>compress_one</code> 对应 <code>fn</code>，<code>runs</code> 对应 <code>iters</code>。</li>
            <li>start、total 的写法可直接迁移；返回变量名换成 <code>avg_latency_ms</code>。</li>
            <li>Notebook 已经写好两个循环，你只补计时与公式，不要重写函数接口。</li>
            <li>测试的 counter 用于确认 warmup=0 时只调用两次正式轮次。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "正式 5 次推理总共耗时 0.36 秒。",
        question: "平均 latency 是多少？",
        options: ["72 ms", "360 ms", "1800 ms"],
        answer: 0,
        revealNote: "0.36 / 5 * 1000 = 72 ms。"
      },
      checkpoint: checkpoint(
        "模型权重从 16 bit 变成 8 bit 后，能否不测就断言 latency 减半？",
        ["不能，kernel、反量化和真实瓶颈都会影响端到端 latency", "能，bit 数和 latency 永远成正比", "能，只要 VRAM 下降"],
        0,
        "量化收益必须通过同口径 benchmark 验证。"
      ),
      homework: [
        "完成 TODO 1，并标注 total 是秒、avg_latency_ms 是毫秒。",
        "测试契约：warmup=0、iters=2 时 counter 为 2，返回平均值非负。",
        "列出真实量化推理比较中必须固定的四项条件。",
        "错误诊断：结果过小检查是否忘记乘 1000；调用次数不对检查是否在循环外额外执行 fn。"
      ]
    }),

    lesson({
      id: "quantized-benefit-error-budget",
      title: "性能收益和误差约束分开算：error 不是越大越好的收益",
      todo: "TODO 2：汇总 latency、throughput、VRAM 与 error_delta，并判断误差预算",
      prerequisite: [
        "latency 与 VRAM 越低越好，差值使用 baseline - quantized。",
        "throughput 越高越好，差值使用 quantized - baseline。",
        "error_delta 使用 quantized error - baseline error，它描述新增误差，不是性能改善。",
        "error_within_budget 要比较 error_delta 与 quant_metrics['error_budget']。"
      ],
      intuition: "量化部署同时有两条线：性能线希望更快、更省、更高吞吐；质量线希望新增误差不超过预算。把 error_delta 也套进“正数越好”会得到危险结论，所以它必须作为约束单独判断。",
      exampleHtml: `<div class="adv-course">
        <table class="adv-shapes">
          <thead><tr><th>指标</th><th>公式</th><th>测试结果</th><th>如何判断</th></tr></thead>
          <tbody>
            <tr><td>latency</td><td><code>base - quant</code></td><td>28 ms</td><td>大于 0 为改善</td></tr>
            <tr><td>throughput</td><td><code>quant - base</code></td><td>40</td><td>大于 0 为改善</td></tr>
            <tr><td>VRAM</td><td><code>base - quant</code></td><td>5000 MB</td><td>大于 0 为改善</td></tr>
            <tr><td>error</td><td><code>quant - base</code></td><td>0.012</td><td>不超过 0.02 预算</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>性能收益</h4>
            <div class="adv-checks">
              <span>latency 降低</span>
              <span>throughput 提高</span>
              <span>VRAM 降低</span>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>质量闸门</h4>
            <div class="adv-contract">
              <span><code>error_delta</code></span><strong>量化方案新增的误差</strong>
              <span><code>error_budget</code></span><strong>部署允许的最大新增误差</strong>
              <span>通过条件</span><strong>error_delta 不超过预算</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">即使 latency、throughput、VRAM 全部改善，只要 error_within_budget 为 False，就不能直接得出“可部署”。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用图片编码方案练习收益与质量预算", [
          "original = {'size_mb': 20.0, 'quality_loss': 0.0}",
          "compressed = {",
          "    'size_mb': 8.0,",
          "    'quality_loss': 0.03,",
          "    'loss_budget': 0.02,",
          "}",
          "",
          "size_gain = original['size_mb'] - compressed['size_mb']",
          "loss_delta = compressed['quality_loss'] - original['quality_loss']",
          "within_budget = loss_delta <= compressed['loss_budget']"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>size_gain</code> 对应 VRAM 节省，沿用低指标的减法方向。</li>
            <li><code>loss_delta</code> 对应 <code>error_delta</code>，表示新增质量代价。</li>
            <li><code>loss_budget</code> 对应 <code>error_budget</code>，使用小于或等于比较。</li>
            <li>再分别按高低偏好补上 latency_delta 与 throughput_delta，并写入 summary 的固定键名。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "量化新增误差 0.025，error_budget=0.02，三项性能都改善。",
        question: "error_within_budget 应是什么，能否仅凭性能进入部署？",
        options: ["False，不能", "True，可以", "False，但仍可忽略"],
        answer: 0,
        revealNote: "0.025 超过 0.02；质量闸门失败，性能收益不能覆盖明确的误差约束。"
      },
      checkpoint: checkpoint(
        "为什么 error_delta 不设置 error_improved = error_delta > 0？",
        ["新增误差通常是代价，应与预算比较，而不是把变大解释为改善", "因为浮点数不能比较大小", "因为 baseline 永远没有 error 键"],
        0,
        "性能差值和质量约束的语义不同，必须分开处理。"
      ),
      homework: [
        "完成 TODO 2，特别核对 error_delta 的方向和小于等于预算的条件。",
        "测试契约：28.0、40.0、5000.0、0.012，三项 improved 与 error_within_budget 均为 True。",
        "把 quantized error 改为 0.03，确认只有 error_within_budget 变成 False。",
        "错误诊断：误差预算判断相反时检查比较符号；数值出现 0.02 而非 0.012 时检查是否误用了 budget 作为 delta。"
      ]
    }),

    lesson({
      id: "quantized-deployment-report",
      title: "部署结论要过双闸门：性能可观，误差也要合格",
      todo: "TODO 3：生成量化方案、四项指标和部署建议组成的 Markdown 报告",
      prerequisite: [
        "quant_name 是 W8A16、INT8、4-bit 等方案名，必须来自函数参数。",
        "rows 需要 latency、throughput、VRAM、error 四行。",
        "前三行读取 improved 布尔键；error 行读取 error_within_budget，显示满足预算或超出预算。",
        "recommendation 是最终部署动作，例如扩大回归、重新校准或暂缓部署。"
      ],
      intuition: "部署报告不是压缩率海报。它要让读者一眼看出方案、性能收益、质量闸门和下一步。只有性能与误差同时达标，才适合进入更大样本回归或线上验证。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>方案名</b>量化对象与 bit 配置</span>
          <span><b>性能三行</b>latency / throughput / VRAM</span>
          <span><b>质量一行</b>error 是否满足预算</span>
          <span><b>部署建议</b>扩大测试、校准或停止</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>测试样例的报告信号</h4>
            <div class="adv-checks">
              <span>W8A16</span>
              <span>表格 header</span>
              <span>error 满足预算</span>
              <span>更大样本回归测试</span>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>不可部署也要如实输出</h4>
            <p>若 error_within_budget 为 False，error 行必须写“超出预算”。recommendation 应指向重新校准、调整粒度或暂缓，而不是隐藏失败。</p>
          </section>
        </div>

        <div class="adv-callout">测试只检查几个关键片段，但四项 rows 是函数契约的一部分。不要省略 error 行，也不要把 W8A16 写死。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用音频压缩方案练习双闸门报告", [
          "codec = 'Codec-X'",
          "speed_gain = 15",
          "quality_ok = False",
          "rows = [",
          "    f\"| speed | {speed_gain} | {'改善' if speed_gain > 0 else '未改善'} |\",",
          "    f\"| quality | 0.08 | {'满足预算' if quality_ok else '超出预算'} |\",",
          "]",
          "advice = '重新调整压缩等级后再测试'",
          "report = \"\\n\".join(",
          "    [f\"方案：{codec}\", '| 指标 | 变化 | 判断 |', '| --- | --- | --- |']",
          "    + rows + [f\"部署建议：{advice}。\"]",
          ")"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>codec</code> 对应 <code>quant_name</code>。</li>
            <li>speed 行扩展为 latency、throughput、VRAM 三个性能行。</li>
            <li><code>quality_ok</code> 对应 <code>summary['error_within_budget']</code>。</li>
            <li><code>advice</code> 对应 <code>recommendation</code>，返回前缀按 Notebook 改为“量化方案”和“部署建议”。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "error_within_budget=False，recommendation 是“重新校准后再测”。",
        question: "error 行与结论应怎样写？",
        options: ["超出预算；重新校准后再测", "改善；直接上线", "删除 error 行；只保留性能"],
        answer: 0,
        revealNote: "报告必须保留质量失败，并让下一步动作回应这个阻塞原因。"
      },
      checkpoint: checkpoint(
        "量化报告为什么必须包含 quant_name？",
        ["不同量化方案的 bit 数、粒度与后端条件不同，指标必须可追溯", "因为字典不能保存方案名", "因为 error_budget 会自动从名字推导"],
        0,
        "没有方案上下文的数字无法支持部署复现和选型。"
      ),
      homework: [
        "完成 TODO 3，生成四项 rows、部署 conclusion，并拼入 quant_name。",
        "测试契约：报告包含 W8A16、Markdown 表头和“更大样本回归测试”。",
        "传入一个误差超预算的 summary，确认 error 行显示“超出预算”。",
        "错误诊断：报告出现 undefined 时核对 summary 键名；方案名不变化时检查是否写死；表格单行时检查换行 join。"
      ]
    })
  ],

  "36": [
    lesson({
      id: "prefix-normalize-and-chunk",
      title: "先把输入变整齐：统一成 list，再按步长切成 tuple 块",
      todo: "TODO 1-2：实现 _normalize 与 _chunk_tokens",
      prerequisite: [
        "Sequence[int] 表示可按顺序读取的整数序列，传入值可以是 list 或 tuple。",
        "list(tokens) 会创建普通列表，让后续切片和比较使用同一种表示。",
        "range(0, len(tokens), block_size) 从 0 开始，每次跨 block_size 个位置。",
        "tokens[i:i+block_size] 是左闭右开的切片；尾部不足一个块时也会保留下来，再用 tuple(...) 固定块表示。"
      ],
      intuition: "缓存管理先要消除表示差异：同样的 token，不能因为一个是 list、一个是 tuple 就让代码到处加特殊分支。统一输入后，再像把长句按固定字数装盒，最后一盒不满也照样保留。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>接收 Sequence</b>list 或 tuple 都可以</span>
          <span><b>normalize</b>统一得到 list[int]</span>
          <span><b>按步长切片</b>0, block_size, 2*block_size...</span>
          <span><b>块转 tuple</b>得到稳定的缓存块列表</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>block_size=2 的 shape / 状态流</h4>
            <div class="adv-flow">
              <span>输入<br>[1,2,3,4,5]</span>
              <span>切片 0:2<br>[1,2]</span>
              <span>切片 2:4<br>[3,4]</span>
              <strong>尾块 4:6<br>(5,)</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>返回类型契约</h4>
            <div class="adv-contract">
              <span><code>_normalize</code></span><strong>List[int]</strong>
              <span><code>_chunk_tokens</code></span><strong>List[Tuple[int, ...]]</strong>
              <span>空输入</span><strong>range 不进入，返回空列表</strong>
              <span>尾块</span><strong>允许小于 block_size</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">单元素 tuple 必须写成 <code>(5,)</code>，逗号决定它是 tuple。测试期望最后一块正是 <code>(5,)</code>，不能补零凑满。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用订单编号练习统一表示与分组", [
          "def group_orders(order_ids, group_size):",
          "    normalized = list(order_ids)",
          "    groups = [",
          "        tuple(normalized[i : i + group_size])",
          "        for i in range(0, len(normalized), group_size)",
          "    ]",
          "    return groups",
          "",
          "print(group_orders((7, 8, 9, 10, 11), 2))",
          "# [(7, 8), (9, 10), (11,)]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>order_ids</code> 对应 <code>tokens</code>，<code>group_size</code> 对应 <code>self.block_size</code>。</li>
            <li>TODO 1 只取 <code>list(tokens)</code> 并返回，不负责切块。</li>
            <li>TODO 2 先调用 <code>self._normalize(tokens)</code>，再把列表推导式赋给 <code>chunks</code>。</li>
            <li>块的 tuple 表示与 Notebook 的 <code>chunked_prefixes</code>、prefill plan 契约一致。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "tokens=[1,2,3,4,5]，block_size=2。",
        question: "_chunk_tokens 应返回什么？",
        options: ["[(1,2),(3,4),(5,)]", "[(1,2),(3,4)]", "[[1,2],[3,4],[5,0]]"],
        answer: 0,
        revealNote: "切片不会丢掉尾部，也不会自动补零；每个块按契约转成 tuple。"
      },
      checkpoint: checkpoint(
        "为什么 _chunk_tokens 要先调用 _normalize？",
        ["让 list/tuple 等输入沿同一条切片逻辑处理", "因为 tuple 不能保存整数", "为了改变 token 的数值"],
        0,
        "统一表示降低后续比较与切片的分支复杂度，不改变 token 内容。"
      ),
      homework: [
        "完成 TODO 1-2，并分别用 list、tuple 输入验证返回结果相同。",
        "测试契约：block_size=2 时 [1,2,3] 必须切成 [(1,2),(3,)]。",
        "边界练习：传入空列表和恰好 4 个 token，预测并验证结果。",
        "错误诊断：尾块丢失时检查 range 的终点；返回嵌套 list 时检查是否漏了 tuple(...)。"
      ]
    }),

    lesson({
      id: "prefix-register-and-longest-match",
      title: "缓存只登记一次，命中只认开头，并且选择最长前缀",
      todo: "TODO 3-4：保存前缀块并实现最长前缀匹配",
      prerequisite: [
        "tuple(list_value) 把完整前缀变成可稳定比较的 tuple。",
        "if prefix not in cached_prefixes 用于去重；只有新前缀才同时追加完整表示和分块表示。",
        "prompt[:n] 只取 prompt 开头 n 个 token，适合判断“是不是以前缀开头”。",
        "best_len = max(best_len, candidate_len) 会在多个命中候选中保留最长长度。"
      ],
      intuition: "Prefix Cache 不是查找任意相同片段。只有从新 prompt 第一个 token 开始连续相同，已有 KV 状态才处在正确上下文上；多个缓存都命中时，复用最长的那个能少算最多后缀。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>登记时维护两份同步视图</h4>
            <div class="adv-contract">
              <span><code>cached_prefixes</code></span><strong>完整 tuple 前缀，用于匹配</strong>
              <span><code>chunked_prefixes</code></span><strong>同一前缀的块列表，用于管理</strong>
              <span>去重条件</span><strong>完整前缀尚未出现</strong>
              <span>顺序约束</span><strong>两个列表相同下标描述同一前缀</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>最长命中状态流</h4>
            <div class="adv-flow">
              <span>prompt<br>[1,2,3,9]</span>
              <span>缓存 [1,2]<br>命中 2</span>
              <span>缓存 [1,2,3]<br>命中 3</span>
              <strong>best_len<br>3</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>长度过滤</code><span>缓存比 prompt 长时不可能完整命中，直接 continue</span></div>
          <div><b>2</b><code>开头切片比较</code><span>prompt 前 n 个 token 与完整 cached_prefix 比较</span></div>
          <div><b>3</b><code>更新 best_len</code><span>只有命中时才保留更长值</span></div>
        </div>

        <div class="adv-callout">[8,1,2,3] 中间包含缓存 [1,2,3]，但命中长度仍应为 0。前缀缓存必须从位置 0 开始。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用路径路由练习最长开头匹配", [
          "known_routes = [('api',), ('api', 'v2'), ('docs',)]",
          "request = ['api', 'v2', 'users']",
          "best_len = 0",
          "",
          "for route in known_routes:",
          "    if len(route) > len(request):",
          "        continue",
          "    is_match = request[: len(route)] == list(route)",
          "    if is_match:",
          "        best_len = max(best_len, len(route))",
          "",
          "print(best_len)  # 2"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>known_routes/request</code> 对应 <code>self.cached_prefixes/prompt</code>。</li>
            <li>TODO 3 先调用 <code>self._chunk_tokens(prefix)</code>，保证两份缓存视图同步追加。</li>
            <li>TODO 4 的 <code>is_match</code> 可直接沿用“开头切片等于候选列表”的写法。</li>
            <li>循环结束返回长度而非前缀本身，后续 split_prompt 会用这个下标切片。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "缓存有 [1,2] 与 [1,2,3]，prompt 是 [1,2,3,4]。",
        question: "match_prefix 应返回多少？",
        options: ["3", "2", "4"],
        answer: 0,
        revealNote: "两个候选都从开头命中，best_len 应保留更长的 3。"
      },
      checkpoint: checkpoint(
        "prompt=[1,2,0]，缓存=[1,2,3] 时为什么先跳过？",
        ["缓存长度大于 prompt，不可能被 prompt 完整包含为前缀", "因为 token 0 不能匹配", "因为缓存只能有偶数长度"],
        0,
        "长度过滤既避免无意义比较，也明确要求缓存前缀必须完整命中。"
      ),
      homework: [
        "完成 TODO 3-4，重复 add 同一前缀后确认两个缓存列表都没有新增项。",
        "测试契约：两条长度为 3 的缓存分别命中对应 prompt；[1,2,0] 返回 0。",
        "额外加入短前缀 [1,2]，确认 [1,2,3,9] 仍选择长度 3。",
        "错误诊断：中间子串也命中时检查是否比较了整个 prompt；总返回最后候选长度时检查是否使用 max。"
      ]
    }),

    lesson({
      id: "prefix-split-and-prefill-plan",
      title: "把命中长度变成工作量：前半复用，后半重算，计划复用同一切块函数",
      todo: "TODO 5-6：拆分 reusable_prefix / suffix，并生成 chunked prefill plan",
      prerequisite: [
        "若 hit_len=3，prompt[:3] 取索引 0、1、2，prompt[3:] 从索引 3 一直取到末尾。",
        "match_prefix 返回 0 时，前缀切片是空列表，suffix 是完整 prompt。",
        "chunked_prefill_plan 应调用 _chunk_tokens，而不是复制一份新的切块算法。",
        "split_prompt 返回三元组 (reusable_prefix, suffix, hit_len)，顺序必须与类型标注和测试一致。"
      ],
      intuition: "命中长度本身只是数字，真正的系统收益来自它把 prompt 划成了两段：已缓存部分不再 prefill，后缀继续计算。执行计划再把需要处理的 token 按统一 block_size 切开，避免缓存组织和调度组织各说一套规则。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>normalize</b>prompt 统一为 list</span>
          <span><b>match</b>得到 hit_len=3</span>
          <span><b>split</b>[1,2,3] + [9]</span>
          <span><b>plan</b>按 block_size 复用切块逻辑</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>命中时</h4>
            <div class="adv-flow">
              <span>prompt<br>[1,2,3,9]</span>
              <strong>reusable<br>[1,2,3]</strong>
              <span>suffix<br>[9]</span>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>未命中时</h4>
            <div class="adv-flow">
              <span>prompt<br>[1,2,0]</span>
              <span>reusable<br>[]</span>
              <strong>suffix<br>[1,2,0]</strong>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span>测试 split</span><strong>[1,2,3,9] -&gt; [1,2,3]、[9]、3</strong>
          <span>测试 plan</span><strong>[1,2,3,4,5] -&gt; [(1,2),(3,4),(5,)]</strong>
          <span>一致性原则</span><strong>缓存分块与 prefill 计划都走 _chunk_tokens</strong>
        </div>

        <div class="adv-callout">Notebook 的 chunked_prefill_plan 对传入的 prompt_tokens 整体切块。不要擅自让它先调用 split_prompt 只切 suffix，这会改变当前函数与测试契约。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用已下载章节练习切分复用与待处理部分", [
          "chapters = ['A', 'B', 'C', 'D', 'E']",
          "downloaded_count = 3",
          "",
          "reusable = chapters[:downloaded_count]",
          "remaining = chapters[downloaded_count:]",
          "",
          "batch_size = 2",
          "plan = [",
          "    tuple(chapters[i : i + batch_size])",
          "    for i in range(0, len(chapters), batch_size)",
          "]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>chapters</code> 对应 normalize 后的 <code>prompt</code>。</li>
            <li><code>downloaded_count</code> 对应 <code>hit_len</code>。</li>
            <li><code>reusable/remaining</code> 对应 TODO 5 的 <code>reusable_prefix/suffix</code>。</li>
            <li>TODO 6 不重写列表推导式，直接把 <code>self._chunk_tokens(prompt_tokens)</code> 赋给 <code>plan</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prompt=[1,2,3,9]，最长命中 hit_len=3。",
        question: "split_prompt 的三个返回值是什么？",
        options: ["[1,2,3]、[9]、3", "[1,2,3,9]、[]、4", "[1,2]、[3,9]、2"],
        answer: 0,
        revealNote: "切片边界正好是 hit_len：边界之前复用，边界及之后继续 prefill。"
      },
      checkpoint: checkpoint(
        "为什么 chunked_prefill_plan 要复用 _chunk_tokens？",
        ["保证缓存块和执行计划使用同一 block_size 与尾块规则", "因为类方法之间不能重复变量名", "为了让 plan 总是只包含 suffix"],
        0,
        "单一切块逻辑避免同一个 manager 出现两套不一致的块边界。"
      ),
      homework: [
        "完成 TODO 5-6，并运行完整 test_prefix_cache_manager。",
        "测试契约：split 返回 [1,2,3]、[9]、3；plan 返回三个 tuple 块，尾块为 (5,)。",
        "手工测试无缓存命中，确认 reusable_prefix 为空、suffix 等于完整 prompt。",
        "错误诊断：prefix/suffix 重叠或丢 token 时检查切片边界；plan 类型错误时确认调用的是 _chunk_tokens。"
      ]
    })
  ],

  "37": [
    lesson({
      id: "multi-token-propose",
      title: "提议阶段只做两件事：统一为 list，再截取本轮上限",
      todo: "TODO 1：实现 propose，返回最多 max_proposal_len 个候选 token",
      prerequisite: [
        "Sequence[int] 可以是 list 或 tuple；list(draft_tokens) 把它统一成普通列表。",
        "values[:n] 取前 n 个元素；当原列表不足 n 个时，Python 会返回全部元素而不报错。",
        "max_proposal_len 在 __init__ 中已验证为正数。",
        "propose 只决定本轮候选序列，不读取概率，也不做接受判断。"
      ],
      intuition: "多 token 解码先控制尝试规模。提议太长可能增加验证和回退成本，因此每轮只拿草稿序列的一个前缀。这里先把“提出哪些 token”和“是否接受”拆开，后面的控制流会更容易验证。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>输入输出契约</h4>
            <div class="adv-contract">
              <span>输入</span><strong>draft_tokens: Sequence[int]</strong>
              <span>配置</span><strong>self.max_proposal_len</strong>
              <span>输出</span><strong>List[int]，长度不超过上限</strong>
              <span>不负责</span><strong>概率比较、验证、回退</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>测试数据流</h4>
            <div class="adv-flow">
              <span>草稿<br>[10,20,30,31]</span>
              <span>上限<br>3</span>
              <strong>proposed<br>[10,20,30]</strong>
            </div>
          </section>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>候选少于上限</h4>
            <p><code>[4,5][:3]</code> 返回 <code>[4,5]</code>，切片不会补 token。</p>
          </section>
          <section class="adv-panel warn">
            <h4>不要在这里验证</h4>
            <p>propose 没有 draft_probs 和 target_probs 参数；擅自读取概率会破坏方法职责。</p>
          </section>
        </div>

        <div class="adv-callout">候选序列具有前缀顺序。不要用 set 去重，也不要排序；token 的位置决定后续概率矩阵使用哪一行。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用待办队列练习截取本轮处理上限", [
          "def propose_tasks(task_ids, max_tasks):",
          "    normalized = list(task_ids)",
          "    proposed = normalized[:max_tasks]",
          "    return proposed",
          "",
          "print(propose_tasks((7, 4, 9, 2), 3))",
          "# [7, 4, 9]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>task_ids</code> 对应 <code>draft_tokens</code>。</li>
            <li><code>max_tasks</code> 对应实例属性 <code>self.max_proposal_len</code>。</li>
            <li>两步可以合并成 <code>list(draft_tokens)[: self.max_proposal_len]</code>。</li>
            <li>把结果赋给 Notebook 已给出的 <code>proposed</code> 并返回。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "max_proposal_len=3，draft_tokens=(10,20,30,31)。",
        question: "propose 返回什么类型和内容？",
        options: ["list [10,20,30]", "tuple (10,20,30)", "list [10,20,30,31]"],
        answer: 0,
        revealNote: "先 list 转换，再取前三个，因此类型是 list，长度为 3。"
      },
      checkpoint: checkpoint(
        "为什么 propose 不能把 draft_tokens 排序后再截取？",
        ["候选顺序对应自回归位置和概率矩阵行，排序会改变语义", "因为 Python 不支持整数排序", "因为 max_proposal_len 只能用于 tuple"],
        0,
        "多 token 候选是有顺序的前缀，不是无序集合。"
      ),
      homework: [
        "完成 TODO 1，并分别传入 list、tuple、少于上限的短序列。",
        "测试契约：上限 3 时 [10,20,30,31] 返回 [10,20,30]。",
        "确认返回值是新 list，不会把原 draft_tokens 改写。",
        "错误诊断：返回 4 个 token 时检查是否漏了切片；返回 tuple 时检查是否漏了 list(...)。"
      ]
    }),

    lesson({
      id: "multi-token-verify-first-rejection",
      title: "验证要沿位置走：取出候选概率，达到比例才接受，首次拒绝立刻停止",
      todo: "TODO 2-3：实现接受阈值与逐 token 验证调用",
      prerequisite: [
        "二维概率张量用 tensor[row, column] 取单个值；这里 row=i，column=token_id。",
        "float(tensor_scalar) 把零维 Tensor 转成 Python 浮点数，便于普通条件判断。",
        "正常接受条件是 target_prob 不低于 draft_prob * min_accept_ratio。",
        "accepted_tokens 只保存连续通过的前缀；第一次 False 时记录 rejected_at=i 并 break。"
      ],
      intuition: "第 i 个候选建立在前 i-1 个候选都成立的上下文上。一旦某个位置被目标模型拒绝，后续候选的前缀基础就失效，所以不能跳过它继续接受后面的 token。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>位置 i</b>读取 proposed[i]</span>
          <span><b>二维索引</b>probs[i, token_id]</span>
          <span><b>比例阈值</b>target 对比 draft * ratio</span>
          <span><b>首次拒绝</b>记录 i，立即 break</span>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>位置 i</th><th>token_id</th><th>draft_prob</th><th>target_prob</th><th>结果</th></tr></thead>
          <tbody>
            <tr><td>0</td><td>10</td><td>0.5</td><td>0.8</td><td>接受</td></tr>
            <tr><td>1</td><td>20</td><td>0.5</td><td>0.8</td><td>接受</td></tr>
            <tr><td>2</td><td>30</td><td>0.5</td><td>0.2</td><td>拒绝并停止</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>正常分支</h4>
            <div class="adv-contract">
              <span>ratio</span><strong>0.6</strong>
              <span>阈值</span><strong>0.5 * 0.6 = 0.3</strong>
              <span>0.31</span><strong>达到阈值，True</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>零草稿概率边界</h4>
            <p>draft_prob 不大于 0 时，Notebook 已写好特殊返回：target_prob 大于 0 才接受。TODO 2 不要覆盖这段分支。</p>
          </section>
        </div>

        <div class="adv-callout">draft_probs 和 target_probs 的 shape 在测试中都是 [4,40]：行表示候选位置，列表示 vocabulary token id。不要误写成 probs[token_id, i]。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用质量抽检练习比例阈值与首次失败", [
          "draft_scores = [0.8, 0.6, 0.5]",
          "review_scores = [0.7, 0.4, 0.2]",
          "ratio = 0.6",
          "accepted_indices = []",
          "rejected_at = None",
          "",
          "for i, draft_score in enumerate(draft_scores):",
          "    accepted = review_scores[i] >= draft_score * ratio",
          "    if accepted:",
          "        accepted_indices = accepted_indices + [i]",
          "    else:",
          "        rejected_at = i",
          "        break"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>draft_score/review_scores[i]</code> 对应单个位置的 draft_prob/target_prob。</li>
            <li>TODO 2 把比例条件赋给 <code>accepted</code>；保留已有的零概率提前返回。</li>
            <li>TODO 3 不重写条件，调用 <code>self._accept_token(draft_prob, target_prob)</code>。</li>
            <li>Notebook 已写好追加、rejected_at 和 break 分支；只补方法调用，避免重复控制流。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "draft_prob=0.5，target_prob=0.31，min_accept_ratio=0.6。",
        question: "_accept_token 应返回什么？",
        options: ["True", "False", "None"],
        answer: 0,
        revealNote: "阈值是 0.5 * 0.6 = 0.3，0.31 大于等于 0.3。"
      },
      checkpoint: checkpoint(
        "为什么位置 2 首次拒绝后不能继续验证位置 3 并直接接受？",
        ["位置 3 的候选依赖包含位置 2 的前缀，前缀已不再成立", "因为 tensor 只能索引三行", "因为 accepted_tokens 必须总是偶数长度"],
        0,
        "自回归候选存在前缀依赖，首次拒绝后的整个后缀都需要回退。"
      ),
      homework: [
        "完成 TODO 2-3，先单测 _accept_token(0.5,0.31) 与 (0.5,0.2)。",
        "测试契约：前两个 token 接受，第三个拒绝，返回 accepted=[10,20]、rejected_at=2。",
        "检查 shape/dtype：测试概率表是 float Tensor [4,40]，索引后显式转为 Python float。",
        "错误诊断：IndexError 先查 [i, token_id] 顺序；错误接受后续 token 时检查拒绝分支是否 break。"
      ]
    }),

    lesson({
      id: "multi-token-decode-rollback-history",
      title: "汇总阶段切开两段：接受前缀进入结果，拒绝位置起整段回退",
      todo: "TODO 4：根据 rejected_at 生成 rejected_suffix，并保持 result/history 契约",
      prerequisite: [
        "rejected_at 是整数索引或 None；None 表示所有 proposed token 都被接受。",
        "判断 None 应写 is not None，避免把合法索引 0 当成 False。",
        "proposed[rejected_at:] 从首次拒绝位置开始保留整个后缀，包含被拒绝 token 本身。",
        "decode 返回 dict，并把同一个 result 追加到 self.history；测试会检查 history 长度。"
      ],
      intuition: "验证结束后要把控制流变成可消费的状态：连续接受的前缀可以推进解码，首次拒绝及其后续候选都不再可靠。全接受时没有回退；第一个就拒绝时则整个 proposed 都属于回退后缀。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>发生首次拒绝</h4>
            <div class="adv-flow">
              <span>proposed<br>[10,20,30]</span>
              <strong>accepted<br>[10,20]</strong>
              <span>rejected_at<br>2</span>
              <span>suffix<br>[30]</span>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>全部接受</h4>
            <div class="adv-flow">
              <span>proposed<br>[10,20,30]</span>
              <strong>accepted<br>[10,20,30]</strong>
              <span>rejected_at<br>None</span>
              <span>suffix<br>[]</span>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>proposed_tokens</code></span><strong>本轮最多 max_proposal_len 个候选</strong>
          <span><code>accepted_tokens</code></span><strong>从左到右连续通过的前缀</strong>
          <span><code>accepted_len</code></span><strong>len(accepted_tokens)</strong>
          <span><code>rejected_at</code></span><strong>首次拒绝索引或 None</strong>
          <span><code>rejected_suffix</code></span><strong>从拒绝索引开始的 proposed 后缀</strong>
        </div>

        <div class="adv-callout">不能写 <code>if rejected_at</code>：当第 0 个 token 被拒绝时 rejected_at=0，Python 会把 0 当作 False，错误地产生空后缀。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用批量审批练习 None 与索引 0", [
          "proposed = ['A', 'B', 'C']",
          "rejected_at = 0",
          "",
          "rejected_suffix = (",
          "    proposed[rejected_at:]",
          "    if rejected_at is not None",
          "    else []",
          ")",
          "result = {",
          "    'rejected_at': rejected_at,",
          "    'rejected_suffix': rejected_suffix,",
          "}",
          "# rejected_suffix 是 ['A', 'B', 'C']"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移到 Notebook</h4>
          <ul>
            <li><code>proposed</code> 对应 decode 已经取得的 <code>proposed</code>。</li>
            <li>条件必须保留 <code>rejected_at is not None</code>，正确覆盖索引 0。</li>
            <li>TODO 4 只补 <code>rejected_suffix</code>；result 字典和 history.append 已经写好。</li>
            <li>测试样例 rejected_at=2，因此后缀应从 token 30 开始，即 [30]。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "proposed=[10,20,30]，第 0 个 token 就被拒绝。",
        question: "rejected_suffix 应是什么？",
        options: ["[10,20,30]", "[]", "[20,30]"],
        answer: 0,
        revealNote: "切片从 rejected_at=0 开始，整个提议都需要回退。"
      },
      checkpoint: checkpoint(
        "全部候选通过时，rejected_at=None，应返回什么 rejected_suffix？",
        ["[]", "完整 proposed", "None"],
        0,
        "没有首次拒绝位置，就没有需要回退的后缀；返回类型仍保持 List[int]。"
      ),
      homework: [
        "完成 TODO 4，并运行 test_multi_token_decoder。",
        "测试契约：result 的五个键值与测试一致，且一次 decode 后 len(sim.history)==1。",
        "补测三个边界：全部接受、第 0 个拒绝、候选短于 max_proposal_len。",
        "错误诊断：第 0 个拒绝却返回空列表时检查是否误用 truthy 判断；history 长度错误时不要删除已有 append。"
      ]
    })
  ]
};
