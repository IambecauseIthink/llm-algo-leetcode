const { advancedStyles, checkpoint, code, lesson } = require("../advanced_lesson_helpers");

module.exports = {
  "38": [
    lesson({
      id: "decode-request-state-and-schedule-key",
      title: "先把请求变成状态，再让 tuple 决定谁先运行",
      todo: "TODO 1-2：构造 RequestState，并定义五级调度排序键",
      prerequisite: [
        "dataclass 会根据字段定义自动生成构造函数；RequestState 没有显式传入 generated_len 和 phase 时，会使用 0 与 'prefill'。",
        "Python 的 tuple 按从左到右逐项比较：第一项不同就不再看后面的项，因此排序规则也有明确的优先级层次。",
        "min(items, key=fn) 比较的不是对象本身，而是 fn(item) 返回的键；键越小，对象越先被选中。",
        "负号可以反转一个数字的排序方向：priority 越大，-priority 越小，所以高业务优先级会排在前面。"
      ],
      intuition: "调度器不能只保存 request_id，因为每次选择都依赖请求当前处于 prefill 还是 decode、是否命中 cache、优先级和长度。先把这些属性封装成 RequestState，再把业务规则翻译成一个从强到弱的 tuple，调度决策就从模糊描述变成了可检查的数据。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>enqueue</b>参数变成 RequestState</span>
          <span><b>phase_rank</b>prefill=0，decode=1</span>
          <span><b>cache_rank</b>hit=0，miss=1</span>
          <span><b>tuple</b>从左到右决定先后</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>TODO 1 的对象契约</h4>
            <div class="adv-contract">
              <span><code>request_id</code></span><strong>请求的稳定标识</strong>
              <span><code>prompt_len</code></span><strong>同时也是本模拟器的目标生成长度</strong>
              <span><code>generated_len</code></span><strong>默认 0，decode 时逐步增加</strong>
              <span><code>phase</code></span><strong>默认从 prefill 开始</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>TODO 2 的比较顺序</h4>
            <div class="adv-steps">
              <div><b>1</b><code>phase_rank</code><span>所有 prefill 都先于 decode</span></div>
              <div><b>2</b><code>cache_rank</code><span>同阶段内 cache hit 优先</span></div>
              <div><b>3</b><code>-req.priority</code><span>同条件下数值更大的优先级先跑</span></div>
              <div><b>4</b><code>total_len, request_id</code><span>用长度与 id 稳定破平局</span></div>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>请求</th><th>状态</th><th>排序键前半段</th><th>为什么</th></tr></thead>
          <tbody>
            <tr><td>A</td><td>prefill, miss, priority=9</td><td><code>(0,1,-9,...)</code></td><td>阶段项是 0</td></tr>
            <tr><td>B</td><td>decode, hit, priority=99</td><td><code>(1,0,-99,...)</code></td><td>阶段项是 1，仍排在 A 后</td></tr>
            <tr><td>C</td><td>prefill, hit, priority=2</td><td><code>(0,0,-2,...)</code></td><td>同为 prefill，命中 cache 使它排在 A 前</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">排序键的第一项权力最大。不要看到 B 的 priority=99 就误判它先运行：Notebook 明确把 phase_rank 放在 tuple 最前面，因此任何 active prefill 都先于 decode。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用维修工单练习 dataclass 与多级排序", [
          "from dataclasses import dataclass",
          "",
          "@dataclass",
          "class Ticket:",
          "    ticket_id: int",
          "    urgent: bool = False",
          "    waiting_minutes: int = 0",
          "",
          "def ticket_key(ticket):",
          "    urgent_rank = 0 if ticket.urgent else 1",
          "    return (urgent_rank, -ticket.waiting_minutes, ticket.ticket_id)",
          "",
          "tickets = [Ticket(8), Ticket(3, True, 4), Ticket(5, True, 9)]",
          "chosen = min(tickets, key=ticket_key)  # ticket_id == 5"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>Ticket(...)</code> 的具名参数构造 -&gt; TODO 1 的 <code>RequestState(request_id=..., prompt_len=..., priority=..., cache_hit=...)</code>。</li>
            <li><code>urgent_rank</code> -&gt; <code>phase_rank</code> 与 <code>cache_rank</code>：把布尔/类别规则变成可排序数字。</li>
            <li><code>-waiting_minutes</code> -&gt; <code>-req.priority</code>：较大的原值获得较小的排序项。</li>
            <li><code>ticket_id</code> -&gt; <code>req.request_id</code>：最后一个字段只负责稳定破平局。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "三个 active 请求分别是 A=(prefill, miss, priority=9)、B=(decode, hit, priority=99)、C=(prefill, hit, priority=2)。",
        question: "按 Notebook 的 key，谁会最先被 min 选中？",
        options: ["C", "B", "A"],
        answer: 0,
        revealNote: "先比 phase：A/C 胜过 B；再比 cache：C 的 hit 胜过 A 的 miss。"
      },
      checkpoint: checkpoint(
        "为什么排序键里要写 -req.priority，而不是 req.priority？",
        ["因为 min 选择较小键，取负后高 priority 会变得更小", "因为 priority 必须是负数类型", "因为负号会把 priority 变成布尔值"],
        0,
        "调度器使用 min；若直接放正数，反而会让低 priority 先被选中。"
      ),
      homework: [
        "完成 TODO 1：只传请求参数给 RequestState，让 generated_len=0、phase='prefill' 使用 dataclass 默认值，再 append 到 self.queue。",
        "完成 TODO 2：严格按 (phase_rank, cache_rank, -req.priority, req.total_len, req.request_id) 返回 tuple，不要擅自交换字段顺序。",
        "契约检查：enqueue 后 queue 长度增加，首个请求 phase 是 prefill；本课只使用 Python 对象和整数，没有 tensor、dtype 或 device。",
        "错误诊断：若测试首事件不是 request 3，打印三个请求的 key，从 tuple 第一项开始找出与预期不同的字段。"
      ]
    }),

    lesson({
      id: "decode-step-run-and-event-contract",
      title: "一次只推进一个状态：从 step 事件走到 run 终止",
      todo: "TODO 3-4：从 active 请求中选择 chosen，并只在成功事件后增加 steps",
      prerequisite: [
        "列表推导式 [req for req in queue if not req.done] 会过滤完成请求；done 是 generated_len >= prompt_len。",
        "step 每次只做一个动作：prefill 请求切换到 decode，decode 请求才会把 generated_len 加 1。",
        "None 是“没有 active 请求”的哨兵值；run 看到 None 必须 break，不能再增加 steps。",
        "self.timeline 是同一个列表对象；run 返回 self.timeline，因此测试中的 sim.timeline is events 检查的是对象身份。"
      ],
      intuition: "把 step 想成调度器的一次时钟滴答。每滴答只选择一个请求并产生一条事件，run 只是反复调用 step。只要单步状态迁移正确、空队列能返回 None、计数器只在真实事件后增加，整个请求生命周期就能自然结束。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>active 过滤<br>排除 done</span>
          <span>min + key<br>选 chosen</span>
          <span>prefill<br>只改 phase</span>
          <span>decode<br>generated_len + 1</span>
          <strong>记录 event<br>append timeline</strong>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>单个 prompt_len=2 请求</h4>
            <div class="adv-steps">
              <div><b>1</b><code>prefill_to_decode</code><span>phase 变 decode，generated_len 仍为 0</span></div>
              <div><b>2</b><code>decode_one_step</code><span>generated_len 变 1，尚未 done</span></div>
              <div><b>3</b><code>finish</code><span>generated_len 变 2，达到 prompt_len</span></div>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>run 的循环不变量</h4>
            <div class="adv-contract">
              <span>有事件</span><strong>追加 timeline，然后 steps = steps + 1</strong>
              <span>无事件</span><strong>step 返回 None，立即 break</strong>
              <span>安全上限</span><strong>steps 永远不超过 max_steps</strong>
              <span>返回对象</span><strong>直接返回 self.timeline，不复制</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>测试输入</th><th>prefill 事件数</th><th>decode 事件数</th><th>总事件数</th></tr></thead>
          <tbody>
            <tr><td>prompt_len 2、3、1 的三个请求</td><td>3</td><td>2+3+1=6</td><td>9</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">本模拟器为了教学，把“目标生成长度”也设成 prompt_len，所以每个请求需要 1 次 prefill 转换加 prompt_len 次 decode。不要把真实服务里的 max_new_tokens 规则自行加进 Notebook。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用下载任务练习哨兵循环", [
          "jobs = [2, 1]",
          "events = []",
          "",
          "def advance_once():",
          "    active = [i for i, left in enumerate(jobs) if left > 0]",
          "    if not active:",
          "        return None",
          "    chosen = min(active)",
          "    jobs[chosen] -= 1",
          "    event = {'job': chosen, 'left': jobs[chosen]}",
          "    events.append(event)",
          "    return event",
          "",
          "steps = 0",
          "while steps < 10:",
          "    event = advance_once()",
          "    if event is None:",
          "        break",
          "    steps = steps + 1"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>active</code> 过滤未完成下载 -&gt; Notebook 已写好的未 done 请求列表。</li>
            <li><code>min(active)</code> -&gt; TODO 3 的 <code>min(active, key=self._schedule_key)</code>。</li>
            <li><code>event is None</code> -&gt; run 的终止条件；None 不代表一次成功调度。</li>
            <li><code>steps = steps + 1</code> -&gt; TODO 4，必须放在 None 检查之后。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "一个请求初始 phase='prefill'、generated_len=0、prompt_len=1。",
        question: "它完整结束需要产生几条事件？",
        options: ["2 条：一次切到 decode，一次 decode 后 finish", "1 条：prefill 时直接 finish", "3 条：还要额外记录空队列事件"],
        answer: 0,
        revealNote: "prefill 只切换阶段；下一次 step 才增加 generated_len 并产生 finish。"
      },
      checkpoint: checkpoint(
        "steps 应该在什么时候加 1？",
        ["step 返回真实 event 后", "每次进入 while 的第一行", "step 返回 None 后"],
        0,
        "这样 max_steps 统计的才是成功执行的调度动作，空队列只负责结束循环。"
      ),
      homework: [
        "完成 TODO 3-4，随后手算三个测试请求的 3 次 prefill 与 6 次 decode，确认 events 长度应为 9。",
        "检查事件契约：第一条是 request 3 的 prefill_to_decode；至少一条 action 是 finish；run 结束后所有 req.done 为 True。",
        "状态检查：prefill 不应增加 generated_len，decode 每次只加 1；本课没有 tensor shape/dtype/device。",
        "错误诊断：无限循环先查 steps 是否更新；事件过少先查 done 条件；首请求错误则回到上一任务检查 tuple 的字段顺序。"
      ]
    })
  ],

  "39": [
    lesson({
      id: "weight-calibration-groups-and-awq-mask",
      title: "先找敏感通道：RMS 校准、向上分组与 AWQ mask",
      todo: "TODO 1-3：统计输入通道重要性、计算 group 数、标记每组 top-k 敏感位置",
      prerequisite: [
        "Linear 权重 shape 是 [out_features,in_features]；校准激活最后一维也必须是 in_features。",
        "RMS 的顺序是平方、求平均、开平方；对 activations 除最后一维外的所有维度归约，才能留下每个输入通道一个分数。",
        "向上取整分组可写成 (length + group_size - 1) // group_size，保证不足一整组的尾部也被覆盖。",
        "torch.topk(...).indices 返回局部 group 内的下标；布尔 mask 用 True 表示该位置被保护。"
      ],
      intuition: "4-bit 可表示的状态很少，误差不能平均分配。Notebook 用校准激活的 RMS 估计输入通道的重要程度：某通道经常出现大激活，它对应的权重误差更容易影响输出。AWQ 教学模式在每个 group 内挑出 top-k，先做一张保护名单，再量化剩余权重。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>activations</b>[样本..., in_features]</span>
          <span><b>RMS reduce</b>只留下 [in_features]</span>
          <span><b>ceil groups</b>覆盖最后不足一组的通道</span>
          <span><b>top-k mask</b>每组保护高重要性位置</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>TODO 1：维度为什么这样消失</h4>
            <table class="adv-shapes">
              <thead><tr><th>变量</th><th>shape</th><th>动作</th></tr></thead>
              <tbody>
                <tr><td><code>act</code></td><td>[16,8]</td><td>校准 batch 有 16 个样本、8 个输入通道</td></tr>
                <tr><td><code>reduce_dims</code></td><td>(0,)</td><td>只归约样本维</td></tr>
                <tr><td><code>importance</code></td><td>[8]</td><td>每个输入通道留下一个 RMS</td></tr>
              </tbody>
            </table>
          </section>
          <section class="adv-panel good">
            <h4>TODO 2-3：每组独立保护</h4>
            <div class="adv-contract">
              <span><code>in_features=8</code></span><strong>输入通道总数</strong>
              <span><code>group_size=4</code></span><strong>得到 n_groups=2</strong>
              <span><code>protect_ratio=.25</code></span><strong>每个 4 元 group 保护 k=1</strong>
              <span><code>mask[topk]=True</code></span><strong>只标记当前 group 的局部位置</strong>
            </div>
          </section>
        </div>

        <div class="adv-flow">
          <span>权重一行<br>[8]</span>
          <span>group 0<br>通道 0:4</span>
          <span>group 1<br>通道 4:8</span>
          <strong>每组各自 top-k<br>写入二维 protected_mask</strong>
        </div>

        <div class="adv-callout">这只是 AWQ 核心直觉的教学模拟，不是完整 AWQ 搜索；GPTQ 模式不会进入 method == "awq" 分支，因此 protected_mask 应全部为 False。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用传感器校准值练习 RMS 与分组 top-k", [
          "samples = torch.tensor([[3.0, 0.0, 1.0, 2.0],",
          "                        [4.0, 0.0, 1.0, 2.0]])",
          "dims = tuple(range(samples.ndim - 1))",
          "importance = samples.pow(2).mean(dim=dims).sqrt()  # [4]",
          "",
          "width = 3",
          "num_groups = (importance.numel() + width - 1) // width",
          "first_group = importance[0:3]",
          "mask = torch.zeros_like(first_group, dtype=torch.bool)",
          "top_index = torch.topk(first_group, k=1).indices",
          "mask[top_index] = True"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>samples</code> -&gt; <code>activations</code>；最后一维就是 in_features。</li>
            <li><code>dims</code> 与 RMS 表达式 -&gt; TODO 1，输出元素数必须等于 in_features。</li>
            <li><code>num_groups</code> -&gt; TODO 2；不能用普通整除丢掉尾组。</li>
            <li><code>mask[top_index] = True</code> -&gt; TODO 3 的 <code>mask[topk] = True</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "in_features=10、group_size=4。",
        question: "n_groups 应该是多少，最后一组有几个通道？",
        options: ["3 组，最后 2 个通道", "2 组，最后 4 个通道", "4 组，最后 0 个通道"],
        answer: 0,
        revealNote: "(10+4-1)//4=3，区间依次是 0:4、4:8、8:10。"
      },
      checkpoint: checkpoint(
        "对 shape [batch,seq,in_features] 的 activations，RMS 应归约哪些维？",
        ["batch 与 seq，保留最后的 in_features", "只归约 in_features", "三个维度全部归约成标量"],
        0,
        "Notebook 用 tuple(range(act.ndim - 1))，也就是保留最后一维的逐通道重要性。"
      ),
      homework: [
        "完成 TODO 1-3：先验证 importance.numel()==in_features，再确保每个 AWQ group 至少保护 1 个且不超过本组长度。",
        "shape/dtype/device：测试中 weight=[4,8]、acts=[16,8]，importance=[8]、protected_mask=[4,8]；fit 会 detach 并转 float，但保持原 device。",
        "测试：AWQ protect_ratio=.25、group_size=4 时 protected_mask.any() 必须为 True；GPTQ 实例的 mask 必须全 False。",
        "错误诊断：importance 变标量说明误归约了最后一维；scales 少一组说明用了向下整除；mask 没生效则检查 topk 下标是否写成 True。"
      ]
    }),

    lesson({
      id: "weight-quantize-dequantize-and-mse",
      title: "低比特闭环：scale 只看普通通道，恢复后再盖回保护值",
      todo: "TODO 4-6：计算分组 scale、反量化 qweight，并用 MSE 检查重构误差",
      prerequisite: [
        "本课采用对称量化：qmax=2^(bits-1)-1；bits=4 时 qmax=7，int8 只是保存这些低比特整数的容器。",
        "scale = absmax / qmax，因此量化是 round(weight / scale)，反量化必须用 q * scale。",
        "AWQ 的保护位置没有写入正常 qweight；反量化后要用 protected_weight 覆盖 mask=True 的位置。",
        "MSE 是 (original - reconstructed)^2 的逐元素平均，返回一个标量 Tensor。"
      ],
      intuition: "量化和反量化必须是一对方向相反的操作。先用一个 group 的普通权重确定步长 scale，把连续值落到有限整数格点；恢复时乘回同一步长。被保护通道绕过格点近似，最后像贴补丁一样用原浮点值盖回去。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>普通通道的对称量化</h4>
            <div class="adv-flow">
              <span>base.abs().max()</span>
              <span>除以 qmax<br>得到 scale</span>
              <span>round(w/scale)</span>
              <strong>clamp 后存 int8</strong>
            </div>
            <p><code>clamp_min(eps)</code> 防止全零 group 产生除零。</p>
          </section>
          <section class="adv-panel good">
            <h4>反量化与保护覆盖</h4>
            <div class="adv-flow">
              <span>q_group<br>转 scales.dtype</span>
              <span>q_group * scale</span>
              <span>mask=True<br>取 protected_weight</span>
              <strong>恢复完整 weight</strong>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>qweight</code></span><strong>shape [out_features,in_features]，dtype int8</strong>
          <span><code>scales</code></span><strong>shape [out_features,n_groups]，每行每组一个 scale</strong>
          <span><code>dequantize()</code></span><strong>恢复 shape [out_features,in_features] 的浮点近似权重</strong>
          <span><code>forward(x)</code></span><strong>F.linear(x, restored_weight)，[2,8] -&gt; [2,4]</strong>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>TODO</th><th>Notebook 表达式方向</th><th>常见反向错误</th></tr></thead>
          <tbody>
            <tr><td>4 scale</td><td><code>absmax / qmax</code></td><td>写成 qmax / absmax</td></tr>
            <tr><td>5 restore</td><td><code>q_group * scale</code></td><td>再次除以 scale</td></tr>
            <tr><td>6 MSE</td><td><code>((weight-recon)**2).mean()</code></td><td>只求差值平均，正负会抵消</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">第 40 课会采用另一种 scale 定义 q=round(x*scale)，所以恢复时除 scale。不要只背“量化都乘或都除”；先看当前 scale 的公式，再写互逆操作。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用一组温度权重练习对称量化闭环", [
          "values = torch.tensor([-1.0, -0.2, 0.6, 1.4])",
          "qmax = 7",
          "eps = 1e-8",
          "",
          "scale = (values.abs().max() / qmax).clamp_min(eps)",
          "codes = torch.clamp(torch.round(values / scale), -qmax, qmax)",
          "restored = codes * scale",
          "error = ((values - restored) ** 2).mean()",
          "",
          "assert restored.shape == values.shape",
          "assert error >= 0"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>values</code> -&gt; TODO 4 的 <code>base</code>，也就是排除保护位置后的普通权重。</li>
            <li><code>scale</code> -&gt; <code>scales[row,g]</code>；Notebook 每行每组都单独保存。</li>
            <li><code>codes * scale</code> -&gt; TODO 5 的 <code>dequant</code>。</li>
            <li><code>error</code> -&gt; TODO 6；Notebook 比较完整原权重与 <code>dequantize()</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "bits=4，某组普通权重 absmax=1.4。",
        question: "Notebook 的 qmax 与 scale 分别是多少？",
        options: ["7 与 0.2", "15 与约 0.093", "127 与约 0.011"],
        answer: 0,
        revealNote: "有符号 4-bit 教学范围使用 qmax=2^(4-1)-1=7，scale=1.4/7=0.2。"
      },
      checkpoint: checkpoint(
        "为什么 AWQ 反量化后还要按 protected_mask 覆盖 protected_weight？",
        ["保护通道没有按普通低比特路径保存，需要恢复原浮点权重", "为了把所有 qweight 变成 bool", "为了改变输出 shape"],
        0,
        "保护位置绕开了普通量化误差；覆盖只替换数值，不改变 shape。"
      ),
      homework: [
        "完成 TODO 4-6，并在一个 group 上打印 base、scale、q_group、dequant，确认除 scale 与乘 scale 互为反向操作。",
        "shape/dtype/device：测试要求 qweight 为 int8、scales=[4,2]、restored=[4,8]、forward([2,8])=[2,4]；输入 x 与保存权重需位于兼容 device。",
        "测试：AWQ 与 GPTQ 都要完成 dequantize；MSE 只要求非负，不能把“非负”误当成精度充分的证明。",
        "错误诊断：输出量级异常先查乘除方向；保护位置变 0 查覆盖逻辑；forward 报 device mismatch 时确保校准权重与 x 在同一设备。"
      ]
    })
  ],

  "40": [
    lesson({
      id: "fp8-sim-symmetric-quantization-state",
      title: "先读 scale 的定义：全局量化、反量化与 shape 状态",
      todo: "TODO 1-4：计算 absmax/scale/int8 量化值，并记录 fp8_shape",
      prerequisite: [
        "Notebook 明确说明这是用 int8 容器做的 FP8 思想模拟，不是 E4M3 或 E5M2 的真实硬件编码。",
        "这里 scale 定义为 qmax / absmax，所以量化使用 x * scale，反量化使用 q / scale。",
        "torch.round 只负责取最近整数，torch.clamp 负责限制范围，to(torch.int8) 才改变存储 dtype。",
        "tuple(x.shape) 把 torch.Size 转成普通 tuple，测试用它检查量化状态是否记住原始结构。"
      ],
      intuition: "低精度状态不只有整数值。要想恢复数据，至少要一起保存 q、scale 和 shape。本任务先做最小全局量化：整个 hidden tensor 共用一个标量 scale，让初学者先跑通可逆方向，再进入 KV Cache 的分组 scale。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>absmax</b>找到输入动态范围</span>
          <span><b>scale</b>qmax / clamp(absmax)</span>
          <span><b>quantize</b>round + clamp + int8</span>
          <span><b>state</b>保存 q、scale、shape</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>本课的乘除方向</h4>
            <div class="adv-contract">
              <span>scale</span><strong><code>qmax / absmax</code></strong>
              <span>量化</span><strong><code>q = round(x * scale)</code></strong>
              <span>反量化</span><strong><code>x_hat = q / scale</code></strong>
              <span>防零</span><strong><code>absmax.clamp_min(eps)</code></strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>fit 后保存的状态</h4>
            <div class="adv-contract">
              <span><code>fp8_q</code></span><strong>与 hidden 同 shape，dtype int8</strong>
              <span><code>fp8_scale</code></span><strong>全局标量 Tensor</strong>
              <span><code>fp8_shape</code></span><strong>tuple(hidden.shape)</strong>
              <span>恢复 dtype</span><strong>跟随 scale.dtype，本测试中为 float32</strong>
            </div>
          </section>
        </div>

        <div class="adv-flow">
          <span>hidden<br>[2,8] float</span>
          <span>全局 absmax<br>标量</span>
          <span>fp8_q<br>[2,8] int8</span>
          <strong>dequantize_fp8<br>[2,8] float</strong>
        </div>

        <div class="adv-callout">名称里有 FP8，不代表 q 的 dtype 会是 torch.float8。测试明确要求 int8；课程只保留“低精度编码 + scale 恢复”的教学闭环。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用音频幅度练习乘 scale 的对称量化", [
          "audio = torch.tensor([-0.8, -0.1, 0.3, 1.0])",
          "limit = 127",
          "eps = 1e-8",
          "",
          "peak = torch.max(torch.abs(audio))",
          "multiplier = limit / peak.clamp_min(eps)",
          "packed = torch.clamp(",
          "    torch.round(audio * multiplier), -limit, limit",
          ").to(torch.int8)",
          "restored = packed.to(multiplier.dtype) / multiplier.clamp_min(eps)",
          "original_shape = tuple(audio.shape)"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>peak</code> -&gt; TODO 1 的 <code>absmax</code>。</li>
            <li><code>multiplier</code> -&gt; TODO 2 的 <code>scale</code>；此处含义是“浮点值要乘多少”。</li>
            <li><code>packed</code> -&gt; TODO 3 的 <code>q</code>，操作顺序不能漏掉 clamp 或 int8 转换。</li>
            <li><code>original_shape</code> -&gt; TODO 4 的 <code>self.fp8_shape</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "输入 absmax=2，qmax=100。",
        question: "scale 是多少？数值 x=1 会量化到约多少？",
        options: ["scale=50，q=50", "scale=0.02，q=0", "scale=200，q=200"],
        answer: 0,
        revealNote: "scale=qmax/absmax=50，量化是 round(x*scale)，所以 1 映射到 50。"
      },
      checkpoint: checkpoint(
        "本 Notebook 的 _sym_dequantize 为什么是 q / scale？",
        ["因为量化阶段使用了 x * scale", "因为 int8 只能做除法", "因为 scale 等于 absmax / qmax"],
        0,
        "恢复操作要撤销量化时的乘法；注意第 39 课采用的是另一种 scale 定义。"
      ),
      homework: [
        "完成 TODO 1-4，并用全零 tensor 额外调用 _sym_quantize，确认 clamp_min(eps) 阻止除零。",
        "shape/dtype/device：测试 hidden=[2,8]，fp8_q 同 shape 且 int8，fp8_shape=(2,8)；_sym_quantize 会 detach().float() 并保持输入 device。",
        "测试：fit 后 dequantize_fp8 与 forward 的 hidden 输出都必须保持 [2,8]；不要改写成真实 FP8 API。",
        "错误诊断：恢复值接近 0 先查是否错误地乘了 scale；溢出先查 clamp；dtype 不对则检查 .to(torch.int8) 是否在最后。"
      ]
    }),

    lesson({
      id: "kv-group-quantization-restore-and-mse",
      title: "沿最后一维分组：每个 KV chunk 必须找回自己的 scale",
      todo: "TODO 5-7：向上计算 KV group 数、逐组恢复，并计算 float MSE",
      prerequisite: [
        "KV Cache 可以有多个前导维；Notebook 只要求 ndim >= 2，并始终沿最后一维分组。",
        "kv.shape[:-1] 保留所有前导维，再拼上 n_groups，得到每个逻辑 row 对应的 scale 表。",
        "reshape(-1,last_dim) 把所有前导维合并为 row，量化完成后必须 reshape(self.kv_shape) 恢复原结构。",
        "每个 group 都由 _sym_quantize 得到自己的 scale；恢复时不能使用全局 fp8_scale。"
      ],
      intuition: "长 KV 向量里的不同区域可能有不同数值范围。分组就像给每个小箱子单独配一把尺子：整数数据放在 kv_q，尺子放在 kv_scale。恢复时按 row 和 group 找回配对的尺子，才能把扁平处理后的数据拼回原 shape。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>测试 shape 展开</h4>
            <table class="adv-shapes">
              <thead><tr><th>对象</th><th>shape</th><th>含义</th></tr></thead>
              <tbody>
                <tr><td><code>kv</code></td><td>[2,3,8]</td><td>6 条长度为 8 的逻辑 row</td></tr>
                <tr><td><code>flat</code></td><td>[6,8]</td><td>只为循环方便，数据次序不变</td></tr>
                <tr><td><code>kv_scale</code></td><td>[2,3,2]</td><td>group_size=4，每条 row 有 2 个 scale</td></tr>
                <tr><td><code>flat_scale</code></td><td>[6,2]</td><td>与 flat 的 row 一一对应</td></tr>
              </tbody>
            </table>
          </section>
          <section class="adv-panel good">
            <h4>TODO 6 的配对规则</h4>
            <div class="adv-steps">
              <div><b>1</b><code>start=g*group_size</code><span>定位当前最后一维区间</span></div>
              <div><b>2</b><code>scale=flat_scale[row,g]</code><span>取同一 row、同一 group 的尺子</span></div>
              <div><b>3</b><code>_sym_dequantize(chunk,scale)</code><span>复用已有的除法恢复</span></div>
              <div><b>4</b><code>reshape(self.kv_shape)</code><span>恢复所有前导维</span></div>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span>TODO 5</span><strong><code>(last_dim + kv_group_size - 1) // kv_group_size</code></strong>
          <span>TODO 6</span><strong>当前 q chunk 与 <code>flat_scale[row,g]</code> 配对</strong>
          <span>TODO 7</span><strong><code>mean((original.float()-restored.float())**2)</code></strong>
          <span>forward 返回</span><strong>有 kv 时返回 (fp8_restored, kv_restored)，无 kv 时只返回 hidden</strong>
        </div>

        <div class="adv-callout">MSE 前显式转 float，避免低精度或整数运算改变误差语义；MSE 非负只是测试底线，真实部署还要评估生成质量。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用分段成绩数据练习 flatten、分组与还原", [
          "scores = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)",
          "last_dim = scores.size(-1)",
          "group_size = 3",
          "groups = (last_dim + group_size - 1) // group_size",
          "",
          "flat = scores.reshape(-1, last_dim)  # [6, 4]",
          "restored = torch.zeros_like(flat)",
          "for row in range(flat.size(0)):",
          "    for group in range(groups):",
          "        start = group * group_size",
          "        end = min(start + group_size, last_dim)",
          "        restored[row, start:end] = flat[row, start:end]",
          "",
          "restored = restored.reshape(scores.shape)",
          "mse = ((scores.float() - restored.float()) ** 2).mean()"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>groups</code> -&gt; TODO 5 的 <code>n_groups</code>，尾部长度 1 仍有独立 group。</li>
            <li><code>row/group/start/end</code> -&gt; quantize 与 dequantize 两个循环的相同索引契约。</li>
            <li>例子中的直接复制 -&gt; TODO 6 要替换为 <code>self._sym_dequantize(flat[row,start:end], scale)</code>。</li>
            <li><code>mse</code> -&gt; TODO 7，必须比较原始与恢复后的同 shape tensor。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "kv.shape=[2,3,10]，kv_group_size=4。",
        question: "kv_scale 的 shape 应该是什么？",
        options: ["[2,3,3]", "[2,3,4]", "[6,10]"],
        answer: 0,
        revealNote: "最后一维 10 向上分成 3 组，前导维 [2,3] 保持不变。"
      },
      checkpoint: checkpoint(
        "恢复 row=2、group=1 的 KV chunk 时应该使用哪个 scale？",
        ["flat_scale[2,1]", "fp8_scale", "flat_scale[1,2]"],
        0,
        "量化时 scale 按逻辑 row 和最后一维 group 保存，恢复必须使用完全相同的索引。"
      ),
      homework: [
        "完成 TODO 5-7，先在纸上写出测试 kv=[2,3,8]、group_size=4 时 flat=[6,8]、kv_scale=[2,3,2]。",
        "shape/dtype/device：kv_q 为 int8，kv_scale 与恢复输出在量化后的 float dtype/device 上；kv_restore 与原 kv shape 完全一致。",
        "测试：fit、dequantize_kv_cache 和 forward 都要通过；再试 last_dim=10 验证尾组没有被遗漏。",
        "错误诊断：scale shape 错查 kv.shape[:-1]；恢复数值错查 row/g 是否配对；reshape 报错查是否给每个 chunk 都写回原长度。"
      ]
    })
  ],

  "41": [
    lesson({
      id: "kv-cache-score-heap-and-stale-records",
      title: "优先级会变化：评分、最小堆与懒删除必须配套",
      todo: "TODO 1-3：计算保留分数、压入最新堆项，并识别过期记录",
      prerequisite: [
        "heapq 是最小堆：heappop 总是先弹出 tuple 最小的记录，因此低 priority 天然先被驱逐。",
        "缓存命中会增加 hits 并更新 last_used，同一个 prefix 的 priority 会随时间变化。",
        "heapq 没有直接修改任意旧项的简单接口；Notebook 采用懒删除：压入新项，弹出时检查旧项是否 stale。",
        "tuple (priority,last_used,prefix) 既定义堆顺序，也携带足够信息与 entries 中的最新对象核对。"
      ],
      intuition: "entries 是当前事实，queue 只是等待处理的历史便签。每次缓存价值变化，就把一张新便签压入堆；旧便签仍在，但真正驱逐前必须与 entries 核对。这样不用在堆中搜索删除，却能保证不会根据过期优先级误删热门缓存。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>TODO 1：可解释评分</h4>
            <div class="adv-contract">
              <span><code>reuse_bonus</code></span><strong>hits，复用越多分越高</strong>
              <span><code>recency</code></span><strong>1/(1+time gap)，越新越接近 1</strong>
              <span><code>size_penalty</code></span><strong>size/capacity，越大成本越高</strong>
              <span><code>score</code></span><strong>reuse + 0.5*recency - 0.25*size penalty</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>TODO 2-3：两份数据各司其职</h4>
            <div class="adv-contract">
              <span><code>entries[prefix]</code></span><strong>唯一的最新 CacheEntry</strong>
              <span><code>queue</code></span><strong>可能含同一 prefix 的多条历史 tuple</strong>
              <span>最新项</span><strong>(priority,last_used) 与 entry 相同</strong>
              <span>过期项</span><strong>二元组不相等，continue 跳过</strong>
            </div>
          </section>
        </div>

        <div class="adv-flow">
          <span>a 首次加入<br>hits=1</span>
          <span>压入旧项<br>(p1,t1,a)</span>
          <span>a 再次命中<br>hits=2,t2</span>
          <span>压入新项<br>(p2,t2,a)</span>
          <strong>弹出旧项时<br>stale check 跳过</strong>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>动作</th><th>堆中 priority 符号</th><th>原因</th></tr></thead>
          <tbody>
            <tr><td>保留高价值项</td><td>不取负</td><td>最小堆先弹出低分项用于驱逐</td></tr>
            <tr><td>snapshot 高分在前</td><td>排序 key 用 -e.priority</td><td>这是展示排序，与驱逐堆方向相反</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">如果在 _refresh_queue 对 priority 取负，最小堆会先弹出绝对值最大的高价值缓存，驱逐方向正好反了。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用文章热度练习最小堆与过期项检查", [
          "import heapq",
          "",
          "latest = {'guide': (3.5, 8)}  # score, last_seen",
          "queue = []",
          "heapq.heappush(queue, (1.2, 3, 'guide'))  # 历史项",
          "heapq.heappush(queue, (3.5, 8, 'guide'))  # 最新项",
          "",
          "score, seen, name = heapq.heappop(queue)",
          "current = latest.get(name)",
          "is_stale = current is not None and (score, seen) != current",
          "if is_stale:",
          "    pass  # 跳过旧记录，继续弹堆"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>heapq.heappush</code> -&gt; TODO 2；queue_item 是 <code>(entry.priority,entry.last_used,prefix)</code>。</li>
            <li><code>latest</code> -&gt; <code>self.entries</code>，它才保存当前真相。</li>
            <li><code>(score,seen) != current</code> -&gt; TODO 3 对 priority 与 last_used 的联合核对。</li>
            <li>例子省略的评分 -&gt; TODO 1 必须使用 Notebook 给定的三个权重系数。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "同一 prefix 的旧堆项是 (1.2,3)，entries 中最新值是 (3.5,8)。",
        question: "弹出旧项后应该怎么做？",
        options: ["判定 stale 并继续弹下一项", "立即驱逐该 prefix", "把 current_bytes 清零"],
        answer: 0,
        revealNote: "queue 可以有历史记录；只有与 entries 最新 priority 和 last_used 同时一致的项才有效。"
      },
      checkpoint: checkpoint(
        "为什么 queue_item 中的 priority 不取负？",
        ["因为这里要让低 priority 先从最小堆弹出并被驱逐", "因为 heapq 会自动把数字取负", "因为 priority 永远小于 0"],
        0,
        "驱逐堆与高分在前的展示排序方向不同，不能混用。"
      ),
      homework: [
        "完成 TODO 1-3，并手算一个 hits=2、当前刚访问、size/capacity=.25 的 score，确认复用和新近性加分、大小扣分。",
        "数据契约：entries 是 Dict[str,CacheEntry]，queue 元素是三元 tuple；本课不存真实 KV tensor，因此没有 shape/dtype/device。",
        "测试前额外连续 touch 同一 prefix 两次，观察 queue 中允许重复项，但 stale check 不会误驱逐最新 entry。",
        "错误诊断：热门项被先删查 priority 是否错误取负；KeyError 查已删除 prefix 是否跳过；旧项生效查 stale 比较是否同时覆盖 priority 与 last_used。"
      ]
    }),

    lesson({
      id: "kv-cache-touch-eviction-and-snapshot",
      title: "容量不足时再驱逐：创建条目、维护字节数、导出稳定快照",
      todo: "TODO 4-5：创建新 CacheEntry，并按高优先级导出 snapshot",
      prerequisite: [
        "新 prefix 第一次 touch 既是写入也是首次使用，所以 hits 从 1 开始，last_used 使用已经加 1 的 self.time。",
        "_evict_until_fit(needed) 的循环条件是 current_bytes + needed > capacity_bytes；每次驱逐都要同步扣减 current_bytes。",
        "CacheEntry 使用 order=True，但 prefix/hits/bytes 标记 compare=False；堆中仍显式保存 tuple，避免依赖整个对象比较。",
        "sorted 的 key 可用 -priority 实现高分在前，再用 last_used 与 prefix 做确定性破平局。"
      ],
      intuition: "touch 是调度器的统一入口：命中时更新价值，未命中时先腾空间再登记新对象。驱逐、entries、current_bytes、log 和 queue 必须一起维护，最后 snapshot 把内部状态转换成容易测试和观察的四元组列表。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>touch</b>time 先加 1</span>
          <span><b>hit</b>hits/last_used 更新并 refresh</span>
          <span><b>miss</b>先 evict_until_fit</span>
          <span><b>add</b>写 entries、bytes、queue、log</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>TODO 4：新条目的初始值</h4>
            <div class="adv-contract">
              <span><code>priority=0.0</code></span><strong>占位，随后 _refresh_queue 统一重算</strong>
              <span><code>last_used=self.time</code></span><strong>记录当前 touch 时刻</strong>
              <span><code>prefix=prefix</code></span><strong>entries 的业务键</strong>
              <span><code>hits=1, bytes=bytes_</code></span><strong>首次使用与真实容量成本</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>TODO 5：快照契约</h4>
            <div class="adv-contract">
              <span>内部排序</span><strong><code>(-priority,last_used,prefix)</code></strong>
              <span>输出 tuple</span><strong>(prefix, bytes, round(priority,4), hits)</strong>
              <span>测试方向</span><strong>priority 列必须降序</strong>
              <span>展示取整</span><strong>只 round 输出，不改内部 priority</strong>
            </div>
          </section>
        </div>

        <div class="adv-flow">
          <span>capacity=128<br>已有 a=40,b=48</span>
          <span>新增 c=56<br>总需求 144</span>
          <span>堆弹低价值项<br>扣除它的 bytes</span>
          <strong>直到 current+56<br>不超过 128</strong>
        </div>

        <div class="adv-callout">单条缓存若比总容量还大，touch 会直接 ValueError；不要靠驱逐所有旧缓存去容纳一个本来就不可能放下的条目。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用书架容量练习先腾空间再新增", [
          "capacity = 10",
          "used = 7",
          "books = {'old': {'size': 7, 'score': 1.0}}",
          "new_size = 6",
          "",
          "while used + new_size > capacity and books:",
          "    name, item = min(books.items(), key=lambda pair: pair[1]['score'])",
          "    used -= item['size']",
          "    books.pop(name)",
          "",
          "books['new'] = {'size': new_size, 'score': 2.0}",
          "used += new_size",
          "ordered = sorted(",
          "    books.items(), key=lambda pair: (-pair[1]['score'], pair[0])",
          ")"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>used + new_size</code> -&gt; <code>current_bytes + needed</code> 的容量判断。</li>
            <li><code>books['new']</code> -&gt; TODO 4 创建 CacheEntry 并写入 entries。</li>
            <li><code>used += new_size</code> -&gt; Notebook 的 current_bytes 同步更新。</li>
            <li><code>-score</code> 排序 -&gt; TODO 5 的高 priority 在前 snapshot。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "capacity=128，current_bytes=88，现在要加入 56 bytes 的新缓存。",
        question: "驱逐循环至少要释放多少 bytes 才能停止？",
        options: ["16 bytes", "56 bytes", "88 bytes"],
        answer: 0,
        revealNote: "88+56=144，超过容量 16；实际会按完整 entry 驱逐，所以释放量可能大于 16。"
      },
      checkpoint: checkpoint(
        "新 CacheEntry 为什么先写 priority=0.0？",
        ["随后 _refresh_queue 会根据 hits、大小和时间统一计算真实分数", "因为所有新缓存永久最低优先级", "因为 heapq 只接受 0"],
        0,
        "初始化与评分职责分开，避免在 touch 中复制 _score 逻辑。"
      ),
      homework: [
        "完成 TODO 4-5，运行给定请求序列，确认日志同时出现 add、reuse:a 与至少一次 evict。",
        "容量/结构契约：任何时刻 current_bytes <= capacity_bytes；snapshot 每项长度为 4，priority 列按降序排列。",
        "边界测试：调用 touch('huge', capacity+1) 应触发 ValueError；这套模拟器存元数据，不涉及真实 KV tensor 的 dtype/device。",
        "错误诊断：容量统计不准查驱逐和新增时是否分别减/加 bytes；快照顺序反了查 -priority；新条目 KeyError 查是否先写 entries 再 refresh。"
      ]
    })
  ],

  "42": [
    lesson({
      id: "nccl-events-and-interval-overlap",
      title: "先把性能问题变成时间区间：记录事件并判断 overlap",
      todo: "TODO 1-3：记录 compute 字典、构造 CommEvent，并判断两个半开区间是否重叠",
      prerequisite: [
        "compute 事件用 dict 保存 name/start/end；通信事件用 dataclass CommEvent 保存 op/start/end/bytes。",
        "duration 属性返回 max(end-start,0.0)，可避免错误时间顺序产生负耗时。",
        "两个区间完全错开的条件是 end <= compute_start 或 start >= compute_end；对这个条件取 not 就得到重叠。",
        "边界刚好接触不算重叠，例如通信 end=3.0、计算 start=3.0 会满足 end <= start。"
      ],
      intuition: "Profiler 的第一步不是优化，而是把发生了什么记录成结构化事件。只要每条事件有类型、起止时间和字节数，就能把“通信似乎很慢”拆成可计算的问题；overlap 判断则回答这段通信有没有被任意计算区间覆盖。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>add_compute</b>保存名称与 [start,end)</span>
          <span><b>add_comm</b>构造 op/时间/bytes</span>
          <span><b>_has_overlap</b>遍历已有 compute</span>
          <span><b>event flag</b>写 overlap_with_compute</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>TODO 1-2 的数据契约</h4>
            <div class="adv-contract">
              <span>compute</span><strong><code>{"name":name,"start":start,"end":end}</code></strong>
              <span>communication</span><strong><code>CommEvent(op=op,start=start,end=end,bytes=bytes)</code></strong>
              <span>overlap 字段</span><strong>add_comm 时调用 _has_overlap 后写入</strong>
              <span>保存位置</span><strong>compute_events 与 events 两个列表分开</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>TODO 3 的反向思考</h4>
            <div class="adv-flow">
              <span>通信完全在左边<br>end &lt;= c.start</span>
              <span>或完全在右边<br>start &gt;= c.end</span>
              <strong>都不是<br>说明有交集</strong>
            </div>
            <p>代码是 <code>not (left_case or right_case)</code>，括号不能漏。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>compute</th><th>comm</th><th>结果</th><th>原因</th></tr></thead>
          <tbody>
            <tr><td>[0.0,2.0)</td><td>[1.0,2.5)</td><td>True</td><td>区间 1.0 到 2.0 相交</td></tr>
            <tr><td>[3.0,5.0)</td><td>[2.6,3.0)</td><td>False</td><td>只在 3.0 接触边界</td></tr>
            <tr><td>[3.0,5.0)</td><td>[3.5,4.3)</td><td>True</td><td>通信落在计算区间内部</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">overlap 标记只在 add_comm 当下根据已经记录的 compute 事件计算，不会因以后新增 compute 自动回溯更新；给定测试按正确顺序添加事件。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用会议日程练习区间是否冲突", [
          "meetings = [",
          "    {'name': 'design', 'start': 9.0, 'end': 10.5},",
          "    {'name': 'review', 'start': 14.0, 'end': 15.0},",
          "]",
          "",
          "candidate_start = 10.0",
          "candidate_end = 11.0",
          "has_conflict = False",
          "for meeting in meetings:",
          "    overlaps = not (",
          "        candidate_end <= meeting['start']",
          "        or candidate_start >= meeting['end']",
          "    )",
          "    if overlaps:",
          "        has_conflict = True",
          "        break"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>meetings</code> 字典 -&gt; TODO 1 的 compute event。</li>
            <li><code>candidate_start/end</code> -&gt; TODO 2 创建的 CommEvent 时间。</li>
            <li><code>overlaps</code> -&gt; TODO 3；字段名替换为 <code>c["start"]</code> 与 <code>c["end"]</code>。</li>
            <li><code>break</code> -&gt; Notebook 找到任一 overlap 后立即 return True。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "compute=[3.0,5.0)，comm=[2.6,3.0)。",
        question: "Notebook 的判断会把它标记成 overlap 吗？",
        options: ["不会，comm_end <= compute_start", "会，因为它们都包含 3.0", "会，因为 bytes 大于 0"],
        answer: 0,
        revealNote: "代码采用半开区间式边界，end 与 start 相等属于完全错开。"
      },
      checkpoint: checkpoint(
        "判断通信 [s,e) 与计算 [cs,ce) 重叠的正确表达式是哪一个？",
        ["not (e <= cs or s >= ce)", "e <= cs and s >= ce", "s < e"],
        0,
        "先写两种完全错开情况，用 or 合并，再整体取反。"
      ),
      homework: [
        "完成 TODO 1-3，并用给定三个通信事件手算 overlap：all_reduce=True、broadcast=False、reduce_scatter=True。",
        "字段契约：compute dict 含 name/start/end；CommEvent 含 op/start/end/bytes，duration 非负；本模拟器没有真实 GPU tensor、dtype 或 device。",
        "测试边界：额外加入一个 end 恰等于 compute start 的事件，应为 False；完全包含的事件应为 True。",
        "错误诊断：所有事件都 True 检查 not 的括号；broadcast 被标 True 检查 <=/>= 边界；后加 compute 不生效是当前 add_comm 时计算的设计。"
      ]
    }),

    lesson({
      id: "nccl-summary-buckets-and-timeline",
      title: "把事件变成结论：按 op 聚合，再输出可排序时间线",
      todo: "TODO 4-6：创建 by_op 聚合桶、累加 count/time/bytes，并导出 timeline 记录",
      prerequisite: [
        "dict.setdefault(key, default) 会在 key 不存在时写入 default，并无论如何返回该 key 对应的对象。",
        "item 是 by_op 中嵌套 dict 的引用；修改 item['count'] 会直接更新 by_op[e.op]。",
        "sorted(events,key=lambda x:(x.start,x.end,x.op)) 用开始时间、结束时间、op 名依次做稳定排序。",
        "overlap_ratio 使用 max(total_comm_time,1e-8) 防止没有通信事件时除零。"
      ],
      intuition: "timeline 回答“什么时候发生”，summary 回答“总体花在哪里”。同一份 CommEvent 既可以按时间排序，也可以按 op 分桶累加。初学者只要先建立一个固定的桶结构，再对每条事件做三次加法，就能得到最小可用的通信分析报告。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>TODO 4-5：聚合桶</h4>
            <div class="adv-contract">
              <span>桶 key</span><strong>通信 op，例如 all_reduce</strong>
              <span>初始值</span><strong><code>{"count":0,"time":0.0,"bytes":0}</code></strong>
              <span>每条事件</span><strong>count +1，time +duration，bytes +e.bytes</strong>
              <span>全局指标</span><strong>total、overlap_time、overlap_ratio</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>TODO 6：timeline 单条记录</h4>
            <div class="adv-contract">
              <span>身份</span><strong>op</strong>
              <span>位置</span><strong>start、end、duration</strong>
              <span>规模</span><strong>bytes</strong>
              <span>关系</span><strong>overlap=overlap_with_compute</strong>
            </div>
          </section>
        </div>

        <div class="adv-flow">
          <span>events<br>3 条 CommEvent</span>
          <span>按 op 分桶<br>count/time/bytes</span>
          <span>按 start 排序<br>生成 records</span>
          <strong>summary + timeline<br>两个观察视角</strong>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>给定测试事件</th><th>duration</th><th>overlap</th><th>进入哪个桶</th></tr></thead>
          <tbody>
            <tr><td>all_reduce 1.0-2.5</td><td>1.5</td><td>True</td><td>by_op["all_reduce"]</td></tr>
            <tr><td>broadcast 2.6-3.0</td><td>0.4</td><td>False</td><td>by_op["broadcast"]</td></tr>
            <tr><td>reduce_scatter 3.5-4.3</td><td>0.8</td><td>True</td><td>by_op["reduce_scatter"]</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">Notebook 的 overlap_time 是教学近似：只要事件有任何重叠，就把该通信事件的完整 duration 计入，不计算精确交集长度。不要自行改成另一套指标，否则测试含义会变化。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用快递记录练习 setdefault 聚合与时间排序", [
          "deliveries = [",
          "    {'kind': 'air', 'start': 3.0, 'end': 5.0, 'boxes': 4},",
          "    {'kind': 'ground', 'start': 1.0, 'end': 4.0, 'boxes': 7},",
          "    {'kind': 'air', 'start': 6.0, 'end': 7.0, 'boxes': 2},",
          "]",
          "",
          "by_kind = {}",
          "for event in deliveries:",
          "    bucket = by_kind.setdefault(",
          "        event['kind'], {'count': 0, 'time': 0.0, 'boxes': 0}",
          "    )",
          "    bucket['count'] += 1",
          "    bucket['time'] += event['end'] - event['start']",
          "    bucket['boxes'] += event['boxes']",
          "",
          "timeline = sorted(deliveries, key=lambda e: (e['start'], e['end']))"
        ])}
        <div class="adv-map">
          <h4>独立例子 -&gt; Notebook TODO</h4>
          <ul>
            <li><code>by_kind</code> -&gt; <code>by_op</code>；<code>event['kind']</code> -&gt; <code>e.op</code>。</li>
            <li><code>bucket</code> -&gt; TODO 4 的 <code>item</code>，默认字段改为 count/time/bytes。</li>
            <li>三次累加 -&gt; TODO 5；duration 使用 <code>e.duration</code> 属性。</li>
            <li><code>timeline</code> -&gt; TODO 6 外层排序；Notebook 还要把 dataclass 转成六字段 dict。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "all_reduce 有两条事件，duration 分别 1.2 与 0.8，bytes 分别 100 与 60。",
        question: "聚合桶应该得到什么？",
        options: ["count=2, time=2.0, bytes=160", "count=1, time=0.8, bytes=60", "count=2, time=0.96, bytes=40"],
        answer: 0,
        revealNote: "同 op 的事件逐项累加：次数 2、耗时 1.2+0.8、数据量 100+60。"
      },
      checkpoint: checkpoint(
        "setdefault 在当前 op 已存在时会做什么？",
        ["返回已有聚合桶，不覆盖其中已累计的数据", "每次都把桶清零", "返回 None"],
        0,
        "因此循环中的 item 始终指向 by_op[e.op] 的当前桶，可以继续累加。"
      ),
      homework: [
        "完成 TODO 4-6，确认 timeline 第一条是 all_reduce，第二条 broadcast 的 overlap 为 False。",
        "汇总契约：num_comm_events=3，total_comm_time>0，overlap_time>0，overlap_ratio 在 [0,1]；all_reduce count=1、bytes=128*1024。",
        "记录结构：timeline 每项包含 op/start/end/duration/bytes/overlap；这些是 Python 标量和 dict，不涉及 tensor shape/dtype/device。",
        "错误诊断：count 总为 1 查是否错误覆盖桶；bytes 不累加查 item 是否引用 by_op；时间线乱序查排序 key；ratio 超界查 overlap_time 是否只累计标记事件。"
      ]
    })
  ]
};
