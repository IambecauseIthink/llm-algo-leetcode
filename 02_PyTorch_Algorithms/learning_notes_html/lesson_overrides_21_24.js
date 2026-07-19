const { advancedStyles, checkpoint, code, lesson } = require("./advanced_lesson_helpers");

module.exports = {
  "21": [
    lesson({
      id: "decoding-temperature-pipeline",
      title: "先认清 logits：温度只改差距，不改候选数量",
      todo: "TODO 1：温度下限与缩放",
      prerequisite: [
        "logits 是模型给词表中每个 token 的原始分数，不是概率；本题输入示例的 shape 是 [1, vocab_size]。",
        "Softmax 才会把最后一维变成总和为 1 的概率。除以温度发生在 Softmax 之前。",
        "temperature 是 Python 浮点数。它不能为 0，否则 logits / temperature 会除零。",
        "decode_next_token 的顺序已经给好：Temperature -> Top-K -> Top-p -> Softmax -> multinomial。TODO 只补三个局部函数。"
      ],
      intuition: "先把每个 logit 想成候选 token 的赛前得分。温度 T 不删除任何选手，只统一缩放所有得分：T 小于 1 时，分数差被放大；T 大于 1 时，分数差被压小。TODO 1 还要给 T 设置一个极小的正数下限，让除法始终有定义。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>logits [B,V]</b>每行是一份词表得分</span>
          <span><b>logits / T</b>只改变同一行内的差距</span>
          <span><b>Softmax</b>稍后才转成概率</span>
          <span><b>multinomial</b>最后按概率抽一个 token</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>先解释变量</h4>
            <div class="adv-contract">
              <span><code>logits</code></span><strong>[batch, vocab_size] 的浮点张量；最后一维对应候选 token</strong>
              <span><code>temperature</code></span><strong>Python float；控制分数差距，T=1 表示不改变</strong>
              <span><code>temp</code></span><strong>加过正数下限后的安全除数</strong>
              <span>返回值</span><strong>shape、dtype、device 都应与 logits 一致</strong>
            </div>
          </section>
          <section class="adv-panel blue">
            <h4>再看差值</h4>
            <div class="adv-flow">
              <span>原得分 4.0</span><span>原得分 3.1</span><strong>差 0.9</strong>
            </div>
            <div class="adv-flow">
              <span>T = 0.5</span><span>两者都除以 0.5</span><strong>新差 1.8</strong>
            </div>
            <p>测试不要求背概率，它直接断言两个候选的差值变成原来的 2 倍。</p>
          </section>
        </div>

        <div class="adv-callout">常见误区：低温不是直接选最大 token，也不是把其他 token 设为零。它只让后续 Softmax 更偏向高分项；真正采样仍在流水线末尾。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("给传感器分数设置安全缩放下限", [
          "sensor_scores = torch.tensor([[1.5, 0.5, -0.5]])",
          "scale = 0.25",
          "safe_scale = max(scale, 1e-5)",
          "scaled_scores = sensor_scores / safe_scale",
          "",
          "print(sensor_scores.shape, scaled_scores.shape)",
          "print(sensor_scores.dtype, scaled_scores.dtype)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>sensor_scores</code> -> <code>logits</code>，都在最后一维保存一组待比较分数。</li>
            <li><code>scale</code> -> <code>temperature</code>，都是 Python 浮点除数。</li>
            <li><code>safe_scale</code> -> TODO 1 的 <code>temp</code>，用 <code>max</code> 设置正数下限。</li>
            <li><code>scaled_scores</code> -> 函数返回值，除法不改变 shape、dtype 或 device。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "同一行里两个 logits 分别是 4.0 和 3.1，现在使用 temperature=0.5。",
        question: "缩放后两者的差值会怎样？",
        options: [
          "从 0.9 变成 1.8",
          "从 0.9 变成 0.45",
          "shape 从 [1,V] 变成 [V,1]"
        ],
        answer: 0,
        revealNote: "两个值都除以 0.5，相当于都乘 2，所以它们的差也乘 2；shape 不变。"
      },
      checkpoint: checkpoint(
        "TODO 1 给 temperature 设置极小正数下限，主要防止什么？",
        ["除以 0", "词表被排序", "batch 维被删除"],
        0,
        "温度是除数。先取正数下限，才能让缩放在 temperature=0 一类输入上仍有定义。"
      ),
      homework: [
        "完成 Notebook TODO 1：先得到安全的 temp，再让 logits 做逐元素缩放；不要在这里调用 Softmax。",
        "核对输入与输出都是 [batch, vocab_size]，浮点 dtype 与 CPU/GPU device 均由 logits 继承；temp 本身是 Python float。",
        "运行 Temperature 断言：它检查 index 5 与 6 的差值是否翻倍。若失败，先排查是乘了 T、除错变量，还是提前做了 Softmax。"
      ]
    }),

    lesson({
      id: "decoding-top-k-threshold",
      title: "Top-K：先找第 K 大门槛，再用掩码保形状",
      todo: "TODO 2：Top-K 截断",
      prerequisite: [
        "Top-K 的 K 是固定保留数量。本题沿 logits 的最后一维筛选，每个 batch 行各算自己的门槛。",
        "torch.topk 返回 values 和 indices；TODO 只需要 values 中最小的那个 Top-K 值作为门槛。",
        "保留最后一维很重要：门槛最好是 [batch, 1]，这样能广播到 [batch, vocab_size]。",
        "把被过滤位置改成 -inf，而不是删除元素；这样输入输出 shape 完全相同，后续 Softmax 会给这些位置概率 0。"
      ],
      intuition: "Top-K 可以分成两件事：先为每一行找一条及格线，再把低于及格线的位置盖成 -inf。及格线就是前 K 个最大值里最小的那个。代码不移动 token，也不缩短词表，所以 token 的原索引仍然有效。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>每行独立找门槛</h4>
            <div class="adv-flow">
              <span>4.0</span><span>3.1</span><span>2.3</span><strong>第 3 大 = 2.3</strong>
            </div>
            <p>K=3 时，2.3 是这一行的门槛。不同 batch 行可能有不同门槛。</p>
          </section>
          <section class="adv-panel warn">
            <h4>低于门槛才过滤</h4>
            <div class="adv-flow">
              <span>大于门槛：保留</span><span>等于门槛：保留</span><strong>小于门槛：-inf</strong>
            </div>
            <p>Notebook 测试数据没有并列门槛，因此最终恰好保留 3 个位置。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>对象</th><th>shape</th><th>用途</th></tr></thead>
          <tbody>
            <tr><td><code>logits</code></td><td>[B,V]</td><td>原始词表得分，原索引不能丢</td></tr>
            <tr><td>Top-K values</td><td>[B,K]</td><td>每行最大的 K 个值</td></tr>
            <tr><td>第 K 大门槛</td><td>[B,1]</td><td>沿 V 维广播比较</td></tr>
            <tr><td>过滤结果</td><td>[B,V]</td><td>被过滤处为 -inf，其余值保持不变</td></tr>
          </tbody>
        </table>

        <div class="adv-checks">
          <span><b>边界 1</b><br>top_k &lt;= 0 时原样返回</span>
          <span><b>边界 2</b><br>top_k &gt;= V 时原样返回</span>
          <span><b>广播</b><br>[B,1] 对比 [B,V]</span>
          <span><b>设备</b><br>替换值要兼容 logits.device</span>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("每位评委只保留得分最高的两个方案", [
          "ratings = torch.tensor([[2.0, 7.0, 5.0, 1.0],",
          "                        [8.0, 3.0, 6.0, 4.0]])",
          "best_values, best_positions = torch.topk(ratings, k=2, dim=-1)",
          "cutoff = best_values[..., -1:]",
          "blocked = torch.tensor(float('-inf'), device=ratings.device)",
          "screened = torch.where(ratings < cutoff, blocked, ratings)",
          "",
          "print(cutoff.shape)   # [2, 1]",
          "print(screened.shape) # [2, 4]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>ratings</code> -> <code>logits</code>，每行独立筛选最后一维。</li>
            <li><code>best_values</code> -> <code>torch.topk</code> 返回的前 K 大值。</li>
            <li><code>cutoff</code> -> TODO 2 的第 K 大门槛；末尾切片保留长度为 1 的维度。</li>
            <li><code>screened</code> -> 过滤后的 logits；条件替换保持原 shape 和索引。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "一行得分是 [4.0, 1.0, 3.0, 2.0]，K=2，第 K 大门槛为 3.0。",
        question: "使用“得分小于门槛就设为 -inf”后，哪些位置保留？",
        options: [
          "值为 4.0 和 3.0 的位置",
          "值为 1.0 和 2.0 的位置",
          "只保留值为 4.0 的位置"
        ],
        answer: 0,
        revealNote: "Top-K 保留最大的两个值。比较条件是小于门槛才移除，所以等于 3.0 的位置也保留。"
      },
      checkpoint: checkpoint(
        "为什么第 K 大门槛常写成 shape [batch, 1]，而不是 [batch]？",
        ["便于沿 vocab 维广播到 [batch, vocab_size]", "为了把 dtype 变成整数", "为了打乱 token 索引"],
        0,
        "保留最后一维后，每行的一条门槛可以自然与该行所有词表分数比较。"
      ),
      homework: [
        "完成 Notebook TODO 2 的 filter_value、kth_values 与条件替换；保留函数顶部已经写好的 top_k 边界分支。",
        "检查 kth_values 从 [batch, top_k] 取末项后仍是 [batch, 1]；结果保持 logits 的浮点 dtype、[batch, vocab_size] shape 和原 device。",
        "运行 Top-K 测试：它统计非 -inf 的位置必须为 3。若数量不对，检查是否取成最大值、比较方向写反，或把等于门槛的位置也删掉。"
      ]
    }),

    lesson({
      id: "decoding-top-p-sort-restore",
      title: "Top-p：在排序坐标里累加，再回到 token 原坐标",
      todo: "TODO 3：Top-p 掩码、右移与 scatter 恢复",
      prerequisite: [
        "Top-p 保留的是累计概率达到阈值所需的最小候选集合，数量会随分布形状变化。",
        "代码先按 logits 降序排序，再对排序后的 logits 做 Softmax 和 cumsum；累加必须发生在概率上。",
        "第一个让累计概率超过 top_p 的 token 仍要保留，因此“超过阈值”的布尔掩码要向右平移一格。",
        "排序只用于决定删谁。最终还要依据 sorted_indices 把值散射回原 token 索引。"
      ],
      intuition: "Top-p 有两个坐标系：概率从大到小的“排名坐标”，以及词表原本的“token 坐标”。累计概率只能在排名坐标里计算；采样却必须回到 token 坐标。中间的右移掩码保证跨过阈值的那一项被收入候选集合。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>1. sort</b>按分数从大到小，并记住原索引</span>
          <span><b>2. softmax</b>把排序后的分数变成概率</span>
          <span><b>3. cumsum</b>计算累计概率并形成移除掩码</span>
          <span><b>4. scatter</b>把过滤结果放回原 token 索引</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>为什么要右移掩码</h4>
            <div class="adv-flow">
              <span>累计 0.54</span><span>累计 0.76</span><strong>累计 0.86，首次越过 0.8</strong><span>后续更大</span>
            </div>
            <p>直接用 cumulative &gt; 0.8 会删掉第三项。右移后第三项保留，从第四项开始删除。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>为什么要恢复顺序</h4>
            <div class="adv-contract">
              <span>排序位置 0</span><strong>可能来自原 token 5</strong>
              <span>排序位置 1</span><strong>可能来自原 token 6</strong>
              <span>排序位置 2</span><strong>可能来自原 token 1</strong>
              <span>恢复后</span><strong>保留值回到原词表位置 5、6、1</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>remove = cumulative_probs &gt; top_p</code><span>先标出首次越界项及后续项</span></div>
          <div><b>2</b><code>remove[..., 1:] 读取旧的 [..., :-1]</code><span>用 clone 避免重叠切片原地写造成数据污染</span></div>
          <div><b>3</b><code>remove[..., 0] = False</code><span>最高概率 token 永远至少保留一个</span></div>
          <div><b>4</b><code>scatter along dim=-1</code><span>按 sorted_indices 恢复到原词表顺序</span></div>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("按累计权重筛选商品，再恢复商品原编号", [
          "scores = torch.tensor([[0.2, 2.0, 1.0, 3.0]])",
          "ranked, original_pos = torch.sort(scores, dim=-1, descending=True)",
          "running = torch.cumsum(torch.softmax(ranked, dim=-1), dim=-1)",
          "drop = running > 0.75",
          "drop[..., 1:] = drop[..., :-1].clone()",
          "drop[..., 0] = False",
          "ranked = ranked.masked_fill(drop, float('-inf'))",
          "restored = torch.zeros_like(scores).scatter(-1, original_pos, ranked)",
          "print(restored.shape)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>ranked</code>、<code>original_pos</code> -> 已给出的 <code>sorted_logits</code>、<code>sorted_indices</code>。</li>
            <li><code>running</code> -> 已给出的 <code>cumulative_probs</code>，它是概率的累计和。</li>
            <li><code>drop</code> -> TODO 3 的 <code>sorted_indices_to_remove</code>，需要右移并保护首项。</li>
            <li><code>restored</code> -> TODO 3 的 <code>restored_logits</code>，沿最后一维按原索引散射。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "排序后累计概率是 [0.54, 0.76, 0.86, 0.94]，top_p=0.8。",
        question: "把“超过阈值”的掩码右移一格后，应保留多少个候选？",
        options: ["3 个", "2 个", "4 个"],
        answer: 0,
        revealNote: "第三个候选使累计概率首次越过 0.8，它仍属于达到阈值所必需的集合；从第四个开始移除。"
      },
      checkpoint: checkpoint(
        "Top-p 中 scatter 恢复原顺序的直接目的是什么？",
        ["让保留的分数重新对应正确 token_id", "把概率总和强制改成 0", "把 batch 维拼到 vocab 维"],
        0,
        "排序位置不是 token_id。按 sorted_indices 散射后，后续 multinomial 的列索引才仍是正确 token_id。"
      ),
      homework: [
        "完成 Notebook TODO 3：从 cumulative_probs 得到移除掩码，右移并保护第一项，再过滤 sorted_logits，最后恢复原索引顺序。",
        "逐步打印所有中间量，它们都应是 [batch, vocab_size]；布尔掩码 dtype 是 bool，其余张量保持 logits 的浮点 dtype 和 device。",
        "运行 Top-p 测试与完整管线测试：应保留 3 个非 -inf 值，next_token shape 为 [1,1]。若失败，检查是否忘了右移、原地右移时漏了 clone，或 scatter 的 dim/index/src 用反。"
      ]
    })
  ],

  "22": [
    lesson({
      id: "paged-prefill-allocation",
      title: "Prefill：先算要几块，再一次性检查够不够",
      todo: "当前练习 TODO 1 + TODO 2：向上取整并分配块；物理池初始化已给出",
      prerequisite: [
        "physical_kv_cache 已由 Notebook 初始化为 [num_blocks, block_size, head_dim]，当前练习不需要重写这行。",
        "req.seq_len 是当前请求已有的 token 数；req.block_table 是 List[int]，按逻辑顺序记录物理 block_id。",
        "free_blocks 是仍可分配的物理块编号列表。pop 会同时返回编号并把它从空闲列表移除。",
        "Prefill 要容纳全部 prompt，所以块数是 seq_len / block_size 的向上取整，而不是向下取整。"
      ],
      intuition: "把物理池想成固定规格的仓位。一个仓位能放 block_size 个 token，哪怕最后只多 1 个 token，也必须再占一个完整仓位。分配前先检查空闲块总数，避免分到一半才发现不够，留下半更新状态。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>物理池</b>[num_blocks, block_size, head_dim]</span>
          <span><b>需求量</b>ceil(seq_len / block_size)</span>
          <span><b>容量检查</b>free 数量必须不小于需求</span>
          <span><b>块表</b>按逻辑先后追加物理 ID</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>长度 6、每块 4 个 token</h4>
            <div class="adv-flow">
              <strong>逻辑 token 0..3</strong><strong>逻辑 token 4..5</strong>
            </div>
            <p>第一块装满 4 个，第二块装 2 个，因此 needed_blocks 是 2。</p>
          </section>
          <section class="adv-panel good">
            <h4>块表只存地址，不存 KV 数值</h4>
            <div class="adv-flow">
              <span>逻辑块 0</span><strong>物理块 ID 0</strong><span>逻辑块 1</span><strong>物理块 ID 1</strong>
            </div>
            <p>真实物理 ID 可以不连续；逻辑顺序由 block_table 的列表顺序表达。</p>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>needed_blocks</code></span><strong>Python int；必须覆盖 req.seq_len 个 token</strong>
          <span><code>free_blocks</code></span><strong>List[int]；分配后长度减少</strong>
          <span><code>req.block_table</code></span><strong>List[int]；分配后按顺序增加物理 ID</strong>
          <span>OOM 契约</span><strong>不足时抛 RuntimeError，且异常文本包含 OOM</strong>
        </div>

        <div class="adv-callout">当前 Notebook 的练习 TODO 编号跳过了 4：Prefill 是 TODO 1、2，Decode 是 TODO 3，缓存拼装标为 TODO 5。请以练习单元本身为准。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("给订单分配固定容量的货箱", [
          "item_count = 11",
          "box_capacity = 5",
          "boxes_needed = (item_count + box_capacity - 1) // box_capacity",
          "available_boxes = [20, 7, 31, 9]",
          "assigned_boxes = []",
          "",
          "if len(available_boxes) < boxes_needed:",
          "    raise RuntimeError('FULL')",
          "for _ in range(boxes_needed):",
          "    assigned_boxes.append(available_boxes.pop(0))"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>item_count</code> -> <code>req.seq_len</code>，表示必须容纳的逻辑元素数。</li>
            <li><code>box_capacity</code> -> <code>self.block_size</code>，表示每个固定块能容纳多少 token。</li>
            <li><code>boxes_needed</code> -> TODO 1 的 <code>needed_blocks</code>，使用整数向上取整。</li>
            <li><code>available_boxes</code> 与 <code>assigned_boxes</code> -> TODO 2 的 <code>self.free_blocks</code> 与 <code>req.block_table</code>；异常文本按 Notebook 要求改为 OOM。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prompt 长度是 6，每个物理块能放 4 个 token。",
        question: "Prefill 必须分配多少个块？",
        options: ["2 个", "1 个", "3 个"],
        answer: 0,
        revealNote: "6 // 4 虽然等于 1，但剩余 2 个 token 仍需要第二块，所以必须向上取整为 2。"
      },
      checkpoint: checkpoint(
        "为什么应在 pop 空闲块之前检查 free_blocks 是否足够？",
        ["避免 OOM 时只完成了一部分分配，留下半更新状态", "为了把 block_id 变成浮点数", "为了让 seq_len 自动归零"],
        0,
        "先完整检查容量，失败时可以直接抛错，不会先从池里拿走若干块再中断。"
      ),
      homework: [
        "完成当前练习 TODO 1 与 TODO 2：算 needed_blocks，先做容量检查，再把相应数量的 block_id 从 free_blocks 移入 req.block_table。",
        "类型账本：needed_blocks 和 block_id 是 Python int，两个表是 List[int]；physical_kv_cache 已是 [num_blocks, block_size, head_dim] 浮点张量，本 TODO 不改变其 dtype/device。",
        "运行 Prefill 与 OOM 测试：长度 6 应拿 2 块并剩 8 块；只有 1 块却要容纳 5 个 token 时必须抛 RuntimeError，消息含 OOM。常见错误是用向下取整或边分配边检查。"
      ]
    }),

    lesson({
      id: "paged-decode-boundary",
      title: "Decode：长度先加一，只在新块第一个位置补块",
      todo: "当前练习 TODO 3：判断跨块并按需追加一个物理块",
      prerequisite: [
        "allocate_for_decode 开头已经执行 req.seq_len += 1，TODO 3 看到的是加入新 token 之后的长度。",
        "一个 block 的逻辑位置可以看成 1 到 block_size；新块的第一个位置对应新长度除以 block_size 余 1。",
        "Decode 每次只生成一个 token，因此最多新增一个物理块，不需要 Prefill 那样一次循环分配多块。",
        "跨块时仍要处理空闲列表为空的 OOM 分支，并把新 block_id 追加到原 block_table 末尾。"
      ],
      intuition: "先盯状态变化，不要先背取模公式。block_size=4 时，长度 4 的最后一个位置还在第一块；加入新 token 后长度变成 5，它正好落在第二块第一个位置，此时才需要补块。长度 6、7、8 都继续使用第二块，长度 9 再补第三块。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>旧 seq_len</code><span>表示调用前已经缓存的 token 数</span></div>
          <div><b>2</b><code>seq_len += 1</code><span>Notebook 已经完成：把本轮新 token 计入长度</span></div>
          <div><b>3</b><code>判断新位置</code><span>是否刚进入一个新块的第一个槽位</span></div>
          <div><b>4</b><code>必要时 append</code><span>只取一个空闲物理块，接到块表末尾</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>加一后的 seq_len</th><th>block_size=4 时所在位置</th><th>是否新分配</th></tr></thead>
          <tbody>
            <tr><td>5</td><td>第 2 块的第 1 个槽位</td><td>是</td></tr>
            <tr><td>6</td><td>第 2 块的第 2 个槽位</td><td>否</td></tr>
            <tr><td>8</td><td>第 2 块的第 4 个槽位</td><td>否</td></tr>
            <tr><td>9</td><td>第 3 块的第 1 个槽位</td><td>是</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>容易写反的时刻</h4>
            <p>余数为 0 表示刚填满一块，不是已经进入下一块。因为本函数先加一，所以需要识别的是新块第一个位置。</p>
          </section>
          <section class="adv-panel good">
            <h4>测试中的两条边界</h4>
            <p>长度 6 变 7 不分块；之后变 8 仍不分，变 9 才新增。另一组测试直接检查 4 变 5 必须新增。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("向固定座位排中加入一位新观众", [
          "people = 8",
          "seats_per_row = 4",
          "free_rows = [12, 18]",
          "used_rows = [3, 9]",
          "",
          "people += 1",
          "starts_new_row = ((people - 1) % seats_per_row) == 0",
          "if starts_new_row:",
          "    if not free_rows:",
          "        raise RuntimeError('NO_ROW')",
          "    used_rows.append(free_rows.pop(0))"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>people</code> -> <code>req.seq_len</code>，都先增加 1，再判断新元素落在哪里。</li>
            <li><code>seats_per_row</code> -> <code>self.block_size</code>，是固定块容量。</li>
            <li><code>starts_new_row</code> -> TODO 3 的 <code>is_new_block_needed</code>；可等价改写为检查新长度对块大小的余数。</li>
            <li><code>free_rows</code>、<code>used_rows</code> -> <code>self.free_blocks</code>、<code>req.block_table</code>；Notebook 的耗尽异常要求为 OOM。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "block_size=4，调用 allocate_for_decode 前 req.seq_len=8；函数开头会先加 1。",
        question: "本次是否需要新物理块？",
        options: ["需要，新增后的长度 9 是新块第一个位置", "不需要，因为 8 能被 4 整除", "需要一次分配两个块"],
        answer: 0,
        revealNote: "判断使用的是加一后的长度 9。逻辑位置 1..4、5..8、9..12 分属三块，因此 9 需要第三块。"
      },
      checkpoint: checkpoint(
        "block_size=4，加一后的 seq_len=8 时应怎样？",
        ["继续使用当前块，不新增", "新增一块", "清空 block_table"],
        0,
        "长度 8 恰好填满第二块，但还没有 token 进入第三块；下一次变成 9 时才新增。"
      ),
      homework: [
        "完成当前练习 TODO 3：基于已经加一后的 req.seq_len 判断是否进入新块；只有需要时才检查 OOM、取一个 ID 并 append。",
        "这里没有新张量：seq_len、block_id 是 Python int，block_table/free_blocks 是 List[int]；不要改 physical_kv_cache 的 [num_blocks, block_size, head_dim]、dtype 或 device。",
        "运行 Decode 两组边界测试：6->7 不新增，8->9 新增；4->5 立即新增。若失败，先查是否在加一前判断、是否把余数 0 当成新块，或每步都错误分配。"
      ]
    }),

    lesson({
      id: "paged-cache-reassembly",
      title: "按块表取数：物理地址离散，逻辑 token 仍连续",
      todo: "当前练习 TODO 5：索引物理块、沿 token 维拼接并截断",
      prerequisite: [
        "physical_kv_cache 的 shape 是 [num_blocks, block_size, head_dim]。用一个 block_id 索引后得到 [block_size, head_dim]。",
        "req.block_table 的列表顺序就是逻辑块顺序，不能按物理 block_id 数值重新排序。",
        "torch.cat(..., dim=0) 会连接 token 槽位，得到 [已分配块数 * block_size, head_dim]。",
        "最后一块可能只使用了一部分，因此返回前必须按 req.seq_len 截去未使用槽位。"
      ],
      intuition: "块表像一张路线单：逻辑第 0 块可能在物理 7 号，逻辑第 1 块可能在物理 2 号。恢复时按路线单依次取出完整块，沿 token 维首尾相接，再把最后一块尚未使用的尾巴裁掉。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>block_table<br>[7, 2, 9]</span><span>物理块 7<br>[B,D]</span><span>物理块 2<br>[B,D]</span><span>物理块 9<br>[B,D]</span><strong>逻辑缓存<br>[seq_len,D]</strong>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>阶段</th><th>shape</th><th>block_size=4、head_dim=64、3 块、seq_len=9</th></tr></thead>
          <tbody>
            <tr><td>物理池</td><td>[N,B,D]</td><td>[10,4,64]</td></tr>
            <tr><td>单个选中块</td><td>[B,D]</td><td>[4,64]</td></tr>
            <tr><td>拼接后</td><td>[3B,D]</td><td>[12,64]</td></tr>
            <tr><td>按真实长度截断</td><td>[S,D]</td><td>[9,64]</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>测试怎样识别顺序</h4>
            <p>三个块分别填入 1.0、2.0、3.0；结果的 0:4、4:8、8:9 必须依次看到这些值。</p>
          </section>
          <section class="adv-panel warn">
            <h4>测试怎样识别截断</h4>
            <p>第三块虽然有 4 行槽位，seq_len=9 时只允许返回其中第 1 行；否则 shape 会错误地变成 [12,64]。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("按仓位清单恢复连续货物记录", [
          "warehouse = torch.arange(5 * 3 * 2).reshape(5, 3, 2)",
          "route = [3, 1]",
          "true_items = 5",
          "",
          "picked_shelves = [warehouse[shelf_id] for shelf_id in route]",
          "joined_items = torch.cat(picked_shelves, dim=0)",
          "visible_items = joined_items[:true_items]",
          "print(joined_items.shape, visible_items.shape)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>warehouse</code> -> <code>self.physical_kv_cache</code>，第一维是物理块编号。</li>
            <li><code>route</code> -> <code>req.block_table</code>，必须按列表顺序逐块索引。</li>
            <li><code>picked_shelves</code> -> TODO 5 的 <code>blocks</code>，每项 shape 为 [block_size, head_dim]。</li>
            <li><code>joined_items</code> -> <code>cat_blocks</code>；<code>true_items</code> -> <code>req.seq_len</code>，最后只返回真实长度。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "block_size=4、head_dim=8，块表中有 2 个物理块，真实 seq_len=5。",
        question: "先拼接再截断后的返回 shape 是什么？",
        options: ["[5,8]", "[8,8]", "[2,4,8]"],
        answer: 0,
        revealNote: "两个块先沿 token 维拼成 [8,8]，再按真实序列长度取前 5 行，得到 [5,8]。"
      },
      checkpoint: checkpoint(
        "恢复逻辑缓存时，为什么不能先对 req.block_table 的物理 ID 排序？",
        ["列表顺序表示逻辑 token 的先后，排序会打乱序列", "排序会把浮点张量变成布尔张量", "排序会自动释放物理块"],
        0,
        "物理编号只表示存放地址，编号大小不表示 token 时间顺序；逻辑顺序由块表本身记录。"
      ),
      homework: [
        "完成当前练习 TODO 5：按 req.block_table 逐块索引，沿 dim=0 拼接；保留函数最后已有的 [:req.seq_len] 截断。",
        "shape 账本必须能说清：池 [N,B,D] -> 单块 [B,D] -> 拼接 [块数*B,D] -> 返回 [seq_len,D]；dtype/device 全程继承 physical_kv_cache。",
        "运行两组拼装断言：分别得到 [9,64] 与 [5,8]，并核对每段填充值。若失败，检查 cat 是否用了 dim=0、块表顺序是否保留、是否漏掉真实长度截断。"
      ]
    })
  ],

  "23": [
    lesson({
      id: "speculative-probability-lookup",
      title: "先读对 p 和 q：每一行只验证草稿实际选中的 token",
      todo: "TODO 1 前置与直接接受分支：读取 [K,V] 中的对应概率并判断 p >= q",
      prerequisite: [
        "draft_probs 和 target_probs 都是 [K, vocab_size]：第 i 行表示草稿第 i 个位置的一整份词表概率。",
        "draft_tokens 是长度 K 的 Python token_id 列表；token_id 用来选中该行真正草拟出的那一列。",
        "二维索引 probs[i, token_id] 得到 0 维张量，.item() 把它转成 Python float。",
        "p 表示目标模型对当前草稿 token 的概率，q 表示草稿模型对同一 token 的概率。先解释两者，再使用接受公式。"
      ],
      intuition: "验证第 i 个草稿时，不比较两整行分布，也不取每行最大值。草稿已经选定 token_id，所以只需分别查询两个模型在这个位置、这个 token 上的概率。若目标模型给出的 p 不小于草稿模型的 q，接受概率已经封顶为 1，可以直接收下，不需要随机数。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>位置 i</b>当前验证草稿链中的第几个 token</span>
          <span><b>token_id</b>草稿模型在该位置实际选中的词</span>
          <span><b>p</b>target_probs[i, token_id]</span>
          <span><b>q</b>draft_probs[i, token_id]</span>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape / 类型</th><th>语义</th></tr></thead>
          <tbody>
            <tr><td><code>draft_probs</code></td><td>[K,V] 浮点张量</td><td>小模型在 K 个位置的词表分布</td></tr>
            <tr><td><code>target_probs</code></td><td>[K,V] 浮点张量</td><td>大模型在同样 K 个位置的词表分布</td></tr>
            <tr><td><code>draft_tokens</code></td><td>长度 K 的 List[int]</td><td>每个位置实际草拟出的 token_id</td></tr>
            <tr><td><code>p</code>、<code>q</code></td><td>Python float</td><td>两个模型对当前草稿 token 的概率</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>p >= q</h4>
            <div class="adv-flow"><span>p = 0.8</span><span>q = 0.5</span><strong>接受概率 = 1</strong></div>
            <p>直接把当前 token_id 加入 accepted_tokens。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>p &lt; q</h4>
            <div class="adv-flow"><span>p = 0.4</span><span>q = 0.5</span><strong>进入随机分支</strong></div>
            <p>这时才需要计算 p/q 并抽随机数；下一课再处理。</p>
          </section>
        </div>

        <div class="adv-callout">分支结构要互斥：直接接受之后不应再进入随机分支，否则同一个 token 可能被追加两次，测试提供的随机数调用次数也会错位。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("查询两位评审对已选方案的分数", [
          "junior_scores = torch.tensor([[0.2, 0.6, 0.2],",
          "                              [0.5, 0.1, 0.4]])",
          "senior_scores = torch.tensor([[0.3, 0.5, 0.2],",
          "                              [0.6, 0.1, 0.3]])",
          "chosen_options = [1, 0]",
          "",
          "row = 0",
          "option_id = chosen_options[row]",
          "senior_value = senior_scores[row, option_id].item()",
          "junior_value = junior_scores[row, option_id].item()",
          "certain = senior_value >= junior_value"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>row</code> -> 循环变量 <code>i</code>，表示当前草稿位置。</li>
            <li><code>chosen_options</code> -> <code>draft_tokens</code>，其中元素映射到 <code>token_id</code>。</li>
            <li><code>senior_value</code> -> <code>p</code>，<code>junior_value</code> -> <code>q</code>。</li>
            <li><code>certain</code> -> TODO 1 的直接接受判断；成立时只做一次追加，不抽随机数。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "当前 token 的目标概率 p=0.8，草稿概率 q=0.5。",
        question: "验证器应该走哪条路径？",
        options: ["直接接受，不调用 torch.rand", "以 0.625 的概率接受", "立即拒绝并停止"],
        answer: 0,
        revealNote: "p/q 大于 1，但概率不能超过 1，所以接受概率封顶为 1；代码可直接接受。"
      },
      checkpoint: checkpoint(
        "为什么索引要写成第 i 行与当前 token_id，而不是比较两整行概率？",
        ["验证对象是草稿在该位置实际选中的 token", "因为整行概率没有 dtype", "因为 token_id 表示 batch 大小"],
        0,
        "投机验证针对已经草拟出的具体 token，p 与 q 都必须是两个模型对同一个位置、同一个 token 的概率。"
      ),
      homework: [
        "完成 TODO 1 的直接接受分支，并为后续随机逻辑保留互斥分支结构；不要让 p>=q 的 token 再抽一次随机数。",
        "核对 draft_probs/target_probs 是 [K,vocab_size] 浮点张量且通常在同一 device；索引后 .item() 得到 Python float，draft_tokens 与 accepted_tokens 是 Python int 列表。",
        "对照测试第 0 个 token：p=0.8、q=0.5，必须接受 token 10。若结果重复或 mock_rand 提前消耗，检查直接分支后是否仍落入随机分支。"
      ]
    }),

    lesson({
      id: "speculative-ratio-coin",
      title: "p 小于 q：把比值变成一枚可复现的概率硬币",
      todo: "TODO 2 前半：生成 r，并用 r < p/q 决定接受",
      prerequisite: [
        "本阶段只在 p < q 时执行，因此 0 <= p/q < 1，可以作为接受概率。",
        "torch.rand(1) 返回 shape [1] 的均匀随机张量，取 .item() 后得到区间 [0,1) 的 Python float。",
        "判断 r < p/q 时，小随机数落在接受区间内；否则表示拒绝。",
        "Notebook 测试会临时替换 torch.rand，使第一次返回 0.5、第二次返回 0.9，所以分支调用次数必须准确。"
      ],
      intuition: "当草稿模型比目标模型更看好当前 token 时，不能总是接受。接受区间从 0 开始，长度是 p/q：比值越接近 1，硬币越容易通过；比值越小，越容易拒绝。随机数 r 就像在 0 到 1 的尺子上落一个点。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>测试中的第二个 token</h4>
            <div class="adv-steps">
              <div><b>1</b><code>p/q = 0.4/0.5 = 0.8</code><span>接受区间长度为 0.8</span></div>
              <div><b>2</b><code>r = 0.5</code><span>随机点落在接受区间</span></div>
              <div><b>3</b><code>0.5 &lt; 0.8</code><span>接受 token 20</span></div>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>测试中的第三个 token</h4>
            <div class="adv-steps">
              <div><b>1</b><code>p/q = 0.1/0.9</code><span>接受区间约为 0.11</span></div>
              <div><b>2</b><code>r = 0.9</code><span>随机点落在接受区间外</span></div>
              <div><b>3</b><code>0.9 不小于 0.11</code><span>拒绝当前 token</span></div>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>p/q</code></span><strong>Python float，当前分支中位于 0 到 1 之间</strong>
          <span><code>r</code></span><strong>Python float，由 torch.rand(1).item() 得到</strong>
          <span><code>r &lt; p/q</code></span><strong>bool；True 接受，False 拒绝</strong>
          <span>随机设备</span><strong>本 Notebook 测试在 CPU 上 monkeypatch torch.rand，按测试给出的调用形式即可</strong>
        </div>

        <div class="adv-callout">这里不要把比较写成 r &lt; q/p。概率校正的分子是目标模型概率 p，分母是产生草稿的模型概率 q。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("按合格率决定是否抽检通过", [
          "trusted_rate = 0.35",
          "claimed_rate = 0.50",
          "accept_chance = trusted_rate / claimed_rate",
          "draw = torch.rand(1).item()",
          "",
          "if draw < accept_chance:",
          "    decision = 'accept'",
          "else:",
          "    decision = 'reject'",
          "print(draw, accept_chance, decision)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>trusted_rate</code> -> 目标模型概率 <code>p</code>。</li>
            <li><code>claimed_rate</code> -> 草稿模型概率 <code>q</code>。</li>
            <li><code>accept_chance</code> -> TODO 2 中的 <code>p / q</code>。</li>
            <li><code>draw</code> -> TODO 2 的 <code>r</code>；accept 分支映射为追加当前 token_id。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "p=0.4、q=0.5，所以 p/q=0.8；测试给出的 r=0.5。",
        question: "当前草稿 token 是否接受？",
        options: ["接受，因为 0.5 < 0.8", "拒绝，因为 p < q", "直接接受且不应读取 r"],
        answer: 0,
        revealNote: "p<q 只说明进入随机分支，不等于必拒绝。随机数小于接受概率 0.8，因此通过。"
      },
      checkpoint: checkpoint(
        "在 p<q 分支中，哪个量是接受概率？",
        ["p/q", "q/p", "p+q"],
        0,
        "目标分布给当前 token 的质量 p，要除以草稿分布产生它的概率 q；当前分支下该比值小于 1。"
      ),
      homework: [
        "完成 TODO 2 的随机接受部分：只在 p<q 分支生成一次 r，并把 r 与 p/q 比较；接受时追加当前 token_id。",
        "shape/type 账本：torch.rand(1) 是 [1] 浮点 CPU 张量，.item() 后 r 是 Python float；p、q 也已是 float，accepted_tokens 仍是 List[int]。不要无故把输入 [K,V] 搬 device 或改 dtype。",
        "用测试数值手算两次：0.5 < 0.8 应接受 token 20，0.9 < 0.1/0.9 为假。若结果相反，检查比值方向、比较符号和 torch.rand 的调用位置。"
      ]
    }),

    lesson({
      id: "speculative-rejection-stop",
      title: "拒绝必须 break：后续草稿依赖已经失效的前缀",
      todo: "TODO 2 后半：拒绝时停止验证并满足 accepted == [10, 20]",
      prerequisite: [
        "for 循环中的 break 会立刻结束整个循环；continue 只跳过当前轮，下一轮仍会执行。",
        "草稿 token 30 之后的 token 40，是在包含 token 30 的前缀上生成的。若 30 被拒绝，40 的上下文前提已不成立。",
        "accepted_tokens 只记录从开头开始连续通过的草稿前缀，不是从 K 个位置里任意挑选的集合。",
        "测试只断言最终列表等于 [10, 20]；这同时检查直接接受、随机接受和拒绝后停止三段控制流。"
      ],
      intuition: "投机验证不是独立批改 K 道选择题，而是在验证一条连续路径。某一步拒绝，就像路线从这里断了；后面的草稿基于错误前缀，不能越过断点继续收集。因此拒绝分支的状态动作只有一个重点：立刻 break。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <strong>token 10<br>直接接受</strong><strong>token 20<br>随机接受</strong><span>token 30<br>随机拒绝</span><span>token 40<br>不再验证</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>如果误写 continue</h4>
            <p>循环会继续验证 token 40，可能得到非连续结果 [10,20,40]，还会多调用一次测试替换过的 torch.rand。</p>
          </section>
          <section class="adv-panel good">
            <h4>使用 break</h4>
            <p>拒绝 token 30 后立即结束，返回已经连续通过的前缀 [10,20]。</p>
          </section>
        </div>

        <div class="adv-contract">
          <span>循环状态</span><strong>i 从 0 向后推进，只要此前所有草稿都已接受</strong>
          <span>接受动作</span><strong>append 当前 token_id，然后进入下一轮</strong>
          <span>拒绝动作</span><strong>不 append，并 break</strong>
          <span>返回契约</span><strong>List[int]，内容是最长连续接受前缀</strong>
        </div>

        <div class="adv-checks">
          <span>第 0 位<br>不消耗随机数</span>
          <span>第 1 位<br>消耗 r=0.5</span>
          <span>第 2 位<br>消耗 r=0.9 后拒绝</span>
          <span>第 3 位<br>完全不执行</span>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("读取连续有效的批次编号", [
          "batch_ids = [101, 102, 103, 104]",
          "is_valid = [True, True, False, True]",
          "continuous_prefix = []",
          "",
          "for batch_id, ok in zip(batch_ids, is_valid):",
          "    if ok:",
          "        continuous_prefix.append(batch_id)",
          "    else:",
          "        break",
          "print(continuous_prefix)  # [101, 102]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>batch_ids</code> -> <code>draft_tokens</code>，顺序具有前缀依赖。</li>
            <li><code>ok</code> -> 当前 token 的直接接受或随机接受结果。</li>
            <li><code>continuous_prefix</code> -> <code>accepted_tokens</code>，只收集连续通过的开头部分。</li>
            <li><code>break</code> -> TODO 2 的拒绝动作；不能替换成 continue。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "接受状态依次为 [True, True, False, True]。",
        question: "投机验证应返回哪段 token？",
        options: ["只返回前两个连续接受的 token", "返回第 1、2、4 个 token", "返回空列表"],
        answer: 0,
        revealNote: "第三个 token 被拒绝后，第四个 token 的生成前缀已经失效，因此循环必须在第三个位置停止。"
      },
      checkpoint: checkpoint(
        "拒绝草稿 token 时，为什么不能用 continue？",
        ["后续草稿依赖被拒绝 token 所在的前缀，已不再有效", "continue 会把列表 dtype 改成 float", "continue 只能用于 while 循环"],
        0,
        "草稿是一条自回归路径。中间断点出现后，后续候选不再对应目标模型接受的上下文。"
      ),
      homework: [
        "补齐 TODO 2 的拒绝分支并使用 break，保证函数返回的是连续接受前缀；随后从头审视 TODO 1/2 是否形成完整互斥控制流。",
        "返回值必须是 List[int]，本步骤不创建新张量，不改变 draft_probs/target_probs 的 [K,V]、浮点 dtype 或 device；随机标量仍通过 .item() 进入 Python 控制流。",
        "运行唯一断言 accepted == [10,20]。若包含 30，检查概率比较；若包含 40，检查是否误用 continue；若随机序列错位，检查 p>=q 分支是否不必要地调用了 torch.rand。"
      ]
    })
  ],

  "24": [
    lesson({
      id: "radix-lcp-scan",
      title: "最长公共前缀：双边界向前走，第一次不同就停",
      todo: "TODO 1：_lcp_len 逐 token 计算最长公共前缀长度",
      prerequisite: [
        "cached_tokens 与 prompt_tokens 都是 Python token_id 列表，不是 PyTorch 张量。索引从 0 开始。",
        "match_len 同时表示“已匹配数量”和“下一次要比较的索引”。初始为 0。",
        "循环继续前必须同时保证 match_len 小于两个列表长度，否则较短列表会越界。",
        "一旦同一位置 token 不相等，公共前缀已经结束，后面即使再次相等也不能计入。"
      ],
      intuition: "把两列 token 从左端对齐，用一根指针一起向右走。相同就把 match_len 加一；不同就立刻停止。公式先放在语义之后：最终 match_len 最大只能到 min(len(cached_tokens), len(prompt_tokens))。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>遇到不同就停</h4>
            <div class="adv-flow">
              <strong>1 = 1<br>长度 1</strong><strong>2 = 2<br>长度 2</strong><span>3 != 4<br>停止</span><span>后面不看</span>
            </div>
            <p>[1,2,3] 与 [1,2,4] 的 LCP 长度是 2。</p>
          </section>
          <section class="adv-panel good">
            <h4>较短列表先结束也要停</h4>
            <div class="adv-flow">
              <strong>7 = 7</strong><strong>8 = 8</strong><span>缓存已结束</span>
            </div>
            <p>[7,8] 是 [7,8,9,10] 的完整前缀，所以长度是 2，不是 4。</p>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>cached_tokens</code></span><strong>List[int]，某条已缓存路径</strong>
          <span><code>prompt_tokens</code></span><strong>List[int]，新请求的 token 序列</strong>
          <span><code>match_len</code></span><strong>Python int，既是下一索引，也是已匹配 token 数</strong>
          <span>device / dtype</span><strong>本 TODO 使用 Python 列表，没有张量 dtype 或 device</strong>
        </div>

        <div class="adv-checks">
          <span>先查缓存边界</span><span>再查 prompt 边界</span><span>相等才加一</span><span>不等立即 break</span>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("比较两条目录路径的公共开头", [
          "saved_path = ['home', 'team', 'docs']",
          "new_path = ['home', 'team', 'images', 'raw']",
          "common = 0",
          "",
          "while common < len(saved_path) and common < len(new_path):",
          "    if saved_path[common] != new_path[common]:",
          "        break",
          "    common += 1",
          "print(common)  # 2"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>saved_path</code> -> <code>cached_tokens</code>。</li>
            <li><code>new_path</code> -> <code>prompt_tokens</code>。</li>
            <li><code>common</code> -> TODO 1 的 <code>match_len</code>，同时充当索引与计数。</li>
            <li>双长度条件与遇到不同就 break -> TODO 1 的安全扫描控制流。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "缓存路径是 [7,8]，新 prompt 是 [7,8,9,10]。",
        question: "最长公共前缀长度是多少？",
        options: ["2", "4", "0"],
        answer: 0,
        revealNote: "较短列表的两个 token 都匹配后，它已经结束；公共前缀长度受较短长度限制。"
      },
      checkpoint: checkpoint(
        "为什么 while 条件必须同时检查两个列表的长度？",
        ["任一列表先结束都应停止，避免索引越界", "为了把列表转成张量", "为了让 token 自动排序"],
        0,
        "比较需要读取两个列表的同一索引，只要一边没有该位置，就不能继续。"
      ),
      homework: [
        "完成 Notebook TODO 1：用 match_len 从 0 开始逐位比较，相等时增长，遇到不等或任一列表结束时停止。",
        "类型账本：cached_tokens/prompt_tokens 是 List[int]，match_len 是 Python int；本函数没有 tensor shape、dtype、device，切勿加入不必要的 torch 操作。",
        "运行两条 LCP 断言：[1,2,3] 对 [1,2,4] 得 2，[7,8] 对更长列表也得 2。若 IndexError，检查双边界；若得到 3 或 4，检查是否在不等时 break。"
      ]
    }),

    lesson({
      id: "radix-best-candidate",
      title: "多条缓存路径：每条算局部命中，全局只保留最大值",
      todo: "TODO 2：遍历 self.root.children 并更新 best_match_len",
      prerequisite: [
        "这个教学版 insert 会把每条完整 tokens 直接挂到 root.children；它是单层暴力匹配，不要求实现真正的树分裂。",
        "每个 child 是 TreeNode，候选缓存路径保存在 child.key_tokens。",
        "_lcp_len 已负责计算一条候选与 prompt_tokens 的局部 match_len。",
        "best_match_len 是跨候选的运行最大值，初始为 0；没有 child 或全部首 token 不同都会保持 0。"
      ],
      intuition: "TODO 2 是一个标准“遍历并取最大值”的归约过程。对每个缓存 child，先问它能从开头匹配几位，再把答案与当前冠军 best_match_len 比较。最终返回的不是某个节点，而是可复用 token 的最大数量。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>候选 [0,1,2,3]<br>LCP=4</span><strong>候选 [0,1,2,3,4]<br>LCP=5</strong><span>候选 [9,9,9]<br>LCP=0</span><strong>best=5</strong>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>局部问题</h4>
            <p>当前 child.key_tokens 与 prompt_tokens 共享多少个开头 token？交给 _lcp_len。</p>
          </section>
          <section class="adv-panel good">
            <h4>全局问题</h4>
            <p>所有局部 match_len 中最大的是多少？持续更新 best_match_len。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>best_match_len = 0</code><span>无命中时的正确默认值</span></div>
          <div><b>2</b><code>遍历 root.children</code><span>本 Notebook 只检查根节点下一层</span></div>
          <div><b>3</b><code>调用 _lcp_len</code><span>得到当前候选的局部命中长度</span></div>
          <div><b>4</b><code>保留更大值</code><span>短候选不能覆盖此前的更长命中</span></div>
        </div>

        <div class="adv-callout">不要在看到第一个正数命中时就 return。测试同时插入长度 4 和长度 5 的相同前缀路径，要求选择后者的 5。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("从多个候选单词中找最长公共开头", [
          "candidates = ['inter', 'internet', 'internal']",
          "query = 'internet cafe'",
          "best = 0",
          "",
          "for candidate in candidates:",
          "    local = 0",
          "    while local < len(candidate) and candidate[local] == query[local]:",
          "        local += 1",
          "    if local > best:",
          "        best = local",
          "print(best)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>candidates</code> -> <code>self.root.children</code>，每个元素是一条缓存候选。</li>
            <li><code>candidate</code> -> <code>child.key_tokens</code>。</li>
            <li><code>query</code> -> <code>prompt_tokens</code>。</li>
            <li><code>local</code> -> 调用 <code>self._lcp_len</code> 得到的 match_len；<code>best</code> -> TODO 2 的 <code>best_match_len</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "三个候选路径对新 prompt 的 LCP 长度依次是 4、5、0。",
        question: "match_prefix 最终应返回什么？",
        options: ["5", "4", "0"],
        answer: 0,
        revealNote: "函数寻找所有缓存路径中的最长命中，因此运行最大值依次变成 4、5，并保持 5。"
      },
      checkpoint: checkpoint(
        "没有任何缓存路径首 token 与 prompt 相同时，best_match_len 应是多少？",
        ["0", "-1", "prompt 的完整长度"],
        0,
        "0 表示没有任何可复用 token，也是初始化值；测试明确检查无命中返回 0。"
      ),
      homework: [
        "完成 Notebook TODO 2：遍历 self.root.children，对 child.key_tokens 调用 _lcp_len，并只在局部长度更大时更新 best_match_len。",
        "这里仍是 Python 数据：children 是 TreeNode 列表，key_tokens 是 List[int]，两个长度是 int；无 tensor shape、dtype、device，也不要实现题目未要求的递归树。",
        "运行多候选断言：对 [0,1,2,3,4,5] 必须命中 5，对 [7,6,5] 必须为 0。若只得 4，检查是否过早 return；若错误命中，检查是否比较前缀而非任意子序列。"
      ]
    }),

    lesson({
      id: "radix-split-hit-miss",
      title: "把命中长度落到数据流：前缀复用，后缀重算",
      todo: "TODO 3：调用 match_prefix，并按 hit_len 切出 hit_prefix 与 miss_suffix",
      prerequisite: [
        "match_prefix(prompt_tokens) 返回 Python int hit_len，表示从开头连续可复用的 token 数。",
        "Python 切片 prompt_tokens[:hit_len] 取得前 hit_len 个元素；prompt_tokens[hit_len:] 取得余下元素。",
        "切片不会修改原列表，而是产生新列表；返回顺序由函数签名固定为 hit_prefix、miss_suffix、hit_len。",
        "hit_len=0 是自然回退：[:0] 得空列表，[0:] 得完整 prompt，不需要另写特殊分支。"
      ],
      intuition: "前两课只算出了一个数字，这一课把数字变成真正的工作边界。切片点左边的 token 已有 KV Cache，可以复用；切片点右边的 token 没命中，需要模型重新计算。命中为 0 时，同一套切片自动退化为“无前缀、全量重算”。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>prompt</b>[0,1,2,3,4,5]</span>
          <span><b>hit_len</b>最长命中为 5</span>
          <span><b>hit_prefix</b>[0,1,2,3,4] 可复用</span>
          <span><b>miss_suffix</b>[5] 待重算</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>有命中</h4>
            <div class="adv-flow"><strong>前 5 个 token<br>复用缓存</strong><span>最后 1 个 token<br>重新计算</span></div>
          </section>
          <section class="adv-panel neutral">
            <h4>无命中</h4>
            <div class="adv-flow"><span>空前缀 []</span><strong>完整 prompt<br>全部重算</strong></div>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>hit_len</code></span><strong>int，满足 0 &lt;= hit_len &lt;= len(prompt_tokens)</strong>
          <span><code>hit_prefix</code></span><strong>List[int]，长度等于 hit_len</strong>
          <span><code>miss_suffix</code></span><strong>List[int]，长度等于 len(prompt_tokens)-hit_len</strong>
          <span>拼回检查</span><strong>hit_prefix + miss_suffix 必须等于原 prompt_tokens</strong>
        </div>

        <div class="adv-checks">
          <span>返回顺序正确</span><span>前缀长度等于 hit_len</span><span>后缀从 hit_len 开始</span><span>无命中自然回退</span>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("按书签把章节拆成已读和未读", [
          "chapters = ['intro', 'tensor', 'cache', 'serving']",
          "bookmark = 2",
          "read_part = chapters[:bookmark]",
          "unread_part = chapters[bookmark:]",
          "",
          "assert len(read_part) == bookmark",
          "assert read_part + unread_part == chapters",
          "print(read_part, unread_part)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -> Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>chapters</code> -> <code>prompt_tokens</code>。</li>
            <li><code>bookmark</code> -> TODO 3 中调用 <code>match_prefix</code> 得到的 <code>hit_len</code>。</li>
            <li><code>read_part</code> -> <code>hit_prefix</code>，使用从开头到切片点的切片。</li>
            <li><code>unread_part</code> -> <code>miss_suffix</code>，使用从切片点到末尾的切片。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prompt_tokens=[7,6,5]，match_prefix 返回 hit_len=0。",
        question: "两次切片会得到什么？",
        options: ["hit_prefix=[]，miss_suffix=[7,6,5]", "hit_prefix=[7]，miss_suffix=[6,5]", "两个列表都为空"],
        answer: 0,
        revealNote: "[:0] 是空列表，[0:] 从开头取到末尾，因此无需为无命中另写分支。"
      },
      checkpoint: checkpoint(
        "哪个关系能同时检查切分没有丢 token、也没有改变顺序？",
        ["hit_prefix + miss_suffix == prompt_tokens", "len(hit_prefix) > len(prompt_tokens)", "hit_len == -1"],
        0,
        "把两段按原顺序拼回去应精确得到原 prompt；同时前缀长度应等于 hit_len。"
      ),
      homework: [
        "完成 Notebook TODO 3：先调用 match_prefix 得 hit_len，再以同一个切片点得到 hit_prefix 和 miss_suffix，按函数已有顺序返回三者。",
        "三个序列都是 List[int]，hit_len 是 int；本 TODO 没有 tensor shape、dtype、device。额外自检 len(hit_prefix)==hit_len 且两段拼回原 prompt。",
        "运行有命中和无命中全部断言：前者应为 [0,1,2,3,4]、[5]、5，后者应为 []、[7,6,5]、0。若错位，检查切片冒号方向或返回元组顺序。"
      ]
    })
  ]
};
