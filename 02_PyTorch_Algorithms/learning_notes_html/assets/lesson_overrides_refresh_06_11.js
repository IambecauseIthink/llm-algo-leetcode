const {
  advancedStyles,
  checkpoint,
  code,
  lesson
} = require("../advanced_lesson_helpers");

module.exports = {
  "06": [
    lesson({
      id: "router-global-probabilities",
      title: "先让每个 token 给所有专家分配概率",
      todo: "TODO 1：routing_probs",
      prerequisite: [
        "输入 hidden_states 的 shape 是 [batch_size, seq_len, hidden_size]。B 是句子数，S 是每句 token 数，H 是每个 token 的特征数。",
        "view(-1, hidden_size) 把 B 和 S 合并成 token 总数 T=B×S；它不改变数据，只把 [B,S,H] 看成 [T,H]。",
        "gate 是 nn.Linear(hidden_size, num_experts, bias=False)，所以每个 token 的 H 个特征会变成 E=num_experts 个打分。",
        "logit 是未归一化分数，可以为负，也不要求总和为 1；softmax 才把一行分数变成概率分布。"
      ],
      intuition: "把每个 token 想成一张问诊单，E 位专家都先给它一个匹配分。Router 要沿专家这一维把分数变成概率，因此同一 token 的 E 个概率相加为 1。Notebook 还先把 logits 转成 FP32，因为 softmax 里的指数运算对低精度更敏感。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>[B,S,H]</b>句子中的 token 特征</span>
          <span><b>[T,H]</b>T=B×S，排成 token 队列</span>
          <span><b>[T,E]</b>gate 给每位专家打分</span>
          <span><b>[T,E]</b>每行变成概率分布</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>先认清最后一维</h4>
            <div class="adv-contract">
              <span><code>hidden_states</code></span><strong>[2,4,16]，共 8 个 token</strong>
              <span><code>router_logits</code></span><strong>[8,8]，每行对应 8 位专家</strong>
              <span><code>dim=-1</code></span><strong>在每行的专家列表内归一化</strong>
              <span>输出 dtype</span><strong>TODO 先用 FP32，返回前再恢复输入 dtype</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>一行只属于一个 token</h4>
            <div class="adv-flow">
              <span>logits<br>[-1, 2, 0, 1]</span>
              <span>转 FP32</span>
              <span>softmax<br>全部为正</span>
              <strong>一行总和 = 1</strong>
            </div>
            <p>不能沿 token 维做 softmax，否则不同 token 会互相争概率，Router 就失去“每个 token 自己选专家”的含义。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape</th><th>最后一维表示什么</th><th>本步是否改变 shape</th></tr></thead>
          <tbody>
            <tr><td><code>hidden_states</code></td><td><code>[T,H]</code></td><td>隐藏特征</td><td>已在前面展平</td></tr>
            <tr><td><code>router_logits</code></td><td><code>[T,E]</code></td><td>专家原始分数</td><td>gate 把 H 改为 E</td></tr>
            <tr><td><code>routing_probs</code></td><td><code>[T,E]</code></td><td>专家全局概率</td><td>softmax 不改 shape</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">这里“全局 Softmax”是指对一个 token 的全部 E 位专家归一化，不是把 batch 中所有 token 混在一起归一化。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("给每位学生的所有兴趣方向分配概率", [
          "scores = torch.tensor([",
          "    [1.0, 3.0, -1.0],",
          "    [2.0, 0.0, 1.0],",
          "])",
          "",
          "# 每一行是一位学生，最后一维是 3 个方向",
          "probabilities = F.softmax(scores.float(), dim=-1)",
          "print(probabilities.shape)       # [2, 3]",
          "print(probabilities.sum(dim=-1)) # [1, 1]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>scores</code> 对应 gate 产出的 <code>router_logits</code>。</li>
            <li><code>3 个方向</code> 对应 <code>num_experts</code> 位专家。</li>
            <li><code>probabilities</code> 对应 TODO 1 要创建的 <code>routing_probs</code>。</li>
            <li>保留 <code>.float()</code> 和 <code>dim=-1</code>：前者提高 softmax 稳定性，后者明确沿专家维计算。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "router_logits.shape=[8,8]，其中第 0 维是 token，第 1 维是专家。",
        question: "softmax 应该沿哪个维度计算，才能让每个 token 独立给所有专家分配概率？",
        options: ["dim=-1", "dim=0", "先把整个张量 flatten 成一维"],
        answer: 0,
        revealNote: "最后一维是专家维。沿 dim=-1 后，每一行 8 个专家概率之和为 1。"
      },
      checkpoint: checkpoint(
        "输入是 [2,4,16]，num_experts=8。展平、gate、softmax 后 routing_probs 的 shape 是什么？",
        ["[8,8]", "[2,4,8]", "[8,16]"],
        0,
        "Notebook 在 gate 前已经把 B×S 合并成 8 个 token；gate 输出每个 token 对 8 位专家的分数，所以是 [8,8]。"
      ),
      homework: [
        "完成 TODO 1：只写出从 router_logits 得到 routing_probs 的表达式，不要在这一步提前截取专家。",
        "shape 自检：routing_probs 与 router_logits 都应为 [batch_size*seq_len, num_experts]。",
        "数值自检：临时查看 routing_probs.sum(dim=-1)，每个 token 都应接近 1。",
        "错误诊断：若不同 token 被绑在一起分配概率，检查 softmax 是否误用了 dim=0；若低精度出现异常，检查是否漏掉 router_logits.float()。"
      ]
    }),

    lesson({
      id: "router-topk-values-indices",
      title: "再用 topk 同时取出权重和专家编号",
      todo: "TODO 2：routing_weights 与 selected_experts",
      prerequisite: [
        "torch.topk(input, k, dim) 会返回两个 Tensor：values 是最大的 k 个值，indices 是这些值原来所在的下标。",
        "TODO 2 的输入是完整概率表 routing_probs，不是原始 hidden_states，也不是已经截断的 logits。",
        "每个 token 只保留 K 位专家，因此最后一维会从 E 变成 K。",
        "权重是浮点 Tensor；专家编号是整数 Tensor，后续 torch.where 会用编号决定 token 去哪个 expert。"
      ],
      intuition: "Top-K 不只回答“最高的两个概率是多少”，还必须回答“这两个概率属于哪两位专家”。values 决定加权比例，indices 决定分发位置；少拿任何一张表，后面的 SparseMoEBlock 都无法工作。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>同一个操作返回两种信息</h4>
            <div class="adv-contract">
              <span>输入一行</span><strong>[0.05, 0.62, 0.08, 0.25]</strong>
              <span><code>values</code></span><strong>[0.62, 0.25]，后续叫 routing_weights</strong>
              <span><code>indices</code></span><strong>[1, 3]，后续叫 selected_experts</strong>
            </div>
          </section>
          <section class="adv-panel blue">
            <h4>shape 只缩短专家维</h4>
            <div class="adv-flow">
              <span>routing_probs<br>[T,E]</span>
              <span>每行选 K 个</span>
              <strong>两个输出<br>[T,K]</strong>
            </div>
            <p>T 个 token 都保留；每个 token 的专家候选从 E 个缩到 K 个。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>input = routing_probs</code><span>使用 TODO 1 的完整专家概率表</span></div>
          <div><b>2</b><code>k = self.top_k</code><span>不要把 Notebook 测试里的 2 硬编码进模块</span></div>
          <div><b>3</b><code>dim = -1</code><span>每个 token 在自己的专家维中选择</span></div>
          <div><b>4</b><code>values, indices = topk(...)</code><span>按 forward 的返回合同给两个结果命名</span></div>
        </div>

        <div class="adv-callout">不要只写 <code>selected_experts = torch.topk(...)</code>。topk 返回的是具名二元结果；本题需要把“值”和“下标”分别接住。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("从每场比赛中选出前两名及其赛道号", [
          "race_scores = torch.tensor([",
          "    [0.10, 0.55, 0.20, 0.15],",
          "    [0.40, 0.05, 0.35, 0.20],",
          "])",
          "",
          "top_scores, lane_ids = torch.topk(",
          "    race_scores, k=2, dim=-1",
          ")",
          "print(top_scores) # 成绩值",
          "print(lane_ids)   # 原来的列下标"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>race_scores</code> 对应 <code>routing_probs</code>。</li>
            <li><code>top_scores</code> 对应浮点的 <code>routing_weights</code>。</li>
            <li><code>lane_ids</code> 对应整数的 <code>selected_experts</code>。</li>
            <li><code>k=2</code> 要改成模块配置 <code>self.top_k</code>，这样测试换 K 时仍能工作。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某 token 的概率是 [0.10, 0.55, 0.20, 0.15]，top_k=2。",
        question: "torch.topk 返回的 values 与 indices 分别是什么？",
        options: ["[0.55,0.20] 与 [1,2]", "[1,2] 与 [0.55,0.20]", "[0.55,0.20] 与 [0,1]"],
        answer: 0,
        revealNote: "values 保留最大的概率值，indices 保留这些值在原专家维中的列号。"
      },
      checkpoint: checkpoint(
        "为什么 selected_experts 必须保留整数索引，而不能只返回 routing_weights？",
        ["后面的分发逻辑要用索引找到每个 token 应执行的专家", "整数索引会让 softmax 更准确", "权重不能参与乘法"],
        0,
        "权重只说明贡献大小；没有专家编号，SparseMoEBlock 不知道该调用哪一个 nn.Linear 专家。"
      ),
      homework: [
        "完成 TODO 2：从 routing_probs 沿最后一维取 self.top_k 个最大概率，同时接住 values 和 indices。",
        "shape/dtype 自检：两个输出都是 [T,K]；routing_weights 是浮点，selected_experts 是整数。",
        "边界自检：selected_experts 的所有值都必须在 [0, num_experts) 内。",
        "错误诊断：若后续 torch.where 比较异常，检查是否把 values 和 indices 的接收顺序写反。"
      ]
    }),

    lesson({
      id: "router-renormalize-and-dispatch",
      title: "最后重归一化，并读懂专家输出如何汇合",
      todo: "TODO 3：Top-K 权重重归一化",
      prerequisite: [
        "全量 routing_probs 每行和为 1，但丢掉未入选专家后，Top-K 权重之和通常小于 1。",
        "sum(dim=-1, keepdim=True) 会把 [T,K] 的每行求和为 [T,1]；保留长度为 1 的维度后可以广播除回 K 个权重。",
        "返回前 routing_weights.to(hidden_states.dtype) 会恢复输入精度；selected_experts 仍保持整数类型。",
        "SparseMoEBlock 已经写好：torch.where 找到选择当前专家的 token，专家计算后乘对应权重，再用 += 汇总多位专家的贡献。"
      ],
      intuition: "只留下两位专家后，他们原来的概率可能合计只有 0.8。重归一化就是在入选者内部重新分配 100% 的话语权，同时保持 0.6:0.2 的相对比例不变。这样不同 token 的专家混合尺度一致。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>截取后还不是完整混合权重</h4>
            <div class="adv-flow">
              <span>Top-2<br>[0.60,0.20]</span>
              <span>行和<br>0.80</span>
              <span>分别除 0.80</span>
              <strong>[0.75,0.25]<br>和为 1</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>keepdim 让广播意图清楚</h4>
            <div class="adv-contract">
              <span>权重</span><strong>[T,K]</strong>
              <span>不 keepdim 的和</span><strong>[T]，不能按预期对齐最后一维</strong>
              <span>keepdim 的和</span><strong>[T,1]，一行的和广播给本行 K 个值</strong>
            </div>
          </section>
        </div>

        <div class="adv-roadmap">
          <span><b>selected_experts</b>决定 token 去哪里</span>
          <span><b>torch.where</b>找 token 行和 Top-K 槽位</span>
          <span><b>unsqueeze(-1)</b>把 [n] 权重变 [n,1]</span>
          <span><b>+=</b>把 K 位专家贡献累加</span>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>测试检查</th><th>期望</th><th>失败时优先检查</th></tr></thead>
          <tbody>
            <tr><td>Router 权重</td><td><code>[B*S,K]</code></td><td>topk 的 dim 和 k</td></tr>
            <tr><td>每行权重和</td><td>接近 1</td><td>TODO 3 的分母与 keepdim</td></tr>
            <tr><td>最终 MoE 输出</td><td>恢复为 <code>[B,S,H]</code></td><td>聚合是否保留 token 与 hidden 维</td></tr>
          </tbody>
        </table>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把入选评委的票数重新归一化", [
          "chosen_votes = torch.tensor([",
          "    [0.60, 0.20],",
          "    [0.30, 0.15],",
          "])",
          "",
          "vote_total = chosen_votes.sum(dim=-1, keepdim=True)",
          "normalized_votes = chosen_votes / vote_total",
          "print(vote_total.shape)                 # [2, 1]",
          "print(normalized_votes.sum(dim=-1))     # [1, 1]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>chosen_votes</code> 对应 TODO 2 产出的 <code>routing_weights</code>。</li>
            <li><code>vote_total</code> 对应每个 token 的 Top-K 概率和。</li>
            <li>TODO 3 可以把求和直接写进除法，但必须保留 <code>dim=-1, keepdim=True</code>。</li>
            <li>完成后不要删掉 Notebook 已给出的 dtype 恢复与返回语句。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "routing_weights.shape=[8,2]，要让每行两个权重重新相加为 1。",
        question: "分母最合适的 shape 是什么？",
        options: ["[8,1]", "[8]", "[1,2]"],
        answer: 0,
        revealNote: "[8,1] 表示每个 token 一个行和，可以沿最后一维广播给该 token 的两个权重。"
      },
      checkpoint: checkpoint(
        "为什么专家聚合使用 final_hidden_states[token_idx] += current_output * current_weight？",
        ["同一 token 会接收 K 位专家的贡献，必须累加", "因为 Linear 只能原地执行", "因为 selected_experts 是浮点数"],
        0,
        "Top-K 路由让一个 token 经过多位专家；每位专家只贡献加权的一部分，最终输出是这些贡献之和。"
      ),
      homework: [
        "完成 TODO 3：沿最后一维计算每个 token 的 Top-K 权重和，并用它重归一化。",
        "运行测试前先手算 [0.60,0.20] 应变成 [0.75,0.25]，确认代码保持相对比例。",
        "测试目标：每行权重和接近 1、索引不越界、SparseMoEBlock 输出 shape 与输入完全相同。",
        "错误诊断：出现广播错误时检查 keepdim；输出整体偏小时检查是否忘了重归一化；只剩一个专家贡献时检查聚合是否误用赋值代替 +=。"
      ]
    })
  ],

  "07": [
    lesson({
      id: "load-balance-probability-mass",
      title: "先累计 P_i：每位专家平均拿到多少路由权重",
      todo: "TODO 1：计算 P_i",
      prerequisite: [
        "函数收到的是稀疏 Top-K 结果：routing_weights 和 selected_experts 都是 [total_tokens, top_k]，不是 [T,E] 的稠密概率表。",
        "同一个位置的 routing_weights[t,k] 与 selected_experts[t,k] 配对：前者是权重，后者是要把这份权重加给哪位专家。",
        "flatten() 会把 T×K 次选择排成一列；两个张量必须一起 flatten，配对关系才不会被打乱。",
        "scatter_add_(0, index, src) 会按 index 指定的目标位置，把 src 中的值累加到长度为 E 的统计向量。"
      ],
      intuition: "想象每个 token 有 K 张带权选票。selected_experts 写着投给谁，routing_weights 写着票有多重。P_i 的第一步是把所有投给专家 i 的票重相加，第二步再除以 token 数 T，得到每个 token 平均贡献给专家 i 的概率质量。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>[T,K]</b>每个 token 的 K 次选择</span>
          <span><b>flatten</b>排成 T×K 对“编号+权重”</span>
          <span><b>scatter_add</b>按专家编号累加</span>
          <span><b>/ T</b>得到 P_i，shape [E]</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>编号和权重必须同位置配对</h4>
            <div class="adv-contract">
              <span><code>selected</code></span><strong>[[0,2],[1,2]]</strong>
              <span><code>weights</code></span><strong>[[0.7,0.3],[0.4,0.6]]</strong>
              <span>专家 2 收到</span><strong>0.3 + 0.6 = 0.9</strong>
              <span>专家 0 收到</span><strong>0.7</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>P_i 的合同</h4>
            <div class="adv-contract">
              <span>初始化</span><strong>长度 E、dtype/device 跟 routing_weights 一致的零向量</strong>
              <span>累加后</span><strong>每位专家在整个 batch 收到的权重总和</strong>
              <span>除以</span><strong>total_tokens，不是 total_tokens×top_k</strong>
              <span>最终和</span><strong>每个 token 的 Top-K 权重和为 1 时，P_i 总和也为 1</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">Notebook 的 P_i 基于传入的稀疏 Top-K 权重统计。不要在函数里重新 softmax，也不要虚构一个 [T,E] 输入。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("按柜台编号累计每笔金额", [
          "counter_ids = torch.tensor([2, 0, 2, 1])",
          "amounts = torch.tensor([3.0, 7.0, 6.0, 4.0])",
          "",
          "totals = torch.zeros(",
          "    3, dtype=amounts.dtype, device=amounts.device",
          ")",
          "totals.scatter_add_(0, counter_ids, amounts)",
          "averages = totals / 2  # 这里有 2 位顾客",
          "print(totals)          # [7, 4, 9]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>counter_ids</code> 对应展平后的 <code>selected_experts</code>。</li>
            <li><code>amounts</code> 对应展平后的 <code>routing_weights</code>。</li>
            <li><code>3</code> 个柜台对应 <code>num_experts</code>。</li>
            <li><code>2</code> 位顾客对应 <code>total_tokens</code>；累计完成后除它得到 <code>P_i</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "有 2 个 token、top_k=2，每个 token 的两个权重之和为 1。",
        question: "所有专家的累计权重总和除以 total_tokens 后，P_i.sum() 应接近多少？",
        options: ["1", "2", "top_k 的平方"],
        answer: 0,
        revealNote: "每个 token 总共贡献 1 的概率质量；T 个 token 共贡献 T，除以 T 后总和为 1。"
      },
      checkpoint: checkpoint(
        "为什么 P_i 的零向量要显式继承 routing_weights 的 dtype 和 device？",
        ["scatter_add_ 的目标和源需要类型、设备兼容", "这样 selected_experts 会自动变成浮点", "只为了打印更漂亮"],
        0,
        "统计向量是 scatter_add_ 的目标。若一个在 CPU、一个在 GPU，或 dtype 不兼容，运算会失败。"
      ),
      homework: [
        "完成 TODO 1：创建长度为 num_experts 的零向量，把展平后的权重按展平后的专家编号累加，再除以 total_tokens。",
        "shape/dtype/device 自检：P_i 是 [num_experts]，dtype/device 与 routing_weights 一致。",
        "数值自检：当前输入每个 token 的权重和为 1，因此 P_i.sum() 应接近 1。",
        "错误诊断：若 shape 不匹配，检查两个输入是否都 flatten；若专家统计错位，检查 index 和 src 是否接反。"
      ]
    }),

    lesson({
      id: "load-balance-selection-frequency",
      title: "再统计 f_i：每位专家被选中的次数占比",
      todo: "TODO 2：expert_mask、tokens_per_expert、f_i",
      prerequisite: [
        "P_i 统计带权概率质量；f_i 只统计被选中的次数，不关心该次权重是 0.9 还是 0.1。",
        "F.one_hot(selected_experts, num_classes=num_experts) 会把 [T,K] 的整数编号变成 [T,K,E]。",
        "在 one-hot 张量上沿 dim=(0,1) 求和，就是把所有 token 与所有 Top-K 槽位都统计掉，只留下 E 位专家。",
        "Top-K 路由共有 T×K 次选择，所以 f_i 的分母必须是 total_tokens * top_k。"
      ],
      intuition: "P_i 像统计每个柜台收到的业务量，f_i 像统计每个柜台被叫号多少次。一次叫号无论业务大小都算 1 次。one-hot 把专家编号翻译成一排只有一个 1 的记分卡，沿 token 和槽位两维求和就得到每位专家的次数。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>one-hot 多出专家维</h4>
            <div class="adv-flow">
              <span>selected<br>[T,K]</span>
              <span>one_hot<br>[T,K,E]</span>
              <span>sum dim=(0,1)</span>
              <strong>counts<br>[E]</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>分母为什么是 T×K</h4>
            <p>T=1000、K=2 时，Router 一共做了 2000 次专家选择。若仍只除以 1000，<code>f_i.sum()</code> 会变成 2，而不是概率分布需要的 1。</p>
            <div class="adv-contract">
              <span>次数总和</span><strong>T×K</strong>
              <span>归一化分母</span><strong>total_tokens * top_k</strong>
              <span>f_i 总和</span><strong>1</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>统计量</th><th>看什么</th><th>是否使用 routing_weights</th><th>归一化分母</th></tr></thead>
          <tbody>
            <tr><td><code>P_i</code></td><td>平均权重</td><td>使用</td><td><code>T</code></td></tr>
            <tr><td><code>f_i</code></td><td>选择次数占比</td><td>不使用</td><td><code>T*K</code></td></tr>
          </tbody>
        </table>

        <div class="adv-callout">tokens_per_expert 要转为浮点再做除法，这样 f_i 才是可与 P_i 相乘的比例向量。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("统计每轮被选中的活动类别", [
          "chosen = torch.tensor([",
          "    [0, 2],",
          "    [1, 2],",
          "])",
          "",
          "# 2 轮、每轮 2 个选择、共 3 个类别",
          "marks = F.one_hot(chosen, num_classes=3)",
          "counts = marks.sum(dim=(0, 1)).float()",
          "fractions = counts / (2 * 2)",
          "print(counts)          # [1, 1, 2]",
          "print(fractions.sum()) # 1"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>chosen</code> 对应 <code>selected_experts</code>。</li>
            <li><code>num_classes=3</code> 对应 <code>num_classes=num_experts</code>。</li>
            <li><code>counts</code> 对应 <code>tokens_per_expert</code>，沿前两维求和。</li>
            <li><code>2*2</code> 对应 <code>total_tokens * top_k</code>，归一化后得到 <code>f_i</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "selected_experts.shape=[1000,2]，每一行都有两个合法专家编号。",
        question: "tokens_per_expert.sum() 应等于多少？",
        options: ["2000", "1000", "8"],
        answer: 0,
        revealNote: "共有 1000×2 次选择；one-hot 求和会逐次计数，所以总次数是 2000。"
      },
      checkpoint: checkpoint(
        "为什么计算 f_i 时不能用 routing_weights 的数值？",
        ["f_i 定义的是被选中次数占比，每次选择都记 1", "routing_weights 没有 device", "one_hot 只能处理浮点权重"],
        0,
        "权重大小属于 P_i 的视角；f_i 的职责是统计实际分配频率。"
      ),
      homework: [
        "完成 TODO 2：one-hot 编码专家编号，沿 token 与 Top-K 槽位求和，转浮点并除以 T×K。",
        "shape 自检：expert_mask 为 [T,K,E]，tokens_per_expert 与 f_i 都为 [E]。",
        "数值自检：tokens_per_expert.sum() 等于 T×K，f_i.sum() 接近 1。",
        "错误诊断：若 f_i.sum() 等于 K，说明分母漏乘 top_k；若 one_hot 报错，检查 selected_experts 是否仍是整数索引。"
      ]
    }),

    lesson({
      id: "load-balance-auxiliary-loss",
      title: "最后点乘两种偏好，读懂均匀与塌缩测试",
      todo: "TODO 3：计算 aux_loss",
      prerequisite: [
        "P_i 和 f_i 都是长度 E 的比例向量，且在本 Notebook 定义下各自总和为 1。",
        "逐元素乘法 f_i * P_i 表示同一位专家的“实际频率 × 平均权重”；sum() 再把 E 位专家的贡献合起来。",
        "公式是 alpha * num_experts * sum(f_i * P_i)，返回值应是 0 维浮点 Tensor。",
        "均匀分布时 f_i=P_i=1/E，因此最终理论值是 alpha；测试还要求严重偏向专家 0、1 时的 loss 大于均匀值的两倍。"
      ],
      intuition: "如果 Router 在权重上偏爱某些专家，同时实际也反复选择它们，那么 P_i 和 f_i 会在同一位置一起变大，点积随之升高。优化器为了减小辅助损失，会避免“想选谁”和“总选谁”长期集中在同一小撮专家上。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>完全均匀</h4>
            <div class="adv-contract">
              <span><code>f_i</code></span><strong>[1/E, ..., 1/E]</strong>
              <span><code>P_i</code></span><strong>[1/E, ..., 1/E]</strong>
              <span>点积</span><strong>E × 1/E² = 1/E</strong>
              <span>aux loss</span><strong>alpha × E × 1/E = alpha</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>严重塌缩</h4>
            <p>测试让所有 token 都选择专家 0 和 1，且各占 0.5。两种统计同时集中在前两位，点积明显变大，因此辅助损失必须高于均匀分配。</p>
            <div class="adv-flow">
              <span>选择集中</span>
              <span>权重集中</span>
              <strong>惩罚升高</strong>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span>函数输出</span><strong>单个标量 Tensor，可加到主任务 loss</strong>
          <span>均匀测试</span><strong>loss_good 接近 alpha=0.01，atol=1e-4</strong>
          <span>塌缩测试</span><strong>loss_bad &gt; loss_good * 2</strong>
          <span>工程用法</span><strong>total_loss = task_loss + aux_loss</strong>
        </div>

        <div class="adv-callout">本课只实现辅助项，不要在函数里创建 CrossEntropyLoss，也不要把 alpha 再乘一次到主任务 loss 上。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把两个类别分布按同位置点乘", [
          "visit_fraction = torch.tensor([0.5, 0.3, 0.2])",
          "interest_mass = torch.tensor([0.4, 0.4, 0.2])",
          "coefficient = 0.01",
          "categories = 3",
          "",
          "overlap = (visit_fraction * interest_mass).sum()",
          "penalty = coefficient * categories * overlap",
          "print(penalty.ndim) # 0，标量 Tensor"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>visit_fraction</code> 对应实际分配比例 <code>f_i</code>。</li>
            <li><code>interest_mass</code> 对应平均路由权重 <code>P_i</code>。</li>
            <li><code>categories</code> 对应 <code>num_experts</code>，<code>coefficient</code> 对应 <code>alpha</code>。</li>
            <li>保持逐元素乘法后再 <code>.sum()</code>，不要误写成矩阵乘法得到不同 shape。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "E=8，完全均匀时 f_i=P_i=1/8，alpha=0.01。",
        question: "按 Notebook 公式计算，均匀分配的 aux_loss 应是多少？",
        options: ["0.01", "0.00125", "0.08"],
        answer: 0,
        revealNote: "点积是 1/E，再乘 E 抵消，最终正好等于 alpha。"
      },
      checkpoint: checkpoint(
        "若均匀测试通过，但 loss_bad 与 loss_good 几乎一样，最应该先检查哪两项？",
        ["P_i 是否按专家累加、f_i 是否按选择次数统计", "是否创建了 optimizer", "matplotlib 是否安装"],
        0,
        "辅助损失的区分能力来自两种专家分布；统计对象或归一化写错，会抹平塌缩与均匀之间的差异。"
      ),
      homework: [
        "完成 TODO 3：严格按 alpha、num_experts、f_i、P_i 的公式返回标量 aux_loss。",
        "先验证均匀输入：loss_good 应在 1e-4 容差内接近 0.01。",
        "再验证塌缩输入：loss_bad 必须大于 loss_good 的两倍。",
        "错误诊断：均匀值多一个 K 时回查 f_i 分母；P_i.sum() 不为 1 时回查 scatter_add 与除数；返回 shape 非标量时检查是否漏掉 sum()。"
      ]
    })
  ],

  "08": [
    lesson({
      id: "gemma-rmsnorm-one-plus-weight",
      title: "Gemma RMSNorm：从纯归一化开始学习缩放",
      todo: "TODO 1：output = normalized × (1 + weight)",
      prerequisite: [
        "x 的 shape 是 [batch_size, seq_len, hidden_size]；RMSNorm 只沿最后的 hidden_size 维计算均方。",
        "variance = x.float().pow(2).mean(-1, keepdim=True) 得到 [B,S,1]，再通过广播作用到 [B,S,H]。",
        "x_norm 已经是 FP32 的归一化结果；self.weight 是长度 H 的可训练参数，会沿 B、S 两维广播。",
        "Gemma 的 weight 初始化为 0，所以真正的缩放因子 1 + weight 初始为 1；return 时必须用 type_as(x) 恢复输入 dtype。"
      ],
      intuition: "标准缩放参数若直接乘 weight，就需要把 weight 初始化为 1。Gemma 换一种参数化：保存从 1 的偏移量，实际乘 (1+w)。当 w=0 时，层先只做 RMS 归一化；训练再学习每个隐藏特征应该从 1 向上或向下调整多少。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>x</b>[B,S,H]，保留原 dtype</span>
          <span><b>x.float()</b>用 FP32 统计平方均值</span>
          <span><b>x_norm</b>每个 token 的 RMS 归一化</span>
          <span><b>1 + weight</b>按隐藏特征缩放</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>shape 与广播</h4>
            <div class="adv-contract">
              <span><code>variance</code></span><strong>[B,S,1]</strong>
              <span><code>x_norm</code></span><strong>[B,S,H]，FP32</strong>
              <span><code>weight</code></span><strong>[H]，每个隐藏特征一个缩放偏移</strong>
              <span><code>output</code></span><strong>[B,S,H]，返回时恢复 x.dtype</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>初始化时发生什么</h4>
            <div class="adv-flow">
              <span>weight = 0</span>
              <span>1 + weight = 1</span>
              <strong>output = x_norm</strong>
            </div>
            <p>这不是恒等映射 <code>output=x</code>；它仍然做了 RMS 归一化，只是没有额外改变归一化后的特征缩放。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>测试</th><th>期望</th><th>它证明什么</th></tr></thead>
          <tbody>
            <tr><td>weight=0</td><td>out 等于手算的 x_norm</td><td>正确使用了 1+w</td></tr>
            <tr><td>weight 非零</td><td>out2 与 out 不同</td><td>可训练缩放真正生效</td></tr>
            <tr><td>FP16 输入</td><td>FP16 输出</td><td>最后恢复了原 dtype</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">TODO 只需在已有 x_norm 上应用 Gemma 缩放。不要重新计算 variance，也不要在乘法前把 self.weight 强制改成不可训练数据。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("从基准亮度 1 学习每个通道的偏移", [
          "normalized_pixels = torch.tensor([",
          "    [0.5, -1.0, 0.25],",
          "])",
          "channel_offset = torch.tensor([0.0, 0.2, -0.1])",
          "",
          "scale = 1 + channel_offset",
          "adjusted = normalized_pixels * scale",
          "print(scale)    # [1.0, 1.2, 0.9]",
          "print(adjusted.shape)"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>normalized_pixels</code> 对应已经算好的 <code>x_norm</code>。</li>
            <li><code>channel_offset</code> 对应可训练的 <code>self.weight</code>。</li>
            <li><code>adjusted</code> 对应 TODO 要创建的 <code>output</code>。</li>
            <li>Notebook 的 return 已负责 <code>type_as(x)</code>，因此 TODO 保持 FP32 运算即可。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "self.weight 初始化为全 0，但 forward 仍执行了 RMS 归一化。",
        question: "此时 output 最准确的描述是什么？",
        options: ["等于 x_norm，而不一定等于原始 x", "等于全 0", "等于 2*x_norm"],
        answer: 0,
        revealNote: "1+0=1，所以缩放不改变 x_norm；RMS 归一化本身仍然存在。"
      },
      checkpoint: checkpoint(
        "为什么 variance 使用 keepdim=True？",
        ["保留 [B,S,1]，便于 rsqrt 结果广播乘回 [B,S,H]", "让 variance 变成长整型", "为了增加一个 batch"],
        0,
        "RMS 是每个 token 一个标量。保留最后一维后，它可以自然广播到该 token 的 H 个特征。"
      ),
      homework: [
        "完成 TODO 1：把 x_norm 与 Gemma 的 (1 + self.weight) 缩放结合，赋给 output。",
        "shape/dtype 自检：output 与 x shape 相同；FP16 输入经过 return 后仍是 FP16。",
        "测试目标：weight=0 时等于测试手算的 expected；weight 改成非零后输出发生变化。",
        "错误诊断：初始化测试差很多时检查是否漏加 1；FP16 测试失败时检查是否改坏了 type_as(x) 返回路径。"
      ]
    }),

    lesson({
      id: "qwen-tied-embedding-parameter",
      title: "Qwen 权重绑定：让两个模块持有同一个 Parameter",
      todo: "TODO 2：lm_head.weight 指向 embed_tokens.weight",
      prerequisite: [
        "nn.Embedding(vocab_size, hidden_size) 的 weight shape 是 [V,H]；nn.Linear(hidden_size, vocab_size, bias=False) 的 weight 也是 [V,H]。",
        "shape 相同只说明可以共享，不代表它们已经共享。两个模块刚创建时默认各有一块独立参数内存。",
        "权重绑定是对象引用赋值，不是复制数值；赋值后两个属性指向同一个 nn.Parameter。",
        "测试用 data_ptr() 验证权重和梯度的物理内存地址，而不只是用 allclose 验证数值相等。"
      ],
      intuition: "Embedding 用矩阵按 token id 查行，LM Head 用同一矩阵把 hidden state 投影回词表。绑定不是每次手动同步两张表，而是给同一张表贴上两个入口标签：从任何入口更新，看到的都是同一份参数与同一份梯度。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>为什么 shape 正好匹配</h4>
            <div class="adv-contract">
              <span>Embedding weight</span><strong>[vocab_size, hidden_size]</strong>
              <span>LM Head weight</span><strong>[vocab_size, hidden_size]</strong>
              <span>输入方向</span><strong>token id -&gt; 查出 hidden 向量</strong>
              <span>输出方向</span><strong>hidden 向量 -&gt; vocab logits</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>复制与绑定不是一回事</h4>
            <div class="adv-contract">
              <span>复制数值</span><strong>当前数值相同，未来更新仍各走各的</strong>
              <span>绑定参数</span><strong>两个 weight 属性就是同一个 Parameter</strong>
              <span>测试证据</span><strong>weight.data_ptr() 相同，grad.data_ptr() 也相同</strong>
            </div>
          </section>
        </div>

        <div class="adv-roadmap">
          <span><b>创建 Embedding</b>得到参数 [V,H]</span>
          <span><b>创建 bias=False Linear</b>得到兼容的 [V,H]</span>
          <span><b>重新指向</b>LM Head 使用 Embedding 参数</span>
          <span><b>backward</b>两条路径的梯度汇入同一 weight.grad</span>
        </div>

        <div class="adv-checks">
          <span>权重指针相同</span>
          <span>修改 Embedding 后 Head 同步</span>
          <span>两端都有 grad</span>
          <span>梯度指针相同</span>
        </div>

        <div class="adv-callout">不要写 <code>lm_head.weight.data = embed_tokens.weight.data.clone()</code>。clone 明确创建了新内存，无法通过指针共享测试。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("让两个模块共享同一个可训练温度参数", [
          "class SharedTemperature(nn.Module):",
          "    def __init__(self):",
          "        super().__init__()",
          "        self.encoder_scale = nn.Parameter(torch.ones(4))",
          "        self.decoder_scale = self.encoder_scale",
          "",
          "module = SharedTemperature()",
          "assert module.encoder_scale.data_ptr() == \\",
          "       module.decoder_scale.data_ptr()"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>encoder_scale</code> 对应 <code>self.embed_tokens.weight</code>。</li>
            <li><code>decoder_scale</code> 对应 <code>self.lm_head.weight</code>。</li>
            <li>TODO 位于 <code>__init__</code>：让后者属性直接引用前者的 Parameter。</li>
            <li>完成后删掉占位的 <code>raise NotImplementedError</code>，否则构造对象时仍会提前中断。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "两个 [V,H] 参数的数值通过 copy_ 变得完全相同。",
        question: "仅凭 torch.allclose 为 True，能否证明权重已经绑定？",
        options: ["不能，还要检查 data_ptr 或对象引用", "能，数值相等就一定共享内存", "只有在 CPU 上能证明"],
        answer: 0,
        revealNote: "独立 Tensor 可以拥有相同数值。Notebook 特意检查 data_ptr，要求物理内存级共享。"
      },
      checkpoint: checkpoint(
        "绑定后，为什么 Embedding 路径与 LM Head 路径的梯度会出现在同一个 weight.grad 中？",
        ["两条计算路径引用同一个 nn.Parameter，autograd 会把贡献累加到它的 grad", "因为 bias=False 会复制梯度", "因为 data_ptr 会自动调用 optimizer"],
        0,
        "共享的是 Parameter 本身，所以它只有一份 grad 存储；两条路径的反向贡献都累积到这里。"
      ),
      homework: [
        "完成 TODO 2：在 __init__ 中让 lm_head.weight 直接指向 embed_tokens.weight，并移除异常占位。",
        "shape 自检：两边权重都应为 [vocab_size, hidden_size]；LM Head 保持 bias=False。",
        "测试目标：权重 data_ptr 相同，修改任一入口时另一端同步，backward 后 grad data_ptr 也相同。",
        "错误诊断：数值相同但指针不同，说明做了 copy/clone；构造对象直接失败，说明未删掉 NotImplementedError。"
      ]
    })
  ],

  "09": [
    lesson({
      id: "sft-label-mask-and-length",
      title: "先构造 labels：只让 response 与有效位置参与学习",
      todo: "TODO 1–2：labels、截断与填充",
      prerequisite: [
        "input_ids 是 prompt_ids + response_ids；它包含模型需要看到的完整上下文。",
        "labels 与 input_ids 长度相同，但 prompt 位置写 -100，response 位置保留真实 token id。",
        "nn.CrossEntropyLoss(ignore_index=-100) 会跳过标签为 -100 的位置，因此 prompt 与 padding 都不会贡献 loss。",
        "截断时 input_ids 与 labels 必须切同一范围；填充时 input_ids 填 pad_id，而 labels 填 -100。最终两者固定为 max_len 个 long 元素。"
      ],
      intuition: "输入像一张完整试卷：题目和标准答案都要给模型看；labels 像评分模板：题目区域打上“不计分”，只给答案区域评分。Padding 只是把纸张补到统一长度，也必须标成不计分。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>同一序列，两种职责</h4>
            <div class="adv-contract">
              <span><code>input_ids</code></span><strong>[prompt tokens, response tokens, pad tokens]</strong>
              <span><code>labels</code></span><strong>[-100..., response tokens, -100...]</strong>
              <span>模型可见</span><strong>prompt、response、padding 都在输入中</strong>
              <span>参与评分</span><strong>只有 labels 中非 -100 的 response token</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>Notebook 样例</h4>
            <div class="adv-contract">
              <span>prompt</span><strong>[10,20,30]</strong>
              <span>response</span><strong>[40,50,60,70]</strong>
              <span>max_len</span><strong>8，需要补 1 个位置</strong>
              <span>labels</span><strong>[-100,-100,-100,40,50,60,70,-100]</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>拼接 input_ids</code><span>保持 prompt 在前、response 在后</span></div>
          <div><b>2</b><code>构造同长度 labels</code><span>prompt 用 -100，response 保留 id</span></div>
          <div><b>3</b><code>若超长：两者一起 [:max_len]</code><span>不能只截输入或只截标签</span></div>
          <div><b>4</b><code>若不足：计算 pad_len</code><span>输入补 pad_id，标签补 -100</span></div>
          <div><b>5</b><code>转 torch.long</code><span>token id 与类别标签都要求整数类型</span></div>
        </div>

        <div class="adv-callout">Notebook 注释写“从末尾截断”，参考实现实际采用 <code>[:max_len]</code>：保留序列前 max_len 个位置、丢弃末尾超出的部分。页面与测试按这一当前实现合同对齐。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("给问答卡制作固定长度评分模板", [
          "question = [7, 8]",
          "answer = [21, 22, 23]",
          "limit = 7",
          "",
          "tokens = question + answer",
          "score_labels = [-100] * len(question) + answer",
          "",
          "if len(tokens) > limit:",
          "    tokens = tokens[:limit]",
          "    score_labels = score_labels[:limit]",
          "else:",
          "    missing = limit - len(tokens)",
          "    tokens = tokens + [0] * missing",
          "    score_labels = score_labels + [-100] * missing"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>question / answer</code> 对应 <code>prompt_ids / response_ids</code>。</li>
            <li><code>tokens / score_labels</code> 对应 <code>input_ids / labels</code>。</li>
            <li><code>limit</code> 对应 <code>max_len</code>，输入填充值 0 要改为函数参数 <code>pad_id</code>。</li>
            <li>返回前把两个列表分别转成 <code>torch.tensor(..., dtype=torch.long)</code>；Notebook 已给好这部分。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prompt 长 3、response 长 4、max_len=8。",
        question: "labels 的最后一个位置应该填什么？",
        options: ["-100", "pad_id=0 并参与 loss", "response 的最后一个 token 70"],
        answer: 0,
        revealNote: "第 8 个位置是 padding；labels 中 padding 必须是 ignore_index=-100。"
      },
      checkpoint: checkpoint(
        "如果序列长度超过 max_len，为什么 input_ids 和 labels 必须使用同一切片？",
        ["保证每个输入位置仍与同位置的评分规则对齐", "为了把 long 转为 float", "因为 prompt 必须全部参与 loss"],
        0,
        "输入与标签是逐位置对应的。截断范围不同会让 response 标签错位，训练目标失真。"
      ),
      homework: [
        "完成 TODO 1：labels 的 prompt 部分全部为 -100，response 部分保留原 token id。",
        "完成 TODO 2：实现超长时同步截断、不足时分别用 pad_id 和 -100 补到 max_len。",
        "合同自检：返回两个 shape=[max_len]、dtype=torch.long 的 Tensor。",
        "错误诊断：标签断言失败时逐段打印 prompt 区、response 区、pad 区；若长度不同，检查是否两张列表都执行了相同截断或补齐。"
      ]
    }),

    lesson({
      id: "sft-next-token-shift",
      title: "再做 shift：第 t 个预测对齐第 t+1 个标签",
      todo: "TODO 3：shift_logits 与 shift_labels",
      prerequisite: [
        "logits.shape=[batch_size, seq_len, vocab_size]；每个位置保存对词表中所有 token 的预测分数。",
        "labels.shape=[batch_size, seq_len]；每个位置只有一个目标 token id 或 ignore_index=-100。",
        "自回归模型在位置 t 的输出用于预测下一个 token t+1，因此不能让 logits[t] 与 labels[t] 对齐。",
        "切片后的 Tensor 可能不是连续内存；Notebook 使用 contiguous() 后再 view 展平。"
      ],
      intuition: "把 logits 当成每个站点发出的“下一站预测”。0 号站的预测应与 1 号站真实标签比较，1 号站与 2 号站比较。最后一个站没有下一个标签，所以丢掉最后一个 logits；第一个标签没有前一站预测，所以丢掉第一个 label。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>logits 位置 0</b>预测 token 1</span>
          <span><b>logits 位置 1</b>预测 token 2</span>
          <span><b>...</b>继续一位错开</span>
          <span><b>logits 位置 S-2</b>预测 token S-1</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>两边各切掉一端</h4>
            <div class="adv-contract">
              <span>原 logits</span><strong>[B,S,V]</strong>
              <span>去掉最后位置</span><strong>[B,S-1,V]</strong>
              <span>原 labels</span><strong>[B,S]</strong>
              <span>去掉第一位置</span><strong>[B,S-1]</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>Notebook 的准确预测放在哪里</h4>
            <p>response 第一个 token 是 40，位于 labels[3]。测试把高分放在 logits[2,40]，正是因为 shift 后 logits 原位置 2 会与 labels 原位置 3 比较。</p>
            <div class="adv-flow">
              <span>logits[2,40]=50</span>
              <span>对齐 labels[3]=40</span>
              <strong>该位置低 loss</strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">不要把 logits 和 labels 向同一方向切。若两边都去掉第一项或都去掉最后一项，shape 仍可能匹配，但语义会变成预测当前位置而不是下一个位置。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把“下一天预测”与真实天气错开一位", [
          "predictions = torch.tensor([10, 20, 30, 40])",
          "actual = torch.tensor([9, 11, 19, 31])",
          "",
          "# 第 0 天的输出用于预测第 1 天",
          "next_predictions = predictions[:-1].contiguous()",
          "next_actual = actual[1:].contiguous()",
          "print(next_predictions) # [10,20,30]",
          "print(next_actual)       # [11,19,31]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>predictions[:-1]</code> 对应保留 batch、去掉 logits 的最后一个序列位置，并保留完整词表维。</li>
            <li><code>actual[1:]</code> 对应保留 batch、去掉 labels 的第一个序列位置。</li>
            <li>Notebook 用省略号索引处理任意前置 batch 维；最后别漏掉 <code>.contiguous()</code>。</li>
            <li>完成后两个结果的有效位置数都必须是 <code>seq_len-1</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "labels 原位置 3 的值是 response token 40。",
        question: "哪一个 logits 原位置应该负责预测这个 40？",
        options: ["位置 2", "位置 3", "位置 4"],
        answer: 0,
        revealNote: "自回归预测向前看一位：位置 2 的输出预测位置 3 的 token。"
      },
      checkpoint: checkpoint(
        "logits.shape=[1,8,100]、labels.shape=[1,8]，shift 后 shape 分别是什么？",
        ["[1,7,100] 与 [1,7]", "[1,8,99] 与 [1,8]", "[7,100] 与 [8]"],
        0,
        "shift 只缩短序列维一格；batch 与 vocab 维不变。"
      ),
      homework: [
        "完成 TODO 3：logits 去掉最后一个序列位置，labels 去掉第一个序列位置，并让切片连续。",
        "shape 自检：shift_logits=[B,S-1,V]，shift_labels=[B,S-1]。",
        "用测试样例追踪：logits 原位置 2、3、4、5 应分别预测 labels 中的 40、50、60、70。",
        "错误诊断：loss 很大但 shape 正确时，优先检查两边切片方向是否写反。"
      ]
    }),

    lesson({
      id: "sft-cross-entropy-contract",
      title: "最后展平：把所有有效位置交给交叉熵",
      todo: "TODO 4：CrossEntropyLoss 与 view",
      prerequisite: [
        "nn.CrossEntropyLoss 的常用输入合同是 logits [N,C]、targets [N]；C 是类别数，在语言模型里就是 vocab_size。",
        "shift_logits 当前是 [B,S-1,V]，要把 B 与 S-1 合并为样本数 N；shift_labels 同样合并为 N。",
        "logits 最后一维 V 必须保留，不能把整个 Tensor flatten 成一列。",
        "ignore_index=-100 让 prompt 与 padding 对齐到的目标位置不计入平均 loss。"
      ],
      intuition: "交叉熵把“每个有效位置”当成一道 V 选 1 的分类题。B 个序列、每个 S-1 道题可以排成 N 道题，但每道题的 V 个选项必须保持在同一行。labels 则每题只给一个正确选项编号。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>展平前后</h4>
            <div class="adv-contract">
              <span><code>shift_logits</code></span><strong>[B,S-1,V] -&gt; [B×(S-1),V]</strong>
              <span><code>shift_labels</code></span><strong>[B,S-1] -&gt; [B×(S-1)]</strong>
              <span>类别维</span><strong>始终保留 V，由 size(-1) 读取</strong>
              <span>忽略位置</span><strong>target=-100，不参与 loss</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>为什么测试 loss 很低</h4>
            <p>测试只在 40、50、60、70 四个 response 目标上放置极高正确 logit；prompt 与 padding 标签为 -100，被 loss 跳过，因此整体 loss 小于 0.01。</p>
            <div class="adv-flow">
              <span>prompt<br>忽略</span>
              <span>4 个 response<br>预测正确</span>
              <span>padding<br>忽略</span>
              <strong>loss 很低</strong>
            </div>
          </section>
        </div>

        <div class="adv-checks">
          <span>loss 是 0 维 Tensor</span>
          <span>词表维没有被压扁</span>
          <span>targets dtype 为 long</span>
          <span>ignore_index 明确为 -100</span>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把两组选择题排成一个题目列表", [
          "# 2 位学生，每人 3 题，每题 4 个选项",
          "choice_logits = torch.randn(2, 3, 4)",
          "answers = torch.tensor([",
          "    [1, 0, -100],",
          "    [2, 3, 1],",
          "])",
          "",
          "criterion = nn.CrossEntropyLoss(ignore_index=-100)",
          "flat_logits = choice_logits.view(-1, choice_logits.size(-1))",
          "flat_answers = answers.view(-1)",
          "loss = criterion(flat_logits, flat_answers)"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>choice_logits</code> 对应 <code>shift_logits</code>，最后一维 4 对应词表大小 V。</li>
            <li><code>answers</code> 对应 <code>shift_labels</code>，-100 的题目自动跳过。</li>
            <li><code>flat_logits</code> 与 <code>flat_answers</code> 是 TODO 4 展平后的合同。</li>
            <li>使用已有变量的 <code>size(-1)</code> 读取 V，不要硬编码测试里的 vocab_size=100。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "shift_logits.shape=[1,7,100]。",
        question: "交给 CrossEntropyLoss 前，正确的 view shape 是什么？",
        options: ["[7,100]", "[700]", "[1,700]"],
        answer: 0,
        revealNote: "7 个位置是 7 道分类题，每道题仍有 100 个词表候选。"
      },
      checkpoint: checkpoint(
        "为什么 labels 中的 padding 要填 -100 而不是 pad_id=0？",
        ["-100 会被 ignore_index 跳过，0 会被当作需要预测的真实类别", "CrossEntropyLoss 不接受 0", "pad_id 只能放在 GPU"],
        0,
        "target=0 是合法词表类别，会产生梯度；target=-100 才表达“不评分”。"
      ),
      homework: [
        "完成 TODO 4：创建 ignore_index=-100 的交叉熵，把 logits 保留词表维展平、labels 展平，再计算 loss。",
        "shape/dtype 自检：flat logits=[B×(S-1),V]，flat labels=[B×(S-1)] 且为 long，loss 为标量 Tensor。",
        "运行测试：手工强化的四个 response 预测应让 loss 小于 0.01。",
        "错误诊断：类别维报错时检查是否把 logits 完全 flatten；loss 偏大时检查 prompt/padding 是否仍为 -100，以及 shift 是否正确。"
      ]
    })
  ],

  "10": [
    lesson({
      id: "lora-parameters-and-initialization",
      title: "先搭两条路：冻结主权重，只训练低秩 A 与 B",
      todo: "TODO 1–2：模块、参数 shape 与初始化",
      prerequisite: [
        "nn.Linear(in_features, out_features, bias=False) 的 weight shape 是 [out_features, in_features]。",
        "LoRA 主分支保留原始 Linear，但把它的 weight.requires_grad 设为 False；旁路的 lora_A、lora_B 则保持可训练。",
        "lora_A shape=[r,in_features] 负责降到低秩空间，lora_B shape=[out_features,r] 负责升回输出维。",
        "主权重和 A 使用 Kaiming uniform；B 必须初始化为 0，使训练开始时 B@A=0，旁路不改变基座输出。"
      ],
      intuition: "LoRA 像在一条冻结的主干道路旁加一条窄支路。A 把高维输入压成 r 维，B 再把 r 维升回输出空间。支路出口 B 初始封为 0，因此刚安装时车辆仍只走主干；训练逐渐打开 B 后，支路才开始提供可学习的修正。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>参数合同</h4>
            <div class="adv-contract">
              <span><code>linear.weight</code></span><strong>[out,in]，冻结，bias=False</strong>
              <span><code>lora_A</code></span><strong>[r,in]，可训练</strong>
              <span><code>lora_B</code></span><strong>[out,r]，可训练</strong>
              <span><code>scaling</code></span><strong>lora_alpha / r</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>为什么 B 必须为 0</h4>
            <div class="adv-flow">
              <span>A 随机初始化</span>
              <span>B = 0</span>
              <span>B@A = 0</span>
              <strong>初始输出 = base 输出</strong>
            </div>
            <p>A 不能也全 0，否则训练第一步两边可能都得不到有用的梯度；随机 A 与零 B 是刻意搭配。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>路径</th><th>参数量</th><th>是否训练</th><th>初始化目的</th></tr></thead>
          <tbody>
            <tr><td>主权重</td><td><code>out×in</code></td><td>否</td><td>提供基座能力</td></tr>
            <tr><td>LoRA A</td><td><code>r×in</code></td><td>是</td><td>提供非零低秩方向</td></tr>
            <tr><td>LoRA B</td><td><code>out×r</code></td><td>是</td><td>初始关闭增量</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">TODO 1 末尾已有 <code>self.reset_parameters()</code>。不要在 TODO 中再调用一次，也不要漏删两个 <code>pass</code> 占位后留下不可达或重复逻辑。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("创建一个低维瓶颈旁路", [
          "input_dim = 12",
          "output_dim = 20",
          "bottleneck = 3",
          "",
          "base = nn.Linear(input_dim, output_dim, bias=False)",
          "base.weight.requires_grad = False",
          "",
          "down = nn.Parameter(torch.empty(bottleneck, input_dim))",
          "up = nn.Parameter(torch.empty(output_dim, bottleneck))",
          "nn.init.kaiming_uniform_(down, a=math.sqrt(5))",
          "nn.init.zeros_(up)"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>base</code> 对应 <code>self.linear</code>，使用函数的 in_features、out_features。</li>
            <li><code>down / up</code> 对应 <code>self.lora_A / self.lora_B</code>，bottleneck 对应 <code>r</code>。</li>
            <li>TODO 2 还要按 Notebook 要求初始化主权重本身，而不只初始化 A/B。</li>
            <li>参数需要用 <code>nn.Parameter</code> 包装，否则不会出现在 <code>named_parameters()</code> 与优化器中。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "in=128、out=256、r=8。",
        question: "lora_A 与 lora_B 的 shape 应分别是什么？",
        options: ["[8,128] 与 [256,8]", "[128,8] 与 [8,256]", "[256,128] 与 [8,8]"],
        answer: 0,
        revealNote: "A 先把输入 128 降到 8，B 再把 8 升到输出 256。参数按 Linear 权重习惯存为 [输出,输入]。"
      },
      checkpoint: checkpoint(
        "测试为什么要求初始化时 layer(x) 与 layer.linear(x) 完全相同？",
        ["B 为零使 LoRA 增量为零，只剩主分支", "主权重 requires_grad=False 会输出零", "scaling 初始化为零"],
        0,
        "冻结只影响是否求梯度，不影响前向数值；初始等价性来自 lora_B=0。"
      ),
      homework: [
        "完成 TODO 1：创建 bias=False 的主 Linear 并冻结 weight，创建 shape 正确的 A/B Parameter。",
        "完成 TODO 2：主权重与 A 使用指定 Kaiming 初始化，B 初始化为全 0。",
        "合同自检：只有 lora_A、lora_B 可训练；linear.weight 不可训练；三者 shape 与 [out,in]、[r,in]、[out,r] 一致。",
        "错误诊断：初始输出不等于 base 时检查 B 是否全零；优化器看不到 A/B 时检查是否漏用 nn.Parameter。"
      ]
    }),

    lesson({
      id: "lora-forward-low-rank-path",
      title: "再写前向：先降维、再升维、最后乘缩放",
      todo: "TODO 3：result 与 lora_out",
      prerequisite: [
        "PyTorch Linear 计算等价于 x @ weight.T，所以手写 A/B 路径也要按存储 shape 使用转置。",
        "x 的前导维可以是 [batch,seq] 等任意形状；矩阵乘法只处理最后一维。",
        "x @ lora_A.T 把最后一维 in 变为 r；再 @ lora_B.T 把 r 变为 out。",
        "scaling=lora_alpha/r 只乘 LoRA 增量，主分支 self.linear(x) 不乘这个系数。"
      ],
      intuition: "前向输出是两条同 shape 路径的相加：主干给出冻结模型原本的答案，LoRA 支路给出任务适配的修正量。支路不能先算 B，因为 x 的最后一维是 in，只有 A 的转置接得上。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>x</b>[B,S,in]</span>
          <span><b>@ A.T</b>[B,S,r]</span>
          <span><b>@ B.T</b>[B,S,out]</span>
          <span><b>× scaling</b>LoRA 增量</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>主分支</h4>
            <div class="adv-flow">
              <span>x [32,10,128]</span>
              <span>linear</span>
              <strong>result [32,10,256]</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>LoRA 分支</h4>
            <div class="adv-flow">
              <span>x [32,10,128]</span>
              <span>A.T -&gt; [32,10,8]</span>
              <span>B.T -&gt; [32,10,256]</span>
              <strong>×2 后加到 result</strong>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span>测试初始化</span><strong>B=0，所以 lora_out=0</strong>
          <span>测试模拟训练</span><strong>手动把 B 改成非零，输出必须变化</strong>
          <span>返回 shape</span><strong>与主 Linear 输出完全相同</strong>
        </div>

        <div class="adv-callout">不要返回 <code>lora_out</code> 单独一条支路，也不要忘了缩放。LoRA 的定义是冻结主输出加上缩放后的低秩增量。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("让三维批量数据通过两级矩阵", [
          "samples = torch.randn(4, 6, 12)",
          "down = torch.randn(3, 12)",
          "up = torch.randn(20, 3)",
          "scale = 2.0",
          "",
          "compressed = samples @ down.T",
          "expanded = compressed @ up.T",
          "update = expanded * scale",
          "print(compressed.shape) # [4,6,3]",
          "print(update.shape)     # [4,6,20]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>samples / down / up</code> 对应 <code>x / self.lora_A / self.lora_B</code>。</li>
            <li><code>scale</code> 对应构造器已算好的 <code>self.scaling</code>。</li>
            <li>TODO 3 还需先用 <code>self.linear(x)</code> 得到主分支 result，再把 update 加进去。</li>
            <li>矩阵相乘使用 <code>@</code>，A/B 都要转置，因为参数按 [输出,输入] 存储。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "x 最后一维 128，A.shape=[8,128]，B.shape=[256,8]。",
        question: "正确的低秩乘法顺序是什么？",
        options: ["(x @ A.T) @ B.T", "(x @ B) @ A", "(A @ x) @ B"],
        answer: 0,
        revealNote: "A.T 是 [128,8]，先把 128 接到 8；B.T 是 [8,256]，再把 8 接到 256。"
      },
      checkpoint: checkpoint(
        "为什么 scaling 只作用在 lora_out 上？",
        ["它控制低秩更新的幅度，不应改变冻结基座的原始输出", "主权重没有 dtype", "linear 已经自动除以 r"],
        0,
        "LoRA 公式是 W0x + (alpha/r)BAx；缩放属于增量项。"
      ),
      homework: [
        "完成 TODO 3：计算主分支 result，按 A.T、B.T 顺序计算低秩输出并乘 self.scaling，再相加返回。",
        "shape 自检：中间为 [...,r]，lora_out 与 result 都为 [...,out_features]。",
        "测试目标：B=0 时输出等于 base；B 改为非零后输出不再相同。",
        "错误诊断：matmul shape 报错时在纸上写出每个矩阵最后两维；输出不变时检查是否真正把 lora_out 加回 result。"
      ]
    }),

    lesson({
      id: "lora-merge-for-inference",
      title: "最后合并：把 B@A 写回主权重",
      todo: "TODO 4：merge_weights",
      prerequisite: [
        "A.shape=[r,in]、B.shape=[out,r]，所以 B@A 的 shape 正好是 [out,in]，与 linear.weight 相同。",
        "前向中的 (x @ A.T) @ B.T 与使用合并矩阵 x @ (B@A).T 数学等价。",
        "合并时同样必须乘 self.scaling，否则合并前后的输出幅度不同。",
        "Notebook 用 .data 原地更新冻结主权重，是一个教学版部署合并；调用一次后测试只使用 layer.linear(x)。"
      ],
      intuition: "训练时把修正拆成 A、B 两个小矩阵是为了省可训练参数；部署时可以先把两级修正乘成一张完整的 ΔW，再加进 W0。这样推理只走一个普通 Linear，不再多做两次矩阵乘法。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>shape 闭环</h4>
            <div class="adv-flow">
              <span>B [out,r]</span>
              <span>@ A [r,in]</span>
              <span>ΔW [out,in]</span>
              <strong>加到 W0 [out,in]</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>测试的等价目标</h4>
            <div class="adv-contract">
              <span>合并前</span><strong>linear(x) + LoRA(x)</strong>
              <span>合并后</span><strong>updated linear(x)</strong>
              <span>断言</span><strong>torch.allclose(..., atol=1e-5)</strong>
            </div>
          </section>
        </div>

        <div class="adv-roadmap">
          <span><b>训练形态</b>W0 冻结，A/B 单独保存</span>
          <span><b>计算增量</b>ΔW=(B@A)×scaling</span>
          <span><b>合并形态</b>Wmerged=W0+ΔW</span>
          <span><b>推理形态</b>只调用普通 Linear</span>
        </div>

        <div class="adv-callout">这个教学实现没有 merged 标志。不要连续调用 merge_weights，否则同一增量会被重复加进主权重；生产实现通常会记录状态并支持安全 merge/unmerge。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把两级低秩改动合成完整更新矩阵", [
          "base_weight = torch.randn(20, 12)",
          "down = torch.randn(3, 12)",
          "up = torch.randn(20, 3)",
          "scale = 2.0",
          "",
          "delta_weight = (up @ down) * scale",
          "merged_weight = base_weight + delta_weight",
          "print(delta_weight.shape)  # [20,12]",
          "print(merged_weight.shape) # [20,12]"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>base_weight</code> 对应 <code>self.linear.weight.data</code>。</li>
            <li><code>up @ down</code> 对应 <code>self.lora_B @ self.lora_A</code>，顺序不能颠倒。</li>
            <li><code>scale</code> 对应 <code>self.scaling</code>。</li>
            <li>TODO 4 要把增量原地加到主权重；测试随后用 <code>layer.linear(x)</code> 验证。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "A=[8,128]，B=[256,8]。",
        question: "哪一个乘积能得到可加到 linear.weight=[256,128] 的增量？",
        options: ["B @ A", "A @ B", "A.T @ B.T"],
        answer: 0,
        revealNote: "[256,8] @ [8,128] = [256,128]，与主权重 shape 完全一致。"
      },
      checkpoint: checkpoint(
        "合并后为何不再调用完整 layer(x) 来做测试输出？",
        ["完整 forward 仍会再加一次 LoRA 支路，造成重复增量；测试只调用已更新的 linear", "完整 layer 不支持 Tensor", "merge 会删除 forward"],
        0,
        "合并已经把 ΔW 加入 linear.weight；若 forward 再加 lora_out，就计算了两次同一修正。"
      ),
      homework: [
        "完成 TODO 4：按 B@A 顺序构造 [out,in] 增量，乘 scaling 后原地加到主权重。",
        "shape 自检：增量与 self.linear.weight 完全相同。",
        "测试目标：合并前完整 layer 输出与合并后 layer.linear 输出在 atol=1e-5 内一致。",
        "错误诊断：合并输出差固定倍数时检查 scaling；矩阵维度错误时检查是否误写 A@B；重复调用造成漂移时重新初始化 layer 再测试。"
      ]
    })
  ],

  "11": [
    lesson({
      id: "wsd-warmup-step-semantics",
      title: "Warmup：先弄清 scheduler 眼中的 step",
      todo: "TODO 1：从 0 线性升到 base_lr",
      prerequisite: [
        "WSD_Scheduler 继承 LRScheduler，base_lrs 来自 optimizer 初始化时每个参数组的学习率。",
        "当前实现使用 step = self._step_count - 1；构造 scheduler 时父类会初始化一次，因此测试收集到的第一个 optimizer lr 必须是 0。",
        "Warmup 条件是 step < num_warmup_steps，当前学习率按 step/num_warmup_steps 线性增长。",
        "get_lr() 必须为每个 base_lr 计算一个 current_lr，追加到 lrs 列表后返回。"
      ],
      intuition: "Warmup 像逐步打开水龙头：第 0 步先关闭，随后按完成的预热步数比例打开。这里最容易错的不是乘法，而是 step 从哪里开始。Notebook 的测试明确把 lrs[0] 当作 0，所以实现必须服从当前 _step_count 的时序。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>线性比例</h4>
            <div class="adv-contract">
              <span><code>step=0</code></span><strong>0 / warmup = 0，lr=0</strong>
              <span><code>step=250</code></span><strong>完成 25%，lr=0.25×base_lr</strong>
              <span><code>step=999</code></span><strong>接近 base_lr</strong>
              <span><code>step=1000</code></span><strong>进入 Stable，恰好 base_lr</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>测试的调用顺序</h4>
            <div class="adv-flow">
              <span>记录当前 optimizer lr</span>
              <span>optimizer.step()</span>
              <span>scheduler.step()</span>
              <strong>下一轮再记录</strong>
            </div>
            <p>因此测试数组索引与 scheduler 内部 step 的边界要一起理解，不能只凭“第几步”口头猜测。</p>
          </section>
        </div>

        <div class="adv-roadmap">
          <span><b>Warmup</b>0 ≤ step &lt; 1000</span>
          <span><b>Stable</b>1000 ≤ step &lt; 8000</span>
          <span><b>Decay</b>step ≥ 8000</span>
          <span><b>Total</b>10000 次记录</span>
        </div>

        <div class="adv-callout">不要照搬其他调度器常见的 <code>(step+1)/warmup</code> 写法。本 Notebook 的断言要求第一条记录严格为 0，并在索引 warmup 处进入 base_lr。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把音量在 4 个预热刻度中线性打开", [
          "max_volume = 0.8",
          "warmup_steps = 4",
          "",
          "for step in range(warmup_steps):",
          "    current = max_volume * step / warmup_steps",
          "    print(step, current)",
          "# step=0 -> 0.0",
          "# step=2 -> 0.4"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>max_volume</code> 对应当前循环中的 <code>base_lr</code>。</li>
            <li><code>warmup_steps</code> 对应 <code>self.num_warmup_steps</code>。</li>
            <li><code>current</code> 对应 TODO 1 的 <code>current_lr</code>。</li>
            <li>Notebook 可以直接用同一线性公式；保留当前外层对每个 base_lr 的循环。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "base_lr=3e-4，warmup_steps=1000，step=500。",
        question: "Warmup 线性公式得到的 current_lr 是多少？",
        options: ["1.5e-4", "3e-4", "0"],
        answer: 0,
        revealNote: "完成比例 500/1000=0.5，因此学习率是 base_lr 的一半。"
      },
      checkpoint: checkpoint(
        "为什么 step=1000 时不再执行 warmup 分支？",
        ["条件是 step < num_warmup_steps，边界 1000 进入 Stable 并使用完整 base_lr", "因为 total_steps 已结束", "因为 min_lr_ratio 为 0.1"],
        0,
        "严格小于把 0..999 留给 Warmup；1000 是 Stable 的第一步。"
      ),
      homework: [
        "完成 TODO 1：在 warmup 分支中按当前 step 比例计算 current_lr，确保 step=0 时为 0。",
        "边界自检：step=0、500、999、1000 分别应为 0、半程、接近最大、最大值。",
        "多参数组自检：代码必须使用 for 循环里的 base_lr，而不是硬编码测试 max_lr。",
        "错误诊断：lrs[0] 非 0 时检查是否误加 1；warmup 边界不等于 max_lr 时检查分支条件与 Stable 赋值。"
      ]
    }),

    lesson({
      id: "wsd-stable-window",
      title: "Stable：用累计边界守住主要学习窗口",
      todo: "TODO 2：保持 base_lr",
      prerequisite: [
        "Stable 从 num_warmup_steps 开始，到 num_warmup_steps + num_stable_steps 之前结束。",
        "elif 分支只有在 warmup 条件已失败时进入，因此不需要重复写下界。",
        "稳定期 current_lr 就是当前参数组的 base_lr，不做比例、余弦或 min_lr 限制。",
        "Notebook 测试检查 lrs[warmup] 与 lrs[warmup+stable-1] 都等于 max_lr。"
      ],
      intuition: "Stable 不是“什么都没做”，而是刻意给模型一段持续吸收数据的固定更新强度。代码上最重要的是用累计步数当右边界：warmup=1000、stable=7000，Stable 结束点是 8000，而不是 7000。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>用绝对 step 划区间</h4>
            <div class="adv-contract">
              <span>左边界</span><strong>num_warmup_steps = 1000</strong>
              <span>右边界</span><strong>warmup + stable = 8000</strong>
              <span>有效 step</span><strong>1000 ... 7999</strong>
              <span>学习率</span><strong>始终为 base_lr</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>常见边界错误</h4>
            <div class="adv-contract">
              <span>误写 <code>step &lt; stable</code></span><strong>会在 7000 就提前进入 Decay</strong>
              <span>误写 <code>&lt;=</code></span><strong>会多保留一个 Stable step</strong>
              <span>误用 min_lr</span><strong>Stable 不应该提前衰减</strong>
            </div>
          </section>
        </div>

        <div class="adv-checks">
          <span>lrs[1000] = 3e-4</span>
          <span>lrs[7999] = 3e-4</span>
          <span>step=8000 进入 Decay</span>
          <span>延长 Stable 只需增大步数配置</span>
        </div>

        <div class="adv-callout">Stable 右边界是两个阶段长度之和。调度器的 step 是从训练开头累计的，不会在每个阶段自动清零。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用累计时间判断恒速巡航阶段", [
          "warmup = 3",
          "cruise = 5",
          "speed = 80",
          "",
          "for step in range(10):",
          "    if step < warmup:",
          "        current = speed * step / warmup",
          "    elif step < warmup + cruise:",
          "        current = speed",
          "    else:",
          "        current = None  # 后续减速阶段",
          "    print(step, current)"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>warmup + cruise</code> 对应 <code>self.num_warmup_steps + self.num_stable_steps</code>。</li>
            <li><code>speed</code> 对应每个参数组的 <code>base_lr</code>。</li>
            <li><code>current</code> 对应 TODO 2 的 <code>current_lr</code>。</li>
            <li>Notebook 已给好 elif 条件，只需正确设置值，不要重算局部 step。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "warmup=1000、stable=7000，当前 step=7500。",
        question: "当前属于哪个阶段，current_lr 应是多少？",
        options: ["Stable，等于 base_lr", "Decay，接近 min_lr", "Warmup，等于 0.75×base_lr"],
        answer: 0,
        revealNote: "Stable 的绝对区间是 [1000,8000)，7500 仍在其中。"
      },
      checkpoint: checkpoint(
        "为什么 Stable 的条件使用 warmup + stable，而不是只用 stable？",
        ["step 是从训练开始累计的绝对步数，Stable 要接在 Warmup 后", "因为 base_lr 要相加", "因为 LRScheduler 只接受偶数"],
        0,
        "阶段长度要转换成累计边界：第二段右边界等于第一段长度加第二段长度。"
      ),
      homework: [
        "完成 TODO 2：Stable 分支直接令 current_lr 等于当前 base_lr。",
        "边界自检：测试配置下 step=1000 与 step=7999 都是最大值，step=8000 才进入 Decay。",
        "检查分支使用累计边界 warmup+stable，不要把 stable 当作从训练开头计数的终点。",
        "错误诊断：平台期提前结束时检查是否漏加 warmup；平台期值变化时检查是否误用了 Warmup/Decay 的比例。"
      ]
    }),

    lesson({
      id: "wsd-cosine-decay-and-test",
      title: "Decay：把阶段进度映射到余弦退火",
      todo: "TODO 3：从 base_lr 余弦降到 min_lr",
      prerequisite: [
        "Decay 的局部步数是全局 step 减去 warmup 与 stable 两段长度；这样进入 Decay 的第一步局部进度为 0。",
        "decay_ratio=decay_step/num_decay_steps，把当前局部步数映射到 0 到接近 1。",
        "0.5*(1+cos(pi*ratio)) 会从 1 平滑降到 0；再用 min_lr + (base_lr-min_lr)*cosine_decay 映射到实际学习率区间。",
        "min_lr=base_lr*min_lr_ratio。Notebook 的当前参考实现和测试都是余弦衰减；末尾文字中的线性公式与代码不一致，不作为本 TODO 合同。"
      ],
      intuition: "先不要直接背完整公式。分三小步：把全局 step 变成 Decay 内部的进度，把进度送进从 1 降到 0 的余弦系数，再把系数拉伸到 [min_lr, base_lr]。每一步都有清楚的输入输出区间。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>全局 step</b>减去前两段长度</span>
          <span><b>decay_ratio</b>局部进度 0 → 1</span>
          <span><b>cosine_decay</b>系数 1 → 0</span>
          <span><b>current_lr</b>base_lr → min_lr</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>三个关键位置</h4>
            <div class="adv-contract">
              <span>ratio=0</span><strong>cos(0)=1，lr=base_lr</strong>
              <span>ratio=0.5</span><strong>cos(pi/2)=0，系数=0.5，lr 在中点</strong>
              <span>ratio=1</span><strong>cos(pi)=-1，系数=0，lr=min_lr</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>Notebook 最后一条记录</h4>
            <p>循环共记录 total 次，最后记录对应 Decay 的末端。测试要求它在 1e-8 内等于 max_lr×0.1，说明边界公式和 scheduler 调用时序必须共同对齐。</p>
            <div class="adv-contract">
              <span>base_lr</span><strong>3e-4</strong>
              <span>min_lr_ratio</span><strong>0.1</strong>
              <span>期望 min_lr</span><strong>3e-5</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>断言位置</th><th>期望学习率</th><th>失败时检查</th></tr></thead>
          <tbody>
            <tr><td><code>lrs[0]</code></td><td>0</td><td>Warmup step 起点</td></tr>
            <tr><td><code>lrs[warmup]</code></td><td>base_lr</td><td>Warmup/Stable 边界</td></tr>
            <tr><td><code>lrs[warmup+stable-1]</code></td><td>base_lr</td><td>Stable 累计边界</td></tr>
            <tr><td><code>lrs[-1]</code></td><td>base_lr×0.1</td><td>Decay 进度、余弦映射与调用时序</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要把 cosine_decay 直接乘 base_lr；那会降到 0。Notebook 要降到非零 min_lr，所以必须先缩放区间宽度 base_lr-min_lr，再加回 min_lr。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把 0→1 的进度映射成平滑减速系数", [
          "start_speed = 80.0",
          "end_speed = 8.0",
          "decay_steps = 100",
          "local_step = 50",
          "",
          "ratio = local_step / decay_steps",
          "cosine_factor = 0.5 * (1 + math.cos(math.pi * ratio))",
          "speed = end_speed + (start_speed - end_speed) * cosine_factor",
          "print(ratio)         # 0.5",
          "print(cosine_factor) # 0.5"
        ])}
        <div class="adv-map">
          <h4>独立例子如何迁移回 Notebook</h4>
          <ul>
            <li><code>local_step</code> 对应全局 step 减去 warmup 与 stable 的 <code>decay_step</code>。</li>
            <li><code>ratio</code> 对应 <code>decay_ratio</code>，分母使用 <code>self.num_decay_steps</code>。</li>
            <li><code>start_speed / end_speed</code> 对应 <code>base_lr / min_lr</code>。</li>
            <li>完成后把得到的 <code>current_lr</code> 留给已有 <code>lrs.append(current_lr)</code>，不要在分支中提前 return。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "Decay 刚开始时 decay_step=0。",
        question: "此时 cosine_decay 与 current_lr 分别应是什么？",
        options: ["1 与 base_lr", "0 与 min_lr", "0.5 与两者中点"],
        answer: 0,
        revealNote: "ratio=0，cos(0)=1，所以余弦系数为 1，区间映射回 base_lr。"
      },
      checkpoint: checkpoint(
        "为什么 current_lr 公式最后要加 min_lr？",
        ["把 0 到区间宽度的结果整体平移，使最低点是 min_lr 而不是 0", "让 step 变成整数", "为了替代 math.cos"],
        0,
        "余弦系数降到 0 时，乘法项消失；加上的 min_lr 就成为最终下限。"
      ),
      homework: [
        "完成 TODO 3：计算 Decay 局部步数、0–1 进度、余弦系数，再把它映射到 [min_lr,base_lr]。",
        "手算自检：ratio=0、0.5、1 时，current_lr 分别是 base_lr、区间中点、min_lr。",
        "运行完整测试：首项、Warmup 边界、Stable 末端和最后 min_lr 四个断言都要通过；绘图单元会创建后关闭图像。",
        "错误诊断：末值到 0 时检查是否漏加 min_lr；一进入 Decay 就骤降时检查是否忘减前两段；曲线方向相反时检查余弦系数公式。"
      ]
    })
  ]
};
