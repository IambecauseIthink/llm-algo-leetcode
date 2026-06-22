const esc = (value) => String(value)
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;");

const code = (title, lines) => `<div class="syntax-card"><h4>语法热身：${esc(title)}</h4><pre><code>${lines.map(esc).join("\n")}</code></pre></div>`;

const checkpoint = (question, options, answer, explain) => ({ question, options, answer, explain });

const lesson = ({ id, title, todo, prerequisite, intuition, exampleHtml, syntaxHtml, predict, checkpoint, homework }) => ({
  id,
  title,
  todo,
  prerequisite,
  intuition,
  exampleHtml,
  syntaxHtml,
  predict,
  checkpoint,
  homework
});

module.exports = {
  "06": [
    lesson({
      id: "moe-router-softmax-topk",
      title: "Router 打分：每个 token 先看所有专家，再选 Top-K",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "MoE 的目标是“大容量、低激活”：模型有很多专家，但每个 token 只去少数几个专家那里计算。",
        "Router 是一个很小的线性层，把每个 token 的 hidden state 映射成 num_experts 个 logit 分数。",
        "本 notebook 的实现路径是：先对所有专家做 softmax 得到完整概率表，再从概率里取 Top-K，最后对 Top-K 权重重归一化。",
        "Router 不负责真正执行专家网络，它只输出两张清单：选了哪些专家 selected_experts，以及每个专家拿多少权重 routing_weights。"
      ],
      intuition: "把 Router 想成分诊台。每个 token 先拿自己的特征名片给所有专家打分，softmax 把分数变成全局概率，再只保留最适合的 K 个专家。Notebook 的 TODO 1/2 本质就是这两步。初学时先盯住一行：一行代表一个 token，这一行里所有列代表它对所有专家的偏好。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先读 forward 的输入输出合同</strong>
              <p><code>TopKRouter.forward</code> 输入是 [batch_size, seq_len, hidden_size]，但输出不再按 batch/seq 分组，而是把所有 token 排成一列：<code>routing_weights</code> 和 <code>selected_experts</code> 都是 [batch_size * seq_len, top_k]。这就是后面 SparseMoEBlock 能按专家找 token 的原因。</p>
              <div class="mini-table">
                <span>输入</span><strong>hidden_states [B,S,H]</strong>
                <span>输出 1</span><strong>routing_weights [B*S,K]</strong>
                <span>输出 2</span><strong>selected_experts [B*S,K]</strong>
              </div>
            </div>
            <div class="story-arrow">第一步：把 batch 和 seq 合并，变成 token 队列</div>
            <div class="story-panel">
              <strong>1. 先把 [B,S,H] 展平成 token 列表</strong>
              <p>Router 不是一次处理“一整句话”，而是逐 token 分配专家。输入 hidden_states 是 [batch, seq, hidden]，展平后变成 [num_tokens, hidden]，其中 <code>num_tokens = batch * seq</code>。</p>
              <div class="mini-flow"><span>hidden_states<br>[2,4,16]</span><span>view(-1,16)</span><strong>flat tokens<br>[8,16]</strong></div>
              <details class="think">
                <summary>为什么 Router 要按 token 展平，而不是按整句话路由？</summary>
                <div class="think-body">
                  <p>因为同一句话里的不同 token 可能需要不同专家。比如数字 token 可能更需要算术专家，代码 token 可能更需要语法专家。</p>
                  <p>把 [B,S,H] 展平成 [B*S,H] 后，每一行都能独立得到自己的专家选择，MoE 才真正做到“稀疏激活”。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">每个 token 对所有专家打一排分数</div>
            <div class="story-panel">
              <strong>2. gate 输出 router_logits：[num_tokens, num_experts]</strong>
              <p>如果有 8 个 token、4 个专家，router_logits 就是一张 8×4 表。每一行是一枚 token 对全部专家的原始打分。</p>
              <svg viewBox="0 0 470 210" width="100%" style="max-width:640px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12" text-anchor="middle">
                  <text x="88" y="24" fill="#2563eb" font-weight="700">token↓ / expert→</text>
                  <text x="170" y="24">E0</text><text x="230" y="24">E1</text><text x="290" y="24">E2</text><text x="350" y="24">E3</text>
                  <text x="105" y="58" fill="#2563eb">t0</text><text x="105" y="93" fill="#2563eb">t1</text><text x="105" y="128" fill="#2563eb">t2</text><text x="105" y="163" fill="#2563eb">...</text>
                </g>
                <g font-size="12" text-anchor="middle">
                  <rect x="145" y="42" width="46" height="24" fill="#fff6e8" stroke="#e8b66f"></rect><text x="168" y="59">1.2</text>
                  <rect x="205" y="42" width="46" height="24" fill="#f8fbff" stroke="#b8c8e8"></rect><text x="228" y="59">0.1</text>
                  <rect x="265" y="42" width="46" height="24" fill="#fde68a" stroke="#d9a15a"></rect><text x="288" y="59">2.0</text>
                  <rect x="325" y="42" width="46" height="24" fill="#f8fbff" stroke="#b8c8e8"></rect><text x="348" y="59">-0.5</text>
                  <rect x="145" y="77" width="46" height="24" fill="#f8fbff" stroke="#b8c8e8"></rect><text x="168" y="94">0.4</text>
                  <rect x="205" y="77" width="46" height="24" fill="#fde68a" stroke="#d9a15a"></rect><text x="228" y="94">2.4</text>
                  <rect x="265" y="77" width="46" height="24" fill="#fff6e8" stroke="#e8b66f"></rect><text x="288" y="94">1.0</text>
                  <rect x="325" y="77" width="46" height="24" fill="#f8fbff" stroke="#b8c8e8"></rect><text x="348" y="94">0.0</text>
                  <rect x="145" y="112" width="46" height="24" fill="#f8fbff" stroke="#b8c8e8"></rect><text x="168" y="129">-1.0</text>
                  <rect x="205" y="112" width="46" height="24" fill="#fff6e8" stroke="#e8b66f"></rect><text x="228" y="129">0.9</text>
                  <rect x="265" y="112" width="46" height="24" fill="#f8fbff" stroke="#b8c8e8"></rect><text x="288" y="129">0.2</text>
                  <rect x="325" y="112" width="46" height="24" fill="#fde68a" stroke="#d9a15a"></rect><text x="348" y="129">1.8</text>
                </g>
                <text x="230" y="190" font-size="13" fill="#657184">一行 softmax，一行 top_k；不同 token 可以选不同专家</text>
              </svg>
            </div>
            <div class="story-arrow">本关写法：先完整概率表，再 topk</div>
            <div class="story-panel">
              <strong>3. 为什么 notebook 先写 softmax，再写 topk？</strong>
              <p>这样代码最直观：先得到每个 token 对所有专家的概率分布，再从这张概率表里挑最大的 K 个。它也方便后续课程继续统计专家使用率、负载均衡等信息。</p>
              <div class="ratio-board"><span>router_logits<br>[tokens,E]</span><span>softmax 概率表<br>[tokens,E]</span><strong>topk 后<br>weights/indices [tokens,K]</strong></div>
              <p>技术上要注意：如果后面一定会对 Top-K 权重重归一化，那么“topk logits 后在局部 softmax”和“全量 softmax 后 topk 再重归一化”对最终 Top-K 权重是等价的。本关按 notebook 的写法来做，不把另一种写法当成数学错误。</p>
              <p>这也是 TODO 1 强调 <code>router_logits.float()</code> 的原因：softmax 对数值很敏感，先转 FP32 更稳，最后再转回 hidden_states 的 dtype。</p>
              <details class="think">
                <summary>softmax 的 dim=-1 到底是什么意思？</summary>
                <div class="think-body">
                  <p>最后一维是 experts。<code>dim=-1</code> 表示对每个 token 的“专家分数行”做归一化，让这一行所有专家概率加起来等于 1。</p>
                  <p>如果维度选错，概率就会跨 token 混在一起，那就变成“不同 token 互相抢概率”，完全不是 Router 想要的行为。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">topk 同时给你“值”和“索引”</div>
            <div class="story-panel">
              <strong>4. routing_weights 是概率，selected_experts 是专家编号</strong>
              <p><code>torch.topk</code> 返回两个张量：第一个是 Top-K 的值，也就是被选专家的概率权重；第二个是这些值原本在专家维的下标，也就是专家 id。后面的聚合逻辑靠专家 id 找人，靠权重决定每个专家输出占多少。</p>
              <div class="mini-table">
                <span>routing_probs 一行</span><span>[0.05, 0.62, 0.08, 0.25]</span>
                <span>top_k=2 values</span><strong>[0.62, 0.25]</strong>
                <span>top_k=2 indices</span><strong>[1, 3]</strong>
              </div>
              <p style="color:#bd6516"><strong>易错点</strong>：notebook TODO 后面有 zero 占位初始化。写完 TODO 后，要让函数返回你真正算出的 <code>routing_weights</code> 和 <code>selected_experts</code>，不要被占位变量覆盖。</p>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：Router 是 MoE 的调度器</div>
              <p>MoE 的专家像很多并行工作台，Router 决定每个 token 送去哪几个工作台。调度得好，模型容量变大但每个 token 只激活少数专家；调度得差，热门专家拥堵、冷门专家闲置，吞吐和效果都会受影响。</p>
              <p>下一关的负载均衡损失，就是专门约束这个调度器不要总偏心同几个专家。</p>
            </div>
          </div>`,
      syntaxHtml: code("先用一行玩具分数练 softmax 和 topk", [
        "toy_logits = torch.tensor([[1.2, 0.1, 2.0, -0.5]])  # 1 个 token, 4 个专家",
        "",
        "# dim=-1：只在“专家这一行”里归一化，得到每个专家的概率",
        "toy_probs = F.softmax(toy_logits.float(), dim=-1)   # [1, 4]",
        "",
        "# topk 同时返回值(values)和下标(indices)",
        "toy_weights, toy_ids = torch.topk(toy_probs, k=2, dim=-1)",
        "print(toy_weights)  # 概率值，例如 [[0.62, 0.25]]",
        "print(toy_ids)      # 专家编号，例如 [[2, 0]]"
      ]) + `
        <div class="syntax-card">
          <h4>举一反三：把玩具张量换成 notebook 变量</h4>
          <p>玩具例子只有 1 个 token。notebook 只是把很多 token 排成 [tokens, experts]，同样沿最后一维做 softmax/topk。</p>
          <div class="mini-table">
            <span>toy_logits</span><strong>router_logits</strong>
            <span>toy_probs</span><strong>routing_probs</strong>
            <span>toy_weights</span><strong>routing_weights</strong>
            <span>toy_ids</span><strong>selected_experts</strong>
            <span>k=2</span><strong>self.top_k</strong>
          </div>
          <p class="syntax-tip">迁移口诀：先把输入展平成 [tokens, hidden]，gate 后得到 [tokens, experts]；之后所有 softmax/topk 都沿 <code>dim=-1</code>，因为最后一维就是专家维。</p>
        </div>`,
      predict: {
        hook: "Router 有多种等价写法。本关 notebook 明确要求先得到完整概率表，再从概率里取 Top-K。",
        question: "先判断：为什么 notebook 要先对全量 router_logits 做 softmax？",
        options: [
          "因为 torch.topk 不能处理 logits，只能处理概率",
          "因为本关要先得到 [tokens, experts] 的完整概率表，再按概率取 Top-K，并为后续负载统计保留清晰接口",
          "因为 softmax 会改变 tensor 的 shape，方便 topk 使用"
        ],
        answer: 1,
        revealNote: "按 notebook 写：F.softmax(router_logits.float(), dim=-1) 得到完整概率表，再 torch.topk。后续 TODO 3 会把 Top-K 权重重新归一化。"
      },
      checkpoint: checkpoint(
        "hidden_states shape 是 [2,4,16]，num_experts=8，top_k=2。展平并 topk 后 selected_experts 的 shape 是？",
        ["[8, 2]", "[2, 4, 8]", "[8, 16]"],
        0,
        "num_tokens = 2*4 = 8；每个 token 选 2 个专家，所以 selected_experts 是 [8,2]。"
      ),
      homework: [
        "TODO 1：对 router_logits.float() 在专家维度 dim=-1 做全局 softmax。",
        "TODO 2：用 torch.topk(routing_probs, self.top_k, dim=-1) 同时取 routing_weights（概率值）和 selected_experts（专家编号）。",
        "注意把 TODO 后面的 zero 占位返回替换掉，不要让它覆盖你算出的 routing_weights 和 selected_experts；这是本关最常见的低级坑。",
        "检查返回 shape：二者都应该是 [batch_size * seq_len, top_k]，专家索引必须落在 [0, num_experts) 内。"
      ]
    }),
    lesson({
      id: "moe-router-renorm-merge",
      title: "Top-K 之后：重归一化，再把专家输出加权合成",
      todo: "TODO 3 / 阅读已给聚合逻辑",
      prerequisite: [
        "全量 softmax 的一整行概率和为 1；但丢掉非 Top-K 专家后，留下来的概率和通常小于 1。",
        "重归一化就是把留下来的 Top-K 权重按比例放大，让每个 token 的这 K 个权重重新加和为 1。",
        "SparseMoEBlock 的聚合逻辑是：找到选中某专家的 token，跑这个专家，再乘对应权重，加回 final_hidden_states。",
        "同一个 token 会被 top_k 个专家各处理一次，所以 final_hidden_states 需要用累加，而不是赋值覆盖。"
      ],
      intuition: "Top-K 像只留下票数最高的两位专家，但他们原本只拿到了全体专家票数的一部分。重归一化把这部分票按比例摊成 100%，保证输出尺度稳定；聚合时每个专家贡献多少，由 routing weight 决定。把这段想成“先重新分配投票权，再按投票权合成专家意见”。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先分清本 Mission 的两条线</strong>
              <p>TODO 3 只要求你把 Top-K 权重重新归一化；后面的 SparseMoEBlock 聚合逻辑已经给好，主要让你读懂它。一个是“权重准备”，一个是“专家输出合成”，不要混成一件事。</p>
              <div class="mini-table">
                <span>你要补的</span><strong>routing_weights 重新除以每行 sum</strong>
                <span>你要读懂的</span><strong>where 找 token → expert 计算 → 加权累加</strong>
              </div>
            </div>
            <div class="story-arrow">先看为什么 Top-K 后的概率和不再是 1</div>
            <div class="story-panel">
              <strong>1. Top-K 权重为什么要重新除以 sum？</strong>
              <p>假设某 token 的全局概率是 [0.60, 0.27, 0.10, 0.03]，Top-2 留下 [0.60, 0.27]，和是 0.87。如果直接拿它们混合专家输出，结果整体会被乘小。重归一化后变成 [0.60/0.87, 0.27/0.87]，两者重新加和为 1。</p>
              <div class="ratio-board"><span>Top-2 原权重<br>.60 + .27 = .87</span><span>重归一化<br>.60/.87, .27/.87</span><strong>新权重<br>.69 + .31 = 1.00</strong></div>
              <details class="think">
                <summary>如果不重归一化，模型一定会崩吗？</summary>
                <div class="think-body">
                  <p>不一定立刻崩，但输出尺度会系统性变小：Top-K 丢掉的概率质量越多，专家混合结果越被压低。</p>
                  <p>深层网络最怕每层尺度悄悄漂移。重归一化相当于告诉模型：既然只留下 K 个专家，那这 K 个专家内部重新按 100% 分配贡献。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">keepdim=True 是为了让广播除法不丢轴</div>
            <div class="story-panel">
              <strong>2. 形状读法：sum 后还要保持 [tokens, 1]</strong>
              <p><code>routing_weights</code> 是 [tokens, top_k]。如果 <code>sum(dim=-1)</code> 得到 [tokens]，再相除容易因为维度不对出错；<code>keepdim=True</code> 保持成 [tokens, 1]，就能自动广播到每个 Top-K 槽位。</p>
              <div class="mini-flow"><span>weights<br>[8,2]</span><span>sum(..., keepdim=True)<br>[8,1]</span><strong>weights / sum<br>[8,2]</strong></div>
              <p style="color:#657184">测试里的 <code>weights.sum(dim=-1)</code> 接近 1，就是专门检查这一行有没有写对。</p>
            </div>
            <div class="story-arrow">读懂 SparseMoEBlock：where 找人，weight 乘输出，再累加</div>
            <div class="story-panel">
              <strong>3. selected_experts 不是概率，是“这个 token 去哪几个专家”</strong>
              <p>聚合代码遍历每个 expert_idx。<code>torch.where(selected_experts == expert_idx)</code> 会返回两列信息：哪些 token 选中了这个专家，以及它在 Top-K 的第几个槽位被选中。</p>
              <div class="mini-table">
                <span>selected_experts</span><span>[[2,0], [1,2], [3,1]]</span>
                <span>expert_idx=2</span><strong>token_idx=[0,1]<br>kth_expert=[0,1]</strong>
              </div>
              <p>然后用 <code>routing_weights[token_idx, kth_expert]</code> 拿到对应权重；<code>unsqueeze(-1)</code> 把 [n] 变成 [n,1]，才能乘到专家输出 [n, hidden] 的每个 hidden 维上。</p>
              <div class="matrix-flow"><span>expert_output<br>[n, hidden]</span><span>current_weight<br>[n, 1]</span><strong>加权后累加到<br>final_hidden_states[token_idx]</strong></div>
            </div>
            <div class="story-arrow">为什么是 +=：一个 token 会收集多个专家的贡献</div>
            <div class="story-panel">
              <strong>4. top_k=2 时，同一个 token 会被写回两次</strong>
              <p>如果 token 0 选了专家 2 和专家 0，那么遍历 expert_idx=2 时会给 token 0 加一次输出；遍历 expert_idx=0 时还会再给 token 0 加一次输出。最终结果是两个专家输出的加权和，所以这里必须是 <code>+=</code> 累加。</p>
              <div class="sum-flow">
                <b>token0 输出</b>
                <em>expert2(x0) * w0,0</em>
                <em>expert0(x0) * w0,1</em>
                <strong>两份贡献相加</strong>
              </div>
              <p style="color:#bd6516"><strong>易错点</strong>：如果写成赋值覆盖，后一个专家会把前一个专家的贡献抹掉，top_k 就退化得很奇怪。</p>
            </div>
            <div class="story-arrow">最后按测试反查三件事</div>
            <div class="story-panel">
              <strong>5. Router 测试其实在帮你做 code review</strong>
              <p>本关测试不关心专家内容学得好不好，它只检查路由协议是否成立：shape 对、权重和为 1、专家 id 合法、MoE 输出回到原 shape。</p>
              <div class="mini-table">
                <span>weights / indices shape</span><strong>[batch_size * seq_len, top_k]</strong>
                <span>weights.sum(dim=-1)</span><strong>每行接近 1</strong>
                <span>final output</span><strong>回到 [batch_size, seq_len, hidden_size]</strong>
              </div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：玩具循环易懂，工业实现会先排队再批处理</div>
              <p>这里用 for 循环遍历专家，是为了让逻辑透明。真实推理框架会把同一个专家的 token 先排序聚到一起，一次性组成大 batch 喂给专家网络，减少小矩阵调用带来的 GPU 空转。</p>
              <p>但无论实现多复杂，核心协议都没变：Router 给专家 id 和权重，专家产出向量，最后按权重加回对应 token。</p>
            </div>
          </div>`,
      syntaxHtml: code("先用一行权重练 keepdim=True", [
        "toy_weights = torch.tensor([[0.60, 0.27], [0.40, 0.10]])  # [tokens=2, top_k=2]",
        "",
        "# keepdim=True：sum 后保留成 [2, 1]，才能逐行广播除回 [2, 2]",
        "row_sum = toy_weights.sum(dim=-1, keepdim=True)            # [[0.87], [0.50]]",
        "toy_normed = toy_weights / row_sum                         # 每行和都变成 1",
        "print(toy_normed.sum(dim=-1))                               # tensor([1., 1.])"
      ]) + code("再用玩具 selected_experts 读懂 where 和 unsqueeze", [
        "selected = torch.tensor([[2, 0], [1, 2], [3, 1]])  # [tokens=3, top_k=2]",
        "expert_idx = 2",
        "",
        "# where 返回两个坐标：第几个 token、第几个 top-k 槽位",
        "token_idx, kth = torch.where(selected == expert_idx)",
        "print(token_idx)  # tensor([0, 1])",
        "print(kth)        # tensor([0, 1])",
        "",
        "weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])",
        "w = weights[token_idx, kth].unsqueeze(-1)  # [2] -> [2, 1]，才能乘 [2, hidden]"
      ]) + `
        <div class="syntax-card">
          <h4>举一反三：把玩具写法搬回 notebook</h4>
          <p>TODO 3 只需要第一张卡的重归一化写法。第二张卡是为了读懂后面已经给出的专家聚合代码。</p>
          <div class="mini-table">
            <span>toy_weights</span><strong>routing_weights</strong>
            <span>selected</span><strong>selected_experts</strong>
            <span>expert_idx</span><strong>for 循环里的 expert_idx</strong>
            <span>w</span><strong>current_weight</strong>
          </div>
          <p class="syntax-tip">迁移口诀：归一化看最后一维 <code>top_k</code>，所以 <code>sum(dim=-1, keepdim=True)</code>；聚合时权重先从 [n] 变 [n,1]，才能广播乘到专家输出的 hidden 维。</p>
        </div>`,
      predict: {
        hook: "Top-K 后，两个留下的概率可能是 0.60 和 0.27。它们已经是概率了，看起来可以直接用。",
        question: "先判断：为什么还要做 routing_weights / routing_weights.sum(..., keepdim=True)？",
        options: [
          "为了把 expert 索引也变成概率",
          "为了让 Top-K 留下的权重重新加和为 1，避免输出尺度被整体压小",
          "为了把 [tokens, top_k] 改成 [tokens, hidden]"
        ],
        answer: 1,
        revealNote: "Top-K 丢掉了部分概率质量，留下的权重和通常小于 1。重归一化后，每个 token 的专家混合仍然是稳定的加权平均。"
      },
      checkpoint: checkpoint(
        "routing_weights[token_idx, kth_expert] 的 shape 是 [n]，要乘 expert 输出 [n, hidden]，应该先做什么？",
        ["unsqueeze(-1) 变成 [n,1]", "argmax() 变成一个专家编号", "flatten() 变成 [n*hidden]"],
        0,
        "专家输出有 hidden 维，权重需要 [n,1] 才能沿 hidden 维广播，给每个 token 的整条向量乘同一个 routing weight。"
      ),
      homework: [
        "TODO 3：用 routing_weights / routing_weights.sum(dim=-1, keepdim=True) 完成 Top-K 权重重归一化。",
        "按顺序阅读 SparseMoEBlock：final_hidden_states 初始化、flat_hidden_states 展平、遍历专家、torch.where 找 token、取权重 unsqueeze、加权累加。",
        "跑 notebook 测试：weights.sum(dim=-1) 应接近 1，专家索引不能越界，输出 shape 应回到原来的 [batch_size, seq_len, hidden_size]。"
      ]
    })
  ],
  "07": [
    lesson({
      id: "moe-balance-token-frequency",
      title: "负载均衡先数人头",
      todo: "TODO 2",
      prerequisite: [
        "MoE 如果总选同一个专家，就会出现拥堵。",
        "f_i 表示第 i 个专家在所有 Top-K 选择里出现的比例。",
        "one_hot 可以把专家 id 转成按专家统计的计数表。"
      ],
      intuition: "像给多个窗口排队：只看 router 觉得谁好还不够，还要统计每个窗口真正排了多少人。",
      exampleHtml: `<div class="mini-table"><span>选中专家</span><span>[0, 2, 2, 1]</span><span>one-hot 计数</span><strong>f=[1/4, 1/4, 2/4]</strong></div>`,
      syntaxHtml: code("one_hot 计数", [
        "ids = torch.tensor([0, 2, 2, 1])",
        "hot = F.one_hot(ids, num_classes=3).float()",
        "freq = hot.mean(dim=0)"
      ]),
      checkpoint: checkpoint(
        "如果每个 token 只选 1 个专家，4 个 token 分给 [0,2,2,1]，专家 2 的 f_i 是多少？",
        ["0.5", "0.25", "2.0"],
        0,
        "专家 2 收到 2 个 token，占 4 个 token 的一半。"
      ),
      homework: [
        "把 selected_experts 转成 one_hot。",
        "沿 token 和 top_k 两个选择维统计每个专家出现次数。",
        "检查所有 f_i 的和接近 1。"
      ]
    }),
    lesson({
      id: "moe-balance-prob-loss",
      title: "P_i 看偏好，f_i 看实际分配",
      todo: "TODO 1 / TODO 3",
      prerequisite: [
        "P_i 是 Top-K 路由权重按专家累加后的平均得分。",
        "f_i 是第 i 个专家实际被选中的比例。",
        "辅助损失会惩罚概率偏好和实际分配过于集中。"
      ],
      intuition: "P_i 像问卷里的偏好，f_i 像真实排队人数；两者都均匀，专家才不会冷热不均。",
      exampleHtml: `<div class="ratio-board"><span>P=[.33, .34, .33]</span><span>f=[.25, .25, .50]</span><strong>loss 关注 P 和 f 的乘积和</strong></div>`,
      syntaxHtml: code("scatter_add 按专家累加", [
        "p_i = torch.zeros(num_experts, device=routing_weights.device)",
        "p_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())",
        "p_i = p_i / (total_tokens * top_k)",
        "aux_loss = alpha * num_experts * (f_i * p_i).sum()"
      ]),
      checkpoint: checkpoint(
        "如果一个专家 P_i 很高且 f_i 也很高，辅助损失会如何看待它？",
        ["说明负载集中，需要惩罚", "说明这个专家应被删除", "说明 softmax 失效"],
        0,
        "负载均衡希望专家被更平均地使用，高偏好和高占用同时出现会推高惩罚项。"
      ),
      homework: [
        "用 scatter_add_ 把 routing_weights 累加到对应专家，得到 P_i。",
        "用 one_hot 统计 f_i。",
        "按 notebook 公式组合 P_i、f_i 和专家数量得到辅助损失。"
      ]
    })
  ],
  "08": [
    lesson({
      id: "gemma-rmsnorm-plus-one",
      title: "Gemma 的 +1 缩放",
      todo: "TODO 1",
      prerequisite: [
        "RMSNorm 输出会再乘一个可学习缩放向量。",
        "普通写法直接学习 weight。",
        "Gemma 写法学习的是 weight + 1。"
      ],
      intuition: "把默认缩放设为 1，参数只负责学“我要比默认多一点还是少一点”。",
      exampleHtml: `<div class="scale-demo"><span>normalized x</span><span>weight=[0.1, -0.2]</span><strong>scale=1+weight=[1.1, 0.8]</strong></div>`,
      syntaxHtml: code("参数和常数相加", [
        "gain_delta = nn.Parameter(torch.zeros(4))",
        "scale = 1.0 + gain_delta",
        "y = x * scale"
      ]),
      checkpoint: checkpoint(
        "Gemma RMSNorm 中，如果 weight 初始化为 0，实际缩放是多少？",
        ["1", "0", "hidden_size"],
        0,
        "实际使用的是 1 + weight，所以初始化为 0 时等价于不改变尺度。"
      ),
      homework: [
        "实现归一化后的 (1 + weight) 缩放。",
        "确认初始化时输出不会被缩放到 0。",
        "用测试检查 Gemma 写法和预期输出一致。"
      ]
    }),
    lesson({
      id: "qwen-tie-embeddings",
      title: "权重绑定：输入词表和输出词表共用一张表",
      todo: "TODO 2",
      prerequisite: [
        "Embedding 权重把 token id 映射成向量。",
        "lm_head 权重把 hidden 向量映射回 vocab logits。",
        "权重绑定要求两个模块指向同一块 Parameter。"
      ],
      intuition: "读词和猜词使用同一套词典坐标，既省参数，也让输入输出空间保持一致。",
      exampleHtml: `<div class="lookup"><span>embed_tokens.weight</span><strong>同一块内存</strong><span>lm_head.weight</span></div>`,
      syntaxHtml: code("让两个属性指向同一个权重", [
        "emb = nn.Embedding(100, 16)",
        "head = nn.Linear(16, 100, bias=False)",
        "head.weight = emb.weight",
        "assert head.weight is emb.weight"
      ]),
      checkpoint: checkpoint(
        "判断权重绑定是否真的共享内存，最直接的检查是？",
        ["lm_head.weight is embed_tokens.weight", "lm_head.weight.shape == embed_tokens.weight.shape", "两个模块名字相同"],
        0,
        "shape 相同不代表共享；is 检查两个引用是否指向同一对象。"
      ),
      homework: [
        "把 lm_head.weight 直接绑定到 embed_tokens.weight。",
        "用 is 检查是否同一块 Parameter。",
        "确认 forward 的 vocab logits shape 没有改变。"
      ]
    })
  ],
  "09": [
    lesson({
      id: "sft-label-mask",
      title: "SFT labels：只让回答部分参与损失",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "SFT 数据通常由 prompt 和 answer 拼接而成。",
        "prompt 是条件，不应该被当成要学习复述的答案。",
        "CrossEntropyLoss 的 ignore_index 常用 -100。"
      ],
      intuition: "题目部分像试卷题干，模型读它；答案部分才是要批改的作答。",
      exampleHtml: `<div class="mini-table"><span>tokens</span><span>[题, 题, 答, 答]</span><span>labels</span><strong>[-100, -100, 答, 答]</strong></div>`,
      syntaxHtml: code("列表拼接、截断和填充", [
        "ids = prompt_ids + answer_ids",
        "labels = [-100] * len(prompt_ids) + answer_ids",
        "ids = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))",
        "labels = labels[:max_len] + [-100] * max(0, max_len - len(labels))"
      ]),
      checkpoint: checkpoint(
        "在 SFT 中，prompt 对应的 label 通常设成什么？",
        ["-100", "pad_id", "prompt token 自己"],
        0,
        "-100 会被 PyTorch 的交叉熵忽略，避免训练模型复述题干。"
      ),
      homework: [
        "构造 input_ids 时拼接 prompt 和 answer。",
        "构造 labels 时把 prompt 位置设为 -100。",
        "按 max_length 做截断和 padding。"
      ]
    }),
    lesson({
      id: "sft-shift-ce",
      title: "Shift 对齐：当前位置预测下一个 token",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "自回归模型用前面的 token 预测下一个 token。",
        "logits 的时间步 t 对应 label 的时间步 t+1。",
        "交叉熵通常需要展平成 [N, vocab] 和 [N]。"
      ],
      intuition: "像遮住下一格让模型猜：第 0 个位置的输出，应该和第 1 个 token 对齐。",
      exampleHtml: `<div class="timeline"><span>logits: 0 1 2</span><span>labels: 1 2 3</span><strong>最后一个 logits 没有下一个 label</strong></div>`,
      syntaxHtml: code("切片和 reshape", [
        "shift_logits = logits[:, :-1, :]",
        "shift_labels = labels[:, 1:]",
        "loss = F.cross_entropy(",
        "    shift_logits.reshape(-1, vocab_size),",
        "    shift_labels.reshape(-1),",
        "    ignore_index=-100,",
        ")"
      ]),
      checkpoint: checkpoint(
        "logits shape 是 [2,5,10]，shift_logits = logits[:, :-1, :] 后 shape 是？",
        ["[2,4,10]", "[2,5,9]", "[1,5,10]"],
        0,
        "去掉最后一个时间步，序列长度从 5 变成 4。"
      ),
      homework: [
        "把 logits 去掉最后一个时间步。",
        "把 labels 去掉第一个时间步。",
        "展平后调用 cross_entropy，并保留 ignore_index=-100。"
      ]
    })
  ],
  "10": [
    lesson({
      id: "lora-low-rank-adapter",
      title: "LoRA：冻结大矩阵，只训练低秩旁路",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "普通 Linear 的权重 shape 是 [out_features, in_features]。",
        "LoRA 用两个小矩阵 A 和 B 近似一个大更新矩阵。",
        "主权重通常冻结，只训练 A/B。"
      ],
      intuition: "不直接改整面墙，只在旁边贴一张可训练的小补丁；补丁由 A 先降维、B 再升维组成。",
      exampleHtml: `<div class="matrix-flow"><span>x</span><span>A: in to r</span><span>B: r to out</span><strong>加到 base linear</strong></div>`,
      syntaxHtml: code("定义可训练和不可训练参数", [
        "self.weight = nn.Parameter(torch.empty(out_features, in_features))",
        "self.weight.requires_grad = False",
        "self.lora_A = nn.Parameter(torch.empty(r, in_features))",
        "self.lora_B = nn.Parameter(torch.empty(out_features, r))"
      ]),
      checkpoint: checkpoint(
        "LoRA 中 r 的主要作用是什么？",
        ["控制低秩旁路的瓶颈大小", "控制 batch size", "控制 vocab size"],
        0,
        "r 越小，可训练参数越少；r 越大，旁路表达能力更强。"
      ),
      homework: [
        "在 __init__ 中创建主权重和 LoRA A/B。",
        "按 notebook 要求初始化权重。",
        "确认只有 LoRA 旁路参数需要训练。"
      ]
    }),
    lesson({
      id: "lora-forward-merge",
      title: "前向时相加，推理时可以合并",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "LoRA 输出是 base_out + lora_out。",
        "lora_out 通常要乘 alpha / r 的缩放。",
        "合并权重就是把低秩更新加回主权重。"
      ],
      intuition: "训练时走两条路方便只改小补丁；部署时把补丁贴回主权重，前向就不多绕路。",
      exampleHtml: `<div class="branches"><span>base: xW</span><span>LoRA: x A^T B^T * scale</span><strong>输出相加</strong></div>`,
      syntaxHtml: code("线性层和权重合并", [
        "base = F.linear(x, self.weight, self.bias)",
        "update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling",
        "merged_weight = self.weight + self.lora_B @ self.lora_A * self.scaling"
      ]),
      checkpoint: checkpoint(
        "合并 LoRA 权重后，推理时还需要单独计算 LoRA 旁路吗？",
        ["不需要，更新已加进主权重", "需要，否则 shape 会变", "只在 batch size 为 1 时需要"],
        0,
        "合并后主权重已经包含低秩更新，推理可以只走普通 Linear。"
      ),
      homework: [
        "实现 base linear 和 LoRA 旁路相加。",
        "注意 LoRA 旁路的缩放系数。",
        "实现 merge 权重并验证合并前后输出接近。"
      ]
    })
  ],
  "11": [
    lesson({
      id: "wsd-warmup-stable",
      title: "Warmup 和 Stable：先慢慢点火，再保持巡航",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "学习率决定每次参数更新步子有多大。",
        "训练刚开始梯度不稳定，warmup 常从 0 线性升到 base_lr。",
        "stable 阶段保持 base_lr 不变。"
      ],
      intuition: "开车先平稳起步，再进入匀速巡航；学习率也先从小到大，再保持一段时间。",
      exampleHtml: `<div class="curve-legend"><span>step 0: 0</span><span>warmup end: base_lr</span><strong>stable: base_lr</strong></div>`,
      syntaxHtml: code("用 if/elif 划阶段", [
        "if step < warmup_steps:",
        "    lr = base_lr * step / warmup_steps",
        "elif step < stable_steps:",
        "    lr = base_lr",
        "else:",
        "    lr = decay_lr"
      ]),
      checkpoint: checkpoint(
        "base_lr=0.01，warmup_steps=100，step=50 时线性 warmup 学习率是多少？",
        ["0.005", "0.01", "0.05"],
        0,
        "走到 warmup 的一半，所以学习率是 base_lr 的一半。"
      ),
      homework: [
        "在 warmup 区间写出线性增长公式。",
        "在 stable 区间直接返回 base_lr。",
        "检查边界 step 是否落在正确阶段。"
      ]
    }),
    lesson({
      id: "wsd-cosine-decay",
      title: "Cosine Decay：最后平滑降落到 min_lr",
      todo: "TODO 3",
      prerequisite: [
        "decay 阶段要知道已经走过 decay 的比例 progress。",
        "cos(pi * progress) 从 1 平滑变到 -1。",
        "min_lr 是训练末尾保留的最低学习率。"
      ],
      intuition: "最后不是突然刹车，而是沿着半个余弦曲线平滑降速。",
      exampleHtml: `<div class="timeline"><span>progress=0: base_lr</span><span>progress=.5: 中间值</span><span>progress=1: min_lr</span></div>`,
      syntaxHtml: code("余弦衰减比例", [
        "progress = (step - decay_start) / decay_steps",
        "cosine = 0.5 * (1 + math.cos(math.pi * progress))",
        "lr = min_lr + (base_lr - min_lr) * cosine"
      ]),
      checkpoint: checkpoint(
        "cosine decay 的 progress 应该限制在哪个范围内最安全？",
        ["0 到 1", "-1 到 1", "任意大于 1"],
        0,
        "progress 表示 decay 阶段完成比例，通常夹在 0 到 1。"
      ),
      homework: [
        "计算 decay 阶段的 progress。",
        "用 cosine 公式从 base_lr 平滑降到 min_lr。",
        "处理超过总步数后的学习率下界。"
      ]
    })
  ],
  "12": [
    lesson({
      id: "ppo-ratio",
      title: "PPO Ratio：新策略相对旧策略变了多少",
      todo: "TODO 1",
      prerequisite: [
        "log_prob 是概率取 log 后的值。",
        "log(a) - log(b) 等于 log(a/b)。",
        "exp(new_logprob - old_logprob) 得到概率比率。"
      ],
      intuition: "ratio 像“新方案相对旧方案的加码倍数”：大于 1 表示新策略更愿意选这个动作。",
      exampleHtml: `<div class="ratio-board"><span>old prob=.20</span><span>new prob=.30</span><strong>ratio=1.5</strong></div>`,
      syntaxHtml: code("从 log 概率还原比率", [
        "old_logp = torch.log(torch.tensor([0.2]))",
        "new_logp = torch.log(torch.tensor([0.3]))",
        "ratio = torch.exp(new_logp - old_logp)"
      ]),
      checkpoint: checkpoint(
        "new_prob=0.1，old_prob=0.2，ratio 是多少？",
        ["0.5", "2.0", "0.1"],
        0,
        "ratio = 0.1 / 0.2 = 0.5。"
      ),
      homework: [
        "用 exp(new_log_probs - old_log_probs) 计算 ratio。",
        "确认 ratio 和 advantages 可以逐元素相乘。",
        "用简单数值检查 ratio 方向是否正确。"
      ]
    }),
    lesson({
      id: "ppo-clip-loss",
      title: "Clip：限制策略一步别迈太大",
      todo: "TODO 2 / TODO 3 / TODO 4",
      prerequisite: [
        "advantage 为正时，想提高这个动作概率。",
        "advantage 为负时，想降低这个动作概率。",
        "clip 把 ratio 限制在 [1-eps, 1+eps]。"
      ],
      intuition: "PPO 像给更新装护栏：允许改进，但不允许新策略突然偏离旧策略太远。",
      exampleHtml: `<div class="clip-box"><span>ratio=1.8</span><span>eps=.2</span><strong>clipped=1.2</strong></div>`,
      syntaxHtml: code("clamp 和 min", [
        "raw = ratio * advantages",
        "clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages",
        "objective = torch.min(raw, clipped)",
        "loss = -objective.mean()"
      ]),
      checkpoint: checkpoint(
        "eps=0.2 时，ratio 会被限制在什么区间？",
        ["[0.8, 1.2]", "[0.2, 1.0]", "[-0.2, 0.2]"],
        0,
        "clip 区间是 [1-eps, 1+eps]。"
      ),
      homework: [
        "计算未截断 surrogate。",
        "计算 clamp 后的 surrogate。",
        "取二者 min 后加负号作为 loss。"
      ]
    })
  ],
  "13": [
    lesson({
      id: "dpo-implicit-reward",
      title: "DPO 隐式奖励：看新模型比参考模型多偏爱多少",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "chosen 是人类偏好的回答，rejected 是较差回答。",
        "policy_logps 来自当前模型。",
        "reference_logps 来自冻结参考模型。"
      ],
      intuition: "DPO 不训练一个单独奖励模型，而是用当前模型和参考模型的 logprob 差值当作隐式奖励。",
      exampleHtml: `<div class="pref-pair"><span>chosen: policy-ref = +1.2</span><span>rejected: policy-ref = +0.2</span><strong>chosen 优势更大</strong></div>`,
      syntaxHtml: code("成对样本相减", [
        "chosen_reward = chosen_policy_logp - chosen_ref_logp",
        "rejected_reward = rejected_policy_logp - rejected_ref_logp",
        "margin = chosen_reward - rejected_reward"
      ]),
      checkpoint: checkpoint(
        "DPO 中 chosen_reward 通常由哪两个量相减得到？",
        ["chosen_policy_logp - chosen_ref_logp", "chosen_loss - rejected_loss", "old_logp - new_logp"],
        0,
        "隐式奖励衡量当前策略相对参考策略对该回答提高了多少概率。"
      ),
      homework: [
        "分别计算 chosen 和 rejected 的 policy-reference 差值。",
        "确认 chosen/rejected 的 shape 可以相减。",
        "把 beta 作为奖励差距的缩放因子。"
      ]
    }),
    lesson({
      id: "dpo-logsigmoid",
      title: "LogSigmoid：把偏好差距变成可优化损失",
      todo: "TODO 3",
      prerequisite: [
        "DPO 关心 chosen_reward 是否大于 rejected_reward。",
        "差距越大，说明模型越符合偏好。",
        "-logsigmoid(diff) 会在 diff 大时变小。"
      ],
      intuition: "像判断一场二选一投票：chosen 赢得越明显，损失越低；输掉时损失会变高。",
      exampleHtml: `<div class="reward-gap"><span>diff 小于 0: 惩罚大</span><span>diff = 0: 不确定</span><strong>diff 大于 0: 损失小</strong></div>`,
      syntaxHtml: code("稳定的 logsigmoid", [
        "diff = beta * (chosen_reward - rejected_reward)",
        "loss_per_pair = -F.logsigmoid(diff)",
        "loss = loss_per_pair.mean()"
      ]),
      checkpoint: checkpoint(
        "如果 chosen_reward 明显大于 rejected_reward，DPO loss 应该怎样？",
        ["更小", "更大", "不变"],
        0,
        "chosen 相对 rejected 越占优，-logsigmoid(diff) 越接近 0。"
      ),
      homework: [
        "计算 beta * (chosen_reward - rejected_reward)。",
        "用 -F.logsigmoid 得到每对样本损失。",
        "对 batch 求 mean。"
      ]
    })
  ],
  "14": [
    lesson({
      id: "attn-backward-pv",
      title: "Attention 反传第一步：O = P @ V",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "前向输出 O 由注意力概率 P 乘 value V 得到。",
        "矩阵乘法反传时，左边和右边都会收到梯度。",
        "dV 和 dP 的 shape 要分别对齐 V 和 P。"
      ],
      intuition: "先从最后的混合结果倒推：输出对 V 的依赖像加权求和，输出对 P 的依赖像每个 value 被选了多少。",
      exampleHtml: `<div class="matrix-flow"><span>dO</span><span>dV = P^T @ dO</span><span>dP = dO @ V^T</span></div>`,
      syntaxHtml: code("矩阵乘法的转置位置", [
        "dV = P.transpose(-2, -1) @ dO",
        "dP = dO @ V.transpose(-2, -1)"
      ]),
      checkpoint: checkpoint(
        "O = P @ V 的反传中，dV 应该使用哪个公式？",
        ["P.transpose(-2,-1) @ dO", "dO @ P", "V @ dO"],
        0,
        "对右矩阵 V 的梯度等于左矩阵 P 的转置乘输出梯度。"
      ),
      homework: [
        "从 ctx.saved_tensors 取出 P 和 V。",
        "写出 dV 和 dP 的矩阵乘法。",
        "检查 dV shape 与 V 一致。"
      ]
    }),
    lesson({
      id: "attn-softmax-qk",
      title: "穿过 Softmax，再回到 Q 和 K",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "P = softmax(S)，softmax 反传不是简单逐元素乘。",
        "S = Q @ K^T * scale。",
        "dQ 和 dK 都要乘回 scale。"
      ],
      intuition: "梯度先穿过概率归一化层，再穿过打分矩阵；打分来自 Q 和 K 的相似度。",
      exampleHtml: `<div class="softmax-row"><span>dP</span><span>softmax backward</span><span>dS</span><strong>dQ, dK</strong></div>`,
      syntaxHtml: code("softmax backward 的常见写法", [
        "dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))",
        "dQ = (dS @ K) * scale",
        "dK = (dS.transpose(-2, -1) @ Q) * scale"
      ]),
      checkpoint: checkpoint(
        "softmax 反传里为什么要用 keepdim=True 求和？",
        ["保持可广播维度", "删除 batch 维", "让梯度停止传播"],
        0,
        "最后一维求和后保留维度，才能和 P、dP 按元素广播运算。"
      ),
      homework: [
        "用 P * (dP - sum(dP*P)) 写 dS。",
        "用 dS @ K 写 dQ。",
        "用 dS^T @ Q 写 dK，并乘以 scale。"
      ]
    })
  ],
  "15": [
    lesson({
      id: "flash-tiling-max",
      title: "FlashAttention 分块：不一次存完整注意力矩阵",
      todo: "TODO 1 / TODO 2 / TODO 3",
      prerequisite: [
        "标准 attention 会生成 [seq, seq] 的分数矩阵。",
        "FlashAttention 按块读取 K/V，减少显存占用。",
        "softmax 为了数值稳定，需要维护最大值 m。"
      ],
      intuition: "像分批扫描一大张表：每次只读一块，但要记住目前见过的最大分数，保证 softmax 稳定。",
      exampleHtml: `<div class="tile-grid"><span>Q block</span><span>K/V block 1</span><span>K/V block 2</span><strong>滚动更新 m</strong></div>`,
      syntaxHtml: code("分块循环", [
        "for start in range(0, seq_len, block_size):",
        "    end = min(start + block_size, seq_len)",
        "    k_block = K[start:end]",
        "    scores = q @ k_block.T",
        "    m_new = torch.maximum(m, scores.max(dim=-1).values)"
      ]),
      checkpoint: checkpoint(
        "FlashAttention 为什么要维护 m_new？",
        ["为了稳定计算 exp，避免溢出", "为了改变 hidden_size", "为了跳过 V 矩阵"],
        0,
        "在线 softmax 需要全局最大值来稳定指数计算，即使分块也不能丢。"
      ),
      homework: [
        "初始化 O、m、l 等全局状态。",
        "在每个块中计算局部分数和 m_block。",
        "用 m_new 更新当前全局最大值。"
      ]
    }),
    lesson({
      id: "flash-online-output",
      title: "Online Softmax：旧输出也要按新最大值重缩放",
      todo: "TODO 4 / TODO 5 / TODO 6",
      prerequisite: [
        "分块 softmax 不能只算当前块，还要合并过去块的贡献。",
        "当全局最大值变了，旧的指数和 l 要重新缩放。",
        "输出 O 也要跟着同一比例调整。"
      ],
      intuition: "新块来了以后，如果发现更大的分数，旧账本要按新的标尺重算，才能和新块公平相加。",
      exampleHtml: `<div class="online-book"><span>旧 l, O</span><span>新块 l_block, O_block</span><strong>按 m_old/m_new 修正后合并</strong></div>`,
      syntaxHtml: code("带状态的累积更新", [
        "old_scale = torch.exp(m_old - m_new)",
        "l_new = l_old * old_scale + l_block",
        "old_part = O_old * (l_old * old_scale / l_new)",
        "new_part = (P_ij @ v_block) / l_new",
        "O_new = old_part + new_part"
      ]),
      checkpoint: checkpoint(
        "当 m_new 大于旧 m 时，旧的 exp 和输出为什么要缩放？",
        ["因为 softmax 的参考最大值变了", "因为 block_size 变了", "因为 V 的 dtype 变了"],
        0,
        "exp(score-m) 的 m 变了，同一个 score 的指数值也会变，旧贡献必须换到同一标尺。"
      ),
      homework: [
        "用 P_ij = exp(S_ij - m_new) 计算当前块贡献。",
        "用 l_old * exp(m_old - m_new) + l_block 更新 l_new。",
        "按 notebook 的 online softmax 公式更新 O_i。"
      ]
    })
  ],
  "16": [
    lesson({
      id: "decode-temperature",
      title: "Temperature：调节 logits 的锋利程度",
      todo: "TODO 1",
      prerequisite: [
        "logits 还不是概率，需要 softmax 后才能采样。",
        "temperature 越小，分布越尖锐。",
        "temperature 不能为 0，否则会除零。"
      ],
      intuition: "温度像创作旋钮：低温更保守，高温更发散；但旋钮不能拧到 0。",
      exampleHtml: `<div class="scale-demo"><span>logits / 0.5</span><span>差距变大</span><strong>更偏向最高分 token</strong></div>`,
      syntaxHtml: code("防止除零", [
        "temperature = max(float(temperature), 1e-6)",
        "scaled_logits = logits / temperature",
        "probs = torch.softmax(scaled_logits, dim=-1)"
      ]),
      checkpoint: checkpoint(
        "temperature 变小后，softmax 分布通常会怎样？",
        ["更尖锐", "更平均", "shape 变短"],
        0,
        "除以小温度会放大 logit 差距，最高分 token 的概率更高。"
      ),
      homework: [
        "给 temperature 加最小值保护。",
        "用 logits / temperature 得到缩放后的分数。",
        "再进入 Top-K 或 Top-P 过滤。"
      ]
    }),
    lesson({
      id: "decode-candidate-filter",
      title: "Top-K 和 Top-P：先筛候选，再采样",
      todo: "TODO 2 / TODO 3",
      prerequisite: [
        "Top-K 保留概率最高的 K 个 token。",
        "Top-P 保留累计概率达到 p 的最小候选集合。",
        "被过滤掉的 token 常用 -inf 屏蔽。"
      ],
      intuition: "Top-K 像固定只看前 K 名；Top-P 像把候选加到概率够一篮子为止。",
      exampleHtml: `<div class="mini-table"><span>概率排序</span><span>.50 .25 .15 .10</span><span>top_p=.8</span><strong>保留前三个</strong></div>`,
      syntaxHtml: code("排序、累计和 mask", [
        "sorted_logits, sorted_idx = torch.sort(logits, descending=True)",
        "cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)",
        "remove = cumulative > top_p",
        "remove[..., 1:] = remove[..., :-1].clone()",
        "remove[..., 0] = 0",
        "sorted_logits[remove] = float('-inf')",
        "filtered_logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)"
      ]),
      checkpoint: checkpoint(
        "Top-K 过滤后，被淘汰 token 的 logits 常设为？",
        ["-inf", "0", "1"],
        0,
        "softmax(-inf) 约为 0，表示不会被采样到。"
      ),
      homework: [
        "用 torch.topk 找到 Top-K 阈值并屏蔽其他 token。",
        "排序后用 softmax + cumsum 实现 Top-P 候选集合。",
        "平移移除 mask 后 scatter 回原始 token 顺序，再 softmax 和 multinomial。"
      ]
    })
  ],
  "17": [
    lesson({
      id: "paged-block-table",
      title: "PagedAttention：逻辑连续，物理可以不连续",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "KV Cache 会随序列长度增长。",
        "直接连续分配容易产生显存碎片。",
        "PagedAttention 用固定大小 block 管理缓存。"
      ],
      intuition: "像操作系统分页：请求看到的是连续的 token 序列，底层实际由一块块物理页拼起来。",
      exampleHtml: `<div class="tile-grid"><span>逻辑 token 0-3</span><span>block 7</span><span>逻辑 token 4-7</span><span>block 2</span></div>`,
      syntaxHtml: code("向上取整算 block 数", [
        "needed = (seq_len + block_size - 1) // block_size",
        "for _ in range(needed):",
        "    block_id = free_blocks.pop(0)",
        "    block_table.append(block_id)"
      ]),
      checkpoint: checkpoint(
        "seq_len=17，block_size=8，需要多少个 block？",
        ["3", "2", "17"],
        0,
        "17 个 token 需要覆盖 0-7、8-15、16 这三段。"
      ),
      homework: [
        "用向上取整计算 prefill 需要的 block 数。",
        "检查 free_blocks 是否足够，不够时抛出 OOM。",
        "从 free_blocks 分配 block_id 到 req.block_table。"
      ]
    }),
    lesson({
      id: "paged-decode-growth",
      title: "Decode 阶段：只有跨入新 block 才分配",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "自回归 decode 每次只新增 1 个 token。",
        "最后一个 block 没满时可以继续写入。",
        "根据 block_table 才能把物理块拼回逻辑序列。"
      ],
      intuition: "不要每来一个 token 就搬家；只有当前页写满，才申请下一页。",
      exampleHtml: `<div class="timeline"><span>len=8: block 满</span><span>len=9: 进入新 block</span><strong>此时分配 1 块</strong></div>`,
      syntaxHtml: code("取模判断边界", [
        "req.seq_len += 1",
        "need_new = (req.seq_len % block_size) == 1",
        "if need_new:",
        "    req.block_table.append(free_blocks.pop(0))",
        "blocks = [cache[i] for i in req.block_table]",
        "logical = torch.cat(blocks, dim=0)[:req.seq_len]"
      ]),
      checkpoint: checkpoint(
        "block_size=4，decode 后 seq_len=9，是否刚进入新 block？",
        ["是，因为 9 % 4 == 1", "否，因为 9 不是 4 的倍数", "只和 request_id 有关"],
        0,
        "长度从 8 到 9 时跨入第三个 block 的第一个位置。"
      ),
      homework: [
        "decode 时先让 seq_len 加 1。",
        "用取模判断是否需要新 block。",
        "按 block_table 取块并 cat，最后截断到真实 seq_len。"
      ]
    })
  ],
  "18": [
    lesson({
      id: "spec-draft-verify",
      title: "Speculative Decoding：小模型先起草，大模型来验收",
      todo: "整体流程",
      prerequisite: [
        "draft model 生成快但不够准。",
        "target model 生成慢但质量高。",
        "本 notebook 只练接受/拒绝前缀；完整算法还会处理拒绝后的修正采样。"
      ],
      intuition: "像先写草稿再让老师批改：草稿正确的部分直接采用，错误处再由老师接管。",
      exampleHtml: `<div class="mini-flow"><span>draft: A B C</span><span>target 验证</span><strong>接受 A B，拒绝 C</strong></div>`,
      syntaxHtml: code("逐个检查候选 token", [
        "accepted = []",
        "for token, p, q in zip(draft_tokens, target_probs, draft_probs):",
        "    if should_accept(p, q):",
        "        accepted.append(token)",
        "    else:",
        "        break"
      ]),
      checkpoint: checkpoint(
        "在本 notebook 的验证函数里，接受概率 p/q 中的 p 来自哪里？",
        ["target model", "draft model", "tokenizer"],
        0,
        "draft 负责提出候选；这里练的是用 target 概率 p 和 draft 概率 q 做前缀接受判断。完整投机解码还要补拒绝后的修正采样。"
      ),
      homework: [
        "读懂 notebook 中 draft 概率 q 和 target 概率 p 的含义。",
        "确认验证过程按候选 token 顺序进行。",
        "理解拒绝后为什么停止接受后续草稿；完整算法的后续采样不在本 notebook 范围内。"
      ]
    }),
    lesson({
      id: "spec-accept-rule",
      title: "接受规则：p >= q 时全收，否则按 p/q 抽签",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "p 是 target model 对草稿 token 的概率。",
        "q 是 draft model 对同一 token 的概率。",
        "当 draft 过于乐观时，需要降低接受概率。"
      ],
      intuition: "如果大模型比小模型还认可这个 token，就直接通过；如果小模型太自信，就按比例抽签。",
      exampleHtml: `<div class="ratio-board"><span>p=.20, q=.50</span><span>accept prob=.40</span><strong>随机数小于 .40 才接受</strong></div>`,
      syntaxHtml: code("条件判断和随机数", [
        "if p >= q:",
        "    accept = True",
        "else:",
        "    accept_prob = p / q",
        "    accept = torch.rand(()) < accept_prob"
      ]),
      checkpoint: checkpoint(
        "p=0.3，q=0.6 时，接受概率是多少？",
        ["0.5", "1.0", "0.3"],
        0,
        "target 概率低于 draft 概率，所以按 p/q = 0.5 接受。"
      ),
      homework: [
        "实现 p >= q 时 100% 接受。",
        "实现 p < q 时按 p/q 接受。",
        "注意 q 很小时的数值安全。"
      ]
    })
  ],
  "19": [
    lesson({
      id: "radix-prefix-cache",
      title: "RadixAttention：缓存共同前缀，少算重复 prompt",
      todo: "核心概念",
      prerequisite: [
        "很多请求可能共享相同系统提示词或长前缀。",
        "KV Cache 可以复用已经计算过的前缀。",
        "Radix Tree 适合按 token 前缀查找。"
      ],
      intuition: "像自动补全目录：多个 prompt 从同一段开头分叉，公共开头只存一份。",
      exampleHtml: `<div class="flow"><span>[1,2]</span><span>到 [3]</span><span>到 [8]</span><strong>另一路到 [4]</strong></div>`,
      syntaxHtml: code("列表前缀比较", [
        "def common_prefix(a, b):",
        "    n = 0",
        "    for x, y in zip(a, b):",
        "        if x != y:",
        "            break",
        "        n += 1",
        "    return n"
      ]),
      checkpoint: checkpoint(
        "[10, 20, 30] 和 [10, 20, 99] 的最长公共前缀长度是？",
        ["2", "1", "3"],
        0,
        "前两个 token 相同，第三个不同。"
      ),
      homework: [
        "先理解节点里存的是 token 片段而不是单个字符。",
        "用公共前缀长度判断是否命中缓存。",
        "把匹配到的 KV Cache 长度返回给调用方。"
      ]
    }),
    lesson({
      id: "radix-tree-walk",
      title: "遍历子节点：找能匹配 prompt 的最长分支",
      todo: "TODO",
      prerequisite: [
        "root.children 存放可能的下一段前缀。",
        "匹配时要比较 prompt_tokens 和 child.tokens。",
        "最长匹配优先，因为它能复用更多缓存。"
      ],
      intuition: "在分叉路口先看哪条路的路牌和 prompt 开头对得上，对得越长越值得走。",
      exampleHtml: `<div class="branches"><span>prompt: A B C D</span><span>child1: A B</span><span>child2: A X</span><strong>走 child1</strong></div>`,
      syntaxHtml: code("遍历节点找最长匹配", [
        "best = None",
        "best_len = 0",
        "for child in self.root.children:",
        "    n = common_prefix(prompt_tokens, child.tokens)",
        "    if n > best_len:",
        "        best = child",
        "        best_len = n"
      ]),
      checkpoint: checkpoint(
        "查 Radix Tree 时为什么要找最长匹配？",
        ["复用尽可能多的 KV Cache", "让 batch size 变大", "避免使用 token id"],
        0,
        "匹配越长，已经缓存的前缀越多，需要新计算的 token 越少。"
      ),
      homework: [
        "遍历 root.children。",
        "对每个 child 计算公共前缀长度。",
        "返回最长命中的节点和匹配长度。"
      ]
    })
  ],
  "20": [
    lesson({
      id: "quant-absmax-scale",
      title: "W8 量化：用绝对最大值确定刻度尺",
      todo: "TODO 1 / TODO 2 / TODO 3",
      prerequisite: [
        "int8 的有效范围通常看作 [-127, 127]。",
        "这里的 scale 表示整数刻度密度：127 / absmax。",
        "round 后还要 clamp，防止越界。"
      ],
      intuition: "先找这组权重里最大的绝对值，再算出 1 个浮点单位对应多少个 int8 刻度。",
      exampleHtml: `<div class="scale-demo"><span>absmax=3.0</span><span>scale=127/3.0</span><strong>w_int8=round(w * scale)</strong></div>`,
      syntaxHtml: code("量化三步", [
        "absmax = weight.abs().max()",
        "scale = 127.0 / absmax",
        "q = torch.round(weight * scale).clamp(-128, 127).to(torch.int8)"
      ]),
      checkpoint: checkpoint(
        "这个 notebook 中为什么 scale 常用 127 / absmax？",
        ["把最大绝对值映射到 int8 边界", "把所有数变成正数", "让矩阵转置"],
        0,
        "对称 int8 量化希望最大幅度刚好落在可表示边界附近。"
      ),
      homework: [
        "计算权重绝对最大值。",
        "用 127.0 / absmax 得到 scale。",
        "先乘 scale，再 round、clamp、转 int8 完成量化。"
      ]
    }),
    lesson({
      id: "quant-w8a16-linear",
      title: "W8A16 前向：权重反量化，激活保持半精度",
      todo: "TODO 4",
      prerequisite: [
        "W8A16 表示 weight 用 int8 存，activation 用 fp16/bf16 算。",
        "矩阵乘法前通常要把 int8 权重反量化回浮点。",
        "本 notebook 的反量化公式是 q_weight / scale。"
      ],
      intuition: "量化时把浮点数乘上刻度密度变成整数；反量化时再除以同一把刻度尺还原。",
      exampleHtml: `<div class="matrix-flow"><span>int8 weight</span><span>除以 scale</span><span>fp weight</span><strong>F.linear(x, fp_weight)</strong></div>`,
      syntaxHtml: code("反量化后做线性层", [
        "dequant_weight = q_weight.float() / scale",
        "dequant_weight = dequant_weight.to(x.dtype)",
        "out = F.linear(x, dequant_weight, bias)"
      ]),
      checkpoint: checkpoint(
        "W8A16 中 A16 指的是什么？",
        ["激活用 16-bit 浮点", "权重有 16 行", "batch size 必须是 16"],
        0,
        "A 是 activation，16 表示激活通常保留 fp16/bf16 精度。"
      ),
      homework: [
        "用 q_weight.float() / scale 反量化。",
        "把反量化权重转成输入 dtype。",
        "调用 F.linear 完成前向并比较误差。"
      ]
    })
  ],
  "21": [
    lesson({
      id: "gc-activation-memory",
      title: "Gradient Checkpointing：少存激活，多算一次",
      todo: "概念准备",
      prerequisite: [
        "反向传播需要用到前向保存的中间激活。",
        "深层网络保存所有激活会占很多显存。",
        "checkpoint 会丢掉部分激活，反向时重算。"
      ],
      intuition: "像不保存每一步草稿，只保存关键页码；需要复查时再把中间步骤重新算一遍。",
      exampleHtml: `<div class="flow"><span>普通: 保存 a1,a2,a3</span><span>checkpoint: 保存输入</span><strong>反向时重算 a1,a2,a3</strong></div>`,
      syntaxHtml: code("包装一个函数", [
        "from torch.utils.checkpoint import checkpoint",
        "def block_forward(x):",
        "    return block(x)",
        "y = checkpoint(block_forward, x, use_reentrant=False)"
      ]),
      checkpoint: checkpoint(
        "Gradient Checkpointing 用什么换显存？",
        ["更多计算时间", "更大的 batch size 自动出现", "更少模型参数"],
        0,
        "它通过反向重算前向片段来减少激活保存。"
      ),
      homework: [
        "先确认需要包裹的是哪段前向计算。",
        "理解 checkpoint 不改变数学结果，只改变保存策略。",
        "注意输入通常需要参与梯度。"
      ]
    }),
    lesson({
      id: "gc-recompute",
      title: "用 checkpoint 包裹 forward：接口很小，含义很重要",
      todo: "TODO",
      prerequisite: [
        "checkpoint 接收一个可调用函数和它的输入张量。",
        "被包裹函数的输出会参与后续计算。",
        "反向时 PyTorch 会重新调用这段函数。"
      ],
      intuition: "把一段计算交给 checkpoint 管家：前向先不保存全部细节，反向需要时再现场复现。",
      exampleHtml: `<div class="mini-flow"><span>x</span><span>checkpoint(block, x)</span><span>y</span><strong>backward 时重算 block(x)</strong></div>`,
      syntaxHtml: code("lambda 包住额外参数", [
        "out = checkpoint(",
        "    lambda hidden: layer(hidden, attention_mask=mask),",
        "    x,",
        "    use_reentrant=False,",
        ")"
      ]),
      checkpoint: checkpoint(
        "checkpoint 包裹的函数在什么时候会被重新执行？",
        ["反向传播时", "导入 torch 时", "保存模型时"],
        0,
        "反向需要中间激活时，会重放 forward 片段来恢复它们。"
      ),
      homework: [
        "导入 torch.utils.checkpoint.checkpoint。",
        "用 checkpoint 包裹 notebook 指定的前向块。",
        "执行 notebook 检查，确认输出和梯度仍然正确。"
      ]
    })
  ],
  "22": [
    lesson({
      id: "qlora-nf4-lookup",
      title: "QLoRA 的 4bit 权重：先查表反量化",
      todo: "TODO 1",
      prerequisite: [
        "4bit 只能表示 16 个离散编号。",
        "NF4 codebook 把编号映射成浮点值。",
        "反量化还要乘上 scale。"
      ],
      intuition: "4bit 权重像色号，不直接是颜色；先用色号查调色盘，再按亮度 scale 调整。",
      exampleHtml: `<div class="lookup"><span>q=[0, 7, 15]</span><span>NF4 codebook</span><strong>float values * scale</strong></div>`,
      syntaxHtml: code("用整数索引查 codebook", [
        "codebook = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])",
        "q = torch.tensor([0, 2, 4])",
        "deq = codebook[q] * scale"
      ]),
      checkpoint: checkpoint(
        "NF4 反量化的第一步通常是什么？",
        ["用 4bit 编号查 codebook", "对编号做 softmax", "把编号当 token id 做 embedding"],
        0,
        "4bit 编号本身只是离散码，需要查 codebook 得到近似浮点值。"
      ),
      homework: [
        "用 quantized_weight 作为索引查 NF4 codebook。",
        "乘上 scale 得到基础权重近似值。",
        "注意 dtype 要能参与后续线性计算。"
      ]
    }),
    lesson({
      id: "qlora-two-path",
      title: "QLoRA 前向：冻结 4bit 基座，加 LoRA 旁路",
      todo: "TODO 2",
      prerequisite: [
        "QLoRA 的基础权重是量化存储的冻结权重。",
        "LoRA 旁路负责训练少量新增参数。",
        "最终输出是基础前向和 LoRA 前向相加。"
      ],
      intuition: "基座模型压缩保存，只负责提供主干能力；小旁路负责学习新任务的改动。",
      exampleHtml: `<div class="branches"><span>base: dequant W</span><span>adapter: A to B</span><strong>base_out + lora_out</strong></div>`,
      syntaxHtml: code("两条前向路径相加", [
        "base_out = F.linear(x, dequant_weight, bias)",
        "lora_out = F.linear(F.linear(x, lora_A), lora_B) * scaling",
        "out = base_out + lora_out"
      ]),
      checkpoint: checkpoint(
        "QLoRA 训练时主要更新哪部分参数？",
        ["LoRA adapter", "4bit 基础权重", "token id"],
        0,
        "基础权重量化并冻结，训练集中在 LoRA adapter。"
      ),
      homework: [
        "分别计算 dequantized base forward 和 LoRA forward。",
        "把两条路径输出相加。",
        "检查输出 shape 和普通 Linear 一致。"
      ]
    })
  ],
  "23": [
    lesson({
      id: "zero-state-shard",
      title: "ZeRO：把优化器状态切给不同 GPU",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "Adam 等优化器会为参数保存额外状态。",
        "每张卡都保存完整状态会浪费显存。",
        "ZeRO 把参数或优化器状态按 shard 分给多张卡。"
      ],
      intuition: "多人共同保管一本大账本，每个人只负责自己的几页，需要时再同步完整结果。",
      exampleHtml: `<div class="gpu-grid"><span>GPU0: 参数 0-3</span><span>GPU1: 参数 4-7</span><span>GPU2: 参数 8-11</span></div>`,
      syntaxHtml: code("用参数列表划分负责范围", [
        "params = list(model.parameters())",
        "half_idx = len(params) // 2",
        "gpu_partitions = {",
        "    0: params[:half_idx],",
        "    1: params[half_idx:],",
        "}"
      ]),
      checkpoint: checkpoint(
        "ZeRO 分片的直接目的是什么？",
        ["减少每张 GPU 的显存占用", "让 loss 不需要反传", "把模型改成 CNN"],
        0,
        "把状态切开后，每张卡只保存自己负责的部分。"
      ),
      homework: [
        "把 model_params 转成参数列表，并按列表切成两份。",
        "为每张 GPU 初始化局部 optimizer state。",
        "确认局部状态长度小于完整参数长度。"
      ]
    }),
    lesson({
      id: "zero-local-step",
      title: "局部更新：单机模拟里原位修改就是同步",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "每张卡只遍历自己负责的参数列表。",
        "TODO 3 要更新局部动量并用动量更新参数。",
        "TODO 4 在本 notebook 中不需要写代码，只要理解真实多卡需要把更新后的权重同步出去。"
      ],
      intuition: "本练习里每个 p 都是同一个 Parameter 的引用；你原位改 p.data，模型里的参数也同步变了。",
      exampleHtml: `<div class="sum-flow"><span>GPU0 负责 fc1</span><span>GPU1 负责 fc2</span><strong>p.data 原位更新</strong><span>真实多卡才需要广播</span></div>`,
      syntaxHtml: code("遍历本 GPU 参数和梯度", [
        "for p, g in zip(params, grads):",
        "    momentum = states[id(p)]",
        "    momentum = momentum + g",
        "    states[id(p)] = momentum",
        "    p.data = p.data - lr * momentum"
      ]),
      checkpoint: checkpoint(
        "为什么 notebook 的 TODO 4 不需要额外写 All-Gather 代码？",
        ["p.data 是原位引用，单机模拟里已经同步", "因为 ZeRO 不需要参数", "因为梯度会自动消失"],
        0,
        "真实多卡需要广播或 All-Gather；这个单机模拟直接修改 Parameter 引用，效果等价于模型参数已经更新。"
      ),
      homework: [
        "只遍历当前 GPU 负责的 params 和对应 grads。",
        "更新 states[id(p)] 中保存的动量。",
        "用 p.data 原位更新参数，并理解 TODO 4 在真实多卡中的同步意义。"
      ]
    })
  ],
  "24": [
    lesson({
      id: "tp-column",
      title: "Tensor Parallel 列切分：每张卡算一部分输出特征",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "Linear 可写成 X @ A。",
        "沿 A 的列切分，相当于把输出特征切给不同 GPU。",
        "每张卡得到部分输出，最后沿特征维拼接。"
      ],
      intuition: "一张大表的列太多，就分给多个人各算几列，最后把列并回去。",
      exampleHtml: `<div class="matrix-flow"><span>X</span><span>A 列切成 A0/A1</span><span>X@A0, X@A1</span><strong>cat dim=1</strong></div>`,
      syntaxHtml: code("按列 chunk 再拼接", [
        "shards = torch.chunk(A, world_size, dim=1)",
        "partials = [X @ shard for shard in shards]",
        "out = torch.cat(partials, dim=1)"
      ]),
      checkpoint: checkpoint(
        "列切分 A 后，每张 GPU 拿到的是哪一部分权重？",
        ["一部分输出列", "一部分 batch 样本", "完整 A 的副本"],
        0,
        "A 的 dim=1 是 out_features，切列就是把不同输出特征分给不同 GPU。"
      ),
      homework: [
        "沿列方向切分权重 A。",
        "每张 GPU 用本地权重分片做矩阵乘法。",
        "确认每个 local_out 的 batch 维不变、输出特征变少。"
      ]
    }),
    lesson({
      id: "tp-allgather",
      title: "All-Gather：把列切分结果拼回完整输出",
      todo: "TODO 3",
      prerequisite: [
        "列切分后，每张卡得到的是不同输出特征列。",
        "这些 partial outputs 的 batch 维相同。",
        "All-Gather 在这个模拟里就是沿特征维拼接。"
      ],
      intuition: "每张卡像各自算出一摞列，最后按原来的列顺序横向贴回去，得到完整 Y。",
      exampleHtml: `<div class="sum-flow"><span>Y0: [batch, out/2]</span><span>Y1: [batch, out/2]</span><strong>cat dim=1 -> [batch, out]</strong></div>`,
      syntaxHtml: code("沿特征维拼接 partial outputs", [
        "gpu_outputs = [X @ shard for shard in weight_shards]",
        "Y_gathered = torch.cat(gpu_outputs, dim=1)",
        "assert Y_gathered.shape[1] == A.shape[1]"
      ]),
      checkpoint: checkpoint(
        "Column Parallel 的 partial outputs 应该如何合并？",
        ["沿特征维拼接", "逐元素相加", "沿 batch 维拼接"],
        0,
        "列切分得到的是不同输出列，所以要沿 dim=1 拼接回完整输出。"
      ),
      homework: [
        "收集每张 GPU 的 local_out。",
        "用 torch.cat(gpu_outputs, dim=1) 模拟 All-Gather。",
        "和单卡 X @ A 的输出做数值比较。"
      ]
    })
  ],
  "25": [
    lesson({
      id: "pp-microbatch",
      title: "Pipeline Parallel：把大模型切成流水线阶段",
      todo: "概念准备",
      prerequisite: [
        "模型可以按层切成多个 stage。",
        "一个 batch 可以再切成多个 micro-batch。",
        "流水线并行让不同 stage 同时处理不同 micro-batch。"
      ],
      intuition: "像工厂流水线：第一段处理第 2 个小批次时，第二段已经在处理第 1 个小批次。",
      exampleHtml: `<div class="timeline"><span>t0: S0-M0</span><span>t1: S0-M1 / S1-M0</span><span>t2: S0-M2 / S1-M1</span></div>`,
      syntaxHtml: code("把 batch 切成 micro-batch", [
        "micro_batches = torch.chunk(batch, chunks=num_micro_batches, dim=0)",
        "for mb in micro_batches:",
        "    hidden = stage0(mb)",
        "    out = stage1(hidden)"
      ]),
      checkpoint: checkpoint(
        "Pipeline Parallel 中 micro-batch 的作用是什么？",
        ["填满流水线，减少空泡", "改变 vocab size", "替代反向传播"],
        0,
        "micro-batch 越多，流水线越容易被填满，空闲时间比例更低。"
      ),
      homework: [
        "理解 stage 数和 micro-batch 数分别代表什么。",
        "用时间表画出前几个 micro-batch 的流动。",
        "观察 micro-batch 增多时 bubble ratio 的变化。"
      ]
    }),
    lesson({
      id: "pp-bubble-ratio",
      title: "Bubble Ratio：流水线空转比例",
      todo: "TODO",
      prerequisite: [
        "stage 越多，填满和排空流水线需要的时间越长。",
        "micro-batch 越多，真正忙碌的时间越多。",
        "bubble ratio 近似衡量空转时间占比。"
      ],
      intuition: "流水线刚启动和收尾时总有人闲着；小批次越多，这段闲着的比例越小。",
      exampleHtml: `<div class="ratio-board"><span>stages=4</span><span>micro_batches=8</span><strong>bubble = (4-1)/(8+4-1)</strong></div>`,
      syntaxHtml: code("写一个比例函数", [
        "def bubble_ratio(num_stages, num_micro_batches):",
        "    bubble = num_stages - 1",
        "    total = num_micro_batches + num_stages - 1",
        "    return bubble / total"
      ]),
      checkpoint: checkpoint(
        "stage 固定时，增加 micro-batch 数通常会让 bubble ratio 怎样？",
        ["降低", "升高", "恒为 1"],
        0,
        "micro-batch 越多，空转时间被更多有效计算摊薄。"
      ),
      homework: [
        "按 notebook 要求实现 bubble ratio 公式。",
        "测试 stage=1 时 bubble ratio 是否为 0。",
        "用不同 micro-batch 数验证趋势。"
      ]
    })
  ]
};
