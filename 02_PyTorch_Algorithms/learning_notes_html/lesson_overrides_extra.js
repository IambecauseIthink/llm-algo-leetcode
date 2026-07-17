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
          <div class="shape-story moe-story">
            <style>
              .moe-story { gap: 12px; }
              .moe-concept-grid,
              .moe-practice-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 12px;
                margin: 12px 0;
                align-items: stretch;
              }
              .moe-practice-grid { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
              .moe-practice-grid .syntax-card { margin-top: 0; }
              .moe-card {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #fff;
                padding: 14px;
                display: grid;
                gap: 10px;
                align-content: start;
              }
              .moe-card > strong {
                color: #1d4ed8;
                font-size: 17px;
              }
              .moe-card.accent {
                border-color: #e8b66f;
                background: #fff8ed;
              }
              .moe-card.good {
                border-color: #99d6bf;
                background: #f4fcf8;
              }
              .moe-card.warn {
                border-color: #e8b66f;
                background: #fff6e8;
              }
              .moe-card p { margin: 0; }
              .moe-route {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                gap: 8px;
                margin: 12px 0;
              }
              .moe-route span,
              .moe-route strong {
                min-height: 76px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #fff;
                padding: 10px;
                display: grid;
                place-items: center;
                text-align: center;
                line-height: 1.45;
              }
              .moe-route b {
                display: block;
                color: #1d4ed8;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              }
              .moe-route strong {
                border-color: #99d6bf;
                background: var(--soft-green);
                color: #123f31;
              }
              .moe-kv {
                display: grid;
                grid-template-columns: minmax(130px, 0.42fr) minmax(0, 1fr);
                gap: 8px;
                margin: 12px 0;
              }
              .moe-kv span,
              .moe-kv strong {
                min-height: 44px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #fff;
                padding: 10px;
                display: grid;
                align-items: center;
                line-height: 1.45;
              }
              .moe-kv span {
                color: var(--muted);
                font-weight: 800;
                text-align: center;
                background: #fbfcfd;
              }
              .moe-kv strong {
                color: #182033;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                overflow-wrap: anywhere;
              }
              .moe-token-row {
                display: grid;
                grid-template-columns: repeat(4, minmax(82px, 1fr));
                gap: 8px;
                margin: 12px 0;
              }
              .moe-token-row span {
                min-height: 58px;
                border: 1px solid #dce2e8;
                border-radius: 8px;
                background: #f8fbff;
                display: grid;
                place-items: center;
                text-align: center;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                font-weight: 800;
              }
              .moe-token-row .hot {
                border-color: #d9a15a;
                background: #fde68a;
                color: #6f3d08;
              }
              @media (max-width: 640px) {
                .moe-kv,
                .moe-token-row { grid-template-columns: 1fr; }
              }
            </style>
            <div class="moe-concept-grid">
              <div class="moe-card">
                <strong>0. 先读 forward 的输入输出合同</strong>
                <p><code>TopKRouter.forward</code> 的输入仍是三维句子张量，但输出是按 token 排好的两张清单。先把这个合同读清楚，后面代码就不会绕。</p>
                <div class="moe-kv">
                  <span>输入</span><strong>hidden_states [B,S,H]</strong>
                  <span>输出权重</span><strong>routing_weights [B*S,K]</strong>
                  <span>输出专家</span><strong>selected_experts [B*S,K]</strong>
                </div>
              </div>
              <div class="moe-card accent">
                <strong>TODO 1 / 2 只负责“选专家”</strong>
                <p>先算每个 token 对所有专家的概率，再保留 Top-K。真正执行专家网络是后面的 <code>SparseMoEBlock</code>，不要在 Router 里混进去。</p>
                <div class="moe-route">
                  <span><b>TODO 1</b>softmax 概率表</span>
                  <strong><b>TODO 2</b>topk 权重 + 专家 id</strong>
                </div>
              </div>
            </div>

            <div class="story-arrow">主线：先把 [B,S,H] 变成 token 队列，再沿专家维做 softmax/topk</div>
            <div class="story-panel">
              <div class="moe-route">
                <span><b>hidden_states</b>[B,S,H]</span>
                <span><b>flatten</b>[B*S,H]</span>
                <span><b>gate</b>router_logits [B*S,E]</span>
                <span><b>softmax</b>routing_probs [B*S,E]</span>
                <strong><b>topk</b>weights / experts [B*S,K]</strong>
              </div>
              <p>这里的关键不是画很多线，而是抓住最后一维的含义：展平后每一行是一枚 token，最后一维是全部专家。</p>
            </div>

            <div class="moe-concept-grid">
              <div class="moe-card">
                <strong>1. 每一行是一枚 token</strong>
                <p>如果有 4 个专家，一行 logits 就是这个 token 对 4 个专家的偏好分数。softmax 和 topk 都只在这一行内部做。</p>
                <div class="moe-token-row">
                  <span>E0<br>1.2</span>
                  <span>E1<br>0.1</span>
                  <span class="hot">E2<br>2.0</span>
                  <span>E3<br>-0.5</span>
                </div>
              </div>
              <div class="moe-card">
                <strong>2. <code>dim=-1</code> 就是专家维</strong>
                <p>不要把不同 token 混在一起归一化。Router 要让“同一个 token 的所有专家概率”加起来等于 1。</p>
                <div class="moe-kv">
                  <span>正确</span><strong>F.softmax(router_logits.float(), dim=-1)</strong>
                  <span>读法</span><strong>每个 token 自己在专家列表里分配概率</strong>
                </div>
              </div>
            </div>

            <div class="story-arrow">topk 返回两张表：values 是权重，indices 是专家编号</div>
            <div class="moe-concept-grid">
              <div class="moe-card good">
                <strong>3. 值和编号要一起拿</strong>
                <div class="moe-kv">
                  <span>routing_probs 一行</span><strong>[0.05, 0.62, 0.08, 0.25]</strong>
                  <span>values</span><strong>routing_weights = [0.62, 0.25]</strong>
                  <span>indices</span><strong>selected_experts = [1, 3]</strong>
                </div>
              </div>
              <div class="moe-card warn">
                <strong>4. 最容易掉坑的地方</strong>
                <p>notebook 里 TODO 后面有 zero 占位变量。写完 TODO 后，要返回你真正算出的 <code>routing_weights</code> 和 <code>selected_experts</code>，不要让占位值覆盖结果。</p>
              </div>
            </div>

            <div class="field-note">
              <div class="fn-title">行业视角：Router 是 MoE 的调度器</div>
              <p>MoE 的专家像并行工作台，Router 只决定每个 token 送去哪几个工作台，以及每个工作台占多少权重。调度得好，模型容量变大但每个 token 只激活少数专家；调度得差，热门专家拥堵，冷门专家闲置。</p>
              <p>下一关的负载均衡损失，就是专门约束这个调度器不要总偏心同几个专家。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="moe-practice-grid">
          ${code("先用一行玩具分数练 softmax 和 topk", [
        "# 1 个 token, 4 个专家",
        "toy_logits = torch.tensor([[1.2, 0.1, 2.0, -0.5]])",
        "",
        "# dim=-1：只在“专家这一行”里归一化，得到每个专家的概率",
        "toy_probs = F.softmax(toy_logits.float(), dim=-1)",
        "",
        "# topk 同时返回值(values)和下标(indices)",
        "toy_weights, toy_ids = torch.topk(",
        "    toy_probs, k=2, dim=-1",
        ")",
        "print(toy_weights)",
        "print(toy_ids)"
      ])}
          <div class="syntax-card">
          <h4>举一反三：把玩具张量换成 notebook 变量</h4>
          <p>玩具例子只有 1 个 token。notebook 只是把很多 token 排成 [tokens, experts]，同样沿最后一维做 softmax/topk。</p>
          <div class="moe-kv">
            <span>toy_logits</span><strong>router_logits</strong>
            <span>toy_probs</span><strong>routing_probs</strong>
            <span>toy_weights</span><strong>routing_weights</strong>
            <span>toy_ids</span><strong>selected_experts</strong>
            <span>k=2</span><strong>self.top_k</strong>
          </div>
          <p class="syntax-tip">迁移口诀：先把输入展平成 [tokens, hidden]，gate 后得到 [tokens, experts]；之后所有 softmax/topk 都沿 <code>dim=-1</code>，因为最后一维就是专家维。</p>
        </div>
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
          <div class="shape-story moe-story">
            <div class="moe-concept-grid">
              <div class="moe-card accent">
                <strong>0. 本 Mission 分两层读</strong>
                <p>你真正要补的是 TODO 3：把 Top-K 权重重新归一化。后面的专家聚合逻辑已经给好，目标是读懂它为什么这样写。</p>
                <div class="moe-kv">
                  <span>要补代码</span><strong>routing_weights / row_sum</strong>
                  <span>要读懂</span><strong>where 找 token, expert 计算, 加权累加</strong>
                </div>
              </div>
              <div class="moe-card">
                <strong>先看权重，再看专家输出</strong>
                <p>不要一开始就盯着 for 循环。Router 先给出权重和专家编号；SparseMoEBlock 再按编号把 token 送到专家，并把结果加回来。</p>
                <div class="moe-route">
                  <span><b>Router</b>weights + experts</span>
                  <strong><b>MoE Block</b>expert outputs weighted sum</strong>
                </div>
              </div>
            </div>

            <div class="story-arrow">第一层：Top-K 留下来的概率要重新分成 100%</div>
            <div class="moe-concept-grid">
              <div class="moe-card">
                <strong>1. 为什么要除以每行 sum</strong>
                <p>全量概率一行加起来是 1，但只保留 Top-2 后可能只剩 0.87。直接混合专家输出会把整体尺度压小，所以要在留下的专家内部重新分配。</p>
                <div class="moe-route">
                  <span><b>Top-2</b>.60 + .27 = .87</span>
                  <span><b>divide</b>.60/.87, .27/.87</span>
                  <strong><b>renorm</b>.69 + .31 = 1.00</strong>
                </div>
              </div>
              <div class="moe-card good">
                <strong>2. <code>keepdim=True</code> 保住可广播维度</strong>
                <p><code>routing_weights</code> 是 [tokens, top_k]。sum 后保留成 [tokens, 1]，才能自动广播除回每个 Top-K 槽位。</p>
                <div class="moe-kv">
                  <span>原权重</span><strong>routing_weights [8,2]</strong>
                  <span>行和</span><strong>routing_weights.sum(dim=-1, keepdim=True) [8,1]</strong>
                  <span>归一化后</span><strong>routing_weights [8,2], 每行和接近 1</strong>
                </div>
              </div>
            </div>

            <div class="story-arrow">第二层：SparseMoEBlock 逐专家找 token，再把加权输出累加回去</div>
            <div class="story-panel">
              <div class="moe-route">
                <span><b>selected_experts</b>专家编号表</span>
                <span><b>torch.where</b>token_idx + kth_expert</span>
                <span><b>routing_weights</b>取当前权重</span>
                <span><b>unsqueeze</b>[n] 变 [n,1]</span>
                <strong><b>+=</b>累加到 final_hidden_states</strong>
              </div>
              <p>这段的读法是“谁选了当前专家，就把谁拿出来跑当前专家；再按对应权重写回”。</p>
            </div>

            <div class="moe-concept-grid">
              <div class="moe-card">
                <strong>3. <code>where</code> 返回两个坐标</strong>
                <p>第一个坐标告诉你第几个 token 选中了当前专家；第二个坐标告诉你它是在 Top-K 的第几个槽位选中的。</p>
                <div class="moe-kv">
                  <span>selected_experts</span><strong>[[2,0], [1,2], [3,1]]</strong>
                  <span>expert_idx=2</span><strong>token_idx=[0,1], kth_expert=[0,1]</strong>
                  <span>取权重</span><strong>routing_weights[token_idx, kth_expert]</strong>
                </div>
              </div>
              <div class="moe-card warn">
                <strong>4. 为什么必须是 <code>+=</code></strong>
                <p>top_k=2 时，同一个 token 会从两个专家收集贡献。第一次写回不能覆盖第二次，第二次也不能抹掉第一次，所以这里是累加。</p>
                <div class="moe-route">
                  <span>expert2(x0) * w0,0</span>
                  <span>expert0(x0) * w0,1</span>
                  <strong>token0 = 两份贡献相加</strong>
                </div>
              </div>
            </div>

            <div class="story-panel">
              <strong>5. 用测试反查三件事</strong>
              <div class="moe-kv">
                <span>weights / indices</span><strong>[batch_size * seq_len, top_k]</strong>
                <span>weights.sum(dim=-1)</span><strong>每行接近 1</strong>
                <span>MoE 输出</span><strong>回到 [batch_size, seq_len, hidden_size]</strong>
              </div>
            </div>

            <div class="field-note">
              <div class="fn-title">行业视角：玩具循环易懂，工业实现会先排队再批处理</div>
              <p>这里用 for 循环遍历专家，是为了让逻辑透明。真实推理框架会把同一个专家的 token 先排序聚到一起，一次性组成大 batch 喂给专家网络，减少小矩阵调用带来的 GPU 空转。</p>
              <p>但核心协议没变：Router 给专家 id 和权重，专家产出向量，最后按权重加回对应 token。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="moe-practice-grid">
          ${code("先用一行权重练 keepdim=True", [
        "toy_weights = torch.tensor([",
        "    [0.60, 0.27],",
        "    [0.40, 0.10],",
        "])",
        "",
        "# keepdim=True：sum 后保留成 [2, 1]",
        "row_sum = toy_weights.sum(",
        "    dim=-1, keepdim=True",
        ")",
        "toy_normed = toy_weights / row_sum",
        "print(toy_normed.sum(dim=-1))"
      ])}
          ${code("再用玩具 selected_experts 读懂 where 和 unsqueeze", [
        "selected = torch.tensor([",
        "    [2, 0],",
        "    [1, 2],",
        "    [3, 1],",
        "])",
        "expert_idx = 2",
        "",
        "# where 返回两个坐标：第几个 token、第几个 top-k 槽位",
        "token_idx, kth = torch.where(selected == expert_idx)",
        "print(token_idx)  # tensor([0, 1])",
        "print(kth)        # tensor([0, 1])",
        "",
        "weights = torch.tensor([",
        "    [0.7, 0.3],",
        "    [0.4, 0.6],",
        "    [0.5, 0.5],",
        "])",
        "w = weights[token_idx, kth]",
        "w = w.unsqueeze(-1)  # [2] -> [2, 1]"
      ])}
          <div class="syntax-card">
          <h4>举一反三：把玩具写法搬回 notebook</h4>
          <p>TODO 3 只需要第一张卡的重归一化写法。第二张卡是为了读懂后面已经给出的专家聚合代码。</p>
          <div class="moe-kv">
            <span>toy_weights</span><strong>routing_weights</strong>
            <span>selected</span><strong>selected_experts</strong>
            <span>expert_idx</span><strong>for 循环里的 expert_idx</strong>
            <span>w</span><strong>current_weight</strong>
          </div>
          <p class="syntax-tip">迁移口诀：归一化看最后一维 <code>top_k</code>，所以 <code>sum(dim=-1, keepdim=True)</code>；聚合时权重先从 [n] 变 [n,1]，才能广播乘到专家输出的 hidden 维。</p>
        </div>
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
      title: "f_i：先学会“数人头”——每个专家实际接了多少活",
      todo: "TODO 2",
      prerequisite: [
        "回忆 MoE：每个 token 不再走同一个 FFN，而是由一个 router 打分，挑出 Top-K 个“专家”（小 FFN）去处理它。",
        "麻烦在于 router 会“偷懒”：训练早期它可能发现专家 0、1 还不错，就把几乎所有 token 都塞给这两个，剩下的专家从没被训练到——这叫路由崩塌 (Router Collapse)。",
        "f_i = 第 i 个专家被选中的次数 ÷ 总的选择次数。它衡量“实际工作量”，是后面惩罚项的一半。",
        "F.one_hot(ids, num_classes=E)：把每个专家 id 变成一个长度为 E 的向量，命中的那一位是 1、其余是 0。把它们加起来，就数出每个专家被点名几次。"
      ],
      intuition: "把 E 个专家想成食堂的 E 个打饭窗口。router 是带位员，selected_experts 记录了每个 token 被带到了哪个窗口。f_i 就是“第 i 号窗口排了几个人 ÷ 总人次”。如果某个窗口排爆了、别的窗口空着，就说明负载不均——这一关只做一件事：把这张“人头统计表”用 one_hot 数出来。",
      exampleHtml: `
          <div class="shape-story">
            <div class="learn-grid">
              <div class="learn-card">
                <strong>0. 先读输入输出合同</strong>
                <p><code>selected_experts</code> 是 [tokens, top_k]。每一行是一枚 token，里面 K 个数字是它选中的专家编号。TODO 2 的目标，是把它统计成长度为 E 的 <code>f_i</code>。</p>
                <div class="learn-kv">
                  <span>输入</span><strong>selected_experts [T,K]</strong>
                  <span>中间</span><strong>one_hot [T,K,E]</strong>
                  <span>输出</span><strong>f_i [E], sum(f_i)=1</strong>
                </div>
              </div>
              <div class="learn-card accent">
                <strong>为什么先“数人头”</strong>
                <p>Router 的概率只说明“偏好”，但真正拥不拥堵，要看 Top-K 之后每个专家实际被选了多少次。<code>f_i</code> 就是这张真实排队表。</p>
                <div class="learn-mini">
                  <span>专家0<br>3票</span>
                  <span>专家1<br>1票</span>
                  <span>专家2<br>2票</span>
                  <strong>专家3<br>0票</strong>
                </div>
              </div>
            </div>

            <div class="story-arrow">主线：专家 id 不能直接相加，先 one_hot 成可计数的票</div>
            <div class="story-panel">
              <div class="learn-flow">
                <span><b>selected</b>[[0,2],[0,1],[0,2]]</span>
                <span><b>one_hot</b>[T,K,E]</span>
                <span><b>sum</b>dim=(0,1)</span>
                <span><b>counts</b>[3,1,2,0]</span>
                <strong><b>divide</b>f_i = counts / (T*K)</strong>
              </div>
              <p>读法很简单：把每个专家编号变成一张投票单，再沿 token 维和 top_k 维一起求和。</p>
            </div>

            <div class="learn-grid">
              <div class="learn-card good">
                <strong>1. one_hot 的作用是“让 id 可以计票”</strong>
                <p>专家 id 本身没有可加意义，<code>0 + 2</code> 不是“专家 0 和专家 2 各一票”。one_hot 后，每一位都对应一个专家，相加才变成计数。</p>
                <div class="learn-kv">
                  <span>[0,2]</span><strong>[1,0,1,0]</strong>
                  <span>[0,1]</span><strong>[1,1,0,0]</strong>
                  <span>[0,2]</span><strong>[1,0,1,0]</strong>
                </div>
              </div>
              <div class="learn-card warn">
                <strong>2. 分母是总票数，不是 token 数</strong>
                <p>top_k=2 时，每个 token 投 2 票。3 个 token 的总票数是 6，所以 <code>f=[3,1,2,0]/6</code>，不是除以 3。</p>
                <div class="learn-note">自检口诀：所有 <code>f_i</code> 加起来必须接近 1；如果接近 top_k，分母就写错了。</div>
              </div>
            </div>

            <div class="field-note">
              <div class="fn-title">行业视角：路由崩塌是真金白银的损失</div>
              <p>Mixtral、DeepSeek-MoE 这类模型的专家常分散在多张 GPU 上。如果 token 都挤给少数专家，少数 GPU 排长队甚至 OOM，多数 GPU 空转。<code>f_i</code> 不是学术装饰，它直接对应集群利用率。</p>
              <p>数好这张人头表，是下一步设计惩罚项、把负载摊平的前提。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="practice-grid">
          ${code("one_hot 计票 + 归一化", [
        "ids = torch.tensor([",
        "    [0, 2],",
        "    [0, 1],",
        "    [0, 2],",
        "])",
        "hot = F.one_hot(ids, num_classes=4)",
        "counts = hot.sum(dim=(0, 1)).float()",
        "total_votes = ids.shape[0] * ids.shape[1]",
        "f_i = counts / total_votes"
      ])}
          <div class="syntax-card">
            <h4>举一反三：搬回 notebook</h4>
            <div class="learn-kv">
              <span>ids</span><strong>selected_experts</strong>
              <span>num_classes=4</span><strong>num_experts</strong>
              <span>dim=(0,1)</span><strong>沿 token 和 top_k 一起计票</strong>
              <span>total_votes</span><strong>total_tokens * top_k</strong>
            </div>
            <p class="syntax-tip">迁移口诀：先 one_hot 让专家编号变成票，再对前两维求和，最后除以总票数。</p>
          </div>
        </div>`,
      predict: {
        hook: "router 自己其实“知道”谁该被多选——它输出了每个专家的打分。可负载均衡损失偏偏还要我们再手动统计一遍 f_i（实际被选中的比例）。",
        question: "先判断：为什么不能只用 router 的打分，而要额外统计 f_i？",
        options: [
          "因为打分是“连续偏好”，可微分可优化；f_i 是“离散计数”，由 argmax/Top-K 选出，本身不可导，但能真实反映拥堵",
          "因为 router 的打分在反向传播时会被清零",
          "因为 f_i 只是用来打印日志，对训练没有作用"
        ],
        answer: 0,
        revealNote: "对：Top-K 这步“选谁”是不可导的，梯度传不回去。损失里用可导的 P_i 去“代表”这次分配，再乘上真实的 f_i 当权重——下一关就拼这个公式。"
      },
      checkpoint: checkpoint(
        "动手算：top_k=2，4 个 token 选中 [[0,1],[0,2],[0,1],[0,3]]，专家 0 的 f_i 是多少？",
        ["0.5", "0.25", "4.0"],
        0,
        "专家 0 被选 4 次，总票数 = 4 token × 2 = 8，f_0 = 4/8 = 0.5。"
      ),
      homework: [
        "TODO 2：用 F.one_hot(selected_experts, num_classes=num_experts) 把专家 id 变成可计数的指示向量。",
        "沿 token 维和 top_k 维一起 sum，得到每个专家的总票数 tokens_per_expert。",
        "除以 (total_tokens * top_k) 得到 f_i；自检所有 f_i 之和≈1。"
      ]
    }),
    lesson({
      id: "moe-balance-prob-loss",
      title: "P_i × f_i：把“偏好”和“实际”相乘，逼出均匀分配",
      todo: "TODO 1 / TODO 3",
      prerequisite: [
        "P_i = 第 i 个专家的“平均路由概率”——把所有 token 给它的 router 权重加起来再平均。它是“软偏好”，可导。",
        "f_i = 上一关数出的“实际被选比例”——“硬计数”，反映真实拥堵，不可导。",
        "scatter_add_(0, idx, src)：按 idx 给出的位置，把 src 的值累加到目标向量上——正好用来把每个 token 的权重加到它选中的专家槽位里。",
        "最终公式：L_aux = α · E · Σ(f_i · P_i)。α 很小(如0.01)，E 是专家数。"
      ],
      intuition: "P_i 像“问卷调查的偏好分”，f_i 像“真实排队人数”。损失把两者逐专家相乘再求和：只要某个专家既被偏好(P高)又被挤爆(f高)，乘积就大、惩罚就重。优化器为了降低它，会主动把 token 往冷门专家赶。神奇之处：当大家都均匀(都=1/E)时，这个乘积和取到理论最小值。",
      exampleHtml: `
          <div class="shape-story">
            <div class="learn-grid">
              <div class="learn-card">
                <strong>0. 先分清 P_i 和 f_i</strong>
                <div class="learn-kv">
                  <span>P_i</span><strong>router 权重按专家累加，可导，负责把梯度传回 router</strong>
                  <span>f_i</span><strong>Top-K 真实计数，不可导，负责指出哪个专家拥堵</strong>
                </div>
              </div>
              <div class="learn-card accent">
                <strong>为什么要相乘</strong>
                <p><code>f_i</code> 像“哪里堵”的标签，<code>P_i</code> 像“能被优化器调整的偏好”。乘起来后，拥堵专家对应的概率会被更重地压下去。</p>
              </div>
            </div>

            <div class="story-arrow">TODO 1：用 scatter_add_ 把每张票的权重倒进对应专家桶</div>
            <div class="story-panel">
              <div class="learn-flow">
                <span><b>zero bucket</b>P_i = zeros(E)</span>
                <span><b>index</b>selected_experts.flatten()</span>
                <span><b>source</b>routing_weights.flatten()</span>
                <span><b>scatter_add</b>按专家 id 累加</span>
                <strong><b>average</b>P_i / (T*K)</strong>
              </div>
              <p>读法：每个 Top-K 槽位都有一个“专家 id”和一个“概率权重”，把权重加到它对应专家的桶里。</p>
            </div>

            <div class="story-arrow">TODO 3：把偏好和真实拥堵组合成辅助损失</div>
            <div class="learn-grid">
              <div class="learn-card good">
                <strong>1. 公式只做一件事：惩罚“又热门又拥堵”</strong>
                <div class="learn-kv">
                  <span>核心</span><strong>aux_loss = alpha * E * sum(f_i * P_i)</strong>
                  <span>集中</span><strong>某专家 f_i 高且 P_i 高，乘积大，被重罚</strong>
                  <span>均匀</span><strong>各专家接近平均，乘积和最小</strong>
                </div>
              </div>
              <div class="learn-card warn">
                <strong>2. f_i 不可导也没关系</strong>
                <p>反向传播真正走的是 <code>P_i</code>。<code>f_i</code> 只作为常数权重，告诉优化器“这个专家已经太忙了，压低它的概率”。</p>
                <div class="learn-note">测试线索：均匀时理论最小值约为 <code>alpha / top_k</code>。</div>
              </div>
            </div>

            <div class="field-note">
              <div class="fn-title">行业视角：一个系数 alpha 的拿捏</div>
              <p>alpha 太大，模型会为了摊平负载牺牲主任务效果；alpha 太小，又压不住路由崩塌。真实训练会把这个 aux loss 加到主 CrossEntropy 上，目标是在效果和吞吐之间找平衡。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="practice-grid">
          ${code("scatter_add 求 P_i，再组合损失", [
        "P_i = torch.zeros(",
        "    num_experts,",
        "    device=routing_weights.device,",
        ")",
        "P_i.scatter_add_(",
        "    0,",
        "    selected_experts.flatten(),",
        "    routing_weights.flatten(),",
        ")",
        "P_i = P_i / (total_tokens * top_k)",
        "aux_loss = alpha * num_experts * (f_i * P_i).sum()"
      ])}
          <div class="syntax-card">
            <h4>举一反三：读 scatter_add 的三个位置</h4>
            <div class="learn-kv">
              <span>目标桶</span><strong>P_i, 长度是 num_experts</strong>
              <span>index</span><strong>selected_experts.flatten()</strong>
              <span>src</span><strong>routing_weights.flatten()</strong>
              <span>除法分母</span><strong>total_tokens * top_k</strong>
            </div>
            <p class="syntax-tip">迁移口诀：上一关数“票数”得 f_i，这一关累加“票的权重”得 P_i；最后逐专家相乘再求和。</p>
          </div>
        </div>`,
      predict: {
        hook: "损失是 α·E·Σ(f_i·P_i)——f 和 P 逐专家相乘再相加。优化器拼命想让它变小。",
        question: "先判断：优化器为了最小化这个乘积和，会把 token 的分配推向什么状态？",
        options: [
          "推向均匀分配——所有专家 f_i、P_i 都趋近 1/E，乘积和取到理论最小",
          "推向极端集中——把所有 token 都给一个专家",
          "推向随机——每步都换不同的专家"
        ],
        answer: 0,
        revealNote: "对。两个总和固定的正向量，逐项相乘求和在“都摊平”时最小、在“集中”时最大。所以最小化它 = 逼出均匀分配，正好对抗路由崩塌。"
      },
      checkpoint: checkpoint(
        "理解：某专家 P_i 高（router 很偏爱）且 f_i 也高（实际接了很多 token），辅助损失会怎么对待它？",
        ["乘积 f_i·P_i 很大，被重罚，逼 router 给它降温", "判定它是最优专家，给予奖励", "认为 softmax 失效并跳过它"],
        0,
        "负载均衡要的是雨露均沾。又偏爱又拥堵的专家乘积最大、惩罚最重，梯度会压低它的 P_i。"
      ),
      homework: [
        "TODO 1：用 scatter_add_ 把 routing_weights 按 selected_experts 累加到 P_i，再除以 (total_tokens*top_k)。",
        "TODO 3：按 aux_loss = alpha * num_experts * (f_i * P_i).sum() 组合（f_i 来自上一关）。",
        "跑测试：不均衡的 loss 应明显大于均匀；均匀时应≈ alpha/top_k（0.01/2=0.005）。"
      ]
    })
  ],
  "08": [
    lesson({
      id: "gemma-rmsnorm-plus-one",
      title: "Gemma 的 (1+w) 缩放：一个 +1 为何能稳住训练",
      todo: "TODO 1",
      prerequisite: [
        "回忆 RMSNorm：先把向量按它自己的均方根归一化到稳定尺度，再乘一个可学习的缩放向量 weight，让模型微调每一维的强弱。",
        "PyTorch 里 weight 常初始化成 0（或极小值）。标准写法 y = x_norm · w，此时 w=0 → 输出被乘成 0，等于把这一层“掐断”了。",
        "Gemma 改成 y = x_norm · (1 + w)。w=0 时 (1+0)=1，相当于“先什么都不缩放、原样通过”。",
        "工程细节：统计量要在 FP32 下算，最后 type_as(x) 转回原精度，避免半精度数值不稳。"
      ],
      intuition: "把缩放因子的“默认值”从 0 挪到 1。标准写法里参数学的是“我要把这维放大到多少”（从 0 起步很尴尬）；Gemma 写法里参数学的是“我要在默认 1 的基础上，多一点还是少一点”。训练刚开始 w≈0，整层就是一个老老实实的纯归一化，梯度平滑、不会一上来就把信号清零。",
      exampleHtml: `
          <div class="shape-story">
            <div class="learn-grid">
              <div class="learn-card warn">
                <strong>0. 先看坑点：标准写法会把信号清零</strong>
                <div class="learn-kv">
                  <span>x_norm * w</span><strong>w=0 时 scale=0，输出被掐断</strong>
                  <span>x_norm * (1+w)</span><strong>w=0 时 scale=1，纯归一化原样通过</strong>
                </div>
              </div>
              <div class="learn-card good">
                <strong>Gemma 的参数语义</strong>
                <p><code>weight</code> 不再表示“绝对缩放值”，而表示“相对默认 1 的偏移”。初始化为 0 时就是健康默认。</p>
                <div class="learn-mini">
                  <span>w=0</span>
                  <span>scale=1</span>
                  <strong>训练平滑启动</strong>
                </div>
              </div>
            </div>

            <div class="story-arrow">TODO 主线：FP32 归一化已经算好，只补最后的 (1 + weight) 缩放</div>
            <div class="story-panel">
              <div class="learn-flow">
                <span><b>x</b>输入 dtype</span>
                <span><b>x.float()</b>稳定算统计量</span>
                <span><b>x_norm</b>RMS 归一化</span>
                <span><b>scale</b>1 + self.weight</span>
                <strong><b>output</b>type_as(x)</strong>
              </div>
              <p>notebook 里前面的归一化已经帮你铺好；TODO 只需要把缩放写对，并转回输入精度。</p>
            </div>

            <div class="learn-grid">
              <div class="learn-card">
                <strong>1. 为什么不直接初始化成 1</strong>
                <p>写成 <code>1+w</code> 后，参数仍能从 0 开始学习偏移，weight decay 拉向 0 时也代表“回到默认 scale=1”，不会把层拉向清零。</p>
              </div>
              <div class="learn-card accent">
                <strong>2. 测试应该怎么反查</strong>
                <p>当 <code>self.weight=0</code>，输出应该等于纯 RMSNorm 结果；如果输出接近 0，说明你写成了 <code>x_norm * weight</code>。</p>
              </div>
            </div>

            <div class="field-note">
              <div class="fn-title">行业视角：训练稳定性是大模型的隐形成本</div>
              <p>Gemma 的 <code>1+w</code>、LLaMA 的 Pre-Norm、各种初始化技巧，目标都一样：让训练早期别突然发散。一个小小的 +1，背后是大模型训练不要崩在前几千步的工程要求。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="practice-grid">
          ${code("把基准 1 写进公式", [
        "gain = nn.Parameter(torch.zeros(4))",
        "scale = 1.0 + gain",
        "y = x_norm * scale",
        "y = y.type_as(x)"
      ])}
          <div class="syntax-card">
            <h4>举一反三：搬回 TODO 1</h4>
            <div class="learn-kv">
              <span>gain</span><strong>self.weight</strong>
              <span>x_norm</span><strong>前面 FP32 算出的归一化结果</strong>
              <span>scale</span><strong>1 + self.weight</strong>
              <span>返回 dtype</span><strong>.type_as(x)</strong>
            </div>
            <p class="syntax-tip">迁移口诀：Gemma 不是乘 weight，而是乘 <code>1 + weight</code>；最后把 dtype 还给输入。</p>
          </div>
        </div>`,
      predict: {
        hook: "Gemma 把 RMSNorm 的缩放从 x·w 改成 x·(1+w)，而且 weight 还是初始化成 0。看着只是个 +1。",
        question: "先判断：这个 +1 最主要解决什么问题？",
        options: [
          "让 weight=0 的初始时刻等价于“纯归一化、原样通过”，使训练早期梯度平滑、不被清零",
          "让 RMSNorm 的计算量翻倍，从而更精确",
          "把输出强行限制在 0~1 之间"
        ],
        answer: 0,
        revealNote: "对。w 初始为 0 时，标准写法 ×0 会掐断信号；(1+w) 让它 ×1 原样通过，参数转为“学相对默认的偏移”，早期训练稳得多。"
      },
      checkpoint: checkpoint(
        "判断：Gemma RMSNorm 中 weight 初始化为 0 时，实际作用在归一化结果上的缩放是多少？",
        ["1（等价于不额外缩放，原样通过）", "0（输出被清零）", "hidden_size"],
        0,
        "实际缩放是 1 + weight = 1 + 0 = 1，所以初始等价于纯归一化层。"
      ),
      homework: [
        "TODO 1：在已算好的 x_norm 上实现 output = x_norm * (1 + self.weight)。",
        "记得 .type_as(x) 把结果转回输入精度（前面是 FP32 计算的）。",
        "跑测试：weight=0 时输出应与“无缩放纯归一化”结果一致。"
      ]
    }),
    lesson({
      id: "qwen-tie-embeddings",
      title: "权重绑定：输入查表和输出预测，共用同一张词表",
      todo: "TODO 2",
      prerequisite: [
        "embed_tokens：把 token id 查成向量，权重形状 [vocab_size, hidden]。",
        "lm_head：把 hidden 向量映射回每个词的分数(logits)，权重形状 [vocab_size, hidden]——和 embedding 一模一样！",
        "正因为两者形状相同、语义相关（都是“词 ↔ 向量”的对应表），可以让它们共用同一块参数。",
        "关键区分：相等(==，值一样) ≠ 共享(is，同一个对象)。绑定要的是“同一块内存”。"
      ],
      intuition: "模型开头“读词”（id→向量）和结尾“猜词”（向量→打分）其实在用同一套“词 ↔ 向量”的对应关系。与其各学一张大表，不如共用一张：参数省一半，而且训练时输入端和输出端的梯度都更新到同一张表，让词向量学得更充分。实现只有一行——让 lm_head.weight 直接指向 embed_tokens.weight。",
      exampleHtml: `
          <div class="shape-story">
            <div class="learn-grid">
              <div class="learn-card">
                <strong>0. 先看为什么能绑</strong>
                <p><code>embed_tokens</code> 负责 id 到向量，<code>lm_head</code> 负责 hidden 到 vocab logits。两者的权重形状都是 [vocab, hidden]，都在描述“词和向量的对应关系”。</p>
                <div class="learn-kv">
                  <span>输入表</span><strong>embed_tokens.weight [vocab, hidden]</strong>
                  <span>输出表</span><strong>lm_head.weight [vocab, hidden]</strong>
                  <span>目标</span><strong>两者指向同一个 Parameter</strong>
                </div>
              </div>
              <div class="learn-card accent">
                <strong>绑定不是复制</strong>
                <p>TODO 2 不是让两张表数值相等，而是让两个属性指向同一块内存。复制一份权重，测试仍会判错。</p>
                <div class="learn-note">要写赋值，不要写 <code>clone()</code>、<code>copy_()</code> 或重新创建 Parameter。</div>
              </div>
            </div>

            <div class="story-arrow">主线：两个名字，指向同一个 Parameter</div>
            <div class="story-panel">
              <div class="learn-flow">
                <span><b>before</b>两张独立表</span>
                <span><b>assign</b>lm_head.weight = embed.weight</span>
                <strong><b>after</b>同一块参数</strong>
              </div>
              <p>从此输入端和输出端的梯度都会更新同一张词表；词表越大，省下的参数越明显。</p>
            </div>

            <div class="learn-grid">
              <div class="learn-card good">
                <strong>1. 怎么验真共享</strong>
                <div class="learn-kv">
                  <span>对象级</span><strong>lm_head.weight is embed_tokens.weight</strong>
                  <span>内存级</span><strong>lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()</strong>
                  <span>不可靠</span><strong>shape 相同或数值相等</strong>
                </div>
              </div>
              <div class="learn-card">
                <strong>2. 参数账</strong>
                <p>词表 V、隐藏维 H，一张表就是 V*H 个参数。绑定后两张变一张，直接省下 V*H；大词表模型能省下好几 GB 显存。</p>
                <div class="learn-mini">
                  <span>不绑定<br>2*V*H</span>
                  <strong>绑定<br>1*V*H</strong>
                </div>
              </div>
            </div>

            <div class="field-note">
              <div class="fn-title">行业视角：绑不绑，是容量与成本的权衡</div>
              <p>Qwen、GPT-2 绑定权重，省参数且让词向量获得输入/输出两端监督；超大模型有时解绑，让输入表征和输出预测各自特化。能讲清这个权衡，比只会写一行赋值更重要。</p>
            </div>
          </div>`,
      syntaxHtml: `<div class="practice-grid">
          ${code("让两个属性指向同一块权重", [
        "emb = nn.Embedding(100, 16)",
        "head = nn.Linear(16, 100, bias=False)",
        "head.weight = emb.weight",
        "assert head.weight is emb.weight",
        "assert head.weight.data_ptr() == emb.weight.data_ptr()"
      ])}
          <div class="syntax-card">
            <h4>举一反三：搬回 TODO 2</h4>
            <div class="learn-kv">
              <span>emb</span><strong>self.embed_tokens</strong>
              <span>head</span><strong>self.lm_head</strong>
              <span>正确动作</span><strong>self.lm_head.weight = self.embed_tokens.weight</strong>
              <span>检查</span><strong>is 或 data_ptr()</strong>
            </div>
            <p class="syntax-tip">迁移口诀：绑定是“同一块内存”，不是“复制一份一样的值”。</p>
          </div>
        </div>`,
      predict: {
        hook: "embed_tokens（读词）和 lm_head（猜词）的权重形状一模一样，都是 [vocab, hidden]。Qwen 干脆让它们共用同一块参数。",
        question: "先判断：权重绑定（weight tying）带来的最直接收益是？",
        options: [
          "省掉一整张 [vocab, hidden] 的参数，词表越大省得越多；且输入/输出梯度共同更新词向量",
          "让模型推理速度翻倍",
          "把 vocab_size 自动缩小一半"
        ],
        answer: 0,
        revealNote: "对。词表动辄十几万，一张表就是几亿参数。共用一张省一半参数、省显存，还让词向量同时接受“读”和“猜”两端的监督。"
      },
      checkpoint: checkpoint(
        "判断权重是否真的绑定（共享同一块内存），最可靠的检查是？",
        ["lm_head.weight is embed_tokens.weight（或 data_ptr 相等）", "两者 .shape 相等", "两个模块的变量名相似"],
        0,
        "shape 相等只说明形状一样，复制品也满足；is / data_ptr 才能确认指向同一个对象、同一块内存。"
      ),
      homework: [
        "TODO 2：在 __init__ 里写 self.lm_head.weight = self.embed_tokens.weight（直接赋值，不要 clone/copy）。",
        "用 is 或 data_ptr() 验证两者共享内存；更新 embedding 后确认 lm_head 同步变化。",
        "确认 forward 的 logits 形状仍是 [..., vocab_size]，没被绑定影响。"
      ]
    })
  ],
  "09": [
    lesson({
      id: "sft-label-mask",
      title: "Loss Masking：为什么只批改“答案”，不批改“题目”",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "回忆预训练 vs SFT：预训练时每个 token 都要预测、都算 loss（读一整本书）；SFT 给的是 [Prompt(问题)] + [Response(答案)]，我们只想让模型学会“答”，不想让它学“背题干”。",
        "做法叫 Loss Masking：把 labels 里属于 prompt 和 padding 的位置全设成 -100。",
        "-100 是 PyTorch CrossEntropyLoss 的默认 ignore_index：凡是 label=-100 的位置，既不算损失、也不回传梯度，等于“这题不批改”。",
        "input_ids 和 labels 是两条等长的序列：input_ids 喂给模型看，labels 告诉损失函数“每个位置的正确答案是什么、要不要算分”。"
      ],
      intuition: "把一条 SFT 样本想成一张试卷：prompt 是印好的题干，response 是学生要写的作答。模型读整张卷子（input_ids 完整保留），但老师只批改作答区（labels 里只有 response 是真值，题干和空白都涂成 -100 表示不批改）。这一关就是构造这张“只在答案区留分”的 labels，再处理超长截断和不足填充。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先看清两条序列的关系</strong>
              <p>input_ids = prompt_ids + response_ids，整条都保留给模型看。labels 长度一样，但只有 response 段是“真答案”，prompt 段和后面 padding 段都填 -100。</p>
              <table class="freq-table">
                <tr><th style="width:110px">位置含义</th><th>input_ids（模型看的）</th><th>labels（算分用的）</th></tr>
                <tr><td>prompt 题干</td><td><code>[10, 20, 30]</code></td><td><code style="color:#be3f5b">[-100, -100, -100]</code></td></tr>
                <tr><td>response 答案</td><td><code>[40, 50, 60, 70]</code></td><td><code style="color:#12805c">[40, 50, 60, 70]</code></td></tr>
                <tr><td>padding 补位</td><td><code>[0]</code></td><td><code style="color:#be3f5b">[-100]</code></td></tr>
              </table>
              <p>这正是 notebook 测试期望的：<code>labels = [-100,-100,-100, 40,50,60,70, -100]</code>（max_len=8）。</p>
            </div>
            <div class="story-arrow">为什么题干要涂掉？想想“背题”和“答题”的区别</div>
            <div class="story-panel">
              <strong>1. 涂掉 prompt = 不让模型把“背题干”当成任务</strong>
              <p>如果 prompt 也算 loss，模型会花容量去学“人类爱怎么提问”，而不是“怎么答得好”。把它设成 -100，梯度就只从答案区回传，模型的全部注意力都用在“给定这个问题，下一个词该答什么”。</p>
              <div class="mini-table"><span>算 loss 的位置</span><strong>只有 response 段</strong><span>其余</span><span>-100 全部跳过</span></div>
              <details class="think">
                <summary>想一想：padding 位为什么也必须设成 -100，而不是随便填个 pad_id？</summary>
                <div class="think-body">
                  <p>padding 只是为了把一个 batch 里长短不一的样本对齐成矩形，它不是真实内容。如果 labels 里 padding 位留着 pad_id，模型就会被要求去“预测 padding”，白白浪费容量、还污染 loss。</p>
                  <p>所以 input_ids 的 padding 填 pad_id（占位），但 labels 的 padding 必须填 -100（不批改）。两条序列在 padding 段的填法故意不同——这是最容易写错的地方。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">TODO 2：超长要截、不足要补</div>
            <div class="story-panel">
              <strong>2. 截断与填充：把每条样本对齐到 max_len</strong>
              <p>拼完可能比 max_len 长（要从末尾截掉）或短（要在末尾补齐）。两条序列同步操作，只是填充值不同：input_ids 补 pad_id，labels 补 -100。</p>
              <div class="mini-flow"><span>超长</span><span>input_ids[:max_len]<br>labels[:max_len]</span><strong>不足<br>input_ids+[pad_id]*n<br>labels+[-100]*n</strong></div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：SFT 数据构造是“脏活”，但决定成败</div>
              <p>面试里“你怎么构造 input_ids 和 labels、为什么 mask 掉 prompt”几乎必问，因为它直接暴露你是不是真做过训练。现实中多轮对话、system prompt、工具调用的 mask 规则更复杂（比如只算最后一轮助手回复），但内核就是这关：<strong>用 -100 精确控制“哪些 token 要学、哪些只读”</strong>。</p>
              <p>一个常见线上事故就是 mask 写错——模型把用户的问法也学了进去，微调完开始“复读”用户，而不是回答。</p>
            </div>
          </div>`,
      syntaxHtml: code("拼接 → 造 labels → 截断/填充", [
        "input_ids = prompt_ids + response_ids",
        "labels = [-100] * len(prompt_ids) + response_ids   # 题干涂掉，答案保留",
        "if len(input_ids) > max_len:                        # 超长：末尾截断",
        "    input_ids, labels = input_ids[:max_len], labels[:max_len]",
        "else:                                               # 不足：末尾填充",
        "    pad = max_len - len(input_ids)",
        "    input_ids += [pad_id] * pad                     # 占位",
        "    labels    += [-100]   * pad                     # 不批改"
      ]),
      predict: {
        hook: "SFT 样本是 [Prompt] + [Response]。构造 labels 时，我们偏偏把 Prompt 那一段全设成 -100。",
        question: "先判断：把 prompt 段的 label 设成 -100，最主要是为了？",
        options: [
          "让 prompt 段不算 loss、不回传梯度，逼模型学“回答”而不是“背题干”",
          "为了让 input_ids 变短，加快训练",
          "因为 -100 是 prompt 的特殊 token id"
        ],
        answer: 0,
        revealNote: "对。-100 是 CrossEntropyLoss 的 ignore_index：这些位置被完全跳过。模型照样能“看到”prompt（input_ids 保留），但不会被要求去预测它。"
      },
      checkpoint: checkpoint(
        "动手算：prompt=[10,20,30]、response=[40,50,60,70]、max_len=8、pad_id=0，labels 应该是？",
        ["[-100,-100,-100, 40,50,60,70, -100]", "[10,20,30, 40,50,60,70, 0]", "[-100,-100,-100, 40,50,60,70, 0]"],
        0,
        "prompt 3 位涂 -100，response 4 位保留原值，最后 1 位 padding 也涂 -100。"
      ),
      homework: [
        "TODO 1：input_ids = prompt_ids + response_ids；labels = [-100]*len(prompt_ids) + response_ids。",
        "TODO 2：超长则两条都 [:max_len]；不足则 input_ids 补 pad_id、labels 补 -100，补到 max_len。",
        "跑测试：labels 应等于 [-100,-100,-100,40,50,60,70,-100]。"
      ]
    }),
    lesson({
      id: "sft-shift-ce",
      title: "Shift 对齐：位置 t 的输出，去预测位置 t+1 的词",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "自回归模型的本质：看着前面的词，预测下一个词。所以位置 t 的输出 logits，目标是“第 t+1 个 token”。",
        "模型一次性输出所有位置的 logits：[batch, seq_len, vocab]。labels 是 [batch, seq_len]。它们没有天然对齐——需要错位一格。",
        "错位做法：logits 砍掉最后一个位置（它没有“下一个词”可预测），labels 砍掉第一个位置（它没有“上一个词”来预测它）。",
        "CrossEntropyLoss 要求 logits 形状 [N, vocab]、labels 形状 [N]，所以要先 flatten；ignore_index=-100 让上一关涂掉的位置自动不算分。"
      ],
      intuition: "把序列想成一排格子，模型站在每个格子上猜“下一格是什么字”。第 0 格的猜测要和第 1 格的真值比、第 1 格的猜测和第 2 格比……最后一格没有下一格，它的猜测作废。于是 logits 去尾、labels 去头，两者就一一对齐了。对齐后拉平成一长条，交给交叉熵。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先看清“错位”从哪来</strong>
              <p>模型在位置 t 看到的是 token[0..t]，它输出的 logits[t] 是在预测 token[t+1]。所以 logits[t] 的正确答案是 labels[t+1]——天生差一格。</p>
              <svg viewBox="0 0 460 130" width="100%" style="max-width:600px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12" text-anchor="middle">
                  <text x="30" y="30" fill="#2563eb">logits</text>
                  <rect x="70" y="16" width="44" height="26" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="92" y="34">t=0</text>
                  <rect x="118" y="16" width="44" height="26" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="140" y="34">t=1</text>
                  <rect x="166" y="16" width="44" height="26" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="188" y="34">t=2</text>
                  <rect x="214" y="16" width="44" height="26" fill="#f3f4f6" stroke="#cbd5e1" stroke-dasharray="4 3"></rect><text x="236" y="34" fill="#94a3b8">t=3✂</text>
                  <text x="30" y="96" fill="#12805c">labels</text>
                  <rect x="70" y="82" width="44" height="26" fill="#f3f4f6" stroke="#cbd5e1" stroke-dasharray="4 3"></rect><text x="92" y="100" fill="#94a3b8">✂t=0</text>
                  <rect x="118" y="82" width="44" height="26" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="140" y="100">t=1</text>
                  <rect x="166" y="82" width="44" height="26" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="188" y="100">t=2</text>
                  <rect x="214" y="82" width="44" height="26" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="236" y="100">t=3</text>
                  <g stroke="#bd6516" stroke-width="1.5">
                    <line x1="92" y1="44" x2="140" y2="80"></line>
                    <line x1="140" y1="44" x2="188" y2="80"></line>
                    <line x1="188" y1="44" x2="236" y2="80"></line>
                  </g>
                  <text x="360" y="62" fill="#bd6516">斜线：谁预测谁</text>
                </g>
              </svg>
              <p>去掉两头灰色虚线格，剩下的就一一对上了：logits[0→2] 对 labels[1→3]。</p>
            </div>
            <div class="story-arrow">TODO 3：用切片实现错位</div>
            <div class="story-panel">
              <strong>1. logits 去尾、labels 去头</strong>
              <div class="mini-flow"><span>shift_logits = logits[..., :-1, :]</span><strong>shift_labels = labels[..., 1:]</strong></div>
              <p>notebook 里 logits 是 [2,5,10]，<code>logits[..., :-1, :]</code> 把 seq 从 5 砍到 4 → [2,4,10]；labels 同理去掉第一个 → [2,4]。数量对齐了。</p>
              <details class="think">
                <summary>想一想：为什么是 logits 去尾、labels 去头，而不是反过来？</summary>
                <div class="think-body">
                  <p>因为“预测方向”是往后看。最后一个位置的 logits 想预测的是“序列外的下一个词”，我们没有它的真值，只能丢掉——所以砍 logits 的尾。</p>
                  <p>第一个 token 是模型的输入起点，没有“前文”来预测它，它不该作为被预测的目标——所以砍 labels 的头。反过来切会把对应关系彻底错乱，loss 变成噪声。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">TODO 4：展平 + 交叉熵</div>
            <div class="story-panel">
              <strong>2. 拉平成一长条，喂给 CrossEntropyLoss</strong>
              <p>CrossEntropyLoss 要的是 [N, vocab] 和 [N]。把 [batch, seq, vocab] 拉成 [batch*seq, vocab]、labels 拉成 [batch*seq]。<code>ignore_index=-100</code> 会自动跳过上一关涂成 -100 的所有位置（prompt+padding）。</p>
              <div class="mini-flow"><span>shift_logits.view(-1, vocab)</span><span>shift_labels.view(-1)</span><strong>CrossEntropyLoss(ignore_index=-100)</strong></div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：这就是所有 LLM 训练的心脏</div>
              <p>不管是预训练、SFT 还是指令微调，最终的监督信号都归结到这一步：shift 一格、拉平、交叉熵。HuggingFace 的 <code>modeling_llama.py</code> 里 loss 段几乎和你写的一模一样。理解它，你就能读懂任何开源模型的训练代码。</p>
              <p>上一关的 -100 mask + 这一关的 shift 对齐，合起来就是“只在答案区、只算下一个词预测”——SFT 的全部秘密。</p>
            </div>
          </div>`,
      syntaxHtml: code("shift 对齐 + 展平 + 交叉熵", [
        "shift_logits = logits[..., :-1, :].contiguous()   # 去掉最后一个位置",
        "shift_labels = labels[..., 1:].contiguous()        # 去掉第一个位置",
        "loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 自动跳过",
        "loss = loss_fct(",
        "    shift_logits.view(-1, shift_logits.size(-1)),   # [N, vocab]",
        "    shift_labels.view(-1),                          # [N]",
        ")"
      ]),
      predict: {
        hook: "模型一次输出 logits[batch, seq, vocab]，labels 是 [batch, seq]。算 loss 前，代码先把 logits 砍掉最后一个位置、labels 砍掉第一个位置。",
        question: "先判断：为什么要这样错位切一刀？",
        options: [
          "因为位置 t 的 logits 预测的是第 t+1 个 token，错位一格才能让“预测”和“真值”对齐",
          "为了让 logits 和 labels 的 vocab 维度相等",
          "为了减少一半计算量"
        ],
        answer: 0,
        revealNote: "对。自回归=用前文猜下一个词。logits[t] 的目标是 labels[t+1]，所以 logits 去尾、labels 去头，两者一一对上。"
      },
      checkpoint: checkpoint(
        "动手算：logits 的 shape 是 [2,5,10]，shift_logits = logits[..., :-1, :] 之后 shape 是？",
        ["[2, 4, 10]", "[2, 5, 9]", "[1, 5, 10]"],
        0,
        "去掉最后一个时间步，seq 从 5 变 4，batch 和 vocab 不变 → [2,4,10]。"
      ),
      homework: [
        "TODO 3：shift_logits = logits[..., :-1, :]，shift_labels = labels[..., 1:]（加 .contiguous() 更稳）。",
        "TODO 4：用 view 展平成 [N,vocab] 和 [N]，调用 nn.CrossEntropyLoss(ignore_index=-100)。",
        "跑测试：只在 response 位置算 loss，正确预测时 loss 应接近 0。"
      ]
    })
  ],
  "10": [
    lesson({
      id: "lora-low-rank-adapter",
      title: "LoRA：冻结大矩阵，旁边挂一条可训练的低秩小路",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "痛点：全参微调一个 7B 模型，光 AdamW 优化器就要存参数的动量+方差，显存约是参数量的十几倍，个人和中小团队根本扛不住。",
        "LoRA 的想法：冻结原始大权重 W₀（不训练它），只在旁边挂两个很小的矩阵 A、B，用它们的乘积 BA 去近似“微调带来的改变量 ΔW”。",
        "A 的形状 [r, in]（降维），B 的形状 [out, r]（升维），中间的 r 叫“秩”，通常很小（8/16）。只有 A、B 需要训练。",
        "关键初始化：B 必须初始化为全 0，A 用随机（Kaiming）。这样一开始 BA=0 → ΔW=0 → 微调前的模型输出和原模型完全一致。"
      ],
      intuition: "别去改那面已经砌好的大墙（W₀ 冻结），而是在墙边搭一条细窄的旁路：输入先被 A 压缩到很低的维度 r（瓶颈），再被 B 放大回原来的宽度。这条旁路的参数量 r×(in+out) 远小于整面墙 in×out，所以又省显存又省优化器状态。这一关先把“冻结的墙 + 可训练的窄旁路”这套零件搭出来。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 为什么“低秩”能省这么多？先看参数量对比</strong>
              <p>原始权重 W₀ 是 [out, in]，参数量 out×in。LoRA 旁路只有 A[r,in] + B[out,r]，参数量 r×(in+out)。当 r 远小于 in、out 时，差距是几个数量级。</p>
              <table class="freq-table">
                <tr><th>矩阵</th><th>形状</th><th>参数量（in=out=4096, r=8）</th></tr>
                <tr><td>W₀（冻结）</td><td>[out, in]</td><td>约 1678 万</td></tr>
                <tr><td>lora_A（训练）</td><td>[r, in]</td><td>32768</td></tr>
                <tr><td>lora_B（训练）</td><td>[out, r]</td><td>32768</td></tr>
                <tr><td><strong>可训练合计</strong></td><td></td><td><strong>约 6.5 万（≈原来的 0.4%）</strong></td></tr>
              </table>
            </div>
            <div class="story-arrow">这条旁路长什么样？降维 → 升维的瓶颈结构</div>
            <div class="story-panel">
              <strong>1. A 先把 in 压到 r，B 再把 r 拉回 out</strong>
              <svg viewBox="0 0 460 130" width="100%" style="max-width:600px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12" text-anchor="middle">
                  <rect x="20" y="46" width="70" height="38" rx="8" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="55" y="69">x (in)</text>
                  <line x1="90" y1="65" x2="140" y2="65" stroke="#657184" stroke-width="2" marker-end="url(#arrow10)"></line>
                  <defs><marker id="arrow10" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#657184"></path></marker></defs>
                  <rect x="140" y="50" width="70" height="30" rx="8" fill="#fff6e8" stroke="#e8b66f"></rect><text x="175" y="69">A: →r</text>
                  <rect x="235" y="54" width="30" height="22" rx="6" fill="#fde68a" stroke="#d9a15a"></rect><text x="250" y="69" font-size="11">r=8</text>
                  <line x1="210" y1="65" x2="235" y2="65" stroke="#657184" stroke-width="2"></line>
                  <line x1="265" y1="65" x2="290" y2="65" stroke="#657184" stroke-width="2" marker-end="url(#arrow10)"></line>
                  <rect x="290" y="50" width="70" height="30" rx="8" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="325" y="69">B: →out</text>
                  <line x1="360" y1="65" x2="400" y2="65" stroke="#657184" stroke-width="2" marker-end="url(#arrow10)"></line>
                  <text x="425" y="69" fill="#657184">ΔWx</text>
                  <text x="230" y="30" fill="#8a4d0a" font-size="12">窄瓶颈 r 就是“低秩”的含义</text>
                </g>
              </svg>
              <p>notebook 里主权重用 <code>nn.Linear(in,out,bias=False)</code> 承载，然后 <code>self.linear.weight.requires_grad = False</code> 冻结它；A、B 是单独的 <code>nn.Parameter</code>。</p>
            </div>
            <div class="story-arrow">TODO 2：初始化——B 为什么必须是 0</div>
            <div class="story-panel">
              <strong>2. B 初始化为 0：让微调“从原模型出发”</strong>
              <p>A 用 kaiming 随机（给旁路一点初始的方向多样性），B 用 <code>zeros_</code> 全 0。于是训练第 0 步 BA=0，ΔW=0，LoRA 模型输出 == 原模型输出。微调是“在原模型基础上慢慢加东西”，而不是“一上来就把输出打乱”。</p>
              <div class="scale-demo"><span>A: kaiming 随机</span><span>B: 全 0</span><strong>初始 BA = 0 → 不改变原输出</strong></div>
              <details class="think">
                <summary>想一想：为什么不能 A、B 都随机初始化？两个都 0 又会怎样？</summary>
                <div class="think-body">
                  <p>都随机 → 初始 ΔW≠0，微调还没开始就把预训练好的输出破坏了，等于丢掉了预训练的起点，训练不稳。</p>
                  <p>都为 0 → BA=0 满足了，但更麻烦：A、B 全 0 时，反向传播里 A 的梯度依赖 B、B 的梯度依赖 A，两个都是 0 会让梯度也卡在 0，旁路永远学不动（对称性无法打破）。所以标准做法是“一个随机打破对称、一个置零保证起点”——A 随机、B 归零。</p>
                </div>
              </details>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：LoRA 让“人人可微调”成为现实</div>
              <p>7B 全参微调约需 112GB 显存（参数+梯度+优化器），LoRA(r=8) 只要约 14GB——一张消费级卡就能跑。更妙的是：一个冻结的基座模型，可以挂载多个不同任务训练出的 A/B 适配器，推理时按需切换，这就是“一基座 + N 适配器”的部署范式（QLoRA 更进一步把基座也量化了，本仓库后面会讲）。</p>
            </div>
          </div>`,
      syntaxHtml: code("冻结主权重，定义可训练的 A/B", [
        "self.linear = nn.Linear(in_features, out_features, bias=False)",
        "self.linear.weight.requires_grad = False       # 冻结主权重",
        "self.lora_A = nn.Parameter(torch.empty(r, in_features))",
        "self.lora_B = nn.Parameter(torch.empty(out_features, r))",
        "# 初始化：A 随机打破对称，B 全 0 保证起点",
        "nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))",
        "nn.init.zeros_(self.lora_B)"
      ]),
      predict: {
        hook: "LoRA 的两个小矩阵 A、B，A 用 kaiming 随机初始化，B 却被要求严格初始化成全 0。",
        question: "先判断：为什么 B 一定要初始化为 0？",
        options: [
          "让初始 BA=0、ΔW=0，微调开始时模型输出与原预训练模型完全一致，训练从稳定起点出发",
          "为了让 B 的参数量减半",
          "因为全 0 矩阵计算更快"
        ],
        answer: 0,
        revealNote: "对。B=0 → BA=0 → 初始等价原模型。A 则要随机（不能也为 0），否则梯度卡在 0、旁路学不动——一个置零定起点，一个随机破对称。"
      },
      checkpoint: checkpoint(
        "理解：in_features=4096, out_features=4096, r=8 时，LoRA 可训练参数量约是原始权重的多少？",
        ["约 0.4%（6.5万 vs 1678万）", "约 50%", "和原始权重一样多"],
        0,
        "LoRA=r×(in+out)=8×8192≈6.5万；原始=in×out≈1678万，比值约 0.39%。这就是省显存的来源。"
      ),
      homework: [
        "TODO 1：self.linear = nn.Linear(in,out,bias=False) 并 requires_grad=False 冻结；A=[r,in]、B=[out,r] 两个 Parameter。",
        "TODO 2：kaiming_uniform_ 初始化 A，zeros_ 初始化 B。",
        "自检：初始 forward 输出应等于 self.linear(x)（因为 B=0）。"
      ]
    }),
    lesson({
      id: "lora-forward-merge",
      title: "前向两条路相加，部署时合并回一条路（零延迟）",
      todo: "TODO 3 / TODO 4",
      prerequisite: [
        "前向公式：h = W₀x + ΔWx = W₀x + (α/r)·BAx。主路照常，旁路 (x→A→B) 乘缩放系数后加上去。",
        "scaling = α/r（lora_alpha / r）：控制旁路影响强度。它让你换 r 时不用重调学习率。",
        "计算顺序：x 先过 A（降到 r 维），再过 B（升回 out 维），最后乘 scaling。用矩阵写就是 (x @ A.T) @ B.T * scaling。",
        "合并：因为 (W₀ + BA·scaling)x 恒等于 W₀x + BA·scaling·x，可以把 BA·scaling 直接加进 W₀，部署时只剩一个普通 Linear。"
      ],
      intuition: "训练时走两条路：主路是冻结的大矩阵，旁路是可训练的小补丁，输出相加。这样只有旁路在学。等训练完要部署了，把旁路的更新 (α/r)·BA 一次性加回主权重——两条路合成一条，推理时和原始 Linear 一模一样，没有任何额外计算，这就是 LoRA“零推理延迟”的杀手锏。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 前向：主路 + 旁路，两股汇合</strong>
              <svg viewBox="0 0 460 150" width="100%" style="max-width:600px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12" text-anchor="middle">
                  <rect x="18" y="58" width="56" height="34" rx="8" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="46" y="80">x</text>
                  <path d="M74 68 C120 30, 300 30, 350 58" fill="none" stroke="#2f7fc4" stroke-width="2.5" marker-end="url(#arrow10b)"></path>
                  <defs><marker id="arrow10b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#657184"></path></marker></defs>
                  <text x="200" y="34" fill="#2f7fc4">主路 W₀x（冻结）</text>
                  <line x1="74" y1="82" x2="120" y2="82" stroke="#657184" stroke-width="2"></line>
                  <rect x="120" y="66" width="46" height="30" rx="7" fill="#fff6e8" stroke="#e8b66f"></rect><text x="143" y="85">A</text>
                  <line x1="166" y1="82" x2="196" y2="82" stroke="#657184" stroke-width="2"></line>
                  <rect x="196" y="66" width="46" height="30" rx="7" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="219" y="85">B</text>
                  <line x1="242" y1="82" x2="272" y2="82" stroke="#657184" stroke-width="2"></line>
                  <rect x="272" y="66" width="70" height="30" rx="7" fill="#f4f1ff" stroke="#c9bdf0"></rect><text x="307" y="85" font-size="11">× α/r</text>
                  <circle cx="372" cy="70" r="16" fill="#fff" stroke="#657184"></circle><text x="372" y="75" font-size="18">+</text>
                  <line x1="342" y1="81" x2="357" y2="74" stroke="#657184" stroke-width="2"></line>
                  <text x="410" y="74" fill="#657184">h</text>
                  <text x="200" y="120" fill="#8a4d0a">旁路 (x→A→B)×scaling</text>
                </g>
              </svg>
              <p>代码：<code>result = self.linear(x)</code>；<code>lora_out = (x @ A.T) @ B.T * scaling</code>；<code>result += lora_out</code>。</p>
              <details class="think">
                <summary>想一想：scaling = α/r 有什么用？为什么换 r 时它能帮你少调一个超参？</summary>
                <div class="think-body">
                  <p>r 越大，BA 这条旁路本身能表达的“改变量”越强、数值也倾向更大。如果不缩放，你每换一个 r 就得重新找合适的学习率。</p>
                  <p>乘上 α/r 后，r 变大时缩放自动变小，把旁路的整体幅度拉回一个大致稳定的范围。于是你固定 α（比如 16），调 r 时更新幅度不会剧烈变化，学习率可以基本不动。这是个“解耦超参”的小工程技巧。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">TODO 4：部署时把旁路焊回主路</div>
            <div class="story-panel">
              <strong>1. 合并：(α/r)·BA 加进 W₀，两条路变一条</strong>
              <p>因为 <code>W₀x + scaling·BAx = (W₀ + scaling·BA)x</code>，训练完直接把 <code>scaling·BA</code> 加到主权重上，之后推理就是一个普通 Linear——没有旁路、没有额外矩阵乘法、零延迟。</p>
              <div class="mini-flow"><span>训练：两条路分开</span><strong>merge: W₀ += (α/r)·B@A</strong><span>推理：一条路，等价但更快</span></div>
              <details class="think">
                <summary>想一想：合并后输出和合并前应该完全一样吗？测试怎么验证？</summary>
                <div class="think-body">
                  <p>数学上完全等价，所以合并前后对同一个 x，输出应该在数值误差内相等（notebook 用 <code>torch.allclose(..., atol=1e-5)</code> 验证）。</p>
                  <p>这也是 LoRA 相比 Adapter 类方法的核心优势：Adapter 在网络里插了额外的层，推理永远多花时间；LoRA 的更新是线性的、能被“焊”回原权重，所以可以做到部署零开销。需要换任务时，再把这份更新减回去、换上另一份即可。</p>
                </div>
              </details>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：可合并性 = LoRA 的护城河</div>
              <p>“训练时省显存”很多方法都能做到，但“推理时零额外延迟”是 LoRA 独有的甜点。生产环境里，你可以为不同客户/任务各训一个几十 MB 的 LoRA，共享同一个几十 GB 的基座；上线时合并进去跑得和原模型一样快，或者用支持多 LoRA 的推理框架动态切换。这套“基座只读、补丁热插拔”的模式，是当前大模型服务化的主流姿势之一。</p>
            </div>
          </div>`,
      syntaxHtml: code("前向相加 + 合并权重", [
        "# forward：主路 + 旁路",
        "result = self.linear(x)                       # W0 x（冻结主路）",
        "lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling",
        "result = result + lora_out",
        "",
        "# merge：把低秩更新焊回主权重（零延迟推理）",
        "self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling"
      ]),
      predict: {
        hook: "LoRA 训练时走“主路 + 旁路”两条线。可它宣传的一大卖点是“推理零额外延迟”。",
        question: "先判断：LoRA 是怎么做到推理时不比原模型慢的？",
        options: [
          "把旁路更新 (α/r)·BA 直接加回主权重 W₀，合并成一个普通 Linear，推理时没有额外矩阵乘法",
          "推理时把 batch size 调大来摊薄旁路开销",
          "推理时直接丢弃旁路、不要那部分效果"
        ],
        answer: 0,
        revealNote: "对。因为 W₀x+scaling·BAx=(W₀+scaling·BA)x，更新是线性的、能焊回主权重。合并后结构与原始 Linear 完全相同，零额外开销——这正是 LoRA 胜过 Adapter 的地方。"
      },
      checkpoint: checkpoint(
        "理解：调用 merge_weights() 把更新合并进主权重后，推理时还需要单独走 A/B 旁路吗？",
        ["不需要，更新已加进主权重，走普通 Linear 即可", "需要，否则输出 shape 会变", "只有 batch_size=1 时才不需要"],
        0,
        "合并后 W₀ 已包含 (α/r)·BA，前向只需一个 Linear，和原模型等价且零额外延迟。"
      ),
      homework: [
        "TODO 3：result = self.linear(x)；lora_out = (x @ lora_A.T) @ lora_B.T * scaling；两者相加。",
        "TODO 4：self.linear.weight.data += (lora_B @ lora_A) * scaling。",
        "跑测试：改动 B 后前向应变化；merge 后输出与 merge 前用 allclose(atol=1e-5) 应一致。"
      ]
    })
  ],
  "11": [
    lesson({
      id: "wsd-warmup-stable",
      title: "Warmup + Stable：先热车再巡航，别一脚油门冲飞",
      todo: "TODO 1 / TODO 2",
      prerequisite: [
        "学习率(lr)控制每步参数更新的步长。步长太大，模型容易被冲飞(loss 变 NaN)；太小，学得慢。",
        "为什么要 Warmup：① 模型刚随机初始化时梯度又大又乱，一上来就用大 lr 会瞬间崩；② AdamW 的分母(二阶动量/方差)开局还没攒够数据、非常小，除以它会让实际步长失控。Warmup 给优化器几千步时间“热身”。",
        "Warmup 做法：lr 从 0 线性升到 base_lr。Stable 做法：保持 base_lr 不动，吃掉大部分训练数据。",
        "本关继承 PyTorch 的 LRScheduler，实现 get_lr()，用 step 判断当前在哪个阶段。"
      ],
      intuition: "把训练想成开长途车：Warmup 是低速热车（lr 从 0 慢慢升上来，别让冷引擎/乱梯度把车干坏），Stable 是上高速匀速巡航（保持最高 lr，高效吃数据）。这一关写前两段——线性升 + 恒定，都是很直白的分段函数。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 先看整条曲线的形状（本关做前两段）</strong>
              <p>WSD = Warmup（爬坡）→ Stable（平台）→ Decay（下坡）。本关负责爬坡和平台，下一关负责下坡。</p>
              <svg viewBox="0 0 460 130" width="100%" style="max-width:620px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12">
                  <line x1="40" y1="100" x2="430" y2="100" stroke="#cbd5e1"></line>
                  <line x1="40" y1="20" x2="40" y2="100" stroke="#cbd5e1"></line>
                  <text x="20" y="30" fill="#657184">lr</text>
                  <polyline points="40,100 110,30" fill="none" stroke="#2f7fc4" stroke-width="3"></polyline>
                  <polyline points="110,30 300,30" fill="none" stroke="#12805c" stroke-width="3"></polyline>
                  <path d="M300 30 Q 360 32, 400 88" fill="none" stroke="#94a3b8" stroke-width="3" stroke-dasharray="5 4"></path>
                  <text x="60" y="118" fill="#2f7fc4">Warmup</text>
                  <text x="180" y="118" fill="#12805c">Stable</text>
                  <text x="345" y="118" fill="#94a3b8">Decay(下关)</text>
                  <text x="112" y="24" fill="#657184" font-size="11">base_lr</text>
                </g>
              </svg>
            </div>
            <div class="story-arrow">TODO 1：Warmup 段，从 0 线性升到 base_lr</div>
            <div class="story-panel">
              <strong>1. 线性爬坡：lr = base_lr × step / warmup_steps</strong>
              <p>step=0 时 lr=0（引擎最冷、最保守），step 走到 warmup_steps 时正好等于 base_lr。中间按比例线性插值。</p>
              <div class="curve-legend"><span>step 0 → 0</span><span>step 半程 → base_lr/2</span><strong>step=warmup → base_lr</strong></div>
              <details class="think">
                <summary>想一想：不要 warmup、一开始就用 base_lr，最坏会发生什么？</summary>
                <div class="think-body">
                  <p>随机初始化的模型梯度方向基本是噪声，幅度还大。直接用最大 lr，第一步就可能把权重推到极端区域，loss 直接飙成 NaN——这就是训练日志里最常见的“loss spike / 开局就崩”。</p>
                  <p>再叠加 AdamW 的问题：它用梯度平方的移动平均当分母，开局这个值极小，<code>lr / (√小数)</code> 让实际步长爆炸。warmup 这几千步就是等这个分母把方差“攒稳”，同时让 lr 慢慢加上来，双保险。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">TODO 2：Stable 段，锁定 base_lr</div>
            <div class="story-panel">
              <strong>2. 平台巡航：lr = base_lr（一行）</strong>
              <p>warmup 结束到 decay 开始之间，lr 恒等于 base_lr。这是训练时间最长、loss 下降最主力的阶段。</p>
              <div class="mini-flow"><span>step < warmup</span><span>线性升</span><strong>warmup ≤ step < warmup+stable → base_lr</strong></div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：Warmup 不是玄学，是被 loss spike 教育出来的</div>
              <p>GPT-3、LLaMA 系列全部带 warmup，典型是总步数的 1%~5%（比如训 10 万步、warmup 1000~5000 步）。它几乎不花额外成本，却能挡住“开局炸炉”这种最昂贵的事故——一次千卡训练崩了重启，烧的是真金白银。所以你看任何大模型训练配置，warmup_steps 都是必填项。</p>
            </div>
          </div>`,
      syntaxHtml: code("get_lr 里用 if/elif 分段", [
        "step = self._step_count - 1",
        "if step < self.num_warmup_steps:                 # Warmup：线性升",
        "    current_lr = base_lr * step / self.num_warmup_steps",
        "elif step < self.num_warmup_steps + self.num_stable_steps:",
        "    current_lr = base_lr                          # Stable：恒定",
        "else:",
        "    ...                                           # Decay：下一关"
      ]),
      predict: {
        hook: "大模型训练一上来不敢用最大学习率，非要先花几千步把 lr 从 0 慢慢升上去（Warmup）。",
        question: "先判断：Warmup 主要在防什么？",
        options: [
          "防开局崩：随机初始化梯度乱、AdamW 方差还没攒稳，直接用大 lr 会让 loss 直接 NaN",
          "防止显存溢出",
          "让 batch size 可以设更大"
        ],
        answer: 0,
        revealNote: "对。两个原因叠加：初始梯度又大又乱 + AdamW 分母(方差)开局过小导致实际步长失控。Warmup 用几千步让 lr 缓升、让优化器攒够统计量。"
      },
      checkpoint: checkpoint(
        "动手算：base_lr=0.01、warmup_steps=100，走到 step=50 时线性 warmup 的 lr 是多少？",
        ["0.005", "0.01", "0.05"],
        0,
        "lr = base_lr × step/warmup_steps = 0.01 × 50/100 = 0.005，正好是一半。"
      ),
      homework: [
        "TODO 1：step < warmup_steps 时 current_lr = base_lr * step / num_warmup_steps（step=0 得 0）。",
        "TODO 2：进入 stable 区间时 current_lr = base_lr。",
        "跑测试：lrs[0]≈0，lrs[warmup] 应等于 max_lr，stable 段维持 max_lr。"
      ]
    }),
    lesson({
      id: "wsd-cosine-decay",
      title: "Decay：最后一程用半条余弦曲线平滑降落",
      todo: "TODO 3",
      prerequisite: [
        "Decay 是 WSD 的最后一段：在训练尾部把 lr 从 base_lr 平滑降到 min_lr，帮模型在最优解附近“精细收拢”。",
        "先算已走过 decay 的比例 decay_ratio = (step − warmup − stable) / decay_steps，范围 0→1。",
        "余弦因子 cosine = 0.5·(1 + cos(π·decay_ratio))：ratio=0 时=1，ratio=1 时=0，中间是平滑的 S 形下降。",
        "最终 lr = min_lr + (base_lr − min_lr)·cosine：把 [0,1] 的因子映射到 [min_lr, base_lr]。"
      ],
      intuition: "训练收尾不能急刹车（lr 突然归零会让模型抖动、丢掉刚学的东西），而是沿着半条余弦曲线平滑滑降。开头降得慢（还在高位多学一会），越接近终点降得越快，最后停在 min_lr。这一关就是把这个余弦公式写对。",
      exampleHtml: `
          <div class="shape-story">
            <div class="story-panel">
              <strong>0. 为什么用余弦而不是直线降？</strong>
              <p>余弦下降在两端“平、中间陡”：起点附近 lr 还维持在高位，让模型多吃一会儿；临近终点快速压到 min_lr，帮助收敛。比直线更“先松后紧”，实践中收敛更稳。</p>
              <svg viewBox="0 0 460 130" width="100%" style="max-width:620px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12">
                  <line x1="40" y1="100" x2="430" y2="100" stroke="#cbd5e1"></line>
                  <line x1="40" y1="20" x2="40" y2="100" stroke="#cbd5e1"></line>
                  <path d="M40 25 Q 235 30, 250 60 Q 265 90, 420 95" fill="none" stroke="#2f7fc4" stroke-width="3"></path>
                  <text x="46" y="20" fill="#657184" font-size="11">base_lr</text>
                  <text x="390" y="90" fill="#657184" font-size="11">min_lr</text>
                  <text x="150" y="120" fill="#2f7fc4">cosine decay（半条余弦）</text>
                </g>
              </svg>
            </div>
            <div class="story-arrow">TODO 3：先算进度，再套余弦</div>
            <div class="story-panel">
              <strong>1. 三步：进度 → 余弦因子 → 映射到 lr</strong>
              <div class="mini-flow"><span>decay_ratio = (step−warmup−stable)/decay_steps</span><span>cosine = 0.5(1+cos(π·ratio))</span><strong>lr = min_lr + (base−min)·cosine</strong></div>
              <p>验证端点：ratio=0 → cos(0)=1 → cosine=1 → lr=base_lr（decay 刚开始，接住 stable 的高位）；ratio=1 → cos(π)=−1 → cosine=0 → lr=min_lr（终点）。</p>
              <details class="think">
                <summary>想一想：为什么因子要写成 0.5·(1+cos(...))，而不是直接 cos(...)？</summary>
                <div class="think-body">
                  <p>cos(π·ratio) 本身的取值范围是 [1, −1]，直接拿它当因子会出现负数——lr 会被压到 min_lr 以下甚至变负，非法。</p>
                  <p>+1 把范围抬到 [2, 0]，再 ×0.5 压成 [1, 0]。这样因子是干净的“从 1 平滑降到 0”，乘上 (base−min) 再加 min，就稳稳落在 [min_lr, base_lr] 区间。这个 0.5·(1+cos) 是余弦退火的标准写法，记下来。</p>
                </div>
              </details>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：WSD 为什么正在取代 Cosine（LLaMA-3 的选择）</div>
              <p>传统 Cosine 退火必须一开始就定死总步数、从头到尾一路下降。麻烦在于：如果训练中途想“加数据继续训”(continued pre-training)，此时 lr 早已降到底，模型几乎学不动新知识了。</p>
              <p>WSD 把“稳定期”和“退火期”解耦：想加数据就无限延长 Stable 段（lr 一直是高位），真正要收尾时才进入这最后 10%~20% 的 Decay。这种灵活性正是 LLaMA-3 等现代持续预训练采用 WSD 的原因——而 Decay 段的余弦公式，就是你这一关写的。</p>
            </div>
          </div>`,
      syntaxHtml: code("Decay 段：余弦退火到 min_lr", [
        "min_lr = base_lr * self.min_lr_ratio",
        "decay_step = step - self.num_warmup_steps - self.num_stable_steps",
        "decay_ratio = decay_step / self.num_decay_steps        # 0 → 1",
        "cosine = 0.5 * (1 + math.cos(math.pi * decay_ratio))   # 1 → 0",
        "current_lr = min_lr + (base_lr - min_lr) * cosine"
      ]),
      predict: {
        hook: "Decay 段的余弦因子写成 cosine = 0.5 × (1 + cos(π·ratio))，那个“0.5×(1+…)”看着有点多余。",
        question: "先判断：为什么不直接用 cos(π·ratio) 当衰减因子？",
        options: [
          "因为 cos 的范围是 [1,−1]，直接用会让 lr 出现负数；0.5·(1+cos) 把它规整到干净的 [1,0]",
          "因为 0.5 能让计算更快",
          "因为 PyTorch 要求因子必须带系数"
        ],
        answer: 0,
        revealNote: "对。cos(π·ratio) 从 1 降到 −1，负半段会把 lr 压到 min_lr 以下甚至为负。+1 变 [2,0]、×0.5 变 [1,0]，才是合法的“从满降到零”因子。"
      },
      checkpoint: checkpoint(
        "理解：cosine = 0.5·(1+cos(π·ratio))，当 decay_ratio=1（decay 走完）时，lr 等于多少？",
        ["min_lr（cos(π)=−1 → 因子=0）", "base_lr（因子=1）", "0（因子无意义）"],
        0,
        "ratio=1 → cos(π)=−1 → 0.5·(1−1)=0 → lr=min_lr+(base−min)·0=min_lr，正好落到最低学习率。"
      ),
      homework: [
        "TODO 3：decay_ratio =(step−warmup−stable)/decay_steps；cosine=0.5*(1+cos(π·ratio))；lr=min_lr+(base−min)*cosine。",
        "端点自检：ratio=0 应得 base_lr，ratio=1 应得 min_lr(=base_lr*min_lr_ratio)。",
        "跑测试并看曲线：应是“升—平—余弦降”三段，末端落在 max_lr*0.1。"
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
        "int8 的有效范围通常看作 [-127, 127]，对称量化会尽量让正负两侧共用同一把刻度尺。",
        "这里的 scale 表示整数刻度密度：scale = 127 / absmax。absmax 越大，每个浮点单位对应的 int8 刻度越少。",
        "量化不是简单 to(torch.int8)：要先乘 scale，把浮点数放大到整数刻度，再 round 到最近整数。",
        "round 后还要 clamp，防止极端值越界；最后才转成 torch.int8 存储。"
      ],
      intuition: "把浮点权重压成 int8，像把真实长度画到一把只有 127 格的尺子上。先找最大绝对值 absmax，决定“最远的点”应该贴到 int8 边界；再用同一把尺子把所有权重换算成整数刻度。",
      exampleHtml: `
          <div class="shape-story">
            <div class="learn-grid">
              <div class="learn-card">
                <strong>0. 先读量化函数的合同</strong>
                <p>输入是一块浮点权重，输出不是一个张量，而是两样东西：压缩后的 int8 权重，以及以后反量化要用的 scale。</p>
                <div class="learn-kv">
                  <span>输入</span><strong>weight float tensor</strong>
                  <span>输出 1</span><strong>q_weight int8 tensor</strong>
                  <span>输出 2</span><strong>scale = 127 / absmax</strong>
                </div>
              </div>
              <div class="learn-card accent">
                <strong>scale 是“刻度密度”</strong>
                <p>量化时乘 scale，反量化时除 scale。这个 notebook 的 scale 不是每个整数代表多少浮点值，而是 1 个浮点单位对应多少 int8 格。</p>
                <div class="learn-note">记忆：<code>q = round(w * scale)</code>，所以还原就是 <code>w ≈ q / scale</code>。</div>
              </div>
            </div>

            <div class="story-arrow">主线：absmax 定尺子，乘 scale 上刻度，round/clamp 存 int8</div>
            <div class="story-panel">
              <div class="learn-flow">
                <span><b>absmax</b>weight.abs().max()</span>
                <span><b>scale</b>127.0 / absmax</span>
                <span><b>multiply</b>weight * scale</span>
                <span><b>round</b>最近整数</span>
                <strong><b>clamp + int8</b>[-128,127]</strong>
              </div>
              <p>这条线就是 TODO 1/2/3 的顺序。不要先转 int8，否则小数会直接被截断，量化误差会变得不可控。</p>
            </div>

            <div class="learn-grid">
              <div class="learn-card good">
                <strong>1. 一个小数字例子</strong>
                <div class="learn-kv">
                  <span>absmax</span><strong>3.0</strong>
                  <span>scale</span><strong>127 / 3.0 ≈ 42.33</strong>
                  <span>w=1.5</span><strong>round(1.5 * 42.33) = 64</strong>
                  <span>w=-3.0</span><strong>round(-3.0 * 42.33) = -127</strong>
                </div>
              </div>
              <div class="learn-card warn">
                <strong>2. 最常见错误</strong>
                <p>把 <code>scale</code> 写反、忘记 <code>round</code>、忘记 <code>clamp</code>，都会让测试里的误差或 dtype 检查失败。</p>
                <div class="learn-mini">
                  <span>先乘</span>
                  <span>再 round</span>
                  <span>再 clamp</span>
                  <strong>最后 int8</strong>
                </div>
              </div>
            </div>
          </div>`,
      syntaxHtml: `<div class="practice-grid">
          ${code("量化三步", [
        "absmax = weight.abs().max()",
        "scale = 127.0 / absmax",
        "q = torch.round(weight * scale)",
        "q = q.clamp(-128, 127)",
        "q = q.to(torch.int8)"
      ])}
          <div class="syntax-card">
            <h4>举一反三：对应 notebook TODO</h4>
            <div class="learn-kv">
              <span>TODO 1</span><strong>absmax = weight.abs().max()</strong>
              <span>TODO 2</span><strong>scale = 127.0 / absmax</strong>
              <span>TODO 3</span><strong>round → clamp → int8</strong>
            </div>
            <p class="syntax-tip">迁移口诀：量化乘 scale，反量化除 scale；scale 方向别写反。</p>
          </div>
        </div>`,
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
        "W8A16 表示 weight 用 int8 存储，activation 仍用 fp16/bf16 等 16-bit 浮点参与计算。",
        "普通 PyTorch 的 F.linear 不能直接拿 int8 权重和 fp16 激活做这个玩具前向，所以先把 q_weight 反量化回浮点。",
        "本 notebook 的量化公式是 q = round(w * scale)，因此反量化公式是 w ≈ q / scale。",
        "反量化后的权重要转成 x.dtype，保证和激活 dtype 对齐。"
      ],
      intuition: "W8A16 的核心是“存的时候省，算的时候稳”：权重平时压成 int8 省显存；真正做线性层前，再用 scale 还原成和激活同 dtype 的浮点权重。你写的 TODO 4 就是这条恢复路径。",
      exampleHtml: `
          <div class="shape-story">
            <div class="learn-grid">
              <div class="learn-card">
                <strong>0. 先看 W8A16 的两种 dtype</strong>
                <div class="learn-kv">
                  <span>W8</span><strong>q_weight 存成 torch.int8</strong>
                  <span>A16</span><strong>x 保持 fp16/bf16 等浮点激活</strong>
                  <span>计算前</span><strong>q_weight.float() / scale</strong>
                </div>
              </div>
              <div class="learn-card accent">
                <strong>TODO 4 不是重新量化</strong>
                <p>前面已经把权重量化好了。这里要做的是反量化，再调用 <code>F.linear</code>，最后输出形状和普通 Linear 一样。</p>
              </div>
            </div>

            <div class="story-arrow">主线：int8 权重除以 scale，还原成和激活同 dtype 的浮点权重</div>
            <div class="story-panel">
              <div class="learn-flow">
                <span><b>q_weight</b>int8</span>
                <span><b>float()</b>避免整数除法/溢出</span>
                <span><b>/ scale</b>反量化</span>
                <span><b>to(x.dtype)</b>对齐激活</span>
                <strong><b>F.linear</b>x, dequant_weight, bias</strong>
              </div>
              <p>量化时是乘 scale，反量化时必须除以同一个 scale。这里最容易写错方向。</p>
            </div>

            <div class="learn-grid">
              <div class="learn-card good">
                <strong>1. 为什么要 <code>.float()</code></strong>
                <p><code>q_weight</code> 是 int8 存储格式，先转浮点再除以 scale，才能得到近似原权重的浮点值。</p>
              </div>
              <div class="learn-card warn">
                <strong>2. 为什么要 <code>.to(x.dtype)</code></strong>
                <p>激活是 A16，权重反量化后对齐到 <code>x.dtype</code>，避免线性层里 dtype 不匹配，也更贴近 W8A16 的含义。</p>
              </div>
            </div>
          </div>`,
      syntaxHtml: `<div class="practice-grid">
          ${code("反量化后做线性层", [
        "dequant_weight = q_weight.float() / scale",
        "dequant_weight = dequant_weight.to(x.dtype)",
        "out = F.linear(",
        "    x,",
        "    dequant_weight,",
        "    bias,",
        ")"
      ])}
          <div class="syntax-card">
            <h4>举一反三：TODO 4 检查表</h4>
            <div class="learn-kv">
              <span>量化方向</span><strong>q = round(w * scale)</strong>
              <span>反量化方向</span><strong>w = q.float() / scale</strong>
              <span>dtype 对齐</span><strong>dequant_weight.to(x.dtype)</strong>
              <span>前向</span><strong>F.linear(x, dequant_weight, bias)</strong>
            </div>
            <p class="syntax-tip">迁移口诀：W8 是存储格式，A16 是计算激活；做 Linear 前先把权重恢复成浮点。</p>
          </div>
        </div>`,
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
