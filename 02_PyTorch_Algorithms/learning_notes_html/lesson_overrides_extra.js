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
            <div class="story-panel">
              <strong>0. 先看清输入长什么样</strong>
              <p>notebook 给你的 <code>selected_experts</code> 形状是 <code>[总token数, top_k]</code>：每行是一个 token，行里的 K 个数字是它选中的 K 个专家 id。我们要把它压成一张长度为 E 的“人头表” f。</p>
              <table class="freq-table">
                <tr><th>token</th><th>选中的专家 (top_k=2)</th></tr>
                <tr><td>t0</td><td>[0, 2]</td></tr>
                <tr><td>t1</td><td>[0, 1]</td></tr>
                <tr><td>t2</td><td>[0, 2]</td></tr>
              </table>
              <p style="color:#657184">直觉上专家 0 被点了 3 次、专家 2 两次、专家 1 一次，专家 3+ 颗粒无收——这就是不均衡。</p>
            </div>
            <div class="story-arrow">怎么用张量数人头？→ one_hot 再求和</div>
            <div class="story-panel">
              <strong>1. one_hot：把“专家 id”变成“可以相加的计数”</strong>
              <p>id 本身没法直接加（把 0 和 2 加起来没意义）。<code>F.one_hot</code> 把每个 id 摊成一行 0/1 指示向量，命中位为 1。这样“相加”就等于“计票”。</p>
              <svg viewBox="0 0 460 150" width="100%" style="max-width:620px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12" text-anchor="middle">
                  <text x="60" y="20" fill="#657184">id → one-hot (E=4)</text>
                  <text x="300" y="20" fill="#657184">沿 token/top_k 求和 = 每个专家票数</text>
                  <text x="20" y="52">[0,2]</text>
                  <text x="20" y="82">[0,1]</text>
                  <text x="20" y="112">[0,2]</text>
                  <g font-family="ui-monospace, monospace">
                    <text x="95" y="52">1 0 1 0</text>
                    <text x="95" y="82">1 1 0 0</text>
                    <text x="95" y="112">1 0 1 0</text>
                  </g>
                  <line x1="150" y1="82" x2="210" y2="82" stroke="#657184" stroke-width="2"></line>
                  <text x="180" y="72" fill="#12805c">Σ</text>
                  <g font-family="ui-monospace, monospace" font-weight="700">
                    <rect x="240" y="68" width="150" height="28" fill="#e7f6ee" stroke="#99d6bf"></rect>
                    <text x="315" y="87" fill="#12805c">[3, 1, 2, 0]</text>
                  </g>
                  <text x="315" y="118" fill="#657184">专家0:3  专家1:1  专家2:2  专家3:0</text>
                </g>
              </svg>
              <p>再除以总选择次数 <code>total_tokens * top_k</code>（这里 3×2=6），就得到比例 <code>f = [3,1,2,0]/6</code>。所有 f_i 加起来正好是 1。</p>
              <details class="think">
                <summary>想一想：为什么要除以 total_tokens × top_k，而不是只除以 token 数？</summary>
                <div class="think-body">
                  <p>因为每个 token 选了 K 个专家，总共投出的“票”是 token 数 × K，不是 token 数。分母要和“总票数”一致，f 才是真正的比例、加起来才等于 1。</p>
                  <p>这是 Top-K MoE 最容易写错的地方：Top-1 时 K=1 看不出问题，一旦 K=2 还按 token 数除，f 的和就会变成 2，后面的 loss 全错。</p>
                </div>
              </details>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：路由崩塌是真金白银的损失</div>
              <p>Mixtral 8x7B、DeepSeek-MoE 这类模型，专家是分散在多张 GPU 上的。如果 router 把 token 都挤给少数专家，少数 GPU 会 OOM 或排长队，多数 GPU 在空转——你为整套集群付了钱，却只用上一小部分算力。</p>
              <p>所以 f_i 不是学术指标，它直接对应“集群利用率”。数好这张人头表，是下一步设计惩罚项、把负载摊平的前提。</p>
            </div>
          </div>`,
      syntaxHtml: code("one_hot 计票 + 归一化", [
        "ids = torch.tensor([[0, 2], [0, 1], [0, 2]])  # [tokens, top_k]",
        "hot = F.one_hot(ids, num_classes=4)           # [tokens, top_k, 4]",
        "counts = hot.sum(dim=(0, 1)).float()          # [4] -> [3,1,2,0]",
        "f_i = counts / (ids.shape[0] * ids.shape[1])  # 除以总票数 3*2=6"
      ]),
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
            <div class="story-panel">
              <strong>0. 两个量、两种性格：一个能教模型，一个只会告状</strong>
              <table class="freq-table">
                <tr><th style="width:70px">量</th><th>怎么来</th><th>性格</th></tr>
                <tr><td><strong>P_i</strong></td><td style="text-align:left">router 权重按专家累加再平均（scatter_add_）</td><td style="text-align:left">连续、<strong>可导</strong>——梯度能顺着它流回 router</td></tr>
                <tr><td><strong>f_i</strong></td><td style="text-align:left">Top-K 选中次数的比例（one_hot 计数）</td><td style="text-align:left">离散、<strong>不可导</strong>——只能当“权重”，不能当被优化对象</td></tr>
              </table>
              <p>这就是为什么公式要把它俩<strong>相乘</strong>：用不可导的 f_i 标出“哪里堵”，用可导的 P_i 把惩罚的梯度送回 router 去改。</p>
            </div>
            <div class="story-arrow">第一步：用 scatter_add_ 把权重“倒进”各自专家的桶里（TODO 1）</div>
            <div class="story-panel">
              <strong>1. scatter_add_：按专家 id 归集 router 权重 → P_i</strong>
              <p>开一个长度 E 的全零桶。遍历每个 token 的每个选择，把它的 routing_weight 加到“它选的那个专家”对应的桶里。最后除以总票数得到平均。</p>
              <svg viewBox="0 0 460 140" width="100%" style="max-width:620px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="12" text-anchor="middle">
                  <text x="100" y="20" fill="#657184">(专家id, 权重)</text>
                  <text x="60" y="48">(0, .6)</text>
                  <text x="60" y="72">(2, .4)</text>
                  <text x="60" y="96">(0, .5)</text>
                  <line x1="120" y1="70" x2="190" y2="70" stroke="#657184" stroke-width="2" marker-end="url(#arrow07)"></line>
                  <defs><marker id="arrow07" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#657184"></path></marker></defs>
                  <text x="155" y="60" fill="#12805c">scatter_add_</text>
                  <g font-family="ui-monospace, monospace">
                    <rect x="210" y="40" width="60" height="26" fill="#fff6e8" stroke="#e8b66f"></rect><text x="240" y="58">桶0: 1.1</text>
                    <rect x="210" y="70" width="60" height="26" fill="#eef2f7" stroke="#cbd5e1"></rect><text x="240" y="88">桶1: 0</text>
                    <rect x="280" y="40" width="60" height="26" fill="#fff6e8" stroke="#e8b66f"></rect><text x="310" y="58">桶2: 0.4</text>
                    <rect x="280" y="70" width="60" height="26" fill="#eef2f7" stroke="#cbd5e1"></rect><text x="310" y="88">桶3: 0</text>
                  </g>
                  <text x="380" y="74" fill="#657184">÷ 总票数<br/>= P_i</text>
                </g>
              </svg>
              <p>代码就三行：<code>P=torch.zeros(E)</code> → <code>P.scatter_add_(0, ids.flatten(), w.flatten())</code> → <code>P/=(total_tokens*top_k)</code>。</p>
            </div>
            <div class="story-arrow">第二步：组合成损失（TODO 3）</div>
            <div class="story-panel">
              <strong>2. L_aux = α · E · Σ(f_i · P_i)：为什么均匀时最小</strong>
              <p>f 和 P 的各自总和都固定（都=1）。由均值不等式，两个“总和固定”的正向量逐项相乘再相加，当它们都被摊成平的(每项都=1/E)时，乘积和最小；越尖锐(集中在少数专家)，乘积和越大。</p>
              <div class="formula"><span>集中: f=[.9,.05,.05,0]<br>Σf·P 大 → 重罚</span><strong>均匀: f=P=[¼,¼,¼,¼]<br>Σf·P 最小 → 不罚</strong></div>
              <details class="think">
                <summary>想一想：f_i 不可导，那这个 loss 到底是怎么把梯度传回 router 的？</summary>
                <div class="think-body">
                  <p>关键在于乘积里 <strong>f_i 被当成常数（detach 掉的系数）</strong>，真正带梯度的是 P_i。反向传播时，loss 对 P_i 求导，梯度顺着 router 权重流回去。</p>
                  <p>效果上：某个专家 f_i 大（堵），它对应的 P_i 就被施加一个“往下压”的梯度，router 于是学会少给它发 token。f_i 只负责“指认谁堵”，P_i 负责“承接惩罚”。这正是上一关 predict 里那个“不可导怎么办”的答案落地。</p>
                </div>
              </details>
              <details class="think">
                <summary>想一想：完全均匀时 loss 等于多少？（提示：代入 f_i=1/E, P_i=1/(E·K)）</summary>
                <div class="think-body">
                  <p>均匀时每个专家 f_i = 1/E；而 P_i 是权重平均，均匀分到 E 个专家、每 token 投 K 票，P_i = 1/(E·K)。</p>
                  <p>Σ(f_i·P_i) = E · (1/E)·(1/(E·K)) = 1/(E·K)。乘上 α·E 得 <strong>α/K</strong>。所以 notebook 测试里 α=0.01、K=2 时，均匀分配的理论最小 loss = 0.005。这就是测试断言的那个数。</p>
                </div>
              </details>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：一个系数 α 的拿捏</div>
              <p>α 太大，模型会为了“摊平负载”牺牲主任务效果（强行把不适合的 token 发给冷门专家）；α 太小，又压不住路由崩塌。Switch Transformer、Mixtral 这类工作里，α≈0.01 是反复试出来的折中。</p>
              <p>这关你写的 aux_loss，在真实训练里是直接加到主 CrossEntropy 上的：<code>total = ce_loss + aux_loss</code>。理解它，你才能解释“为什么我的 MoE 训着训着专家就废了”。</p>
            </div>
          </div>`,
      syntaxHtml: code("scatter_add 求 P_i，再组合损失", [
        "P_i = torch.zeros(num_experts, device=routing_weights.device)",
        "P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())",
        "P_i = P_i / (total_tokens * top_k)        # 平均路由概率",
        "# f_i 来自上一关的 one_hot 计数",
        "aux_loss = alpha * num_experts * (f_i * P_i).sum()"
      ]),
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
            <div class="story-panel">
              <strong>0. 先看清坑在哪：w 初始化为 0 时，两种写法天差地别</strong>
              <table class="freq-table">
                <tr><th style="width:130px">写法</th><th>w=0 时的缩放</th><th>后果</th></tr>
                <tr><td>标准 x_norm · w</td><td><strong style="color:#be3f5b">× 0</strong></td><td style="text-align:left">输出全被乘成 0，信号被掐断，早期梯度很差</td></tr>
                <tr><td>Gemma x_norm · (1+w)</td><td><strong style="color:#12805c">× 1</strong></td><td style="text-align:left">原样通过，等价纯归一化，训练平滑启动</td></tr>
              </table>
            </div>
            <div class="story-arrow">为什么“默认 1”这么关键？想象训练第 0 步</div>
            <div class="story-panel">
              <strong>1. 把“学绝对值”改成“学相对默认的偏移”</strong>
              <p>参数从 0 开始更新是常态。标准写法逼着 w 从 0 慢慢爬到 ~1 才能让信号正常通过，这段时间整层都在“半掐断”状态；Gemma 让 w=0 就已经是“正常通过”，w 只需学习“相对 1 的微调量”，起点就健康。</p>
              <div class="scale-demo"><span>x_norm</span><span>w=[0.1, -0.2]</span><strong>scale = 1+w = [1.1, 0.8]</strong></div>
              <details class="think">
                <summary>想一想：为什么不直接把 weight 初始化成 1，效果不就一样了吗？</summary>
                <div class="think-body">
                  <p>数值上确实可以——把 w 初始化成全 1、用标准 x_norm·w，起点也是“×1”。Gemma 选择“初始化成 0 + 写成 (1+w)”，好处是让“零初始化”这个通用、稳健的默认习惯继续适用，参数语义也更统一：所有可学习量都从 0 出发，代表“相对基准的偏移”。</p>
                  <p>很多框架/优化器、权重衰减(weight decay)都默认把参数往 0 拉。如果你把基准烤进“初始化值=1”，weight decay 会一直想把它拽回 0（=掐断）；而写成 (1+w)、对 w 做 decay，拉向的是“w=0 即 ×1”的健康默认。这就是把基准写进公式、而非写进初始化的深层原因。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">写代码时别忘了精度</div>
            <div class="story-panel">
              <strong>2. 顺序：FP32 归一化 → ×(1+w) → type_as 转回</strong>
              <p>notebook 已经帮你算好了 FP32 下的 <code>x_norm</code>。你的 TODO 只差最后一步：乘上 <code>(1 + self.weight)</code>，再 <code>.type_as(x)</code> 转回输入精度。一行就够。</p>
              <div class="mini-flow"><span>x.float()</span><span>归一化得 x_norm</span><strong>x_norm * (1 + weight) → type_as(x)</strong></div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：训练稳定性是大模型的隐形成本</div>
              <p>千亿参数模型训练一次要烧掉海量算力，一旦中途 loss 发散，重启的代价极高。Gemma 的 (1+w)、LLaMA 的 Pre-Norm、各种初始化技巧，目标都一样：让训练在最脆弱的早期稳稳启动。这些看似“一个 +1”的小改动，背后是“别让我几百万的训练任务崩在第一千步”的工程焦虑。</p>
            </div>
          </div>`,
      syntaxHtml: code("把基准 1 写进公式，参数从 0 学偏移", [
        "gain = nn.Parameter(torch.zeros(4))   # 默认 0",
        "scale = 1.0 + gain                    # w=0 时 scale=1（原样通过）",
        "y = x_norm * scale                    # 再 .type_as(x) 转回原精度"
      ]),
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
            <div class="story-panel">
              <strong>0. 为什么这两张表能合并？先看形状</strong>
              <p>词表常有 15 万+ 个词。embed_tokens 的权重是 [vocab, hidden]，lm_head（bias=False）的权重也是 [vocab, hidden]。形状完全一致，且都在描述“每个词对应哪个向量”——天然适合共用。</p>
              <div class="lookup"><span>embed_tokens.weight<br>[vocab, hidden]</span><strong>同一块内存</strong><span>lm_head.weight<br>[vocab, hidden]</span></div>
            </div>
            <div class="story-arrow">怎么“共用”？不是复制，是让两个名字指向同一个对象</div>
            <div class="story-panel">
              <strong>1. 一行赋值：lm_head.weight = embed_tokens.weight</strong>
              <p>这行不复制数据，而是让 lm_head 的 weight 属性指向 embedding 那块 Parameter。从此改一个，另一个自动同步——因为它们本就是同一块内存。</p>
              <svg viewBox="0 0 420 120" width="100%" style="max-width:560px;border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:8px">
                <g font-size="13" text-anchor="middle">
                  <rect x="20" y="30" width="120" height="34" rx="8" fill="#eaf2ff" stroke="#a9c2f6"></rect><text x="80" y="52">embed_tokens.weight</text>
                  <rect x="20" y="74" width="120" height="34" rx="8" fill="#e7f6ee" stroke="#99d6bf"></rect><text x="80" y="96">lm_head.weight</text>
                  <line x1="140" y1="47" x2="250" y2="64" stroke="#657184" stroke-width="2"></line>
                  <line x1="140" y1="91" x2="250" y2="68" stroke="#657184" stroke-width="2"></line>
                  <rect x="250" y="48" width="150" height="34" rx="8" fill="#f4f1ff" stroke="#c9bdf0"></rect><text x="325" y="70">同一个 Parameter</text>
                </g>
              </svg>
              <details class="think">
                <summary>想一想：怎么确认是“真共享”而不是“值刚好相等”？</summary>
                <div class="think-body">
                  <p>用 <code>is</code> 或 <code>data_ptr()</code>。<code>a is b</code> 检查两个引用是不是同一个对象；<code>a.data_ptr() == b.data_ptr()</code> 检查底层数据是不是同一块内存地址。</p>
                  <p>注意 <code>shape 相同</code> 或 <code>值相等(==)</code> 都骗不过：复制一份权重，shape 和值都一样，但改了原表副本不会跟着变——那不是绑定。notebook 测试就是用更新 embedding 后看 lm_head 是否同步、以及 data_ptr 是否相等来验真的。</p>
                </div>
              </details>
            </div>
            <div class="story-arrow">省了多少？算一笔账</div>
            <div class="story-panel">
              <strong>2. 参数账：词表越大，省得越狠</strong>
              <p>词表 V、隐藏维 H，一张表就是 V×H 个参数。绑定后两张变一张，直接省下 V×H。V=150k、H=4096 时，约 6.1 亿参数——单这一项就省下好几 GB 显存。</p>
              <div class="ratio-board"><span>不绑定<br>2 × V×H</span><span>绑定<br>1 × V×H</span><strong>省下 V×H<br>(≈6.1亿 @150k×4096)</strong></div>
            </div>
            <div class="field-note">
              <div class="fn-title">行业视角：绑不绑，是容量与成本的权衡</div>
              <p>Qwen、GPT-2 绑定权重，主打省参数、让词向量获得双倍监督信号——在中小模型上很划算。但到了 LLaMA-70B 这种超大模型，参数预算充足，<strong>解绑</strong>反而能让输入表征和输出预测各自特化，换来更好的效果。</p>
              <p>所以这不是“哪个更对”，而是“在你的规模和预算下，省参数重要还是表达力重要”。能讲清这个权衡，面试里就比只会写一行赋值的人高一档。</p>
            </div>
          </div>`,
      syntaxHtml: code("让两个属性指向同一块权重", [
        "emb = nn.Embedding(100, 16)",
        "head = nn.Linear(16, 100, bias=False)",
        "head.weight = emb.weight              # 指针级共享，不是复制",
        "assert head.weight is emb.weight      # is 检查：同一个对象",
        "assert head.weight.data_ptr() == emb.weight.data_ptr()  # 同一块内存"
      ]),
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
