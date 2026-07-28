const {
  advancedStyles,
  checkpoint,
  code,
  lesson
} = require("../advanced_lesson_helpers");

const practice = (title, lines, mappings) => `<div class="adv-practice">
  ${code(title, lines)}
  <div class="adv-map">
    <h4>把独立例子迁移回 Notebook</h4>
    <ul>${mappings.map((item) => `<li>${item}</li>`).join("")}</ul>
  </div>
</div>`;

module.exports = {
  "12": [
    lesson({
      id: "gradient-accumulation-slice",
      title: "先拆 batch：输入与标签必须切同一段",
      todo: "TODO 1：切分当前 micro-batch",
      prerequisite: [
        "张量的第 0 维是 batch 维。Notebook 中 x.shape=[8,4]、y.shape=[8,2]，同一行的 x 和 y 属于同一条样本。",
        "accum_steps 表示把完整 batch 分成几段；micro_size=x.size(0)//accum_steps 表示每段有几条样本。",
        "Python 切片 start:stop 包含 start、不包含 stop。普通切片会保留原张量的 dtype 和 device。",
        "Notebook 已先检查 batch size 能被 accum_steps 整除，因此本题不需要处理最后一段大小不同的情况。"
      ],
      intuition: "把 8 份练习按顺序分成 4 叠，每叠 2 份。第 idx 轮只取第 idx 叠，而且题目 x 与答案 y 必须使用相同起止下标；否则模型会拿一条样本的输入去拟合另一条样本的目标。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>完整 batch</b>x 有 8 行，y 也有 8 行</span>
          <span><b>每段大小</b>micro_size = 8 // 4 = 2</span>
          <span><b>第 idx 段</b>start=idx×2，stop=(idx+1)×2</span>
          <span><b>本轮 forward</b>只处理对应的 2 行</span>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>对象</th><th>完整 shape</th><th>idx=2 的切片</th><th>切片后 shape</th></tr></thead>
          <tbody>
            <tr><td><code>x → xb</code></td><td><code>[8,4]</code></td><td><code>x[4:6]</code></td><td><code>[2,4]</code></td></tr>
            <tr><td><code>y → yb</code></td><td><code>[8,2]</code></td><td><code>y[4:6]</code></td><td><code>[2,2]</code></td></tr>
          </tbody>
        </table>
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>正确：沿样本维切</h4>
            <p><code>tensor[start:stop]</code> 省略了其余维度，等价于沿第 0 维取行。</p>
          </section>
          <section class="adv-panel warn">
            <h4>错误：沿特征维切</h4>
            <p><code>x[:, start:stop]</code> 会改变每条样本的特征数，第一层 Linear 的输入维度会对不上。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: practice(
        "把 12 位读者分成 3 个阅读小组",
        [
          "pages = torch.randn(12, 5)",
          "ratings = torch.randn(12, 1)",
          "group_count = 3",
          "group_size = pages.size(0) // group_count",
          "",
          "for group_idx in range(group_count):",
          "    start = group_idx * group_size",
          "    stop = (group_idx + 1) * group_size",
          "    page_group = pages[start:stop]",
          "    rating_group = ratings[start:stop]",
          "    print(page_group.shape, rating_group.shape)"
        ],
        [
          "<code>pages / ratings</code> 对应 <code>x / y</code>；两者必须共享同一个 <code>start:stop</code>。",
          "<code>group_count / group_idx</code> 对应 <code>accum_steps / idx</code>。",
          "<code>group_size</code> 对应 Notebook 已经算好的 <code>micro_size</code>。",
          "<code>page_group / rating_group</code> 对应当前循环的 <code>xb / yb</code>。"
        ]
      ),
      predict: {
        hook: "batch_size=8、accum_steps=4，循环正在处理 idx=3。",
        question: "这一轮 xb 应该取哪一段？",
        options: ["x[6:8]", "x[3:5]", "x[:, 6:8]"],
        answer: 0,
        revealNote: "micro_size=2，所以 start=3×2=6、stop=4×2=8；切的是第 0 维。"
      },
      checkpoint: checkpoint(
        "四轮切片 [0:2]、[2:4]、[4:6]、[6:8] 共同满足什么条件？",
        ["覆盖全部 8 行，且没有重复或遗漏", "每轮都会看见完整 batch", "把特征维从 4 降到 2"],
        0,
        "梯度累积需要每条样本在一次有效更新中恰好参与一次。"
      ),
      homework: [
        "完成 TODO 1，并在每轮临时打印 xb.shape、yb.shape；它们应分别是 [2,4] 和 [2,2]。",
        "确认 xb/yb 的 dtype 与 device 分别继承 x/y；不要调用 cpu、to 或重新构造 Tensor。",
        "错误诊断：Linear 维度不匹配时检查是否切了特征维；监督错位时检查 x、y 是否用了完全相同的 start 和 stop。",
        "最终测试会逐参数比较 full batch 与累积路径；切片遗漏或重复会让 torch.allclose 失败。"
      ]
    }),

    lesson({
      id: "gradient-accumulation-scale",
      title: "再攒梯度：先缩放 loss，再连续 backward",
      todo: "TODO 2：计算、缩放、反传并记录 loss",
      prerequisite: [
        "MSELoss(reduction='mean') 为每个 micro-batch 返回一个 0 维浮点 Tensor。",
        "PyTorch 的 parameter.grad 默认会累加；连续 backward 不会自动清空前一次梯度。",
        "等大 micro-batch 的 full-batch mean 梯度，等于各段 mean 梯度的平均，而不是总和。",
        "detach().item() 会得到只用于日志的 Python float；它没有计算图，不能调用 backward。"
      ],
      intuition: "四个 micro-batch 会给参数提交四份梯度。若每份都按完整权重相加，更新量会放大四倍；把每段 loss 先除以 accum_steps，相当于每份只占 1/4，四份累加后才与完整 batch 的平均梯度同尺度。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>漏掉缩放</h4>
            <div class="adv-flow"><span>g1</span><span>g2</span><span>g3</span><strong>Σg</strong></div>
            <p>四段 mean 梯度直接求和，更新幅度约放大 accum_steps 倍。</p>
          </section>
          <section class="adv-panel good">
            <h4>先除 accum_steps</h4>
            <div class="adv-flow"><span>g1/K</span><span>g2/K</span><span>g3/K</span><strong>Σg/K</strong></div>
            <p>缩放后的梯度累积起来，等价于完整 batch 的 mean。</p>
          </section>
        </div>
        <div class="adv-steps">
          <div><b>1</b><code>pred = model(xb)</code><span>输入 [micro_size,4]，输出 [micro_size,2]</span></div>
          <div><b>2</b><code>criterion(pred, yb)</code><span>得到连接计算图的标量 loss</span></div>
          <div><b>3</b><code>loss / accum_steps</code><span>在反传前恢复大 batch 的梯度尺度</span></div>
          <div><b>4</b><code>loss.backward()</code><span>把当前段梯度加进 parameter.grad</span></div>
          <div><b>5</b><code>loss.detach().item()</code><span>只把数值加进 total_loss</span></div>
        </div>
        <div class="adv-callout">计算图和日志是两条用途不同的路径：缩放后的 Tensor 负责 backward；detach 后的 Python 数值负责显示和返回。</div>
      </div>`,
      syntaxHtml: practice(
        "两批传感器样本共同形成一次平均更新",
        [
          "parts = [(sensor_a, target_a), (sensor_b, target_b)]",
          "reported = 0.0",
          "",
          "for part_x, part_y in parts:",
          "    estimate = net(part_x)",
          "    part_loss = criterion(estimate, part_y) / len(parts)",
          "    part_loss.backward()",
          "    reported += part_loss.detach().item()"
        ],
        [
          "<code>parts</code> 的数量对应 <code>accum_steps</code>。",
          "<code>part_x / part_y</code> 对应 <code>xb / yb</code>，<code>estimate</code> 对应 <code>pred</code>。",
          "<code>part_loss</code> 对应 Notebook 的 <code>loss</code>；除法必须发生在 backward 之前。",
          "<code>reported</code> 对应 <code>total_loss</code>，只累计 detach 后的数值。"
        ]
      ),
      predict: {
        hook: "四个等大 micro-batch 都用了 mean loss，但没有除以 accum_steps。",
        question: "累积梯度相对完整 batch 的 mean 梯度最可能怎样？",
        options: ["约放大 4 倍", "缩小到 1/4", "完全不变"],
        answer: 0,
        revealNote: "完整 batch 对应四段梯度的平均；未缩放的连续 backward 得到四段梯度的和。"
      },
      checkpoint: checkpoint(
        "下面哪个对象应该调用 backward？",
        ["除以 accum_steps 后、仍连接计算图的 loss Tensor", "loss.detach().item() 得到的 float", "total_loss 日志变量"],
        0,
        "只有仍连接计算图的 Tensor 能把梯度传回模型参数。"
      ),
      homework: [
        "完成 TODO 2：计算当前段 loss、按 accum_steps 缩放、反传，并把 detach 后的数值加入 total_loss。",
        "检查 pred 与 yb 都是 [micro_size,2] 且 device 一致；loss 是 0 维浮点 Tensor，total_loss 是 Python float。",
        "错误诊断：参数差固定约 4 倍时检查缩放；出现 float 没有 backward 时检查是否过早 item；图被长期保留时检查日志是否漏 detach。",
        "不要在这一段调用 optimizer.step；参数要等所有 micro-batch 的梯度到齐后才更新。"
      ]
    }),

    lesson({
      id: "gradient-accumulation-step",
      title: "最后只更新一次：让参数更新真正等价",
      todo: "TODO 3：统一 step、清梯度并返回 total_loss",
      prerequisite: [
        "optimizer.step() 读取参数当前的 .grad 并更新参数；它不会重新计算梯度。",
        "optimizer.zero_grad() 清除已使用的梯度，为下一个有效 batch 做准备。",
        "一次完整 batch 更新只对应一次 optimizer.step，因此累积路径也必须在循环外 step 一次。",
        "测试从同一个 base_model 深拷贝两个模型，并给两边相同的 SGD 学习率，最后逐参数 allclose。"
      ],
      intuition: "micro-batch 循环只负责收集四份建议，不能中途改模型。四段都在同一组旧参数上完成 forward/backward 后，优化器才统一移动一次；这样才能和 full-batch 路径的一次更新比较。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>周期开始</b>zero_grad 清旧梯度</span>
          <span><b>循环内部</b>4 次 forward/backward，只累加</span>
          <span><b>循环结束</b>optimizer.step 更新一次</span>
          <span><b>收尾</b>zero_grad 并返回 total_loss</span>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>Full batch</h4>
            <p>8 条样本 → 1 次 backward → 1 次 step。</p>
          </section>
          <section class="adv-panel good">
            <h4>Accumulation</h4>
            <p>4×2 条样本 → 4 次 backward → 1 次 step。</p>
          </section>
        </div>
        <div class="adv-contract">
          <span>核心断言</span><strong>每对参数都满足 torch.allclose(..., atol=1e-6)</strong>
          <span>测试含义</span><strong>切片、loss 缩放与 step 时机必须同时正确</strong>
          <span>返回值</span><strong>total_loss 用于显示，不直接替代参数一致性检查</strong>
        </div>
      </div>`,
      syntaxHtml: practice(
        "三段销量数据只触发一次参数更新",
        [
          "optimizer.zero_grad()",
          "for sales_x, sales_y in sales_parts:",
          "    prediction = model(sales_x)",
          "    scaled = criterion(prediction, sales_y) / len(sales_parts)",
          "    scaled.backward()",
          "",
          "optimizer.step()",
          "optimizer.zero_grad()"
        ],
        [
          "<code>sales_parts</code> 循环对应 Notebook 的 micro-batch 循环。",
          "循环内部只有 forward、loss 和 backward；这里不能 step。",
          "循环后的 <code>optimizer.step()</code> 对应 TODO 3 的唯一参数更新。",
          "更新后清梯度，再返回前一任务累计好的 <code>total_loss</code>。"
        ]
      ),
      predict: {
        hook: "若把 optimizer.step() 放进 micro-batch 循环，每一段后都更新参数。",
        question: "为什么会破坏与 full batch 的等价性？",
        options: ["后续 micro-batch 会在已经改变的参数上计算梯度", "step 只影响日志，不影响参数", "只要最后返回 total_loss 就仍然等价"],
        answer: 0,
        revealNote: "四段梯度必须来自同一参数状态，才能代表一个完整 batch 的单次更新。"
      },
      checkpoint: checkpoint(
        "Notebook 最终如何验收梯度累积？",
        ["比较两个模型更新后的每个参数，atol=1e-6", "只看两个 loss 都是正数", "只要求循环执行四次"],
        0,
        "逐参数一致才证明训练语义对齐；日志相近并不足以证明更新正确。"
      ),
      homework: [
        "在 micro-batch 循环外完成一次 optimizer.step，然后按 Notebook 要求 zero_grad 并返回 total_loss。",
        "运行 test_gradient_accumulation，确认 full batch 与 accumulation 的每一对参数都通过 atol=1e-6。",
        "错误诊断顺序：先查 step 是否在循环外，再查 loss 是否除 accum_steps，最后查切片是否完整且 x/y 对齐。",
        "用 accum_steps=2 再试一次，验证实现没有把 4 写死。"
      ]
    })
  ],

  "13": [
    lesson({
      id: "sft-build-batch",
      title: "把 prompt 和 response 变成可监督的 SFT 样本",
      todo: "TODO 1：构造 input_ids 与 labels",
      prerequisite: [
        "input_ids 是模型真正读取的 token 序列；labels 决定每个位置是否产生训练损失。",
        "CrossEntropyLoss 的 ignore_index=-100 会忽略 label 为 -100 的位置。",
        "Python 列表用 + 表示拼接；[value] * n 会生成 n 个相同值。",
        "函数最终返回两个 dtype=torch.long 的一维 Tensor，长度都必须恰好等于 max_len。"
      ],
      intuition: "模型需要读完整的 prompt+response 才知道上下文，但我们只想让它因回答部分的预测好坏而受罚。因此 input_ids 保留整段内容，labels 在 prompt 和 padding 位置放 -100，只在 response 位置保留真实 token。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>拼输入</b>prompt_ids + response_ids</span>
          <span><b>做监督</b>prompt → -100，response → 原 token</span>
          <span><b>统一长度</b>超长从开头截断，不足则 padding</span>
          <span><b>转 Tensor</b>两个结果都是 long、shape [max_len]</span>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>位置</th><th>input_ids</th><th>labels</th><th>是否计算 loss</th></tr></thead>
          <tbody>
            <tr><td>prompt</td><td>原 prompt token</td><td><code>-100</code></td><td>否</td></tr>
            <tr><td>response</td><td>原 response token</td><td>原 response token</td><td>是</td></tr>
            <tr><td>padding</td><td><code>pad_id</code></td><td><code>-100</code></td><td>否</td></tr>
          </tbody>
        </table>
        <div class="adv-callout">Notebook 明确采用“从开头直接截断”的简化策略。不要自行实现保留 response 的复杂截断，否则可能偏离本题测试契约。</div>
      </div>`,
      syntaxHtml: practice(
        "为问答卡片构造输入与评分掩码",
        [
          "question = [8, 9]",
          "answer = [10, 11, 12]",
          "limit = 7",
          "",
          "tokens = question + answer",
          "targets = [-100] * len(question) + answer",
          "padding = limit - len(tokens)",
          "tokens = tokens + [0] * padding",
          "targets = targets + [-100] * padding"
        ],
        [
          "<code>question / answer</code> 对应 <code>prompt_ids / response_ids</code>。",
          "<code>tokens / targets</code> 对应 <code>input_ids / labels</code>。",
          "<code>limit</code> 对应 <code>max_len</code>，Notebook 还要求先判断截断或 padding。",
          "最后把两个列表分别转成 <code>torch.long</code> Tensor。"
        ]
      ),
      predict: {
        hook: "prompt=[1,2,3]，response=[4,5]，max_len=7，pad_id=0。",
        question: "正确的 labels 是哪一个？",
        options: ["[-100,-100,-100,4,5,-100,-100]", "[1,2,3,4,5,0,0]", "[-100,-100,-100,-100,-100,0,0]"],
        answer: 0,
        revealNote: "prompt 与 padding 都忽略，只有 response token 4、5 保留为监督目标。"
      },
      checkpoint: checkpoint(
        "为什么 prompt token 仍放进 input_ids，却在 labels 中写成 -100？",
        ["模型需要它作为上下文，但本实验不让 prompt 位置贡献 loss", "Embedding 不能读取 response", "-100 会让 token 自动移到 GPU"],
        0,
        "输入上下文与监督范围是两件事；mask labels 不会删除模型看到的 token。"
      ),
      homework: [
        "完成 input_ids=prompt+response，以及 labels=prompt 对应 -100、response 保留原 token。",
        "按 Notebook 顺序处理长度：超长时两个列表都取 [:max_len]；不足时 input 补 pad_id、labels 补 -100。",
        "返回两个 torch.long Tensor，并确认测试样本得到 shape [8]；随后 repeat 后 batch shape 是 [4,8]。",
        "错误诊断：长度不同时检查两个列表是否同步截断/补齐；prompt 也产生 loss 时检查 labels mask；Embedding 报 dtype 错时检查是否用了 long。"
      ]
    }),

    lesson({
      id: "sft-next-token-loss",
      title: "错开一位：让位置 t 预测位置 t+1",
      todo: "TODO 2：shift logits、shift labels 并计算交叉熵",
      prerequisite: [
        "TinyCausalLM 输出 logits.shape=[B,T,V]，最后一维 V 是词表类别数；labels.shape=[B,T]。",
        "自回归训练中，位置 t 的 logits 用来预测下一个 token，即 labels 的位置 t+1。",
        "切片后的 Tensor 可能不连续；contiguous 后再 view 展平更稳妥。",
        "CrossEntropyLoss 接收 [N,C] 浮点 logits 与 [N] long targets，并通过 ignore_index=-100 跳过 mask。"
      ],
      intuition: "把 logits 行看成每个位置写下的“下一词猜测”。第 0 个位置的猜测应与第 1 个 token 对答案，第 1 个位置与第 2 个 token 对答案，所以 logits 去掉最后一步，labels 去掉第一步，然后再按 token 做分类。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>logits[:,0]</span><strong>预测 labels[:,1]</strong>
          <span>logits[:,1]</span><strong>预测 labels[:,2]</strong>
          <span>...</span><strong>直到 T-2 → T-1</strong>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>切片前</th><th>切片后</th><th>送入 CE 前</th></tr></thead>
          <tbody>
            <tr><td><code>logits</code></td><td><code>[B,T,V]</code></td><td><code>[B,T-1,V]</code></td><td><code>[B(T-1),V]</code></td></tr>
            <tr><td><code>labels</code></td><td><code>[B,T]</code></td><td><code>[B,T-1]</code></td><td><code>[B(T-1)]</code></td></tr>
          </tbody>
        </table>
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>监督有效位置</h4>
            <p>response label 保留词表下标，CrossEntropy 会计算分类误差。</p>
          </section>
          <section class="adv-panel warn">
            <h4>忽略位置</h4>
            <p>prompt 与 padding 的 label 是 -100；损失函数必须显式设置 ignore_index=-100。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: practice(
        "把天气序列的当前时刻与下一时刻对齐",
        [
          "scores = torch.randn(2, 5, 4)",
          "next_state = torch.tensor([[0, 1, 2, 3, 1], [2, 2, 0, 1, 3]])",
          "",
          "usable_scores = scores[..., :-1, :].contiguous()",
          "future_targets = next_state[..., 1:].contiguous()",
          "loss_fn = nn.CrossEntropyLoss()",
          "loss = loss_fn(",
          "    usable_scores.view(-1, usable_scores.size(-1)),",
          "    future_targets.view(-1),",
          ")"
        ],
        [
          "<code>scores / next_state</code> 对应 <code>logits / labels</code>。",
          "<code>usable_scores</code> 对应 <code>shift_logits</code>；去掉最后一个预测位置。",
          "<code>future_targets</code> 对应 <code>shift_labels</code>；去掉第一个目标位置。",
          "Notebook 的 loss_fn 还必须加入 <code>ignore_index=-100</code>。"
        ]
      ),
      predict: {
        hook: "logits.shape=[4,8,64]，labels.shape=[4,8]。",
        question: "完成 shift 并展平后，交叉熵的两个输入 shape 是什么？",
        options: ["[28,64] 与 [28]", "[32,64] 与 [32]", "[4,7] 与 [4,7]"],
        answer: 0,
        revealNote: "每条序列保留 7 个预测位置，4×7=28；类别维 64 不变。"
      },
      checkpoint: checkpoint(
        "为何不能让 logits[...,1:,:] 对 labels[...,:-1]？",
        ["那会让后一个位置预测前一个 token，方向与 causal LM 目标相反", "因为 logits 不能切片", "因为 labels 必须是 float"],
        0,
        "当前时刻预测下一时刻：保留较早的 logits，配对较晚的 labels。"
      ),
      homework: [
        "完成两个 shift，并在 view 前调用 contiguous；检查 shift_logits=[B,T-1,V]、shift_labels=[B,T-1]。",
        "用 nn.CrossEntropyLoss(ignore_index=-100) 对展平后的 Tensor 计算标量 loss 并返回。",
        "错误诊断：target size 报错时检查时间维切片与展平；prompt/pad 也计入 loss 时检查 ignore_index；view 报 stride 时检查 contiguous。",
        "测试会在训练前后重复调用该函数，因此实现不能修改传入的 logits 或 labels。"
      ]
    }),

    lesson({
      id: "sft-training-loop",
      title: "合成四层闭环：数据、模型、loss 与优化器",
      todo: "TODO 3：micro-batch 训练、梯度累积与 history",
      prerequisite: [
        "input_ids 与 labels 的 batch size 必须能被 accum_steps 整除；Notebook 已为非法情况抛 ValueError。",
        "每个 update 开始先 model.train 和 optimizer.zero_grad；每个 micro-batch 都 forward、loss、backward。",
        "每段 loss 必须除以 accum_steps，循环结束后才 optimizer.step。",
        "history 每个 update 追加一个 Python 数值；测试要求长度等于 num_updates，并要求最终 loss 小于初始 loss。"
      ],
      intuition: "这一题不是再造一个新算法，而是把前两项和第 12 课的梯度累积接好：同一批重复样本经过多次更新后能被小模型记住，loss 就应下降。这个“小样本过拟合”是验证训练闭环是否真的通电的工程方法。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>数据层</b>[4,8] 的 input_ids 与 labels</span>
          <span><b>模型层</b>Embedding → GRU → LM head，输出 [B,T,64]</span>
          <span><b>优化层</b>shifted SFT loss → backward → step</span>
          <span><b>控制层</b>2 个 micro-batch、30 次 update、记录 history</span>
        </div>
        <div class="adv-steps">
          <div><b>1</b><code>每个 update 清梯度</code><span>旧更新不能污染新更新</span></div>
          <div><b>2</b><code>切 mb_input / mb_labels</code><span>两者沿 batch 维使用同一范围</span></div>
          <div><b>3</b><code>model → compute_sft_loss</code><span>把当前 micro-batch 接进前两项函数</span></div>
          <div><b>4</b><code>loss / accum_steps → backward</code><span>累积等价平均梯度</span></div>
          <div><b>5</b><code>step → history.append</code><span>每轮只更新一次并记录一个数值</span></div>
        </div>
        <div class="adv-contract">
          <span>测试 batch</span><strong>同一条 [8] 样本 repeat 4 次，便于快速过拟合</strong>
          <span>训练配置</span><strong>accum_steps=2，num_updates=30，AdamW lr=0.05</strong>
          <span>行为断言</span><strong>len(history)==30 且 final_loss&lt;init_loss</strong>
        </div>
      </div>`,
      syntaxHtml: practice(
        "让一个小分类器反复学习同一批图形",
        [
          "history = []",
          "for update in range(update_count):",
          "    optimizer.zero_grad()",
          "    reported = 0.0",
          "    for part_x, part_y in mini_parts:",
          "        logits = classifier(part_x)",
          "        loss = loss_fn(logits, part_y) / len(mini_parts)",
          "        loss.backward()",
          "        reported += loss.detach().item()",
          "    optimizer.step()",
          "    history.append(reported)"
        ],
        [
          "<code>mini_parts</code> 对应通过 idx 与 micro_size 切出的 <code>mb_input / mb_labels</code>。",
          "<code>classifier</code> 对应 <code>model</code>，随后调用 Notebook 的 <code>compute_sft_loss</code>。",
          "<code>len(mini_parts)</code> 对应 <code>accum_steps</code>。",
          "<code>history</code> 必须每个 update 追加一次，而不是每个 micro-batch 追加一次。"
        ]
      ),
      predict: {
        hook: "input_ids.shape=[4,8]、accum_steps=2。",
        question: "每个 micro-batch 的 mb_input 与模型 logits shape 分别是什么？",
        options: ["[2,8] 与 [2,8,64]", "[4,4] 与 [4,4,64]", "[2,64] 与 [2,8]"],
        answer: 0,
        revealNote: "只切 batch 维；序列长度 8 不变，模型再增加词表维 64。"
      },
      checkpoint: checkpoint(
        "为什么测试使用重复样本，并只检查最终 loss 小于初始 loss？",
        ["重复样本容易快速过拟合，可用来验证整条训练链路是否接通", "为了测试模型泛化能力", "因为重复会自动修复 label mask"],
        0,
        "这里验收的是最小闭环，不是下游效果；连重复样本都学不会通常说明接口或更新路径有断点。"
      ),
      homework: [
        "完成 micro-batch 切片、forward、缩放 loss、backward、total_loss 累加，并在内层循环后 step。",
        "每个 update 只向 history 追加一次 total_loss；测试传入 num_updates=30，所以返回长度必须是 30。",
        "运行 test_end_to_end_finetuning，确认 final_loss<init_loss；不要依赖固定 loss 数值，因为初始化和优化过程才决定它。",
        "错误诊断：history 长度过长时检查 append 层级；loss 不降时依次检查 labels mask、shift 方向、loss 缩放、backward 和 step。"
      ]
    })
  ],

  "14": [
    lesson({
      id: "ppo-log-ratio",
      title: "从 log probability 恢复新旧策略比率",
      todo: "TODO 1：计算重要性采样比率 ratio",
      prerequisite: [
        "log_probs_new、log_probs_old 与 advantages 的 shape 都是 [batch_size,seq_len]，每个元素对应同一个采样 token。",
        "对数规则 log(a/b)=log(a)-log(b)，因此概率比可由对数概率相减后取 exp 得到。",
        "ratio=1 表示新旧策略对该 token 的概率相同；大于 1 表示新策略提高了它的概率。",
        "所有运算都应保持 Tensor、dtype 与 device，不需要转成 Python 数字。"
      ],
      intuition: "旧策略负责采样，新策略正在被更新。ratio 是一把刻度尺：它不是评价 token 好坏，而是量新策略相对旧策略把这个 token 的概率改了多少；好坏方向由 advantage 决定。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>同一 token</b>读取 new 与 old 的 log probability</span>
          <span><b>先相减</b>log_new - log_old</span>
          <span><b>再取指数</b>恢复 probability ratio</span>
          <span><b>解释大小</b>&gt;1 提高，&lt;1 降低，=1 不变</span>
        </div>
        <div class="adv-contract">
          <span><code>log_probs_new</code></span><strong>[B,S]，当前 Actor 的 token 对数概率</strong>
          <span><code>log_probs_old</code></span><strong>[B,S]，采样时旧 Actor 的 token 对数概率</strong>
          <span><code>ratio</code></span><strong>[B,S] 正数 Tensor，继续保留 new 策略的梯度路径</strong>
        </div>
        <div class="adv-callout">不要直接写 log_probs_new/log_probs_old；那是两个负对数值的比，不是两个概率的比。</div>
      </div>`,
      syntaxHtml: practice(
        "比较新旧广告策略的选择概率",
        [
          "log_new = torch.log(torch.tensor([0.30, 0.45]))",
          "log_old = torch.log(torch.tensor([0.25, 0.50]))",
          "change = torch.exp(log_new - log_old)",
          "print(change)  # [1.2, 0.9]"
        ],
        [
          "<code>log_new / log_old</code> 对应 <code>log_probs_new / log_probs_old</code>。",
          "<code>change</code> 对应 <code>ratio</code>，写法天然支持 Notebook 的二维 [B,S]。",
          "第一个 ratio=1.2 表示概率提高 20%；它是否值得鼓励还要结合 advantages。"
        ]
      ),
      predict: {
        hook: "某 token 的 log_probs_new 与 log_probs_old 完全相同。",
        question: "它的 ratio 是多少？",
        options: ["1", "0", "取决于 advantage"],
        answer: 0,
        revealNote: "两者相减为 0，exp(0)=1，表示策略概率没有变化。"
      },
      checkpoint: checkpoint(
        "为什么使用 exp(log_new-log_old)？",
        ["它等价于 new_prob/old_prob，且从已有 log probability 直接计算", "它会自动把 advantage 归一化", "它会把 shape 变成标量"],
        0,
        "这一步只恢复策略概率比，不做 advantage 处理或聚合。"
      ),
      homework: [
        "完成 ratio，并确认 shape、dtype、device 与 log_probs_new 一致。",
        "用 Notebook 的第一项手算：exp(-2.0-(-2.1))=exp(0.1)，应略大于 1。",
        "错误诊断：出现负 ratio 说明公式错误；ratio 全为 1 时检查是否错误地用同一个 Tensor 相减。",
        "保留 log_probs_new 的计算图；不要 detach，否则最终 loss 无法训练 Actor。"
      ]
    }),

    lesson({
      id: "ppo-clipped-surrogates",
      title: "让 advantage 决定方向，让 clip 限制步幅",
      todo: "TODO 2 + TODO 3：计算 surr1 与 surr2",
      prerequisite: [
        "advantages 与 ratio 同 shape；正 advantage 表示该 token 比预期好，负值表示比预期差。",
        "surr1=ratio*advantages 是未限制的新策略目标。",
        "torch.clamp(ratio,1-clip_range,1+clip_range) 只限制 ratio，不改变 advantage。",
        "默认 clip_range=0.2，所以 ratio 被限制在 [0.8,1.2]。"
      ],
      intuition: "advantage 像方向标：正值希望提高概率，负值希望降低概率。ratio 像已经走出的距离；clip 给距离加护栏，避免一次更新把策略推得太远。",
      exampleHtml: `<div class="adv-course">
        <table class="adv-shapes">
          <thead><tr><th>advantage</th><th>ratio 变化</th><th>训练意图</th><th>clip 作用</th></tr></thead>
          <tbody>
            <tr><td>正</td><td>增大</td><td>更常选择好 token</td><td>超过 1+ε 后不再奖励过大增长</td></tr>
            <tr><td>负</td><td>减小</td><td>更少选择差 token</td><td>低于 1-ε 后限制过度下降</td></tr>
          </tbody>
        </table>
        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>未截断目标</h4>
            <p><code>surr1 = ratio * advantages</code></p>
            <p>反映新策略真实变化带来的目标值。</p>
          </section>
          <section class="adv-panel good">
            <h4>截断目标</h4>
            <p><code>surr2 = clipped_ratio * advantages</code></p>
            <p>把 ratio 限制在允许的单步变化区间。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: practice(
        "给推荐系统的更新幅度加护栏",
        [
          "change = torch.tensor([1.35, 0.72, 1.05])",
          "quality = torch.tensor([1.0, -0.5, 0.2])",
          "limit = 0.2",
          "",
          "raw_score = change * quality",
          "safe_change = torch.clamp(change, 1.0 - limit, 1.0 + limit)",
          "safe_score = safe_change * quality"
        ],
        [
          "<code>change / quality / limit</code> 对应 <code>ratio / advantages / clip_range</code>。",
          "<code>raw_score</code> 对应 <code>surr1</code>。",
          "<code>safe_score</code> 对应 <code>surr2</code>。",
          "clamp 只作用于 change/ratio，不能先把 ratio×advantage 的结果截断。"
        ]
      ),
      predict: {
        hook: "clip_range=0.2，ratio=1.5，advantage=2。",
        question: "surr1 与 surr2 分别是多少？",
        options: ["3.0 与 2.4", "2.4 与 3.0", "1.5 与 1.2"],
        answer: 0,
        revealNote: "surr1=1.5×2=3；ratio 截到 1.2 后，surr2=1.2×2=2.4。"
      },
      checkpoint: checkpoint(
        "PPO 应该截断哪个对象？",
        ["ratio，再分别与原 advantages 相乘", "advantages", "最终 mean loss"],
        0,
        "clip 限制的是策略概率变化幅度，不是奖励信号或最终标量。"
      ),
      homework: [
        "完成 surr1 与 surr2，确认二者 shape 都是 [B,S]。",
        "分别造一个正 advantage 且 ratio>1.2、一个负 advantage 且 ratio<0.8 的元素，手算两条 surrogate。",
        "错误诊断：数值无变化时检查 clamp 上下界；方向反转时检查是否误对 advantage clamp。",
        "不要在此处 mean；逐 token 的 min 选择要留给下一任务。"
      ]
    }),

    lesson({
      id: "ppo-loss-and-contract",
      title: "取更保守的目标，再转成优化器要最小化的 loss",
      todo: "TODO 4：-min(surr1,surr2).mean()",
      prerequisite: [
        "torch.min(surr1,surr2) 做逐元素选择，保留 [B,S] shape。",
        "mean 把所有 batch/token 位置聚合成 0 维标量，满足 optimizer 对标量 loss 的常见契约。",
        "PPO 原目标希望最大化 surrogate，而 PyTorch 优化器通过最小化 loss 工作，所以前面要加负号。",
        "测试会用同一公式构造 expected_loss 并 torch.allclose 比较。"
      ],
      intuition: "对每个 token，PPO 都在原目标与受护栏限制的目标中选更保守的那个；把这些值平均后本应最大化，再乘负号交给最小化型优化器。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>逐 token</b>比较 surr1 与 surr2</span>
          <span><b>保守选择</b>取 element-wise minimum</span>
          <span><b>批量聚合</b>对 [B,S] 求 mean</span>
          <span><b>目标转 loss</b>最前面加负号</span>
        </div>
        <div class="adv-contract">
          <span>输入</span><strong>三组 [B,S] 浮点 Tensor；clip_range 为 float</strong>
          <span>输出</span><strong>0 维 actor_loss，保留到 log_probs_new 的梯度路径</strong>
          <span>测试</span><strong>与 -torch.min(s1,s2).mean() 完全同公式比较</strong>
        </div>
        <div class="adv-callout">Notebook 只实现 Actor clip loss。Reference、Reward Model、Critic 的显存流转是理解背景，不要在 compute_actor_loss 中虚构四模型 API。</div>
      </div>`,
      syntaxHtml: practice(
        "把保守收益变成可最小化的训练目标",
        [
          "raw = torch.tensor([1.4, -0.3, 0.8])",
          "bounded = torch.tensor([1.2, -0.5, 0.8])",
          "conservative = torch.min(raw, bounded)",
          "loss = -conservative.mean()"
        ],
        [
          "<code>raw / bounded</code> 对应 <code>surr1 / surr2</code>。",
          "<code>conservative</code> 是逐元素 min，不是先各自求 mean 再比较。",
          "最后的负号对应“最大化目标 → 最小化 loss”。"
        ]
      ),
      predict: {
        hook: "已得到 surr1 与 surr2，二者 shape 都是 [2,2]。",
        question: "最终 actor loss 的 shape 应是什么？",
        options: ["[]，0 维标量", "[2,2]", "[2]"],
        answer: 0,
        revealNote: "逐元素 min 后对所有位置 mean，最终得到一个标量。"
      },
      checkpoint: checkpoint(
        "为什么最终公式前有负号？",
        ["PPO 要最大化 surrogate，而训练代码通常最小化 loss", "为了把 ratio 变成正数", "为了消除 clip_range"],
        0,
        "负号只负责优化方向转换，不改变 ratio 的定义或截断区间。"
      ),
      homework: [
        "完成最终 loss，并运行 test_ppo_actor_loss，让结果与测试现场计算的 expected_loss allclose。",
        "确认输出 loss.ndim==0，且若 log_probs_new.requires_grad=True，loss 仍可 backward。",
        "错误诊断：差一个负号时检查最大化/最小化转换；只差聚合尺度时检查是否漏 mean；数值奇怪时回查 ratio 是否由 log 差取 exp。",
        "用一句话串起四步：log ratio → 两条 surrogate → 逐元素 min → 负均值。"
      ]
    })
  ],

  "15": [
    lesson({
      id: "dpo-implicit-rewards",
      title: "先与参考模型比较：得到 chosen 与 rejected 的隐式奖励",
      todo: "TODO 1 + TODO 2：两组 policy-reference log-ratio",
      prerequisite: [
        "四个输入都是 shape [batch_size] 的序列级 log probability 总和，同一索引属于同一个 prompt 的 chosen/rejected 对。",
        "policy-reference 的对数差表示当前策略相对参考策略提高或降低了某个回答的概率。",
        "beta 是标量温度系数；Notebook 先算 log-ratio，再分别乘 beta 得到两个 reward Tensor。",
        "输出奖励仍是 [batch_size]，不应在这里 mean。"
      ],
      intuition: "DPO 不先训练单独的奖励模型，而是把参考模型当基线：当前策略比参考模型更愿意生成某回答，隐式奖励就偏正；更不愿意，奖励就偏负。chosen 与 rejected 必须分别比较自己的参考概率。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>Chosen 支路</h4>
            <p><code>policy_chosen - reference_chosen</code></p>
            <p>再乘 beta 得到 chosen_rewards。</p>
          </section>
          <section class="adv-panel warn">
            <h4>Rejected 支路</h4>
            <p><code>policy_rejected - reference_rejected</code></p>
            <p>再乘 beta 得到 rejected_rewards。</p>
          </section>
        </div>
        <div class="adv-flow">
          <span>四个 [B] logps</span><span>两条 policy-reference 差</span><span>分别乘 beta</span><strong>两个 [B] rewards</strong>
        </div>
        <div class="adv-callout">不要先做 chosen-rejected。第一层比较是“同一个回答的 policy 与 reference”；第二层才比较 chosen reward 与 rejected reward。</div>
      </div>`,
      syntaxHtml: practice(
        "比较新旧评分器对两版摘要的偏好变化",
        [
          "new_good = torch.tensor([-0.8, -1.1])",
          "old_good = torch.tensor([-1.3, -1.2])",
          "new_bad = torch.tensor([-2.0, -1.8])",
          "old_bad = torch.tensor([-1.5, -1.6])",
          "temperature = 0.2",
          "",
          "good_reward = temperature * (new_good - old_good)",
          "bad_reward = temperature * (new_bad - old_bad)"
        ],
        [
          "<code>new_good / old_good</code> 对应 policy/reference chosen logps。",
          "<code>new_bad / old_bad</code> 对应 policy/reference rejected logps。",
          "<code>temperature</code> 对应 <code>beta</code>。",
          "两个 reward 都保留 batch 维，下一任务才做成对差值。"
        ]
      ),
      predict: {
        hook: "policy_chosen=-1.0、reference_chosen=-2.0、beta=0.1。",
        question: "chosen_rewards 的该元素是多少？",
        options: ["0.1", "-0.1", "1.0"],
        answer: 0,
        revealNote: "policy-reference=(-1)-(-2)=1，再乘 0.1 得 0.1。"
      },
      checkpoint: checkpoint(
        "DPO 的第一层比较应该怎样配对？",
        ["policy chosen 对 reference chosen；policy rejected 对 reference rejected", "policy chosen 对 policy rejected", "reference chosen 对 policy rejected"],
        0,
        "先对同一回答做策略相对参考的变化，才能得到可比较的隐式奖励。"
      ),
      homework: [
        "完成两个 pi_logratios，并确认 chosen/rejected 两条支路没有交叉配错。",
        "保留 Notebook 已写好的 beta 乘法；chosen_rewards、rejected_rewards 的 shape 都应为 [B]。",
        "用测试第 0 项手算：chosen reward=0.1、rejected reward=-0.1。",
        "错误诊断：奖励符号相反时检查减法顺序；shape 变成标量时检查是否误加 mean/sum。"
      ]
    }),

    lesson({
      id: "dpo-pairwise-loss",
      title: "再比较好坏回答：逐样本得到稳定的 DPO loss",
      todo: "TODO 3：chosen-rejected logits 与 -F.logsigmoid",
      prerequisite: [
        "chosen_rewards 与 rejected_rewards 都是 [B]，逐元素相减会得到每对回答的偏好 margin。",
        "margin 越大，表示当前策略相对参考模型更偏向 chosen，DPO loss 应越小。",
        "F.logsigmoid 比 torch.log(torch.sigmoid(...)) 数值更稳定。",
        "函数契约返回 losses、chosen_rewards、rejected_rewards 三个 [B] Tensor；Notebook 不要求在函数内 mean。"
      ],
      intuition: "每条偏好数据像一道二选一题。DPO 关心 chosen reward 比 rejected reward 高多少；把差值送进 sigmoid，相当于把它解释成“chosen 胜出”的可信度，再取负对数作为惩罚。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>成对 margin</b>chosen_rewards - rejected_rewards</span>
          <span><b>映射可信度</b>sigmoid(margin)</span>
          <span><b>稳定负对数</b>-F.logsigmoid(margin)</span>
          <span><b>保留 batch</b>返回每一对的 loss</span>
        </div>
        <div class="adv-contract">
          <span>测试 shape</span><strong>losses、c_rewards、r_rewards 都是 (4,)</strong>
          <span>测试数值</span><strong>第 0 项 margin=0.1-(-0.1)=0.2</strong>
          <span>期望 loss</span><strong>-F.logsigmoid(torch.tensor(0.2))</strong>
        </div>
        <div class="adv-callout">本函数返回逐样本 losses，不要自行 mean。上层训练循环可以决定如何聚合；这里的测试明确保留 batch 维。</div>
      </div>`,
      syntaxHtml: practice(
        "把两版方案的相对评分变成逐对损失",
        [
          "preferred = torch.tensor([0.4, -0.1, 0.8])",
          "other = torch.tensor([-0.2, 0.3, 0.1])",
          "margin = preferred - other",
          "pair_losses = -F.logsigmoid(margin)"
        ],
        [
          "<code>preferred / other</code> 对应 <code>chosen_rewards / rejected_rewards</code>。",
          "<code>margin</code> 对应 Notebook 的 <code>logits</code>，它仍是 [B]。",
          "<code>pair_losses</code> 对应 <code>losses</code>；不要对它 mean。",
          "最终按契约返回 losses 以及前一任务得到的两组 rewards。"
        ]
      ),
      predict: {
        hook: "某一对回答的 chosen reward 提高，而 rejected reward 不变。",
        question: "该样本的 DPO loss 通常怎样变化？",
        options: ["减小", "增大", "shape 变成标量"],
        answer: 0,
        revealNote: "margin 变大，logsigmoid 更接近 0，负号后的损失下降。"
      },
      checkpoint: checkpoint(
        "为什么 Notebook 不在 dpo_loss 内对 losses 求 mean？",
        ["函数契约要求返回每个样本的 [B] loss，测试也检查该 shape", "因为 mean 不支持梯度", "因为 beta 必须是 Tensor"],
        0,
        "聚合策略由调用方决定；本题明确验收逐样本损失。"
      ),
      homework: [
        "完成 logits=chosen_rewards-rejected_rewards，以及 losses=-F.logsigmoid(logits)。",
        "按顺序返回 losses、chosen_rewards、rejected_rewards，并保持三个 shape 都为 [batch_size]。",
        "运行 test_dpo_loss，核对第 0 项两个 reward 与 expected_loss_0。",
        "错误诊断：loss 数值相反时检查 margin 顺序；shape 为 [] 时删除误加的 mean；出现数值不稳时确认使用 F.logsigmoid。"
      ]
    })
  ],

  "16": [
    lesson({
      id: "grpo-group-advantages",
      title: "先在每个 prompt 组内建立相对坐标",
      todo: "TODO 1：按 group_ids 计算标准化 advantages",
      prerequisite: [
        "rewards 与 group_ids 都是 shape [N]；相同 group_id 表示候选回答来自同一个 prompt。",
        "group_ids.unique(sorted=True) 给出实际出现的组编号；布尔 mask 可选出一组 reward。",
        "std(unbiased=False) 使用总体标准差，和 Notebook 给出的公式及参考实现一致。",
        "clamp_min(eps) 防止同组奖励完全相同、标准差为 0 时除零。"
      ],
      intuition: "不同 prompt 的奖励刻度可能不同，所以不能直接全局比较。GRPO 先在每个 prompt 的候选组里算均值和标准差：高于组均值就是正 advantage，低于组均值就是负 advantage。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>找组</b>遍历 unique group id</span>
          <span><b>选成员</b>mask = group_ids == gid</span>
          <span><b>去均值</b>reward - group mean</span>
          <span><b>除尺度</b>除总体 std，并写回原位置</span>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>Notebook 组</th><th>rewards</th><th>组均值</th><th>相对优势性质</th></tr></thead>
          <tbody>
            <tr><td><code>gid=0</code></td><td><code>[1.0,2.0]</code></td><td><code>1.5</code></td><td>一负一正，均值接近 0</td></tr>
            <tr><td><code>gid=1</code></td><td><code>[0.5,1.5]</code></td><td><code>1.0</code></td><td>一负一正，均值接近 0</td></tr>
          </tbody>
        </table>
        <div class="adv-callout">不要用 rewards.mean() 做全局归一化；那会让不同 prompt 的奖励尺度相互污染，失去“group relative”的含义。</div>
      </div>`,
      syntaxHtml: practice(
        "按班级标准化学生的组内表现",
        [
          "scores = torch.tensor([70.0, 90.0, 60.0, 80.0])",
          "class_ids = torch.tensor([3, 3, 8, 8])",
          "relative = torch.zeros_like(scores)",
          "",
          "for class_id in class_ids.unique(sorted=True):",
          "    mask = class_ids == class_id",
          "    group = scores[mask]",
          "    centered = group - group.mean()",
          "    scale = group.std(unbiased=False).clamp_min(1e-6)",
          "    relative[mask] = centered / scale"
        ],
        [
          "<code>scores / class_ids</code> 对应 <code>rewards / group_ids</code>。",
          "<code>relative</code> 对应先用 <code>zeros_like(rewards)</code> 创建的 advantages。",
          "<code>unbiased=False</code> 与 <code>clamp_min(eps)</code> 都是 Notebook 契约的一部分。",
          "布尔 mask 写回可保持 advantages 与原 rewards 的样本顺序一致。"
        ]
      ),
      predict: {
        hook: "同一组 rewards=[4,4]，eps=1e-6。",
        question: "标准化后的 advantages 应是什么？",
        options: ["[0,0]", "[NaN,NaN]", "[4,4]"],
        answer: 0,
        revealNote: "centered 全为 0；分母被 clamp 到 eps 后仍得到 0，避免 NaN。"
      },
      checkpoint: checkpoint(
        "测试为什么检查 adv[group_ids==0].mean() 接近 0？",
        ["组内去均值后再除同组尺度，组内平均应保持为 0", "因为所有 reward 必须为 0", "因为 ratio 会被清零"],
        0,
        "这是判断是否真的按组归一化的直接信号。"
      ),
      homework: [
        "初始化 zeros_like(rewards)，逐 unique gid 取 mask、centered 和总体 std，并把结果写回 advantages[mask]。",
        "确认 advantages.shape=[N]、dtype/device 与 rewards 一致；group_ids 只用于索引，不参与浮点计算。",
        "运行组 0 的均值断言；再用一组相同 rewards 检查 clamp_min(eps) 是否避免 NaN。",
        "错误诊断：组均值不为 0 时检查是否误用全局 mean；两元素组幅度异常时检查 unbiased 是否写成 True。"
      ]
    }),

    lesson({
      id: "grpo-clipped-objective",
      title: "把组内优势接入 PPO 式策略护栏",
      todo: "TODO 2：计算 ratio、surr1 与 surr2",
      prerequisite: [
        "log_probs_new、log_probs_old、advantages 都是 shape [N]，同一位置对应同一个候选回答。",
        "ratio=exp(log_new-log_old) 衡量策略概率变化，必须保持到 log_probs_new 的梯度路径。",
        "surr1 使用原 ratio，surr2 使用 clamp 到 [1-clip_range,1+clip_range] 的 ratio。",
        "advantages 在上一任务已完成组内标准化，本任务不能再次全局归一化。"
      ],
      intuition: "组内 advantage 说明向哪里走，ratio 说明已经走了多远。GRPO 借用 PPO 的两条 surrogate：一条观察真实变化，一条把变化限制在护栏内。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>log_new-log_old</span><strong>exp → ratio [N]</strong>
          <span>ratio×advantage</span><strong>surr1 [N]</strong>
          <span>clamp(ratio)×advantage</span><strong>surr2 [N]</strong>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>正 advantage</h4>
            <p>鼓励提高组内较好候选的概率，但 ratio 上升过多时受 clip 限制。</p>
          </section>
          <section class="adv-panel warn">
            <h4>负 advantage</h4>
            <p>鼓励降低组内较差候选的概率，但 ratio 下降过多时同样受限制。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: practice(
        "给两组方案的相对得分加更新护栏",
        [
          "log_current = torch.tensor([-0.7, -1.2, -0.4])",
          "log_previous = torch.tensor([-0.8, -1.0, -0.5])",
          "relative_quality = torch.tensor([1.0, -1.0, 0.5])",
          "epsilon = 0.2",
          "",
          "ratio = torch.exp(log_current - log_previous)",
          "raw = ratio * relative_quality",
          "bounded = torch.clamp(ratio, 1-epsilon, 1+epsilon) * relative_quality"
        ],
        [
          "<code>log_current / log_previous</code> 对应 new/old log probabilities。",
          "<code>relative_quality</code> 对应上一任务算出的 <code>advantages</code>。",
          "<code>raw / bounded</code> 对应 <code>surr1 / surr2</code>。",
          "所有结果保持 [N]，下一任务才聚合成标量。"
        ]
      ),
      predict: {
        hook: "log_new-log_old=0，advantage=-1。",
        question: "ratio 与 surr1 分别是多少？",
        options: ["1 与 -1", "0 与 0", "-1 与 1"],
        answer: 0,
        revealNote: "exp(0)=1；surr1=1×(-1)=-1。"
      },
      checkpoint: checkpoint(
        "GRPO 与本 Notebook 的 PPO clip 思路在哪一步相同？",
        ["都用原 ratio 和截断 ratio 分别乘 advantage", "都必须训练显式 Critic", "都把 rewards 直接当 loss"],
        0,
        "差异在 advantage 来源；clip surrogate 的构造方式相同。"
      ),
      homework: [
        "完成 ratio、surr1、surr2，确认三个 Tensor 都是 [N] 且 finite。",
        "不要 detach log_probs_new；测试随后会调用 loss.backward 检查梯度。",
        "用 ratio 超过 1.2 与低于 0.8 的例子手算 clamp 是否正确。",
        "错误诊断：梯度为空时检查 detach/item；ratio 非正时检查是否漏 exp；shape 变标量时检查是否提前 mean。"
      ]
    }),

    lesson({
      id: "grpo-loss-test",
      title: "保守聚合并读懂 GRPO 的四个测试信号",
      todo: "TODO 3：返回标量 loss 与逐样本 advantages",
      prerequisite: [
        "torch.min(surr1,surr2) 是逐元素保守选择，随后 mean 得到标量。",
        "前置负号把最大化 surrogate 转为最小化 loss。",
        "函数必须同时返回 loss 与 advantages，分别是 shape [] 和 [N]。",
        "测试不仅看数值 finite，还会 backward 并确认 log_new.grad 不为 None。"
      ],
      intuition: "一个正确的 GRPO 函数既要有合理统计性质，也要能训练：组内 advantage 均值接近 0，loss 是有限标量，且从 loss 能沿 ratio 回到新策略 log probability。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-contract">
          <span>断言 1</span><strong>loss.ndim == 0：优化目标是标量</strong>
          <span>断言 2</span><strong>torch.isfinite(loss)：没有除零或数值爆炸</strong>
          <span>断言 3</span><strong>组 0 advantage 均值在 atol=1e-6 内为 0</strong>
          <span>断言 4</span><strong>loss.backward 后 log_new.grad 不是 None</strong>
        </div>
        <div class="adv-roadmap">
          <span><b>组内统计</b>rewards → advantages</span>
          <span><b>策略变化</b>log probs → ratio</span>
          <span><b>保守选择</b>min(surr1,surr2)</span>
          <span><b>可训练输出</b>负均值 loss + advantages</span>
        </div>
      </div>`,
      syntaxHtml: practice(
        "验证一个可训练的保守目标",
        [
          "raw = torch.tensor([0.8, -0.6], requires_grad=True)",
          "bounded = torch.tensor([0.7, -0.7])",
          "objective = -torch.min(raw, bounded).mean()",
          "",
          "assert objective.ndim == 0",
          "assert torch.isfinite(objective)",
          "objective.backward()",
          "assert raw.grad is not None"
        ],
        [
          "<code>raw / bounded</code> 对应 <code>surr1 / surr2</code>。",
          "<code>objective</code> 对应最终 <code>loss</code>。",
          "Notebook 还需把第一任务的 <code>advantages</code> 一起返回。",
          "梯度检查要求 loss 的路径不能被 detach 或 item 截断。"
        ]
      ),
      predict: {
        hook: "最终 loss 被错误写成 -torch.min(surr1,surr2)，没有 mean。",
        question: "最先失败的测试是什么？",
        options: ["loss.ndim==0", "组内 advantage 均值", "group_ids 的 dtype"],
        answer: 0,
        revealNote: "未聚合时 loss 仍是 [N]，不是标量。"
      },
      checkpoint: checkpoint(
        "哪组返回 shape 与 Notebook 契约一致？",
        ["loss: []，advantages: [N]", "loss: [N]，advantages: []", "两者都是 []"],
        0,
        "优化器需要标量目标，但测试还要检查每个样本的组内相对优势。"
      ),
      homework: [
        "完成 loss=-torch.min(surr1,surr2).mean()，并返回 loss、advantages。",
        "运行 test_grpo_loss，依次确认标量、finite、组均值与梯度四类条件。",
        "错误诊断：NaN 时检查 std 的 clamp_min；grad None 时检查 detach；组均值失败时检查是否按 group_ids 分组。",
        "用一句话比较 DPO 与 GRPO：前者比较 chosen/rejected 对，后者先比较同 prompt 的候选组。"
      ]
    })
  ],

  "17": [
    lesson({
      id: "attention-backward-output-branch",
      title: "从 out=P@V 分出 dV 与 dP 两条梯度支路",
      todo: "TODO 1 + TODO 2：计算 dV 与 dP",
      prerequisite: [
        "前向中 q/k/v.shape=[B,N,d]，scores/p.shape=[B,N,N]，out/dout.shape=[B,N,d]。",
        "矩阵乘法反向会使用转置：out=P@V，所以 dV=P^T@dout，dP=dout@V^T。",
        "transpose(-2,-1) 只交换最后两个矩阵维，保留 batch 维。",
        "ctx.saved_tensors 按 forward 保存顺序取回 q、k、v、p；它们的 dtype/device 与前向一致。"
      ],
      intuition: "out 的每个值同时依赖注意力概率 P 和内容 V。上游梯度 dout 到达矩阵乘法后分成两路：一路告诉 V 应怎样变化，另一路告诉 P 应怎样变化；之后 P 的梯度还要继续穿过 Softmax。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>out = P @ V</span><strong>dout [B,N,d]</strong>
          <span>P 支路：dP=dout@Vᵀ</span><span>V 支路：dV=Pᵀ@dout</span>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>计算</th><th>左输入</th><th>右输入</th><th>结果</th></tr></thead>
          <tbody>
            <tr><td><code>dV=Pᵀ@dout</code></td><td><code>[B,N,N]</code></td><td><code>[B,N,d]</code></td><td><code>[B,N,d]</code></td></tr>
            <tr><td><code>dP=dout@Vᵀ</code></td><td><code>[B,N,d]</code></td><td><code>[B,d,N]</code></td><td><code>[B,N,N]</code></td></tr>
          </tbody>
        </table>
        <div class="adv-callout">不要使用 .T；对三维 Tensor，.T 的行为不等同于只转置矩阵最后两维。这里明确使用 transpose(-2,-1)。</div>
      </div>`,
      syntaxHtml: practice(
        "手算批量矩阵乘法的两条反向支路",
        [
          "weights = torch.randn(3, 4, 5)",
          "values = torch.randn(3, 5, 2)",
          "grad_output = torch.randn(3, 4, 2)",
          "",
          "grad_values = weights.transpose(-2, -1) @ grad_output",
          "grad_weights = grad_output @ values.transpose(-2, -1)",
          "print(grad_values.shape, grad_weights.shape)"
        ],
        [
          "<code>weights / values / grad_output</code> 对应 <code>p / v / dout</code>。",
          "<code>grad_values</code> 对应 <code>dv</code>，shape 跟 v 一致。",
          "<code>grad_weights</code> 对应 <code>dp</code>，shape 跟 p 一致。",
          "例子故意使用非方形矩阵，帮助你从维度而不是死记位置判断转置。"
        ]
      ),
      predict: {
        hook: "p.shape=[2,8,8]、v.shape=[2,8,16]、dout.shape=[2,8,16]。",
        question: "dv 与 dp 的 shape 分别是什么？",
        options: ["[2,8,16] 与 [2,8,8]", "[2,8,8] 与 [2,8,16]", "[8,16] 与 [8,8]"],
        answer: 0,
        revealNote: "每个输入梯度必须与对应输入 shape 相同；batch 维也保留。"
      },
      checkpoint: checkpoint(
        "dP 为什么使用 dout @ v.transpose(-2,-1)？",
        ["因为 out=P@V，对左矩阵 P 求梯度要把右矩阵 V 转置乘回去", "为了把 batch 维移到最后", "因为 Softmax 只接受二维 Tensor"],
        0,
        "这是批量矩阵乘法的链式法则，转置只发生在最后两个矩阵维。"
      ),
      homework: [
        "完成 dv 与 dp，检查 dv.shape==v.shape、dp.shape==p.shape，且 dtype/device 与输入一致。",
        "在纸上写出 [B,N,N]@[B,N,d] 和 [B,N,d]@[B,d,N] 的维度消去过程。",
        "错误诊断：batch 维丢失时检查是否误用 .T；matmul 维度报错时检查转置对象和先后顺序。",
        "先不要处理 scale；它来自 scores=QK^T×scale，只影响之后的 dQ/dK。"
      ]
    }),

    lesson({
      id: "attention-backward-softmax",
      title: "穿过 Softmax：用逐行修正替代完整雅可比",
      todo: "TODO 3：计算 dp_mul_p、row_sum 与 dS",
      prerequisite: [
        "Softmax 沿 dim=-1 逐行计算，因此反向修正项也必须沿最后一维求和。",
        "dp 与 p 都是 [B,N,N]；逐元素乘法 dp*p 不改变 shape。",
        "sum(dim=-1,keepdim=True) 得到 [B,N,1]，可广播回 [B,N,N]。",
        "公式是 ds=p*(dp-row_sum)，括号与乘法顺序不能随意改变。"
      ],
      intuition: "Softmax 一行中的概率相互耦合：提高一个分数会挤压同一行其他概率。row_sum 汇总这行共同的影响，再从每个 dp 中减去，最后乘 p，得到真正回到 score 的梯度。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>dp_mul_p = dp * p</code><span>先计算每个位置的加权上游梯度</span></div>
          <div><b>2</b><code>row_sum = ...sum(-1, keepdim=True)</code><span>得到每行一个共享修正量</span></div>
          <div><b>3</b><code>ds = p * (dp - row_sum)</code><span>利用广播把修正量应用到整行</span></div>
        </div>
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>keepdim=True</h4>
            <p>row_sum 为 [B,N,1]，可自然广播到每一列。</p>
          </section>
          <section class="adv-panel warn">
            <h4>漏掉 keepdim</h4>
            <p>row_sum 变成 [B,N]，在一般 batch/sequence shape 下可能广播错轴或直接报错。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: practice(
        "对一批类别概率手写 Softmax 向量积",
        [
          "prob = torch.softmax(torch.randn(2, 5), dim=-1)",
          "grad_prob = torch.randn(2, 5)",
          "",
          "weighted = grad_prob * prob",
          "shared = weighted.sum(dim=-1, keepdim=True)",
          "grad_score = prob * (grad_prob - shared)",
          "print(grad_score.shape)"
        ],
        [
          "<code>prob / grad_prob</code> 对应 <code>p / dp</code>。",
          "<code>weighted / shared / grad_score</code> 对应 <code>dp_mul_p / row_sum / ds</code>。",
          "独立例子是 [2,5]，Notebook 是 [B,N,N]；只要始终沿最后一维，公式可直接泛化。",
          "keepdim=True 是让 shared 正确广播的语法关键。"
        ]
      ),
      predict: {
        hook: "dp_mul_p.shape=[2,8,8]。",
        question: "沿 dim=-1 求和并 keepdim=True 后，row_sum.shape 是什么？",
        options: ["[2,8,1]", "[2,8]", "[2,1,8]"],
        answer: 0,
        revealNote: "最后一维被压成长度 1，batch 与 query 行维保留。"
      },
      checkpoint: checkpoint(
        "Softmax backward 为什么不直接写 ds=dp*p？",
        ["同一行概率相互耦合，还要减去行级共享修正项", "因为 dp 与 p 的 dtype 不同", "因为 Softmax 没有梯度"],
        0,
        "逐元素项只是雅可比向量积的一部分；row_sum 体现同一行概率总和为 1 的耦合。"
      ),
      homework: [
        "按顺序完成 dp_mul_p、row_sum 与 ds，并确认三个关键 shape 分别为 [B,N,N]、[B,N,1]、[B,N,N]。",
        "不要显式构造 [N,N] 的 Softmax 雅可比；Notebook 要的是高效的雅可比向量积公式。",
        "错误诊断：广播报错时查 dim/keepdim；gradcheck 数值偏差时查 ds 公式括号和 p 是否乘了两次。",
        "可检查 ds 每行求和应接近 0，这是 Softmax shift-invariance 带来的有用自检。"
      ]
    }),

    lesson({
      id: "attention-backward-qk-gradcheck",
      title: "回到 Q/K：补回 scale，并让 gradcheck 做最终裁判",
      todo: "TODO 4：计算 dQ、dK 并按输入顺序返回",
      prerequisite: [
        "scores=Q@K^T×scale，scale=1/sqrt(d) 已保存在 ctx.scale。",
        "由矩阵乘法可得 dq=ds@k×scale，dk=ds^T@q×scale。",
        "CustomAttention.forward 的输入顺序是 q、k、v，所以 backward 必须返回 dq、dk、dv。",
        "gradcheck 使用 float64、eps=1e-6、atol=1e-4，将解析梯度与有限差分数值梯度比较。"
      ],
      intuition: "dS 已经到达打分矩阵，最后沿 QK^T 的两条支路回传。前向把 score 缩小了 scale，链式法则要求两条输入梯度也各乘一次同样的 scale；返回顺序则决定 Autograd 把梯度交给谁。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>ds [B,N,N]</span><strong>ds@k×scale → dq [B,N,d]</strong>
          <span>dsᵀ [B,N,N]</span><strong>dsᵀ@q×scale → dk [B,N,d]</strong>
        </div>
        <div class="adv-contract">
          <span>前向验收</span><strong>custom_out 与原生 attention ref_out allclose</strong>
          <span>反向验收</span><strong>torch.autograd.gradcheck(CustomAttention.apply,...)</strong>
          <span>测试输入</span><strong>B=2、N=8、d=16、dtype=float64、requires_grad=True</strong>
          <span>返回契约</span><strong>return dq, dk, dv</strong>
        </div>
        <div class="adv-callout">forward allclose 通过不代表 backward 正确。漏 scale、转置错或返回顺序错，都只能由 gradcheck 揭示。</div>
      </div>`,
      syntaxHtml: practice(
        "为缩放双线性打分手写两侧梯度",
        [
          "left = torch.randn(2, 3, 4)",
          "right = torch.randn(2, 5, 4)",
          "grad_score = torch.randn(2, 3, 5)",
          "factor = 0.5",
          "",
          "grad_left = grad_score @ right * factor",
          "grad_right = grad_score.transpose(-2, -1) @ left * factor"
        ],
        [
          "<code>left / right</code> 对应 <code>q / k</code>。",
          "<code>grad_score</code> 对应 <code>ds</code>，<code>factor</code> 对应 <code>scale</code>。",
          "<code>grad_left / grad_right</code> 对应 <code>dq / dk</code>。",
          "Notebook 最后还要把前一任务的 dv 按 q、k、v 顺序一起返回。"
        ]
      ),
      predict: {
        hook: "dq、dk 的矩阵乘法都正确，但漏乘了 scale。",
        question: "forward allclose 与 gradcheck 最可能怎样？",
        options: ["forward 通过，gradcheck 失败", "两者都通过", "forward 失败，gradcheck 通过"],
        answer: 0,
        revealNote: "TODO 只影响 backward；解析梯度会比真实梯度大 sqrt(d) 倍。"
      },
      checkpoint: checkpoint(
        "正确的 backward 返回顺序是什么？",
        ["dq, dk, dv", "dv, dk, dq", "ds, dp, dout"],
        0,
        "Autograd 按 forward 参数位置接收梯度，顺序错误会把正确数值交给错误输入。"
      ),
      homework: [
        "完成 dq 与 dk，两者都乘一次 ctx.scale；确认 dq/dk/dv 分别与 q/k/v shape、dtype、device 一致。",
        "按 q、k、v 的前向输入顺序返回 dq、dk、dv。",
        "运行前向 allclose 和完整 gradcheck；不要通过放宽测试容差掩盖公式错误。",
        "排错顺序：dV/dP 的转置 → Softmax 的 keepdim 与括号 → dQ/dK 的 scale → 返回顺序。"
      ]
    })
  ],

  "18": [
    lesson({
      id: "relu-backward-gate",
      title: "ReLU 反向是一扇逐元素的门",
      todo: "TODO 1：构造 x>0 的 mask",
      prerequisite: [
        "grad_out 是后续层传回的上游梯度，x 是 ReLU 前输入；Notebook 中两者 shape 都是 [5]。",
        "x>0 返回布尔 Tensor；PyTorch 对 x=0 选择局部导数 0，所以不能写 x>=0。",
        "mask.to(grad_out.dtype) 把布尔门转成与上游梯度相同的浮点类型。",
        "逐元素相乘保留原 shape 与 device；门只决定放行或阻断，不改变被放行梯度的符号。"
      ],
      intuition: "前向时，ReLU 只让正输入通过；反向时回看原输入 x，正数位置把 grad_out 原样放行，零和负数位置把梯度关成 0。即使上游梯度是负数，只要 x>0，也应该保留这个负值。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <table class="adv-shapes">
          <thead><tr><th>x</th><th>x&gt;0</th><th>grad_out</th><th>grad_in</th></tr></thead>
          <tbody>
            <tr><td><code>-2</code></td><td><code>False</code></td><td><code>5</code></td><td><code>0</code></td></tr>
            <tr><td><code>0</code></td><td><code>False</code></td><td><code>-3</code></td><td><code>0</code></td></tr>
            <tr><td><code>4</code></td><td><code>True</code></td><td><code>-2</code></td><td><code>-2</code></td></tr>
          </tbody>
        </table>
        <div class="adv-roadmap">
          <span><b>看 x</b>决定哪些位置开门</span>
          <span><b>比较 x&gt;0</b>得到 bool mask</span>
          <span><b>对齐 dtype</b>转成 grad_out.dtype</span>
          <span><b>逐元素乘</b>返回与输入同 shape 的梯度</span>
        </div>
        <div class="adv-callout">mask 由 x 决定，不由 grad_out 的正负决定；写 grad_out&gt;0 会错误删除合法的负梯度。</div>
      </div>`,
      syntaxHtml: practice(
        "用阈值门控制传感器梯度",
        [
          "preactivation = torch.tensor([-1.5, 0.0, 2.5])",
          "upstream = torch.tensor([4.0, -3.0, -0.5])",
          "gate = (preactivation > 0).to(upstream.dtype)",
          "grad_input = upstream * gate",
          "print(grad_input)  # [0.0, -0.0, -0.5]"
        ],
        [
          "<code>preactivation</code> 对应 <code>x</code>，它决定门的开关。",
          "<code>upstream</code> 对应 <code>grad_out</code>。",
          "<code>gate</code> 对应 TODO 的 <code>mask</code>。",
          "函数现有 return 已经完成逐元素乘法，只需正确构造 mask。"
        ]
      ),
      predict: {
        hook: "x=[2,-1]，grad_out=[-4,7]。",
        question: "relu_backward 应返回什么？",
        options: ["[-4,0]", "[0,7]", "[4,0]"],
        answer: 0,
        revealNote: "第一个位置 x>0，负梯度 -4 仍原样通过；第二个位置被门阻断。"
      },
      checkpoint: checkpoint(
        "为什么使用 x>0 而不是 x>=0？",
        ["要与 PyTorch 在 x=0 处采用梯度 0 的约定一致", "因为 >= 不能用于 Tensor", "为了把 dtype 变成 long"],
        0,
        "测试输入明确包含 0，会检查边界位置是否与 F.relu 自动求导一致。"
      ),
      homework: [
        "完成 mask=(x>0).to(grad_out.dtype)，返回值应与 x/grad_out 都是 shape [5]。",
        "运行测试，确认包含 -2、-0.5、0、1、3 的手写梯度与 F.relu 自动求导 allclose。",
        "错误诊断：0 位置不一致时检查 >=；dtype 异常时检查 .to；负上游梯度消失时检查是否误用 grad_out 构造门。",
        "确认实现不原地修改 x 或 grad_out。"
      ]
    }),

    lesson({
      id: "softmax-ce-loss-grad",
      title: "从分类误差得到 logits 梯度：probs-one_hot",
      todo: "TODO 2：计算 probs、one_hot、loss 与 grad",
      prerequisite: [
        "logits.shape=[B,C] 是浮点类别分数，labels.shape=[B] 是 long 类别下标。",
        "Softmax 沿 dim=-1 计算，使每个样本的一行概率和为 1。",
        "zeros_like(probs) 创建同 shape/dtype/device 的 one_hot 容器；scatter_ 需要 labels.unsqueeze(1) 提供 [B,1] 索引。",
        "Notebook 的 loss 对 batch 取 mean，因此梯度是 (probs-one_hot)/B，而不是只做 probs-one_hot。"
      ],
      intuition: "one_hot 是目标分布，probs 是模型分布。两者相减表示每个类别预测多了还是少了；正确类别通常得到负梯度，其他类别得到正梯度。因为总 loss 是 B 个样本的平均，每行贡献还要除以 B。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>probs=softmax(logits,dim=-1)</code><span>[B,C]，每行和为 1</span></div>
          <div><b>2</b><code>one_hot.scatter_(1,labels[:,None],1)</code><span>[B,C]，每行正确类为 1</span></div>
          <div><b>3</b><code>负对数似然再 mean</code><span>得到 shape [] 的批量平均 loss</span></div>
          <div><b>4</b><code>(probs-one_hot)/B</code><span>得到与 logits 同 shape 的手写梯度</span></div>
        </div>
        <table class="adv-shapes">
          <thead><tr><th>对象</th><th>Notebook shape</th><th>dtype</th><th>关键性质</th></tr></thead>
          <tbody>
            <tr><td><code>logits</code></td><td><code>[2,3]</code></td><td>float</td><td>requires_grad=True</td></tr>
            <tr><td><code>labels</code></td><td><code>[2]</code></td><td>long</td><td>值为 0 与 2</td></tr>
            <tr><td><code>probs/one_hot/grad</code></td><td><code>[2,3]</code></td><td>跟随 logits</td><td>每行 grad 和接近 0</td></tr>
            <tr><td><code>loss</code></td><td><code>[]</code></td><td>跟随 logits</td><td>对 batch 求 mean</td></tr>
          </tbody>
        </table>
        <div class="adv-callout">log(probs+1e-12) 中的小常数只防止 log(0)；手写 grad 仍使用化简后的 (probs-one_hot)/B。</div>
      </div>`,
      syntaxHtml: practice(
        "为三条天气样本手写四分类损失与梯度",
        [
          "scores = torch.tensor([[1.2, 0.1, -0.4, 0.7],",
          "                       [0.0, 0.5, 1.4, -0.2],",
          "                       [-0.5, 1.1, 0.3, 0.2]])",
          "targets = torch.tensor([3, 2, 1])",
          "",
          "dist = torch.softmax(scores, dim=-1)",
          "indicator = torch.zeros_like(dist)",
          "indicator.scatter_(1, targets.unsqueeze(1), 1.0)",
          "loss = -(indicator * torch.log(dist + 1e-12)).sum(dim=1).mean()",
          "grad_scores = (dist - indicator) / scores.size(0)"
        ],
        [
          "<code>scores / targets</code> 对应 <code>logits / labels</code>。",
          "<code>dist / indicator</code> 对应 <code>probs / one_hot</code>。",
          "<code>loss / grad_scores</code> 对应函数返回的 <code>loss / grad</code>。",
          "例子 B=3、C=4；Notebook B=2、C=3，所以 batch 除数必须用 logits.size(0)，不能写死。"
        ]
      ),
      predict: {
        hook: "某样本 probs=[0.7,0.2,0.1]，正确类是 0，batch size=2。",
        question: "该样本对应的批量平均 logits 梯度行是什么？",
        options: ["[-0.15,0.10,0.05]", "[-0.30,0.20,0.10]", "[0.35,0.10,0.05]"],
        answer: 0,
        revealNote: "probs-one_hot=[-0.3,0.2,0.1]，再除以 B=2。"
      },
      checkpoint: checkpoint(
        "labels.unsqueeze(1) 的作用是什么？",
        ["把 [B] 类别索引变成 [B,1]，让 scatter_ 沿类别维逐行写入 1", "把标签转成浮点概率", "把类别数扩大一倍"],
        0,
        "unsqueeze 只增加长度为 1 的维度，不改变标签值或 long dtype。"
      ),
      homework: [
        "完成 probs、one_hot、loss、grad；确认前三个矩阵 shape 为 [B,C]，loss 为 0 维标量。",
        "one_hot 使用 zeros_like 与 scatter_；loss 先按类别求和，再按 batch mean；grad 除以 logits.size(0)。",
        "运行测试，让 manual_grad 与 F.cross_entropy 自动梯度在 atol=1e-6 内一致。",
        "错误诊断：差固定 B 倍时检查除 batch；类别错位时检查 Softmax dim、scatter dim 和 unsqueeze；loss 为 Inf 时检查 1e-12。"
      ]
    })
  ]
};
