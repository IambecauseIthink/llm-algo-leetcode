const { advancedStyles, checkpoint, code, lesson } = require("./advanced_lesson_helpers");

module.exports = {
  "12": [
    lesson({
      id: "gradient-accumulation-slice",
      title: "先拆 batch：让输入和标签走同一段切片",
      todo: "TODO 1：切分当前 micro-batch",
      prerequisite: [
        "x 的 shape 是 [batch_size, 4]，每一行是一条输入；y 的 shape 是 [batch_size, 2]，同一行是这条输入的回归目标。",
        "accum_steps 表示一个完整 batch 要拆成几段，不是每段有多少条样本。Notebook 已先检查 batch_size 能被 accum_steps 整除。",
        "micro_size = batch_size // accum_steps 表示每段的样本数。切片只改变当前 forward 看见的数据量，不会复制模型。",
        "x、y 原本是什么 dtype、在什么 device，普通切片得到的 xb、yb 就会继续保持一致。"
      ],
      intuition: "把 8 份作业分成 4 叠，每叠 2 份。第 idx 轮只拿第 idx 叠输入和答案：起点是 idx 乘每叠大小，终点是下一叠的起点。输入 xb 与标签 yb 必须使用完全相同的行范围，否则模型会拿甲的输入去对乙的答案。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>完整 batch</b>8 行输入与 8 行标签</span>
          <span><b>micro_size</b>8 / 4 = 2 行</span>
          <span><b>第 idx 段</b>[idx×2 : (idx+1)×2]</span>
          <span><b>模型本轮</b>只接收 2 行输入</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>变量先对号入座</h4>
            <div class="adv-contract">
              <span><code>x</code></span><strong>[8, 4]，8 条样本，每条 4 个特征</strong>
              <span><code>y</code></span><strong>[8, 2]，每条样本对应 2 个回归目标</strong>
              <span><code>accum_steps</code></span><strong>4，循环会处理 4 个 micro-batch</strong>
              <span><code>micro_size</code></span><strong>2，每次 forward 的 batch 维长度</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>四轮切片覆盖且不重叠</h4>
            <div class="adv-flow">
              <span>idx=0<br>[0:2]</span>
              <span>idx=1<br>[2:4]</span>
              <span>idx=2<br>[4:6]</span>
              <strong>idx=3<br>[6:8]</strong>
            </div>
            <p>四段首尾相接，合起来恰好覆盖原 batch 的全部 8 行。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>张量</th><th>进入循环前</th><th>每轮切片后</th><th>dtype / device</th></tr></thead>
          <tbody>
            <tr><td><code>x → xb</code></td><td><code>[8, 4]</code></td><td><code>[2, 4]</code></td><td>继承 x</td></tr>
            <tr><td><code>y → yb</code></td><td><code>[8, 2]</code></td><td><code>[2, 2]</code></td><td>继承 y</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">先检查 batch 维，也就是第 0 维。这里不能沿特征维切；切成 <code>x[:, start:end]</code> 会改变每条样本的特征数，第一层 Linear 会立刻对不上。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把 12 位观众分成 3 组", [
          "features = torch.randn(12, 5)",
          "scores = torch.randn(12, 1)",
          "groups = 3",
          "group_size = features.size(0) // groups",
          "",
          "for group_idx in range(groups):",
          "    start = group_idx * group_size",
          "    stop = (group_idx + 1) * group_size",
          "    feature_group = features[start:stop]",
          "    score_group = scores[start:stop]",
          "    print(feature_group.shape, score_group.shape)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>features / scores</code> -&gt; <code>x / y</code>，两者必须沿第 0 维使用同一切片。</li>
            <li><code>groups / group_idx</code> -&gt; <code>accum_steps / idx</code>。</li>
            <li><code>group_size</code> -&gt; Notebook 已算好的 <code>micro_size</code>。</li>
            <li><code>feature_group / score_group</code> -&gt; 当前轮要交给模型和损失函数的 <code>xb / yb</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "Notebook 中 batch_size=8、accum_steps=4，因此 micro_size=2。循环正运行到 idx=2。",
        question: "这一轮 xb 应该取原 x 的哪些行？",
        options: ["x[4:6]", "x[2:4]", "x[:, 4:6]"],
        answer: 0,
        revealNote: "起点是 2×2=4，终点是 (2+1)×2=6；切的是第 0 维样本行。"
      },
      checkpoint: checkpoint(
        "若 x.shape=[8,4]、y.shape=[8,2]、accum_steps=4，第一轮 xb 与 yb 的 shape 分别是什么？",
        ["[2,4] 与 [2,2]", "[8,1] 与 [8,1]", "[4,2] 与 [2,4]"],
        0,
        "micro_size=2，只缩短 batch 维；输入特征维 4 和目标维 2 都不变。"
      ),
      homework: [
        "完成 TODO 1：先在纸上列出 idx=0..3 的起止下标，再为 xb、yb 写同一段第 0 维切片。",
        "shape/dtype/device 自检：xb 应为 [micro_size,4]，yb 应为 [micro_size,2]；切片后 dtype 与 device 分别继承 x、y。",
        "测试目标：Notebook 最终会比较累积版和 full-batch 版的每个参数；本阶段先确认 4 轮合起来覆盖全部 8 行且没有重复。",
        "常见错误排查：若 Linear 报矩阵维度不匹配，检查是否误切特征维；若监督错位，检查 xb、yb 的 start/stop 是否完全一致。"
      ]
    }),

    lesson({
      id: "gradient-accumulation-scale",
      title: "再攒梯度：loss 先缩小，backward 再相加",
      todo: "TODO 2：计算、缩放、反传并记录 loss",
      prerequisite: [
        "criterion(pred, yb) 使用 MSELoss(reduction='mean')，会返回一个标量损失；pred 与 yb 都是 [micro_size, 2]。",
        "PyTorch 默认不会在每次 backward 前清空参数的 .grad。连续调用 backward，会把新梯度加到已有梯度上。",
        "完整 batch 被均匀拆成 K 段时，full-batch mean loss 等于 K 个 micro-batch mean loss 的平均值。",
        "detach().item() 只把当前标量数值取出来用于记录，不应拿它继续 backward。"
      ],
      intuition: "每个 micro-batch 都给模型参数投一张“更新建议票”。如果直接把 K 张票全部相加，票数会放大 K 倍；先把每段 loss 除以 K，相当于每张票只占 1/K，累积后正好得到完整 batch 的平均梯度。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>不缩放：更新会被放大</h4>
            <div class="adv-flow">
              <span>grad₁</span><span>grad₂</span><span>grad₃</span><strong>总和</strong>
            </div>
            <p>每段 loss 都是 mean；直接累加 K 段梯度，相当于少除了一个 K。</p>
          </section>
          <section class="adv-panel good">
            <h4>先除 K：恢复 full-batch mean</h4>
            <div class="adv-flow">
              <span>grad₁/K</span><span>grad₂/K</span><span>grad₃/K</span><strong>平均梯度</strong>
            </div>
            <p>Notebook 的分段大小相同，因此这个缩放与完整 batch 的均值损失对齐。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>pred = model(xb)</code><span>[micro_size,4] 经过模型得到 [micro_size,2]</span></div>
          <div><b>2</b><code>raw micro loss</code><span>criterion 比较 pred 与 yb，得到 0 维浮点标量</span></div>
          <div><b>3</b><code>scaled loss = raw / accum_steps</code><span>把当前段的梯度权重缩成 1/K</span></div>
          <div><b>4</b><code>backward</code><span>梯度累加进每个可训练参数的 .grad</span></div>
          <div><b>5</b><code>record detached scalar</code><span>累计数值用于返回，不保留计算图</span></div>
        </div>

        <div class="adv-contract">
          <span><code>loss</code></span><strong>0 维浮点 Tensor，仍连接计算图，可 backward</strong>
          <span><code>loss.detach()</code></span><strong>不再追踪梯度，但仍是 Tensor</strong>
          <span><code>loss.detach().item()</code></span><strong>Python float，只适合日志和 total_loss</strong>
        </div>

        <div class="adv-callout">顺序不能反：对缩放后的 Tensor 调用 backward；不要先 item()，因为 Python 数字没有计算图，也没有 backward 方法。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("分两篮水果累计一个平均梯度", [
          "net = nn.Linear(3, 1)",
          "measure = nn.L1Loss(reduction='mean')",
          "baskets = 2",
          "logged_loss = 0.0",
          "",
          "for fruit_x, fruit_y in data_parts:",
          "    estimate = net(fruit_x)",
          "    part_loss = measure(estimate, fruit_y) / baskets",
          "    part_loss.backward()",
          "    logged_loss += part_loss.detach().item()"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>measure(estimate, fruit_y)</code> -&gt; 用 <code>criterion</code> 比较 <code>pred</code> 和当前 <code>yb</code>。</li>
            <li><code>baskets</code> -&gt; <code>accum_steps</code>，缩放发生在 backward 之前。</li>
            <li><code>part_loss.backward()</code> -&gt; Notebook 已保留的反传位置；TODO 要保证这里拿到的是可求导 Tensor。</li>
            <li><code>logged_loss</code> -&gt; <code>total_loss</code>，只累计脱离计算图后的 Python 数值。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "四个等大的 micro-batch 都计算了 mean loss，随后会连续 backward 四次。",
        question: "若每段 loss 不除以 accum_steps，累积梯度相对 full-batch mean 大约会怎样？",
        options: ["放大 4 倍", "缩小到 1/4", "完全不变"],
        answer: 0,
        revealNote: "四段 mean 梯度直接求和，而完整 batch 的 mean 对应四段梯度的平均，所以会多一个 4 倍因子。"
      },
      checkpoint: checkpoint(
        "哪一个对象应该调用 backward()？",
        ["除以 accum_steps 后、仍连接计算图的 loss Tensor", "loss.detach().item() 得到的 Python float", "total_loss 这个日志变量"],
        0,
        "反传需要计算图；detach 或 item 之后只适合记录数值。"
      ),
      homework: [
        "完成 TODO 2：用当前 pred 与 yb 得到 loss，先按 accum_steps 缩放，再让现有 backward 语句累积梯度，并更新 total_loss。",
        "shape/dtype/device 自检：pred、yb 都应为 [micro_size,2] 且 device 一致；loss 是同 device 的浮点标量，total_loss 最终是 Python float。",
        "测试目标：四段缩放 loss 的数值之和应接近 full-batch loss，参数梯度应对应完整 batch 的平均梯度。",
        "常见错误排查：参数更新差约 4 倍时检查是否漏除 accum_steps；出现无 backward 属性时检查是否过早调用 item()；日志持续持有图时检查是否漏 detach。"
      ]
    }),

    lesson({
      id: "gradient-accumulation-step-test",
      title: "最后只更新一次：读懂参数一致性测试",
      todo: "TODO 3：optimizer.step、zero_grad 与返回值",
      prerequisite: [
        "optimizer.step() 会读取参数当前累积的 .grad 并更新参数；它不会替你重新计算梯度。",
        "optimizer.zero_grad() 清理旧梯度，为下一次参数更新做准备。Notebook 在进入循环前已经清过一次。",
        "一次有效大 batch 应对应一次 step，而不是每个 micro-batch 都 step。否则后面的 micro-batch 会看到已经变化的模型。",
        "测试从同一个 base_model 深拷贝出两个模型，并使用相同 SGD 学习率，因此可以逐参数比较更新结果。"
      ],
      intuition: "前两段是在“攒方向”，这一步才真正移动参数。把 step 放在 micro-batch 循环外，四段数据看到的是同一份旧参数；四段建议全部收齐后只移动一次，才和完整 batch 的单次更新可比。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>循环前</b>zero_grad 清空旧账</span>
          <span><b>4 次 backward</b>只累加 .grad，不改参数</span>
          <span><b>循环后</b>step 读取总梯度并更新一次</span>
          <span><b>收尾</b>zero_grad 后返回 total_loss</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>full-batch 路径</h4>
            <div class="adv-flow">
              <span>8 条样本</span><span>1 次 forward</span><span>1 次 backward</span><strong>1 次 step</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>accumulation 路径</h4>
            <div class="adv-flow">
              <span>4×2 条样本</span><span>4 次 forward</span><span>4 次 backward</span><strong>1 次 step</strong>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span>测试初始条件</span><strong>两个模型来自同一个 base_model，初始参数完全相同</strong>
          <span>优化器</span><strong>两边都是 SGD，学习率都是 0.1</strong>
          <span>核心断言</span><strong>逐参数 torch.allclose，绝对容差 atol=1e-6</strong>
          <span>返回值</span><strong>total_loss 用于打印，不直接决定参数断言是否通过</strong>
        </div>

        <div class="adv-checks">
          <span>step 是否只在循环外出现一次</span>
          <span>zero_grad 是否在更新周期边界执行</span>
          <span>loss 是否先除 accum_steps</span>
          <span>所有 8 条样本是否恰好处理一次</span>
        </div>

        <div class="adv-callout">若 loss 打印正确但参数断言失败，优先检查“缩放”和“step 位置”。日志值接近不代表训练语义一定等价。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("两段传感器数据只触发一次更新", [
          "optimizer.zero_grad()",
          "reported = 0.0",
          "",
          "for sensor_x, sensor_y in sensor_parts:",
          "    output = model(sensor_x)",
          "    scaled = criterion(output, sensor_y) / len(sensor_parts)",
          "    scaled.backward()",
          "    reported += scaled.detach().item()",
          "",
          "optimizer.step()",
          "optimizer.zero_grad()",
          "print(reported)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>sensor_parts</code> 循环结束的位置 -&gt; Notebook 的 <code>for idx in range(accum_steps)</code> 结束处。</li>
            <li>循环后的 <code>optimizer.step()</code> -&gt; TODO 3 的唯一参数更新。</li>
            <li>step 后的 <code>optimizer.zero_grad()</code> -&gt; 清掉本轮已使用的累积梯度。</li>
            <li><code>reported</code> -&gt; 最终返回的 <code>total_loss</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "假设把 optimizer.step() 错放进 micro-batch 循环，每轮 backward 后都更新一次。",
        question: "这样还能和 full-batch 的一次更新严格对齐吗？",
        options: ["不能，后续 micro-batch 会在已经改变的参数上计算", "能，因为 step 放哪里都只看 total_loss", "能，只要最后不返回 loss"],
        answer: 0,
        revealNote: "梯度是在当前参数处计算的。中途改参数后，四段梯度不再来自同一个模型状态。"
      },
      checkpoint: checkpoint(
        "Notebook 用什么条件判定梯度累积真正等价？",
        ["两个模型更新后的每个对应参数在 atol=1e-6 内一致", "只要求两个打印 loss 都大于 0", "只要求函数没有抛异常"],
        0,
        "测试逐个 zip 两个模型的参数，并调用 torch.allclose；这是本页最关键的行为验收。"
      ),
      homework: [
        "完成 TODO 3：在所有 micro-batch 都 backward 之后执行一次 step，清理梯度，并返回 total_loss。",
        "shape/dtype/device 自检：参数与 .grad 的 shape/dtype/device 必须一致；返回值应是可格式化打印的 Python float，不要返回仍连接图的 Tensor。",
        "测试目标：运行 test_gradient_accumulation，确认 full 与 accumulated loss 接近，并让每一对参数通过 atol=1e-6 的 allclose 断言。",
        "常见错误排查：差异成固定倍数先查 loss 缩放；差异杂乱先查 step 是否在循环内、切片是否错位；第二次调用受旧梯度影响时查 zero_grad。"
      ]
    })
  ],

  "13": [
    lesson({
      id: "sft-build-batch",
      title: "先造监督样本：上下文要保留，prompt 不计分",
      todo: "TODO 1：构造 SFT 的 input_ids 与 labels",
      prerequisite: [
        "prompt_ids 是用户提示的 token id 列表，response_ids 是希望模型学会生成的回答 token id 列表。两者都先是 Python list。",
        "input_ids 决定模型能看到什么；labels 决定哪些位置产生训练误差。它们长度必须始终相同。",
        "CrossEntropyLoss 的 ignore_index=-100 会跳过标签值为 -100 的位置，因此 -100 是监督开关，不是词表里的 token。",
        "函数随后已有统一的裁剪或补齐逻辑：input 用 pad_id 补，labels 用 -100 补，最终都变成 torch.long。"
      ],
      intuition: "模型需要先读到问题，才能写回答，所以 prompt 必须出现在 input_ids；但这页只训练回答，不要求模型复述问题，所以 prompt 对应的 labels 放 -100。response 同时出现在输入和标签中，经过下一课的一位错位后，就形成“看前文，猜下一个回答 token”。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>input_ids：模型读到的完整序列</h4>
            <div class="adv-flow">
              <span>prompt<br>1, 2, 3</span><strong>response<br>4, 5, 6, 7</strong><span>padding<br>0</span>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>labels：只有回答与训练有关</h4>
            <div class="adv-flow">
              <span>忽略<br>-100, -100, -100</span><strong>监督<br>4, 5, 6, 7</strong><span>忽略<br>-100</span>
            </div>
          </section>
        </div>

        <div class="adv-contract">
          <span><code>prompt_ids</code></span><strong>上下文 token；进入 input，但对应 label 为 -100</strong>
          <span><code>response_ids</code></span><strong>目标回答 token；进入 input，也保留为监督标签</strong>
          <span><code>pad_id</code></span><strong>只用于把 input 补到 max_len</strong>
          <span><code>-100</code></span><strong>告诉损失函数跳过 prompt 和 padding 位置</strong>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>阶段</th><th>input_ids</th><th>labels</th></tr></thead>
          <tbody>
            <tr><td>拼接后</td><td>长度 = prompt 长度 + response 长度</td><td>同长度，前段为 -100</td></tr>
            <tr><td>裁剪后</td><td>最多 max_len</td><td>使用相同边界裁剪</td></tr>
            <tr><td>补齐后</td><td>pad_id 补到 max_len</td><td>-100 补到 max_len</td></tr>
            <tr><td>返回时</td><td><code>[max_len]</code>，torch.long</td><td><code>[max_len]</code>，torch.long</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要把 prompt 从 input_ids 删除。mask 的是损失，不是上下文；模型仍需要看到 prompt 才能条件生成 response。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把题干与答案做成另一条监督序列", [
          "question_tokens = [9, 8]",
          "answer_tokens = [7, 6, 5]",
          "sequence = question_tokens + answer_tokens",
          "targets = [-100] * len(question_tokens) + answer_tokens",
          "",
          "limit = 7",
          "missing = limit - len(sequence)",
          "sequence = sequence + [0] * missing",
          "targets = targets + [-100] * missing",
          "",
          "sequence = torch.tensor(sequence, dtype=torch.long)",
          "targets = torch.tensor(targets, dtype=torch.long)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>question_tokens / answer_tokens</code> -&gt; <code>prompt_ids / response_ids</code>。</li>
            <li><code>sequence</code> -&gt; TODO 1 的 <code>input_ids</code> 拼接结果。</li>
            <li><code>targets</code> -&gt; TODO 1 的 <code>labels</code>；prompt 长度决定前面有几个 -100。</li>
            <li><code>limit</code> -&gt; <code>max_len</code>；Notebook 已提供裁剪与补齐分支，不要在 TODO 中重复一遍。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "prompt 有 3 个 token，response 有 4 个 token，max_len=8。",
        question: "补齐完成后，labels 中应该有几个 -100？",
        options: ["4 个：3 个 prompt 位置加 1 个 padding 位置", "3 个：只忽略 prompt", "1 个：只忽略 padding"],
        answer: 0,
        revealNote: "prompt 不参与监督，padding 也不参与监督，所以两部分都使用 -100。"
      },
      checkpoint: checkpoint(
        "为什么 prompt token 要留在 input_ids，却在 labels 中写成 -100？",
        ["模型需要 prompt 作为上下文，但本页只让 response 位置贡献 loss", "因为 -100 是 prompt 的 pad_id", "因为模型不能读取正整数 token"],
        0,
        "输入可见范围与损失监督范围是两件事；mask label 不会删除模型上下文。"
      ),
      homework: [
        "完成 TODO 1：只负责 prompt/response 的拼接与 label mask，把已有 max_len 裁剪和补齐逻辑留在后面统一处理。",
        "shape/dtype/device 自检：返回的 input_ids、labels 都应为 [max_len]、torch.long；torch.tensor 默认创建在 CPU，符合本 Notebook 的 CPU-first 测试。",
        "为测试样例手算 max_len=8 的两个返回序列；随后 unsqueeze/repeat 后应得到 batch shape [4,8]。",
        "常见错误排查：若 prompt 也产生 loss，检查 labels 前缀；若长度不一致，检查拼接和 padding；若 Embedding 报 dtype 错，检查是否返回 long。"
      ]
    }),

    lesson({
      id: "sft-next-token-loss",
      title: "错开一位：让当前位置预测下一个 token",
      todo: "TODO 2：shift logits、shift labels 与交叉熵",
      prerequisite: [
        "TinyCausalLM 输出 logits，shape 是 [batch_size, seq_len, vocab_size]；最后一维是每个候选 token 的未归一化分数。",
        "labels 的 shape 是 [batch_size, seq_len]；每个位置存正确 token id 或 -100。",
        "自回归训练要求位置 t 的输出去预测位置 t+1 的 token，所以 logits 去掉最后一步，labels 去掉第一步。",
        "CrossEntropyLoss 接收分类分数 [N,C] 与类别索引 [N]；这里 C=vocab_size，需要把 batch 和时间两维展平。"
      ],
      intuition: "序列像接龙：读到第 0 个 token 后预测第 1 个，读到第 1 个后预测第 2 个。最后一个位置没有“下一个标签”，所以 logits 丢掉末尾；第一个 token 前没有本函数提供的预测位置，所以 labels 丢掉开头。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>原 logits</b>[B, T, V]</span>
          <span><b>去最后时间步</b>[B, T-1, V]</span>
          <span><b>labels 去第一项</b>[B, T-1]</span>
          <span><b>展平分类</b>[B×(T-1), V] 对 [B×(T-1)]</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>时间位置如何配对</h4>
            <div class="adv-contract">
              <span>logits 位置 0</span><strong>对齐 labels 位置 1</strong>
              <span>logits 位置 1</span><strong>对齐 labels 位置 2</strong>
              <span>logits 位置 T-2</span><strong>对齐 labels 位置 T-1</strong>
              <span>logits 位置 T-1</span><strong>没有下一个标签，丢弃</strong>
            </div>
          </section>
          <section class="adv-panel blue">
            <h4>分类损失眼中的数据</h4>
            <div class="adv-flow">
              <span>B 个样本</span><span>T-1 个位置</span><strong>B×(T-1) 道 V 选 1 的分类题</strong>
            </div>
            <p><code>ignore_index=-100</code> 会跳过 prompt 与 padding 对应的题目。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>Notebook 测试中的 shape</th><th>语义</th></tr></thead>
          <tbody>
            <tr><td><code>logits</code></td><td><code>[4,8,64]</code></td><td>每条序列每个位置对 64 个词的分数</td></tr>
            <tr><td><code>shift_logits</code></td><td><code>[4,7,64]</code></td><td>去掉没有下一个目标的最后一步</td></tr>
            <tr><td><code>shift_labels</code></td><td><code>[4,7]</code></td><td>去掉无法由序列内前一步预测的首 token</td></tr>
            <tr><td><code>loss</code></td><td><code>[]</code></td><td>忽略 -100 后的平均交叉熵标量</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要在词表维上 shift。省略号切片中的倒数第二维是时间，最后一维 V 必须完整保留给 CrossEntropyLoss。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用天气序列练 next-step 分类", [
          "batch, days, states = 2, 6, 5",
          "weather_logits = torch.randn(batch, days, states)",
          "weather_labels = torch.tensor([",
          "    [-100, 1, 2, 3, 4, -100],",
          "    [-100, 2, 2, 1, 0, -100],",
          "])",
          "",
          "predict_next = weather_logits[..., :-1, :].contiguous()",
          "next_state = weather_labels[..., 1:].contiguous()",
          "criterion = nn.CrossEntropyLoss(ignore_index=-100)",
          "loss = criterion(",
          "    predict_next.view(-1, predict_next.size(-1)),",
          "    next_state.view(-1),",
          ")"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>weather_logits</code> -&gt; <code>logits</code>，最后一维 <code>states</code> -&gt; <code>vocab_size</code>。</li>
            <li><code>predict_next</code> -&gt; <code>shift_logits</code>，保留所有类别列，只去掉最后时间步。</li>
            <li><code>next_state</code> -&gt; <code>shift_labels</code>，去掉第一时间步。</li>
            <li>两次 <code>view</code> -&gt; TODO 2 中把 token 位置合成 N，再交给 ignore_index=-100 的交叉熵。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "logits.shape=[4,8,64]，labels.shape=[4,8]。",
        question: "完成 next-token shift 后，哪组 shape 正确？",
        options: ["shift_logits=[4,7,64]，shift_labels=[4,7]", "shift_logits=[4,8,63]，shift_labels=[4,7]", "shift_logits=[3,8,64]，shift_labels=[4,8]"],
        answer: 0,
        revealNote: "只在时间维错开一位；batch 维与 vocab 维都保持不变。"
      },
      checkpoint: checkpoint(
        "送入 CrossEntropyLoss 前，为什么 shift_logits 的最后一维不能展平掉？",
        ["它是分类类别数 vocab_size，必须保留为 C 维", "它表示 batch_size", "它只包含被忽略的位置"],
        0,
        "交叉熵需要每个样本对所有类别的分数 [N,C]，因此只合并 batch 与时间维。"
      ),
      homework: [
        "完成 TODO 2：先沿时间维构造 shift_logits 与 shift_labels，再用 ignore_index=-100 的交叉熵得到标量 loss；可按语法支架自行组织 loss_fct。",
        "shape/dtype/device 自检：测试时两者应从 [4,8,64]/[4,8] 变为 [4,7,64]/[4,7]；logits 为浮点，labels 为 long，且必须同 device。",
        "先单独调用 compute_sft_loss，确认返回 0 维有限 Tensor；端到端测试会比较训练前后的 .item() 数值。",
        "常见错误排查：类别越界或 dtype 报错查 labels；size mismatch 查 shift 方向；prompt 被计入 loss 查 ignore_index；view 报错时确认切片后 contiguous。"
      ]
    }),

    lesson({
      id: "sft-training-loop",
      title: "合上训练闭环：每次 update 包含两次 micro-backward",
      todo: "TODO 3：梯度累积训练循环与 history",
      prerequisite: [
        "input_ids 与 labels 的 batch shape 都是 [4,8]；accum_steps=2，所以每个 micro-batch 有 2 条序列。",
        "一个 update 开始时先 zero_grad，随后两段数据各自 forward、算 loss、缩放并 backward，最后只 step 一次。",
        "history 每个 update 追加一个 total_loss，因此 num_updates=30 时长度必须恰好是 30。",
        "测试重复同一条样本 4 次并使用较大学习率，是为了让小模型快速过拟合；验收重点是 final_loss 小于 init_loss。"
      ],
      intuition: "这一课不再发明新公式，而是把前两课接上线：切两段 batch，各自经过模型和 SFT loss，把两份缩放梯度攒起来，再更新一次。外层循环重复 30 次，history 就像 30 张训练收据，证明每次 update 都完整走过。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>update 开始</b>train 模式 + zero_grad</span>
          <span><b>micro 0</b>切片 → logits → loss/2 → backward</span>
          <span><b>micro 1</b>切片 → logits → loss/2 → backward</span>
          <span><b>update 结束</b>step → history 追加一次</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>循环层级</h4>
            <div class="adv-contract">
              <span>外层 30 次</span><strong>控制参数更新次数，history 最终长度为 30</strong>
              <span>内层 2 次</span><strong>控制一个 update 内的 micro-batch 数量</strong>
              <span>step 次数</span><strong>每个外层循环 1 次，总共 30 次</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>测试为什么重复同一条样本</h4>
            <div class="adv-flow">
              <span>同一目标 ×4</span><span>干扰更少</span><span>反复训练</span><strong>loss 应明显下降</strong>
            </div>
            <p>它验证的是链路能学习，不是在评估泛化能力。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape / 类型</th><th>何时变化</th></tr></thead>
          <tbody>
            <tr><td><code>mb_input</code></td><td><code>[2,8]</code> long</td><td>每个 idx 切出不同两行</td></tr>
            <tr><td><code>mb_labels</code></td><td><code>[2,8]</code> long</td><td>与 mb_input 使用同一范围</td></tr>
            <tr><td><code>logits</code></td><td><code>[2,8,64]</code> float</td><td>每次 forward 重新计算</td></tr>
            <tr><td><code>loss</code></td><td>0 维 float Tensor</td><td>每段先除 accum_steps</td></tr>
            <tr><td><code>history</code></td><td>Python list[float]</td><td>每个 update 只 append 一次</td></tr>
          </tbody>
        </table>

        <div class="adv-checks">
          <span>len(history) == 30</span>
          <span>final_loss &lt; init_loss</span>
          <span>每个 update 只 step 一次</span>
          <span>total_loss 不持有计算图</span>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用两段图像样本组成一次 update", [
          "parts = 2",
          "part_size = images.size(0) // parts",
          "curve = []",
          "",
          "for epoch_step in range(12):",
          "    optimizer.zero_grad()",
          "    measured = 0.0",
          "    for part_idx in range(parts):",
          "        start = part_idx * part_size",
          "        stop = (part_idx + 1) * part_size",
          "        image_part = images[start:stop]",
          "        label_part = classes[start:stop]",
          "        scores = classifier(image_part)",
          "        scaled = classification_loss(scores, label_part) / parts",
          "        scaled.backward()",
          "        measured += scaled.detach().item()",
          "    optimizer.step()",
          "    curve.append(measured)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>images / classes</code> -&gt; <code>input_ids / labels</code>，按同一 batch 范围切成 micro-batch。</li>
            <li><code>classifier(image_part)</code> -&gt; <code>model(mb_input)</code> 得到 logits。</li>
            <li><code>classification_loss</code> -&gt; 前一课实现的 <code>compute_sft_loss</code>。</li>
            <li><code>curve.append(measured)</code> -&gt; 每次参数更新后向 <code>history</code> 追加一次 total_loss。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "num_updates=30、accum_steps=2。每个内层 micro-batch 都 backward，但 step 在内层结束后。",
        question: "完整实验会调用多少次 optimizer.step()？",
        options: ["30 次", "60 次", "2 次"],
        answer: 0,
        revealNote: "accum_steps 控制每次更新前累计几段梯度，不会把参数更新次数翻倍。"
      },
      checkpoint: checkpoint(
        "Notebook 的 final_loss < init_loss 断言主要证明什么？",
        ["数据、SFT loss、反传与优化器更新已经组成可学习的闭环", "history 中每个值都必须严格单调下降", "模型已经具备真实任务泛化能力"],
        0,
        "重复样本实验只验证最小训练链路能降低目标损失，不要求每一步单调，也不证明泛化。"
      ),
      homework: [
        "完成 TODO 3：在内层依次切 mb_input/mb_labels、前向、调用 compute_sft_loss、按 accum_steps 缩放、backward 并累计 total_loss；内层后 step，外层末 append。",
        "shape/dtype/device 自检：测试中 micro 输入/标签为 [2,8] long，logits 为 [2,8,64] float；模型、输入、标签须在同一 device，history 元素用 detach().item() 变成 float。",
        "运行 test_end_to_end_finetuning：必须同时满足 len(history)==30 与 final_loss&lt;init_loss，打印值只作观察。",
        "常见错误排查：history 长度 60 查 append 是否在内层；loss 不降查 step、缩放、shift 和 label mask；NameError 查是否仍保留 pass 或漏定义 micro 变量。"
      ]
    })
  ],

  "14": [
    lesson({
      id: "ppo-log-ratio",
      title: "先读概率变化：对数相减，再还原成 ratio",
      todo: "TODO 1：计算重要性采样比率 ratio",
      prerequisite: [
        "log_probs_new 与 log_probs_old 都是 [batch_size, seq_len]，每个元素对应同一个已采样 token 在新旧 Actor 下的对数概率。",
        "对数概率通常是非正数；数值越接近 0，原概率越大。不能直接用两个 log 值相除来得到概率比。",
        "log(a/b) = log(a) - log(b)，所以概率比可由对数概率之差再取 exp 得到。",
        "ratio 是无单位的浮点张量，shape、dtype、device 都与输入逐元素运算保持一致。"
      ],
      intuition: "ratio 是“新 Actor 比旧 Actor 更愿意生成这个 token 多少倍”。ratio=1 表示态度没变；大于 1 表示新模型更偏爱；小于 1 表示更不偏爱。先在 log 空间做减法，比先还原两个很小的概率再相除更稳。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>旧 Actor 采样</b>记录每个 token 的 old log prob</span>
          <span><b>当前 Actor 重算</b>得到对应 token 的 new log prob</span>
          <span><b>逐元素相减</b>new log prob - old log prob</span>
          <span><b>exp</b>还原成概率倍数 ratio</span>
        </div>

        <div class="adv-grid three">
          <section class="adv-panel good">
            <h4>ratio &gt; 1</h4>
            <p>新策略给该 token 更高概率。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>ratio = 1</h4>
            <p>新旧策略对该 token 的概率相同。</p>
          </section>
          <section class="adv-panel warn">
            <h4>ratio &lt; 1</h4>
            <p>新策略降低了该 token 的概率。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>delta_logp = log_probs_new - log_probs_old</code><span>[B,S]，记录对数概率改变了多少</span></div>
          <div><b>2</b><code>ratio = exp(delta_logp)</code><span>[B,S]，把加法尺度还原成概率倍数</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>Notebook 测试首行</th><th>log 差</th><th>ratio 含义</th></tr></thead>
          <tbody>
            <tr><td>new=-2.0，old=-2.1</td><td>0.1</td><td>exp(0.1)&gt;1，新策略略微提高概率</td></tr>
            <tr><td>new=-1.5，old=-1.4</td><td>-0.1</td><td>exp(-0.1)&lt;1，新策略略微降低概率</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要写 <code>log_probs_new / log_probs_old</code>。那是两个对数值的比，不是两个概率的比。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("比较新版推荐器与旧版的点击概率", [
          "new_log_click = torch.tensor([[-0.7, -1.1, -0.4]])",
          "old_log_click = torch.tensor([[-0.9, -1.1, -0.2]])",
          "log_change = new_log_click - old_log_click",
          "probability_ratio = torch.exp(log_change)",
          "print(probability_ratio.shape)  # [1, 3]"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>new_log_click / old_log_click</code> -&gt; <code>log_probs_new / log_probs_old</code>。</li>
            <li><code>log_change</code> -&gt; 新旧 Actor 对同一 token 的 log probability 差。</li>
            <li><code>probability_ratio</code> -&gt; TODO 1 的 <code>ratio</code>。</li>
            <li>例子的 [1,3] -&gt; Notebook 的 [batch_size,seq_len]；所有运算都逐元素进行。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某 token 的 new log prob 与 old log prob 完全相同。",
        question: "这个 token 的 ratio 是多少？",
        options: ["1", "0", "无法从 log prob 得到"],
        answer: 0,
        revealNote: "两者之差为 0，而 exp(0)=1，表示概率没有变化。"
      },
      checkpoint: checkpoint(
        "哪条计算表达了 new probability / old probability？",
        ["exp(log_probs_new - log_probs_old)", "log_probs_new / log_probs_old", "exp(log_probs_old + log_probs_new)"],
        0,
        "概率之比在 log 空间对应相减，再用 exp 返回概率尺度。"
      ),
      homework: [
        "完成 TODO 1：根据对数比恒等式构造 ratio，先不要提前做 mean 或改变 shape。",
        "shape/dtype/device 自检：三个输入测试张量都是 [2,2]；ratio 也必须是 [2,2] 浮点 Tensor，并跟随 log_probs_new 的 device。",
        "用测试中的第一个位置手算 exp(-2.0-(-2.1))，并与函数中 ratio 对应元素核对。",
        "常见错误排查：ratio 出现负数说明公式错误；shape 变成标量说明过早归约；全为 1 时检查是否误做同一变量相减。"
      ]
    }),

    lesson({
      id: "ppo-surrogates-and-clip",
      title: "再看好坏方向：advantage 决定该鼓励还是抑制",
      todo: "TODO 2 + TODO 3：无截断与截断 surrogate",
      prerequisite: [
        "advantages 与 ratio 同为 [batch_size, seq_len]；每个 token 都有一个优势值。正值表示这个动作比基线好，负值表示比基线差。",
        "surrogate 不是最终 loss，而是一个希望最大化的代理目标。无截断版本是 ratio 与 advantage 的逐元素乘积。",
        "clip_range=0.2 时，允许的 ratio 区间是 [0.8,1.2]；torch.clamp 会逐元素限制 ratio。",
        "截断的是 ratio，不是 advantage，也不是 log probability。"
      ],
      intuition: "advantage 像方向牌：正数说“这个 token 值得更常出现”，负数说“这个 token 应该少出现”。ratio 记录新策略已经走了多远。PPO 同时算原始方案和限速方案，下一课再挑更保守的结果。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>advantage 为正</h4>
            <div class="adv-flow">
              <span>token 比基线好</span><span>希望提高概率</span><strong>但 ratio 不应无限上涨</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>advantage 为负</h4>
            <div class="adv-flow">
              <span>token 比基线差</span><span>希望降低概率</span><strong>但 ratio 不应无限下跌</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>surr1 = ratio × advantages</code><span>按新旧概率变化修正每个 token 的优势</span></div>
          <div><b>2</b><code>bounded_ratio = clamp(ratio, 1-ε, 1+ε)</code><span>把概率变化限制在信赖区间</span></div>
          <div><b>3</b><code>surr2 = bounded_ratio × advantages</code><span>得到限速后的代理目标</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape</th><th>数值角色</th></tr></thead>
          <tbody>
            <tr><td><code>ratio</code></td><td><code>[B,S]</code></td><td>新概率 / 旧概率</td></tr>
            <tr><td><code>advantages</code></td><td><code>[B,S]</code></td><td>每个 token 的好坏方向和强度</td></tr>
            <tr><td><code>surr1</code></td><td><code>[B,S]</code></td><td>未经限制的目标</td></tr>
            <tr><td><code>surr2</code></td><td><code>[B,S]</code></td><td>ratio 被限制后的目标</td></tr>
          </tbody>
        </table>

        <div class="adv-contract">
          <span>默认 ε</span><strong>0.2</strong>
          <span>ratio 下界</span><strong>1.0 - ε = 0.8</strong>
          <span>ratio 上界</span><strong>1.0 + ε = 1.2</strong>
          <span>advantage</span><strong>原值保留，不做 clamp</strong>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("给价格调整设置上下浮动边界", [
          "change_ratio = torch.tensor([[1.35, 0.72, 1.05]])",
          "benefit = torch.tensor([[0.8, -0.4, 0.2]])",
          "limit = 0.15",
          "",
          "free_score = change_ratio * benefit",
          "safe_ratio = torch.clamp(",
          "    change_ratio,",
          "    1.0 - limit,",
          "    1.0 + limit,",
          ")",
          "safe_score = safe_ratio * benefit"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>change_ratio / benefit</code> -&gt; <code>ratio / advantages</code>。</li>
            <li><code>free_score</code> -&gt; TODO 2 的 <code>surr1</code>。</li>
            <li><code>limit</code> -&gt; <code>clip_range</code>；上下界都围绕 1.0。</li>
            <li><code>safe_score</code> -&gt; TODO 3 的 <code>surr2</code>，shape 不因 clamp 改变。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "clip_range=0.2，某位置 ratio=1.35、advantage=0.5。",
        question: "该位置 surr2 使用的 ratio 是多少？",
        options: ["1.2", "1.35", "0.5"],
        answer: 0,
        revealNote: "ratio 被限制到上界 1+0.2=1.2，advantage 仍保持 0.5。"
      },
      checkpoint: checkpoint(
        "PPO 的 TODO 3 应该 clamp 哪个量？",
        ["ratio，范围为 [1-clip_range, 1+clip_range]", "advantages，范围为 [-clip_range, clip_range]", "surr1，范围为 [0,1]"],
        0,
        "PPO Clip 限制的是新旧策略概率比的变化幅度。"
      ),
      homework: [
        "完成 TODO 2/3：保持逐元素运算，分别构造未截断 surr1 与 ratio 截断后的 surr2，不要在此阶段 mean。",
        "shape/dtype/device 自检：ratio、advantages、surr1、surr2 在测试中都应为 [2,2] 浮点 Tensor，device 一致；clip_range 是 Python float，可直接参与张量运算。",
        "拿 ratio=[1.35,0.72]、clip_range=0.2 手算 clamp 结果 [1.2,0.8]，再核对正负 advantage 下的乘积符号。",
        "常见错误排查：上下界写反会报错；结果完全未截断时查是否 clamp 了错误变量；shape 变小查是否误用了 sum/mean。"
      ]
    }),

    lesson({
      id: "ppo-pessimistic-loss-memory",
      title: "最后取保守目标：逐 token 取 min，再变成标量 loss",
      todo: "TODO 4：PPO Actor Loss；串联四模型流转与测试",
      prerequisite: [
        "surr1 与 surr2 都是 [batch_size, seq_len]；torch.min(surr1, surr2) 在这里需要逐元素比较，而不是一次性求全局最小值。",
        "PPO 想最大化代理目标，但 PyTorch 优化器默认最小化 loss，因此最终要加负号。",
        "mean 会把 batch 和 seq_len 上所有 token 汇总成 0 维标量，正好可用于 backward。",
        "本 Notebook 只实现 Actor Clip Loss；真实 RLHF 还会让 Actor、Reference、Reward、Critic 在采样与训练阶段传递 log prob、reward、value 和 advantage。这里的 old policy 是 PPO 更新前的 Actor 快照，不等同于专门计算 KL 约束的 Reference。"
      ],
      intuition: "对每个 token，PPO 在“原始目标”和“限速目标”之间挑较小的一个，相当于不提前兑现过于乐观的收益。随后对所有 token 求平均得到整批分数；因为代码要最小化，所以再取负号。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>token_objective = minimum(surr1, surr2)</code><span>[B,S]，每个 token 单独选择更保守的值</span></div>
          <div><b>2</b><code>batch_objective = mean(token_objective)</code><span>把全部 token 汇总为 0 维标量</span></div>
          <div><b>3</b><code>actor_loss = -batch_objective</code><span>把最大化目标改写成优化器可最小化的 loss</span></div>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>本函数的数据流</h4>
            <div class="adv-flow">
              <span>new/old log probs</span><span>ratio</span><span>surr1 与 surr2</span><strong>标量 actor loss</strong>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>真实 RLHF 四个角色</h4>
            <div class="adv-contract">
              <span>Actor</span><strong>生成 token，并作为当前策略接受梯度</strong>
              <span>Reference</span><strong>提供 KL 对照，通常冻结；它不是 ratio 中的 old policy 快照</strong>
              <span>Reward Model</span><strong>给整段回复提供奖励信号</strong>
              <span>Critic</span><strong>估计 value，帮助得到 advantage</strong>
            </div>
          </section>
        </div>

        <div class="adv-roadmap">
          <span><b>Actor 生成</b>token 与采样时 old log prob</span>
          <span><b>Reward 打分</b>回复得到 reward</span>
          <span><b>Critic 估值</b>组合出 advantages</span>
          <span><b>Actor 更新</b>本页 clip loss 约束步幅</span>
        </div>

        <div class="adv-checks">
          <span>ratio 解析式与测试一致</span>
          <span>min 是逐元素比较</span>
          <span>mean 后 loss 是标量</span>
          <span>负号放在最终目标外</span>
        </div>

        <div class="adv-callout">ratio 中的 old log prob 来自采样时的旧 Actor；Reference 则是限制策略长期偏离的冻结锚点。显存压力来自多个模型及其激活、参数和优化器状态。本 TODO 不要求估算显存，也不要额外实现 Critic 或 KL 项。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("从两套限额方案中逐项选保守收益", [
          "open_plan = torch.tensor([[0.9, -0.6], [0.4, -0.2]])",
          "limited_plan = torch.tensor([[0.7, -0.4], [0.4, -0.3]])",
          "conservative = torch.minimum(open_plan, limited_plan)",
          "objective = conservative.mean()",
          "cost_to_minimize = -objective"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>open_plan / limited_plan</code> -&gt; <code>surr1 / surr2</code>。</li>
            <li><code>torch.minimum</code> -&gt; TODO 4 的逐元素保守选择；也可使用能返回逐元素最小值的等价 PyTorch 写法。</li>
            <li><code>objective.mean()</code> -&gt; 对 batch 与 seq_len 中所有 token 求平均。</li>
            <li><code>cost_to_minimize</code> -&gt; 函数最终返回的标量 <code>loss</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "surr1 与 surr2 都是 [2,2]。先逐元素 min，再 mean。",
        question: "最终 loss 的 shape 是什么？",
        options: ["[]，0 维标量", "[2,2]", "[2]"],
        answer: 0,
        revealNote: "不指定 dim 的 mean 会聚合张量全部四个元素，负号不会改变 shape。"
      },
      checkpoint: checkpoint(
        "最终为什么是 -torch.min(surr1, surr2).mean() 这一类结构？",
        ["逐 token 取保守目标并求平均，再用负号把最大化改成最小化", "先找整个 batch 唯一最小 token，再复制到全部位置", "让 loss 保持 [batch,seq] 方便打印"],
        0,
        "min 的逐元素结果仍为 [B,S]，mean 汇总为标量，负号转换优化方向。"
      ),
      homework: [
        "完成 TODO 4：对 surr1/surr2 做逐元素最小选择，聚合全部 token，并加上正确优化方向的符号；不要扩展 Notebook 未要求的 KL 或 Critic loss。",
        "shape/dtype/device 自检：min 前两项为 [2,2]，最终 loss 必须是 0 维浮点 Tensor，且仍连接 log_probs_new 的梯度图与 device。",
        "运行 test_ppo_actor_loss：测试会独立重算 ratio、s1、s2、expected_loss，并用 torch.allclose 对照你的返回值。",
        "常见错误排查：数值仅差一个负号就查优化方向；得到单个极端元素查是否误用全局 min；loss 无梯度查是否 detach/item；不匹配再查 clamp 区间。"
      ]
    })
  ],

  "15": [
    lesson({
      id: "dpo-implicit-rewards",
      title: "先做四列减法：Policy 相对 Reference 改变了多少",
      todo: "TODO 1 + TODO 2：chosen / rejected 的 log-ratio 与隐式奖励",
      prerequisite: [
        "四个输入都是 [batch_size]：每个元素已经是一整段回复的 token log probability 之和，不需要在本函数里再按 token 求和。",
        "chosen 表示偏好数据中更好的回复，rejected 表示同一 prompt 下较差的回复；两者必须按 batch 行一一配对。",
        "Policy 是正在训练的模型，Reference 是冻结的对照模型。log policy - log reference 表示 Policy 相对参考分布提高或降低了这段回复的倾向。",
        "beta 是 Python float；乘上 [B] 的 log-ratio 后得到 [B] 浮点隐式奖励，不会改变 shape 或 device。"
      ],
      intuition: "先别急着比较 chosen 与 rejected。对每条回复先问一个更基础的问题：当前 Policy 相比 Reference，更愿意还是更不愿意生成它？chosen 做一次 Policy-Reference，rejected 也做一次，得到两条站在同一参考线上的分数。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>chosen 分支</h4>
            <div class="adv-flow">
              <span>policy chosen logps [B]</span><span>减 reference chosen logps [B]</span><strong>chosen log-ratio [B]</strong>
            </div>
          </section>
          <section class="adv-panel warn">
            <h4>rejected 分支</h4>
            <div class="adv-flow">
              <span>policy rejected logps [B]</span><span>减 reference rejected logps [B]</span><strong>rejected log-ratio [B]</strong>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>chosen log-ratio</code><span>Policy chosen - Reference chosen</span></div>
          <div><b>2</b><code>rejected log-ratio</code><span>Policy rejected - Reference rejected</span></div>
          <div><b>3</b><code>implicit rewards</code><span>两个 log-ratio 分别乘 beta</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>测试首样本</th><th>Policy</th><th>Reference</th><th>beta=0.1 后</th></tr></thead>
          <tbody>
            <tr><td>chosen</td><td>-1.0</td><td>-2.0</td><td>0.1 × (1.0) = 0.1</td></tr>
            <tr><td>rejected</td><td>-3.0</td><td>-2.0</td><td>0.1 × (-1.0) = -0.1</td></tr>
          </tbody>
        </table>

        <div class="adv-contract">
          <span>正 log-ratio</span><strong>Policy 比 Reference 更偏爱这条回复</strong>
          <span>零 log-ratio</span><strong>Policy 与 Reference 对这条回复倾向相同</strong>
          <span>负 log-ratio</span><strong>Policy 比 Reference 更不偏爱这条回复</strong>
          <span><code>beta</code></span><strong>统一缩放两条分支，得到用于比较的隐式奖励</strong>
        </div>

        <div class="adv-callout">不要交叉相减：policy_chosen 必须对 reference_chosen，policy_rejected 必须对 reference_rejected。先建立同类对照，下一课才做 chosen-rejected 偏好比较。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("比较新版与基线对两种文案的倾向", [
          "new_preferred = torch.tensor([-0.6, -1.4, -0.9])",
          "new_skipped = torch.tensor([-2.1, -1.8, -2.5])",
          "base_preferred = torch.tensor([-1.0, -1.0, -1.0])",
          "base_skipped = torch.tensor([-1.7, -1.7, -1.7])",
          "temperature = 0.25",
          "",
          "preferred_change = new_preferred - base_preferred",
          "skipped_change = new_skipped - base_skipped",
          "preferred_score = temperature * preferred_change",
          "skipped_score = temperature * skipped_change"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>new_preferred / new_skipped</code> -&gt; Policy 的 chosen / rejected logps。</li>
            <li><code>base_preferred / base_skipped</code> -&gt; Reference 的 chosen / rejected logps。</li>
            <li><code>preferred_change / skipped_change</code> -&gt; TODO 1/2 的两条 <code>pi_logratios</code>。</li>
            <li><code>temperature</code> -&gt; <code>beta</code>；Notebook 已在 TODO 后写好两次乘法。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "某条 chosen 回复的 policy logps=-1.2，reference logps=-2.0，beta=0.1。",
        question: "它的 chosen_rewards 是多少？",
        options: ["0.08", "-0.08", "0.32"],
        answer: 0,
        revealNote: "先算 -1.2-(-2.0)=0.8，再乘 0.1 得到 0.08。"
      },
      checkpoint: checkpoint(
        "TODO 1 的 chosen log-ratio 应比较哪两个量？",
        ["policy_chosen_logps - reference_chosen_logps", "policy_chosen_logps - policy_rejected_logps", "reference_rejected_logps - policy_chosen_logps"],
        0,
        "隐式奖励先衡量同一回复在 Policy 与 Reference 下的相对倾向。"
      ),
      homework: [
        "完成 TODO 1/2：分别做 chosen 同类相减与 rejected 同类相减；beta 乘法已在 Notebook 后续代码中，不要重复乘。",
        "shape/dtype/device 自检：四个输入、两条 log-ratio、chosen_rewards、rejected_rewards 都应为 [batch_size]=[4] 浮点 Tensor，并保持同 device。",
        "运行前先手算测试首项：chosen reward=0.1、rejected reward=-0.1；测试会用 torch.allclose 精确核对这两项。",
        "常见错误排查：符号相反查减法顺序；结果多一个 0.1 因子查是否重复乘 beta；shape 变标量查是否误加 mean/sum；样本错配查 chosen/rejected 分支。"
      ]
    }),

    lesson({
      id: "dpo-pairwise-loss",
      title: "再做偏好比较：chosen 奖励减 rejected 奖励",
      todo: "TODO 3：DPO logits 与逐样本 losses；读懂全部断言",
      prerequisite: [
        "chosen_rewards 与 rejected_rewards 都是 [batch_size]，同一个下标属于同一个 prompt 的一对回复。",
        "DPO logits 是 chosen 隐式奖励减 rejected 隐式奖励；正值表示当前模型相对参考模型更符合偏好顺序。",
        "F.logsigmoid(logits) 稳定地计算 log(sigmoid(logits))；取负后得到要最小化的非负损失。",
        "本函数返回逐样本 losses [B]，不能在函数内 mean，因为 Notebook 明确断言 shape==(batch_size,)。"
      ],
      intuition: "DPO 把每对回答变成一道二选一题：chosen 应该排在 rejected 前面。两者奖励差越大且为正，模型答对得越有把握，损失越小；差值为负，说明排序反了，损失会变大并推动 Policy 修正。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>两条隐式奖励</b>chosen [B] 与 rejected [B]</span>
          <span><b>逐样本相减</b>logits = chosen - rejected</span>
          <span><b>映射成把握度</b>sigmoid(logits)</span>
          <span><b>训练损失</b>-logsigmoid，仍保留 [B]</span>
        </div>

        <div class="adv-grid three">
          <section class="adv-panel good">
            <h4>logits &gt; 0</h4>
            <p>chosen 排在 rejected 前，损失小于 logits=0 时。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>logits = 0</h4>
            <p>两条回复无法区分，sigmoid=0.5，loss 约为 0.693。</p>
          </section>
          <section class="adv-panel warn">
            <h4>logits &lt; 0</h4>
            <p>偏好顺序颠倒，损失更大。</p>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>logits = chosen_rewards - rejected_rewards</code><span>[B]，每一项是一对回复的奖励间隔</span></div>
          <div><b>2</b><code>losses = -logsigmoid(logits)</code><span>[B]，每对偏好样本各保留一个损失</span></div>
          <div><b>3</b><code>return three vectors</code><span>losses、chosen_rewards、rejected_rewards 都是 [B]</span></div>
        </div>

        <div class="adv-contract">
          <span>首样本奖励</span><strong>chosen=0.1，rejected=-0.1</strong>
          <span>首样本 logits</span><strong>0.1 - (-0.1) = 0.2</strong>
          <span>首样本期望损失</span><strong>-F.logsigmoid(0.2)</strong>
          <span>函数内是否 mean</span><strong>否；测试要求 losses.shape == (4,)</strong>
        </div>

        <div class="adv-checks">
          <span>losses.shape == (4,)</span>
          <span>chosen_rewards.shape == (4,)</span>
          <span>rejected_rewards.shape == (4,)</span>
          <span>首样本三个数值通过 allclose</span>
        </div>

        <div class="adv-callout">训练循环可以在函数外对 losses.mean() 再 backward，但这个 TODO 的返回契约是逐样本损失。提前 mean 会让数值看似合理，却直接破坏 shape 断言。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把每位评审的 A/B 排序变成损失", [
          "score_a = torch.tensor([0.35, -0.10, 0.60])",
          "score_b = torch.tensor([-0.20, 0.25, 0.10])",
          "preference_margin = score_a - score_b",
          "per_reviewer_loss = -F.logsigmoid(preference_margin)",
          "print(per_reviewer_loss.shape)  # [3]",
          "",
          "# 真正训练时可由外部决定如何聚合",
          "training_loss = per_reviewer_loss.mean()"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>score_a / score_b</code> -&gt; <code>chosen_rewards / rejected_rewards</code>。</li>
            <li><code>preference_margin</code> -&gt; TODO 3 的 <code>logits</code>，相减顺序是 chosen 减 rejected。</li>
            <li><code>per_reviewer_loss</code> -&gt; 函数应返回的逐样本 <code>losses</code>。</li>
            <li>例子最后的外部 mean -&gt; Notebook 函数内不要照搬；它说明逐样本返回后，调用者仍可自行聚合。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "chosen_reward=0.4，rejected_reward=-0.2。",
        question: "若把二者差距继续拉大为 chosen=0.8、rejected=-0.4，DPO loss 会怎样？",
        options: ["变小，因为正的偏好 margin 更大", "变大，因为 reward 绝对值更大", "保持不变，因为 shape 没变"],
        answer: 0,
        revealNote: "margin 从 0.6 增加到 1.2，sigmoid 更接近 1，负对数因此变小。"
      },
      checkpoint: checkpoint(
        "为什么 dpo_loss 不能在返回前对 losses 调用 mean()？",
        ["Notebook 契约要求每个偏好对一个 loss，测试断言 shape 为 [batch_size]", "mean 会改变浮点 dtype 为整数", "F.logsigmoid 只能处理二维张量"],
        0,
        "本页返回逐样本向量；聚合可以由外部训练循环决定。"
      ),
      homework: [
        "完成 TODO 3：用 chosen_rewards 与 rejected_rewards 构造逐样本 logits，再用数值稳定的 F.logsigmoid 得到逐样本 losses；保留已有三返回值。",
        "shape/dtype/device 自检：logits、losses、chosen_rewards、rejected_rewards 都应为 [4] 浮点 Tensor，device 与四个输入一致；不要创建额外 CPU 常量张量。",
        "测试目标：运行 test_dpo_loss，先过三个 shape 断言，再过首样本 chosen=0.1、rejected=-0.1、loss=-F.logsigmoid(0.2) 的数值断言。",
        "常见错误排查：loss 偏大且排序反向查 logits 减法顺序；shape=[] 查是否提前 mean；数值不稳查是否手写 log(sigmoid)；奖励错位回到 TODO 1/2 检查同类相减。"
      ]
    })
  ]
};
