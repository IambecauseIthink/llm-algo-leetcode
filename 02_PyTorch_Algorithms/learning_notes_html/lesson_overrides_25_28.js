const { advancedStyles, checkpoint, code, lesson } = require("./advanced_lesson_helpers");

module.exports = {
  "25": [
    lesson({
      id: "w8a16-range-and-scale",
      title: "先定标尺：用 absmax 把浮点范围对齐到 INT8",
      todo: "TODO 1 / TODO 2：absmax、零值保护与 scale",
      prerequisite: [
        "张量中的每个权重原本是浮点数；量化不是直接改类型，而是先决定浮点数与整数之间的换算比例。",
        "本页采用 per-tensor 对称量化：整个 x 共享一个标量 scale，正负两侧都围绕 0。",
        "torch.abs(x) 逐元素取绝对值，torch.max(...) 再从所有元素中找出最大的幅度。",
        "如果 x 全为 0，absmax 也是 0；直接计算 127 / absmax 会产生 Inf，所以必须先做除零保护。"
      ],
      intuition: "把 INT8 的 -127 到 127 想成一把只有 255 个刻度的尺子。absmax 告诉我们原张量最远离 0 的位置，scale 则说明每 1 个浮点单位要放大成多少个整数刻度。先把尺子定好，后面才能舍入和存储。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>读范围</b>找到 x 中最大的绝对值 absmax</span>
          <span><b>防除零</b>全 0 时换成很小的正数</span>
          <span><b>定标尺</b>scale = 127 / absmax</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>变量先翻译成人话</h4>
            <div class="adv-contract">
              <span><code>x</code></span><strong>任意 shape 的浮点权重张量</strong>
              <span><code>absmax</code></span><strong>标量张量，表示最大绝对幅度</strong>
              <span><code>scale</code></span><strong>标量张量，表示浮点值到整数刻度的放大倍数</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>一个只算标尺的小例子</h4>
            <p>若权重是 [-2.0, 0.5, 1.0]，最大的绝对值是 2.0。</p>
            <div class="adv-flow">
              <span>abs(x)<br>[2.0, 0.5, 1.0]</span>
              <span>absmax<br>2.0</span>
              <strong>scale<br>127 / 2</strong>
            </div>
            <p>最大幅度 2.0 会被放大到 127；其他数按同一比例缩放，所以相对大小仍被保留。</p>
          </section>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>全 0 是测试里的明确边界</h4>
            <p>全 0 张量没有可用范围。把 absmax 临时替换成很小的正数后，scale 保持有限，0 乘任何有限 scale 仍是 0。</p>
          </section>
          <section class="adv-panel blue">
            <h4>为什么使用 127</h4>
            <p>Notebook 要做对称量化，让最大正幅度落在 127。虽然 int8 还能表示 -128，但这里的正负有效刻度围绕 0 对称。</p>
          </section>
        </div>

        <div class="adv-callout">先确认语义再写公式：scale 是“量化时乘上去”的放大倍数，因此后面的反量化要除以同一个 scale。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用温度读数练习 absmax、标量判断和安全 scale", [
          "readings = torch.tensor([-1.2, 0.0, 0.8])",
          "largest = torch.abs(readings).max()",
          "",
          "if largest == 0:",
          "    largest = torch.tensor(1e-8, device=readings.device)",
          "",
          "gain = 31.0 / largest  # 练习映射到另一套整数刻度",
          "print(largest.shape)   # torch.Size([])，标量张量"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>readings</code> -&gt; <code>x</code>：输入数据不同，但都先保留原 shape。</li>
            <li><code>largest</code> -&gt; <code>absmax</code>：先逐元素取绝对值，再做全局最大值归约。</li>
            <li><code>31.0</code> -&gt; Notebook 的 INT8 正上界 <code>127.0</code>。</li>
            <li><code>gain</code> -&gt; <code>scale</code>：都是“整数上界 / 最大绝对值”。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "输入 x 是一个全 0 张量，此时 torch.abs(x).max() 也是 0。",
        question: "如果不做保护就计算 127.0 / absmax，最可能发生什么？",
        options: ["scale 变成 Inf，测试中的有限性断言失败", "scale 自动变成 1", "x 自动改成 int8"],
        answer: 0,
        revealNote: "除以 0 不会得到可用标尺。Notebook 首个边界测试明确要求 zero_scale 是有限值。"
      },
      checkpoint: checkpoint(
        "对张量 [-0.8, 1.5, -3.0, 2.5, 0.0] 做本页对称量化时，absmax 应是多少？",
        ["3.0", "2.5", "0.0"],
        0,
        "先取绝对值得到 [0.8, 1.5, 3.0, 2.5, 0.0]，其中最大值是 3.0。"
      ),
      homework: [
        "完成 Notebook TODO 1 和 TODO 2：让 absmax 与 scale 都是标量，并保留与 x 兼容的 dtype/device 语义。",
        "先单独运行全 0 输入；检查量化结果未来仍能保持原 shape、最终 dtype 为 torch.int8，并确认 scale 不是 NaN/Inf。",
        "若有限性测试失败，优先打印 absmax 和 scale；常见错误是先做除法、后做零值保护，顺序反了。"
      ]
    }),

    lesson({
      id: "w8a16-round-clamp-cast",
      title: "再离散化：乘、舍入、截断、转 int8 一个都不能乱",
      todo: "TODO 3：x_scaled 与 x_quant",
      prerequisite: [
        "scale 只是浮点标尺；x * scale 的结果仍是浮点张量，还不能获得 INT8 的存储收益。",
        "torch.round 负责把连续值落到最近的整数刻度，但返回 dtype 通常仍是浮点类型。",
        "torch.clamp 把异常值限制到允许范围；最后的 .to(torch.int8) 才改变存储类型。",
        "这些操作都是逐元素的，因此 x_quant 的 shape 必须与 x 完全相同。"
      ],
      intuition: "量化像把连续的刻度读数抄到只能写整数的表格：先按 scale 放大，再四舍五入；为了不写出表格范围之外的数，再做截断；最后才把整张表换成 int8 存储。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>x_scaled = x * scale</code><span>浮点值对齐到 INT8 刻度</span></div>
          <div><b>2</b><code>rounded = round(x_scaled)</code><span>把连续值变成整数数值，但尚未改 dtype</span></div>
          <div><b>3</b><code>bounded = clamp(...)</code><span>保证数值落在 Notebook 指定区间</span></div>
          <div><b>4</b><code>x_quant = ...to(int8)</code><span>shape 不变，存储类型变成 1 字节整数</span></div>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>阶段</th><th>示例值</th><th>shape</th><th>dtype 关注点</th></tr></thead>
          <tbody>
            <tr><td>原值</td><td><code>2.5</code></td><td>与 x 相同</td><td>浮点</td></tr>
            <tr><td>乘 scale</td><td><code>105.833...</code></td><td>与 x 相同</td><td>仍是浮点</td></tr>
            <tr><td>round</td><td><code>106</code></td><td>与 x 相同</td><td>数值像整数，dtype 未必是 int8</td></tr>
            <tr><td>cast</td><td><code>106</code></td><td>与 x 相同</td><td><code>torch.int8</code></td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel warn">
            <h4>顺序错误会改变答案</h4>
            <p>若在 round 之前就转 int8，小数部分会直接丢失；若完全不 clamp，超范围转换可能产生错误的整数表示。</p>
          </section>
          <section class="adv-panel good">
            <h4>测试中的确定值</h4>
            <p>absmax=3.0 时，scale=127/3。样本 2.5 被放大为约 105.83，四舍五入后应为 106。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("把音频幅度映射到较小的整数区间", [
          "audio = torch.tensor([-1.0, -0.26, 0.24, 1.2])",
          "gain = 15.0 / audio.abs().max()",
          "scaled = audio * gain",
          "rounded = torch.round(scaled)",
          "codes = torch.clamp(rounded, -16, 15).to(torch.int8)",
          "",
          "assert codes.shape == audio.shape",
          "assert codes.dtype == torch.int8"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>audio</code> -&gt; <code>x</code>：都按元素量化，shape 不变。</li>
            <li><code>gain</code> -&gt; <code>scale</code>：Notebook 使用 127 对齐 INT8 范围。</li>
            <li><code>scaled</code> -&gt; <code>x_scaled</code>：保持浮点，交给 round。</li>
            <li><code>codes</code> -&gt; <code>x_quant</code>：Notebook 的 clamp 边界按 TODO 提示设置，并最终转成 torch.int8。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "torch.round(torch.tensor(105.8)) 的数值已经是 106。",
        question: "只调用 torch.round 后，是否已经保证结果 dtype 是 torch.int8？",
        options: ["没有，还需要显式转换为 torch.int8", "是，round 总会返回 int8", "是，但 shape 会变成标量"],
        answer: 0,
        revealNote: "round 改的是数值，不等于改存储类型。Notebook 会直接断言 x_q.dtype == torch.int8。"
      },
      checkpoint: checkpoint(
        "TODO 3 中哪一种顺序符合 Notebook 的量化链路？",
        ["乘 scale → round → clamp → 转 int8", "转 int8 → 乘 scale → round", "clamp → 除以 scale → 转 float"],
        0,
        "先在浮点域完成缩放、舍入和边界限制，最后再切换到紧凑的整数存储。"
      ),
      homework: [
        "完成 TODO 3，并逐行检查 x_scaled 与 x_quant 的 shape 都等于 x.shape、device 不变，最终 x_quant.dtype 必须是 torch.int8。",
        "运行带符号样本测试，确认 scale 等于 127/3，索引 3 的量化值等于 106；全 0 输入量化后非零元素数应为 0。",
        "若 106 断言失败，按“是否乘 scale、是否先 round、clamp 边界、是否过早 cast”四步排查。"
      ]
    }),

    lesson({
      id: "w8a16-dequant-linear",
      title: "最后做前向：INT8 负责存，输入 dtype 负责算",
      todo: "TODO 4：weight_int8 反量化与 F.linear",
      prerequisite: [
        "W8A16 中 W8 表示权重以 INT8 存储；本 Notebook 的激活 x 保持浮点，测试实际会传入 float32。",
        "nn.Linear 的权重 shape 是 [out_features, in_features]；F.linear 会用 x 的最后一维与 in_features 对齐。",
        "量化时做的是 x * scale，所以反量化必须先转回浮点，再除以同一个 scale。",
        "register_buffer 创建的 weight_int8 与 scale 不参与梯度更新，但会跟随模块移动 device；bias 仍是 Parameter。"
      ],
      intuition: "这个模拟层把权重长期放在紧凑的 INT8 仓库里。每次 forward 时，先按输入 x 的计算类型取出一份近似浮点权重，再调用普通线性层。这样展示的是“存储省内存、计算前恢复”的链路。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span><code>weight_int8</code><br>[out, in] / int8</span>
          <span>转成 x.dtype<br>[out, in]</span>
          <span>除以 scale<br>近似浮点权重</span>
          <strong>F.linear(x, w, bias)<br>[batch, seq, out]</strong>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>Notebook shape</th><th>dtype/device 契约</th></tr></thead>
          <tbody>
            <tr><td><code>x</code></td><td><code>[batch, seq, in_features]</code></td><td>激活的浮点 dtype；应与层位于同一 device</td></tr>
            <tr><td><code>weight_int8</code></td><td><code>[out_features, in_features]</code></td><td>存储时固定为 torch.int8</td></tr>
            <tr><td><code>w_dequant</code></td><td><code>[out_features, in_features]</code></td><td>计算时跟随 x.dtype</td></tr>
            <tr><td><code>out</code></td><td><code>[batch, seq, out_features]</code></td><td>浮点输出</td></tr>
          </tbody>
        </table>

        <div class="adv-grid three">
          <section class="adv-panel good">
            <h4>存储测试</h4>
            <p>同样数量的 FP32 权重每个 4 字节，INT8 每个 1 字节，所以 weight_int8 字节数应为四分之一。</p>
          </section>
          <section class="adv-panel blue">
            <h4>近似测试</h4>
            <p>随机层比较原输出与量化层输出的余弦相似度，要求大于 0.99，而不是逐元素完全相等。</p>
          </section>
          <section class="adv-panel neutral">
            <h4>公式测试</h4>
            <p>确定性小矩阵会用同一份反量化权重调用 F.linear，要求你的 forward 与参考结果在 1e-6 内一致。</p>
          </section>
        </div>

        <div class="adv-callout">不要把 scale 再乘回去：本 Notebook 保存的是量化时的放大倍数，恢复数值范围使用除法。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用压缩库存权重练习恢复后做线性计算", [
          "packed_weight = torch.tensor([[6, -3, 9], [2, 4, -8]], dtype=torch.int8)",
          "restore_factor = torch.tensor(3.0)",
          "features = torch.randn(5, 3, dtype=torch.float32)",
          "offset = torch.zeros(2, dtype=features.dtype)",
          "",
          "weight_for_compute = packed_weight.to(features.dtype) / restore_factor",
          "scores = F.linear(features, weight_for_compute, offset)",
          "assert scores.shape == (5, 2)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>packed_weight</code> -&gt; <code>self.weight_int8</code>：都是 [out, in] 的整数存储。</li>
            <li><code>features.dtype</code> -&gt; <code>x.dtype</code>：反量化权重先跟随输入的计算类型。</li>
            <li><code>restore_factor</code> -&gt; <code>self.scale</code>：恢复时使用除法。</li>
            <li><code>scores</code> -&gt; <code>out</code>：使用 F.linear，并带上 Notebook 的 <code>self.bias</code>。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "weight_int8 的 shape 是 [64, 128]，x 的 shape 是 [2, 10, 128]。",
        question: "F.linear(x, w_dequant, bias) 的输出 shape 是什么？",
        options: ["[2, 10, 64]", "[64, 128]", "[2, 128, 10]"],
        answer: 0,
        revealNote: "F.linear 只替换输入的最后一维：in_features=128 被 out_features=64 替换，前面的 batch 与 seq 保留。"
      },
      checkpoint: checkpoint(
        "量化时使用 x_quant ≈ round(x * scale)，反量化权重应采用哪种操作？",
        ["weight_int8.to(x.dtype) / scale", "weight_int8 * scale 后保持 int8", "只调用 weight_int8.long()"],
        0,
        "先回到输入的浮点计算类型，再除以量化放大倍数，才能近似恢复原权重范围。"
      ),
      homework: [
        "完成 TODO 4：w_fp、w_dequant 和 out 的 shape 依次保持 [out_features, in_features]、[out_features, in_features]、[..., out_features]。",
        "检查 dtype/device：w_fp 跟随 x.dtype，weight_int8 与 scale 作为 buffer 应和模块同 device，x 与层也必须位于同一 device。",
        "运行全部测试；若小矩阵失败先核对除法与 bias，若余弦低再核对量化顺序，若内存断言失败则确认 weight_int8 真的是 torch.int8。"
      ]
    })
  ],

  "26": [
    lesson({
      id: "qlora-nf4-lookup",
      title: "先读懂 NF4：存的是 0–15 索引，查表后才是权重",
      todo: "TODO 1：indices 与 dequantized_base_weight",
      prerequisite: [
        "本 Notebook 用 torch.int8 保存 0 到 15 的索引来模拟 4-bit；它没有真的把两个 4-bit 值打包进一个字节。",
        "nf4_table 是长度为 16 的浮点 buffer，位置 0 到 15 分别存放一个 NF4 代表值。",
        "PyTorch 张量索引需要整数索引类型；把 int8 索引转换为 long 后才能稳定地用于查表。",
        "weight_scale 是标量 buffer。查表结果乘以它，会恢复基础权重的整体数值范围。"
      ],
      intuition: "NF4 索引像一本字典的页码：索引 3 本身不是权重 -0.395，它只是告诉我们去 nf4_table 的第 3 个位置取值。整张索引矩阵查表后 shape 不变，但元素从整数编号变成浮点代表值，再乘 scale 才得到用于计算的基础权重。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>索引存储</b>weight_nf4_indices<br>[out, in] / int8</span>
          <span><b>转索引类型</b>indices.long()<br>[out, in] / int64</span>
          <span><b>查表</b>nf4_table[indices]<br>[out, in] / float</span>
          <span><b>恢复范围</b>查表值 × weight_scale</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>二维索引会得到二维权重</h4>
            <p>若一行索引是 [0, 7, 15]，查表会逐位置得到 [-1.0, 0.0, 1.0]。索引矩阵有多少行列，查表结果就有多少行列。</p>
            <div class="adv-flow">
              <span>[0, 7, 15]</span>
              <span>查长度 16 的表</span>
              <strong>[-1.0, 0.0, 1.0]</strong>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>三个 buffer 的职责</h4>
            <div class="adv-contract">
              <span><code>weight_nf4_indices</code></span><strong>冻结的离散编号</strong>
              <span><code>nf4_table</code></span><strong>编号到浮点代表值的映射</strong>
              <span><code>weight_scale</code></span><strong>把标准化代表值放回权重范围</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>变量</th><th>shape</th><th>dtype</th><th>是否训练参数</th></tr></thead>
          <tbody>
            <tr><td><code>weight_nf4_indices</code></td><td><code>[out_features, in_features]</code></td><td>torch.int8</td><td>否，buffer</td></tr>
            <tr><td><code>indices</code></td><td><code>[out_features, in_features]</code></td><td>torch.int64</td><td>否，临时索引</td></tr>
            <tr><td><code>dequantized_base_weight</code></td><td><code>[out_features, in_features]</code></td><td>浮点</td><td>由冻结 buffer 计算</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">这里的“4-bit”是教学模拟的语义，不要误以为 torch.int8 已经自动完成两个索引打包；TODO 只要求正确演示查表反量化。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用颜色编号练习张量查表", [
          "palette = torch.tensor([-0.8, -0.1, 0.4, 0.9])",
          "pixel_codes = torch.tensor([[0, 2], [3, 1]], dtype=torch.int8)",
          "brightness = torch.tensor(0.75)",
          "",
          "lookup_indices = pixel_codes.long()",
          "pixels = palette[lookup_indices] * brightness",
          "assert pixels.shape == pixel_codes.shape"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>pixel_codes</code> -&gt; <code>self.weight_nf4_indices</code>：存的都是查表编号。</li>
            <li><code>lookup_indices</code> -&gt; <code>indices</code>：调用 long 得到合法索引 dtype。</li>
            <li><code>palette</code> -&gt; <code>self.nf4_table</code>：Notebook 的表有 16 个代表值。</li>
            <li><code>brightness</code> -&gt; <code>self.weight_scale</code>：查表后按标量恢复范围。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "weight_nf4_indices 的 shape 是 [3, 4]，nf4_table 的 shape 是 [16]。",
        question: "执行 nf4_table[indices] 后，结果 shape 是什么？",
        options: ["[3, 4]", "[16, 3, 4]", "[12]"],
        answer: 0,
        revealNote: "每个索引位置被替换成一个表值，所以输出继承索引张量的 [3, 4] 布局。"
      },
      checkpoint: checkpoint(
        "为什么 TODO 1 要先把 weight_nf4_indices 转成 long？",
        ["long 是 PyTorch 查表所需的整数索引类型", "long 会让索引开始计算梯度", "long 会把 16 个值压成 4-bit"],
        0,
        "转换的目的只是得到合法索引 dtype；它不会让 buffer 变成可训练参数，也不负责真实位打包。"
      ),
      homework: [
        "完成 TODO 1：检查 indices 与 weight_nf4_indices shape 相同，dequantized_base_weight 为 [out_features, in_features] 的浮点 dtype 张量。",
        "确认 weight_nf4_indices、nf4_table、weight_scale 都是 buffer，随层移动 device，但 requires_grad 应保持 False。",
        "若查表报索引类型错误，检查是否调用 long；若数值测试失败，检查 scale 是乘在查表结果上，而不是乘在索引编号上。"
      ]
    }),

    lesson({
      id: "qlora-two-branches",
      title: "再走双分支：基础层给底座，LoRA 用低秩增量纠偏",
      todo: "TODO 2：base_out、lora_out 与相加返回",
      prerequisite: [
        "输入 x 的 shape 是 [batch, seq, in_features]；矩阵乘法总是让最后一维与右侧矩阵倒数第二维对齐。",
        "lora_A 的 shape 是 [r, in_features]，先把 in_features 压到较小的秩 r。",
        "lora_B 的 shape 是 [out_features, r]，再把 r 展开到 out_features。",
        "scaling = alpha / r 是一个标量，用来控制 LoRA 增量相对基础分支的强度。"
      ],
      intuition: "QLoRA 的输出由两条路汇合。主路使用冻结后再反量化的基础权重，保留原模型能力；旁路只训练两个小矩阵 A 和 B，先降维、再升维，产生一个同 shape 的修正量。两条路终点相同，才能逐元素相加。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>基础分支</h4>
            <div class="adv-flow">
              <span>x<br>[B, S, in]</span>
              <span>反量化权重<br>[out, in]</span>
              <strong>base_out<br>[B, S, out]</strong>
            </div>
            <p>F.linear 把 x 的最后一维 in_features 替换成 out_features。</p>
          </section>
          <section class="adv-panel good">
            <h4>LoRA 旁路</h4>
            <div class="adv-flow">
              <span>x<br>[B, S, in]</span>
              <span>@ A.T<br>[B, S, r]</span>
              <span>@ B.T<br>[B, S, out]</span>
              <strong>× scaling</strong>
            </div>
            <p>A 做低秩压缩，B 再还原输出宽度；旁路输出与基础输出 shape 相同。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>运算</th><th>左侧 shape</th><th>右侧 shape</th><th>结果 shape</th></tr></thead>
          <tbody>
            <tr><td>基础线性层</td><td><code>[B,S,in]</code></td><td><code>[out,in]</code></td><td><code>[B,S,out]</code></td></tr>
            <tr><td>LoRA 第一步</td><td><code>[B,S,in]</code></td><td><code>A.T [in,r]</code></td><td><code>[B,S,r]</code></td></tr>
            <tr><td>LoRA 第二步</td><td><code>[B,S,r]</code></td><td><code>B.T [r,out]</code></td><td><code>[B,S,out]</code></td></tr>
          </tbody>
        </table>

        <div class="adv-contract">
          <span>Notebook 测试配置</span><strong>B=1，S=2，in=4，r=2，out=3</strong>
          <span>两条分支输出</span><strong>都必须是 [1, 2, 3]</strong>
          <span>最终输出</span><strong>base_out + lora_out，shape 仍为 [1, 2, 3]</strong>
        </div>

        <div class="adv-callout">转置不可凭感觉省略：A 存成 [r,in]、B 存成 [out,r]，而 x 需要依次右乘 [in,r] 和 [r,out]。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用推荐系统的小适配器练双分支 shape", [
          "users = torch.randn(2, 5, 6)  # [batch, items, in=6]",
          "base_weight = torch.randn(4, 6)",
          "adapter_down = torch.randn(3, 6)  # [r=3, in=6]",
          "adapter_up = torch.randn(4, 3)    # [out=4, r=3]",
          "strength = 2.0",
          "",
          "base_scores = F.linear(users, base_weight)",
          "hidden = users @ adapter_down.T",
          "delta_scores = (hidden @ adapter_up.T) * strength",
          "combined = base_scores + delta_scores"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>users</code> -&gt; <code>x</code>；前导维度可以不同，最后一维必须是 in_features。</li>
            <li><code>base_weight</code> -&gt; <code>dequantized_base_weight</code>，交给 F.linear 得到基础分支。</li>
            <li><code>adapter_down/up</code> -&gt; <code>self.lora_A/B</code>，按 A.T 后 B.T 的顺序相乘。</li>
            <li><code>strength</code> -&gt; <code>self.scaling</code>，乘在 LoRA 分支上再与 base_out 相加。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "测试中 x=[1,2,4]、lora_A=[2,4]、lora_B=[3,2]。",
        question: "执行 x @ lora_A.T 后，中间张量的 shape 是什么？",
        options: ["[1, 2, 2]", "[1, 2, 3]", "[2, 4]"],
        answer: 0,
        revealNote: "A.T 的 shape 是 [4,2]，所以 x 的最后一维 4 被低秩维度 r=2 替换。"
      },
      checkpoint: checkpoint(
        "base_out 与 lora_out 可以直接相加的关键条件是什么？",
        ["二者 shape 都是 [batch, seq, out_features]", "二者都必须是 int8", "lora_A 与 lora_B 必须完全同 shape"],
        0,
        "基础分支和旁路都落到 out_features，才表示对同一输出位置的基础值与增量。"
      ),
      homework: [
        "完成 TODO 2，先在注释中写出 base_out、x @ A.T、lora_out 的 shape，再组合返回值。",
        "保持 dtype/device 一致：x、反量化权重、lora_A、lora_B 应能在同一设备和浮点计算类型中参与运算；不要把 LoRA 参数转成整数。",
        "运行 Notebook 数值测试；若输出 shape 错误，逐个打印转置后的 A/B shape；若数值不一致，检查 scaling 是否只乘在 LoRA 分支，以及最终是否真的把两条分支相加。"
      ]
    }),

    lesson({
      id: "qlora-gradient-contract",
      title: "最后看梯度：底座冻结，学习能力留在 A 和 B",
      todo: "测试契约：数值对齐、旁路生效与 backward 梯度流",
      prerequisite: [
        "nn.Parameter 默认 requires_grad=True；register_buffer 默认不是可训练参数，也不会由优化器更新。",
        "只要前向结果由 x、lora_A、lora_B 的可微运算得到，out.sum().backward() 就能沿旁路回传梯度。",
        "整数索引不能承担普通浮点梯度；本页恰好希望 weight_nf4_indices 始终冻结。",
        "测试会给 lora_B 写入非零值，因此 lora_A 也应收到非空梯度；不要用 detach 或 no_grad 包住 LoRA 计算。"
      ],
      intuition: "冻结不是让基础分支消失，而是让它只参与前向、不给它保存可训练梯度。反向传播遇到两条分支时，会继续沿 x 和 LoRA 参数构成的可微路径传播；查表底座来自 buffer，所以停在那里。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-flow">
          <span>out.sum()</span>
          <span>base_out + lora_out</span>
          <strong>x.grad<br>存在</strong>
          <strong>lora_A.grad<br>存在</strong>
          <strong>lora_B.grad<br>存在</strong>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>应该获得梯度</h4>
            <div class="adv-checks">
              <span><code>x</code><br>测试显式设置 requires_grad=True</span>
              <span><code>lora_A</code><br>nn.Parameter</span>
              <span><code>lora_B</code><br>nn.Parameter</span>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>应该保持冻结</h4>
            <div class="adv-checks">
              <span><code>weight_nf4_indices</code><br>requires_grad=False，grad=None</span>
              <span><code>weight_scale</code><br>buffer，grad=None</span>
            </div>
          </section>
        </div>

        <div class="adv-steps">
          <div><b>1</b><code>out.shape == (1, 2, 3)</code><span>先守住接口 shape</span></div>
          <div><b>2</b><code>allclose(out, out_ref)</code><span>查表、scale、基础分支、旁路、scaling 全部对齐</span></div>
          <div><b>3</b><code>out != base_out_ref</code><span>LoRA 旁路不能被漏掉</span></div>
          <div><b>4</b><code>out.sum().backward()</code><span>检查梯度到达可训练端，冻结端没有 grad</span></div>
        </div>

        <div class="adv-callout">grad 不是 None 只说明计算图连通；本测试还先做精确数值对照，能防止“梯度能跑但公式写错”的假通过。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用冻结查表与可训练偏移练习梯度边界", [
          "table = torch.tensor([-1.0, 0.0, 1.0])  # 普通张量，不训练",
          "codes = torch.tensor([0, 2], dtype=torch.long)",
          "gain = nn.Parameter(torch.tensor([0.2, -0.3]))",
          "inputs = torch.tensor([2.0, 4.0], requires_grad=True)",
          "",
          "base = table[codes] * inputs",
          "result = base + gain * inputs",
          "result.sum().backward()",
          "",
          "assert inputs.grad is not None",
          "assert gain.grad is not None",
          "assert table.grad is None"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>table/codes</code> -&gt; <code>nf4_table/weight_nf4_indices</code>：提供冻结基础值。</li>
            <li><code>gain</code> -&gt; <code>lora_A、lora_B</code>：都是需要获得梯度的 Parameter。</li>
            <li><code>inputs</code> -&gt; <code>x</code>：测试要求输入梯度存在。</li>
            <li><code>result.sum().backward()</code> -&gt; Notebook 的梯度断点测试；前向中不要人为 detach LoRA 路径。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "weight_nf4_indices 通过 register_buffer 注册，lora_A 通过 nn.Parameter 注册。",
        question: "调用 backward 后，哪组预期符合 Notebook？",
        options: ["lora_A.grad 存在，weight_nf4_indices.grad 为 None", "两者 grad 都存在", "两者 grad 都必须为 None"],
        answer: 0,
        revealNote: "QLoRA 的核心训练边界就是冻结量化底座，只让高精度适配器学习。"
      },
      checkpoint: checkpoint(
        "测试为什么还要断言 out 不能与 base_out_ref 完全相同？",
        ["确认 LoRA 旁路确实参与了最终输出", "确认索引已经真实压成 4-bit", "确认所有 buffer 都获得梯度"],
        0,
        "只算基础分支也可能得到正确 shape；这个断言专门防止遗漏 lora_out。"
      ),
      homework: [
        "完成两个 TODO 后运行完整测试，确认输出 shape 为 [batch, seq, out_dim]，x 与层内浮点参数保持兼容的 dtype/device，且与测试构造的 out_ref 在 atol=1e-5 内一致。",
        "检查梯度契约：x、lora_A、lora_B 的 grad 非 None；weight_nf4_indices.requires_grad=False，indices、weight_scale 都不应积累 grad。",
        "若梯度失败，排查是否在 forward 使用 torch.no_grad、detach、.data 或重新 torch.tensor(...) 切断 LoRA 路径；若只退化成 base_out，检查返回表达式是否遗漏 lora_out。"
      ]
    })
  ],

  "27": [
    lesson({
      id: "zero1-parameter-partitions",
      title: "先分清对象：两张逻辑 GPU 各负责一半 Parameter",
      todo: "TODO 1：收集参数、计算 half_idx、建立 gpu_partitions",
      prerequisite: [
        "model.parameters() 返回一个可迭代对象；先转成 list，后面才能按下标切片并反复访问。",
        "当前 SimpleModel 有 fc1.weight 和 fc2.weight 两个 Parameter，因为两个 Linear 都设置了 bias=False。",
        "Notebook 文字用 flatten 帮助理解，但当前代码并没有把每个 Parameter 的元素拼成大向量；它实际切分的是 Parameter 对象列表。",
        "列表切片保留原 Parameter 的引用，不会复制权重。后面通过这些引用更新时，SimpleModel 里的权重也会改变。"
      ],
      intuition: "把优化器看成一个任务分派员。它先把模型参数按出现顺序排成列表，再从中点切成前后两组：逻辑 GPU 0 负责前半，逻辑 GPU 1 负责后半。这里没有真实 CUDA 通信，字典只是模拟每张卡的责任范围。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>收集</b>model_params -&gt; self.params 列表</span>
          <span><b>找中点</b>half_idx = 参数对象数 // 2</span>
          <span><b>切前半</b>逻辑 GPU 0 的参数引用</span>
          <span><b>切后半</b>逻辑 GPU 1 的参数引用</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>当前模型到底有几个参数对象</h4>
            <div class="adv-contract">
              <span><code>fc1.weight</code></span><strong>一个 Parameter，测试中 shape=[4,4]</strong>
              <span><code>fc2.weight</code></span><strong>一个 Parameter，测试中 shape=[4,4]</strong>
              <span><code>bias</code></span><strong>不存在，因为 bias=False</strong>
              <span><code>len(self.params)</code></span><strong>2，而不是 32 个标量</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>切分后的所有权</h4>
            <div class="adv-flow">
              <span>参数列表<br>[fc1.weight, fc2.weight]</span>
              <strong>GPU 0<br>[fc1.weight]</strong>
              <strong>GPU 1<br>[fc2.weight]</strong>
            </div>
            <p>测试正是按这个顺序提供梯度：键 0 对应 fc1，键 1 对应 fc2。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>结构</th><th>外层类型</th><th>内容</th><th>是否复制 Tensor</th></tr></thead>
          <tbody>
            <tr><td><code>self.params</code></td><td>list</td><td>全部 Parameter 引用</td><td>否</td></tr>
            <tr><td><code>self.gpu_partitions</code></td><td>dict</td><td>gpu_id -&gt; Parameter 列表</td><td>否</td></tr>
            <tr><td><code>self.gpu_partitions[0]</code></td><td>list</td><td>当前测试中只有 fc1.weight</td><td>否</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">不要对每个 [4,4] 权重张量沿元素维度切半；可见测试希望每张逻辑 GPU 恰好负责一个完整 Parameter。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用待处理任务列表练习中点切分", [
          "tasks = ['embed', 'attention', 'mlp', 'head']",
          "middle = len(tasks) // 2",
          "owners = {",
          "    0: tasks[:middle],",
          "    1: tasks[middle:],",
          "}",
          "",
          "assert owners[0] == ['embed', 'attention']",
          "assert owners[1] == ['mlp', 'head']"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>tasks</code> -&gt; <code>self.params</code>：Notebook 先用 list 收集 Parameter 引用。</li>
            <li><code>middle</code> -&gt; <code>half_idx</code>：都使用列表长度的整除中点。</li>
            <li><code>owners</code> -&gt; <code>self.gpu_partitions</code>：键是 gpu_id，值是它负责的列表切片。</li>
            <li>字符串任务没有 shape；Notebook 列表元素是 Parameter，各自保留原 shape、dtype、device。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "SimpleModel(dim=4) 有两个无 bias 的 Linear 层。",
        question: "list(model.parameters()) 中有多少个 Parameter 对象？",
        options: ["2 个", "32 个", "4 个"],
        answer: 0,
        revealNote: "每个 [4,4] weight 是一个 Parameter 对象。Notebook 当前按对象切分，不按 16 个内部元素切分。"
      },
      checkpoint: checkpoint(
        "为什么 gpu_partitions 中应保存原 Parameter 引用？",
        ["局部更新才能直接反映到原 SimpleModel 权重", "为了把权重自动转成字符串", "为了让每个 GPU 保存完整的重复状态"],
        0,
        "列表切片只创建新的列表容器，里面仍指向原参数；这让模拟更新无需真正 All-Gather 也能被模型看到。"
      ),
      homework: [
        "完成 TODO 1，确认 self.params 长度为 2，gpu_partitions 的键为 0/1，两个值各含一个 shape=[4,4] 的 Parameter 引用。",
        "不要在这里改 dtype/device 或 clone 参数；分区只描述所有权，参数仍保持模型原有 dtype 与 device。",
        "先打印每个分区的参数数量和 id；若后续状态数量断言不是 1，排查是否错误地按 Tensor 元素切分，或忘记模型没有 bias。"
      ]
    }),

    lesson({
      id: "zero1-local-states",
      title: "再建局部账本：每张卡只给自己的参数保存动量",
      todo: "TODO 2：optimizer_states 的两层字典与 zeros_like",
      prerequisite: [
        "优化器状态与参数不是同一个东西：Parameter 保存权重，状态 Tensor 保存历史动量。",
        "外层字典按 gpu_id 找到某张逻辑卡，内层字典按 id(p) 找到某个参数的状态。",
        "id(p) 是当前 Python 进程中对象身份的整数标识；同一个 Parameter 引用在分区和更新阶段会得到同一个 id。",
        "torch.zeros_like(p.data) 会复制 p 的 shape、dtype 和 device，但把数值全部初始化为 0。"
      ],
      intuition: "分区只是写下“谁负责谁”，局部状态才是真正省重复存储的地方。GPU 0 不为 fc2 建动量，GPU 1 也不为 fc1 建动量。每个状态必须长得和对应参数一样，后面才能与梯度逐元素相加。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid two">
          <section class="adv-panel blue">
            <h4>分区告诉我们该为谁建状态</h4>
            <div class="adv-flow">
              <span>GPU 0<br>fc1.weight</span>
              <strong>一个 [4,4] 零动量</strong>
            </div>
            <div class="adv-flow">
              <span>GPU 1<br>fc2.weight</span>
              <strong>一个 [4,4] 零动量</strong>
            </div>
          </section>
          <section class="adv-panel neutral">
            <h4>两层字典的读取路径</h4>
            <div class="adv-contract">
              <span>第一层</span><strong><code>optimizer_states[gpu_id]</code> 找到本卡状态表</strong>
              <span>第二层</span><strong><code>states[id(p)]</code> 找到参数 p 的动量</strong>
              <span>状态值</span><strong>与 p 同 shape/dtype/device 的 Tensor</strong>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>测试对象</th><th>状态条目数</th><th>状态 shape</th><th>初始数值</th></tr></thead>
          <tbody>
            <tr><td>GPU 0 / fc1.weight</td><td>1</td><td><code>[4,4]</code></td><td>全 0</td></tr>
            <tr><td>GPU 1 / fc2.weight</td><td>1</td><td><code>[4,4]</code></td><td>全 0</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>为什么用 zeros_like</h4>
            <p>它一次满足数值、shape、dtype、device 四项初始化契约，梯度到来时可以直接逐元素相加。</p>
          </section>
          <section class="adv-panel warn">
            <h4>为什么不用同一个零 Tensor</h4>
            <p>每个 Parameter 必须拥有独立历史。共享同一个状态会让一个参数的梯度污染另一个参数。</p>
          </section>
        </div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("为两个独立计数器建立按对象身份索引的状态", [
          "counter_a = torch.randn(2, 3)",
          "counter_b = torch.randn(4)",
          "assigned = {0: [counter_a], 1: [counter_b]}",
          "",
          "history = {}",
          "for worker_id, tensors in assigned.items():",
          "    history[worker_id] = {",
          "        id(t): torch.zeros_like(t) for t in tensors",
          "    }",
          "",
          "assert history[0][id(counter_a)].shape == counter_a.shape"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>assigned</code> -&gt; <code>self.gpu_partitions</code>：外层键表示负责者。</li>
            <li><code>history</code> -&gt; <code>self.optimizer_states</code>：每个负责者拥有自己的内层状态表。</li>
            <li><code>id(t)</code> -&gt; <code>id(p)</code>：用原对象身份取回对应状态。</li>
            <li><code>zeros_like(t)</code> -&gt; <code>zeros_like(p.data)</code>：状态继承参数的 shape、dtype、device。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "fc1.weight 在 CPU 上是 float32，shape=[4,4]。",
        question: "torch.zeros_like(fc1.weight.data) 会得到什么？",
        options: ["CPU float32、shape=[4,4] 的全 0 Tensor", "标量整数 0", "GPU int8、shape=[1] 的 Tensor"],
        answer: 0,
        revealNote: "zeros_like 的价值正是继承参照 Tensor 的 shape、dtype 与 device，只把内容置零。"
      },
      checkpoint: checkpoint(
        "当前测试为什么要求 len(optimizer.optimizer_states[0]) == 1？",
        ["GPU 0 只负责 fc1.weight，因此只保存一个局部状态", "GPU 0 必须复制 GPU 1 的所有状态", "字典只能存一个键"],
        0,
        "ZeRO-1 的核心节省来自不重复保存别的分区的优化器状态。当前每个分区恰好有一个 Parameter。"
      ),
      homework: [
        "完成 TODO 2：为每个分区参数建立独立零状态；内层键使用 id(p)，值与 p.data 的 shape/dtype/device 一致。",
        "运行测试前检查两个外层分区的状态长度都等于 1，并确认状态全 0、参数本身未被改动。",
        "若 KeyError，打印建立状态和读取状态时的 id(p)；常见错误是 clone 参数后对象身份改变，或把 p 与 id(p) 混作不同键类型。"
      ]
    }),

    lesson({
      id: "zero1-local-update",
      title: "最后更新原权重：对齐 params、grads、states 三条列表",
      todo: "TODO 3：局部动量累加与参数更新",
      prerequisite: [
        "gradients_from_all_gpus 已经模拟 Reduce-Scatter 的结果：每个 gpu_id 只拿到自己负责参数的平均梯度列表。",
        "同一分区中 params 与 grads 的顺序必须一一对应；zip(params, grads) 正好按位置配对。",
        "本页的 momentum 是教学简化版，只做 old_momentum + gradient，没有 beta 系数，也没有 Adam 的二阶状态。",
        "更新公式是新权重 = 旧权重 - lr * momentum；减号表示沿降低损失的方向移动。"
      ],
      intuition: "每张逻辑卡打开自己的三样东西：负责的参数、收到的梯度、保存的动量。先把新梯度记进动量，再用学习率缩小这一步，最后从原权重减掉。因为 params 保存的是原模型引用，改完后测试直接读取 model.fc1.weight 就能看到变化。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>按 gpu_id</b>取本卡 params</span>
          <span><b>同一个键</b>取本卡 grads 与 states</span>
          <span><b>逐对 zip</b>参数 p 对齐梯度 g</span>
          <span><b>先状态后参数</b>momentum += g，p -= lr × momentum</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>GPU 0 的第一步</h4>
            <div class="adv-steps">
              <div><b>1</b><code>momentum: 0 + 1 = 1</code><span>梯度是全 1 Tensor</span></div>
              <div><b>2</b><code>update: 0.1 × 1 = 0.1</code><span>每个权重元素都减 0.1</span></div>
              <div><b>3</b><code>initial_w1 - new_w1 = 0.1</code><span>测试比较的正是这个差值</span></div>
            </div>
          </section>
          <section class="adv-panel blue">
            <h4>GPU 1 的第一步</h4>
            <div class="adv-steps">
              <div><b>1</b><code>momentum: 0 + 2 = 2</code><span>梯度是全 2 Tensor</span></div>
              <div><b>2</b><code>update: 0.1 × 2 = 0.2</code><span>每个权重元素都减 0.2</span></div>
              <div><b>3</b><code>initial_w2 - new_w2 = 0.2</code><span>与第二条 allclose 断言对齐</span></div>
            </div>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>配对项</th><th>当前测试 shape</th><th>dtype/device</th><th>作用</th></tr></thead>
          <tbody>
            <tr><td><code>p</code></td><td><code>[4,4]</code></td><td>模型参数原属性</td><td>被更新的原权重</td></tr>
            <tr><td><code>g</code></td><td><code>[4,4]</code></td><td>由 ones_like/full_like 对齐</td><td>Reduce-Scatter 模拟梯度</td></tr>
            <tr><td><code>momentum</code></td><td><code>[4,4]</code></td><td>由 zeros_like 对齐</td><td>累积历史并参与更新</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">states[id(p)] 必须写回新的 momentum。只更新局部变量而不写回字典，会让第二次 step 忘记历史，破坏“状态”的含义。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用两个货架的库存与累计修正量练局部更新", [
          "shelves = {0: [torch.tensor([5.0, 7.0])], 1: [torch.tensor([9.0])]}",
          "corrections = {0: [torch.tensor([1.0, 2.0])], 1: [torch.tensor([3.0])]}",
          "memory = {worker: {id(item): torch.zeros_like(item) for item in items}",
          "          for worker, items in shelves.items()}",
          "rate = 0.25",
          "",
          "for worker in (0, 1):",
          "    local_memory = memory[worker]",
          "    for item, change in zip(shelves[worker], corrections[worker]):",
          "        accumulated = local_memory[id(item)] + change",
          "        local_memory[id(item)] = accumulated",
          "        item.sub_(rate * accumulated)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>shelves[worker]</code> -&gt; <code>params</code>，<code>corrections[worker]</code> -&gt; <code>grads</code>。</li>
            <li><code>local_memory</code> -&gt; <code>states</code>，通过 id(p) 读取并写回 momentum。</li>
            <li><code>zip(item, change)</code> -&gt; <code>zip(params, grads)</code>：顺序必须一致，shape 才逐项匹配。</li>
            <li><code>rate</code> -&gt; <code>self.lr</code>；Notebook 使用 Parameter 的 data 更新原模型引用。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "初始 momentum 为 0，梯度每个元素都是 2.0，学习率是 0.1。",
        question: "第一次 step 后，旧权重减新权重的每个元素是多少？",
        options: ["0.2", "2.0", "-0.2"],
        answer: 0,
        revealNote: "先得 momentum=2，再从权重减去 0.1×2，所以 initial - new = 0.2。"
      },
      checkpoint: checkpoint(
        "为什么 TODO 3 要同时执行 states[id(p)] = momentum？",
        ["保存本次累积结果，供后续 step 继续使用", "把参数 dtype 改成 id 类型", "把梯度广播给所有 GPU"],
        0,
        "momentum 是跨 step 的优化器状态。只改变局部变量，字典中的历史仍会停留在旧值。"
      ),
      homework: [
        "完成 TODO 3：每个 gpu_id 分别取 params、grads、states；每对 p/g 的 shape、dtype、device 应兼容，然后先更新状态、再更新 p.data。",
        "运行可见测试，确认 diff_w1 全为 0.1、diff_w2 全为 0.2；这也证明分区里保存的是原模型参数引用。",
        "常见错误排查：若差值符号相反，检查更新公式的减号；若只有一层变化，检查 gpu 循环和梯度字典键；若第二次手工 step 不累积，检查是否写回 states[id(p)]。"
      ]
    })
  ],

  "28": [
    lesson({
      id: "pipeline-diagonal-timeline",
      title: "先画时间轴：stage 越靠后，同一 micro-batch 越晚到达",
      todo: "TODO 1：total_steps、t/stage 循环与 timeline",
      prerequisite: [
        "p 表示 Pipeline stage 数量，m 表示 micro-batch 数量；编号都从 0 开始。",
        "timeline 是 Python 列表；timeline[t] 又是一个列表，保存该时间步活跃的 (stage, micro_idx) 元组。",
        "micro-batch 每经过一个 stage 就晚一个时间步，因此在时间 t、阶段 stage 上，对应编号是 micro_idx = t - stage。",
        "只有 0 &lt;= micro_idx &lt; m 时该任务真实存在；负编号表示还没到达，编号达到 m 表示所有 micro-batch 已经过完。"
      ],
      intuition: "把 micro-batch 想成依次进入流水线的小批任务。第 0 个任务在 t=0 进入 stage 0，t=1 到 stage 1；与此同时第 1 个任务进入 stage 0。于是活跃项沿着“时间向右、stage 向下”的对角线移动。",
      styles: advancedStyles,
      exampleHtml: `<div class="adv-course">
        <div class="adv-roadmap">
          <span><b>入口</b>每个时间步可有一个新 micro-batch 进入 stage 0</span>
          <span><b>传播</b>同一 micro-batch 每过一步进入下一个 stage</span>
          <span><b>边界</b>只记录 0 &lt;= micro_idx &lt; m</span>
          <span><b>结束</b>最后一个 micro-batch 离开最后一个 stage</span>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>时间 t</th><th>p=3、m=4 时的 active 列表</th><th>活跃槽位数</th></tr></thead>
          <tbody>
            <tr><td>0</td><td><code>[(0,0)]</code></td><td>1</td></tr>
            <tr><td>1</td><td><code>[(0,1),(1,0)]</code></td><td>2</td></tr>
            <tr><td>2</td><td><code>[(0,2),(1,1),(2,0)]</code></td><td>3</td></tr>
            <tr><td>3</td><td><code>[(0,3),(1,2),(2,1)]</code></td><td>3</td></tr>
            <tr><td>4</td><td><code>[(1,3),(2,2)]</code></td><td>2</td></tr>
            <tr><td>5</td><td><code>[(2,3)]</code></td><td>1</td></tr>
          </tbody>
        </table>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>为什么有 m + p - 1 步</h4>
            <p>4 个 micro-batch 依次进入需要 4 步；最后一个进入后，还要再穿过剩余 p-1=2 个 stage，所以总共 6 步。</p>
          </section>
          <section class="adv-panel blue">
            <h4>一条公式决定格子内容</h4>
            <div class="adv-contract">
              <span>已知</span><strong><code>t</code> 与 <code>stage</code></strong>
              <span>计算</span><strong><code>micro_idx = t - stage</code></strong>
              <span>保留条件</span><strong><code>0 &lt;= micro_idx &lt; m</code></strong>
            </div>
          </section>
        </div>

        <div class="adv-callout">timeline 不是 Tensor，没有 shape/dtype/device；它的结构契约是“长度为总时间步数的 list，每个元素是若干二元 tuple”。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用两道工序和五个订单练对角线调度", [
          "stations, orders = 2, 5",
          "steps = stations + orders - 1",
          "schedule = []",
          "",
          "for tick in range(steps):",
          "    working = []",
          "    for station in range(stations):",
          "        order_id = tick - station",
          "        if 0 <= order_id < orders:",
          "            working.append((station, order_id))",
          "    schedule.append(working)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>stations</code> -&gt; <code>p</code>，<code>orders</code> -&gt; <code>m</code>。</li>
            <li><code>steps</code> -&gt; Notebook 的总时间步数，来自 stage 数与 micro-batch 数。</li>
            <li><code>tick/station/order_id</code> -&gt; <code>t/stage/micro_idx</code>。</li>
            <li><code>schedule/working</code> -&gt; <code>timeline/active</code>：都是嵌套 Python list，不涉及 Tensor dtype 或 device。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "p=3、m=4，在时间 t=4 查看 stage=2。",
        question: "此时 micro_idx=t-stage 是多少，是否应记录？",
        options: ["2，应记录为 (2,2)", "6，不应记录", "-2，应记录"],
        answer: 0,
        revealNote: "4-2=2，且 0<=2<4，所以 stage 2 正在处理 micro-batch 2。"
      },
      checkpoint: checkpoint(
        "p=3、m=4 时，timeline 应包含多少个时间步？",
        ["6", "12", "4"],
        0,
        "总时间步是 p+m-1=3+4-1=6，包括灌满与排空阶段。"
      ),
      homework: [
        "完成 TODO 1，返回 timeline；确认 len(timeline)=p+m-1，每个 timeline[t] 是 (stage, micro_idx) 元组列表。",
        "本 TODO 不创建 Tensor，因此没有 Tensor shape/dtype/device；请改为检查容器类型、元组长度和两个编号都是 Python int。",
        "把 p=3、m=4 的六行时间轴作为 TODO 1 小测试；常见错误排查：出现负编号或编号 m 时检查边界条件，首尾缺步时检查总步数中的 -1。"
      ]
    }),

    lesson({
      id: "pipeline-count-bubbles",
      title: "再数槽位：空闲比例 = 1 - 活跃槽位 / 全部槽位",
      todo: "TODO 2：timeline、active_slots、total_slots 与 bubble",
      prerequisite: [
        "一个槽位表示“某个时间步里的某个 stage”；每步有 p 个槽位。",
        "len(step) 是该时间步的活跃 stage 数，把所有 step 的长度相加就是 active_slots。",
        "timeline 一共有 m+p-1 步，所以 total_slots = len(timeline) * p。",
        "bubble 是空闲占比。先求活跃占比 active_slots/total_slots，再用 1 减去它。"
      ],
      intuition: "时间轴已经告诉我们每一刻有几台 stage 在工作。现在只做一次考勤：所有可能的“时间 × stage”格子是总槽位，有任务的格子是活跃槽位，剩下的格子就是气泡。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-steps">
          <div><b>1</b><code>timeline = build_pipeline_timeline(p, m)</code><span>先使用 TODO 1 的真实调度结果</span></div>
          <div><b>2</b><code>active_slots = sum(len(step) ...)</code><span>每个二元组代表一个活跃槽位</span></div>
          <div><b>3</b><code>total_slots = len(timeline) * p</code><span>每一步最多 p 个 stage 同时工作</span></div>
          <div><b>4</b><code>bubble = 1 - active / total</code><span>返回 0 到 1 之间的 Python 浮点比例</span></div>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel good">
            <h4>继续 p=3、m=4 的例子</h4>
            <div class="adv-contract">
              <span>每步活跃数</span><strong>1 + 2 + 3 + 3 + 2 + 1 = 12</strong>
              <span>全部槽位</span><strong>6 个时间步 × 3 个 stage = 18</strong>
              <span>气泡比例</span><strong>1 - 12/18 = 1/3</strong>
            </div>
          </section>
          <section class="adv-panel blue">
            <h4>和闭式公式对上</h4>
            <p>每个 m 个 micro-batch 都经过 p 个 stage，所以 active_slots=m×p。</p>
            <p>代入总槽位 p×(m+p-1)，约掉 p 后得到 (p-1)/(m+p-1)。</p>
          </section>
        </div>

        <table class="adv-shapes">
          <thead><tr><th>量</th><th>p=8、m=32</th><th>语义</th></tr></thead>
          <tbody>
            <tr><td><code>len(timeline)</code></td><td>39</td><td>32+8-1 个时间步</td></tr>
            <tr><td><code>active_slots</code></td><td>256</td><td>32×8 次 stage 计算</td></tr>
            <tr><td><code>total_slots</code></td><td>312</td><td>39×8 个可用格子</td></tr>
            <tr><td><code>bubble</code></td><td><code>7/39 ≈ 0.1795</code></td><td>落在测试要求的 0.15 到 0.25 之间</td></tr>
          </tbody>
        </table>

        <div class="adv-callout">分母不是 active_slots，也不是单独的 m；它必须覆盖完整时间轴上的全部 stage 槽位。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用教室考勤表练活跃格与空闲格统计", [
          "attendance = [",
          "    [('A', 0)],",
          "    [('A', 1), ('B', 0)],",
          "    [('B', 1)],",
          "]",
          "rooms = 2",
          "present = sum(len(period) for period in attendance)",
          "capacity = len(attendance) * rooms",
          "idle_ratio = 1.0 - present / capacity",
          "assert idle_ratio == 1.0 / 3.0"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>attendance</code> -&gt; <code>timeline</code>：每个内层列表记录当前时刻的活跃项。</li>
            <li><code>rooms</code> -&gt; <code>p</code>：每个时间步最多可用的并行槽位数。</li>
            <li><code>present</code> -&gt; <code>active_slots</code>，通过各步 len 求和。</li>
            <li><code>capacity/idle_ratio</code> -&gt; <code>total_slots/bubble</code>；返回普通 float，不涉及 Tensor device。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "p=3、m=4 的时间轴每步活跃数是 1、2、3、3、2、1。",
        question: "全部槽位 total_slots 应如何计算？",
        options: ["6 个时间步 × 3 个 stage = 18", "只把活跃数相加得到 12", "4 个 micro-batch × 1 = 4"],
        answer: 0,
        revealNote: "total_slots 包含活跃和空闲格子；每个时间步都有 p 个潜在 stage 槽位。"
      },
      checkpoint: checkpoint(
        "p=8、m=32 时，基于时间轴得到的精确 bubble ratio 是哪一个？",
        ["7/39，约 0.1795", "7/8，约 0.875", "32/39，约 0.8205"],
        0,
        "公式是 (p-1)/(m+p-1)，代入得到 7/(32+7)=7/39。"
      ),
      homework: [
        "完成 TODO 2：必须调用 TODO 1 的 timeline，再统计 active_slots、total_slots 和 bubble；返回普通 Python 浮点数而不是 None。",
        "本课数据结构没有 Tensor shape/dtype/device；检查的是 timeline 嵌套长度、整数槽位计数和最终 float 比例。",
        "运行 p=8、m=32 测试；常见错误排查：结果接近 0.82 说明活跃/空闲方向写反，大于 1 时检查 total_slots，除零时检查 timeline 是否为空。"
      ]
    }),

    lesson({
      id: "pipeline-test-and-tradeoff",
      title: "最后用测试反推：micro-batch 越多，固定灌排成本越薄",
      todo: "测试与排错：p=8、m=32 的合理区间及边界直觉",
      prerequisite: [
        "气泡来自开头灌满 p 个 stage 和结尾排空的固定开销，公式分子 p-1 不随 m 增长。",
        "m 增大时，分母 m+p-1 增大，所以气泡占比下降；这不代表现实系统可以无限增加 m。",
        "可见测试为了兼容精确公式与大 m 近似公式，只要求 0.15 &lt; ratio &lt; 0.25。",
        "当前模拟忽略通信、前后向差异、显存和真实 1F1B 细节；TODO 的目标是读懂简化时间轴，而不是实现完整分布式调度器。"
      ],
      intuition: "p-1 个灌排空档像固定成本。只处理很少 micro-batch 时，这部分占比很大；让更多 micro-batch 连续穿过流水线，固定空档被更多有效计算摊薄。测试选择 m=32、p=8，就是在检查这个趋势是否落到合理数值。",
      exampleHtml: `<div class="adv-course">
        <div class="adv-grid three">
          <section class="adv-panel warn">
            <h4>m 很小</h4>
            <p>p=8、m=1 时，气泡是 7/8。只有一个 micro-batch 沿对角线移动，大部分 stage 槽位都在等待。</p>
          </section>
          <section class="adv-panel blue">
            <h4>测试规模</h4>
            <p>p=8、m=32 时，精确值是 7/39，约 0.1795，能通过 0.15 到 0.25 的严格区间。</p>
          </section>
          <section class="adv-panel good">
            <h4>m 继续增大</h4>
            <p>p 固定时，分母继续增长，气泡趋近 0；但现实里还要考虑 micro-batch 太小带来的算力利用率和通信问题。</p>
          </section>
        </div>

        <div class="adv-checks">
          <span><strong>非 None</strong><br>函数确实返回结果</span>
          <span><strong>大于 0.15</strong><br>没有把气泡低估到不合理</span>
          <span><strong>小于 0.25</strong><br>没有把空闲比例算得过高</span>
          <span><strong>精确参考</strong><br>7/39 ≈ 0.1795</span>
        </div>

        <div class="adv-grid two">
          <section class="adv-panel neutral">
            <h4>两个边界自检</h4>
            <div class="adv-contract">
              <span><code>p=1</code></span><strong>没有跨 stage 灌排，bubble 应为 0</strong>
              <span><code>m=1</code></span><strong>只有一条对角线，bubble 应为 (p-1)/p</strong>
            </div>
          </section>
          <section class="adv-panel good">
            <h4>调试顺序</h4>
            <p>先打印每个 timeline step，再核对 active_slots，最后才看 ratio。这样能区分是调度构造错，还是统计公式错。</p>
          </section>
        </div>

        <div class="adv-callout">测试区间较宽不等于任意公式都可以。基于 TODO 1 的时间轴逐格统计，才能让代码结构、精确公式和调度直觉三者一致。</div>
      </div>`,
      syntaxHtml: `<div class="adv-practice">
        ${code("用固定准备时间观察任务数增加后的空闲占比", [
          "setup_steps = 3",
          "for jobs in [1, 4, 16]:",
          "    idle = setup_steps / (jobs + setup_steps)",
          "    print(jobs, idle)",
          "",
          "# jobs 增加时，固定 setup_steps 被更多有效工作摊薄",
          "assert 3 / (16 + 3) < 3 / (4 + 3)"
        ])}
        <div class="adv-map">
          <h4>例子中的变量 -&gt; Notebook TODO 变量/操作</h4>
          <ul>
            <li><code>setup_steps</code> -&gt; <code>p - 1</code>：流水线灌满与排空带来的固定项。</li>
            <li><code>jobs</code> -&gt; <code>m</code>：连续送入的 micro-batch 数量。</li>
            <li><code>idle</code> -&gt; <code>bubble</code>：用于理解趋势，Notebook 仍要求从 timeline 统计得到它。</li>
            <li>这些都是 Python 数值，没有 Tensor shape/dtype/device；最终测试只比较 ratio 的数值区间。</li>
          </ul>
        </div>
      </div>`,
      predict: {
        hook: "保持 p=8 不变，把 m 从 32 增大到 64。",
        question: "简化模型中的 bubble ratio 会怎样变化？",
        options: ["下降，因为固定的 p-1 被更大的分母摊薄", "上升到大于 1", "完全不变"],
        answer: 0,
        revealNote: "公式分子仍是 7，分母从 39 增大到 71，所以比例下降。"
      },
      checkpoint: checkpoint(
        "compute_bubble_ratio(p=8, m=32) 返回 0.1795 左右时，可见测试结果是什么？",
        ["通过，因为它严格位于 0.15 与 0.25 之间", "失败，因为必须恰好等于 0.218", "失败，因为函数必须返回 Tensor"],
        0,
        "测试只要求非 None 且落在开区间 (0.15, 0.25)；精确时间轴结果约为 0.1795。"
      ),
      homework: [
        "完成并联调 TODO 1/2，先检查 p=3、m=4 的 timeline，再运行 Notebook 的 p=8、m=32 测试，目标精确值约为 0.1795。",
        "确认返回值是可与浮点上下界比较的 Python 数值；本 TODO 不涉及 Tensor shape/dtype/device，也不应无故创建 CUDA Tensor。",
        "常见错误排查：ratio 为 None 看 return；约 0.82 看是否算成 active ratio；时间步不等于 39 看 p+m-1；active_slots 不等于 256 看 micro_idx 边界和每步 append 位置。"
      ]
    })
  ]
};
