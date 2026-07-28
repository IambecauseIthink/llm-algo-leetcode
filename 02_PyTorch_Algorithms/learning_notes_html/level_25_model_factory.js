const esc = (value) => String(value)
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

module.exports = function renderLevel25ModelFactory(level, prev, next, slug) {
  const prevLink = prev
    ? `<a class="mf-button quiet" href="${slug(prev)}">← L${esc(prev.id)}</a>`
    : "";
  const nextLink = next
    ? `<a class="mf-button quiet" href="${slug(next)}">L${esc(next.id)} →</a>`
    : "";

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#071014">
  <meta name="description" content="进入 Transformer 模型工厂，逐层拆解 Linear 权重并完成 W8A16 INT8 量化改造。">
  <title>Level 25 | 模型工厂 · W8A16 量化</title>
  <link rel="stylesheet" href="../assets/level25_model_factory.css">
</head>
<body>
  <canvas id="factory-canvas" aria-hidden="true"></canvas>
  <header class="mf-topbar">
    <div class="mf-topbar-left">
      <a class="mf-button quiet" href="25_quantization_w8a16.html">← Level 25 导学</a>
      <div class="mf-brand">
        <span class="mf-brand-mark">25</span>
        <span><b>MODEL FACTORY</b><small>W8A16 改造任务</small></span>
      </div>
    </div>
    <div class="mf-controls" aria-label="工厂控制台">
      <div class="mf-segmented" role="group" aria-label="探索模式">
        <button class="active" data-mode="guided">跟随 Token</button>
        <button data-mode="free">自由剖视</button>
      </div>
      <div class="mf-segmented" role="group" aria-label="模型规模">
        <button class="active" data-preset="toy">玩具模型</button>
        <button data-preset="real">真实估算</button>
      </div>
      <button class="mf-button quiet" id="reset-level" type="button">重置本关</button>
    </div>
  </header>

  <main class="mf-shell">
    <section class="mf-hero">
      <div>
        <p class="mf-kicker">DECODER-ONLY TRANSFORMER · 教学剖面</p>
        <h1>显存正在爆仓。<br><span>救回模型，但别毁掉它的答案。</span></h1>
        <p class="mf-lead">你的明确目标：把权重从 <b>21.61 GiB</b> 压进 <b>6.50 GiB</b> 预算，同时让 8 组 Toy 探针的最差 logit cosine ≥ 0.99、Top-1 零翻转，并满足 W8A16 规格。</p>
      </div>
      <div class="mf-crisis">
        <div class="mf-crisis-head"><span>显存警报</span><strong id="budget-state">OVER BUDGET</strong></div>
        <div class="mf-memory-gauge"><i id="memory-gauge"></i></div>
        <div class="mf-crisis-stats">
          <span><small>FP32 基线</small><b id="fp32-total">—</b></span>
          <span><small>当前方案</small><b id="int8-total">—</b></span>
          <span><small>当前压缩率</small><b id="compression-total">1.00×</b></span>
        </div>
        <p><b>预算线：<span id="budget-limit">—</span></b> · <span id="preset-note">玩具模型会真实计算每个数值；比例与大模型一致。</span></p>
      </div>
    </section>

    <section class="mf-rescue" id="rescue-console" aria-labelledby="rescue-title">
      <header class="mf-rescue-head">
        <div>
          <p class="mf-kicker">MISSION 00 · MEMORY RESCUE</p>
          <h2 id="rescue-title">你是量化工程师：决定每类机器怎样存权重</h2>
          <p>点击方案后，仓库字节会按真实 shape 重算；8 组不同 hidden 输入也会真正重新量化、执行前向并比较 logits。显存达标不等于成功。</p>
        </div>
        <div class="mf-rescue-status" id="rescue-status">
          <small>当前任务</small>
          <b id="current-objective">先压缩最大的 MLP Linear 仓库</b>
          <span id="next-action">在 MLP Linear 行选择 INT8 · 1 byte</span>
        </div>
      </header>

      <div class="mf-goal-strip" aria-label="通关条件">
        <div data-rescue-goal="memory"><i></i><span>显存</span><b id="goal-memory">21.61 / 6.50 GiB</b></div>
        <div data-rescue-goal="quality"><i></i><span>最差 cosine</span><b id="goal-quality">1.000000 / ≥0.990000</b></div>
        <div data-rescue-goal="token"><i></i><span>Top-1 翻转</span><b id="goal-token">0 / 8 翻转</b></div>
        <div data-rescue-goal="spec"><i></i><span>任务规格</span><b id="goal-spec">W8A16</b></div>
      </div>

      <div class="mf-rescue-grid">
        <section class="mf-warehouse-panel" aria-label="权重仓库体积">
          <div class="mf-panel-title"><span>3D 权重仓库</span><b id="warehouse-total">21.61 GiB</b></div>
          <div class="mf-cargo-stack">
            <div class="mf-cargo" data-cargo="mlp"><span>MLP Linear × 32</span><i></i><b>16.13 GiB</b></div>
            <div class="mf-cargo" data-cargo="attention"><span>Attention Linear × 32</span><i></i><b>5.00 GiB</b></div>
            <div class="mf-cargo" data-cargo="lm_head"><span>LM Head</span><i></i><b>0.49 GiB</b></div>
            <div class="mf-cargo tiny" data-cargo="norm"><span>全部 RMSNorm</span><i></i><b>0.001 GiB</b></div>
          </div>
          <p class="mf-sim-note">箱体长度按真实规模权重字节近似缩放。INT8 是 1 byte；INT4 是半个 byte，但通常需要额外 scale 元数据。</p>
        </section>

        <section class="mf-plan-panel" aria-label="量化方案控制">
          <div class="mf-panel-title"><span>改造开关</span><b>点击立即试验</b></div>
          <div class="mf-plan-row" data-plan-row="mlp">
            <div><b>MLP Linear</b><small>gate / up / down · 最大仓库</small></div>
            <div class="mf-scheme-buttons" role="group" aria-label="MLP Linear 存储精度">
              <button class="active" data-target="mlp" data-scheme="fp32">FP32<small>4B</small></button>
              <button data-target="mlp" data-scheme="int8">INT8<small>1B</small></button>
              <button data-target="mlp" data-scheme="int4">INT4<small>0.5B</small></button>
            </div>
          </div>
          <div class="mf-plan-row" data-plan-row="attention">
            <div><b>Attention Linear</b><small>q / k / v / o · 所有 Block</small></div>
            <div class="mf-scheme-buttons" role="group" aria-label="Attention Linear 存储精度">
              <button class="active" data-target="attention" data-scheme="fp32">FP32<small>4B</small></button>
              <button data-target="attention" data-scheme="int8">INT8<small>1B</small></button>
              <button data-target="attention" data-scheme="int4">INT4<small>0.5B</small></button>
            </div>
          </div>
          <div class="mf-plan-row" data-plan-row="lm_head">
            <div><b>LM Head</b><small>hidden → vocab · 直接影响词排名</small></div>
            <div class="mf-scheme-buttons" role="group" aria-label="LM Head 存储精度">
              <button class="active" data-target="lm_head" data-scheme="fp32">FP32<small>4B</small></button>
              <button data-target="lm_head" data-scheme="int8">INT8<small>1B</small></button>
              <button data-target="lm_head" data-scheme="int4">INT4<small>0.5B</small></button>
            </div>
          </div>
          <div class="mf-plan-row risky" data-plan-row="norm">
            <div><b>RMSNorm</b><small>只有 H 个参数 · 非 Linear 改造目标</small></div>
            <div class="mf-scheme-buttons" role="group" aria-label="RMSNorm 存储精度">
              <button class="active" data-target="norm" data-scheme="fp32">FP32<small>保留</small></button>
              <button data-target="norm" data-scheme="int8">INT8<small>收益极小</small></button>
            </div>
          </div>
          <div class="mf-plan-row risky" data-plan-row="activation">
            <div><b>Residual / Activation</b><small>不是权重仓 · 不减少模型权重</small></div>
            <div class="mf-scheme-buttons" role="group" aria-label="Activation 精度">
              <button class="active" data-target="activation" data-scheme="a16">A16<small>浮点</small></button>
              <button data-target="activation" data-scheme="a8">A8<small>另一个课题</small></button>
            </div>
          </div>
          <div class="mf-granularity">
            <div><b>标尺粒度</b><small>改变误差与 scale 元数据，不改变整数 bit 数</small></div>
            <div class="mf-scheme-buttons" role="group" aria-label="量化标尺粒度">
              <button class="active" data-granularity="tensor">整仓 1 把尺<small>Notebook</small></button>
              <button data-granularity="channel">每行 1 把尺<small>局部 scale</small></button>
              <button data-granularity="group">每 64 个 1 把尺<small>group scale</small></button>
            </div>
          </div>
        </section>

        <section class="mf-output-probe" aria-label="模型输出精度探针">
          <div class="mf-panel-title"><span>8 组 Toy Transformer 探针 · 浏览器实算</span><b id="quality-state">BASELINE</b></div>
          <p class="mf-probe-prompt">Top-5 展示第 1 组；下方指标聚合全部 8 组不同 hidden 输入</p>
          <div class="mf-token-compare">
            <div>
              <small>FP32 基线 Top-5</small>
              <ol id="baseline-tokens"></ol>
            </div>
            <div>
              <small>当前方案 Top-5</small>
              <ol id="current-tokens"></ol>
            </div>
          </div>
          <div class="mf-quality-meter"><i id="quality-meter"></i><span>最差 logit cosine <b id="quality-score">1.000000</b></span></div>
          <dl class="mf-probe-facts">
            <div><dt>8 组中最差 cosine</dt><dd id="deployment-cosine">1.000000</dd></div>
            <div><dt>平均 |Δlogit|（越低越好）</dt><dd id="mean-logit-error">0.000000</dd></div>
            <div><dt>最大 |Δlogit|</dt><dd id="max-logit-error">0.000000</dd></div>
            <div><dt>Top-1 翻转数</dt><dd id="top1-state">0 / 8</dd></div>
            <div><dt>权重节省</dt><dd id="saved-memory">0 GiB</dd></div>
          </dl>
          <p class="mf-sim-note">这些是小型 LLaMA-style block 的真实数值，不是预设得分。cosine 不是单调“分数”：单样本可能因误差抵消而微升，所以必须与平均/最大误差和翻转数一起看。</p>
        </section>
      </div>

      <footer class="mf-rescue-footer">
        <div class="mf-feedback" id="rescue-feedback">从最大的 MLP Linear 开始。目标不是“越低 bit 越好”，而是在四条约束之间找到可上线方案。</div>
        <button class="mf-button boss-button" id="validate-rescue" type="button">验证当前改造方案</button>
      </footer>
    </section>

    <section class="mf-evidence" aria-labelledby="evidence-title">
      <header>
        <p class="mf-kicker">ENGINEERING EVIDENCE · 哪些结论能迁移？</p>
        <h2 id="evidence-title">实算、论文结论和未知项，必须分开</h2>
        <p>每张卡片已经把论文压缩成“问题 → 方法 → 证据 → 边界 → 本关启示”。这些论文支持的是量化的一般规律，不是“某一层必然掉多少精度”；真实 checkpoint 仍需用校准集和任务评测逐层消融。</p>
      </header>
      <div class="mf-evidence-grid">
        <article class="mf-paper-card">
          <div class="mf-paper-head">
            <span>W8 · MIXED PRECISION</span><b>LLM.int8()</b>
            <small>Dettmers et al. · NeurIPS 2022</small>
          </div>
          <p class="mf-paper-thesis">一句话：真正难的不是大多数普通值，而是少量、系统性出现的异常特征维度。</p>
          <dl class="mf-paper-notes">
            <div><dt>它在解决什么？</dt><dd>大模型推理显存太高；直接 INT8 又会被异常值破坏。</dd></div>
            <div><dt>核心动作</dt><dd>大部分内积做 vector-wise INT8；检测出的异常维度单独走 FP16 矩阵乘。</dd></div>
            <div><dt>论文证据</dt><dd>超过 99.9% 的值参与 8-bit 计算；论文报告最高到 175B 模型无性能退化，并把相对 16-bit 的推理内存减半。</dd></div>
            <div class="boundary"><dt>不能误读</dt><dd>它不是“整个张量共用一把尺”的朴素 weight-only 量化，也不是所有运算都变 INT8。</dd></div>
          </dl>
          <p class="mf-paper-lesson"><b>带回本关：</b>看到异常值时，不应只继续压低 bit；先决定是否拆分异常通道。</p>
          <a href="https://arxiv.org/abs/2208.07339" target="_blank" rel="noreferrer">打开论文原文 ↗</a>
        </article>
        <article class="mf-paper-card">
          <div class="mf-paper-head">
            <span>W8A8 · TRAINING-FREE PTQ</span><b>SmoothQuant</b>
            <small>Xiao et al. · ICML 2023</small>
          </div>
          <p class="mf-paper-thesis">一句话：activation 比 weight 难量化，可以用数学等价的缩放把难度从 activation 迁移到 weight。</p>
          <dl class="mf-paper-notes">
            <div><dt>它在解决什么？</dt><dd>activation 中的异常值让 W8A8 很难同时保持精度和硬件效率。</dd></div>
            <div><dt>核心动作</dt><dd>离线收集统计量，用等价缩放平滑 activation；权重吸收反向缩放，不需要重新训练。</dd></div>
            <div><dt>论文证据</dt><dd>论文报告最高 1.56× 加速、2× 内存降低且精度损失可忽略，并展示 530B 模型单节点服务。</dd></div>
            <div class="boundary"><dt>不能误读</dt><dd>A8 不是一个可以随手打开的开关；需要校准变换以及真正支持 INT8 的推理内核。</dd></div>
          </dl>
          <p class="mf-paper-lesson"><b>带回本关：</b>Notebook 是 W8A16；切成 A8 已经进入 SmoothQuant 类问题，而不是继续压权重仓。</p>
          <a href="https://arxiv.org/abs/2211.10438" target="_blank" rel="noreferrer">打开论文原文 ↗</a>
        </article>
        <article class="mf-paper-card">
          <div class="mf-paper-head">
            <span>W4 · WEIGHT-ONLY</span><b>AWQ</b>
            <small>Lin et al. · MLSys 2024 Best Paper</small>
          </div>
          <p class="mf-paper-thesis">一句话：权重的重要性并不相同，而且“谁重要”要看它乘到的 activation，而不能只看权重大小。</p>
          <dl class="mf-paper-notes">
            <div><dt>它在解决什么？</dt><dd>4-bit weight-only 很省空间，但少数关键通道的误差可能主导模型输出。</dd></div>
            <div><dt>核心动作</dt><dd>用离线 activation 统计定位显著通道，再通过等价缩放保护它们；不依赖反向传播或重构。</dd></div>
            <div><dt>论文证据</dt><dd>论文发现保护约 1% 显著权重即可大幅降低量化误差；配套 TinyChat 相对 Hugging Face FP16 在桌面与移动 GPU 上报告超过 3× 加速。</dd></div>
            <div class="boundary"><dt>不能误读</dt><dd>AWQ 的 W4 结果来自 activation-aware scaling、权重打包和融合内核，不等于把 absmax 的 qmax 从 127 改成 7。</dd></div>
          </dl>
          <p class="mf-paper-lesson"><b>带回本关：</b>bit 数只是存储格式；保护哪些通道、scale 怎样分组同样决定误差。</p>
          <a href="https://arxiv.org/abs/2306.00978" target="_blank" rel="noreferrer">打开论文原文 ↗</a>
        </article>
        <article class="mf-paper-card">
          <div class="mf-paper-head">
            <span>W3/W4 · SECOND-ORDER PTQ</span><b>GPTQ</b>
            <small>Frantar et al. · ICLR 2023</small>
          </div>
          <p class="mf-paper-thesis">一句话：量化一个权重后，可以利用近似二阶信息调整后续权重，主动补偿它造成的输出误差。</p>
          <dl class="mf-paper-notes">
            <div><dt>它在解决什么？</dt><dd>把超大 Transformer 压到 3/4 bit，同时让一次性 PTQ 在可接受时间内完成。</dd></div>
            <div><dt>核心动作</dt><dd>利用近似二阶信息选择并补偿量化误差，而不是逐个权重独立做 round。</dd></div>
            <div><dt>论文证据</dt><dd>论文报告约 4 GPU 小时量化 175B 模型到每权重 3/4 bit，精度退化可忽略，并实现单 GPU 生成推理。</dd></div>
            <div class="boundary"><dt>不能误读</dt><dd>GPTQ 需要代表性校准数据和误差补偿；本页的 plain absmax INT4 只是故意设置的朴素对照组。</dd></div>
          </dl>
          <p class="mf-paper-lesson"><b>带回本关：</b>round 之后的误差不一定只能接受；更高级 PTQ 会利用层输入信息补偿误差。</p>
          <a href="https://arxiv.org/abs/2210.17323" target="_blank" rel="noreferrer">打开论文原文 ↗</a>
        </article>
      </div>
      <details>
        <summary>21.61 GiB 是怎样算出来的？</summary>
        <div class="mf-memory-proof">
          <code>MLP = L × (I×H + I×H + H×I)</code>
          <code>Attention = L × (H×H + KV×H + KV×H + H×H)</code>
          <code>LM Head = V×H</code>
          <code>RMSNorm = (2L+1)×H</code>
        </div>
        <p>本页真实规模档取 L=32、H=4096、I=11008、KV=1024、V=32000；再乘 FP32 的 4 byte 或 INT8 的 1 byte。per-channel/group 的 FP32 scale 元数据也计入预算。</p>
      </details>
    </section>

    <nav class="mf-mission-rail" aria-label="任务进度">
      <button class="active" data-mission-tab="0"><span>00</span><b>抢救模型</b><small>显存与精度同时达标</small></button>
      <button data-mission-tab="1"><span>01</span><b>搭建 Block</b><small>再拆到字节</small></button>
      <button data-mission-tab="2"><span>02</span><b>校准标尺</b><small>absmax + scale</small></button>
      <button data-mission-tab="3"><span>03</span><b>整数铸造</b><small>量化流水线</small></button>
      <button data-mission-tab="4"><span>04</span><b>推理反应堆</b><small>反量化 + Linear</small></button>
      <button data-mission-tab="5"><span>★</span><b>上线挑战</b><small>同时满足五项指标</small></button>
    </nav>

    <section class="mf-workbench">
      <aside class="mf-depth-rail" aria-label="剖视深度">
        <p>剖视深度</p>
        <button class="active" data-view="0"><b>01</b><span>完整模型</span></button>
        <button data-view="1"><b>02</b><span>Block</span></button>
        <button data-view="2"><b>03</b><span>Linear</span></button>
        <button data-view="3"><b>04</b><span>权重矩阵</span></button>
        <button data-view="4"><b>05</b><span>一个权重</span></button>
      </aside>

      <section class="mf-viewport" aria-label="模型工厂剖视图">
        <div class="mf-viewport-head">
          <div>
            <p id="view-path">MODEL / OVERVIEW</p>
            <h2 id="view-title">Decoder Transformer 工厂</h2>
          </div>
          <div class="mf-lenses" role="group" aria-label="观察镜片">
            <button class="active" data-lens="flow">数据流</button>
            <button data-lens="shape">Shape</button>
            <button data-lens="memory">内存</button>
            <button data-lens="numeric">数值</button>
          </div>
        </div>

        <div class="mf-world" id="factory-world" data-view="0" data-lens="flow">
          <div class="mf-scene model-scene" data-scene="0">
            <div class="mf-model-line">
              <button class="mf-station token-station" data-node="token"><span>token</span><b>“量化”</b><small>ID 7</small></button>
              <i class="mf-pipe"><em></em></i>
              <button class="mf-station" data-node="embedding"><span>Embedding</span><b>ID → 向量</b><small>[B,S,H]</small></button>
              <i class="mf-pipe"><em></em></i>
              <div class="mf-block-stack" aria-label="重复 Transformer Block">
                <button class="mf-station block-machine" data-node="block"><span>Block × <i data-layers>2</i></span><b>Attention + MLP</b><small>点击进入内部</small></button>
                <span class="mf-stack-layer one"></span><span class="mf-stack-layer two"></span>
              </div>
              <i class="mf-pipe"><em></em></i>
              <button class="mf-station" data-node="final_norm"><span>RMSNorm</span><b>稳定尺度</b><small>小参数</small></button>
              <i class="mf-pipe"><em></em></i>
              <button class="mf-station quantizable" data-node="lm_head"><span>LM Head</span><b>H → Vocab</b><small>大型 Linear</small></button>
              <i class="mf-pipe"><em></em></i>
              <button class="mf-station logits-station" data-node="logits"><span>Logits</span><b>下个 token</b><small>[B,S,V]</small></button>
            </div>
            <div class="mf-factory-floor"></div>
          </div>

          <div class="mf-scene block-scene" data-scene="1">
            <div class="mf-residual-track top"><span>residual 主干</span></div>
            <div class="mf-block-flow">
              <button class="mf-machine" data-node="rms1"><small>PRE-NORM</small><b>RMSNorm</b><span>无大型矩阵</span></button>
              <i class="mf-arrow"></i>
              <div class="mf-machine-group attention-group">
                <strong>SELF ATTENTION</strong>
                <div>
                  <button class="mf-mini-machine quantizable" data-node="q_proj">q_proj</button>
                  <button class="mf-mini-machine quantizable" data-node="k_proj">k_proj</button>
                  <button class="mf-mini-machine quantizable" data-node="v_proj">v_proj</button>
                </div>
                <button class="mf-process" data-node="rope">Q/K + RoPE → Attention</button>
                <button class="mf-mini-machine quantizable wide" data-node="o_proj">o_proj</button>
              </div>
              <i class="mf-arrow"></i>
              <button class="mf-machine residual-machine" data-node="residual1"><small>ADD</small><b>Residual</b><span>无权重</span></button>
              <i class="mf-arrow"></i>
              <button class="mf-machine" data-node="rms2"><small>PRE-NORM</small><b>RMSNorm</b><span>无大型矩阵</span></button>
              <i class="mf-arrow"></i>
              <div class="mf-machine-group mlp-group">
                <strong>SWIGLU MLP</strong>
                <div>
                  <button class="mf-mini-machine quantizable" data-node="gate_proj">gate_proj</button>
                  <button class="mf-mini-machine quantizable" data-node="up_proj">up_proj</button>
                </div>
                <button class="mf-process" data-node="silu">SiLU(gate) ⊙ up</button>
                <button class="mf-mini-machine quantizable wide" data-node="down_proj">down_proj</button>
              </div>
              <i class="mf-arrow"></i>
              <button class="mf-machine residual-machine" data-node="residual2"><small>ADD</small><b>Residual</b><span>无权重</span></button>
            </div>
            <div class="mf-residual-track bottom"><span>residual 主干</span></div>
          </div>

          <div class="mf-scene linear-scene" data-scene="2">
            <div class="mf-linear-picker" aria-label="选择投影层">
              <button class="active" data-linear="q_proj">q_proj</button>
              <button data-linear="o_proj">o_proj</button>
              <button data-linear="gate_proj">gate_proj</button>
              <button data-linear="down_proj">down_proj</button>
            </div>
            <div class="mf-linear-machine">
              <div class="mf-port input-port"><small>入口 x</small><b id="linear-input-shape">[B,S,H]</b><span>activation · x.dtype</span></div>
              <div class="mf-rotor">
                <span class="mf-rotor-ring"></span>
                <b>W</b>
                <small id="linear-weight-shape">[out,in]</small>
              </div>
              <div class="mf-port output-port"><small>出口 y</small><b id="linear-output-shape">[B,S,out]</b><span>floating output</span></div>
            </div>
            <div class="mf-equation"><code>y = x Wᵀ + b</code><span id="linear-equation-note">最后一维 in_features 被 out_features 替换</span></div>
          </div>

          <div class="mf-scene matrix-scene" data-scene="3">
            <div class="mf-matrix-wrap">
              <div class="mf-matrix-label top">in_features →</div>
              <div class="mf-matrix-label side">out_features ↓</div>
              <div class="mf-matrix" id="weight-matrix" aria-label="玩具权重矩阵"></div>
              <div class="mf-matrix-scan"></div>
            </div>
            <div class="mf-matrix-stats">
              <span><small>权重 shape</small><b id="matrix-shape">—</b></span>
              <span><small>参数量</small><b id="matrix-params">—</b></span>
              <span><small>FP32 仓库</small><b id="matrix-fp32">—</b></span>
              <span><small>INT8 仓库</small><b id="matrix-int8">—</b></span>
            </div>
          </div>

          <div class="mf-scene byte-scene" data-scene="4">
            <div class="mf-byte-lab">
              <section>
                <p>原始 FP32 权重 <code id="byte-fp-value">2.5</code></p>
                <div class="mf-byte-row fp32-bytes" id="byte-fp32-row"></div>
                <small><span id="byte-fp-hex">0x40200000</span> · 32 bit = 4 个不同职责的 byte</small>
              </section>
              <div class="mf-compressor"><span>× scale</span><b>量化压缩舱</b><span>round · clamp</span></div>
              <section>
                <p>INT8 code <code id="byte-int-value">106</code></p>
                <div class="mf-byte-row int8-bytes" id="byte-int8-row"></div>
                <small>恢复值 <b id="byte-restored-value">2.50394</b> · 误差 <b id="byte-error">0.00394</b></small>
              </section>
            </div>
            <div class="mf-byte-verdict"><b id="byte-selection">当前 W[0,3] = 2.5</b><span>FP32：4 个 byte 保存实际浮点位</span><span>INT8：1 个 byte 保存整数 code；scale 由一组权重共享</span></div>
          </div>

          <div class="mf-lens-readout" id="lens-readout" aria-live="polite"></div>
          <div class="mf-zoom-controls">
            <button class="mf-button quiet" id="zoom-out" type="button">← 向外一层</button>
            <span id="zoom-hint">点击 Block，进入模型内部</span>
            <button class="mf-button hot" id="zoom-in" type="button">深入一层 →</button>
          </div>
        </div>
      </section>

      <aside class="mf-inspector" aria-label="机器检查器">
        <div class="mf-inspector-head">
          <span id="node-type">MODEL</span>
          <i id="quant-badge">教学剖面</i>
        </div>
        <h3 id="node-title">Decoder Transformer</h3>
        <p id="node-description">一个 token 会先变成 hidden state，再穿过重复 Block，最后由 LM Head 产生下一个 token 的 logits。</p>
        <dl class="mf-contract">
          <div><dt>输入</dt><dd id="node-input">token ids [B,S]</dd></div>
          <div><dt>输出</dt><dd id="node-output">logits [B,S,V]</dd></div>
          <div><dt>权重</dt><dd id="node-weight">Embedding + Linear 为主</dd></div>
          <div><dt>量化</dt><dd id="node-quant">逐台检查大型权重仓</dd></div>
        </dl>
        <div class="mf-console" id="factory-console" role="status">
          <span>SYSTEM</span>
          <p>显存预算不足。先找出模型里真正的大型权重仓库。</p>
        </div>
        <details class="mf-manual">
          <summary>工程师手册：这层到底在做什么？</summary>
          <div id="manual-content"></div>
        </details>
      </aside>
    </section>

    <section class="mf-mission-deck">
      <article class="mf-mission-panel active" data-mission-panel="0">
        <div class="mf-mission-copy">
          <p class="mf-kicker">PROLOGUE · MEMORY CRISIS</p>
          <h2>先在上方驾驶舱救回模型，再进入它的内部</h2>
          <p>把 MLP、Attention 和 LM Head 的大型 Linear 仓库改为 INT8；RMSNorm 不属于本 Notebook 的 Linear 替换范围，A8 也不是 W8 权重压缩。显存、8 组数值探针、Top-1 和规格四项同时通过才算成功。</p>
        </div>
        <div class="mf-choice-grid">
          <button data-hunt="norm"><b>为什么不压 RMSNorm？</b><span>权重极少、不是本 Notebook 的 Linear 目标；真实精度影响需实测</span></button>
          <button data-hunt="residual"><b>为什么 A8 不算 W8？</b><span>它改变中间 activation，不减少权重仓库</span></button>
          <button data-hunt="linear"><b>带我找到 Linear</b><span>进入 Block，看见真正的大型二维矩阵</span></button>
        </div>
        <div class="mf-feedback" id="hunt-feedback">驾驶舱会给出下一步提示；这个区域负责解释“为什么”。</div>
      </article>

      <article class="mf-mission-panel" data-mission-panel="1">
        <div class="mf-mission-copy">
          <p class="mf-kicker">MISSION 01 · BUILD THE BLOCK</p>
          <h2>先照说明书搭出一个 Block，再钻进它的一颗权重</h2>
          <p>积木顺序对应真实的 Pre-Norm 残差数据流。每次只装下一块；搭错时工厂会指出被破坏的 shape 或 residual 契约。</p>
        </div>
        <div class="mf-assembly">
          <aside class="mf-instruction-card">
            <small>ASSEMBLY MANUAL · 当前步骤</small>
            <b id="assembly-step">STEP 1 / 6</b>
            <h3 id="assembly-instruction">先在 Attention 之前安装 RMSNorm</h3>
            <p id="assembly-contract">输入和输出都是 [B,S,H]；只有 [H] 个可学习缩放参数。</p>
            <div class="mf-hidden-vector"><span>TOKEN HIDDEN</span><code id="assembly-vector">[0.80, −1.10, 0.35, 1.60]</code></div>
          </aside>
          <section class="mf-brick-yard" aria-label="Transformer Block 积木">
            <div class="mf-brick-bin">
              <button data-block-piece="rms1"><b>RMSNorm A</b><small>[H] 小权重</small></button>
              <button data-block-piece="attention"><b>Attention</b><small>q/k/v/o Linear</small></button>
              <button data-block-piece="residual1"><b>Residual + A</b><small>0 权重</small></button>
              <button data-block-piece="rms2"><b>RMSNorm B</b><small>[H] 小权重</small></button>
              <button data-block-piece="mlp"><b>SwiGLU MLP</b><small>gate/up/down</small></button>
              <button data-block-piece="residual2"><b>Residual + B</b><small>0 权重</small></button>
            </div>
            <div class="mf-assembly-line" id="block-assembly-line" aria-label="Block 组装槽">
              <span data-block-slot="0">01</span><i></i><span data-block-slot="1">02</span><i></i><span data-block-slot="2">03</span><i></i><span data-block-slot="3">04</span><i></i><span data-block-slot="4">05</span><i></i><span data-block-slot="5">06</span>
              <em id="assembly-token" aria-hidden="true">●</em>
            </div>
            <div class="mf-action-row">
              <button class="mf-button quiet" id="clear-assembly" type="button">拆掉重搭</button>
              <button class="mf-button hot" id="run-assembly" type="button" disabled>让 hidden state 跑一遍</button>
            </div>
          </section>
        </div>
        <div class="mf-feedback" id="assembly-feedback">说明书不是死记顺序：观察每块积木有没有权重、怎样保持 [B,S,H] 主干。</div>
        <div class="mf-mission-copy mf-autopsy-copy">
          <h3>组装完成后：沿五级剖视钻到单个权重</h3>
          <p>选择矩阵里的不同数值，页面会用 DataView 读取该值真实的 IEEE-754 FP32 四字节，再显示对应的 INT8 二进制。</p>
        </div>
        <div class="mf-action-row">
          <button class="mf-button hot" id="unfold-bytes" type="button" disabled>展开一个 FP32 权重</button>
          <button class="mf-button hot" id="compress-byte" type="button" disabled>装配 INT8 存储单元</button>
          <span class="mf-status-chip" id="autopsy-status">尚未到达字节层</span>
        </div>
      </article>

      <article class="mf-mission-panel" data-mission-panel="2">
        <div class="mf-mission-copy">
          <p class="mf-kicker">MISSION 02 · CALIBRATION TOWER</p>
          <h2>一把 INT8 尺子，怎样量遍整张权重矩阵？</h2>
          <p>本 Notebook 使用 per-tensor 对称量化：整个张量共享一个 scale。先扫描最大绝对值，再把它对齐到 127。</p>
        </div>
        <div class="mf-calibration-grid">
          <section class="mf-lab-card">
            <div class="mf-card-head"><b>输入权重</b><button class="mf-button tiny" id="restore-sample" type="button">恢复样本</button></div>
            <div class="mf-sample-values" id="sample-values"></div>
            <label class="mf-switch"><input type="checkbox" id="zero-mode"><span></span>切换为全零张量，制造除零故障</label>
            <label class="mf-slider">异常值幅度 <b id="outlier-value">3.0</b><input type="range" id="outlier-slider" min="3" max="20" value="3" step="0.5"></label>
          </section>
          <section class="mf-lab-card calibration-tower">
            <div class="mf-tower-dial"><i id="dial-needle"></i><span>127</span></div>
            <dl>
              <div><dt>absmax</dt><dd id="absmax-readout">等待扫描</dd></div>
              <div><dt>scale = 127 / absmax</dt><dd id="scale-readout">—</dd></div>
              <div><dt>2.5 的量化值</dt><dd id="sample-quant-readout">—</dd></div>
            </dl>
            <label class="mf-switch danger"><input type="checkbox" id="zero-guard"><span></span>安装零值保护模块</label>
            <button class="mf-button hot" id="run-calibration" type="button">启动 absmax 扫描</button>
          </section>
          <section class="mf-lab-card">
            <div class="mf-card-head"><b>量化误差显微镜</b><span id="error-average">MAE —</span></div>
            <div class="mf-error-bars" id="error-bars"></div>
            <p class="mf-card-note">把异常值拖大：scale 会照顾最远的值，小权重能分到的整数刻度随之减少。</p>
          </section>
        </div>
        <div class="mf-feedback" id="calibration-feedback">先用原始样本扫描一次，再试试全零和异常值。</div>
      </article>

      <article class="mf-mission-panel" data-mission-panel="3">
        <div class="mf-mission-copy">
          <p class="mf-kicker">MISSION 03 · INTEGER FOUNDRY</p>
          <h2>把四台机器排成正确的铸造流水线</h2>
          <p>点击或拖动模块进入流水线。错误顺序不会只说“错了”，机器会展示它破坏了哪条 dtype 或数值契约。</p>
        </div>
        <div class="mf-foundry">
          <div class="mf-module-bin" id="quant-module-bin" aria-label="可用量化模块">
            <button draggable="true" data-module="scale">× scale</button>
            <button draggable="true" data-module="round">round</button>
            <button draggable="true" data-module="clamp">clamp [-128,127]</button>
            <button draggable="true" data-module="cast">cast int8</button>
          </div>
          <div class="mf-pipeline" id="quant-pipeline" aria-label="量化流水线" tabindex="0">
            <span data-slot="0">01</span><i></i><span data-slot="1">02</span><i></i><span data-slot="2">03</span><i></i><span data-slot="3">04</span>
          </div>
          <div class="mf-particle-trace" id="quant-trace"><span>输入 2.5 / float</span></div>
          <div class="mf-action-row">
            <button class="mf-button quiet" id="clear-quant-pipeline" type="button">清空</button>
            <button class="mf-button hot" id="run-quant-pipeline" type="button">点火运行</button>
          </div>
        </div>
        <div class="mf-feedback" id="quant-feedback">正确链路必须在浮点域完成缩放、舍入与边界限制，最后才改存储 dtype。</div>
      </article>

      <article class="mf-mission-panel" data-mission-panel="4">
        <div class="mf-mission-copy">
          <p class="mf-kicker">MISSION 04 · INFERENCE REACTOR</p>
          <h2>INT8 负责存，x.dtype 负责算</h2>
          <p>Notebook 模拟层不会直接调用 INT8 Tensor Core。它先恢复近似浮点权重，再交给普通 F.linear。</p>
        </div>
        <div class="mf-reactor-grid">
          <section class="mf-lab-card">
            <b>连接反应堆</b>
            <div class="mf-module-bin runtime-bin" id="runtime-module-bin">
              <button data-runtime="cast">to(x.dtype)</button>
              <button data-runtime="divide">÷ scale</button>
              <button data-runtime="linear">F.linear</button>
            </div>
            <div class="mf-runtime-chain" id="runtime-chain"><span>weight_int8</span><i></i><span>?</span><i></i><span>?</span><i></i><span>?</span></div>
            <div class="mf-action-row">
              <button class="mf-button quiet" id="clear-runtime" type="button">清空</button>
              <button class="mf-button hot" id="run-runtime" type="button">启动推理</button>
            </div>
          </section>
          <section class="mf-reactor mf-lab-card">
            <div class="mf-reactor-core"><span></span><b>F.linear</b><small>FLOAT COMPUTE</small></div>
            <div class="mf-shape-transform"><code>[2,10,128]</code><i>×</i><code>[64,128]</code><b>→</b><code>[2,10,64]</code></div>
          </section>
          <section class="mf-lab-card mf-metrics">
            <div><span>FP32 输出</span><b id="fp-output">—</b></div>
            <div><span>W8A16 输出</span><b id="quant-output">—</b></div>
            <div><span>Cosine Similarity</span><b id="cosine-output">—</b></div>
            <p>生产环境若要获得真正 INT8 计算加速，需要融合量化 GEMM 内核与硬件支持。</p>
          </section>
        </div>
        <div class="mf-feedback" id="runtime-feedback">反量化方向很重要：量化时乘 scale，恢复时必须除以同一个 scale。</div>
      </article>

      <article class="mf-mission-panel boss-panel" data-mission-panel="5">
        <div class="mf-mission-copy">
          <p class="mf-kicker">BOSS CHALLENGE · SHIP THE MODEL</p>
          <h2>让整座模型在预算内上线</h2>
          <p>五条工程约束必须同时满足。通关后，模型里所有可采用 weight-only 量化的 Linear 仓库都会点亮。</p>
        </div>
        <div class="mf-boss-grid">
          <div class="mf-boss-core"><span class="orbit one"></span><span class="orbit two"></span><b>INT8</b><small>DEPLOY CORE</small></div>
          <ul class="mf-boss-checks" id="boss-checks">
            <li data-boss="memory"><i></i><span>当前方案进入 6.50 GiB 预算</span><b id="boss-memory">—</b></li>
            <li data-boss="rescue"><i></i><span>8 组探针、Top-1 与 W8A16 规格通过</span><b id="boss-quality">—</b></li>
            <li data-boss="scale"><i></i><span>全零输入的 scale 保持有限</span><b>finite</b></li>
            <li data-boss="cosine"><i></i><span>输出余弦相似度超过 0.99</span><b id="boss-cosine">—</b></li>
            <li data-boss="shape"><i></i><span>输出 shape 为 [2,10,64]</span><b>[2,10,64]</b></li>
          </ul>
          <button class="mf-button boss-button" id="launch-model" type="button">执行上线检查</button>
        </div>
        <div class="mf-feedback" id="boss-feedback">完成 Mission 01–04 后执行最终检查。</div>
        <div class="mf-notebook-bridge" id="notebook-bridge" hidden>
          <div>
            <p class="mf-kicker">MISSION COMPLETE · RETURN TO NOTEBOOK</p>
            <h3>你已经看见了权重的完整生命周期</h3>
            <p>下面只给检查点，不直接泄露完整代码。回到 Notebook 完成真正的 PyTorch 实现。</p>
          </div>
          <ol>
            <li><b>TODO 1</b><span>absmax 是标量；全零输入先保护再做除法。</span></li>
            <li><b>TODO 2</b><span>scale 表示量化时乘上的放大倍数：127 / absmax。</span></li>
            <li><b>TODO 3</b><span>shape 不变，最后 dtype 必须是 torch.int8。</span></li>
            <li><b>TODO 4</b><span>weight_int8 跟随 x.dtype 后除以 scale，再调用 F.linear。</span></li>
          </ol>
          <a class="mf-button hot" href="../../25_Quantization_W8A16.ipynb">打开 Notebook 作业</a>
        </div>
      </article>
    </section>

    <footer class="mf-footer">
      <div>
        <b>教学剖面，不是特定模型配置声明</b>
        <p>本工厂使用 LLaMA 风格组件建立心智模型；真实模型的层数、GQA 比例、权重共享和量化粒度可能不同。</p>
      </div>
      <nav>${prevLink}${nextLink}</nav>
    </footer>
  </main>

  <script src="../assets/level25_model_factory.js"></script>
</body>
</html>`;
};
