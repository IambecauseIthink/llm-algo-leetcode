(function attachLevel25Factory(root, buildFactory) {
  const api = buildFactory(root);
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }
  root.Level25ModelFactory = api;
  const start = () => api.init(document);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function buildLevel25Factory(root) {
  "use strict";

  const QUANT_ORDER = ["scale", "round", "clamp", "cast"];
  const RUNTIME_ORDER = ["cast", "divide", "linear"];
  const SAMPLE = [-0.8, 1.5, -3.0, 2.5, 0.0];
  const MATRIX_SAMPLE = [
    [1.0, -2.0, 3.0, -4.0],
    [0.5, 0.25, -0.75, 1.5],
    [-1.0, 0.0, 1.0, -2.0]
  ];
  const MATRIX_BIAS = [0.1, -0.2, 0.3];
  const MATRIX_INPUT = [
    [1.0, -1.0, 0.5, 2.0],
    [0.0, 1.0, -1.0, 3.0]
  ];

  const PRESETS = {
    toy: {
      label: "玩具模型",
      hidden: 8,
      kv: 4,
      intermediate: 16,
      vocab: 32,
      layers: 2,
      batchShape: "[1,4",
      note: "玩具模型会真实计算每个数值；比例与大模型一致。"
    },
    real: {
      label: "真实规模估算",
      hidden: 4096,
      kv: 1024,
      intermediate: 11008,
      vocab: 32000,
      layers: 32,
      batchShape: "[B,S",
      note: "典型 LLaMA 风格估算：不在浏览器创建十亿级矩阵，且未计 bias、Norm、权重共享等架构差异。"
    }
  };

  const MODEL_GRAPH = {
    model: {
      title: "Decoder Transformer",
      type: "MODEL",
      description: "token ids 先变成 hidden states，穿过重复的 Transformer Blocks，最后由 LM Head 产生下一个 token 的 logits。",
      input: "token ids [B,S]",
      output: "logits [B,S,V]",
      weight: "Embedding + Linear 权重占绝大多数",
      quant: "逐台检查大型权重仓",
      manual: "这是一张 LLaMA 风格教学剖面，不代表某个具体 checkpoint。真实模型会改变层数、hidden size、GQA 比例、词表和权重共享策略。"
    },
    token: {
      title: "输入 Token",
      type: "DATA",
      description: "tokenizer 输出的整数 ID。它是 Embedding 表的行地址，还不是模型能计算的连续向量。",
      input: "文本片段",
      output: "token ids [B,S]",
      weight: "无",
      quant: "不是 weight-only 量化对象",
      manual: "本页跟踪的是一次前向数据流。W8A16 里的 A 指 activation；输入 token id 本身是整数索引，不等同于 INT8 activation。"
    },
    embedding: {
      title: "Token Embedding",
      type: "LOOKUP",
      description: "按 token id 从词向量表中取行，得到每个 token 的 hidden state。",
      input: "token ids [B,S]",
      output: "hidden [B,S,H]",
      weight: "[V,H] 查找表",
      quant: "生产系统可量化，但本 Notebook 聚焦 Linear",
      manual: "Embedding 也可能很大，但它的核心操作是查表而不是 F.linear。本实验只把 Transformer 内的大型 Linear 投影视为当前改造对象。"
    },
    block: {
      title: "Transformer Block",
      type: "REPEATED UNIT",
      description: "Pre-Norm Attention 与 Pre-Norm SwiGLU MLP 两个残差子层。真实模型会堆叠很多个 Block。",
      input: "hidden [B,S,H]",
      output: "hidden [B,S,H]",
      weight: "7 个主要 Linear 投影 / Block",
      quant: "进入内部逐台标记",
      manual: "典型顺序是 h=x+Attention(RMSNorm(x))，out=h+MLP(RMSNorm(h))。Residual 主干让 shape 始终保持 [B,S,H]。"
    },
    rms1: normNode("Attention 前 RMSNorm"),
    rms2: normNode("MLP 前 RMSNorm"),
    final_norm: normNode("输出 RMSNorm"),
    residual1: residualNode("Attention Residual"),
    residual2: residualNode("MLP Residual"),
    rope: {
      title: "RoPE + Attention",
      type: "ATTENTION CORE",
      description: "RoPE 旋转 Q/K 后计算注意力；它操作 activation，不保存 [out,in] 的大型 Linear 权重。",
      input: "Q/K/V activations",
      output: "context [B,S,H]",
      weight: "RoPE 通常无可学习大矩阵",
      quant: "不是本页 weight-only 对象",
      manual: "大型权重在它前后的 q/k/v/o_proj 中。不要因为一个算子计算复杂，就把它误判为主要权重仓。"
    },
    silu: {
      title: "SiLU(gate) ⊙ up",
      type: "ACTIVATION",
      description: "对 gate 分支做 SiLU，再与 up 分支逐元素相乘。这里处理 activation，不保存大型权重。",
      input: "[B,S,I] + [B,S,I]",
      output: "[B,S,I]",
      weight: "无",
      quant: "不是本页 weight-only 对象",
      manual: "SwiGLU 的大矩阵在 gate_proj、up_proj、down_proj；SiLU 与逐元素乘法本身没有二维权重仓。"
    },
    logits: {
      title: "输出 Logits",
      type: "DATA",
      description: "每个位置对整个词表的未归一化分数，随后可用于采样下一个 token。",
      input: "hidden [B,S,H]",
      output: "logits [B,S,V]",
      weight: "数据本身无权重",
      quant: "不是权重",
      manual: "logits 是 activation。Weight-only 量化压缩的是产生 logits 的 LM Head 权重，而不是把 logits 当作模型参数存储。"
    },
    q_proj: linearNode("q_proj", "H", "H", "生成 Query；每个 attention head 用它询问当前位置需要什么信息。"),
    k_proj: linearNode("k_proj", "H", "KV", "生成 Key；GQA 下输出宽度可小于 H。"),
    v_proj: linearNode("v_proj", "H", "KV", "生成 Value；GQA 下与 Key 共享较少的 KV heads。"),
    o_proj: linearNode("o_proj", "H", "H", "合并 attention heads 后投影回 residual 主干宽度 H。"),
    gate_proj: linearNode("gate_proj", "H", "I", "生成 SwiGLU 的可学习门控分支。"),
    up_proj: linearNode("up_proj", "H", "I", "把 hidden state 升到中间维度 I，提供内容分支。"),
    down_proj: linearNode("down_proj", "I", "H", "把门控后的中间表示降回 residual 主干宽度 H。"),
    lm_head: linearNode("LM Head", "H", "V", "把最后 hidden state 投影到词表 logits；某些模型会与 Embedding 共享权重。")
  };

  function normNode(title) {
    return {
      title,
      type: "NORMALIZATION",
      description: "按 hidden 维度的 RMS 缩放 activation，稳定数值尺度。",
      input: "hidden [B,S,H]",
      output: "hidden [B,S,H]",
      weight: "通常仅 H 个缩放参数",
      quant: "参数很小，不是主要压缩目标",
      manual: "RMSNorm 有可学习 weight，但只有 [H]，而 Linear 常有 [H,H] 或 [I,H]。有参数不等于它是显存主仓库。"
    };
  }

  function residualNode(title) {
    return {
      title,
      type: "ELEMENTWISE ADD",
      description: "把支路输出加回主干，让信息和梯度拥有直接通路。",
      input: "两个 [B,S,H]",
      output: "hidden [B,S,H]",
      weight: "无",
      quant: "不是权重",
      manual: "Residual Add 可能消耗带宽和计算，但它没有持久化权重矩阵。当前任务的 W8 指模型参数存储，不是所有中间 activation。"
    };
  }

  function linearNode(title, inputDim, outputDim, description) {
    return {
      title,
      type: "LINEAR PROJECTION",
      description,
      input: `[B,S,${inputDim}]`,
      output: `[B,S,${outputDim}]`,
      weight: `[${outputDim},${inputDim}]`,
      quant: "适合当前 weight-only INT8 改造",
      quantizable: true,
      inputDim,
      outputDim,
      manual: "PyTorch Linear 保存 [out_features,in_features] 权重，并执行 y=xWᵀ+b。量化只改变权重的存储表示；输入 activation 仍保持浮点。"
    };
  }

  function roundToEven(value) {
    if (!Number.isFinite(value)) return value;
    const floor = Math.floor(value);
    const fraction = value - floor;
    if (Math.abs(fraction - 0.5) <= Number.EPSILON * Math.max(1, Math.abs(value)) * 4) {
      return Math.abs(floor % 2) === 0 ? floor : floor + 1;
    }
    return Math.round(value);
  }

  function absmaxQuantize(values, zeroGuard) {
    const clean = values.map(Number);
    const absmax = clean.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
    const safeAbsmax = absmax === 0 && zeroGuard ? 1e-8 : absmax;
    const scale = 127 / safeAbsmax;
    const quantized = clean.map((value) => {
      const scaled = value * scale;
      if (!Number.isFinite(scaled)) return Number.NaN;
      return Math.max(-128, Math.min(127, roundToEven(scaled)));
    });
    const dequantized = quantized.map((value) => (
      Number.isFinite(value) && Number.isFinite(scale) ? value / scale : Number.NaN
    ));
    const errors = dequantized.map((value, index) => (
      Number.isFinite(value) ? Math.abs(value - clean[index]) : Number.NaN
    ));
    const finiteErrors = errors.filter(Number.isFinite);
    const mae = finiteErrors.length === errors.length && errors.length
      ? finiteErrors.reduce((sum, value) => sum + value, 0) / errors.length
      : Number.NaN;
    return { absmax, safeAbsmax, scale, quantized, dequantized, errors, mae };
  }

  function dequantize(quantized, scale) {
    return quantized.map((row) => row.map((value) => value / scale));
  }

  function linear(inputs, weight, bias) {
    return inputs.map((inputRow) => weight.map((weightRow, outputIndex) => {
      const dot = inputRow.reduce((sum, value, inputIndex) => sum + value * weightRow[inputIndex], 0);
      return dot + (bias ? bias[outputIndex] : 0);
    }));
  }

  function cosineSimilarity(a, b) {
    const flatA = a.flat(Infinity);
    const flatB = b.flat(Infinity);
    if (flatA.length !== flatB.length || !flatA.length) return Number.NaN;
    const dot = flatA.reduce((sum, value, index) => sum + value * flatB[index], 0);
    const normA = Math.sqrt(flatA.reduce((sum, value) => sum + value * value, 0));
    const normB = Math.sqrt(flatB.reduce((sum, value) => sum + value * value, 0));
    return dot / (normA * normB);
  }

  function memoryBytes(elements, bits) {
    return elements * bits / 8;
  }

  function flattenMatrix(matrix) {
    return matrix.reduce((values, row) => values.concat(row), []);
  }

  function quantizeMatrix(matrix, zeroGuard) {
    const result = absmaxQuantize(flattenMatrix(matrix), zeroGuard);
    let cursor = 0;
    const quantized = matrix.map((row) => row.map(() => result.quantized[cursor++]));
    return { ...result, quantized };
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "—";
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KiB`;
    return `${Math.round(bytes)} B`;
  }

  function resolveDim(symbol, preset) {
    if (symbol === "H") return preset.hidden;
    if (symbol === "KV") return preset.kv;
    if (symbol === "I") return preset.intermediate;
    if (symbol === "V") return preset.vocab;
    return Number(symbol);
  }

  function weightDims(nodeId, preset) {
    const node = MODEL_GRAPH[nodeId];
    if (!node || !node.quantizable) return null;
    return [resolveDim(node.outputDim, preset), resolveDim(node.inputDim, preset)];
  }

  function linearWeightCount(preset) {
    const projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
    const perBlock = projections.reduce((sum, id) => {
      const [outDim, inDim] = weightDims(id, preset);
      return sum + outDim * inDim;
    }, 0);
    const [lmOut, lmIn] = weightDims("lm_head", preset);
    return perBlock * preset.layers + lmOut * lmIn;
  }

  function init(doc) {
    const one = (selector, scope) => (scope || doc).querySelector(selector);
    const all = (selector, scope) => [...(scope || doc).querySelectorAll(selector)];
    const gameKey = "pytorch-v2-level-25-factory-v1";
    const checkpointKey = "pytorch-v2-level-25-checkpoints";
    const homeworkKey = "pytorch-v2-level-25-homework";
    const predictKey = "pytorch-v2-level-25-predicts";
    const completeKey = "pytorch-levels-complete-v2";
    const checkpointMission = {
      2: "w8a16-range-and-scale",
      3: "w8a16-round-clamp-cast",
      4: "w8a16-dequant-linear"
    };

    const defaultState = {
      version: 1,
      mode: "guided",
      preset: "toy",
      view: 0,
      lens: "flow",
      mission: 0,
      selectedNode: "model",
      selectedLinear: "q_proj",
      completed: [],
      unfolded: false,
      compressed: false,
      calibrated: false,
      zeroSafe: false,
      zeroMode: false,
      zeroGuard: false,
      outlier: 3,
      quantPipeline: [],
      runtimePipeline: [],
      lastCosine: null,
      outputShapeOk: false,
      shipped: false
    };

    function readJson(key, fallback) {
      try {
        const value = JSON.parse(localStorage.getItem(key));
        return value == null ? fallback : value;
      } catch (_) {
        return fallback;
      }
    }

    const loaded = readJson(gameKey, {});
    const state = { ...defaultState, ...loaded };
    state.completed = Array.isArray(state.completed) ? state.completed : [];
    state.quantPipeline = Array.isArray(state.quantPipeline) ? state.quantPipeline : [];
    state.runtimePipeline = Array.isArray(state.runtimePipeline) ? state.runtimePipeline : [];

    const oldCheckpoints = new Set(readJson(checkpointKey, []));
    Object.entries(checkpointMission).forEach(([mission, checkpoint]) => {
      if (oldCheckpoints.has(checkpoint) && !state.completed.includes(Number(mission))) {
        state.completed.push(Number(mission));
      }
    });

    function save() {
      localStorage.setItem(gameKey, JSON.stringify(state));
    }

    function setConsole(kind, message, label) {
      const consoleBox = one("#factory-console");
      consoleBox.className = `mf-console${kind ? ` ${kind}` : ""}`;
      one("span", consoleBox).textContent = label || (kind === "warn" ? "FAULT" : kind === "good" ? "PASS" : "SYSTEM");
      one("p", consoleBox).textContent = message;
    }

    function setFeedback(id, kind, message) {
      const element = one(id);
      if (!element) return;
      element.className = `mf-feedback${kind ? ` ${kind}` : ""}`;
      element.textContent = message;
    }

    function completeMission(mission, message) {
      if (!state.completed.includes(mission)) state.completed.push(mission);
      const checkpoint = checkpointMission[mission];
      if (checkpoint) {
        const checkpoints = new Set(readJson(checkpointKey, []));
        checkpoints.add(checkpoint);
        localStorage.setItem(checkpointKey, JSON.stringify([...checkpoints]));
      }
      save();
      renderMissionRail();
      setConsole("good", message, `MISSION ${String(mission).padStart(2, "0")} COMPLETE`);
    }

    function setMission(mission) {
      state.mission = Math.max(0, Math.min(5, mission));
      all("[data-mission-panel]").forEach((panel) => {
        panel.classList.toggle("active", Number(panel.dataset.missionPanel) === state.mission);
      });
      all("[data-mission-tab]").forEach((button) => {
        button.classList.toggle("active", Number(button.dataset.missionTab) === state.mission);
      });
      save();
    }

    function setView(view) {
      state.view = Math.max(0, Math.min(4, Number(view)));
      one("#factory-world").dataset.view = String(state.view);
      const titles = [
        ["MODEL / OVERVIEW", "Decoder Transformer 工厂"],
        ["MODEL / BLOCK 01", "Transformer Block 剖面"],
        ["MODEL / BLOCK / LINEAR", `${MODEL_GRAPH[state.selectedLinear].title} 机器`],
        ["MODEL / BLOCK / LINEAR / WEIGHT", "二维权重矩阵"],
        ["MODEL / BLOCK / LINEAR / WEIGHT / BYTE", "单个权重的存储单元"]
      ];
      one("#view-path").textContent = titles[state.view][0];
      one("#view-title").textContent = titles[state.view][1];
      all("[data-view]").forEach((button) => button.classList.toggle("active", Number(button.dataset.view) === state.view));
      one("#zoom-out").disabled = state.view === 0;
      one("#zoom-in").disabled = state.view === 4;
      const hints = [
        "点击 Block，进入模型内部",
        "选择一个橙色 Linear 投影继续深入",
        "拆开转子里的二维权重矩阵",
        "点击矩阵单元，看见真实存储字节",
        "你已经到达模型参数的最小存储尺度"
      ];
      one("#zoom-hint").textContent = hints[state.view];
      if (state.view === 2) selectNode(state.selectedLinear, false);
      renderLens();
      save();
    }

    function setLens(lens) {
      state.lens = lens;
      one("#factory-world").dataset.lens = lens;
      all("[data-lens]").forEach((button) => button.classList.toggle("active", button.dataset.lens === lens));
      renderLens();
      save();
    }

    function selectNode(nodeId, updateView) {
      const node = MODEL_GRAPH[nodeId];
      if (!node) return;
      state.selectedNode = nodeId;
      if (node.quantizable) state.selectedLinear = nodeId;
      all("[data-node]").forEach((element) => element.classList.toggle("selected", element.dataset.node === nodeId));
      one("#node-type").textContent = node.type;
      one("#node-title").textContent = node.title;
      one("#node-description").textContent = node.description;
      one("#node-input").textContent = resolveShape(node.input, PRESETS[state.preset]);
      one("#node-output").textContent = resolveShape(node.output, PRESETS[state.preset]);
      one("#node-weight").textContent = resolveShape(node.weight, PRESETS[state.preset]);
      one("#node-quant").textContent = node.quant;
      one("#quant-badge").textContent = node.quantizable ? "INT8 候选仓" : "非主要目标";
      one("#quant-badge").classList.toggle("yes", Boolean(node.quantizable));
      one("#manual-content").innerHTML = `<p>${node.manual}</p>${node.quantizable
        ? "<ul><li>权重长期以 INT8 存储。</li><li>本 Notebook 前向时转回 x.dtype，并除以 scale。</li><li>生产融合内核可能不显式创建完整反量化矩阵。</li></ul>"
        : ""}`;
      if (node.quantizable) renderLinear(nodeId);
      if (updateView) {
        if (nodeId === "block") setView(1);
        else if (node.quantizable) setView(2);
      }
      renderLens();
      save();
    }

    function resolveShape(text, preset) {
      if (!text) return "—";
      return String(text)
        .replace(/\bKV\b/g, String(preset.kv))
        .replace(/\bH\b/g, String(preset.hidden))
        .replace(/\bI\b/g, String(preset.intermediate))
        .replace(/\bV\b/g, String(preset.vocab));
    }

    function renderLinear(nodeId) {
      const preset = PRESETS[state.preset];
      const node = MODEL_GRAPH[nodeId];
      if (!node || !node.quantizable) return;
      const inDim = resolveDim(node.inputDim, preset);
      const outDim = resolveDim(node.outputDim, preset);
      one("#linear-input-shape").textContent = `[B,S,${inDim}]`;
      one("#linear-weight-shape").textContent = `[${outDim},${inDim}]`;
      one("#linear-output-shape").textContent = `[B,S,${outDim}]`;
      one("#linear-equation-note").textContent = `最后一维 ${inDim} 被 out_features=${outDim} 替换`;
      all("[data-linear]").forEach((button) => button.classList.toggle("active", button.dataset.linear === nodeId));
      one("#matrix-shape").textContent = `[${outDim},${inDim}]`;
      one("#matrix-params").textContent = formatNumber(outDim * inDim);
      one("#matrix-fp32").textContent = formatBytes(memoryBytes(outDim * inDim, 32));
      one("#matrix-int8").textContent = formatBytes(memoryBytes(outDim * inDim, 8));
      buildMatrix();
    }

    function buildMatrix() {
      const matrix = one("#weight-matrix");
      if (matrix.children.length) return;
      const values = [-.8, 1.5, -3, 2.5, 0, .25, -.5, 1];
      for (let index = 0; index < 64; index += 1) {
        const cell = doc.createElement("button");
        const value = values[index % values.length];
        cell.type = "button";
        cell.textContent = String(value);
        cell.title = `W[${Math.floor(index / 8)},${index % 8}] = ${value}`;
        cell.addEventListener("click", () => {
          all("button", matrix).forEach((item) => item.classList.remove("active"));
          cell.classList.add("active");
          setConsole("", `选中 ${cell.title}。FP32 用 4 byte；量化后对应 1 个 INT8 code 与共享 scale。`, "WEIGHT CELL");
          setView(4);
        });
        matrix.appendChild(cell);
      }
    }

    function renderLens() {
      const node = MODEL_GRAPH[state.selectedNode] || MODEL_GRAPH.model;
      const preset = PRESETS[state.preset];
      const readout = one("#lens-readout");
      if (state.lens === "flow") {
        readout.innerHTML = `<b>数据流镜片：</b>${node.description}`;
      } else if (state.lens === "shape") {
        readout.innerHTML = `<b>Shape 镜片：</b>${resolveShape(node.input, preset)} → ${resolveShape(node.output, preset)}；权重 ${resolveShape(node.weight, preset)}。`;
      } else if (state.lens === "memory") {
        const dims = weightDims(state.selectedNode, preset);
        readout.innerHTML = dims
          ? `<b>内存镜片：</b>${formatNumber(dims[0] * dims[1])} 个权重；FP32 ${formatBytes(memoryBytes(dims[0] * dims[1], 32))} → INT8 ${formatBytes(memoryBytes(dims[0] * dims[1], 8))}。`
          : `<b>内存镜片：</b>${node.weight}。它不是当前实验中的大型 Linear 权重仓。`;
      } else {
        const calibration = absmaxQuantize(currentSample(), state.zeroGuard);
        readout.innerHTML = `<b>数值镜片：</b>当前样本 absmax=${formatNumeric(calibration.absmax)}，scale=${formatNumeric(calibration.scale)}；per-tensor 意味着所有值共享这一个 scale。`;
      }
    }

    function renderPreset() {
      const preset = PRESETS[state.preset];
      all("[data-preset]").forEach((button) => button.classList.toggle("active", button.dataset.preset === state.preset));
      all("[data-layers]").forEach((element) => { element.textContent = String(preset.layers); });
      const elements = linearWeightCount(preset);
      const fp32 = memoryBytes(elements, 32);
      const int8 = memoryBytes(elements, 8);
      one("#fp32-total").textContent = formatBytes(fp32);
      one("#int8-total").textContent = formatBytes(int8);
      one("#preset-note").textContent = preset.note;
      one("#memory-gauge").style.width = state.shipped ? "24%" : "94%";
      one(".mf-crisis").classList.toggle("compact", state.shipped);
      one("#budget-state").textContent = state.shipped ? "WITHIN BUDGET" : "OVER BUDGET";
      renderLinear(state.selectedLinear);
      selectNode(state.selectedNode, false);
    }

    function renderMissionRail() {
      all("[data-mission-tab]").forEach((button) => {
        const mission = Number(button.dataset.missionTab);
        button.classList.toggle("complete", state.completed.includes(mission) || (mission === 5 && state.shipped));
      });
    }

    function currentSample() {
      if (state.zeroMode) return [0, 0, 0, 0, 0];
      return [-0.8, 1.5, -Number(state.outlier), 2.5, 0];
    }

    function renderCalibration(result) {
      const values = currentSample();
      const maxAbs = values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
      one("#sample-values").innerHTML = values.map((value) => (
        `<span class="${Math.abs(value) === maxAbs && maxAbs !== 0 ? "absmax" : ""}">${value.toFixed(2)}</span>`
      )).join("");
      one("#outlier-value").textContent = Number(state.outlier).toFixed(1);
      one("#outlier-slider").value = String(state.outlier);
      one("#zero-mode").checked = state.zeroMode;
      one("#zero-guard").checked = state.zeroGuard;
      if (!result) {
        one("#absmax-readout").textContent = "等待扫描";
        one("#scale-readout").textContent = "—";
        one("#sample-quant-readout").textContent = "—";
        one("#error-average").textContent = "MAE —";
        one("#error-bars").innerHTML = "";
        return;
      }
      one("#absmax-readout").textContent = formatNumeric(result.absmax);
      one("#scale-readout").textContent = formatNumeric(result.scale);
      one("#sample-quant-readout").textContent = formatNumeric(result.quantized[3]);
      one("#dial-needle").style.transform = `rotate(${Number.isFinite(result.scale) ? Math.min(70, -70 + Math.log10(result.scale + 1) * 75) : 70}deg)`;
      one("#error-average").textContent = `MAE ${formatNumeric(result.mae)}`;
      const finiteErrors = result.errors.filter(Number.isFinite);
      const maxError = Math.max(0.0001, ...finiteErrors);
      one("#error-bars").innerHTML = result.errors.map((error, index) => {
        const height = Number.isFinite(error) ? 12 + error / maxError * 105 : 120;
        return `<span style="height:${height}px"><b>${Number.isFinite(error) ? error.toFixed(4) : "NaN"}<br>x${index}</b></span>`;
      }).join("");
      renderLens();
    }

    function renderQuantPipeline() {
      all("[data-slot]").forEach((slot, index) => {
        const moduleId = state.quantPipeline[index];
        slot.textContent = moduleId ? moduleLabel(moduleId) : String(index + 1).padStart(2, "0");
        slot.classList.toggle("filled", Boolean(moduleId));
      });
      all("[data-module]").forEach((button) => {
        button.disabled = state.quantPipeline.includes(button.dataset.module);
      });
    }

    function renderRuntimePipeline() {
      const slots = all("span", one("#runtime-chain"));
      slots.slice(1).forEach((slot, index) => {
        const moduleId = state.runtimePipeline[index];
        slot.textContent = moduleId ? runtimeLabel(moduleId) : "?";
        slot.classList.toggle("filled", Boolean(moduleId));
      });
      all("[data-runtime]").forEach((button) => {
        button.disabled = state.runtimePipeline.includes(button.dataset.runtime);
      });
    }

    function moduleLabel(id) {
      return { scale: "× scale", round: "round", clamp: "clamp", cast: "cast int8" }[id] || id;
    }

    function runtimeLabel(id) {
      return { cast: "to(x.dtype)", divide: "÷ scale", linear: "F.linear" }[id] || id;
    }

    function formatNumeric(value) {
      if (Number.isNaN(value)) return "NaN";
      if (value === Infinity) return "Inf";
      if (value === -Infinity) return "-Inf";
      if (!Number.isFinite(value)) return "—";
      if (Number.isInteger(value)) return String(value);
      return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
    }

    function formatNumber(value) {
      return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(value);
    }

    function sameOrder(actual, expected) {
      return actual.length === expected.length && actual.every((item, index) => item === expected[index]);
    }

    function computeReactor() {
      const matrixQuant = quantizeMatrix(MATRIX_SAMPLE, true);
      const restored = dequantize(matrixQuant.quantized, matrixQuant.scale);
      const fpOutput = linear(MATRIX_INPUT, MATRIX_SAMPLE, MATRIX_BIAS);
      const quantOutput = linear(MATRIX_INPUT, restored, MATRIX_BIAS);
      const cosine = cosineSimilarity(fpOutput, quantOutput);
      return { fpOutput, quantOutput, cosine, scale: matrixQuant.scale };
    }

    function renderReactor(result) {
      if (!result) return;
      one("#fp-output").textContent = compactMatrix(result.fpOutput);
      one("#quant-output").textContent = compactMatrix(result.quantOutput);
      one("#cosine-output").textContent = result.cosine.toFixed(6);
      one("#boss-cosine").textContent = result.cosine.toFixed(6);
    }

    function compactMatrix(matrix) {
      return `[${matrix.map((row) => `[${row.map((value) => value.toFixed(3)).join(",")}]`).join(",")}]`;
    }

    function renderBoss() {
      const missionReady = [1, 2, 3, 4].every((mission) => state.completed.includes(mission));
      const checks = {
        memory: state.compressed,
        scale: state.zeroSafe,
        cosine: Number(state.lastCosine) > 0.99,
        shape: state.outputShapeOk
      };
      Object.entries(checks).forEach(([id, passed]) => {
        one(`[data-boss="${id}"]`).classList.toggle("pass", Boolean(passed));
      });
      one("#launch-model").disabled = !missionReady;
      one(".boss-panel").classList.toggle("shipped", state.shipped);
      one("#notebook-bridge").hidden = !state.shipped;
      if (state.lastCosine) one("#boss-cosine").textContent = Number(state.lastCosine).toFixed(6);
    }

    function bindInteractions() {
      all("[data-mode]").forEach((button) => button.addEventListener("click", () => {
        state.mode = button.dataset.mode;
        all("[data-mode]").forEach((item) => item.classList.toggle("active", item.dataset.mode === state.mode));
        setConsole("", state.mode === "guided"
          ? "导览模式会按权重生命周期组织任务，但所有剖视层仍可自由访问。"
          : "自由剖视已开启：可从任意机器直接下钻，不影响任务进度。", "NAVIGATION");
        save();
      }));
      all("[data-preset]").forEach((button) => button.addEventListener("click", () => {
        state.preset = button.dataset.preset;
        renderPreset();
        save();
      }));
      all("[data-mission-tab]").forEach((button) => button.addEventListener("click", () => setMission(Number(button.dataset.missionTab))));
      all("[data-view]").forEach((button) => button.addEventListener("click", () => setView(Number(button.dataset.view))));
      all("[data-lens]").forEach((button) => button.addEventListener("click", () => setLens(button.dataset.lens)));
      all("[data-node]").forEach((element) => element.addEventListener("click", () => selectNode(element.dataset.node, true)));
      all("[data-linear]").forEach((button) => button.addEventListener("click", () => {
        state.selectedLinear = button.dataset.linear;
        selectNode(state.selectedLinear, false);
        setView(2);
      }));
      one("#zoom-out").addEventListener("click", () => setView(state.view - 1));
      one("#zoom-in").addEventListener("click", () => setView(state.view + 1));

      all("[data-hunt]").forEach((button) => button.addEventListener("click", () => {
        const choice = button.dataset.hunt;
        if (choice === "linear") {
          setFeedback("#hunt-feedback", "good", "正确。Linear 的 [out,in] 二维权重会随 hidden size 平方级增长，是本次 weight-only 量化的主要仓库。");
          completeMission(0, "找到大型 Linear 权重仓。RMSNorm 有少量参数，Residual Add 没有参数。");
          selectNode("block", true);
        } else if (choice === "norm") {
          setFeedback("#hunt-feedback", "warn", "RMSNorm 确实有可学习缩放参数，但通常只有 H 个；Linear 常有 H×H 或 I×H 个，量级完全不同。");
          setConsole("warn", "目标有参数，但不是主仓库。比较 [H] 与 [H,H] 的参数量。", "MISIDENTIFIED");
        } else {
          setFeedback("#hunt-feedback", "warn", "Residual Add 会搬运并相加 activation，但它没有持久化权重矩阵，因此不是 W8 的压缩对象。");
          setConsole("warn", "计算过程复杂不等于拥有模型参数。Residual Add 的权重数是 0。", "NO WEIGHT");
        }
      }));

      one("#unfold-bytes").addEventListener("click", () => {
        state.unfolded = true;
        state.view = 4;
        one("#compress-byte").disabled = false;
        one("#autopsy-status").textContent = "FP32 = 4 byte 已展开";
        selectNode(state.selectedLinear, false);
        setView(4);
        setConsole("", "一个 FP32 权重由 32 bit 构成。现在把同一个量化 code 装入 8 bit 存储单元。", "BYTE VIEW");
        save();
      });
      one("#compress-byte").addEventListener("click", () => {
        if (!state.unfolded) return;
        state.compressed = true;
        one("#autopsy-status").textContent = "FP32 4 byte → INT8 1 byte";
        completeMission(1, "完成 4 byte → 1 byte 的权重存储改造。相对 FP16 则是 2 byte → 1 byte。");
        renderBoss();
      });

      one("#zero-mode").addEventListener("change", (event) => {
        state.zeroMode = event.target.checked;
        renderCalibration();
        save();
      });
      one("#zero-guard").addEventListener("change", (event) => {
        state.zeroGuard = event.target.checked;
        renderCalibration();
        save();
      });
      one("#outlier-slider").addEventListener("input", (event) => {
        state.outlier = Number(event.target.value);
        state.zeroMode = false;
        renderCalibration(absmaxQuantize(currentSample(), state.zeroGuard));
        save();
      });
      one("#restore-sample").addEventListener("click", () => {
        state.outlier = 3;
        state.zeroMode = false;
        renderCalibration();
        setFeedback("#calibration-feedback", "", "样本已恢复。扫描后要得到 absmax=3、scale=127/3、2.5→106。");
        save();
      });
      one("#run-calibration").addEventListener("click", () => {
        const result = absmaxQuantize(currentSample(), state.zeroGuard);
        renderCalibration(result);
        if (!Number.isFinite(result.scale)) {
          setFeedback("#calibration-feedback", "warn", "故障复现：absmax=0，127/0 得到 Inf；0×Inf 进一步产生 NaN。先安装零值保护，再扫描。");
          setConsole("warn", "zero_scale 不是有限值，Notebook 的 torch.isfinite 断言会失败。", "DIVIDE BY ZERO");
          return;
        }
        if (state.zeroMode) {
          state.zeroSafe = result.quantized.every((value) => value === 0);
          setFeedback("#calibration-feedback", "good", "全零边界已修复：scale 有限，量化结果仍全为 0。");
        } else if (Number(state.outlier) === 3 && result.quantized[3] === 106) {
          state.calibrated = true;
          setFeedback("#calibration-feedback", "good", "基准校准正确：absmax=3，scale=42.333…，2.5 被舍入为 106。现在切换全零输入验证保护。");
        } else {
          setFeedback("#calibration-feedback", "", "异常值实验正在运行：absmax 变大后 scale 变小，小权重分配到的整数刻度更少。恢复样本可继续基准任务。");
        }
        if (state.calibrated && state.zeroSafe) {
          completeMission(2, "基准样本与全零边界都通过。你已掌握 per-tensor absmax 标尺及其异常值代价。");
        }
        save();
        renderBoss();
      });

      all("[data-module]").forEach((button) => {
        button.addEventListener("click", () => appendQuantModule(button.dataset.module));
        button.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", button.dataset.module));
      });
      one("#quant-pipeline").addEventListener("dragover", (event) => event.preventDefault());
      one("#quant-pipeline").addEventListener("drop", (event) => {
        event.preventDefault();
        appendQuantModule(event.dataTransfer.getData("text/plain"));
      });
      one("#clear-quant-pipeline").addEventListener("click", () => {
        state.quantPipeline = [];
        one("#quant-trace").innerHTML = "<span>输入 2.5 / float</span>";
        renderQuantPipeline();
        save();
      });
      one("#run-quant-pipeline").addEventListener("click", runQuantPipeline);

      all("[data-runtime]").forEach((button) => button.addEventListener("click", () => {
        if (state.runtimePipeline.length >= 3 || state.runtimePipeline.includes(button.dataset.runtime)) return;
        state.runtimePipeline.push(button.dataset.runtime);
        renderRuntimePipeline();
        save();
      }));
      one("#clear-runtime").addEventListener("click", () => {
        state.runtimePipeline = [];
        renderRuntimePipeline();
        save();
      });
      one("#run-runtime").addEventListener("click", runRuntime);

      one("#launch-model").addEventListener("click", () => {
        const ready = state.compressed && state.zeroSafe && Number(state.lastCosine) > 0.99 && state.outputShapeOk
          && [1, 2, 3, 4].every((mission) => state.completed.includes(mission));
        if (!ready) {
          setFeedback("#boss-feedback", "warn", "上线检查未通过。查看未点亮的指标，并回到对应 Mission 修复。");
          return;
        }
        state.shipped = true;
        if (!state.completed.includes(5)) state.completed.push(5);
        const completedLevels = new Set(readJson(completeKey, []));
        completedLevels.add("25");
        localStorage.setItem(completeKey, JSON.stringify([...completedLevels]));
        save();
        renderBoss();
        renderPreset();
        renderMissionRail();
        setFeedback("#boss-feedback", "good", "上线成功：所有可量化 Linear 仓库已点亮。现在回 Notebook 完成 TODO 1–4。");
        setConsole("good", "模型进入预算。注意：本 Notebook 展示存储与反量化链路，真实 INT8 加速仍需要融合内核。", "DEPLOYED");
      });

      one("#reset-level").addEventListener("click", () => {
        if (!root.confirm("重置 Level 25 的模型工厂、闯关题和作业记录？此操作只影响本关。")) return;
        localStorage.removeItem(gameKey);
        localStorage.removeItem(checkpointKey);
        localStorage.removeItem(homeworkKey);
        localStorage.removeItem(predictKey);
        const completedLevels = new Set(readJson(completeKey, []));
        completedLevels.delete("25");
        localStorage.setItem(completeKey, JSON.stringify([...completedLevels]));
        root.location.reload();
      });
    }

    function appendQuantModule(moduleId) {
      if (!QUANT_ORDER.includes(moduleId) || state.quantPipeline.includes(moduleId) || state.quantPipeline.length >= 4) return;
      state.quantPipeline.push(moduleId);
      renderQuantPipeline();
      save();
    }

    function runQuantPipeline() {
      if (state.quantPipeline.length < 4) {
        setFeedback("#quant-feedback", "warn", "流水线尚未装满。缩放、舍入、截断和 dtype 转换四个动作缺一不可。");
        return;
      }
      if (!sameOrder(state.quantPipeline, QUANT_ORDER)) {
        let explanation = "顺序错误会改变数值或 dtype 契约。";
        if (state.quantPipeline.indexOf("cast") < 3) {
          explanation = "过早 cast：小数会先被截掉，后面的 round 已经无法恢复信息。";
        } else if (state.quantPipeline.indexOf("round") < state.quantPipeline.indexOf("scale")) {
          explanation = "过早 round：2.5 先变成 2，再乘 scale，结果不再是 106。";
        } else if (state.quantPipeline.indexOf("clamp") < state.quantPipeline.indexOf("scale")) {
          explanation = "过早 clamp：限制的是原始浮点范围，而不是映射后的 INT8 code。";
        }
        setFeedback("#quant-feedback", "warn", explanation);
        setConsole("warn", explanation, "PIPELINE ORDER");
        return;
      }
      const scale = 127 / 3;
      const scaled = 2.5 * scale;
      const rounded = roundToEven(scaled);
      const clamped = Math.max(-128, Math.min(127, rounded));
      one("#quant-trace").innerHTML = [
        "输入 2.5 / float",
        `× ${scale.toFixed(3)} = ${scaled.toFixed(3)}`,
        `round = ${rounded}`,
        `clamp = ${clamped}`,
        `${clamped} / int8`
      ].map((value) => `<span>${value}</span>`).join("");
      setFeedback("#quant-feedback", "good", "铸造成功：先在浮点域完成 ×scale、round、clamp，最后才切换到 torch.int8 存储。");
      completeMission(3, "量化流水线顺序正确，2.5 最终成为 INT8 code 106。");
      renderBoss();
    }

    function runRuntime() {
      if (!sameOrder(state.runtimePipeline, RUNTIME_ORDER)) {
        const explanation = state.runtimePipeline[0] === "divide"
          ? "int8 直接参与除法会模糊计算 dtype 契约；先转成 x.dtype，再恢复数值范围。"
          : state.runtimePipeline.includes("linear") && state.runtimePipeline.indexOf("linear") < 2
            ? "F.linear 启动太早：它拿到的仍是 INT8 code，而不是近似浮点权重。"
            : "正确链路是 to(x.dtype) → 除以 scale → F.linear。量化时乘，反量化时除。";
        setFeedback("#runtime-feedback", "warn", explanation);
        setConsole("warn", explanation, "REACTOR WIRING");
        return;
      }
      const result = computeReactor();
      renderReactor(result);
      state.lastCosine = result.cosine;
      state.outputShapeOk = result.quantOutput.length === 2 && result.quantOutput[0].length === 3;
      if (result.cosine > 0.99 && state.outputShapeOk) {
        setFeedback("#runtime-feedback", "good", `反应堆稳定：小矩阵输出余弦相似度 ${result.cosine.toFixed(6)}。Notebook 的大 shape 契约是 [2,10,128] → [2,10,64]。`);
        completeMission(4, "权重先跟随 x.dtype，再除以 scale，最终交给浮点 F.linear。");
      } else {
        setFeedback("#runtime-feedback", "warn", "相似度或 shape 未达标，请检查反量化方向和 Linear 权重布局。");
      }
      save();
      renderBoss();
    }

    function setupCanvas() {
      const canvas = one("#factory-canvas");
      const context = canvas.getContext && canvas.getContext("2d");
      if (!context || root.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      let width = 0;
      let height = 0;
      let points = [];
      function resize() {
        const ratio = Math.min(root.devicePixelRatio || 1, 2);
        width = root.innerWidth;
        height = root.innerHeight;
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        points = Array.from({ length: Math.min(52, Math.ceil(width / 26)) }, (_, index) => ({
          x: (index * 97) % width,
          y: (index * 173) % height,
          speed: .08 + (index % 7) * .025,
          size: 1 + index % 2
        }));
      }
      function draw() {
        context.clearRect(0, 0, width, height);
        context.fillStyle = "rgba(86,230,211,.42)";
        points.forEach((point) => {
          point.y -= point.speed;
          if (point.y < -4) point.y = height + 4;
          context.fillRect(point.x, point.y, point.size, point.size);
        });
        root.requestAnimationFrame(draw);
      }
      resize();
      root.addEventListener("resize", resize, { passive: true });
      draw();
    }

    all("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === state.mode));
    buildMatrix();
    bindInteractions();
    renderPreset();
    renderMissionRail();
    renderQuantPipeline();
    renderRuntimePipeline();
    renderCalibration();
    renderReactor(state.lastCosine ? computeReactor() : null);
    renderBoss();
    setMission(state.mission);
    setLens(state.lens);
    setView(state.view);
    selectNode(state.selectedNode, false);
    setupCanvas();

    if (state.unfolded) {
      one("#compress-byte").disabled = false;
      one("#autopsy-status").textContent = state.compressed ? "FP32 4 byte → INT8 1 byte" : "FP32 = 4 byte 已展开";
    }
    if (state.shipped) {
      setConsole(
        "good",
        "模型已在预算内上线。可继续自由剖视，或回到 Notebook 完成 TODO 1–4。",
        "DEPLOYED"
      );
    }

    return { state, MODEL_GRAPH };
  }

  return {
    init,
    MODEL_GRAPH,
    PRESETS,
    roundToEven,
    absmaxQuantize,
    dequantize,
    linear,
    cosineSimilarity,
    memoryBytes,
    weightDims,
    linearWeightCount,
    quantizeMatrix
  };
});
