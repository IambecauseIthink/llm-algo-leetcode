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
  const PROBE_TOKENS = ["量化", "模型", "压缩", "速度", "误差"];

  const PRESETS = {
    toy: {
      label: "玩具模型",
      hidden: 8,
      kv: 4,
      intermediate: 16,
      vocab: 32,
      layers: 2,
      budgetBytes: 1800,
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
      budgetBytes: 6.5 * 1024 ** 3,
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

  function float32Bytes(value) {
    const buffer = new ArrayBuffer(4);
    const view = new DataView(buffer);
    view.setFloat32(0, Number(value), false);
    return Array.from({ length: 4 }, (_, index) => view.getUint8(index));
  }

  function byteBits(value) {
    return (Number(value) & 0xff).toString(2).padStart(8, "0");
  }

  function deploymentGroups(preset) {
    const h = preset.hidden;
    const kv = preset.kv;
    const intermediate = preset.intermediate;
    const attention = preset.layers * (
      h * h + kv * h + kv * h + h * h
    );
    const mlp = preset.layers * (
      intermediate * h + intermediate * h + h * intermediate
    );
    const lmHead = preset.vocab * h;
    const norm = (preset.layers * 2 + 1) * h;
    return { attention, mlp, lm_head: lmHead, norm };
  }

  function scaleCount(target, elements, preset, granularity) {
    if (granularity === "group") return Math.ceil(elements / 64);
    if (granularity === "channel") {
      if (target === "attention") {
        return preset.layers * (preset.hidden * 2 + preset.kv * 2);
      }
      if (target === "mlp") {
        return preset.layers * (preset.intermediate * 2 + preset.hidden);
      }
      if (target === "lm_head") return preset.vocab;
    }
    if (target === "attention") return preset.layers * 4;
    if (target === "mlp") return preset.layers * 3;
    return 1;
  }

  function makeProbeMatrix(rows, columns, phase, amplitude) {
    return Array.from({ length: rows }, (_, row) => (
      Array.from({ length: columns }, (_, column) => {
        const wave = Math.sin((row + 1) * 1.73 + (column + 1) * 0.91 + phase);
        const cross = Math.cos((row + 1) * (column + 2) * 0.37 + phase * 0.6);
        return Number(((wave + cross * 0.35) * amplitude).toFixed(7));
      })
    ));
  }

  const TOY_PROBE = {
    input: [0.8, -1.1, 0.35, 1.6],
    attention: makeProbeMatrix(4, 4, 0.2, 0.46),
    gate: makeProbeMatrix(8, 4, 0.9, 0.38),
    up: makeProbeMatrix(8, 4, 1.7, 0.41),
    down: makeProbeMatrix(4, 8, 2.3, 0.34),
    norm1: [1.06, 0.93, 1.11, 0.89],
    norm2: [0.96, 1.08, 0.91, 1.04],
    lmHead: makeProbeMatrix(5, 4, 3.1, 0.52),
    lmBias: [0.02, -0.04, 0.01, 0.05, -0.02]
  };
  const TOY_PROBE_INPUTS = [
    TOY_PROBE.input,
    [-1.4, 0.2, 1.05, -0.55],
    [0.18, 1.75, -0.9, 0.4],
    [2.1, -0.35, -1.2, 0.65],
    [-0.7, -1.3, 1.9, 0.15],
    [0.05, 0.12, -0.08, 0.2],
    [1.3, 0.85, -1.6, -0.45],
    [-2.0, 1.1, 0.3, 0.95]
  ];
  const TOY_QUALITY_LIMITS = {
    worstCosine: 0.99,
    maxLogitError: 0.1,
    meanAbsoluteError: 0.02
  };
  const BLOCK_ASSEMBLY = [
    { id: "rms1", label: "RMSNorm", role: "Attention 前归一化", shape: "[B,S,H] → [B,S,H]", weight: "[H] 小权重" },
    { id: "attention", label: "Attention", role: "Q/K/V + RoPE + O", shape: "[B,S,H] → [B,S,H]", weight: "4 类 Linear 大矩阵" },
    { id: "residual1", label: "Residual +", role: "加回 Block 原输入", shape: "[B,S,H] + [B,S,H]", weight: "0 个权重" },
    { id: "rms2", label: "RMSNorm", role: "MLP 前归一化", shape: "[B,S,H] → [B,S,H]", weight: "[H] 小权重" },
    { id: "mlp", label: "SwiGLU MLP", role: "gate/up/down", shape: "H → I → H", weight: "3 个 Linear 大矩阵" },
    { id: "residual2", label: "Residual +", role: "加回 Attention 后主干", shape: "[B,S,H] + [B,S,H]", weight: "0 个权重" }
  ];

  function quantizeValuesAtBits(values, bits) {
    const qmax = 2 ** (bits - 1) - 1;
    const qmin = -(2 ** (bits - 1));
    const absmax = values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
    if (absmax === 0) return values.map(() => 0);
    const scale = qmax / absmax;
    return values.map((value) => (
      Math.max(qmin, Math.min(qmax, roundToEven(value * scale))) / scale
    ));
  }

  function quantizeMatrixAtBits(matrix, bits, granularity) {
    if (granularity === "channel") {
      return matrix.map((row) => quantizeValuesAtBits(row, bits));
    }
    if (granularity === "group") {
      const flat = flattenMatrix(matrix);
      const groupSize = 4;
      const restored = [];
      for (let start = 0; start < flat.length; start += groupSize) {
        restored.push(...quantizeValuesAtBits(flat.slice(start, start + groupSize), bits));
      }
      let cursor = 0;
      return matrix.map((row) => row.map(() => restored[cursor++]));
    }
    const restored = quantizeValuesAtBits(flattenMatrix(matrix), bits);
    let cursor = 0;
    return matrix.map((row) => row.map(() => restored[cursor++]));
  }

  function matrixForScheme(matrix, scheme, granularity) {
    if (scheme === "int8") return quantizeMatrixAtBits(matrix, 8, granularity);
    if (scheme === "int4") return quantizeMatrixAtBits(matrix, 4, granularity);
    return matrix.map((row) => row.slice());
  }

  function vectorForScheme(vector, scheme) {
    return scheme === "int8" ? quantizeValuesAtBits(vector, 8) : vector.slice();
  }

  function matVec(matrix, vector) {
    return matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
  }

  function rmsNorm(vector, weight) {
    const rms = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0) / vector.length + 1e-6);
    return vector.map((value, index) => value / rms * weight[index]);
  }

  function softmax(values) {
    const maxValue = Math.max(...values);
    const exponentials = values.map((value) => Math.exp(value - maxValue));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / total);
  }

  function runToyProbe(plan, input) {
    const granularity = plan.granularity || "tensor";
    const attentionWeight = matrixForScheme(TOY_PROBE.attention, plan.attention || "fp32", granularity);
    const gateWeight = matrixForScheme(TOY_PROBE.gate, plan.mlp || "fp32", granularity);
    const upWeight = matrixForScheme(TOY_PROBE.up, plan.mlp || "fp32", granularity);
    const downWeight = matrixForScheme(TOY_PROBE.down, plan.mlp || "fp32", granularity);
    const lmHeadWeight = matrixForScheme(TOY_PROBE.lmHead, plan.lm_head || "fp32", granularity);
    const norm1Weight = vectorForScheme(TOY_PROBE.norm1, plan.norm || "fp32");
    const norm2Weight = vectorForScheme(TOY_PROBE.norm2, plan.norm || "fp32");

    const trace = [];
    let hidden = (input || TOY_PROBE.input).slice();
    let residual = hidden.slice();
    let normalized = rmsNorm(hidden, norm1Weight);
    trace.push(normalized.slice());
    const attentionOutput = matVec(attentionWeight, normalized);
    trace.push(attentionOutput.slice());
    hidden = residual.map((value, index) => value + attentionOutput[index]);
    trace.push(hidden.slice());
    if (plan.activation === "a8") hidden = quantizeValuesAtBits(hidden, 8);

    residual = hidden.slice();
    normalized = rmsNorm(hidden, norm2Weight);
    trace.push(normalized.slice());
    const gate = matVec(gateWeight, normalized).map((value) => value / (1 + Math.exp(-value)));
    const up = matVec(upWeight, normalized);
    const mlpHidden = gate.map((value, index) => value * up[index]);
    const mlpOutput = matVec(downWeight, mlpHidden);
    trace.push(mlpOutput.slice());
    hidden = residual.map((value, index) => value + mlpOutput[index]);
    trace.push(hidden.slice());
    if (plan.activation === "a8") hidden = quantizeValuesAtBits(hidden, 8);

    const logits = matVec(lmHeadWeight, hidden).map((value, index) => value + TOY_PROBE.lmBias[index]);
    const probabilities = softmax(logits);
    const tokens = PROBE_TOKENS.map((token, index) => ({
      token,
      probability: probabilities[index],
      logit: logits[index]
    })).sort((a, b) => b.probability - a.probability);
    return { hidden, logits, probabilities, tokens, trace };
  }

  function evaluateToyProbeSuite(plan) {
    const baselinePlan = {
      mlp: "fp32",
      attention: "fp32",
      lm_head: "fp32",
      norm: "fp32",
      activation: "a16",
      granularity: "tensor"
    };
    const samples = TOY_PROBE_INPUTS.map((input) => {
      const baseline = runToyProbe(baselinePlan, input);
      const current = runToyProbe(plan, input);
      const absoluteErrors = baseline.logits.map((value, index) => Math.abs(value - current.logits[index]));
      return {
        baseline,
        current,
        cosine: cosineSimilarity(baseline.logits, current.logits),
        maxLogitError: Math.max(...absoluteErrors),
        absoluteErrors,
        top1Changed: baseline.tokens[0].token !== current.tokens[0].token
      };
    });
    const allAbsoluteErrors = samples.flatMap((sample) => sample.absoluteErrors);
    const worstCosine = Math.min(...samples.map((sample) => sample.cosine));
    const meanCosine = samples.reduce((sum, sample) => sum + sample.cosine, 0) / samples.length;
    const maxLogitError = Math.max(...samples.map((sample) => sample.maxLogitError));
    const meanAbsoluteError = allAbsoluteErrors.reduce((sum, value) => sum + value, 0) / allAbsoluteErrors.length;
    const top1Flips = samples.filter((sample) => sample.top1Changed).length;
    return {
      samples,
      worstCosine,
      meanCosine,
      maxLogitError,
      meanAbsoluteError,
      top1Flips,
      sampleCount: samples.length,
      baselineTokens: samples[0].baseline.tokens,
      tokens: samples[0].current.tokens
    };
  }

  function simulateDeployment(plan, preset) {
    const groups = deploymentGroups(preset);
    const bytesPerElement = { fp32: 4, int8: 1, int4: 0.5 };
    const granularity = plan.granularity || "tensor";
    const memoryByGroup = {};
    ["mlp", "attention", "lm_head"].forEach((target) => {
      const scheme = plan[target] || "fp32";
      const weightBytes = groups[target] * bytesPerElement[scheme];
      const metadata = scheme === "fp32"
        ? 0
        : scaleCount(target, groups[target], preset, granularity) * 4;
      memoryByGroup[target] = weightBytes + metadata;
    });
    memoryByGroup.norm = groups.norm * (plan.norm === "int8" ? 1 : 4);
    const baselineBytes = (
      groups.mlp + groups.attention + groups.lm_head + groups.norm
    ) * 4;
    const currentBytes = Object.values(memoryByGroup).reduce((sum, value) => sum + value, 0);

    const probeSuite = evaluateToyProbeSuite(plan);
    const cosine = probeSuite.worstCosine;
    const maxLogitError = probeSuite.maxLogitError;
    const top1Unchanged = probeSuite.top1Flips === 0;
    const specPass = plan.mlp === "int8"
      && plan.attention === "int8"
      && plan.lm_head === "int8"
      && plan.norm === "fp32"
      && plan.activation === "a16";
    const memoryPass = currentBytes <= preset.budgetBytes;
    const qualityPass = cosine >= TOY_QUALITY_LIMITS.worstCosine
      && maxLogitError <= TOY_QUALITY_LIMITS.maxLogitError
      && probeSuite.meanAbsoluteError <= TOY_QUALITY_LIMITS.meanAbsoluteError;
    const passed = memoryPass && qualityPass && top1Unchanged && specPass;
    return {
      baselineBytes,
      currentBytes,
      memoryByGroup,
      cosine,
      meanCosine: probeSuite.meanCosine,
      maxLogitError,
      meanAbsoluteError: probeSuite.meanAbsoluteError,
      top1Flips: probeSuite.top1Flips,
      sampleCount: probeSuite.sampleCount,
      top1Unchanged,
      specPass,
      memoryPass,
      qualityPass,
      passed,
      baselineTokens: probeSuite.baselineTokens,
      tokens: probeSuite.tokens,
      savedBytes: baselineBytes - currentBytes
    };
  }

  function init(doc) {
    const one = (selector, scope) => (scope || doc).querySelector(selector);
    const all = (selector, scope) => [...(scope || doc).querySelectorAll(selector)];
    const gameKey = "pytorch-v2-level-25-factory-v3";
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
      version: 2,
      mode: "guided",
      preset: "real",
      view: 0,
      lens: "flow",
      mission: 0,
      selectedNode: "model",
      selectedLinear: "q_proj",
      selectedWeight: 2.5,
      selectedWeightIndex: 3,
      deployment: {
        mlp: "fp32",
        attention: "fp32",
        lm_head: "fp32",
        norm: "fp32",
        activation: "a16",
        granularity: "tensor"
      },
      rescuePassed: false,
      completed: [],
      blockAssembly: [],
      assemblyRan: false,
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
    const state = {
      ...defaultState,
      ...loaded,
      deployment: { ...defaultState.deployment, ...(loaded.deployment || {}) }
    };
    state.completed = Array.isArray(state.completed) ? state.completed : [];
    state.quantPipeline = Array.isArray(state.quantPipeline) ? state.quantPipeline : [];
    state.runtimePipeline = Array.isArray(state.runtimePipeline) ? state.runtimePipeline : [];
    state.blockAssembly = Array.isArray(state.blockAssembly) ? state.blockAssembly : [];

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
      const targetMission = Math.max(0, Math.min(5, mission));
      const prerequisites = Array.from({ length: targetMission }, (_, index) => index);
      const locked = state.mode === "guided" && prerequisites.some((item) => !state.completed.includes(item));
      if (locked) {
        const firstMissing = prerequisites.find((item) => !state.completed.includes(item));
        setConsole("warn", `引导模式下请先完成 Mission ${String(firstMissing).padStart(2, "0")}。切换“自由剖视”仍可查看任何结构。`, "MISSION LOCKED");
        return;
      }
      state.mission = targetMission;
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
        cell.dataset.weightIndex = String(index);
        cell.textContent = String(value);
        cell.title = `W[${Math.floor(index / 8)},${index % 8}] = ${value}`;
        cell.classList.toggle("active", index === state.selectedWeightIndex);
        cell.addEventListener("click", () => {
          all("button", matrix).forEach((item) => item.classList.remove("active"));
          cell.classList.add("active");
          state.selectedWeight = value;
          state.selectedWeightIndex = index;
          renderByteView();
          setConsole("", `选中 ${cell.title}。下方字节现在由这个值实时计算，不再使用固定示例。`, "WEIGHT CELL");
          setView(4);
          save();
        });
        matrix.appendChild(cell);
      }
      renderByteView();
    }

    function renderByteView() {
      const value = Number(state.selectedWeight);
      const scale = 127 / 3;
      const code = Math.max(-128, Math.min(127, roundToEven(value * scale)));
      const restored = code / scale;
      const error = Math.abs(restored - value);
      const bytes = float32Bytes(value);
      one("#byte-fp-value").textContent = formatNumeric(value);
      one("#byte-fp32-row").innerHTML = bytes.map((byte, index) => (
        `<span><b>${byteBits(byte).slice(0, 4)} ${byteBits(byte).slice(4)}</b><small>B${index + 1}</small></span>`
      )).join("");
      one("#byte-fp-hex").textContent = `0x${bytes.map((byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
      one("#byte-int-value").textContent = String(code);
      one("#byte-int8-row").innerHTML = `<span><b>${byteBits(code).slice(0, 4)} ${byteBits(code).slice(4)}</b><small>1 byte</small></span>`;
      one("#byte-restored-value").textContent = formatNumeric(restored);
      one("#byte-error").textContent = formatNumeric(error);
      one("#byte-selection").textContent = `当前 W[${Math.floor(state.selectedWeightIndex / 8)},${state.selectedWeightIndex % 8}] = ${formatNumeric(value)}`;
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

    function renderTokenList(element, tokens, baseline) {
      element.innerHTML = tokens.map((item, index) => (
        `<li class="${index === 0 ? "top" : ""}">
          <span>${item.token}</span>
          <i><em style="width:${(item.probability * 100).toFixed(1)}%"></em></i>
          <b>${(item.probability * 100).toFixed(1)}%</b>
          ${!baseline && index === 0 ? "<small>模型选择</small>" : ""}
        </li>`
      )).join("");
    }

    function formatVector(vector) {
      return `[${vector.map((value) => Number(value).toFixed(2).replace("-", "−")).join(", ")}]`;
    }

    function renderAssembly() {
      const nextIndex = state.blockAssembly.length;
      all("[data-block-slot]").forEach((slot, index) => {
        const piece = BLOCK_ASSEMBLY[index];
        const installed = state.blockAssembly[index] === piece.id;
        slot.classList.toggle("installed", installed);
        slot.innerHTML = installed
          ? `<b>${piece.label}</b><small>${piece.shape}</small>`
          : String(index + 1).padStart(2, "0");
      });
      all("[data-block-piece]").forEach((button) => {
        button.classList.toggle("used", state.blockAssembly.includes(button.dataset.blockPiece));
        button.disabled = state.blockAssembly.includes(button.dataset.blockPiece);
      });
      if (nextIndex < BLOCK_ASSEMBLY.length) {
        const next = BLOCK_ASSEMBLY[nextIndex];
        one("#assembly-step").textContent = `STEP ${nextIndex + 1} / ${BLOCK_ASSEMBLY.length}`;
        one("#assembly-instruction").textContent = `${next.role}：安装 ${next.label}`;
        one("#assembly-contract").textContent = `${next.shape}；${next.weight}。`;
      } else {
        one("#assembly-step").textContent = "BLOCK ASSEMBLED";
        one("#assembly-instruction").textContent = "Pre-Norm 残差 Block 已正确闭合";
        one("#assembly-contract").textContent = "两条子层都先归一化、再变换、最后与各自的 residual 主干相加。";
      }
      one("#run-assembly").disabled = nextIndex !== BLOCK_ASSEMBLY.length;
      one("#unfold-bytes").disabled = !state.assemblyRan;
      if (!state.assemblyRan) {
        one("#assembly-vector").textContent = formatVector(TOY_PROBE.input);
      }
    }

    function maybeCompleteMissionOne() {
      if (state.assemblyRan && state.compressed) {
        completeMission(1, "已正确组装并运行 Pre-Norm Block，也完成了 FP32 4 byte → INT8 1 byte 的权重存储改造。");
      }
    }

    function rescueGuidance(simulation) {
      const plan = state.deployment;
      if (plan.norm === "int8") {
        return [
          "撤销 RMSNorm INT8：它不属于本关的 Linear 改造",
          `只多省 ${formatBytes(simulation.memoryByGroup.norm * 3)}；8 组实算的最差 cosine=${simulation.cosine.toFixed(6)}，平均 |Δlogit|=${simulation.meanAbsoluteError.toFixed(6)}。单样本 cosine 偶尔微升只是误差抵消，不是质量提升。`
        ];
      }
      if (plan.activation === "a8") {
        return [
          "A8 不是本关的 W8 权重压缩",
          `切回 A16。权重仓字节没有变化；8 组实算平均 |Δlogit|=${simulation.meanAbsoluteError.toFixed(6)}，真实 W8A8 还需要校准与适配内核。`
        ];
      }
      const int4Target = ["mlp", "attention", "lm_head"].find((target) => plan[target] === "int4");
      if (int4Target) {
        const label = { mlp: "MLP Linear", attention: "Attention Linear", lm_head: "LM Head" }[int4Target];
        return [
          `${label} 已变成 INT4，不符合 W8A16`,
          `8 组探针的最差 cosine=${simulation.cosine.toFixed(6)}、平均 |Δlogit|=${simulation.meanAbsoluteError.toFixed(6)}。朴素 absmax INT4 不等于 GPTQ/AWQ。`
        ];
      }
      const fp32Target = ["mlp", "attention", "lm_head"].find((target) => plan[target] === "fp32");
      if (fp32Target) {
        const label = { mlp: "MLP Linear", attention: "Attention Linear", lm_head: "LM Head" }[fp32Target];
        return [`下一步：量化 ${label}`, `把 ${label} 从 FP32 · 4 byte 切到 INT8 · 1 byte，观察仓库缩短多少。`];
      }
      if (!simulation.memoryPass) {
        return ["显存仍超预算", "三类大型 Linear 都已是 INT8；尝试更细的 scale 粒度会增加少量元数据，不会带来 4 倍以上的免费压缩。"];
      }
      if (!simulation.qualityPass || !simulation.top1Unchanged) {
        return [
          "显存已达标，但 Toy 输出越过验收线",
          `8 组中最差 cosine=${simulation.cosine.toFixed(6)}、最大 |Δlogit|=${simulation.maxLogitError.toFixed(6)}、Top-1 翻转 ${simulation.top1Flips}/${simulation.sampleCount}；调整方案后再验证。`
        ];
      }
      if (state.rescuePassed) {
        return ["抢救完成：显存警报已解除", "大型 Linear 已换成 INT8；继续 Mission 01，拆开一个权重看清 4 byte 如何变成 1 byte。"];
      }
      return ["四项条件已满足", "点击“验证当前改造方案”，关闭显存警报并进入模型内部。"];
    }

    function renderRescue() {
      const preset = PRESETS[state.preset];
      const simulation = simulateDeployment(state.deployment, preset);
      const baseline = simulation.baselineBytes;
      one("#fp32-total").textContent = formatBytes(baseline);
      one("#int8-total").textContent = formatBytes(simulation.currentBytes);
      one("#compression-total").textContent = `${(baseline / simulation.currentBytes).toFixed(2)}×`;
      one("#budget-limit").textContent = formatBytes(preset.budgetBytes);
      one("#warehouse-total").textContent = formatBytes(simulation.currentBytes);
      one("#goal-memory").textContent = `${formatBytes(simulation.currentBytes)} / ${formatBytes(preset.budgetBytes)}`;
      one("#goal-quality").textContent = `${simulation.cosine.toFixed(6)} / ≥0.990000`;
      one("#goal-token").textContent = `${simulation.top1Flips} / ${simulation.sampleCount} 翻转`;
      one("#goal-spec").textContent = simulation.specPass ? "W8A16" : "不符合 W8A16";
      one("#quality-score").textContent = simulation.cosine.toFixed(6);
      one("#deployment-cosine").textContent = simulation.cosine.toFixed(6);
      one("#mean-logit-error").textContent = simulation.meanAbsoluteError.toFixed(6);
      one("#max-logit-error").textContent = simulation.maxLogitError.toFixed(6);
      one("#top1-state").textContent = `${simulation.top1Flips} / ${simulation.sampleCount}`;
      one("#saved-memory").textContent = formatBytes(simulation.savedBytes);
      one("#quality-meter").style.width = `${Math.max(0, Math.min(100, simulation.cosine * 100))}%`;
      one("#quality-state").textContent = simulation.qualityPass && simulation.top1Unchanged ? "OUTPUT STABLE" : "OUTPUT DRIFT";
      one("#quality-state").classList.toggle("warn", !simulation.qualityPass || !simulation.top1Unchanged);
      renderTokenList(one("#baseline-tokens"), simulation.baselineTokens, true);
      renderTokenList(one("#current-tokens"), simulation.tokens, false);

      const goals = {
        memory: simulation.memoryPass,
        quality: simulation.qualityPass,
        token: simulation.top1Unchanged,
        spec: simulation.specPass
      };
      Object.entries(goals).forEach(([goal, passed]) => {
        const item = one(`[data-rescue-goal="${goal}"]`);
        item.classList.toggle("pass", passed);
        item.classList.toggle("fail", !passed);
      });

      all("[data-target][data-scheme]").forEach((button) => {
        button.classList.toggle("active", state.deployment[button.dataset.target] === button.dataset.scheme);
      });
      all("[data-granularity]").forEach((button) => {
        button.classList.toggle("active", state.deployment.granularity === button.dataset.granularity);
      });

      const groups = deploymentGroups(preset);
      const largestBytes = groups.mlp * 4;
      ["mlp", "attention", "lm_head", "norm"].forEach((target) => {
        const cargo = one(`[data-cargo="${target}"]`);
        const current = simulation.memoryByGroup[target];
        const full = groups[target] * 4;
        cargo.style.setProperty("--cargo-base", `${Math.max(6, full / largestBytes * 100)}%`);
        cargo.style.setProperty("--cargo-ratio", String(Math.max(0.04, current / full)));
        one("b", cargo).textContent = `${formatBytes(full)} → ${formatBytes(current)}`;
        cargo.classList.toggle("compressed", current < full);
        cargo.classList.toggle("danger", target === "norm" && state.deployment.norm === "int8");
      });

      const memoryRatio = simulation.currentBytes / baseline;
      one("#memory-gauge").style.width = `${Math.min(100, memoryRatio * 100)}%`;
      one(".mf-memory-gauge").style.setProperty("--budget-position", `${Math.min(100, preset.budgetBytes / baseline * 100)}%`);
      let budgetState = "OVER BUDGET";
      if (simulation.memoryPass && (!simulation.qualityPass || !simulation.top1Unchanged || !simulation.specPass)) {
        budgetState = "MEMORY OK · MODEL AT RISK";
      } else if (simulation.passed) {
        budgetState = state.rescuePassed ? "RESCUE COMPLETE" : "READY TO VALIDATE";
      }
      one("#budget-state").textContent = budgetState;
      one(".mf-crisis").classList.toggle("compact", simulation.passed);
      one(".mf-crisis").classList.toggle("quality-risk", simulation.memoryPass && !simulation.passed);
      const [objective, nextAction] = rescueGuidance(simulation);
      one("#current-objective").textContent = objective;
      one("#next-action").textContent = nextAction;
      one("#rescue-feedback").textContent = nextAction;
      one("#validate-rescue").disabled = false;
      return simulation;
    }

    function renderPreset() {
      const preset = PRESETS[state.preset];
      all("[data-preset]").forEach((button) => button.classList.toggle("active", button.dataset.preset === state.preset));
      all("[data-layers]").forEach((element) => { element.textContent = String(preset.layers); });
      one("#preset-note").textContent = preset.note;
      renderRescue();
      renderLinear(state.selectedLinear);
      selectNode(state.selectedNode, false);
    }

    function renderMissionRail() {
      all("[data-mission-tab]").forEach((button) => {
        const mission = Number(button.dataset.missionTab);
        const prerequisites = Array.from({ length: mission }, (_, index) => index);
        const locked = state.mode === "guided" && prerequisites.some((item) => !state.completed.includes(item));
        button.classList.toggle("complete", state.completed.includes(mission) || (mission === 5 && state.shipped));
        button.classList.toggle("locked", locked);
        button.setAttribute("aria-disabled", String(locked));
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
      const deployment = simulateDeployment(state.deployment, PRESETS[state.preset]);
      const missionReady = [0, 1, 2, 3, 4].every((mission) => state.completed.includes(mission));
      const checks = {
        memory: deployment.memoryPass,
        rescue: state.rescuePassed && deployment.qualityPass && deployment.top1Unchanged && deployment.specPass,
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
      one("#boss-memory").textContent = formatBytes(deployment.currentBytes);
      one("#boss-quality").textContent = `worst cos ${deployment.cosine.toFixed(6)}`;
    }

    function bindInteractions() {
      all("[data-mode]").forEach((button) => button.addEventListener("click", () => {
        state.mode = button.dataset.mode;
        all("[data-mode]").forEach((item) => item.classList.toggle("active", item.dataset.mode === state.mode));
        renderMissionRail();
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
      all("[data-target][data-scheme]").forEach((button) => button.addEventListener("click", () => {
        state.deployment[button.dataset.target] = button.dataset.scheme;
        const simulation = renderRescue();
        if (!simulation.passed) {
          state.rescuePassed = false;
          state.shipped = false;
          state.completed = state.completed.filter((mission) => mission !== 0 && mission !== 5);
        }
        renderMissionRail();
        renderBoss();
        save();
      }));
      all("[data-granularity]").forEach((button) => button.addEventListener("click", () => {
        state.deployment.granularity = button.dataset.granularity;
        const simulation = renderRescue();
        if (!simulation.passed) {
          state.rescuePassed = false;
          state.shipped = false;
          state.completed = state.completed.filter((mission) => mission !== 0 && mission !== 5);
        }
        renderMissionRail();
        renderBoss();
        save();
      }));
      one("#validate-rescue").addEventListener("click", () => {
        const simulation = renderRescue();
        if (!simulation.passed) {
          const [objective, action] = rescueGuidance(simulation);
          setFeedback("#rescue-feedback", "warn", `${objective}：${action}`);
          setConsole("warn", "显存、8 组 Toy 探针、Top-1 与 W8A16 规格必须同时通过；只让显存变绿不算完成。", "CONFIG REJECTED");
          return;
        }
        state.rescuePassed = true;
        completeMission(0, `显存进入预算；8 组探针最差 cosine=${simulation.cosine.toFixed(6)}，Top-1 零翻转，且方案符合 W8A16。`);
        setFeedback("#rescue-feedback", "good", "抢救成功：大型 Linear 使用 INT8；RMSNorm 保留 FP32；activation 保持 A16。现在进入模型内部，理解为什么。");
        renderRescue();
        renderBoss();
        setMission(1);
        save();
      });
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
          setFeedback("#hunt-feedback", "good", "Linear 的 [out,in] 二维权重才是主仓库。上方 MLP 箱体约 16.13 GiB，RMSNorm 总共约 0.001 GiB。");
          setConsole("good", "已定位大型 Linear。回到救援驾驶舱选择实际存储方案，显存才会真正变化。", "TARGET LOCATED");
          selectNode("block", true);
        } else if (choice === "norm") {
          const normFp32Bytes = deploymentGroups(PRESETS[state.preset]).norm * 4;
          setFeedback("#hunt-feedback", "warn", `全部 RMSNorm 的 FP32 权重只有 ${formatBytes(normFp32Bytes)}；改成 INT8 只省 75%，远小于二维 Linear 仓库。本 Notebook 的 W8A16Linear 也不替换 Norm。`);
          setConsole("warn", "这里能确定的是“持久权重收益极小、超出本实验实现范围”。真实精度影响必须对目标 checkpoint 做逐层消融，不能仅凭层名断言。", "OUT OF SCOPE");
        } else {
          setFeedback("#hunt-feedback", "warn", "A8 可以减少部分运行时 activation 缓冲区，但 W8A16 的模型权重仓不会因此变小；它还是另一套精度与内核问题。");
          setConsole("warn", "Residual Add 的持久化权重数是 0。不要把运行时 activation 内存与模型权重存储混为一谈。", "WRONG BUDGET");
        }
      }));

      all("[data-block-piece]").forEach((button) => button.addEventListener("click", () => {
        const pieceId = button.dataset.blockPiece;
        const next = BLOCK_ASSEMBLY[state.blockAssembly.length];
        if (!next) return;
        if (pieceId !== next.id) {
          const chosen = BLOCK_ASSEMBLY.find((piece) => piece.id === pieceId);
          let reason = `说明书当前需要 ${next.label}（${next.role}），而不是 ${chosen.label}。`;
          if (pieceId === "attention" && next.id === "rms1") {
            reason = "LLaMA-style Pre-Norm Block 要先对输入做 RMSNorm，再送进 Attention；否则你搭成了另一种架构。";
          } else if (pieceId.startsWith("residual") && !state.blockAssembly.includes("attention")) {
            reason = "Residual Add 必须等子层产生 [B,S,H] 输出后，才能与保存的主干相加；现在第二个加数还不存在。";
          } else if (pieceId === "mlp" && next.id !== "mlp") {
            reason = "SwiGLU MLP 前还有第二个 RMSNorm；Pre-Norm 的位置不能跳过。";
          }
          setFeedback("#assembly-feedback", "warn", reason);
          setConsole("warn", reason, "ASSEMBLY MISMATCH");
          return;
        }
        state.blockAssembly.push(pieceId);
        state.assemblyRan = false;
        setFeedback("#assembly-feedback", "good", `${next.label} 已锁定：${next.shape}；${next.weight}。`);
        renderAssembly();
        save();
      }));

      one("#clear-assembly").addEventListener("click", () => {
        state.blockAssembly = [];
        state.assemblyRan = false;
        state.completed = state.completed.filter((mission) => mission !== 1);
        one("#assembly-token").classList.remove("running");
        setFeedback("#assembly-feedback", "", "积木已拆回零件区。跟随说明书重新搭建 Pre-Norm Block。");
        renderAssembly();
        renderMissionRail();
        save();
      });

      one("#run-assembly").addEventListener("click", () => {
        if (state.blockAssembly.length !== BLOCK_ASSEMBLY.length) return;
        const baselinePlan = {
          mlp: "fp32",
          attention: "fp32",
          lm_head: "fp32",
          norm: "fp32",
          activation: "a16",
          granularity: "tensor"
        };
        const probe = runToyProbe(baselinePlan);
        const token = one("#assembly-token");
        const reduceMotion = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
        token.classList.remove("running");
        void token.offsetWidth;
        token.classList.add("running");
        probe.trace.forEach((vector, index) => {
          root.setTimeout(() => {
            one("#assembly-vector").textContent = formatVector(vector);
            all("[data-block-slot]").forEach((slot, slotIndex) => slot.classList.toggle("active", slotIndex === index));
            if (index === probe.trace.length - 1) {
              state.assemblyRan = true;
              all("[data-block-slot]").forEach((slot) => slot.classList.remove("active"));
              setFeedback("#assembly-feedback", "good", "hidden state 已沿 6 块积木完成一次真实前向：主干 shape 始终保持 [B,S,H]。现在可以下钻到单个权重。");
              renderAssembly();
              maybeCompleteMissionOne();
              save();
            }
          }, reduceMotion ? 0 : index * 420);
        });
      });

      one("#unfold-bytes").addEventListener("click", () => {
        if (!state.assemblyRan) {
          setFeedback("#assembly-feedback", "warn", "请先按说明书搭完 Block，并让 hidden state 跑一遍。");
          return;
        }
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
        maybeCompleteMissionOne();
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
        const deployment = simulateDeployment(state.deployment, PRESETS[state.preset]);
        const ready = state.rescuePassed && deployment.passed
          && state.compressed && state.zeroSafe && Number(state.lastCosine) > 0.99 && state.outputShapeOk
          && [0, 1, 2, 3, 4].every((mission) => state.completed.includes(mission));
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
    renderAssembly();
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
    quantizeMatrix,
    float32Bytes,
    byteBits,
    deploymentGroups,
    simulateDeployment,
    quantizeMatrixAtBits,
    runToyProbe,
    evaluateToyProbeSuite,
    TOY_PROBE_INPUTS,
    TOY_QUALITY_LIMITS,
    BLOCK_ASSEMBLY
  };
});
