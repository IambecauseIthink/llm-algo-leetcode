const assert = require("assert");
const fs = require("fs");
const path = require("path");
const factory = require("./assets/level25_model_factory");

const root = __dirname;
const notes = path.join(root, "notes");

assert.strictEqual(factory.roundToEven(2.5), 2, "torch.round uses ties-to-even for 2.5");
assert.strictEqual(factory.roundToEven(3.5), 4, "torch.round uses ties-to-even for 3.5");
assert.strictEqual(factory.roundToEven(-1.5), -2, "negative half ties also round to even");
assert.strictEqual(factory.roundToEven(-2.5), -2, "negative even tie is preserved");

const sample = factory.absmaxQuantize([-0.8, 1.5, -3.0, 2.5, 0.0], false);
assert.strictEqual(sample.absmax, 3);
assert.ok(Math.abs(sample.scale - 127 / 3) < 1e-12);
assert.strictEqual(sample.quantized[3], 106);
assert.ok(sample.quantized.every((value) => Number.isInteger(value) && value >= -128 && value <= 127));

const unsafeZero = factory.absmaxQuantize([0, 0, 0], false);
assert.strictEqual(unsafeZero.scale, Infinity);
assert.ok(unsafeZero.quantized.every(Number.isNaN));

const safeZero = factory.absmaxQuantize([0, 0, 0], true);
assert.ok(Number.isFinite(safeZero.scale));
assert.deepStrictEqual(safeZero.quantized, [0, 0, 0]);

const weights = [
  [1.0, -2.0, 3.0, -4.0],
  [0.5, 0.25, -0.75, 1.5],
  [-1.0, 0.0, 1.0, -2.0]
];
const inputs = [
  [1.0, -1.0, 0.5, 2.0],
  [0.0, 1.0, -1.0, 3.0]
];
const bias = [0.1, -0.2, 0.3];
const quantizedWeights = factory.quantizeMatrix(weights, true);
const restored = factory.dequantize(quantizedWeights.quantized, quantizedWeights.scale);
const fpOutput = factory.linear(inputs, weights, bias);
const quantOutput = factory.linear(inputs, restored, bias);
const cosine = factory.cosineSimilarity(fpOutput, quantOutput);
assert.ok(cosine > 0.99, `deterministic W8A16 cosine should exceed 0.99, got ${cosine}`);
assert.deepStrictEqual(fpOutput.map((row) => row.length), [3, 3]);

assert.strictEqual(factory.memoryBytes(1024, 32) / factory.memoryBytes(1024, 8), 4);
assert.strictEqual(factory.memoryBytes(1024, 16) / factory.memoryBytes(1024, 8), 2);
assert.deepStrictEqual(factory.weightDims("q_proj", factory.PRESETS.toy), [8, 8]);
assert.deepStrictEqual(factory.weightDims("k_proj", factory.PRESETS.real), [1024, 4096]);
assert.deepStrictEqual(factory.weightDims("down_proj", factory.PRESETS.real), [4096, 11008]);
assert.ok(factory.linearWeightCount(factory.PRESETS.real) > factory.linearWeightCount(factory.PRESETS.toy));
assert.strictEqual(factory.MODEL_GRAPH.q_proj.quantizable, true);
assert.ok(!factory.MODEL_GRAPH.rms1.quantizable);
assert.ok(!factory.MODEL_GRAPH.residual1.quantizable);
assert.deepStrictEqual(
  factory.BLOCK_ASSEMBLY.map((piece) => piece.id),
  ["rms1", "attention", "residual1", "rms2", "mlp", "residual2"],
  "LLaMA-style block uses two Pre-Norm residual sublayers"
);

assert.deepStrictEqual(factory.float32Bytes(2.5), [0x40, 0x20, 0x00, 0x00]);
assert.notDeepStrictEqual(factory.float32Bytes(2.5), factory.float32Bytes(-0.8));
assert.strictEqual(factory.byteBits(106), "01101010");
assert.strictEqual(factory.byteBits(-34), "11011110");

const baselinePlan = {
  mlp: "fp32",
  attention: "fp32",
  lm_head: "fp32",
  norm: "fp32",
  activation: "a16",
  granularity: "tensor"
};
const w8a16Plan = {
  ...baselinePlan,
  mlp: "int8",
  attention: "int8",
  lm_head: "int8"
};
const correctDeployment = factory.simulateDeployment(w8a16Plan, factory.PRESETS.real);
assert.ok(correctDeployment.memoryPass, "W8A16 Linear plan must fit the 6.5 GiB budget");
assert.ok(correctDeployment.qualityPass);
assert.ok(correctDeployment.top1Unchanged);
assert.ok(correctDeployment.specPass);
assert.ok(correctDeployment.passed);
assert.ok(correctDeployment.baselineBytes / correctDeployment.currentBytes > 3.9);
assert.ok(correctDeployment.maxLogitError > 0, "INT8 probe must be recomputed, not copied from FP32");
assert.ok(correctDeployment.cosine < 1, "INT8 probe should expose its actual non-zero drift");
assert.strictEqual(correctDeployment.sampleCount, 8);
assert.strictEqual(correctDeployment.top1Flips, 0);
assert.ok(correctDeployment.meanAbsoluteError > 0);

const baselineProbe = factory.runToyProbe(baselinePlan);
const w8Probe = factory.runToyProbe(w8a16Plan);
assert.strictEqual(baselineProbe.trace.length, 6);
assert.ok(baselineProbe.trace.every((vector) => vector.length === 4));
assert.notDeepStrictEqual(w8Probe.logits, baselineProbe.logits);

const wrongNormPlan = factory.simulateDeployment(
  { ...w8a16Plan, norm: "int8" },
  factory.PRESETS.real
);
assert.ok(wrongNormPlan.memoryPass, "quantizing Norm does not undo the Linear memory saving");
assert.ok(!wrongNormPlan.specPass, "Notebook W8A16Linear scope does not replace RMSNorm");
assert.ok(wrongNormPlan.currentBytes < correctDeployment.currentBytes);
assert.ok(
  wrongNormPlan.cosine < correctDeployment.cosine,
  "the multi-input worst-case cosine must not present RMSNorm INT8 as an improvement"
);
assert.ok(
  wrongNormPlan.meanAbsoluteError > correctDeployment.meanAbsoluteError,
  "the multi-input mean logit error must expose RMSNorm INT8's additional drift"
);
assert.ok(
  correctDeployment.currentBytes - wrongNormPlan.currentBytes < 1024 * 1024,
  "all RMSNorm weights save less than 1 MiB in this declared real-size preset"
);

const activationPlan = factory.simulateDeployment(
  { ...w8a16Plan, activation: "a8" },
  factory.PRESETS.real
);
assert.strictEqual(
  activationPlan.currentBytes,
  correctDeployment.currentBytes,
  "A8 changes activation precision, not persistent weight bytes"
);
assert.ok(!activationPlan.specPass);

const naiveInt4Plan = factory.simulateDeployment(
  { ...baselinePlan, mlp: "int4", attention: "int4", lm_head: "int4" },
  factory.PRESETS.real
);
assert.ok(naiveInt4Plan.currentBytes < correctDeployment.currentBytes);
assert.ok(
  naiveInt4Plan.maxLogitError > correctDeployment.maxLogitError,
  "for the fixed probe, plain absmax INT4 must visibly drift more than INT8"
);

const realGroups = factory.deploymentGroups(factory.PRESETS.real);
assert.strictEqual(realGroups.mlp, 32 * 3 * 11008 * 4096);
assert.strictEqual(realGroups.attention, 32 * (4096 * 4096 + 1024 * 4096 * 2 + 4096 * 4096));
assert.strictEqual(realGroups.lm_head, 32000 * 4096);
assert.strictEqual(realGroups.norm, (32 * 2 + 1) * 4096);

const lessonHtml = fs.readFileSync(path.join(notes, "25_quantization_w8a16.html"), "utf8");
const factoryHtml = fs.readFileSync(path.join(notes, "25_quantization_model_factory.html"), "utf8");
const factoryCss = fs.readFileSync(path.join(root, "assets", "level25_model_factory.css"), "utf8");
const notebook = JSON.parse(fs.readFileSync(path.join(root, "..", "25_Quantization_W8A16.ipynb"), "utf8"));
const notebookText = JSON.stringify(notebook);

assert.ok(lessonHtml.includes('href="25_quantization_model_factory.html"'));
assert.ok(lessonHtml.includes("启动模型工厂"));
assert.ok(factoryHtml.includes("Decoder Transformer 工厂"));
assert.ok(factoryHtml.includes("../assets/level25_model_factory.js"));
assert.ok(factoryHtml.includes("MISSION 04 · INFERENCE REACTOR"));
assert.ok(factoryHtml.includes("显存正在爆仓"));
assert.ok(factoryHtml.includes("8 组 Toy Transformer 探针 · 浏览器实算"));
assert.ok(factoryHtml.includes("平均 |Δlogit|（越低越好）"));
assert.ok(factoryHtml.includes('data-target="norm" data-scheme="int8"'));
assert.ok(factoryHtml.includes('data-block-piece="attention"'));
assert.ok(factoryHtml.includes("实算、论文结论和未知项，必须分开"));
assert.ok(factoryHtml.includes("https://arxiv.org/abs/2208.07339"));
assert.ok(factoryHtml.includes("超过 99.9% 的值参与 8-bit 计算"));
assert.ok(factoryHtml.includes("最高 1.56× 加速、2× 内存降低"));
assert.ok(factoryHtml.includes("保护约 1% 显著权重"));
assert.ok(factoryHtml.includes("约 4 GPU 小时量化 175B 模型"));
assert.ok(factoryHtml.includes("不能误读"));
assert.ok(factoryHtml.includes('id="byte-fp32-row"'));
assert.ok(factoryHtml.includes("per-tensor"));
assert.ok(factoryHtml.includes("to(x.dtype)"));
assert.ok(factoryCss.includes("@media (max-width: 640px)"));
assert.ok(factoryCss.includes("@media (prefers-reduced-motion: reduce)"));
assert.ok(!notebookText.includes("反量化时再把 INT8 乘回"));
assert.ok(notebookText.includes("除以 self.scale 恢复其数值范围"));
assert.ok(notebookText.includes("weight_int8.to(x.dtype)"));

console.log(`Level 25 model factory checks passed (cosine=${cosine.toFixed(6)})`);
