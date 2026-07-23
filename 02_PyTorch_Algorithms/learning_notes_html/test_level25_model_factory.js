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
assert.ok(factoryHtml.includes("per-tensor"));
assert.ok(factoryHtml.includes("to(x.dtype)"));
assert.ok(factoryCss.includes("@media (max-width: 640px)"));
assert.ok(factoryCss.includes("@media (prefers-reduced-motion: reduce)"));
assert.ok(!notebookText.includes("反量化时再把 INT8 乘回"));
assert.ok(notebookText.includes("反量化时把 INT8 除以同一个"));
assert.ok(notebookText.includes("不能写死 FP16"));

console.log(`Level 25 model factory checks passed (cosine=${cosine.toFixed(6)})`);
