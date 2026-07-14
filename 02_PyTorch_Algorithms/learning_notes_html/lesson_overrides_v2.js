// Reuse the detailed lessons whose notebook topic survived the upstream renumbering.
// Brand-new topics intentionally fall back to curriculum_v2 metadata + notebook guide.

const legacy = require("./lesson_overrides_extra");

module.exports = {
  "12": null,         // New: Gradient Accumulation
  "13": null,         // New: End-to-End Fine-Tuning Experiment
  "14": legacy["12"], // PPO
  "15": legacy["13"], // DPO
  "16": null,         // New: GRPO
  "17": legacy["14"], // Attention backward, now introduced as Autograd Basics
  "18": null,         // New: Activation and Loss Backward
  "19": legacy["21"], // Activation checkpointing; notebook guide adds offload context
  "20": legacy["15"], // FlashAttention
  "21": legacy["16"], // Decoding
  "22": legacy["17"], // PagedAttention
  "23": legacy["18"], // Speculative decoding
  "24": legacy["19"], // RadixAttention
  "25": legacy["20"], // W8A16
  "26": legacy["22"], // QLoRA
  "27": legacy["23"], // ZeRO
  "28": legacy["25"], // Pipeline parallelism
  "29": legacy["24"]  // Tensor parallelism
};
