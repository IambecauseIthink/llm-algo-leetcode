# 显存优化图片资产维护记录

> 该文件只供维护者核对图片资产，不是学习路线中的第 7 个学习页面。学习者应直接从 `intro`、`01–06` 和 `walkthrough` 进入内容。

## 资产清单

| 资产 | 使用位置 | 作用 |
|:---|:---|:---|
| `memory_optimization_roadmap.svg` | `intro.md` | 展示 Task0–6、训练分支、推理 / 量化分支和系统扩展 |
| `memory_optimization_knowledge_map.svg` | `intro.md` | 展示显存对象、策略和证据升级 |
| `memory_ledger.svg` | `01_vram_ledger_and_metrics.md`、`casebook.md` | 对象 → 生命周期 → 证据 |
| `training_pressure_diagnosis.svg` | `02_training_memory_pressure.md` | 改变输入 → 定位压力对象 → 选择候选策略 |
| `checkpoint_offload_tradeoff.svg` | `03_checkpointing_and_offload.md` | 重算与搬运的代价转移 |
| `kv_cache_budget.svg` | `04_inference_cache_and_memory_budget.md` | 请求增长 → Cache 状态 → 组织复用 → 容量证据 |
| `quantization_memory_tool.svg` | `05_quantization_as_a_memory_tool.md` | 量化对象 → 处理时机 → 部署证据 |
| `benchmark_tradeoff_decision.svg` | `06_benchmark_and_tradeoff_decision.md`、`casebook.md` | 固定条件 → 对照 → 预算 → Profiling → 决策 |

`ledger_evidence_bridge.svg`、`training_memory_pressure.svg`、`checkpoint_offload_lifecycle.svg`、`kv_cache_roles.svg`、`quantization_object_timing.svg` 和 `memory_evidence_decision_loop.svg` 是历史细分图，暂保留在资产目录，正文不再引用。

## 维护规则

- 正式图片使用 SVG，文件放在 `docs/public/topic_discussion/memory_performance_tuning/`。
- 源 Markdown 只引用实际使用的图片，不在正文中保留图片占位说明。
- 图片只表达概念关系，不承载代码实现、完整实验步骤或过长解释。
- 修改源文件后，最后统一运行文档镜像同步脚本；不手动编辑 `docs/topic_discussion/` 下的镜像正文。
