# Topic Discussion Visual Assets Audit

Last updated: 2026-08-17

## Scope

这份清单服务 `topic_discussion/` 下的横向专题与基础支撑专题图片资产审核。

当前先不做全量逐图细审，先锁三件事：

- 图片也必须先分级，再决定是否进入正文
- 未审核图片不进入正文主叙事位置
- `visual_assets` 页可以作为图片缓冲区，但不等于自动批准进入正文

## Current Rule

从现在开始，`topic_discussion` 图片统一遵守下面的入口规则：

1. 未审核图片，不进入正文。
2. 未审核图片如需暂存，只放在：
   - `*_visual_assets.md`
   - 附录页
   - 素材汇总页
3. 只有在完成下面三步后，图片才可以进入正文：
   - 职责分级
   - 可读性初审
   - 动作判断：保留 / 减字 / 中文化 / 重画 / 后置

## Role Model

和 `Part02` 保持同一套三层职责模型：

### 1. 核心教学图

- 直接解释某个专题正文中的核心机制
- 一张图优先只讲一个主线
- 不替代正文大段解释

### 2. 路线收束图 / 决策图

- 负责说明专题页之间的连接、比较框架或决策路径
- 不承担底层机制主解释

### 3. 结构占位图 / 过渡图

- 用于暂时占位或说明页面边界
- 不优先精修
- 不在未定稿前进入正文

## Existing Asset Clusters

当前 `topic_discussion` 已经存在较多图片引用，主要分布在这些专题：

- `fine_tuning_training`
- `inference_optimization`
- `memory_performance_tuning`
- `communication_parallel`
- `profiling`
- `compiler_graph_optimization`
- `post_training_alignment`
- `quantization`
- `backpropagation_training_mechanism`
- `model_architecture`

另有已知本地素材文件：

- `topic_discussion/backpropagation_training_mechanism/attention_backward_impl_compare.svg`
- `topic_discussion/model_architecture/qwen_version_split.svg`
- `topic_discussion/model_architecture/deepseek_version_split.svg`

## Inference Optimization Audit

当前先审 `topic_discussion/inference_optimization` 这一组。

### 已进入正文的图片

| 资产名 | 所在正文页 | 当前职责判断 | 是否建议继续留在正文 | 当前主要问题 | 建议动作 |
|:---|:---|:---|:---:|:---|:---|
| `request_lifecycle.svg` | `01_request_path_and_metrics.md` | 核心教学图 | Y | 待审图本身，但页内职责合理 | 保留在正文，后续做可读性初审 |
| `prefill_attention.svg` | `02_prefill_and_attention_kernel.md` | 核心教学图 | Y | 待审图本身，但页内职责合理 | 保留在正文，后续做可读性初审 |
| `decode_strategies.svg` | `03_decoding_strategies.md` | 核心教学图 | Y | 待审图本身，但页内职责合理 | 保留在正文，后续做可读性初审 |
| `kv_cache_lifecycle_zh.svg` | `04_kv_cache_lifecycle_and_reuse.md` | 核心教学图 | Y | 已重做：覆盖增长、复用、容量边界，并桥接 07 | 保留在正文，后续同步 docs 镜像 |
| `kv_cache_backend_mechanisms_zh.svg` | `04_kv_cache_lifecycle_and_reuse.md` | vLLM / SGLang 机制对照图 | Y | 新增：区分物理分页与前缀复用 | 保留在正文，后续同步 docs 镜像 |
| `quantized_deployment.svg` | `05_quantized_inference_and_deployment.md` | 核心教学图 | Y | 待审图本身，但页内职责合理 | 保留在正文，后续做可读性初审 |
| `benchmark_decision.svg` | `06_benchmark_and_decision.md` | 路线收束图 / 决策图 | Y | 待审图本身，但页内职责合理 | 保留在正文，后续做可读性初审 |

### 当前结论

- `01-05` 的图当前都属于“正文主解释图”，职责是成立的。
- `06` 的图更像“收口决策图”，放在正文也是成立的。
- 这一组目前**没有明显应该先撤回 `07_visual_assets.md` 的图片**。

也就是说，`inference_optimization` 这组的主要矛盾不是“图不该进正文”，而是：

- 图本身是否已经过审
- 图内文字是否过多
- 图是否开始替代正文解释

### 本组后续规则

在这一组里，后续图片要进入正文，必须满足：

1. 图只解释当前页的一个主问题。
2. 图不把跨页路线、算法细节和最终决策混在一起。
3. 图先经过资产表登记，再决定是否放正文。

当前建议推进顺序：

1. 先逐图审核 `request_lifecycle.svg`
2. 再审 `prefill_attention.svg`
3. 再审 `decode_strategies.svg`
4. 再审 `kv_cache_scheduling.svg`
5. 再审 `quantized_deployment.svg`
6. 最后审 `benchmark_decision.svg`

## Next Steps

建议后续按专题批次推进，而不是把 `topic_discussion` 全量图片一起处理：

1. 先审 `inference_optimization`
2. 再审 `memory_performance_tuning`
3. 再审 `communication_parallel / profiling`
4. 最后审其余基础支撑专题

每批次最少输出：

- 该专题有哪些图片已经进入正文
- 哪些是核心教学图
- 哪些只是路线图或占位图
- 哪些应该先撤回到 `visual_assets` 页
