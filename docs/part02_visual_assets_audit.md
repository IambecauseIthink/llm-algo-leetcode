# Part02 Visual Assets Audit

Last updated: 2026-08-17

## Scope

这份清单只记录 `Part02` 当前**已被正文实际引用**的 SVG 资产，目标是服务后续三类工作：

- 渲染与可用性复核
- 术语一致性与是否中文化
- 是否保留、重画或降级为附录图

暂不把两类内容混进来：

- `docs/public/02_PyTorch_Algorithms/` 中尚未被正文引用的 SVG
- notebook 内嵌 ASCII / text 流程块

但从当前收口角度看，`Part02` 的图解问题不能只看 SVG 单体，还要同时看三种表达格式的边界：

- 正文主图用的 `SVG`
- notebook / docs 内局部解释用的 `ASCII / text block`
- 尚未形成稳定模板的 `Mermaid`

当前优先任务不是继续加图，而是先把三者的职责切开。

## Current Format Inventory

按 2026-08-17 当前仓库状态，`Part02` 图解表达大致是下面这套：

- `SVG`
  - 已经是 `docs/02_PyTorch_Algorithms` 正文主图的主格式
  - 当前约有 `31` 张 SVG 被正文实际引用
  - 这是后续应继续保留的正式教学图格式
- `ASCII / text block`
  - 主要存在于源 notebook 中
  - 常见用途是补一段局部结构、维度流向或简化版流程
  - 当前问题不是“不能存在”，而是它有时会和正式 SVG 重复承担主解释职责
- `Mermaid`
  - 当前在 `Part02` 正文里基本不是实际主格式
  - 因此这轮优先级不是“批量改 Mermaid”，而是先冻结 Mermaid 继续进入正文主叙事

结论上，`Part02` 当前真正的混合问题更接近：

1. 正文正式主图已经大多是 SVG。
2. 但部分页面同时保留了较长的 `ASCII / text block`，让“临时结构块”和“正式教学图”边界变模糊。
3. `Mermaid` 目前不是清理主战场，应先停止新增，而不是先投入大规模转换。

## Format Policy

`Part02` 后续统一按下面的格式口径收口：

### 1. SVG：正文正式主图

- 只要一张图承担本节的核心机制解释，它就应进入 `SVG` 体系。
- `SVG` 才允许进入正文主叙事位置。
- 进入正文前，仍需先经过这份审计表的职责分级与可读性复核。

### 2. ASCII / text block：局部辅助结构，不做主图

- 可以继续存在于 source notebook / docs 中。
- 适合承载：
  - 维度流向
  - 很短的两三步局部流程
  - 图前的“最小骨架说明”
- 不适合承载：
  - 本节唯一主图
  - 长段规则解释
  - 与 SVG 重复的完整机制图

如果某段 `ASCII / text block` 已经承担“没有它读者就看不懂本节主机制”的职责，就应考虑收编成正式 `SVG`，或者反过来删掉重复 SVG。

### 3. Mermaid：暂停进入 Part02 正文主图

- 在没有稳定模板前，`Mermaid` 不再作为 `Part02` 正文主图格式继续扩散。
- 如果后续要重新启用，前提是：
  - 明确它和 `SVG` 的职责边界
  - 有统一模板
  - 有稳定渲染与维护成本评估

当前阶段更稳的策略是：`Mermaid` 冻结新增，先把现有 `SVG + ASCII` 关系理顺。

## Role Model

当前先用三层职责模型给 `Part02` 图片分级，避免一边改图一边反复返工。

### 1. 核心教学图

这类图优先级最高，直接服务某一节最核心的一个机制。

应该做什么：

- 解释结构位置
- 解释数据流
- 解释组件关系

不该做什么：

- 不替代正文大段解释
- 不把完整算法规则全塞进图里
- 不同时承担机制、案例和结论三种职责

### 2. 路线收束图 / 项目图

这类图主要服务路线导航、项目闭环和 benchmark 收口。

应该做什么：

- 说明这页和前后页如何连接
- 说明比较框架或交付框架
- 帮读者快速定位“这页在路线里的作用”

不该做什么：

- 不承担单一机制主解释
- 不把路线图写成一页迷你教材

### 3. 结构占位图 / 过渡图

这类图优先级最低。通常页面定位、正文深度或最终结构还没完全稳定。

应该做什么：

- 先占住结构位置
- 暂时说明这一页讨论的对象或边界

不该做什么：

- 不急着精修文案
- 不急着中文化
- 不在页面职责未定时投入重画成本

## Status Legend

| 状态 | 含义 |
|:---|:---|
| `ok` | 静态结构正常，可进入人工视觉复核 |
| `dense` | 结构可用，但信息密度偏高，优先查可读性 |
| `mismatch` | 图和当前页面职责可能不完全匹配 |
| `reserved-style` | 图仍明显带预留页 / 过渡页语气，不建议先中文化 |

## Audit Table

| 资产名 | 来源页 | 当前是否被引用 | 类型 | 当前状态 | 主要问题 | 建议动作 | 优先级 | 备注 |
|:---|:---|:---:|:---|:---|:---|:---|:---:|:---|
| `01_rmsnorm_diagram.svg` | `01_RMSNorm_Tutorial` | Y | 教学图 | `ok` | 无明显静态问题 | 进入人工视觉复核；术语可后续统一 | P0 | 图内文字为英文 |
| `02_swiglu_gate.svg` | `02_SwiGLU_Activation` | Y | 教学图 | `ok` | 无明显静态问题 | 进入人工视觉复核；术语可后续统一 | P0 | 图内文字为英文 |
| `03_rope_rotation.svg` | `03_RoPE_Tutorial` | Y | 教学图 | `ok` | 无明显静态问题 | 进入人工视觉复核；术语可后续统一 | P0 | 图内文字为英文 |
| `04_attention_heads.svg` | `04_Attention_MHA_GQA` | Y | 教学图 | `ok` | 文本节点较多，但仍属主解释图 | 进入人工视觉复核；确认术语与正文一致 | P0 | 图内文字为英文 |
| `05_llama_block.svg` | `05_LLaMA3_Block_Tutorial` | Y | 教学图 | `ok` | 使用虚线 residual 路径，需人工确认视觉清晰度 | 进入人工视觉复核；可作为中文化候选 | P0 | 主结构图 |
| `06_moe_router.svg` | `06_MoE_Router` | Y | 教学图 | `ok` | 无明显静态问题 | 进入人工视觉复核 | P1 | 图内文字为英文 |
| `07_moe_balance.svg` | `07_MoE_Load_Balancing_Loss` | Y | 教学图 | `ok` | 无明显静态问题 | 进入人工视觉复核 | P1 | 图内文字为英文 |
| `08_architecture_tricks.svg` | `08_Architecture_Tricks` | Y | 教学图 | `ok` | 文本略长，需人工确认换行与拥挤度 | 进入人工视觉复核 | P1 | 图内文字为英文 |
| `09_sft_alignment.svg` | `09_SFT_Training_Loop` | Y | 教学图 | `ok` | token 行清楚，但下半部分规则说明过多，图开始承担正文解释职责 | 保留结构；优先减字；后续可中文化 | P0 | 主教学链路图 |
| `10_lora_adapter.svg` | `10_LoRA_Tutorial` | Y | 教学图 | `ok` | 结构清楚，但底部总结区过满，旁路说明文字偏多 | 保留结构；精简底部总结；后续可中文化 | P0 | 主教学链路图 |
| `11_wsd_curve.svg` | `11_LR_Schedulers_WSD_Cosine` | Y | 教学图 | `ok` | 长说明句偏多 | 进入人工视觉复核；评估是否简化文字 | P0 | 图内文字为英文 |
| `12_gradient_accumulation.svg` | `12_Gradient_Accumulation` | Y | 教学图 | `ok` | 虚线路径较多，需人工确认阅读顺序 | 进入人工视觉复核 | P0 | 图内文字为英文 |
| `13_training_loop.svg` | `13_End_to_End_Fine_Tuning_Experiment` | Y | 教学图 | `dense` | 节点与文字较多，闭环叙事可能偏密 | 先做人工视觉复核，再决定是否拆图或减字 | P0 | 端到端闭环图 |
| `14_rlhf_memory_flow.svg` | `14_RLHF_PPO_Memory` | Y | 教学图 | `dense` | 信息密度高，图内文字较长 | 先做人工视觉复核，再决定是否分成两图 | P0 | 对齐与显存主解释图 |
| `17_autograd_attention_backward.svg` | `17_Autograd_Basics` | Y | 教学图 | `ok` | 反向路径较多，需人工确认箭头和标签清晰度 | 进入人工视觉复核 | P0 | 图内文字为英文 |
| `19_checkpoint_offload.svg` | `19_Activation_Checkpointing_and_Activation_Offload` | Y | 教学图 | `ok` | 文本长度偏长，checkpoint 与 offload 对照需人工确认 | 进入人工视觉复核 | P1 | 图内文字为英文 |
| `20_flashattention_tiling.svg` | `20_FlashAttention_Sim` | Y | 教学图 | `ok` | 结构主线清楚，但两块说明卡文字过长，像迷你讲义 | 保留结构；优先压缩说明卡文案；术语后续统一 | P0 | 推理优化关键图 |
| `22_paged_attention_blocks.svg` | `22_vLLM_PagedAttention` | Y | 教学图 | `ok` | 逻辑块与物理块对照需人工确认易读性 | 进入人工视觉复核；优先术语统一 | P0 | 推理优化关键图 |
| `23_speculative_decoding_flow.svg` | `23_Speculative_Decoding` | Y | 教学图 | `dense` | `Acceptance rule` 与底部说明区过满，算法细节写进图里太多 | 保留主流程；优先减字；必要时把接受规则收回正文 | P0 | 推理优化关键图 |
| `24_radix_attention_tree.svg` | `24_SGLang_RadixAttention` | Y | 教学图 | `dense` | 树结构和说明语句较多 | 先做人工视觉复核，再决定是否简化文本 | P0 | 推理优化关键图 |
| `25_quantization_pipeline.svg` | `25_Quantization_W8A16` | Y | 教学图 | `dense` | 主流程清楚，但下半部分两块解释区使图过满 | 保留主流程；优先删减总结卡与 reading rule 文案 | P0 | 量化路线关键图 |
| `26_qlora_flow.svg` | `26_QLoRA_and_4bit_Quantization` | Y | 教学图 | `dense` | 流程与说明均偏多 | 先做人工视觉复核；后续统一术语 | P1 | 图内文字为英文 |
| `27_zero_sharding.svg` | `27_ZeRO_Optimizer_Sim` | Y | 教学图 | `ok` | 需人工确认分片对照是否直观 | 进入人工视觉复核 | P1 | 图内文字为英文 |
| `28_pipeline_bubble.svg` | `28_Pipeline_Parallelism_MicroBatch` | Y | 教学图 | `dense` | bubble 解释较长，需确认时序图是否拥挤 | 先做人工视觉复核 | P1 | 图内文字为英文 |
| `29_tensor_parallel_split.svg` | `29_Tensor_Parallelism_Sim` | Y | 教学图 | `ok` | 需人工确认列并行/行并行对照是否足够清楚 | 进入人工视觉复核 | P1 | 图内文字为英文 |
| `30_long_context_budget.svg` | `30_Long_Context_Fine_Tuning` | Y | 教学图 | `dense` | 更像决策卡而非单一结构图，文本较多 | 人工复核后决定是否保留原样 | P1 | 风格与前面 01-29 不完全一致 |
| `31_lora_variants_map.svg` | `31_LoRA_Variants_Theory` | Y | 比较图 | `dense` | 文本多、风格不同于前一批 SVG | 人工复核后决定是否需要统一模板 | P1 | 使用 `Arial`，非统一模板 |
| `32_sft_data_engineering_flow.svg` | `32_Data_Engineering_for_SFT` | Y | 流程图 | `dense` | 说明句过长，存在压缩文案空间 | 先做人工视觉复核，再决定是否简化 | P1 | 使用 `Arial`，非统一模板 |
| `38_pd_disaggregation.svg` | `38_Prefill_Decode_Disaggregation` | Y | 结构图 | `reserved-style` | 正文已是正式教学页，但图仍写“reserved page keeps space for”，职责明显错位 | 优先重写图文案；保留三段结构，不先中文化 | P2 | 先去掉“reserved / future page”语气 |
| `43_unified_memory_map.svg` | `43_Unified_Memory_Management` | Y | 结构图 | `reserved-style` | 正文讲统一账本，但图仍是预留页说明卡，且文本过长 | 优先重写图文案；保留 tier 结构，去掉预留页语气 | P2 | 不建议先翻译旧文案 |
| `44_auto_tuning_loop.svg` | `44_Auto_Tuning_Framework` | Y | 流程图 | `reserved-style` | 正文已转教学框架页，但图仍在解释“future page must prevent” | 优先重写图文案；保留 Goal/Search/Measure/Decide 骨架 | P2 | 先改职责，再决定中文化 |

## Role Groups

### A. 核心教学图

- `01_rmsnorm_diagram.svg`
- `02_swiglu_gate.svg`
- `03_rope_rotation.svg`
- `04_attention_heads.svg`
- `05_llama_block.svg`
- `06_moe_router.svg`
- `07_moe_balance.svg`
- `08_architecture_tricks.svg`
- `09_sft_alignment.svg`
- `10_lora_adapter.svg`
- `11_wsd_curve.svg`
- `12_gradient_accumulation.svg`
- `13_training_loop.svg`
- `14_rlhf_memory_flow.svg`
- `17_autograd_attention_backward.svg`
- `19_checkpoint_offload.svg`
- `20_flashattention_tiling.svg`
- `22_paged_attention_blocks.svg`
- `23_speculative_decoding_flow.svg`
- `24_radix_attention_tree.svg`
- `25_quantization_pipeline.svg`
- `26_qlora_flow.svg`
- `27_zero_sharding.svg`
- `28_pipeline_bubble.svg`
- `29_tensor_parallel_split.svg`

这组的统一原则：

- 一张图优先只讲一个机制主线
- 能删掉的说明句尽量收回正文
- 先做“减字”，再决定是否中文化

### B. 路线收束图 / 项目图

- `30_long_context_budget.svg`
- `31_lora_variants_map.svg`
- `32_sft_data_engineering_flow.svg`

这组的统一原则：

- 图主要承担路线定位、比较框架或交付收束
- 不要求像核心教学图那样完整解释底层机制
- 如果文本继续增长，应优先拆回正文或表格

### C. 结构占位图 / 过渡图

- `38_pd_disaggregation.svg`
- `43_unified_memory_map.svg`
- `44_auto_tuning_loop.svg`

这组的统一原则：

- 当前先解决“页面职责和图职责错位”
- 不把中文化作为第一任务
- 先决定这页究竟是正式教学页、弱教学页，还是过渡页

## Current Decision

当前不建议直接启动“全部 SVG 中文化”。更稳的顺序是：

1. 先定清 `SVG / ASCII / Mermaid` 的职责边界
2. 先对 `P0 / P1` 的正式 `SVG` 主图做人工视觉复核
3. 先确认 `38 / 43 / 44` 的页面定位是否稳定
4. 再从 `P0` 里挑真正长期保留的关键教学图做术语统一或中文化

当前新增结论：

- `Part02` 当前优先处理的不是“图内先翻成中文”，而是“不要继续混用主图格式”
- `Mermaid` 当前不应作为下一轮正文主图扩张方向
- `ASCII / text block` 可以保留，但应降回辅助说明，不再和正式主图竞争
- `38` 这类图不应先被当成正文重点图推进
- 在全量图片职责未理清前，局部精修单张过渡图的收益较低
- 下一步应优先按职责批次推进，而不是按文件顺序零散修改

## P0 First Batch Notes

已先审第一批高价值图：`01 / 05 / 09 / 10 / 20 / 22 / 23 / 24 / 25`。

当前结论：

- `01 / 05`
  - 可直接保留
  - 主要问题是标题、副标题或总结句偏长，不需要重画
- `09 / 10`
  - 教学价值高
  - 当前更适合“保留结构、压缩文字”，而不是先重画
- `20 / 22`
  - 图的主结构清楚
  - 说明卡片文案明显偏多，建议先减字
- `23 / 24 / 25`
  - 属于当前最容易信息过密的一批
  - 优先方向应是“图只保留结构主线，算法细节回收正文”

当前推荐的图整改策略：

1. 图保留结构主线，不继续让图承担大段正文解释。
2. 对 `09 / 10 / 20 / 23 / 25` 先做“减字”版，再决定是否中文化。
3. 真要中文化，也应在减字后做，而不是直接把长英文说明逐句翻译成中文。

状态更新（2026-08-17）：

- `01_rmsnorm_diagram.svg`
  - 已做一轮减字
  - 当前保留“短 ASCII 骨架 + 正式 SVG”结构
- `02_swiglu_gate.svg`
  - 已做一轮减字
  - 当前保留“短 ASCII 骨架 + 正式 SVG”结构
- `09_sft_alignment.svg`
  - 已做一轮减字
  - 底部教学清单已收成一条短收口
- `10_lora_adapter.svg`
  - 已做一轮减字
  - 底部总结区已收成一条短收口
- `20_flashattention_tiling.svg`
  - 已做一轮减字
  - 当前仍保留主结构，说明卡已压缩
- `22_paged_attention_blocks.svg`
  - 已做一轮减字
  - 当前仍保留“逻辑序列 -> 块表 -> 物理块池”主结构
- `23_speculative_decoding_flow.svg`
  - 已做一轮减字
  - 接受规则和教学清单已从大段说明卡收成短收口
- `24_radix_attention_tree.svg`
  - 已做一轮减字
  - 右侧和底部说明已压缩，主树结构保留
- `25_quantization_pipeline.svg`
  - 已做一轮减字
  - 机制主线保留，教学说明卡已压缩

## ASCII / Text Block First Batch

这一轮先不把 `ASCII / text block` 当成“必须全部删除”的对象，而是先看它有没有和 `SVG` 抢主解释职责。

首批先看：`01 / 02 / 20`。

### 01_RMSNorm_Tutorial

当前有两段 `ASCII / text block`：

1. `x [B, T, D] -> RMS over D -> output [B, T, D]`
2. `x -> RMSNorm -> Attention / MLP -> residual add`

当前判断：

- 两段都比较短。
- 第一段主要补“沿 hidden dimension 做归一化”这个维度直觉。
- 第二段主要补“RMSNorm 在 block 里的位置”这个最小骨架。
- 它们没有单独承担完整主解释，仍然是 `SVG` 在承担正文主图职责。

处理结论：

- 保留。
- 不继续扩写成更长的 ASCII 流程图。
- 后续如果 `SVG` 已能完整覆盖“位置 + 维度”两层信息，再考虑删掉其中一段最重复的骨架。

### 02_SwiGLU_Activation

当前有一段 `ASCII / text block`：

1. `gate_proj / up_proj / multiply / down_proj`

当前判断：

- 这段是图前骨架，作用是先把三路分支关系讲清。
- 它和 `SVG` 有重复，但重复量可控。
- 这类重复目前是“先用最小骨架降低读图门槛”，不属于职责冲突。

处理结论：

- 保留。
- 不继续增加第二段或更复杂的 ASCII 结构块。
- 如果后续 `02_swiglu_gate.svg` 再做一轮精简并能把主路径讲得更直白，可以再评估是否把这段收掉。

### 20_FlashAttention_Sim

当前判断：

- 这页正文没有额外的 `ASCII / text block` 主结构图。
- 当前格式问题不在 `ASCII vs SVG` 冲突，而在 `20_flashattention_tiling.svg` 自身说明卡文案偏长。
- 因此这页不应被纳入“先删 ASCII”的优先批次。

处理结论：

- 不需要做 ASCII 清理。
- 后续继续沿“保留结构、压缩 SVG 文案”的方向处理。

### Batch Conclusion

对 `01 / 02 / 20` 这批来说，当前最稳的口径是：

- `01 / 02`
  - `ASCII / text block` 仍可保留
  - 但只能作为图前骨架，不再继续扩写
- `20`
  - 当前不属于混合格式冲突页
  - 主要问题仍是 `SVG` 信息密度

因此下一批真正该优先检查的，不是继续盯 `01 / 02 / 20`，而是：

1. `23_Speculative_Decoding`
2. `24_SGLang_RadixAttention`
3. `25_Quantization_W8A16`

这三页更可能出现“图里塞了太多规则，正文和图职责缠在一起”的问题。

状态更新（2026-08-17）：

- `01 / 02`
  - 结论维持不变：ASCII 先保留为图前骨架
  - 当前未进入“删 ASCII”阶段
- `20`
  - 结论维持不变：主问题是 SVG 文案密度，不是格式冲突
  - 已完成一轮 SVG 减字

## Dense SVG Second Batch

第二批转看：`23 / 24 / 25`。

这一批的核心判断和 `01 / 02 / 20` 不一样：

- 主要问题不是 `ASCII / text block` 和 `SVG` 混用。
- 主要问题是 `SVG` 自己已经开始承担过多正文解释。
- 因此动作重点不是“删 ASCII”，而是“把规则、边界和长解释收回正文”。

### 23_Speculative_Decoding

当前判断：

- 正文里没有额外的 `ASCII / text block` 主结构图冲突。
- 页内已经先用正文给出接受概率公式和“为什么无损”的解释。
- `23_speculative_decoding_flow.svg` 如果还继续承载 `acceptance rule`、拒绝分支细节和长说明区，就会和正文形成重复解释。

处理结论：

- 这页不需要优先做 ASCII 清理。
- `SVG` 应只保留：
  - draft model 产出候选
  - target model 验证
  - accepted / rejected 后的控制流主线
- 接受概率公式、无损性解释、拒绝后如何回退，优先留在正文，不再堆进图里。

### 24_SGLang_RadixAttention

当前判断：

- 正文已经先讲了：
  - vLLM 与 SGLang 的机制对比
  - 最长前缀匹配公式
  - 为什么适合多轮对话
- 因此 `24_radix_attention_tree.svg` 不应再重复承载完整概念说明。
- 这页最容易出现的问题是：树结构、路径说明和多段解释同时进图，导致读者在图里看一遍、正文再看一遍。

处理结论：

- 这页也不是 ASCII 冲突页。
- `SVG` 应主要承担：
  - 共享前缀路径
  - 命中节点
  - 未命中新分支
- `Longest Prefix Match` 的公式、和 PagedAttention 的边界对比、适用场景判断，优先放正文。

### 25_Quantization_W8A16

当前判断：

- 正文已经按 Step 1-3 讲了：
  - Weight-only Quantization 为什么先做
  - `absmax_quantize` 和 `W8A16Linear` 的组件职责
  - `absmax / scale / quantize / dequantize` 数学公式
- 如果 `25_quantization_pipeline.svg` 再继续放长段 reading rule、边界说明和公式口径，就会明显替代正文。

处理结论：

- 这页同样不属于优先清理 ASCII 的对象。
- `SVG` 应主要承担：
  - float weight
  - absmax / scale
  - int8 storage
  - dequantize -> linear forward
- PTQ / QAT 区别、为什么选 W8A16、`127 vs 128` 的口径说明，优先留在正文。

### Batch Conclusion

对 `23 / 24 / 25` 这批的统一结论是：

- 问题主轴不是“格式混用”，而是“正式 SVG 过度正文化”。
- 当前不建议优先删正文中的短结构块。
- 更合适的动作是：
  1. 图只保留结构主线
  2. 公式和规则解释回收正文
  3. 边界判断和适用条件不再塞进图底部说明卡

这批的后续编辑优先级建议：

1. `23_speculative_decoding_flow.svg`
2. `25_quantization_pipeline.svg`
3. `24_radix_attention_tree.svg`

原因：

- `23 / 25` 更像典型的“规则被写进图里太多”
- `24` 的问题更多是树结构和说明语句同时偏多，但正文边界已经相对清楚

状态更新（2026-08-17）：

- `23`
  - 已完成一轮减字
  - 当前图只保留主控制流，规则解释已明显回收到正文侧
- `24`
  - 已完成一轮减字
  - 当前图主要承担“共享前缀 -> 命中 -> 新分支”结构主线
- `25`
  - 已完成一轮减字
  - 当前图主要承担“量化存储 -> 反量化 -> 线性前向”主线

## Reduced SVG Follow-up

在前两批基础上，`09 / 10 / 13 / 14 / 20 / 22 / 23 / 24 / 25` 已经进入“先减字一版”的状态。

### 13_End_to_End_Fine_Tuning_Experiment

当前判断：

- `13_training_loop.svg` 的主结构仍然成立。
- 主要问题不是流程错，而是底部“证明项”写成了半页讲义。
- 正文已经提供了最小报告模板、判据和链路解释，所以图不需要再重复承担这些说明。

处理结论：

- 已完成一轮减字。
- 保留主闭环和三条侧链：
  - evaluation path
  - report path
  - control path
- 底部说明已压成一句话收口。

### 14_RLHF_PPO_Memory

当前判断：

- `14_rlhf_memory_flow.svg` 的主流程和“多对象共存”的结构价值仍然很高。
- 主要问题是两整块说明卡把正文又讲了一遍。
- 当前还不需要拆成两张图，先减字足够有效。

处理结论：

- 已完成一轮减字。
- 保留 `prompt -> actor -> reward -> critic -> advantage -> PPO update` 主链。
- 两块说明卡已压成：
  - memory ledger 的短说明
  - 和 SFT 的一句话对比

### Follow-up Conclusion

截至 2026-08-17，当前已经处理过的一批高价值教学图可以分成两类：

1. 已减字且结构稳定：
   - `09 / 10 / 20 / 22 / 23 / 24 / 25 / 13 / 14`
2. 已审但后续仍可继续观察：
   - `01 / 02`
   - 目前结论仍是保留短 ASCII 骨架，不急着继续动

这意味着下一步不该回头反复修同一批，而应转向两类后续工作之一：

- 补做剩余 `P0 / P1` 教学图的轻量复核
- 或开始做“哪些图值得长期中文化”的名单筛选

## Next Suggested Steps

1. 先做 `ASCII / text block` 盘点：
   - 哪些只是短辅助骨架
   - 哪些已经和 SVG 重复
   - 哪些其实在替代主图职责
2. 对 `A. 核心教学图` 逐个截图或页面预览复核：
   - 文字是否溢出
   - 框和箭头是否拥挤
   - 长句是否影响阅读顺序
3. 对 `B. 路线收束图 / 项目图` 先判断：
   - 这张图是不是只服务路线定位
   - 是否把机制解释塞得过多
   - 是否应拆回正文 / 表格
4. 对 `38 / 43 / 44` 先做页面职责判断：
   - 是否继续保留为正式正文
   - 是否需要重画为更教学化的图
   - 是否应降级为过渡图 / 附录图
5. 等页面定位稳定后，再制定中文化名单，而不是一刀切。

## Reserved-Style Follow-up

`38 / 43 / 44` 这一组的当前结论已经明确：

- 正文本身已经不是预留页，而是正式教学页。
- 但图资产仍延续了“reserved page keeps space for / future page should...” 的旧语气。
- 因此这组图的第一优先级不是翻译，而是**改职责**。

当前推荐动作：

1. `38_pd_disaggregation.svg`
   - 保留 `Prefill Service -> Handoff -> Decode Service` 三段骨架
   - 删掉所有 “reserved / future page” 语气
   - 改成真正回答 “为什么拆池、代价是什么、什么时候值得拆”
2. `43_unified_memory_map.svg`
   - 保留 `GPU Tier / Host Tier / Policy` 结构
   - 删掉 “this reserved page should eventually explain”
   - 改成真正回答 “哪些对象常驻、哪些可迁移、迁移代价如何进入预算”
3. `44_auto_tuning_loop.svg`
   - 保留 `Goal / Search / Measure / Decide` 骨架
   - 删掉 “future page must prevent”
   - 改成真正回答 “约束先筛、统一评分、再输出推荐配置”
