# 60–65 训练微调项目验证清单

本页统一 60–65 的项目验证口径。它不要求所有项目使用相同指标，而是要求所有项目使用相同的报告骨架：

```text
13 训练基线 → 60–65 问题分流 → 05 项目交付与决策
```

## 项目分工

| 项目 | 问题类型 | 核心证据 | 项目级别 |
|:---|:---|:---|:---|
| 60 | 第一个完整 LoRA 交付 | adapter、质量、资源、复现和采用决策 | L1 核心项目 |
| 61 | 模型结构与 target modules 探索 | 结构差异、参数量、显存、步时、任务得分 | L1 扩展项目 |
| 62 | 指令格式与任务适配 | 格式通过率、样例、训练与验证结果 | L1 扩展项目 |
| 63 | LoRA 变体比较 | rank、alpha、dropout、target modules、质量/资源对比 | L1 扩展项目 |
| 64 | SFT 数据质量审计 | 空样本、重复样本、长度、清洗比例、readiness | L0/L1 准入项目 |
| 65 | QLoRA 与显存预算选型 | 位宽、显存、吞吐、质量下限、可行性 | L1/L2 扩展项目 |

`13` 先提供可比较的训练基线；`60–65` 只在对应问题出现时进入，不要求全部连续运行；`05` 最后检查产物是否完整并输出交付决策。

## 统一报告骨架

每个项目可以保留自己的策略字段，但结果至少应包含：

```text
schema_version / project / stage
config / baseline / candidates
quality / resources / artifacts
decision / environment
```

| 区域 | 最少记录 | 作用 |
|:---|:---|:---|
| `config` | model、dataset、dtype、batch、seq_len、seed | 固定实验口径 |
| `baseline` | 无微调或当前默认方案 | 提供比较参照 |
| `candidates` | 每个候选方案及状态 | 保留比较过程，不只保存赢家 |
| `quality` | train/val loss、任务指标、样例检查 | 判断效果是否达标 |
| `resources` | 可训练参数、显存、步时、吞吐 | 判断成本是否可接受 |
| `artifacts` | adapter、tokenizer、config、metrics、report | 支撑复现和交付 |
| `decision` | accept / tune / reject、原因、下一步 | 把实验变成项目结论 |
| `environment` | Python、PyTorch、CUDA、设备、运行入口 | 解释环境差异 |

公共协议由 `tools/fine_tuning_result_schema.py` 提供。它只规范外层结构，不会替项目补造缺失的质量或资源数据。

## 61–65 自动化入口

61–65 是 CPU-first 的项目模板，不会默认下载模型或启动训练。每节末尾的可选导出单元统一使用 `tools/fine_tuning_project_runtime.py`：

| 配置 | 默认值 | 作用 |
|:---|:---:|:---|
| `PROJECT_ID` | 当前项目编号 | 写入报告的项目标识 |
| `PROJECT_RESULT_PATH` | `benchmarks/results/<project>.json` | 自动创建目录并保存报告 |
| `PROJECT_CONFIG` | Notebook 示例配置 | 校验 model、dtype、batch、seq_len、steps、seed |
| `RUN_PROJECT_EXPORT` | `False` | 只有在结果已经完成后才导出 JSON |

推荐流程：

1. 先完成本节的 baseline、candidate、quality 和 decision 计算。
2. 将这些结果组装成 `PROJECT_REPORT`，不要用模板演示数据代替实测数据。
3. 确认 `RUN_PROJECT_EXPORT = True` 后运行导出单元。
4. 检查生成的 JSON 是否包含 `baseline`、`candidates`、`quality`、`resources` 和 `decision`。

如果 Notebook 不是从仓库根目录启动，CPU-first 题目测试仍可运行；要导出正式报告，应从仓库根目录运行，或先将仓库根目录加入 `sys.path`。

### 最小报告模板

下面模板只表示报告外层结构，不包含任何实验结果。学习者应在完成本节实验后，用真实的 baseline、candidate、quality 和 resources 替换占位内容，再运行导出单元：

```python
PROJECT_REPORT = {
    "schema_version": "fine-tuning-project/v1",
    "project": PROJECT_ID,
    "stage": "project_decision",
    "config": PROJECT_CONFIG,
    "baseline": {"status": "not_run"},
    "candidates": [],
    "quality": {"task_metrics": {}},
    "resources": {},
    "artifacts": {"report": PROJECT_RESULT_PATH},
    "decision": {
        "decision": "tune",
        "reason": "evidence_is_incomplete",
        "next_action": "complete_validation",
    },
    "environment": runtime_snapshot(),
}
```

项目特有字段直接放在对应区域：61 记录结构差异，62 记录格式与样例评测，63 记录 LoRA 变体，64 记录数据审计，65 记录量化和预算约束。不要把空模板直接当作正式实验报告。

## 项目特有字段

- `60`：LoRA 配置、adapter 路径、merge 状态。
- `61`：架构差异、参数量、部署成本和结构得分。
- `62`：instruction/chat template、格式检查和样本通过率。
- `63`：rank、alpha、dropout、target modules 和变体排名。
- `64`：数据审计计数、长度分位数、重复率和清洗规则。
- `65`：量化格式、bit width、显存预算、吞吐和质量阈值。

`64` 的主要产出是数据准入结论，不应为了“看齐”其他项目而虚构模型吞吐；`61` 的主要产出是结构选择依据，也不必强行套用数据质量指标。

## 验证顺序

在仓库根目录执行：

```bash
python -m py_compile tools/fine_tuning_result_schema.py
python tools/test_notebook_answers.py \
  --dir 02_PyTorch_Algorithms \
  --mode answer
python tools/check_docs_links.py \
  --skip-convert --skip-mirror-check --skip-build
```

真实 GPU 或 Colab / ModelScope 运行时，还应保存模型、数据、dtype、batch、seq_len、seed、训练步数和环境信息；CPU-first 模板只能证明流程可运行，不能替代真实训练结论。

61–65 的统一 runtime 只负责配置校验、环境快照和报告保存，不负责自动调参、自动改 batch，也不会在 OOM 后偷偷降低实验规模。这样可以避免“程序跑完了”被误读成“实验结论成立”。

## 60 真实模型 smoke test

60 的可选 Step 6 默认关闭。将 `RUN_REAL_TRAINING = True` 后，依赖安装单元会默认使用当前 Notebook 内核自动安装 `transformers`、`peft`、`accelerate`、`datasets` 和 `httpx[socks]`；若模型或数据源使用 ModelScope，还会自动安装 `modelscope`。其中 `httpx[socks]` 用于兼容带有 `socks5` / `all_proxy` 的网络环境。缓存、结果和本地数据路径均由 Notebook 根据仓库根目录自动推导，不需要学习者填写路径。受限网络环境可以将 `AUTO_INSTALL_REAL_DEPS = False`，改用平台预装或手动安装。之后 Notebook 会：

1. 通过 `MODEL_SOURCE` 自动解析本地、Hugging Face 或 ModelScope 模型。
2. 加载 tokenizer 和真实基座模型。
3. 挂载 `q_proj / v_proj` LoRA adapter。
4. 在固定小批量样本上运行少量训练 step。
5. 保存 adapter、tokenizer 和 `benchmarks/results/60_real_lora/60_real_lora.json`。

默认路径约定为：模型缓存放在仓库根目录的 `model_cache/`，结果放在 `benchmarks/results/60_real_lora/`；选择 `local` 数据源时，Notebook 会自动搜索 `benchmarks/data/` 和 `data/` 下的 JSON/JSONL 文件。

如果关闭自动安装，Colab / ModelScope 可按平台手动安装依赖：

```bash
pip install -U transformers peft accelerate
# ModelScope 网络或模型源：
pip install -U modelscope
```

真实 smoke test 的决策默认是 `tune`，因为它没有运行同口径 baseline，也没有验证集。只有补齐匹配 baseline、验证集和多步稳定性后，才可以把结果升级为正式 `accept / reject` 结论。

真实数据集可选 `inline`、Hugging Face、ModelScope 或本地 JSON/JSONL。远程或本地记录至少应能映射到 `instruction / input / output` 或 `prompt / response`；正式实验还应固定数据集版本、抽样数量、字段映射和 train/validation 划分，并把数据审计结果写入 `quality.data_audit`。

### 60 matched baseline 验证

在 60 的 Step 7 中设置 `RUN_REAL_MATCHED = True`。Notebook 会自动按 `MATCHED_VAL_RATIO` 划分数据，使用 `MATCHED_BATCH_SIZE` 分批，分别运行 full-parameter baseline 和 LoRA，并保存：

```text
benchmarks/results/60_real_lora/60_real_lora_matched.json
benchmarks/results/60_real_lora/matched_adapter/
```

建议第一次使用 `MATCHED_BATCH_SIZE = 1`、`MATCHED_STEPS = 10`、`REAL_MAX_SEQ_LEN = 256`。12 GB 左右显存环境先保持小模型和小步数；如果 baseline OOM，保留 LoRA smoke test，并把 baseline 标记为不可行，不要强行比较。

## 决策边界

- `accept`：质量达到目标，资源成本可接受，artifact 和报告齐全。
- `tune`：方向成立，但数据、配置、训练控制或证据链仍需补齐。
- `reject`：没有证明优于 baseline，或结果无法复现和交付。

不要把“loss 下降”直接当成 `accept`，也不要把“某个候选较快”当成最终采用依据。
