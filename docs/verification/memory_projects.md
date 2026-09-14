# 73–76 显存优化项目验证清单

本页是 `Part 02` 显存优化项目的执行清单，负责说明运行顺序、结果文件、预算敏感性和 74 的 profiling 收口要求；它不替代各 Notebook 的机制讲解和题目答案。

## 验证顺序

```text
静态检查 → 答案区回归 → 73 训练基线 → 76 策略 benchmark → 75 预算决策 → 74 profiling 收口
```

## 1. 静态检查与答案区回归

在仓库根目录执行：

```bash
python -m py_compile \
  tools/training_memory_result_schema.py \
  tools/normalize_training_memory_results.py \
  tools/profiling_result_schema.py

for f in \
  73_Training_Performance_Analysis.ipynb \
  74_Profiling_Driven_End_to_End_Optimization.ipynb \
  75_Memory_Budget_Compression_Project.ipynb \
  76_Activation_Checkpoint_Offload_Benchmark.ipynb; do
  python tools/test_notebook_answers.py \
    "02_PyTorch_Algorithms/$f" --mode answer
done

python tools/check_docs_links.py \
  --skip-convert --skip-mirror-check --skip-build
```

真实 GPU 单元默认关闭；答案区通过不代表已经完成真实 GPU 验证。

## 2. 项目分工与结果文件

| 项目 | 作用 | 运行开关 | 原始结果 | 统一旁路结果 |
|---|---|---|---|---|
| 73 | 建立训练性能与显存 baseline | `RUN_REAL_GPU = True` | `73_real_gpu_training.json` | `73_real_gpu_training_v1.json` |
| 76 | 比较 checkpoint / offload / hybrid | `RUN_REAL_GPU = True` | `76_real_gpu_memory.json` | `76_real_gpu_memory_v1.json` |
| 75 | 在预算和质量约束下做决策 | `RUN_REAL_PROJECT = True` | `75_memory_budget_decision.json` | `75_memory_budget_decision_v1.json` |
| 74 | 用 profiling 解释并验证最终方案 | 当前需补真实运行入口 | `74_profiling_optimization.json` | 使用 `profiling-optimization/v1` |

原始 JSON 是实验记录，不能覆盖；`*_v1.json` 是兼容归一化结果。当前已有的 73、76、75 结果无需因为 schema 统一而重跑。

## 3. 73 训练基线

固定以下条件：

- 模型、模型版本、GPU、驱动、PyTorch/CUDA；
- batch size、seq_len、dtype、optimizer；
- warmup、iters、seed 和输入数据；
- baseline 与 tuned 只改变一个变量。

每个配置建议重复 3 次，记录：

```text
step_time_ms / samples_per_s
peak_memory_mb / peak_reserved_mb
train_loss / eval_loss / loss_delta
model / dtype / device / torch / CUDA
```

73 的 AMP 对比必须同时检查 loss 或 eval loss，不能只依据时间和峰值显存判断优化成功。

## 4. 76 策略 benchmark

核心候选为：

```text
baseline / checkpoint / offload / hybrid
```

建议分两档运行：

| workload | 建议 seq_len | 目的 |
|---|---:|---|
| `smoke` | 512 | 快速确认策略和保存链路可运行 |
| `pressure` | 768 | 比较显存收益与吞吐代价 |

条件允许时继续增大 seq_len，记录 OOM 边界。每个候选记录：

- `status`、OOM 错误摘要；
- step time、samples/s；
- peak allocated、peak reserved；
- train/eval loss；
- checkpoint 重算、offload 搬运等策略专属代价。

75/76 已处理 OOM、缺失字段和无效 baseline：异常候选会计入 `oom_count` 或 `invalid_count`，不会导致汇总单元直接崩溃。

## 5. 75 预算敏感性

75 不重新测量模型，而是使用 76 的候选结果测试不同约束。建议运行以下网格：

| 参数 | 建议取值 |
|---|---|
| `memory_cap_mb` | `9500 / 10000 / 11200` |
| `min_samples_per_s` | baseline 的 `0.5 / 0.7 / 0.9` |
| `max_val_loss` | baseline 的 `+1% / +2% / +5%` |

每个组合至少保存：

```text
feasible_names
best_candidate
memory_saving_mb
throughput_ratio
decision
```

如果候选只在很窄的阈值范围内 `accept`，应标记为敏感结论；如果显存收益低于显著阈值，通常输出 `tune`，而不是强行 `accept`。

当前本地结果中 checkpoint 约节省 332 MB、吞吐约为 baseline 的 80%，低于 512 MB 显著显存收益阈值，因此当前 `tune` 结论合理。

## 6. 74 profiling 收口

74 使用 76 选出的候选作为 tuned，与 baseline 使用同一 workload。建议：

```text
seq_len=768
warmup=3
iters=10
每个候选重复 3 次
```

除 73/76 的公共指标外，还要保存：

```text
profile.tool
profile.trace_path
profile.top_operators
profile.compute_ratio
profile.memory_ratio
profile.communication_ratio
bottleneck.category
bottleneck.evidence
bottleneck.optimization
validation
decision
```

单 GPU 没有真实通信时使用 `not_applicable`，不要将未测量的通信占比写成 `0`。建议结果路径为：

```text
benchmarks/results/74_profiling_optimization.json
```

结果协议由 `tools/profiling_result_schema.py` 提供。当前 74 只有 CPU-first 模板和协议定义，尚无真实 profiling JSON；因此 73–76 链路仍未完全收口。

## 7. 结果检查

检查原始和旁路结果：

```bash
ls -lh benchmarks/results/73_real_gpu_training*.json \
  benchmarks/results/76_real_gpu_memory*.json \
  benchmarks/results/75_memory_budget_decision*.json \
  benchmarks/results/74_profiling_optimization*.json
```

统一判断顺序为：

```text
73 是否建立可复现 baseline
→ 76 是否比较了候选和代价
→ 75 是否在多组预算下保持决策稳定
→ 74 是否用 profiling 证据解释端到端收益
```

