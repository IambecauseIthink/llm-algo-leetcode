# 使用指南

本页只保留三件事：用什么环境、用什么测试脚本、哪些页需要 GPU。

## 一句话结论

| 目标 | 推荐环境 | 备注 |
|---|---|---|
| 第零部分 / 第一部分 | `llm_algo` 或在线 Notebook | 以 Python、Jupyter、NumPy、PyTorch 为主 |
| 第二部分 | `llm_algo` | CPU-first；少数页需要 GPU |
| 第三部分 | 独立 GPU 环境 | 需要 Linux + NVIDIA GPU + CUDA / Triton |
| 团队统一验证 | CNB / Docker / 云端 GPU | 用于复现和协作 |

## 环境规则

- `conda activate llm_algo` 是第零部分、第一部分、第二部分的共享验证环境。
- 第三部分单独使用 GPU 环境，但要尽量兼容前面章节的基础依赖。
- 第零部分更偏入门和轻量验证，优先保持简单。
- 第二部分整体是 CPU-first，`21_Gradient_Checkpointing` 需要真实 CUDA 显存峰值。
- 第三部分是 GPU-required，不应把 GPU 需求扩散到前面章节。

## 测试脚本

- 第零部分 / 第一部分：`test_chapter0_1_notebooks.py`
- 第二部分 / 第三部分：`test_notebook_answers.py`

### 约定

- 公开 notebook 要尽量保持可跑、可测、结构完整。
- 第零部分 / 第一部分的公开 notebook 先向 Chapter 2 / 3 的模板靠拢，但不强制变成 GPU-only。
- 第三部分保持 GPU-aware 的运行模式开关，不把硬件要求写死在单一脚本里。

## Notebook 状态

- **有 notebook 的页面**
  - 第零部分：`00, 01, 04, 05, 07, 08, 09, 12, 13`
  - 第一部分：`03, 06, 21, 22, 29`
- **先按正文维护**
  - 第零部分：`02, 03, 06, 10, 11`
- **占位页**
  - 第一部分扩展池：`24, 25, 27, 28, 30, 31, 32`

## 相关阅读

- [维护与发布手册](./maintenance.md)
- [第零部分导学](./00_Prerequisites/intro.md)
- [第一部分导学](./01_Hardware_Math_and_Systems/intro.md)
- [第二部分导学](./02_PyTorch_Algorithms/intro.md)
- [第三部分导学](./03_CUDA_and_Triton_Kernels/intro.md)
