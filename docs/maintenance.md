# 维护与发布手册

本页只保留当前需要执行的维护规则。目标是让导航、正文、Notebook、测试脚本和环境边界保持一致。

## 当前维护对象

- `README.md`：仓库总入口
- `docs/index.md`：站点首页
- `docs/guide.md`：环境与学习方式
- `docs/.vitepress/config.mts`：站点导航
- `00_Prerequisites/intro.md`、`01_Hardware_Math_and_Systems/intro.md`、`02_PyTorch_Algorithms/intro.md`、`03_CUDA_and_Triton_Kernels/intro.md`：各部分导学
- `docs/` 下对应页面：站点侧镜像
- `test_chapter0_1_notebooks.py`：第零部分 / 第一部分验证
- `test_notebook_answers.py`：第二部分 / 第三部分验证
- `check_chapter_links.py`：站内链接检查
- `SESSION_HANDOFF.md`：当前交接记录

## 当前结构

- 第零部分：前置知识与环境准备
- 第一部分：硬件、算力推导与系统级理论
- 第二部分：PyTorch 算法实战
- 第三部分：CUDA C++ 与 Triton 算子开发

Chapter / 第几章 这类表述只在正文桥接语境里保留，导航页统一使用“部分”。

## 导航规则

- `README.md`、`docs/index.md`、各部分 `intro.md` 视为同一套导航总览
- 组别名称和入口要可点击
- 导航页优先展示“部分 / 简介 / 组别 / 状态”
- 组级入口默认折叠，用户点开再看细节

## Notebook 与测试脚本

- 第零部分 / 第一部分当前共用 `test_chapter0_1_notebooks.py`
- 第二部分 / 第三部分当前共用 `test_notebook_answers.py`
- 公开 notebook 需要可跑、可测、结构完整
- 第一部分的公开 notebook 应尽量靠近第二部分 / 第三部分的 `TODO` + `🛑 **STOP HERE** 🛑` + `test_...()` 风格
- 第一部分后续如果完全收敛，再评估是否并入 `test_notebook_answers.py`
- 第零部分维持独立入口更清晰
- 理论页可以先占位，Notebook 不建议长期只占位

### 第零部分 Notebook 现状

- 已公开 notebook：`00`、`01`、`04`、`05`、`07`、`08`、`09`、`12`、`13`
- 仅正文、暂不挂 notebook：`02`、`03`、`06`、`10`、`11`
- 暂不挂 notebook 的页面先按正文页维护，不强行加入口
- 如果后续要补 notebook，再按 Chapter 2 风格统一补齐入口和测试

## 环境规则

- 第零部分 / 第一部分 / 第二部分共用 `conda activate llm_algo`
- 第三部分单独 GPU 环境，但要兼容基础依赖
- 第零部分和第一部分以 CPU-first 为主
- 第二部分以 CPU-first 为主，个别 notebook 需要 GPU
- 第三部分为 GPU-required

## 当前收尾顺序

1. 先收尾第一部分
2. 第一部分结束后单独检查第零部分
3. 再处理 `feat/platform-content-restructure` 分支上的其他改动

## P0 / P1 / P2 约定

- `P0`：已有正文草稿，优先收口
- `P1` / `P2`：先保持占位可达，只保留简短说明和后续更新提示
- 只有准备好的公开内容才挂更完整的入口

## 合并前检查

提交前至少确认：

- 站内链接通过
- `docs` 构建通过
- Notebook 测试通过
- 导航页口径与 `README.md` 一致
- 公开内容没有明显的强词或过时表述
