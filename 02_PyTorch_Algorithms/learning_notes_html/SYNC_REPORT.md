# 官方同步与 HTML 更新记录 · 2026-09-14

官方来源：[datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)，本次 `git fetch upstream` 获取的 main 为 [`4fa6262`](https://github.com/datawhalechina/llm-algo-leetcode/commit/4fa62623ae7f2f207871b43c3f254a765765bf2e)，提交时间 2026-09-11 21:57:48 +0800。

本地起点：`b27d7a4`。同步前工作区干净；完整旧版本保留在本地分支 `backup/pre-upstream-sync-2026-09-14`，包括旧 Notebook、练习实现和 HTML。仅完成本地同步，没有推送远程仓库。

## 官方有哪些变化

直接比较本地起点和官方 main（不含本地独有 HTML）：934 个文件发生变化，增加 97,188 行、删除 15,520 行。此前的同步未完整保留官方合并关系，因此不能把 `HEAD..upstream/main` 的提交数解释为本次新增内容数量；本次使用真正的双亲合并建立后续同步关系。

| 范围 | 本次采用的官方版本 |
|---|---|
| Part 00 | 20 个前置 Notebook，完善 Python、PyTorch、训练接口和调试/显存账本 |
| Part 01 | 33 个硬件、数学与系统 Notebook，更新图解和文档 |
| Part 02 | 00–89 共 90 个 Notebook：75 个正式主题，15 个结构化预留页；分成 10 个专题组 |
| 方法扩展 | 长上下文、LoRA 变体、SFT 数据、微调准备、PD 分离、fallback、offload、统一内存、自动调优、专家并行、在线对齐等 |
| 项目路线 | 60 起承接微调、架构、推理、量化、缓存、调度、MLA、显存、并行和偏好优化项目 |
| 专题与工具 | 更新推理、显存、量化、通信、profiling、架构和对齐路线；引入/更新 benchmark 结果、运行时、依赖配置和 CI 验证 |

官方 Notebook 全部采用当前版本。原先按旧编号保留的重复 Notebook 和文档由新版对应课程替代，旧内容可从备份分支恢复。

## HTML 如何对应新版

- 43 个旧主题精讲按文件主题匹配到新版编号，新增 32 个正式课程页。总计 75 课；15 个预留页以明确的预留入口显示，不伪装成已完成课程。
- 每课展示最新 Notebook 标题、导读、TODO、题目与测试代码，提供官方固定版本和本地 Notebook 入口。以 `STOP HERE` / 参考答案标题作为题目区边界，不提前展开官方答案。
- 新课程采用官方原理与图解，补充主题检查题。原有精讲保持为概念练习，Notebook 作业统一指向当前题目区，避免继续把旧 TODO 签名当作新版要求。
- Markdown 使用正式解析器，数学公式由 KaTeX 静态渲染，样式与字体随页面保存。官方图片链接解析到本仓库 `docs/public`，同章课间链接指向对应 HTML。
- 地图增加中英文/编号搜索、新版分层路线与预留章节列表。手机视口下图、代码、公式、表格可在各自容器内缩放或滚动。
- 所有 13 个改编号主题保留旧 HTML 跳转。通关/答题记录从 v1/v2 迁移到 v3；原存储不删除，新编号不会继承原编号的不同主题进度。新版作业记录按源文件哈希隔离。
- 第 25 课的量化工厂保留。

| 旧编号 | 新编号 | 主题 |
|---|---|---|
| 30 | 60 | LoRA 微调项目 |
| 31 | 66 | 推理性能对比 |
| 32 | 73 | 训练性能分析 |
| 33 | 74 | Profiling 驱动端到端优化 |
| 34 | 79 | 分布式并行基准 |
| 35 | 67 | 量化推理与部署 |
| 36 | 34 | 前缀缓存与分块 Prefill |
| 37 | 35 | 多 Token 解码 |
| 38 | 36 | Decode 调度 |
| 39 | 40 | GPTQ / AWQ |
| 40 | 41 | FP8 / KV Cache 量化 |
| 41 | 37 | KV Cache 调度 |
| 42 | 46 | NCCL 通信 Profiling |

完整文件映射和源文件 SHA-256 见 `course_manifest.json`。

## 本地修复与验证

额外修复官方 docs 中 10 个 requirements 链接和 1 个错误的 Part 00/Part 02 章节链接；镜像检查排除本地独立 HTML 工具目录与 node_modules。首页由官方同步脚本重新生成。

| 检查 | 结果 |
|---|---|
| `npm test --prefix 02_PyTorch_Algorithms/learning_notes_html` | 75 课、15 预留、954 个本地链接、77 段脚本语法、旧地址映射、进度迁移及幂等性通过 |
| 量化工厂原有测试 | 通过；cosine=0.999993，为工厂自身确定性示例验证 |
| `python3 tools/check_docs_links.py --skip-convert --skip-build` | Source/docs mirror 通过；missing_count=0 |
| `npm run docs:build`（docs 目录） | VitePress 构建成功 |
| 浏览器 | 搜索“长上下文”、新课答题反馈、旧 LoRA 地址转到新 60、390px 宽度无页面横向溢出、未记录到控制台 error |
| Notebook/GPU 执行 | 本次不执行模型、Profiler 或 benchmark；上述结果是内容、生成器和网页验证，不代表课程中的性能结论由本机复测 |
