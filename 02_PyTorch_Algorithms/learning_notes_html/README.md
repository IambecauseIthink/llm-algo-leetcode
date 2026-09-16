# Learning Notes HTML

这个目录用于存放“Notebook 课前导学 + 闯关检查”HTML。

建议结构：

- `index.html`: 闯关地图，每个 notebook 对应一个关卡入口。
- `notes/`: 按 notebook 拆分的关卡页面。
- `assets/`: 图片、样式、数据或小型可视化资源。
- `generate_levels.js`: 关卡生成器，维护地图、页面模板和通用样式。
- `curriculum_v2.js`: upstream 新版 12–42 课程的元数据与 Notebook 映射。
- `lesson_overrides.js` / `lesson_overrides_extra.js`: 各关的零基础导学、图例、语法热身、闯关题和 notebook 作业数据。

当前地图对应官方 `518cc451c51a1e86d6e17f304147135d2a8cb379`（2026-09-16）：**75 个正式课程 + 15 个预留入口**。

- 所有 Notebook 入口直接指向 Datawhale 官方仓库的固定版本。
- `upstream_source.json` 固定官方提交；`curriculum_current.js` 只通过 Git blob 读取官方题目，不读取本地 Notebook。个人作业不会进入生成的 HTML。
- 后半段 45 课（30–52、60–86 中正式课程）按「生活问题 → 术语 → 数字推演 → 交互实验 → 最小 Python → 纠错 → 官方练习」重写。`teaching_specs.js` 维护逐课教学内容。
- `assets/learning_lab.js` 提供 21 类教学实验，支持参数调节、逐步播放、暂停、重置和计算反馈；所有数字都是简化模型计算，不是实际 GPU 测量。
- 新版进度使用 v4；基础课保留旧进度，重写的后半段重新答题。v1/v2/v3 原数据不删除。
- 页面支持桌面和手机、键盘操作及减少动态效果偏好。HTML 可离线阅读，官方 Notebook 链接需要联网。
- 旧 HTML 地址保留跳转；原有第 25 课量化工厂保留。
- 发布前执行 `npm run check:notebooks --prefix 02_PyTorch_Algorithms/learning_notes_html`，比对 Git 暂存区与官方所有 Notebook 的路径和 blob，拒绝个人改动或额外 Notebook。

每个关卡页面包含：

- 课程输入：根据 notebook TODO 反向拆出的必要知识
- 闯关答题：少量题目覆盖关键概念
- Notebook 作业：回到 `.ipynb` 中完成真正的代码练习

HTML 不提供在线写代码环境；最后的写代码是 notebook 作业，用来做举一反三和加强记忆。

## 视觉与教学布局原则

后续优化课程页面时，优先服务学习路径，不要为了“画图”而画图。页面里的每个视觉块都应该回答一个明确问题：学生现在要理解什么、下一步要写什么、怎么从玩具例子迁移到 notebook TODO。

- 短标签 + 长说明不要使用等宽卡片网格。比如 `TODO 1` / `TODO 2` 后面跟较长解释时，等宽 `.mini-table` 会造成短标签区域空太长、说明区域太挤。应改用语义表格、任务清单或左右比例明确的布局。
- `.mini-table` 只适合短词映射，例如 `toy_logits -> router_logits`、`switch -> gate_proj` 这类两边都很短的对应关系。若任一侧需要完整句子，改用 `<table class="freq-table">` 或普通段落。
- 图解原理的第 0 步通常承担“任务拆解”和“学习路线”职责，应该清楚、稳定、可扫读。优先用 `任务 / 要完成什么 / 检查点` 这种结构，而不是把多个短卡片硬凑成视觉图。
- 语法热身要遵循“玩具例子 -> notebook 变量映射 -> TODO 检查点”的顺序。先让初学者看懂 PyTorch 语法，再让他们举一反三完成 notebook。
- 视觉块的长度和权重应匹配内容价值。核心逻辑、常见错法、测试排查可以占更宽空间；装饰性流程块、图标和箭头要少用，避免打断阅读节奏。
- 每次新增视觉组件后，检查移动端和桌面端是否会出现明显空白、文本挤压或信息断裂。布局看起来不顺时，优先调整信息结构，而不是继续加图。

进度、答题状态和作业 checklist 都保存在访问者自己的浏览器 `localStorage` 中，不会写回 HTML 文件。因此你把这个目录分享出去时，别人看到的是原始初始版本，而不是你的个人记录。

可以直接在浏览器中打开 `index.html` 预览。修改课程内容后，运行下面命令重新生成静态页面：

```bash
npm ci --prefix 02_PyTorch_Algorithms/learning_notes_html
npm run generate --prefix 02_PyTorch_Algorithms/learning_notes_html
npm test --prefix 02_PyTorch_Algorithms/learning_notes_html
```

历史同步记录见 [SYNC_REPORT.md](SYNC_REPORT.md)，本轮教学与 UI 验收见 [UI_REVIEW.md](UI_REVIEW.md)。
