# Learning Notes HTML

这个目录用于存放“Notebook 课前导学 + 闯关检查”HTML。

建议结构：

- `index.html`: 闯关地图，每个 notebook 对应一个关卡入口。
- `notes/`: 按 notebook 拆分的关卡页面。
- `assets/`: 图片、样式、数据或小型可视化资源。
- `generate_levels.js`: 关卡生成器，维护地图、页面模板和通用样式。
- `lesson_overrides.js` / `lesson_overrides_extra.js`: 各关的零基础导学、图例、语法热身、闯关题和 notebook 作业数据。

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
node 02_PyTorch_Algorithms/learning_notes_html/generate_levels.js
```
