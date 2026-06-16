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

进度、答题状态和作业 checklist 都保存在访问者自己的浏览器 `localStorage` 中，不会写回 HTML 文件。因此你把这个目录分享出去时，别人看到的是原始初始版本，而不是你的个人记录。

可以直接在浏览器中打开 `index.html` 预览。修改课程内容后，运行下面命令重新生成静态页面：

```bash
node 02_PyTorch_Algorithms/learning_notes_html/generate_levels.js
```
