# SVG 制图组件与颜色规范

本文件是教程 SVG 的轻量组件参考。它不要求所有图片使用同一套布局，但要求颜色语义、文字层级和箭头行为保持一致。

## 最小画布骨架

```svg
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 1200 520"
     role="img"
     aria-labelledby="title desc">
  <title id="title">中文图片标题</title>
  <desc id="desc">用一句话描述图片展示的关系。</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10"
            refX="8" refY="3" orient="auto">
      <path d="M0,0 L9,3 L0,6 Z" fill="#475569" />
    </marker>
  </defs>
  <rect width="1200" height="520" fill="#F8FAFC" />
</svg>
```

## 组件约定

```css
/* 标题、节点标题、节点正文、辅助说明 */
.title { font-size: 25px; font-weight: 700; }
.heading { font-size: 18px; font-weight: 700; }
.body { font-size: 15px; font-weight: 500; }
.small { font-size: 14px; font-weight: 500; }

/* 节点和连线 */
.node { stroke-width: 2; rx: 14; }
.flow { stroke: #475569; stroke-width: 2.5; fill: none;
        marker-end: url(#arrow); }
.support { stroke: #94A3B8; stroke-width: 2; fill: none;
           stroke-dasharray: 7 6; marker-end: url(#arrow); }
```

字体优先使用：

```css
font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
```

## 颜色语义

| 语义 | 填充色 | 边框色 |
|:---|:---|:---|
| 页面背景 | `#F8FAFC` | — |
| 共同输入 / 前置 | `#DBEAFE` | `#2563EB` |
| 计算 / 机制 | `#FEF3C7` | `#D97706` |
| 候选 / 可行 | `#DCFCE7` | `#16A34A` |
| 观察 / 指标 | `#E0F2FE` | `#0284C7` |
| 决策 / 交付 | `#EDE9FE` | `#7C3AED` |
| 警告 / 调整 | `#FEF3C7` | `#F59E0B` |
| 拒绝 / 失败 | `#FFF1F2` | `#E11D48` |

## 使用规则

1. 主图回答“这些概念如何联系”；第二图回答“如何实验、比较或决策”。
2. 一个节点尽量只有概念名和一行解释；字段列表、参数和结果放 Markdown 表格。
3. 采用总分结构，不在图片底部重复增加总结节点。
4. 连接线不能穿过节点或文字；分支尽量使用直线，必要时扩大节点间距。
5. SVG 只表达结构关系，不表达未经实测的速度、显存或质量收益。
6. 修改后至少运行 XML 校验、路径检查和 `git diff --check`。
