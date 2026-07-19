const esc = (value) => String(value)
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;");

const code = (title, lines) => `<div class="syntax-card"><h4>语法热身：${esc(title)}</h4><pre><code>${lines.map(esc).join("\n")}</code></pre></div>`;

const checkpoint = (question, options, answer, explain) => ({ question, options, answer, explain });

const lesson = ({ id, title, todo, prerequisite, intuition, exampleHtml, syntaxHtml, styles, predict, checkpoint: lessonCheckpoint, homework }) => ({
  id,
  title,
  todo,
  prerequisite,
  intuition,
  exampleHtml,
  syntaxHtml,
  styles,
  predict,
  checkpoint: lessonCheckpoint,
  homework
});

const advancedStyles = `
    .hero,
    .hero > *,
    .hero .meta,
    .hero .map-steps {
      min-width: 0;
    }
    .hero .formula,
    .hero .pill {
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .lesson-card.inquiry,
    .lesson-card.inquiry .lesson-top,
    .lesson-card.inquiry .gated,
    .lesson-card.inquiry .lesson-section {
      min-width: 0;
    }
    .lesson-card.inquiry .todo-pill,
    .lesson-card.inquiry h2,
    .lesson-card.inquiry code {
      overflow-wrap: anywhere;
    }
    .lesson-card.inquiry .predict-option > span,
    .lesson-card.inquiry .checkpoint-option > span,
    .lesson-card.inquiry .homework-item > span {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .adv-course,
    .adv-practice {
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }
    .adv-course h4,
    .adv-practice h4 {
      margin: 0;
      font-size: 17px;
      color: #182033;
    }
    .adv-course p,
    .adv-practice p { margin: 0; }
    .adv-roadmap {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      border: 1px solid #cdd9ef;
      border-radius: 8px;
      overflow: hidden;
      background: #f7faff;
    }
    .adv-roadmap > span {
      min-height: 78px;
      padding: 12px;
      display: grid;
      align-content: center;
      gap: 4px;
      border-right: 1px solid #cdd9ef;
      color: #42516b;
      line-height: 1.45;
    }
    .adv-roadmap > span:last-child { border-right: 0; }
    .adv-roadmap b {
      color: #1d4ed8;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .adv-grid {
      display: grid;
      gap: 12px;
      align-items: stretch;
    }
    .adv-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .adv-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .adv-panel,
    .adv-example,
    .adv-map {
      padding: 14px;
      border: 1px solid #dce3eb;
      border-radius: 8px;
      background: #fff;
      display: grid;
      gap: 10px;
      align-content: start;
      min-width: 0;
    }
    .adv-panel.neutral { border-top: 4px solid #657184; }
    .adv-panel.good,
    .adv-example { border-top: 4px solid #16835f; background: #f5fbf8; }
    .adv-panel.warn { border-top: 4px solid #d97706; background: #fff9ef; }
    .adv-panel.blue { border-top: 4px solid #2563eb; background: #f7faff; }
    .adv-flow {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 8px;
    }
    .adv-flow > span,
    .adv-flow > strong {
      min-height: 62px;
      padding: 9px;
      border: 1px solid #dce3eb;
      border-radius: 6px;
      background: #fff;
      display: grid;
      place-items: center;
      text-align: center;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .adv-flow > strong {
      border-color: #99d6bf;
      background: #e9f7f0;
      color: #125b45;
    }
    .adv-contract {
      display: grid;
      grid-template-columns: minmax(130px, 0.34fr) minmax(0, 1fr);
      border: 1px solid #dce3eb;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .adv-contract > span,
    .adv-contract > strong {
      min-height: 48px;
      padding: 10px 12px;
      display: grid;
      align-items: center;
      border-bottom: 1px solid #dce3eb;
      line-height: 1.45;
    }
    .adv-contract > span {
      border-right: 1px solid #dce3eb;
      background: #f7f9fc;
      color: #526077;
      font-weight: 800;
    }
    .adv-contract > strong {
      overflow-wrap: anywhere;
    }
    .adv-contract > span:nth-last-child(-n+2),
    .adv-contract > strong:nth-last-child(-n+2) { border-bottom: 0; }
    .adv-contract code,
    .adv-mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .adv-shapes {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      background: #fff;
      border: 1px solid #dce3eb;
    }
    .adv-shapes th,
    .adv-shapes td {
      padding: 10px;
      border: 1px solid #dce3eb;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    .adv-shapes th {
      background: #eaf1ff;
      color: #1d4ed8;
    }
    .adv-shapes code { white-space: normal; }
    .adv-steps {
      display: grid;
      gap: 8px;
    }
    .adv-steps > div {
      display: grid;
      grid-template-columns: 36px minmax(180px, 0.72fr) minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid #dce3eb;
      border-radius: 7px;
      background: #fff;
    }
    .adv-steps > div > b {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: #2563eb;
      color: #fff;
    }
    .adv-steps code {
      color: #172033;
      font-weight: 800;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .adv-steps span { color: #5f6d82; }
    .adv-callout {
      padding: 12px 14px;
      border-left: 4px solid #d97706;
      border-radius: 6px;
      background: #fff6e7;
      color: #74400d;
      line-height: 1.6;
      font-weight: 750;
    }
    .adv-map { background: #f7f9fc; }
    .adv-map ul,
    .adv-map ol { margin: 0; padding-left: 20px; }
    .adv-map li { margin: 6px 0; line-height: 1.5; }
    .adv-practice {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: stretch;
    }
    .adv-practice .syntax-card { margin-top: 0; }
    .adv-checks {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .adv-checks > span {
      min-height: 70px;
      padding: 10px;
      border: 1px solid #dce3eb;
      border-radius: 7px;
      background: #fff;
      display: grid;
      place-items: center;
      text-align: center;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    @media (max-width: 760px) {
      .adv-grid.two,
      .adv-grid.three,
      .adv-practice { grid-template-columns: 1fr; }
      .adv-roadmap { grid-template-columns: 1fr; }
      .adv-roadmap > span {
        border-right: 0;
        border-bottom: 1px solid #cdd9ef;
      }
      .adv-roadmap > span:last-child { border-bottom: 0; }
      .adv-contract { grid-template-columns: 1fr; }
      .adv-contract > span { border-right: 0; }
      .adv-contract > span:nth-last-child(-n+2) { border-bottom: 1px solid #dce3eb; }
      .adv-steps > div { grid-template-columns: 36px minmax(0, 1fr); }
      .adv-steps > div > span { grid-column: 2; }
      .adv-shapes { font-size: 13px; }
      .adv-shapes th,
      .adv-shapes td { padding: 7px; }
    }`;

module.exports = {
  advancedStyles,
  checkpoint,
  code,
  esc,
  lesson
};
