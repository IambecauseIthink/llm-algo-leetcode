const plans = require('./teaching_specs');
const {models} = require('./assets/learning_lab');
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function labHtml(kind,id) {
  const model=models[kind];
  return `<section class="learning-lab" data-lab="${kind}" aria-label="${esc(model.title)}">
    <header><span class="lab-eyebrow">动手实验室 / 只在浏览器中演示</span><h4>${esc(model.title)}</h4><p>${esc(model.purpose)}</p></header>
    <div class="lab-controls">${model.controls.map(c=>`<label for="lab-${id}-${c.key}"><span>${esc(c.label)} <output data-value="${c.key}" for="lab-${id}-${c.key}">${c.value}</output> ${esc(c.unit)}</span><input type="range" id="lab-${id}-${c.key}" name="${c.key}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}"></label>`).join('')}</div>
    <div class="lab-metrics" data-metrics></div><div class="lab-bars" data-bars></div>
    <div class="lab-tokens" data-tokens aria-label="计算对象"></div>
    <div class="lab-phase-track" aria-hidden="true"><i data-phase-progress></i></div><div class="lab-phase-labels" data-phase-labels></div>
    <div class="lab-story" data-frame aria-live="polite"><small data-step></small><h4 data-frame-title></h4><p data-frame-text></p></div>
    <div class="lab-actions"><button type="button" data-prev>上一步</button><button type="button" data-next>下一步</button><button type="button" data-play>播放推演</button><button type="button" data-reset>重置实验</button></div>
    <form class="lab-challenge" onsubmit="return false"><label for="answer-${id}" data-challenge-label>算一算</label><div><input id="answer-${id}" data-answer type="number" step="any" inputmode="decimal" placeholder="填一个数字"><button type="button" data-check-answer>检查我的计算</button></div><p data-answer-feedback role="status"></p></form>
    <p class="lab-caption">教学简化模型：所有数值来自上方规则，不是 GPU 性能实测。播放可以暂停，键盘方向键可调整滑块。</p>
  </section>`;
}
function makeAdvanced(level) {
  const p=plans[Number(level.id)];if(!p)throw new Error(`Missing teaching plan ${level.id}`);
  const [question,correct,wrong,explain]=p.check;
  const [firstTerm,firstMeaning]=p.terms[0];
  const toQuiz=(q,options,answer,why)=>({question:q,options,answer,explain:why});
  const prerequisite=p.terms.map(([term,meaning])=>term+'：'+meaning);
  const officialWork=[`到官方 Notebook 阅读题目函数与测试；先写出输入、输出和一个最小例子。`, `把本课手算过程转换成实现，先跑最小测试，再检查空输入、边界值或未达门槛的情况。`, `记录一个与预期不同的结果，并用本课术语说明原因。`];
  return [
    {id:'v4-story',title:p.hook,todo:'01 / 建立直觉',prerequisite:[],intuition:p.analogy,
      exampleHtml:`<div class="concept-glossary"><h4>先认识这三个词</h4><dl>${p.terms.map(([term,meaning])=>`<div><dt>${esc(term)}</dt><dd>${esc(meaning)}</dd></div>`).join('')}</dl></div><div class="mental-prompt"><b>试着说给朋友听</b><p>先不看公式：用上面的生活场景，说一说这节课想减少哪种浪费、需要付出什么代价。</p></div>`,
      checkpoint:toQuiz(`本课中的「${firstTerm}」指什么？`,[firstMeaning,p.terms[1][1],p.terms[2][1]],0,`它指的是：${firstMeaning}。另外两个选项分别解释「${p.terms[1][0]}」和「${p.terms[2][0]}」，注意区分。`),
      homework:['不看术语表，用一句话解释三个新词之间的区别。']},
    {id:'v4-experiment',title:'把一个小例子算到最后',todo:'02 / 操作与推演',prerequisite:[],intuition:'先用默认数值手算，再只改一个参数。让结果来检验你的猜想。',
      exampleHtml:`<ol class="worked-example">${p.steps.map(([title,text])=>`<li><h4>${esc(title)}</h4><p>${esc(text)}</p></li>`).join('')}</ol>${labHtml(p.lab,level.id)}`,
      checkpoint:toQuiz(question,[correct,...wrong],0,explain),homework:['改一个滑块，先预测结果，再按步骤核对。说明哪一项变了、哪一项没变。']},
    {id:'v4-transfer',title:'从会看，走到会写与会判断',todo:'03 / 纠错与迁移',prerequisite:[],intuition:'下面是一段独立的小练习。它把计算关系写清楚，帮助你进入官方题目；不是整份作业的答案。',
      exampleHtml:`<div class="code-walk"><h4>读懂这段最小 Python</h4><pre><code>${esc(p.code)}</code></pre><p>先找输入变量，再找中间量，最后核对注释里的输出。改一个输入，手算后再运行。</p></div><div class="misconception"><h4>最容易踩的坑</h4><p>${esc(p.trap)}</p></div><div class="transfer-map"><h4>去官方题目做什么</h4><ol>${officialWork.map(x=>`<li>${esc(x)}</li>`).join('')}</ol></div>`,
      checkpoint:toQuiz('同学提出下面的做法，哪一项会导致本课讨论的误判？',[p.trap,`先用小例子检查「${firstTerm}」的定义和计算。`,'改变条件后重新计算，并解释结论的适用范围。'],0,`需要避免的是：${p.trap}`),homework:['打开本课官方 Notebook，完成题目并检查测试结果。','用一段话解释：课堂玩具例子与真实应用之间，还差哪些条件。']}
  ];
}
module.exports={makeAdvanced,plans,labHtml};
