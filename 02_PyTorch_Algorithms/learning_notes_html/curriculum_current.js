// Read the official notebooks on each build; course identity follows the topic,
// not its numeric slot. Reserved notebooks are listed separately, never graded.
const fs = require('node:fs');
const {execFileSync} = require('node:child_process');
const upstream = require('./upstream_source.json');
const repoRoot = pathRoot();
function pathRoot() { return require('node:path').resolve(__dirname, '../..'); }
function officialGit(...args) { return execFileSync('git', args, {cwd:repoRoot, encoding:'utf8', maxBuffer:32*1024*1024}); }
function officialNotebook(file) { return officialGit('show', upstream.commit + ':02_PyTorch_Algorithms/' + file); }
const path = require('node:path');
const crypto = require('node:crypto');
const legacy = require('./curriculum_v2');
const notebookDir = path.join(__dirname, '..');
const topic = file => file.replace(/^\d+_/, '').replace(/\.ipynb$/, '').replace('Activation_Checkpointing_and_Activation_Offload', 'Activation_Checkpointing');
const source = cell => Array.isArray(cell.source) ? cell.source.join('') : cell.source || '';
function questionCells(notebook) {
  const stop = notebook.cells.findIndex(c => /STOP\s*HERE|^##\s*参考代码与解析/m.test(source(c)));
  if (stop < 0) throw new Error('Notebook has no answer boundary');
  return notebook.cells.slice(0, stop);
}
const reserved = [];
function build(early) {
  const previous = [...early, ...legacy];
  const oldByTopic = new Map(previous.map(l => [topic(l.file), l]));
  const filenames = officialGit('ls-tree', '--name-only', upstream.commit + ':02_PyTorch_Algorithms').trim().split('\n').filter(f => /^\d{2}_.*\.ipynb$/.test(f)).sort();
  const current = filenames.flatMap(file => {
    const data = officialNotebook(file);
    const notebook = JSON.parse(data);
    const id = file.slice(0, 2);
    const cells = questionCells(notebook);
    const first = source(cells.find(c => c.cell_type === 'markdown'));
    const title = (first.match(/^#\s*\d+[.、]?\s*(.+)/m) || [null, topic(file).replaceAll('_', ' ')])[1];
    if (/_Reserved_/.test(file)) { reserved.push({id, file, title}); return []; }
    const old = oldByTopic.get(topic(file));
    const markdown = cells.filter(c => c.cell_type === 'markdown').map(source).join('\n\n');
    const introText = (markdown.match(/##\s*本节导读\s*\n([\s\S]*?)(?=\n#{1,3}\s|$)/) || [, ''])[1];
    const intro = introText.split(/\n\s*\n/).find(s => s.trim() && !/^[-#*]/.test(s.trim())) || title;
    const tags = [...((first.match(/\*\*标签：\*\*\s*([^\n|]*(?:\|[^\n]*)?)/) || [,''])[1]).matchAll(/`([^`]+)`/g)].map(m => m[1]);
    const category = /推理|Cache|解码/.test(tags.join(' ')) ? 'inference' : /训练|对齐|LoRA|SFT/.test(tags.join(' ')) ? 'training' : old?.category || 'architecture';
    const todos = [...new Set(cells.filter(c => c.cell_type === 'code').flatMap(c => source(c).split('\n').filter(s => /^\s*(?:#\s*|["']{3})?TODO\s*\d+[^\n]*[:：]/.test(s)).map(s => s.trim().replace(/^(?:#\s*|["']{3})/, '').replace(/["']{3}$/, ''))))];
    return [{...old, id, title, file, oldId: old?.id, category, tags: tags.length ? tags : old?.tags || [],
      difficulty: (first.match(/难度：\*\*\s*([^|\n]+)/) || [, 'Medium'])[1].trim(),
      summary: intro.replace(/[*`]/g, '').trim(),
      concepts: old?.concepts || [intro], handsOn: todos,
      formula: old?.formula || '读懂任务 → 完成实现 → 核对测试与适用范围',
      sourceHash: crypto.createHash('sha256').update(data).digest('hex'), cells,
      steps: cells.filter(c => c.cell_type === 'markdown' && /### Step/.test(source(c))).map(source),
    }];
  });
  const moves = previous.map(old => {
    const next = current.find(l => topic(l.file) === topic(old.file));
    if (!next) throw new Error(`Unmapped local course: ${old.file}`);
    return {oldId: old.id, id: next.id, oldFile: old.file, file: next.file};
  });
  return {levels: current, reserved, moves};
}
module.exports = {build, source, officialNotebook, upstream};
