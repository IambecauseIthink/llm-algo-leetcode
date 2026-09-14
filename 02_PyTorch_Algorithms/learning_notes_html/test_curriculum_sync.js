const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = __dirname;
const manifest = require('./course_manifest.json');
const index = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const notebooks = fs.readdirSync(path.join(root, '..')).filter(f => /^\d{2}_.*\.ipynb$/.test(f));
assert.deepEqual(new Set([...manifest.levels, ...manifest.reserved].map(l => l.file)), new Set(notebooks));
assert.equal(manifest.levels.length, 75);
assert.equal(manifest.reserved.length, 15);
assert.equal(new Set(manifest.levels.map(l => l.id)).size, 75);
for (const l of manifest.levels) {
  const html = fs.readFileSync(path.join(root, 'notes', l.page), 'utf8');
  assert(index.includes(`href="notes/${l.page}"`), l.file);
  assert.equal((html.match(/<h2>新版 Notebook 任务<\/h2>/g) || []).length, 1, l.file);
  assert(html.includes(`href="../../${l.file}"`), l.file);
  assert.equal(l.sourceHash, crypto.createHash('sha256').update(fs.readFileSync(path.join(root, '..', l.file))).digest('hex'));
  assert(html.includes(l.sourceHash));
  assert(!html.includes('<h2>参考代码与解析</h2>'));
  assert(!html.includes('>undefined<'), `${l.file}: undefined output`);
}
for (const r of manifest.reserved) assert(!index.includes(`data-id="${r.id}"`));
let scriptCount = 0;
let linkCount = 0;
for (const file of ['index.html', ...fs.readdirSync(path.join(root, 'notes')).filter(f => f.endsWith('.html')).map(f => 'notes/' + f)]) {
  const html = fs.readFileSync(path.join(root, file), 'utf8');
  for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
    new vm.Script(match[1], {filename: file}); scriptCount++;
  }
  for (const match of html.matchAll(/\b(?:src|href)="([^"<>]+)"/g)) {
    const target = match[1].replaceAll('&amp;', '&');
    if (/^(?:[a-z]+:|#|\/\/)/i.test(target)) continue;
    const relative = decodeURIComponent(target.split(/[?#]/)[0]);
    assert(fs.existsSync(path.resolve(root, path.dirname(file), relative)), `${file} -> ${target}`);
    linkCount++;
  }
}
for (const move of manifest.moves) {
  assert.equal(move.oldFile.replace(/^\d+_/, ''), move.file.replace(/^\d+_/, ''));
  if (move.oldId === move.id) continue;
  const oldPage = move.oldFile.replace('.ipynb', '.html').toLowerCase();
  const newPage = move.file.replace('.ipynb', '.html').toLowerCase();
  assert(fs.readFileSync(path.join(root, 'notes', oldPage), 'utf8').includes(`0;url=${newPage}`));
}
// Run the actual generated migration, using isolated storage rather than a user's browser.
const migration = index.match(/const courseMoves = [\s\S]*?(?=\n\s*const completed =)/)[0];
function migrate(seed) {
  const store = new Map(Object.entries(seed));
  const localStorage = {getItem: key => store.get(key) ?? null, setItem: (key, val) => store.set(key, val)};
  vm.runInNewContext(migration, {localStorage});
  const first = store.get('pytorch-levels-complete-v3');
  vm.runInNewContext(migration, {localStorage});
  assert.equal(store.get('pytorch-levels-complete-v3'), first);
  return JSON.parse(first);
}
assert.deepEqual(migrate({'pytorch-levels-complete-v2': '["00","25","30","31","41","42"]'}), ['00','25','60','66','37','46']);
assert.deepEqual(migrate({'pytorch-levels-complete': '["00","11","12","30"]'}), ['00','11']);
assert.deepEqual(migrate({'pytorch-levels-complete-v3': '["30"]','pytorch-levels-complete-v2':'["30"]'}), ['30']);
assert.deepEqual(migrate({}), []);
// Re-numbered quiz state follows the same topic; changed homework never inherits old checks.
const moved = fs.readFileSync(path.join(root, 'notes/60_lora_fine_tuning_project.html'), 'utf8');
const stateSetup = moved.match(/const levelId = [\s\S]*?(?=\n\s*const checkpointsDone)/)[0];
const store = new Map([['pytorch-v2-level-30-checkpoints', '["example"]'], ['pytorch-v2-level-30-homework', '["obsolete-task"]']]);
vm.runInNewContext(stateSetup, {localStorage: {getItem:k => store.get(k) ?? null,setItem:(k,v)=>store.set(k,v)}});
assert.equal(store.get('pytorch-v3-level-60-checkpoints'), '["example"]');
assert(![...store.keys()].some(k => k.startsWith('pytorch-v3-level-60-homework')));
console.log(`Sync checks passed: 75 courses, 15 reserves, ${linkCount} local links, ${scriptCount} scripts; topic redirects and progress migration verified.`);
