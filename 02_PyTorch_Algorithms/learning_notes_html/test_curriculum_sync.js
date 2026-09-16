const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const root = __dirname;
const manifest = require('./course_manifest.json');
const {officialNotebook} = require('./curriculum_current');
const {plans} = require('./advanced_course');
const index = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const git = (...args) => execFileSync('git', args, {cwd:path.resolve(root,'../..'),encoding:'utf8',maxBuffer:32*1024*1024});
const files = new Set(git('ls-tree','-r','--name-only',manifest.upstream.commit).trim().split('\n'));
const notebooks = [...files].filter(f=>/^02_PyTorch_Algorithms\/\d{2}_.*\.ipynb$/.test(f)).map(f=>path.basename(f));
assert.deepEqual(new Set([...manifest.levels, ...manifest.reserved].map(l => l.file)), new Set(notebooks));
assert.equal(manifest.levels.length,75); assert.equal(manifest.reserved.length,15);
assert.equal(Object.keys(plans).length,45);
for (const l of manifest.levels) {
  const html = fs.readFileSync(path.join(root,'notes',l.page),'utf8');
  assert(index.includes(`href="notes/${l.page}"`),l.file);
  assert.equal((html.match(/<h1[ >]/g)||[]).length,1,`${l.file}: repeated h1`);
  assert(html.includes(`https://github.com/datawhalechina/llm-algo-leetcode/blob/${manifest.upstream.commit}/02_PyTorch_Algorithms/${l.file}`));
  assert.equal(l.sourceHash,crypto.createHash('sha256').update(officialNotebook(l.file)).digest('hex'));
  assert(html.includes(l.sourceHash)); assert(!html.includes('打开本课 Notebook'));
  assert(!html.includes('katex-error')); assert(!html.includes('>undefined<'));
  if(Number(l.id)>=30){
    assert.equal((html.match(/data-lesson="/g)||[]).length,3,l.file);
    for(const marker of ['concept-glossary','worked-example','data-lab=','misconception','code-walk']) assert(html.includes(marker),`${l.file}: ${marker}`);
    const p=plans[Number(l.id)];assert(p.terms.length>=3&&p.steps.length>=3&&p.code&&p.trap);
  }
}
for(const r of manifest.reserved)assert(!index.includes(`data-id="${r.id}"`));
let scriptCount=0,localLinks=0,officialLinks=0;
for(const file of ['index.html',...fs.readdirSync(path.join(root,'notes')).filter(f=>f.endsWith('.html')).map(f=>'notes/'+f)]){
  const html=fs.readFileSync(path.join(root,file),'utf8');
  for(const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)){new vm.Script(match[1],{filename:file});scriptCount++;}
  for(const match of html.matchAll(/\b(?:src|href)="([^"<>]+)"/g)){
    const target=match[1].replaceAll('&amp;','&');
    if(target.includes('.ipynb')){
      const u=new URL(target);assert.equal(u.origin,'https://github.com');
      const parts=decodeURIComponent(u.pathname).split('/');assert.deepEqual(parts.slice(1,4),['datawhalechina','llm-algo-leetcode','blob']);
      const ref=parts[4],repoPath=parts.slice(5).join('/');
      if(ref===manifest.upstream.commit)assert(files.has(repoPath),`Missing official notebook ${repoPath}`);
      else git('cat-file','-e',ref+':'+repoPath);
      officialLinks++;
    }
    if(/^(?:[a-z]+:|#|\/\/)/i.test(target))continue;
    const relative=decodeURIComponent(target.split(/[?#]/)[0]);
    assert(fs.existsSync(path.resolve(root,path.dirname(file),relative)),`${file} -> ${target}`);localLinks++;
  }
}
for(const m of manifest.moves){
  const oldPage=m.oldFile.replace('.ipynb','.html').toLowerCase();
  const newPage=m.file.replace('.ipynb','.html').toLowerCase();
  if(oldPage!==newPage)assert(fs.readFileSync(path.join(root,'notes',oldPage),'utf8').includes(`0;url=${newPage}`));
}
const migration=index.match(/const courseMoves = [\s\S]*?(?=\n\s*const completed =)/)[0];
function migrate(seed){const store=new Map(Object.entries(seed));const localStorage={getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,v)};
 vm.runInNewContext(migration,{localStorage});const first=store.get('pytorch-levels-complete-v4');vm.runInNewContext(migration,{localStorage});assert.equal(first,store.get('pytorch-levels-complete-v4'));return {value:JSON.parse(first),store};}
assert.deepEqual(migrate({'pytorch-levels-complete-v3':'["00","25","30","60"]'}).value,['00','25']);
assert.deepEqual(migrate({'pytorch-levels-complete-v2':'["00","25","30","31"]'}).value,['00','25']);
assert.deepEqual(migrate({'pytorch-levels-complete':'["00","11","12","30"]'}).value,['00','11']);
assert.deepEqual(migrate({'pytorch-levels-complete-v4':'["30"]'}).value,['30']);
assert.deepEqual(migrate({}).value,[]);
assert.equal(migrate({'pytorch-levels-complete-v3':'["60"]'}).store.get('pytorch-levels-complete-v3'),'["60"]');
// Rebuilt advanced quizzes must not inherit the old one-question completion.
const moved=fs.readFileSync(path.join(root,'notes/60_lora_fine_tuning_project.html'),'utf8');
const stateSetup=moved.match(/const levelId = [\s\S]*?(?=\n\s*const checkpointsDone)/)[0];
const store=new Map([['pytorch-v3-level-60-checkpoints','["current-core"]']]);
vm.runInNewContext(stateSetup,{localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,v)}});
assert(!store.has('pytorch-v4-level-60-checkpoints'));
// Build source is an official git blob; guard against reintroducing local notebook reads.
const source=fs.readFileSync(path.join(root,'curriculum_current.js'),'utf8');
assert(source.includes('const data = officialNotebook(file)'));assert(!source.includes('fs.readFileSync'));
console.log(`Course checks passed: 75 courses, 45 teaching plans, 15 reserves; ${localLinks} local resources, ${officialLinks} official notebook links, ${scriptCount} scripts; migration and official-source isolation verified.`);
