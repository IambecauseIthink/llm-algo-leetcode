const assert=require('node:assert/strict');
const {models,calculate}=require('./assets/learning_lab');
const metric=(r,label)=>r.metrics.find(x=>x.label===label).value;
// Test all input corners, not just default render success.
let cases=0;
for(const [kind,model] of Object.entries(models)){
 const inputs=model.controls.reduce((all,c)=>all.flatMap(x=>[c.min,c.value,c.max].map(v=>({...x,[c.key]:v}))),[{}]);
 for(const input of inputs){const r=calculate(kind,input);assert(r.frames.length>=4);assert(Number.isFinite(r.challenge.answer));for(const x of r.rows)assert(Number.isFinite(x.value)&&x.value>=0,kind);for(const x of r.metrics)if(typeof x.value==='number')assert(Number.isFinite(x.value),kind);cases++;}
 for(const c of model.controls)assert.throws(()=>calculate(kind,{[c.key]:NaN}));
}
assert.equal(metric(calculate('context',{window:16,reserve:4}),'装得下'),'3/4');
assert.equal(calculate('lora',{rank:2,width:8}).challenge.answer,28);
assert.equal(calculate('data').challenge.answer,2);
assert.equal(calculate('data',{trim:0,dedupe:0}).challenge.answer,4);
assert.equal(calculate('gate',{budget:6,quality:98}).challenge.answer,0);
assert.equal(calculate('prefix',{hit:10,chunk:3}).challenge.answer,0);
assert.equal(calculate('prefix',{hit:3,chunk:3}).challenge.answer,3);
assert.equal(calculate('cache',{capacity:2}).challenge.answer,1);
assert.deepEqual(calculate('cache',{capacity:2}).tokens.map(x=>x.text),['A','C']);
assert.equal(calculate('schedule',{long:6,shortFirst:0}).challenge.answer,8);
assert.equal(calculate('schedule',{long:6,shortFirst:1}).challenge.answer,3);
assert.equal(metric(calculate('overlap',{start:8}),'最后完成'),12);
assert.equal(metric(calculate('overlap',{start:8}),'重叠'),0);
assert.equal(calculate('offload',{moved:8,bandwidth:8}).challenge.answer,16);
assert.equal(metric(calculate('offload',{moved:8,bandwidth:8}),'串行往返'),2);
assert.equal(metric(calculate('kv',{tokens:8,heads:2}),'MHA'),512);
assert.equal(metric(calculate('kv',{tokens:8,heads:2}),'GQA'),256);
assert.equal(metric(calculate('kv',{tokens:8,heads:2}),'简化 MLA'),192);
assert.equal(metric(calculate('pipeline',{prefill:16,decode:4}),'理想完成率上限'),4);
assert.equal(calculate('bubble',{batches:4}).challenge.answer,6);
assert.equal(calculate('latency',{ttft:100,tpot:20}).challenge.answer,180);
assert.equal(metric(calculate('amdahl',{fraction:20,speed:4}),'新耗时'),85);
assert.equal(metric(calculate('preference',{invalid:10}),'有效样本胜率'),80);
assert.equal(metric(calculate('preference',{invalid:10}),'有效覆盖'),50);
assert.equal(metric(calculate('dpo',{margin:1,beta:1}),'DPO loss'),.69);
assert(metric(calculate('dpo',{margin:3,beta:1}),'DPO loss')<metric(calculate('dpo',{margin:2,beta:1}),'DPO loss'));
assert(calculate('grpo',{third:3}).tokens.some(x=>x.text==='A2=0'));
assert.equal(calculate('quant',{bits:4}).challenge.answer,2);
assert.equal(metric(calculate('parallel',{left:4}),'理想利用率'),100);
console.log(`Lab checks passed: ${Object.keys(models).length} mechanisms, ${cases} boundary/default combinations, numerical contracts and trade-offs verified.`);
