/* Pure classroom models + a DOM adapter. No model, network request, or GPU execution. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else { root.LearningLab = api; if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', api.mount); else api.mount(); }
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';
  const control = (key,label,min,max,value,step=1,unit='') => ({key,label,min,max,value,step,unit});
  const f = (n,d=2) => Number(n.toFixed(d));
  const row = (label,value,unit='') => ({label,value:typeof value==='number'?f(value):value,unit});
  const frame = (title,text) => ({title,text});
  const challenge = (question,answer,unit,explain) => ({question,answer:f(answer),unit,explain});
  const models = {
    context: {
      title:'给回答留一块空位', purpose:'拖动窗口和回答预算，看看哪些样本能装进去。',
      controls:[control('window','上下文窗口',8,32,16,1,'token'),control('reserve','回答预留',1,8,4,1,'token')],
      compute:({window:w,reserve:r})=>{const budget=w-r, lengths=[4,8,12,15], count=lengths.filter(n=>n<=budget).length;return {
        metrics:[row('prompt 可用',budget,'token'),row('装得下',count+'/4'),row('fit rate',count/4*100,'%')],
        rows:[row('窗口',w,'token'),row('回答预留',r,'token'),row('prompt 预算',budget,'token')],
        tokens:lengths.map(n=>({text:n+' token',kind:n<=budget?'good':'bad'})),
        frames:[frame('先占住回答的位置',`窗口有 ${w} 格，回答预留 ${r} 格。`),frame('算可用空间',`prompt 最多 ${w}−${r}=${budget} token。`),frame('逐个检查样本',`4、8、12、15 中有 ${count} 个不超过 ${budget}。`),frame('别把装得下当学得好','这个实验只算预算；质量和实际速度仍要另外验证。')],
        challenge:challenge('prompt 最多能放多少 token？',budget,'token',`${w}−${r}=${budget}。`)};}
    },
    lora:{
      title:'把一个大矩阵拆成两块补丁',purpose:'改变 rank，观察旁路的两段形状和参数规模。',
      controls:[control('rank','旁路 rank',1,8,2),control('width','输入维度',4,32,8)],
      compute:({rank:r,width:d})=>{const out=6, a=r*d,b=out*r,full=d*out;return {
        metrics:[row('完整矩阵',full,'个参数'),row('A + B',a+b,'个参数'),row('可训练参数节省',(1-(a+b)/full)*100,'%')],
        rows:[row('完整 W',full,'参数'),row('补丁 A',a,'参数'),row('补丁 B',b,'参数')],
        tokens:[{text:`输入 ${d}`,kind:'base'},{text:`A: ${d}→${r}`,kind:'good'},{text:`B: ${r}→6`,kind:'good'},{text:'加回 W 路径',kind:'base'}],
        frames:[frame('原来的一步',`W 将 ${d} 维映射到 6 维，共 ${d}×6=${full} 个数。`),frame('经过窄通道',`A 用 ${d}×${r}=${a} 个数，把输入映射到 rank=${r}。`),frame('回到输出维度',`B 用 ${r}×6=${b} 个数，旁路合计 ${a+b}。`),frame('观察边界',a+b<full?'旁路参数更少；它仍与冻结的 W 一起参与前向。':'rank 太大时，这个小矩阵的旁路不再省参数。')],
        challenge:challenge('A 与 B 一共有多少参数？',a+b,'个',`${d}×${r}+${r}×6=${a+b}。`)};}
    },
    data:{title:'把原始记录送过三道检查',purpose:'切换空白清理和去重，观察保留样本怎样变化。',
      controls:[control('trim','清理首尾空格（0 关 / 1 开）',0,1,1),control('dedupe','精确去重（0 关 / 1 开）',0,1,1)],
      compute:({trim,dedupe})=>{const raw=[['问A','答A'],['问B','   '],['问A','答A'],[' 问C ','答C']];let seen=new Set();const verdict=raw.map(([p,r])=>{if(trim){p=p.trim();r=r.trim();}const key=JSON.stringify([p,r]);const ok=Boolean(p&&r)&&!(dedupe&&seen.has(key));if(ok)seen.add(key);return ok;});const kept=verdict.filter(Boolean).length;return {
        metrics:[row('原始',4,'条'),row('保留',kept,'条'),row('剔除',4-kept,'条')],rows:[row('输入记录',4,'条'),row('有效输出',kept,'条')],
        tokens:raw.map((x,i)=>({text:`${String.fromCharCode(65+i)} ${verdict[i]?'保留':'剔除'}`,kind:verdict[i]?'good':'bad'})),
        frames:[frame('看输入','A 正常、B 只有空格、C 与 A 重复、D 正常但带空白。'),frame('空白影响判断',trim?'strip 后 B 变成空字符串，不能作为有效回答。':'没有 strip，空格字符串仍为真，会漏掉空回答。'),frame('检查重复',dedupe?'C 与 A 的字段组合相同，只保留第一次。':'关闭去重后 C 仍留下；样本数多并不代表信息更多。'),frame('进入下一步',`当前保留 ${kept} 条；还需要 tokenizer、模板、mask 和跨集合泄漏检查。`)],
        challenge:challenge('当前规则保留几条？',kept,'条','按当前开关逐条判断，注意空格字符串与空字符串不同。')};}
    },
    gate:{title:'当一次有原则的方案评审员',purpose:'改变显存和质量门槛。先过滤，再在合格者中比较速度。',
      controls:[control('budget','显存上限',6,16,10,1,'GiB'),control('quality','质量下限',80,98,90,1,'%')],
      compute:({budget,quality})=>{const runs=[{name:'A',mem:12,q:95,tps:120},{name:'B',mem:8,q:93,tps:100},{name:'C',mem:6,q:85,tps:150}];const valid=runs.filter(x=>x.mem<=budget&&x.q>=quality);const best=[...valid].sort((a,b)=>b.tps-a.tps)[0];return {
        metrics:[row('可行候选',valid.length,'个'),{label:'推荐',value:best?best.name:'暂无',unit:''}],rows:runs.map(x=>row(`${x.name} · ${x.mem}GiB / 质量${x.q}%`,x.tps,'tokens/s')),
        tokens:runs.map(x=>({text:`${x.name} ${valid.includes(x)?'通过':x.mem>budget?'超显存':'质量不足'}`,kind:valid.includes(x)?'good':'bad'})),
        frames:[frame('记录候选','A=12GiB/95%/120；B=8GiB/93%/100；C=6GiB/85%/150。数值是课堂假设。'),frame('先看硬门槛',`显存≤${budget}GiB 且质量≥${quality}%，需要同时满足。`),frame('比较可行集合',valid.length?`可行的是 ${valid.map(x=>x.name).join('、')}，现在才比较吞吐。`:'没有可行候选时，应返回暂无，而不是偷偷降低门槛。'),frame('说明结论',best?`在当前门槛和这张假设表里选 ${best.name}。条件一变，结论可能改变。`:'下一步是调整计划或获得新候选；未知数据不能当作通过。')],
        challenge:challenge('有几个候选同时满足两项门槛？',valid.length,'个','逐个检查显存与质量，任一项失败都排除。')};}
    },
    prefix:{title:'只计算还没读过的后半段',purpose:'移动前缀命中长度和 chunk 大小，观察剩余工作。',
      controls:[control('hit','前缀命中',0,10,4,1,'token'),control('chunk','每块最多',1,6,3,1,'token')],
      compute:({hit,chunk})=>{const n=10-hit,chunks=[];for(let i=0;i<n;i+=chunk)chunks.push(Math.min(chunk,n-i));return {
        metrics:[row('复用',hit,'token'),row('还需计算',n,'token'),row('新 prefill',chunks.length,'块')],rows:[row('完整 prompt',10,'token'),row('已复用',hit,'token'),row('新计算',n,'token')],
        tokens:Array.from({length:10},(_,i)=>({text:i<hit?'复用':`块${Math.floor((i-hit)/chunk)+1}`,kind:i<hit?'good':'base'})),
        frames:[frame('输入十个 token','假设前缀已通过完整一致性检查，可以安全复用。'),frame('去掉已算部分',`10−${hit}=${n} 个 token 仍需要 prefill。`),frame('把剩余切块',chunks.length?`按 ${chunk} 切成 ${chunks.join(' + ')}，总量仍是 ${n}。`:'全部命中，不需要新增 prefill 块。'),frame('理解优化对象','缓存减少重复计算；分块改变计算组织，不会凭空消除未命中 token。')],
        challenge:challenge('还需要几个 prefill 块？',chunks.length,'块',`ceil((${10}−${hit})/${chunk})=${chunks.length}。`)};}
    },
    speculate:{title:'草稿接受到哪里就在哪里停',purpose:'调整首次拒绝位置与验证成本，看看有效推进量如何影响收益。',
      controls:[control('accepted','连续接受的草稿',0,4,2,1,'token'),control('cost','一轮 draft + verify 成本',10,50,30,1,'时间单位')],
      compute:({accepted:a,cost})=>{const progress=a+1, per=cost/progress;return {
        metrics:[row('草稿接受',a,'token'),row('含目标补发的推进',progress,'token'),row('平均成本',per,'/token')],rows:[row('普通解码 / token',10,'时间单位'),row('本轮平均 / token',per,'时间单位')],tokens:Array.from({length:4},(_,i)=>({text:i<a?'接受':i===a?'拒绝':'丢弃',kind:i<a?'good':'bad'})).concat([{text:'目标补发 1',kind:'base'}]),
        frames:[frame('草稿提出四个候选','本实验固定每轮提议 4 个候选，暂不模拟真实概率分布。'),frame('接受连续前缀',`接受前 ${a} 个，后面${a===4?'没有拒绝':'从首错处丢弃'}。`),frame('目标模型补一步',`按本实验约定，拒绝时补发/全接受时奖励 1 个，推进 ${a}+1=${progress}。`),frame('把验证开销算进去',`${cost}/${progress}=${f(per)} 单位/token，相对 baseline 10 ${per<10?'更少':per>10?'更多':'相同'}。真实系统需按实际协议统计。`)],
        challenge:challenge('按本实验约定，本轮总推进多少 token？',progress,'token',`${a} 个被接受草稿 + 1 个目标补发。`)};}
    },
    schedule:{title:'重新排一次队，等待会怎样变？',purpose:'三个任务同时到达，比较 FIFO 与短任务优先。',
      controls:[control('long','长任务时长',3,12,6,1,'时间单位'),control('shortFirst','短任务优先（0 关 / 1 开）',0,1,0)],
      compute:({long,shortFirst})=>{let jobs=[{name:'A',t:long},{name:'B',t:2},{name:'C',t:1}];if(shortFirst)jobs.sort((a,b)=>a.t-b.t);let clock=0;const waits=jobs.map(j=>{const w=clock;clock+=j.t;return w;});return {
        metrics:[row('平均等待',waits.reduce((a,b)=>a+b,0)/3,'单位'),row('最后完成',clock,'单位')],rows:jobs.map((j,i)=>row(`${j.name} 等待`,waits[i],'单位')),tokens:jobs.map(j=>({text:`${j.name} · ${j.t}`,kind:'base'})),
        frames:[frame('请求同时到达',`A 服务 ${long}，B 服务 2，C 服务 1。单服务台、不可抢占。`),frame('按规则取队首',`当前顺序是 ${jobs.map(j=>j.name).join(' → ')}。`),frame('累加前人的执行',`等待分别是 ${waits.join('、')}，不包含请求自身执行时间。`),frame('比较公平性','短任务优先可改善这个例子的平均等待；真实线上还需考虑长任务饥饿、长度估计和优先级。')],
        challenge:challenge('当前队列最后一个请求等待多久？',waits[2],'单位','把前两个任务的服务时长相加。')};}
    },
    cache:{title:'亲手看一次 LRU 淘汰',purpose:'按下一步播放 A→B→A→C 的访问，改变书架容量。',
      controls:[control('capacity','缓存容量',1,3,2,1,'条')],
      compute:({capacity})=>{let list=[],hits=0;const frames=[];for(const key of ['A','B','A','C']){const hit=list.includes(key);if(hit)hits++;list=list.filter(x=>x!==key);list.push(key);const evict=list.length>capacity?list.shift():null;frames.push({...frame(`访问 ${key} · ${hit?'命中':'未命中'}`,`从最旧到最新：${list.join(' → ')}。${evict?'容量不足，淘汰 '+evict+'。':'无需淘汰。'}`),tokens:list.map(x=>({text:x,kind:x===key?'good':'base'}))});}return {
        metrics:[row('访问',4,'次'),row('命中',hits,'次'),row('最终缓存',list.length,'条')],rows:[row('容量',capacity,'条'),row('驻留',list.length,'条')],tokens:list.map(x=>({text:x,kind:'good'})),frames,
        challenge:challenge('这四次访问中命中几次？',hits,'次','只有访问时仍在缓存中的条目才算命中。')};}
    },
    quant:{title:'用更粗的尺子记录四个权重',purpose:'改变位宽，观察整数刻度、恢复误差和存储量。',
      controls:[control('bits','对称有符号位宽',2,8,4,1,'bit')],
      compute:({bits})=>{const values=[-.9,-.2,.4,1],qmax=2**(bits-1)-1,scale=1/qmax,restored=values.map(x=>Math.round(x/scale)*scale),mse=values.reduce((s,x,i)=>s+(x-restored[i])**2,0)/4;return {
        metrics:[row('scale',scale),{label:'均方误差',value:mse.toFixed(6),unit:''},row('4 个数的理想存储',bits*4/8,'字节')],rows:[row('FP32 纯数据',16,'字节'),row('低比特纯数据',bits*4/8,'字节')],tokens:values.map((x,i)=>({text:`${x} → ${f(restored[i],3)}`,kind:'base'})),
        frames:[frame('固定原值','原始权重为 −0.9、−0.2、0.4、1.0；本实验使用对称整数舍入。'),frame('建立刻度',`${bits} 位使用 ±${qmax} 的对称范围，scale=1/${qmax}≈${f(scale,4)}。`),frame('整数再恢复','先 round(x/scale)，再乘 scale；结果落在有限的刻度上。'),frame('比较代价','位宽下降减少纯数据字节，但误差通常变大。未计 scale、对齐与元数据，也不模拟 FP8、GPTQ 或 AWQ 内核。')],
        challenge:challenge('只算 4 个数的理想数据区，需要多少字节？',bits*4/8,'字节',`4×${bits}/8=${bits*4/8}。`)};}
    },
    kv:{title:'让 KV 缓存的账本动起来',purpose:'固定一层，改变 token 数和 KV heads，对照三种表示。',
      controls:[control('tokens','序列长度',4,32,8,1,'token'),control('heads','GQA 的 KV heads',1,4,2)],
      compute:({tokens:t,heads:h})=>{const mha=2*t*4*4*2,gqa=2*t*h*4*2,mla=t*(8+4)*2;return {
        metrics:[row('MHA',mha,'字节'),row('GQA',gqa,'字节'),row('简化 MLA',mla,'字节')],rows:[row('MHA · 4 KV heads',mha,'字节'),row(`GQA · ${h} KV heads`,gqa,'字节'),row('MLA · latent8 + pos4',mla,'字节')],
        tokens:[{text:'batch 1',kind:'base'},{text:'层数 1',kind:'base'},{text:'head dim 4',kind:'base'},{text:'每元素 2 字节',kind:'base'}],
        frames:[frame('先固定口径','batch=1、层数=1、query heads=4、head dim=4、FP16。'),frame('数普通缓存',`K/V 两份 × ${t} token × ${h} KV heads × 4 维 × 2 字节 = ${gqa}。`),frame('换一种表示',`课堂 MLA：${t}×(latent 8 + 位置 4)×2 = ${mla} 字节；结构必须由模型本身支持。`),frame('别忘了乘法边界','实际多层、多请求要再乘相应因子；这里是理论数据区，未计分配器、分页和 kernel 开销。')],
        challenge:challenge('当前 GQA 理论数据区是多少字节？',gqa,'字节',`2×${t}×${h}×4×2=${gqa}。`)};}
    },
    offload:{title:'腾空间之前，先算一趟往返',purpose:'改变搬运量和有效带宽，与 1.5 秒的假设重算比较。',
      controls:[control('moved','卸载的激活',1,12,8,1,'GiB'),control('bandwidth','有效带宽',2,32,8,2,'GiB/s')],
      compute:({moved:m,bandwidth:b})=>{const trip=2*m/b;return {metrics:[row('GPU 腾出',m,'GiB'),row('串行往返',trip,'秒'),row('假设重算',1.5,'秒')],rows:[row('搬运往返',trip,'秒'),row('重算',1.5,'秒')],tokens:[{text:'GPU 激活',kind:'base'},{text:`→ ${m}GiB →`,kind:'good'},{text:'CPU 暂存',kind:'base'},{text:'← 反向前取回',kind:'good'}],
        frames:[frame('前向之后',`${m}GiB 激活暂时不在 GPU，容量下降但数据仍需保存。`),frame('搬到 CPU',`${m}/${b}=${f(m/b)} 秒，使用有效带宽而非标称值。`),frame('反向前取回',`再花 ${f(m/b)} 秒，串行合计 ${f(trip)}。`),frame('比较下一步',`假设重算 1.5 秒，当前${trip<1.5?'搬运更短':trip>1.5?'重算更短':'二者相同'}。重叠、缓冲与调度会改变实际结果。`)],
        challenge:challenge('往返需传输多少 GiB？',2*m,'GiB',`去 ${m} + 回 ${m} = ${2*m}。`)};}
    },
    memory:{title:'搭一张不会漏项的显存账本',purpose:'把同一时刻的状态加起来，观察何时跨过 12 GiB 预算。',
      controls:[control('activation','激活占用',1,12,6,1,'GiB'),control('cache','缓存/其他占用',0,6,2,1,'GiB')],
      compute:({activation:a,cache:c})=>{const total=4+a+c;return {metrics:[row('合计',total,'GiB'),row('预算',12,'GiB'),row('余量',12-total,'GiB')],rows:[row('权重',4,'GiB'),row('激活',a,'GiB'),row('缓存/其他',c,'GiB')],tokens:[{text:'权重固定 4',kind:'base'},{text:`激活 ${a}`,kind:'base'},{text:`其他 ${c}`,kind:'base'},{text:total<=12?'预算内':'超过预算',kind:total<=12?'good':'bad'}],frames:[frame('固定权重','这个简化例子先固定 4 GiB 权重。'),frame('加入变化项',`激活 ${a}，缓存/其他 ${c}，假设这些对象同时存活。`),frame('核对预算',`4+${a}+${c}=${total}，相对 12GiB ${total<=12?'剩余 '+(12-total):'超出 '+(total-12)}。`),frame('连接真实测量','真实峰值还需按时间轴观察梯度、optimizer、工作区等；不能把这个简化账本当完整训练模型。')],challenge:challenge('当前总占用多少 GiB？',total,'GiB',`4+${a}+${c}=${total}。`)};}
    },
    overlap:{title:'在两条时间轨道上找交集',purpose:'计算固定在 [0,6]，通信长 4；移动通信起点。',
      controls:[control('start','通信开始时刻',0,8,4,1,'时间单位')],
      compute:({start:s})=>{const overlap=Math.max(0,Math.min(6,s+4)-Math.max(0,s)),end=Math.max(6,s+4);return {metrics:[row('重叠',overlap,'单位'),row('最后完成',end,'单位')],rows:[row('计算结束',6,'单位'),row('通信开始',s,'单位'),row('通信结束',s+4,'单位')],tokens:Array.from({length:12},(_,i)=>({text:String(i),kind:i<6&&i>=s&&i<s+4?'good':i<6||i>=s&&i<s+4?'base':'idle'})),frames:[frame('放下计算区间','计算固定在 [0,6]。'),frame('放下通信区间',`通信位于 [${s},${s+4}]，时长 4。`),frame('数重叠区间',`max(0,min(6,${s+4})−max(0,${s}))=${overlap}。绿色格是重叠。`),frame('看端到端','最终完成时刻还包含中间空隙，不能永远用计算+通信−重叠代替 makespan。')],challenge:challenge('两个区间重叠多少时间单位？',overlap,'单位','只计算两条轨道共同覆盖的区间。')};}
    },
    parallel:{title:'把八份工作分到两个窗口',purpose:'改变任务分配，观察最忙窗口怎样决定完成时间。',
      controls:[control('left','分给窗口 A',1,7,6,1,'份')],
      compute:({left:a})=>{const b=8-a,t=Math.max(a,b),eff=8/(2*t);return {metrics:[row('完成时间',t,'单位'),row('理想利用率',eff*100,'%')],rows:[row('窗口 A',a,'份'),row('窗口 B',b,'份')],tokens:Array.from({length:8},(_,i)=>({text:i<a?'A':'B',kind:i<a?'base':'good'})),frames:[frame('固定总工作','总计 8 份，每份耗时 1，两个窗口同速且无通信。'),frame('分派工作',`A 做 ${a}，B 做 ${b}。`),frame('等待最忙的人',`整批完成时间 max(${a},${b})=${t}，另一个窗口提前完成后空闲。`),frame('加回现实成本',`理想利用率=8/(2×${t})=${f(eff*100)}%。通信与同步还会增加成本。`)],challenge:challenge('整批完成需要多少时间单位？',t,'单位',`max(${a},${b})=${t}。`)};}
    },
    pipeline:{title:'两段生产线，谁限制了完成率？',purpose:'只改变一个池的处理能力，观察瓶颈何时转移。',
      controls:[control('prefill','Prefill 池能力',1,16,8,1,'请求/s'),control('decode','Decode 池能力',1,16,4,1,'请求/s')],
      compute:({prefill:p,decode:d})=>({metrics:[row('理想完成率上限',Math.min(p,d),'请求/s'),row('两池能力差',Math.abs(p-d),'请求/s')],rows:[row('Prefill',p,'请求/s'),row('Decode',d,'请求/s')],tokens:[{text:'请求到达',kind:'base'},{text:'Prefill '+p,kind:p<=d?'bad':'good'},{text:'状态交接',kind:'idle'},{text:'Decode '+d,kind:d<=p?'bad':'good'}],frames:[frame('每个请求经过两段','先处理输入，再生成输出；本实验固定每请求工作量。'),frame('比较能力',`Prefill 每秒 ${p}，Decode 每秒 ${d}。`),frame('寻找瓶颈',`稳定完成率的理想上界是 min(${p},${d})=${Math.min(p,d)}。`),frame('加回真实成本','实际还有排队、KV 传输和变长请求，本实验不预测 TTFT 或真实容量。')],challenge:challenge('当前理想完成率上限是多少？',Math.min(p,d),'请求/s','每个请求必须经过两段，较慢的池限制整体。')})
    },
    bubble:{title:'给三段流水线持续喂任务',purpose:'三阶段等时长前向模型，增加 microbatch，观察空泡占比。',
      controls:[control('batches','Microbatch 数量',1,12,4)],
      compute:({batches:m})=>{const ticks=m+2,util=m/ticks;return {metrics:[row('完成时间',ticks,'单位'),row('利用率',util*100,'%'),row('空闲比例',(1-util)*100,'%')],rows:[row('有效槽位',3*m,'个'),row('空闲槽位',6,'个')],tokens:Array.from({length:ticks},(_,i)=>({text:i<2?'填充':`完成 ${i-1}`,kind:i<2?'idle':'good'})),frames:[frame('第一批进入','第一个 microbatch 依次经过三个阶段，需要 3 个时间单位。'),frame('后续接力',`剩余 ${m-1} 批，每单位完成一批。`),frame('算总时间',`3+${m-1}=${ticks}；总槽位 3×${ticks}，有效槽位 3×${m}。`),frame('看公式边界','这是仅前向、等时长、不计通信的流水线。训练 1F1B、阶段不均衡和调度开销需要另算。')],challenge:challenge('总共需要多少时间单位？',ticks,'单位',`${m}+3−1=${ticks}。`)};}
    },
    preference:{title:'胜率的分母由谁决定？',purpose:'8 胜、2 负固定，增加无效记录，区分胜率与覆盖率。',
      controls:[control('invalid','无效评测记录',0,10,2,1,'条')],
      compute:({invalid:i})=>({metrics:[row('有效样本胜率',80,'%'),row('有效覆盖',10/(10+i)*100,'%'),row('若误除以全部',8/(10+i)*100,'%')],rows:[row('胜',8,'条'),row('负',2,'条'),row('无效',i,'条')],tokens:[{text:'8 胜',kind:'good'},{text:'2 负',kind:'bad'},{text:`${i} 无效`,kind:'idle'}],frames:[frame('固定有效结果','有效结果始终为 8 胜、2 负，约定无效不计入胜率。'),frame('计算胜率','8/(8+2)=80%，增加无效记录不会改变这项有效样本胜率。'),frame('报告覆盖',`有效覆盖=10/${10+i}=${f(10/(10+i)*100)}%，它会下降。`),frame('避免选择性报告','如果只留下容易评的样本，胜率会误导；必须同时记录失效原因与场景覆盖。')],challenge:challenge('有效结果一共有多少条？',10,'条','8 胜+2 负，无效另报。')})
    },
    latency:{title:'看五个 token 怎样陆续出现',purpose:'移动首 token 等待与后续间隔，分清 TTFT、TPOT、E2E。',
      controls:[control('ttft','首 token 等待',20,200,100,10,'ms'),control('tpot','后续 token 间隔',5,50,20,5,'ms')],
      compute:({ttft,tpot})=>({metrics:[row('TTFT',ttft,'ms'),row('TPOT',tpot,'ms'),row('5 token E2E',ttft+4*tpot,'ms')],rows:[row('等待首 token',ttft,'ms'),row('后续 4 段',4*tpot,'ms')],tokens:Array.from({length:5},(_,i)=>({text:`T${i+1} · ${ttft+i*tpot}ms`,kind:'base'})),frames:[frame('开始请求','本实验从请求开始计时，TTFT 已包含到首 token 的全部等待。'),frame('首 token 出现',`在 ${ttft}ms 时 T1 出现，不再给它额外加一次 TPOT。`),frame('生成后四个',`T2 至 T5 之间增加 4×${tpot}=${4*tpot}ms。`),frame('完整结束',`E2E=${ttft}+${4*tpot}=${ttft+4*tpot}ms。真实间隔可能变化，这里固定为常数。`)],challenge:challenge('T5 在多少毫秒出现？',ttft+4*tpot,'ms',`${ttft}+4×${tpot}=${ttft+4*tpot}。`)})
    },
    amdahl:{title:'局部加速能带来多少总收益？',purpose:'改变瓶颈占比和局部加速倍数，其余工作保持不变。',
      controls:[control('fraction','可优化部分占比',10,90,20,10,'%'),control('speed','局部加速',1,10,4,1,'倍')],
      compute:({fraction:p,speed:s})=>{const total=100-p+p/s;return {metrics:[row('原耗时',100,'秒'),row('新耗时',total,'秒'),row('整体加速',100/total,'倍')],rows:[row('未优化',100-p,'秒'),row('优化后部分',p/s,'秒')],tokens:[{text:`${100-p}s 不变`,kind:'idle'},{text:`${p}s → ${f(p/s)}s`,kind:'good'}],frames:[frame('原任务 100 秒',`其中 ${p} 秒可优化，${100-p} 秒不变。`),frame('局部加速',`${p}/${s}=${f(p/s)} 秒，不是整个任务除以 ${s}。`),frame('重新加总',`${100-p}+${f(p/s)}=${f(total)} 秒。`),frame('判断投资方向',`总加速只有 ${f(100/total)} 倍；越不占主导的部分，对整体的上限越小。`)],challenge:challenge('未优化部分仍耗时多少秒？',100-p,'秒',`100−${p}=${100-p}，与局部加速倍数无关。`)};}
    },
    dpo:{title:'把两种相对偏好放到同一把尺上',purpose:'改变 policy margin，reference margin 固定为 1，观察 DPO 玩具损失。',
      controls:[control('margin','Policy margin',-3,4,2,0.5),control('beta','beta',0.5,2,1,0.5)],
      compute:({margin:m,beta:b})=>{const z=b*(m-1),loss=Math.log1p(Math.exp(-z));return {metrics:[row('调整后 margin',m-1),row('乘 beta 后',z),row('DPO loss',loss)],rows:[row('loss',loss)],tokens:[{text:`policy ${m}`,kind:'base'},{text:'reference 1',kind:'idle'},{text:`差值 ${f(m-1)}`,kind:m>=1?'good':'bad'}],frames:[frame('比较两个回答','这里的 margin 已经是 chosen 与 rejected 的 log probability 差。'),frame('扣掉参照',`policy ${m} − reference 1 = ${f(m-1)}。`),frame('缩放与损失',`z=${b}×${f(m-1)}=${f(z)}，loss=log(1+exp(−z))≈${f(loss,3)}。`),frame('解释信号','这对样本的调整后偏好更强时损失下降；它不直接证明泛化、格式或安全表现。')],challenge:challenge('扣除 reference 后的 margin 是多少？',m-1,'',`${m}−1=${f(m-1)}。`)};}
    },
    grpo:{title:'同一组里，谁比平均更好？',purpose:'前两个奖励固定 1、2，改变第三个，观察组内相对优势。',
      controls:[control('third','第三个奖励',0,5,3,0.5)],
      compute:({third})=>{const rewards=[1,2,third],mean=(3+third)/3,std=Math.sqrt(rewards.reduce((s,r)=>s+(r-mean)**2,0)/3),adv=rewards.map(r=>(r-mean)/(std+1e-8));return {metrics:[row('组均值',mean),row('总体标准差',std)],rows:rewards.map((r,i)=>row(`奖励 ${i+1}`,r)),tokens:adv.map((a,i)=>({text:`A${i+1}=${f(a,3)}`,kind:a>0?'good':a<0?'bad':'idle'})),frames:[frame('确定比较组',`同组奖励为 1、2、${third}，不要和其他 prompt 的候选混算。`),frame('找平均线',`均值=(1+2+${third})/3=${f(mean,3)}。`),frame('计算相对优势',`减均值再除以标准差：${adv.map(a=>f(a,3)).join('、')}。本实验采用总体标准差。`),frame('理解正负号','正值比组平均好，负值比平均差；绝对奖励高低与组内相对优势不是同一指标。')],challenge:challenge('三个奖励之和是多少？',3+third,'',`1+2+${third}=${3+third}。`)};}
    }
  };
  // Labels can be strings (e.g. "3/4"); only numerical chart values are normalized.
  function calculate(kind, values={}) {
    const model=models[kind]; if(!model) throw new Error('Unknown model '+kind);
    const input={};
    for(const c of model.controls){const val=values[c.key]===undefined?c.value:Number(values[c.key]);if(!Number.isFinite(val)||val<c.min||val>c.max) throw new Error('Invalid '+c.key);input[c.key]=val;}
    return model.compute(input);
  }
  function mount() {
    if(typeof document==='undefined')return;
    const esc=s=>String(s).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    for (const el of document.querySelectorAll('[data-lab]')) {
      if(el.dataset.mounted)continue;el.dataset.mounted='true';
      const kind=el.dataset.lab,model=models[kind];if(!model)continue;
      let values={},step=0,timer=null,result;
      const stop=()=>{if(timer)clearInterval(timer);timer=null;el.querySelector('[data-play]').textContent='播放推演';};
      function paint() {
        result=calculate(kind,values);step=Math.min(step,result.frames.length-1);
        const max=Math.max(1,...result.rows.map(r=>Math.abs(r.value)));
        el.querySelector('[data-metrics]').innerHTML=result.metrics.map(m=>`<div><small>${esc(m.label)}</small><strong>${esc(m.value)} <span>${esc(m.unit)}</span></strong></div>`).join('');
        el.querySelector('[data-bars]').innerHTML=result.rows.map(r=>`<div class="lab-bar"><div><span>${esc(r.label)}</span><b>${esc(r.value)} ${esc(r.unit)}</b></div><div class="lab-track"><i style="width:${Math.max(0,r.value)/max*100}%"></i></div></div>`).join('');
        el.querySelector('[data-tokens]').innerHTML=(result.frames[step].tokens||result.tokens||[]).map(t=>`<span class="lab-token ${esc(t.kind)}">${esc(t.text)}</span>`).join('');
        el.querySelector('[data-step]').textContent=`${step+1} / ${result.frames.length}`;
        el.querySelector('[data-frame-title]').textContent=result.frames[step].title;
        el.querySelector('[data-frame-text]').textContent=result.frames[step].text;
        el.querySelector('[data-frame]').style.setProperty('--step',step);
        el.querySelector('[data-phase-progress]').style.width=(step/(result.frames.length-1)*100)+'%';
        el.querySelector('[data-phase-labels]').innerHTML=result.frames.map((item,i)=>`<span class="${i===step?'active':''}">${i+1}. ${esc(item.title)}</span>`).join('');
        el.querySelector('[data-next]').disabled=step===result.frames.length-1;
        el.querySelector('[data-prev]').disabled=step===0;
        el.querySelector('[data-challenge-label]').textContent=result.challenge.question+'（'+(result.challenge.unit||'数值')+'）';
      }
      el.querySelectorAll('input[type=range]').forEach(input=>{
        input.addEventListener('input',()=>{values[input.name]=Number(input.value);el.querySelector(`[data-value="${input.name}"]`).textContent=input.value;step=0;stop();delete el.querySelector('[data-answer-feedback]').dataset.correct;el.querySelector('[data-answer-feedback]').textContent='参数变了，重新算一次试试。';el.querySelector('[data-answer]').value='';paint();});
      });
      el.querySelector('[data-next]').addEventListener('click',()=>{stop();step++;paint();});
      el.querySelector('[data-prev]').addEventListener('click',()=>{stop();step--;paint();});
      el.querySelector('[data-play]').addEventListener('click',()=>{if(timer){stop();return;}if(step>=result.frames.length-1)step=0;paint();el.querySelector('[data-play]').textContent='暂停推演';timer=setInterval(()=>{step++;paint();if(step===result.frames.length-1)stop();},3200);});
      el.querySelector('[data-reset]').addEventListener('click',()=>{stop();values={};step=0;for(const c of model.controls){el.querySelector(`input[name="${c.key}"]`).value=c.value;el.querySelector(`[data-value="${c.key}"]`).textContent=c.value;}el.querySelector('[data-answer]').value='';el.querySelector('[data-answer-feedback]').textContent='';paint();});
      el.querySelector('[data-check-answer]').addEventListener('click',()=>{const value=el.querySelector('[data-answer]').value,feedback=el.querySelector('[data-answer-feedback]');if(value.trim()===''){feedback.textContent='先填一个数字，猜错也没关系。';return;}const ok=Number.isFinite(Number(value))&&Math.abs(Number(value)-result.challenge.answer)<0.015;feedback.textContent=(ok?'算对了。':'还差一点。')+' '+result.challenge.explain;feedback.dataset.correct=String(ok);});
      document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
      paint();
    }
  }
  return {models,calculate,mount};
});
