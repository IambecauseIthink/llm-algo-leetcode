// Check the exact staged publication payload, independently of local working copies.
const {execFileSync}=require('child_process');
const path=require('path');
const source=require('./upstream_source.json');
const cwd=path.resolve(__dirname,'../..');
const git=(...args)=>execFileSync('git',args,{cwd,encoding:'utf8'});
const official=new Map(git('ls-tree','-rz',source.commit).split('\0').filter(Boolean).filter(x=>x.endsWith('.ipynb')).map(x=>{const [meta,name]=x.split('\t');return [name,meta.split(' ')[2]];}));
const staged=new Map(git('ls-files','--stage','-z').split('\0').filter(Boolean).filter(x=>x.endsWith('.ipynb')).map(x=>{const [meta,name]=x.split('\t');return [name,meta.split(' ')[1]];}));
const changed=[...new Set([...official.keys(),...staged.keys()])].filter(name=>official.get(name)!==staged.get(name));
if(changed.length){console.error('Notebook publication blocked: staged files differ from official source:\n'+changed.join('\n'));process.exit(1);}
console.log(`Publication check passed: ${staged.size} staged notebooks exactly match official ${source.commit.slice(0,7)}.`);
