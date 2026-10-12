import test from 'node:test';import assert from 'node:assert/strict';import { once } from 'node:events';import { createServer } from '../server.mjs';
test('Standalone portfolio routes and AI integration',async()=>{const server=await createServer({});server.listen(0,'127.0.0.1');await once(server,'listening');const base=`http://127.0.0.1:${server.address().port}`,original=globalThis.fetch;
try{
 let r=await fetch(base+'/');assert.equal(r.status,200);assert.match(await r.text(),/theme.css/);
 r=await fetch(base+'/resume-preview.jpg');assert.equal(r.status,200);assert.equal(r.headers.get('Content-Type'),'image/jpeg');
 for(const name of ['harjot_singh_resume.pdf','voice-project.zip','churn-project.zip','spam-project.zip'])assert.equal((await fetch(base+'/'+name)).status,200);
 for(const path of ['/.env','/server.mjs','/data/knowledge.json','/%2e%2e%2f.env'])assert.equal((await fetch(base+path)).status,404);
 r=await fetch(base+'/api/status');assert.deepEqual(await r.json(),{configured:false});
 r=await fetch(base+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:'Who is Harjot?'})});assert.equal(r.status,503);
 r=await fetch(base+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json',Origin:'https://other.test'},body:'{}'});assert.equal(r.status,403);
 let model;globalThis.fetch=async(url,options)=>{if(String(url).startsWith('https://api.groq.com/')){const data=JSON.parse(options.body);model=data.model;assert.match(data.messages[0].content,/DAV UNIVERSITY/);return new Response(JSON.stringify({choices:[{message:{content:'Harjot studied at DAV University; result awaited.'}}]}));}return original(url,options);};
 r=await fetch(base+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:'Tell me about his education',apiKey:'test-key-for-mock-provider'})});assert.equal(r.status,200);assert.match((await r.json()).answer,/DAV/);assert.equal(model,'openai/gpt-oss-20b');
 console.log('HTTP routes, files, grounding, Groq model selection, and secret-file protection passed.');
}finally{globalThis.fetch=original;await new Promise(resolve=>server.close(resolve));}});
test('Optional password protection',async()=>{const server=await createServer({BASIC_AUTH_USER:'owner',BASIC_AUTH_PASSWORD:'test-password'});server.listen(0,'127.0.0.1');await once(server,'listening');const base=`http://127.0.0.1:${server.address().port}`;try{assert.equal((await fetch(base+'/')).status,401);assert.equal((await fetch(base+'/',{headers:{Authorization:'Basic '+Buffer.from('owner:test-password').toString('base64')}})).status,200);}finally{await new Promise(resolve=>server.close(resolve));}});
