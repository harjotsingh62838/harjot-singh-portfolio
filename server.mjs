import http from 'node:http';
import { readFile, readdir } from 'node:fs/promises';
import { timingSafeEqual } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';
import { Readable } from 'node:stream';
import { pathToFileURL } from 'node:url';
import chat from './chat.mjs';
const publicDirectory=fileURLToPath(new URL('./public/',import.meta.url));
const mime={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.pdf':'application/pdf','.jpg':'image/jpeg','.zip':'application/zip'};
function equal(a,b){const x=Buffer.from(a),y=Buffer.from(b);return x.length===y.length&&timingSafeEqual(x,y);}
export async function createServer(config=process.env){
 const assets=new Set(await readdir(publicDirectory));
 if((config.BASIC_AUTH_USER&&!config.BASIC_AUTH_PASSWORD)||(!config.BASIC_AUTH_USER&&config.BASIC_AUTH_PASSWORD))throw new Error('Set both BASIC_AUTH_USER and BASIC_AUTH_PASSWORD, or leave both empty.');
 const handler=async(req,res)=>{
  const finish=(status,data,headers={})=>{res.writeHead(status,{'X-Content-Type-Options':'nosniff','Referrer-Policy':'strict-origin-when-cross-origin',...headers});res.end(data);};
  try{
   if(config.BASIC_AUTH_USER){const expected='Basic '+Buffer.from(config.BASIC_AUTH_USER+':'+config.BASIC_AUTH_PASSWORD).toString('base64');if(!equal(req.headers.authorization||'',expected))return finish(401,'Sign in to this portfolio.',{'WWW-Authenticate':'Basic realm="Harjot Portfolio"'});}
   const host=req.headers.host||'localhost';const origin=config.PUBLIC_ORIGIN||`http://${host}`;
   const url=new URL(req.url,origin);
   if(url.pathname==='/health')return finish(200,'{"ok":true}',{'Content-Type':'application/json','Cache-Control':'no-store'});
   if(url.pathname.startsWith('/api/')){
    let bytes=0;const chunks=[];
    for await(const chunk of req){bytes+=chunk.length;if(bytes>24000)return finish(413,'{"error":"Request is too large"}',{'Content-Type':'application/json'});chunks.push(chunk);}
    const headers=new Headers();for(const [key,value]of Object.entries(req.headers)){if(value!==undefined&&!['host','connection','content-length','transfer-encoding','x-internal-client-ip','oai-authenticated-user-id','cf-connecting-ip'].includes(key))headers.set(key,Array.isArray(value)?value.join(','):value);}
    headers.set('x-internal-client-ip',req.socket.remoteAddress||'local');
    const body=Buffer.concat(chunks);const request=new Request(new URL(req.url,origin),{method:req.method,headers,...(!['GET','HEAD'].includes(req.method)?{body}:{} )});
    const response=await chat.fetch(request,{GROQ_API_KEY:config.GROQ_API_KEY,GROQ_MODEL:config.GROQ_MODEL||'openai/gpt-oss-20b'});
    res.writeHead(response.status,Object.fromEntries(response.headers));if(req.method==='HEAD')res.end();else if(response.body)Readable.fromWeb(response.body).pipe(res);else res.end();return;
   }
   if(!['GET','HEAD'].includes(req.method))return finish(405,'Method not allowed');
   let name;try{name=decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname).slice(1);}catch{return finish(400,'Invalid URL');}
   if(!assets.has(name)||name.includes('/')||name.includes('\\')||name.startsWith('.'))return finish(404,'Not found');
   const data=await readFile(join(publicDirectory,name));const extension=name.slice(name.lastIndexOf('.'));
   finish(200,req.method==='HEAD'?undefined:data,{'Content-Type':mime[extension]||'application/octet-stream','Cache-Control':'no-cache',...(extension==='.zip'?{'Content-Disposition':`attachment; filename="${name}"`}:{})});
  }catch(error){console.error('Request failed:',error.name);if(!res.headersSent)finish(500,'{"error":"The server could not process this request."}',{'Content-Type':'application/json'});else res.end();}
 };
 return http.createServer(handler);
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){const port=Number(process.env.PORT||3000);if(!Number.isInteger(port)||port<1||port>65535)throw new Error('PORT must be between 1 and 65535.');const server=await createServer();server.listen(port,process.env.HOST||'0.0.0.0',()=>console.log(`Portfolio running on port ${port}. Open http://localhost:${port}`));for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{server.close(()=>process.exit(0));setTimeout(()=>process.exit(0),5000).unref()});}
