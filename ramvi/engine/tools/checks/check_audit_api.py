"""HTTP/API и грамматика на настоящей модели; сервер живёт только внутри проверки."""
import json,subprocess,sys,threading,urllib.request,urllib.error
import torch
from transformers import AutoTokenizer
from ramvi_engine.core.gengine import open_gguf
from ramvi_engine.formats.model import find_model
from ramvi_engine.server.app import App,LimitedHTTPServer,make_handler
from ramvi_engine.formats.tokenizer_check import validate_tokenizer,validate_draft

def main():
    torch.set_grad_enabled(False)
    tk=AutoTokenizer.from_pretrained(find_model('Qwen3-8B'))
    draft=open_gguf('qwen3:0.6b',maxt=1024,vram_reserve=0)
    eng=open_gguf('qwen3:8b',maxt=1024,cpu='hybrid')
    validate_draft(eng,draft,tk,tk)
    app=App(eng,tk,'audit',draft=draft,api_key='audit-key',force_greedy=True)
    server=LimitedHTTPServer(('127.0.0.1',0),make_handler(app))
    thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.05));thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    def request(path,body,key='audit-key'):
        data=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request(base+path,data,{'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=90) as response:return response.read().decode()
    try:
        assert json.loads(request('/v1/models',None))['data'][0]['id']=='audit'
        try:request('/v1/models',None,key='wrong');raise AssertionError('auth accepted')
        except urllib.error.HTTPError as exc:assert exc.code==401
        body={'model':'audit','messages':[{'role':'user','content':'Say exactly: audit-ok'}], 'temperature':0,'max_tokens':32}
        first=json.loads(request('/v1/chat/completions',body))
        text=first['choices'][0]['message']['content'];assert text and first['usage']['completion_tokens']>0
        stream=request('/v1/chat/completions',{**body,'stream':True})
        pieces=[]
        for line in stream.splitlines():
            if line.startswith('data: ') and line[6:]!='[DONE]':
                event=json.loads(line[6:])
                if event.get('choices'):pieces.append(event['choices'][0].get('delta',{}).get('content') or '')
        assert ''.join(pieces)==text,('stream differs',text,''.join(pieces))
        zero=json.loads(request('/v1/chat/completions',{**body,'max_tokens':0}))
        assert zero['usage']['completion_tokens']==0 and zero['choices'][0]['message']['content']==''
        schema={'type':'object','properties':{'x':{'type':'integer','const':17}},'required':['x'],'additionalProperties':False}
        constrained={**body,'max_tokens':64,'response_format':{'type':'json_schema','json_schema':{'name':'answer','schema':schema}}}
        result=json.loads(request('/v1/chat/completions',constrained))
        assert json.loads(result['choices'][0]['message']['content'])=={'x':17},result
        short=json.loads(request('/v1/chat/completions',{**constrained,'max_tokens':1}))
        assert short['choices'][0]['finish_reason']=='length'
        for path in ('/api/chat','/v1/messages','/v1/completions'):
            req={**body,'max_tokens':32,'prompt':'Say exactly: audit-ok','stream':False}
            result=json.loads(request(path,req));assert result
        if '--extended' in sys.argv:
            cancelled=app.run(app.build_ids([{'role':'user','content':'Write a long numbered list of 1000 animals.'}]),256,0.0)
            next(cancelled)
            cancelled.close()
            assert not any(t.name=='llm-response' and t.is_alive() for t in threading.enumerate())
            assert app.lock.acquire(timeout=1),'model lock was not released after cancellation'
            app.lock.release()
            again=json.loads(request('/v1/chat/completions',body))
            assert again['choices'][0]['message']['content']==text
            print('REAL_CANCEL_OK',flush=True)
            subprocess.run([sys.executable,'-m','ramvi.engine.tools.checks.api_test',base,'audit-key'],check=True,timeout=180)
        print('AUDIT_API_OK: auth, stream equality, zero budget, JSON schema, truncation, endpoints',flush=True)
    finally:
        server.shutdown();thread.join(timeout=5);server.server_close()
        eng.close();draft.close()
        assert not thread.is_alive()
if __name__=='__main__':main()
