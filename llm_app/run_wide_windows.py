"""Полный WIDE30 через HTTP, с повтором и ограниченным временем процесса."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

from .run_audit_windows import available, kill_tree


def write_json(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--deadline',required=True)
    ap.add_argument('--serve',action='store_true')
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--out',default='llm_app/results/wide_current')
    args=ap.parse_args()
    remaining=datetime.datetime.fromisoformat(args.deadline).timestamp()-time.time()
    if remaining<30:raise SystemExit('deadline expired')
    available()
    watchdog=threading.Timer(remaining,lambda:kill_tree(os.getpid()))
    watchdog.daemon=True;watchdog.start()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:
        if args.serve:server(out)
        else:run(args,out)
    finally:watchdog.cancel();watchdog.join(timeout=2)


def server(out):
    import torch
    from transformers import AutoTokenizer
    from .gengine import GGUFEngine
    from .model import find_model
    from .server import App,serve
    from .tokenizer_check import validate_draft
    from .bench_wide import WIDE
    torch.set_grad_enabled(False)
    # Новая версия целиком; настройки заморожены на оба полных круга.
    os.environ['LLM_PREFILL']='blocked'
    os.environ['LLM_PREFILL_MIN_TOKENS']='256'
    tk=AutoTokenizer.from_pretrained(find_model('Qwen3-8B'))
    draft=GGUFEngine('qwen3:0.6b',maxt=2048,vram_reserve=0)
    eng=GGUFEngine('qwen3:14b',maxt=2048,cpu='hybrid')
    eng.autofit()
    validate_draft(eng,draft,tk,tk)
    if os.environ.get('WIDE_SKIP_PROFILE')=='1':
        profile={'enabled':False,'reason':'calibration needs additional RAM; standard application placement'}
    else:
        profile=eng.calibrate_placement(batches=(1,4,16),repeats=3)
        eng.configure_placement(profile)
    write_json(out/'profile.json',profile)
    # Решаемый контроль и ленивая стратегия: plain target без draft.
    controls=[]
    questions=[('arithmetic','Reply with just the number: 2 + 2 ='),
               ('fresh_numbers','Reply with just the number: 137 * 29 =')]+[(str(i),WIDE[i][1]) for i in (0,8,12,19,28)]
    for label,q in questions:
        available()
        ids=tk.encode(tk.apply_chat_template([{'role':'user','content':q}],tokenize=False,
                         add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        eng.ctx=[];ref=list(eng.generate(ids,32))
        eng.ctx=[];draft.ctx=[];got=list(eng.generate(ids,32,draft=draft))
        if ref!=got:raise AssertionError('plain/draft token mismatch '+label)
        text=tk.decode(got,skip_special_tokens=True)
        if label=='arithmetic' and text.strip()!='4':raise AssertionError('solvable arithmetic control: '+repr(text))
        if label=='fresh_numbers' and text.strip()!='3973':raise AssertionError('fresh arithmetic control: '+repr(text))
        controls.append(dict(label=label,q=q,same=True,tokens=got,text=text))
        write_json(out/'controls.json',controls)
        print('CONTROL',label,'OK',flush=True)
    eng._cost={'td':None,'pts':[]};eng.ctx=[];draft.ctx=[]
    app=App(eng,tk,'qwen3:14b',draft=draft)
    meta=app.metadata()
    meta.update(prefill='blocked',prefill_min_tokens=256,measured_routes=getattr(eng,'_measured_routes',{}),
                deadline_env=os.environ.get('WIDE_DEADLINE'))
    write_json(out/'server_metadata.json',meta)
    print('WIDE_SERVER_READY',flush=True)
    serve(app,'127.0.0.1',11435)


def run(args,out):
    from .bench import chat
    from .bench_wide import WIDE
    from .bench_validation import paired_rows,validate_speed
    deadline=datetime.datetime.fromisoformat(args.deadline).timestamp()
    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1','NUMBA_NUM_THREADS':'2',
         'WIDE_DEADLINE':args.deadline}
    with (out/'server.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','llm_app.run_wide_windows','--serve',
                                '--deadline',args.deadline,'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            ready_until=min(deadline-30,time.time()+600)
            while time.time()<ready_until:
                if child.poll() is not None:raise RuntimeError('server exited: '+(out/'server.log').read_text(encoding='utf-8',errors='replace')[-5000:])
                try:
                    with urllib.request.urlopen('http://127.0.0.1:11435/health',timeout=1) as response:
                        health=json.load(response)
                    break
                except OSError:time.sleep(1)
            else:raise TimeoutError('server did not become ready')
            write_json(out/'health.json',health)
            chat('http://127.0.0.1:11435','qwen3:14b','Hello!',8)
            reference=json.loads(Path('llm_app/results/wide_ollama_14b.json').read_text(encoding='utf-8'))
            paired_rows([dict(q=q) for _,q in WIDE],reference,'q');validate_speed(reference)
            for repeat in range(1,args.repeats+1):
                rows=[]
                for index,(cat,q) in enumerate(WIDE):
                    available()
                    if deadline-time.time()<60:raise TimeoutError('deadline too close for another question')
                    text,fin,ttft,wall=chat('http://127.0.0.1:11435','qwen3:14b',q,200)
                    n=fin['eval_count'];seconds=fin['eval_duration']/1e9
                    if not text or n<=0 or seconds<=0:raise AssertionError('empty or untimed result')
                    row=dict(cat=cat,q=q,out=text,n=n,gen_s=seconds,tps=n/seconds,ttft=ttft,wall=wall,
                             run=fin['_run'],total_output_tokens=fin.get('total_output_tokens'),
                             prompt_s=fin.get('prompt_eval_duration',0)/1e9,done_reason=fin.get('done_reason'))
                    rows.append(row)
                    write_json(out/f'wide_current_14b_r{repeat}.json',rows)
                    print(json.dumps(dict(repeat=repeat,question=index+1,cat=cat,n=n,tps=row['tps'],
                                          ttft=ttft,wall=wall),ensure_ascii=True),flush=True)
                paired_rows(rows,reference,'q')
                print('WIDE_FULL_DONE',repeat,len(rows),flush=True)
            write_json(out/'completed.json',dict(repeats=args.repeats,questions=len(WIDE),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        finally:
            if child.poll() is None:kill_tree(child.pid)
            child.wait(timeout=15)
    print('WIDE_RUN_FINISHED',flush=True)

if __name__=='__main__':main()
