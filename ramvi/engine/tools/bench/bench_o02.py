"""Парное сравнение O02 с точным снимком прежнего generate, без смены размещения."""
from ramvi_engine.config import results_dir
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from ramvi.engine.tools.bench.bench_wide import WIDE
from ramvi_engine.core.engine import Engine
from ramvi_engine.core.gengine import GGUFEngine
from ramvi_engine.formats.model import find_model
from ramvi.engine.tools.processes import available, kill_tree

FRESH = [
 ('fresh', 'Вычисли 137 * 29 и объясни вычисление.'),
 ('fresh', 'Return JSON with primes between 211 and 251, sorted ascending.'),
 ('fresh', 'Write a Rust function computing the greatest common divisor of 1234567 and 891011.'),
 ('fresh', 'Переведи на английский: В четверг 19 ноября курьер доставит семь коробок в 08:43.'),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--tokens', type=int, default=48)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cases', default='', help='индексы в WIDE+FRESH для точного повтора')
    args=ap.parse_args()
    remaining=datetime.datetime.fromisoformat(args.deadline).timestamp()-time.time()
    if remaining<30: raise SystemExit('deadline expired')
    available()
    watchdog=threading.Timer(remaining,lambda:kill_tree(os.getpid()))
    watchdog.daemon=True;watchdog.start()
    try: run(args)
    finally: watchdog.cancel();watchdog.join(timeout=2)

def run(args):
    source=Path(args.baseline).read_text(encoding='utf-8')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    scope=dict(np=np,time=time,torch=torch)
    exec(compile(ast.Module(body=[method],type_ignores=[]),args.baseline,'exec'),scope)
    methods={'old':scope['generate'],'new':Engine.generate}
    tk=AutoTokenizer.from_pretrained(find_model('Qwen3-8B'))
    torch.set_grad_enabled(False)
    draft=GGUFEngine('qwen3:0.6b',maxt=1024,vram_reserve=0)
    eng=GGUFEngine('qwen3:14b',maxt=1024,cpu='hybrid');eng.autofit()
    rows=[];states={'old':{},'new':{}}
    out=(results_dir() / 'o02/paired.json');out.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),
              new_sha256=hashlib.sha256(Path(__import__('ramvi_engine.core.engine', fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
              place=eng.place,tokens=args.tokens,repeats=args.repeats)
    def save(): out.write_text(json.dumps(dict(meta=meta,rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    def measure(label, ids, n):
        available(); eng.ctx=[];draft.ctx=[]
        eng._cost=states.get(label,{}).copy() or {'td':None,'pts':[]}
        st={};calls=0;draft_s=0.;original_draft=draft.draft
        def counted(*a,**kw):
            nonlocal calls,draft_s
            calls+=1;start=time.perf_counter()
            try:return original_draft(*a,**kw)
            finally:draft_s+=time.perf_counter()-start
        draft.draft=counted
        previous_gate=os.environ.get('RAMVI_DRAFT_GATE')
        os.environ['RAMVI_DRAFT_GATE']='1' if label=='new' else '0'
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            if label=='plain': tokens=list(eng.generate(ids,n,stats=st))
            else: tokens=list(methods[label](eng,ids,n,draft=draft,stats=st))
            torch.cuda.synchronize();wall=time.perf_counter()-start
        finally:
            draft.draft=original_draft
            if previous_gate is None:os.environ.pop('RAMVI_DRAFT_GATE',None)
            else:os.environ['RAMVI_DRAFT_GATE']=previous_gate
        st['measured_draft_calls']=calls;st['measured_draft_ms']=draft_s*1000
        if label in states:states[label]=eng._cost.copy()
        return dict(mode=label,ids=tokens,stats=st,wall_s=wall)
    # Заведомо решаемый контроль: один токен не зависит от политики декодирования.
    for label in ('plain','old','new'):
        r=measure(label,tk.encode('Hello'),1)
        rows.append(dict(case='control',repeat=-1,**r))
    assert rows[0]['ids']==rows[1]['ids']==rows[2]['ids']
    cases=WIDE+FRESH
    if args.limit:cases=cases[:args.limit]+FRESH
    if args.cases:cases=[(WIDE+FRESH)[int(i)] for i in args.cases.split(',')]
    for i,(cat,q) in enumerate(cases):
        ids=tk.encode(tk.apply_chat_template([{'role':'user','content':q}],tokenize=False,
                      add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        plain=measure('plain',ids,args.tokens)
        rows.append(dict(case=i,category=cat,q=q,repeat=-1,**plain));save()
        for rep in range(args.repeats):
            order=('old','new') if (i+rep)%2==0 else ('new','old')
            for label in order:
                r=measure(label,ids,args.tokens)
                rows.append(dict(case=i,category=cat,q=q,repeat=rep,**r));save()
                assert r['ids']==plain['ids'],f'token mismatch {i} {rep} {label}'
                print(json.dumps(dict(case=i,repeat=rep,mode=label,stats=r['stats'],wall_s=r['wall_s']),ensure_ascii=True),flush=True)
    print('O02_DONE',flush=True)
if __name__=='__main__':main()
