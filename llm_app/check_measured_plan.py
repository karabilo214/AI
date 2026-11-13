"""Калибровка маршрутов и парный контроль против прежнего плана."""
import argparse,datetime,json,threading,time
from pathlib import Path
import torch
from .run_audit_windows import available,kill_tree

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',required=True)
    ap.add_argument('--model',default='qwen3:14b');ap.add_argument('--resident',type=int);ap.add_argument('--tokens',type=int,default=16)
    a=ap.parse_args();remaining=datetime.datetime.fromisoformat(a.deadline).timestamp()-time.time()
    if remaining<30:raise SystemExit('deadline expired')
    available();timer=threading.Timer(remaining,lambda:kill_tree(os.getpid()));timer.daemon=True;timer.start()
    try:run(a)
    finally:timer.cancel();timer.join(timeout=2)

def run(a):
    from .gengine import GGUFEngine
    from .model import find_model
    from transformers import AutoTokenizer
    torch.set_grad_enabled(False)
    tk=AutoTokenizer.from_pretrained(find_model('Qwen3-8B'))
    e=GGUFEngine(a.model,maxt=1024,cpu='hybrid',resident=a.resident)
    if a.resident is None:e.autofit()
    # Решаемый контроль: одно слово, обе политики совпадают с исходным target.
    control=list(e.generate(tk.encode('Hello'),1))
    profile=e.calibrate_placement()
    p=Path('llm_app/results/measured_plan');p.mkdir(parents=True,exist_ok=True)
    (p/'profile.json').write_text(json.dumps(profile,indent=2))
    e.configure_placement(profile)
    assert list(e.generate(tk.encode('Hello'),1))==control
    routes=e._measured_routes.copy();rows=[]
    # Непрофилированные размеры и длинный контекст тоже должны сохранить
    # переходы между CPU/GPU и обе копии KV.
    static={int(t):v['baseline_route'] for t,v in profile['batches'].items()}
    checks=[]
    for prefix,t in ((1,1),(35,4),(47,16),(270,4)):
        ids=[100+(i*503)%3000 for i in range(prefix+t)]
        answers=[]
        for mapping in (static,routes):
            e._reset_loader();e.ctx=[];e._measured_routes=mapping
            e.process(ids[:prefix],0)
            tokens=e.process(ids[prefix:],prefix)
            following=e.process([tokens[-1]],prefix+t)
            answers.append(tokens+following)
        assert answers[0]==answers[1],f'batch/context mismatch {prefix} {t}'
        checks.append(dict(prefix=prefix,t=t,same=True))
    (p/'checks.json').write_text(json.dumps(checks,indent=2))
    prompts=['Вычисли 137 * 29 и объясни вычисление.','Write a Python function merging two sorted lists.',
             'Return a JSON object with primes from 211 to 251.']
    for rep in range(3):
        for q in prompts:
            ids=tk.encode(tk.apply_chat_template([{'role':'user','content':q}],tokenize=False,
                          add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            pair={}
            for label in (('static','measured') if rep%2==0 else ('measured','static')):
                available();e._reset_loader();e.ctx=[];e._measured_routes={int(t):s['baseline_route'] for t,s in profile['batches'].items()} if label=='static' else routes
                st={};torch.cuda.synchronize();t=time.perf_counter()
                tokens=list(e.generate(ids,a.tokens,stats=st))
                torch.cuda.synchronize()
                pair[label]=dict(tokens=tokens,wall_s=time.perf_counter()-t,stats=st)
            assert pair['static']['tokens']==pair['measured']['tokens'],'different tokens'
            rows.append(dict(q=q,repeat=rep,**pair))
            (p/'paired.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(rows[-1],ensure_ascii=True),flush=True)
    # Sampling без угадывателя — с одинаковым seed и свежим prompt.
    answers=[]
    for mapping in (static,routes):
        e._reset_loader();e.ctx=[];e._measured_routes=mapping
        answers.append(list(e.generate(tk.encode('Compute 137 * 29.'),16,temperature=.7,seed=319)))
    assert answers[0]==answers[1],'sampling mismatch'
    print('MEASURED_DONE',flush=True)
if __name__=='__main__':main()
