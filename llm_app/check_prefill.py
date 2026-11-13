"""NVIDIA A/B prefill: входы, KV, greedy и sampling; JSON после каждого опыта."""
import argparse,datetime,json,os,threading,time
from pathlib import Path
import numpy as np
import torch
from .run_audit_windows import available,kill_tree

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',required=True)
    ap.add_argument('--model',default='qwen3:0.6b');ap.add_argument('--hf',action='store_true')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--only-long',action='store_true')
    args=ap.parse_args();remaining=datetime.datetime.fromisoformat(args.deadline).timestamp()-time.time()
    if remaining<30:raise SystemExit('deadline expired')
    available();watchdog=threading.Timer(remaining,lambda:kill_tree(os.getpid()));watchdog.daemon=True;watchdog.start()
    try:run(args)
    finally:watchdog.cancel();watchdog.join(timeout=2)

def kernel_checks():
    from .prefill_kernels import attention,matmul,decode_rows
    from .gkernels import repack
    from .wcomp import compress
    from .gguf import dequant
    # Заведомо решаемая матрица: identity должна вернуть исходный X.
    eye=torch.eye(256,device='cuda');x=torch.arange(256,device='cuda').float()[None,:].repeat(17,1)
    assert torch.equal(matmul(x,eye),x)
    for seed in (19,71,509):
        rng=np.random.default_rng(seed)
        for typ,bs in (('Q4_K',144),('Q6_K',210)):
            raw=rng.integers(0,256,7*2*bs,dtype=np.uint8);blocks=raw.reshape(-1,bs)
            if typ=='Q4_K':blocks[:,:4]=np.array([.003,.002],np.float16).view(np.uint8)
            else:blocks[:,-2:]=np.array([-.004],np.float16).view(np.uint8)
            packed=repack(raw,7,512,typ)
            actual=decode_rows(packed,2,7).cpu().numpy()
            ref=dequant(raw,typ,7*512).reshape(7,512)[2:7]
            np.testing.assert_allclose(actual,ref,rtol=2e-7,atol=1e-7)
        torch.manual_seed(seed)
        w=torch.randn(9,512,device='cuda').bfloat16();w[3,11]=1e-30;w[4,92]=1e20
        assert torch.equal(decode_rows(compress(w),2,7),w[2:7].float())
        n,start,h,kv,hd=33,7,4,2,64
        q=torch.randn(n,h*hd,device='cuda');k=torch.randn(kv,80,hd,device='cuda');v=torch.randn_like(k)
        out=torch.empty_like(q);attention(q,k,v,out,start,h,kv,hd)
        ref=torch.empty_like(q)
        for head in range(h):
            for t in range(n):
                kk=k[head//2,:start+t+1].double();vv=v[head//2,:start+t+1].double()
                probs=torch.softmax(kk@q[t,head*hd:(head+1)*hd].double()/hd**.5,0)
                ref[t,head*hd:(head+1)*hd]=(probs@vv).float()
        torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6)
    print('PREFILL_KERNELS_OK',flush=True)

def run(args):
    kernel_checks()
    if args.hf:
        from .engine import Engine
        eng=Engine(args.model,maxt=1024,resident=1)
    else:
        from .gengine import GGUFEngine
        eng=GGUFEngine(args.model,maxt=1024,cpu='hybrid')
    rows=[];out=Path('llm_app/results/prefill');out.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False)
    os.environ['LLM_PREFILL_MIN_TOKENS']='0'  # измеряем и короткие размеры кандидата
    def once(ids,start,mode):
        available();os.environ['LLM_PREFILL']=mode
        eng.ctx=[]
        if start:eng.process(ids[:start],0)
        torch.cuda.synchronize();t=time.perf_counter()
        result=eng.process(ids[start:],start)
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        continuation=[];pos=len(ids);token=result[-1]
        for _ in range(16):
            token=eng.process([token],pos)[-1];continuation.append(token);pos+=1
        kv=[(eng.Kc[l][:,:len(ids)].float().cpu().numpy(),eng.Vc[l][:,:len(ids)].float().cpu().numpy()) for l in eng.Kc]
        for l in getattr(eng,'cpu_ids',[]):
            cl=eng.W[l]
            kv.append((cl.K[:,:len(ids)].copy(),cl.V[:,:len(ids)].copy()))
        return result+continuation,seconds,kv
    for seed in (13,61,197,311):
        rng=np.random.default_rng(seed)
        for n,start in (((259,19),) if args.only_long else ((17,0),(259,19),(513,31)) if args.quick else ((17,0),(33,7),(129,0),(259,19),(513,31))):
            ids=rng.integers(100,min(10000,getattr(eng,'V',eng.emb_in.shape[0] if hasattr(eng,'emb_in') else 10000)),n).tolist()
            results={}
            for mode in (('legacy','blocked') if seed%3 else ('blocked','legacy')):
                results[mode]=once(ids,start,mode)
            old,new=results['legacy'],results['blocked']
            diff=max((float(np.abs(a-b).max()) for left,right in zip(old[2],new[2]) for a,b in zip(left,right)),default=0.)
            row=dict(seed=seed,n=n,start=start,legacy_s=old[1],blocked_s=new[1],same=old[0]==new[0],kv_max=diff)
            rows.append(row);(out/f'{"hf" if args.hf else "gguf"}.json').write_text(json.dumps(rows,indent=2))
            print(row,flush=True)
            if not row['same']:raise AssertionError('different token')
    # Ленивая стратегия: plain target, следующий decode и sampling при тёплом prefix.
    for mode in ('legacy','blocked'):
        os.environ['LLM_PREFILL']=mode;eng.ctx=[]
        ids=[100,203,307]*57
        a=list(eng.generate(ids,16,temperature=.8,seed=147))
        b=list(eng.generate(ids,16,temperature=.8,seed=147))
        assert a==b,'prefix-dependent sampling'
    print('PREFILL_DONE',flush=True)
if __name__=='__main__':main()
