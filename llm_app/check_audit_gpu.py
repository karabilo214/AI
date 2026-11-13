"""Границы графа, сохранность KV и F32 на настоящей NVIDIA. Без Ollama-сервера."""
import torch,numpy as np
import threading
from .engine import Loader
from .gengine import GGUFEngine
from .gkernels import repack,ggemv

def main():
    torch.set_grad_enabled(False)
    for seed in (3,41,99):
        rng=np.random.default_rng(seed)
        w=rng.standard_normal((32,256)).astype(np.float32)
        w[0,:2]=[70000,1.0001]
        packed=repack(w.view(np.uint8).ravel(),32,256,'F32')
        for t in (1,3,16):
            x=torch.from_numpy(rng.standard_normal((t,256)).astype(np.float32)).cuda()
            y=torch.empty(t,32,device='cuda');ggemv(x,packed,y,0,t)
            ref=x.double()@torch.from_numpy(w).cuda().double().T
            err=float((y.double()-ref).abs().max()/ref.abs().max())
            assert torch.isfinite(y).all() and err<1e-4,(seed,t,err)
    print('F32: 9 cases OK',flush=True)
    del packed,x,y,ref;torch.cuda.empty_cache()
    eng=GGUFEngine('qwen3:0.6b',maxt=64,cpu=False)
    assert eng.all_gpu,'small model must fit for graph test'
    tokens=[1000+(i*7919)%50000 for i in range(64)]
    loader=Loader();stop=threading.Event();failures=[]
    src=torch.zeros(40*2**20,dtype=torch.uint8).pin_memory();dst=torch.empty_like(src,device='cuda')
    def copy_loop():
        try:
            while not stop.is_set():
                job={'wait':None,'pairs':[(src,dst)],'done':threading.Event(),'rel':{'done':threading.Event()}}
                loader.q.put(job);loader.wait(job);job['ev'].synchronize()
        except Exception as exc:failures.append(exc)
    copier=threading.Thread(target=copy_loop);copier.start()
    try:
        for b in (1,3,5,7,9,15,16):
            eng.process(tokens,0)
            saved=[(cache,cache.clone()) for d in (eng.Kc,eng.Vc) for cache in d.values()]
            eng._capture(b)
            torch.cuda.synchronize()
            assert all(torch.equal(cache,old) for cache,old in saved),('capture corrupted KV',b)
            eng.process(tokens[:64-b],0)
            out=eng.process(tokens[64-b:],64-b)
            assert len(out)==b and torch.isfinite(eng.logits[:b]).all(),b
            print(f'graph T={b}: KV preserved, boundary OK',flush=True)
    finally:
        stop.set();copier.join(timeout=5);loader.close()
    assert not copier.is_alive() and not failures,failures
    eng.close()
    print('AUDIT_GPU_OK',flush=True)
if __name__=='__main__':main()
