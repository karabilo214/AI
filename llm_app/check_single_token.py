"""Сравнение старой/новой DLL на конечных GGUF-блоках; абсолютный дедлайн."""
import argparse,ctypes,datetime,json,os,statistics,threading,time
from pathlib import Path
import numpy as np
from .run_audit_windows import available,kill_tree
from .gguf import dequant

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',required=True);a=ap.parse_args()
    remaining=datetime.datetime.fromisoformat(a.deadline).timestamp()-time.time()
    if remaining<20:raise SystemExit('deadline expired')
    available();timer=threading.Timer(remaining,lambda:kill_tree(os.getpid()));timer.daemon=True;timer.start()
    try:run()
    finally:timer.cancel();timer.join(timeout=2)

def run():
    libs=[ctypes.CDLL(str(Path(p).resolve())) for p in ('llm_app/results/single_token/baseline.dll','llm_app/cpp/qmm.dll')]
    for lib in libs:
        lib.set_threads(16)
        lib.moe_ffn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*7+[ctypes.c_void_p]*5
        lib.moe_ffn.restype=ctypes.c_int
    rows=[]
    try:
        for seed in (17,53,211):
            rng=np.random.default_rng(seed)
            for d,f,k in ((256,256,1),(512,768,3),(2048,768,8)):
                for types in ((0,0,0),(1,1,1),(0,1,0),(1,0,1)):
                    e=8;raw=[]
                    for typ,n,kk in zip(types,(e*f,e*f,e*d),(d,d,f)):
                        bs=144 if typ==0 else 210
                        w=rng.integers(0,256,n*(kk//256)*bs,dtype=np.uint8)
                        blk=w.reshape(-1,bs)
                        if typ==0:blk[:,:4]=np.array([.0001,.0001],np.float16).view(np.uint8)
                        else:blk[:,-2:]=np.array([.0001],np.float16).view(np.uint8)
                        raw.append(w)
                    ids=rng.choice(e,k,replace=False).astype(np.int32);weights=rng.random(k,dtype=np.float32);weights[-1]=0 if k>1 else 1
                    x=rng.normal(size=d).astype(np.float32)
                    hs=[np.full((k,f),np.nan,np.float32) for _ in libs];zs=[np.empty((k,d),np.float32) for _ in libs]
                    funcs=[]
                    for lib,h,z in zip(libs,hs,zs):
                        def call(lib=lib,h=h,z=z):
                            rc=lib.moe_ffn(*[w.ctypes.data for w in raw],*types,d,f,1,k,ids.ctypes.data,weights.ctypes.data,x.ctypes.data,h.ctypes.data,z.ctypes.data)
                            assert rc==0 and lib.take_failed()==0
                        funcs.append(call);call()
                    samples=[[],[]]
                    for rep in range(15):
                        for i in ((0,1) if rep%2 else (1,0)):
                            t=time.perf_counter();funcs[i]();samples[i].append(time.perf_counter()-t)
                    times=[statistics.median(v) for v in samples]
                    assert np.array_equal(zs[0],zs[1]),'single-token mismatch'
                    if k>2:
                        saved=ids.copy();ids[1]=ids[0]
                        for fn in funcs:fn()
                        assert np.array_equal(zs[0],zs[1]),'duplicate expert mismatch'
                        ids[:]=saved
                        weights_saved=weights.copy();weights[:]=0
                        for fn in funcs:fn()
                        assert np.array_equal(zs[0],zs[1]) and not zs[1].any(),'zero weights mismatch'
                        weights[:]=weights_saved
                        for fn in funcs:fn()
                    # Независимый решаемый контроль на маленькой матрице.
                    error=None
                    if d==256:
                        ws=[dequant(w,'Q4_K' if ty==0 else 'Q6_K',n*kk).reshape(e,-1,kk).astype(np.float64)
                            for w,ty,n,kk in zip(raw,types,(e*f,e*f,e*d),(d,d,f))]
                        g=ws[0][ids[0]]@x;u=ws[1][ids[0]]@x
                        ref=ws[2][ids[0]]@(g/(1+np.exp(-g))*u)
                        error=float(np.max(np.abs(zs[1][0]-ref)))
                        assert error / max(float(np.max(np.abs(ref))),1e-12) < 1e-5, (seed,types,error)
                    rows.append(dict(seed=seed,d=d,f=f,k=k,types=types,old_s=times[0],new_s=times[1],same=True,reference_error=error))
        out=Path('llm_app/results/single_token/paired.json');out.write_text(json.dumps(rows,indent=2))
        print('SINGLE_DONE',len(rows),flush=True)
    finally:
        for lib in libs:lib.pool_shutdown()
if __name__=='__main__':main()
