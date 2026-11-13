import types,time,unittest,numpy as np
from .audit_support import definition
Engine=definition('engine.py','Engine',dict(np=np,time=time,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None))))
class SamplingEngine(Engine):
    def __init__(self):
        self.ctx=[];self.maxt=100;self.ck=types.SimpleNamespace(eos={-1});self.hb=np.zeros((16,2),np.float32)
        self._decode=lambda tokens,pos:len(tokens)
        self.head=types.SimpleNamespace(pick=lambda h,t,temp,p,rng,**kw:rng.integers(1,10000,size=t).tolist())
class PromptSelectionTest(unittest.TestCase):
    def test_cache_does_not_change_sampling_rng(self):
        e=SamplingEngine();prompt=[1,2,3,4,5]
        for seed in (3,41,99):
            e.ctx=[]
            cold=list(e.generate(prompt,4,temperature=1,seed=seed))
            warm=list(e.generate(prompt,4,temperature=1,seed=seed))
            self.assertEqual(cold,warm)
    def test_constraint_mask_applies_only_to_last_prompt_row(self):
        cls=definition('gengine.py','GGUFEngine',dict(Engine=Engine,GiB=2**30))
        e=object.__new__(cls);e.ctx=[];e.maxt=100;e._decode=lambda tokens,pos:len(tokens)
        mask=object();calls=[]
        def pick(t,temp,p,rng,actual_mask,row0=0):
            calls.append((t,row0));self.assertIs(actual_mask,mask);return [9]
        e._pick=pick
        self.assertEqual(e._process([1,2,3,4,5],0,mask=mask,last_only=True),[9])
        self.assertEqual(calls,[(1,4)])
if __name__=='__main__':unittest.main()
