import types,time,unittest
import numpy as np
from .audit_support import definition
ns=dict(np=np,time=time,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None)))
Engine=definition('engine.py','Engine',ns)
class Toy(Engine):
    def __init__(self,maxt):self.maxt=maxt;self.ctx=[];self.ck=types.SimpleNamespace(eos={999});self.calls=[]
    def process(self,tokens,pos0,*a,**kw):
        if pos0<0 or pos0+len(tokens)>self.maxt:raise AssertionError('outside context')
        self.calls.append((len(tokens),pos0));self.ctx=self.ctx[:pos0]+list(tokens);return [7]*len(tokens)
    def draft(self,seq,k,cum=None):
        if len(seq)+k>self.maxt:raise AssertionError('draft outside context')
        self.draft_confs=[1.0]*k;return [7]*k
class ContextTest(unittest.TestCase):
    def test_step_and_draft_finish_at_boundary(self):
        for p in (1,3,8):
            for dm in (None,4,8,100):
                e=Toy(8);d=Toy(dm) if dm else None
                out=list(e.generate([1]*p,100,draft=d,k=7))
                self.assertEqual(out,[7]*(8-p+1))
                self.assertTrue(all(pos+n<=8 for n,pos in e.calls))
    def test_long_prompt_rejected(self):
        with self.assertRaises(ValueError):list(Toy(8).generate([1]*9))
if __name__=='__main__':unittest.main()
