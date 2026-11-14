import unittest,numpy as np,torch
from .audit_support import definition
from ramvi_engine.cpu.cpu_layer import bf
import numba as nb
ns=dict(np=np,torch=torch,nb=nb,bf=bf)
definition("head.py","exact_rows",ns)
ns["exact_rows"] = ns["exact_rows"].py_func
Head=definition("head.py","Head",ns)
class SamplerTest(unittest.TestCase):
    def test_sampling_keeps_low_probability_vocabulary_tail(self):
        h=object.__new__(Head);h.HI=torch.tensor([[0.,-21.]]);h.LO=torch.tensor([0.])
        h.Wc=torch.tensor([[0.],[-21.]]).bfloat16().view(torch.int16).numpy().view(np.uint16)
        class Rng:
            def choice(self,n,p):
                self.n=n;self.p=p;return n-1
        rng=Rng();result=h.pick(torch.ones(1,1),1,temperature=1,top_p=1,rng=rng,run_bounds=False)
        self.assertEqual(result,[1]);self.assertEqual(rng.n,2)
        self.assertGreater(rng.p[-1],0)
if __name__=='__main__':unittest.main()
