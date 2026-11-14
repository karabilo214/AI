import unittest
import torch
import types
from .audit_support import definition
class GraphBoundsTest(unittest.TestCase):
    def test_graph_uses_real_batch_at_context_boundary(self):
        eng=definition('engine.py','Engine',{})
        gg=definition('gengine.py','GGUFEngine',dict(Engine=eng,GiB=2**30))
        for cls in (eng,gg):
            for size in (1,3,5,7,9,15,16):
                e=object.__new__(cls);e.all_gpu=True;e.graphs={};e.maxt=20
                e.xb=torch.zeros(16,2);e.pos=torch.zeros(1,dtype=torch.long)
                e._set_phase=lambda phase:None;e._emb=lambda tokens:torch.ones(len(tokens),2)
                seen=[]
                def capture(b):
                    seen.append(b);e.graphs[b]=types.SimpleNamespace(replay=lambda:None)
                e._capture=capture
                self.assertEqual(e._decode([1]*size,20-size),size)
                self.assertEqual(seen,[size]);self.assertLessEqual(e.pos.item()+size,e.maxt)
if __name__=='__main__':unittest.main()
