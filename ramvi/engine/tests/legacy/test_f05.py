import unittest
import torch
from .audit_support import definition
class CaptureTest(unittest.TestCase):
    def test_capture_restores_live_kv_and_position_even_on_error(self):
        eng=definition('engine.py','Engine',dict(torch=torch))
        gg=definition('gengine.py','GGUFEngine',dict(Engine=eng,GiB=2**30,torch=torch))
        for cls in (eng,gg):
            for size in (1,3,8):
                for fail in (False,True):
                    e=object.__new__(cls);e.maxt=8;e.pos=torch.tensor([2])
                    e.Kc={0:torch.arange(16).reshape(1,8,2).float()};e.Vc={0:e.Kc[0].clone()*2}
                    original=(e.Kc[0].clone(),e.Vc[0].clone())
                    def capture(b):
                        start=e.pos.item();e.Kc[0][:,start:].fill_(99);e.Vc[0][:,start:].zero_()
                        if fail:raise RuntimeError('capture')
                    e._capture_at=capture
                    if fail:
                        with self.assertRaises(RuntimeError):e._capture(size)
                    else:e._capture(size)
                    self.assertTrue(torch.equal(e.Kc[0],original[0]));self.assertTrue(torch.equal(e.Vc[0],original[1]));self.assertEqual(e.pos.item(),2)
if __name__=='__main__':unittest.main()
