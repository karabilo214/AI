import unittest
import torch
from .audit_support import definition

class OOMTest(unittest.TestCase):
    def test_moe_preserves_original_error_and_invalidates_cache(self):
        cls=definition('moe.py','MoEEngine',dict(GGUFEngine=object,GiB=2**30,torch=torch))
        eng=object.__new__(cls);eng.ctx=[1,2];eng.graphs={1:object()}
        error=torch.OutOfMemoryError('test OOM')
        def fail(*args): raise error
        eng._process=fail
        with self.assertRaises(torch.OutOfMemoryError) as caught: eng.process([3],2)
        self.assertIs(caught.exception,error)
        self.assertEqual(eng.ctx,[]);self.assertEqual(eng.graphs,{})
    def test_normal_path(self):
        cls=definition('moe.py','MoEEngine',dict(GGUFEngine=object,GiB=2**30,torch=torch))
        eng=object.__new__(cls);eng._process=lambda *a:[9]
        self.assertEqual(eng.process([1],0),[9])
if __name__=='__main__':unittest.main()
