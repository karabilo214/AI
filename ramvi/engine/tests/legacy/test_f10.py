import unittest,types
import torch
from .audit_support import definition
class RecoveryTest(unittest.TestCase):
    def cls(self):
        shim=types.SimpleNamespace(OutOfMemoryError=torch.OutOfMemoryError,cuda=types.SimpleNamespace(empty_cache=lambda:None))
        return definition('gengine.py','GGUFEngine',dict(Engine=object,GiB=2**30,torch=shim))
    def test_retry_rebuilds_prefix_and_is_bounded(self):
        cls=self.cls();e=object.__new__(cls)
        e.L=4;e.ctx=[1,2];e.place=['gpu']*4;e.all_gpu=True;e.log=lambda *a:None
        calls=[]
        def process(tokens,pos,*args):
            calls.append((list(tokens),pos))
            if len(calls)==1:raise torch.OutOfMemoryError('first')
            e.ctx=e.ctx[:pos]+list(tokens);return [9]
        e._process=process;e._demote=lambda l:e.place.__setitem__(l,'stream')
        self.assertEqual(e.process([3],2),[9]);self.assertEqual(calls,[([3],2),([1,2],0),([3],2)])
        self.assertEqual(e.ctx,[1,2,3])
    def test_failure_in_recovery_preserves_oom(self):
        cls=self.cls();e=object.__new__(cls);e.L=1;e.ctx=[];e.place=['gpu'];e.all_gpu=True
        error=torch.OutOfMemoryError('original')
        def fail(*a):raise error
        e._process=fail
        def demote(*a):raise RuntimeError('no buffer')
        e._demote=demote
        with self.assertRaises(torch.OutOfMemoryError) as c:e.process([1],0)
        self.assertIs(c.exception,error);self.assertIsInstance(c.exception.__cause__,RuntimeError)
    def test_all_gpu_loader_phase_resets(self):
        e=object.__new__(self.cls());e.all_gpu=True;e.pending=[];e.phase='dN'
        e._reset_loader();self.assertIsNone(e.phase)
if __name__=='__main__':unittest.main()
