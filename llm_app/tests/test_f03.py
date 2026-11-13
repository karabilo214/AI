import unittest
import types
from .audit_support import definition
class CacheTest(unittest.TestCase):
    def classes(self):
        eng=definition('engine.py','Engine',{})
        gg=definition('gengine.py','GGUFEngine',dict(Engine=eng,GiB=2**30))
        return [(eng,'process'),(gg,'_process')]
    def test_failed_decode_and_prefill_clear_cache(self):
        for cls,method in self.classes():
            for n in (1,17):
                with self.subTest(cls=cls.__name__,n=n):
                    e=object.__new__(cls);e.ctx=[1,2];e.maxt=100
                    def fail(*a): raise RuntimeError('simulated compute error')
                    e._decode=e._prefill=fail
                    with self.assertRaisesRegex(RuntimeError,'simulated'):
                        getattr(e,method)([3]*n,1)
                    self.assertEqual(e.ctx,[])
    def test_success_publishes_only_after_pick(self):
        for cls,method in self.classes():
            e=object.__new__(cls);e.ctx=[1,2];e.maxt=100;e.hb=[0]
            e._decode=lambda *a:1
            def pick(*a,**k):
                self.assertEqual(e.ctx,[1,2]);return [7]
            e.head=types.SimpleNamespace(pick=pick);e._pick=pick
            self.assertEqual(getattr(e,method)([3],1),[7]);self.assertEqual(e.ctx,[1,3])
if __name__=='__main__': unittest.main()
