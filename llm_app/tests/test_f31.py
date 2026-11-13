import hashlib,json,math,os,tempfile,types,unittest
from pathlib import Path
from .audit_support import definition,ROOT
class FakeTensor:
    def __mul__(self,other):return self
    def bfloat16(self):return self
class TuneTest(unittest.TestCase):
    def test_corrupted_cache_is_ignored_and_all_failures_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tune.json';p.write_text('{bad')
            wcomp=types.SimpleNamespace(__file__=str(ROOT/'wcomp.py'),CFG1={},CFGT={})
            torch=types.SimpleNamespace(__version__='test',cuda=types.SimpleNamespace(get_device_name=lambda:'test'),
                                       randn=lambda *a,**kw:FakeTensor(),zeros=lambda *a,**kw:FakeTensor())
            ns=dict(torch=torch,hashlib=hashlib,Path=Path,json=json,os=os,math=math,CACHE=str(p),wcomp=wcomp,
                    compress=lambda x:None,_try=lambda f:float('inf'))
            tune=definition('tune.py','tune',ns);logs=[]
            tune([],log=logs.append);self.assertTrue(logs)
            with self.assertRaises(RuntimeError):tune([(16,256,0)],log=logs.append)
            self.assertEqual(p.read_text(),'{bad')
if __name__=='__main__':unittest.main()
