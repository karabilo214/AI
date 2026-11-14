"""Настоящие Numba-ядра: контроль на известном решении и свежих случайных данных."""
import types,unittest,numpy as np
from ramvi_engine.cpu.gcpu import CpuMat
from ramvi_engine.formats.gguf import dequant
class ArithmeticTest(unittest.TestCase):
    def test_zero_matrix_and_random_quantized_batches(self):
        for seed in (3,41,20261113):
            rng=np.random.default_rng(seed)
            for kind,bs in (('Q4_K',144),('Q6_K',210)):
                raw=rng.integers(0,256,(4,2,bs),np.uint8)
                if kind=='Q4_K':
                    raw[:,:,:2]=np.array([.01],np.float16).view(np.uint8)
                    raw[:,:,2:4]=np.array([.005],np.float16).view(np.uint8)
                else:raw[:,:,208:210]=np.array([.01],np.float16).view(np.uint8)
                raw=raw.ravel();g=types.SimpleNamespace(raw=lambda name:(raw,4,512,kind))
                mat=CpuMat(g,'test');weight=dequant(raw,kind,2048).reshape(4,512).astype(np.float64)
                for t in (1,3,16):
                    x=rng.standard_normal((t,512)).astype(np.float32)
                    for add in (False,True):
                        storage=rng.standard_normal((t,12)).astype(np.float32);y=storage[:,4:8];original=y.copy()
                        mat.mm(x,y,None,add)
                        ref=x.astype(np.float64)@weight.T+(original if add else 0)
                        self.assertLess(float(np.abs(y-ref).max()/max(np.abs(ref).max(),1e-30)),1e-5)
        raw=np.zeros(4*2*144,np.uint8)
        g=types.SimpleNamespace(raw=lambda name:(raw,4,512,'Q4_K'));mat=CpuMat(g,'zero')
        y=np.full((3,4),99,np.float32);mat.mm(np.ones((3,512),np.float32),y,None)
        np.testing.assert_array_equal(y,np.zeros((3,4),np.float32))
if __name__=='__main__':unittest.main()
