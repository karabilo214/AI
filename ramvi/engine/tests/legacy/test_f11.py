import unittest,numpy as np
from ramvi_engine.cpu.cpu_layer import qk_rope_attn
class AttentionPrecisionTest(unittest.TestCase):
    def test_real_numba_rounding_and_step_batch_agree(self):
        for seed in (0,17,99):
            qkv=np.random.default_rng(seed).standard_normal((3,16)).astype(np.float32)
            qkv[:,12:]+=np.float32(.0001)
            inv=np.array([1,.1],np.float32);norm=np.ones(4,np.float32)
            k=np.zeros((1,8,4),np.float32);v=np.zeros_like(k);out=np.zeros((3,8),np.float32)
            qk_rope_attn(qkv,norm,norm,inv,0,k,v,out,2,1,4,1e-6,True)
            np.testing.assert_array_equal(v[0,:3],qkv[:,12:].astype(np.float16).astype(np.float32))
            np.testing.assert_array_equal(k,k.astype(np.float16).astype(np.float32))
            ks=np.zeros_like(k);vs=np.zeros_like(v);steps=np.zeros_like(out)
            for i in range(3):qk_rope_attn(qkv[i:i+1],norm,norm,inv,i,ks,vs,steps[i:i+1],2,1,4,1e-6,True)
            np.testing.assert_array_equal(out,steps)
if __name__=='__main__':unittest.main()
