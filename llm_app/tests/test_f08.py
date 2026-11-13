import unittest,numpy as np,torch
from .audit_support import definition
class RepackTest(unittest.TestCase):
    def test_f32_preserves_bits_and_range(self):
        repack=definition('gkernels.py','repack',dict(np=np,torch=torch))
        for seed in (0,1,89):
            a=np.random.default_rng(seed).standard_normal(256).astype(np.float32)
            a[:3]=[1.0001,70000,-70000]
            w=repack(a.view(np.uint8),1,256,'F32',dev='cpu')
            self.assertEqual(w['t'],'F32');self.assertEqual(w['W'].dtype,torch.float32)
            np.testing.assert_array_equal(w['W'].numpy().ravel().view(np.uint32),a.view(np.uint32))
if __name__=='__main__':unittest.main()
