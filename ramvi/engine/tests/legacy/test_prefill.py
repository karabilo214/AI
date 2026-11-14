import unittest
import numpy as np
import torch
from .audit_support import definition
from ramvi_engine.formats.gguf import dequant
from ramvi_engine.kernels.prefill import weight_rows, matrix
repack=definition('gkernels.py','repack',dict(np=np,torch=torch))
compress=definition('wcomp.py','compress',dict(torch=torch))

class PrefillTest(unittest.TestCase):
    def test_gguf_rows_against_independent_decoder(self):
        for seed in (11,37,103):
            rng=np.random.default_rng(seed)
            for typ,bs in [('Q4_K',144),('Q6_K',210)]:
                n,k=7,768
                raw=rng.integers(0,256,n*(k//256)*bs,dtype=np.uint8)
                blocks=raw.reshape(-1,bs)
                # Конечные масштабы, включая отрицательные Q6.
                if typ=='Q4_K':
                    blocks[:,:4]=np.array([.003,.002],dtype=np.float16).view(np.uint8)
                else:blocks[:,-2:]=np.array([-.004],dtype=np.float16).view(np.uint8)
                w=repack(raw,n,k,typ,dev='cpu')
                ref=dequant(raw,typ,n*k).reshape(n,k)
                np.testing.assert_allclose(weight_rows(w,2,6).numpy(),ref[2:6],rtol=2e-7,atol=1e-7)
    def test_compressed_rows_include_exceptions(self):
        for seed in (7,43,109):
            torch.manual_seed(seed)
            w=torch.randn(9,512).bfloat16();w[3,11]=1e-30;w[4,92]=1e20
            packed=compress(w)
            self.assertTrue(torch.equal(weight_rows(packed,2,7),w[2:7].float()))
    def test_matrix_rows_tail_add_and_strides(self):
        torch.manual_seed(23)
        for n,k,t in ((1,256,1),(133,512,37),(257,256,131)):
            w=torch.randn(n,k).bfloat16();x=torch.randn(t,k)
            for packed in (w,compress(w)):
                target=torch.randn(t,n+9);before=target.clone()
                matrix(x,packed,target[:,:n],add=True)
                torch.testing.assert_close(target[:,:n],before[:,:n]+x@w.float().T,rtol=1e-5,atol=5e-5)
                self.assertTrue(torch.equal(target[:,n:],before[:,n:]))

    def test_short_prompt_fallback_and_explicit_mode(self):
        from unittest.mock import patch
        from ramvi_engine.kernels.prefill import enabled
        with patch.dict('os.environ',{'LLM_PREFILL':'blocked','LLM_PREFILL_MIN_TOKENS':'256'}):
            self.assertFalse(enabled(255));self.assertTrue(enabled(256))
        with patch.dict('os.environ',{'LLM_PREFILL':'legacy'}):self.assertFalse(enabled(1024))

    def test_consecutive_prefills_restart_finite_loader_order(self):
        import collections
        from .audit_support import definition
        engine=definition('engine.py','Engine',dict(itertools=__import__('itertools')))
        e=object.__new__(engine);e.all_gpu=False;e.phase='prefill';e.pending=collections.deque()
        e.L=3;e.place=['gpu','stream','cpu'];e.stream_ids=[1];requests=[];e._request=requests.append
        e._set_phase('prefill');e._set_phase('prefill')
        self.assertEqual(requests,[1,2,1,2])
        e.phase='decode';e._set_phase('decode')
        self.assertEqual(requests,[1,2,1,2])
