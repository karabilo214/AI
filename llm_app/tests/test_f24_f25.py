import sys,types,unittest
from unittest.mock import patch
from .audit_support import definition
class PlacementTest(unittest.TestCase):
    def test_factory_keeps_moe_options(self):
        fn=definition('gengine.py','open_gguf',dict(GGUF=lambda p:types.SimpleNamespace(meta={'general.architecture':'qwen3moe'}),find_ollama=lambda n:n))
        fn.__globals__['__package__']='llm_app'
        with patch.dict(sys.modules,{'llm_app.moe':types.SimpleNamespace(MoEEngine=lambda name,**kw:kw)}):
            self.assertEqual(fn('model.gguf',resident=0,cpu=False,vram_reserve=123),dict(resident=0,cpu=False,vram_reserve=123))
    def test_hf_forced_all_layers_stays_all_layers(self):
        ns=dict(GiB=2**30,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda:(10**12,10**12))),
                ram_info=lambda:(10**12,10**12),sys=types.SimpleNamespace(platform='linux'))
        fn=definition('plan.py','make_plan',ns)
        ck=types.SimpleNamespace(L=4,KV=1,HD=2,D=2,cfg={'vocab_size':4},layer_params=lambda:{'m':(4,4)})
        self.assertEqual(fn(ck,10,force_resident=4,log=lambda *a:None)['place'],['gpu']*4)
        with self.assertRaises(ValueError):fn(ck,10,force_resident=5)
if __name__=='__main__':unittest.main()
