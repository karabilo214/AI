"""Настоящий отказ выделения CUDA-памяти после реального прохода; KV и переносы."""
import torch
import gc
import threading
from .gengine import GGUFEngine

def main():
    torch.set_grad_enabled(False)
    for seed in (3,41,99):
        eng=GGUFEngine('qwen3:0.6b',maxt=128,cpu=False)
        prompt=[1000+(i*(seed+7919))%50000 for i in range(40)]
        reference=list(eng.generate(prompt,12))
        pick=eng._pick;injected=[False]
        def failing_pick(*args,**kwargs):
            out=pick(*args,**kwargs)
            if not injected[0]:
                injected[0]=True
                torch.empty(1 << 60, dtype=torch.uint8, device='cuda')
                raise AssertionError('oversized allocation unexpectedly succeeded')
            return out
        eng._pick=failing_pick
        actual=list(eng.generate(prompt,12))
        assert injected[0] and actual==reference,(seed,reference,actual)
        assert eng.place.count('stream')>=1
        assert eng.ctx==(prompt+actual)[:-1]
        del eng._pick
        eng.close()
        assert not any(t.name == 'llm-loader' and t.is_alive() for t in threading.enumerate())
        del eng, pick, failing_pick
        gc.collect();torch.cuda.empty_cache()
        print(f'recovery seed={seed}: tokens, prefix and demotion OK',flush=True)
    print('AUDIT_RECOVERY_OK',flush=True)
if __name__=='__main__':main()
