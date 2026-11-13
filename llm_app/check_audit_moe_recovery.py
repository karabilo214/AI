"""MoE: OOM остаётся OOM, а следующий запрос пересчитывает повреждённый префикс."""
import torch
from .gengine import open_gguf

def main():
    torch.set_grad_enabled(False)
    eng=open_gguf('qwen3:30b',maxt=256)
    try:
        for seed in (3,41,99):
            prompt=[1000+(i*(seed+7919))%50000 for i in range(24)]
            ref=list(eng.generate(prompt,8))
            pick=eng._pick
            def fail(*args,**kwargs):
                pick(*args,**kwargs)
                torch.empty(1 << 60,dtype=torch.uint8,device='cuda')
                raise AssertionError('allocation should fail')
            eng._pick=fail
            try:
                list(eng.generate(prompt,8))
            except torch.OutOfMemoryError:
                assert eng.ctx==[]
            else:raise AssertionError('MoE should preserve original OOM')
            finally:del eng._pick
            actual=list(eng.generate(prompt,8))
            assert actual==ref,(seed,ref,actual)
            print(f'MoE OOM seed={seed}: original error, cleared prefix, next request OK',flush=True)
    finally:eng.close()
    print('AUDIT_MOE_RECOVERY_OK',flush=True)
if __name__=='__main__':main()
