"""Проверка архитектур/плана по реальным файлам без загрузки весов в GPU."""
import torch
from ramvi_engine.formats.gguf import GGUF,find_ollama,validate_qwen,eos_ids
from ramvi_engine.planning.plan import plan_gguf

def main():
    for name in ('qwen3:0.6b','qwen3:8b','qwen3:14b','qwen3:30b'):
        g=GGUF(find_ollama(name))
        validate_qwen(g);assert eos_ids(g)
        g.close()
        before=torch.cuda.memory_allocated()
        plan=plan_gguf(name,maxt=256,log=lambda *a:None)
        assert len(plan['place'])==len(plan['layer_file_bytes'])
        assert torch.cuda.memory_allocated()==before
        print(name+': architecture, EOS, metadata-only plan OK',flush=True)
    print('AUDIT_METADATA_OK',flush=True)
if __name__=='__main__':main()
