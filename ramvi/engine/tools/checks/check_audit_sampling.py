"""HF sampling: повторяемость seed при свежем/повторном префиксе, полный словарь."""
import torch
import sys
from transformers import AutoTokenizer
from ramvi_engine.core.engine import Engine
from ramvi_engine.formats.model import find_model

def main():
    torch.set_grad_enabled(False)
    name='Qwen3-0.6B';tk=AutoTokenizer.from_pretrained(find_model(name))
    if '--gguf' in sys.argv:
        from ramvi_engine.core.gengine import GGUFEngine
        eng=GGUFEngine('qwen3:0.6b',maxt=256,cpu=False)
    else:
        eng=Engine(name,maxt=256,cpu=False)
    outputs=[]
    try:
        for seed,text in ((3,'Write a short sentence beginning with The.'),(41,'List three colors.'),(99,'A story begins like this:')):
            prompt=tk.encode(tk.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            a=list(eng.generate(prompt,8,temperature=1,top_p=1,seed=seed))
            b=list(eng.generate(prompt,8,temperature=1,top_p=1,seed=seed))
            assert a==b,(seed,a,b)
            outputs.append(tuple(a))
            print(f'sampling seed={seed}: repeated tokens match, n={len(a)}',flush=True)
        assert len(set(outputs))>1,'different inputs unexpectedly give one fixed output'
    finally:eng.close()
    print('AUDIT_SAMPLING_OK',flush=True)
if __name__=='__main__':main()
