"""Проверка эталона Reference против HF transformers fp32 на Qwen3-0.6B: самые вероятные слова на 2 текстах."""
import torch
from transformers import AutoModelForCausalLM

from big_engine import Reference, snap
from core import wiki_tokens

test = wiki_tokens("test").tolist()
ref = Reference("Qwen3-0.6B")
hf = AutoModelForCausalLM.from_pretrained(snap("Qwen3-0.6B"), dtype=torch.float32).cuda().eval()
for j in range(2):
    t = test[5000 + j * 15000: 5000 + j * 15000 + 192]
    a, _ = ref.argmax(t)
    with torch.no_grad():
        b = hf(torch.tensor([t], device="cuda")).logits[0].argmax(-1).tolist()
    print(f"текст {j}: совпало {sum(x == y for x, y in zip(a, b))}/{len(t)}")
