"""Заведомо решаемая проверка: своя реализация == transformers (логиты и пошаговая генерация с кэшем)."""
import torch
from transformers import AutoModelForCausalLM

from core import SNAP, Qwen, loss_of, wiki_tokens, windows

ids = windows(wiki_tokens("test"), 2, T=256)
hf = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    ref = hf(ids).logits
import sys; me = Qwen(sys.argv[1] if len(sys.argv) > 1 else "cpu", torch.float32); ids = ids.to(me.device); ref = ref.to(me.device)
mine = me.forward(ids)
print("макс. разница логитов:", (ref - mine).abs().max().item(), " (масштаб логитов", ref.abs().max().item(), ")")
print("совпадение топ-1:", (ref.argmax(-1) == mine.argmax(-1)).float().mean().item())
print("потеря HF / своя:", loss_of(ref, ids), loss_of(mine, ids))

# пошагово с кэшем: последние 16 токенов по одному
x = ids[:1]
cache = [None] * 28
me.forward(x[:, :240], cache=cache)
steps = torch.cat([me.forward(x[:, t:t + 1], cache=cache) for t in range(240, 256)], 1)
print("кэш: макс. разница с полным проходом:", (steps - mine[:1, 240:256]).abs().max().item())
