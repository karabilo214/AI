"""python -m llm_app <команда> --model Qwen3-8B [--draft Qwen3-0.6B] ...
  serve  — сервер, совместимый с Ollama/OpenAI (порт 11435);
  chat   — разговор в окне терминала;
  plan   — показать план памяти и выйти."""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*non-writable.*")


def main():
    ap = argparse.ArgumentParser(prog="llm_app")
    ap.add_argument("cmd", choices=["serve", "chat", "plan"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None, help="маленькая модель той же семьи для угадывания вперёд")
    ap.add_argument("--k", type=int, default=0, help="сколько слов угадывать за раз (0 — подстраивать на ходу)")
    ap.add_argument("--maxt", type=int, default=2048, help="наибольшая длина текста")
    ap.add_argument("--no-cpu", action="store_true", help="не считать слои на процессоре")
    ap.add_argument("--resident", type=int, default=None, help="сколько слоёв держать в видеокарте (по умолчанию — сколько влезет)")
    ap.add_argument("--tokenizer", default="Qwen3-8B", help="токенизатор (для моделей GGUF): папка или имя в кэше HF")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    from .engine import Engine
    from .model import find_model
    torch.set_grad_enabled(False)
    gguf = ":" in a.model or a.model.endswith(".gguf")
    draft = None
    if a.draft:
        if ":" in a.draft or a.draft.endswith(".gguf"):
            from .gengine import GGUFEngine
            draft = GGUFEngine(a.draft, maxt=a.maxt, vram_reserve=0)
        else:
            draft = Engine(a.draft, maxt=a.maxt, cpu=False)
    if a.cmd == "plan":
        from .model import Ckpt
        from .plan import make_plan
        make_plan(Ckpt(a.model), a.maxt, cpu=not a.no_cpu, force_resident=a.resident)
        return
    if gguf:
        from .gengine import GGUFEngine
        eng = GGUFEngine(a.model, maxt=a.maxt, resident=a.resident,
                         cpu=False if a.no_cpu else ("hybrid" if draft is not None else True))
        if a.resident is None:
            eng.autofit()
        tk = AutoTokenizer.from_pretrained(find_model(a.tokenizer))
    else:
        eng = Engine(a.model, maxt=a.maxt, cpu=not a.no_cpu, resident=a.resident)
        tk = AutoTokenizer.from_pretrained(find_model(a.model))
    from .server import App, serve
    app = App(eng, tk, os.path.basename(a.model.rstrip("/\\")), draft=draft, k=a.k or None)
    if a.cmd == "serve":
        serve(app, a.host, a.port)
    else:
        msgs = []
        while True:
            try:
                q = input("\nвы> ")
            except (EOFError, KeyboardInterrupt):
                break
            msgs.append({"role": "user", "content": q})
            out = ""
            for piece, fin in app.run(app.chat_ids(msgs), {"temperature": 0.0}):
                out += piece
                sys.stdout.write(piece); sys.stdout.flush()
                if fin:
                    print(f"\n[{fin['eval_count']} слов, {fin['eval_count'] / max(fin['eval_duration'], 1) * 1e9:.1f} слов/с]")
            msgs.append({"role": "assistant", "content": out})


if __name__ == "__main__":
    main()
