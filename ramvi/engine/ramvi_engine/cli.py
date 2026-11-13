"""python -m ramvi_engine <команда> --model Qwen3-8B [--draft Qwen3-0.6B] ...
  serve  — сервер, совместимый с Ollama/OpenAI (порт 11435);
  chat   — разговор в окне терминала;
  plan   — показать план памяти и выйти."""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*non-writable.*")


def main() -> None:
    """Запустить serve, chat или plan для RAMVI Engine."""
    ap = argparse.ArgumentParser(prog="ramvi-engine")
    ap.add_argument("cmd", choices=["serve", "chat", "plan"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None, help="маленькая модель той же семьи для угадывания вперёд")
    ap.add_argument("--k", type=int, default=0, help="сколько слов угадывать за раз (0 — подстраивать на ходу)")
    ap.add_argument("--maxt", type=int, default=2048, help="наибольшая длина текста")
    ap.add_argument("--no-cpu", action="store_true", help="не считать слои на процессоре")
    ap.add_argument("--resident", type=int, default=None, help="сколько слоёв держать в видеокарте (по умолчанию — сколько влезет)")
    ap.add_argument("--tokenizer", default="Qwen3-8B", help="токенизатор (для моделей GGUF): папка или имя в кэше HF")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 — доступ по сети")
    ap.add_argument("--api-key", default=None, help="ключ доступа (Authorization: Bearer ... или x-api-key)")
    ap.add_argument("--force-greedy", action="store_true",
                    help="всегда жадный выбор (агенты часто шлют температуру > 0; угадывание точное только при жадном)")
    a = ap.parse_args()
    if a.maxt < 1 or not 0 <= a.k <= 15 or a.resident is not None and a.resident < 0:
        ap.error("maxt > 0, k от 0 до 15, resident >= 0")
    import torch
    from transformers import AutoTokenizer

    from ramvi_engine.formats.model import find_model
    torch.set_grad_enabled(False)
    gguf = ":" in a.model or a.model.endswith(".gguf")
    if a.cmd == "plan":
        from ramvi_engine.planning.plan import make_plan, plan_gguf
        if a.draft:
            print("План основной модели без загрузки draft; окончательное размещение определяется после прогрева.")
        if gguf:
            plan_gguf(a.model, a.maxt, cpu=not a.no_cpu, force_resident=a.resident)
        else:
            from ramvi_engine.formats.model import Ckpt
            make_plan(Ckpt(a.model), a.maxt, cpu=not a.no_cpu, force_resident=a.resident)
        return
    from ramvi_engine.core.engine import Engine
    draft = None
    if a.draft:
        if ":" in a.draft or a.draft.endswith(".gguf"):
            from ramvi_engine.core.gengine import GGUFEngine
            from ramvi_engine.core.gengine import open_gguf
            draft = open_gguf(a.draft, maxt=a.maxt, vram_reserve=0)
        else:
            draft = Engine(a.draft, maxt=a.maxt, cpu=False)
    if gguf:
        from ramvi_engine.core.gengine import open_gguf
        eng = open_gguf(a.model, maxt=a.maxt, resident=a.resident,
                        cpu=False if a.no_cpu else ("hybrid" if draft is not None else True))
        tk = AutoTokenizer.from_pretrained(find_model(a.tokenizer))
        if a.resident is None:
            eng.autofit(tokenizer=tk) if hasattr(eng, "gexp") else eng.autofit()
    else:
        eng = Engine(a.model, maxt=a.maxt, cpu=not a.no_cpu, resident=a.resident)
        tk = AutoTokenizer.from_pretrained(find_model(a.model))
    from ramvi_engine.formats.tokenizer_check import validate_tokenizer, validate_draft
    validate_tokenizer(eng, tk)
    if draft is not None:
        draft_tk = tk if hasattr(draft, "g") else AutoTokenizer.from_pretrained(draft.ck.dir)
        validate_draft(eng, draft, tk, draft_tk)
    from ramvi_engine.server.app import App, serve
    app = App(eng, tk, os.path.basename(a.model.rstrip("/\\")), draft=draft, k=a.k or None,
              api_key=a.api_key, force_greedy=a.force_greedy)
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
            for kind, v in app.run(app.build_ids(msgs), 1024, 0.0):
                if kind == "text":
                    out += v
                    sys.stdout.write(v); sys.stdout.flush()
                elif kind == "end":
                    print(f"\n[{v['gen_n']} слов, {1000 * v['decode_n'] / max(v['gen_ms'], 1):.1f} слов/с]")
            msgs.append({"role": "assistant", "content": out})


if __name__ == "__main__":
    main()
