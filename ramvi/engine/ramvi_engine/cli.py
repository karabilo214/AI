"""python -m ramvi_engine <команда> --model Qwen3-8B [--draft Qwen3-0.6B] ...
  serve  — сервер, совместимый с Ollama/OpenAI (порт 11435);
  chat   — разговор в окне терминала;
  plan   — показать план памяти и выйти;
  capabilities — JSON-карта возможностей без загрузки модели."""
import argparse
import os
import sys
import warnings
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.model_location import is_gguf_model

warnings.filterwarnings("ignore", message=".*non-writable.*")


def _planning_config(a: argparse.Namespace) -> ModelConfig:
    """Lightweight parsing; planning itself runs in the engine process."""
    import math
    from ramvi_engine.model_loader import ModelConfig
    from ramvi_engine.planning.automatic import PlanningOptions
    if a.reserve_gib is not None and (not math.isfinite(a.reserve_gib * 2**30) or a.reserve_gib < 0):
        raise ValueError("reserve-gib должен быть конечным неотрицательным числом")
    if a.plan_mode == "manual" and a.resident is None:
        raise ValueError("manual требует --resident")
    if a.plan_mode == "legacy" and (a.reserve_gib is not None or a.max_batch is not None or a.draft_policy != "auto" or a.prefill != "auto" or a.no_paging or a.load_attempts != 4):
        raise ValueError("planning flags не применяются в legacy режиме")
    options = None if a.plan_mode == "legacy" else PlanningOptions(
        mode=a.plan_mode, reserve_bytes=None if a.reserve_gib is None else int(a.reserve_gib * 2**30),
        max_batch=a.max_batch, draft=a.draft_policy, prefill=a.prefill, allow_paging=not a.no_paging, max_load_attempts=a.load_attempts)
    return ModelConfig(a.model, a.draft, a.maxt, not a.no_cpu, a.resident, a.tokenizer, options)


def main() -> None:
    """Запустить serve, chat, plan или capabilities для RAMVI Engine."""
    from ramvi_engine.console import configure_output
    configure_output()
    ap = argparse.ArgumentParser(prog="ramvi-engine")
    ap.add_argument("cmd", choices=["serve", "chat", "plan", "capabilities"],
                    help="serve — API-сервер, chat — диалог в консоли, plan — план размещения без загрузки, capabilities — возможности движка")
    ap.add_argument("--model", help="модель: папка HF, файл GGUF или имя модели Ollama")
    ap.add_argument("--draft", default=None, help="маленькая модель той же семьи для угадывания вперёд")
    ap.add_argument("--k", type=int, default=0, help="сколько слов угадывать за раз (0 — подстраивать на ходу)")
    ap.add_argument("--maxt", type=int, default=2048, help="наибольшая длина текста")
    ap.add_argument("--no-cpu", action="store_true", help="не считать слои на процессоре")
    ap.add_argument("--resident", type=int, default=None, help="сколько слоёв держать в видеокарте (по умолчанию — сколько влезет)")
    ap.add_argument("--tokenizer", default="Qwen3-8B", help="токенизатор (для моделей GGUF): папка или имя в кэше HF")
    ap.add_argument("--port", type=int, default=11435, help="порт API-сервера")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 — доступ по сети")
    ap.add_argument("--api-key", default=os.environ.get("RAMVI_API_KEY"), help="ключ доступа (Authorization: Bearer ... или x-api-key)")
    ap.add_argument("--force-greedy", action="store_true",
                    help="всегда жадный выбор (агенты часто шлют температуру > 0; угадывание точное только при жадном)")
    ap.add_argument("--plan-mode", choices=["auto", "manual", "legacy"], default="auto",
                    help="план размещения: auto — сам по памяти, manual — по --resident, legacy — прежний план этапа 1")
    ap.add_argument("--reserve-gib", type=float, default=None, help="запас видеопамяти в ГиБ, который план не занимает")
    ap.add_argument("--max-batch", type=int, choices=[1, 2, 4, 8, 16], default=None, help="наибольший размер пачки при проверке угаданных слов")
    ap.add_argument("--draft-policy", choices=["auto", "on", "off"], default="auto", help="угадывание маленькой моделью: auto — по замеру, on — всегда, off — никогда")
    ap.add_argument("--prefill", choices=["auto", "legacy", "blocked"], default="auto", help="способ обработки запроса до первого слова: auto, legacy (прежний) или blocked (блоками)")
    ap.add_argument("--no-paging", action="store_true", help="не подгружать слои по шине во время генерации")
    ap.add_argument("--load-attempts", type=int, choices=range(1, 17), default=4, help="сколько попыток загрузки с уменьшением плана при нехватке памяти")
    a = ap.parse_args()
    if a.cmd == "capabilities":
        import json
        from ramvi_engine.capabilities import capability_catalog
        print(json.dumps(capability_catalog(), ensure_ascii=False, indent=2))
        return
    if not a.model:
        ap.error("--model обязателен для serve, chat и plan")
    if a.maxt < 1 or not 0 <= a.k <= 15 or a.resident is not None and a.resident < 0:
        ap.error("maxt > 0, k от 0 до 15, resident >= 0")
    try:
        config = _planning_config(a)
    except ValueError as exc:
        ap.error(str(exc))
    if a.cmd == "plan" and config.planning is not None:
        import json
        from ramvi_engine.model_loader import preview_plan
        print(json.dumps(preview_plan(config), ensure_ascii=False, indent=2))
        return
    if a.cmd == "serve":
        # 3.3: следить за владельцем ещё до долгой загрузки весов.
        from ramvi_engine.processes import ParentWatch
        ParentWatch.from_env()
    import torch

    torch.set_grad_enabled(False)
    gguf = is_gguf_model(a.model)
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
    from ramvi_engine.model_loader import ModelConfig, create_session
    eng = create_session(config)
    try:
        eng.load()
        tk = eng.tokenizer
        draft = eng.draft_handle if a.draft else None
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
    finally:
        eng.close()


if __name__ == "__main__":
    main()
