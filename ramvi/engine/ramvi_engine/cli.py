"""python -m ramvi_engine <команда> --model Qwen3-8B [--draft Qwen3-0.6B] ...
  serve  — сервер, совместимый с Ollama/OpenAI (порт 11435);
  chat   — разговор в окне терминала;
  plan   — показать план памяти и выйти;
  capabilities — JSON-карта возможностей без загрузки модели."""
import argparse
import json
import os
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.model_location import is_gguf_model

warnings.filterwarnings("ignore", message=".*non-writable.*")
# B.1: справка и ошибки CLI — из общих каталогов RAMVI (ramvi/i18n, рядом с engine/ в комплекте).
I18N_DIR = Path(__file__).resolve().parents[2] / "i18n"


def translator(language: str | None = None) -> Callable[..., str]:
    """Текст ключа на языке RAMVI_LANGUAGE; нет каталога (движок отдельно) — сам ключ."""
    try:
        path = I18N_DIR / f"{language or os.environ.get('RAMVI_LANGUAGE', 'ru')}.json"
        texts = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        texts = {}

    def t(key: str, **params: object) -> str:
        text = str(texts.get(key, key))
        for name, value in params.items():
            text = text.replace("{" + name + "}", str(value))
        return text

    return t


def with_defaults(parser: argparse.ArgumentParser, t: Callable[..., str]) -> None:
    """B.1: в справке у параметра — значение по умолчанию (как ramvi_service.i18n.with_defaults);
    ключ доступа из RAMVI_API_KEY в справку не попадает."""
    for action in parser._actions:
        if action.dest != "api_key" and action.help and action.default not in (None, False) and isinstance(action.default, (str, int, float)):
            action.help = f"{action.help} ({t('cli.default', value=action.default)})"


def _planning_config(a: argparse.Namespace) -> ModelConfig:
    """Lightweight parsing; planning itself runs in the engine process."""
    import math
    from ramvi_engine.model_loader import ModelConfig
    from ramvi_engine.planning.automatic import PlanningOptions
    if a.reserve_gib is not None and (not math.isfinite(a.reserve_gib * 2**30) or a.reserve_gib < 0):
        raise ValueError("cli.engine.bad_reserve")
    if a.plan_mode == "manual" and a.resident is None:
        raise ValueError("cli.engine.manual_needs_resident")
    if a.plan_mode == "legacy" and (a.reserve_gib is not None or a.max_batch is not None or a.draft_policy != "auto" or a.prefill != "auto" or a.no_paging or a.load_attempts != 4):
        raise ValueError("cli.engine.legacy_flags")
    options = None if a.plan_mode == "legacy" else PlanningOptions(
        mode=a.plan_mode, reserve_bytes=None if a.reserve_gib is None else int(a.reserve_gib * 2**30),
        max_batch=a.max_batch, draft=a.draft_policy, prefill=a.prefill, allow_paging=not a.no_paging, max_load_attempts=a.load_attempts)
    return ModelConfig(a.model, a.draft, a.maxt, not a.no_cpu, a.resident, a.tokenizer, options)


def main() -> None:
    """Запустить serve, chat, plan или capabilities для RAMVI Engine."""
    from ramvi_engine.console import configure_output
    configure_output()
    t = translator()
    ap = argparse.ArgumentParser(prog="ramvi-engine", description=t("cli.engine.about"))
    ap.add_argument("cmd", choices=["serve", "chat", "plan", "capabilities"],
                    help=t("cli.engine.cmd"))
    ap.add_argument("--model", help=t("cli.engine.model"))
    ap.add_argument("--draft", default=None, help=t("cli.engine.draft"))
    ap.add_argument("--k", type=int, default=0, help=t("cli.engine.k"))
    ap.add_argument("--maxt", type=int, default=2048, help=t("cli.engine.maxt"))
    ap.add_argument("--no-cpu", action="store_true", help=t("cli.engine.no_cpu"))
    ap.add_argument("--resident", type=int, default=None, help=t("cli.engine.resident"))
    ap.add_argument("--tokenizer", default="Qwen3-8B", help=t("cli.engine.tokenizer"))
    ap.add_argument("--port", type=int, default=11435, help=t("cli.engine.port"))
    ap.add_argument("--host", default="127.0.0.1", help=t("cli.engine.host"))
    ap.add_argument("--api-key", default=os.environ.get("RAMVI_API_KEY"), help=t("cli.engine.api_key"))
    ap.add_argument("--force-greedy", action="store_true",
                    help=t("cli.engine.force_greedy"))
    ap.add_argument("--plan-mode", choices=["auto", "manual", "legacy"], default="auto",
                    help=t("cli.engine.plan_mode"))
    ap.add_argument("--reserve-gib", type=float, default=None, help=t("cli.engine.reserve_gib"))
    ap.add_argument("--max-batch", type=int, choices=[1, 2, 4, 8, 16], default=None, help=t("cli.engine.max_batch"))
    ap.add_argument("--draft-policy", choices=["auto", "on", "off"], default="auto", help=t("cli.engine.draft_policy"))
    ap.add_argument("--prefill", choices=["auto", "legacy", "blocked"], default="auto", help=t("cli.engine.prefill"))
    ap.add_argument("--no-paging", action="store_true", help=t("cli.engine.no_paging"))
    ap.add_argument("--load-attempts", type=int, choices=range(1, 17), default=4, help=t("cli.engine.load_attempts"))
    with_defaults(ap, t)
    a = ap.parse_args()
    if a.cmd == "capabilities":
        import json
        from ramvi_engine.capabilities import capability_catalog
        print(json.dumps(capability_catalog(), ensure_ascii=False, indent=2))
        return
    if not a.model:
        ap.error(t("cli.engine.need_model"))
    if a.maxt < 1 or not 0 <= a.k <= 15 or a.resident is not None and a.resident < 0:
        ap.error(t("cli.engine.bad_numbers"))
    try:
        config = _planning_config(a)
    except ValueError as exc:
        ap.error(t(str(exc)))
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
    if torch.cuda.is_available():  # 8.7: без ptxas ядра Triton собирает драйвер
        from ramvi_engine.kernels.driver_jit import enable

        if enable():
            print("ядра Triton собирает драйвер NVIDIA (ptxas нет)", file=sys.stderr, flush=True)
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
