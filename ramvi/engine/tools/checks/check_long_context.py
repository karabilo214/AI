"""B.8 (1): длинный контекст на малой видеопамяти — что даёт план и сколько стоит контекст.

Для каждого размера контекста: план автоматического размещения (слои в видеокарте, по шине,
на процессоре, байты кэша k, v), затем загрузка и настоящий запрос на ~90% контекста:
время чтения запроса и скорость генерации 32 слов. Жадный выбор, без угадывателя.

usage: python -m tools.checks.check_long_context --model <GGUF> --tokenizer <папка>
       --contexts 4096,8192,16384,32768 --out <json>
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--fill", type=float, default=0.9)
    parser.add_argument("--new", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from ramvi_engine.model_loader import ModelConfig, create_session, preview_plan

    report: dict[str, Any] = {"model": args.model, "rows": []}
    filler = "Абзац {i}: в журнале склада записано, что полки проверены и всё на месте. "
    for context in (int(c) for c in args.contexts.split(",")):
        config = ModelConfig(args.model, maxt=context, tokenizer=args.tokenizer)
        row: dict[str, Any] = {"context": context}
        try:
            row["plan"] = preview_plan(config)
        except Exception as error:  # план не сложился — записываем причину
            row["plan_error"] = repr(error)
        session = create_session(config)
        try:
            started = time.perf_counter()
            session.load()
            row["load_s"] = round(time.perf_counter() - started, 1)
            engine = session._get_bundle().engine
            row["place"] = {k: list(engine.place).count(k) for k in ("gpu", "stream", "cpu", "hyb")}
            tok = session.tokenizer
            target = int(context * args.fill) - args.new
            text, i = "", 0
            while len(tok.encode(text)) < target:
                text += "".join(filler.format(i=i + j) for j in range(50))
                i += 50
            ids = tok.encode(
                tok.apply_chat_template(
                    [{"role": "user", "content": text + "\nСколько абзацев в журнале?"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                add_special_tokens=False,
            )[-target:]
            engine.ctx = []
            stats: dict[str, Any] = {}
            out = list(engine.generate(ids, args.new, stats=stats))
            row.update(
                prompt_tokens=len(ids),
                prompt_s=round(stats["prompt_ms"] / 1000, 1),
                prompt_tok_s=round(len(ids) / max(stats["prompt_ms"] / 1000, 1e-9), 1),
                gen_tokens=len(out),
                gen_tok_s=round(
                    max(stats.get("gen_n", len(out)) - 1, 1)
                    / max(stats.get("gen_ms", 0) / 1000, 1e-9),
                    2,
                ),
                vram_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            )
        except Exception as error:
            row["error"] = repr(error)[:400]
        finally:
            session.close()  # выгрузка модели и кэшей движка
            del session
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        report["rows"].append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "plan"}, ensure_ascii=False),
            flush=True,
        )
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
