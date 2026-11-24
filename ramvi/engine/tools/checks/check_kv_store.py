"""B.7: возврат в длинный чат после другого чата — время до первого слова и тот же ответ.

Сценарий на одном загруженном движке, жадный выбор, без угадывателя:
  same   — чат A, сразу продолжение A (как раньше: общее начало текущего чата);
  off    — A, чат B, возврат в A без хранилища (вся история A читается заново);
  ram    — то же с кэшем неактивных чатов в RAM;
  disk   — то же, RAM 0, только диск.
Ответ возврата сравнивается с «same» (то же продолжение без переключения).

usage: python -m tools.checks.check_kv_store --model <GGUF> --tokenizer <папка> --out <json>
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any


def turns(topic: str, count: int) -> list[dict[str, str]]:
    messages = []
    for i in range(count):
        messages.append(
            {"role": "user", "content": f"Расскажи подробно про {topic}, часть {i + 1}."}
        )
        messages.append(
            {
                "role": "assistant",
                "content": " ".join(
                    f"Пункт {i + 1}.{j}: {topic} — важная тема, о ней стоит знать многое."
                    for j in range(12)
                ),
            }
        )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--new", type=int, default=24, help="слов ответа")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from ramvi_engine.generation.kv_store import KvStore
    from ramvi_engine.model_loader import ModelConfig, create_session

    session = create_session(ModelConfig(args.model, maxt=args.context, tokenizer=args.tokenizer))
    session.load()
    tok = session.tokenizer
    engine = session._get_bundle().engine

    def ids(messages: list[dict[str, str]]) -> list[int]:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return list(tok.encode(text, add_special_tokens=False))

    history_a = turns("устройство видеокарты", 14)
    history_b = turns("выращивание томатов", 10)
    a = ids([*history_a, {"role": "user", "content": "Коротко: что главное?"}])
    a2_messages = [*history_a, {"role": "user", "content": "А сколько в ней памяти обычно?"}]
    a2 = ids(a2_messages)
    b = ids([*history_b, {"role": "user", "content": "Коротко: что главное?"}])

    def ask(prompt: list[int]) -> tuple[list[int], float]:
        stats: dict[str, Any] = {}
        out = list(engine.generate(prompt, args.new, stats=stats))
        return out, round(float(stats["first_token_ms"]), 1)

    report: dict[str, Any] = {
        "model": Path(args.model).name,
        "context": args.context,
        "tokens": {"a": len(a), "a_return": len(a2), "b": len(b)},
        "plan": list(engine.place),
        "runs": {},
    }
    folder = Path(tempfile.mkdtemp(prefix="kv-check-"))
    modes = {
        "same": None,
        "off": KvStore(0, 0, None),
        "ram": KvStore(4 * 2**30, 0, None),
        "disk": KvStore(0, 16 * 2**30, folder),
    }
    for name, store in modes.items():
        engine.kv_store = store or KvStore(0, 0, None)
        engine.ctx = []
        first, _ = ask(a)
        if name != "same":
            ask(b)
        started = time.perf_counter()
        answer, ttft = ask(a2)
        row = {
            "first_token_ms": ttft,
            "seconds": round(time.perf_counter() - started, 2),
            "text": tok.decode(answer, skip_special_tokens=True),
            "tokens": answer,
        }
        if store is not None:
            row["saved"], row["restored"] = store.saved, store.restored
            row["ram_mb"] = round(store.used("ram") / 2**20, 1)
            row["disk_mb"] = round(store.used("disk") / 2**20, 1)
            store.close()
        report["runs"][name] = row
        print(name, ttft, row["text"][:80], flush=True)
    same = report["runs"]["same"]["tokens"]
    report["identical_to_same"] = {
        m: report["runs"][m]["tokens"] == same for m in ("off", "ram", "disk")
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
