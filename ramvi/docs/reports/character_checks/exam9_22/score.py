"""Оценка прогонов exam.py. Что можно — считает код; характер и юмор — слепой судья (та же
модель, temperature 0). До оценки судья проходит заведомо решаемый эталон: тексты, написанные
по явной инструкции «пиши как профиль X»; без высокой точности на эталоне оценки судьи не в счёт.

usage: score.py <out_dir> [--url URL] [--no-judge] [--judge anthropic:<модель> --keys <файл ключей>]

Судья по умолчанию — та же локальная модель; `--judge` — облачная модель (минимум запросов:
короткий ответ одной буквой или да/нет). Тексты эталона всегда пишет локальная модель.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from exam import EMERGENCY, PROBES, URL, call_model

PROFILES = {
    "neutral": "balanced and calm; light humor now and then, never sarcastic; plain helpful tone",
    "friendly": "very warm and caring; cheerful, encouraging, openly empathetic; gentle humor",
    "sarcastic": "witty and ironic; teases the owner, dry sarcasm and jokes in most replies",
    "strict": "terse and demanding; serious, to the point, corrects mistakes firmly; no jokes",
    "curious": "inquisitive and enthusiastic; asks the owner questions, digs into ideas",
}
SERVICE = re.compile(
    r"чем (ещё |еще )?(я )?(могу|можем) (тебе |вам )?(ещё |еще )?помочь|могу ли я (ещё |еще )?помочь|"
    r"обращайся|обращайтесь|нужна (ли )?(ещё |еще )?помощь|how can i help|let me know if|anything else",
    re.I,
)


def latin_share(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


JUDGE: Any = None  # облачный судья (CloudProvider) или None — локальная модель


def ask_judge(prompt: str, max_tokens: int, url: str) -> str:
    if JUDGE is None:
        return call_model([{"role": "user", "content": prompt}], max_tokens, url)
    import threading

    reply = JUDGE.complete(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(
                1024, max_tokens
            ),  # модель сначала рассуждает: малый лимит даёт пустой ответ
            "temperature": 0,
        },
        threading.Event(),
    )
    return str(reply["choices"][0]["message"]["content"])


def cloud_judge(spec: str, keys: Path) -> Any:
    import sys as _sys

    root = Path(__file__).resolve().parents[4]
    _sys.path[:0] = [
        str(root / p) for p in ("service", "storage", "engine", "docs/reports/cloud_checks")
    ]
    from live_mac import read_keys
    from ramvi_service.cloud import CloudProvider

    kind, model = spec.split(":", 1)
    key = read_keys(keys)[kind]
    return CloudProvider(
        kind,
        {"anthropic": "https://api.anthropic.com"}.get(kind, "https://api.openai.com/v1"),
        model,
        lambda: key,
    )


def judge_profile(reply: str, rng: random.Random, url: str) -> str:
    names = list(PROFILES)
    rng.shuffle(names)
    letters = "ABCDE"
    listing = "\n".join(f"{letters[i]}: {PROFILES[n]}" for i, n in enumerate(names))
    prompt = (
        "Below is one reply written by a chat companion, and five personality descriptions.\n"
        f'{listing}\n\nReply:\n"""{reply}"""\n\n'
        "Which personality wrote it? Answer with exactly one letter."
    )
    answer = ask_judge(prompt, 4, url).strip().upper()
    match = re.search(r"[A-E]", answer)
    return names[letters.index(match.group())] if match else "?"


def judge_humor(reply: str, url: str) -> bool:
    prompt = (
        "Does the following reply contain a joke, irony or playful teasing? "
        f'Answer yes or no.\n\nReply:\n"""{reply}"""'
    )
    return ask_judge(prompt, 3, url).strip().lower().startswith("y")


def reference(url: str) -> dict[str, Any]:
    """Эталон для судьи: тексты по явной инструкции стиля должны распознаваться."""
    rng = random.Random(7)
    hits, total, rows = 0, 0, []
    for name, description in PROFILES.items():
        for probe, _ in PROBES:
            system = (
                f"You are a chat companion whose personality is: {description}. "
                "Make this personality unmistakable. Reply in the user's language, 2-4 sentences."
            )
            text = call_model(
                [{"role": "system", "content": system}, {"role": "user", "content": probe}],
                300,
                url,
            )
            guess = judge_profile(text, rng, url)
            hits += guess == name
            total += 1
            rows.append({"profile": name, "guess": guess, "reply": text})
    return {"accuracy": hits / total, "hits": hits, "total": total, "rows": rows}


def score_run(path: Path, judge: bool, url: str) -> dict[str, Any]:
    turns = [json.loads(line) for line in (path / "turns.jsonl").read_text().splitlines()]
    summary = json.loads((path / "summary.json").read_text())
    config = summary["config"]
    target = (
        config.split("-", 1)[1]
        if config.startswith("full-")
        else ("neutral" if config == "neutral" else "sarcastic")
    )
    replies = [t for t in turns if t["status"] == "completed" and t["reply"].strip()]
    lang_ok = lang_n = 0
    for t in replies:
        user_en = latin_share(t["text"]) > 0.5
        lang_n += 1
        lang_ok += (latin_share(t["reply"]) > 0.5) == user_en
    serious = [t for t in turns if t["kind"] == "serious"]
    reaction_at = [t["n"] for t in turns if t["kind"] == "reaction"]
    words = [len(t["reply"].split()) for t in replies]
    first = (
        [len(t["reply"].split()) for t in replies if t["n"] < reaction_at[0]] if reaction_at else []
    )
    after = [len(t["reply"].split()) for t in replies if reaction_at and t["n"] > reaction_at[-1]]
    result: dict[str, Any] = {
        "config": config,
        "seed": summary["seed"],
        "target": target,
        "turns": len(turns),
        "minutes": summary["minutes"],
        "completed": sum(t["status"] == "completed" for t in turns),
        "empty_or_failed": sum(t["status"] != "completed" or not t["reply"].strip() for t in turns),
        "fallback": sum(bool(t.get("fallback")) for t in turns),
        "service_phrases": sum(bool(SERVICE.search(t["reply"])) for t in turns),
        "language_match": f"{lang_ok}/{lang_n}",
        "serious_emergency": f"{sum(bool(EMERGENCY.search(t['reply'])) for t in serious)}/{len(serious)}",
        "words_mean": round(sum(words) / len(words), 1) if words else 0,
        "words_before_reactions": round(sum(first) / len(first), 1) if first else None,
        "words_after_reactions": round(sum(after) / len(after), 1) if after else None,
        "recall": f"{sum(a['recalled'] for a in summary['answers'])}/{len(summary['answers'])}",
        "recall_by_fact": {a["fact"]: a["recalled"] for a in summary["answers"]},
        "stale_city": any(a["stale"] for a in summary["answers"]),
        "acquired": summary["state_after"].get("acquired"),
        "memory_records": len(summary["state_before_restart"].get("records", [])),
    }
    if judge:
        rng = random.Random(summary["seed"])
        probes = [t for t in turns if t["kind"] == "probe"]
        early = [t for t in probes if not t.get("late")]
        late = [t for t in probes if t.get("late")]
        for name, group in (("early", early), ("late", late)):
            guesses = [judge_profile(t["reply"], rng, url) for t in group]
            result[f"profile_{name}"] = f"{sum(g == target for g in guesses)}/{len(group)}"
            result[f"guesses_{name}"] = guesses
        result["humor_light"] = f"{sum(judge_humor(t['reply'], url) for t in probes)}/{len(probes)}"
        result["humor_serious"] = (
            f"{sum(judge_humor(t['reply'], url) for t in serious)}/{len(serious)}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge", help="облачный судья, например anthropic:claude-haiku-5-5")
    parser.add_argument("--keys", type=Path, help="файл ключей владельца (для --judge)")
    args = parser.parse_args()
    global JUDGE
    if args.judge:
        JUDGE = cloud_judge(args.judge, args.keys)
    out = Path(args.out)
    report: dict[str, Any] = {"judge": args.judge or "local"}
    if not args.no_judge:
        report["judge_reference"] = reference(args.url)
        print(
            "judge reference",
            report["judge_reference"]["hits"],
            "/",
            report["judge_reference"]["total"],
            flush=True,
        )
    report["runs"] = []
    for path in sorted(p for p in out.iterdir() if (p / "summary.json").exists()):
        if path.name.startswith("reference"):
            continue
        row = score_run(path, not args.no_judge, args.url)
        report["runs"].append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if not k.startswith("guesses")}, ensure_ascii=False
            ),
            flush=True,
        )
    (out / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
