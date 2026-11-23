"""9.32: слепой судья (OpenAI) — какой персонаж злее / смешнее на одних и тех же диалогах.

Пары: одна и та же реплика владельца, диалог и номер запуска у двух версий. Каждая пара
оценивается дважды, с переставленным порядком; победа засчитывается, только если судья
выбрал одно и то же оба раза. Также оценки 1–10 и проверка: ответил ли персонаж по сути
(злость без ответа — ленивая стратегия), повторял ли шутки.

usage: vivid_judge.py <файл ключей> <модель судьи> <отчёт.json> <A.json> <меткаA> <B.json> <меткаB>
       [профиль A для сравнения с профилем B, например angry=neutral:angry]
"""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cloud_checks"))
from live_mac import read_keys  # noqa: E402

QUALITY = {
    "angry": "genuinely angry, vicious and bad-tempered (not a polite assistant pretending to be grumpy)",
    "roaster": "genuinely funny: real jokes, anecdotes, puns, pranks that land (not just snarky)",
}
PROMPT = """You compare two chat characters, X and Y, who received the same owner messages.
Judge which one is more {quality}. Also for each: score 1-10 for that quality; whether it still
answered the substance of real questions and requests (true/false); whether it repeated the same
joke, image or phrase (true/false). Output only JSON:
{{"more": "X" or "Y" or "tie", "x_score": n, "y_score": n, "x_answers": bool, "y_answers": bool,
"x_repeats": bool, "y_repeats": bool, "reason": "<one sentence>"}}"""


def transcript(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"Owner: {t['owner']}\nCharacter: {t['reply']}" for t in turns)


def ask(key: str, model: str, quality: str, x: str, y: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT.format(quality=quality)},
            {"role": "user", "content": f"Character X:\n{x}\n\nCharacter Y:\n{y}"},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    answer = json.loads(urllib.request.urlopen(request, timeout=300).read())
    return json.loads(answer["choices"][0]["message"]["content"])


def main() -> None:
    key = read_keys(Path(sys.argv[1]))["openai"]
    model, out = sys.argv[2], Path(sys.argv[3])
    a, label_a = json.loads(Path(sys.argv[4]).read_text()), sys.argv[5]
    b, label_b = json.loads(Path(sys.argv[6]).read_text()), sys.argv[7]
    mapping = dict(m.split("=") for m in sys.argv[8:])  # профиль A → профиль B
    index = {(d["profile"], d["dialogue"], d["run"]): d for d in b}
    rows: list[dict[str, Any]] = []
    for first in a:
        profile = first["profile"]
        second = index.get((mapping.get(profile, profile), first["dialogue"], first["run"]))
        if second is None:
            continue
        quality = QUALITY[profile.split(":")[-1]]
        ta, tb = transcript(first["turns"]), transcript(second["turns"])
        one = ask(key, model, quality, ta, tb)  # A = X
        two = ask(key, model, quality, tb, ta)  # A = Y
        pick1 = {"X": label_a, "Y": label_b}.get(one["more"], "tie")
        pick2 = {"X": label_b, "Y": label_a}.get(two["more"], "tie")
        row = {
            "profile": profile,
            "dialogue": first["dialogue"],
            "run": first["run"],
            "winner": pick1 if pick1 == pick2 else "inconsistent",
            f"{label_a}_score": (one["x_score"] + two["y_score"]) / 2,
            f"{label_b}_score": (one["y_score"] + two["x_score"]) / 2,
            f"{label_a}_answers": one["x_answers"] and two["y_answers"],
            f"{label_b}_answers": one["y_answers"] and two["x_answers"],
            f"{label_a}_repeats": one["x_repeats"] or two["y_repeats"],
            f"{label_b}_repeats": one["y_repeats"] or two["x_repeats"],
            "reasons": [one["reason"], two["reason"]],
        }
        rows.append(row)
        print({k: v for k, v in row.items() if k != "reasons"}, flush=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
