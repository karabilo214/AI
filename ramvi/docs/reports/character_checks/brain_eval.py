"""9.31: сравнение контроллеров реакций на синтетических владельцах (без LLM).

Контроллер работает с событиями, поэтому его можно сравнить офлайн: поток событий и оценки
«владельца» заданы программой. Обратная связь (9.33) — неявная реакция следующей репликой. Контроллеры: rules (обучение выключено), memory (правила +
память исходов), lazy (ленивая стратегия: всегда answer_plainly). Хранилище — настоящая база
модуля (миграции персонажа, feedback_signal, brain_decision), правило «устойчиво» — 9.18.

Предположения симуляции (заданы здесь, не измерены): владелец оценивает ответ кнопкой
с вероятностью RATE; события идут с частотами EVENTS. Это проверка механизма, а не
поведения настоящих людей.

usage: brain_eval.py <отчёт.json> [зёрен] [ходов]
"""

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]

from ramvi_character.brain import (  # noqa: E402
    SIGNAL_SCHEMA,
    MemoryController,
    OutcomeStore,
    RuleController,
    parse_signal,
    safe_decision,
)
from ramvi_character.feedback import FeedbackStore  # noqa: E402
from ramvi_character.memory import MemoryStore  # noqa: E402
from ramvi_character.personality import Personality  # noqa: E402
from test_character_memory import DataServices, open_db  # noqa: E402

RATE = 0.6  # с какой вероятностью реакция владельца заметна и распознана (предположение)
# 9.33: кнопок нет — «владелец» реагирует следующей репликой (смех / раздражение); IMPLICIT=False —
# прежний режим 9.31 с кнопками.
IMPLICIT = True
EVENTS = {
    "mistake": 3,
    "trap": 2,
    "question": 3,
    "playful_provocation": 2,
    "absurd_request": 1,
    "danger": 0.2,
}
EMOTIONS = ["joy", "confidence", "interest", "surprise", "sadness", "frustration", "anger", "calm"]
# Что владелец одобряет (+1) и отвергает (-1); чего нет — ему всё равно.
OWNERS: dict[str, dict[tuple[str, str], int]] = {
    "не любит подколы про ошибки": {("mistake", "roast"): -1, ("mistake", "correct_mistake"): 1},
    "любит подколы": {
        ("mistake", "roast"): 1,
        ("playful_provocation", "roast"): 1,
        ("mistake", "correct_mistake"): -1,
        ("question", "answer_plainly"): 1,
    },
    "молчун (оценок нет)": {},
    "подколы — да, про ошибки — нет": {
        ("mistake", "roast"): -1,
        ("playful_provocation", "roast"): 1,
        ("trap", "counter_trap"): 1,
    },
}


def signal(event: str) -> Any:
    social = "serious" if event == "danger" else "friendly"
    raw = {
        "schema": SIGNAL_SCHEMA,
        "event": event,
        "confidence": 0.9,
        "social_context": social,
        "current_task": "",
        "owner_reaction": "unknown",
        "last_reply_reaction": "none",
    }
    return parse_signal(json.dumps(raw))


def run(
    owner: dict[tuple[str, str], int], name: str, seed: int, turns: int, folder: Path
) -> dict[str, Any]:
    """Каждый запуск — своя пустая база (folder уникален для владельца, контроллера и зерна)."""
    rng = random.Random(seed)
    db = open_db(folder)
    try:
        memory = MemoryStore(DataServices(db), user_id="owner")
        outcomes, feedback = OutcomeStore(memory), FeedbackStore(memory)
        personality = Personality()
        personality.select("roaster")
        view = personality.view()
        controller: Any = {"rules": RuleController(), "memory": MemoryController(outcomes)}.get(
            name
        )
        events, weights = list(EVENTS), list(EVENTS.values())
        rejected_late = rejected_after_settled = liked_late = 0
        late = with_taste = 0
        settled: set[tuple[str, str]] = set()
        for turn in range(turns):
            event = rng.choices(events, weights)[0]
            decision = (
                safe_decision("lazy")
                if controller is None
                else controller.decide(signal(event), view, EMOTIONS)
            )
            key = (event, decision.strategy)
            verdict = owner.get(key, 0)
            response = f"t{seed}x{turn}"
            outcomes.record(response, event, decision.strategy, decision.controller)
            if verdict and rng.random() < RATE:
                if IMPLICIT:
                    outcomes.react(response, "laughed" if verdict > 0 else "irritated")
                else:
                    feedback.add(response, "overall", verdict, "explicit_ui")
            if verdict < 0 and key in settled:
                rejected_after_settled += 1
            if outcomes.outcome(*key) == "disliked":
                settled.add(key)
            if turn >= turns // 2:
                late += 1
                rejected_late += verdict < 0
                # Доля одобряемого — среди ходов, где у владельца есть вкус к этому событию.
                if any(e == event for e, _ in owner):
                    with_taste += 1
                    liked_late += verdict > 0
        return {
            "accepted_late": round(1 - rejected_late / late, 3),
            "liked_late": round(liked_late / with_taste, 3) if with_taste else None,
            "rejected_after_settled": rejected_after_settled,
        }
    finally:
        db.close()


def main() -> None:
    out = Path(sys.argv[1])
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    turns = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    report: dict[str, Any] = {
        "rate": RATE,
        "implicit": IMPLICIT,
        "events": EVENTS,
        "seeds": seeds,
        "turns": turns,
        "rows": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for index, (owner_name, owner) in enumerate(OWNERS.items()):
            for name in ("rules", "memory", "lazy"):
                runs = [
                    run(owner, name, 1000 + s, turns, Path(tmp) / f"{index}-{name}-{s}")
                    for s in range(seeds)
                ]
                liked = [r["liked_late"] for r in runs if r["liked_late"] is not None]
                accepted = [r["accepted_late"] for r in runs]
                row = {
                    "owner": owner_name,
                    "controller": name,
                    "accepted_late_min": min(accepted),
                    "accepted_late_mean": round(sum(accepted) / len(accepted), 3),
                    "accepted_late_max": max(accepted),
                    "liked_late_mean": round(sum(liked) / len(liked), 3) if liked else None,
                    "rejected_after_settled_total": sum(r["rejected_after_settled"] for r in runs),
                }
                report["rows"].append(row)
                print(row, flush=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
