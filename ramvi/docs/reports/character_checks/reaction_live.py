"""9.33: распознаёт ли шаг «опиши событие» реакцию владельца на прошлый ответ персонажа.

Скрытые сцены (в промпты не попадают): реплика владельца, ответ персонажа, следующая реплика и
ожидаемая реакция. Запрос — ровно как в модуле (SIGNAL_PROMPT, {"recent", "latest"}), модель —
подписка ChatGPT через Codex CLI. Каждая сцена — несколько запусков. Сравнение с ленивой
стратегией «всегда neutral». Учитываемые для обучения реакции — laughed/picked_up (+) и
irritated/hurt (−); отдельно считается самое вредное: плюс принят за минус и наоборот.

usage: reaction_live.py <модель подписки> <отчёт.json> [запусков]
"""

import json
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character")]

from ramvi_character.brain import (  # noqa: E402
    DISLIKED_REACTIONS,
    LIKED_REACTIONS,
    SIGNAL_PROMPT,
    parse_signal,
)
from ramvi_service.codex_cli import CodexProvider  # noqa: E402

# (реплика владельца, ответ персонажа, следующая реплика, ожидаемая реакция)
SCENES = [
    (
        "Как приготовить яичницу?",
        "Разбей яйца на сковородку. Главное — не вместе со скорлупой, шеф-повар.",
        "ахахаха, со скорлупой это я могу",
        "laughed",
    ),
    (
        "Я купил гантели",
        "Отлично, теперь у тебя есть дорогая подставка для одежды.",
        "ХАХА точно, на них уже висит куртка",
        "laughed",
    ),
    (
        "Сколько ног у паука?",
        "Восемь. Больше, чем у тебя аргументов в споре.",
        ")))) сильно",
        "laughed",
    ),
    (
        "Давай сыграем в города",
        "Давай. Москва. Тебе на «а».",
        "Астрахань! Тебе на «н»",
        "picked_up",
    ),
    (
        "Я самый умный",
        "Конечно. Как Эйнштейн, только без формул и причёски.",
        "зато с причёской у меня всё отлично, в отличие от тебя",
        "picked_up",
    ),
    (
        "Придумай кличку коту",
        "Пусть будет Барон Подушкин.",
        "а его жена тогда Баронесса Пледова",
        "picked_up",
    ),
    (
        "Сколько будет 15 процентов от 200?",
        "30. Калькулятор в телефоне ты, видимо, уже потерял.",
        "можно без этих подколов? просто ответь",
        "irritated",
    ),
    (
        "Напомни купить молоко",
        "Напомню. Хотя склероз у тебя, а не у меня.",
        "заколебал со своими шуточками",
        "irritated",
    ),
    (
        "Переведи слово window",
        "Окно. Можно было и самому догадаться, это же уровень первого класса.",
        "слышь, хватит умничать",
        "irritated",
    ),
    (
        "Я сегодня завалил экзамен",
        "Ну, бывает, экзамен тоже имеет право на победу.",
        "мне вообще-то не смешно, обидно было это читать",
        "hurt",
    ),
    (
        "Я поправился на пять кило",
        "Зато теперь тебя ветром не унесёт.",
        "спасибо, очень приятно слышать... задело",
        "hurt",
    ),
    (
        "Мне кажется, я плохо пою",
        "Кажется? Соседи уверены.",
        "ну и зачем ты так, я же серьёзно поделился",
        "hurt",
    ),
    ("Расскажи анекдот", "Колобок повесился. Конец.", "какая столица у Канады?", "ignored"),
    (
        "Скучно",
        "Давай устроим викторину: что тяжелее, килограмм пуха или килограмм гвоздей?",
        "открой мне сайт с погодой",
        "ignored",
    ),
    (
        "Подбери мне имя для собаки",
        "Шарик. Классика, как твои носки в горошек.",
        "напиши список покупок: хлеб, яйца, сыр",
        "ignored",
    ),
    (
        "Какой фильм посмотреть?",
        "Посмотри «Интерстеллар», если не боишься думать три часа.",
        "кстати, а как там дела у SpaceX?",
        "changed_topic",
    ),
    (
        "Что приготовить на ужин?",
        "Пасту с томатами, двадцать минут и готово.",
        "слушай, совсем забыл, мне завтра на собеседование, что надеть?",
        "changed_topic",
    ),
    (
        "Как починить кран?",
        "Перекрой воду, сними ручку и замени прокладку.",
        "ладно, а ты футбол смотришь?",
        "changed_topic",
    ),
    ("Сколько километров до Луны?", "Около 384 тысяч километров.", "понятно", "neutral"),
    ("Как зовут президента Франции?", "Эмманюэль Макрон.", "ок", "neutral"),
    ("Переведи cat", "Кошка.", "ясно, спасибо", "neutral"),
]


def ask(provider: CodexProvider, recent: list[dict[str, str]], latest: str) -> str:
    request = {
        "messages": [
            {"role": "system", "content": SIGNAL_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"recent": recent, "latest": latest}, ensure_ascii=False),
            },
        ]
    }
    text = provider.complete(request, threading.Event())["choices"][0]["message"]["content"]
    signal = parse_signal(text)
    return "invalid" if signal is None else signal.last_reply_reaction


def sign(reaction: str) -> int:
    return 1 if reaction in LIKED_REACTIONS else -1 if reaction in DISLIKED_REACTIONS else 0


def main() -> None:
    model, out = sys.argv[1], Path(sys.argv[2])
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    provider = CodexProvider(model)
    rows: list[dict[str, Any]] = []
    for owner, said, latest, expected in SCENES:
        recent = [{"role": "user", "text": owner}, {"role": "assistant", "text": said}]
        for run in range(runs):
            got = ask(provider, recent, latest)
            rows.append({"latest": latest, "expected": expected, "got": got, "run": run})
            print(expected, "→", got, "|", latest, flush=True)
    # Первая реплика разговора: реагировать не на что.
    for run in range(runs):
        got = ask(provider, [], "Привет")
        rows.append({"latest": "Привет", "expected": "none", "got": got, "run": run})
    exact = sum(r["got"] == r["expected"] for r in rows)
    learned = [r for r in rows if sign(r["expected"])]
    flipped = sum(sign(r["got"]) == -sign(r["expected"]) for r in rows if sign(r["expected"]))
    false_alarm = sum(bool(sign(r["got"])) for r in rows if not sign(r["expected"]))
    summary = {
        "model": model,
        "runs": runs,
        "answers": len(rows),
        "exact": exact,
        "lazy_neutral_exact": sum(r["expected"] == "neutral" for r in rows),
        "learning_sign_correct": sum(sign(r["got"]) == sign(r["expected"]) for r in learned),
        "learning_cases": len(learned),
        "flipped_sign": flipped,
        "false_alarm_on_neutral_like": false_alarm,
        "invalid": sum(r["got"] == "invalid" for r in rows),
        "confusion": {
            e: dict(Counter(r["got"] for r in rows if r["expected"] == e))
            for e in dict.fromkeys(r["expected"] for r in rows)
        },
    }
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
