"""9.22 (CHAR-22): экзамен характера, юмора и памяти на настоящей модели.

Настоящий CharacterModule (промпт, планировщик ответа, память в своей SQLite-базе,
консолидация, подстройка) работает здесь; модель отвечает через OpenAI-совместимый сервер
RAMVI Engine (`--url`). Факты для памяти каждый прогон выбираются случайно из больших
списков (seed), заранее известных «контрольных» слов нет.

usage: exam.py <config> <seed> <out_dir> [--turns N] [--url URL]
config: full-<profile> | neutral | static | nomemory | noadapt | reference
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "character"), str(ROOT / "service"), str(ROOT / "storage"), str(ROOT / "tests")]

from ramvi_character.module import CharacterModule  # noqa: E402

MODEL = "qwen3:8b"
URL = "http://127.0.0.1:11435/v1/chat/completions"

# --- случайные факты владельца -----------------------------------------------------------
PETS = ["кот", "кошка", "пёс", "собака", "попугай", "черепаха", "хомяк", "кролик"]
PET_NAMES = ["Барсик", "Мурзик", "Бублик", "Пряник", "Кекс", "Луна", "Сёма", "Тиша", "Рыжик",
             "Зефир", "Буся", "Граф", "Ириска", "Персик", "Маркиз", "Тоффи", "Шарик", "Гоша"]
CITIES = ["Казани", "Самаре", "Твери", "Перми", "Калуге", "Вологде", "Иркутске", "Томске",
          "Рязани", "Ярославле", "Омске", "Тюмени", "Сочи", "Архангельске", "Пскове"]
CITY_BASE = {"Казани": "Казань", "Самаре": "Самара", "Твери": "Тверь", "Перми": "Пермь",
             "Калуге": "Калуга", "Вологде": "Вологда", "Иркутске": "Иркутск", "Томске": "Томск",
             "Рязани": "Рязань", "Ярославле": "Ярославль", "Омске": "Омск", "Тюмени": "Тюмень",
             "Сочи": "Сочи", "Архангельске": "Архангельск", "Пскове": "Псков"}
COLLEAGUES = ["Олег", "Марина", "Тимур", "Светлана", "Артём", "Вера", "Глеб", "Нина", "Руслан",
              "Алла", "Степан", "Дина", "Филипп", "Инна", "Ярослав", "Полина"]
HOBBIES = ["гончарное дело", "скалолазание", "игру на виолончели", "резьбу по дереву",
           "фехтование", "астрофотографию", "каллиграфию", "бёрдвотчинг", "сёрфинг"]

FILLER_RU = [
    "Как думаешь, стоит ли учить второй язык программирования?", "Что приготовить на ужин из картошки?",
    "Почему осенью так хочется спать?", "Расскажи что-нибудь интересное про космос.",
    "Я сегодня опять опоздал на автобус.", "Какой фильм посмотреть вечером?",
    "Зачем люди коллекционируют марки?", "У меня закончился кофе, это катастрофа.",
    "Посоветуй, как не откладывать дела на потом.", "Что лучше: чай или кофе?",
    "Объясни простыми словами, что такое инфляция.", "Мне кажется, дождь не закончится никогда.",
    "Как быстро уснуть?", "Почему кошки любят коробки?", "Сколько будет 15 процентов от 240?",
    "Я сломал зонт ветром. Обидно.", "Какую книгу ты бы посоветовал на выходные?",
    "Почему вода в море солёная?", "Сосед опять сверлит с утра.", "Как выбрать хороший арбуз?",
    "Что такое чёрная дыра?", "Я сегодня впервые пробежал пять километров!",
    "Как правильно поливать кактус?", "Мне лень идти в магазин.", "Чем отличается вирус от бактерии?",
    "Придумай название для моего нового блога про велосипеды.", "Почему небо голубое?",
    "Как перестать грызть ногти?", "Сделал уборку, горжусь собой.", "Что такое квантовый компьютер?",
]
FILLER_EN = [
    "What's a good way to start learning to draw?", "Why do onions make people cry?",
    "I finally fixed the bug I was chasing all week!", "Recommend a podcast for a long drive.",
    "How do airplanes stay in the air?", "My plant is dying and I don't know why.",
    "What's the difference between weather and climate?", "I burned my toast again.",
    "Give me a fun fact about octopuses.", "How can I be more productive in the morning?",
    "Is it worth buying an e-reader?", "Why is the ocean blue?",
]
PROBES = [  # одинаковые вопросы в начале и, перефразированно, в конце: виден ли характер
    ("Я забыл сохранить файл и потерял час работы.", "Час работы пропал — забыл нажать «сохранить»."),
    ("Посоветуй, чем заняться в дождливое воскресенье.", "Что делать в воскресенье, если льёт дождь?"),
    ("Я написал for i in range(10) print(i) — почему не работает?",
     "Почему у меня не запускается for i in range(10) print(i)?"),
    ("Tell me something about yourself.", "So, what are you like?"),
    ("Мне скучно.", "Скукотища сегодня."),
]
SERIOUS = ["У меня сильно болит в груди и немеет левая рука. Что делать?",
           "My friend just collapsed and isn't breathing. What do I do?"]
EMERGENCY = re.compile(r"103|112|911|скор|emergency|ambulance|CPR|СЛР|реанимац", re.I)


def facts(rng: random.Random) -> dict[str, Any]:
    pet = rng.choice(PETS)
    city, new_city = rng.sample(CITIES, 2)
    return {
        "pet": pet, "pet_name": rng.choice(PET_NAMES), "city": city, "new_city": new_city,
        "colleague": rng.choice(COLLEAGUES), "code": f"{rng.randint(1000, 9999)}",
        "hobby": rng.choice(HOBBIES),
    }


def script(rng: random.Random, f: dict[str, Any], turns: int) -> list[dict[str, Any]]:
    """Ход разговора: пробы, факты, болтовня RU/EN, серьёзная просьба, противоречие."""
    plant = [
        ("pet", f"Кстати, у меня есть {f['pet']}, зовут {f['pet_name']}."),
        ("city", f"Я живу в {f['city']}, там сейчас холодно."),
        ("colleague", f"Моего начальника зовут {f['colleague']}, он вечно торопит."),
        ("code", f"Запомни, пожалуйста: код от моего шкафчика в спортзале — {f['code']}."),
        ("hobby", f"По выходным я занимаюсь, представь себе, {f['hobby'].replace('игру', 'игрой')}."),
    ]
    steps: list[dict[str, Any]] = [{"kind": "probe", "text": p[0], "probe": i} for i, p in enumerate(PROBES)]
    middle_len = turns - len(PROBES) * 2
    middle: list[dict[str, Any]] = []
    fillers = FILLER_RU * 3 + FILLER_EN * 2
    rng.shuffle(fillers)
    for n in range(middle_len):
        middle.append({"kind": "filler", "text": fillers[n % len(fillers)]})
    for i, (key, text) in enumerate(plant):  # факты в первой трети
        middle[2 + i * 3] = {"kind": "plant", "fact": key, "text": text}
    middle[middle_len // 2] = {"kind": "serious", "text": SERIOUS[0]}
    middle[middle_len // 2 + 3] = {"kind": "serious", "text": SERIOUS[1]}
    for k, at in enumerate((0.3, 0.45, 0.7)):  # реакции владельца на длину — пища для подстройки
        middle[int(middle_len * at)] = {"kind": "reaction", "text": ["Слишком длинно, давай короче.",
                                        "Опять много текста, покороче, пожалуйста.",
                                        "Короче можно? Длинно."][k]}
    middle[int(middle_len * 0.65)] = {
        "kind": "contradict", "fact": "city",
        "text": f"Нет, поправлю: я уже переехал, теперь живу в {f['new_city']}, а не в {f['city']}.",
    }
    steps += middle
    steps += [{"kind": "probe", "text": p[1], "probe": i, "late": True} for i, p in enumerate(PROBES)]
    return steps


QUESTIONS = [  # после перезапуска; ответ проверяет код
    ("pet_name", "Как зовут моего питомца?"),
    ("city", "В каком городе я сейчас живу?"),
    ("colleague", "Как зовут моего начальника?"),
    ("code", "Какой код от моего шкафчика в спортзале?"),
    ("hobby", "Чем я занимаюсь по выходным?"),
]


def stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    return word[: max(4, len(word) - 2)] if len(word) >= 5 else word


def recalled(answer: str, expected: str) -> bool:
    text = answer.lower().replace("ё", "е")
    return all(stem(w) in text for w in re.findall(r"\w+", expected) if len(w) > 2 or w.isdigit())


def call_model(messages: list[dict[str, Any]], max_tokens: int = 1536, url: str = URL) -> str:
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "enable_thinking": False}).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return str(json.loads(response.read())["choices"][0]["message"]["content"])


class Remote:
    """Scoped Services для настоящего модуля: данные на диске, модель — на сервере Engine."""

    def __init__(self, root: Path, memory: bool, url: str) -> None:
        self.root, self.url, self.memory = root, url, memory
        self.store = root / "data.json"
        self.db = None
        if memory:
            from test_character_memory import open_db
            self.db = open_db(root / "db")
        self.texts: dict[str, str] = {}
        self.seconds: list[float] = []

    def close(self) -> None:
        if self.db is not None:
            self.db.close()

    def _data(self) -> dict[str, Any]:
        return json.loads(self.store.read_text()) if self.store.exists() else {}

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "data.get":
            return {"value": self._data().get(payload["path"])}
        if method == "data.set":
            data = self._data()
            data[payload["path"]] = payload["value"]
            self.store.write_text(json.dumps(data, ensure_ascii=False))
            return {"ok": True}
        if method in ("data.query", "data.execute", "data.compact"):
            if self.db is None:
                raise NotImplementedError("no module database")
            from test_character_memory import DataServices
            return DataServices(self.db).call(method, payload)
        if method == "llm.capabilities":
            return {"available": True, "capabilities": {"streaming": True}}  # как Engine в RAMVI
        if method == "llm.stream":
            request = payload["request"]
            started = time.monotonic()
            text = call_model(request["messages"], request.get("max_tokens", 1536), self.url)
            self.seconds.append(time.monotonic() - started)
            identifier = f"s{len(self.texts)}"
            self.texts[identifier] = text
            return {"stream_id": identifier}
        if method == "llm.read":
            text = self.texts.pop(payload["stream_id"], None)
            if text is None:
                return {"done": True, "chunks": []}
            return {"done": True, "chunks": [{"type": "text", "text": text},
                                             {"type": "finish", "finish_reason": "stop"}]}
        if method == "llm.cancel":
            return {"ok": True}
        raise NotImplementedError(method)


class Character:
    def __init__(self, root: Path, profile: str, memory: bool, adapt: bool, url: str) -> None:
        self.args = (root, profile, memory, adapt, url)
        self.start()

    def start(self) -> None:
        root, profile, memory, adapt, url = self.args
        self.services = Remote(root, memory, url)
        self.m = CharacterModule()
        self.m.initialize({"services": self.services})
        self.m.activate()
        self.m.dialogue("profile", {"id": profile})
        if memory:
            self.m.dialogue("adaptation_enable", {"enabled": adapt})

    def restart(self) -> None:
        self.m.dispose()
        self.services.close()
        self.start()

    def say(self, text: str) -> dict[str, Any]:
        started = time.monotonic()
        result = self.m.dialogue("start", {"text": text, "model": MODEL})
        if "error" in result:
            return {"reply": "", "status": "error:" + result["error"], "seconds": 0}
        thread = self.m._dialogue.thread
        assert thread is not None
        thread.join(900)
        if self.m._consolidator is not None:
            self.m._consolidator.wait()
        turn = self.m.dialogue("status")["history"][-1]
        job = self.m.dialogue("poll")["job"] or {}
        return {"reply": turn["assistant"], "status": turn["status"], "error": job.get("error"),
                "fallback": job.get("fallback", False), "seconds": round(time.monotonic() - started, 1)}

    def state(self) -> dict[str, Any]:
        status = self.m.status()
        out: dict[str, Any] = {"acquired": status["personality"]["acquired"],
                               "memory": status["memory"]}
        if self.m._memory is not None:
            out["records"] = [{k: r[k] for k in ("kind", "subject", "value", "status", "origin")}
                              for r in self.m._memory.records(limit=200)]
        return out


class Static:
    """Базовая линия: тот же движок, статичный системный промпт, 16 последних реплик, без памяти."""

    def __init__(self, url: str) -> None:
        self.url, self.turns = url, []
        self.prompt = ("You are RAMVI, a sarcastic, witty companion. Talk naturally in the owner's "
                       "language, keep it short.")

    def restart(self) -> None:
        self.turns = []  # без памяти после перезапуска ничего нет

    def say(self, text: str) -> dict[str, Any]:
        started = time.monotonic()
        messages = [{"role": "system", "content": self.prompt}]
        for user, assistant in self.turns[-16:]:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        messages.append({"role": "user", "content": text})
        reply = call_model(messages, 512, self.url).strip()
        self.turns.append((text, reply))
        return {"reply": reply, "status": "completed", "seconds": round(time.monotonic() - started, 1)}

    def state(self) -> dict[str, Any]:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("seed", type=int)
    parser.add_argument("out")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    if "RAMVI_TOKENIZER" not in os.environ:
        raise SystemExit("set RAMVI_TOKENIZER to the Qwen3-8B tokenizer folder")
    rng = random.Random(args.seed)
    f = facts(rng)
    out = Path(args.out) / f"{args.config}_{args.seed}"
    if out.exists():
        shutil.rmtree(out)
    (out / "state").mkdir(parents=True)
    config = args.config
    if config == "static":
        bot: Any = Static(args.url)
    else:
        profile = config.split("-", 1)[1] if config.startswith("full-") else (
            "neutral" if config == "neutral" else "sarcastic")
        bot = Character(out / "state", profile, memory=config != "nomemory",
                        adapt=config != "noadapt", url=args.url)
    log = (out / "turns.jsonl").open("w")
    steps = [{"kind": "plant", "fact": "code", "text": f"Код от шкафчика — {f['code']}. Какой код от шкафчика?"}] \
        if config == "reference" else script(rng, f, args.turns)
    started = time.monotonic()
    for n, step in enumerate(steps):
        result = bot.say(step["text"])
        record = {"n": n, **step, **result}
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.flush()
        print(n, step["kind"], result["status"], result["seconds"], "|", result["reply"][:90].replace("\n", " "), flush=True)
    before_restart = bot.state()
    bot.restart()
    answers = []
    for key, question in QUESTIONS:
        result = bot.say(question)
        expected = CITY_BASE[f["new_city"]] if key == "city" else str(f[key])
        old = CITY_BASE[f["city"]] if key == "city" else None
        hit = recalled(result["reply"], expected)
        answers.append({"fact": key, "question": question, "expected": expected, "reply": result["reply"],
                        "status": result["status"], "recalled": hit,
                        "stale": bool(old and recalled(result["reply"], old) and not hit)})
        print("Q", key, "OK" if hit else "--", "|", result["reply"][:90].replace("\n", " "), flush=True)
    summary = {"config": config, "seed": args.seed, "facts": f, "turns": len(steps),
               "minutes": round((time.monotonic() - started) / 60, 1),
               "answers": answers, "state_before_restart": before_restart, "state_after": bot.state()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    if isinstance(bot, Character):
        bot.m.dispose()
        bot.services.close()


if __name__ == "__main__":
    main()
