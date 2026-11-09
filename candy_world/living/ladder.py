"""Растёт ли лестница? Каждая найденная программа становится словом библиотеки для следующих ступеней.

Ступени: 1) по 2 каждому, 2) по N одному гостю, 3) по N нескольким гостям (0-3, надо сохранить N),
4) по N каждому (умножение). В библиотеку идут и программы целиком, и каждый найденный повтор отдельно.

Допуск в библиотеку: найденная программа должна быть верна и на большем, чем требовала ступень
(больше чисел, больше гостей, большие N). Иначе это подгонка, попытка не засчитывается.
Режимы: "scratch" только базовые действия; "ladder" базовые действия плюс слова, найденные на прежних ступенях.
Слова берутся только из найденного отбором, ручные программы служат лишь проверкой, что ступень решаема.
Поиск с перезапусками: серия коротких попыток, застрявшую бросаем.

    python ladder.py --mode ladder --seed 0
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import compose as C

CHAINS = ("C", "R", "S")
STEPS = ("double", "one", "two", "multiply")
EVENTS = {"double": ("guest", "thing"), "one": ("tick", "guest", "thing"), "two": ("tick", "guest", "thing"),
          "multiply": ("tick", "guest", "thing")}
TRIES, GENERATIONS = 16, 300
POPULATION = CHILDREN = 16
MAX_LEN = 5


def make_days(step, count, limit, rng, need_range=(1, 3), guests_max=3):
    if step not in ("one", "two"):
        return C.make_days(step, count, limit, rng, need_range)
    days = []
    for _ in range(count):
        need, things = rng.randint(*need_range), rng.randint(0, limit)
        guests = 1 if step == "one" else rng.randint(0, guests_max)
        events = ["thing"] * things + ["guest"] * guests
        rng.shuffle(events)
        days.append((["tick"] * need + events, max(guests * need - things, 0)))
    return days


class Machine(C.Machine):
    def __init__(self, width, limit, words):
        super().__init__(width, limit)
        self.words = words

    def run(self, program):
        for s in program:
            if s[0] == "call":
                body, names = self.words[s[1]]
                self.run(rename(body, dict(zip(names, s[2]))))
            elif s[0] == "inc":
                self.call(s[1], C.INC)
            elif s[0] == "dec":
                self.call(s[1], C.DEC)
            else:
                while self.call(s[1], C.CHECK):
                    self.run(s[2])


def rename(body, mapping):
    out = []
    for s in body:
        if s[0] == "call":
            out.append(("call", s[1], tuple(mapping.get(c, c) for c in s[2])))
        elif s[0] == "while":
            out.append(("while", mapping.get(s[1], s[1]), rename(s[2], mapping)))
        else:
            out.append((s[0], mapping.get(s[1], s[1])))
    return out


def chains_in(body, seen=None):
    seen = [] if seen is None else seen
    for s in body:
        names = s[2] if s[0] == "call" else (s[1],)
        for c in names:
            if c not in seen:
                seen.append(c)
        if s[0] == "while":
            chains_in(s[2], seen)
    return seen


def make_word(body):
    names = chains_in(body)
    placeholders = [f"x{i}" for i in range(len(names))]
    return rename(body, dict(zip(names, placeholders))), placeholders


def answer(program, events, words, width=12, limit=6000):
    m = Machine(width, limit, words)
    try:
        for e in events:
            m.run(program[e])
        trips = 0
        while m.call("C", C.CHECK):
            trips += 1
        return trips
    except (C.Budget, RecursionError):
        return -1


def score(program, days, words, width=12):
    got = [answer(program, e, words, width) for e, _ in days]
    errors = sum(abs(g - r) if g >= 0 else 100 for g, (_, r) in zip(got, days))
    return -errors / len(days), sum(g == r for g, (_, r) in zip(got, days)) / len(days)


def random_statement(rng, words, depth=0):
    if words and rng.random() < 0.3:
        name = rng.choice(list(words))
        return ("call", name, tuple(rng.choice(CHAINS) for _ in words[name][1]))
    if depth < 1 and rng.random() < 0.25:
        return ("while", rng.choice(CHAINS), [random_statement(rng, words, depth + 1) for _ in range(rng.randint(1, 2))])
    return (rng.choice(("inc", "dec")), rng.choice(CHAINS))


def mutate(stmts, rng, words, depth=0):
    stmts = list(stmts)
    roll = rng.random()
    if (roll < 0.3 or not stmts) and len(stmts) < MAX_LEN:
        stmts.insert(rng.randint(0, len(stmts)), random_statement(rng, words, depth))
    elif roll < 0.45 and stmts:
        stmts.pop(rng.randrange(len(stmts)))
    elif roll < 0.7 and stmts:
        stmts[rng.randrange(len(stmts))] = random_statement(rng, words, depth)
    elif stmts:
        i = rng.randrange(len(stmts))
        s = stmts[i]
        if s[0] == "while":
            stmts[i] = ("while", s[1], mutate(s[2], rng, words, depth + 1) or [random_statement(rng, words, depth + 1)]) \
                if rng.random() < 0.6 else ("while", rng.choice(CHAINS), s[2])
        elif s[0] == "call":
            stmts[i] = ("call", s[1], tuple(rng.choice(CHAINS) for _ in s[2]))
        else:
            stmts[i] = (s[0], rng.choice(CHAINS))
    return stmts


def size(stmts):
    return sum(1 + (size(s[2]) if s[0] == "while" else 0) for s in stmts)


def admitted(step, program, words):
    """Проверка допуска: больше, чем требовала ступень. Отдельные дни, не те, что у итогового экзамена."""
    rng = random.Random(4242)
    if step == "double":
        days = make_days(step, 60, 30, rng)
    elif step == "one":
        days = make_days(step, 60, 20, rng, (4, 7))
    elif step == "two":
        days = make_days(step, 60, 15, rng, (1, 6), guests_max=6)
    else:
        days = make_days(step, 60, 20, rng, (4, 7))
    return score(program, days, words, 16)[1] == 1.0


def search(step, words, seed):
    """Перезапуски: до TRIES попыток по GENERATIONS поколений. Возвращает программу и сколько поколений ушло."""
    train = make_days(step, 60, 6, random.Random(500 + seed))
    used = 0
    search.rejected = 0
    for attempt in range(TRIES):
        rng = random.Random(seed * 1000 + attempt)
        population = [{e: [random_statement(rng, words)] for e in EVENTS[step]} for _ in range(POPULATION)]
        scored = [((lambda v: (v[0] - 0.001 * sum(size(p) for p in prog.values()), v[1]))(score(prog, train, words)), prog)
                  for prog in population]
        for generation in range(1, GENERATIONS + 1):
            children = []
            for _ in range(CHILDREN):
                child = copy.deepcopy(rng.choice(scored)[1])
                event = rng.choice(EVENTS[step])
                child[event] = mutate(child[event], rng, words)
                value = score(child, train, words)
                children.append(((value[0] - 0.001 * sum(size(p) for p in child.values()), value[1]), child))
            scored = sorted(scored + children, key=lambda x: x[0][0], reverse=True)[:POPULATION]
            used += 1
            if scored[0][0][1] == 1.0:
                if admitted(step, scored[0][1], words):
                    return scored[0][1], used, attempt + 1
                search.rejected += 1      # подгонка под ступень: в библиотеку не берём, начинаем заново
                break
    return None, used, TRIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["scratch", "ladder"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    words, report = {}, []
    for step in STEPS:
        started = time.perf_counter()
        program, generations, attempts = search(step, words if args.mode == "ladder" else {}, args.seed)
        row = {"ступень": step, "найдено": program is not None, "поколений всего": generations, "попыток": attempts,
               "отвергнуто на допуске": search.rejected,
               "секунд": round(time.perf_counter() - started), "слов в библиотеке": len(words)}
        if program is not None:
            exam_rng = random.Random(77)
            row["экзамен до 50, цепочка 16"] = score(program, make_days(step, 60, 50, exam_rng), words, 16)[1]
            if step in ("one", "two", "multiply"):
                row["экзамен по 4-7 (не учился)"] = score(program, make_days(step, 60, 20, exam_rng, (4, 7)), words, 16)[1]
            row["программа"] = program
            row["использует слова"] = sorted({s[1] for p in program.values() for s in p if s[0] == "call"})
            if args.mode == "ladder":
                for event, body in program.items():
                    if body:
                        words[f"{step}_{event}"] = make_word(body)
                    for k, statement in enumerate(body):
                        if statement[0] == "while":
                            words[f"{step}_{event}_loop{k}"] = make_word([statement])
        report.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "программа"}, ensure_ascii=False), flush=True)
    out = Path(__file__).parent / "results_ladder"
    out.mkdir(exist_ok=True)
    (out / f"{args.mode}__seed_{args.seed}.json").write_text(
        json.dumps({"режим": args.mode, "seed": args.seed, "ступени": report, "слова": words,
                    "хеш библиотеки не менялся": C.TABLE_HASH == C.hashlib.sha256(C.TABLE.tobytes()).hexdigest()},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
