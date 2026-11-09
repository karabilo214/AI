"""Отбор программ из вызовов библиотеки. Библиотека (таблица счётчика) не меняется.

    python compose_search.py --task multiply --seed 0
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import compose as C

EVENTS = {"double": (C.GUEST, C.THING), "triple": (C.GUEST, C.THING), "multiply": (C.TICK, C.GUEST, C.THING)}
GENERATIONS = 1500
POPULATION, CHILDREN = 24, 24
MAX_LEN = 5


def random_statement(rng, depth=0):
    chain = rng.choice(C.CHAINS)
    if depth < 1 and rng.random() < 0.25:
        return ("while", chain, [random_statement(rng, depth + 1) for _ in range(rng.randint(1, 2))])
    return (rng.choice(("inc", "dec")), chain)


def mutate_list(stmts, rng, depth=0):
    stmts = list(stmts)
    roll = rng.random()
    if (roll < 0.3 or not stmts) and len(stmts) < MAX_LEN:
        stmts.insert(rng.randint(0, len(stmts)), random_statement(rng, depth))
    elif roll < 0.45 and stmts:
        stmts.pop(rng.randrange(len(stmts)))
    elif roll < 0.7 and stmts:
        stmts[rng.randrange(len(stmts))] = random_statement(rng, depth)
    elif stmts:
        i = rng.randrange(len(stmts))
        s = stmts[i]
        if s[0] == "while":
            if rng.random() < 0.5:
                stmts[i] = ("while", rng.choice(C.CHAINS), s[2])
            else:
                stmts[i] = ("while", s[1], mutate_list(s[2], rng, depth + 1) or [random_statement(rng, depth + 1)])
        else:
            stmts[i] = (s[0], rng.choice(C.CHAINS))
    return stmts


def size(stmts):
    return sum(1 + (size(s[2]) if s[0] == "while" else 0) for s in stmts)


def fitness(program, days):
    value, exact = C.score(program, days)
    return value - 0.001 * sum(size(p) for p in program.values()), exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(EVENTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    train = C.make_days(args.task, 60, 6, random.Random(1000 + args.seed))
    population = [{e: [random_statement(rng)] for e in EVENTS[args.task]} for _ in range(POPULATION)]
    scored = [(fitness(p, train), p) for p in population]
    started, solved = time.perf_counter(), None
    for generation in range(1, GENERATIONS + 1):
        children = []
        for _ in range(CHILDREN):
            parent = rng.choice(scored)[1]
            child = copy.deepcopy(parent)
            event = rng.choice(EVENTS[args.task])
            child[event] = mutate_list(child[event], rng)
            children.append((fitness(child, train), child))
        scored = sorted(scored + children, key=lambda x: x[0][0], reverse=True)[:POPULATION]
        best_exact = scored[0][0][1]
        if generation % 50 == 0:
            print(f"{args.task}/{args.seed} поколение {generation}: точно {best_exact:.3f}, "
                  f"секунд {time.perf_counter() - started:.0f}", flush=True)
        if best_exact == 1.0:
            solved = generation
            break
    best = scored[0][1]
    exam_rng = random.Random(77)
    exam = {"знакомое, числа до 10": C.score(best, C.make_days(args.task, 150, 10, exam_rng))[1],
            "числа до 50, цепочка 16": C.score(best, C.make_days(args.task, 60, 50, exam_rng), 16)[1]}
    if args.task == "multiply":
        exam["по 4-7 (не учился), до 30, цепочка 16"] = C.score(best, C.make_days(args.task, 60, 30, exam_rng, (4, 7)), 16)[1]
    result = {"задача": args.task, "seed": args.seed, "решено на поколении": solved, "секунд": round(time.perf_counter() - started),
              "программа": best, "экзамен": exam, "хеш библиотеки не менялся": C.TABLE_HASH == C.hashlib.sha256(C.TABLE.tobytes()).hexdigest()}
    out = Path(__file__).parent / "results_compose"
    out.mkdir(exist_ok=True)
    (out / f"{args.task}__seed_{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
