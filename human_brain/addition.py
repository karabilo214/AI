"""Сложение в мире из candy_world/arith/arith.py: учим до 10, проверяем любые числа.

День (как в arith.py, задача «сложение»): карточка (каждому по 1), гости
приходят группами (дверь, потом гости по одному; групп не меньше двух, в
группе не меньше двух гостей), вопрос «сколько принести?». Потом ответ шагами:
на каждом шаге показывается сигнал «отвечай» (GO), мозг говорит «ещё» (MORE)
или «готово» (DONE). Ответ = число «ещё» до первого «готово».

Мозг: входы CARD, CANDY, DOOR, GUEST, ASK, GO; накопитель ACC (Accumulator.py,
врождённое устройство: целое число, вход прибавляет round(w) порций, «ещё»
отнимает одну, на GO загорается ACC, если не пусто); действия MORE и DONE
соревнуются (взаимное торможение, короткое: inh_short). Связи при рождении: все входы и ACC ко
всем действиям, как в задачах по награде (tasks.make_reward_brain). Рост,
любопытство, изменение нейронов, забывание, запрет — исходные (tasks.attach);
знакомость выключена (иначе десятки одинаковых гостей перестают замечаться),
у входов и действий нет подстройки порога (make_same; иначе GO и GUEST
глохнут, а пороги «ещё» и «готово» разъезжаются и перекашивают выбор).

Учится по последствиям:
    накопитель — после праздника известно верное число (принёс + не хватило
        - лишнее), по ошибке правит, какие входы сколько прибавляют;
    действия — та же подсказка задним числом, что в arith.py: на шагах до
        верного числа надо было «ещё», на верном — «готово»; за каждый шаг
        награда +1 / -1 (без ответа 0). Действие, когда не спрашивали (во
        время событий и в паузах), — наказание. Включены правки головы
        elig_reset и last_action (метки не переходят к следующей награде) и
        boredom (скука выводит из ловушки молчания). Обучаемое торможение
        вход -> действие (правка головы inhibit) — по --inhibit, пока хуже.
Ещё для этого мозга включены выключатели ядра (по умолчанию выключены):
budget_above_floor (запас сверх минимума путей к действиям),
reward_only_actions (пути к действиям меняются только по награде и не тают
от простоя) и Growth.sensory_regions = входы (в органы чувств связи не растут).
Учим на днях с числом гостей до 10. Экзамен на копии без шума и без
обучения: свежие дни до 10, 11-20, 21-50, 51-100, 101-1000. Рядом ленивая
стратегия «всегда 0» (для сложения всегда 0%, гостей не меньше 4).
--hand: ручная сборка (заведомо решаемая проверка).

Запуск из папки human_brain:
    ../.venv/bin/python addition.py [--seed 0] [--days 300] [--hand]
"""
import argparse
import copy
import random
import time

import tasks
import Head
from Brain import Brain
from Accumulator import Accumulator

EVENTS = ("CARD", "CANDY", "DOOR", "GUEST", "ASK")
ACTIONS = ("MORE", "DONE")
EVENT_STEPS = 3      # шагов на одно событие
ANSWER_WAIT = 6      # сколько шагов ждать ответа на GO
ANSWER_REST = 4      # пауза после ответа


# ---------- мир (как arith.py, задача «сложение») ----------

def groups(guests, rng, at_least=2):
    sizes = [2] * max(at_least, rng.randint(1, guests // 2))
    while sum(sizes) > guests:
        sizes.pop()
    for _ in range(guests - sum(sizes)):
        sizes[rng.randrange(len(sizes))] += 1
    out = []
    for s in sizes:
        out += ["DOOR"] + ["GUEST"] * s
    return out


def make_day(rng, top=10, low=0):
    g = rng.randint(max(4, low), top)
    return ["CARD"] + groups(g, rng) + ["ASK"], g


# ---------- мозг ----------

INHIBIT = False   # обучаемое торможение вход -> действие (--inhibit): пока делает хуже


def make_brain(p, seed, hand=False):
    b = Brain()
    b.add_region("входы")
    for n in EVENTS + ("GO", "ACC"):
        b.add_neuron(n, region="входы")
    if hand:
        b.add_region("действия, жёсткие", learn=0.0)   # ручные силы не меняются
    b.add_region("действия")
    for n in ACTIONS:
        b.add_neuron(n, explores=True, region="действия, жёсткие" if hand else "действия")
    if hand:
        # с учётом пола 1.0 у путей к действиям (Synapse.ACTION_MIN_WEIGHT):
        # «ещё» = GO + ACC = 11, «готово» = GO + ACC = 10; один GO: «готово» 9 > порога 8
        w = {("GO", "MORE"): 5.0, ("ACC", "MORE"): 6.0, ("GO", "DONE"): 9.0}
        for s in EVENTS + ("GO", "ACC"):
            for o in ACTIONS:
                b.connect(s, o, weight=w.get((s, o), 1.0), myelin_thickness=1.0)
    else:
        for s in EVENTS + ("GO", "ACC"):
            for o in ACTIONS:
                b.connect(s, o, weight=p["init_weight"], myelin_thickness=1.0,
                          plasticity=p["init_plasticity"])
    if not hand and p.get("inhibit_on", INHIBIT):
        # обучаемое торможение вход -> действие (правка головы inhibit): с 0,
        # наказание усиливает, награда ослабляет. Без него «есть ещё» (ACC) не
        # может подавить «готово»: возбуждение одно различает «GO» и «GO + ACC»
        # только разницей сил, которая у предела силы сжимается почти до нуля
        for s in EVENTS + ("GO", "ACC"):
            for o in ACTIONS:
                b.connect(s, o, weight=0.0, inhibitory=True, myelin_thickness=1.0,
                          plasticity=p.get("inh_plasticity", Head.INH_PLASTICITY))
        b.inh_window = Head.INH_WINDOW
        b.inh_delayed = True
    b.connect("MORE", "DONE", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("DONE", "MORE", weight=p["inhibit_weight"], inhibitory=True)
    # торможение не оставляет заряда ниже покоя (правка из головы): иначе
    # «готово», заторможенное прошлым «ещё», не может ответить на следующий GO
    b.inh_short = True
    # правки головы: награда сбрасывает метки, новое действие стирает метки
    # прошлого (иначе «готово», сработавшее в паузе после «ещё», получало
    # награду за следующее «ещё»)
    b.reset_eligibility = True
    b.last_action_only = True
    # запас входов сверх минимума путей к действиям: иначе на экзамене (без
    # наград) GO -> DONE таял 20 -> 10 за 7 дней
    b.budget_above_floor = True
    # пути к действиям меняются только по награде: иначе на экзамене (без
    # наград) сотни шагов ответа растили GO -> MORE (9.6 -> 12.6 за день с 626)
    b.reward_only_actions = p.get("reward_only_actions", True)
    if not hand:
        tasks.attach(b, p, seed)
        # входы без подстройки порога: иначе часто показываемые GO и GUEST
        # поднимают порог выше толчка 20 и перестают замечаться (GO через раз)
        # и у действий (правка головы same_act): частое «ещё» поднимало свой порог,
        # редкое «готово» опускало, и «готово» выигрывало соревнование при меньшем входе
        b.neuron_change.make_same(("входы", "действия"))
        # скука (правка головы boredom): ответы «никто» подряд подталкивают
        # действия; иначе «готово», сведённое наказаниями к минимуму, больше
        # никогда не срабатывает и не может получить награду (ловушка молчания)
        b.curiosity.boredom = p.get("boredom", Head.BOREDOM)
        # рост не проводит связи внутрь входов: иначе после ASK нейрон GO
        # загорался сам (выученный ритм дня) и сбивал такт ответа
        b.growth.sensory_regions = ["входы"]
    acc = Accumulator(b, "ACC", EVENTS, decrement=["MORE"], query="GO")
    if hand:
        acc.w["GUEST"] = 1.0
    return b


def acted(fired):
    a = [n for n in ACTIONS if n in fired]
    return a[0] if len(a) == 1 else ("оба" if a else None)


def day(b, events, answer, learn, cap=None, truth=None):
    """Прожить день. Возвращает (ответ мозга, закончил ли «готово», действий не вовремя).
    truth: верное число со знаком для накопителя (по умолчанию = answer)."""
    acc = b.accumulator
    acc.new_day()
    if b.latch is not None:
        b.latch.new_day()
    if b.units is not None:
        b.units.new_day()
    early = [0]

    def unasked(n):
        """n шагов, когда не спрашивали: действие — ошибка (при обучении наказание)."""
        for _ in range(n):
            if acted(b.step()) is not None:
                early[0] += 1
                if learn:
                    b.reward(-1.0)

    for e in events:
        b.stimulate(e, 20.0)
        unasked(EVENT_STEPS)
    cap = answer + 10 if cap is None else cap
    more, done = 0, False
    for k in range(cap):
        b.stimulate("GO", 20.0)
        out = None
        for _ in range(ANSWER_WAIT):
            out = acted(b.step())
            if out is not None:
                break
        if learn:
            want = "MORE" if more < answer else "DONE"
            b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
        unasked(ANSWER_REST)
        if out == "MORE":
            more += 1
        elif out == "DONE" or out == "оба":
            done = out == "DONE"
            break
    if b.units is not None:
        b.units.after_day(answer, answer if truth is None else truth, learn)
    elif learn:
        acc.learn(answer if truth is None else truth)
    unasked(20)
    return more, done, early[0]


def exam_sets(seed, n=30):
    rng = random.Random(1000 + seed)
    return {"до 10, свежие дни": [make_day(rng) for _ in range(n)],
            "11-20": [make_day(rng, 20, 11) for _ in range(n)],
            "21-50": [make_day(rng, 50, 21) for _ in range(n)],
            "51-100": [make_day(rng, 100, 51) for _ in range(10)],
            "101-1000": [make_day(rng, 1000, 101) for _ in range(5)]}


def exam(b, days):
    t = copy.deepcopy(b)
    t.curiosity = None
    ok = 0
    for events, answer in days:
        more, done, _ = day(t, events, answer, learn=False)
        ok += more == answer and done
    return ok


def run(p, seed, days=300, sets=None):
    """Обучить мозг на days днях (до 10) и сдать экзамены. Возвращает {набор: доля верных}."""
    tasks.apply(p)
    b = make_brain(p, seed)
    rng = random.Random(seed)
    for _ in range(days):
        events, answer = make_day(rng)
        day(b, events, answer, learn=True)
    ex = exam_sets(seed)
    names = list(ex) if sets is None else sets
    return {k: exam(b, ex[k]) / len(ex[k]) for k in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--inhibit", action="store_true")
    a = ap.parse_args()
    global INHIBIT
    INHIBIT = a.inhibit
    p = dict(tasks.PARAMS_DEFAULT)
    p["fam_on"] = False
    tasks.apply(p)
    b = make_brain(p, a.seed, hand=a.hand)
    rng = random.Random(a.seed)
    start = time.time()
    if not a.hand:
        check = [make_day(random.Random(77)) for _ in range(20)]
        right = 0
        for d in range(1, a.days + 1):
            events, answer = make_day(rng)
            more, done, early = day(b, events, answer, learn=True)
            right += more == answer and done
            if d % 50 == 0:
                acc = b.accumulator
                print(f"день {d}: верно за последние 50 {right}/50, экзамен до 10 "
                      f"{exam(b, check)}/20, накопитель "
                      + " ".join(f"{k} {v:.2f}" for k, v in acc.w.items())
                      + f"  [{time.time() - start:.0f} с]", flush=True)
                right = 0
    print("ЭКЗАМЕН (мозг / ленивая «всегда 0» = 0):")
    for name, days in exam_sets(a.seed).items():
        print(f"  {name:18s} {exam(b, days)}/{len(days)}", flush=True)


if __name__ == "__main__":
    main()
