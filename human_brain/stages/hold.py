"""Удержание: может ли активность нейронов сама держаться между событиями.

Сейчас всё, что живёт дольше нескольких шагов (число дня, вопрос), держат
устройства, которые каждое утро обнуляет код мира. Здесь проверяется, может
ли это делать сама сеть: кольцо нейронов, раз загоревшись, горит само, пока
его не погасит торможение.

Проверка 1 (заведомо решаемая, --ring): кольцо из RING нейронов, связи по
кругу сильнее порога; толчок CUE зажигает первый, STOP тормозит все. Сколько
шагов кольцо горит само и гаснет ли по STOP. Два варианта: голое ядро и мозг
со всеми механизмами (рост, любопытство, изменение нейронов, запас входов,
обучение связей) — они могут кольцо разрушить.

Проверка 2 (--random): никаких колец заранее. Область «память» из N_MEM
нейронов, у каждого OUT случайных связей к другим со случайной задержкой
1..4 шага (важна длина петли во времени, а не число нейронов в ней), сила —
доля g от «до порога» (разброс 0.5..1.5). Факт A зажигает свои CUE_SIZE
нейронов, факт B — другие. Мерится: горит ли что-то через HOLD шагов, не
взрыв ли (горит почти всё), различаются ли следы A и B (доля общих нейронов).
До обучения и после показов фактов (рост связей и обучение могут сами
выделить петли).

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.hold --ring [--steps 200] [--stop 150]
    .venv/bin/python -m human_brain.stages.hold --random [--seed 0]
"""
import argparse
import copy
import random

from human_brain.stages import tasks
from human_brain.core.Brain import Brain

RING = 3
N_MEM = 40
OUT = 4
CUE_SIZE = 5
HOLD = 50
N_INH = 0      # тормозных нейронов в области (0 = без торможения)
INH_IN = 8     # от скольких нейронов памяти каждый тормозной получает возбуждение
INH_OUT = 10   # скольких нейронов памяти каждый тормозной гасит
INH_W = 1.0    # сила торможения, доля «до порога»
CUE_HOLD = 1   # сколько шагов подряд показывается факт (ансамбли: несколько)


def make_ring_brain(p, seed, full):
    b = Brain()
    b.add_region("входы")
    for n in ("CUE", "STOP"):
        b.add_neuron(n, region="входы")
    b.add_region("кольцо")
    names = [f"M{k}" for k in range(RING)]
    for n in names:
        b.add_neuron(n, region="кольцо")
    th = b.reach(b.neurons["M0"].index)
    b.connect("CUE", "M0", weight=th + 4.0, myelin_thickness=1.0)
    for k in range(RING):
        b.connect(names[k], names[(k + 1) % RING], weight=th + 4.0, myelin_thickness=1.0)
        b.connect("STOP", names[k], weight=3 * th, inhibitory=True, myelin_thickness=1.0)
    if full:
        tasks.attach(b, p, seed)
        b.growth.sensory_regions = ["входы"]
    return b


def ring_test(p, seed, full, steps, stop):
    b = make_ring_brain(p, seed, full)
    b.stimulate("CUE", 20.0)
    alive = []
    for t in range(steps):
        if t == stop:
            b.stimulate("STOP", 20.0)
        fired = b.step()
        alive.append(any(n.startswith("M") for n in fired))
    held = next((t for t in range(1, stop) if not any(alive[t:t + RING + 1])), stop)
    after = any(alive[stop + RING + 2:])
    w = [round(c.synapse.weight, 1) for c in b.connections
         if c.source.startswith("M") and c.target.startswith("M")]
    return held, after, w


def make_random_brain(p, seed, g, full):
    rnd = random.Random(seed)
    b = Brain()
    b.add_region("входы")
    for n in ("CUE_A", "CUE_B"):
        b.add_neuron(n, region="входы")
    b.add_region("память")
    mem = [f"M{k}" for k in range(N_MEM)]
    for n in mem:
        b.add_neuron(n, region="память")
    th = b.reach(b.neurons["M0"].index)
    for k, n in enumerate(mem):
        for m in rnd.sample([x for x in mem if x != n], OUT):
            d = rnd.randint(1, 4)
            b.connect(n, m, weight=g * th * rnd.uniform(0.5, 1.5),
                      myelin_thickness=(5.0 - d) / 4.0)
    if N_INH:
        # тормозные нейроны (общий механизм равновесия): их зовут нейроны
        # памяти, а они гасят случайных соседей — побеждает одна группа
        b.add_region("тормоз")
        for k in range(N_INH):
            iname = f"I{k}"
            b.add_neuron(iname, region="тормоз")
            for m in rnd.sample(mem, INH_IN):
                b.connect(m, iname, weight=0.5 * th, myelin_thickness=1.0)
            for m in rnd.sample(mem, INH_OUT):
                b.connect(iname, m, weight=INH_W * th, inhibitory=True, myelin_thickness=1.0)
    order = rnd.sample(mem, 2 * CUE_SIZE)
    for cue, group in (("CUE_A", order[:CUE_SIZE]), ("CUE_B", order[CUE_SIZE:])):
        for m in group:
            b.connect(cue, m, weight=th + 4.0, myelin_thickness=1.0)
    if full:
        tasks.attach(b, p, seed)
        b.growth.sensory_regions = ["входы", "тормоз"]
        b.growth.pair_regions = []          # без нейронов-сочетаний: смотрим саму область
    return b


def trace(b, cue, steps=HOLD + 10):
    """Показать факт на копии; доля горящих по шагам и кто горел около шага HOLD."""
    t = copy.deepcopy(b)
    t.curiosity = None
    for _ in range(CUE_HOLD - 1):
        t.stimulate(cue, 20.0)
        t.step()
    t.stimulate(cue, 20.0)
    frac, late = [], set()
    for k in range(steps):
        f = [n for n in t.step() if n.startswith("M")]
        frac.append(len(f) / N_MEM)
        if HOLD - 5 <= k <= HOLD + 5:
            late |= set(f)
    return frac, late


def random_test(p, seed, g, full, shows=300):
    """(когда, доля горящих около шага HOLD, наибольшая, горит у A, у B, общих) до и после показов."""
    b = make_random_brain(p, seed, g, full)
    out = []
    for stage in ("до", "после"):
        if stage == "после":
            rnd = random.Random(seed + 1)
            for _ in range(shows):
                cue = rnd.choice(("CUE_A", "CUE_B"))
                for _ in range(CUE_HOLD - 1):    # факт держится на входе несколько шагов
                    b.stimulate(cue, 20.0)
                    b.step()
                b.stimulate(cue, 20.0)
                for _ in range(HOLD + 30):
                    b.step()
        fa, la = trace(b, "CUE_A")
        fb, lb = trace(b, "CUE_B")
        both = len(la & lb) / max(1, len(la | lb))
        out.append((stage, round(max(fa[HOLD - 5:HOLD + 5]), 2), round(max(fa), 2), len(la), len(lb), round(both, 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--stop", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    if a.random:
        print("g, механизмы: (когда, доля горящих около шага 50, наибольшая доля, горит у A, у B, общих)")
        for full in (False, True):
            for g in (0.5, 0.8, 1.0, 1.3, 1.7):
                print(g, "все" if full else "голое", random_test(p, a.seed, g, full))
        return
    for full in (False, True):
        held, after, w = ring_test(p, a.seed, full, a.steps, a.stop)
        print(f"{'все механизмы' if full else 'голое ядро':14s} горело само до шага {held} "
              f"(STOP на {a.stop}); после STOP горит: {after}; связи кольца {w}")


if __name__ == "__main__":
    main()
