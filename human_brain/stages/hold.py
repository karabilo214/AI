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

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.hold --ring [--steps 200] [--stop 150]
"""
import argparse

from human_brain.stages import tasks
from human_brain.core.Brain import Brain

RING = 3


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--stop", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    for full in (False, True):
        held, after, w = ring_test(p, a.seed, full, a.steps, a.stop)
        print(f"{'все механизмы' if full else 'голое ядро':14s} горело само до шага {held} "
              f"(STOP на {a.stop}); после STOP горит: {after}; связи кольца {w}")


if __name__ == "__main__":
    main()
