"""«По N каждому» с плотным сигналом о хранимом N (идея 1, 2026-11-11).

Мир тот же, что в task_multiply (штрихи карточки задают N, гость хочет N,
предмет даёт 1, поход). Устройство цепочки то же (DESIGN, 512 бит таблицы).
Новое только в обучении: после каждого гостя мир задаёт два вопроса —
PROBE0 «бит 0 твоего N?» и PROBE1 «бит 1 твоего N?» (N от 1 до 3 = 2 бита);
верный ответ (решение цепочки на этом событии) — этот бит N. Так у задачи
появляется плотный сигнал именно там, где все прошлые поиски застревали:
использовать N, не разрушив его. На экзамене вопросов нет (make_exam_days).
"""
import numpy as np

from task_multiply import TICK, GUEST, THING, FETCH, DESIGN, hand_table as mul_hand

PROBE0, PROBE1 = 5, 6
CODES = {"fetch": FETCH}


def make_days(items, rng, probes=True):
    days, right, empty = [], [], []
    for things, guests, need in items:
        events = [GUEST] * guests + [THING] * things
        rng.shuffle(events)
        events = [TICK] * need + events
        out, row, balance = [], [], 0
        for e in events:
            if e == GUEST:
                balance += need
            elif e == THING:
                balance -= 1
            out.append(e)
            row.append(int(balance >= 0) if e in (GUEST, THING) else None)
            if probes and e == GUEST:
                out += [PROBE0, PROBE1]
                row += [need & 1, (need >> 1) & 1]
        days.append(out)
        right.append(max(guests * need - things, 0))
        empty.append(row)
    return days, np.array(right), empty


def make_exam_days(items, rng):
    return make_days(items, rng, probes=False)


def hand_table():
    """Ручная таблица умножения + ответы на вопросы: N хранится двоично,
    бит k — в элементе k. Вопрос k: первый элемент получает слева a=1, b=1;
    PROBE0 — он отдаёт свой бит N по линии b, a=0; PROBE1 — он передаёт
    a=1, b=0, и бит отдаёт второй элемент (видит a=1, b=0). Остальные
    (a=0) передают b дальше. Память при вопросах не меняется."""
    d = DESIGN
    t = mul_hand().reshape(d.rows, d.out_bits).copy()
    for code in (PROBE0, PROBE1):
        for c in (0, 1):
            for n in (0, 1):
                for a in (0, 1):
                    for b in (0, 1):
                        row = (code << 4) | (c << 3) | (n << 2) | (a << 1) | b
                        if a == 1 and b == 1:
                            out = (0, n) if code == PROBE0 else (1, 0)
                        elif a == 1 and b == 0:
                            out = (0, n)
                        else:
                            out = (0, b)
                        t[row] = (c, n, out[0], out[1])
    return t.reshape(-1)
