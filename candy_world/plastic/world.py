"""Мир, где правило дня скрыто. В один день несколько попыток, после каждой виден итог.

Агенту не говорят правило. Он может понять его только из своих подзатыльников за этот день.
"""
import random

THING, BROKEN, GUEST, GONE, CARD, ACT, RESULT = range(7)
MARKS = 7
NEEDS = (1, 2, 3)
TOKEN_SIZE = MARKS + len(NEEDS) + 3   # отметка, карточка, итог попытки
MAX_ANSWER = 60

# Скрытые правила: (сколько ещё каждому, сломанные годятся, ушедшим тоже надо)
TRAIN_RULES = {"обычное": (0, False, False), "на 1 больше": (1, False, False),
               "сломанные годятся": (0, True, False), "ушедшим тоже": (0, False, True)}
NEW_RULE = {"на 2 больше (не учился)": (2, False, False)}


def right_answer(scene, rule):
    things, broken, guests, gone, need = scene
    extra, broken_good, gone_get = rule
    wanted = (guests + (gone if gone_get else 0)) * (need + extra)
    return max(wanted - things - (broken if broken_good else 0), 0)


def make_scene(rng):
    return (rng.randint(0, 6), rng.randint(0, 3), rng.randint(0, 6), rng.randint(0, 3), rng.choice(NEEDS))


def token(mark, need=None, result=(0, 0, 0)):
    row = [0.0] * TOKEN_SIZE
    row[mark] = 1.0
    if need is not None:
        row[MARKS + NEEDS.index(need)] = 1.0
    row[MARKS + len(NEEDS):] = list(result)
    return row


def scene_tokens(scene, rng):
    things, broken, guests, gone, need = scene
    items = [token(THING)] * things + [token(BROKEN)] * broken + [token(GUEST)] * guests + [token(GONE)] * gone
    rng.shuffle(items)
    return [token(CARD, need=need)] + items + [token(ACT)]


def result_token(answer, right):
    """Что он узнаёт: не хватило, лишнее и насколько. Правильного числа ему не называют."""
    gap = answer - right
    return token(RESULT, result=(float(gap < 0), float(gap > 0), min(abs(gap), 10) / 10))
