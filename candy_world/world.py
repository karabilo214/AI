"""Законы мира. Здесь нет агента, и мир ничего не подсказывает, кроме последствий.

Как идёт один день:
1. Агент смотрит: мимо него по одному проходят предметы (конфеты, булки, туфли) и гости
   (друзья, кошечки, собачки), вперемешку. Числа ему не сообщают.
2. Агент готовится: каждый шаг либо идёт за ещё одним предметом, либо говорит "готово".
3. Праздник: каждому гостю даётся один предмет. Кому не хватило, обижается. Лишние остаются.

Поход стоит заметно дорого, лишний предмет пропадает, обида гостя хуже всего.
"""
import random

# Что видно у каждого предмета: две роли и внешний вид. Роль видна сразу, как в жизни:
# конфету можно отдать, а гость её хочет. Внешний вид у разных предметов разный.
LOOK_SIZE = 8
TOKEN_SIZE = 3 + LOOK_SIZE
THING, GUEST, TIME_TO_ACT = 0, 1, 2

LOOKS = {
    "конфета": [1, 0, 0, 1, 0, 1, 1, 0], "булка": [0, 1, 0, 1, 1, 0, 1, 0],
    "туфля": [1, 1, 1, 0, 0, 0, 1, 0], "мячик": [0, 0, 1, 0, 1, 1, 0, 1],
    "друг": [0, 1, 1, 0, 1, 0, 0, 1], "кошечка": [1, 0, 1, 1, 0, 0, 0, 1],
    "собачка": [0, 1, 0, 0, 0, 1, 1, 1], "сосед": [1, 1, 0, 1, 0, 1, 0, 0],
    # Этих в обучении нет: только для проверки.
    "яблоко": [1, 1, 0, 0, 1, 1, 0, 0], "ложка": [0, 0, 0, 1, 1, 1, 1, 0],
    "гость": [0, 0, 1, 1, 0, 0, 1, 1], "зайчик": [1, 0, 0, 0, 1, 0, 1, 1],
}
THINGS = ("конфета", "булка", "туфля", "мячик")     # что можно отдать
GUESTS = ("друг", "кошечка", "собачка", "сосед")    # кто хочет получить
NEW_THINGS = ("яблоко", "ложка")
NEW_GUESTS = ("гость", "зайчик")


FETCH_COST = 0.2        # один поход: заметно дорого, но дешевле обиженного гостя
LEFTOVER_COST = 0.25    # одна лишняя конфета
UPSET_COST = 1.0        # один обиженный гость
MAX_FETCHES = 100       # больше сходить нельзя


def token(role, look=None):
    roles = [0, 0, 0]
    roles[role] = 1
    return roles + (list(look) if look else [0] * LOOK_SIZE)


def show(candies, friends, things=None, guests=None):
    """Что агент увидит: всё вперемешку, по одному. В один день бывает один вид или несколько."""
    things = things or random.sample(THINGS, random.randint(1, len(THINGS)))
    guests = guests or random.sample(GUESTS, random.randint(1, len(GUESTS)))
    items = [token(THING, LOOKS[random.choice(things)]) for _ in range(candies)]
    items += [token(GUEST, LOOKS[random.choice(guests)]) for _ in range(friends)]
    random.shuffle(items)
    return items


def party(candies, friends, fetched):
    """Закон мира: чем кончился день."""
    have = candies + fetched
    upset = max(friends - have, 0)
    leftover = max(have - friends, 0)
    score = -(upset * UPSET_COST + leftover * LEFTOVER_COST + fetched * FETCH_COST)
    return {"upset": upset, "leftover": leftover, "fetched": fetched, "score": score}
