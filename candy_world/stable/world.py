"""Законы мира: 100 предметов и существ, 100 ситуаций. Агента здесь нет.

Один день:
1. Показывают карточку дня: сколько нужно каждому гостю и сколько штук в одной упаковке.
2. Мимо идут по одному, вперемешку: целые предметы, сломанные предметы, гости, ушедшие гости.
3. Агент берёт упаковки: каждый шаг либо ещё одну, либо "готово".
4. Праздник: кому не хватило, обижается, лишнее пропадает, каждый поход стоит времени.

Что значат карточка и отметки, ему никто не говорит: он узнаёт это только по последствиям.
"""
import math
import random
from dataclasses import dataclass

THING_NAMES = (
    "конфета булка туфля мячик ложка яблоко кепка книга чашка ключ носок шарф кубик свеча лента пуговица "
    "карандаш тарелка груша слива подушка зонт варежка шишка ракушка монетка морковка печенье пряник вилка "
    "ботинок перчатка лампа ведро гребень кружка блокнот линейка ластик тетрадь рюкзак колпак свисток флажок "
    "пирог сухарь орех вафля батарейка кисточка"
).split()
GUEST_NAMES = (
    "друг кошечка собачка сосед зайчик гость ёжик бабушка внук дедушка медведь лисичка попугай учитель "
    "хомяк черепаха кузен белочка мышонок козлик тётя дядя пингвин енот сова слонёнок ослик цыплёнок барсук утёнок "
    "повар врач почтальон садовник котёнок щенок племянник подруга тигрёнок волчонок олень барашек жираф дельфин "
    "моряк пилот художник сестра брат малыш"
).split()
assert len(THING_NAMES) == 50 and len(GUEST_NAMES) == 50
assert len(set(THING_NAMES + GUEST_NAMES)) == 100

NEW_THINGS = tuple(THING_NAMES[-10:])    # их нет в обучении
NEW_GUESTS = tuple(GUEST_NAMES[-10:])    # их нет в обучении
OLD_THINGS = tuple(THING_NAMES[:-10])
OLD_GUESTS = tuple(GUEST_NAMES[:-10])

# Внешний вид: у каждого свой постоянный набор из 16 признаков.
LOOK_SIZE = 16
_rng = random.Random(2026)
LOOKS = {name: [_rng.randint(0, 1) for _ in range(LOOK_SIZE)] for name in THING_NAMES + GUEST_NAMES}

# Отметка на предмете. Это видно как в жизни: сломан он или целый, ушёл гость или здесь.
# Карточку показывают штрихами: "нужно каждому" столько штрихов, сколько нужно, и так же упаковка.
# Пауза: ничего не происходит, время идёт.
THING, BROKEN, GUEST, GONE, CARD, TIME_TO_ACT, NEED_TICK, PACK_TICK, PAUSE = range(9)
MARKS = 9
NEEDS = (1, 2, 3, 4, 5)
PACKS = (1, 2, 3, 4, 5)
MODES = ("обычный", "часть гостей ушла", "часть предметов сломана", "и то и другое")
TOKEN_SIZE = MARKS + LOOK_SIZE

SITUATIONS = [(n, p, m) for n in NEEDS for p in PACKS for m in MODES]
# В обучении нет ни "по 5 каждому", ни "по 5 в упаковке": это проверка переноса на новое значение.
OLD_SITUATIONS = tuple(x for x in SITUATIONS if x[0] < 5 and x[1] < 5)
NEW_SITUATIONS = tuple(x for x in SITUATIONS if x[0] == 5 or x[1] == 5)

FETCH_COST = 0.2      # один поход: заметно дорого, но дешевле обиженного гостя
LEFTOVER_COST = 0.25  # одна лишняя штука
UPSET_COST = 1.0      # одна недоданная штука
MAX_FETCHES = 100


@dataclass(frozen=True)
class Day:
    things: int
    broken: int
    guests: int
    gone: int
    need: int
    pack: int

    def required_units(self):
        return max(self.guests * self.need - self.things * self.pack, 0)

    def right_fetches(self):
        """Только для проверки: агент этого не видит."""
        return math.ceil(self.required_units() / self.pack)


def token(mark, look=None):
    marks = [0] * MARKS
    marks[mark] = 1
    return marks + (list(look) if look else [0] * LOOK_SIZE)


def make_day(limit, situations, rng=random):
    need, pack, mode = rng.choice(situations)
    extra = limit // 2
    broken = rng.randint(1, max(extra, 1)) if mode in MODES[2:] else 0
    gone = rng.randint(1, max(extra, 1)) if mode in (MODES[1], MODES[3]) else 0
    return Day(rng.randint(0, limit), broken, rng.randint(0, limit), gone, need, pack)


def show(day, thing_names, guest_names, rng=random, pauses=False):
    """Что агент увидит за день. С паузами между событиями иногда ничего не происходит."""
    things = rng.sample(thing_names, rng.randint(1, min(4, len(thing_names))))
    guests = rng.sample(guest_names, rng.randint(1, min(4, len(guest_names))))
    items = [token(THING, LOOKS[rng.choice(things)]) for _ in range(day.things)]
    items += [token(BROKEN, LOOKS[rng.choice(things)]) for _ in range(day.broken)]
    items += [token(GUEST, LOOKS[rng.choice(guests)]) for _ in range(day.guests)]
    items += [token(GONE, LOOKS[rng.choice(guests)]) for _ in range(day.gone)]
    rng.shuffle(items)
    if pauses:
        for _ in range(rng.randint(0, len(items) + 3)):
            items.insert(rng.randint(0, len(items)), token(PAUSE))
    card = [token(CARD)] + [token(NEED_TICK)] * day.need + [token(PACK_TICK)] * day.pack
    return card + items


def party(day, fetched):
    """Закон мира: чем кончился день."""
    have = (day.things + fetched) * day.pack
    wanted = day.guests * day.need
    upset = max(wanted - have, 0)
    leftover = max(have - wanted, 0)
    score = -(upset * UPSET_COST + leftover * LEFTOVER_COST + fetched * FETCH_COST)
    return {"upset": upset, "leftover": leftover, "fetched": fetched, "score": score}
