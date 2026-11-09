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
# Последствие прошлого события, если мир его показывает: "сейчас кто-то ждёт" (1) или "все довольны" (0).
TOKEN_SIZE = MARKS + LOOK_SIZE + 2

SITUATIONS = [(n, p, m) for n in NEEDS for p in PACKS for m in MODES]
# В обучении нет ни "по 5 каждому", ни "по 5 в упаковке": это проверка переноса на новое значение.
OLD_SITUATIONS = tuple(x for x in SITUATIONS if x[0] < 5 and x[1] < 5)
NEW_SITUATIONS = tuple(x for x in SITUATIONS if x[0] == 5 or x[1] == 5)

@dataclass(frozen=True)
class Rules:
    """Законы одного мира. Агенту их не называют."""

    name: str = "основной"
    extra_need: int = 0
    broken_is_good: bool = False
    gone_still_get: bool = False
    pack_is_one: bool = False


MAIN = Rules()
TRAIN_WORLDS = (MAIN, Rules("на 1 больше", extra_need=1), Rules("сломанные годятся", broken_is_good=True),
                Rules("упаковки врут", pack_is_one=True))
NEW_WORLDS = (Rules("ушедшим тоже надо (не учился)", gone_still_get=True), Rules("на 2 больше (не учился)", extra_need=2))

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

    def right_fetches(self, rules=None):
        """Сколько упаковок нужно по законам мира. Агент этого не видит."""
        rules = rules or MAIN
        guests = self.guests + (self.gone if rules.gone_still_get else 0)
        pack = 1 if rules.pack_is_one else self.pack
        have = (self.things + (self.broken if rules.broken_is_good else 0)) * pack
        return math.ceil(max(guests * (self.need + rules.extra_need) - have, 0) / pack)


def token(mark, look=None, seen=None):
    """seen: что мир показал после прошлого события (None: ничего не показал)."""
    marks = [0] * MARKS
    marks[mark] = 1
    shown = [0, 0] if seen is None else [1, int(seen)]
    return marks + (list(look) if look else [0] * LOOK_SIZE) + shown


def make_day(limit, situations, rng=random):
    need, pack, mode = rng.choice(situations)
    extra = limit // 2
    broken = rng.randint(1, max(extra, 1)) if mode in MODES[2:] else 0
    gone = rng.randint(1, max(extra, 1)) if mode in (MODES[1], MODES[3]) else 0
    return Day(rng.randint(0, limit), broken, rng.randint(0, limit), gone, need, pack)


def show(day, thing_names, guest_names, rng=random, pauses=False, consequences=False, rules=None):
    """Что агент увидит за день, и что можно было предсказать после каждого события.

    Возвращает (события, цели). Цель для события: будет ли после него кто-то ждать. Если мир
    показывает последствия, это видно в следующем событии, уже после того как его можно было
    предсказать. None: тут предсказывать нечего.
    """
    things = rng.sample(thing_names, rng.randint(1, min(4, len(thing_names))))
    guests = rng.sample(guest_names, rng.randint(1, min(4, len(guest_names))))
    events = [(THING, rng.choice(things)) for _ in range(day.things)]
    events += [(BROKEN, rng.choice(things)) for _ in range(day.broken)]
    events += [(GUEST, rng.choice(guests)) for _ in range(day.guests)]
    events += [(GONE, rng.choice(guests)) for _ in range(day.gone)]
    rng.shuffle(events)
    if pauses:
        for _ in range(rng.randint(0, len(events) + 3)):
            events.insert(rng.randint(0, len(events)), (PAUSE, None))
    return sequence(day, events, consequences, rules)


def sequence(day, events, consequences=False, rules=None):
    """События по порядку: (отметка, имя). Мир сам ведёт счёт, агент его не видит."""
    tokens = [token(CARD)] + [token(NEED_TICK)] * day.need + [token(PACK_TICK)] * day.pack
    targets = [None] * len(tokens)
    rules = rules or MAIN
    pack = 1 if rules.pack_is_one else day.pack
    need = day.need + rules.extra_need
    wanted = have = 0
    seen = None
    for mark, name in events:
        tokens.append(token(mark, LOOKS[name] if name else None, seen if consequences else None))
        if mark == THING or (mark == BROKEN and rules.broken_is_good):
            have += pack
        elif mark == GUEST or (mark == GONE and rules.gone_still_get):
            wanted += need
        if mark in (THING, GUEST, BROKEN, GONE):
            seen = wanted > have
            targets.append(int(seen))
        else:
            targets.append(None)
    return tokens, targets


def party(day, fetched, rules=None):
    """Подзатыльник за каждую упаковку мимо, в любую сторону, и цена похода."""
    right = day.right_fetches(rules)
    short, extra = max(right - fetched, 0), max(fetched - right, 0)
    return {"short": short, "extra": extra, "fetched": fetched, "score": -((short + extra) * 1.0 + fetched * FETCH_COST)}
