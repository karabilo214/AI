"""Законы мира: 100 предметов и существ, 100 ситуаций. Агента здесь нет.

Один день:
1. Показывают карточку дня: сколько нужно каждому гостю и сколько штук в одной упаковке.
2. Мимо идут по одному, вперемешку: целые предметы, сломанные предметы, гости, ушедшие гости.
3. Агент берёт упаковки: каждый шаг либо ещё одну, либо "готово".
4. Итог: за каждую упаковку мимо нужного числа подзатыльник, в любую сторону. Поход тоже стоит времени.

Законы могут быть разными: это разные миры. Агенту не говорят, в каком он мире.

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
THING, BROKEN, GUEST, GONE, CARD, TIME_TO_ACT = range(6)
MARKS = 6
NEEDS = (1, 2, 3, 4, 5)
PACKS = (1, 2, 3, 4, 5)
MODES = ("обычный", "часть гостей ушла", "часть предметов сломана", "и то и другое")
TOKEN_SIZE = MARKS + len(NEEDS) + len(PACKS) + LOOK_SIZE

SITUATIONS = [(n, p, m) for n in NEEDS for p in PACKS for m in MODES]
assert len(SITUATIONS) == 100
_shuffled = random.Random(7).sample(SITUATIONS, 100)
NEW_SITUATIONS = tuple(_shuffled[:20])   # их нет в обучении
OLD_SITUATIONS = tuple(_shuffled[20:])



SLAP = 1.0        # подзатыльник за одну упаковку мимо
FETCH_COST = 0.2  # один поход
MAX_FETCHES = 100


@dataclass(frozen=True)
class Rules:
    """Законы одного мира."""

    name: str = "основной"
    extra_need: int = 0            # сколько ещё каждому гостю сверх карточки
    broken_is_good: bool = False   # сломанные тоже годятся
    gone_still_get: bool = False   # ушедшим гостям тоже надо дать
    pack_is_one: bool = False      # в упаковке всегда одна штука, что бы ни писали на карточке


MAIN = Rules()
OTHER_WORLDS = (
    Rules("рыбка съедает на 1 больше", extra_need=1),
    Rules("сломанные годятся", broken_is_good=True),
    Rules("ушедшим тоже надо", gone_still_get=True),
    Rules("упаковки врут: всегда 1 штука", pack_is_one=True),
)

# Третий мир: предметов и гостей больше нет. Жуки и многоножки это еда, рыбки в аквариуме ждут её.
BUG_NAMES = (
    "жук многоножка муравей кузнечик таракан сверчок мокрица богомол божья_коровка стрекоза "
    "комар мотылёк клоп оса пчела бабочка гусеница червяк паук шершень"
).split()
FISH_NAMES = (
    "гуппи карп карась сом окунь щука неон скалярия барбус петушок "
    "золотая_рыбка меченосец данио тетра дискус лещ форель судак налим линь"
).split()
_fish_rng = random.Random(2027)
for _name in BUG_NAMES + FISH_NAMES:
    LOOKS[_name] = [_fish_rng.randint(0, 1) for _ in range(LOOK_SIZE)]
OLD_BUGS, NEW_BUGS = tuple(BUG_NAMES[:-5]), tuple(BUG_NAMES[-5:])
OLD_FISH, NEW_FISH = tuple(FISH_NAMES[:-5]), tuple(FISH_NAMES[-5:])


@dataclass(frozen=True)
class Day:
    things: int
    broken: int
    guests: int
    gone: int
    need: int
    pack: int

    def right_fetches(self, rules=MAIN):
        """Сколько упаковок нужно по законам этого мира. Агент этого не видит."""
        guests = self.guests + (self.gone if rules.gone_still_get else 0)
        wanted = guests * (self.need + rules.extra_need)
        pack = 1 if rules.pack_is_one else self.pack
        have = (self.things + (self.broken if rules.broken_is_good else 0)) * pack
        return math.ceil(max(wanted - have, 0) / pack)


def token(mark, look=None, need=None, pack=None):
    marks = [0] * MARKS
    marks[mark] = 1
    needs = [int(need == n) for n in NEEDS]
    packs = [int(pack == p) for p in PACKS]
    return marks + needs + packs + (list(look) if look else [0] * LOOK_SIZE)


def make_day(limit, situations, rng=random):
    need, pack, mode = rng.choice(situations)
    extra = limit // 2
    broken = rng.randint(1, max(extra, 1)) if mode in MODES[2:] else 0
    gone = rng.randint(1, max(extra, 1)) if mode in (MODES[1], MODES[3]) else 0
    return Day(rng.randint(0, limit), broken, rng.randint(0, limit), gone, need, pack)


def show(day, thing_names, guest_names, rng=random):
    """Что агент увидит за день. В один день бывает один вид или несколько."""
    things = rng.sample(thing_names, rng.randint(1, min(4, len(thing_names))))
    guests = rng.sample(guest_names, rng.randint(1, min(4, len(guest_names))))
    items = [token(THING, LOOKS[rng.choice(things)]) for _ in range(day.things)]
    items += [token(BROKEN, LOOKS[rng.choice(things)]) for _ in range(day.broken)]
    items += [token(GUEST, LOOKS[rng.choice(guests)]) for _ in range(day.guests)]
    items += [token(GONE, LOOKS[rng.choice(guests)]) for _ in range(day.gone)]
    rng.shuffle(items)
    return [token(CARD, need=day.need, pack=day.pack)] + items


def party(day, fetched, rules=MAIN):
    """Закон мира: чем кончился день. Подзатыльник одинаковый, взял меньше или взял больше."""
    right = day.right_fetches(rules)
    short = max(right - fetched, 0)
    extra = max(fetched - right, 0)
    score = -((short + extra) * SLAP + fetched * FETCH_COST)
    return {"short": short, "extra": extra, "fetched": fetched, "score": score}
