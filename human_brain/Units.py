"""Найм элементов из заготовок: система сама создаёт устройство для нового вопроса.

Прежний накопитель (Accumulator) «владеет» вопросами, на которых он выучен
(own), и учится только на них: ошибка нового вопроса его не переписывает.

Для каждого вопроса (запомненного защёлкой) система после дня сравнивает, что
насчитало её устройство (ответ при идеальном «ещё/готово»), с верным числом.
Вопрос без своего устройства сначала отвечается прежним накопителем (без
обучения). Если за patience дней этого вопроса доля ошибок устройства выше
max_error, нанимается заготовка, привязанная к вопросу, со своим нейроном-
выходом (U1, U2, ...: загорается на «отвечай», если «ещё есть»). Заготовка:
    вид «по одной»     — уровень; «ещё» отнимает 1; «ещё есть», пока уровень > 0;
    вид «по регистру»  — уровень и регистр; «ещё» отнимает регистр;
                         «ещё есть», пока уровень >= регистра (> 0);
    регистр            — счётчик одного события (reg_event).
Что прибавлять к уровню (w), заготовка учит по ошибке после праздника (как
накопитель): цель уровня — верный ответ A («по одной») или A * регистр +
(регистр - 1) / 2 («по регистру»: середина допустимого). Если через trial дней
своего вопроса доля ошибок за последние judge дней выше max_error, заготовку
увольняют и нанимают следующую (порядок перебора случайный, свой у мозга).
Удачная (ошибок не больше max_error) остаётся. Пока вопрос привязан к
заготовке, прежний накопитель на нём молчит.
"""
import random
from collections import deque

VARIANTS = ("по регистру", "по одной")


class Unit:
    def __init__(self, inputs, variant, reg_event, rate=0.5):
        self.inputs = list(inputs)
        self.variant = variant
        self.reg_event = reg_event
        self.rate = rate
        self.w = {k: 0.0 for k in self.inputs}
        self.level = 0
        self.reg = 0
        self.counts = {k: 0 for k in self.inputs}

    def new_day(self):
        self.level = 0
        self.reg = 0
        self.counts = {k: 0 for k in self.inputs}

    def add(self, name):
        self.level += int(round(self.w[name]))
        if name == self.reg_event:
            self.reg += 1
        self.counts[name] += 1

    def more(self):
        if self.variant == "по одной":
            if self.level > 0:
                self.level -= 1
        else:
            self.level -= self.reg

    def has_more(self):
        if self.variant == "по одной":
            return self.level > 0
        return self.reg > 0 and self.level >= self.reg

    def predicted(self):
        if self.variant == "по одной":
            return max(self.level, 0)
        return self.level // self.reg if self.reg > 0 and self.level > 0 else 0

    def learn(self, answer):
        n = self.counts
        denom = sum(v * v for v in n.values())
        if denom == 0:
            return
        if self.variant == "по одной":
            target = answer
        else:
            target = answer * self.reg + (self.reg - 1) / 2.0
        err = sum(self.w[k] * n[k] for k in self.inputs) - target
        for k in self.inputs:
            self.w[k] -= self.rate * err * n[k] / denom

    def state(self):
        return {"inputs": self.inputs, "variant": self.variant, "reg_event": self.reg_event,
                "rate": self.rate, "w": self.w}

    @classmethod
    def from_state(cls, st):
        u = cls(st["inputs"], st["variant"], st["reg_event"], st["rate"])
        u.w = dict(st["w"])
        return u


class Units:
    PARAMS = ("inputs", "own", "query", "actions", "decrement", "patience", "trial",
              "judge", "max_error", "seed")

    def __init__(self, brain, inputs, own, query="GO", actions=("MORE", "DONE"),
                 decrement="MORE", patience=100, trial=100, judge=40, max_error=0.1,
                 seed=0):
        self.brain = brain
        brain.units = self
        self.inputs = list(inputs)
        self.own = list(own)            # вопросы прежнего накопителя
        self.query = query
        self.actions = list(actions)
        self.decrement = decrement
        self.patience = patience
        self.trial = trial
        self.judge = judge
        self.max_error = max_error
        self.seed = seed
        rnd = random.Random(seed)
        self.candidates = [[v, e] for v in VARIANTS for e in self.inputs]
        rnd.shuffle(self.candidates)
        self.next_candidate = 0
        self.bound = {}                 # вопрос -> {"unit", "neuron", "days", "since"}
        self.history = {}               # вопрос -> последние результаты устройства (True = верно)
        self.days_seen = {}             # вопрос -> сколько дней было
        self.log = []
        self._question = None
        self._prediction = None
        self._asked = False

    # ---------- день ----------

    def new_day(self):
        self._question = None
        self._prediction = None
        self._asked = False
        self.brain.accumulator.mute = False
        for rec in self.bound.values():
            rec["unit"].new_day()

    def _current_unit(self):
        rec = self.bound.get(self._question)
        return rec["unit"] if rec else None

    def before_step(self):
        b = self.brain
        if b.stim_time[b.neurons[self.query].index] != b.time:
            return
        if not self._asked:
            self._asked = True
            self._prediction = self._device_answer()
        rec = self.bound.get(self._question)
        if rec is not None and rec["unit"].has_more():
            b.stimulate(rec["neuron"], 20.0)

    def observe(self):
        b = self.brain
        t = b.time
        for name in self.inputs:
            i = b.neurons[name].index
            if b._fired[i] and b.stim_time[i] == t:
                for rec in self.bound.values():
                    rec["unit"].add(name)
        if b.latch.held is not None:
            self._question = b.latch.inputs[b.latch.held]
            b.accumulator.mute = self._question in self.bound
        if b._fired[b.neurons[self.decrement].index]:
            u = self._current_unit()
            if u is not None:
                u.more()

    def _device_answer(self):
        u = self._current_unit()
        if u is not None:
            return u.predicted()
        lvl = self.brain.accumulator.level
        return max(-lvl, 0) if self._question == "ASK_LEFT" else max(lvl, 0)

    def after_day(self, answer, truth, learn):
        """Конец дня: обучение устройства вопроса, учёт ошибок, найм/увольнение."""
        if not learn:
            return
        q = self._question
        acc = self.brain.accumulator
        if q is None:
            return
        if q in self.own:
            acc.learn(truth)
        rec = self.bound.get(q)
        if rec is not None:
            rec["unit"].learn(answer)
            rec["days"] += 1
        ok = self._prediction == answer
        h = self.history.setdefault(q, deque(maxlen=self.judge))
        h.append(ok)
        self.days_seen[q] = self.days_seen.get(q, 0) + 1
        if q in self.own:
            return
        err = 1.0 - sum(h) / len(h)
        if rec is None:
            if self.days_seen[q] >= self.patience and err > self.max_error:
                self._hire(q)
        elif not rec.get("kept") and rec["days"] >= self.trial:
            if err > self.max_error:
                self._fire(q)
                self._hire(q)
            else:
                rec["kept"] = True
                self.log.append(f"{q}: оставлена заготовка {rec['unit'].variant}, регистр "
                                f"{rec['unit'].reg_event} (ошибок {err:.2f})")

    # ---------- найм ----------

    def _hire(self, q):
        b = self.brain
        if self.next_candidate >= len(self.candidates):
            self.log.append(f"{q}: заготовки кончились")
            return
        variant, reg_event = self.candidates[self.next_candidate]
        self.next_candidate += 1
        name = f"U{len([r for r in self.bound]) + 1}"
        if name not in b.neurons:
            self._make_neuron(name)
        self.bound[q] = {"unit": Unit(self.inputs, variant, reg_event), "neuron": name,
                         "days": 0}
        self.history[q] = deque(maxlen=self.judge)
        self.log.append(f"день {self.days_seen.get(q, 0)} вопроса {q}: нанята заготовка "
                        f"{variant}, регистр {reg_event} -> {name}")

    def _fire(self, q):
        rec = self.bound.pop(q)
        self.log.append(f"день {self.days_seen.get(q, 0)} вопроса {q}: уволена "
                        f"{rec['unit'].variant}, регистр {rec['unit'].reg_event}")

    def _make_neuron(self, name):
        """Выход заготовки: как датчик (слабые неизменные связи к действиям)."""
        b = self.brain
        b.add_neuron(name, region="входы")
        w = b.rules["Synapse"]["ACTION_MIN_WEIGHT"]
        for o in self.actions:
            b.connect(name, o, weight=w, myelin_thickness=1.0, fixed=True)
            b.connect(name, o, weight=0.0, inhibitory=True, myelin_thickness=1.0, plasticity=0.0)
        if b.growth is not None:
            b.growth.max_neurons += 6

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "candidates": self.candidates, "next_candidate": self.next_candidate,
                "bound": {q: {"unit": r["unit"].state(), "neuron": r["neuron"], "days": r["days"],
                              "kept": r.get("kept", False)} for q, r in self.bound.items()},
                "history": {q: list(h) for q, h in self.history.items()},
                "days_seen": self.days_seen, "log": self.log}

    @classmethod
    def from_state(cls, brain, st):
        u = cls(brain, **st["params"])
        u.candidates = st["candidates"]
        u.next_candidate = st["next_candidate"]
        u.bound = {q: {"unit": Unit.from_state(r["unit"]), "neuron": r["neuron"],
                       "days": r["days"], "kept": r["kept"]} for q, r in st["bound"].items()}
        u.history = {q: deque(h, maxlen=u.judge) for q, h in st["history"].items()}
        u.days_seen = dict(st["days_seen"])
        u.log = list(st["log"])
        return u
