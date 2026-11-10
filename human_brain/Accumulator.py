"""Накопитель: врождённое устройство, которое держит целое число (сколько угодно большое).

Устройство (не учится):
    level — целое число >= 0, на ночь (new_day) обнуляется;
    вход i, сработавший от толчка извне, прибавляет round(w[i]) порций
        (связь либо считается — порция за импульс, либо нет);
    действие из decrement, сработав, отнимает одну порцию;
    когда сработал сигнал «отвечай» (query) и level > 0, на следующем шаге
        загорается нейрон накопителя (neuron): «ещё есть».
    Счёт целыми порциями — это и есть повторяемая операция: она одинакова
    для 3 и для 1000, поэтому число не ограничено обучением.
Чему учится (по последствиям, learn): какие входы и сколько порций
прибавляют. После праздника известно верное число (принёс + не хватило -
лишнее); ошибка L = сумма w[i] * n[i] - верное (n[i] — сколько раз вход
сработал за день) правит силы: w[i] -= rate * L * n[i] / сумма n^2.
Входы, которые не связаны с ответом (дверь, карточка), уходят к 0, гость к 1.

Со знаком (signed=True, для вычитания): level может быть меньше 0; на сигнал
«отвечай» загорается neuron, если level > 0, или neg_neuron, если level < 0;
действие из decrement сдвигает level на одну порцию к 0. Верное число для
обучения тогда со знаком: не хватило минус осталось (конфета уходит к -1).

С регистром (register=True, для умножения): второе целое число R (на ночь 0).
Вход i кладёт в регистр round(r[i]) и прибавляет к уровню round(w[i]) +
round(m[i]) * R («прибавить содержимое регистра»). Учится всё по той же
ошибке после праздника (градиент квадрата ошибки по w, m, r; начальные m и r
0.1, иначе из нуля не сдвинуться). Карточки уходят в регистр (r -> 1), гость
прибавляет регистр (m -> 1, w -> 0): «каждому по N». Сложение и вычитание —
частный случай: одна карточка = по 1.
"""


class Accumulator:
    PARAMS = ("neuron", "inputs", "decrement", "query", "rate", "signed", "neg_neuron",
              "register", "seen", "plast")

    def __init__(self, brain, neuron, inputs, decrement, query, rate=0.5, signed=False,
                 neg_neuron=None, register=False, seen=(), plast=None):
        self.brain = brain
        brain.accumulator = self
        self.neuron = neuron
        self.inputs = list(inputs)
        self.decrement = list(decrement)
        self.query = query
        self.rate = rate
        self.signed = signed
        self.neg_neuron = neg_neuron
        self.register = register
        # «увиденные» входы (зрительные детекторы): считаются по срабатыванию
        # (их зажигает зрение, а не толчок извне) и могут только прибавлять
        # порции (w): в регистр не кладутся и регистр не прибавляют
        self.seen = list(seen)
        # пластичность входа (множитель обучения): закреплённые старые входы
        # учатся медленно, новые — в полную силу (как молодые и закреплённые связи)
        self.plast = dict(plast or {})
        self.w = {name: 0.0 for name in self.inputs}
        self.m = {name: 0.1 if register else 0.0 for name in self.inputs}
        self.r = {name: 0.1 if register else 0.0 for name in self.inputs}
        self.level = 0
        self.reg = 0
        self.counts = {name: 0 for name in self.inputs}
        self._q = {}      # (i, j) -> сумма по появлениям i числа j, бывших до него
        self.mute = False  # вопрос дня отдан нанятой заготовке (Units): накопитель молчит

    def units(self, name):
        return int(round(self.w[name]))

    def new_day(self):
        self.level = 0
        self.reg = 0
        self.counts = {name: 0 for name in self.inputs}
        self._q = {}

    def before_step(self):
        """Сигнал «отвечай» показан на этом шаге: накопитель отвечает, если не пуст."""
        b = self.brain
        q = b.neurons[self.query].index
        if b.stim_time[q] == b.time and not self.mute:
            if self.level > 0:
                b.stimulate(self.neuron, 20.0)
            elif self.level < 0 and self.signed:
                b.stimulate(self.neg_neuron, 20.0)

    def observe(self):
        """Вызывается мозгом в конце каждого шага."""
        b = self.brain
        t = b.time
        for name in self.inputs:
            i = b.neurons[name].index
            if b._fired[i] and (b.stim_time[i] == t or name in self.seen):
                self.level += self.units(name)
                if self.register:
                    self.level += int(round(self.m[name])) * self.reg
                    for j, cnt in self.counts.items():
                        if cnt:
                            self._q[(name, j)] = self._q.get((name, j), 0) + cnt
                    self.reg += int(round(self.r[name]))
                self.counts[name] += 1
        for name in self.decrement:
            if b._fired[b.neurons[name].index]:
                if self.level > 0:
                    self.level -= 1
                elif self.level < 0 and self.signed:
                    self.level += 1
        if not self.signed:
            self.level = max(0, self.level)

    def learn(self, correct):
        """После праздника: верное число известно, силы входов поправляются."""
        n = self.counts
        P = {k: self.plast.get(k, 1.0) for k in self.inputs}
        top = max(P.values()) if P else 1.0
        young = [k for k in self.inputs if P[k] >= top]
        if not self.register:
            # молодые (самые пластичные) входы исправляют ошибку так, как будто
            # закреплённые неизменны (знаменатель только по ним); закреплённые
            # учатся медленно с обычным знаменателем. При равной пластичности —
            # обычное правило
            denom = sum(n[k] * n[k] for k in self.inputs)
            denom_y = sum(n[k] * n[k] for k in young)
            if denom == 0:
                return 0.0
            err = sum(self.w[k] * n[k] for k in self.inputs) - correct
            for k in self.inputs:
                d = denom_y if (k in young and denom_y > 0) else denom
                self.w[k] -= P[k] * self.rate * err * n[k] / d
            return err
        # с регистром: прогноз = сумма w_i n_i + сумма m_i * сумма_j r_j q_ij
        gm = {i: sum(self.r[j] * q for (ii, j), q in self._q.items() if ii == i) for i in self.inputs}
        gr = {j: sum(self.m[i] * q for (i, jj), q in self._q.items() if jj == j) for j in self.inputs}
        pred = sum(self.w[k] * n[k] for k in self.inputs) + sum(self.m[i] * gm[i] for i in self.inputs)
        err = pred - correct
        def sq(k):
            return n[k] * n[k] + (0.0 if k in self.seen else gm[k] * gm[k] + gr[k] * gr[k])
        denom = sum(sq(k) for k in self.inputs)
        denom_y = sum(sq(k) for k in young)
        if denom == 0:
            return 0.0
        for k in self.inputs:
            d = denom_y if (k in young and denom_y > 0) else denom
            k_ = P[k] * self.rate * err / d
            self.w[k] -= k_ * n[k]
            if k not in self.seen:
                self.m[k] -= k_ * gm[k]
                self.r[k] -= k_ * gr[k]
        return err

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "w": self.w, "m": self.m, "r": self.r, "level": self.level, "reg": self.reg,
                "counts": self.counts}

    @classmethod
    def from_state(cls, brain, state):
        a = cls(brain, **state["params"])
        a.w = dict(state["w"])
        a.m = dict(state.get("m", a.m))
        a.r = dict(state.get("r", a.r))
        a.reg = int(state.get("reg", 0))
        a.level = int(state["level"])
        a.counts = dict(state["counts"])
        return a
