"""Доверие к знанию по ситуации: общий механизм «когда применять знание».

Знание здесь — устройство со своим выходом: прежний накопитель (ACC / ACC_NEG)
и заготовки (U1..U14, как в NeuralChoice). Таблиц «вопрос -> устройство» нет:
ни владения вопросами (own), ни весов по вопросу (w[q]), ни закрепления по
серии удач, ни ограничения «сочетания только с новым вопросом».

Ситуация — все нейроны мозга, сработавшие с утра до первого «отвечай»
(какие из них важны, никто не говорит). У каждого устройства свои ворота:
    доверие g = сигмоида(смещение + сумма весов сработавших нейронов).
Ворота делают две вещи:
    применять — толчок выхода устройства = 20 * g (при малом доверии выход
        не доходит до порога и молчит);
    учить     — устройство учится с множителем ответственности
        λ = g * exp(-e) / max по устройствам (e — ошибка самого устройства в
        этот день, в порциях); там, где оно ошибается, а другое право, оно не
        переучивается.
Чему учатся ворота: после праздника каждое устройство сверяет, что ответило
бы оно само (в тени, даже если отвечало другое): верно (e = 0) -> цель 1,
иначе 0; веса сработавших нейронов и смещение сдвигаются на lr * (цель - g).
Ошибку каждое устройство считает само, так же, как уже учится (накопитель —
число со знаком, заготовка — свой ответ); правила под вид задачи нет.

Новое число: lr (скорость ворот). hand: функция (имя устройства, ситуация)
-> доверие, задаётся руками для заведомо решаемой проверки.
Что остаётся ручным: сами устройства (накопитель, 14 заготовок) и то, что
выходы заготовок сочетаются только с нейронами защёлки (иначе сочетаний сотни).
"""
import math

from human_brain.parts.NeuralChoice import NeuralChoice, Candidate

ANY = "*"   # у заготовки одни веса на все ситуации


class Gate:
    def __init__(self, lr):
        self.lr = lr
        self.bias = 0.0
        self.w = {}          # имя нейрона -> вес

    def value(self, x):
        s = self.bias + sum(self.w.get(n, 0.0) for n in x)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))

    def learn(self, x, target, g):
        d = self.lr * (target - g)
        self.bias += d
        for n in x:
            self.w[n] = self.w.get(n, 0.0) + d

    def state(self):
        return {"lr": self.lr, "bias": self.bias, "w": self.w}

    @classmethod
    def from_state(cls, st):
        g = cls(st["lr"])
        g.bias = st["bias"]
        g.w = dict(st["w"])
        return g


class TrustChoice(NeuralChoice):
    def __init__(self, brain, inputs, lr=0.2, hand=None, make=True, consistent=False,
                 verified_max=50, **kw):
        kw.pop("own", None)
        super().__init__(brain, inputs, own=[], make=make, **kw)
        self.streak = 0            # закрепления по серии нет
        self.mute_old = False
        self.partners_all = True
        self.lr = lr
        self.hand = hand
        # consistent: вместо ответственности λ каждое устройство само отличает
        # ошибку знания от ошибки применения (поправка не должна ломать дни,
        # где оно было право; накопитель — Accumulator.consistent)
        self.consistent = consistent
        self.verified_max = verified_max
        self.verified = {}         # имя заготовки -> [(счётчики, ответ)]
        acc = brain.accumulator
        self.acc_name = acc.neuron
        self.gates = {n: Gate(lr) for n in [self.acc_name] + self.names}
        self.log = []
        self._x = None
        self._seen = set()
        self._g = {}
        self._level0 = None
        if make:
            self.refresh_partners()

    # ---------- день ----------

    def new_day(self):
        for c in self.cands:
            c.new_day()
        self._x = None
        self._seen = set()
        self._g = {}
        self._level0 = None
        acc = self.brain.accumulator
        acc.mute = True            # выход накопителя подаёт этот модуль, с доверием

    def trust(self, name):
        if self.hand is not None:
            return self.hand(name, self._x)
        return self.gates[name].value(self._x)

    def before_step(self):
        b = self.brain
        if b.stim_time[b.neurons[self.query].index] != b.time:
            return
        acc = b.accumulator
        if self._x is None:
            self._x = frozenset(self._seen)
            self._g = {n: self.trust(n) for n in self.gates}
            self._level0 = acc.level
        g = self._g[self.acc_name]
        if acc.level > 0:
            b.stimulate(acc.neuron, 20.0 * g)
        elif acc.level < 0 and acc.signed:
            b.stimulate(acc.neg_neuron, 20.0 * g)
        for c, name in zip(self.cands, self.names):
            if c.has_more(ANY):
                b.stimulate(name, 20.0 * self._g[name])

    def observe(self):
        b = self.brain
        t = b.time
        fired = b._fired[:b.n_neurons]
        if self._x is None:
            for i in fired.nonzero()[0]:
                self._seen.add(b.names[i])
        for name in self.inputs:
            i = b.neurons[name].index
            if fired[i] and b.stim_time[i] == t:
                for c in self.cands:
                    c.counts[name] += 1
        if fired[b.neurons[self.decrement].index]:
            for c in self.cands:
                c.mores += 1

    def errors(self, answer, truth):
        e = {self.acc_name: abs(self._level0 - truth)}
        for c, name in zip(self.cands, self.names):
            e[name] = abs(c.predicted(ANY) - answer)
        return e

    def after_day(self, answer, truth, learn, result=None):
        if not learn or self._x is None:
            return
        e = self.errors(answer, truth)
        s = {n: self._g[n] * math.exp(-e[n]) for n in e}
        top = max(s.values())
        lam = {n: (v / top if top > 0 else 0.0) for n, v in s.items()}
        # ворота: верно ли было бы само устройство в этой ситуации
        if self.hand is None:
            for n, gate in self.gates.items():
                gate.learn(self._x, 1.0 if e[n] == 0 else 0.0, self._g[n])
        # устройства учатся с ответственностью
        if self.consistent:
            self.brain.accumulator.learn(truth)
            for c, name in zip(self.cands, self.names):
                self._learn_checked(c, name, answer, e[name] == 0)
            return
        self.brain.accumulator.learn(truth, scale=lam[self.acc_name])
        for c, name in zip(self.cands, self.names):
            old = c.rate
            c.rate = old * lam[name]
            c.learn(ANY, answer)
            c.rate = old

    @staticmethod
    def _answer(c, counts, w):
        """Что ответила бы заготовка c с весами w в дне с такими счётчиками."""
        lv = sum(int(round(w[k])) * n for k, n in counts.items())
        if c.variant == "по одной":
            return max(lv, 0)
        r = counts.get(c.reg_event, 0)
        return lv // r if r > 0 and lv > 0 else 0

    def _learn_checked(self, c, name, answer, right):
        """Поправка заготовки — наибольшая доля, не ломающая дни, где она была права."""
        counts = dict(c.counts)
        old = dict(c.weights(ANY))
        c.learn(ANY, answer)
        new = dict(c.weights(ANY))
        mem = self.verified.setdefault(name, [])
        good = [(n, a) for n, a in mem if self._answer(c, n, old) == a]
        if good:
            # повернуть поправку: без части, меняющей уровень на проверенных днях
            import numpy as np
            keys = list(new)
            d = np.array([new[k] - old[k] for k in keys])
            from human_brain.parts.gem import project
            G = np.array([[n.get(k, 0) for k in keys] for n, _ in good], dtype=float)
            def target(n, a):
                r = n.get(c.reg_event, 0)
                return a if c.variant == "по одной" else a * r + (r - 1) / 2.0
            errs = np.array([sum(old[k] * n.get(k, 0) for k in keys) - target(n, a) for n, a in good])
            d = project(d, errs[:, None] * G)
            new = {k: old[k] + float(v) for k, v in zip(keys, d)}
        alpha = 1.0
        while True:
            cand = {k: old[k] + alpha * (new[k] - old[k]) for k in new}
            if all(self._answer(c, n, cand) == a for n, a in good):
                break
            alpha = alpha / 2 if alpha > 1.0 / 64 else 0.0
            if alpha == 0.0:
                cand = old
                break
        c.w[ANY] = cand
        if right:
            mem.append((counts, answer))
            if len(mem) > self.verified_max:
                mem.pop(0)

    # ---------- разбор ----------

    def report(self, x):
        """Доверие каждого устройства в ситуации x (набор имён нейронов)."""
        return {n: round(g.value(x), 3) for n, g in self.gates.items()}

    # ---------- сохранение ----------

    def state(self):
        st = super().state()
        st.update(kind="trust", lr=self.lr, gates={n: g.state() for n, g in self.gates.items()},
                  consistent=self.consistent, verified_max=self.verified_max,
                  verified={n: [[c, a] for c, a in m] for n, m in self.verified.items()})
        return st

    @classmethod
    def from_state(cls, brain, st):
        p = dict(st["params"])
        p.pop("own", None)
        x = cls(brain, lr=st["lr"], make=False, **p)
        x.cands = [Candidate.from_state(c) for c in st["cands"]]
        x.gates = {n: Gate.from_state(g) for n, g in st["gates"].items()}
        x.log = list(st.get("log", []))
        x.consistent = st.get("consistent", False)
        x.verified_max = st.get("verified_max", 50)
        x.verified = {n: [(dict(c), a) for c, a in m] for n, m in st.get("verified", {}).items()}
        return x
