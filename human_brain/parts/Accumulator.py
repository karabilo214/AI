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

Роли символов (role_of, по умолчанию пусто = выключено). Общая врождённая
способность: число-символ означает «столько-то раз событие». Какое событие —
решают нейроны: у каждого события мира свой нейрон роли (role_of: нейрон ->
событие), к ним идут связи от всех «увиденных» символов (буквы, цифры, знаки).
Если перед символом j (после предыдущего символа, без событий мира между
ними) сработал нейрон роли события e, символ j работает как round(w[j]) раз
событие e (прибавляет порции, кладёт в регистр — всё, что делает e). Связь
«символ -> роль» — настоящий синапс мозга (сначала 0); устройство учит её по
последствиям, перебирая в уме (как заготовки деления, которые считают все
сразу): после праздника для каждой цифры j, стоявшей после символа i, оно
прикидывает, каким вышло бы число, если бы j была столько-то раз каждым
событием (или без роли). Если одна роль дала бы самое верное число и лучше,
чем без роли, связь i -> её нейрон крепнет на role_step (доля «до порога»);
сработавшая роль, которая не лучшая, слабеет. (Пробовалось: случайный поиск
роли и подкрепление по «сошлось ли число» — знак связывался с дверью и
карточкой: пока роль искалась, новый вес самого знака уходил к -5.)
Ошибку, которую объяснила бы лучшая роль, веса входов не учат: накопитель
правит веса по числу, каким оно вышло бы при лучших ролях (иначе новый вес
знака впитывал ошибку роли раньше, чем роль успевала выучиться).
"""
import random


class Accumulator:
    PARAMS = ("neuron", "inputs", "decrement", "query", "rate", "signed", "neg_neuron",
              "register", "seen", "plast", "role_of", "role_explore", "role_step",
              "confidence", "conf_p0", "conf_r", "conf_q", "consistent", "verified_max")

    def __init__(self, brain, neuron, inputs, decrement, query, rate=0.5, signed=False,
                 neg_neuron=None, register=False, seen=(), plast=None, role_of=None, role_explore=0.0,
                 role_step=0.25, roles=None, confidence=False, conf_p0=1.0, conf_r=1.0,
                 conf_q=0.001, consistent=False, verified_max=50):
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
        self.role_of = dict(role_of or {})
        self.role_explore = role_explore
        self.role_step = role_step        # шаг силы связи, доля от «до порога»
        # уверенность в каждом весе (confidence=True, вместо plast): неуверенность
        # P (новый вес conf_p0) убывает от каждого дня, где вес участвовал, и
        # чуть растёт (conf_q) — уверенный вес меняется медленно, но не застывает;
        # ошибку дня исправляют прежде всего неуверенные веса (см. learn)
        self.confidence = confidence
        self.conf_p0, self.conf_r, self.conf_q = conf_p0, conf_r, conf_q
        import numpy as np
        # ошибка знания или ошибка применения (consistent=True): поправку дня
        # сначала примеряют к последним verified_max дням, где накопитель был
        # прав; берётся самая большая доля поправки, которая ни один из них не
        # ломает (0 — ошибка применения: знание не трогается)
        self.consistent = consistent
        self.verified_max = verified_max
        self.verified = []    # [(counts, q, верное число)]
        self.Pkeys = []       # "w:имя" / "m:имя" / "r:имя"
        self._pidx = {}
        self.Pm = np.zeros((0, 0))   # совместная неуверенность весов
        self.role_rng = random.Random(0)
        self._role = None      # (нейрон роли, символ, вызвавший её)
        self._sym = None       # (последний символ, шаг)
        self._used = []        # номера связей «символ -> роль», использованных за день
        self._added = 0
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
        self._role = None
        self._sym = None
        self._used = []
        self._added = 0

    def before_step(self):
        """Сигнал «отвечай» показан на этом шаге: накопитель отвечает, если не пуст."""
        b = self.brain
        q = b.neurons[self.query].index
        if b.stim_time[q] == b.time and not self.mute:
            if self.level > 0:
                b.stimulate(self.neuron, 20.0)
            elif self.level < 0 and self.signed:
                b.stimulate(self.neg_neuron, 20.0)

    def _apply(self, name, times=1):
        """Событие name случилось times раз (прибавить, регистр — как у события)."""
        for _ in range(times):
            add = self.units(name)
            if self.register:
                add += int(round(self.m[name])) * self.reg
                for j, cnt in self.counts.items():
                    if cnt:
                        self._q[(name, j)] = self._q.get((name, j), 0) + cnt
                self.reg += int(round(self.r[name]))
            self.level += add
            self._added += add
            self.counts[name] += 1

    def _role_conn(self, sym, role):
        b = self.brain
        s, r = b.neurons[sym].index, b.neurons[role].index
        for c in range(b.n_conn):
            if b.src[c] == s and b.tgt[c] == r and b.alive[c] and not b.inhibitory[c]:
                return c
        return None

    def _observe_roles(self):
        """Какой нейрон роли сработал (сильнейшая связь от последнего символа)."""
        b = self.brain
        fired = [r for r in self.role_of if b._fired[b.neurons[r].index]]
        if not fired:
            return
        sym = self._sym[0] if self._sym else None
        if len(fired) > 1 and sym is not None:
            fired.sort(key=lambda r: -(b.weight[c] if (c := self._role_conn(sym, r)) is not None else 0.0))
        self._role = (fired[0], sym)

    def observe(self):
        """Вызывается мозгом в конце каждого шага."""
        b = self.brain
        t = b.time
        if self.role_of:
            self._observe_roles()
        for name in self.inputs:
            i = b.neurons[name].index
            if not b._fired[i]:
                continue
            if name in self.seen:
                role = self._role
                reps = self.units(name)
                before = self._added
                used = None
                if role is not None and role[1] != name:
                    self._apply(self.role_of[role[0]], abs(reps))
                    used = role[0]
                    self._role = None
                else:
                    self.level += reps
                    self._added += reps
                    self.counts[name] += 1   # через роль цифра считается событиями роли
                prev = self._sym[0] if self._sym is not None else None
                if self.role_of and prev is not None and prev != name and reps != 0:
                    # для разбора после праздника: (символ, сколько раз, вклад, роль)
                    self._used.append((prev, abs(reps), self._added - before, used, name))
                self._sym = (name, t)
            elif b.stim_time[i] == t:
                self._apply(name)
                self._role = None      # событие мира между символами снимает роль
                self._sym = None
        if (self.role_of and self._sym is not None and self._role is None
                and t - self._sym[1] == 2 and b.curiosity is not None
                and self.role_rng.random() < self.role_explore):
            # символ ни к какой роли не привёл: пробуем случайную
            b.stimulate(self.role_rng.choice(sorted(self.role_of)), 20.0)
        for name in self.decrement:
            if b._fired[b.neurons[name].index]:
                if self.level > 0:
                    self.level -= 1
                elif self.level < 0 and self.signed:
                    self.level += 1
        if not self.signed:
            self.level = max(0, self.level)

    def learn_roles(self, correct):
        """После праздника: использованные связи «символ -> роль» по последствиям."""
        if not self._used:
            return 0
        b = self.brain
        err = self._added - correct
        shift = 0
        for sym, reps, got, used, _ in self._used:
            base = err - got
            opts = {None: abs(base + reps)}
            for r, e in self.role_of.items():
                opts[r] = abs(base + reps * self.units(e))
            best = min(opts.values())
            winners = [r for r, v in opts.items() if v == best]
            # ничья: остаётся объяснение, которое уже работало (сработавшая роль,
            # иначе самая сильная связь); без него — ничья не учит
            win = None
            if used in winners:
                win = used
            elif len(winners) == 1:
                win = winners[0]
            else:
                ws = {r: b.weight[self._role_conn(sym, r)] for r in winners if r is not None}
                top = max(ws.values(), default=0.0)
                if top > 0 and sum(v == top for v in ws.values()) == 1:
                    win = max(ws, key=ws.get)
            if win is not None:
                self._shift(sym, win, +1.0)
                # эту часть ошибки объясняет роль, а не веса
                shift += got - reps * self.units(self.role_of[win])
            if used is not None and used not in winners:
                self._shift(sym, used, -1.0)
        return shift

    def _shift(self, sym, role, sign):
        b = self.brain
        c = self._role_conn(sym, role)
        if c is None:
            return
        reach = b.reach(b.tgt[c])
        b.weight[c] = max(0.0, min(1.5 * reach, b.weight[c] + sign * self.role_step * reach))

    def learn(self, correct, scale=1.0):
        """После праздника: верное число известно, силы входов поправляются.
        scale — множитель обучения (ответственность устройства, Trust.py)."""
        correct = correct + self.learn_roles(correct)
        if self.consistent:
            return self._learn_consistent(correct, scale)
        if self.confidence:
            return self._learn_confident(correct, scale)
        if scale != 1.0:
            old = self.rate
            self.rate = old * scale
            try:
                return self._learn_plain(correct)
            finally:
                self.rate = old
        return self._learn_plain(correct)

    def predict(self, counts, q, w=None, m=None, r=None):
        """Прогноз числа (без округления) для дня с такими счётчиками."""
        w, m, r = w or self.w, m or self.m, r or self.r
        pred = sum(w[k] * n for k, n in counts.items() if k in w)
        if self.register:
            for (i, j), x in q.items():
                pred += m.get(i, 0.0) * r.get(j, 0.0) * x
        return pred

    def answer_of(self, counts, q, w=None, m=None, r=None):
        """Что насчитало бы само устройство (веса округляются, как в работе)."""
        w, m, r = w or self.w, m or self.m, r or self.r
        lv = sum(int(round(w[k])) * n for k, n in counts.items() if k in w)
        if self.register:
            for (i, j), x in q.items():
                lv += int(round(m.get(i, 0.0))) * int(round(r.get(j, 0.0))) * x
        return lv

    def _learn_consistent(self, correct, scale):
        """Ошибка знания или ошибка применения. Поправка — по прежнему правилу
        (или по уверенности, если confidence). Потом она примеряется к памяти
        дней, где устройство было право: берётся наибольшая доля (1, 1/2, 1/4,
        ... 1/64, иначе 0), при которой ни один из этих дней не становится
        неверным (прогноз отходит от верного на 0.5 и больше). Поправку,
        которая ломает проверенное, вызвала не ошибка знания, а то, что знание
        применили не там (или вход не сработал): знание остаётся прежним."""
        counts = {k: v for k, v in self.counts.items() if v}
        q = dict(self._q)
        before = self.predict(counts, q)
        ok_before = self.answer_of(counts, q) == correct
        old = (dict(self.w), dict(self.m), dict(self.r))
        if self.confidence:
            saved_P = (list(self.Pkeys), dict(self._pidx), self.Pm.copy())
            err = self._learn_confident(correct, scale)
        else:
            err = self._learn_plain(correct)
        new = (dict(self.w), dict(self.m), dict(self.r))
        good = [(c, qq, t) for c, qq, t in self.verified
                if self.answer_of(c, qq, *old) == t]
        new = self._project(old, new, good)
        def mix(a):
            return tuple({k: o[k] + a * (n[k] - o[k]) for k in n} for o, n in zip(old, new))
        alpha = 1.0
        while alpha > 1e-9:
            cand = mix(alpha)
            if all(self.answer_of(c, qq, *cand) == t for c, qq, t in good):
                break
            alpha = alpha / 2 if alpha > 1.0 / 64 else 0.0
        self.w, self.m, self.r = mix(alpha)
        if alpha == 0.0 and self.confidence:
            self.Pkeys, self._pidx, self.Pm = saved_P
        self.last_alpha = alpha
        if ok_before:
            self.verified.append((counts, q, correct))
            if len(self.verified) > self.verified_max:
                self.verified.pop(0)
        return err

    def _project(self, old, new, good):
        """Повернуть поправку так, чтобы (в первом приближении) ошибка ни на
        одном проверенном дне не росла (условие метода GEM, parts/gem.py)."""
        import numpy as np
        if not good:
            return new
        w, m, r = old
        keys = [("w", k) for k in self.inputs]
        if self.register:
            keys += [("m", k) for k in self.inputs] + [("r", k) for k in self.inputs]
        idx = {key: i for i, key in enumerate(keys)}
        tab_o = {"w": old[0], "m": old[1], "r": old[2]}
        tab_n = {"w": new[0], "m": new[1], "r": new[2]}
        d = np.array([tab_n[t].get(k, 0.0) - tab_o[t].get(k, 0.0) for t, k in keys])
        G = np.zeros((len(good), len(keys)))
        errs = np.array([self.predict(c, qq, *old) - t for c, qq, t in good])
        for row, (c, qq, _) in enumerate(good):
            for k, n in c.items():
                if ("w", k) in idx:
                    G[row, idx[("w", k)]] += n
            if self.register:
                for (i, j), x in qq.items():
                    G[row, idx[("m", i)]] += r.get(j, 0.0) * x
                    G[row, idx[("r", j)]] += m.get(i, 0.0) * x
        from human_brain.parts.gem import project
        d = project(d, errs[:, None] * G)
        out = (dict(old[0]), dict(old[1]), dict(old[2]))
        tab = {"w": out[0], "m": out[1], "r": out[2]}
        for (t, k), v in zip(keys, d):
            tab[t][k] = tab_o[t].get(k, 0.0) + float(v)
        return out

    def _features(self):
        """(вид, вход, производная прогноза по весу) для участвовавших весов и прогноз."""
        n = self.counts
        feats = [("w", k, float(n[k])) for k in self.inputs if n[k]]
        pred = sum(self.w[k] * n[k] for k in self.inputs)
        if self.register:
            gm = {i: sum(self.r[j] * q for (ii, j), q in self._q.items() if ii == i) for i in self.inputs}
            gr = {j: sum(self.m[i] * q for (i, jj), q in self._q.items() if jj == j) for j in self.inputs}
            pred += sum(self.m[i] * gm[i] for i in self.inputs)
            for k in self.inputs:
                if k in self.seen:
                    continue
                if gm[k]:
                    feats.append(("m", k, gm[k]))
                if gr[k]:
                    feats.append(("r", k, gr[k]))
        return feats, pred

    def _learn_confident(self, correct, scale):
        """Все веса — одна совместная оценка с неуверенностью (фильтр Калмана с
        полной матрицей P: неуверенность каждого веса и связи между ними).
        Поправка = K * ошибка, K = P f / S, S = f P f + conf_r (f — производные
        прогноза по весам). Вес, уже проверенный в днях, где он менялся, а
        ошибки не было, уверен и почти не двигается; ошибку забирает то, чего
        прежние дни не объясняли. Новый вес — неуверенность conf_p0; у
        участвовавших весов неуверенность растёт на conf_q (уверенное может
        медленно переучиться, если ошибка не уходит)."""
        import numpy as np
        feats, pred = self._features()
        if not feats:
            return 0.0
        err = pred - correct
        keys = self.Pkeys
        for t, k, _ in feats:
            if f"{t}:{k}" not in self._pidx:
                n = len(keys)
                keys.append(f"{t}:{k}")
                self._pidx[f"{t}:{k}"] = n
                P = np.zeros((n + 1, n + 1))
                P[:n, :n] = self.Pm
                P[n, n] = self.conf_p0
                self.Pm = P
        f = np.zeros(len(keys))
        for t, k, d in feats:
            f[self._pidx[f"{t}:{k}"]] = d
        Pf = self.Pm @ f
        S = float(f @ Pf) + self.conf_r
        K = Pf / S
        table = {"w": self.w, "m": self.m, "r": self.r}
        for i, key in enumerate(keys):
            if K[i] != 0.0:
                t, k = key.split(":", 1)
                table[t][k] -= scale * K[i] * err
        self.Pm = self.Pm - scale * np.outer(K, Pf)
        self.Pm = 0.5 * (self.Pm + self.Pm.T)
        for t, k, _ in feats:
            i = self._pidx[f"{t}:{k}"]
            self.Pm[i, i] += scale * self.conf_q
        return err

    @property
    def P(self):
        """Неуверенность каждого веса (диагональ P), для разбора."""
        return {k: float(self.Pm[i, i]) for i, k in enumerate(self.Pkeys)}

    def _learn_plain(self, correct):
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
                "counts": self.counts, "Pkeys": self.Pkeys, "Pm": self.Pm.tolist(),
                "verified": [[c, [[i, j, x] for (i, j), x in q.items()], t] for c, q, t in self.verified]}

    @classmethod
    def from_state(cls, brain, state):
        a = cls(brain, **state["params"])
        a.w = dict(state["w"])
        a.m = dict(state.get("m", a.m))
        a.r = dict(state.get("r", a.r))
        a.reg = int(state.get("reg", 0))
        a.level = int(state["level"])
        a.counts = dict(state["counts"])
        a.verified = [(dict(c), {(i, j): x for i, j, x in q}, t) for c, q, t in state.get("verified", [])]
        if "Pkeys" in state:
            import numpy as np
            a.Pkeys = list(state["Pkeys"])
            a._pidx = {k: i for i, k in enumerate(a.Pkeys)}
            a.Pm = np.array(state["Pm"], dtype=float).reshape(len(a.Pkeys), len(a.Pkeys))
        return a
