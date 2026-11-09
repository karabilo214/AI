class Association:
    """Правило «бывает вместе» между двумя группами нейронов (например, зрение и слух).

    Не важно, что пришло раньше и с каким промежутком: если нейрон X одной
    группы и нейрон Y другой были активны в пределах window шагов друг от
    друга, это совместное появление (каждое считается один раз).

    Связи растут в обе стороны (X -> Y и Y -> X), если совместных появлений
    не меньше min_count и они случаются хотя бы в min_lift раз чаще, чем
    случайно. Сила X -> Y = scale * доля появлений X, рядом с которыми был Y
    (для каждого направления свой счёт: в скольких появлениях X был Y, и
    наоборот; поэтому доля никогда не больше 1).
    Нейрон, общий для двух картинок, получает связь вдвое слабее; решает
    сумма: звук загорается, когда горит бо́льшая часть картинки.

    external_only: считаются только срабатывания от внешнего толчка (увидел,
    услышал), а не эхо внутри мозга; иначе эхо по ассоциациям само
    становится «совместным появлением» и связывает всё со всем.

    Сила пересчитывается при каждом новом появлении X: если X всё чаще
    бывает без Y, связь слабеет (до 0). Обычные правила (усиление,
    «опоздал», награда, миелин, удаление) такие связи не трогают.
    Журнал: self.log (выросла / ослабла ассоциация).
    """

    PARAMS = ("window", "min_count", "min_lift", "scale", "external_only")

    def __init__(self, brain, group_a, group_b, window=4, min_count=3, min_lift=1.5, scale=None,
                 external_only=True):
        self.brain = brain
        brain.association = self
        self.group_a = self._resolve(group_a)
        self.group_b = self._resolve(group_b)
        self.window = window
        self.min_count = min_count
        self.min_lift = min_lift
        gap = brain.rules["Neuron"]["THRESHOLD"] - brain.rules["Neuron"]["REST"]
        self.scale = 0.75 * gap if scale is None else scale
        self.external_only = external_only   # учиться только на увиденном/услышанном, не на эхе
        self.episodes = {}      # номер нейрона -> сколько раз появлялся
        self.last = {}          # номер нейрона -> когда последний раз срабатывал
        # (a, b) -> [появлений a, рядом с которыми был b; появлений b, рядом с
        #            которыми был a; последний учтённый эпизод a; эпизод b]
        self.joint = {}
        self.partners = {}      # номер -> множество номеров из другой группы
        self.links = {}         # (источник, цель) -> Connection
        self.log = []

    def _resolve(self, group):
        """Группа: имя области или список имён нейронов -> множество номеров
        (номера пересчитываются при каждом шаге: в области могут расти нейроны)."""
        return group

    def _members(self, group):
        b = self.brain
        if isinstance(group, str):
            return {i for i, r in enumerate(b.region_of) if r == group}
        return {b.neurons[n].index for n in group}

    # ---------- наблюдение ----------

    def observe(self):
        """Вызывается мозгом в конце каждого шага."""
        b = self.brain
        t = b.time
        ga, gb = self._members(self.group_a), self._members(self.group_b)
        self._ga, self._gb = ga, gb
        fired = [i for i in ga | gb if b._fired[i]
                 and (not self.external_only or b.stim_time[i] == t)]
        if not fired:
            return
        new_episode = []
        for i in fired:
            if t - self.last.get(i, -10 ** 9) > self.window:
                self.episodes[i] = self.episodes.get(i, 0) + 1
                new_episode.append(i)
            self.last[i] = t
        touched = set()
        for i in fired:
            other = gb if i in ga else ga
            for j in other:
                if t - self.last.get(j, -10 ** 9) <= self.window:
                    a, bb = (i, j) if i in ga else (j, i)
                    rec = self.joint.setdefault((a, bb), [0, 0, -1, -1])
                    changed = False
                    if rec[2] != self.episodes[a]:      # это появление a ещё не учтено
                        rec[0] += 1
                        rec[2] = self.episodes[a]
                        changed = True
                    if rec[3] != self.episodes[bb]:     # это появление b ещё не учтено
                        rec[1] += 1
                        rec[3] = self.episodes[bb]
                        changed = True
                    if changed:
                        self.partners.setdefault(a, set()).add(bb)
                        self.partners.setdefault(bb, set()).add(a)
                        touched.add((a, bb))
        for i in new_episode:                     # X появился: пересчитать все его связи
            for j in self.partners.get(i, ()):
                touched.add((i, j) if i in ga else (j, i))
        for a, bb in touched:
            self._update(a, bb)

    def _update(self, a, bb):
        b = self.brain
        rec = self.joint[(a, bb)]
        steps = b.time + 1
        for x, y, n_ab in ((a, bb, rec[0]), (bb, a, rec[1])):
            p = n_ab / self.episodes[x]
            chance = min(1.0, self.episodes[y] / steps * (2 * self.window + 1))
            ok = n_ab >= self.min_count and p >= self.min_lift * chance
            w = self.scale * p if ok else 0.0
            conn = self.links.get((x, y))
            if conn is None:
                if ok:
                    conn = b.connect(b.names[x], b.names[y], weight=w, myelin_thickness=1.0,
                                     assoc=True, recall=True)
                    self.links[(x, y)] = conn
                    self.log.append({"time": int(b.time), "kind": "выросла ассоциация",
                                     "text": f"{b.names[x]}->{b.names[y]}: вместе {n_ab} раз "
                                             f"({100 * p:.0f}% появлений {b.names[x]}), сила {w:.1f}"})
            else:
                if w == 0.0 and conn.synapse.weight > 0.0:
                    self.log.append({"time": int(b.time), "kind": "ослабла ассоциация",
                                     "text": f"{b.names[x]}->{b.names[y]}: теперь лишь {100 * p:.0f}%"})
                conn.synapse.weight = w

    def report(self):
        return [f"шаг {e['time']}: {e['kind']} {e['text']}" for e in self.log]

    # ---------- сохранение (вызывается из Brain.save / Brain.load) ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "groups": [self.group_a, self.group_b],
                "episodes": [[i, n] for i, n in self.episodes.items()],
                "last": [[i, t] for i, t in self.last.items()],
                "joint": [[a, b, *rec] for (a, b), rec in self.joint.items()],
                "partners": [[i, sorted(s)] for i, s in self.partners.items()],
                "links": [[x, y, c.index] for (x, y), c in self.links.items()],
                "log": list(self.log)}

    @classmethod
    def from_state(cls, brain, state):
        a = cls(brain, *state["groups"], **state["params"])
        a.episodes = {i: n for i, n in state["episodes"]}
        a.last = {i: t for i, t in state["last"]}
        a.joint = {}
        for row in state["joint"]:
            if len(row) == 5:     # старый формат: один общий счёт на оба направления
                x, y, n, ea, eb = row
                a.joint[(x, y)] = [n, n, ea, eb]
            else:
                x, y, na, nb, ea, eb = row
                a.joint[(x, y)] = [na, nb, ea, eb]
        a.partners = {i: set(s) for i, s in state["partners"]}
        a.links = {(x, y): brain._all_connections[c] for x, y, c in state["links"]}
        a.log = list(state["log"])
        return a
