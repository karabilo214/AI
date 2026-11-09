import copy


class Episode:
    """Один эпизод: какие входы были показаны, какое действие, чем кончилось."""

    def __init__(self, inputs, action, value, time):
        self.inputs = tuple(inputs)   # имена показанных нейронов
        self.action = action          # имя выхода-действия (или None)
        self.value = value            # знак результата: +1 награда, -1 наказание
        self.count = 1                # сколько раз так было
        self.first_time = time
        self.last_time = time
        self.outdated = False         # противоречит более новому опыту
        self.replays = 0              # сколько раз повторён во сне

    @property
    def importance(self):
        return self.count * abs(self.value)

    def __repr__(self):
        res = "награда" if self.value > 0 else "наказание"
        state = ", устарел" if self.outdated else ""
        return (f"Эпизод({'+'.join(self.inputs)} -> {self.action}: {res}, "
                f"было {self.count} раз{state})")


class Episodes:
    """Память эпизодов (как гиппокамп): запись, вспоминание по части, сон.

    Запись: при каждой награде/наказании (brain.reward) запоминается эпизод:
    входы, показанные за последние horizon шагов, выход-действие (explores),
    сработавший последним за это время, и знак результата. Одинаковые
    эпизоды сливаются (растёт счётчик и важность).

    Устаревание (управляемое забывание эпизодов): если в той же ситуации
    (те же входы) новый опыт противоречит старому (то же действие дало
    другой результат, или награду дало другое действие), старый эпизод
    помечается устаревшим и во сне не повторяется. При переполнении
    забываются сначала устаревшие, потом наименее важные.

    Вспоминание: recall(части ситуации) -> самый похожий не устаревший эпизод.

    Сон (выборочное восстановление): sleep(n) проверяет на копии мозга,
    помнит ли он каждый эпизод (даёт ли нужное действие без шума), и
    повторяет до n забытых, начиная с самых важных. Повтор = входы эпизода
    подаются изнутри (без «показа», знакомость не растёт), мозг действует,
    и результат из памяти подкрепляет или наказывает его действие.
    """

    PARAMS = ("horizon", "capacity", "strength")

    def __init__(self, brain, horizon=10, capacity=100, strength=20.0):
        self.brain = brain
        brain.episodes = self
        self.horizon = horizon
        self.capacity = capacity
        self.strength = strength
        self.items = []
        self.replaying = False

    # ---------- запись ----------

    def on_reward(self, value):
        """Вызывается мозгом из brain.reward(value)."""
        if self.replaying or value == 0:
            return
        b = self.brain
        t = b.time
        inputs = sorted(name for name, n in b.neurons.items()
                        if t - b.stim_time[n.index] <= self.horizon and not b.explores[n.index])
        if not inputs:
            return
        action, best = None, -1
        for name, n in b.neurons.items():
            ft = b.fire_time[n.index]
            if b.explores[n.index] and t - ft <= self.horizon and ft > best:
                action, best = name, ft
        self.record(inputs, action, 1 if value > 0 else -1, t)

    def record(self, inputs, action, value, time):
        inputs = tuple(inputs)
        same = None
        for ep in self.items:
            if ep.inputs != inputs:
                continue
            if ep.action == action and ep.value == value:
                same = ep
            elif ((ep.action == action and ep.value != value)
                  or (ep.action != action and ep.value > 0 and value > 0)):
                ep.outdated = True     # новый опыт в той же ситуации противоречит старому
        if same is not None:
            same.count += 1
            same.last_time = time
            same.outdated = False
            return same
        ep = Episode(inputs, action, value, time)
        self.items.append(ep)
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda e: (not e.outdated, e.importance, e.last_time))
            self.items.pop(0)
        return ep

    def active(self):
        return [ep for ep in self.items if not ep.outdated]

    # ---------- вспоминание ----------

    def recall(self, cue):
        """Самый похожий не устаревший эпизод по части ситуации (имена входов)."""
        cue = set(cue)
        best, best_score = None, -1.0
        for ep in self.active():
            inp = set(ep.inputs)
            score = len(cue & inp) / len(cue | inp)
            key = (score, ep.value > 0, ep.importance, ep.last_time)
            if best is None or key > best_key:
                best, best_key, best_score = ep, key, score
        return best if best_score > 0 else None

    # ---------- проверка и повтор ----------

    def _act(self, b, inputs):
        """Подать входы изнутри и вернуть набор выходов-действий, сработавших
        первыми (с первого шага после подачи, за 7 шагов); пустой — никто."""
        for name in inputs:
            b.potential[b.neurons[name].index] += self.strength
        for _ in range(7):
            fired = b.step()
            acts = frozenset(n for n in fired if b.explores[b.neurons[n].index])
            if acts:
                return acts
        return frozenset()

    @staticmethod
    def _done(ep, acts):
        """Выполнен ли эпизод набором действий acts.
        Награда: сделано ровно нужное действие. Наказание: сделано что-то,
        и наказанного действия среди сделанного нет."""
        if ep.value > 0:
            return acts == {ep.action}
        return bool(acts) and ep.action not in acts

    def remembers(self, ep):
        """Помнит ли мозг эпизод (на копии, без шума любопытства)."""
        t = copy.deepcopy(self.brain)
        t.curiosity = None
        t.episodes = None
        return self._done(ep, self._act(t, ep.inputs))

    def replay(self, ep):
        """Повторить эпизод на настоящем мозге (с обучением). Возвращает набор действий."""
        b = self.brain
        self.replaying = True
        act = self._act(b, ep.inputs)
        if ep.value > 0:
            r = 1.0 if act == {ep.action} else (-1.0 if act else 0.0)
        else:
            r = -1.0 if ep.action in act else 0.0
        b.reward(r)
        for _ in range(20):
            b.step()
        ep.replays += 1
        self.replaying = False
        return act

    def sleep(self, n=5):
        """Повторить до n забытых эпизодов, начиная с самых важных. Возвращает число повторов."""
        need = [ep for ep in self.active() if not self.remembers(ep)]
        need.sort(key=lambda e: (-e.importance, -e.last_time))
        for ep in need[:n]:
            self.replay(ep)
        return min(n, len(need))

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "items": [ep.__dict__ for ep in self.items]}

    @classmethod
    def from_state(cls, brain, state):
        e = cls(brain, **state["params"])
        for d in state["items"]:
            ep = Episode(d["inputs"], d["action"], d["value"], d["first_time"])
            ep.__dict__.update(d)
            ep.inputs = tuple(d["inputs"])
            e.items.append(ep)
        return e
