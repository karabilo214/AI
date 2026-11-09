class Hearing:
    """Память слуха: нейроны порядка звуков на линиях задержки (миелин).

    Если мозг min_count раз слышал «x, потом через gap шагов y», в области
    memory_region вырастает нейрон порядка «x→y»:
        x -> «x→y»  медленная связь (задержка gap + 1),
        y -> «x→y»  быстрая связь (задержка 1).
    Сигналы приходят одновременно, только если x прозвучал на gap шагов
    раньше y; каждый по отдельности до порога не дотягивает (0.6 расстояния
    до порога). Нейрон порядка — детектор совпадения: заряд на следующий шаг
    не переходит (иначе два разошедшихся сигнала складывались, и «y, потом x»
    при gap = 1 тоже зажигало нейрон). Для «y, потом x» сигналы разойдутся,
    и нейрон промолчит.

    Проговаривание: «x→y» -> x (задержка 1) и -> y (задержка gap + 1), чтобы
    по картинке слово прозвучало в правильном порядке.

    Все эти связи жёсткие (ассоциации, их не учат обычные правила). В нейрон
    порядка проходят только услышанные звуки, а не проговорённые, поэтому
    по кругу ничего не ходит. Активность памяти слуха считается восприятием
    (область perceptual), и по ассоциациям она связывается, например, с
    картинками (Association(brain, "зрение", "память слуха")).

    Выбор победителя (winner=True, по умолчанию): нейроны порядка — одна группа
    соревнования (brain.pool). Если никто из них не дошёл до порога сам,
    срабатывает получивший больше всех по ассоциациям, если он единственный
    лидер и сигнал пришёл от большинства воспринятого на прошлом шаге (см.
    kernel.choose_winners). Без этого при многих картинках, делящих нейроны
    зрения, сила ассоциаций делится между ними, и картинка -> слово молчит
    (32 зрения, 64 пары: 1/320; с выбором 317/320, capacity.py).
    """

    PARAMS = ("sounds", "memory", "max_gap", "min_count", "winner")

    def __init__(self, brain, sounds="слух", memory="память слуха", max_gap=4, min_count=2,
                 winner=True):
        self.brain = brain
        brain.hearing = self
        self.sounds = sounds
        self.memory = memory
        self.max_gap = max_gap      # не больше 4: задержка связи не больше 5 шагов
        self.min_count = min_count
        # выбор победителя: нейроны порядка соревнуются; при вспоминании (по
        # картинке) срабатывает тот, кому ассоциации дали больше всех (kernel)
        self.winner = winner
        self.heard = {}             # номер звука -> когда последний раз услышан
        self.heard_before = {}      # номер звука -> {шаг: услышан}
        self.counts = {}            # (x, y, gap) -> сколько раз
        self.units = {}             # (x, y, gap) -> имя нейрона порядка
        self.log = []

    def observe(self):
        """Вызывается мозгом в конце каждого шага."""
        b = self.brain
        t = b.time
        now = [i for i, r in enumerate(b.region_of)
               if r == self.sounds and b._fired[i] and b.stim_time[i] == t]
        # нейрон порядка, зажжённый действительно услышанным (x, потом y), = восприятие:
        # помечаем как «услышано сейчас», чтобы ассоциации учились на нём;
        # вызванный изнутри (например, картинкой) восприятием не считается
        for (x, y, gap), name in self.units.items():
            u = b.neurons[name].index
            if (b._fired[u] and self.heard.get(y) == t - 1
                    and self.heard_before.get(x, {}).get(t - 1 - gap)):
                b.stim_time[u] = t
        for y in now:
            for x, tx in list(self.heard.items()):
                gap = t - tx
                if x != y and 1 <= gap <= self.max_gap:
                    key = (x, y, gap)
                    self.counts[key] = self.counts.get(key, 0) + 1
                    if self.counts[key] >= self.min_count and key not in self.units:
                        self._grow(x, y, gap)
        for y in now:
            self.heard[y] = t
            self.heard_before.setdefault(y, {})[t] = True
        for x in [x for x, tx in self.heard.items() if t - tx > self.max_gap + 2]:
            del self.heard[x]
        for x in self.heard_before:
            for old in [k for k in self.heard_before[x] if t - k > self.max_gap + 2]:
                del self.heard_before[x][old]

    def _grow(self, x, y, gap):
        b = self.brain
        nx, ny = b.names[x], b.names[y]
        name = f"{nx}→{ny}" + ("" if gap == 2 else f"/{gap}")
        k = 2
        while name in b.neurons:
            name = f"{nx}→{ny}/{gap}#{k}"
            k += 1
        slow = (5.0 - (gap + 1)) / 4.0                  # миелин для задержки gap + 1
        neuron = b.add_neuron(name, region=self.memory)
        b.make_coincidence(neuron.index)
        if self.winner:
            b.pool[neuron.index] = 0
        reach = b.reach(neuron.index)                   # сколько не хватает до порога
        b.connect(nx, name, weight=0.6 * reach, myelin_thickness=slow, assoc=True)
        b.connect(ny, name, weight=0.6 * reach, myelin_thickness=1.0, assoc=True)
        # проговаривание: чуть больше расстояния до порога самого звука
        b.connect(name, nx, weight=b.reach(x) + 1.0, myelin_thickness=1.0, assoc=True)
        b.connect(name, ny, weight=b.reach(y) + 1.0, myelin_thickness=slow, assoc=True)
        self.units[(x, y, gap)] = name
        self.log.append({"time": int(b.time), "kind": "вырос нейрон порядка",
                         "text": f"{name}: {nx}, потом через {gap} шага {ny} "
                                 f"(слышал {self.counts[(x, y, gap)]} раза)"})

    def report(self):
        return [f"шаг {e['time']}: {e['kind']} {e['text']}" for e in self.log]

    # ---------- сохранение (вызывается из Brain.save / Brain.load) ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "heard": [[x, t] for x, t in self.heard.items()],
                "heard_before": [[x, sorted(d)] for x, d in self.heard_before.items()],
                "counts": [[x, y, g, c] for (x, y, g), c in self.counts.items()],
                "units": [[x, y, g, name] for (x, y, g), name in self.units.items()],
                "log": list(self.log)}

    @classmethod
    def from_state(cls, brain, state):
        params = dict(state["params"])
        params.setdefault("winner", False)   # старые файлы: нейроны порядка без соревнования
        h = cls(brain, **params)
        h.heard = {x: t for x, t in state["heard"]}
        h.heard_before = {x: {t: True for t in ts} for x, ts in state["heard_before"]}
        h.counts = {(x, y, g): c for x, y, g, c in state["counts"]}
        h.units = {(x, y, g): name for x, y, g, name in state["units"]}
        h.log = list(state["log"])
        return h
