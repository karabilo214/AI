import numpy as np

from human_brain.core.Neuron import Neuron


class NeuronChange:
    """Разные и меняющиеся нейроны.

    Разные: при подключении (и при добавлении нового нейрона) каждый
    нейрон получает свои случайные отличия:
        порог     = обычный + N(0, threshold_spread)  мВ
        стекание  = обычное + N(0, decay_spread), в пределах 0.85..0.99
        отдых     = обычный + случайно -1, 0 или +1 шаг (если refractory_jitter)

    Меняющиеся (как в мозге, «внутренняя пластичность»): нейрон следит,
    как часто он срабатывает (доля шагов, окно ~1/(1-window) шагов).
    Чаще target -> порог растёт (нейрон менее чувствителен),
    реже -> порог падает (нейрон чувствительнее).
    Скорость: rate мВ за шаг при отклонении доли на 1.
    Порог держится в пределах min_threshold..max_threshold.
    Считается в kernel.py.
    """

    PARAMS = ("threshold_spread", "decay_spread", "refractory_jitter",
              "rate", "target", "window", "min_threshold", "max_threshold", "same_regions")

    def __init__(self, brain,
                 threshold_spread=3.468095,
                 decay_spread=0.01,
                 refractory_jitter=True,
                 rate=0.030059,
                 target=0.043085,
                 window=0.99,
                 min_threshold=-65.0,
                 max_threshold=-45.0,
                 seed=0,
                 vary_existing=True,
                 same_regions=()):
        self.brain = brain
        brain.neuron_change = self
        self.threshold_spread = threshold_spread
        self.decay_spread = decay_spread
        self.refractory_jitter = refractory_jitter
        self.rate = rate
        self.target = target
        self.window = window
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.rng = np.random.default_rng(seed)
        # области, нейроны которых случайного разброса не получают (одинаковые)
        self.same_regions = sorted(same_regions)
        if vary_existing:
            for i in range(brain.n_neurons):
                self.vary(i)

    def vary(self, i):
        """Даёт нейрону i свои случайные отличия."""
        b = self.brain
        if b.coincidence[i]:
            return                   # детектор совпадения: свои стекание и порог
        if self.same_regions and i < len(b.region_of) and b.region_of[i] in self.same_regions:
            b.ip_fixed[i] = True     # одинаковые и не меняются со временем
            return
        t = b.threshold[i] + self.rng.normal(0.0, self.threshold_spread)
        b.threshold[i] = min(self.max_threshold, max(self.min_threshold, t))
        d = b.decay[i] + self.rng.normal(0.0, self.decay_spread)
        b.decay[i] = min(0.99, max(0.85, d))
        if self.refractory_jitter:
            b.refractory_len[i] = max(1, b.refractory_len[i] + int(self.rng.integers(-1, 2)))

    def make_same(self, regions):
        """Нейроны этих областей одинаковые: без случайного разброса (уже
        существующим возвращаются значения области) и без подстройки порога
        от частоты срабатывания."""
        b = self.brain
        self.same_regions = sorted(set(self.same_regions) | set(regions))
        for i in range(b.n_neurons):
            name = b.region_of[i]
            if name in self.same_regions and not b.coincidence[i]:
                b.ip_fixed[i] = True
                reg = b.regions[name]
                rule = b.rules["Neuron"]
                b.threshold[i] = rule["THRESHOLD"] if reg.get("threshold") is None else reg["threshold"]
                b.decay[i] = rule["DECAY"] if reg.get("decay") is None else reg["decay"]
                b.refractory_len[i] = (rule["REFRACTORY"] if reg.get("refractory") is None
                                       else reg["refractory"])

    def describe(self):
        """Чем нейроны отличаются сейчас."""
        b = self.brain
        lines = []
        for name, n in b.neurons.items():
            lines.append(f"{name}: порог {n.threshold:.1f} (обычный {b.rules['Neuron']['THRESHOLD']:.0f}), "
                         f"стекание {n.decay:.3f}, отдых {n.refractory_len}, "
                         f"срабатывает в {100 * n.activity_rate:.1f}% шагов")
        return lines

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "rng": self.rng.bit_generator.state}

    @classmethod
    def from_state(cls, brain, state):
        change = cls(brain, vary_existing=False, **state["params"])
        change.rng.bit_generator.state = state["rng"]
        return change
