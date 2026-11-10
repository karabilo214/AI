"""Ожидание награды: обучение по ошибке предсказания, а не по самой награде.

Мозг ждёт награду V по тому, что недавно видел, слышал и помнил:
    признаки  = нейроны областей regions, сработавшие за последние horizon шагов
    ожидание  V = среднее value[i] по этим нейронам (0, если никого)
    ошибка    delta = награда - V
Связи меняются на delta (kernel.reward): ожидаемая награда ничему не учит,
неожиданная учит сильно, а не пришедшая ожидаемая награда (0 при V > 0)
ослабляет то, что к ней вело. Ценности учатся на ту же ошибку:
    value[i] += rate * delta  для каждого признака.
Ожидание зависит от ситуации, а не от действия.

Остальные модули (любопытство, знакомость, эпизоды) получают саму награду.
"""
import numpy as np

from human_brain.core import kernel


class RewardPrediction:
    PARAMS = ("regions", "horizon", "rate")

    def __init__(self, brain, regions, horizon=15, rate=0.2):
        self.brain = brain
        brain.critic = self
        self.regions = sorted(regions)
        self.horizon = horizon
        self.rate = rate
        self.value = {}          # номер нейрона -> ожидаемая награда
        self.last = None         # (ожидание, ошибка) последней награды

    def _features(self):
        b = self.brain
        return [i for i in range(b.n_neurons)
                if b.region_of[i] in self.regions and b.fire_time[i] != kernel.NEVER
                and b.time - b.fire_time[i] <= self.horizon]

    def delta(self, reward):
        """Ошибка предсказания для этой награды; заодно учит ожидание."""
        feats = self._features()
        expect = float(np.mean([self.value.get(i, 0.0) for i in feats])) if feats else 0.0
        d = reward - expect
        for i in feats:
            self.value[i] = self.value.get(i, 0.0) + self.rate * d
        self.last = (expect, d)
        return d

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "value": [[i, v] for i, v in self.value.items()]}

    @classmethod
    def from_state(cls, brain, state):
        r = cls(brain, **state["params"])
        r.value = {i: v for i, v in state["value"]}
        return r
