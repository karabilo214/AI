import numpy as np


class Curiosity:
    """Любопытство: собственный шум мозга, который растёт после неудач.

    Нейроны, добавленные с explores=True (обычно выходы, то есть действия),
    каждый шаг получают случайный шум (нормальный, со средним 0).
    Сила шума = calm + level * (curious - calm), где level от 0 до 1:
    - наказание поднимает level (неудача -> пробовать другое);
    - награда опускает level (получилось -> делать так же);
    - без событий level медленно гаснет (decay за шаг);
    - скука: результат 0 (мозг ничего не сделал) тоже поднимает level на idle,
      иначе бездействие ничем не наказано и мозг может застрять.
    Ловушка молчания (boredom > 0): шума мало, чтобы довести действие до порога,
    если входы его не дотягивают. Поэтому каждый результат 0 подряд (попытка
    ничего не дала) поднимает скуку: bored = 1 - 0.5 ** (число таких попыток
    подряд), и все действия каждый шаг получают постоянный толчок
    boredom * bored мВ. Скуку сбрасывает только результат (награда или
    наказание): действие, после которого ничего не произошло, её не снимает.
    0 = выключено.
    Избегание (avoid > 0, «проиграл — смени»): наказанное действие (последнее
    сработавшее) получает каждый шаг отрицательный толчок avoid мВ, который
    гаснет на avoid_decay за шаг; награда снимает избегание. Иначе привычное
    действие, которое в новой ситуации наказывают, всё равно побеждает, и
    другое ни разу не пробуется (нечему дать награду). 0 = выключено.
    """

    PARAMS = ("calm", "curious", "up", "down", "decay", "idle", "boredom", "avoid",
              "avoid_decay")

    def __init__(self, brain, calm=0.141609, curious=1.434077, up=0.05, down=1.0,
                 decay=0.998041, idle=0.2, seed=0, boredom=0.0, avoid=0.0, avoid_decay=0.97):
        self.brain = brain
        brain.curiosity = self
        self.calm = calm          # шум в спокойном состоянии
        self.curious = curious    # шум при полном любопытстве
        self.up = up              # насколько поднимает наказание
        self.down = down          # насколько опускает награда
        self.decay = decay        # угасание за шаг
        self.idle = idle          # насколько поднимает бездействие (скука)
        self.boredom = boredom    # толчок действиям при полной скуке, мВ за шаг
        self.avoid = avoid        # толчок против наказанного действия, мВ за шаг
        self.avoid_decay = avoid_decay
        self.avoiding = {}        # номер действия -> текущая сила избегания
        self.silent = 0           # тихих попыток подряд
        self.level = 0.0
        self.rng = np.random.default_rng(seed)

    @property
    def noise(self):
        return self.calm + self.level * (self.curious - self.calm)

    def before_step(self):
        """Вызывается мозгом перед каждым шагом."""
        brain = self.brain
        explorers = np.flatnonzero(brain.explores[:brain.n_neurons])
        if len(explorers):
            brain.potential[explorers] += self.rng.normal(0.0, self.noise, len(explorers))
            if self.boredom > 0.0 and self.silent > 0:
                brain.potential[explorers] += self.boredom * (1.0 - 0.5 ** self.silent)
        for i in list(self.avoiding):
            brain.potential[i] -= self.avoiding[i]
            self.avoiding[i] *= self.avoid_decay
            if self.avoiding[i] < 0.01:
                del self.avoiding[i]
        self.level *= self.decay

    def on_reward(self, value):
        """Вызывается мозгом из brain.reward(value)."""
        if value != 0:
            self.silent = 0
        if value < 0:
            self.level += self.up * min(1.0, -value) * (1.0 - self.level)
            if self.avoid > 0.0:
                b = self.brain
                acts = np.flatnonzero(b.explores[:b.n_neurons])
                if len(acts):
                    last = acts[np.argmax(b.fire_time[acts])]
                    if b.time - b.fire_time[last] <= 15:
                        self.avoiding[int(last)] = self.avoid
        elif value > 0:
            self.level -= self.down * min(1.0, value) * self.level
            self.avoiding = {}
        else:
            self.level += self.idle * (1.0 - self.level)
            if self.boredom > 0.0:
                self.silent += 1

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "level": self.level, "silent": self.silent,
                "avoiding": [[i, v] for i, v in self.avoiding.items()],
                "rng": self.rng.bit_generator.state}

    @classmethod
    def from_state(cls, brain, state):
        curiosity = cls(brain, **state["params"])
        curiosity.level = state["level"]
        curiosity.silent = state.get("silent", 0)
        curiosity.avoiding = {int(i): v for i, v in state.get("avoiding", [])}
        curiosity.rng.bit_generator.state = state["rng"]
        return curiosity
