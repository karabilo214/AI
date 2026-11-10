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
    Скука по ситуации (situational=True): счётчик тихих попыток свой у каждой
    ситуации — набора нейронов (не действий), горевших за последние window
    шагов. Неудача в ситуации (молчание или наказание) прибавляет её счётчику,
    награда в ней сбрасывает его; толчок действиям — по счётчику той
    ситуации, в которой мозг сейчас, и нарастает (порция за каждый шаг в ней,
    пока ничего не сработало): ситуация длится несколько шагов, а не весь цикл. Иначе скука копится только от молчания
    подряд: если ситуация, где мозг молчит, перемешана с другими, где он
    отвечает, счётчик всё время сбрасывался, и в ней ничего не пробовалось
    (ИЛИ: «ни A, ни B»). Ситуации хранятся таблицей «набор -> счётчик», как
    события у найма детекторов (Recruit).
    """

    PARAMS = ("calm", "curious", "up", "down", "decay", "idle", "boredom", "avoid",
              "avoid_decay", "situational", "window")

    def __init__(self, brain, calm=0.141609, curious=1.434077, up=0.05, down=1.0,
                 decay=0.998041, idle=0.2, seed=0, boredom=0.0, avoid=0.0, avoid_decay=0.97,
                 situational=False, window=10):
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
        self.situational = situational
        self.window = window
        self.silent_by = {}       # ситуация (набор номеров нейронов) -> тихих попыток
        self.gate = None          # имя нейрона-вопроса: поиск только после него (None = всегда)
        self.gate_window = 8
        self._key = None
        self._ramp = 0
        self.level = 0.0
        self.rng = np.random.default_rng(seed)

    @property
    def noise(self):
        return self.calm + self.level * (self.curious - self.calm)

    def before_step(self):
        """Вызывается мозгом перед каждым шагом."""
        brain = self.brain
        explorers = np.flatnonzero(brain.explores[:brain.n_neurons])
        if self.gate is not None:
            # поиск только по вопросу: шум и скука — пока сигнал gate (вопрос)
            # горел не дальше gate_window шагов назад
            gates = [self.gate] if isinstance(self.gate, str) else self.gate
            last = max(brain.fire_time[brain.neurons[g].index] for g in gates)
            if brain.time - last > self.gate_window:
                explorers = explorers[:0]
        if len(explorers):
            brain.potential[explorers] += self.rng.normal(0.0, self.noise, len(explorers))
            n_silent = self.silent
            ramp = 1
            if self.situational:
                key = self.situation()
                n_silent = self.silent_by.get(key, 0)
                # желание попробовать нарастает, пока мозг стоит в ситуации, где
                # раньше молчал: толчок одной порцией за шаг дошёл бы до порога
                # только за ~20 шагов (обычная скука толкает и в паузах), а
                # ситуация длится ~6
                acted = (brain.time - brain.fire_time[explorers] <= 1).any()
                if key != self._key or acted or n_silent == 0:
                    self._ramp = 0
                self._key = key
                self._ramp += 1
                ramp = self._ramp
            if self.boredom > 0.0 and n_silent > 0:
                brain.potential[explorers] += self.boredom * (1.0 - 0.5 ** n_silent) * ramp
        for i in list(self.avoiding):
            brain.potential[i] -= self.avoiding[i]
            self.avoiding[i] *= self.avoid_decay
            if self.avoiding[i] < 0.01:
                del self.avoiding[i]
        self.level *= self.decay

    def situation(self):
        """Нейроны (не действия), горевшие за последние window шагов."""
        b = self.brain
        n = b.n_neurons
        recent = (b.time - b.fire_time[:n] <= self.window) & (b.fire_time[:n] >= 0) & ~b.explores[:n]
        return frozenset(np.flatnonzero(recent).tolist())

    def on_reward(self, value):
        """Вызывается мозгом из brain.reward(value)."""
        if self.situational:
            key = self.situation()
            if key:
                # копится, пока в этой ситуации не получилось: и молчание, и
                # наказание; обнуляет только награда. Иначе «да» (наказано) —
                # молчание (избегание «да») — «да» … : наказание обнуляло
                # счётчик, он не рос выше 1, и «нет» не пробовалось (ИЛИ, пустой
                # случай, знаки-картинки)
                if value > 0:
                    self.silent_by[key] = 0
                elif self.boredom > 0.0:
                    self.silent_by[key] = self.silent_by.get(key, 0) + 1
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
                "silent_by": [[sorted(k), v] for k, v in self.silent_by.items()],
                "rng": self.rng.bit_generator.state}

    @classmethod
    def from_state(cls, brain, state):
        curiosity = cls(brain, **state["params"])
        curiosity.level = state["level"]
        curiosity.silent = state.get("silent", 0)
        curiosity.avoiding = {int(i): v for i, v in state.get("avoiding", [])}
        curiosity.silent_by = {frozenset(k): v for k, v in state.get("silent_by", [])}
        curiosity.rng.bit_generator.state = state["rng"]
        return curiosity
