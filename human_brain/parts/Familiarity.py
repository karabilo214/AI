class Familiarity:
    """Знакомость: мозг перестаёт замечать то, что видел много раз без последствий.

    Показ: внешний толчок нейрона (brain.stimulate). С каждым показом
    знакомость нейрона растёт: f += habituation * (1 - f), а толчок действует
    слабее: сила * (1 - max_ignore * f). Знакомое в итоге не доводит нейрон
    до порога: мозг его не замечает и ничему на нём не учится («трава в
    сотый раз»).

    Новость: награда или наказание, или внешний толчок, который сеть не
    предсказала (нейрон сработал, а по связям получил меньше explain_weight).
    Новость снижает знакомость входов, показанных незадолго до неё
    (след показа гаснет на trace_decay за шаг):
        f *= 1 - dishabituation * сила новости * след.
    Так знакомое снова становится интересным, если за ним что-то последовало.
    Когда сеть выучила, что за A следует B, B уже предсказан и новостью не
    считается, и A постепенно становится знакомым.

    Восстановление со временем: каждый шаг знакомость умножается на recovery
    (0.9999: половина уходит примерно за 6900 шагов). Давно не виденное снова
    становится новым и замечается («спонтанное восстановление»).
    """

    PARAMS = ("habituation", "max_ignore", "trace_decay", "dishabituation", "explain_weight",
              "recovery", "one_event")

    def __init__(self, brain, habituation=0.05, max_ignore=0.717818, trace_decay=0.756039,
                 dishabituation=1.0, explain_weight=11.899689, recovery=0.999859,
                 one_event=False):
        self.brain = brain
        brain.familiarity = self
        self.habituation = habituation
        self.max_ignore = max_ignore
        self.trace_decay = trace_decay
        self.dishabituation = dishabituation
        self.explain_weight = explain_weight
        self.recovery = recovery          # знакомость стекает к «новому» за шаг
        self.news_count = 0   # сколько новостей было (для сведения)
        # one_event: всё показанное на одном шаге — одно событие (например,
        # нейроны одной картинки): оно не новость само для себя и не сбрасывает
        # знакомость своих частей; новость считается один раз за шаг.
        # False — как раньше: каждый неожиданный нейрон сбрасывает остальных.
        self.one_event = one_event

    def gain(self, i):
        return 1.0 - self.max_ignore * self.brain.familiar[i]

    def on_stimulate(self, i):
        """Показ нейрона i (вызывается из brain.stimulate после толчка)."""
        b = self.brain
        b.familiar[i] += self.habituation * (1.0 - b.familiar[i])
        b.shown_trace[i] = 1.0
        b.shown_now[i] = True

    def _dishabituate(self, strength, skip=-1, skip_now=False):
        b = self.brain
        for i in range(b.n_neurons):
            if i != skip and not (skip_now and b.shown_now[i]):
                k = min(1.0, self.dishabituation * strength * b.shown_trace[i])
                b.familiar[i] *= 1.0 - k

    def after_step(self):
        """Вызывается мозгом после каждого шага."""
        b = self.brain
        if self.one_event:
            if any(b.shown_now[j] and b._fired[j] and b.received[j] < self.explain_weight
                   for j in range(b.n_neurons)):
                self.news_count += 1
                self._dishabituate(1.0, skip_now=True)
        else:
            for j in range(b.n_neurons):
                if b.shown_now[j] and b._fired[j] and b.received[j] < self.explain_weight:
                    self.news_count += 1
                    self._dishabituate(1.0, skip=j)
        for i in range(b.n_neurons):
            b.shown_trace[i] *= self.trace_decay
            b.shown_now[i] = False
            b.familiar[i] *= self.recovery

    def on_reward(self, value):
        if value != 0:
            self.news_count += 1
            self._dishabituate(abs(value))

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "news_count": self.news_count}

    @classmethod
    def from_state(cls, brain, state):
        f = cls(brain, **state["params"])
        f.news_count = state["news_count"]
        return f
