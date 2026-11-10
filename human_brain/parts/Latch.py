"""Защёлка: врождённое устройство, которое держит, какой был вопрос, до конца дня.

Входы-вопросы (inputs) и их нейроны памяти (outputs, по одному на вопрос).
Сработал вопрос от толчка извне — защёлка запоминает его (последний за день).
Когда показан сигнал «отвечай» (query), загорается нейрон памяти запомненного
вопроса. На ночь (new_day) защёлка пустеет. Сама не учится: что делать с
запомненным вопросом, мозг учит по награде. Нужна потому, что ответ длится
сотни шагов, а следы рабочей памяти гаснут примерно за 30.
"""


class Latch:
    PARAMS = ("inputs", "outputs", "query")

    def __init__(self, brain, inputs, outputs, query):
        self.brain = brain
        brain.latch = self
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.query = query
        self.held = None          # номер запомненного вопроса в inputs или None

    def new_day(self):
        self.held = None

    def before_step(self):
        b = self.brain
        if self.held is not None and b.stim_time[b.neurons[self.query].index] == b.time:
            b.stimulate(self.outputs[self.held], 20.0)

    def observe(self):
        b = self.brain
        for k, name in enumerate(self.inputs):
            i = b.neurons[name].index
            if b._fired[i] and b.stim_time[i] == b.time:
                self.held = k

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS}, "held": self.held}

    @classmethod
    def from_state(cls, brain, state):
        x = cls(brain, **state["params"])
        x.held = state["held"]
        return x
