
class Neuron:
    """Упрощённая импульсная нервная клетка.

    Заряд стекает к покою. Дошёл до порога -> импульс, сброс
    и несколько шагов отдыха. Порог, скорость стекания и отдых у каждого
    нейрона свои (по умолчанию одинаковые; разными их делает NeuronChange).
    Сам расчёт в kernel.py, данные в массивах мозга; этот объект только окно в них.
    """

    REST = -70.0
    THRESHOLD = -62.0
    DECAY = 0.85
    REFRACTORY = 1

    def __init__(self, brain, index):
        self._brain = brain
        self.index = index

    @property
    def potential(self):
        return float(self._brain.potential[self.index])

    @potential.setter
    def potential(self, value):
        self._brain.potential[self.index] = value

    @property
    def threshold(self):
        return float(self._brain.threshold[self.index])

    @threshold.setter
    def threshold(self, value):
        self._brain.threshold[self.index] = value

    @property
    def refractory(self):
        return int(self._brain.refractory[self.index])

    @property
    def decay(self):
        return float(self._brain.decay[self.index])

    @property
    def refractory_len(self):
        return int(self._brain.refractory_len[self.index])

    @property
    def activity_rate(self):
        """Доля шагов со срабатыванием за последнее время (если подключён NeuronChange)."""
        nc = self._brain.neuron_change
        return float(self._brain.activity[self.index] * (1 - nc.window)) if nc else 0.0

    @property
    def spikes(self):
        return int(self._brain.spikes[self.index])

    def receive(self, signal):
        self._brain.potential[self.index] += signal
