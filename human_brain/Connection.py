from Myelin import Myelin
from Synapse import Synapse


class Connection:
    """Связь между двумя нейронами: синапс и миелин (окно в массивы мозга)."""

    def __init__(self, brain, index, source, target):
        self._brain = brain
        self.index = index
        self.source = source
        self.target = target
        self.synapse = Synapse(brain, index)
        self.myelin = Myelin(brain, index)

    @property
    def alive(self):
        return bool(self._brain.alive[self.index])

    @property
    def inhibitory(self):
        return bool(self._brain.inhibitory[self.index])

    @property
    def hypothesis(self):
        """Ещё не подтверждённая гипотеза (немая связь)."""
        return bool(self._brain.hyp[self.index])

    def __repr__(self):
        state = ("" if self.alive else ", удалена") + (
            f", гипотеза: {self._brain.hyp_hits[self.index]} из {self._brain.hyp_tests[self.index]}"
            if self.hypothesis else "")
        if self.inhibitory:
            return f"Connection({self.source}-|{self.target}, тормозит на {self.synapse.weight:.2f}{state})"
        return (f"Connection({self.source}->{self.target}, "
                f"вес {self.synapse.weight:.2f}, задержка {self.myelin.delay()}{state})")
