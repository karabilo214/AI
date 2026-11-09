from kernel import delay


class Myelin:
    """Скорость передачи и её долговременная адаптация.

    Меняется по-разному в зависимости от того, как используется связь:
    - сигнал помог получателю сработать -> слой растёт (тем медленнее, чем он толще);
    - сигнал пришёл впустую -> слой немного тоньше
      (но не если получателя только что затормозил соперник);
    - связь не используется -> слой очень медленно тает.

    Сам расчёт в kernel.py, данные в массивах мозга; этот объект только окно в них.
    """

    THICKNESS = 0.309776
    GROW = 0.024373      # за полезный сигнал
    SHRINK = 0.006042   # за бесполезный сигнал
    FADE = 9.9e-05    # за шаг без сигнала

    # Сигнал пришёл, а получатель промолчал, потому что его только что
    # затормозил соперник: это не бесполезный сигнал, а проигравший.
    # Такой связи миелин не уменьшаем (если PROTECT_INHIBITED).
    # Получатель считается заторможенным SUPPRESS_STEPS шагов после торможения.
    PROTECT_INHIBITED = True
    SUPPRESS_STEPS = 3

    # Миелин по времени (TIMING): подстраивает время доставки под цепь,
    # а не повторяет успех синапса. Сигнал пришёл раньше срабатывания
    # получателя (в пределах TIMING_WINDOW шагов) -> медленнее (SHRINK),
    # после срабатывания -> быстрее (GROW), получатель молчал -> без изменений.
    TIMING = True
    TIMING_WINDOW = 4
    IDLE_STEPS = 501   # при TIMING тает только связь без сигнала столько шагов

    def __init__(self, brain, index):
        self._brain = brain
        self.index = index

    @property
    def thickness(self):
        return float(self._brain.thickness[self.index])

    @thickness.setter
    def thickness(self, value):
        self._brain.thickness[self.index] = value

    def delay(self):
        return int(delay(self._brain.thickness[self.index]))
