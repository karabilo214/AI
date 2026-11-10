
class Synapse:
    """Передача сигнала и синаптическая память, меняющаяся в обе стороны.

    Сигнал пришёл ДО срабатывания получателя (помог) -> связь сильнее.
    Сигнал пришёл ПОСЛЕ срабатывания получателя (опоздал) -> связь слабее.

    Пластичность: множитель скорости обучения. У обычной связи 1.
    Молодая (выросшая) связь начинает с большой пластичностью, быстро
    крепнет и быстро слабеет; когда сила дошла до CONSOLIDATE_WEIGHT
    (хватает, чтобы одной зажечь получателя), связь закрепляется:
    пластичность становится 1.

    Сам расчёт в kernel.py, данные в массивах мозга; этот объект только окно в них.
    """

    STRENGTHEN = 0.022685
    WEAKEN = 0.185019
    TRACE_DECAY = 0.658229
    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 24.303764
    CONSOLIDATE_WEIGHT = 17.655948   # порог (-55) минус покой (-70)

    # Обучение по награде. Связь, которая помогла получателю сработать,
    # получает метку (сколько помогла), метка гаснет на ELIGIBILITY_DECAY
    # за шаг. brain.reward(value) меняет силу на
    # REWARD_RATE * value * пластичность * метка.
    # Награда в 3 раза сильнее обычного усиления (STRENGTHEN): иначе
    # наказанное действие всё равно крепло бы от того, что случилось.
    REWARD_RATE = 1.0
    ELIGIBILITY_DECAY = 0.99

    # Точная заслуга (CREDIT): метка = доля связи в срабатывании получателя;
    # награда идёт только по связям, ведущим к выходу-действию (explores),
    # сработавшему за последние CREDIT_HORIZON шагов. См. kernel.reward.
    CREDIT = True
    # Необходимость (NECESSITY): заслугу получает связь, без которой получатель
    # не сработал бы (остального заряда не хватило бы до порога).
    # Выключено: при выбранных параметрах вход один почти всегда дотягивает
    # до порога, «необходимых» связей нет, и обучение встаёт (B 0.84 -> 0.00).
    NECESSITY = False
    CREDIT_HORIZON = 15

    # Контекстные ветви (CONTEXTS): у связи кроме общей силы есть добавка на
    # каждый контекст; награда в контексте меняет только его добавку.
    # Контекст активен CONTEXT_HORIZON шагов после срабатывания его нейрона
    # (нейроны-контексты: brain.add_neuron(..., context=True)). См. kernel.py.
    CONTEXTS = True
    CONTEXT_HORIZON = 13

    # Управляемое забывание (FORGET): важность связи копится от наград и
    # гаснет на IMPORTANCE_DECAY за шаг; связь с важностью >= IMPORTANT не
    # удаляется как мёртвая. Закреплённая связь, DOUBT_NEEDED раз подряд
    # участвовавшая в наказанном действии, получает пластичность
    # RELEARN_PLASTICITY (можно переучить). См. kernel.reward.
    FORGET = True
    IMPORTANCE_DECAY = 0.9999
    IMPORTANT = 3.0
    DOUBT_NEEDED = 3
    RELEARN_PLASTICITY = 20.0
    # Во сколько раз слабее меняется закреплённая связь. 1.0 = так же: с выбранными
    # параметрами 0.05 ломает обучение (B 0.84 -> 0.32, разворот контекстов 36/50 -> 0/50).
    CONSOLIDATED_RATE = 1.0

    # Защита путей к действиям: связь к выходу-действию (explores) не
    # удаляется и не слабеет ниже ACTION_MIN_WEIGHT (0 = выключено).
    ACTION_MIN_WEIGHT = 1.579161
    # Отрицательная память: ослабленный закреплённый путь к действию меняется
    # от награды в NEG_RATE раз слабее, пока его не освободят подтверждения.
    NEG_RATE = 0.05
    # Сколько наказаний подряд нужно, чтобы путь стал «так нельзя». 0 = сразу.
    # 3 лучше для переучивания (20/20 против 16/20), но ломает задачу B (0.92 -> 0.30).
    NEG_NEEDED = 0
    # Изменчивый запрет (BAN): вместо закрепления «нельзя» у пути к действию
    # уверенность запрета от 0 до 1; наказание повышает (BAN_UP), награда, где
    # путь сам главная причина, снижает (BAN_DOWN). См. kernel.reward.
    BAN = False
    BAN_UP = 0.5
    BAN_DOWN = 0.465469
    BAN_POWER = 7.591039   # рост запрещённого пути умножается на (1 - запрет) ** BAN_POWER

    def __init__(self, brain, index):
        self._brain = brain
        self.index = index

    @property
    def weight(self):
        return float(self._brain.weight[self.index])

    @weight.setter
    def weight(self, value):
        self._brain.weight[self.index] = value

    @property
    def plasticity(self):
        return float(self._brain.plasticity[self.index])

    def context_weight(self, k):
        """Добавка силы в контексте k."""
        return float(self._brain.ctx_w[self.index, k])

    @property
    def importance(self):
        return float(self._brain.importance[self.index])

    @property
    def ban(self):
        """Уверенность запрета «так нельзя» (0..1)."""
        return float(self._brain.ban[self.index])

    @property
    def confirm(self):
        return int(self._brain.confirm[self.index])

    @property
    def doubt(self):
        return int(self._brain.doubt[self.index])

    @property
    def eligibility(self):
        return float(self._brain.eligibility[self.index])

    @property
    def consolidated(self):
        return self.plasticity <= 1.0

    @property
    def pre_trace(self):
        return float(self._brain.pre_trace[self.index])

    @property
    def post_trace(self):
        return float(self._brain.post_trace[self.index])
