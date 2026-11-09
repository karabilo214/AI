from collections import deque


PAIR_SHARE = 0.6   # доля расстояния до порога на каждый вход нейрона-сочетания


class Growth:
    """Наращивает связи и нейроны по наблюдаемой активности мозга.

    Новая связь X -> Y (связи X -> Y ещё нет). Растёт только то, чего сеть
    ещё не объясняет: срабатывание Y учитывается, только если по связям
    Y получил на этом шаге меньше explain_weight (то есть сеть сама его
    не зажгла). Так не растут обходы (X -> Z -> Y уже есть) и связи от
    общего источника. Дальше строгий отбор, нужны все три:
        1. постоянный промежуток: Y сработал через одно и то же число шагов
           (1..max_gap) после X не меньше connection_threshold раз;
           разные промежутки считаются отдельно;
        2. надёжность: так было хотя бы в min_reliability срабатываний X;
        3. выше случайности: надёжность хотя бы в min_lift раз больше
           доли шагов, на которых Y срабатывает вообще.
        Кандидат, у которого после forget_after срабатываний X надёжность
        ниже нужной, забывается (счёт начнётся заново).
        Связь создаётся слабой, а её задержка равна найденному промежутку,
        чтобы сигнал приходил к моменту срабатывания Y.
        Если hypotheses: связь сначала гипотеза (немая): она проверяется на
        следующих hyp_needed приходах сигнала, то есть на событиях, по которым
        её не выращивали. Подтвердилась (Y срабатывал в >= hyp_min_rel случаев
        и в hyp_lift раз чаще, чем вообще) -> начинает действовать, иначе
        удаляется. Проверку считает kernel.py.
        Это память: молодая связь гибкая (young_plasticity), быстро крепнет
        и быстро слабеет, а набрав силу зажигать Y одна, закрепляется
        (см. Synapse.py). Дальше её судьбу решает обычное обучение.

    Новый нейрон:
        X и Y несколько раз сработали на одном шаге, а нейрона,
        объединяющего эту пару, ещё нет. Создаётся нейрон «X+Y», который
        загорается только когда сигналы от X и Y приходят вместе:
        детектор совпадения (заряд на следующий шаг не переходит, порог не
        меняется), входы закреплённые (не учатся и не удаляются), быстрые
        (задержка 1 шаг), каждый PAIR_SHARE расстояния до порога: один вход
        не зажигает, два вместе зажигают. (Раньше входы были по pair_weight =
        11.1 при пороге 8: зажигал и один родитель; pair_weight больше не
        используется, оставлен для старых наборов параметров.)
        Чтобы новый нейрон на что-то влиял, он встраивается в сеть:
        получает слабые молодые связи туда же, куда ведут возбуждающие
        связи X и Y (inherit_outputs); эти связи быстрые, задержка 1
        (fast_new_outputs), иначе он опаздывал бы. Какие из них нужны, решат
        обычное обучение и награда; ненужные ослабнут и удалятся.

    Журнал: каждое событие (выросла связь, вырос нейрон, удалена связь)
    с причиной пишется в self.log; growth.report() выводит его словами.

    Удаление мёртвой связи:
        сила связи держится ниже dead_weight подряд dead_patience шагов.
        Считается в kernel.py; удалённые связи в brain.removed_connections.
        Если совпадения продолжатся, связь может вырасти заново.
    """

    def __init__(self, brain,
                 connection_threshold=3,
                 min_reliability=0.443501,
                 min_lift=3.328922,
                 forget_after=20,
                 explain_weight=6.153709,
                 neuron_threshold=2,
                 max_gap=4,
                 new_weight=1.452021,
                 young_plasticity=100.0,
                 pair_weight=11.127045,
                 max_neurons=100,
                 inherit_outputs=True,
                 fast_new_outputs=True,
                 hypotheses=True,
                 hyp_needed=5,
                 hyp_min_rel=0.818103,
                 hyp_lift=2.186701,
                 dead_weight=2.0,
                 dead_patience=92,
                 pair_regions=None,
                 sensory_regions=None,
                 inherit_inhibition=False):
        self.brain = brain
        brain.growth = self
        brain.dead_weight = dead_weight      # ниже этой силы связь считается слабой
        brain.dead_patience = dead_patience  # сколько шагов подряд слабой до удаления (0 = не удалять)

        self.connection_threshold = connection_threshold  # сколько совпадений нужно для связи
        self.min_reliability = min_reliability            # доля срабатываний X, за которыми шёл Y
        self.min_lift = min_lift                          # во сколько раз чаще случайного
        self.forget_after = forget_after                  # когда забывать ненадёжного кандидата
        self.explain_weight = explain_weight              # столько по связям = сеть сама зажгла Y
        self.neuron_threshold = neuron_threshold          # сколько совпадений нужно для нейрона
        self.max_gap = max_gap                            # самый большой промежуток X -> Y
        self.new_weight = new_weight                      # сила новой связи (одна не зажигает)
        self.young_plasticity = young_plasticity          # во сколько раз быстрее учится молодая связь
        self.pair_weight = pair_weight                    # сила входа нового нейрона (зажигают вдвоём)
        self.max_neurons = max_neurons
        self.inherit_outputs = inherit_outputs            # встраивать новый нейрон в сеть
        self.fast_new_outputs = fast_new_outputs          # его выходы быстрые (задержка 1)
        self.hypotheses = hypotheses                      # новая связь сначала гипотеза
        self.hyp_needed = hyp_needed                      # сколько проверок до решения
        self.hyp_min_rel = hyp_min_rel                    # доля попаданий для подтверждения
        self.hyp_lift = hyp_lift                          # во сколько раз чаще случайного
        # из каких областей берутся нейроны для сочетаний (None = из любых,
        # включая другие сочетания; иначе только «настоящие» нейроны этих областей)
        self.pair_regions = None if pair_regions is None else sorted(pair_regions)
        # области органов чувств: в их нейроны связи не растут (они сообщают
        # только пришедшее извне; иначе ожидание «после вопроса идёт GO»
        # зажигает GO само и сбивает такт, а DONE -> GO замыкает петлю)
        self.sensory_regions = None if sensory_regions is None else sorted(sensory_regions)
        # новый нейрон-сочетание получает тормозящие связи к действиям, к которым
        # они есть у родителей (с силы 0): тормозить можно по сочетанию, а не по
        # одному входу. False = нет; True = пластичность как у родителя; число =
        # такая пластичность (родительские тогда могут вовсе не учиться)
        self.inherit_inhibition = inherit_inhibition
        brain.hyp_needed = hyp_needed if hypotheses else 0
        brain.hyp_min_rel = hyp_min_rel
        brain.hyp_lift = hyp_lift

        self.history = deque(maxlen=max_gap)  # срабатывания за последние шаги
        self.order_counts = {}   # (X, Y, промежуток) -> [совпадений, срабатываний X до начала счёта]
        self.together_counts = {}  # (X, Y) -> число совместных срабатываний
        self.pair_neurons = {}   # (X, Y) -> имя созданного нейрона
        self.grown_connections = []
        self.grown_neurons = []
        self.log = []            # события роста: {"time", "kind", "text"}

    # ---------- журнал ----------

    def _note(self, kind, text):
        self.log.append({"time": int(self.brain.time), "kind": kind, "text": text})

    def on_removed(self, gone):
        """Вызывается мозгом, когда удалены связи (мёртвые или неподтверждённые гипотезы)."""
        b = self.brain
        for c in gone:
            if c.hypothesis:
                self._note("гипотеза отвергнута",
                           f"{c.source}->{c.target}: {c.target} сработал в "
                           f"{b.hyp_hits[c.index]} из {b.hyp_tests[c.index]} проверок")
            else:
                self._note("удалена связь",
                           f"{c.source}->{c.target}: сила держалась ниже {b.dead_weight} "
                           f"{b.dead_patience} шагов подряд")

    def on_confirmed(self, confirmed):
        """Вызывается мозгом, когда гипотезы подтвердились."""
        b = self.brain
        for c in confirmed:
            self._note("гипотеза подтверждена",
                       f"{c.source}->{c.target}: {c.target} сработал в "
                       f"{b.hyp_hits[c.index]} из {b.hyp_tests[c.index]} проверок, "
                       f"связь начинает действовать")

    def report(self, kinds=None, last=None):
        """Журнал роста словами. kinds: какие события показать; last: сколько последних."""
        events = [e for e in self.log if kinds is None or e["kind"] in kinds]
        if last is not None:
            events = events[-last:]
        lines = [f"шаг {e['time']}: {e['kind']} {e['text']}" for e in events]
        counts = {}
        for e in self.log:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        lines.append("всего: " + (", ".join(f"{k} {v}" for k, v in counts.items()) or "ничего не выросло"))
        return lines

    # ---------- сохранение (вызывается из Brain.save / Brain.load) ----------

    PARAMS = ("connection_threshold", "min_reliability", "min_lift", "forget_after",
              "explain_weight", "neuron_threshold", "max_gap", "new_weight",
              "young_plasticity", "pair_weight", "max_neurons", "inherit_outputs",
              "fast_new_outputs", "hypotheses", "hyp_needed", "hyp_min_rel", "hyp_lift",
              "pair_regions", "sensory_regions", "inherit_inhibition")

    def state(self):
        return {
            "params": {name: getattr(self, name) for name in self.PARAMS},
            "history": [list(step) for step in self.history],
            "order_counts": [[s, t, gap, int(c[0]), int(c[1])]
                             for (s, t, gap), c in self.order_counts.items()],
            "together_counts": [[a, b, int(v)] for (a, b), v in self.together_counts.items()],
            "pair_neurons": [[a, b, name] for (a, b), name in self.pair_neurons.items()],
            "grown_connections": [c.index for c in self.grown_connections],
            "grown_neurons": list(self.grown_neurons),
            "log": list(self.log),
        }

    @classmethod
    def from_state(cls, brain, state):
        growth = cls(brain, dead_weight=brain.dead_weight,
                     dead_patience=brain.dead_patience, **state["params"])
        growth.history.extend(state["history"])
        growth.order_counts = {(s, t, gap): [c0, c1]
                               for s, t, gap, c0, c1 in state["order_counts"]}
        growth.together_counts = {(a, b): v for a, b, v in state["together_counts"]}
        growth.pair_neurons = {(a, b): name for a, b, name in state["pair_neurons"]}
        growth.grown_connections = [brain._all_connections[c]
                                    for c in state["grown_connections"]]
        growth.grown_neurons = list(state["grown_neurons"])
        growth.log = list(state.get("log", []))
        return growth

    def observe(self, fired):
        """Вызывается мозгом в конце каждого шага."""
        brain = self.brain
        # порядок по номеру нейрона (так же считает быстрая версия в fast.py)
        fired = sorted(fired, key=lambda n: brain.neurons[n].index)
        fired_all = fired

        # 1. Кто сработал раньше и кто сейчас: кандидаты на связь
        steps_seen = brain.time + 1
        # срабатывания, которые сеть уже объясняет своими сигналами, не учим
        surprising = [
            n for n in fired
            if brain.received[brain.neurons[n].index] < self.explain_weight
            and not (self.sensory_regions is not None
                     and brain.region_of[brain.neurons[n].index] in self.sensory_regions)
            # в детектор совпадения (сочетание, порядок звуков, нанятая память)
            # связи не растут: его смысл задан входами при рождении, а выросшая
            # связь зажигала бы его без родителей (GO -> «ACC+M_BRING» = 24.3)
            and not brain.coincidence[brain.neurons[n].index]
        ]
        for gap, past in enumerate(reversed(self.history), start=1):
            for source in past:
                source_spikes = brain.spikes[brain.neurons[source].index]
                for target in surprising:
                    if source == target or brain.has_connection(source, target):
                        continue
                    key = (source, target, gap)
                    count = self.order_counts.get(key)
                    if count is None:
                        # счёт начинается с того срабатывания X, которое дало совпадение
                        count = self.order_counts[key] = [0, source_spikes - 1]
                    count[0] += 1

                    source_fires = source_spikes - count[1]
                    reliability = count[0] / source_fires
                    target_rate = brain.spikes[brain.neurons[target].index] / steps_seen

                    if (count[0] >= self.connection_threshold
                            and reliability >= self.min_reliability
                            and reliability >= self.min_lift * target_rate):
                        self._grow_connection(source, target, gap)
                        self._note("выросла гипотеза" if self.hypotheses else "выросла связь",
                                   f"{source}->{target}: {target} {count[0]} раз сработал через {gap} шаг. "
                                   f"после {source} ({100 * reliability:.0f}% срабатываний {source}, "
                                   f"а сам по себе в {100 * target_rate:.1f}% шагов), "
                                   f"сеть этого не объясняла")
                        for g in range(1, self.max_gap + 1):
                            self.order_counts.pop((source, target, g), None)
                    elif source_fires >= self.forget_after and reliability < self.min_reliability:
                        del self.order_counts[key]

        # 2. Кто сработал вместе: кандидаты на новый нейрон
        if self.pair_regions is not None:
            fired = [n for n in fired
                     if brain.region_of[brain.neurons[n].index] in self.pair_regions]
        for i, first in enumerate(fired):
            for second in fired[i + 1:]:
                pair = (first, second)
                if pair in self.pair_neurons:
                    continue
                self.together_counts[pair] = self.together_counts.get(pair, 0) + 1
                if self.together_counts[pair] >= self.neuron_threshold:
                    self._grow_neuron(pair, self.together_counts[pair])
                    del self.together_counts[pair]

        self.history.append(fired_all)

    def _grow_connection(self, source, target, gap):
        # Задержка = round(5 - 4 * толщина), подбираем толщину под промежуток
        thickness = min(1.0, max(0.0, (5.0 - gap) / 4.0))
        connection = self.brain.connect(
            source, target,
            weight=self.new_weight,
            myelin_thickness=thickness,
            plasticity=self.young_plasticity,
            hypothesis=self.hypotheses
        )
        self.grown_connections.append(connection)

    def _grow_neuron(self, pair, together):
        brain = self.brain
        if len(brain.neurons) >= self.max_neurons:
            return
        name = f"{pair[0]}+{pair[1]}"
        k = 2
        while name in brain.neurons:   # разные пары могут дать одно имя: L + R+S1, L+R + S1
            name = f"{pair[0]}+{pair[1]}#{k}"
            k += 1
        # куда вели возбуждающие связи родителей (до появления нового нейрона)
        outputs = {}
        inh_targets = {}
        if self.inherit_inhibition:
            for c in brain.connections:
                if (c.source in pair and c.inhibitory and brain.explores[brain.neurons[c.target].index]
                        and not brain.explores[brain.neurons[c.source].index]):
                    inh_targets[c.target] = (brain.plasticity[c.index]
                                             if self.inherit_inhibition is True
                                             else float(self.inherit_inhibition))
        if self.inherit_outputs:
            for c in brain.connections:
                if (c.source in pair and not c.inhibitory
                        and c.target not in pair and c.target not in outputs):
                    outputs[c.target] = c.myelin.thickness
        neuron = brain.add_neuron(name)
        brain.make_coincidence(neuron.index)
        w_in = PAIR_SHARE * brain.reach(neuron.index)
        for source in pair:
            brain.connect(source, name, weight=w_in, myelin_thickness=1.0, fixed=True)
        self.pair_neurons[pair] = name
        self.grown_neurons.append(name)
        self._note("вырос нейрон",
                   f"{name}: {pair[0]} и {pair[1]} {together} раз сработали на одном шаге; "
                   f"входы от обоих по {w_in:.2f} (загорается только от двух сразу)")
        for target, plast in inh_targets.items():
            # быстрая, как и возбуждающие выходы сочетания (иначе миелин по
            # умолчанию даёт задержку 4, и тормоз приходит после решения)
            brain.connect(name, target, weight=0.0, inhibitory=True, plasticity=plast,
                          myelin_thickness=1.0)
        for target, thickness in outputs.items():
            if self.fast_new_outputs:
                thickness = 1.0   # короткая новая связь: задержка 1 шаг
            c = brain.connect(name, target, weight=self.new_weight,
                              myelin_thickness=thickness, plasticity=self.young_plasticity)
            self.grown_connections.append(c)
            self._note("выросла связь",
                       f"{name}->{target}: новый нейрон встраивается туда, куда вели связи "
                       f"{pair[0]} или {pair[1]} (сила {self.new_weight}, молодая)")
