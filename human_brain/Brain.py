import json

import numpy as np

import kernel
from Neuron import Neuron
from Connection import Connection
from Synapse import Synapse
from Myelin import Myelin

# Массивы, из которых состоит мозг (всё, что он знает и делает)
NEURON_FIELDS = ("potential", "refractory", "spikes", "threshold", "received", "explores",
                 "decay", "refractory_len", "activity", "suppressed",
                 "familiar", "shown_trace", "shown_now", "drive", "fire_time", "ctx_slot",
                 "stim_time", "learn_gain", "relax", "perceptual", "ip_fixed",
                 "ctx_trace", "push_in", "coincidence", "pool")
CONNECTION_FIELDS = ("src", "tgt", "weight", "pre_trace", "post_trace", "thickness",
                     "alive", "weak_steps", "pending", "plasticity", "eligibility",
                     "inhibitory", "hyp", "hyp_tests", "hyp_hits", "arr_time", "ctx_w",
                     "importance", "doubt", "confirm", "last_share", "ban", "w0", "assoc",
                     "fixed", "recall", "block_elig")

# Правила (константы классов), которые мозг запоминает при создании
RULE_CLASSES = {"Neuron": Neuron, "Synapse": Synapse, "Myelin": Myelin}


def current_rules():
    """Снимок правил Neuron, Synapse, Myelin (то, что выставил tasks.apply):
    {"Neuron": {"REST": ...}, "Synapse": {...}, "Myelin": {...}}."""
    return {cname: {key: getattr(cls, key) for key in vars(cls) if key.isupper()}
            for cname, cls in RULE_CLASSES.items()}


def _grow_capacity(capacity):
    return max(1, 2 * capacity)


def _resize(array, capacity):
    new = np.zeros((capacity,) + array.shape[1:], dtype=array.dtype)
    new[:len(array)] = array
    return new


class Brain:
    """Управляет нейронами, связями и течением времени.

    Данные лежат в массивах, шаг считается на Numba (kernel.py).
    Выученное сохраняется в файл: brain.save(path), Brain.load(path).

    Правила (константы Neuron, Synapse, Myelin) мозг запоминает при создании
    в self.rules и дальше читает только их: tasks.apply для другого опыта
    этот мозг не меняет, а save/load хранят правила вместе с мозгом.
    """

    def __init__(self, neuron_capacity=16, connection_capacity=64, rules=None):
        if neuron_capacity < 0 or connection_capacity < 0:
            raise ValueError("ёмкость не может быть отрицательной")
        self.rules = current_rules() if rules is None else {k: dict(v) for k, v in rules.items()}
        self.neurons = {}        # имя -> Neuron
        self.names = []          # номер -> имя
        self.connections = []    # живые связи
        self.removed_connections = []
        self._all_connections = []   # номер -> Connection (и удалённые)
        self._pair_count = {}        # (источник, цель) -> число живых связей

        self.n_neurons = 0
        self.potential = np.zeros(neuron_capacity)
        self.refractory = np.zeros(neuron_capacity, dtype=np.int64)
        self.spikes = np.zeros(neuron_capacity, dtype=np.int64)
        self.threshold = np.zeros(neuron_capacity)
        self.received = np.zeros(neuron_capacity)  # получено по связям на последнем шаге
        self.explores = np.zeros(neuron_capacity, dtype=np.bool_)  # получает шум любопытства
        self.decay = np.zeros(neuron_capacity)                     # своя скорость стекания заряда
        self.refractory_len = np.zeros(neuron_capacity, dtype=np.int64)  # свой отдых после импульса
        self.activity = np.zeros(neuron_capacity)                  # след недавних срабатываний
        self.suppressed = np.zeros(neuron_capacity, dtype=np.int64)  # шагов до конца торможения
        self.familiar = np.zeros(neuron_capacity)        # знакомость (0 новое .. 1 приелось)
        self.shown_trace = np.zeros(neuron_capacity)     # след недавнего показа
        self.shown_now = np.zeros(neuron_capacity, dtype=np.bool_)  # показан на этом шаге
        self.drive = np.zeros(neuron_capacity)           # заряд перед последним импульсом
        self.fire_time = np.full(neuron_capacity, kernel.NEVER, dtype=np.int64)  # когда срабатывал
        self._relevance = np.zeros(neuron_capacity)      # значимость для награды (рабочий)
        self.ctx_slot = np.full(neuron_capacity, -1, dtype=np.int64)  # номер контекста или -1
        self.stim_time = np.full(neuron_capacity, kernel.NEVER, dtype=np.int64)  # когда толкнули снаружи
        self.learn_gain = np.ones(neuron_capacity)   # множитель обучения входящих связей (область)
        self.relax = np.zeros(neuron_capacity)       # возврат входящих связей к исходной силе (область)
        self.perceptual = np.zeros(neuron_capacity, dtype=np.bool_)  # его активность = восприятие
        self.ip_fixed = np.zeros(neuron_capacity, dtype=np.bool_)    # порог не подстраивается
        # детектор совпадения: заряд не переходит на следующий шаг (стекание 0),
        # порог не меняется; NeuronChange его не трогает
        self.coincidence = np.zeros(neuron_capacity, dtype=np.bool_)
        # группа соревнования (выбор победителя), -1 = не участвует
        self.pool = np.full(neuron_capacity, -1, dtype=np.int64)
        self.recall_in = np.zeros(neuron_capacity)   # получено по связям вспоминания (рабочий)
        self.regions = {}                            # имя области -> её свойства
        self.region_of = []                          # номер нейрона -> имя области или None
        self.n_contexts = 0
        self._active = np.zeros(kernel.MAX_CONTEXTS, dtype=np.bool_)
        self._ctx_level = np.zeros(kernel.MAX_CONTEXTS)
        self.ctx_trace = np.zeros(neuron_capacity)   # след активности нейрона-контекста
        self.push_in = np.zeros(neuron_capacity)     # внешний толчок за этот шаг
        # Контекстная модуляция (плавные контексты): затухание следа нейрона-
        # контекста за шаг (0 = обычные контексты «включён/выключен»).
        self.ctx_graded = 0.0
        # Подстройка порога по полученному толчку, а не по срабатыванию.
        self.ip_push = False
        # Нижняя граница заряда при торможении (-1e18 = без границы).
        self.v_floor = -1e18
        # Правки разбора кода (по умолчанию выключены, включает голова):
        self.inh_short = False          # торможение не оставляет заряда ниже покоя
        self.myelin_keep = False        # сигнал вовремя поддерживает миелин
        self.last_action_only = False   # новое действие стирает метки прошлого
        self.decision_ctx = False       # награда учит контекст момента решения
        self.reset_eligibility = False  # награда сбрасывает метки
        # запас входов считать сверх минимума путей к действиям (иначе минимум
        # возвращает силу, которую запас снимает, и сильный путь тает); меняет
        # основные задачи (B 0.92 -> 0.86), поэтому по умолчанию выключено
        self.budget_above_floor = False
        # обучаемое торможение приходит с задержкой, как возбуждение (kernel)
        self.inh_delayed = False
        # связи к действиям меняются только по награде (kernel)
        self.reward_only_actions = False
        # метка обучаемого торможения делится между участниками (kernel)
        self.inh_share = False
        # метка «помешал» по всему торможению, по долям (kernel)
        self.block_collective = False
        self._dec_level = np.zeros(kernel.MAX_CONTEXTS)
        self._dec_active = np.zeros(kernel.MAX_CONTEXTS, dtype=np.bool_)
        self._dec_n = np.zeros(1, dtype=np.int64)
        self.critic = None   # ожидание награды (RewardPrediction), подключается само
        self._syn_drive = np.zeros(neuron_capacity)
        self._fired = np.zeros(neuron_capacity, dtype=np.bool_)

        self.n_conn = 0
        self.src = np.zeros(connection_capacity, dtype=np.int64)
        self.tgt = np.zeros(connection_capacity, dtype=np.int64)
        self.weight = np.zeros(connection_capacity)
        self.pre_trace = np.zeros(connection_capacity)
        self.post_trace = np.zeros(connection_capacity)
        self.thickness = np.zeros(connection_capacity)
        self.alive = np.zeros(connection_capacity, dtype=np.bool_)
        self.weak_steps = np.zeros(connection_capacity, dtype=np.int64)
        self.pending = np.zeros((connection_capacity, kernel.RING), dtype=np.int64)
        self.plasticity = np.ones(connection_capacity)
        self.eligibility = np.zeros(connection_capacity)  # метка для награды
        self.inhibitory = np.zeros(connection_capacity, dtype=np.bool_)  # тормозящая связь
        self.n_inhibitory = 0
        self.hyp = np.zeros(connection_capacity, dtype=np.bool_)      # связь-гипотеза (немая)
        self.hyp_tests = np.zeros(connection_capacity, dtype=np.int64)  # сколько раз проверена
        self.hyp_hits = np.zeros(connection_capacity, dtype=np.int64)   # сколько раз подтвердилась
        self.arr_time = np.full(connection_capacity, kernel.NEVER, dtype=np.int64)  # когда пришёл сигнал
        self._confirmed = np.zeros(connection_capacity, dtype=np.int64)
        self.ctx_w = np.zeros((connection_capacity, kernel.MAX_CONTEXTS))  # добавки по контекстам
        self.importance = np.zeros(connection_capacity)  # важность (сколько награды принесла)
        self.doubt = np.zeros(connection_capacity, dtype=np.int64)  # наказаний подряд
        self.confirm = np.zeros(connection_capacity, dtype=np.int64)  # подтверждений подряд
        self.last_share = np.zeros(connection_capacity)  # доля среди сигналов при срабатывании
        self.ban = np.zeros(connection_capacity)  # уверенность запрета «так нельзя»
        self.w0 = np.zeros(connection_capacity)   # исходная сила связи
        self.assoc = np.zeros(connection_capacity, dtype=np.bool_)  # связь «бывает вместе»
        self.fixed = np.zeros(connection_capacity, dtype=np.bool_)  # закреплённый вход детектора
        self.recall = np.zeros(connection_capacity, dtype=np.bool_)  # связь вспоминания
        self.block_elig = np.zeros(connection_capacity)  # метка «помешал» (торможение)
        self._delivered = np.zeros(connection_capacity, dtype=np.bool_)
        self._removed = np.zeros(connection_capacity, dtype=np.int64)

        # Удаление мёртвых связей: включает Growth (0 = выключено)
        self.dead_weight = 0.5
        self.dead_patience = 0

        # Общий запас силы входов нейрона: сумма сил возбуждающих входов
        # не больше input_budget; рост одного входа отнимает у остальных.
        # 0 = выключено. Разумное значение 30 (= самая большая сила одной связи).
        self.input_budget = 40.811013
        # Запас считать только на входы, сигнал по которым пришёл за последние
        # budget_window шагов (0 = на все входы). Включает голова.
        self.budget_window = 0
        # Обучаемое торможение вход/память -> действие (0 = выключено):
        # метка, если действие сработало не позже inh_window шагов после
        # источника торможения; наказание усиливает, награда ослабляет.
        self.inh_window = 0

        # Гипотезы: сколько проверок нужно и какой результат подтверждает
        # (0 = гипотез нет; включает Growth). См. kernel.py.
        self.hyp_needed = 0
        self.hyp_min_rel = 0.6
        self.hyp_lift = 2.0
        # Лишнее срезать сначала с молодых связей, закреплённые беречь
        self.protect_consolidated = True
        self._input_total = np.zeros(neuron_capacity)

        self.time = 0
        self.growth = None      # подключается классом Growth
        self.curiosity = None   # подключается классом Curiosity
        self.neuron_change = None  # подключается классом NeuronChange
        self.familiarity = None    # подключается классом Familiarity
        self.episodes = None       # подключается классом Episodes
        self.association = None    # подключается классом Association
        self.hearing = None        # подключается классом Hearing (память слуха)
        self.recruit = None        # подключается классом Recruit (найм памяти)
        self.accumulator = None    # подключается классом Accumulator (накопитель)
        self.latch = None          # подключается классом Latch (защёлка вопроса)
        self.units = None          # подключается классом Units (найм заготовок)

    # ---------- строение ----------

    def add_region(self, name, threshold=None, decay=None, refractory=None,
                   learn=1.0, relax=0.0, perceptual=False):
        """Область: свойства её нейронов и обучения их входящих связей.

        threshold, decay, refractory: порог, стекание заряда, отдых (None = обычные);
        learn: множитель обучения входящих связей (0 = жёсткие связи);
        relax: доля за шаг, на которую сила входящей связи возвращается к исходной
        (больше 0 = быстрая память, которая сама забывает);
        perceptual: активность её нейронов считается восприятием (по ассоциациям
        от неё сигнал идёт), как у увиденного/услышанного.
        """
        self.regions[name] = dict(threshold=threshold, decay=decay, refractory=refractory,
                                  learn=learn, relax=relax, perceptual=perceptual)

    def add_neuron(self, name, threshold=None, explores=False, decay=None, refractory=None,
                   context=False, region=None):
        """explores=True: нейрон получает шум любопытства (обычно выходы-действия).

        Если подключён NeuronChange, нейрон получает свои случайные отличия.
        context=True: нейрон-контекст (см. Synapse.CONTEXTS), не больше
        kernel.MAX_CONTEXTS.
        """
        if name in self.neurons:
            raise ValueError(f"Нейрон {name} уже есть")
        if self.n_neurons == len(self.potential):
            capacity = _grow_capacity(len(self.potential))
            for field in NEURON_FIELDS + ("_fired", "_input_total", "_relevance", "_syn_drive",
                                          "recall_in"):
                setattr(self, field, _resize(getattr(self, field), capacity))

        # значения по умолчанию: из области, иначе из Neuron в момент вызова
        reg = self.regions[region] if region is not None else {}
        threshold = reg.get("threshold") if threshold is None else threshold
        decay = reg.get("decay") if decay is None else decay
        refractory = reg.get("refractory") if refractory is None else refractory
        threshold = self.rules["Neuron"]["THRESHOLD"] if threshold is None else threshold
        decay = self.rules["Neuron"]["DECAY"] if decay is None else decay
        refractory = self.rules["Neuron"]["REFRACTORY"] if refractory is None else refractory
        i = self.n_neurons
        self.potential[i] = self.rules["Neuron"]["REST"]
        self.refractory[i] = 0
        self.spikes[i] = 0
        self.threshold[i] = threshold
        self.explores[i] = explores
        self.decay[i] = decay
        self.refractory_len[i] = refractory
        self.activity[i] = 0.0
        self.drive[i] = 0.0
        self.fire_time[i] = kernel.NEVER
        self.ctx_slot[i] = -1
        self.stim_time[i] = kernel.NEVER
        self.learn_gain[i] = reg.get("learn", 1.0)
        self.relax[i] = reg.get("relax", 0.0)
        self.perceptual[i] = reg.get("perceptual", False)
        self.ip_fixed[i] = False
        self.coincidence[i] = False
        self.pool[i] = -1
        if context:
            if self.n_contexts >= kernel.MAX_CONTEXTS:
                raise ValueError("слишком много контекстов")
            self.ctx_slot[i] = self.n_contexts
            self.n_contexts += 1
        self.n_neurons += 1
        neuron = self._register_neuron(name, i, region)
        if self.neuron_change is not None:
            self.neuron_change.vary(i)
        return neuron

    @property
    def rest(self):
        return self.rules["Neuron"]["REST"]

    def reach(self, i):
        """Сколько нейрону i не хватает от покоя до порога."""
        return float(self.threshold[i] - self.rules["Neuron"]["REST"])

    def make_coincidence(self, i):
        """Нейрон i становится детектором совпадения: срабатывает только от
        сигналов, пришедших на одном шаге (заряд на следующий шаг не
        переходит), порог не подстраивается."""
        self.decay[i] = 0.0
        self.ip_fixed[i] = True
        self.coincidence[i] = True
        self.potential[i] = self.rules["Neuron"]["REST"]

    def _register_neuron(self, name, i, region=None):
        neuron = Neuron(self, i)
        self.region_of.append(region)
        self.neurons[name] = neuron
        self.names.append(name)
        return neuron

    def connect(self, source, target, weight=18.0, myelin_thickness=None,
                plasticity=1.0, inhibitory=False, hypothesis=False, assoc=False, fixed=False,
                recall=False):
        """Связь source -> target.

        inhibitory=True: тормозящая связь. Действует в тот же шаг, когда
        source сработал: снижает заряд target на weight. Не учится и не удаляется.
        hypothesis=True: связь-гипотеза: пока не подтвердится проверками,
        сигнал по ней не меняет заряд target (см. kernel.py). Нужен подключённый
        Growth (он задаёт правила проверки).
        fixed=True: закреплённый вход детектора (не учится, не удаляется).
        recall=True: связь вспоминания: её вход считается при выборе победителя.
        """
        if source not in self.neurons or target not in self.neurons:
            raise ValueError("Неизвестный нейрон")
        if hypothesis and self.hyp_needed <= 0:
            raise ValueError("связь-гипотеза требует правил проверки: подключите Growth "
                             "с hypotheses=True (brain.hyp_needed > 0)")
        if fixed and (hypothesis or inhibitory or assoc):
            raise ValueError("закреплённой может быть только обычная возбуждающая связь")
        if self.n_conn == len(self.weight):
            capacity = _grow_capacity(len(self.weight))
            for field in CONNECTION_FIELDS + ("_delivered", "_removed", "_confirmed"):
                setattr(self, field, _resize(getattr(self, field), capacity))

        if myelin_thickness is None:
            myelin_thickness = self.rules["Myelin"]["THICKNESS"]
        c = self.n_conn
        self.src[c] = self.neurons[source].index
        self.tgt[c] = self.neurons[target].index
        self.weight[c] = weight
        self.pre_trace[c] = 0.0
        self.post_trace[c] = 0.0
        self.thickness[c] = myelin_thickness
        self.alive[c] = True
        self.weak_steps[c] = 0
        self.pending[c, :] = 0
        self.plasticity[c] = plasticity
        self.eligibility[c] = 0.0
        self.inhibitory[c] = inhibitory
        self.hyp[c] = hypothesis
        self.hyp_tests[c] = 0
        self.hyp_hits[c] = 0
        self.arr_time[c] = kernel.NEVER
        self.ctx_w[c, :] = 0.0
        self.importance[c] = 0.0
        self.doubt[c] = 0
        self.confirm[c] = 0
        self.last_share[c] = 0.0
        self.ban[c] = 0.0
        self.w0[c] = weight
        self.assoc[c] = assoc
        self.fixed[c] = fixed
        self.recall[c] = recall
        self.block_elig[c] = 0.0
        if inhibitory:
            self.n_inhibitory += 1
        self.n_conn += 1
        return self._register_connection(c)

    def _register_connection(self, c):
        source, target = self.names[self.src[c]], self.names[self.tgt[c]]
        connection = Connection(self, c, source, target)
        self._all_connections.append(connection)
        if self.alive[c]:
            self.connections.append(connection)
            pair = (source, target)
            self._pair_count[pair] = self._pair_count.get(pair, 0) + 1
        else:
            self.removed_connections.append(connection)
        return connection

    def has_connection(self, source, target):
        return self._pair_count.get((source, target), 0) > 0

    # ---------- работа ----------

    def stimulate(self, name, strength):
        """Воздействие извне на нейрон (показ).

        Если подключена Familiarity, знакомое действует слабее.
        """
        i = self.neurons[name].index
        self.stim_time[i] = self.time
        if self.familiarity is not None:
            push = strength * self.familiarity.gain(i)
            self.familiarity.on_stimulate(i)
        else:
            push = strength
        self.potential[i] += push
        self.push_in[i] += push

    def reward(self, value):
        """Награда (value > 0) или наказание (value < 0).

        Меняет связи, которые недавно помогли своим получателям сработать
        (см. Synapse.REWARD_RATE и ELIGIBILITY_DECAY).
        """
        # с ожиданием награды связи учатся на ошибку предсказания
        learn = self.critic.delta(value) if self.critic is not None else value
        S = self.rules["Synapse"]
        kernel.reward(
            float(learn), float(value), self.n_neurons, self.n_conn, self.src, self.tgt,
            self.weight, self.eligibility, self.plasticity, self.alive,
            self.inhibitory, self.explores, self.fire_time, self.time,
            bool(S["CREDIT"]), int(S["CREDIT_HORIZON"]), self._relevance,
            float(S["REWARD_RATE"]), float(S["MIN_WEIGHT"]), float(S["MAX_WEIGHT"]),
            float(S["CONSOLIDATE_WEIGHT"]),
            self._n_ctx(), self.ctx_slot, self.ctx_w, int(S["CONTEXT_HORIZON"]), self._active,
            self._ctx_level, float(self.ctx_graded), self.ctx_trace,
            bool(S["FORGET"]), self.importance, self.doubt, int(S["DOUBT_NEEDED"]),
            float(S["RELEARN_PLASTICITY"]), float(S["CONSOLIDATED_RATE"]),
            self.last_share, self.confirm, float(S["NEG_RATE"]),
            bool(S["BAN"]), self.ban, float(S["BAN_UP"]), float(S["BAN_DOWN"]),
            float(S["BAN_POWER"]), self.learn_gain,
            bool(self.decision_ctx), self._dec_level, self._dec_active, self._dec_n,
            self.block_elig,
        )
        if self.reset_eligibility:
            # награда «съела» метки: следующая награда не учит прошлую попытку
            self.eligibility[:self.n_conn] = 0.0
            self.block_elig[:self.n_conn] = 0.0
        if self.input_budget > 0:
            kernel.limit_inputs(self.n_neurons, self.n_conn, self.tgt, self.weight,
                                self.alive, self.inhibitory, float(self.input_budget),
                                self._input_total, self.plasticity,
                                bool(self.protect_consolidated), self.hyp, self.assoc,
                                int(self.budget_window), self.time, self.arr_time, self.fixed,
                                self._budget_floor(), self.explores)
        if self.curiosity is not None:
            self.curiosity.on_reward(value)
        if self.familiarity is not None:
            self.familiarity.on_reward(value)
        if self.episodes is not None:
            self.episodes.on_reward(value)

    def step(self):
        """Один временной шаг работы мозга."""
        if self.curiosity is not None:
            self.curiosity.before_step()
        if self.accumulator is not None:
            self.accumulator.before_step()
        if self.latch is not None:
            self.latch.before_step()
        if self.units is not None:
            self.units.before_step()
        S, M = self.rules["Synapse"], self.rules["Myelin"]
        _, n_removed, n_confirmed = kernel.step(
            self.time,
            self.n_neurons, self.potential, self.refractory, self.spikes,
            self.threshold, float(self.rules["Neuron"]["REST"]), self.decay, self.refractory_len,
            self.activity, *self._neuron_change_params(), self.ip_fixed,
            bool(self.ip_push), self.push_in, float(self.v_floor),
            bool(self.inh_short), bool(self.last_action_only), bool(self.myelin_keep),
            bool(self.decision_ctx), self._dec_level, self._dec_active, self._dec_n,
            self.n_conn, self.src, self.tgt, self.weight,
            self.pre_trace, self.post_trace, self.thickness,
            self.alive, self.weak_steps, self.pending, self.plasticity,
            self.eligibility, self.inhibitory, self.n_inhibitory > 0,
            self.suppressed, int(M["SUPPRESS_STEPS"]), bool(M["PROTECT_INHIBITED"]),
            float(S["STRENGTHEN"]), float(S["WEAKEN"]), float(S["TRACE_DECAY"]),
            float(S["MIN_WEIGHT"]), float(S["MAX_WEIGHT"]), float(S["CONSOLIDATE_WEIGHT"]),
            float(S["ELIGIBILITY_DECAY"]),
            float(M["GROW"]), float(M["SHRINK"]), float(M["FADE"]),
            self.dead_weight, self.dead_patience,
            float(self.input_budget), self._input_total, bool(self.protect_consolidated),
            int(self.budget_window), int(self.inh_window),
            self.hyp, self.hyp_tests, self.hyp_hits, int(self.hyp_needed),
            float(self.hyp_min_rel), float(self.hyp_lift),
            self.arr_time, self.drive, self.fire_time,
            bool(S["CREDIT"]), bool(M["TIMING"]), int(M["TIMING_WINDOW"]),
            int(M["IDLE_STEPS"]),
            self._n_ctx(), self.ctx_slot, self.ctx_w, int(S["CONTEXT_HORIZON"]), self._active,
            self._ctx_level, float(self.ctx_graded), self.ctx_trace,
            self.importance, *self._forget_step_params(), self.doubt,
            self.explores, float(S["ACTION_MIN_WEIGHT"]), self.last_share, self.confirm,
            self._syn_drive, int(S["NEG_NEEDED"]), bool(S["NECESSITY"]), bool(S["BAN"]),
            self.ban, float(S["BAN_POWER"]),
            self.learn_gain, self.relax, self.w0, self.assoc, self.stim_time, self.perceptual,
            self.fixed,
            bool((self.pool[:self.n_neurons] >= 0).any()), self.pool, self.recall, self.recall_in,
            self._budget_floor(), bool(self.inh_delayed), self.block_elig,
            bool(self.reward_only_actions), bool(self.inh_share), bool(self.block_collective),
            self._fired, self._delivered, self._removed, self.received, self._confirmed,
        )
        if n_confirmed and self.growth is not None:
            self.growth.on_confirmed([self._all_connections[c]
                                      for c in self._confirmed[:n_confirmed]])

        # Убираем удалённые связи из списков
        if n_removed:
            gone = [self._all_connections[c] for c in self._removed[:n_removed]]
            gone_ids = {id(x) for x in gone}
            self.connections = [x for x in self.connections if id(x) not in gone_ids]
            for connection in gone:
                self._pair_count[(connection.source, connection.target)] -= 1
            self.removed_connections.extend(gone)
            if self.growth is not None:
                self.growth.on_removed(gone)

        if self.familiarity is not None:
            self.familiarity.after_step()

        fired = {self.names[i] for i in np.flatnonzero(self._fired[:self.n_neurons])}

        # Рост новых связей и нейронов, ассоциации
        if self.growth is not None:
            self.growth.observe(fired)
        if self.hearing is not None:
            self.hearing.observe()
        if self.association is not None:
            self.association.observe()
        if self.recruit is not None:
            self.recruit.observe()
        if self.accumulator is not None:
            self.accumulator.observe()
        if self.latch is not None:
            self.latch.observe()
        if self.units is not None:
            self.units.observe()

        self.time += 1
        return fired

    def _budget_floor(self):
        return float(self.rules["Synapse"]["ACTION_MIN_WEIGHT"]) if self.budget_above_floor else 0.0

    def _forget_step_params(self):
        if self.rules["Synapse"]["FORGET"]:
            return float(self.rules["Synapse"]["IMPORTANCE_DECAY"]), float(self.rules["Synapse"]["IMPORTANT"])
        return 1.0, 1e18   # без забывания: важность не гаснет и от удаления не защищает

    def _n_ctx(self):
        return self.n_contexts if self.rules["Synapse"]["CONTEXTS"] else 0

    def _neuron_change_params(self):
        nc = self.neuron_change
        if nc is None:
            return 0.0, 0.0, 0.0, -1e9, 1e9
        return nc.rate, nc.target, nc.window, nc.min_threshold, nc.max_threshold

    def run(self, steps, verbose=True):
        for _ in range(steps):
            fired = self.step()
            if fired and verbose:
                print(f"Время {self.time - 1}: {sorted(fired)}")

    # ---------- сохранение ----------

    def save(self, path):
        """Сохраняет всё состояние мозга (и роста, если подключён) в .npz."""
        data = {field: getattr(self, field)[:self.n_neurons] for field in NEURON_FIELDS}
        data.update({field: getattr(self, field)[:self.n_conn] for field in CONNECTION_FIELDS})
        data["names"] = np.array(self.names, dtype=str)
        data["format"] = np.int64(2)
        data["rules"] = np.array(json.dumps(self.rules))
        data["time"] = np.int64(self.time)
        data["dead_weight"] = np.float64(self.dead_weight)
        data["dead_patience"] = np.int64(self.dead_patience)
        data["input_budget"] = np.float64(self.input_budget)
        data["windows"] = np.array([self.budget_window, self.inh_window], dtype=np.int64)
        data["modes"] = np.array([self.ctx_graded, float(self.ip_push), self.v_floor,
                                  float(self.inh_short), float(self.last_action_only),
                                  float(self.decision_ctx), float(self.reset_eligibility),
                                  float(self.myelin_keep), float(self.budget_above_floor),
                                  float(self.inh_delayed), float(self.reward_only_actions),
                                  float(self.inh_share), float(self.block_collective)])
        data["decision"] = np.concatenate([self._dec_level, self._dec_active.astype(float),
                                           self._dec_n.astype(float)])
        critic = self.critic.state() if self.critic is not None else None
        data["critic"] = np.array(json.dumps(critic))
        rec = self.recruit.state() if self.recruit is not None else None
        data["recruit"] = np.array(json.dumps(rec))
        data["hyp_rules"] = np.array([self.hyp_needed, self.hyp_min_rel, self.hyp_lift])
        data["regions"] = np.array(json.dumps({"regions": self.regions, "of": self.region_of}))
        data["protect_consolidated"] = np.bool_(self.protect_consolidated)
        growth = self.growth.state() if self.growth is not None else None
        data["growth"] = np.array(json.dumps(growth))
        curiosity = self.curiosity.state() if self.curiosity is not None else None
        data["curiosity"] = np.array(json.dumps(curiosity))
        change = self.neuron_change.state() if self.neuron_change is not None else None
        data["neuron_change"] = np.array(json.dumps(change))
        fam = self.familiarity.state() if self.familiarity is not None else None
        data["familiarity"] = np.array(json.dumps(fam))
        eps = self.episodes.state() if self.episodes is not None else None
        data["episodes"] = np.array(json.dumps(eps))
        hear = self.hearing.state() if self.hearing is not None else None
        data["hearing"] = np.array(json.dumps(hear))
        assoc = self.association.state() if self.association is not None else None
        data["association"] = np.array(json.dumps(assoc))
        acc = self.accumulator.state() if self.accumulator is not None else None
        data["accumulator"] = np.array(json.dumps(acc))
        latch = self.latch.state() if self.latch is not None else None
        data["latch"] = np.array(json.dumps(latch))
        units = self.units.state() if self.units is not None else None
        data["units"] = np.array(json.dumps(units))
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        """Восстанавливает мозг из файла brain.save(); продолжает ровно с того же места."""
        with np.load(path, allow_pickle=False) as d:
            names = [str(x) for x in d["names"]]
            n, m = len(names), len(d["src"])
            # правила из файла; старые файлы без правил берут текущие константы классов
            rules = json.loads(str(d["rules"])) if "rules" in d.files else None
            if rules is not None:
                missing = [f"{c}.{k}" for c, keys in current_rules().items()
                           for k in keys if k not in rules.get(c, {})]
                if missing:
                    raise ValueError("в файле нет правил: " + ", ".join(sorted(missing)))
            brain = cls(neuron_capacity=max(16, n), connection_capacity=max(64, m), rules=rules)
            for field in NEURON_FIELDS:
                if field in d.files:          # старые файлы без новых полей
                    getattr(brain, field)[:n] = d[field]
            for field in CONNECTION_FIELDS:
                if field in d.files:
                    getattr(brain, field)[:m] = d[field]
            brain.n_neurons, brain.n_conn = n, m
            brain.n_inhibitory = int(brain.inhibitory[:m].sum())
            brain.n_contexts = int((brain.ctx_slot[:n] >= 0).sum())
            brain.time = int(d["time"])
            brain.dead_weight = float(d["dead_weight"])
            brain.dead_patience = int(d["dead_patience"])
            brain.input_budget = float(d["input_budget"]) if "input_budget" in d.files else 0.0
            if "windows" in d.files:
                brain.budget_window, brain.inh_window = (int(x) for x in d["windows"])
            if "modes" in d.files:
                brain.ctx_graded, brain.ip_push = float(d["modes"][0]), bool(d["modes"][1])
                if len(d["modes"]) > 2:
                    brain.v_floor = float(d["modes"][2])
                if len(d["modes"]) > 3:
                    (brain.inh_short, brain.last_action_only, brain.decision_ctx,
                     brain.reset_eligibility) = (bool(x) for x in d["modes"][3:7])
                if len(d["modes"]) > 7:
                    brain.myelin_keep = bool(d["modes"][7])
                if len(d["modes"]) > 8:
                    brain.budget_above_floor = bool(d["modes"][8])
                if len(d["modes"]) > 9:
                    brain.inh_delayed = bool(d["modes"][9])
                if len(d["modes"]) > 10:
                    brain.reward_only_actions = bool(d["modes"][10])
                if len(d["modes"]) > 11:
                    brain.inh_share = bool(d["modes"][11])
                if len(d["modes"]) > 12:
                    brain.block_collective = bool(d["modes"][12])
            if "decision" in d.files:
                k = kernel.MAX_CONTEXTS
                brain._dec_level[:] = d["decision"][:k]
                brain._dec_active[:] = d["decision"][k:2 * k] > 0.5
                brain._dec_n[0] = int(d["decision"][2 * k])
            critic = json.loads(str(d["critic"])) if "critic" in d.files else None
            rec = json.loads(str(d["recruit"])) if "recruit" in d.files else None
            regions = json.loads(str(d["regions"])) if "regions" in d.files else None
            if "hyp_rules" in d.files:
                brain.hyp_needed = int(d["hyp_rules"][0])
                brain.hyp_min_rel = float(d["hyp_rules"][1])
                brain.hyp_lift = float(d["hyp_rules"][2])
            if "protect_consolidated" in d.files:
                brain.protect_consolidated = bool(d["protect_consolidated"])
            growth = json.loads(str(d["growth"]))
            curiosity = json.loads(str(d["curiosity"])) if "curiosity" in d.files else None
            change = json.loads(str(d["neuron_change"])) if "neuron_change" in d.files else None
            fam = json.loads(str(d["familiarity"])) if "familiarity" in d.files else None
            eps = json.loads(str(d["episodes"])) if "episodes" in d.files else None
            hear = json.loads(str(d["hearing"])) if "hearing" in d.files else None
            assoc = json.loads(str(d["association"])) if "association" in d.files else None
            acc = json.loads(str(d["accumulator"])) if "accumulator" in d.files else None
            latch = json.loads(str(d["latch"])) if "latch" in d.files else None
            units = json.loads(str(d["units"])) if "units" in d.files else None

        if regions is not None:
            brain.regions = regions["regions"]
        for i, name in enumerate(names):
            brain._register_neuron(name, i, regions["of"][i] if regions else None)
        for c in range(m):
            brain._register_connection(c)
        if growth is not None:
            from Growth import Growth
            Growth.from_state(brain, growth)
        if curiosity is not None:
            from Curiosity import Curiosity
            Curiosity.from_state(brain, curiosity)
        if change is not None:
            from NeuronChange import NeuronChange
            NeuronChange.from_state(brain, change)
        if fam is not None:
            from Familiarity import Familiarity
            Familiarity.from_state(brain, fam)
        if eps is not None:
            from Episodes import Episodes
            Episodes.from_state(brain, eps)
        if rec is not None:
            from Recruit import Recruit
            Recruit.from_state(brain, rec)
        if critic is not None:
            from RewardPrediction import RewardPrediction
            RewardPrediction.from_state(brain, critic)
        if hear is not None:
            from Hearing import Hearing
            Hearing.from_state(brain, hear)
        if assoc is not None:
            from Association import Association
            Association.from_state(brain, assoc)
        if acc is not None:
            from Accumulator import Accumulator
            Accumulator.from_state(brain, acc)
        if latch is not None:
            from Latch import Latch
            Latch.from_state(brain, latch)
        if units is not None:
            if units.get("kind") == "neural":
                from NeuralChoice import NeuralChoice
                NeuralChoice.from_state(brain, units)
            else:
                from Units import Units
                Units.from_state(brain, units)
        return brain
