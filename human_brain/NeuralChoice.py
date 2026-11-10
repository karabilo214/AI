"""Выбор заготовки делает сама импульсная сеть (вместо правила найма Units.py).

Все заготовки живут сразу (вид «по одной» / «по регистру» × регистр = счётчик
одного события). Каждая считает на каждом дне и учит веса уровня отдельно для
каждого вопроса (по ошибке после праздника, как в Units.Unit). У каждой свой
нейрон-выход U1..Un (область «заготовки»): на «отвечай» он загорается, если
заготовка для запомненного вопроса считает, что можно дать ещё. Какой
заготовке верить на каком вопросе, никто не говорит: мозг выращивает сочетания
«выход + вопрос» (Ui + M_*; другие сочетания с выходами запрещены, иначе их
сотни) и по награде учит их связи к «ещё»/«готово», как всё остальное.
Закрепление выбора (streak > 0, идея владельца: внутреннее повторение): после
streak удач подряд на новом вопросе выбор сети запоминается (удачей считается
день с ответом больше 0 без ошибки — день с ответом 0 заготовку не проверяет;
ошибка в любой день обнуляет серию): победитель — заготовка,
чьё сочетание с вопросом сильнее всех ведёт к «ещё» (возбуждение минус
торможение); связи этого сочетания к действиям становятся неизменными, а
остальные заготовки на этом вопросе больше не загораются. Выбирает сеть,
правило только закрепляет результат повторением.
Привыкание выходов (habituate, решение владельца: соперников выключает сам
мозг, а не программа): каждый выход заготовки привыкает, отдельно для каждого
вопроса — при каждом загорании его заметность f растёт (f += hab * (1 - f)),
толчок выхода = 20 * (1 - max_ignore * f); со временем f сама гаснет (recovery
за шаг). Привыкание снимается (f *= 1 - dishab) только у тех, чьё сочетание с
вопросом было главной причиной награждённого действия (доля сигнала при его
срабатывании, last_share, не меньше cause_share). Числа — из выбранных
параметров привыкания мозга («трава»: fam_habituation, fam_max_ignore,
fam_dishabituation, fam_recovery). Победителя никто не назначает: верная
заготовка остаётся заметной, остальные горят без последствий и затихают.
Прежний накопитель учится и говорит (ACC / ACC_NEG) только на своих вопросах
(own) — это защита выученного, а не выбор; среди заготовок есть и вид «по
одной», то есть такой же счётчик. Иначе его «минус» на чужом вопросе давал
сочетание, выучившее тормоз против «ещё», и ответы на деление качались.
"""
from Units import VARIANTS


class Candidate:
    def __init__(self, inputs, variant, reg_event, rate=0.5):
        self.inputs = list(inputs)
        self.variant = variant
        self.reg_event = reg_event
        self.rate = rate
        self.w = {}            # вопрос -> {вход: вес уровня}
        self.new_day()

    def new_day(self):
        self.counts = {k: 0 for k in self.inputs}
        self.mores = 0

    def weights(self, q):
        return self.w.setdefault(q, {k: 0.0 for k in self.inputs})

    def level0(self, q):
        w = self.weights(q)
        return sum(int(round(w[k])) * n for k, n in self.counts.items())

    def reg(self):
        return self.counts.get(self.reg_event, 0)

    def has_more(self, q):
        if self.variant == "по одной":
            return self.level0(q) - self.mores > 0
        r = self.reg()
        return r > 0 and self.level0(q) - self.mores * r >= r

    def predicted(self, q):
        lv = self.level0(q)
        if self.variant == "по одной":
            return max(lv, 0)
        r = self.reg()
        return lv // r if r > 0 and lv > 0 else 0

    def learn(self, q, answer):
        n = self.counts
        denom = sum(v * v for v in n.values())
        if denom == 0:
            return
        r = self.reg()
        target = answer if self.variant == "по одной" else answer * r + (r - 1) / 2.0
        w = self.weights(q)
        err = sum(w[k] * n[k] for k in self.inputs) - target
        for k in self.inputs:
            w[k] -= self.rate * err * n[k] / denom

    def state(self):
        return {"inputs": self.inputs, "variant": self.variant, "reg_event": self.reg_event,
                "rate": self.rate, "w": self.w}

    @classmethod
    def from_state(cls, st):
        c = cls(st["inputs"], st["variant"], st["reg_event"], st["rate"])
        c.w = {q: dict(v) for q, v in st["w"].items()}
        return c


class NeuralChoice:
    PARAMS = ("inputs", "own", "query", "actions", "decrement", "region")

    def __init__(self, brain, inputs, own, query="GO", actions=("MORE", "DONE"),
                 decrement="MORE", region="заготовки", make=True):
        self.brain = brain
        brain.units = self               # тот же разъём, что у Units
        self.inputs = list(inputs)
        self.own = list(own)
        self.query = query
        self.actions = list(actions)
        self.decrement = decrement
        self.region = region
        self.mute_old = True
        self.partners_all = False
        self.streak = 10           # дней подряд без ошибки до закрепления (0 = выключено)
        self.habituate = False     # привыкание выходов (вместо закрепления программой)
        self.hab = 0.05
        self.max_ignore = 0.717818
        self.dishab = 1.0
        self.recovery = 0.999859
        self.cause_share = 0.3
        self.fam = {}              # (вопрос, номер) -> привыкание 0..1
        self._pair_out = {}        # сочетание -> номера его выходных связей
        self.run_ok = {}           # вопрос -> текущая серия
        self.chosen = {}           # вопрос -> номер закреплённой заготовки
        self.cands = [Candidate(self.inputs, v, e) for v in VARIANTS for e in self.inputs]
        self.names = [f"U{k + 1}" for k in range(len(self.cands))]
        self._question = None
        if make:
            self._make_neurons()

    def _make_neurons(self):
        b = self.brain
        # десятки сочетаний тормозят одно действие: метку торможения делить
        b.inh_share = True
        # «помешал» по сумме тормозов здесь хуже (с закреплением выбора 16/24
        # против 19/24), поэтому выключено
        b.block_collective = False
        if self.region not in b.regions:
            b.add_region(self.region)
        for name in self.names:
            b.add_neuron(name, region=self.region)
            for o in self.actions:
                # сила 0: только образец целей для сочетаний (иначе 14 выходов
                # вместе толкали бы оба действия)
                b.connect(name, o, weight=0.0, myelin_thickness=1.0, fixed=True)
                b.connect(name, o, weight=0.0, inhibitory=True, myelin_thickness=1.0,
                          plasticity=0.0)
        if b.neuron_change is not None:
            b.neuron_change.make_same((self.region,))
        g = b.growth
        if g is not None:
            g.max_neurons += 3 * len(self.names) + 6
            if g.pair_regions is not None and self.region not in g.pair_regions:
                g.pair_regions = sorted(set(g.pair_regions) | {self.region})
            g.pair_partners = {self.region: self._new_question_neurons()}
            g.sensory_regions = sorted(set(g.sensory_regions or []) | {self.region})

    def _new_question_neurons(self):
        """Нейроны защёлки тех вопросов, которыми прежние устройства не владеют:
        выходы заготовок сочетаются только с ними (прежние навыки не переподключаются)."""
        lt = self.brain.latch
        if self.partners_all:
            return list(lt.outputs)
        return [o for q, o in zip(lt.inputs, lt.outputs) if q not in self.own]

    def refresh_partners(self):
        if self.brain.growth is not None:
            self.brain.growth.pair_partners = {self.region: self._new_question_neurons()}

    def new_day(self):
        self._question = None
        self.brain.accumulator.mute = False
        for c in self.cands:
            c.new_day()

    def before_step(self):
        b = self.brain
        if b.stim_time[b.neurons[self.query].index] != b.time or self._question is None:
            return
        chosen = self.chosen.get(self._question)
        q = self._question
        for k, (c, name) in enumerate(zip(self.cands, self.names)):
            if chosen is not None and k != chosen:
                continue           # выбор закреплён: остальные заготовки молчат
            if c.has_more(q):
                if self.habituate:
                    f = self.fam.get((q, k), 0.0)
                    b.stimulate(name, 20.0 * (1.0 - self.max_ignore * f))
                    self.fam[(q, k)] = f + self.hab * (1.0 - f)
                else:
                    b.stimulate(name, 20.0)

    def on_reward(self, value):
        """Награда: снять привыкание у выходов, чьё сочетание с вопросом было
        главной причиной награждённого действия."""
        if not self.habituate or value <= 0 or self._question is None:
            return
        b = self.brain
        acts = [b.neurons[a].index for a in self.actions]
        act = max(acts, key=lambda i: b.fire_time[i])
        if b.time - b.fire_time[act] > 15:
            return
        lt = b.latch
        mq = lt.outputs[lt.inputs.index(self._question)]
        for k, name in enumerate(self.names):
            for pair in (f"{mq}+{name}", f"{name}+{mq}"):
                if pair not in b.neurons:
                    continue
                for c in self._pair_out.get(pair) or self._cache_out(pair):
                    if b.tgt[c] == act and not b.inhibitory[c] and b.last_share[c] >= self.cause_share:
                        key = (self._question, k)
                        self.fam[key] = self.fam.get(key, 0.0) * (1.0 - self.dishab)

    def _cache_out(self, pair):
        b = self.brain
        idx = [c.index for c in b.connections if c.source == pair]
        self._pair_out[pair] = idx
        return idx

    def observe(self):
        b = self.brain
        t = b.time
        if self.habituate and self.fam:
            for key in self.fam:
                self.fam[key] *= self.recovery
        for name in self.inputs:
            i = b.neurons[name].index
            if b._fired[i] and b.stim_time[i] == t:
                for c in self.cands:
                    c.counts[name] += 1
        if b.latch.held is not None:
            self._question = b.latch.inputs[b.latch.held]
            b.accumulator.mute = self.mute_old and self._question not in self.own
        if b._fired[b.neurons[self.decrement].index]:
            for c in self.cands:
                c.mores += 1

    def after_day(self, answer, truth, learn, result=None):
        if not learn or self._question is None:
            return
        q = self._question
        if q in self.own:
            self.brain.accumulator.learn(truth)
        for c in self.cands:
            c.learn(q, answer)
        if self.streak and q not in self.own and q not in self.chosen and result is not None:
            more, done = result
            ok = more == answer and done
            if not ok:
                self.run_ok[q] = 0
            elif answer > 0:
                # засчитывается только день, где надо было сказать «ещё»: день с
                # ответом 0 проходит и без заготовки (сразу «готово»), он её не проверяет
                self.run_ok[q] = self.run_ok.get(q, 0) + 1
            if self.run_ok.get(q, 0) >= self.streak:
                self._consolidate(q)

    def _consolidate(self, q):
        """Запомнить выбор сети на вопросе q (см. описание модуля)."""
        b = self.brain
        lt = b.latch
        mq = lt.outputs[lt.inputs.index(q)]
        score = {}
        for c in b.connections:
            parts = c.source.split("+")
            if c.target == self.decrement and mq in parts and len(parts) == 2:
                u = parts[1] if parts[0] == mq else parts[0]
                if u in self.names:
                    s = score.get(u, 0.0)
                    score[u] = s + (-c.synapse.weight if c.inhibitory else c.synapse.weight)
        if not score:
            return
        win = max(score, key=score.get)
        k = self.names.index(win)
        self.chosen[q] = k
        pair = {f"{mq}+{win}", f"{win}+{mq}"}
        for c in b.connections:
            if c.source in pair and c.target in self.actions:
                b.fixed[c.index] = not c.inhibitory   # возбуждение — как закреплённый вход
                if c.inhibitory:
                    b.plasticity[c.index] = 0.0      # торможение больше не учится
        self.log = getattr(self, "log", [])
        self.log.append(f"{q}: закреплён выбор {win} ({self.cands[k].variant}, регистр "
                        f"{self.cands[k].reg_event}) после {self.streak} удач подряд")

    def accuracy(self, q, days):
        """Для разбора: сколько дней каждая заготовка насчитала верно (на копиях дней)."""
        return {name: c.variant + "/" + c.reg_event for name, c in zip(self.names, self.cands)}

    # ---------- сохранение ----------

    def state(self):
        return {"kind": "neural", "params": {k: getattr(self, k) for k in self.PARAMS},
                "cands": [c.state() for c in self.cands], "mute_old": self.mute_old,
                "partners_all": self.partners_all, "streak": self.streak,
                "run_ok": self.run_ok, "chosen": self.chosen, "log": getattr(self, "log", []),
                "habituate": self.habituate,
                "fam": [[q, k, f] for (q, k), f in self.fam.items()]}

    @classmethod
    def from_state(cls, brain, st):
        x = cls(brain, make=False, **st["params"])
        x.cands = [Candidate.from_state(c) for c in st["cands"]]
        x.mute_old = st.get("mute_old", True)
        x.partners_all = st.get("partners_all", False)
        x.streak = st.get("streak", 10)
        x.run_ok = dict(st.get("run_ok", {}))
        x.chosen = {q: int(k) for q, k in st.get("chosen", {}).items()}
        x.log = list(st.get("log", []))
        x.habituate = st.get("habituate", False)
        x.fam = {(q, int(k)): f for q, k, f in st.get("fam", [])}
        return x
