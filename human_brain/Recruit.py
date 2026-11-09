"""Найм памяти: новое событие получает свой нейрон рабочей памяти.

Событие = входы (зрение, слух), показанные и сработавшие на одном шаге.
Связь доходит до нейрона памяти за 1..5 шагов (миелин меняет задержку), поэтому
событие ждёт узнавания до window шагов:
    - сработал нейрон, закреплённый за этим же событием: всё в порядке;
    - сработал свободный нейрон: он закрепляется за событием;
    - за window шагов не узнал никто: если у события уже есть владелец
      (не узнал, например, из-за торможения), он закрепляется заново; иначе
      событие новое, нанимается свободный нейрон с наибольшим зарядом (ближе
      всех к порогу); если свободных нет — тот, что дольше всех не работал.
      У одного события всегда не больше одного владельца; когда событие
      получило нейрон, остальные ожидающие заявки того же события решены.
Закрепление: входы события дают нейрону по (1 + EPS * k) / k расстояния до
порога (k — число входов события, EPS = 1 / (2 * MAX_EVENT * (MAX_EVENT - 1))):
все k вместе чуть превышают порог, без любого одного не хватает при любом
k <= MAX_EVENT. Запас над порогом растёт с k: если показан набор, содержащий
событие другого нейрона, выигрывает соревнование (взаимное торможение памяти)
более полный детектор. Остальные входы 0. Нейрон становится детектором
совпадения (заряд на следующий шаг не переходит, порог не меняется), входы
события закреплённые (не учатся, миелин не меняется, не удаляются) и быстрые
(задержка 1): нужны все входы на одном шаге. Посторонние входы его не
тормозят: набор, содержащий событие целиком, нейрон тоже зажигает.
Переназначение на другое событие стирает прежний смысл нейрона: заряд,
отдых, сигналы в пути, следы и метки его связей, контекстные добавки и
ожидание награды. Повторное закрепление того же события только выравнивает
входы, выученное не трогает.
Нанятый нейрон сразу активен (след 1): память о событии появляется сейчас.
"""


MAX_EVENT = 32                                   # больше входов в одном событии нельзя
EPS = 1.0 / (2 * MAX_EVENT * (MAX_EVENT - 1))    # запас над порогом на один вход


class Recruit:
    PARAMS = ("inputs", "memory", "margin", "window")

    def __init__(self, brain, inputs=("зрение", "слух"), memory="рабочая память", margin=1.1,
                 window=5):
        self.brain = brain
        brain.recruit = self
        self.inputs = sorted(inputs)
        self.memory = memory
        self.margin = margin  # не используется (сила входов см. выше), оставлен для старых файлов
        self.owner = {}       # номер нейрона памяти -> отсортированный список номеров входов
        self.used = {}        # номер нейрона памяти -> когда последний раз работал
        self.window = window  # сколько шагов событие ждёт узнавания (наибольшая задержка)
        self.pending = []     # [время события, входы] — ещё не узнанные события
        self.log = []

    def observe(self):
        """Вызывается мозгом в конце каждого шага."""
        b = self.brain
        t = b.time
        mem = [i for i, r in enumerate(b.region_of) if r == self.memory]
        fired_mem = [i for i in mem if b._fired[i]]
        for i in fired_mem:
            self.used[i] = t
        left = []
        settled = []        # события, решённые на этом шаге
        for t0, event in self.pending:
            if event in settled:
                continue    # тот же набор уже получил нейрон на этом шаге
            if self._settle(event, mem, fired_mem, t - t0 >= self.window):
                settled.append(event)
            else:
                left.append([t0, event])
        # заявки того же события, что уже решено, тоже решены (нейрон один)
        self.pending = [x for x in left if x[1] not in settled]
        now = [i for i, r in enumerate(b.region_of)
               if r in self.inputs and b._fired[i] and b.stim_time[i] == t]
        if now:
            self.pending.append([t, now])

    def owner_of(self, event):
        """Нейрон памяти, закреплённый за событием, или None."""
        for w, ev in self.owner.items():
            if ev == event:
                return w
        return None

    def _settle(self, event, mem, fired_mem, last_chance):
        """True, если событие решено (узнано или нанят нейрон)."""
        b = self.brain
        own = self.owner_of(event)
        if own is not None and own in fired_mem:
            return True
        free_fired = [i for i in fired_mem if i not in self.owner]
        if own is not None:
            if not last_chance:
                return False
            w = own
            how = "закреплён заново (не узнал)"
        elif free_fired:
            w = max(free_fired, key=lambda i: b.drive[i])
            how = "сработал свободный"
        elif not last_chance:
            return False
        else:
            free = [i for i in mem if i not in self.owner]
            if free:
                w = max(free, key=lambda i: b.potential[i])
                how = "нанят свободный"
            else:
                w = min(mem, key=lambda i: self.used.get(i, -1))
                how = "переназначен давно не работавший"
        self._claim(w, event)
        self.log.append({"time": int(b.time), "text": f"{b.names[w]}: {how} для "
                         + "+".join(b.names[i] for i in event)})
        return True

    def _claim(self, w, event):
        b = self.brain
        name = b.names[w]
        old = self.owner.get(w)
        # прежний владелец этого события (если был другой нейрон) освобождается
        prev = self.owner_of(event)
        if prev is not None and prev != w:
            del self.owner[prev]
        k_in = len(event)
        if k_in > MAX_EVENT:
            raise ValueError(f"событие из {k_in} входов, больше MAX_EVENT = {MAX_EVENT}")
        b.make_coincidence(w)
        reach = b.reach(w)
        strength = reach * (1.0 + EPS * k_in) / k_in
        have = set()
        for c in b.connections:
            if c.target != name or c.inhibitory:
                continue
            src = b.neurons[c.source].index
            if b.region_of[src] not in self.inputs or b.assoc[c.index]:
                continue
            k = c.index
            if src in event and src not in have:
                b.weight[k] = strength
                b.thickness[k] = 1.0          # все входы события приходят через 1 шаг
                b.hyp[k] = False
                b.fixed[k] = True
                b.plasticity[k] = 1.0
                have.add(src)
            else:
                b.weight[k] = 0.0             # не входит в событие (или лишний дубль)
                b.fixed[k] = False
        for src in event:
            if src not in have:
                b.connect(b.names[src], name, weight=strength, myelin_thickness=1.0, fixed=True)
        self.owner[w] = list(event)
        self.used[w] = b.time
        b.ctx_trace[w] = 1.0
        if old == list(event):
            return                    # то же событие: выученное (g, ожидание) сохраняется
        # нейрон получает новый смысл: прежнее состояние к делу не относится
        rest = b.rest
        b.potential[w] = rest
        b.refractory[w] = 0
        incoming = b.tgt[:b.n_conn] == w
        outgoing = b.src[:b.n_conn] == w
        b.pending[:b.n_conn][incoming] = 0              # старые сигналы в пути пропадают
        b.pre_trace[:b.n_conn][incoming | outgoing] = 0.0
        b.post_trace[:b.n_conn][incoming | outgoing] = 0.0
        b.eligibility[:b.n_conn][incoming | outgoing] = 0.0
        slot = b.ctx_slot[w]
        if slot >= 0:
            b.ctx_w[:b.n_conn, slot] = 0.0            # контекстные добавки (g)
            b._dec_level[slot] = 0.0                  # снимок контекста момента решения
            if b._dec_active[slot]:
                b._dec_active[slot] = False
                b._dec_n[0] -= 1
        if b.critic is not None:
            b.critic.value.pop(w, None)               # ожидание награды
        if old is not None and old != event:
            self.log.append({"time": int(b.time), "text": f"{name}: забыто "
                             + "+".join(b.names[i] for i in old)})

    # ---------- сохранение ----------

    def state(self):
        return {"params": {k: getattr(self, k) for k in self.PARAMS},
                "owner": [[w, ev] for w, ev in self.owner.items()],
                "used": [[w, t] for w, t in self.used.items()],
                "pending": self.pending, "log": list(self.log)}

    @classmethod
    def from_state(cls, brain, state):
        r = cls(brain, **state["params"])
        r.owner = {w: list(ev) for w, ev in state["owner"]}
        r.used = {w: t for w, t in state["used"]}
        r.pending = [list(x) for x in state.get("pending", [])]
        r.log = list(state["log"])
        return r
