"""Быстрая оценка параметров: весь цикл задач внутри Numba.

Те же правила, что у Brain + Growth + Curiosity + NeuronChange (шаг считает
тот же kernel.py; рост повторяет Growth.observe шаг в шаг, но на массивах),
только без Python в цикле. Используется эволюцией: evolve.py --fast.

Отличие от обычной версии одно: случайные числа (шум любопытства, отличия
нейронов) берутся из генератора Numba, а не numpy. Поэтому при шуме ответы
совпадают с обычной версией только по статистике; без шума совпадают
точно (проверка: check_fast.py).
"""
import random

import numpy as np
from numba import njit, int64, float64, boolean
from numba.experimental import jitclass

from human_brain.core import kernel
from human_brain.parts.Growth import PAIR_SHARE

# ---------- параметры: порядок в массиве P ----------

P_NAMES = [
    "rest", "threshold", "decay", "refractory",
    "strengthen", "weaken", "trace_decay", "min_weight", "max_weight",
    "consolidate_weight", "eligibility_decay", "reward_rate",
    "myelin_grow", "myelin_shrink", "myelin_fade", "suppress_steps",
    "protect_inhibited", "default_thickness",
    "dead_weight", "dead_patience", "input_budget", "protect_consolidated",
    "calm", "curious", "curiosity_up", "curiosity_down", "curiosity_decay",
    "connection_threshold", "min_reliability", "min_lift", "forget_after",
    "explain_weight", "young_plasticity", "new_weight", "pair_weight",
    "neuron_threshold", "max_gap", "max_neurons", "inherit_outputs", "fast_new_outputs",
    "neurons_on", "nc_rate", "nc_target", "nc_window", "nc_min", "nc_max",
    "nc_threshold_spread", "nc_decay_spread", "nc_refractory_jitter",
    "init_weight", "init_plasticity", "inhibit_weight",
    "fam_on", "fam_habituation", "fam_max_ignore", "fam_trace_decay",
    "fam_dishabituation", "fam_explain",
    "hyp_on", "hyp_needed", "hyp_min_rel", "hyp_lift",
    "credit_on", "credit_horizon", "myelin_timing", "timing_window", "myelin_idle",
    "ctx_on", "ctx_horizon", "fam_recovery",
    "forget_on", "imp_decay", "important", "doubt_needed", "relearn_plast", "consolidated_rate",
    "action_min", "curiosity_idle", "neg_rate", "neg_needed", "necessity_on",
    "ban_on", "ban_up", "ban_down", "ban_power",
]
I = {name: k for k, name in enumerate(P_NAMES)}
(REST, THRESHOLD, DECAY, REFRACTORY, STRENGTHEN, WEAKEN, TRACE_DECAY, MIN_WEIGHT,
 MAX_WEIGHT, CONSOLIDATE, ELIG_DECAY, REWARD_RATE, M_GROW, M_SHRINK, M_FADE,
 SUPPRESS_STEPS, PROTECT_INHIBITED, DEFAULT_THICK, DEAD_WEIGHT, DEAD_PATIENCE,
 BUDGET, PROTECT_CONS, CALM, CURIOUS, CUR_UP, CUR_DOWN, CUR_DECAY, CONN_THRESHOLD,
 MIN_REL, MIN_LIFT, FORGET_AFTER, EXPLAIN, YOUNG_PLAST, NEW_WEIGHT, PAIR_WEIGHT,
 NEURON_THRESHOLD, MAX_GAP, MAX_NEURONS, INHERIT, FAST_NEW, NC_ON, NC_RATE, NC_TARGET,
 NC_WINDOW, NC_MIN, NC_MAX, NC_TSPREAD, NC_DSPREAD, NC_RJITTER,
 INIT_WEIGHT, INIT_PLAST, INHIBIT_WEIGHT,
 FAM_ON, FAM_HAB, FAM_MAX, FAM_TRACE, FAM_DISHAB, FAM_EXPLAIN,
 HYP_ON, HYP_NEEDED, HYP_MIN_REL, HYP_LIFT,
 CREDIT_ON, CREDIT_HORIZON, MYELIN_TIMING, TIMING_WINDOW, MYELIN_IDLE,
 CTX_ON, CTX_HORIZON, FAM_RECOVERY,
 FORGET_ON, IMP_DECAY, IMPORTANT, DOUBT_NEEDED, RELEARN_PLAST, CONSOLIDATED_RATE,
 ACTION_MIN, CUR_IDLE, NEG_RATE, NEG_NEEDED, NECESSITY_ON,
 BAN_ON, BAN_UP, BAN_DOWN, BAN_POWER) = range(len(P_NAMES))


def params_array(p):
    """Набор параметров tasks (словарь) -> массив P для быстрой версии."""
    v = {
        "rest": -70.0, "min_weight": 0.0, "forget_after": 20, "max_gap": 4,
        "max_neurons": 12, "inherit_outputs": True, "nc_window": 0.99,
        "nc_min": -65.0, "nc_max": -45.0, "nc_decay_spread": 0.01,
        "nc_refractory_jitter": True,
    }
    v.update(p)
    v["input_budget"] = p["input_budget"] if p["budget_on"] else 0.0
    return np.array([float(v[name]) for name in P_NAMES])


# ---------- мозг на массивах ----------

spec = [
    ("NC", int64), ("CC", int64), ("G", int64), ("P", float64[:]),
    ("n", int64), ("potential", float64[:]), ("refractory", int64[:]), ("spikes", int64[:]),
    ("threshold", float64[:]), ("received", float64[:]), ("explores", boolean[:]),
    ("decay", float64[:]), ("refr_len", int64[:]), ("activity", float64[:]), ("ip_fixed", boolean[:]), ("ctx_level", float64[:]), ("dec_n", int64[:]),
    ("ctx_trace", float64[:]), ("push_in", float64[:]),
    ("suppressed", int64[:]), ("fired", boolean[:]), ("input_total", float64[:]),
    ("m", int64), ("src", int64[:]), ("tgt", int64[:]), ("weight", float64[:]),
    ("pre", float64[:]), ("post", float64[:]), ("thick", float64[:]), ("alive", boolean[:]),
    ("weak", int64[:]), ("pending", int64[:, :]), ("plast", float64[:]), ("elig", float64[:]),
    ("inhib", boolean[:]), ("n_inhib", int64), ("delivered", boolean[:]), ("removed", int64[:]),
    ("time", int64), ("pair_count", int64[:, :]),
    ("growth_on", boolean), ("curiosity_on", boolean), ("nc_on", boolean), ("level", float64),
    ("budget", float64), ("protect_cons", boolean), ("dead_patience", int64),
    ("hist", boolean[:, :]), ("hist_head", int64), ("hist_count", int64),
    ("oc_active", boolean[:, :, :]), ("oc_count", int64[:, :, :]), ("oc_start", int64[:, :, :]),
    ("tog", int64[:, :]), ("pair_made", boolean[:, :]),
    ("grown_src", int64[:]), ("grown_tgt", int64[:]), ("n_grown", int64),
    ("n_grown_neurons", int64),
    ("fam_on", boolean), ("familiar", float64[:]), ("shown_trace", float64[:]),
    ("shown_now", boolean[:]), ("news_count", int64),
    ("hyp", boolean[:]), ("hyp_tests", int64[:]), ("hyp_hits", int64[:]), ("arr_time", int64[:]),
    ("drive", float64[:]), ("fire_time", int64[:]), ("confirmed", int64[:]),
    ("relevance", float64[:]), ("grown_c", int64[:]), ("hyp_needed", int64),
    ("ctx_slot", int64[:]), ("ctx_w", float64[:, :]), ("n_contexts", int64), ("active", boolean[:]),
    ("importance", float64[:]), ("doubt", int64[:]),
    ("confirm", int64[:]), ("last_share", float64[:]), ("syn_drive", float64[:]),
    ("ban", float64[:]), ("learn_gain", float64[:]), ("relax", float64[:]), ("w0", float64[:]),
    ("assoc", boolean[:]), ("stim_time", int64[:]), ("perceptual", boolean[:]),
    ("fixed", boolean[:]), ("pool", int64[:]), ("recall", boolean[:]), ("recall_in", float64[:]),
    ("block_elig", float64[:]),
    ("coincidence", boolean[:]),
]


@jitclass(spec)
class Sim:
    def __init__(self, NC, CC, P):
        self.NC = NC
        self.CC = CC
        self.P = P
        self.G = int(P[MAX_GAP])
        self.n = 0
        self.potential = np.zeros(NC)
        self.refractory = np.zeros(NC, dtype=np.int64)
        self.spikes = np.zeros(NC, dtype=np.int64)
        self.threshold = np.zeros(NC)
        self.received = np.zeros(NC)
        self.explores = np.zeros(NC, dtype=np.bool_)
        self.decay = np.zeros(NC)
        self.refr_len = np.zeros(NC, dtype=np.int64)
        self.activity = np.zeros(NC)
        self.ip_fixed = np.zeros(NC, dtype=np.bool_)
        self.suppressed = np.zeros(NC, dtype=np.int64)
        self.fired = np.zeros(NC, dtype=np.bool_)
        self.input_total = np.zeros(NC)
        self.m = 0
        self.src = np.zeros(CC, dtype=np.int64)
        self.tgt = np.zeros(CC, dtype=np.int64)
        self.weight = np.zeros(CC)
        self.pre = np.zeros(CC)
        self.post = np.zeros(CC)
        self.thick = np.zeros(CC)
        self.alive = np.zeros(CC, dtype=np.bool_)
        self.weak = np.zeros(CC, dtype=np.int64)
        self.pending = np.zeros((CC, kernel.RING), dtype=np.int64)
        self.plast = np.ones(CC)
        self.elig = np.zeros(CC)
        self.inhib = np.zeros(CC, dtype=np.bool_)
        self.n_inhib = 0
        self.delivered = np.zeros(CC, dtype=np.bool_)
        self.removed = np.zeros(CC, dtype=np.int64)
        self.time = 0
        self.pair_count = np.zeros((NC, NC), dtype=np.int64)
        self.growth_on = False
        self.curiosity_on = False
        self.nc_on = False
        self.level = 0.0
        self.budget = 0.0
        self.protect_cons = True
        self.dead_patience = 0
        self.hist = np.zeros((self.G, NC), dtype=np.bool_)
        self.hist_head = 0
        self.hist_count = 0
        self.oc_active = np.zeros((NC, NC, self.G + 1), dtype=np.bool_)
        self.oc_count = np.zeros((NC, NC, self.G + 1), dtype=np.int64)
        self.oc_start = np.zeros((NC, NC, self.G + 1), dtype=np.int64)
        self.tog = np.zeros((NC, NC), dtype=np.int64)
        self.pair_made = np.zeros((NC, NC), dtype=np.bool_)
        self.grown_src = np.zeros(CC, dtype=np.int64)
        self.grown_tgt = np.zeros(CC, dtype=np.int64)
        self.n_grown = 0
        self.n_grown_neurons = 0
        self.fam_on = False
        self.familiar = np.zeros(NC)
        self.shown_trace = np.zeros(NC)
        self.shown_now = np.zeros(NC, dtype=np.bool_)
        self.news_count = 0
        self.hyp = np.zeros(CC, dtype=np.bool_)
        self.hyp_tests = np.zeros(CC, dtype=np.int64)
        self.hyp_hits = np.zeros(CC, dtype=np.int64)
        self.arr_time = np.full(CC, kernel.NEVER, dtype=np.int64)
        self.drive = np.zeros(NC)
        self.fire_time = np.full(NC, kernel.NEVER, dtype=np.int64)
        self.confirmed = np.zeros(CC, dtype=np.int64)
        self.relevance = np.zeros(NC)
        self.grown_c = np.zeros(CC, dtype=np.int64)
        self.hyp_needed = 0
        self.ctx_slot = np.full(NC, -1, dtype=np.int64)
        self.ctx_w = np.zeros((CC, kernel.MAX_CONTEXTS))
        self.n_contexts = 0
        self.active = np.zeros(kernel.MAX_CONTEXTS, dtype=np.bool_)
        self.ctx_level = np.zeros(kernel.MAX_CONTEXTS)
        self.dec_n = np.zeros(1, dtype=np.int64)
        self.ctx_trace = np.zeros(NC)
        self.push_in = np.zeros(NC)
        self.importance = np.zeros(CC)
        self.doubt = np.zeros(CC, dtype=np.int64)
        self.confirm = np.zeros(CC, dtype=np.int64)
        self.last_share = np.zeros(CC)
        self.syn_drive = np.zeros(NC)
        self.ban = np.zeros(CC)
        self.learn_gain = np.ones(NC)
        self.relax = np.zeros(NC)
        self.w0 = np.zeros(CC)
        self.assoc = np.zeros(CC, dtype=np.bool_)
        self.stim_time = np.full(NC, kernel.NEVER, dtype=np.int64)
        self.perceptual = np.zeros(NC, dtype=np.bool_)
        self.fixed = np.zeros(CC, dtype=np.bool_)
        self.pool = np.full(NC, -1, dtype=np.int64)
        self.recall = np.zeros(CC, dtype=np.bool_)
        self.recall_in = np.zeros(NC)
        self.block_elig = np.zeros(CC)
        self.coincidence = np.zeros(NC, dtype=np.bool_)

    # ----- знакомость (то же, что Familiarity) -----

    def stimulate(self, i, strength):
        self.stim_time[i] = self.time
        if self.fam_on:
            self.potential[i] += strength * (1.0 - self.P[FAM_MAX] * self.familiar[i])
            self.familiar[i] += self.P[FAM_HAB] * (1.0 - self.familiar[i])
            self.shown_trace[i] = 1.0
            self.shown_now[i] = True
        else:
            self.potential[i] += strength

    def dishabituate(self, strength, skip):
        for i in range(self.n):
            if i != skip:
                k = min(1.0, self.P[FAM_DISHAB] * strength * self.shown_trace[i])
                self.familiar[i] *= 1.0 - k

    def fam_after_step(self):
        for j in range(self.n):
            if self.shown_now[j] and self.fired[j] and self.received[j] < self.P[FAM_EXPLAIN]:
                self.news_count += 1
                self.dishabituate(1.0, j)
        for i in range(self.n):
            self.shown_trace[i] *= self.P[FAM_TRACE]
            self.shown_now[i] = False
            self.familiar[i] *= self.P[FAM_RECOVERY]

    # ----- строение -----

    def n_ctx(self):
        return self.n_contexts if self.P[CTX_ON] > 0 else 0

    def add_neuron(self, explores, context):
        i = self.n
        P = self.P
        self.potential[i] = P[REST]
        self.refractory[i] = 0
        self.spikes[i] = 0
        self.threshold[i] = P[THRESHOLD]
        self.explores[i] = explores
        self.decay[i] = P[DECAY]
        self.refr_len[i] = int(P[REFRACTORY])
        self.activity[i] = 0.0
        self.drive[i] = 0.0
        self.fire_time[i] = kernel.NEVER
        self.ctx_slot[i] = -1
        self.learn_gain[i] = 1.0
        self.relax[i] = 0.0
        if context and self.n_contexts < kernel.MAX_CONTEXTS:
            self.ctx_slot[i] = self.n_contexts
            self.n_contexts += 1
        self.n += 1
        if self.nc_on:
            self.vary(i)
        return i

    def vary(self, i):
        P = self.P
        t = self.threshold[i] + np.random.normal(0.0, P[NC_TSPREAD])
        self.threshold[i] = min(P[NC_MAX], max(P[NC_MIN], t))
        d = self.decay[i] + np.random.normal(0.0, P[NC_DSPREAD])
        self.decay[i] = min(0.99, max(0.85, d))
        if P[NC_RJITTER] > 0:
            self.refr_len[i] = max(1, self.refr_len[i] + np.random.randint(-1, 2))

    def connect(self, s, t, w, thick, plast, inhib, hyp, fixed=False):
        if self.m >= self.CC:
            # обычный Brain расширил бы массивы; молча строить неполную сеть нельзя
            raise ValueError("fast.Sim: не хватает места для связей (CC)")
        c = self.m
        self.src[c] = s
        self.tgt[c] = t
        self.weight[c] = w
        self.pre[c] = 0.0
        self.post[c] = 0.0
        self.thick[c] = thick
        self.alive[c] = True
        self.weak[c] = 0
        self.pending[c, :] = 0
        self.plast[c] = plast
        self.elig[c] = 0.0
        self.inhib[c] = inhib
        self.hyp[c] = hyp
        self.hyp_tests[c] = 0
        self.hyp_hits[c] = 0
        self.arr_time[c] = kernel.NEVER
        self.ctx_w[c, :] = 0.0
        self.importance[c] = 0.0
        self.doubt[c] = 0
        self.confirm[c] = 0
        self.last_share[c] = 0.0
        self.ban[c] = 0.0
        self.w0[c] = w
        self.fixed[c] = fixed
        if inhib:
            self.n_inhib += 1
        self.m += 1
        self.pair_count[s, t] += 1
        return c

    # ----- работа -----

    def reward(self, value):
        P = self.P
        kernel.reward(value, value, self.n, self.m, self.src, self.tgt, self.weight, self.elig,
                      self.plast, self.alive, self.inhib, self.explores, self.fire_time,
                      self.time, P[CREDIT_ON] > 0, int(P[CREDIT_HORIZON]), self.relevance,
                      P[REWARD_RATE], P[MIN_WEIGHT], P[MAX_WEIGHT], P[CONSOLIDATE],
                      self.n_ctx(), self.ctx_slot, self.ctx_w, int(P[CTX_HORIZON]), self.active,
            self.ctx_level, 0.0, self.ctx_trace,
                      P[FORGET_ON] > 0, self.importance, self.doubt, int(P[DOUBT_NEEDED]),
                      P[RELEARN_PLAST], P[CONSOLIDATED_RATE], self.last_share, self.confirm,
                      P[NEG_RATE], P[BAN_ON] > 0, self.ban, P[BAN_UP], P[BAN_DOWN],
                      P[BAN_POWER], self.learn_gain,
                      False, self.ctx_level, self.active, self.dec_n, self.block_elig,
                      self.doubt, -1)
        if self.budget > 0:
            kernel.limit_inputs(self.n, self.m, self.tgt, self.weight, self.alive,
                                self.inhib, self.budget, self.input_total, self.plast,
                                self.protect_cons, self.hyp, self.assoc,
                                0, self.time, self.arr_time, self.fixed,
                                0.0, self.explores)
        if self.curiosity_on:
            if value < 0:
                self.level += P[CUR_UP] * min(1.0, -value) * (1.0 - self.level)
            elif value > 0:
                self.level -= P[CUR_DOWN] * min(1.0, value) * self.level
            else:
                self.level += P[CUR_IDLE] * (1.0 - self.level)
        if self.fam_on and value != 0:
            self.news_count += 1
            self.dishabituate(abs(value), -1)

    def step(self):
        P = self.P
        if self.curiosity_on:
            noise = P[CALM] + self.level * (P[CURIOUS] - P[CALM])
            for i in range(self.n):
                if self.explores[i]:
                    self.potential[i] += np.random.normal(0.0, noise)
            self.level *= P[CUR_DECAY]
        if self.nc_on:
            ip_rate, ip_target, window = P[NC_RATE], P[NC_TARGET], P[NC_WINDOW]
            lo, hi = P[NC_MIN], P[NC_MAX]
        else:
            ip_rate, ip_target, window, lo, hi = 0.0, 0.0, 0.0, -1e9, 1e9
        imp_decay = 1.0
        important = 1e18
        if P[FORGET_ON] > 0:
            imp_decay = P[IMP_DECAY]
            important = P[IMPORTANT]
        _, n_removed, n_confirmed = kernel.step(
            self.time,
            self.n, self.potential, self.refractory, self.spikes, self.threshold,
            P[REST], self.decay, self.refr_len,
            self.activity, ip_rate, ip_target, window, lo, hi, self.ip_fixed, False, self.push_in, -1e18,
            False, False, False, False, self.ctx_level, self.active, self.dec_n,
            self.m, self.src, self.tgt, self.weight, self.pre, self.post, self.thick,
            self.alive, self.weak, self.pending, self.plast, self.elig,
            self.inhib, self.n_inhib > 0, self.suppressed, int(P[SUPPRESS_STEPS]),
            P[PROTECT_INHIBITED] > 0,
            P[STRENGTHEN], P[WEAKEN], P[TRACE_DECAY], P[MIN_WEIGHT], P[MAX_WEIGHT],
            P[CONSOLIDATE], P[ELIG_DECAY],
            P[M_GROW], P[M_SHRINK], P[M_FADE],
            P[DEAD_WEIGHT], self.dead_patience,
            self.budget, self.input_total, self.protect_cons, 0, 0,
            self.hyp, self.hyp_tests, self.hyp_hits, self.hyp_needed,
            P[HYP_MIN_REL], P[HYP_LIFT], self.arr_time, self.drive, self.fire_time,
            P[CREDIT_ON] > 0, P[MYELIN_TIMING] > 0, int(P[TIMING_WINDOW]), int(P[MYELIN_IDLE]),
            self.n_ctx(), self.ctx_slot, self.ctx_w, int(P[CTX_HORIZON]), self.active,
            self.ctx_level, 0.0, self.ctx_trace,
            self.importance, imp_decay, important, self.doubt,
            self.explores, P[ACTION_MIN], self.last_share, self.confirm, self.syn_drive,
            int(P[NEG_NEEDED]), P[NECESSITY_ON] > 0, P[BAN_ON] > 0, self.ban, P[BAN_POWER],
            self.learn_gain, self.relax, self.w0, self.assoc, self.stim_time, self.perceptual,
            self.fixed, False, self.pool, self.recall, self.recall_in, 0.0, False, self.block_elig, False, False, False, True, self.fired, self.delivered, self.removed, self.received, self.confirmed)
        for k in range(n_removed):
            c = self.removed[k]
            self.pair_count[self.src[c], self.tgt[c]] -= 1
        if self.fam_on:
            self.fam_after_step()
        if self.growth_on:
            self.observe()
        self.time += 1

    # ----- рост (то же, что Growth.observe) -----

    def observe(self):
        P = self.P
        n0 = self.n
        steps_seen = self.time + 1
        fired_idx = np.empty(n0, dtype=np.int64)
        nf = 0
        for i in range(n0):
            if self.fired[i]:
                fired_idx[nf] = i
                nf += 1
        surprising = np.empty(nf, dtype=np.int64)
        ns = 0
        for k in range(nf):
            i = fired_idx[k]
            if self.received[i] < P[EXPLAIN] and not self.coincidence[i]:
                surprising[ns] = i
                ns += 1
        for gap in range(1, self.hist_count + 1):
            slot = (self.hist_head - gap) % self.G
            for s in range(n0):
                if not self.hist[slot, s]:
                    continue
                source_spikes = self.spikes[s]
                for k in range(ns):
                    t = surprising[k]
                    if s == t or self.pair_count[s, t] > 0:
                        continue
                    if not self.oc_active[s, t, gap]:
                        self.oc_active[s, t, gap] = True
                        self.oc_count[s, t, gap] = 0
                        self.oc_start[s, t, gap] = source_spikes - 1
                    self.oc_count[s, t, gap] += 1
                    count = self.oc_count[s, t, gap]
                    source_fires = source_spikes - self.oc_start[s, t, gap]
                    reliability = count / source_fires
                    target_rate = self.spikes[t] / steps_seen
                    if (count >= P[CONN_THRESHOLD] and reliability >= P[MIN_REL]
                            and reliability >= P[MIN_LIFT] * target_rate):
                        thickness = min(1.0, max(0.0, (5.0 - gap) / 4.0))
                        c = self.connect(s, t, P[NEW_WEIGHT], thickness, P[YOUNG_PLAST], False,
                                         P[HYP_ON] > 0)
                        self.grown_src[self.n_grown] = s
                        self.grown_tgt[self.n_grown] = t
                        self.grown_c[self.n_grown] = c
                        self.n_grown += 1
                        for g in range(1, self.G + 1):
                            self.oc_active[s, t, g] = False
                    elif source_fires >= P[FORGET_AFTER] and reliability < P[MIN_REL]:
                        self.oc_active[s, t, gap] = False
        for a in range(nf):
            for b in range(a + 1, nf):
                x = fired_idx[a]
                y = fired_idx[b]
                if self.pair_made[x, y]:
                    continue
                self.tog[x, y] += 1
                if self.tog[x, y] >= P[NEURON_THRESHOLD]:
                    self.grow_neuron(x, y)
                    self.tog[x, y] = 0
        # история: кто сработал на этом шаге (без только что выросших)
        for i in range(self.NC):
            self.hist[self.hist_head, i] = self.fired[i] if i < n0 else False
        self.hist_head = (self.hist_head + 1) % self.G
        self.hist_count = min(self.hist_count + 1, self.G)

    def grow_neuron(self, x, y):
        P = self.P
        if self.n >= P[MAX_NEURONS] or self.n >= self.NC:
            return
        out_t = np.empty(self.NC, dtype=np.int64)
        out_th = np.empty(self.NC)
        no = 0
        if P[INHERIT] > 0:
            for c in range(self.m):
                if not self.alive[c] or self.inhib[c]:
                    continue
                s = self.src[c]
                t = self.tgt[c]
                if (s == x or s == y) and t != x and t != y:
                    seen = False
                    for k in range(no):
                        if out_t[k] == t:
                            seen = True
                    if not seen:
                        out_t[no] = t
                        out_th[no] = self.thick[c]
                        no += 1
        if self.m + 2 + no > self.CC:
            raise ValueError("fast.Sim: не хватает места для связей нового нейрона (CC)")
        new = self.add_neuron(False, False)
        # детектор совпадения (как Brain.make_coincidence и Growth._grow_neuron)
        self.decay[new] = 0.0
        self.ip_fixed[new] = True
        self.coincidence[new] = True
        self.potential[new] = P[REST]
        w_in = PAIR_SHARE * (self.threshold[new] - P[REST])
        self.connect(x, new, w_in, 1.0, 1.0, False, False, True)
        self.connect(y, new, w_in, 1.0, 1.0, False, False, True)
        self.pair_made[x, y] = True
        self.n_grown_neurons += 1
        for k in range(no):
            th = 1.0 if P[FAST_NEW] > 0 else out_th[k]
            c = self.connect(new, out_t[k], P[NEW_WEIGHT], th, P[YOUNG_PLAST], False, False)
            self.grown_src[self.n_grown] = new
            self.grown_tgt[self.n_grown] = out_t[k]
            self.grown_c[self.n_grown] = c
            self.n_grown += 1


def _make_clone():
    """clone(a): полная копия мозга (для экзамена, чтобы не менять оригинал).

    Копируются все поля spec (список один, поэтому новое поле не потеряется)."""
    lines = ["def clone(a):", "    b = Sim(a.NC, a.CC, a.P)"]
    for name, typ in spec:
        if name in ("NC", "CC", "P"):
            continue
        if hasattr(typ, "ndim"):
            lines.append(f"    b.{name}[...] = a.{name}")
        else:
            lines.append(f"    b.{name} = a.{name}")
    lines.append("    return b")
    env = {"Sim": Sim}
    exec("\n".join(lines), env)
    return njit(env["clone"])


clone = _make_clone()


@njit
def attach(sim, curiosity, neurons):
    """Как tasks.attach: рост, любопытство, запас, изменение нейронов."""
    P = sim.P
    sim.growth_on = True
    sim.dead_patience = int(P[DEAD_PATIENCE])
    sim.curiosity_on = curiosity
    sim.level = 0.0
    sim.budget = P[BUDGET]
    sim.protect_cons = P[PROTECT_CONS] > 0
    sim.fam_on = P[FAM_ON] > 0
    sim.hyp_needed = int(P[HYP_NEEDED]) if P[HYP_ON] > 0 else 0
    if neurons and P[NC_ON] > 0:
        sim.nc_on = True
        for i in range(sim.n):
            sim.vary(i)


# ---------- задачи (как в tasks.py) ----------

NONE, LEFT, RIGHT, BOTH = 0, 1, 2, 3


@njit
def trial(sim, inputs, L, R):
    for i in range(inputs.shape[0]):
        if inputs[i]:
            sim.stimulate(i, 20.0)
    for _ in range(7):     # ответ смотрим с первого шага после показа
        sim.step()
        fl = sim.fired[L]
        fr = sim.fired[R]
        if fl or fr:
            if fl and fr:
                return BOTH
            return LEFT if fl else RIGHT
    return NONE


@njit
def rest(sim, steps):
    for _ in range(steps):
        sim.step()


@njit
def exam(sim, inputs, L, R):
    t = clone(sim)
    t.curiosity_on = False
    return trial(t, inputs, L, R)


@njit
def learn(sim, inputs, good, L, R, n):
    hits = 0
    for _ in range(n):
        out = trial(sim, inputs, L, R)
        if out == good:
            hits += 1
            sim.reward(1.0)
        elif out != NONE:
            sim.reward(-1.0)
        else:
            sim.reward(0.0)
        rest(sim, 20)
    return hits


@njit
def run_reward_task(P, NC, CC, seed, parts_inputs, parts_good):
    """Как tasks.run_task. Возвращает (все части верны, доля верных при обучении,
    вторая часть верна до обучения: 1/0, или -1 если части одна)."""
    np.random.seed(seed)
    sim = Sim(NC, CC, P)
    s1 = sim.add_neuron(False, False)
    s2 = sim.add_neuron(False, False)
    L = sim.add_neuron(True, False)
    R = sim.add_neuron(True, False)
    for s in (s1, s2):
        for o in (L, R):
            sim.connect(s, o, P[INIT_WEIGHT], 1.0, P[INIT_PLAST], False, False)
    sim.connect(L, R, P[INHIBIT_WEIGHT], P[DEFAULT_THICK], 1.0, True, False)
    sim.connect(R, L, P[INHIBIT_WEIGHT], P[DEFAULT_THICK], 1.0, True, False)
    attach(sim, True, True)
    n_parts = parts_good.shape[0]
    hits = 0
    before = -1
    for k in range(n_parts):
        if k == 1:
            before = 1 if exam(sim, parts_inputs[k], L, R) == parts_good[k] else 0
        hits += learn(sim, parts_inputs[k], parts_good[k], L, R, 60)
    ok = True
    for k in range(n_parts):
        if exam(sim, parts_inputs[k], L, R) != parts_good[k]:
            ok = False
    return ok, hits / (60.0 * n_parts), before, sim.weight[:sim.m].copy()


@njit
def run_context_task_fast(P, NC, CC, seed, items_inputs, items_good, block_of, n_blocks, n):
    """Как tasks.run_context_task. items_*: все пары подряд, block_of[k] = номер блока."""
    np.random.seed(seed)
    sim = Sim(NC, CC, P)
    for k in range(4):
        sim.add_neuron(False, k >= 2)     # S1, S2 стимулы; C1, C2 контексты
    L = sim.add_neuron(True, False)
    R = sim.add_neuron(True, False)
    for s in range(4):
        if P[CTX_ON] > 0 and s >= 2:
            continue   # с ветвями контекст только настраивает пути, действием не командует
        for o in (L, R):
            sim.connect(s, o, P[INIT_WEIGHT], 1.0, P[INIT_PLAST], False, False)
    sim.connect(L, R, P[INHIBIT_WEIGHT], P[DEFAULT_THICK], 1.0, True, False)
    sim.connect(R, L, P[INHIBIT_WEIGHT], P[DEFAULT_THICK], 1.0, True, False)
    attach(sim, True, True)
    hits = 0
    for blk in range(n_blocks):
        idx = np.empty(items_good.shape[0], dtype=np.int64)
        m = 0
        for k in range(items_good.shape[0]):
            if block_of[k] == blk:
                idx[m] = k
                m += 1
        for t in range(n):
            k = idx[t % m]
            out = trial(sim, items_inputs[k], L, R)
            if out == items_good[k]:
                hits += 1
                sim.reward(1.0)
            elif out != NONE:
                sim.reward(-1.0)
            else:
                sim.reward(0.0)
            rest(sim, 20)
    ok = True
    first_ok = 0
    first_n = 0
    for k in range(items_good.shape[0]):
        right = exam(sim, items_inputs[k], L, R) == items_good[k]
        if not right:
            ok = False
        if block_of[k] == 0:
            first_n += 1
            if right:
                first_ok += 1
    return ok, hits / (1.0 * n * n_blocks), first_ok / first_n, sim.weight[:sim.m].copy()


@njit
def memory_test(P, NC, CC):
    good = 0
    for gap in range(1, 5):
        sim = Sim(NC, CC, P)
        a = sim.add_neuron(False, False)
        b = sim.add_neuron(False, False)
        attach(sim, False, False)
        for rep in range(10):
            sim.stimulate(a, 20.0)
            for s in range(30):
                if s == gap:
                    sim.stimulate(b, 20.0)
                sim.step()
        t = clone(sim)
        before = t.spikes[b]
        t.stimulate(a, 20.0)
        rest(t, 10)
        ok10 = t.spikes[b] > before
        rest(sim, 2000)
        before = sim.spikes[b]
        sim.stimulate(a, 20.0)
        rest(sim, 10)
        if ok10 and sim.spikes[b] > before:
            good += 1
    return good / 4.0


@njit
def growth_run(P, NC, CC, stim):
    """Мозг из stim.shape[1] нейронов, толчки по расписанию stim[t, i].
    Возвращает выросшие пары, которые в конце живы и не являются непроверенной гипотезой."""
    sim = Sim(NC, CC, P)
    for i in range(stim.shape[1]):
        sim.add_neuron(False, False)
    attach(sim, False, False)
    for t in range(stim.shape[0]):
        for i in range(stim.shape[1]):
            if stim[t, i] != 0.0:
                sim.stimulate(i, stim[t, i])
        sim.step()
    keep = np.zeros(sim.n_grown, dtype=np.bool_)
    for k in range(sim.n_grown):
        c = sim.grown_c[k]
        keep[k] = sim.alive[c] and not sim.hyp[c]
    return sim.grown_src[:sim.n_grown][keep].copy(), sim.grown_tgt[:sim.n_grown][keep].copy()


@njit
def grass_test(P, NC, CC, presentations):
    """Как tasks.grass_test. Возвращает (рано X срабатывал, поздно, Y заметили,
    X снова замечают после награды)."""
    sim = Sim(NC, CC, P)
    x = sim.add_neuron(False, False)
    y = sim.add_neuron(False, False)
    o = sim.add_neuron(False, False)
    sim.connect(x, o, 18.0, P[DEFAULT_THICK], 1.0, False, False)
    sim.connect(y, o, 18.0, P[DEFAULT_THICK], 1.0, False, False)
    attach(sim, False, False)
    early = 0
    late = 0
    for k in range(presentations):
        before = sim.spikes[x]
        sim.stimulate(x, 20.0)
        rest(sim, 30)
        fired = sim.spikes[x] > before
        if k < 10 and fired:
            early += 1
        if k >= presentations - 10 and fired:
            late += 1
    before = sim.spikes[y]
    sim.stimulate(y, 20.0)
    rest(sim, 30)
    y_noticed = sim.spikes[y] > before
    for k in range(5):
        sim.stimulate(x, 20.0)
        rest(sim, 3)
        sim.reward(1.0)
        rest(sim, 27)
    before = sim.spikes[x]
    sim.stimulate(x, 20.0)
    rest(sim, 30)
    return early, late, y_noticed, sim.spikes[x] > before


def grass_score(P):
    early, late, y_noticed, regained = grass_test(P, NC_TASKS, CC_TASKS, 200)
    ignored = early >= 8 and late <= 1
    return (float(ignored) + float(y_noticed) + float(regained)) / 3.0


# ---------- обёртки (тот же вид результата, что у tasks) ----------

NC_TASKS = 16
CC_TASKS = 256


def task_arrays(parts):
    inputs = np.zeros((len(parts), NC_TASKS), dtype=np.bool_)
    good = np.zeros(len(parts), dtype=np.int64)
    for k, (names, answer) in enumerate(parts):
        for name in names:
            inputs[k, {"S1": 0, "S2": 1}[name]] = True
        good[k] = LEFT if answer == "L" else RIGHT
    return inputs, good


def growth_schedule(seed, link, steps=2000):
    """То же расписание толчков, что в tasks.growth_test (A=0, B=1, шум 2..11)."""
    rnd = random.Random(seed)
    stim = np.zeros((steps, 12))
    plan = {}
    for t in range(steps):
        for i in range(2, 12):
            if rnd.random() < 0.02:
                stim[t, i] += 20.0
        if rnd.random() < 0.02:
            stim[t, 0] += 20.0
            plan[t + 2] = True
        if link and plan.pop(t, False):
            stim[t, 1] += 20.0
    return stim


def growth_test(P, seed):
    found, false = False, 0
    for link in (True, False):
        src, tgt = growth_run(P, NC_TASKS, CC_TASKS, growth_schedule(seed, link))
        pairs = list(zip(src.tolist(), tgt.tolist()))
        false += sum(1 for g in pairs if g != (0, 1))
        if link:
            found = (0, 1) in pairs
    return (1.0 if found else 0.0) * max(0.0, 1.0 - 0.25 * false)


def context_arrays(blocks):
    names = {"S1": 0, "S2": 1, "C1": 2, "C2": 3}
    items = [(i, g, b) for b, block in enumerate(blocks) for i, g in block]
    inputs = np.zeros((len(items), NC_TASKS), dtype=np.bool_)
    good = np.zeros(len(items), dtype=np.int64)
    block_of = np.zeros(len(items), dtype=np.int64)
    for k, (i, g, b) in enumerate(items):
        for name in i:
            inputs[k, names[name]] = True
        good[k] = LEFT if g == "L" else RIGHT
        block_of[k] = b
    return inputs, good, block_of, len(blocks)


def run_context_task(p_or_P, name, seed, n=60):
    from human_brain.stages import tasks
    P = p_or_P if isinstance(p_or_P, np.ndarray) else params_array(p_or_P)
    inputs, good, block_of, nb = context_arrays(tasks.CONTEXT_TASKS[name])
    ok, hits, first, _ = run_context_task_fast(P, NC_TASKS, CC_TASKS, seed, inputs, good,
                                               block_of, nb, n)
    return bool(ok), float(hits), float(first)


def run_task(p_or_P, name, seed):
    from human_brain.stages import tasks
    P = p_or_P if isinstance(p_or_P, np.ndarray) else params_array(p_or_P)
    inputs, good = task_arrays(tasks.TASKS[name])
    ok, hits, before, _ = run_reward_task(P, NC_TASKS, CC_TASKS, seed, inputs, good)
    return bool(ok), float(hits), (None if before < 0 else bool(before))


def evaluate(p, seeds):
    """Как tasks.evaluate, но быстро."""
    from human_brain.stages import tasks
    P = params_array(p)
    detail = {}
    hits, before = [], []
    for name in tasks.REWARD_TASKS:
        res = [run_task(P, name, s) for s in seeds]
        detail[name] = sum(ok for ok, _, _ in res) / len(seeds)
        hits += [h for _, h, _ in res]
        before += [b for _, _, b in res if b is not None]
    detail["reward"] = sum(detail[n] for n in tasks.REWARD_TASKS) / len(tasks.REWARD_TASKS)
    detail["memory"] = float(memory_test(P, NC_TASKS, CC_TASKS))
    detail["growth"] = growth_test(P, seeds[0])
    detail["grass"] = grass_score(P)
    detail["hits"] = sum(hits) / len(hits)
    detail["before"] = sum(before) / len(before)
    score = (3 * detail["reward"] + detail["memory"] + detail["growth"] + detail["grass"]
             + 0.1 * detail["hits"])
    return score, detail
