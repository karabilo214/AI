"""Расчёт одного шага мозга на Numba.

Данные лежат в массивах (см. Brain.py), правила те же, что описаны
в Neuron.py, Synapse.py и Myelin.py.

Очередь сигналов: задержка всегда от 1 до 5 шагов, поэтому вместо кучи
у каждой связи есть кольцо на RING ячеек: pending[c, t % RING] = сколько
сигналов придёт по связи c в момент t.

Три механизма с переключателями (выключены -> поведение как раньше):
  гипотезы (hyp_needed > 0): связь с hyp[c] = True «немая»: сигнал по ней
      доставляется, но заряд получателя не меняет. Каждый приход сигнала =
      проверка предсказания «получатель сработает сейчас». После hyp_needed
      проверок: доля попаданий >= hyp_min_rel и в hyp_lift раз выше, чем
      получатель срабатывает вообще -> связь подтверждена (начинает
      действовать), иначе удалена. Проверяются только события после
      появления связи.
  заслуга (credit_on): метка связи для награды = её доля в срабатывании
      получателя (сила * след / заряд перед импульсом), а не просто «пришёл
      сигнал». necessity_on: и только если связь была нужна (без её вклада
      остального заряда не хватило бы до порога; доля от 0 до 1). При награде заслуга идёт назад от выходов-действий
      (explores) по цепочке: связи, не ведущие к действию, не меняются.
  миелин по времени (myelin_timing): миелин подстраивает время доставки,
      а не повторяет успех синапса: сигнал пришёл раньше срабатывания
      получателя -> связь медленнее; пришёл после -> быстрее; получатель
      молчал -> не меняется. Тает миелин только у связи, по которой
      idle_steps шагов не было сигнала (иначе таяние без противовеса
      медленно замедляло бы всю цепь).
  контексты (n_ctx > 0): у связи кроме общей силы есть добавка на каждый
      контекст ctx_w[c, k] («ветвь»). Контекст k активен, если его нейрон
      (ctx_slot == k) срабатывал за последние ctx_horizon шагов. Сигнал идёт
      с силой «общая + добавки активных контекстов». Награда в контексте
      меняет только добавку этого контекста, общая сила не трогается:
      обучение в одном контексте не стирает другой.
  управляемое забывание (forget_on): у связи есть важность (сколько
      награды она принесла, медленно гаснет без использования, imp_decay за
      шаг) и счёт сомнений. Важная связь (importance >= important) не
      удаляется как мёртвая. Закреплённая связь меняется от награды в
      consolidated_rate раз слабее (переучиваться трудно). Закреплённая связь, которая doubt_needed раз
      подряд заметно участвовала в наказанном действии, снова становится
      гибкой (пластичность relearn_plast): мир изменился, её можно
      переучить. Награда сбрасывает сомнения.
  защита путей к действиям (action_min > 0): связь к выходу-действию
      (explores) не удаляется как мёртвая, и её общая сила не опускается
      ниже action_min. Иначе после наказаний мог пропасть единственный путь
      к другому ответу, и переучиться было бы нельзя.
      Отрицательная память: путь, сведённый наказаниями к минимуму,
      закрепляется слабым (пластичность 1), если перед этим было neg_needed
      наказаний подряд с его участием: знание «так нельзя» хранится и
      наградой почти не меняется (изменения * neg_rate, пока путь закреплён
      и слабее порога закрепления). Освобождение по подтверждениям (при
      forget_on): если слабый закреплённый путь doubt_needed раз подряд был
      главной причиной (доля >= 0.5 среди сигналов по связям, без шума)
      действия, которое наградили, он снова гибкий (relearn_plast).
      Изменчивый запрет (ban_on, заменяет закрепление «нельзя»): у пути к
      действию запрет ban от 0 до 1. Наказание с его участием: ban += ban_up *
      (1 - ban). Награда, где он главная причина (доля >= 0.5): ban -= ban_down
      * ban. Рост силы (от награды и от «сработали друг за другом»)
      умножается на (1 - ban) ** ban_power.
  области: у нейрона t свой множитель обучения входящих связей
      learn_gain[t] (0 = связи жёсткие) и возврат relax[t]: каждый шаг сила
      входящей связи стекает к исходной w0 на долю relax[t] (быстрая память
      быстро учится и сама забывает). По умолчанию 1 и 0.
  закреплённые связи (fixed[c]): входы детекторов (нейрон-сочетание, нанятый
      нейрон памяти). Не учатся (сила, миелин), не удаляются как мёртвые и не
      сжимаются общим запасом входов: иначе детектор перестаёт узнавать своё.
  выбор победителя (pool_on): нейроны с одинаковым pool[i] >= 0 соревнуются.
      Если на шаге ни один нейрон группы не дошёл до порога сам, срабатывает
      тот, кто получил больше всех по связям вспоминания (recall[c], их
      ставит Association), если он единственный лидер и сигнал пришёл к нему
      от большинства (больше половины) того, что воспринято на прошлом шаге
      (сработало от толчка извне, вне групп соревнования): один общий
      признак из трёх слово не вызывает (иначе в начале обучения угадывается
      и проговаривается чужое, а новая картинка называет слово по одному
      знакомому нейрону).
      Знание «какое слово ближе всех к картинке» есть и при слабых связях,
      а порог при многих картинках, делящих нейроны, не достигается.
  ассоциации (assoc[c]): связь «бывает вместе» (Association.py). Её силу
      задаёт модуль ассоциаций; обычные правила (усиление, «опоздал»,
      награда, миелин, удаление) её не меняют. Сигнал по ней идёт, только
      если источник сработал от внешнего толчка (stim_time == time) или
      сам часть восприятия (perceptual, например память слуха).
"""
import numpy as np
from numba import njit

RING = 8           # больше самой длинной задержки (5)
MAX_CONTEXTS = 16  # сколько контекстов может различать мозг
NEVER = -10 ** 9   # «давно не срабатывал / не приходил сигнал»
LEVEL_MIN = 0.05   # плавный контекст слабее этого считается выключенным


@njit(cache=True)
def active_contexts(n_neurons, ctx_slot, fire_time, time, ctx_horizon, active,
                    level, graded, ctx_trace):
    """active[k] = контекст k активен, level[k] = его уровень. Возвращает число активных.

    Обычный режим (graded = 0): контекст активен, если его нейрон срабатывал
    за последние ctx_horizon шагов, уровень 1.
    Плавный режим (graded > 0, контекстная модуляция): уровень = медленный
    след активности нейрона (ctx_trace, 0..1), активен при уровне >= LEVEL_MIN.
    """
    for k in range(MAX_CONTEXTS):
        active[k] = False
        level[k] = 0.0
    n = 0
    for i in range(n_neurons):
        k = ctx_slot[i]
        if k < 0:
            continue
        if graded > 0.0:
            if ctx_trace[i] > level[k]:
                level[k] = ctx_trace[i]
        elif time - fire_time[i] <= ctx_horizon and not active[k]:
            active[k] = True
            level[k] = 1.0
            n += 1
    if graded > 0.0:
        for k in range(MAX_CONTEXTS):
            if level[k] >= LEVEL_MIN:
                active[k] = True
                n += 1
            else:
                level[k] = 0.0
    return n


@njit(cache=True)
def effective(c, weight, ctx_w, level, n_active, min_weight, max_weight):
    """Сила связи с учётом добавок активных контекстов: w + сумма level[k] * ctx_w[c, k]
    (в обычном режиме level = 1 у активных, 0 у остальных)."""
    w = weight[c]
    if n_active > 0:
        for k in range(MAX_CONTEXTS):
            if level[k] > 0.0:
                w += level[k] * ctx_w[c, k]
        w = max(min_weight, min(max_weight, w))
    return w


@njit(cache=True)
def step(time,
         # нейроны
         n_neurons, potential, refractory, spikes, threshold,
         rest, decay, refractory_len,
         # изменение нейронов (ip_rate = 0 -> выключено)
         activity, ip_rate, ip_target, activity_window, min_threshold, max_threshold,
         ip_fixed,   # True -> порог этого нейрона не подстраивается
         # ip_push: подстройка по полученному толчку (push_in + received), а не
         # по срабатыванию; push_in копит внешние толчки шага (Brain.stimulate)
         ip_push, push_in,
         v_floor,    # заряд не опускается ниже (торможение), -1e18 = без границы
         # inh_short: торможение короткое — снимает текущий заряд, но заряд ниже
         # покоя после шага не остаётся (иначе соперник ~10 шагов не может ответить)
         inh_short,
         # last_action_only: новое действие стирает метки связей к действиям
         # (награда относится к последнему действию, а не к рефлексу до него)
         last_action_only,
         # myelin_keep: при миелине по времени сигнал, пришедший вовремя (получатель
         # сработал в тот же шаг), поддерживает миелин; иначе связь «вовремя»
         # ничем не укрепляется и в долгих паузах тает, задержка растёт
         myelin_keep,
         # decision_ctx: при срабатывании действия запомнить уровни контекстов
         # (dec_level, dec_active, dec_n[0]); награда учит контекст момента решения
         decision_ctx, dec_level, dec_active, dec_n,
         # связи
         n_conn, src, tgt, weight, pre_trace, post_trace, thickness,
         alive, weak_steps, pending, plasticity, eligibility,
         inhibitory, has_inhibition, suppressed, suppress_steps, protect_inhibited,
         # правила синапса
         strengthen, weaken, trace_decay, min_weight, max_weight,
         consolidate_weight, eligibility_decay,
         # правила миелина
         grow, shrink, fade,
         # удаление мёртвых связей (dead_patience = 0 -> выключено)
         dead_weight, dead_patience,
         # общий запас силы входов нейрона (0 -> выключено)
         input_budget, input_total, protect_consolidated,
         # запас только на входы, сигнал по которым пришёл за последние
         # budget_window шагов (0 -> на все входы, как раньше)
         budget_window,
         # обучаемое торможение: окно (0 -> выключено), см. шаг 3
         inh_window,
         # гипотезы, заслуга, миелин по времени
         hyp, hyp_tests, hyp_hits, hyp_needed, hyp_min_rel, hyp_lift,
         arr_time, drive, fire_time, credit_on, myelin_timing, timing_window, idle_steps,
         # контексты
         n_ctx, ctx_slot, ctx_w, ctx_horizon, active,
         # уровни контекстов; плавный режим (ctx_graded = затухание следа, 0 = обычный)
         ctx_level, ctx_graded, ctx_trace,
         # управляемое забывание
         importance, imp_decay, important, doubt,
         # защита путей к действиям
         explores, action_min, last_share, confirm, syn_drive, neg_needed,
         # необходимость для заслуги; изменчивый запрет
         necessity_on, ban_on, ban, ban_power,
         # области
         learn_gain, relax, w0, assoc, stim_time, perceptual, fixed,
         # выбор победителя
         pool_on, pool, recall, recall_in,
         # запас входов считать сверх минимума путей к действиям (0 = как раньше)
         budget_floor,
         # inh_delayed: обучаемое торможение (вход -> действие) приходит по
         # очереди с задержкой, как возбуждение, и вычитается при доставке;
         # иначе оно бьёт в шаг срабатывания источника, на шаг раньше
         # возбуждения от того же источника, и ничего не подавляет
         inh_delayed,
         # block_elig: метка «помешал» у запаздывающего обучаемого торможения:
         # без этой связи цель сработала бы. Награда её усиливает, наказание
         # ослабляет (kernel.reward); иначе торможение, однажды подавившее цель,
         # уже нечем ослабить (метка ставится, только если цель сработала).
         # Проверка «без всего торможения, по долям» на 24 запусках вычитания
         # хуже (всё вместе 8/24 против 15/24), не использовать
         block_elig,
         # reward_only_actions: связи к действиям (explores) меняются только по
         # награде (три фактора), без «сработали друг за другом» и «опоздал»,
         # и их миелин не тает от простоя: иначе сотни повторов без наград
         # превращают выбор в привычку, а долгий простой сбивает такт
         reward_only_actions,
         # inh_share: метка обучаемого торможения делится поровну между всеми
         # тормозящими связями, участвовавшими в срабатывании цели (как доля
         # заслуги у возбуждающих); иначе при десятках источников одно
         # наказание добавляет каждому полный шаг, и торможение взрывается
         inh_share,
         # block_collective: метка «помешал» по всему пришедшему обучаемому
         # торможению (без него цель сработала бы), каждой связи по доле силы;
         # нужно, когда тормозов много (выбор заготовки нейронами). На вычитании
         # (2–3 тормоза) хуже: 8/24 против 15/24, там выключено
         block_collective,
         # выходы
         fired, delivered, removed, received, confirmed):
    slot = time % RING
    n_active = 0
    if n_ctx > 0:
        n_active = active_contexts(n_neurons, ctx_slot, fire_time, time, ctx_horizon, active,
                                   ctx_level, ctx_graded, ctx_trace)

    # 1. Доставляем сигналы, пришедшие к этому шагу
    #    received[i] = сколько нейрон i получил по связям на этом шаге
    for i in range(n_neurons):
        received[i] = 0.0
        recall_in[i] = 0.0
    for c in range(n_conn):
        delivered[c] = False
        if not alive[c]:
            continue
        k = pending[c, slot]
        if k == 0:
            continue
        pending[c, slot] = 0
        delivered[c] = True
        t = tgt[c]
        if inhibitory[c]:
            # (сюда попадает только запаздывающее обучаемое торможение)
            wi = effective(c, weight, ctx_w, ctx_level, n_active, 0.0, max_weight)
            for _ in range(k):
                potential[t] = max(v_floor, potential[t] - wi)
            arr_time[c] = time
            continue
        for _ in range(k):
            # получатель недавно сработал без этого сигнала: сигнал опоздал
            if not assoc[c] and not fixed[c] and not (reward_only_actions and explores[t]):
                w = weight[c] - weaken * plasticity[c] * post_trace[c] * learn_gain[t]
                weight[c] = max(min_weight, min(max_weight, w))
            pre_trace[c] += 1.0
            if not hyp[c]:            # гипотеза «немая»: заряд не меняет
                we = effective(c, weight, ctx_w, ctx_level, n_active, min_weight, max_weight)
                potential[t] += we
                received[t] += we
                if recall[c]:
                    recall_in[t] += we
        if myelin_timing and not hyp[c] and not assoc[c] and not fixed[c]:
            late = time - fire_time[t]
            if late >= 1 and late <= timing_window:
                # пришёл после срабатывания получателя: ускорить
                thickness[c] += grow * (1.0 - thickness[c])
        arr_time[c] = time

    # заряд после доставки (для метки «помешал») и всё пришедшее обучаемое торможение
    pot_before = np.empty(n_neurons if inh_delayed else 0)
    inh_total = np.zeros(n_neurons if (inh_delayed and block_collective) else 0)
    if inh_delayed:
        for i in range(n_neurons):
            pot_before[i] = potential[i]
        if block_collective:
            for c in range(n_conn):
                if delivered[c] and alive[c] and inhibitory[c]:
                    inh_total[tgt[c]] += effective(c, weight, ctx_w, ctx_level, n_active,
                                                   0.0, max_weight)

    # 1б. Выбор победителя в группах соревнования
    if pool_on:
        choose_winners(n_neurons, pool, potential, threshold, refractory, recall_in,
                       n_conn, src, tgt, alive, delivered, recall, weight, hyp,
                       time, fire_time, stim_time)

    # 2. Обновляем нейроны
    if has_inhibition:
        n_fired = fire_with_inhibition(
            n_neurons, potential, refractory, spikes, threshold,
            rest, decay, refractory_len,
            n_conn, src, tgt, weight, alive, inhibitory, fired,
            suppressed, suppress_steps, drive, fire_time, time, v_floor,
            inh_short, explores, ctx_w, ctx_level, n_active, max_weight, inh_delayed)
    else:
        n_fired = 0
        for i in range(n_neurons):
            fired[i] = False
            if refractory[i] > 0:
                refractory[i] -= 1
                potential[i] = rest
                continue
            if potential[i] >= threshold[i]:
                spikes[i] += 1
                drive[i] = potential[i] - rest
                fire_time[i] = time
                potential[i] = rest
                refractory[i] = refractory_len[i]
                fired[i] = True
                n_fired += 1
                continue
            potential[i] = rest + (potential[i] - rest) * decay[i]

    # 2б. Нейроны меняются: часто срабатывающий повышает порог, молчащий понижает
    if ip_rate > 0.0:
        for i in range(n_neurons):
            activity[i] = activity[i] * activity_window + (1.0 if fired[i] else 0.0)
            if ip_fixed[i]:
                continue
            if ip_push:
                # шаг считается активным, если нейрон сработал или получил толчок
                # не меньше половины расстояния до порога: знакомое, которое
                # толкают, но уже не доводят до порога, не считается «молчащим»;
                # постоянный слабый вход активностью не считается (единицы те же,
                # что у целевой доли срабатываний)
                x = (push_in[i] + max(0.0, received[i])) / max(threshold[i] - rest, 1e-9)
                act = 1.0 if (fired[i] or x >= 0.5) else 0.0
                activity[i] = (activity[i] - (1.0 if fired[i] else 0.0)) + act
            rate = activity[i] * (1.0 - activity_window)   # доля шагов со срабатыванием
            t = threshold[i] + ip_rate * (rate - ip_target)
            threshold[i] = max(min_threshold, min(max_threshold, t))

    for i in range(n_neurons):
        push_in[i] = 0.0
    action_now = False
    if last_action_only or decision_ctx:
        for i in range(n_neurons):
            if fired[i] and explores[i]:
                action_now = True
                break
    if decision_ctx and action_now:
        for k in range(MAX_CONTEXTS):
            dec_level[k] = ctx_level[k]
            dec_active[k] = active[k]
        dec_n[0] = n_active
    clear_actions = last_action_only and action_now
    # 2б'. След активности контекстных нейронов (плавный режим)
    if ctx_graded > 0.0:
        for i in range(n_neurons):
            if ctx_slot[i] >= 0:
                ctx_trace[i] = 1.0 if fired[i] else ctx_trace[i] * ctx_graded

    # 2в. Доля каждой связи среди сигналов по связям у сработавших нейронов
    if action_min > 0.0:
        for i in range(n_neurons):
            syn_drive[i] = 0.0
        for c in range(n_conn):
            if alive[c] and not inhibitory[c] and not hyp[c] and fired[tgt[c]]:
                syn_drive[tgt[c]] += effective(c, weight, ctx_w, ctx_level, n_active,
                                               min_weight, max_weight) * pre_trace[c]

    # число обучаемых тормозящих связей, участвовавших в срабатывании цели
    inh_part = np.zeros(n_neurons if (inh_share and inh_window > 0) else 0)
    if inh_share and inh_window > 0:
        for c in range(n_conn):
            if (alive[c] and inhibitory[c] and learn_inhibit(c, src, tgt, explores)
                    and fired[tgt[c]] and time - fire_time[src[c]] <= inh_window):
                inh_part[tgt[c]] += 1.0

    # 3. Синапсы, миелин, проверка гипотез и удаление мёртвых связей
    #    (тормозящие связи постоянные: не учатся и не удаляются)
    n_removed = 0
    n_confirmed = 0
    for c in range(n_conn):
        if clear_actions and alive[c] and explores[tgt[c]] and not hyp[c] and not assoc[c]:
            eligibility[c] = 0.0       # новое действие: метки прошлого действия стираются
            if inhibitory[c]:
                block_elig[c] = 0.0
        if inh_window > 0 and alive[c] and inhibitory[c] and learn_inhibit(c, src, tgt, explores):
            # обучаемое торможение (вход/память -> действие): метка, если действие
            # сработало, когда источник торможения срабатывал не больше inh_window
            # шагов назад. Наказание усиливает торможение, награда ослабляет
            # (kernel.reward). Торможение между действиями постоянное.
            eligibility[c] *= eligibility_decay
            block_elig[c] *= eligibility_decay
            if fired[tgt[c]] and time - fire_time[src[c]] <= inh_window:
                eligibility[c] += 1.0 / inh_part[tgt[c]] if inh_share else 1.0
            if inh_delayed and delivered[c] and not fired[tgt[c]]:
                wi = effective(c, weight, ctx_w, ctx_level, n_active, 0.0, max_weight)
                if block_collective:
                    tot = inh_total[tgt[c]]
                    if wi > 0.0 and pot_before[tgt[c]] + tot >= threshold[tgt[c]]:
                        block_elig[c] += wi / tot   # без торможения цель сработала бы
                elif wi > 0.0 and pot_before[tgt[c]] + wi >= threshold[tgt[c]]:
                    block_elig[c] += 1.0     # без этой связи цель сработала бы
            continue
        if not alive[c] or inhibitory[c]:
            continue
        t = tgt[c]
        target_fired = fired[t]

        if assoc[c] or fixed[c]:
            # силу ассоциации задаёт модуль ассоциаций; закреплённая не меняется
            pre_trace[c] *= trace_decay
            continue

        if hyp[c]:
            # гипотеза учится тихо (сила копится), но награды и миелина нет
            if target_fired:
                w = weight[c] + strengthen * plasticity[c] * pre_trace[c] * learn_gain[t]
                weight[c] = max(min_weight, min(max_weight, w))
                post_trace[c] += 1.0
                if plasticity[c] > 1.0 and weight[c] >= consolidate_weight:
                    plasticity[c] = 1.0
            pre_trace[c] *= trace_decay
            post_trace[c] *= trace_decay
            if delivered[c]:
                hyp_tests[c] += 1
                if target_fired:
                    hyp_hits[c] += 1
            if hyp_needed > 0 and hyp_tests[c] >= hyp_needed:
                reliability = hyp_hits[c] / hyp_tests[c]
                target_rate = spikes[t] / (time + 1.0)
                if reliability >= hyp_min_rel and reliability >= hyp_lift * target_rate:
                    hyp[c] = False                 # подтверждена: начинает действовать
                    confirmed[n_confirmed] = c
                    n_confirmed += 1
                else:
                    alive[c] = False               # не подтвердилась: удалить
                    pending[c, :] = 0
                    removed[n_removed] = c
                    n_removed += 1
            continue

        # метка для награды гаснет; связь, которая помогла, получает новую
        eligibility[c] *= eligibility_decay
        if target_fired and action_min > 0.0:
            we0 = effective(c, weight, ctx_w, ctx_level, n_active, min_weight, max_weight)
            last_share[c] = we0 * pre_trace[c] / syn_drive[t] if syn_drive[t] > 0.0 else 0.0
        if target_fired:
            if credit_on:
                # заслуга = доля этой связи в заряде, который зажёг получателя
                we = effective(c, weight, ctx_w, ctx_level, n_active, min_weight, max_weight)
                x = we * pre_trace[c]
                share = x / max(drive[t], 1e-9)
                need = 1.0
                if necessity_on and x > 0.0:
                    # была ли связь нужна: хватило бы получателю остального заряда?
                    gap = threshold[t] - rest
                    need = min(1.0, max(0.0, (gap - (drive[t] - x)) / x))
                eligibility[c] += pre_trace[c] * min(1.0, share) * need
            else:
                eligibility[c] += pre_trace[c]
            # получатель сработал вскоре после прихода сигнала: сигнал помог
            grow_w = strengthen * plasticity[c] * pre_trace[c] * learn_gain[t]
            if reward_only_actions and explores[t]:
                grow_w = 0.0      # путь к действию меняет только награда
            if ban_on and explores[t]:
                grow_w *= (1.0 - ban[c]) ** ban_power   # запрет сдерживает и этот рост
            w = weight[c] + grow_w
            weight[c] = max(min_weight, min(max_weight, w))
            post_trace[c] += 1.0
            # молодая связь набрала силу, чтобы одной зажигать получателя: закрепляется
            # (освобождённая сомнениями, doubt < 0, ждёт подтверждения наградой)
            if plasticity[c] > 1.0 and weight[c] >= consolidate_weight and doubt[c] >= 0:
                plasticity[c] = 1.0
                confirm[c] = 0
        pre_trace[c] *= trace_decay
        post_trace[c] *= trace_decay

        if myelin_timing:
            if target_fired:
                early = time - arr_time[c]
                if early >= 1 and early <= timing_window:
                    # пришёл раньше срабатывания получателя: замедлить
                    thickness[c] -= shrink * thickness[c]
                elif myelin_keep and early == 0:
                    # пришёл вовремя: связь в деле, миелин держится
                    thickness[c] += grow * (1.0 - thickness[c])
            elif (not delivered[c] and time - arr_time[c] > idle_steps
                    and not (reward_only_actions and explores[t])):
                # тает только у связи, по которой давно не было сигнала (путь к
                # действию при reward_only_actions не тает: долгий простой между
                # вопросами иначе удлиняет задержку и сбивает такт ответа)
                thickness[c] -= fade * thickness[c]
        elif delivered[c]:
            if target_fired:
                thickness[c] += grow * (1.0 - thickness[c])
            elif not (protect_inhibited and suppressed[t] > 0):
                # сигнал пришёл впустую; но если получателя только что
                # затормозил соперник, сигнал не бесполезен, а проиграл
                thickness[c] -= shrink * thickness[c]
        else:
            thickness[c] -= fade * thickness[c]

        importance[c] *= imp_decay
        if dead_patience > 0:
            if weight[c] < dead_weight:
                weak_steps[c] += 1
            else:
                weak_steps[c] = 0
            if (weak_steps[c] >= dead_patience and importance[c] < important
                    and not (action_min > 0.0 and explores[t])):
                alive[c] = False
                pending[c, :] = 0   # сигналы в пути пропадают
                removed[n_removed] = c
                n_removed += 1

    # 3а. Возврат к исходной силе (быстрая память сама забывает)
    for c in range(n_conn):
        if alive[c] and not inhibitory[c] and not fixed[c] and relax[tgt[c]] > 0.0:
            weight[c] += relax[tgt[c]] * (w0[c] - weight[c])

    # 3б. Общий запас силы входов: если входы нейрона вместе сильнее запаса,
    #     все они пропорционально уменьшаются (рост одного отнимает у других)
    if input_budget > 0.0:
        limit_inputs(n_neurons, n_conn, tgt, weight, alive, inhibitory,
                     input_budget, input_total, plasticity, protect_consolidated, hyp, assoc,
                     budget_window, time, arr_time, fixed, budget_floor, explores)

    # 3в. Пути к действиям не слабеют ниже минимума; сведённый к минимуму
    #     путь закрепляется слабым (отрицательная память), если он не освобождён
    #     подтверждениями (confirm < 0)
    if action_min > 0.0:
        for c in range(n_conn):
            if (alive[c] and not inhibitory[c] and not hyp[c] and not fixed[c]
                    and explores[tgt[c]]):
                if weight[c] <= action_min:
                    weight[c] = action_min
                    # «так нельзя» только после neg_needed наказаний подряд,
                    # а не от одной (возможно, ошибочной) награды
                    if (not ban_on and confirm[c] >= 0 and plasticity[c] > 1.0
                            and doubt[c] >= neg_needed):
                        plasticity[c] = 1.0
                        doubt[c] = 0

    # 4. Отправляем новые сигналы (тормозящие уже сработали в шаге 2).
    #    Ассоциация передаёт сигнал, только если источник сработал от внешнего
    #    толчка (увидел/услышал): вызванное изнутри дальше по ассоциациям не идёт,
    #    иначе эхо зрение <-> слух ходит по кругу без конца.
    for c in range(n_conn):
        if assoc[c] and stim_time[src[c]] != time and not perceptual[src[c]]:
            continue
        if (alive[c] and fired[src[c]] and inhibitory[c] and inh_delayed
                and learn_inhibit(c, src, tgt, explores)):
            d = delay(thickness[c])
            pending[c, (time + d) % RING] += 1
            continue
        if alive[c] and not inhibitory[c] and fired[src[c]]:
            d = delay(thickness[c])
            pending[c, (time + d) % RING] += 1

    return n_fired, n_removed, n_confirmed


@njit(cache=True)
def choose_winners(n_neurons, pool, potential, threshold, refractory, recall_in,
                   n_conn, src, tgt, alive, delivered, recall, weight, hyp,
                   time, fire_time, stim_time):
    """В каждой группе, где никто не дошёл до порога сам, единственный лидер
    по входу вспоминания, получивший сигнал от большинства воспринятого на
    прошлом шаге, доводится до порога (сработает в шаге 2)."""
    n_pools = 0
    for i in range(n_neurons):
        if pool[i] + 1 > n_pools:
            n_pools = pool[i] + 1
    if n_pools == 0:
        return
    # воспринято на прошлом шаге: сработало от толчка извне (вне групп)
    n_sent = 0
    for i in range(n_neurons):
        if pool[i] < 0 and fire_time[i] == time - 1 and stim_time[i] == time - 1:
            n_sent += 1
    # от скольких из них дошёл сигнал вспоминания до каждого нейрона групп
    got = np.zeros(n_neurons, dtype=np.int64)
    for c in range(n_conn):
        s = src[c]
        if (alive[c] and delivered[c] and recall[c] and not hyp[c] and weight[c] > 0.0
                and pool[tgt[c]] >= 0 and fire_time[s] == time - 1 and stim_time[s] == time - 1):
            got[tgt[c]] += 1
    crossed = np.zeros(n_pools, dtype=np.bool_)
    best = np.full(n_pools, -1, dtype=np.int64)
    best_v = np.zeros(n_pools)
    second_v = np.zeros(n_pools)
    for i in range(n_neurons):
        g = pool[i]
        if g < 0 or refractory[i] > 0:
            continue
        if potential[i] >= threshold[i]:
            crossed[g] = True
        v = recall_in[i]
        if v > best_v[g]:
            second_v[g] = best_v[g]
            best_v[g] = v
            best[g] = i
        elif v > second_v[g]:
            second_v[g] = v
    for g in range(n_pools):
        if (not crossed[g] and best[g] >= 0 and best_v[g] > second_v[g]
                and 2 * got[best[g]] > n_sent):
            w = best[g]
            potential[w] = max(potential[w], threshold[w])


@njit(cache=True)
def fire_with_inhibition(n_neurons, potential, refractory, spikes, threshold,
                         rest, decay, refractory_len,
                         n_conn, src, tgt, weight, alive, inhibitory, fired,
                         suppressed, suppress_steps, drive, fire_time, time, v_floor,
                         inh_short, explores, ctx_w, ctx_level, n_active, max_weight,
                         inh_delayed):
    """Срабатывание нейронов с быстрым торможением (в тот же шаг).

    Нейроны выше порога срабатывают по очереди: первым тот, кто выше
    всех превысил порог. Сработавший сразу снижает заряд тех, к кому
    ведут его тормозящие связи, и они могут уже не сработать.
    """
    resting = np.zeros(n_neurons, dtype=np.bool_)  # отдыхают на этом шаге
    hit = np.zeros(n_neurons, dtype=np.bool_)      # заторможены на этом шаге
    for i in range(n_neurons):
        fired[i] = False
        if suppressed[i] > 0:      # сколько ещё шагов нейрон считается заторможенным
            suppressed[i] -= 1
        if refractory[i] > 0:
            refractory[i] -= 1
            potential[i] = rest
            resting[i] = True

    n_fired = 0
    while True:
        best = -1
        best_margin = 0.0
        for i in range(n_neurons):
            if resting[i] or fired[i]:
                continue
            margin = potential[i] - threshold[i]
            if margin >= 0.0 and (best < 0 or margin > best_margin):
                best = i
                best_margin = margin
        if best < 0:
            break
        spikes[best] += 1
        drive[best] = potential[best] - rest
        fire_time[best] = time
        potential[best] = rest
        refractory[best] = refractory_len[best]
        fired[best] = True
        n_fired += 1
        for c in range(n_conn):
            if alive[c] and inhibitory[c] and src[c] == best:
                if inh_delayed and explores[tgt[c]] and not explores[src[c]]:
                    continue      # обучаемое торможение придёт по очереди
                w = weight[c]
                if n_active > 0 and explores[tgt[c]] and not explores[src[c]]:
                    # обучаемое торможение тоже зависит от контекста (w + сумма m * g)
                    w = effective(c, weight, ctx_w, ctx_level, n_active, 0.0, max_weight)
                if w > 0.0:
                    # торможение не опускает заряд ниже v_floor (как в мозге:
                    # ниже равновесного потенциала торможения не уйти)
                    potential[tgt[c]] = max(v_floor, potential[tgt[c]] - w)
                    suppressed[tgt[c]] = suppress_steps
                    hit[tgt[c]] = True

    for i in range(n_neurons):
        if not resting[i] and not fired[i]:
            if inh_short and hit[i] and potential[i] < rest:
                potential[i] = rest      # короткое торможение: долга ниже покоя не остаётся
            potential[i] = rest + (potential[i] - rest) * decay[i]
    return n_fired


@njit(cache=True)
def learn_inhibit(c, src, tgt, explores):
    """Учится ли тормозящая связь: к действию и не от другого действия."""
    return explores[tgt[c]] and not explores[src[c]]


@njit(cache=True)
def limit_inputs(n_neurons, n_conn, tgt, weight, alive, inhibitory, budget, total,
                 plasticity, protect_consolidated, hyp, assoc,
                 window, time, arr_time, fixed, floor, explores):
    """Общая сила возбуждающих входов каждого нейрона не больше budget.

    protect_consolidated: лишнее срезается сначала с молодых связей
    (пластичность > 1); закреплённые урезаются, только если они одни
    больше запаса (тогда молодые обнуляются, закреплённые сжимаются).
    Иначе все входы сжимаются пропорционально. Гипотезы (немые) и ассоциации
    (у них своя нормировка) и закреплённые входы детекторов не считаются и
    не сжимаются.
    window > 0: считаются и сжимаются только входы, сигнал по которым пришёл
    за последние window шагов (работали вместе); остальные не трогаются,
    так что неиспользуемые пути в большой сети не душатся.
    floor > 0 (Brain.budget_above_floor): пути к действиям не опускаются ниже
    минимума floor, поэтому запас считается и сжимается только сверх него: иначе минимум
    каждый шаг возвращал бы силу, которую запас снимает, и запас медленно
    съедал бы сильный путь (замечено в addition.py: GO -> DONE 20 -> 10 за
    7 дней без наград).
    """
    young = np.zeros(n_neurons)
    for i in range(n_neurons):
        total[i] = 0.0
    for c in range(n_conn):
        if alive[c] and not inhibitory[c] and not hyp[c] and not assoc[c] and not fixed[c]:
            if window > 0 and time - arr_time[c] > window:
                continue
            base = floor if (floor > 0.0 and explores[tgt[c]]) else 0.0
            x = max(0.0, weight[c] - base)
            total[tgt[c]] += x
            if plasticity[c] > 1.0:
                young[tgt[c]] += x
    for c in range(n_conn):
        if not alive[c] or inhibitory[c] or hyp[c] or assoc[c] or fixed[c]:
            continue
        if window > 0 and time - arr_time[c] > window:
            continue
        t = tgt[c]
        if total[t] <= budget:
            continue
        base = floor if (floor > 0.0 and explores[t]) else 0.0
        x = max(0.0, weight[c] - base)
        if not protect_consolidated:
            weight[c] = base + x * budget / total[t]
            continue
        excess = total[t] - budget
        old = total[t] - young[t]
        if young[t] >= excess:
            if plasticity[c] > 1.0:
                weight[c] = base + x * (young[t] - excess) / young[t]
        elif plasticity[c] > 1.0:
            weight[c] = base
        else:
            weight[c] = base + x * budget / old


@njit(cache=True)
def reward(value, outcome, n_neurons, n_conn, src, tgt, weight, eligibility, plasticity, alive,
           inhibitory, explores, fire_time, time, credit_on, horizon, relevance,
           rate, min_weight, max_weight, consolidate_weight,
           n_ctx, ctx_slot, ctx_w, ctx_horizon, active, ctx_level, ctx_graded, ctx_trace,
           forget_on, importance, doubt, doubt_needed, relearn_plast, consolidated_rate,
           last_share, confirm, neg_rate, ban_on, ban, ban_up, ban_down, ban_power,
           learn_gain, decision_ctx, dec_level, dec_active, dec_n, block_elig):
    """Награда (value > 0) или наказание (value < 0) помеченным связям.

    Изменение = rate * value * пластичность * метка (* значимость получателя,
    если credit_on): молодые связи учатся от награды быстро, закреплённые
    почти не меняются.

    Значимость (credit_on): выход-действие (explores), сработавший за
    последние horizon шагов, = 1. Назад по цепочке: значимость источника =
    наибольшее (метка связи, не больше 1) * значимость получателя, 3 шага.
    Каждый шаг считается по снимку прошлого шага, поэтому глубина ровно 3
    ребра и не зависит от порядка, в котором создавались связи.
    Связи к нейронам, которые на действие не повлияли, не меняются.
    Если выходов-действий нет, значимость у всех 1 (как без credit_on).

    Контексты (n_ctx > 0): если какой-то контекст активен, меняется не общая
    сила, а добавка активных контекстов (поровну между ними).
    Плавные контексты: добавка k меняется на dw * m_k, общая сила — на
    dw * (1 - сумма m), если сумма меньше 1 (то, что память не объясняет).
    decision_ctx: уровни контекстов берутся из момента решения (dec_*).

    value — чему учить (награда или ошибка её предсказания); outcome — сама
    награда: правила, которые смотрят на знак (запрет, сомнения, важность),
    берут знак у неё, а не у ошибки предсказания.
    """
    n_active = 0
    if n_ctx > 0:
        n_active = active_contexts(n_neurons, ctx_slot, fire_time, time, ctx_horizon, active,
                                   ctx_level, ctx_graded, ctx_trace)
        if decision_ctx:
            for k in range(MAX_CONTEXTS):
                ctx_level[k] = dec_level[k]
                active[k] = dec_active[k]
            n_active = dec_n[0]
    base_share = 1.0
    if n_active > 0:
        base_share = 0.0
        if ctx_graded > 0.0:
            total = 0.0
            for k in range(MAX_CONTEXTS):
                if active[k]:
                    total += ctx_level[k]
            base_share = max(0.0, 1.0 - total)
    if credit_on:
        any_action = False
        for i in range(n_neurons):
            relevance[i] = 0.0
            if explores[i]:
                any_action = True
                if time - fire_time[i] <= horizon:
                    relevance[i] = 1.0
        if not any_action:
            for i in range(n_neurons):
                relevance[i] = 1.0
        else:
            prev = np.empty(n_neurons)
            for _ in range(3):
                for i in range(n_neurons):
                    prev[i] = relevance[i]
                for c in range(n_conn):
                    if alive[c] and not inhibitory[c] and eligibility[c] > 0.0:
                        v = min(1.0, eligibility[c]) * prev[tgt[c]]
                        if v > relevance[src[c]]:
                            relevance[src[c]] = v
    for c in range(n_conn):
        if alive[c] and inhibitory[c] and block_elig[c] > 0.0:
            # «помешал»: награда — правильно не дал (сильнее), наказание — мешал (слабее)
            db = rate * value * plasticity[c] * block_elig[c] * learn_gain[tgt[c]]
            weight[c] = max(0.0, min(max_weight, weight[c] + db))
        if not alive[c] or eligibility[c] == 0.0:   # у постоянных тормозящих и гипотез метка 0
            continue
        k = relevance[tgt[c]] if credit_on else 1.0
        if inhibitory[c]:
            # обучаемое торможение: наказание усиливает, награда ослабляет;
            # в контексте меняется его добавка (как у возбуждающих связей)
            dw = rate * value * plasticity[c] * eligibility[c] * k * learn_gain[tgt[c]]
            if n_active > 0:
                for j in range(MAX_CONTEXTS):
                    if active[j]:
                        a = ctx_w[c, j] - (dw * ctx_level[j] if ctx_graded > 0.0 else dw / n_active)
                        ctx_w[c, j] = max(0.0 - weight[c], min(max_weight - weight[c], a))
            if base_share > 0.0:
                weight[c] = max(0.0, min(max_weight, weight[c] - dw * base_share))
            continue
        dw = rate * value * plasticity[c] * eligibility[c] * k * learn_gain[tgt[c]]
        if forget_on and plasticity[c] <= 1.0:
            dw *= consolidated_rate   # закреплённое меняется трудно («полная чаша»)
        dw_base = dw * base_share      # доля общей силы (без контекста — вся)
        if ban_on:
            # изменчивый запрет: награда укрепляет запрещённый путь тем слабее,
            # чем увереннее запрет (только общая сила)
            if dw_base > 0.0 and explores[tgt[c]]:
                dw_base *= (1.0 - ban[c]) ** ban_power
        elif (neg_rate < 1.0 and plasticity[c] <= 1.0 and explores[tgt[c]]
                and weight[c] < consolidate_weight):
            # отрицательная память: ослабленный путь меняется медленно
            # (только общая сила; контекстные ветви учатся как обычно)
            dw_base *= neg_rate
        if n_active > 0:
            for j in range(MAX_CONTEXTS):
                if active[j]:
                    # обычный режим: поровну между активными; плавный: по уровню
                    # (производная от w + level * g по g равна level)
                    a = ctx_w[c, j] + (dw * ctx_level[j] if ctx_graded > 0.0 else dw / n_active)
                    ctx_w[c, j] = max(min_weight - weight[c], min(max_weight - weight[c], a))
        if base_share > 0.0:
            weight[c] = max(min_weight, min(max_weight, weight[c] + dw_base))
        we = effective(c, weight, ctx_w, ctx_level, n_active, min_weight, max_weight)
        if ban_on and explores[tgt[c]]:
            part_b = eligibility[c] * k
            if outcome < 0 and part_b >= 0.3:
                ban[c] += ban_up * (1.0 - ban[c])     # участвовал в наказанном: скорее нельзя
            elif outcome > 0 and k > 0.0 and last_share[c] >= 0.5:
                ban[c] -= ban_down * ban[c]           # сам привёл к награде: запрет снимается
        if forget_on and outcome > 0 and doubt[c] < 0:
            doubt[c] = 0                    # освобождённая связь снова подтвердилась
        if plasticity[c] > 1.0 and we >= consolidate_weight and doubt[c] >= 0:
            plasticity[c] = 1.0
            confirm[c] = 0
        if forget_on:
            part = eligibility[c] * k        # насколько связь участвовала в действии
            if outcome > 0:
                importance[c] += part * outcome
                doubt[c] = 0
                if (plasticity[c] <= 1.0 and we < consolidate_weight and k > 0.0
                        and confirm[c] >= 0 and last_share[c] >= 0.5):
                    # слабый закреплённый путь сам привёл к награде
                    confirm[c] += 1
                    if confirm[c] >= doubt_needed:
                        plasticity[c] = relearn_plast
                        confirm[c] = -1   # освобождён: не закрепится слабым снова
            elif outcome < 0 and confirm[c] > 0:
                confirm[c] = 0
            if outcome < 0 and part >= 0.3 and doubt[c] >= 0:
                doubt[c] += 1
                if doubt[c] >= doubt_needed and plasticity[c] <= 1.0:
                    # закреплённое снова и снова ведёт к наказанию: ошибочное,
                    # освобождаем для переучивания
                    plasticity[c] = relearn_plast
                    importance[c] *= 0.5
                    doubt[c] = -1   # освобождена: не закрепится, пока не подтвердится наградой


@njit(cache=True)
def delay(thickness):
    # Чем больше миелинизация, тем меньше задержка (1..5)
    # np.round округляет половину к чётному, как round() в Python
    return max(1, int(np.round(5.0 - 4.0 * thickness)))
