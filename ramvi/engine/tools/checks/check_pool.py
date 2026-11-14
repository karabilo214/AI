"""Нагрузочная проверка пула потоков C++ (qmm.dll): тысячи случайных вызовов q4k_mm/q6k_mm/f16_mm и moe_ffn
(разные пачки, добавление/запись, два потока Python одновременно) против numpy по распакованным весам.
python -m ramvi.engine.tools check pool -- [итераций]"""
import sys
import threading
import time

import numpy as np

from ramvi_engine.cpu.gcpu import CpuMat, _dll
from ramvi_engine.formats.gguf import GGUF, find_ollama


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    d = _dll()
    assert d is not None, "нет qmm.dll"
    g = GGUF(find_ollama("qwen3:0.6b"))
    mats = []
    for n in ("blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight", "blk.1.ffn_gate.weight"):
        m = CpuMat(g, n)
        mats.append((n, m, g.full(n).astype(np.float64)))
    print("матрицы:", [(n.split(".")[-2], m.t, m.N, m.K) for n, m, _ in mats])
    rng = np.random.default_rng(0)
    worst, bad = 0.0, 0
    result_lock = threading.Lock()
    total_bad = 0

    def one(r):
        nonlocal worst, bad
        n, m, W = mats[r.integers(len(mats))]
        T = int(r.integers(1, 17))
        add = bool(r.integers(2))
        X = r.standard_normal((T, m.K)).astype(np.float32)
        Y0 = r.standard_normal((T, m.N)).astype(np.float32)
        Y = Y0.copy()
        m.mm(X, Y, None, add)
        ref = X.astype(np.float64) @ W.T + (Y0 if add else 0)
        e = float(np.abs(Y - ref).max() / (np.abs(ref).max() + 1e-9))
        with result_lock:
            worst = max(worst, e)
            bad += e > 1e-4

    t0 = time.perf_counter()
    for _ in range(iters):
        one(rng)
    print(f"один поток: {iters} вызовов за {time.perf_counter() - t0:.1f} с, наибольшая ошибка {worst:.1e}, плохих {bad}")
    total_bad += bad
    worst, bad = 0.0, 0
    errors = []
    def threaded(seed):
        try:
            for i in range(iters // 2):
                one(np.random.default_rng(seed * 7 + i))
        except Exception as exc:
            errors.append(exc)
    ths = [threading.Thread(target=threaded, args=(seed,)) for seed in (1, 2)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"два потока Python одновременно: {iters} вызовов за {time.perf_counter() - t0:.1f} с, наибольшая ошибка {worst:.1e}, плохих {bad}")

    total_bad += bad + len(errors)
    if errors:
        raise RuntimeError("проверка потока завершилась ошибкой") from errors[0]
    # moe_ffn: эксперты слоя 0 модели MoE
    try:
        gm = GGUF(find_ollama("qwen3:30b"))
    except FileNotFoundError:
        print("qwen3:30b нет — moe_ffn не проверяется")
        raise SystemExit(1 if total_bad else 0)
    from ramvi_engine.core.moe import TYP
    a = gm.meta["general.architecture"]
    E, K, F, D = gm.meta[f"{a}.expert_count"], gm.meta[f"{a}.expert_used_count"], gm.meta[f"{a}.expert_feed_forward_length"], gm.meta[f"{a}.embedding_length"]
    raw = {}
    for s in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        b, N, KK, t = gm.raw(f"blk.0.{s}.weight")
        raw[s] = (np.asarray(b), TYP[t])
    pool = list(range(0, E, E // 12))[:12]  # 12 экспертов — для них считаем эталон
    ref_w = {}
    for e in pool:
        ref_w[e] = (gm.rows("blk.0.ffn_gate_exps.weight", range(e * F, (e + 1) * F)).astype(np.float64),
                    gm.rows("blk.0.ffn_up_exps.weight", range(e * F, (e + 1) * F)).astype(np.float64),
                    gm.rows("blk.0.ffn_down_exps.weight", range(e * D, (e + 1) * D)).astype(np.float64))
    Hs = np.empty((16 * K, F), np.float32)
    Zs = np.empty((16 * K, D), np.float32)
    worst, bad = 0.0, 0
    t0 = time.perf_counter()
    nm = max(50, iters // 20)
    for it in range(nm):
        T = int(rng.integers(1, 17))
        ids = np.array([rng.choice(pool, K, replace=False) for _ in range(T)], np.int32)
        w = rng.random((T, K)).astype(np.float32)
        w[rng.random((T, K)) < 0.2] = 0.0  # часть пар «в видеокарте» — пропускаются
        X = rng.standard_normal((T, D)).astype(np.float32)
        rc = d.moe_ffn(raw["ffn_gate_exps"][0].ctypes.data, raw["ffn_up_exps"][0].ctypes.data, raw["ffn_down_exps"][0].ctypes.data,
                  raw["ffn_gate_exps"][1], raw["ffn_up_exps"][1], raw["ffn_down_exps"][1], D, F, T, K,
                  ids.ctypes.data, w.ctypes.data, X.ctypes.data, Hs.ctypes.data, Zs.ctypes.data)
        bad += rc != 0
        Z = Zs[:T * K].reshape(T, K, D)
        ref = np.zeros((T, K, D))
        for t in range(T):
            for j in range(K):
                if w[t, j] == 0:
                    continue
                Gw, Uw, Dw = ref_w[int(ids[t, j])]
                x = X[t].astype(np.float64)
                gg, uu = Gw @ x, Uw @ x
                ref[t, j] = w[t, j] * (Dw @ (gg / (1 + np.exp(-gg)) * uu))
        e = float(np.abs(Z - ref).max() / (np.abs(ref).max() + 1e-9))
        worst = max(worst, e)
        bad += e > 1e-4
    # недопустимые размеры должны отклоняться, а не писать за границы
    X = np.zeros((17, 1024), np.float32)
    Y = np.zeros((17, mats[0][1].N), np.float32)
    try:
        mats[0][1].mm(X, Y, None)
        print("ПРОВАЛ: пачка из 17 не отклонена")
        bad += 1
    except ValueError:
        print("пачка из 17 отклонена — OK")
    print(f"moe_ffn: {nm} вызовов за {time.perf_counter() - t0:.1f} с, наибольшая ошибка {worst:.1e}, плохих {bad}; "
          f"исключений в рабочих потоках: {d.take_failed()}")
    sys.exit(1 if bad + total_bad else 0)


if __name__ == "__main__":
    main()
