"""Поворот поправки, чтобы она не увеличивала ошибку на проверенных днях
(условие метода GEM: на днях из памяти удач ошибка не должна расти).

A — строки (ошибка дня) * (производная прогноза дня по весам); поправка d
допустима, если A @ d <= 0 для всех строк. Нарушенные условия собираются в
активный набор, и d проецируется так, чтобы по ним изменение было 0.
"""
import numpy as np


def project(d, A, iters=20):
    if len(A) == 0:
        return d
    d0 = d.copy()
    active = np.zeros(len(A), dtype=bool)
    for _ in range(iters):
        viol = (A @ d > 1e-12) & ~active
        if not viol.any():
            break
        active |= viol
        G = A[active]
        d = d0 - G.T @ np.linalg.lstsq(G @ G.T + 1e-12 * np.eye(len(G)), G @ d0, rcond=None)[0]
    return d
