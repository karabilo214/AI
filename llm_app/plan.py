"""План памяти: какие слои держать в видеокарте (сжатые 12 бит), какие подкачивать по шине, какие считать на
процессоре — по свободной памяти видеокарты и RAM. Слои вне видеокарты по умолчанию чередуются: процессор
считает один, пока следующий едет по шине."""
import ctypes
import os
import sys

import torch

GiB = 2**30


def ram_info():
    """(всего, доступно) байт RAM."""
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys, m.ullAvailPhys
    if os.path.exists("/proc/meminfo"):
        d = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo")}
        return d["MemTotal"], d.get("MemAvailable", d["MemFree"])
    tot = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return tot, tot // 2


def make_plan(ck, maxt, draft_bytes=0, cpu=True, vram_reserve=0.8 * GiB, force_resident=None, log=print):
    L = ck.L
    P = ck.layer_params()
    n = sum(a * b for a, b in P.values())
    c12 = int(n * 1.505)  # 12 бит + исключения
    b16 = n * 2
    kv = 2 * ck.KV * maxt * ck.HD * 4
    V = ck.cfg["vocab_size"]
    head = V * ck.D * 1.0 + V * 16  # int8 + масштабы/границы
    free, total = torch.cuda.mem_get_info()
    base = vram_reserve + head + draft_bytes
    if force_resident is None and base + L * (c12 + kv) <= free:
        R = L
    else:
        slots = 2 * c12 + (b16 if cpu else 0)  # буферы подкачки (+ буфер bf16 для слоёв процессора при чтении запроса)
        R = int((free - base - slots - L * kv) // c12) if force_resident is None else force_resident
        R = max(0, min(L - 1, R))
    place = ["gpu"] * R
    for i in range(R, L):
        place.append("cpu" if (cpu and (i - R) % 2 == 1) else "stream")
    tot_ram, avail = ram_info()
    need = place.count("stream") * c12 + place.count("cpu") * b16
    plan = {"place": place, "R": R, "vram_free": free, "ram_avail": avail, "ram_need": need,
            "pin_budget": int(0.4 * tot_ram) if sys.platform == "win32" else int(0.6 * tot_ram)}
    log(f"план: видеокарта свободно {free / GiB:.1f} ГиБ, RAM доступно {avail / GiB:.1f} ГиБ; слоёв в видеокарте {R} из {L}, "
        f"по шине {place.count('stream')}, на процессоре {place.count('cpu')}; RAM под слои {need / GiB:.1f} ГиБ")
    if need > avail:
        log("  внимание: RAM может не хватить — слои будут читаться с диска (медленно)")
    return plan
