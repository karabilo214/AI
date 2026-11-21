"""AUTO-01: локальный профиль железа. Только чтение системы, без сети.

Неизвестное значение — None, а не ошибка: профиль собирается всегда.
"""

import ctypes
import ntpath
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MIB = 1024 * 1024


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vendor: str
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    driver: str | None = None


@dataclass(frozen=True)
class HardwareProfile:
    os: str
    arch: str
    cpu_name: str | None
    isa: tuple[str, ...]
    cores_physical: int | None
    cores_logical: int | None
    ram_total_mb: int | None
    ram_available_mb: int | None
    disk_path: str
    disk_total_mb: int | None
    disk_free_mb: int | None
    gpus: tuple[GPUInfo, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(args: list[str], timeout: float = 5.0) -> str | None:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _int(text: Any) -> int | None:
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None


# Флаги, важные для выбора сборки движка.
_ISA_FLAGS = ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "neon")


def _isa_from_flags(flags: set[str]) -> tuple[str, ...]:
    return tuple(name for name in _ISA_FLAGS if name in flags)


def parse_nvidia_smi(text: str) -> tuple[GPUInfo, ...]:
    """Строки CSV: name, memory.total, memory.free, driver_version (MiB)."""
    gpus = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[0]:
            continue
        gpus.append(
            GPUInfo(
                name=parts[0],
                vendor="NVIDIA",
                vram_total_mb=_int(parts[1]),
                vram_free_mb=_int(parts[2]),
                driver=parts[3] or None,
            )
        )
    return tuple(gpus)


def _nvidia_gpus() -> tuple[GPUInfo, ...]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ()
    out = _run(
        [
            exe,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_nvidia_smi(out) if out else ()


def nvidia_free_mb() -> int | None:
    """5.4: свободная видеопамять самой свободной NVIDIA сейчас (МиБ); None — NVIDIA нет."""
    free = [g.vram_free_mb for g in _nvidia_gpus() if g.vram_free_mb is not None]
    return max(free) if free else None


def ram_available_mb() -> int | None:
    """5.4: доступная оперативная память сейчас (МиБ), без медленных опросов (WMI и т.п.)."""
    info: dict[str, Any] = {"ram_available_mb": None}
    try:
        if sys.platform.startswith("linux"):
            info["ram_total_mb"], info["ram_available_mb"] = parse_meminfo(
                Path("/proc/meminfo").read_text()
            )
        elif sys.platform == "win32":
            _windows_memory(info)
        elif sys.platform == "darwin":
            _macos_memory(info)
    except Exception:  # замер не должен ронять запись
        return None
    value = info["ram_available_mb"]
    return int(value) if value is not None else None


# --- Linux -----------------------------------------------------------------


def parse_meminfo(text: str) -> tuple[int | None, int | None]:
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        number = _int(rest.split()[0]) if rest.split() else None
        values[key.strip()] = number
    total = values.get("MemTotal")
    avail = values.get("MemAvailable")
    return (
        total // 1024 if total is not None else None,
        avail // 1024 if avail is not None else None,
    )


def parse_cpuinfo(text: str) -> tuple[str | None, set[str], int | None]:
    name = None
    flags: set[str] = set()
    cores = set()
    physical = core = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and name is None:
            name = value
        elif key in ("flags", "Features") and not flags:
            flags = set(value.lower().split())
            if "asimd" in flags:
                flags.add("neon")
        elif key == "physical id":
            physical = value
        elif key == "core id":
            core = value
        elif not line.strip() and core is not None:
            cores.add((physical, core))
            physical = core = None
    if core is not None:
        cores.add((physical, core))
    return name, flags, (len(cores) or None)


def _linux(info: dict[str, Any]) -> None:
    try:
        info["ram_total_mb"], info["ram_available_mb"] = parse_meminfo(
            Path("/proc/meminfo").read_text()
        )
    except OSError:
        pass
    try:
        name, flags, physical = parse_cpuinfo(Path("/proc/cpuinfo").read_text())
        info.update(cpu_name=name, isa=_isa_from_flags(flags), cores_physical=physical)
    except OSError:
        pass
    info["other_gpus"] = _linux_gpus()


# --- macOS -----------------------------------------------------------------


def _sysctl(name: str) -> str | None:
    out = _run(["sysctl", "-n", name])
    return out.strip() if out else None


def _macos_memory(info: dict[str, Any]) -> None:
    vm = _run(["vm_stat"])
    if vm:
        page = _int(_sysctl("hw.pagesize")) or 4096
        pages = {}
        for line in vm.splitlines():
            key, _, value = line.partition(":")
            pages[key.strip()] = _int(value.strip().rstrip("."))
        free = sum(pages.get(k) or 0 for k in ("Pages free", "Pages inactive", "Pages speculative"))
        info["ram_available_mb"] = free * page // MIB


def _macos(info: dict[str, Any]) -> None:
    info["cpu_name"] = _sysctl("machdep.cpu.brand_string")
    info["cores_physical"] = _int(_sysctl("hw.physicalcpu"))
    total = _int(_sysctl("hw.memsize"))
    info["ram_total_mb"] = total // MIB if total else None
    flags = set((_sysctl("machdep.cpu.features") or "").lower().split())
    flags |= set((_sysctl("machdep.cpu.leaf7_features") or "").lower().split())
    flags = {flag.replace(".", "_") for flag in flags}
    if platform.machine() == "arm64":
        flags.add("neon")
    info["isa"] = _isa_from_flags(flags)
    _macos_memory(info)
    info["other_gpus"] = parse_system_profiler(
        _run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=15) or ""
    )


# --- Windows ---------------------------------------------------------------


def _windows_memory(info: dict[str, Any]) -> None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        info["ram_total_mb"] = status.ullTotalPhys // MIB
        info["ram_available_mb"] = status.ullAvailPhys // MIB


def _windows(info: dict[str, Any]) -> None:
    _windows_memory(info)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    # IsProcessorFeaturePresent: 10=SSE2(база x64), 39=AVX, 40=AVX2, 41=AVX512F.
    flags = {"sse4_2"} if platform.machine().lower() in ("amd64", "x86_64") else set()
    for code, name in ((39, "avx"), (40, "avx2"), (41, "avx512f")):
        if kernel32.IsProcessorFeaturePresent(code):
            flags.add(name)
    info["isa"] = _isa_from_flags(flags)
    try:
        import winreg

        key = winreg.OpenKey(  # type: ignore[attr-defined,unused-ignore]
            winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined,unused-ignore]
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0]  # type: ignore[attr-defined,unused-ignore]
        info["cpu_name"] = str(name).strip()
    except OSError:
        pass
    out = _run(["powershell", "-NoProfile", "-Command", _WMI_PS], timeout=15)
    if out:
        info["cores_physical"], info["other_gpus"] = parse_wmi_json(out)


_WMI_PS = (
    "@{cpu=@(Get-CimInstance Win32_Processor | Select-Object NumberOfCores);"
    "gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)}"
    " | ConvertTo-Json -Compress -Depth 3"
)


def _vendor(name: str) -> str:
    low = name.lower()
    for key, vendor in (
        ("nvidia", "NVIDIA"),
        ("geforce", "NVIDIA"),
        ("radeon", "AMD"),
        ("amd", "AMD"),
        ("intel", "Intel"),
        ("apple", "Apple"),
    ):
        if key in low:
            return vendor
    return "unknown"


def nvidia_driver_from_wmi(version: str | None) -> str | None:
    """Версия драйвера NVIDIA из формата Windows: «32.0.15.6692» → «566.92».

    У драйвера NVIDIA в WMI старшая часть ≥ 20, вторая — 0, третья — 1x; последние пять
    цифр двух младших частей — версия NVIDIA. Иное (например, виртуальный драйвер
    Microsoft «10.0.26100.1150» в Windows Sandbox) — не драйвер NVIDIA: None."""
    if not version:
        return None
    parts = version.strip().split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return None
    major, minor, build, revision = (int(p) for p in parts)
    if major < 20 or minor != 0 or not 10 <= build <= 19 or revision > 9999:
        return None
    digits = f"{build % 10}{revision:04d}"
    return f"{int(digits[:3])}.{digits[3:]}"


def parse_wmi_json(text: str) -> tuple[int | None, tuple[GPUInfo, ...]]:
    """VRAM из WMI не берём: AdapterRAM ограничен 4 ГБ, у встроенных память общая."""
    import json

    try:
        data = json.loads(text)
    except ValueError:
        return None, ()
    cores = sum(_int(c.get("NumberOfCores")) or 0 for c in data.get("cpu") or [])
    gpus = tuple(
        GPUInfo(
            name=g["Name"].strip(),
            vendor=_vendor(g["Name"]),
            # 8.1: у NVIDIA — версия в привычном виде или None (не настоящий драйвер NVIDIA).
            driver=nvidia_driver_from_wmi(g.get("DriverVersion"))
            if _vendor(g["Name"]) == "NVIDIA"
            else g.get("DriverVersion"),
        )
        for g in data.get("gpu") or []
        if g.get("Name")
    )
    return cores or None, gpus


def parse_system_profiler(text: str) -> tuple[GPUInfo, ...]:
    import json

    try:
        items = json.loads(text).get("SPDisplaysDataType") or []
    except (ValueError, AttributeError):
        return ()
    gpus = []
    for item in items:
        name = item.get("sppci_model") or item.get("_name")
        if name:
            vram = item.get("spdisplays_vram") or item.get("spdisplays_vram_shared")
            mb = _int(str(vram).split()[0]) if vram else None
            if mb is not None and "GB" in str(vram):
                mb *= 1024
            gpus.append(GPUInfo(name=name, vendor=_vendor(name), vram_total_mb=mb))
    return tuple(gpus)


def _linux_gpus() -> tuple[GPUInfo, ...]:
    out = _run(["lspci", "-mm"]) or ""
    gpus = []
    for line in out.splitlines():
        if any(k in line for k in ('"VGA', '"3D', '"Display')):
            parts = line.split('"')
            name = f"{parts[3]} {parts[5]}" if len(parts) > 5 else line
            gpus.append(GPUInfo(name=name, vendor=_vendor(name)))
    return tuple(gpus)


def collect(disk_path: str | os.PathLike[str] | None = None) -> HardwareProfile:
    """Собрать профиль. disk_path — папка данных/моделей (по умолчанию текущая)."""
    path = str(Path(disk_path or os.getcwd()).resolve())
    info: dict[str, Any] = dict(
        cpu_name=None,
        isa=(),
        cores_physical=None,
        ram_total_mb=None,
        ram_available_mb=None,
        other_gpus=(),
    )
    try:
        if sys.platform.startswith("linux"):
            _linux(info)
        elif sys.platform == "darwin":
            _macos(info)
        elif sys.platform == "win32":
            _windows(info)
    except Exception:  # профиль не должен ронять запуск
        pass
    disk_total: int | None
    disk_free: int | None
    try:
        usage = shutil.disk_usage(path)
        disk_total, disk_free = usage.total // MIB, usage.free // MIB
    except OSError:
        disk_total = disk_free = None
    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu_name=info["cpu_name"] or platform.processor() or None,
        isa=tuple(info["isa"]),
        cores_physical=info["cores_physical"],
        cores_logical=os.cpu_count(),
        ram_total_mb=info["ram_total_mb"],
        ram_available_mb=info["ram_available_mb"],
        disk_path=path,
        disk_total_mb=disk_total,
        disk_free_mb=disk_free,
        gpus=_merge_gpus(_nvidia_gpus(), info["other_gpus"]),
    )


def _merge_gpus(nvidia: tuple[GPUInfo, ...], other: tuple[GPUInfo, ...]) -> tuple[GPUInfo, ...]:
    """Данные nvidia-smi точнее; NVIDIA из общего списка берём, только если nvidia-smi нет."""
    if not nvidia:
        return tuple(other)
    return tuple(nvidia) + tuple(g for g in other if g.vendor != "NVIDIA")


if __name__ == "__main__":
    import json

    print(json.dumps(collect().to_dict(), ensure_ascii=False, indent=2))


# --- 8.2 (PORT-03): вид диска папки моделей -----------------------------------------------

_NETWORK_FS = ("nfs", "nfs4", "cifs", "smbfs", "smb3", "fuse.sshfs", "afpfs", "9p", "davfs")
# GetDriveTypeW: 0 — неизвестно, 1 — нет такого корня, 2 — съёмный, 3 — встроенный,
# 4 — сетевой, 5 — CD/DVD, 6 — RAM-диск.
_WINDOWS_DRIVES = {2: "removable", 3: "local", 4: "network", 5: "removable", 6: "local"}


def _linux_kind(path: str, mounts: str) -> str:
    best, kind = "", "local"
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        point, fs = parts[1].replace("\\040", " "), parts[2]
        inside = path == point or path.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best = point
            kind = "network" if fs in _NETWORK_FS else "local"
    return kind


def drive_kind(
    path: str | os.PathLike[str], *, system: str | None = None, mounts: str | None = None
) -> str:
    """Где лежит папка: `local`, `removable` (флешка, внешний диск), `network` или `unknown`.

    Сетевой и съёмный диск допустимы, но модели с них грузятся заметно дольше — интерфейс
    предупреждает (PORT-03). Windows — GetDriveTypeW и UNC-пути; Linux — /proc/mounts;
    macOS — тома в /Volumes (внешние или сетевые)."""
    system = system or platform.system()
    raw = os.fspath(path)
    if system == "Windows":
        if raw.startswith(("\\\\", "//")):
            return "network"
        drive = ntpath.splitdrive(ntpath.abspath(raw) if os.name == "nt" else raw)[0]
        if not drive:
            return "unknown"
        try:
            kind = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")  # type: ignore[attr-defined,unused-ignore]
        except (AttributeError, OSError):
            return "unknown"
        return _WINDOWS_DRIVES.get(kind, "unknown")
    text = os.path.abspath(raw)
    if system == "Darwin":
        return "removable" if text.startswith("/Volumes/") else "local"
    if system == "Linux":
        if mounts is None:
            try:
                mounts = Path("/proc/mounts").read_text(encoding="utf-8")
            except OSError:
                return "unknown"
        return _linux_kind(text, mounts)
    return "unknown"
