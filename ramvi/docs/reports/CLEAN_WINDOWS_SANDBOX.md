# Переносимый комплект на чистой Windows (Windows Sandbox), 8.1

Дата: 2026-11-18/19. Хост: ASUS (Windows 11 Pro 25H2, RTX 5060 Laptop, драйвер NVIDIA 616.92). Гость: Windows Sandbox — Windows 11 Enterprise 10.0.26100, 8 ГБ, сеть выключена, vGPU включён. Комплект — копия рабочего комплекта (4,3 ГБ: torch 2.15.0+cu132, numba 0.68.1, triton 3.9.0). Скрипты — [stage8_checks/](stage8_checks/).

## Система действительно чистая

В System32 нет `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`; Python нет; `nvidia-smi` нет. Видеокарта RTX 5060 видна через виртуальный драйвер Microsoft (10.0.26100.1150).

## Прогон 1 — комплект как есть

| Проверка | Итог |
|---|---|
| копирование комплекта на диск Sandbox | 358 с |
| встроенный Python 3.12.10, numpy 2.5.4 | работают |
| `import torch` | **ошибка**: `WinError 126` при загрузке `torch\lib\c10.dll` — нет библиотек Microsoft Visual C++ |
| сервис | стартовал, отвечает на `/ui/v1/status` и `/ui/v1/system` |
| окно RAMVI через 45 с | открыто, интерфейс отрисован; модель не запустилась (`engine process exited during startup` — та же причина) |
| проверка драйвера (новый код) | **ошибка проверки**: показала «драйвер NVIDIA 10.0.26100.1150» — это версия виртуального драйвера Microsoft из WMI |

## Исправления

1. **Комплект:** сборка кладёт библиотеки Visual C++ 2015–2022 x64 рядом с `python.exe` (`build_portable.VC_RUNTIME`: msvcp140, msvcp140_1/_2/_atomic_wait/_codecvt_ids, vcruntime140, vcruntime140_1, vcruntime140_threads, concrt140, vcomp140; Microsoft разрешает распространять их вместе с приложением). Нет хотя бы одной на машине сборки — сборка останавливается.
2. **Драйвер из WMI:** версия NVIDIA в записи Windows (`32.0.16.1692` → 616.92, совпало с `nvidia-smi` на хосте) принимается, только если она в формате NVIDIA; иначе (виртуальный драйвер Microsoft) — драйвера NVIDIA нет.
3. **Драйвер без nvidia-smi при работающей CUDA** (Sandbox пробрасывает драйвер хоста) — «драйвер NVIDIA работает», не ошибка.
4. **Нехватка Visual C++** у torch — отдельное понятное сообщение диагностики «комплект неполный» (`system.backend_vcredist`).

## Прогон 2 — библиотеки Visual C++ рядом с python.exe (копия, без пересборки)

| Проверка | Итог |
|---|---|
| `import torch` | 2.15.0+cu132, CUDA 13.2, **`torch.cuda.is_available()` = True** (vGPU Sandbox) |
| `import numba, triton` | 0.68.1, 3.9.0 |
| диагностика | ОС, CPU, RAM, GPU — ok; бэкенд `system.backend_cuda` — ok; драйвер — в этом прогоне ещё `driver_missing` (исправлено после прогона, п. 3) |
| сервис | стартовал, `/ui/v1/system` ответил |

## Что осталось

Пересобрать комплект с библиотеками Visual C++ и в Sandbox пройти мастер и получить ответ модели (генерация на vGPU не проверялась).
