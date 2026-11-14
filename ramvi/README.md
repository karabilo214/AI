# RAMVI

RAMVI — приложение для локальной работы с языковыми моделями. Вычислительная часть — RAMVI Engine.

Этап 1 завершён: ядро находится в устанавливаемом пакете `engine/ramvi_engine`, проверки и инструменты отделены от продукта. Итог проверен на Mac, Windows и NVIDIA; подробности — в отчёте приёмки. Portable-приложение и графическая оболочка ещё не реализованы.

## Документация

- [План выполнения и статус пунктов](docs/PLAN.md).
- [Техническое задание — PDF](docs/spec/RAMVI_technical_spec_v1.pdf) и [DOCX](docs/spec/RAMVI_technical_spec_v1.docx).
- [Структура проекта и правила зависимостей](docs/adr/ADR-0001-structure.md).
- [Перенос документов](docs/DOCUMENT_MIGRATION.md).
- [Итог этапа 1: WIDE30 × 4, API и сравнение с Ollama](docs/reports/STAGE1_ACCEPTANCE.md).
- [Установка движка и команды запуска](engine/README.md).
- [Устройство переноса и проверки совместимости](docs/architecture/ENGINE_MIGRATION.md).
- [Инструменты проверки](docs/architecture/TOOLS_MIGRATION.md) и [каталоги данных, настройки](docs/architecture/CONFIGURATION.md).

## Подготовка окружения разработчика

Нужен Python 3.12. Команды выполняются из каталога `ramvi`. Окружение инструментов разработки не требует моделей, CUDA, PyTorch или Triton.

Linux / macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade "pip>=25.1"
.venv/bin/python -m pip install --group dev
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade "pip>=25.1"
.venv\Scripts\python.exe -m pip install --group dev
```

Версии инструментов зафиксированы в `pyproject.toml`. Это отдельное окружение; зависимости действующего движка описаны в его README.

## Проверки

Из `ramvi` с Python созданного окружения:

```sh
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
```

Здесь `python` означает `.venv/bin/python` на Linux/macOS или `.venv\Scripts\python.exe` на Windows. Проверяются ссылки документации, структура пакета, CLI и совместимость импортов. Проверки, требующие вычислительных зависимостей, пропускаются в лёгком окружении. Тесты движка находятся в `engine/tests/`, проверки и замеры — в `engine/tools/`. Общий запуск из корня репозитория: `python -m ramvi.engine.tools`.

## Структура

- `docs/spec/` — техническое задание.
- `docs/reports/` — сводные отчёты; сырые результаты и логи не входят в git.
- `docs/architecture/` — описание подсистем.
- `docs/adr/` — принятые архитектурные решения.
- `tests/` — общие проверки проекта.
- `engine/` — устанавливаемый RAMVI Engine и проверки пакета.

Остальные пакеты создаются последовательно по плану. Выбор графической оболочки остаётся открытым до пункта 3.1.

## Лицензия

[MIT](LICENSE). Зависимости и веса моделей сохраняют собственные лицензии.
