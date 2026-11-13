# RAMVI

RAMVI — приложение для локальной работы с языковыми моделями. Вычислительная часть — RAMVI Engine.

Сейчас подготовлены документация и каркас проекта. Исполняемый движок пока находится в `llm_app`; перенос пакетов — следующий пункт плана. Portable-приложение и графическая оболочка ещё не реализованы.

## Документация

- [План выполнения и статус пунктов](docs/PLAN.md).
- [Техническое задание — PDF](docs/spec/RAMVI_technical_spec_v1.pdf) и [DOCX](docs/spec/RAMVI_technical_spec_v1.docx).
- [Структура проекта и правила зависимостей](docs/adr/ADR-0001-structure.md).
- [Перенос документов](docs/DOCUMENT_MIGRATION.md).
- [Последнее сравнение на 30 вопросах с Ollama](docs/reports/WIDE30_OLLAMA_COMPARISON.md).
- [Действующий движок и команды запуска](../llm_app/README.md).

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

Здесь `python` означает `.venv/bin/python` на Linux/macOS или `.venv\Scripts\python.exe` на Windows. Текущий тест проверяет локальные ссылки документации. Тесты и GPU-экзамены движка остаются в `llm_app` до соответствующих пунктов переноса.

## Структура

- `docs/spec/` — техническое задание.
- `docs/reports/` — сводные отчёты; сырые результаты и логи не входят в git.
- `docs/architecture/` — описание подсистем.
- `docs/adr/` — принятые архитектурные решения.
- `tests/` — общие проверки проекта.

Пакеты движка, сервиса, агента, голоса и хранилища создаются последовательно по плану. Выбор графической оболочки остаётся открытым до пункта 3.1.

## Лицензия

[MIT](LICENSE). Зависимости и веса моделей сохраняют собственные лицензии.
