# RAMVI Engine

Перенесённое ядро `llm_app`. Алгоритмы вычисления, форматы весов, численные режимы и значения переменных окружения сохраняются. Этот этап не добавляет новые модели или GPU-backend.

## Установка при переносе

Из корня репозитория, в существующее окружение движка:

```sh
python -m pip install --no-deps -e ramvi/engine
python -m ramvi_engine --help
ramvi-engine --help
```

`--no-deps` сохраняет уже проверенное окружение: команда не меняет PyTorch, Triton и зависимости действующего движка. Для нового окружения общие зависимости перечислены в `pyproject.toml` и `requirements.txt`; GPU-зависимости выбираются под ОС и драйвер. Группа `gpu` содержит Triton для Linux и triton-windows для Windows. На Mac выполняются проверки без GPU.

## Команды

```sh
python -m ramvi_engine serve --model qwen3:14b --draft qwen3:0.6b
python -m ramvi_engine chat --model qwen3:14b --draft qwen3:0.6b
python -m ramvi_engine plan --model qwen3:14b
```

Импорты абсолютные, например `ramvi_engine.core.gengine.GGUFEngine`. Проверки и замеры размещены в `tools/`; общий запуск из корня репозитория — `python -m ramvi.engine.tools`. [Список команд](../docs/architecture/TOOLS_MIGRATION.md).

## Native

Исходник, скрипт сборки и генератор манифеста находятся в `ramvi_engine/native/`. Из корня репозитория:

```sh
sh ramvi/engine/ramvi_engine/native/build.sh
```

Библиотека создаётся рядом с исходником; загрузчик проверяет ISA, ABI и хеши манифеста. Готовый бинарник не хранится в git.

## Проверка переноса

Из корня репозитория, в окружении движка:

```sh
python -m ramvi.engine.tools unit -- -q
python -m pytest -c ramvi/pyproject.toml ramvi/tests ramvi/engine/tests
```

Итоговая Windows/CUDA-проверка этапа 1 выполнена: четыре полных ряда WIDE30, тексты совпали на 30/30 вопросах, общая скорость изменилась на +0,56%. [Полный отчёт](../docs/reports/STAGE1_ACCEPTANCE.md) содержит fp32, API, условия, повторы и сравнение с сохранённой Ollama. Эта небольшая разница не доказывает ускорение алгоритма. После последующих изменений движка экзамен повторяется.

[План](../docs/PLAN.md) · [Карта переноса](../docs/architecture/ENGINE_MIGRATION.md) · [Лицензия](LICENSE).


## Данные и настройки

Переменные движка используют `RAMVI_*`; прежние `LLM_*` читаются как запасной вариант. `RAMVI_DATA_DIR` задаёт каталог данных, внутри него находятся cache/work/results. [Правила выбора каталогов](../docs/architecture/CONFIGURATION.md). Внешние кэши и пути моделей HF/Ollama сохраняются.
