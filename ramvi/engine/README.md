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

## Карта возможностей

`python -m ramvi_engine capabilities` возвращает JSON без загрузки модели и CUDA. Запущенный сервер предоставляет авторизованный `GET /capabilities`. Условия поддержки и ограничения: [карта возможностей](../docs/architecture/capabilities.md).

## Оценка памяти

Планировщики возвращают `resources` и `resource_fit` на основе общего `ResourceEstimate`. Тот же контракт используется для предварительного подбора моделей: [компоненты, API и ограничения](../docs/architecture/RESOURCE_ESTIMATE.md). Оценка не является гарантией отсутствия OOM.

## Жизненный цикл

CLI `serve`/`chat` используют `EngineSession`: load, warmup, cancel, unload и restart в одном процессе. HTTP-контракт, отмена и условия освобождения памяти: [жизненный цикл](../docs/architecture/ENGINE_LIFECYCLE.md).

## Метрики генерации

Все ответы генерации содержат `ramvi_metrics`; в потоке снимок приходит в завершающем событии. `GET /engine/metrics` возвращает авторизованный статус. Определения TTFT, tok/s, draft, выборочного RSS и CUDA allocator: [контракт метрик](../docs/architecture/ENGINE_METRICS.md).

## Контракты этапа 2

[Версии и события](../docs/architecture/ENGINE_PROTOCOL.md) · [Автоматическое размещение](../docs/architecture/AUTOMATIC_PLANNING.md) · [Независимый fp32 oracle](../docs/architecture/REFERENCE_VALIDATION.md) · [Изолированный процесс](../docs/architecture/ENGINE_PROCESS.md) · [Module Host](../docs/architecture/MODULE_CORE.md).
