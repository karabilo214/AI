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

В переходный период команды `python -m llm_app ...` и старые импорты направляются к тем же модулям. Новые импорты — абсолютные, например `ramvi_engine.core.gengine.GGUFEngine`. Проверочные скрипты пока остаются в `llm_app` до пункта 1.6.

## Native

Исходник, скрипт сборки и генератор манифеста находятся в `ramvi_engine/native/`. Из корня репозитория:

```sh
sh ramvi/engine/ramvi_engine/native/build.sh
```

Библиотека создаётся рядом с исходником; загрузчик проверяет ISA, ABI и хеши манифеста. Готовый бинарник не хранится в git. Прежний `llm_app/cpp/build.sh` временно делегирует сборку в этот каталог.

## Проверка переноса

Из корня репозитория, в окружении движка:

```sh
python -m unittest discover -s llm_app/tests -t .
python -m pytest -c ramvi/pyproject.toml ramvi/tests ramvi/engine/tests
```

Локальные проверки не заменяют Windows/CUDA-экзамен и полный WIDE30 из пункта 1.9. Перенос не считается доказательством сохранения скорости на GPU.

[План](../docs/PLAN.md) · [Карта переноса](../docs/architecture/ENGINE_MIGRATION.md) · [Лицензия](LICENSE).
