# Перенос ядра RAMVI Engine (1.4)

Вычислительные алгоритмы сохранены. Изменения ограничены размещением модулей, абсолютными импортами, путями native и точкой входа CLI. HTTP-модуль пока перенесён целиком; его разделение относится к пункту 1.5.

| Прежний модуль | Канонический модуль |
|---|---|
| `llm_app.head` | `ramvi_engine.generation.head` |
| `llm_app.engine` | `ramvi_engine.core.engine` |
| `llm_app.gengine` | `ramvi_engine.core.gengine` |
| `llm_app.moe` | `ramvi_engine.core.moe` |
| `llm_app.reference` | `ramvi_engine.core.reference` |
| `llm_app.tkernels` | `ramvi_engine.kernels.tkernels` |
| `llm_app.wcomp` | `ramvi_engine.kernels.wcomp` |
| `llm_app.gkernels` | `ramvi_engine.kernels.gkernels` |
| `llm_app.vocab8` | `ramvi_engine.kernels.vocab8` |
| `llm_app.q4` | `ramvi_engine.kernels.q4` |
| `llm_app.prefill` | `ramvi_engine.kernels.prefill` |
| `llm_app.prefill_kernels` | `ramvi_engine.kernels.prefill_kernels` |
| `llm_app.cpu_layer` | `ramvi_engine.cpu.cpu_layer` |
| `llm_app.gcpu` | `ramvi_engine.cpu.gcpu` |
| `llm_app.gguf` | `ramvi_engine.formats.gguf` |
| `llm_app.model` | `ramvi_engine.formats.model` |
| `llm_app.tokenizer_check` | `ramvi_engine.formats.tokenizer_check` |
| `llm_app.plan` | `ramvi_engine.planning.plan` |
| `llm_app.tune` | `ramvi_engine.planning.tune` |
| `llm_app.measured_plan` | `ramvi_engine.planning.measured_plan` |
| `llm_app.constrain` | `ramvi_engine.generation.constrain` |
| `llm_app.grammar_worker` | `ramvi_engine.generation.grammar_worker` |
| `llm_app.server` | `ramvi_engine.server.app` |
| `llm_app.request_validation` | `ramvi_engine.server.validation` |
| `llm_app.__main__` | `ramvi_engine.cli` |

## Совместимость

Старые имена остаются тонкими переходными файлами. Они возвращают тот же объект модуля, поэтому monkeypatch, глобальное состояние и кэши не дублируются. Только `llm_app` знает расположение нового пакета в исходном рабочем дереве; RAMVI Engine не импортирует старый пакет. Этот слой удаляется по пункту 1.11 после экзамена.

Новые команды доступны после установки `pip install --no-deps -e ramvi/engine`: `python -m ramvi_engine` и `ramvi-engine`. Импорт корневого пакета и `--help` не загружают CUDA или модели.

## Ресурсы и проверка

Генератор грамматики находится рядом с `generation/constrain.py` и включён в пакет. Native-загрузчик ищет библиотеку и манифест в `ramvi_engine/native`. C++-исходник и флаги сборки сохранены. Метаданные сервера вычисляют hash всего нового дерева Python-модулей, а не одного подпакета `server`.

Имена переменных окружения и путь к кэшу автонастройки пока прежние: их смена относится к пункту 1.8. Байты вычислительных Triton-модулей без внутренних импортов сохраняются, чтобы не менять ключ существующего кэша автонастройки. Массовое форматирование перенесённого кода не совмещается с переносом.

Для нового кода применяются Ruff и строгий mypy. Перенесённые модули получают исключение из стилевого переформатирования; критические ошибки имён и синтаксиса продолжают проверяться. Изменение стиля старого ядра должно быть отдельной работой.

Обязательные проверки: старые модульные тесты через новые исходники; запуск CLI без GPU; идентичность переходных модулей; загрузка native по новому пути с проверкой манифеста; состав wheel; сравнение вычислительных AST и хешей C++/ядер до и после переноса. Windows, эталон fp32 и WIDE30 требуют отдельного разрешённого прогона, описанного в пункте 1.9.


## Локальная сверка исходников

Сравнение с коммитом `7e33302` выполнено для всех 25 перенесённых Python-модулей. AST без операторов импорта совпал с исходным после применения только перечисленных выше правок размещения native, метаданных и CLI. Вычислительные тела не изменены. Шесть независимых Triton-модулей (`wcomp`, `gkernels`, `tkernels`, `q4`, `vocab8`, `prefill_kernels`), C++ и оба файла сборки совпали побайтно. Это структурная проверка, а не измерение скорости GPU.

## Результат локальных проверок

На macOS прошли 89 прежних модульных тестов и 11 проверок нового пакета в окружении с вычислительными зависимостями. В лёгком окружении: 6 проверок прошли, 5 пропущены из-за отсутствия NumPy/Torch. Ruff, проверка форматирования и строгий mypy для новых тестов проходят.

Wheel установлен без зависимостей в отдельное временное окружение вне рабочего дерева: импорт, обе точки входа CLI и наличие native/grammar-ресурсов проверены. Windows DLL успешно пересобрана новым `native/build.sh` на Mac; исполнение DLL на Windows этим не проверено. Скорость и ответы GPU после переноса пока не измерены; пункт 1.4 остаётся открытым до экзамена.

## Повторение структурной сверки

Проверка сохранена в `engine/tests/test_migration_sources.py`: 25 сравнений вычислительных AST, 6 побайтных сравнений Triton-модулей, 3 сравнения native-файлов и контроль обнаружения изменения арифметики. Для неё нужен git с исходным коммитом `7e3330232b4b83e13f1456cb05c579fef52ef9a9`; при shallow checkout историю следует получить до запуска. Проверка ничего не запускает из старых исходников и не требует моделей или CUDA.

Из каталога `ramvi`:

```sh
python -m pytest engine/tests/test_migration_sources.py
```

Это временный контракт приёмки переноса: до экзамена 1.9 вычисления должны оставаться прежними. После приёмки будущие изменения алгоритмов потребуют отдельного пересмотра этой проверки, а не молчаливой смены эталона. Абсолютные импорты проверяются отдельно в `test_package_layout.py`. Там же проверяется, что метаданные сервера замечают изменение вложенного исходника и его переименование, но не изменение логов.

В `.gitattributes` продукта для Python, C++, shell и TOML закреплены окончания строк LF. Checkout на Windows должен сохранять байты исходников, участвующие в проверке native-манифестов и ключах автонастройки, независимо от пользовательского `core.autocrlf`.
