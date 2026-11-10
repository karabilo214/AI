# MyAI

Опыты с моделью мозга, которая учится во время работы: импульсные нейроны, рост
связей, торможение, награда. Главная часть — `human_brain/`.

## Структура

```
human_brain/
  core/        ядро: нейрон, синапс, миелин, мозг (Brain), kernel и fast (numba),
               сохранение мозга (brain_store)
  parts/       врождённые устройства и механизмы обучения: рост связей (Growth),
               любопытство (Curiosity), знакомость, эпизоды, слух, «бывает вместе»
               (Association), найм детекторов (Recruit), накопитель, защёлка,
               заготовки (Units, NeuralChoice), правки головы (Head)
  stages/      ступени обучения по порядку: сложение, вычитание, умножение,
               деление, чтение, правило мягкости, цифры, знак «−», логика (И/ИЛИ/НЕ);
               tasks — общие параметры и простые задачи с наградой
  evolution/   эволюция параметров (evolve, evolve_addition, params_chosen.json)
  tests/       проверки и замеры (check_*, mixed_exam, capacity, ...)
  examples/    маленький пример работы мозга
  docs/        отчёт аудита кода
  data/        сохранённые мозги и результаты эволюции (не в git)
candy_world/   мир с конфетами (arith — мир ступеней арифметики) и ранние опыты
simple_counter/ первый простой счётчик
docs/          техническое задание ANM
results/       результаты опытов (не в git)
```

## Запуск

Всё запускается из корня проекта как модули:

```
.venv/bin/python -m human_brain.stages.minus --seed 1000 --days 2000
.venv/bin/python -m human_brain.stages.logic --op AND --seed 0
.venv/bin/python -m human_brain.tests.check_store
.venv/bin/python -m human_brain.tests.mixed_exam --seed 1000
```

Каждая ступень берёт мозг, сохранённый после предыдущей (`human_brain/data/brains/`),
а если его нет — обучает всю цепочку с нуля и сохраняет. Продолжение после
загрузки совпадает с обучением подряд до бита (`tests/check_store`).
