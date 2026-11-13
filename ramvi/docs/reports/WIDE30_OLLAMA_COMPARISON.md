# 30 вопросов: текущий llm_app и Ollama

После этого замера O02 выключен по умолчанию по просьбе владельца. Теперь он включается только через `LLM_DRAFT_GATE=1`; скорость восстановленной обычной политики здесь ещё не перемерялась.

Qwen3‑14B, угадыватель Qwen3‑0.6B. Два полных прогона по 30 вопросам, максимум 200 токенов на ответ. Температура 0, seed 0, контекст 2048, размышления выключены. Запросы и их порядок совпадают с сохранённым набором Ollama.

| Вариант | Скорость, токенов/с | Первое слово, медиана, с | Время 30 ответов, с |
|---|---:|---:|---:|
| Ollama, сохранённый замер | 6.27 | 0.332 | 833.2 |
| Наш движок до последних изменений | 8.41 | 0.592 | 632.0 |
| Наш текущий, прогон 1 | 7.80 | 0.593 | 680.1 |
| Наш текущий, прогон 2 | 7.82 | 0.589 | 678.1 |
| Наш текущий, оба прогона вместе | 7.81 | 0.590 | 679.1 (среднее) |

Текущая скорость относительно Ollama: **1.246×**; относительно нашей предыдущей версии: **0.928×**. Медиана отношения по вопросам — 1.187×. Не медленнее Ollama на 19/30 вопросах.

Худший случай: №9, творчество, 0.774×. Лучший: №29, json, 3.507×.
По текущему замеру скорость выше сохранённой Ollama на 24.6%, но ниже нашей предыдущей версии на 7.2%. Последние изменения не подтвердили ускорение относительно прежнего нашего результата.

## По каждому вопросу

| № | Вопрос | Ollama, ток/с | Наш 1, ток/с | Наш 2, ток/с | Наш / Ollama | Первое слово: Ollama / наш, с |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Explain how the internet works to a 12-year-old. | 6.23 | 5.38 | 5.23 | 0.852× | 0.309 / 0.576 |
| 2 | What is the difference between a virus and a bacterium? | 6.27 | 6.62 | 6.64 | 1.058× | 0.305 / 0.574 |
| 3 | Объясни простыми словами, что такое инфляция и почему она возникает. | 6.25 | 5.66 | 5.58 | 0.898× | 0.352 / 0.601 |
| 4 | Who was Marie Curie and what is she known for? | 6.24 | 6.38 | 6.39 | 1.023× | 0.317 / 0.578 |
| 5 | Расскажи кратко историю Санкт-Петербурга. | 6.32 | 5.71 | 5.75 | 0.907× | 0.318 / 0.568 |
| 6 | Give me practical advice on how to prepare for a job interview. | 6.27 | 7.03 | 7.06 | 1.123× | 0.317 / 0.577 |
| 7 | Как правильно начать бегать, если раньше не занимался спортом? | 6.31 | 5.17 | 5.01 | 0.808× | 0.340 / 0.591 |
| 8 | Write a short poem about autumn in the city. | 6.24 | 4.87 | 4.95 | 0.786× | 0.323 / 0.568 |
| 9 | Напиши короткую сказку про кота, который хотел стать капитаном корабля. | 6.29 | 4.86 | 4.88 | 0.774× | 0.349 / 0.753 |
| 10 | Write the opening paragraph of a mystery novel set in Venice. | 6.21 | 4.98 | 4.90 | 0.796× | 0.331 / 0.576 |
| 11 | Write a polite email to a landlord asking to fix a broken heater. | 6.25 | 5.93 | 6.00 | 0.954× | 0.332 / 0.566 |
| 12 | Напиши деловое письмо с просьбой перенести встречу на следующую неделю. | 6.22 | 6.14 | 6.30 | 1.000× | 0.354 / 0.762 |
| 13 | Write a Python function that merges two sorted lists into one sorted list, with tests. | 6.27 | 13.78 | 14.00 | 2.216× | 0.388 / 0.591 |
| 14 | Write a JavaScript function that debounces another function, and explain how it works. | 6.24 | 9.11 | 9.12 | 1.459× | 0.330 / 0.570 |
| 15 | Write a SQL query to find the top 5 customers by total order amount from tables customers and orders. | 6.28 | 12.69 | 12.77 | 2.028× | 0.334 / 0.601 |
| 16 | Write a regular expression that validates an email address and explain each part. | 6.23 | 10.78 | 10.78 | 1.732× | 0.326 / 0.563 |
| 17 | Напиши на Go функцию, которая читает файл построчно и считает количество слов. | 6.26 | 13.74 | 13.80 | 2.199× | 0.358 / 0.584 |
| 18 | Write a C function that reverses a string in place. | 6.24 | 10.51 | 10.58 | 1.690× | 0.318 / 0.561 |
| 19 | A store sells apples at 3 for $2. How much do 18 apples cost? Explain step by step. | 6.25 | 15.44 | 15.64 | 2.485× | 0.322 / 0.758 |
| 20 | Solve the equation 2x^2 - 8x + 6 = 0 and show the steps. | 6.30 | 12.07 | 12.13 | 1.921× | 0.325 / 0.608 |
| 21 | Поезд проехал 240 км за 3 часа. С какой скоростью он ехал и сколько проедет за 5 часов? | 6.28 | 14.61 | 14.76 | 2.339× | 0.370 / 0.765 |
| 22 | If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain. | 6.24 | 10.18 | 10.35 | 1.644× | 0.334 / 0.597 |
| 23 | Translate into Russian: 'The early bird catches the worm, but the second mouse gets the cheese.' | 6.45 | 5.70 | 5.92 | 0.900× | 0.326 / 0.593 |
| 24 | Translate into English: 'Тише едешь — дальше будешь. Не откладывай на завтра то, что можно сделать сегодня.' | 6.52 | 6.64 | 6.69 | 1.022× | 0.341 / 0.756 |
| 25 | Summarize the following text in two sentences: The Industrial Revolution began in Britain in the late 18th century and spread to Europe and North America. It transformed economies that had been based on agriculture and handicrafts into economies based on large-scale industry, mechanized manufacturing, and the factory system. New machines, new power sources, and new ways of organizing work made existing industries more productive and efficient. | 6.29 | 8.02 | 8.07 | 1.279× | 0.373 / 1.487 |
| 26 | Перескажи этот текст по-русски в трёх предложениях: The Industrial Revolution began in Britain in the late 18th century and spread to Europe and North America. It transformed economies that had been based on agriculture and handicrafts into economies based on large-scale industry, mechanized manufacturing, and the factory system. New machines, new power sources, and new ways of organizing work made existing industries more productive and efficient. | 6.31 | 7.71 | 8.08 | 1.250× | 0.383 / 1.722 |
| 27 | List 15 countries in Africa with their capitals. | 6.32 | 14.27 | 14.40 | 2.269× | 0.330 / 0.576 |
| 28 | Перечисли 10 полезных привычек для здоровья с кратким пояснением. | 6.29 | 8.83 | 8.94 | 1.413× | 0.339 / 0.749 |
| 29 | Return a JSON array of 5 books with fields title, author, year. Output only JSON. | 6.33 | 22.21 | 22.16 | 3.507× | 0.332 / 0.593 |
| 30 | Write a short dialogue between a customer and a barista in a coffee shop. | 6.20 | 5.39 | 5.42 | 0.872× | 0.334 / 0.591 |

## По типам вопросов

| Тип | Медиана наш / Ollama |
|---|---:|
| объяснение | 0.898× |
| факты | 0.965× |
| совет | 0.965× |
| творчество | 0.786× |
| письмо | 0.977× |
| код | 1.880× |
| математика | 2.339× |
| рассуждение | 1.644× |
| перевод | 0.961× |
| пересказ | 1.265× |
| список | 1.841× |
| json | 3.507× |
| диалог | 0.872× |

## Проверка ответов и условия

- Тексты между двумя текущими прогонами совпали на 30/30 вопросах.
- С нашей предыдущей версией тексты обоих прогонов совпали на 30/30 вопросах.
- До измерения проверены 7 контролей: ответы target без угадывателя и с ним совпали. Проверки с известным ответом: 2+2=4 и 137×29=3973.
- Совпадение текста не является оценкой истинности всех ответов; это проверка сохранения поведения движка.
- Размещение: GPU 21, подкачка 8, гибрид 11, CPU-only 0.
- Prefill: blocked, порог 256 токенов. Эта проверка не заменяет отдельный тест длинных запросов.
- Дополнительная калибровка измеренного планировщика упёрлась в RAM. Основной тест выполнен после перезапуска процесса, на штатном распределении слоёв; измеренные маршруты не включены.
- Ollama не запускалась: использован сохранённый результат на том же наборе. Это историческое сравнение, а не одновременный замер двух серверов. Исторический JSON Ollama не содержит полного манифеста оборудования и версии.
- Скорость вычислена как сумма eval_count / сумма eval_duration, а не среднее скоростей отдельных вопросов. Как и в предыдущем отчёте, счётчик нашего сервера исключает первый токен; архив Ollama не выделяет его отдельно.
- Время первого слова измерено HTTP-клиентом. Загрузка модели, прогрев и контрольные запросы вынесены за измеряемый набор.
- Во время обоих прогонов код и настройки не менялись. На каждый прогон сохранены все ответы и параметры.

Длины полных запросов: 22–102 токена. Порог нового prefill достигают 0/30: ускорение длинного prefill этим набором не измеряется.

## Файлы и воспроизводимость

- Архив Ollama: `llm_app/results/wide_ollama_14b.json`; SHA256 `5950073e9f4da446176d3a1f439b9e0c78ff935322b73c84ef29c04312272079`.
- Архив нашей предыдущей версии: `llm_app/results/wide_ours_14b_fixed3.json`.
- Веса: `sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e`; токенизатор `a429c8857a5dd281eebb7c91843ba4888a8fe42d63135d3d504170ab72b3fd09`.
- Манифест сервера: `server_metadata.json`; hash кода `2509b536af16388992dc4150cedfae8706ed0099cdceafd3e26f26c5f0c931f2`.
- Полные ответы: `wide_current_14b_r1.json`, `wide_current_14b_r2.json`.
- Таблица для обработки: `comparison.csv`; исходные контроли: `controls.json`.
