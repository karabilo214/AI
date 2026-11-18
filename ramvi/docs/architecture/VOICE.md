# Голосовой ввод (этап 5)

Решение — [ADR-0004](../adr/ADR-0004-stt-backend.md): whisper.cpp, модель large-v3-turbo q5_0, запасная small q5_1. Пункт 5.2 [PLAN.md](../PLAN.md).

## Как идёт запись

Микрофон включается только кнопкой или Ctrl+M (работает и во время набора текста); Esc во время записи отменяет её (5.3).

1. Кнопка микрофона у поля ввода (`VoiceViewModel`) → `POST /voice/start`. Сервис открывает микрофон и сразу, пока человек говорит, запускает процесс распознавания с моделью.
2. Пока идёт запись, окно раз в 200 мс спрашивает `GET /voice`: длительность и громкость для индикатора; на пределе (120 с) запись останавливается сама.
3. Второе нажатие → `POST /voice/stop`: сервис закрывает микрофон, передаёт звук процессу распознавания и возвращает текст. Текст **дописывается в поле ввода**, в чат сам не уходит. Крестик → `POST /voice/cancel`: звук забывается без распознавания.

## Части

| Часть | Файл | Что делает |
|---|---|---|
| Модели | `service/ramvi_service/voice/models.py` | Файл, адрес, размер, SHA-256; папка `models/stt/`; детектор речи Silero скачивается вместе с моделью, без него модель не считается установленной |
| Запись | `voice/audio.py` | PortAudio (`sounddevice`): на Windows — WASAPI (полные имена, 16 кГц через `auto_convert`), иначе интерфейс системного устройства; запись 16 бит моно в память, предел длины. Если 16 кГц не открылась — частота устройства, пересчёт делает процесс распознавания |
| Процесс распознавания | `voice/worker.py`, `voice/stt.py` | Отдельный процесс с whisper.cpp (`pywhispercpp`), только CPU, детектор речи с запасом 1000 мс (иначе на тишине whisper выдумывает «Продолжение следует...»); RPC общего `ProcessSupervisor`. Звук передаётся через общую память, не через диск. Без дела 5 минут — процесс останавливается и освобождает память; смена модели — новый процесс |
| Выбор способа (5.4) | `voice/service.py`, `voice/accel.py`, `voice/server.py`, `hardware.ram_available_mb/nvidia_free_mb` | При каждой записи: видеокарта (`whisper-server` из ускорителя, одна запись — один процесс), если ускоритель скачан и хватает видеопамяти и RAM; иначе CPU; не хватает RAM — small; не хватает и ей — `voice.no_memory` до открытия микрофона |
| Сервис | `voice/service.py` | Состояния idle → recording → transcribing; ошибки `voice.*`; скачивание модели через `downloads.Download` (HTTPS, SHA-256 до активации) |
| Оболочка | `VoiceViewModel.cs`, `SoundSettingsViewModel.cs` | Кнопка, индикатор, раздел «Звук» в настройках |

`sounddevice`, `pywhispercpp` и `numpy` импортируются только при записи и в процессе распознавания: сервис и проверки CI работают без них (`pip install ./service[voice]` — с голосом). В переносимый комплект оба пакета входят (`packaging/build_portable.py`).

## API

| Запрос | Ответ |
|---|---|
| `GET /voice` | `state`, `seconds`, `level`, `limit_reached`, `language`, `device`, `model` (выбранная: `installed`, `loaded`), `models` (все: `installed`), `download` |
| `GET /voice/devices` | `devices: [{name, default}]` |
| `POST /voice/start` `{device?}` | как `GET /voice`; микрофон по умолчанию — из настроек |
| `POST /voice/stop` | `text`, `language`, `audio_s`, `proc_s` |
| `POST /voice/cancel` | как `GET /voice` |
| `POST /voice/model/download` `{model?}`, `POST /voice/model/cancel` | как `GET /voice` |
| `POST /voice/accel/download` | как `GET /voice`; ускоритель на видеокарте (только Windows + NVIDIA) |

В `GET /voice` также `backend` (где идёт текущая запись), `last` (`backend`, `model`, `proc_s` прошлого распознавания), `accel` (`offered`, `installed`, `size`); в ответе `stop` — `backend` и `model`.

Настройки (`settings.json`): `voice_device` («» — системный), `voice_language` (`auto`/`ru`/`en`/`uk`), `voice_model`.

## Контракт для модулей (5.6)

`service/ramvi_service/contracts/voice.py` — без аудиобиблиотек, whisper, TTS и Character:

- `SpeechCapabilities` — языки (`auto`, `ru`, `en`, `uk`), авто-язык, `streaming=False` (потоковая транскрипция — VOI-04), `cancel=True`, `backends` (`cpu` или `cpu`+`gpu`, если скачан ускоритель), предел длины;
- `AudioCapture`, `AudioStream`, `Device` — захват (реализация `voice/audio.py`); `SpeechRecognizer` — распознавание (`voice/stt.py` на CPU, `voice/server.py` на видеокарте);
- `MICROPHONE = "audio_input"` — единица ресурса в общем `ResourceBroker` хоста модулей (ёмкость 1). Запись берёт аренду и отдаёт её после остановки, отмены или ошибки открытия микрофона; занят — `voice.mic_busy`.

Модули пользуются голосом через Scoped Services (право `voice`): `voice.capabilities`, `voice.listen.start` `{device?}`, `voice.listen.stop` (текст, язык, где распознано), `voice.listen.cancel`. Модуль останавливает только свою запись (`voice.not_owner`); общий Stop интерфейса отменяет любую. Остановленный модуль не держит микрофон: его запись отменяется вместе с остальными ресурсами. Character повторно использует захват и распознавание, а оркестрацию и эмоциональную политику держит у себя.

Отмена во время распознавания останавливает процесс распознавания; `stop` возвращает `voice.cancelled`.

## Ошибки

`voice.model_missing` и `voice.no_memory` (до открытия микрофона), `voice.accel_unsupported`, `voice.no_device`, `voice.device_missing`, `voice.device_error`, `voice.unavailable` (нет пакетов), `voice.mic_busy`, `voice.not_owner`, `voice.cancelled`, `voice.too_short` (< 0,3 с), `voice.no_signal` (одни нули — так macOS отдаёт звук без разрешения на микрофон), `voice.no_speech`, `voice.stt_failed`, `voice.busy`. После любой ошибки состояние — idle, чат работает.

## Проверено

- Тесты сервиса `tests/test_voice.py` (подставной микрофон и распознаватель; настоящий дочерний процесс с подставной моделью — общая память и RPC), намеренные порчи кода 6/6 пойманы. Тесты оболочки `VoiceTests.cs`, стык с настоящим сервисом в `ServiceHostTests`, снимки в обеих темах.
- Mac M1 Pro, настоящий whisper.cpp large-v3-turbo q5_0, звук из файлов вместо микрофона (FLEURS RU/UK/EN и синтетика): язык верно 12/12; фраза 2–14 с — 3,0 с с заданным языком, 5,8–6,9 с с «auto» (5 потоков). После закрытия сервиса процессов распознавания не остаётся.
- Найдено и исправлено проверкой с настоящим whisper: PortAudio может прислать звук до того, как запись узнала частоту устройства.
- Windows (ноутбук, Ryzen 7 260), комплект `runtime\python` + колёса голоса: настоящий микрофон через WASAPI (полные имена, уровень звука идёт), 3 с тишины → «речь не распознана» (без детектора было «Продолжение следует...»); речь из файлов — тексты как на Mac, 7,5–7,9 с на фразу с заданным языком, 14,6–15,6 с с «auto»; после закрытия процессов не остаётся. Скорость на Windows CPU — задача 5.4 ([ADR-0004](../adr/ADR-0004-stt-backend.md)).
- 5.4 на ASUS: ускоритель скачан сервисом с GitHub с проверкой SHA-256 и распакован (66 с); распознавание на видеокарте 0,30–0,55 с после остановки записи; рядом с 14B — без нехватки памяти, генерация медленнее на 1,5 %; CPU при 14B отказал по памяти с понятной ошибкой ([ADR-0004](../adr/ADR-0004-stt-backend.md)).
