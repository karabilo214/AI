# Чат (UI-02, пункт 3.4)

Оболочка Avalonia (`desktop/`) запускает процесс сервиса и говорит с ним по локальному HTTP.
Модель, история и лимиты живут в сервисе; оболочка только показывает и отправляет.

## Процессы

```
RAMVI (C#) ──запуск──▶ python -m ramvi_service.app --data-dir …   (ключ в RAMVI_UI_TOKEN)
    │                     │  печатает {"event":"ready","port":N}
    └─HTTP 127.0.0.1:N────┘──▶ EngineManager ──▶ EngineProcess (ramvi-engine serve)
```

- `desktop/src/Ramvi.Desktop/Services/ServiceHost.cs` — запуск сервиса, ожидание `ready`, остановка.
  Сервис получает `RAMVI_EXIT_WITH_PARENT=1` и выходит сам, если оболочка умерла.
- Для разработки: `RAMVI_ROOT`, `RAMVI_PYTHON`, `RAMVI_DATA_DIR`, `RAMVI_SERVICE_ARGS_JSON`
  (например `["--engine-command-json", "[…fake_engine.py]"]`), `RAMVI_LANGUAGE`.

## API `/ui/v1` (`service/ramvi_service/ui_api.py`)

Только 127.0.0.1, заголовок `Authorization: Bearer <ключ>`, проверка `Host`.

| Запрос | Что делает |
|---|---|
| `GET /status` | состояние движка и лимиты вложений |
| `GET /models` | установленные модели |
| `GET /chats`, `POST /chats` | список (новые сверху, флаг `generating`), новый диалог |
| `GET /chats/{id}/messages`, `DELETE /chats/{id}` | история, удаление |
| `POST /chats/{id}/send` | поток SSE: `user`, `status`, `reasoning`, `text`, `done` |
| `POST /chats/{id}/regenerate` | поток SSE: `regenerate` (какие ответы убраны), затем как send |
| `POST /chats/{id}/stop`, `/clear`, `/model`, `/title` | остановка, очистка, смена модели, заголовок |

Ошибки — `{"error":{"code","params"}}`; код — ключ перевода из `i18n/<язык>.json`.
Обрыв соединения клиента отменяет генерацию; ответ сохраняется с `finish = cancelled`.

## Хранилище

Миграция ядра v2 (`storage/ramvi_storage/schema.py`): у сообщений есть `reasoning`, `finish`
(`stop` / `length` / `cancelled` / `error`) и вложения. Thinking хранится отдельно от ответа.

## Вложения

Лимиты по умолчанию (`ramvi_service.chat.Limits`): 5 файлов, 128 КБ на файл, 256 КБ всего,
32 000 символов текста. Оболочка проверяет те же лимиты до отправки (`AttachmentReader`):
размер — до чтения файла, только UTF-8 без нулевых байтов, BOM убирается. Сервис проверяет снова.

## Проверки

- `tests/test_chat.py` — сервис, HTTP и поддельный движок `tests/fixtures/fake_engine.py`
  в настоящих процессах.
- `desktop/tests` (`dotnet test desktop/tests/Ramvi.Desktop.Tests`): модель представления,
  вложения, переводы (ru = en по ключам и параметрам, все ключи из кода есть), HTTP-клиент,
  окно без экрана (Avalonia.Headless: привязки, свёрнутое thinking, Enter отправляет).

Не проверено: настоящая модель через оболочку, Windows, перетаскивание мышью в живом окне.
Состояния интерфейса с действиями восстановления — пункт 3.5.
