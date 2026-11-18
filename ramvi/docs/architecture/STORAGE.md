# Хранилище RAMVI (пункт 3.2)

Пакет `ramvi_storage` (`ramvi/storage/`) — локальная база приложения по разделу 10 ТЗ: SQLite в режиме WAL, миграции схемы и репозитории сущностей. Зависимостей, кроме стандартной библиотеки Python, нет; нужен SQLite не старше 3.37 (STRICT-таблицы).

## Где лежит

`open_core(data_dir)` открывает `<data_dir>/ramvi.sqlite3` и доводит схему ядра до текущей версии. Папка данных берётся из настроек (`Settings.data_dir`, 3.7). В базе нет абсолютного пути к ней самой: закрытую базу можно перенести вместе с папкой (проверено тестом). Пути к файлам моделей (`models.path`) записывает вызывающий код — внутри папки приложения их следует хранить относительными, как в настройках.

Файлы моделей и JSON-кэш каталога в базу не входят.

## Соединение

`Database`:

- `journal_mode=WAL`, `foreign_keys=ON`, `synchronous=NORMAL`, `busy_timeout` 5 с;
- `transaction()` — `BEGIN IMMEDIATE`; вложенный вызов в том же потоке — часть внешней транзакции; исключение — `ROLLBACK`;
- нарушения ограничений (внешний ключ, уникальность, CHECK) превращаются в `ConstraintError`, а не в сырую ошибку `sqlite3`;
- `backup(path)` — согласованная копия через online backup API, без остановки работы; `integrity_ok()` — `PRAGMA integrity_check`.

Одно соединение на процесс, все обращения под общей блокировкой. Два соединения (например, два процесса) могут писать одновременно: тест — 2 × 150 записей без потерь и с целой базой.

## Миграции

`migrate(db, owner, migrations, backup_dir=None)`. У каждого владельца своя цепочка версий 1, 2, 3 … в общей таблице `schema_migrations (owner, version, name, checksum, applied_at)`. Владелец ядра — `core`; модуль использует свой идентификатор. Цепочки независимы: обновление модуля не меняет версию ядра и других модулей.

| Правило | Что происходит |
|---|---|
| Миграция падает | откатывается целиком, версия не меняется, частично созданных таблиц нет |
| Применённая миграция изменена | `MigrationError` «изменена после применения» — нужна новая миграция |
| База новее кода | `MigrationError` «понижение не поддерживается»; база не трогается |
| Версии не подряд с 1, неверный владелец | `MigrationError` до любых изменений |
| Есть что применить к непустой базе | сначала копия `backups/<имя>.<владелец>.v<N>.<время>.bak` (`open_core` делает это всегда) |

Миграция — SQL-скрипт и/или функция `run(db)` в той же транзакции. Контрольная сумма — по имени и SQL; у функции учитывается `code_version`, его увеличивают при изменении функции.

Пространства данных модулей (3.12) строятся на этих цепочках: там добавляются проверка, что модуль создаёт и читает только свои таблицы, экспорт, отдельный сброс и удаление. Специфические схемы Character регистрирует сам модуль.

## Схема ядра, версия 1

Время — секунды Unix (REAL, UTC). JSON-поля — TEXT с `json_valid`. Все таблицы STRICT.

| Таблица | Поля по ТЗ | Дополнительно |
|---|---|---|
| `models` | id, provider, architecture, quant, sha256, bytes, path, license, installed_at, capabilities | sha256 — 64 символа |
| `model_profiles` | model_id, hardware_hash_local, context, plan, ttft, tokps, peak_ram, peak_vram, tested_at | удаляются вместе с моделью |
| `chats` | id, title, model_id, created_at, updated_at | при удалении модели model_id = NULL |
| `messages` | id, chat_id, role, content, token_usage, created_at | `seq` — порядок добавления (часы могут идти назад); удаляются вместе с чатом |
| `agent_runs` | id, objective, workspace, model_id, status, steps, token_usage, started_at, ended_at, result, stop_reason (v3) | статусы planned/running/waiting_approval/done/failed/cancelled |
| `agent_events` | seq, agent_run_id, kind, step, created_at, payload (v4) | хронология запуска агента (6.7); удаляется вместе с запуском |
| `tool_calls` | id, agent_run_id, tool_name, args_redacted, permission, result, duration | `seq`; permission auto/approved/denied |
| `client_keys` | id, client_name, permissions, secret_hash, created_at, last_seen, revoked_at | `requests` — счётчик запросов (7.2) |
| `downloads` | id, model_id, source, temp_path, bytes_done, total_bytes, state, checksum | model_id — id каталога, без внешнего ключа: записи модели до конца загрузки нет |

## Репозитории

`ramvi_storage.repositories` — только чтение и запись строк, без бизнес-логики:

- `Chats`: создать, `recent()` (сначала недавно изменённые), переименовать/сменить модель, удалить; `add_message` обновляет `updated_at` чата; `messages()` по порядку добавления; `delete_messages_from()` — для Regenerate; `clear()`.
- `Models`: `upsert`, `all`, `delete` (файл не трогается); профили — `add_profile`, `latest_profile(model, hardware_hash)`.
- `Downloads`: создать (`queued`), `update` состояния и прогресса, `unfinished()` — что продолжить после перезапуска.
- `AgentRuns`: создать (`planned`), `update`; конечный статус ставит `ended_at` и больше не меняется; `add_tool_call` / `tool_calls` по порядку. Аргументы инструмента должны быть очищены от секретов до записи — это задача Agent Runtime.

`KeyStore` из `ramvi_service.client_keys` хранит ключи в таблице `client_keys` общей базы: `KeyStore(db)` использует открытую базу и не закрывает её; `KeyStore(path)` открывает отдельный файл со схемой ядра (для проверок и совместимости).

## Проверки

`tests/test_storage.py` (20 тестов) и прежние `tests/test_client_keys.py` (10, без изменений). Проверка мутациями: отключение внешних ключей, `busy_timeout=0`, COMMIT вместо ROLLBACK, отключение проверки контрольной суммы, проверки «база новее кода», резервной копии, порядок сообщений по времени вместо `seq`, смена статуса после завершения — каждая из 8 поломок роняет тесты.

## Не входит в 3.2

- Шифрование секретов средствами ОС (ТЗ §10) — отдельно, с учётом portable-режима; ключи клиентов хранятся только как SHA-256.
- Изоляция и экспорт данных модулей — 3.12.
- Подключение базы к сервису при запуске — супервизор и сервис (3.3).
