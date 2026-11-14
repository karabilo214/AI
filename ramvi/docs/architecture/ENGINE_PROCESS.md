# Изолированный Engine process — 2.8

`ramvi_engine.engine_process.EngineProcess` — лёгкий контроллер для сервиса/будущего UI. Вызов start запускает отдельный Python-процесс `ramvi_engine serve`; PyTorch, CUDA, веса и Loader находятся в нём. Контроллер не импортирует inference и работает без установленного/импортированного torch до запуска worker. Desktop UI ещё относится к этапу 3: здесь готов и проверен процессный контракт, а не окно приложения.

```python
from ramvi_engine.engine_process import EngineProcess
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.planning.automatic import PlanningOptions

engine = EngineProcess(ModelConfig("Qwen3-0.6B", planning=PlanningOptions()))
try:
    engine.start()
    result = engine.call(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    status = engine.status()
finally:
    engine.close()
```

Engine слушает только loopback и получает случайный API-ключ через окружение, не argv. Старт проверяет instance ID и major API version; занявший порт другой сервер не принимается за нужный Engine. Локальный транспорт не использует HTTP proxy и не следует redirect. Контроллер сообщает starting/ready/failed/stopped, exit code, ограниченную диагностику и возможность restart. После падения worker родительский сервис/контроллер продолжает работать; рестарт явный, без бесконечного crash-loop. Долгий start следует вызывать в задаче сервиса/фоновом executor UI, а не в event loop. Повторный одновременный start отклоняется, не создаёт второй потерянный процесс.

Общий `ramvi_engine.processes.ProcessSupervisor` также используется Module Host. Он поддерживает bounded stdio JSONL RPC, ограниченные stdout/stderr и очереди, единый timeout на ожидание RPC/запись/чтение, stop/restart. На POSIX процессы имеют собственную session/process group. На Windows actual command запускается только после присоединения доверенного bootstrap к Job Object с kill-on-close, затем наследует Job. Bootstrap использует базовый интерпретатор: venv-redirector мог бы создать настоящий Python до назначения Job. Это устраняет окно, в котором мгновенный потомок мог бы уйти до назначения Job. Остановка явно вызывает TerminateJobObject: удерживаемая копия handle не должна сохранять потомков. Windows stdout/stderr читаются через PeekNamedPipe и ограниченный os.read; stop не ждёт блокировку TextIO.readline при закрытии канала. Завершается всё принадлежащее контроллеру дерево, а не процессы по имени; старый запрос не имеет права остановить уже новый worker.

Сбой/зависание запроса останавливает принадлежащий worker; следующие операции могут перезапустить его. При остановке reader/writer закрываются, нет неограниченного чтения логов или висящих команд. Контроллер не принимает произвольные shell-команды из модельного payload. Module entrypoint задаёт доверенная регистрация Host. Процессная изоляция защищает родителя от аварии и позволяет завершить CUDA-контекст целиком; это не OS sandbox для недоверенного Python-кода.

Локальные проверки используют настоящие процессы: RPC, повторный restart, timeout, crash HTTP-worker, сохранение родителя, завершение потомков, oversized output и Windows Job fixture. Аппаратный сквозной check `stage2_runtime` использует настоящую модель/HTTP и пишет отдельный JSON. Синхронный native CUDA kernel нельзя безопасно убить Python-потоком; остановка worker process является последней границей восстановления.
