# RAMVI Service

Лёгкий Module Host и публичные контракты платформы. Engine не зависит от этого пакета. Character, UI и аудио не загружаются при импорте.

[Архитектура и приёмка модулей](../docs/architecture/MODULE_CORE.md).

Установка из репозитория после Engine: `python -m pip install -e ramvi/service`.

Пакет предоставляет ModuleHost, Manifest/Authority, Scoped Services, EventBus, FIFO ResourceBroker/SharedLLM и optional ContextProvider. Worker использует общий ProcessSupervisor движка. Доверенные модули регистрируются явно; постоянное хранилище секретов и OS context adapters подключаются отдельно.
