"""6.2 (AG-02): единая версионируемая схема инструментов, журнал и исполнение.

Инструмент — `ToolSpec` (имя, описание, JSON Schema аргументов, нужное право,
эффект) и обработчик. `ToolRunner.execute` — единственный путь исполнения:

1. аргументы проверяются по схеме — некорректный вызов не исполняется;
2. вызов записывается в журнал **до** исполнения; не записался — не исполняется;
3. обработчик выполняется; результат или ошибка становятся `ToolResult`,
   который возвращается в контекст модели и дописывается в журнал.

Эффекты `write` и `command` ничего не меняют сами: обработчик готовит предложение
(diff или команду), применение — после подтверждения пользователя (6.4).

Опасные инструменты (риск high/critical) — переключатели в настройках доступа
агента (`ToolAccess`, вкладка «Агент»). Модели предлагается инструмент, только если
он включён и его право выдано; выключенный вызов не исполняется.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ramvi_service.agent.workspace import AccessDenied, Permission, Workspace

SCHEMA_VERSION = "ramvi.agent.tools/1"
Effect = Literal["read", "write", "command"]
Risk = Literal["low", "medium", "high", "critical"]

MAX_RESULT_CHARS = 16_000  # что видит модель
MAX_JOURNAL_CHARS = 4_000  # что хранит журнал о результате
MAX_JOURNAL_ARG_CHARS = 2_000
SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|authorization)")


class ToolError(Exception):
    """Ожидаемая ошибка инструмента с ключом (`agent.tool.*`) — уходит модели."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code, self.params = code, params


@dataclass(frozen=True)
class ToolSpec:
    """Описание инструмента; `parameters` — JSON Schema объекта аргументов."""

    name: str
    description: str
    parameters: dict[str, Any]
    permission: Permission
    effect: Effect = "read"
    risk: Risk = "low"
    version: int = 1

    @property
    def configurable(self) -> bool:
        """Опасный инструмент: пользователь включает и выключает его в настройках."""
        return self.risk in ("high", "critical")

    def to_openai(self) -> dict[str, Any]:
        """Описание для `tools` в OpenAI-совместимом запросе к модели."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolAccess:
    """Настройки доступа к опасным инструментам (вкладка «Агент» в настройках).

    `disabled` — выключенные инструменты; `allow_destructive_commands` — разрешить
    команды, которые удаляют данные, повышают права или исполняют произвольный код
    (по умолчанию заблокированы). Меняет только пользователь.
    """

    disabled: frozenset[str] = frozenset()
    allow_destructive_commands: bool = False


@dataclass(frozen=True)
class ToolResult:
    """Итог вызова. `content` — текст для модели; `data` — для runtime и интерфейса."""

    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    truncated: bool = False

    @classmethod
    def failure(cls, code: str, **params: Any) -> "ToolResult":
        detail = f" {json.dumps(params, ensure_ascii=False, default=str)}" if params else ""
        return cls(False, f"ERROR {code}{detail}", {"params": params}, error=code)

    def for_model(self) -> str:
        if len(self.content) <= MAX_RESULT_CHARS:
            return self.content
        return self.content[:MAX_RESULT_CHARS] + "\n[... output truncated]"


# --- проверка аргументов --------------------------------------------------------------

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def validate(schema: Mapping[str, Any], value: Any, where: str = "$") -> list[str]:
    """Проверить значение по подмножеству JSON Schema, которое используют инструменты.

    Поддержано: type, properties, required, additionalProperties: false, enum,
    minLength/maxLength, minimum/maximum, items, maxItems. Возвращает список ошибок.
    """
    expected = schema.get("type")
    if expected is not None:
        kinds = _TYPES[expected]
        if not isinstance(value, kinds) or (expected == "integer" and isinstance(value, bool)):
            return [f"{where}: expected {expected}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: must be one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{where}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{where}: too long")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{where}: above {schema['maximum']}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{where}: too many items")
        for index, item in enumerate(value):
            errors += validate(schema.get("items", {}), item, f"{where}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{where}.{name}: required")
        for name, item in value.items():
            if name in properties:
                errors += validate(properties[name], item, f"{where}.{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}.{name}: unknown argument")
    return errors


# --- журнал ---------------------------------------------------------------------------


class Journal(Protocol):
    """Журнал шагов: запись до исполнения, результат — после."""

    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str: ...

    def finish(self, call_id: str, result: str, duration: float) -> None: ...


def redact(args: Mapping[str, Any]) -> dict[str, Any]:
    """Аргументы для журнала: секретные ключи скрыты, длинный текст — длина и хеш."""
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if SECRET_KEY.search(key):
            clean[key] = "***"
        elif isinstance(value, str) and len(value) > MAX_JOURNAL_ARG_CHARS:
            digest = hashlib.sha256(value.encode()).hexdigest()[:16]
            clean[key] = {"chars": len(value), "sha256": digest, "head": value[:200]}
        elif isinstance(value, Mapping):
            clean[key] = redact(value)
        else:
            clean[key] = value
    return clean


class StorageJournal:
    """Журнал в SQLite (таблица `tool_calls`) для одного запуска агента."""

    def __init__(self, runs: Any, run_id: str) -> None:
        self.runs, self.run_id = runs, run_id

    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
        return str(self.runs.add_tool_call(self.run_id, tool, args, permission).id)

    def finish(self, call_id: str, result: str, duration: float) -> None:
        self.runs.finish_tool_call(call_id, result, duration)


def _journal_result(result: ToolResult) -> str:
    summary = {
        "ok": result.ok,
        "error": result.error,
        "truncated": result.truncated or len(result.content) > MAX_RESULT_CHARS,
        "chars": len(result.content),
        "content": result.content[:MAX_JOURNAL_CHARS],
    }
    return json.dumps(summary, ensure_ascii=False)


# --- исполнение -----------------------------------------------------------------------

Handler = Callable[[dict[str, Any]], ToolResult]


class ToolRunner:
    """Реестр инструментов и единственный путь их исполнения."""

    def __init__(
        self, workspace: Workspace | None = None, access: ToolAccess | None = None
    ) -> None:
        self._tools: dict[str, tuple[ToolSpec, Handler]] = {}
        self.workspace = workspace
        self.access = access or ToolAccess()

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        if spec.parameters.get("type") != "object":
            raise ValueError(f"tool {spec.name}: parameters must be an object schema")
        self._tools[spec.name] = (spec, handler)

    @property
    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def is_available(self, spec: ToolSpec) -> bool:
        """Инструмент включён в настройках, его право выдано и grant не отозван."""
        if spec.name in self.access.disabled:
            return False
        ws = self.workspace
        return ws is None or (not ws.revoked and ws.permissions.allows(spec.permission))

    def available(self) -> list[ToolSpec]:
        return [spec for spec in self.specs if self.is_available(spec)]

    def openai_tools(self) -> list[dict[str, Any]]:
        """Только доступные инструменты — то, что предлагается модели."""
        return [spec.to_openai() for spec in self.available()]

    def schema(self) -> dict[str, Any]:
        """Версионированное описание всех инструментов (для модели и интерфейса)."""
        return {
            "version": SCHEMA_VERSION,
            "tools": [
                {
                    "name": s.name,
                    "version": s.version,
                    "permission": s.permission,
                    "effect": s.effect,
                    "risk": s.risk,
                    "configurable": s.configurable,
                    "available": self.is_available(s),
                    "description": s.description,
                    "parameters": s.parameters,
                }
                for s in self.specs
            ],
        }

    def execute(
        self, name: str, args: Any, journal: Journal, *, permission: str = "auto"
    ) -> ToolResult:
        """Проверить, записать в журнал и выполнить вызов; ошибки — в результат."""
        entry = self._tools.get(name)
        if entry is None:
            return self._reject(journal, name, args, ToolResult.failure("agent.tool.unknown"))
        spec, handler = entry
        if spec.name in self.access.disabled:
            return self._reject(journal, name, args, ToolResult.failure("agent.tool.disabled"))
        if self.workspace is not None:
            try:
                self.workspace.require(spec.permission)
            except AccessDenied as error:
                failure = ToolResult.failure(error.code, **error.params)
                return self._reject(journal, name, args, failure)
        errors = validate(spec.parameters, args)
        if errors:
            failure = ToolResult.failure("agent.tool.invalid_args", errors=errors)
            return self._reject(journal, name, args, failure)
        call_id = journal.begin(name, redact(args), permission)
        started = time.monotonic()
        try:
            result = handler(dict(args))
        except AccessDenied as error:
            result = ToolResult.failure(error.code, **error.params)
        except ToolError as error:
            result = ToolResult.failure(error.code, **error.params)
        except Exception as error:  # noqa: BLE001 — любая ошибка возвращается модели
            result = ToolResult.failure("agent.tool.failed", error=type(error).__name__)
        journal.finish(call_id, _journal_result(result), time.monotonic() - started)
        return result

    @staticmethod
    def _reject(journal: Journal, name: str, args: Any, result: ToolResult) -> ToolResult:
        """Некорректный вызов не исполняется, но остаётся в журнале как отклонённый."""
        shown = redact(args) if isinstance(args, dict) else {"raw": repr(args)[:200]}
        call_id = journal.begin(str(name)[:100], shown, "denied")
        journal.finish(call_id, _journal_result(result), 0.0)
        return result
