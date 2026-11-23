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
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from ramvi_service.agent.workspace import PERMISSIONS, AccessDenied, Permission, Workspace

SCHEMA_VERSION = "ramvi.agent.tools/1"
Effect = Literal["read", "write", "command", "control"]
ToolPermission = Permission | Literal["context.ui", "agent.control"]
Risk = Literal["low", "medium", "high", "critical"]

MAX_RESULT_CHARS = 16_000  # что видит модель
MAX_JOURNAL_CHARS = 4_000  # что хранит журнал о результате
MAX_JOURNAL_ARG_CHARS = 2_000
SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|authorization)")


class StopRequested(Exception):
    """Пользователь остановил запуск агента (например, в запросе подтверждения)."""


class ToolError(Exception):
    """Ожидаемая ошибка инструмента с ключом (`agent.tool.*`) — уходит модели."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code, self.params = code, params


# AG-06: повтор имеет смысл только при исправимой ошибке. Остальные — запрет или
# решение пользователя: повторять бесполезно, нужен другой путь или честный отказ.
CORRECTABLE = frozenset(
    {
        "agent.tool.invalid_args",
        "agent.tool.unknown",
        "agent.tool.edit_not_found",
        "agent.tool.edit_ambiguous",
        "agent.tool.edit_empty",
        "agent.tool.bad_pattern",
        "agent.tool.pattern_too_long",
        "agent.tool.shell_syntax",
        "agent.tool.bad_command",
        "agent.tool.command_not_found",
        "agent.tool.not_text",
        "agent.tool.not_directory",
        "agent.tool.no_change",
        "agent.tool.too_large",
        "agent.tool.too_many_files",
        "agent.access.not_found",
        "agent.access.not_directory",
        "agent.access.not_file",
        "agent.access.parent",
        "agent.access.absolute",
        "agent.access.invalid",
        "agent.change.conflict",
        "agent.run.too_many_calls",
    }
)
HINT_FIX = "(Fix the call and try again.)"
HINT_STOP = (
    "(Do not repeat this call: it is not allowed or was refused. Choose another way, "
    "or call finish with success=false and explain why.)"
)
HINT_DENIED = "(The user refused this action. Follow the reason they gave.)"
# 6.14: выключенный пользователем инструмент можно попросить включить, но не обойти.
HINT_REQUEST = (
    "(This tool is turned off by the user. If the task really needs it, call request_tool "
    "with its name and the reason; otherwise choose another way.)"
)


@dataclass(frozen=True)
class ToolSpec:
    """Описание инструмента; `parameters` — JSON Schema объекта аргументов."""

    name: str
    description: str
    parameters: dict[str, Any]
    permission: ToolPermission
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
    (по умолчанию заблокированы). `confirm_*` — спрашивать подтверждение перед
    записью, командой и разрушительной командой (6.4); `max_series` — сколько
    следующих действий можно разрешить одним подтверждением. `command_timeout` —
    предел одной команды в секундах, None — без ограничения (тогда команду
    останавливает только пользователь). Меняет только пользователь.
    """

    disabled: frozenset[str] = frozenset()
    allow_destructive_commands: bool = False
    confirm_writes: bool = True
    confirm_commands: bool = True
    confirm_controls: bool = True
    confirm_destructive: bool = True
    max_series: int = 20
    command_timeout: float | None = 600.0
    # 6.15: «Авто мод» на экране агента — действия без подтверждений. Выключенные инструменты,
    # блок разрушительных команд и контрольные точки он не снимает.
    auto_mode: bool = False

    def effective(self) -> "ToolAccess":
        """Доступ с учётом «Авто мода»: подтверждения выключены, остальное как задано."""
        if not self.auto_mode:
            return self
        return replace(
            self,
            confirm_writes=False,
            confirm_commands=False,
            confirm_controls=False,
            confirm_destructive=False,
        )


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
        if code in ("agent.approval.denied", "agent.tool.request_denied"):
            hint = HINT_DENIED
        elif code == "agent.tool.disabled":
            hint = HINT_REQUEST
        elif code in CORRECTABLE:
            hint = HINT_FIX
        else:
            hint = HINT_STOP
        return cls(False, f"ERROR {code}{detail}\n{hint}", {"params": params}, error=code)

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

    def finish(
        self, call_id: str, result: str, duration: float, permission: str | None = None
    ) -> None: ...


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

    def __init__(self, runs: Any, run_id: str, *, private: bool = False) -> None:
        self.runs, self.run_id, self.private = runs, run_id, private

    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
        return str(
            self.runs.add_tool_call(
                self.run_id,
                tool,
                private_payload({"args": args})["args"] if self.private else args,
                permission,
            ).id
        )

    def finish(
        self, call_id: str, result: str, duration: float, permission: str | None = None
    ) -> None:
        if self.private:
            result = json.dumps(private_payload(json.loads(result)), ensure_ascii=False)
        self.runs.finish_tool_call(call_id, result, duration, permission=permission)


def private_payload(value: Any) -> Any:
    """UI-задача: сохранять метаданные, не экран/выделение/введённый текст."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "text",
                "content",
                "diff",
                "summary",
                "image",
                "structure",
                "value",
                "plan",
                "title",
                "command",
                "reason",
            }:
                clean[key] = "[private UI content]"
            elif key == "args":
                clean[key] = (
                    {
                        k: v
                        for k, v in item.items()
                        if k in {"application", "window_id", "snapshot_id", "element_id", "action"}
                    }
                    if isinstance(item, dict)
                    else {}
                )
            elif key == "result" and isinstance(item, str):
                clean[key] = "[private UI content]"
            else:
                clean[key] = private_payload(item)
        return clean
    if isinstance(value, list | tuple):
        return [private_payload(item) for item in value]
    return value


def _journal_result(result: ToolResult) -> str:
    summary = {
        "ok": result.ok,
        "error": result.error,
        "truncated": result.truncated or len(result.content) > MAX_RESULT_CHARS,
        "chars": len(result.content),
        "change_id": result.data.get("change_id"),
        "content": result.content[:MAX_JOURNAL_CHARS],
    }
    return json.dumps(summary, ensure_ascii=False)


# --- исполнение -----------------------------------------------------------------------

Handler = Callable[[dict[str, Any]], ToolResult]


class ToolRunner:
    """Реестр инструментов и единственный путь их исполнения."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        access: ToolAccess | None = None,
        gate: Any = None,
    ) -> None:
        """`gate` — `ApprovalGate` (6.4); без него write/command только возвращают предложение."""
        self._tools: dict[str, tuple[ToolSpec, Handler]] = {}
        self.workspace = workspace
        self.access = access or ToolAccess()
        self.gate = gate
        self.private = False
        self._authorizers: dict[str, Callable[[], None]] = {}
        self.model_context: Callable[[threading.Event], AbstractContextManager[threading.Event]] = (
            nullcontext
        )

    def authorize_with(self, permission: str, check: Callable[[], None]) -> None:
        """Только доверенная сборка tools: независимый грант вне рабочей папки."""
        self._authorizers[permission] = check

    def _require(self, spec: ToolSpec) -> None:
        if spec.permission in PERMISSIONS:
            if self.workspace is not None:
                self.workspace.require(spec.permission)
        elif spec.permission in self._authorizers:
            try:
                self._authorizers[spec.permission]()
            except PermissionError:
                raise AccessDenied("agent.access.permission") from None
        else:
            raise AccessDenied("agent.access.permission")

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
        try:
            self._require(spec)
            return True
        except AccessDenied:
            return False

    def status(self, spec: ToolSpec) -> str:
        """6.14: enabled | disabled (выключен пользователем) | unavailable (нет права)."""
        if spec.name in self.access.disabled:
            return "disabled"
        return "enabled" if self.is_available(spec) else "unavailable"

    def enable_for_run(self, name: str) -> None:
        """6.14: пользователь разрешил выключенный инструмент — только на эту задачу; настройки
        не меняются (снимать ограничения в настройках может только пользователь, не модель)."""
        self.access = replace(self.access, disabled=self.access.disabled - {name})

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
        try:
            self._require(spec)
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
        decided: str | None = None
        proposal = result.data.get("proposal")
        if result.ok and spec.effect != "read" and self.gate is not None and proposal:
            try:
                result, decided = self.gate.handle(spec, proposal)
            except Exception as error:  # noqa: BLE001
                result = ToolResult.failure("agent.tool.failed", error=type(error).__name__)
        journal.finish(call_id, _journal_result(result), time.monotonic() - started, decided)
        if result.error == "agent.approval.stopped":
            raise StopRequested
        return result

    @staticmethod
    def _reject(journal: Journal, name: str, args: Any, result: ToolResult) -> ToolResult:
        """Некорректный вызов не исполняется, но остаётся в журнале как отклонённый."""
        shown = redact(args) if isinstance(args, dict) else {"raw": repr(args)[:200]}
        call_id = journal.begin(str(name)[:100], shown, "denied")
        journal.finish(call_id, _journal_result(result), 0.0)
        return result
