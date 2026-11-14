"""Манифест расширения и идентичность, выдаваемая доверенным Host."""

import re
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

MODULE_API_VERSION = "1.0"
ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def module_id(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("неверный module_id")
    return value


@dataclass(frozen=True)
class Manifest:
    module_id: str
    version: str
    entrypoint: str
    api_version: str = MODULE_API_VERSION
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    data_schema: int = 1
    ui: dict[str, Any] | None = None
    resource_limits: dict[str, int] = field(default_factory=dict)
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module_id(self.module_id)
        if not VERSION.fullmatch(self.version):
            raise ValueError("version должна быть major.minor.patch")
        if self.api_version != MODULE_API_VERSION:
            raise ValueError("несовместимая версия Module API")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", self.entrypoint):
            raise ValueError("entrypoint должен иметь вид package.module:factory")
        for dependency in self.required + self.optional:
            module_id(dependency)
        if self.module_id in self.required + self.optional:
            raise ValueError("модуль зависит от себя")
        if len(set(self.required + self.optional)) != len(self.required + self.optional):
            raise ValueError("повтор зависимости")
        if (
            isinstance(self.data_schema, bool)
            or not isinstance(self.data_schema, int)
            or self.data_schema < 1
        ):
            raise ValueError("неверная версия данных")
        for dependency, version in self.dependency_versions.items():
            if dependency not in self.required + self.optional or not VERSION.fullmatch(version):
                raise ValueError("неверная версия зависимости")
        for name, value in self.resource_limits.items():
            if not name or isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("неверная квота ресурса")


@dataclass(frozen=True)
class Owner:
    module_id: str
    principal_id: str
    session_id: str
    token: str = field(repr=False)


class Authority:
    """Идентичность берётся из Host/bearer token, не из модельного payload."""

    def __init__(self) -> None:
        self._owners: dict[str, Owner] = {}
        self._lock = threading.RLock()

    def issue(self, module: str, principal: str, session: str) -> Owner:
        module_id(module)
        if not principal or not session:
            raise ValueError("нужны principal и session")
        owner = Owner(module, principal, session, secrets.token_urlsafe(32))
        with self._lock:
            self._owners[owner.token] = owner
        return owner

    def resolve(self, token: str) -> Owner:
        with self._lock:
            if token not in self._owners:
                raise PermissionError("владелец отозван или не авторизован")
            return self._owners[token]

    def require(self, owner: Owner) -> None:
        if self.resolve(owner.token) != owner:
            raise PermissionError("подмена владельца")

    def revoke(self, owner: Owner) -> None:
        with self._lock:
            self._owners.pop(owner.token, None)


class Module(Protocol):
    def initialize(self, context: dict[str, Any]) -> Any: ...
    def activate(self) -> Any: ...
    def deactivate(self) -> Any: ...
    def suspend(self) -> Any: ...
    def resume(self) -> Any: ...
    def dispose(self) -> Any: ...


@dataclass(frozen=True)
class ProviderCapabilities:
    text: bool = True
    image_input: bool = False
    streaming: bool = False
    json_schema: bool = False
    tools: bool = False
    external: bool = False


class LLMProvider(Protocol):
    """Возможности провайдера независимы от архитектуры/GPU backend движка."""

    capabilities: ProviderCapabilities

    def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]: ...
    def cancel(self) -> None: ...
    def unload(self) -> None: ...
