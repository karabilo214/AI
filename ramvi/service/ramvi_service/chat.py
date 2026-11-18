"""3.4 (UI-02): чат поверх общего доступа к модели и хранилища 3.2.

Чат — обычный владелец `SharedLLM` (как модули в 2.11): очередь, отмена и смена
модели общие. Ответ идёт потоком событий; остановленный ответ сохраняется с
`finish="cancelled"`, thinking — отдельно от текста. Пользовательских строк здесь
нет: ошибки — ключи сообщений с параметрами, перевод делает интерфейс (3.14).
"""

import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from ramvi_storage.database import Database
from ramvi_storage.repositories import Chat, Chats, Message, Models, NotFound

from ramvi_service.contracts.modules import Owner
from ramvi_service.engines import MODEL_FIELD, EngineError
from ramvi_service.resources import ResourceBusy, SharedLLM

TITLE_CHARS = 60


class ChatError(RuntimeError):
    """Ошибка с ключом сообщения для интерфейса."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code, self.params = code, params


@dataclass(frozen=True)
class Limits:
    """Лимиты сообщения и перетаскиваемых текстовых файлов (UI проверяет те же)."""

    max_text_chars: int = 32_000
    max_files: int = 5
    max_file_bytes: int = 128 * 1024
    max_total_bytes: int = 256 * 1024
    max_name_chars: int = 255


def check_attachments(
    files: list[dict[str, Any]], limits: Limits | None = None
) -> list[dict[str, Any]]:
    limits = limits or Limits()
    if not isinstance(files, list):
        raise ChatError("chat.attachment_invalid")
    if len(files) > limits.max_files:
        raise ChatError("chat.too_many_files", limit=limits.max_files)
    result, total = [], 0
    for item in files:
        if not isinstance(item, dict) or set(item) - {"name", "text"}:
            raise ChatError("chat.attachment_invalid")
        name, body = item.get("name"), item.get("text")
        if not isinstance(name, str) or not isinstance(body, str):
            raise ChatError("chat.attachment_invalid")
        name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or len(name) > limits.max_name_chars:
            raise ChatError("chat.attachment_invalid")
        if "\x00" in body:
            raise ChatError("chat.attachment_not_text", name=name)
        size = len(body.encode("utf-8"))
        if size > limits.max_file_bytes:
            raise ChatError("chat.file_too_large", name=name, limit=limits.max_file_bytes)
        total += size
        if total > limits.max_total_bytes:
            raise ChatError("chat.files_too_large", limit=limits.max_total_bytes)
        result.append({"name": name, "bytes": size, "text": body})
    return result


class Generation:
    """Поток событий ответа. `close()` освобождает чат, даже если чтение не начиналось."""

    def __init__(self, events: Iterator[dict[str, Any]], release: Callable[[], None]) -> None:
        self._events, self._release = events, release
        self._started = False

    def __iter__(self) -> "Generation":
        return self

    def __next__(self) -> dict[str, Any]:
        self._started = True
        return next(self._events)

    def close(self) -> None:
        if not self._started:
            self._release()
        close = getattr(self._events, "close", None)
        if close is not None:
            close()


def message_json(message: Message) -> dict[str, Any]:
    return asdict(message)


def chat_json(chat: Chat) -> dict[str, Any]:
    return asdict(chat)


class ChatService:
    def __init__(
        self,
        db: Database,
        llm: SharedLLM,
        principal: str = "local_user",
        *,
        limits: Limits | None = None,
        default_model: Callable[[], str | None] | None = None,
        request_timeout: float = 600.0,
    ) -> None:
        self.chats_repo, self.models_repo = Chats(db), Models(db)
        self.llm, self.principal = llm, principal
        self._owners: dict[str, Owner] = {}
        self.limits = limits or Limits()
        self._default_model = default_model
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self._active: dict[str, threading.Event] = {}

    # --- список и настройки ------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "architecture": m.architecture,
                "quant": m.quant,
                "bytes": m.bytes,
                "capabilities": m.capabilities,
            }
            for m in self.models_repo.all()
        ]

    def list_chats(self) -> list[dict[str, Any]]:
        return [
            dict(chat_json(c), generating=c.id in self._active) for c in self.chats_repo.recent()
        ]

    def create(self, model_id: str | None = None) -> dict[str, Any]:
        model = model_id if model_id is not None else self._fallback_model()
        if model is not None:
            self._require_model(model)
        return chat_json(self.chats_repo.create("", model))

    def get(self, chat_id: str) -> Chat:
        try:
            return self.chats_repo.get(chat_id)
        except NotFound:
            raise ChatError("chat.not_found", chat_id=chat_id) from None

    def messages(self, chat_id: str) -> list[dict[str, Any]]:
        self.get(chat_id)
        return [message_json(m) for m in self.chats_repo.messages(chat_id)]

    def set_model(self, chat_id: str, model_id: str) -> dict[str, Any]:
        self.get(chat_id)
        self._require_model(model_id)
        return chat_json(self.chats_repo.update(chat_id, model_id=model_id))

    def rename(self, chat_id: str, title: str) -> dict[str, Any]:
        self.get(chat_id)
        title = " ".join(title.split())[:TITLE_CHARS]
        return chat_json(self.chats_repo.update(chat_id, title=title))

    def clear(self, chat_id: str) -> int:
        self._idle(chat_id)
        return self.chats_repo.clear(chat_id)

    def delete(self, chat_id: str) -> None:
        self._idle(chat_id)
        self.chats_repo.delete(chat_id)
        with self._lock:
            owner = self._owners.pop(chat_id, None)
        if owner is not None:
            self.llm.release_owner(owner)
            self.llm.authority.revoke(owner)

    def stop_all(self) -> int:
        """Остановить ответы во всех чатах (перед «Выгрузить модель», 3.17)."""
        with self._lock:
            active = list(self._active.values())
        for cancel in active:
            cancel.set()
        return len(active)

    def has_model(self, model_id: str) -> bool:
        try:
            self._require_model(model_id)
        except ChatError:
            return False
        return True

    def stop(self, chat_id: str) -> bool:
        """Остановить текущий ответ; False — нечего останавливать."""
        self.get(chat_id)
        with self._lock:
            cancel = self._active.get(chat_id)
        if cancel is None:
            return False
        cancel.set()
        return True

    # --- генерация -----------------------------------------------------------------

    def send(
        self, chat_id: str, text: str, attachments: list[dict[str, Any]] | None = None
    ) -> Generation:
        """Сохранить сообщение пользователя и выдать ответ потоком событий.

        Проверки (лимиты, модель, занятость) выполняются сразу, до первого события.
        """
        chat = self.get(chat_id)
        files = self.check_attachments(attachments or [])
        if not isinstance(text, str) or len(text) > self.limits.max_text_chars:
            raise ChatError("chat.text_too_long", limit=self.limits.max_text_chars)
        if not text.strip() and not files:
            raise ChatError("chat.empty_message")
        model_id = self._chat_model(chat)
        cancel = self._claim(chat_id)
        try:
            user = self.chats_repo.add_message(chat_id, "user", text, attachments=files or None)
            if not chat.title:
                first = text.strip().splitlines()[0] if text.strip() else files[0]["name"]
                self.chats_repo.update(chat_id, title=" ".join(first.split())[:TITLE_CHARS])
        except BaseException:
            self._release(chat_id, cancel)
            raise
        return Generation(
            self._run(chat_id, model_id, cancel, {"type": "user", "message": message_json(user)}),
            lambda: self._release(chat_id, cancel),
        )

    def regenerate(self, chat_id: str) -> Generation:
        """Удалить последний ответ и получить новый на то же сообщение пользователя."""
        chat = self.get(chat_id)
        model_id = self._chat_model(chat)
        cancel = self._claim(chat_id)
        try:
            history = self.chats_repo.messages(chat_id)
            last_user = max((i for i, m in enumerate(history) if m.role == "user"), default=None)
            if last_user is None:
                raise ChatError("chat.nothing_to_regenerate")
            removed: list[str] = []
            if last_user + 1 < len(history):
                removed = [m.id for m in history[last_user + 1 :]]
                self.chats_repo.delete_messages_from(chat_id, removed[0])
        except BaseException:
            self._release(chat_id, cancel)
            raise
        return Generation(
            self._run(chat_id, model_id, cancel, {"type": "regenerate", "removed": removed}),
            lambda: self._release(chat_id, cancel),
        )

    def _run(
        self, chat_id: str, model_id: str, cancel: threading.Event, first: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        # Жадный выбор: ответы повторяемы, а угадывание предсказателем точное только при нём.
        request = {"messages": self._prompt(chat_id), "temperature": 0, MODEL_FIELD: model_id}
        text: list[str] = []
        reasoning: list[str] = []
        finish: str | None = None
        usage: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        saved: Message | None = None
        stream = self.llm.stream(
            self._owner(chat_id), request, timeout=self.request_timeout, cancel=cancel
        )
        try:
            yield first
            try:
                for chunk in stream:
                    kind = chunk.get("type")
                    if kind == "text":
                        text.append(chunk["text"])
                    elif kind == "reasoning":
                        reasoning.append(chunk["text"])
                    elif kind == "end":
                        finish = "length" if chunk.get("finish_reason") == "length" else "stop"
                        usage = chunk.get("usage")
                        continue
                    elif kind != "status":
                        continue  # tool_calls в чате не используются
                    yield chunk
            except InterruptedError:
                finish = "cancelled"
            except EngineError as exc:
                finish, error = "error", {"code": exc.code, "params": exc.params}
            except ChatError as exc:
                finish, error = "error", {"code": exc.code, "params": exc.params}
            except TimeoutError:
                finish, error = "error", {"code": "chat.busy_timeout", "params": {}}
            except ResourceBusy:
                finish, error = "error", {"code": "api.busy", "params": {}}
            except Exception as exc:
                finish = "error"
                error = {"code": "chat.generation_failed", "params": {"detail": str(exc)}}
            saved = self._save(chat_id, text, reasoning, finish, usage)
            yield {
                "type": "done",
                "finish": finish,
                "message": None if saved is None else message_json(saved),
                "error": error,
            }
        finally:
            # Клиент ушёл посреди ответа (GeneratorExit): остановить генерацию
            # и сохранить то, что успело прийти, как остановленный ответ.
            if finish is None:
                cancel.set()
            stream.close()
            if finish is None and saved is None:
                self._save(chat_id, text, reasoning, "cancelled", None)
            self._release(chat_id, cancel)

    def _save(
        self,
        chat_id: str,
        text: list[str],
        reasoning: list[str],
        finish: str | None,
        usage: dict[str, Any] | None,
    ) -> Message | None:
        content, thought = "".join(text), "".join(reasoning)
        if finish in ("cancelled", "error") and not content and not thought:
            return None  # пустой обрыв не засоряет историю
        try:
            return self.chats_repo.add_message(
                chat_id, "assistant", content, reasoning=thought, finish=finish, token_usage=usage
            )
        except NotFound:
            return None  # чат удалили во время ответа

    # --- вспомогательное -------------------------------------------------------------

    def check_attachments(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return check_attachments(files, self.limits)

    def _prompt(self, chat_id: str) -> list[dict[str, str]]:
        prompt = []
        for m in self.chats_repo.messages(chat_id):
            if m.role == "assistant" and not m.content:
                continue
            content = m.content
            for f in m.attachments or []:
                content += f'\n\n<file name="{f["name"]}">\n{f["text"]}\n</file>'
            prompt.append({"role": m.role, "content": content.strip()})
        return prompt

    def _chat_model(self, chat: Chat) -> str:
        model_id = chat.model_id or self._fallback_model()
        if model_id is None:
            raise ChatError("chat.no_model")
        self._require_model(model_id)
        if chat.model_id is None:
            self.chats_repo.update(chat.id, model_id=model_id)
        return model_id

    def _fallback_model(self) -> str | None:
        if self._default_model is not None:
            chosen = self._default_model()
            if chosen is not None:
                return chosen
        for chat in self.chats_repo.recent(limit=20):
            if chat.model_id is not None:
                return chat.model_id
        models = self.models_repo.all()
        return models[0].id if models else None

    def _require_model(self, model_id: str) -> None:
        try:
            self.models_repo.get(model_id)
        except NotFound:
            raise ChatError("chat.model_not_found", model_id=model_id) from None

    def _owner(self, chat_id: str) -> Owner:
        """Каждый чат — отдельная сессия владельца `chat`: запросы разных чатов
        встают в общую справедливую очередь модели, а не упираются в квоту."""
        with self._lock:
            owner = self._owners.get(chat_id)
            if owner is None:
                owner = self.llm.authority.issue("chat", self.principal, chat_id)
                self.llm.retain(owner)
                self._owners[chat_id] = owner
            return owner

    def _claim(self, chat_id: str) -> threading.Event:
        with self._lock:
            if chat_id in self._active:
                raise ChatError("chat.busy")
            cancel = threading.Event()
            self._active[chat_id] = cancel
            return cancel

    def _release(self, chat_id: str, cancel: threading.Event) -> None:
        with self._lock:
            if self._active.get(chat_id) is cancel:
                del self._active[chat_id]

    def _idle(self, chat_id: str) -> None:
        self.get(chat_id)
        with self._lock:
            if chat_id in self._active:
                raise ChatError("chat.busy")
