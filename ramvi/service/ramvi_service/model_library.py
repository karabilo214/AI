"""4.11: действия с установленными моделями для экрана «Модели» и настроек.

- Модель по умолчанию — модель новых чатов (тот же выбор, что запоминает чат, 3.6).
- Удалить — забыть модель (её чаты остаются без модели, замеры 4.7 удаляются). Файл
  удаляется только по явному выбору и только у модели, которую скачал сам RAMVI
  из каталога в папку моделей. Подключённый на месте файл пользователя
  (импорт, папка Ollama) не удаляется никогда.
- Загруженная сейчас модель сначала выгружается — так же, как кнопкой «Выгрузить
  модель» (ответы останавливаются, память освобождается).
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ramvi_storage.repositories import Model, Models, NotFound

from ramvi_service.chat import ChatError
from ramvi_service.engines import resolve_model


class ModelLibrary:
    def __init__(
        self,
        models: Models,
        remember: Callable[[str], None],
        models_folder: Callable[[], Path],
        app_dir: Path | None,
        loaded: Callable[[], str | None],
        unload: Callable[[], Any],
    ) -> None:
        self.models, self.remember, self.models_folder = models, remember, models_folder
        self.app_dir, self.loaded, self.unload = app_dir, loaded, unload

    def _get(self, model_id: str) -> Model:
        try:
            return self.models.get(model_id)
        except NotFound:
            raise ChatError("chat.model_not_found", model_id=model_id) from None

    def set_default(self, model_id: str) -> dict[str, Any]:
        self._get(model_id)
        self.remember(model_id)
        return {"default_model": model_id}

    def owns_file(self, model: Model) -> bool:
        """Файл скачан RAMVI из каталога и лежит в папке моделей — его можно удалить."""
        if model.capabilities.get("imported", True) or not model.capabilities.get("catalog_id"):
            return False
        path = Path(resolve_model(model, self.app_dir).path).resolve()
        folder = self.models_folder().resolve()
        return folder in path.parents

    def delete(self, model_id: str, delete_file: bool) -> dict[str, Any]:
        model = self._get(model_id)
        if delete_file and not self.owns_file(model):
            raise ChatError("models.file_not_ours", model_id=model_id)
        if self.loaded() == model_id:
            self.unload()
        path = resolve_model(model, self.app_dir).path
        self.models.delete(model_id)
        removed = False
        if delete_file:
            try:
                os.remove(path)
                removed = True
            except FileNotFoundError:
                pass
        return {"deleted": True, "file_deleted": removed}
