"""HTTP correlation создаёт обработчик; модель не выбирает owner envelope."""

import uuid
from typing import Any

from ramvi_engine.protocol import EventContext


def generation_context(handler: Any) -> EventContext:
    return EventContext(
        getattr(handler, "_request_id", None) or uuid.uuid4().hex,
        module_id="engine",
        response_id=uuid.uuid4().hex,
    )
