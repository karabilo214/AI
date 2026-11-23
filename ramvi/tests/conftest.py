"""Общие настройки тестов."""

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _reaction_controller_off(request: pytest.FixtureRequest) -> Iterator[None]:
    """9.30: контроллер реакций делает отдельный запрос к модели перед ответом. Тесты другого
    поведения персонажа рассчитаны на один запрос за ход — для них он выключен; его проверяют
    тесты с меткой brain."""
    if request.node.get_closest_marker("brain") is not None:
        yield
        return
    try:
        import ramvi_character.module as module
    except ImportError:
        yield
        return
    previous = module.DEFAULT_BRAIN
    module.DEFAULT_BRAIN = "off"
    try:
        yield
    finally:
        module.DEFAULT_BRAIN = previous
