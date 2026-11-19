"""8.7: без драйвера NVIDIA или со старым — понятное сообщение до запуска движка."""

from typing import Any

from ramvi_service import driver_gate
from ramvi_service.engines import EngineError
from ramvi_service.i18n import load_catalog, msg, render


def test_check_versions() -> None:
    assert driver_gate.check(lambda: None) == ("engine.no_nvidia_driver", "13.2")
    assert driver_gate.check(lambda: (13, 0)) == ("engine.driver_cuda_old", "13.0")
    assert driver_gate.check(lambda: (12, 9)) == ("engine.driver_cuda_old", "12.9")
    assert driver_gate.check(lambda: (13, 2)) is None
    assert driver_gate.check(lambda: (14, 0)) is None


def test_messages_tell_what_to_install() -> None:
    for lang in ("ru", "en"):
        catalog = load_catalog(lang)
        for code, detail in (
            ("engine.no_nvidia_driver", "13.2"),
            ("engine.driver_cuda_old", "12.4"),
        ):
            text = render(msg(code, detail=detail), catalog)
            assert "13.2" in text and "nvidia.com/drivers" in text and detail in text


def test_engine_refuses_before_start_with_driver_code(tmp_path: Any) -> None:
    from ramvi_service.app import build
    from test_chat import ENGINE, TOKEN, add_models

    add_models(tmp_path, "model-a")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    try:
        engines.preflight = lambda model: driver_gate.check(lambda: (12, 4))
        try:
            engines.ensure("model-a")
        except EngineError as error:
            assert error.code == "engine.driver_cuda_old" and error.params["detail"] == "12.4"
        else:
            raise AssertionError("движок не должен запускаться со старым драйвером")
        assert engines._supervisor is None  # движок не запускался
    finally:
        engines.stop()
