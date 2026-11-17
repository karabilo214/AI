"""4.8 AUTO-06: невыполнимый план — сразу понятная ошибка; подсказки при нехватке памяти."""

import time
from pathlib import Path
from typing import Any

from ramvi_service import compat
from ramvi_service.app import build
from ramvi_service.guard import Guard
from ramvi_service.hardware import GPUInfo
from ramvi_service.task_profile import TaskProfile
from ramvi_storage.repositories import Model, NotFound
from test_chat import ENGINE, TOKEN, Client
from test_compat import LAPTOP
from test_control_run import add_gguf


class StubFit:
    """Оценка по таблице: (модель, контекст) → (статус, помещается ли сейчас, размер)."""

    def __init__(self, table: dict[tuple[str, int], tuple[str, bool]], sizes: dict[str, Any]):
        self.table, self.sizes = table, sizes

    def for_file(
        self, path: Any, name: str, model_id: Any = None, task: Any = None
    ) -> dict[str, Any]:
        context = (task or TaskProfile()).effective_context
        status, now = self.table.get((name, context), (compat.NO_VRAM, False))
        return {
            "status": status,
            "now": {"fits": now} if status in compat.RUNNABLE else None,
            "reasons": [{"code": "compat.reason.vram", "params": {}}],
            "model": {"size_b": self.sizes.get(name, 0)},
        }


class StubModels:
    def __init__(self, *ids: str) -> None:
        self.items = {i: Model(i, "local", "qwen3", f"/m/{i}.gguf", 0.0) for i in ids}

    def get(self, model_id: str) -> Model:
        if model_id not in self.items:
            raise NotFound(model_id)
        return self.items[model_id]

    def all(self) -> list[Model]:
        return list(self.items.values())


def guard(table: dict[Any, Any], sizes: dict[str, Any], profile: TaskProfile, *ids: str) -> Guard:
    return Guard(StubFit(table, sizes), StubModels(*ids), None, lambda: profile)  # type: ignore[arg-type]


def actions(found: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [(s["action"], s["params"]) for s in found]


def test_suggests_largest_context_that_fits_now() -> None:
    table = {("14b", 4096): (compat.SLOW, True), ("14b", 2048): (compat.SLOW, True)}
    found = guard(table, {}, TaskProfile("long_context"), "14b").suggestions("14b")
    reduce = [a for a in actions(found) if a[0] == "reduce_context"]
    assert reduce == [("reduce_context", {"context": 4096})]  # одна, наибольшая: не 2048
    assert ("economy_mode", {"mode": "economy"}) in actions(found)


def test_suggests_smaller_installed_models_that_fit_now() -> None:
    table = {
        ("14b", 2048): (compat.NO_VRAM, False),
        ("8b", 2048): (compat.RECOMMENDED, True),
        ("4b", 2048): (compat.RECOMMENDED, True),
        ("1b", 2048): (compat.RECOMMENDED, True),
        ("busy", 2048): (compat.SLOW, False),
    }
    sizes = {"14b": 14, "8b": 8, "4b": 4, "1b": 1, "busy": 6}
    found = guard(table, sizes, TaskProfile(), "14b", "8b", "4b", "1b", "busy").suggestions("14b")
    assert actions(found) == [  # две самые крупные из помещающихся сейчас
        ("smaller_model", {"model_id": "8b"}),
        ("smaller_model", {"model_id": "4b"}),
    ]


def test_suggests_closing_programs_when_pc_could_but_memory_is_busy() -> None:
    table = {("14b", 2048): (compat.SLOW, False)}
    found = guard(table, {}, TaskProfile(), "14b").suggestions("14b")
    assert actions(found) == [("close_programs", {})]
    assert found[0]["message"]["code"] == "guard.close_programs"


def test_no_made_up_suggestions() -> None:
    assert guard({}, {}, TaskProfile(), "14b").suggestions("14b") == []
    assert guard({}, {}, TaskProfile(), "14b").suggestions("missing") == []


def test_preflight_blocks_only_what_cannot_fit_at_all() -> None:
    table = {
        ("big", 2048): (compat.NO_VRAM, False),
        ("slow", 2048): (compat.SLOW, False),
        ("mac", 2048): (compat.UNSUPPORTED, False),
    }
    g = guard(table, {}, TaskProfile(), "big", "slow", "mac")
    assert g.preflight(Model("big", "local", "qwen3", "/m/big.gguf", 0.0)) == (
        "engine.no_memory",
        "compat.reason.vram",
    )
    # «Медленно» и «не поддерживается» решает движок: оценка не должна блокировать лишнее.
    assert g.preflight(Model("slow", "local", "qwen3", "/m/slow.gguf", 0.0)) is None
    assert g.preflight(Model("mac", "local", "qwen3", "/m/mac.gguf", 0.0)) is None


def test_infeasible_model_fails_fast_and_keeps_the_working_one(tmp_path: Path) -> None:
    add_gguf(tmp_path, "model-plain", "model-other")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    tiny_gpu = GPUInfo("Tiny NVIDIA", "NVIDIA", 8, 8, "1")
    cramped = LAPTOP.__class__(**{**LAPTOP.__dict__, "gpus": (tiny_gpu,)})
    assert server.fit is not None
    server.start()
    try:
        client = Client(server.port)
        client.ok("POST", "/engine/load", {"model_id": "model-plain"})
        deadline = time.monotonic() + 20
        while engines.status()["state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.05)
        server.fit.hardware = lambda: cramped  # теперь карта на 8 МБ: ничего не поместится
        server.fit._hw = None
        started = time.monotonic()
        client.ok("POST", "/engine/load", {"model_id": "model-other"})
        while time.monotonic() - started < 10:
            status = client.ok("GET", "/status")["engine"]
            if status["loading"] is None and (status["problem"] or {}).get("code"):
                break
            time.sleep(0.02)
        assert time.monotonic() - started < 5  # без запуска движка и ожидания
        assert status["problem"]["code"] == "engine.no_memory"
        assert status["problem"]["params"]["model_id"] == "model-other"
        assert "suggestions" in status["problem"]
        assert engines.model_id == "model-plain"  # рабочая модель не выгружена
    finally:
        server.close()
        engines.stop()
