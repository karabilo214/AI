"""3.10: предсказатель (draft) подбирается автоматически, чат просит жадный выбор."""

from pathlib import Path

from ramvi_service.app import build
from ramvi_service.engines import EngineManager, default_engine
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client

GB = 10**9


def add(data: Path, *models: tuple[str, str, int]) -> None:
    with open_core(data) as db:
        for i, (model_id, architecture, size) in enumerate(models):
            Models(db).upsert(
                Model(
                    model_id, "local", architecture, f"models/{model_id}.gguf", float(i), bytes=size
                )
            )


def manager(data: Path) -> EngineManager:
    return EngineManager(Models(open_core(data)), data / "run", model_root=data)


def test_smallest_dense_qwen3_at_least_8x_smaller_is_chosen(tmp_path: Path) -> None:
    add(
        tmp_path,
        ("qwen3-14b", "qwen3", 9 * GB),
        ("qwen3-8b", "qwen3", 5 * GB),
        ("qwen3-0.6b-fp16", "qwen3", int(1.4 * GB)),
        ("qwen3-0.6b", "qwen3", int(0.52 * GB)),
        ("qwen3-30b", "qwen3moe", 18 * GB),
    )
    m = manager(tmp_path)
    get = m.models.get

    assert m.draft_for(get("qwen3-14b")).id == "qwen3-0.6b"  # type: ignore[union-attr]
    assert m.draft_for(get("qwen3-8b")).id == "qwen3-0.6b"  # type: ignore[union-attr]
    assert m.draft_for(get("qwen3-0.6b-fp16")) is None  # 0.52 ГБ меньше всего в 2,7 раза
    assert m.draft_for(get("qwen3-0.6b")) is None  # себе не предсказатель
    assert m.draft_for(get("qwen3-30b")) is None  # MoE: предсказатель не подбирается
    draft = m.draft_for(get("qwen3-14b"))
    assert draft is not None and Path(draft.path).is_absolute()  # от папки приложения


def test_default_engine_passes_draft(tmp_path: Path) -> None:
    main = Model("m", "local", "qwen3", str(tmp_path / "m.gguf"), 0.0)
    small = Model("s", "local", "qwen3", str(tmp_path / "s.gguf"), 0.0)

    assert default_engine(main).config.draft is None  # type: ignore[attr-defined]
    engine = default_engine(main, small)
    args = engine._arguments()  # type: ignore[attr-defined]
    assert args[args.index("--draft") + 1] == str(tmp_path / "s.gguf")


def test_chat_uses_draft_and_greedy_requests(tmp_path: Path) -> None:
    add(tmp_path, ("model-a", "qwen3", 9 * GB), ("model-tiny", "qwen3", GB // 2))
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        client = Client(server.port)
        chat = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
        # Поддельный движок отвечает 400 на запрос без temperature=0.
        done = client.events(f"/chats/{chat}/send", {"text": "hi"})[-1]
        assert done["finish"] == "stop", done
        status = client.ok("GET", "/status")["engine"]
        assert status["model_id"] == "model-a" and status["draft_model_id"] == "model-tiny"
    finally:
        server.close()
        engines.stop()
