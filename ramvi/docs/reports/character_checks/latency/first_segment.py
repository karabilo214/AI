"""9.23: через сколько персонаж получает первый готовый сегмент ответа от облачной модели.

Настоящий модуль Character на Scoped Services сервиса (как в экзамене 7.11), профиль
«дружелюбный», три короткие реплики Claude; `first_segment_ms` — из задания модуля.

usage: RAMVI_TOKENIZER=<Qwen3-8B> first_segment.py <файл ключей> <отчёт.json>
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "cloud_checks"))
import exam7_11 as exam  # noqa: E402  (сервис, ключи и модуль Character — оттуда)
from ramvi_service.app import build  # noqa: E402

TURNS = [
    "Привет! Как дела?",
    "Расскажи коротко, чем ты можешь помочь.",
    "Посоветуй, чем заняться вечером.",
]


def main() -> None:
    keys = exam.read_keys(Path(sys.argv[1]))
    data = Path(tempfile.mkdtemp(prefix="ramvi-first-"))
    server, engines = build(data, exam.TOKEN, voice_options={"secrets": exam.Secrets()})
    server.start()
    client = exam.Client(server.port)
    rows = []
    try:
        client.ok(
            "POST",
            "/cloud/connections",
            {"kind": "anthropic", "name": "claude", "key": keys["anthropic"]},
        )
        client.ok(
            "POST",
            "/cloud/connections/claude/models",
            {"model": "claude-sonnet-5-5", "enabled": True},
        )
        exam.CHARACTER_TURNS[:] = [(f"t{i}", t) for i, t in enumerate(TURNS)]
        from ramvi_character.module import CharacterModule

        original = CharacterModule.dialogue

        def spy(self, operation, arguments=None):  # type: ignore[no-untyped-def]
            result = original(self, operation, arguments)
            if operation == "poll" and isinstance(result, dict) and result.get("job"):
                rows.append(result["job"].get("first_segment_ms"))
            return result

        CharacterModule.dialogue = spy  # type: ignore[method-assign]
        replies = exam.character(server, "cloud:claude:claude-sonnet-5-5", data / "char")
    finally:
        server.close()
        engines.stop()
    report = {
        "first_segment_ms": rows,
        "turns": [{k: r[k] for k in ("turn", "status", "fallback", "seconds")} for r in replies],
    }
    Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
