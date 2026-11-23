"""Одна реплика персонажу на облачной модели: профиль, уровень мата, фраза владельца.

usage: say_once.py <файл ключей> <токенизатор Qwen3> <модель OpenAI, например gpt-5.5 | claude:...> <профиль>
                   <мат 0-3, 0 — запрещён> <фраза> [повторов] [keep — повторы в одном диалоге]
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))
sys.path.insert(0, str(Path(__file__).parent))

from live_mac import Secrets, read_keys  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from roast_live import pick, say  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    os.environ["RAMVI_TOKENIZER"] = sys.argv[2]
    choice, profile, level, text = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
    repeats = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    keep = len(sys.argv) > 8 and sys.argv[8] == "keep"
    app = Path(tempfile.mkdtemp(prefix="ramvi-say-"))
    shutil.copytree(ROOT / "modules" / "character", app / "modules" / "character")
    server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
    server.start()
    client = Client(server.port)
    try:
        model = pick(client, keys, choice)
        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        client.ok("POST", path, {"operation": "profile", "arguments": {"id": profile}})
        allowed = {"name": "profanity", "allowed": level > 0}
        client.ok("POST", path, {"operation": "boundary", "arguments": allowed})
        if level:
            client.ok("POST", path, {"operation": "profanity_level", "arguments": {"level": level}})
        mode = "в одном диалоге" if keep else "с чистой историей"
        print(f"модель {model.rsplit(':', 1)[-1]}, профиль {profile}, мат {level}, {mode}")
        for number in range(repeats):
            # Без keep каждый повтор — заново, без истории: видно, разные ли ответы на одну фразу;
            # с keep — та же фраза подряд в одном диалоге.
            if not keep:
                client.ok("POST", path, {"operation": "history_clear", "arguments": {}})
            reply, seconds, error = say(client, path, text, model)
            print(f"{number + 1}. ({seconds} с) {reply or error}", flush=True)
    finally:
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        shutil.rmtree(app, ignore_errors=True)


if __name__ == "__main__":
    main()
