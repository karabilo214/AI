"""Формат промпта на локальной 8B: разбирается ли структурный ответ (без запасного текста)."""

import json
import sys
import time
from pathlib import Path

APP = Path(sys.argv[1]).resolve()
sys.path[:0] = [
    str(APP / "ramvi" / p) for p in ("service", "storage", "engine", "character", "tests")
]
from ramvi_service.app import build  # noqa: E402

TOKEN = "t" * 43
LINES = [
    "Здорово, чем занят?",
    "Скучно, расскажи что-нибудь",
    "Я вчера пробежал два километра, я теперь спортсмен",
    "Ты чё такой тормозной, а?",
]


def main() -> None:
    import http.client

    out, label = Path(sys.argv[2]), sys.argv[3]
    server, engines = build(APP / "data", TOKEN, app_dir=APP)
    server.start()

    def ok(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=900)
        c.request(
            method,
            "/ui/v1" + path,
            json.dumps(body or {}) if method != "GET" else None,
            {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
        )
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        assert r.status == 200, (path, r.status, data)
        return data

    rep = {"label": label, "turns": []}
    try:
        models = [m["id"] for m in ok("GET", "/models")["models"]]
        model = next(m for m in models if "8b" in m.lower() and "f16" not in m.lower())
        rep["model"], rep["models"] = model, models
        assert ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        ok("POST", path, {"operation": "profile", "arguments": {"id": "roaster"}})
        ok(
            "POST",
            path,
            {"operation": "boundary", "arguments": {"name": "profanity", "allowed": True}},
        )
        for text in LINES:
            t0 = time.monotonic()
            ok("POST", path, {"operation": "start", "arguments": {"text": text, "model": model}})
            while True:
                job = ok("POST", path, {"operation": "poll"})["job"]
                if job and job["state"] != "running" and job.get("settled", True):
                    break
                time.sleep(1)
            rep["turns"].append(
                {
                    "owner": text,
                    "reply": " ".join(
                        e["text"] for e in job["events"] if e.get("type") == "segment"
                    ),
                    "segments": sum(e.get("type") == "segment" for e in job["events"]),
                    "fallback": bool(job.get("fallback")),
                    "strategy": ((job.get("brain") or {}).get("decision") or {}).get("strategy"),
                    "error": job.get("error_detail"),
                    "seconds": round(time.monotonic() - t0, 1),
                }
            )
            out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        server.close()
        server.sections.host.close()
        engines.stop()
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


main()
