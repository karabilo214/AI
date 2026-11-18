"""Экзамены 7 и 8 ТЗ + проверка 7.1 официальными клиентами на настоящем движке.

Часть A (7.1): сервис без окна (--headless), модель грузится через API интерфейса,
ответы потоком через официальные клиенты openai, anthropic, ollama.
Часть B (экзамены 7 и 8): сервис в этом процессе с настоящим движком.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["USERPROFILE"]) / "Desktop" / "RAMVI"
PYTHON = BUNDLE / "runtime" / "python" / "python.exe"
OLLAMA = Path(os.environ["USERPROFILE"]) / ".ollama" / "models"
TOKENIZER = BUNDLE / "runtime" / "tokenizers" / "Qwen3-8B"
CODE = [HERE / "service", HERE / "storage", BUNDLE / "ramvi" / "engine"]
sys.path[:0] = [str(HERE / "clients"), *map(str, CODE)]
os.environ["RAMVI_TOKENIZER"] = str(TOKENIZER)

import anthropic  # noqa: E402
import ollama  # noqa: E402
import openai  # noqa: E402

TOKEN = "exam-ui-token-" + "x" * 24
RESULTS: dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
OUT = HERE / ("exam7_a_results.json" if "--only-a" in sys.argv else "exam7_8_results.json")
PROMPTS = [
    "Сколько будет 17*23? Ответь числом и объясни в двух предложениях.",
    "Name three planets of the solar system and one fact about each.",
    "Переведи на английский: «Утро было холодным, но солнечным».",
]
LONG = "Напиши подробный рассказ о маяке на берегу моря, не меньше двадцати предложений."


def save() -> None:
    OUT.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def log(*parts: Any) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def settings(app_dir: Path, data: Path, port: int) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "settings.json").write_text(
        json.dumps({"data_dir": str(data), "models_dir": str(OLLAMA), "api_port": port}),
        encoding="utf-8",
    )


def pick_model(models: list[dict[str, Any]]) -> str:
    ids = [m["id"] for m in models]
    RESULTS.setdefault("model_ids", ids)
    for model_id in ids:
        low = model_id.lower()
        if "8b" in low and "f16" not in low:
            return model_id
    raise SystemExit(f"qwen3 8b q4 not found among {ids}")


# --- часть A: без окна, официальные клиенты ---------------------------------------------


class Ui:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}/ui/v1"

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path,
            data,
            method=method,
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read())


def wait_ready(ui: Ui, limit: float = 900) -> dict[str, Any]:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        engine = ui.call("GET", "/status")["engine"]
        if engine.get("state") == "ready":
            return engine
        if engine.get("state") == "failed":
            raise RuntimeError(f"engine failed: {engine}")
        time.sleep(2)
    raise TimeoutError("engine did not load")


def official_clients(api: str, key: str, model: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    question = "Привет! Назови столицу Франции одним словом."
    client = openai.OpenAI(base_url=api + "/v1", api_key=key, max_retries=0, timeout=600)
    started, first, text = time.monotonic(), None, ""
    for chunk in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0,
        max_tokens=64,
        stream=True,
    ):
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            first = first or time.monotonic() - started
            text += delta
    out["openai_stream"] = {
        "text": text,
        "first_token_s": first,
        "total_s": time.monotonic() - started,
    }
    whole = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}], temperature=0, max_tokens=64
    )
    out["openai"] = {"text": whole.choices[0].message.content, "usage": whole.usage.model_dump()}
    out["openai_models"] = [m.id for m in client.models.list().data]

    claude = anthropic.Anthropic(base_url=api, api_key=key, max_retries=0, timeout=600)
    started, first, text = time.monotonic(), None, ""
    usage: dict[str, Any] = {}
    for event in claude.messages.create(
        model=model,
        max_tokens=64,
        stream=True,
        messages=[{"role": "user", "content": question}],
    ):
        if event.type == "content_block_delta" and getattr(event.delta, "text", None):
            first = first or time.monotonic() - started
            text += event.delta.text
        elif event.type == "message_start":
            usage["input_tokens"] = event.message.usage.input_tokens
        elif event.type == "message_delta" and event.usage is not None:
            usage["output_tokens"] = event.usage.output_tokens
    out["anthropic_stream"] = {"text": text, "first_token_s": first, "usage": usage}

    local = ollama.Client(host=api, headers={"Authorization": f"Bearer {key}"}, timeout=600)
    started, first, text, last = time.monotonic(), None, "", None
    for part in local.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        stream=True,
        options={"temperature": 0, "num_predict": 64},
    ):
        piece = part["message"]["content"]
        if piece:
            first = first or time.monotonic() - started
            text += piece
        last = part
    out["ollama_stream"] = {
        "text": text,
        "first_token_s": first,
        "prompt_eval_count": getattr(last, "prompt_eval_count", None),
        "eval_count": getattr(last, "eval_count", None),
    }
    out["ollama_tags"] = [m.model for m in local.list().models]
    return out


def part_a() -> None:
    log("part A: headless service")
    work = HERE / "work_a"
    shutil.rmtree(work, ignore_errors=True)
    app_dir, data = work / "app", work / "data"
    settings(app_dir, data, 11441)
    env = {
        **os.environ,
        "RAMVI_UI_TOKEN": TOKEN,
        "RAMVI_EXIT_WITH_PARENT": "1",
        "PYTHONPATH": os.pathsep.join(map(str, CODE)),
    }
    errors = open(HERE / "service_a.log", "wb")
    # Встроенный Python читает путь импорта из ._pth и не смотрит PYTHONPATH:
    # свежий код добавляет короткий загрузчик.
    launcher = (
        "import sys, runpy; sys.path[:0] = sys.argv.pop(1).split(';'); "
        "sys.argv[0] = 'ramvi_service.app'; runpy.run_module('ramvi_service.app', run_name='__main__')"
    )
    process = subprocess.Popen(
        [
            str(PYTHON),
            "-c",
            launcher,
            ";".join(map(str, CODE)),
            "--headless",
            "--data-dir",
            str(data),
            "--app-dir",
            str(app_dir),
            "--api-port",
            "11441",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=errors,
        env=env,
    )
    try:
        assert process.stdout is not None
        ready = json.loads(process.stdout.readline())
        RESULTS["a_ready"] = ready
        api, ui = ready["api"], Ui(ready["port"])
        RESULTS["a_scan"] = ui.call("POST", "/models/scan")
        model = pick_model(ui.call("GET", "/models")["models"])
        RESULTS["model"] = model
        log("loading", model)
        started = time.monotonic()
        ui.call("POST", "/engine/load", {"model_id": model})
        RESULTS["a_engine"] = wait_ready(ui)
        RESULTS["a_load_s"] = time.monotonic() - started
        created = ui.call(
            "POST", "/connections/keys", {"name": "exam IDE", "permissions": ["chat", "models"]}
        )
        RESULTS["a_clients"] = official_clients(api, created["key"], model)
        RESULTS["a_counters"] = ui.call("GET", "/connections")["clients"]
        RESULTS["a_discovery"] = json.loads((data / "run" / "api.json").read_text("utf-8"))
        log("part A done")
    finally:
        save()
        assert process.stdin is not None
        process.stdin.close()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        RESULTS["a_exit"] = process.returncode
        errors.close()


# --- часть B: экзамены 7 и 8 -------------------------------------------------------------


def part_b() -> None:
    from ramvi_service.api_server import ApiOptions
    from ramvi_service.app import build

    log("part B: in-process service")
    work = HERE / "work_b"
    shutil.rmtree(work, ignore_errors=True)
    app_dir, data = work / "app", work / "data"
    settings(app_dir, data, 11442)
    server, engines = build(data, TOKEN, app_dir=app_dir)
    server.start()
    api = server.api
    api.options = ApiOptions(port=11442)
    api.start()
    url = api.url
    try:
        assert server.onboarding is not None
        server.onboarding.scan_models_dir()
        model = pick_model([{"id": m.id} for m in server.onboarding.models.all()])
        llm = api.llm
        switcher = llm.authority.issue("core", "exam", "switch")
        llm.retain(switcher)
        switch_log: list[dict[str, Any]] = []

        def switch(resident: int | None) -> None:
            asked = time.monotonic()
            with llm.exclusive(switcher, timeout=900):
                got = time.monotonic()
                engines.ensure(model, resident=resident)
            switch_log.append(
                {
                    "resident": resident,
                    "asked": asked,
                    "slot": got,
                    "ready": time.monotonic(),
                    "status": engines.status(),
                    "plan": engine_plan(),
                    "vram_mib": vram(),
                }
            )

        def engine_plan() -> Any:
            provider = engines.provider()
            request = urllib.request.Request(
                provider.base_url + "/engine/metrics",
                headers={"Authorization": f"Bearer {provider.api_key}"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read()).get("plan")

        def vram() -> int:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout
            return int(out.strip().splitlines()[0])

        log("loading full plan")
        switch(None)
        keys = {}
        for name in ("base", "A", "B"):
            record, key = api.keys.create(f"exam {name}", {"chat", "models"})
            keys[name] = (record, key)

        def client(name: str) -> openai.OpenAI:
            return openai.OpenAI(
                base_url=url + "/v1", api_key=keys[name][1], max_retries=0, timeout=900
            )

        def ask(name: str, prompt: str, max_tokens: int = 80) -> tuple[str, dict[str, int]]:
            r = client(name).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content or "", r.usage.model_dump()

        # Экзамен 8: эталон и повторяемость.
        base1 = {p: ask("base", p)[0] for p in PROMPTS}
        base2 = {p: ask("base", p)[0] for p in PROMPTS}
        long_base = ask("base", LONG, 200)[0]
        RESULTS["e8_baseline"] = base1
        RESULTS["e8_repeatable"] = base1 == base2
        log("baseline repeatable:", base1 == base2)

        # Экзамен 8: два клиента вперемешку и одновременно.
        mixed: list[dict[str, Any]] = []
        lock = threading.Lock()

        def run(name: str, order: list[int]) -> None:
            for index in order:
                text, _ = ask(name, PROMPTS[index])
                with lock:
                    mixed.append(
                        {"client": name, "prompt": index, "same": text == base1[PROMPTS[index]]}
                    )

        threads = [
            threading.Thread(target=run, args=("A", [0, 1, 2, 0, 1, 2])),
            threading.Thread(target=run, args=("B", [2, 1, 0, 2, 1, 0])),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        RESULTS["e8_interleaved"] = {
            "answers": len(mixed),
            "all_same": all(m["same"] for m in mixed),
            "detail": mixed,
        }
        log("interleaved all same:", all(m["same"] for m in mixed), len(mixed))

        # Экзамен 8: смена размещения посреди чужого ответа.
        stream_out: dict[str, Any] = {}

        def long_stream() -> None:
            text, started = "", time.monotonic()
            for chunk in client("A").chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": LONG}],
                temperature=0,
                max_tokens=200,
                stream=True,
            ):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    stream_out.setdefault("first", time.monotonic())
                    text += delta
            stream_out.update(text=text, end=time.monotonic(), started=started)

        streamer = threading.Thread(target=long_stream)
        streamer.start()
        while "first" not in stream_out:
            time.sleep(0.05)
        time.sleep(0.5)
        switch(20)  # гибрид: 20 слоёв в видеопамяти, остальные на процессоре
        streamer.join()
        hybrid = switch_log[-1]
        RESULTS["e8_switch_mid_request"] = {
            "answer_same_as_baseline": stream_out["text"] == long_base,
            "switch_asked_before_answer_end": hybrid["asked"] < stream_out["end"],
            "switch_took_slot_after_answer_end": hybrid["slot"] >= stream_out["end"],
            "hybrid_plan": hybrid["plan"],
            "hybrid_vram_mib": hybrid["vram_mib"],
            "full_plan": switch_log[0]["plan"],
            "full_vram_mib": switch_log[0]["vram_mib"],
        }
        log("mid-request switch:", RESULTS["e8_switch_mid_request"])
        hybrid1 = {p: ask("B", p)[0] for p in PROMPTS}
        hybrid2 = {p: ask("A", p)[0] for p in PROMPTS}
        RESULTS["e8_hybrid"] = {
            "repeatable": hybrid1 == hybrid2,
            "same_as_full_gpu": {p: hybrid1[p] == base1[p] for p in PROMPTS},
            "answers": hybrid1,
        }
        switch(None)
        restored = {p: ask("A", p)[0] for p in PROMPTS}
        RESULTS["e8_back_to_full"] = {
            "same_as_baseline": restored == base1,
            "status": switch_log[-1]["status"],
        }
        log("back to full same:", restored == base1)

        # Экзамен 7: два ключа раздельно, счётчики по каждому.
        a_rec, b_rec = keys["A"][0], keys["B"][0]
        before = {n: api.keys.get(keys[n][0].id) for n in ("A", "B")}
        usage: dict[str, list[dict[str, int]]] = {"A": [], "B": []}

        def counted(name: str, prompt: str) -> None:
            _, used = ask(name, prompt)
            with lock:
                usage[name].append(used)

        threads = [
            threading.Thread(target=counted, args=("A", PROMPTS[0])),
            threading.Thread(target=counted, args=("B", PROMPTS[1])),
            threading.Thread(target=counted, args=("A", PROMPTS[2])),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        after = {n: api.keys.get(keys[n][0].id) for n in ("A", "B")}
        counters = {}
        for name in ("A", "B"):
            counters[name] = {
                "requests": after[name].requests - before[name].requests,
                "prompt_tokens": after[name].prompt_tokens - before[name].prompt_tokens,
                "completion_tokens": after[name].completion_tokens - before[name].completion_tokens,
                "busy_seconds": after[name].busy_seconds - before[name].busy_seconds,
                "client_saw_prompt": sum(u["prompt_tokens"] for u in usage[name]),
                "client_saw_completion": sum(u["completion_tokens"] for u in usage[name]),
            }
        RESULTS["e7_counters"] = counters
        log("counters:", counters)

        # Экзамен 7: отзыв одного не останавливает другой.
        revoke_out: dict[str, Any] = {}

        def a_long() -> None:
            text = ""
            try:
                for chunk in client("A").chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": LONG}],
                    temperature=0,
                    max_tokens=200,
                    stream=True,
                ):
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        revoke_out.setdefault("a_first", True)
                        text += delta
                revoke_out["a_error"] = None
            except Exception as error:  # noqa: BLE001
                revoke_out["a_error"] = type(error).__name__
            revoke_out["a_text_len"] = len(text)
            revoke_out["a_full_len"] = len(long_base)

        def b_wait() -> None:
            text, _ = ask("B", PROMPTS[1])
            revoke_out["b_same"] = text == base1[PROMPTS[1]]

        b_before = api.keys.get(b_rec.id)
        ta = threading.Thread(target=a_long)
        ta.start()
        while "a_first" not in revoke_out:
            time.sleep(0.05)
        tb = threading.Thread(target=b_wait)
        tb.start()
        time.sleep(1.0)
        revoke_out["stopped"] = api.revoke(a_rec.id)
        ta.join()
        tb.join()
        try:
            ask("A", PROMPTS[0])
            revoke_out["a_after_revoke"] = "answered"
        except openai.AuthenticationError:
            revoke_out["a_after_revoke"] = "401"
        revoke_out["b_after_revoke_same"] = ask("B", PROMPTS[2])[0] == base1[PROMPTS[2]]
        b_after = api.keys.get(b_rec.id)
        revoke_out["b_requests_delta"] = b_after.requests - b_before.requests
        revoke_out["b_errors_delta"] = b_after.errors - b_before.errors
        RESULTS["e7_revoke"] = revoke_out
        RESULTS["switch_log"] = switch_log
        log("revoke:", revoke_out)
    finally:
        save()
        server.close()
        server.sections.host.close() if server.sections is not None else None
        engines.stop()


def main() -> None:
    try:
        part_a()
    except Exception:  # noqa: BLE001
        RESULTS["a_failure"] = traceback.format_exc()
        log("part A failed")
        print(RESULTS["a_failure"], flush=True)
    save()
    if "--only-a" in sys.argv:
        RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        return
    try:
        part_b()
    except Exception:  # noqa: BLE001
        RESULTS["b_failure"] = traceback.format_exc()
        log("part B failed")
        print(RESULTS["b_failure"], flush=True)
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()


if __name__ == "__main__":
    main()
