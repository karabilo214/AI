"""Bounded local speech + DAC telemetry check; no microphone, LLM or remote API."""

import json
import tempfile
import threading
import time
from pathlib import Path

from ramvi_character.speech import SpeechOutput
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.speech.native import SystemSpeech
from ramvi_service.speech.output import SoundOutput
from ramvi_service.speech.service import SpeechService


def run():
    outputs = []
    sources = []
    records = []

    def output(rate):
        device = SoundOutput(rate)
        outputs.append(device)
        return device

    def provider(info, key):
        native = SystemSpeech()
        sources.append(native)
        return native

    with tempfile.TemporaryDirectory(prefix="ramvi-sync-check-") as directory:
        core = SpeechService(provider_factory=provider, output_factory=output)
        data = ScopedData(Path(directory), "character")
        secrets = ScopedSecrets(MemorySecrets(), "character")

        class Bridge:
            def call(self, method, payload):
                if method == "data.get":
                    return {"value": data.read_json(payload["path"], None)}
                if method == "data.set":
                    data.write_json(payload["path"], payload["value"])
                    return {"ok": True}
                name = method.removeprefix("speech.")
                if name == "configure":
                    return core.configure("check", data, payload["settings"])
                if name == "advance":
                    return core.advance("check", payload["generation"], payload["response_id"])
                if name == "start":
                    return core.start("check", data, secrets, payload)
                if name == "status":
                    return core.status("check", payload["job_id"])
                if name == "cancel":
                    return core.cancel("check")
                raise ValueError(name)

        speech = SpeechOutput(Bridge(), "native-clock-check")
        guard = threading.Timer(55, lambda: (speech.close(), core.close()))
        guard.daemon = True
        guard.start()
        try:
            voices = SystemSpeech()
            try:
                available = voices.voices()
            finally:
                voices.close()
            selected = next(
                (v["id"] for v in available if "ru" in v["languages"]), available[0]["id"]
            )
            speech.configure({"enabled": True, "voice": selected})
            for number, emotion, interrupt in [
                (65539, "sad", False),
                (104723, "cheerful", False),
                (999983, "neutral", False),
                (702841, "thinking", True),
                (524287, "neutral", False),
            ]:
                identifier = f"fresh-{number}"
                generation = speech.begin(identifier)
                speech.offer(
                    identifier,
                    {
                        "id": "s1",
                        "text": f"Проверка. Число {number}.",
                        "emotion": emotion,
                        "animation": "talk",
                        "intensity": 0.4,
                        "voice_style": "neutral",
                    },
                    generation,
                )
                speech.finish(identifier, generation)
                deadline = time.monotonic() + 15
                trace = []
                seen_playback = False
                while time.monotonic() < deadline:
                    speech.touch()
                    packet = speech.view()
                    if packet["error"]:
                        raise RuntimeError(packet["error"])
                    sync = packet["sync"]
                    if sync and sync["playback"]["windows"]:
                        assert sync["emotion"] == emotion
                        seen_playback = True
                        trace.append(sync)
                    audible = bool(
                        sync
                        and any(
                            w["start_ms"] <= 0 < w["end_ms"] and w["amplitude"] > 0.012
                            for w in sync["playback"]["windows"]
                        )
                    )
                    if interrupt and audible:
                        break
                    if seen_playback and not speech.queue and speech.state in {"idle", "completed"}:
                        break
                    time.sleep(0.02)
                else:
                    raise TimeoutError("native clock check")
                assert seen_playback
                job = core.status("check")["job"]
                assert job["metadata"]["requested_emotion"] == emotion
                assert job["submitted_audio_s"] > 0
                cancelled_at = time.monotonic()
                speech.cancel()
                fence_ack_ms = (time.monotonic() - cancelled_at) * 1000
                assert speech.view()["sync"] is None
                while not core.status("check", job["id"])["job"]["settled"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("cancelled audio did not settle")
                    time.sleep(0.01)
                if interrupt:
                    assert core.status("check", job["id"])["job"]["state"] == "cancelled"
                assert any(w["amplitude"] > 0.012 for s in trace for w in s["playback"]["windows"])
                records.append(
                    {
                        "interrupted": interrupt,
                        "fence_ack_ms": round(fence_ack_ms, 3),
                        "response_id": identifier,
                        "emotion": emotion,
                        "rate": job["rate"],
                        "submitted_audio_s": job["submitted_audio_s"],
                        "packets": len(trace),
                        "nonzero_windows": sum(
                            w["amplitude"] > 0.012 for s in trace for w in s["playback"]["windows"]
                        ),
                        "trace": trace,
                    }
                )
        finally:
            speech.close()
            core.close()
            guard.cancel()
            guard.join(timeout=1)
        assert not speech.thread.is_alive()
        assert all(o.worker.status()["state"] == "stopped" for o in outputs)
        assert all(p.worker.status()["state"] == "stopped" for p in sources)
    return {"runs": records, "workers_stopped": True, "character_thread_stopped": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "trace"} for r in result["runs"]],
            ensure_ascii=False,
        )
    )
