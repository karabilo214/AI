"""Local delivery mechanism check; scripted context categories, no LLM quality claim."""

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
    rows = []
    probe = SystemSpeech()
    try:
        voices = probe.voices()
    finally:
        probe.close()
    voice = next((v["id"] for v in voices if "ru" in v["languages"]), voices[0]["id"])
    with tempfile.TemporaryDirectory(prefix="ramvi-delivery-") as directory:
        data = ScopedData(Path(directory), "character")
        secrets = ScopedSecrets(MemorySecrets(), "character")
        for index, category in enumerate(["write", "calculate", "write"]):
            outputs, providers, starts = [], [], []

            def provider(info, key, providers=providers):
                obj = SystemSpeech()
                providers.append(obj)
                return obj

            def output(rate, outputs=outputs):
                obj = SoundOutput(rate)
                outputs.append(obj)
                return obj

            core = SpeechService(provider_factory=provider, output_factory=output)
            owner = f"check-{index}"

            class Bridge:
                def __init__(self, core=core, owner=owner, starts=starts):
                    self.core, self.owner, self.starts = core, owner, starts

                def call(self, method, payload):
                    if method == "data.get":
                        return {"value": data.read_json(payload["path"], None)}
                    if method == "data.set":
                        data.write_json(payload["path"], payload["value"])
                        return {"ok": True}
                    if method == "speech.configure":
                        return self.core.configure(self.owner, data, payload["settings"])
                    if method == "speech.advance":
                        return self.core.advance(
                            self.owner, payload["generation"], payload["response_id"]
                        )
                    if method == "speech.start":
                        self.starts.append(payload)
                        return self.core.start(self.owner, data, secrets, payload)
                    if method == "speech.status":
                        return self.core.status(self.owner, payload["job_id"])
                    if method == "speech.cancel":
                        return self.core.cancel(self.owner)
                    raise ValueError(method)

            speech = SpeechOutput(Bridge())
            guard = threading.Timer(
                20, lambda speech=speech, core=core: (speech.close(), core.close())
            )
            guard.daemon = True
            guard.start()
            try:
                speech.configure({"enabled": True, "voice": voice})
                identifier = f"fresh-{[65539, 999983, 524287][index]}"
                generation = speech.begin(identifier)
                speech.react(
                    identifier,
                    {
                        "reaction": category,
                        "reaction_language": "ru",
                        "emotion": "neutral",
                        "intensity": 0.2,
                    },
                    generation,
                )
                deadline = time.monotonic() + 15
                audible_reaction = False
                while time.monotonic() < deadline:
                    speech.touch()
                    packet = speech.view()
                    if packet["error"]:
                        raise RuntimeError(packet["error"])
                    sync = packet["sync"]
                    if (
                        sync
                        and sync["segment_id"] == "~reaction"
                        and any(
                            w["start_ms"] <= 0 < w["end_ms"] and w["amplitude"] > 0.012
                            for w in sync["playback"]["windows"]
                        )
                    ):
                        audible_reaction = True
                        break
                    time.sleep(0.02)
                assert audible_reaction
                speech.offer(
                    identifier,
                    {
                        "id": "s1",
                        "text": f"Готовая фраза. {identifier.split('-')[1]}.",
                        "emotion": "neutral",
                        "intensity": 0,
                        "animation": "none",
                        "voice_style": "neutral",
                    },
                    generation,
                )
                speech.finish(identifier, generation)
                while time.monotonic() < deadline:
                    speech.touch()
                    if speech.error:
                        raise RuntimeError(speech.error)
                    if len(starts) == 2 and core.status(owner)["job"]["settled"]:
                        break
                    time.sleep(0.02)
                else:
                    raise TimeoutError("delivery check")
                assert starts[0]["preface"] and not starts[1]["preface"]
                assert core.status(owner)["job"]["state"] == "completed"
                rows.append(
                    {
                        "category": category,
                        "reaction": starts[0]["request"]["text"],
                        "reaction_audible_window": True,
                        "answer_completed": True,
                        "jobs": len(starts),
                    }
                )
            finally:
                speech.close()
                core.close()
                guard.cancel()
                guard.join(timeout=1)
            assert not speech.thread.is_alive()
            assert all(o.worker.status()["state"] == "stopped" for o in outputs)
            assert all(p.worker.status()["state"] == "stopped" for p in providers)
    assert rows[0]["reaction"] != rows[2]["reaction"]
    return {"runs": rows, "workers_stopped": True, "llm_context_selection": "not_tested"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
