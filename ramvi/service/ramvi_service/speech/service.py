"""Scoped speech synthesis/playback, separate credentials and remote-text opt-in."""

import copy
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_service.contracts.speech import SPEAKER, SpeechRequest
from ramvi_service.speech.cloud import CloudSpeech
from ramvi_service.speech.native import SystemSpeech
from ramvi_service.speech.output import SoundOutput

CATALOG = json.loads(Path(__file__).with_name("providers.json").read_text())
DEFAULTS = {"enabled": False, "provider": "system", "voice": "", "remote_text_enabled": False}


class SpeechService:
    def __init__(
        self,
        resources: Any = None,
        voice: Any = None,
        provider_factory: Callable[..., Any] | None = None,
        output_factory: Callable[[int], Any] = SoundOutput,
    ) -> None:
        self.resources, self.voice = resources, voice
        self.provider_factory = provider_factory or self._provider
        self.output_factory = output_factory
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.voice_cache: list[dict[str, Any]] | None = None

    @staticmethod
    def _provider(info: dict[str, Any], key: str) -> Any:
        return (
            SystemSpeech()
            if info["id"] == "system"
            else CloudSpeech(info["id"], info["model"], key)
        )

    def settings(self, data: Any) -> dict[str, Any]:
        saved = data.read_json("speech.json", {})
        if not isinstance(saved, dict) or set(saved) - set(DEFAULTS):
            raise ValueError("invalid speech settings")
        result = {**DEFAULTS, **saved}
        if (
            result["provider"] not in {p["id"] for p in CATALOG["providers"]}
            or any(not isinstance(result[k], bool) for k in ["enabled", "remote_text_enabled"])
            or not isinstance(result["voice"], str)
            or len(result["voice"].encode()) > 255
        ):
            raise ValueError("invalid speech selection")
        return result

    def configure(self, owner: Any, data: Any, settings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
            raise ValueError("invalid speech settings")
        proposed = {**self.settings(data), **settings}

        # Validate without committing, using the same settings parser.
        class Proposed:
            def read_json(self, path: str, default: Any = None) -> Any:
                return proposed

        self.settings(Proposed())
        self.cancel(owner)
        data.write_json("speech.json", proposed)
        return {"settings": proposed}

    def catalog(self, data: Any) -> dict[str, Any]:
        result: dict[str, Any] = copy.deepcopy(CATALOG)
        result["settings"] = self.settings(data)
        with self.lock:
            if self.voice_cache is None:
                provider = self.provider_factory(result["providers"][0], "")
                try:
                    self.voice_cache = provider.voices()
                except (RuntimeError, OSError, NotImplementedError):
                    self.voice_cache = []
                finally:
                    provider.close()
            result["providers"][0]["voices"] = copy.deepcopy(self.voice_cache)
        for item in result["providers"][1:]:
            item["voices"] = [
                {"id": v, "name": v, "languages": ["ru", "en"]} for v in item["voices"]
            ]
        return result

    def key(self, secrets: Any, provider: str, value: str) -> dict[str, Any]:
        if (
            provider not in {"openai", "gemini", "elevenlabs"}
            or not isinstance(value, str)
            or len(value) > 4096
        ):
            raise ValueError("invalid TTS credential")
        try:
            if value.strip():
                secrets.set("tts_" + provider, value.strip())
            else:
                secrets.delete("tts_" + provider)
        except RuntimeError:
            raise RuntimeError("speech.secrets_unavailable") from None
        return {"key_set": bool(value.strip())}  # never return the secret

    def start(self, owner: Any, data: Any, secrets: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - {
            "request",
            "response_id",
            "segment_id",
            "preview",
        }:
            raise ValueError("invalid speech job")
        if not isinstance(payload.get("preview", False), bool):
            raise ValueError("invalid preview flag")
        request = SpeechRequest(**payload["request"])
        config = self.settings(data)
        if not config["enabled"] and not payload.get("preview", False):
            raise RuntimeError("speech.disabled")
        info = copy.deepcopy(next(p for p in CATALOG["providers"] if p["id"] == config["provider"]))
        if info["external"] and not config["remote_text_enabled"]:
            raise PermissionError("speech.remote_text_disabled")
        key = ""
        if info["external"]:
            key = secrets.get("tts_" + info["id"]) or ""
            if not key:
                raise RuntimeError("speech.key_missing")
        response_id, segment_id = (
            payload.get("response_id", "preview"),
            payload.get("segment_id", "preview"),
        )
        if any(not isinstance(v, str) or not v or len(v) > 64 for v in [response_id, segment_id]):
            raise ValueError("invalid speech response/segment id")
        voice = config["voice"] or (
            "coral"
            if info["id"] == "openai"
            else "Kore"
            if info["id"] == "gemini"
            else "JBFqnCBsd6RMkjVDRZzb"
            if info["id"] == "elevenlabs"
            else ""
        )
        if info["external"] and voice not in info["voices"]:
            raise ValueError("speech.voice_not_supported")
        applied = (
            "prompt"
            if info["emotion_control"] == "prompt" and request.style
            else "provider_default"
        )
        adapted = replace(request, voice=voice, style=request.style if applied == "prompt" else "")
        with self.lock:
            # One active request per owner; bounded terminal metadata, no accumulated audio/text.
            if any(j["owner"] == owner and j["thread"].is_alive() for j in self.jobs.values()):
                raise RuntimeError("speech.busy")
            for identifier, old in list(self.jobs.items()):
                if old["owner"] == owner and not old["thread"].is_alive():
                    del self.jobs[identifier]
            if len(self.jobs) >= 16:
                raise RuntimeError("speech.busy")
            identifier = uuid.uuid4().hex
            job: dict[str, Any] = {
                "id": identifier,
                "owner": owner,
                "state": "queued",
                "cancel": threading.Event(),
                "provider": None,
                "output": None,
                "response_id": response_id,
                "segment_id": segment_id,
                "error": None,
                "metadata": {
                    "provider": info["id"],
                    "model": info["model"],
                    "voice": voice,
                    "requested_emotion": request.emotion,
                    "emotion_control": applied,
                    "fallback": bool(request.style and applied == "provider_default"),
                    "streaming": info["streaming"],
                    "timing": info["timecodes"],
                    "pricing": info["pricing"],
                },
                "bytes": 0,
                "rate": None,
                "submitted_audio_s": 0.0,
            }
            provider = self.provider_factory(info, key)
            job["provider"] = provider
            thread = threading.Thread(
                target=self._run, args=(job, adapted), name="speech-output", daemon=True
            )
            job["thread"] = thread
            self.jobs[identifier] = job
            thread.start()
            return {"job_id": identifier, "metadata": copy.deepcopy(job["metadata"])}

    def _run(self, job: dict[str, Any], request: SpeechRequest) -> None:
        lease = None
        gated = False
        carry = b""
        started = time.monotonic()
        try:
            job["state"] = "synthesizing"
            for piece in job["provider"].synthesize(request, job["cancel"]):
                if job["cancel"].is_set():
                    raise InterruptedError("speech cancelled")
                if time.monotonic() - started > 180:
                    raise TimeoutError("speech deadline")
                if job["output"] is None:
                    if self.resources:
                        lease = self.resources.acquire(
                            job["owner"], {SPEAKER: 1}, timeout=10, cancel=job["cancel"]
                        )
                    if self.voice:
                        try:
                            self.voice.session_playback(True, owner=job["owner"])
                            gated = True
                        except Exception:
                            pass  # no own input session; never suppress somebody else's microphone
                    if job["cancel"].is_set():
                        raise InterruptedError("speech cancelled")
                    job["output"] = self.output_factory(piece.rate)
                    job["rate"] = piece.rate
                    job["state"] = "playing"
                if piece.rate != job["rate"]:
                    raise ValueError("speech format changed midstream")
                pcm = carry + piece.pcm
                carry = pcm[len(pcm) // 2 * 2 :]
                pcm = pcm[: len(pcm) // 2 * 2]
                job["bytes"] += len(pcm)
                if job["bytes"] > piece.rate * 2 * 120:
                    raise ValueError("speech duration exceeds limit")
                if job["cancel"].is_set():
                    raise InterruptedError("speech cancelled")
                if pcm:
                    job["output"].write(pcm)
                job["submitted_audio_s"] = job["bytes"] / piece.rate / 2
            if job["cancel"].is_set():
                raise InterruptedError("speech cancelled")
            if carry or not job["bytes"]:
                raise ValueError("truncated or empty speech audio")
            job["metadata"]["alignment"] = getattr(job["provider"], "timings", None)
            job["state"] = "completed"
        except InterruptedError:
            job["state"] = "cancelled"
        except Exception as error:
            job["state"] = "cancelled" if job["cancel"].is_set() else "error"
            if not job["cancel"].is_set():
                code = str(error)
                job["error"] = (
                    code if code.startswith("speech.") and len(code) < 100 else "speech.failed"
                )
        finally:
            if job["output"]:
                try:
                    job["output"].close()
                except Exception:
                    pass
            if gated:
                try:
                    self.voice.session_playback(False, owner=job["owner"])
                except Exception:
                    pass
            if lease:
                self.resources.release(job["owner"], lease)
            job["provider"].close()

    def status(self, owner: Any, identifier: str | None = None) -> dict[str, Any]:
        with self.lock:
            matches = [
                j
                for j in self.jobs.values()
                if j["owner"] == owner and (identifier is None or j["id"] == identifier)
            ]
            if identifier is not None and not matches:
                raise PermissionError("speech.not_owner")
            if not matches:
                return {"job": None}
            job = matches[-1]
            result = {
                k: copy.deepcopy(job[k])
                for k in [
                    "id",
                    "state",
                    "response_id",
                    "segment_id",
                    "error",
                    "metadata",
                    "bytes",
                    "rate",
                    "submitted_audio_s",
                ]
            }
            result["settled"] = not job["thread"].is_alive()
            return {"job": result}

    def cancel(self, owner: Any, identifier: str | None = None) -> dict[str, Any]:
        with self.lock:
            jobs = [
                j
                for j in self.jobs.values()
                if j["owner"] == owner and (identifier is None or j["id"] == identifier)
            ]
            if identifier is not None and not jobs:
                raise PermissionError("speech.not_owner")
            for job in jobs:
                job["cancel"].set()
                if job["state"] not in {"completed", "error"}:
                    job["state"] = "cancelled"
        for job in jobs:
            if job["output"]:
                try:
                    job["output"].cancel()
                except Exception:
                    pass
            job["provider"].cancel()
        return {"ok": True}

    def release(self, owner: Any) -> None:
        self.cancel(owner)
        with self.lock:
            jobs = [j for j in self.jobs.values() if j["owner"] == owner]
        for job in jobs:
            job["thread"].join(timeout=3)
        with self.lock:
            for job in jobs:
                if not job["thread"].is_alive():
                    self.jobs.pop(job["id"], None)

    def close(self) -> None:
        with self.lock:
            owners = {j["owner"] for j in self.jobs.values()}
        for owner in owners:
            self.release(owner)
