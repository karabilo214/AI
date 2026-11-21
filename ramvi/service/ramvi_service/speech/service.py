"""Scoped speech synthesis/playback, separate credentials and remote-text opt-in."""

import copy
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_service.contracts.speech import SPEAKER, SpeechRequest
from ramvi_service.speech.buffer import PCM_BLOCK_MS, PCM_BLOCKS, PCMBuffer
from ramvi_service.speech.cloud import CloudSpeech
from ramvi_service.speech.errors import failure
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
        first_audio_timeout: float = 45,
        audio_timeout: float = 20,
        job_timeout: float = 180,
    ) -> None:
        self.resources, self.voice = resources, voice
        self.provider_factory = provider_factory or self._provider
        self.output_factory = output_factory
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.generations: dict[Any, dict[str, Any]] = {}
        self.voice_cache: list[dict[str, Any]] | None = None
        self.inventory: dict[str, Any] = {"state": "idle", "error": None}
        self.inventory_thread: threading.Thread | None = None
        self.inventory_provider: Any = None
        self.closed = False
        self.first_audio_timeout, self.audio_timeout, self.job_timeout = (
            first_audio_timeout,
            audio_timeout,
            job_timeout,
        )

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
        current = self.settings(data)
        proposed = {**current, **settings}
        if proposed["provider"] != current["provider"]:
            proposed["remote_text_enabled"] = settings.get("remote_text_enabled", False)

        # Validate without committing, using the same settings parser.
        class Proposed:
            def read_json(self, path: str, default: Any = None) -> Any:
                return proposed

        self.settings(Proposed())
        self.cancel(owner)
        data.write_json("speech.json", proposed)
        return {"settings": proposed}

    def _inventory(self) -> None:
        provider = None
        try:
            provider = self.provider_factory(CATALOG["providers"][0], "")
            with self.lock:
                self.inventory_provider = provider
                if self.closed:
                    return
            voices = provider.voices()[:128]
            with self.lock:
                self.voice_cache = voices
                self.inventory = {"state": "ready", "error": None}
        except Exception:
            with self.lock:
                self.inventory = {"state": "error", "error": "speech.provider_failed"}
        finally:
            if provider is not None:
                try:
                    provider.close()
                except Exception:
                    pass
            with self.lock:
                self.inventory_provider = None

    def catalog(self, data: Any, refresh: bool = False) -> dict[str, Any]:
        if not isinstance(refresh, bool):
            raise ValueError("invalid inventory refresh")
        result: dict[str, Any] = copy.deepcopy(CATALOG)
        result["settings"] = self.settings(data)
        with self.lock:
            if (
                not self.closed
                and (self.inventory["state"] == "idle" or refresh)
                and not (self.inventory_thread and self.inventory_thread.is_alive())
            ):
                self.inventory = {"state": "loading", "error": None}
                self.inventory_thread = threading.Thread(
                    target=self._inventory, name="speech-inventory", daemon=True
                )
                self.inventory_thread.start()
            result["local_voices"] = copy.deepcopy(self.inventory)
            result["providers"][0]["voices"] = copy.deepcopy(self.voice_cache or [])
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
            "generation",
            "preface",
        }:
            raise ValueError("invalid speech job")
        if not isinstance(payload.get("preview", False), bool):
            raise ValueError("invalid preview flag")
        preface = payload.get("preface", False)
        if not isinstance(preface, bool):
            raise ValueError("invalid speech preface")
        request = SpeechRequest(**payload["request"])
        config = self.settings(data)
        if not config["enabled"] and not payload.get("preview", False):
            raise RuntimeError("speech.disabled")
        info = copy.deepcopy(next(p for p in CATALOG["providers"] if p["id"] == config["provider"]))
        if preface and info["external"]:
            raise PermissionError("speech.preface_local_only")
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
        generation = payload.get("generation")
        fingerprint = hashlib.sha256(
            json.dumps(
                {"request": adapted.to_dict(), "config": config, "preface": preface}, sort_keys=True
            ).encode()
        ).hexdigest()
        with self.lock:
            if self.closed:
                raise RuntimeError("speech.closed")
            fence = self.generations.get(owner)
            # Legacy callers are isolated only until they opt into generation fencing.
            if generation is None and fence is None:
                generation = 0
                fence = {"generation": 0, "response_id": response_id, "seen": {}}
            if (
                generation is None
                or isinstance(generation, bool)
                or not isinstance(generation, int)
            ):
                raise ValueError("speech.generation_required")
            if (
                fence is None
                or generation != fence["generation"]
                or response_id != fence["response_id"]
            ):
                raise RuntimeError("speech.stale_generation")
            previous = fence["seen"].get(segment_id)
            if previous:
                if previous[0] != fingerprint:
                    raise ValueError("speech.segment_conflict")
                old = self.jobs[previous[1]]
                return {
                    "job_id": old["id"],
                    "metadata": copy.deepcopy(old["metadata"]),
                    "duplicate": True,
                }
            preface_id = fence.get("preface")
            if (
                preface
                and preface_id is not None
                or not preface
                and len(fence["seen"]) - int(preface_id is not None) >= 8
            ):
                raise RuntimeError("speech.segment_limit")
            # Одна фраза звучит, следующая того же ответа может синтезироваться заранее:
            # пауза между фразами — не время синтеза. Больше одной в очереди нельзя.
            active = [j for j in self.jobs.values() if j["owner"] == owner and not self._settled(j)]
            after = None
            if active:
                if (
                    len(active) > 1
                    or preface
                    or active[0]["generation"] != generation
                    or active[0]["response_id"] != response_id
                ):
                    raise RuntimeError("speech.busy")
                after = active[0]
            self._prune(owner)
            if len(self.jobs) >= 128:
                raise RuntimeError("speech.busy")
            identifier = uuid.uuid4().hex
            job: dict[str, Any] = {
                "id": identifier,
                "owner": owner,
                "generation": generation,
                "io_lock": threading.RLock(),
                "producer": None,
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
                "after": after,
            }
            provider = self.provider_factory(info, key)
            job["provider"] = provider
            job["buffer"] = PCMBuffer(job["cancel"], 6000) if after else PCMBuffer(job["cancel"])
            thread = threading.Thread(
                target=self._run, args=(job, adapted), name="speech-output", daemon=True
            )
            job["thread"] = thread
            self.jobs[identifier] = job
            fence["seen"][segment_id] = (fingerprint, identifier)
            if preface:
                fence["preface"] = segment_id
            thread.start()
            return {"job_id": identifier, "metadata": copy.deepcopy(job["metadata"])}

    @staticmethod
    def _settled(job: dict[str, Any]) -> bool:
        return not job["thread"].is_alive() and not (job["producer"] and job["producer"].is_alive())

    def _prune(self, owner: Any) -> None:
        current = self.generations.get(owner, {}).get("generation")
        for identifier, old in list(self.jobs.items()):
            if old["owner"] == owner and old["generation"] != current and self._settled(old):
                del self.jobs[identifier]

    def advance(self, owner: Any, generation: int, response_id: str | None) -> dict[str, Any]:
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or not 0 < generation < 2**53
        ):
            raise ValueError("invalid speech generation")
        if response_id is not None and (
            not isinstance(response_id, str) or not response_id or len(response_id) > 64
        ):
            raise ValueError("invalid speech response id")
        with self.lock:
            current = self.generations.get(owner)
            if current and generation < current["generation"]:
                raise RuntimeError("speech.stale_generation")
            if current and generation == current["generation"]:
                if current["response_id"] != response_id:
                    raise ValueError("speech.generation_conflict")
                return {"generation": generation}
            if current is None and len(self.generations) >= 64:
                raise RuntimeError("speech.busy")
            self.generations[owner] = {
                "generation": generation,
                "response_id": response_id,
                "seen": {},
            }
            jobs = [j for j in self.jobs.values() if j["owner"] == owner and not self._settled(j)]
            for job in jobs:
                job["cancel"].set()
        # Fence publication + the per-device lock form a barrier: after acknowledgement
        # no older device can start or submit PCM. Provider cancellation is owned by
        # the consumer's finally, so network/process teardown never blocks this barrier.
        for job in jobs:
            self._abort(job)
        return {"generation": generation}

    @staticmethod
    def _abort(job: dict[str, Any]) -> None:
        output = job["output"]
        if output is not None and callable(getattr(output, "interrupt", None)):
            try:
                output.interrupt()
            except Exception:
                pass
        with job["io_lock"]:
            if job["output"]:
                try:
                    job["output"].cancel()
                except Exception:
                    pass

    def _run(self, job: dict[str, Any], request: SpeechRequest) -> None:
        lease = None
        gated = False
        output = None
        buffer = job["buffer"]
        started = time.monotonic()
        stage = "provider"
        try:
            job["state"] = "synthesizing"
            producer = threading.Thread(
                target=buffer.produce,
                args=(job["provider"], request),
                name="speech-producer",
                daemon=True,
            )
            job["producer"] = producer
            producer.start()
            deadline = min(started + self.first_audio_timeout, started + self.job_timeout)
            while (
                piece := buffer.get(
                    deadline,
                    "speech.provider_timeout" if output is None else "speech.audio_timeout",
                )
            ) is not None:
                if time.monotonic() - started > self.job_timeout:
                    raise TimeoutError("speech.deadline")
                job["metadata"]["alignment"] = getattr(job["provider"], "timings", None)
                if output is None:
                    job["metadata"]["first_pcm_ms"] = (time.monotonic() - started) * 1000
                    previous = job.pop("after", None)
                    if previous is not None:
                        # Звук готов — ждём, пока доиграет предыдущая фраза (её устройство
                        # и аренда динамика освобождаются до конца её потока).
                        job["state"] = "waiting"
                        waited = time.monotonic()
                        while not self._settled(previous):
                            if job["cancel"].is_set():
                                raise InterruptedError("speech cancelled")
                            time.sleep(0.005)
                        started += time.monotonic() - waited  # ожидание — не время задания
                    if self.resources:
                        lease = self.resources.acquire(
                            job["owner"], {SPEAKER: 1}, timeout=10, cancel=job["cancel"]
                        )
                    if self.voice:
                        try:
                            self.voice.session_playback(True, owner=job["owner"])
                            gated = True
                        except Exception as error:
                            if str(error) != "voice.not_owner":
                                raise RuntimeError("speech.device_unavailable") from None
                    # Construction must not start playback. A late constructor can
                    # finish after advance; its cancelled job will never call start.
                    stage = "device_open"
                    output = self.output_factory(piece.rate)
                    with job["io_lock"]:
                        job["output"] = output
                        if job["cancel"].is_set():
                            raise InterruptedError("speech cancelled")
                        stage = "device_start"
                        output.start()
                        job["rate"] = piece.rate
                        job["state"] = "playing"
                with job["io_lock"]:
                    if job["cancel"].is_set():
                        raise InterruptedError("speech cancelled")
                    stage = "device_write"
                    output.write(piece.pcm)
                    telemetry = getattr(output, "telemetry", None)
                    if (
                        telemetry is not None
                        and telemetry[0].get("windows")
                        and "first_dac_ms" not in job["metadata"]
                    ):
                        raw, observed = telemetry
                        job["metadata"]["first_dac_ms"] = max(
                            0,
                            (observed + raw["windows"][0][0] - raw["device_time"] - started) * 1000,
                        )
                    job["bytes"] += len(piece.pcm)
                    job["submitted_audio_s"] = job["bytes"] / piece.rate / 2
                stage = "provider"
                deadline = min(time.monotonic() + self.audio_timeout, started + self.job_timeout)
            if job["cancel"].is_set():
                raise InterruptedError("speech cancelled")
            drain = getattr(output, "drain", None)
            if callable(drain):
                with job["io_lock"]:
                    if job["cancel"].is_set():
                        raise InterruptedError("speech cancelled")
                    stage = "device_write"
                    drain()
            job["metadata"]["alignment"] = getattr(job["provider"], "timings", None)
            job["state"] = "completed"
        except InterruptedError:
            job["state"] = "cancelled"
        except Exception as error:
            job["state"] = "cancelled" if job["cancel"].is_set() else "error"
            if not job["cancel"].is_set():
                job["error"] = failure(error, stage)
        finally:
            if job["state"] != "completed":
                job["cancel"].set()
                self._abort(job)
            # Every cleanup runs even if another one fails. Device facade always
            # terminates its process after a bounded close/drain RPC.
            cleanup = []
            if output:

                def close_output() -> None:
                    with job["io_lock"]:
                        output.close()

                cleanup.append(close_output)
            if gated:
                cleanup.append(lambda: self.voice.session_playback(False, owner=job["owner"]))
            if lease:
                cleanup.append(lambda: self.resources.release(job["owner"], lease))
            cleanup.append(job["provider"].close)
            for close in cleanup:
                try:
                    close()
                except Exception:
                    if job["error"] is None and not job["cancel"].is_set():
                        job["error"] = "speech.cleanup_failed"
                        job["state"] = "error"
                    job["cancel"].set()
                    self._abort(job)
            if job["producer"]:
                job["producer"].join(timeout=3)

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
            # Без id — звучащая фраза, а не заранее синтезируемая следующая.
            job = next((j for j in matches if not self._settled(j)), matches[-1])
            result = {
                k: copy.deepcopy(job[k])
                for k in [
                    "id",
                    "state",
                    "generation",
                    "response_id",
                    "segment_id",
                    "error",
                    "metadata",
                    "bytes",
                    "rate",
                    "submitted_audio_s",
                ]
            }
            view = getattr(job["output"], "playback_view", None)
            result["playback"] = (
                view(job["metadata"].get("alignment"))
                if callable(view) and job["state"] == "playing" and not job["cancel"].is_set()
                else {"clock": "unavailable", "windows": []}
            )
            result["settled"] = self._settled(job)
            result["pcm_buffer"] = {
                "blocks": job["buffer"].queue.qsize(),
                "high_water": job["buffer"].high_water,
                "capacity": PCM_BLOCKS,
                "block_ms": PCM_BLOCK_MS,
            }
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
            self._abort(job)
        return {"ok": True}

    def release(self, owner: Any) -> None:
        self.cancel(owner)
        with self.lock:
            jobs = [j for j in self.jobs.values() if j["owner"] == owner]
        for job in jobs:
            job["thread"].join(timeout=3)
        with self.lock:
            for job in jobs:
                if self._settled(job):
                    self.jobs.pop(job["id"], None)
            if not any(j["owner"] == owner for j in self.jobs.values()):
                self.generations.pop(owner, None)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            inventory = self.inventory_provider
            owners = {j["owner"] for j in self.jobs.values()} | set(self.generations)
        for owner in owners:
            self.release(owner)
        if inventory is not None:
            try:
                inventory.close()
            except Exception:
                pass
        if self.inventory_thread:
            self.inventory_thread.join(timeout=3)
