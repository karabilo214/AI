"""Character lifecycle and text state; devices/providers are optional later adapters."""

import copy
import json
from typing import Any

from ramvi_character.context import Budget, ContextOrchestrator, text
from ramvi_character.dialogue import Dialogue
from ramvi_character.emotions import EmotionalState
from ramvi_character.initiative import InitiativeGate
from ramvi_character.pack import load_pack, presentation
from ramvi_character.personality import Personality
from ramvi_character.speech import SpeechOutput
from ramvi_character.voice import VoiceConversation


class CharacterModule:
    def __init__(self) -> None:
        self._personality = Personality()
        self._emotions = EmotionalState()
        self._initiative = InitiativeGate()
        self._speech = SpeechOutput()
        self._dialogue = Dialogue(self)
        self._voice = VoiceConversation(self)
        self._scope = "uninitialized"
        self._task = ""
        self._state = "new"
        self._draft = ""
        self._model: str | None = None
        self._emotion = "neutral"
        self._presentation: dict[str, Any] | None = None
        self._pack: dict[str, Any] | None = None
        self._presentation_settings = {"pack_path": "", "show_avatar": True}

    def initialize(self, context: dict[str, Any]) -> dict[str, Any]:
        saved = context.get("checkpoint", {})
        if not isinstance(saved, dict) or (
            saved
            and saved.get("schema")
            not in {"ramvi.character.state/1", "ramvi.character.state/2", "ramvi.character.state/3"}
        ):
            raise ValueError("unsupported character state schema")
        draft, model = saved.get("draft", ""), saved.get("model")
        if not isinstance(draft, str) or len(draft.encode("utf-8")) > 32_000:
            raise ValueError("invalid character draft")
        if model is not None and (not isinstance(model, str) or len(model) > 255):
            raise ValueError("invalid model selection")
        expression = saved.get("emotion", "neutral")
        if not isinstance(expression, str) or len(expression) > 64:
            raise ValueError("invalid emotion")
        personality = Personality(saved.get("personality"))
        emotions = EmotionalState(saved.get("emotional_state"))
        initiative = InitiativeGate(saved.get("initiative"))
        task = text(saved.get("task", ""), 4096)
        scope = text(
            context.get("module_id", "character") + ":" + context.get("session_id", "local"), 255
        )
        self._draft, self._model, self._emotion = draft, model, expression
        self._personality, self._emotions = personality, emotions
        self._initiative, self._task, self._scope = initiative, task, scope
        self._presentation = None
        self._pack = None
        self._speech = SpeechOutput(context.get("services"), context.get("session_id"))
        self._dialogue = Dialogue(self, context.get("services"))
        self._voice = VoiceConversation(self, context.get("services"))
        preferences = saved.get("presentation_settings", self._presentation_settings)
        if context.get("services") is not None:
            stored = (
                context["services"].call("data.get", {"path": "presentation.json"}).get("value")
            )
            if stored is not None:
                preferences = stored
        self._presentation_settings = self._validate_presentation(preferences)
        # Text activation does not import/load animation, STT, TTS or an LLM.
        self._state = "initialized"
        return self.status()

    def activate(self) -> dict[str, Any]:
        self._state = "active"
        return self.status()

    def deactivate(self) -> None:
        self._speech.close()
        self._voice.end(wait=True)
        self._dialogue.cancel(wait=True)
        self._initiative.end()
        self._presentation = None
        self._pack = None
        self._state = "inactive"

    def suspend(self) -> None:
        self._speech.close()
        self._voice.end(wait=True)
        self._dialogue.cancel(wait=True)
        self._initiative.end()
        self._presentation = None
        self._pack = None
        self._state = "suspended"

    def resume(self) -> dict[str, Any]:
        self._state = "active"
        return self.status()

    def dispose(self) -> None:
        self.deactivate()
        self._state = "disposed"

    def checkpoint(self) -> dict[str, Any]:
        self._speech.close()
        self._voice.end(wait=True)
        self._dialogue.cancel(wait=True)
        # No sound, animation queue or unfinished response resumes after a restart.
        return {
            "schema": "ramvi.character.state/3",
            "draft": self._draft,
            "model": self._model,
            "emotion": self._emotion,
            "presentation_settings": copy.deepcopy(self._presentation_settings),
            "personality": self._personality.checkpoint(),
            "emotional_state": self._emotions.checkpoint(),
            "task": self._task,
            "initiative": self._initiative.checkpoint(),
        }

    def status(self) -> dict[str, Any]:
        return {
            "mode": "text",
            "state": self._state,
            "draft": self._draft,
            "model": self._model,
            "emotion": self._emotion,
            "presentation": copy.deepcopy(self._presentation),
            "personality": self._personality.view(),
            "emotional_state": self._emotions.view(),
            "task": self._task,
            "initiative": self._initiative.status(),
            "devices_open": self._voice.state in {"listening", "hearing", "recognizing", "stopping"}
            or self._speech.state == "playing",
            "provider_connected": self._dialogue.connected
            or self._speech.state in {"synthesizing", "playing"},
        }

    def _require_active(self) -> None:
        if self._state != "active":
            raise RuntimeError("character is not active")

    def set_task(self, task: str) -> dict[str, Any]:
        self._require_active()
        self._dialogue.cancel()
        self._task = text(task, 4096)
        return self.status()

    def prepare_context(
        self,
        user_input: str,
        recent: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_active()
        options = Budget(**budget) if budget is not None else Budget()
        return (
            ContextOrchestrator(self._scope)
            .build(
                task=self._task,
                user_input=user_input,
                personality=self._personality.view(),
                state=self._emotions.view(),
                recent=recent,
                memories=memories,
                budget=options,
            )
            .view()
        )

    def configure_initiative(self, preferences: dict[str, Any]) -> dict[str, Any]:
        self._require_active()
        self._initiative.configure(preferences)
        return self.status()

    def begin_session(self) -> dict[str, Any]:
        self._require_active()
        self._initiative.begin()
        return self.status()

    def end_session(self) -> dict[str, Any]:
        self._initiative.end()
        return self.status()

    def claim_initiative(self, tokens: int) -> dict[str, Any]:
        self._require_active()
        view = self._personality.view()
        return self._initiative.claim(
            tokens,
            permitted=(
                view["boundaries"]["initiative"] and view["effective_traits"]["initiative"] > 0
            ),
        )

    def profiles(self) -> dict[str, Any]:
        return {"schema": "ramvi.character.profiles/1", "profiles": self._personality.catalog()}

    def select_profile(self, profile_id: str, version: str | None = None) -> dict[str, Any]:
        self._require_active()
        self._dialogue.cancel()
        self._personality.select(profile_id, version)
        return self.status()

    def add_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        self._require_active()
        self._personality.add_custom(profile)
        return self.profiles()

    def set_preferences(self, traits: dict[str, float]) -> dict[str, Any]:
        self._require_active()
        self._dialogue.cancel()
        self._personality.preferences(traits)
        return self.status()

    def set_boundary(self, name: str, allowed: bool) -> dict[str, Any]:
        # This extension is for explicit owner controls, never an LLM tool.
        self._require_active()
        self._dialogue.cancel()
        self._personality.boundary(name, allowed)
        return self.status()

    def react(
        self,
        event: str,
        strength: float = 0.5,
        relevance: float = 1,
        novelty: float = 0.5,
        certainty: float = 1,
    ) -> dict[str, Any]:
        self._require_active()
        self._dialogue.cancel()
        self._emotions.react(
            event, strength=strength, relevance=relevance, novelty=novelty, certainty=certainty
        )
        return self.status()

    def appraise(self, changes: dict[str, float]) -> dict[str, Any]:
        self._require_active()
        self._dialogue.cancel()
        self._emotions.influence(changes)
        return self.status()

    def add_emotion(self, definition: dict[str, Any]) -> dict[str, Any]:
        self._require_active()
        self._dialogue.cancel()
        self._emotions.add(definition)
        return self.status()

    def configure(self, draft: str, model: str | None = None) -> dict[str, Any]:
        if self._state != "active":
            raise RuntimeError("character is not active")
        if not isinstance(draft, str) or len(draft.encode("utf-8")) > 32_000:
            raise ValueError("invalid character draft")
        if model is not None and (not isinstance(model, str) or not model or len(model) > 255):
            raise ValueError("invalid model selection")
        if model != self._model:
            self._dialogue.cancel()
        self._draft, self._model = draft, model
        return self.status()

    @staticmethod
    def _validate_presentation(value: Any) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != {"pack_path", "show_avatar"}
            or not isinstance(value["pack_path"], str)
            or len(value["pack_path"].encode()) > 4096
            or type(value["show_avatar"]) is not bool
        ):
            raise ValueError("invalid presentation preferences")
        return copy.deepcopy(value)

    def dialogue(
        self, operation: str = "status", arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Declared UI command; expected owner errors do not crash the module worker."""
        try:
            self._require_active()
            args = arguments or {}
            if operation == "start":
                model = args.get("model", self._model)
                if not isinstance(model, str):
                    raise ValueError("select a model")
                self.configure("", model)
                return self._dialogue.start(args["text"], model)
            if operation == "cancel":
                self._dialogue.cancel()
            elif operation == "configure":
                self.configure(args.get("draft", self._draft), args.get("model", self._model))
            elif operation == "profile":
                self.select_profile(args["id"])
            elif operation == "boundary":
                self.set_boundary(args["name"], args["allowed"])
            elif operation == "presentation":
                preferences = self._validate_presentation(args["settings"])
                if self._dialogue.services is not None:
                    self._dialogue.services.call(
                        "data.set", {"path": "presentation.json", "value": preferences}
                    )
                self._presentation_settings = preferences
            elif operation not in {"status", "poll"}:
                raise ValueError("unknown dialogue operation")
            result = self._dialogue.view(args.get("response_id"), args.get("cursor", 0))
            if operation == "poll":
                return {"job": result["job"]}
            result["presentation_settings"] = copy.deepcopy(self._presentation_settings)
            result["character"] = {
                "draft": self._draft,
                "model": self._model,
                "profile_id": self._personality.view()["base"]["id"],
            }
            result["profiles"] = [
                {"id": p["id"], "name": p["name"]} for p in self.profiles()["profiles"]
            ]
            if result["job"]:
                result["job"]["events"] = []
                result["job"]["cursor"] = 0
            # History and owner draft are separately bounded; keep one RPC below 64 KiB.
            while (
                len(json.dumps(result, ensure_ascii=False).encode()) > 60_000 and result["history"]
            ):
                result["history"].pop(0)
            return result
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            return {"error": type(error).__name__, "code": "character.request_error"}

    def voice(
        self, operation: str = "status", arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            self._require_active()
            args = arguments or {}
            if not isinstance(args, dict):
                raise ValueError("invalid voice arguments")
            if operation == "capabilities":
                return self._voice.capabilities()
            if operation == "start":
                if "model" in args:
                    self.configure(self._draft, args["model"])
                return self._voice.start()
            if operation == "stop":
                return self._voice.end()
            if operation != "status":
                raise ValueError("unknown voice operation")
            self._voice.touch()
            return self._voice.view()
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            return {"error": type(error).__name__, "code": "character.voice_error"}

    def speech(
        self, operation: str = "status", arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            self._require_active()
            args = arguments or {}
            if not isinstance(args, dict):
                raise ValueError("invalid speech arguments")
            if operation == "policy":
                return self._speech.policy(args.get("settings"))
            if operation == "catalog":
                return self._speech.catalog(args.get("refresh", False))
            if operation == "configure":
                return self._speech.configure(args["settings"])
            if operation == "resume":
                return self._speech.arm()
            if operation == "stop":
                return self._speech.pause()
            if operation == "key":
                if self._speech.services is None:
                    raise RuntimeError("speech is unavailable")
                return dict(self._speech.services.call("speech.key", args))
            if operation == "preview":
                return self._speech.preview()
            if operation != "status":
                raise ValueError("unknown speech operation")
            self._speech.touch()
            return self._speech.view()
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
            return {"error": type(error).__name__, "code": "speech.failed"}

    def present(self, emotion: str, animation: str = "idle") -> dict[str, Any]:
        if self._state != "active":
            raise RuntimeError("character is not active")
        pack = self._pack if self._pack is not None else load_pack()
        directive = presentation(pack, emotion, animation)
        self._pack = pack
        self._emotion, self._presentation = emotion, directive
        return self.status()

    def stop_presentation(self) -> dict[str, Any]:
        self._presentation = None
        return self.status()


def create() -> CharacterModule:
    return CharacterModule()
