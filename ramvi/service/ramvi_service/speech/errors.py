"""Stable, non-sensitive diagnostic codes. Never expose provider/device exception text."""

import re

CODES = frozenset(
    {
        "speech.disabled",
        "speech.remote_text_disabled",
        "speech.key_missing",
        "speech.secrets_unavailable",
        "speech.voice_not_supported",
        "speech.voice_missing",
        "speech.busy",
        "speech.stale_generation",
        "speech.device_unavailable",
        "speech.device_lost",
        "speech.device_timeout",
        "speech.device_close_failed",
        "speech.provider_timeout",
        "speech.audio_timeout",
        "speech.deadline",
        "speech.provider_failed",
        "speech.cleanup_failed",
        "speech.failed",
        "speech.closed",
    }
)


def failure(error: BaseException, stage: str = "provider") -> str:
    message = str(error)
    if message in CODES or re.fullmatch(r"speech\.api_http_[1-5]\d\d", message):
        return message
    if isinstance(error, TimeoutError):
        return "speech.device_timeout" if stage.startswith("device") else "speech.provider_timeout"
    if stage.startswith("device"):
        return "speech.device_lost" if stage == "device_write" else "speech.device_unavailable"
    return "speech.provider_failed"
