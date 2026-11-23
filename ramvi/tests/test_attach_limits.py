"""B.15: лимиты прикрепляемых файлов задаёт пользователь в настройках приложения."""

from pathlib import Path

import pytest
from ramvi_service import settings as st
from ramvi_service.app_config import AppConfig
from ramvi_service.chat import ChatError, Limits, attachment_limits, check_attachments
from ramvi_service.ui_api import MAX_BODY, max_body


def files(*sizes: int) -> list[dict[str, str]]:
    return [{"name": f"{i}.txt", "text": "x" * size} for i, size in enumerate(sizes)]


def test_defaults_keep_previous_chat_limits() -> None:
    assert attachment_limits(st.Settings()) == Limits()


def test_saved_limits_apply_and_zero_means_no_limit(tmp_path: Path) -> None:
    config = AppConfig(tmp_path, tmp_path / "data")
    config.update({"attach_max_files": 1, "attach_max_file_kb": 1, "attach_max_total_kb": 1})
    limits = attachment_limits(st.load(tmp_path))
    check_attachments(files(1024), limits)
    with pytest.raises(ChatError) as error:
        check_attachments(files(1025), limits)
    assert error.value.code == "chat.file_too_large"
    with pytest.raises(ChatError):
        check_attachments(files(1, 1), limits)
    config.update({"attach_max_files": 0, "attach_max_file_kb": 0, "attach_max_total_kb": 0})
    unlimited = attachment_limits(st.load(tmp_path))
    # Больше прежних 5 файлов / 128 КБ / 256 КБ — проходит.
    assert len(check_attachments(files(*[300 * 1024] * 6), unlimited)) == 6
    assert max_body(unlimited) is None


@pytest.mark.parametrize("value", [-1, True, 1.5, "5"])
def test_bad_limit_is_refused_and_nothing_is_written(tmp_path: Path, value: object) -> None:
    config = AppConfig(tmp_path, tmp_path / "data")
    with pytest.raises(ChatError):
        config.update({"attach_max_files": value})
    assert st.load(tmp_path).attach_max_files == 5


def test_request_ceiling_follows_the_total_limit() -> None:
    assert max_body(Limits()) == 256 * 1024 * 6 + MAX_BODY
    big = Limits(max_total_bytes=10 * 2**20)
    assert max_body(big) is not None and max_body(big) > 10 * 2**20 * 6  # type: ignore[operator]


def test_engine_context_overflow_becomes_a_translated_error_with_numbers() -> None:
    from ramvi_service.i18n import UserError
    from ramvi_service.providers import stream_chunks

    event = {"error": {"message": "context overflow: needed=9100 available=8192"}}
    with pytest.raises(UserError) as error:
        list(stream_chunks(event))
    assert error.value.code == "chat.context_overflow"
    assert error.value.params == {"needed": 9100, "available": 8192}
    with pytest.raises(RuntimeError, match="other"):
        list(stream_chunks({"error": "other failure"}))


def test_engine_refusal_before_the_stream_is_also_explained() -> None:
    import io
    import urllib.error

    from ramvi_service.i18n import UserError
    from ramvi_service.providers import refused

    body = b'{"error": {"message": "context overflow: needed=40000 available=32768"}}'
    error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, io.BytesIO(body))  # type: ignore[arg-type]
    result = refused(error)
    assert isinstance(result, UserError) and result.params == {"needed": 40000, "available": 32768}
    other = urllib.error.HTTPError("http://x", 500, "x", {}, io.BytesIO(b"not json"))  # type: ignore[arg-type]
    assert refused(other) is other
