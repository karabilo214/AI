"""B.1 CLI-HELP-01: у каждой консольной команды RAMVI справка из каталогов переводов —
описание команды, каждого параметра, значения по умолчанию и допустимые варианты."""

import argparse
import contextlib
import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from ramvi_engine import cli as engine_cli  # noqa: E402
from ramvi_service import app, updates  # noqa: E402
from ramvi_service.i18n import LANGUAGES, translator  # noqa: E402
from ramvi_service.voice import vad_worker, worker  # noqa: E402


def engine_parser(t: Callable[..., str]) -> argparse.ArgumentParser:
    """Парсер движка строится внутри main — перехватываем его до разбора."""
    caught: list[argparse.ArgumentParser] = []

    def stop(self: argparse.ArgumentParser, *_: Any) -> None:
        caught.append(self)
        raise SystemExit(0)

    english = t("cli.default", value="").startswith("default")
    previous = engine_cli.translator
    with (
        patch.object(argparse.ArgumentParser, "parse_args", stop),
        patch.object(engine_cli, "translator", lambda *_: previous("en" if english else "ru")),
        pytest.raises(SystemExit),
    ):
        engine_cli.main()
    return caught[0]


COMMANDS: dict[str, Callable[[Callable[..., str]], argparse.ArgumentParser]] = {
    "ramvi-engine": engine_parser,
    "ramvi-service": app.build_parser,
    "ramvi-update": updates.build_parser,
    "ramvi-stt-worker": worker.build_parser,
    "ramvi-vad-worker": vad_worker.build_parser,
}


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_and_parameter_has_translated_help(command: str, language: str) -> None:
    t = translator(language)
    parser = COMMANDS[command](t)
    assert parser.prog == command
    assert parser.description and not parser.description.startswith("cli."), command
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--help"])
    assert stopped.value.code == 0
    text = " ".join(out.getvalue().split())
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        name = action.option_strings[-1] if action.option_strings else action.dest
        assert action.help and "cli." not in action.help, (command, name)  # ключ без перевода
        if action.option_strings or not action.choices:  # команда с вариантами видна списком
            assert name in text, (command, name)
        for choice in action.choices or ():
            assert str(choice) in text, (command, name, choice)
        if (
            action.dest != "api_key"
            and action.default not in (None, False)
            and isinstance(action.default, (str, int, float))
        ):
            assert t("cli.default", value=action.default) in action.help, (command, name)


def test_engine_help_never_shows_the_access_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMVI_API_KEY", "secret-key-65537")
    parser = engine_parser(translator("ru"))
    assert "secret-key-65537" not in parser.format_help()
