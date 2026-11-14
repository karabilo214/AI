"""Model template policy uses explicit checkpoint metadata, without loading weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ramvi_engine.model_template import configure_model_template

LEGACY = "legacy tokenizer template"
THINKING = "{% if add_generation_prompt %}assistant\n<think>\n{% endif %}"


def tokenizer() -> SimpleNamespace:
    return SimpleNamespace(chat_template=LEGACY)


def gguf(metadata: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(g=SimpleNamespace(meta=metadata))


def test_thinking_model_uses_its_embedded_template() -> None:
    engine = gguf({"general.finetune": "Thinking", "tokenizer.chat_template": THINKING})
    tok = tokenizer()
    configure_model_template(engine, tok)
    assert tok.chat_template == THINKING
    assert tok.ramvi_thinking_required is True
    assert engine.thinking_required is True
    assert tok.ramvi_legacy_chat_template == LEGACY
    assert tok.ramvi_template_source == engine.chat_template_source
    assert "gguf" in tok.ramvi_template_source.lower()


def test_policy_uses_metadata_instead_of_model_name() -> None:
    engine = gguf(
        {
            "general.name": "Unrelated Name",
            "general.finetune": "Thinking",
            "tokenizer.chat_template": THINKING,
        }
    )
    tok = tokenizer()
    configure_model_template(engine, tok)
    assert tok.ramvi_thinking_required is True
    assert tok.chat_template == THINKING


@pytest.mark.parametrize(
    "template",
    [
        None,
        "",
        "   ",
        1,
        True,
        [],
        {},
        "assistant without thinking opener",
        "<think>" + "x" * 65536,
        "<think>" + "я" * 32768,
    ],
)
def test_thinking_template_missing_or_invalid_is_rejected(template: object) -> None:
    metadata: dict[str, object] = {"general.finetune": "Thinking"}
    if template is not None:
        metadata["tokenizer.chat_template"] = template
    tok = tokenizer()
    with pytest.raises(ValueError):
        configure_model_template(gguf(metadata), tok)
    assert tok.chat_template == LEGACY


@pytest.mark.parametrize("finetune", [None, "Instruct", "", False, True, 1])
def test_ordinary_gguf_keeps_legacy_template(finetune: object) -> None:
    metadata: dict[str, object] = {
        "general.name": "Thinking named but not declared",
        "tokenizer.chat_template": THINKING,
    }
    if finetune is not None:
        metadata["general.finetune"] = finetune
    engine = gguf(metadata)
    tok = tokenizer()
    configure_model_template(engine, tok)
    assert tok.chat_template == LEGACY
    assert tok.ramvi_thinking_required is False
    assert engine.thinking_required is False


def test_hf_engine_template_is_unchanged() -> None:
    engine = SimpleNamespace()
    tok = tokenizer()
    configure_model_template(engine, tok)
    assert tok.chat_template == LEGACY
    assert tok.ramvi_thinking_required is False
    assert engine.thinking_required is False


def test_policy_import_has_no_model_or_cuda_dependencies() -> None:
    engine_path = Path(__file__).resolve().parents[1] / "engine"
    code = """
import sys
from ramvi_engine.model_template import configure_model_template
assert callable(configure_model_template)
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
assert not any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, " + repr(str(engine_path)) + ");" + code,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_required_thinking_is_parsed_even_when_request_disables_it() -> None:
    from ramvi_engine.capabilities import engine_capabilities
    from ramvi_engine.server.app import App

    class Tokenizer:
        def decode(self, tokens: list[int], **kwargs: Any) -> str:
            return "".join({0: "reasoned", 1: "</think>", 2: "4", 99: ""}[t] for t in tokens)

    class Engine:
        thinking_required = True
        chat_template_source = "gguf"
        supports_mask = True
        maxt = 64
        pos = None
        place: list[str] = []
        ck = SimpleNamespace(eos={99}, cfg={"model_type": "qwen3"})

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            stats = kwargs["stats"]
            stats.update(prompt_n=1, prompt_ms=1.0, first_token_ms=1.0, gen_ms=4.0)
            for i, token in enumerate([0, 1, 2, 99]):
                stats["gen_n"] = i + 1
                yield token

    engine = Engine()
    app = App(engine, Tokenizer(), "thinking-only")
    for args in (([7], 32, 0), ([7], 32, 0, 1, None, None, False)):
        events = list(app.run(*args))
        assert "".join(v for k, v in events if k == "text") == "4"
        assert "".join(v for k, v in events if k == "reasoning") == "reasoned"
    assert app.metadata()["thinking_required"] is True
    assert app.metadata()["chat_template_source"] == "gguf"
    capabilities = engine_capabilities(engine)["loaded_model"]
    assert capabilities["thinking_required"] and not capabilities["grammar_mask"]
    with pytest.raises(ValueError, match="Thinking-only"):
        app.constraint("json")
