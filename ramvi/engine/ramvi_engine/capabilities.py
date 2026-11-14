"""Карта реализованных возможностей без импорта вычислительных библиотек."""

from typing import Any


def capability_catalog() -> dict[str, Any]:
    """Вернуть новый JSON-совместимый снимок контракта, не проверяя GPU/веса."""
    rows = []
    for model_format, architecture, model_type in (
        ("safetensors", "qwen3", "dense"),
        ("gguf", "qwen3", "dense"),
        ("gguf", "qwen3moe", "moe"),
    ):
        moe = model_type == "moe"
        gguf = model_format == "gguf"
        rows.append(
            {
                "format": model_format,
                "architecture": architecture,
                "model_type": model_type,
                "backend": "cuda",
                "execution_modes": ["gpu", "cpu_gpu"]
                if moe
                else (
                    ["gpu", "stream", "cpu_gpu", "hybrid"] if gguf else ["gpu", "stream", "cpu_gpu"]
                ),
                "cpu_only": False,
                "draft": {
                    "supported": True,
                    "conditions": ["compatible_tokenizer", "greedy_decoding", "memory_budget"],
                },
                "json": {
                    "supported": gguf,
                    "mode": "grammar" if gguf else "unsupported",
                    "conditions": [
                        "xgrammar",
                        "compatible_tokenizer",
                        "supported_schema",
                        "output_budget",
                    ]
                    if gguf
                    else [],
                },
                "tools": {
                    "supported": True,
                    "mode": "template_and_parser",
                    "conditions": ["tool_capable_chat_template", "model_quality"],
                    "executes_tools": False,
                    "guaranteed_call": False,
                },
                "weight_constraints": (
                    "BF16 CPU matrices; model config and tensor layout must match Qwen3"
                    if not gguf
                    else "validate_qwen: dense matrices Q4_K/Q6_K/F16/F32; expert matrices Q4_K/Q6_K"
                ),
                "requirements": ["cuda_device", "torch_cuda", "triton", "compatible_weights"]
                + (["native_qmm"] if moe else []),
            }
        )
    return {
        "schema_version": 1,
        "scope": "implementation_catalog",
        "runtime_verified": False,
        "unknown_combinations": "unsupported",
        "backends": {
            "cuda": {"implemented": True, "requires_runtime_check": True},
            "cpu": {"implemented": False, "note": "CPU layers require CUDA engine"},
            "metal": {"implemented": False},
            "rocm": {"implemented": False},
        },
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "rows": rows,
        "limits": {"parallel_generations": 1, "n": 1, "context": "fixed_at_load"},
    }


def engine_capabilities(engine: Any, *, draft_loaded: bool = False) -> dict[str, Any]:
    """Добавить фактические метаданные загруженного движка без пробного inference."""
    result = capability_catalog()
    g = getattr(engine, "g", None)
    ck = getattr(engine, "ck", None)
    model_format = "gguf" if g is not None else "safetensors"
    metadata = getattr(g, "meta", {}) if g is not None else getattr(ck, "cfg", {})
    architecture = metadata.get("general.architecture" if g is not None else "model_type")
    match = next(
        (
            row
            for row in result["rows"]
            if row["format"] == model_format and row["architecture"] == architecture
        ),
        None,
    )
    result["loaded_model"] = {
        "format": model_format if match is not None else None,
        "architecture": architecture,
        "recognized": match is not None,
        "context_tokens": getattr(engine, "maxt", None),
        "placement": getattr(engine, "place", None),
        "draft_loaded": draft_loaded,
        "grammar_mask": bool(getattr(engine, "supports_mask", False))
        and not bool(getattr(engine, "thinking_required", False)),
        "readiness": "not_probed",
        "thinking_required": bool(getattr(engine, "thinking_required", False)),
        "chat_template_source": getattr(engine, "chat_template_source", "tokenizer"),
    }
    return result
