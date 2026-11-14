"""Снимок native-логитов для одинакового teacher-forced prefix."""

from importlib import import_module
from typing import Any


def native_logits(engine: Any, reference: Any, tokens: list[int]) -> tuple[Any, int]:
    """HF: exact projection native hidden; GGUF: actual native output kernel.

    Никакого сравнения текстов: отдельно проверяется выбор native head, в том
    числе HF bound-based exact selection, относительно полного argmax.
    """
    engine.ctx = []
    selected = int(engine.process(tokens, 0, last_only=True)[-1])
    row = 0 if len(tokens) > 16 else len(tokens) - 1
    if hasattr(engine, "logits"):
        logits = engine.logits[row].detach().float().cpu().clone()
    else:
        logits = reference.project(engine.hb[row : row + 1].detach().float().to(reference.device))[
            0
        ].cpu()
    return logits, selected


def native_trace(engine: Any, reference: Any, tokens: list[int]) -> tuple[Any, int, dict[int, Any]]:
    """Диагностический hook на выходах слоёв; восстанавливает все instance attrs."""
    torch = import_module("torch")

    history: dict[int, Any] = {}
    saved: dict[str, tuple[bool, Any]] = {}
    cpu_saved: dict[int, Any] = {}

    def capture(layer: int, x: Any, position: int) -> None:
        if position + len(x) == len(tokens):
            value = x[-1]
            history[layer] = (
                value.detach().float().cpu().clone()
                if torch.is_tensor(value)
                else torch.tensor(value.copy())
            )

    def install(name: str, build: Any) -> None:
        if not hasattr(engine, name):
            return
        saved[name] = (name in vars(engine), vars(engine).get(name))
        setattr(engine, name, build(getattr(engine, name)))

    def batch(original: Any) -> Any:
        def call(layer: int, x: Any, count: int, *args: Any, **kwargs: Any) -> Any:
            result = original(layer, x, count, *args, **kwargs)
            capture(layer, x, int(engine.pos.item()))
            return result

        return call

    def gpu(original: Any) -> Any:
        def call(layer: int, weights: Any, x: Any, count: int) -> Any:
            result = original(layer, weights, x, count)
            capture(layer, x, int(engine.pos.item()))
            return result

        return call

    def cpu(original: Any) -> Any:
        def call(layer: int, x: Any, position: int) -> Any:
            result = original(layer, x, position)
            capture(layer, x, position)
            return result

        return call

    try:
        if hasattr(engine, "cpul"):
            for layer, original in list(engine.cpul.items()):
                cpu_saved[layer] = original

                def wrap_cpu(
                    value: Any, pos: int, inv: Any, *, original: Any = original, layer: int = layer
                ) -> Any:
                    result = original(value, pos, inv)
                    capture(layer, value, pos)
                    return result

                engine.cpul[layer] = wrap_cpu
            install("_moe", batch)
        else:
            install("_batch_layer", batch)
            install("_gpu_layer", gpu)
            install("_cpu_layer", cpu)
        logits, selected = native_logits(engine, reference, tokens)
        return logits, selected, history
    finally:
        for name, (existed, value) in saved.items():
            if existed:
                setattr(engine, name, value)
            else:
                vars(engine).pop(name, None)
        for layer, original in cpu_saved.items():
            engine.cpul[layer] = original
