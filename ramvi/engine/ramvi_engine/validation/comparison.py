"""Сравнение логитов и ID без вывода о правильности по совпадению текста."""

import math
from collections.abc import Sequence
from typing import Any


def compare_logits(
    actual: Sequence[float], expected: Sequence[float], *, atol: float, rtol: float
) -> dict[str, Any]:
    """Конечные полные векторы; tie-break argmax — минимальный ID."""
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("допуски должны быть конечными неотрицательными")
    if not actual or len(actual) != len(expected):
        raise ValueError("векторы логитов должны быть непустыми и одной длины")
    a, b = list(actual), list(expected)
    if any(not math.isfinite(value) for value in a + b):
        raise ValueError("логиты должны быть конечными")
    errors = [abs(x - y) for x, y in zip(a, b, strict=True)]
    violations = [
        i
        for i, (error, ref) in enumerate(zip(errors, b, strict=True))
        if error > atol + rtol * abs(ref)
    ]
    ai = max(range(len(a)), key=lambda i: a[i])
    bi = max(range(len(b)), key=lambda i: b[i])
    ordered = sorted(range(len(b)), key=lambda i: (-b[i], i))
    margin = b[ordered[0]] - b[ordered[1]] if len(b) > 1 else None
    return {
        "passed": not violations and ai == bi,
        "logits_within_tolerance": not violations,
        "argmax_matches": ai == bi,
        "actual_argmax": ai,
        "expected_argmax": bi,
        "max_abs_error": max(errors),
        "rms_error": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "violating_logits": len(violations),
        "first_violation": violations[0] if violations else None,
        "reference_top2_margin": margin,
        "atol": atol,
        "rtol": rtol,
        "vocab_size": len(a),
    }


def compare_tokens(actual: Sequence[int], expected: Sequence[int]) -> dict[str, Any]:
    """ID токенов, включая EOS; не игнорировать суффикс более длинного ответа."""
    for token in [*actual, *expected]:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError("ID токенов должны быть неотрицательными целыми")
    first = next(
        (i for i, (x, y) in enumerate(zip(actual, expected, strict=False)) if x != y),
        min(len(actual), len(expected)),
    )
    match = list(actual) == list(expected)
    return {
        "passed": match,
        "actual_count": len(actual),
        "expected_count": len(expected),
        "first_mismatch": None if match else first,
    }
