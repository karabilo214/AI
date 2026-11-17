"""B.6: ширина куска чтения запроса (без видеокарты — правила выбора; с CUDA — побитовое совпадение)."""

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from ramvi_engine.core.gengine import GM, prefill_rows, wide_ok  # noqa: E402
from ramvi_engine.kernels.gkernels import tc_width  # noqa: E402


def test_rows_follow_batch_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAMVI_PREFILL_ROWS", raising=False)
    assert prefill_rows(16) == 64
    assert prefill_rows(8) == 8  # план урезал пачку после нехватки памяти — кусок тоже
    monkeypatch.setenv("RAMVI_PREFILL_ROWS", "16")
    assert prefill_rows(16) == 16
    monkeypatch.setenv("RAMVI_PREFILL_ROWS", "48")
    with pytest.raises(ValueError):
        prefill_rows(16)


def test_wide_only_for_tensor_core_layers() -> None:
    quant = {n: {"t": "Q4_K"} for n in GM}
    assert wide_ok(quant)
    assert not wide_ok({**quant, "v": {"t": "F16"}})
    assert [tc_width(t) for t in (2, 16, 17, 32, 33, 64)] == [16, 16, 32, 32, 64, 64]
