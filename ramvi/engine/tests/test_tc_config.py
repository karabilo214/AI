"""B.6: подобранный блок ядра тензорных блоков (правило по замеру; с CUDA — побайтная сверка)."""

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from ramvi_engine.kernels.gkernels import _cfg_tc  # noqa: E402


def test_measured_rule_and_fixed_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAMVI_TC_CFG", raising=False)
    assert _cfg_tc("Q4_K", 12288, 4096) == (128, 8)  # 8B gate
    assert _cfg_tc("Q4_K", 4096, 4096) == (64, 4)  # 8B q
    assert _cfg_tc("Q6_K", 5120, 17408) == (64, 4)  # 14B down
    assert _cfg_tc("Q4_K", 1024, 5120) == (64, 4)  # k
    monkeypatch.setenv("RAMVI_TC_CFG", "fixed")
    assert _cfg_tc("Q4_K", 12288, 4096) == (32, 4)
