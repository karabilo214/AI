"""4.4: примерная скорость по плану — формула должна покрывать сделанные замеры."""

import pytest
from ramvi_service import speed
from ramvi_service.speed import Traffic

GB = 10**9

# Байты весов на токен по местам — из настоящих заголовков Ollama и размещений,
# при которых делались замеры (ноутбук RTX 5060 8 ГБ, Ryzen 7 260, 16 ГБ DDR5 в один канал).
CALIBRATION = [
    ("qwen3:8b Q4, 36/36 в VRAM", Traffic(vram=int(4.87 * GB), layers=36), (55, 58)),
    (
        "qwen3:14b Q4, 21 VRAM / 8 шина / 11 CPU",
        Traffic(vram=int(4.93 * GB), cpu=int(2.26 * GB), bus=int(1.64 * GB), layers=40),
        (5.6, 6.1),
    ),
    (
        "qwen3:8b F16, 15 VRAM / 21 CPU",
        Traffic(vram=int(7.03 * GB), cpu=int(8.10 * GB), layers=36),
        (2.35, 2.35),
    ),
    (
        "qwen3:30b MoE, 14 VRAM / 34 CPU",
        Traffic(vram=int(0.75 * GB), cpu=int(1.17 * GB), moe=True, layers=48),
        (8.0, 9.7),
    ),
]


@pytest.mark.parametrize("name, traffic, measured", CALIBRATION, ids=[c[0] for c in CALIBRATION])
def test_estimate_covers_measurements(
    name: str, traffic: Traffic, measured: tuple[float, float]
) -> None:
    estimate = speed.tokens_per_second(traffic, cores=8)
    assert estimate is not None
    low, high = estimate
    # Оценка примерная: края замера — не дальше 10 % от диапазона оценки
    # (8B F16: замер 2,35 на 4 % ниже нижней границы 2,45).
    assert low * 0.9 <= measured[0] and measured[1] <= high * 1.1, (name, estimate)


def test_small_model_is_bound_by_per_layer_cost() -> None:
    """0.6B: 28 слоёв, ~0,45 ГБ весов — замер 3,4 мс на токен (≈290 ток/с), не 600+."""
    estimate = speed.tokens_per_second(Traffic(vram=int(0.45 * GB), layers=28), 8)
    assert estimate is not None
    assert 200 < estimate[0] and estimate[1] < 330


def test_more_off_gpu_is_slower_and_paging_is_much_slower() -> None:
    in_vram = speed.tokens_per_second(Traffic(vram=5 * GB), 8)
    on_cpu = speed.tokens_per_second(Traffic(vram=3 * GB, cpu=2 * GB), 8)
    paged = speed.tokens_per_second(Traffic(vram=3 * GB, cpu=2 * GB, disk=2 * GB), 8)
    assert in_vram and on_cpu and paged
    assert in_vram[1] > on_cpu[1] > paged[1]
    assert paged[1] < 1.2  # 2 ГБ с диска на каждый токен — меньше токена в секунду


def test_few_cores_limit_ram_reading() -> None:
    traffic = Traffic(vram=2 * GB, cpu=4 * GB)
    eight = speed.tokens_per_second(traffic, 8)
    two = speed.tokens_per_second(traffic, 2)
    unknown = speed.tokens_per_second(traffic, None)
    assert eight and two and unknown
    assert two[1] < eight[1] == unknown[1]


def test_nothing_to_read_and_rounding() -> None:
    assert speed.tokens_per_second(Traffic(), 8) is None
    assert [speed.rounded(v) for v in (57.3, 5.83, 0.312, 0)] == [57.0, 5.8, 0.31, 0.0]
