"""4.10 ENG-MULTIGPU: раскладка слоёв по картам (без видеокарты)."""

import pytest
from ramvi_engine.planning.multigpu import split_layers, visible_devices

GB = 10**9


def test_one_card_takes_layers_in_order_until_full() -> None:
    assert split_layers([GB] * 10, [4 * GB]) == [0, 0, 0, 0]


def test_model_that_does_not_fit_one_card_fits_two() -> None:
    """Qwen3-32B Q4 (~19,8 ГБ, 64 слоя) на двух картах по 12 ГБ: всё в видеопамяти."""
    layers = [300 * 10**6] * 64  # ≈19,2 ГБ слоёв вместе с KV
    budgets = [11 * GB, 11 * GB]
    one = split_layers(layers, budgets[:1])
    two = split_layers(layers, budgets)
    assert len(one) < 64  # одной карте мало
    assert len(two) == 64
    assert two == sorted(two)  # подряд идущие блоки
    assert two.count(0) == two.count(1) == 32  # пропорционально равным бюджетам


def test_proportional_to_unequal_budgets() -> None:
    # 9 ГБ на бюджеты 6/3/3: цели 4,5/2,25/2,25 ГБ; последняя карта добирает остаток.
    assert split_layers([GB] * 9, [6 * GB, 3 * GB, 3 * GB]) == [0] * 4 + [1] * 2 + [2] * 3


def test_overflow_fills_cards_in_turn_and_reports_how_many_fit() -> None:
    plan = split_layers([GB] * 20, [5 * GB, 3 * GB])
    assert plan == [0] * 5 + [1] * 3  # 8 слоёв в видеопамяти, остальные — шина/CPU


def test_resident_limit_and_edge_cases() -> None:
    assert split_layers([GB] * 10, [20 * GB, 20 * GB], resident=4) == [0, 0, 1, 1]
    assert split_layers([GB] * 10, [20 * GB], resident=0) == []
    assert split_layers([], [GB]) == []
    assert split_layers([GB], []) == []
    assert split_layers([2 * GB], [GB, GB]) == []  # слой не делится между картами
    with pytest.raises(ValueError):
        split_layers([-1], [GB])


def test_device_selection() -> None:
    assert visible_devices(2, "") == [0, 1]
    assert visible_devices(3, "1") == [1]  # «одна карта» для сравнения на той же машине
    assert visible_devices(3, "2, 0, 2") == [2, 0]
    with pytest.raises(ValueError):
        visible_devices(1, "1")


def test_planner_budget_adds_secondary_cards_only_for_dense_gguf() -> None:
    from ramvi_engine.planning.automatic import automatic_reserve
    from ramvi_engine.planning.multigpu import SECONDARY_OVERHEAD, usable_vram

    free = [11 * GB, 11 * GB]
    assert usable_vram(free[:1], automatic_reserve) == 11 * GB  # одна карта — как раньше
    two = usable_vram(free, automatic_reserve)
    assert two == 11 * GB + 11 * GB - automatic_reserve(11 * GB) - SECONDARY_OVERHEAD
    assert usable_vram(free, automatic_reserve, gguf=False) == 11 * GB  # MoE/HF — одна карта
    assert usable_vram([], automatic_reserve) == 0
